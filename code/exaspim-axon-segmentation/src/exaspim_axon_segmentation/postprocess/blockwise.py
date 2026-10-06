"""Skeletonize probability maps too large for memory, block by block.

Same result as :func:`~exaspim_axon_segmentation.postprocess.skeleton.skeletonize_probability`,
up to small differences where blocks meet, in three passes:

1. **Local, per block** (parallel). Each block is read with a ``halo`` of context and thinned,
   spur-pruned and tip-to-tip bridged (:func:`thin_probability`, then the bridge join), all of
   which look no further than ~100 µm. A block keeps only the nodes in its own *core*; an edge
   from a core node into the halo is recorded as a *cut* and later attached to the neighbouring
   block's node at that position (or the nearest one within ``stitch_vox``).
2. **Global, on the stitched graph**. Piece-level filters need whole pieces: minimum length,
   single-plane speckle and the anchor / attach isolation rule (:func:`filter_pieces`).
3. **Local again, per block** (sequential). The final gap filling routes along the probability
   map; it runs block by block on the global graph with global piece labels and a shared
   union-find, so joins never close a loop across blocks.

Coordinates: voxel indices are global within ``region`` (the probability array's own index
space); world µm = (region.start + voxel) × scale + translation, as for the whole-volume path.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.io import Box, LevelTransform
from exaspim_axon_segmentation.postprocess.skeleton import (
    AxonSkeleton,
    SkeletonConfig,
    _adjacency,
    _join_pieces,
    filter_pieces,
    thin_probability,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockConfig:
    """Block layout of :func:`skeletonize_blockwise` (sizes in voxels of the probability grid)."""

    block_vox: int = 384  # core side; ~1.5 x 1.15 x 1.15 mm at level 2
    halo_vox: int = 32  # context read around each core (> bridge / connect gaps, ~100 µm)
    stitch_vox: float = 2.5  # a cut edge attaches to the neighbour's node within this
    workers: int = 4  # processes for the local pass


def _open(source):
    """Return the probability array of a numpy array or a zarr URI (opened in each worker)."""
    if isinstance(source, str):
        import zarr

        return zarr.open_group(source, mode="r")["0"]
    return source


_WORKER_MASK = None  # set once per worker process (the mask can be a large region array)


def _init_worker(mask_fn) -> None:
    global _WORKER_MASK
    _WORKER_MASK = mask_fn


def _read(source, lo: NDArray, hi: NDArray, mask_fn=None, min_value: int = 0):
    """Read a block; None if no voxel exceeds ``min_value`` (checked before masking)."""
    array = _open(source)
    block = np.asarray(array[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))])
    if min_value and not (block > min_value).any():
        return None
    if mask_fn is not None:
        block = np.where(mask_fn(lo, hi), block, 0).astype(np.uint8)
    return block


def _local_pass(task) -> dict | None:
    """Thin and bridge one padded block; return its core nodes, their edges and cut edges."""
    source, lo, hi, core_lo, core_hi, scale, config, mask_fn = task
    mask_fn = _WORKER_MASK if isinstance(mask_fn, str) else mask_fn
    core_value = round(config.core * 255)
    probability = _read(source, lo, hi, mask_fn, min_value=core_value)
    if probability is None or not (probability > core_value).any():
        return None
    voxel, edges = thin_probability(probability, scale, config)
    if not len(voxel):
        return None
    filled = np.zeros(len(voxel), dtype=bool)
    voxel, edges, filled = _join_pieces(
        voxel, edges, filled, probability, scale,
        gap_um=config.bridge_um, min_cos=config.bridge_min_cos, mutual=True, min_p=0.0,
    )  # fmt: skip
    g = voxel + lo  # global voxel indices
    own = np.all((g >= core_lo) & (g < core_hi), axis=1)
    if not own.any():
        return None
    remap = np.full(len(g), -1, dtype=np.int64)
    remap[own] = np.arange(int(own.sum()))
    a, b = own[edges[:, 0]], own[edges[:, 1]]
    inner = remap[edges[a & b]]
    cut_from = np.r_[remap[edges[a & ~b, 0]], remap[edges[b & ~a, 1]]]
    cut_to = np.r_[g[edges[a & ~b, 1]], g[edges[b & ~a, 0]]].reshape(-1, 3)
    return {"voxel": g[own].astype(np.int32), "edges": inner, "filled": filled[own],
            "cut_from": cut_from, "cut_to": cut_to.astype(np.int32)}  # fmt: skip


def _blocks(shape: NDArray, block: BlockConfig):
    side = np.array([block.block_vox] * 3)
    counts = -(-shape // side)
    for idx in np.ndindex(*counts):
        core_lo = np.array(idx) * side
        core_hi = np.minimum(core_lo + side, shape)
        lo = np.maximum(core_lo - block.halo_vox, 0)
        hi = np.minimum(core_hi + block.halo_vox, shape)
        yield lo, hi, core_lo, core_hi


def skeletonize_blockwise(
    source,
    grid: LevelTransform,
    region: Box,
    config: SkeletonConfig,
    block: BlockConfig = BlockConfig(),  # noqa: B008
    *,
    mask_fn=None,
) -> AxonSkeleton:
    """Skeletonize a uint8 (p × 255) probability array covering ``region`` of ``grid``.

    ``source`` is a numpy array or the URI of a probability OME-Zarr (array ``0``);
    ``mask_fn(lo, hi)`` optionally returns the voxels of a box to keep (e.g. inside a region).
    """
    t0 = time.time()
    scale = np.asarray(grid.scale, dtype=np.float64)
    shape = np.asarray(_open(source).shape[-3:])
    parallel = block.workers > 1 and isinstance(source, str)
    task_mask = "worker" if parallel and mask_fn is not None else mask_fn
    tasks = [
        (source, lo, hi, cl, ch, scale, config, task_mask)
        for lo, hi, cl, ch in _blocks(shape, block)
    ]
    log.info("Blockwise skeleton: %d blocks of %d³ (+%d halo) over %s voxels",
             len(tasks), block.block_vox, block.halo_vox, shape.tolist())  # fmt: skip

    # pass 1: local thinning and bridging, in parallel
    parts = []
    if parallel:
        pool_args = dict(initializer=_init_worker, initargs=(mask_fn,))
        with ProcessPoolExecutor(block.workers, **pool_args) as pool:
            for k, part in enumerate(pool.map(_local_pass, tasks, chunksize=1)):
                if part is not None:
                    parts.append(part)
                if (k + 1) % 50 == 0:
                    log.info(
                        "  local pass %d/%d blocks · %.0f s", k + 1, len(tasks), time.time() - t0
                    )
    else:
        parts = [p for p in map(_local_pass, tasks) if p is not None]
    if not parts:
        empty = np.zeros((0, 3))
        return AxonSkeleton(np.zeros((1, 1, 1), bool), empty, np.zeros((0, 2), np.int64),
                            config, np.zeros(0, bool))  # fmt: skip

    # stitch: cut edges attach to the neighbouring block's node at (or next to) that voxel
    offsets = np.cumsum([0] + [len(p["voxel"]) for p in parts])
    voxel = np.concatenate([p["voxel"] for p in parts]).astype(np.int64)
    filled = np.concatenate([p["filled"] for p in parts])
    edges = [p["edges"] + o for p, o in zip(parts, offsets[:-1], strict=True)]
    cut_from = np.concatenate([p["cut_from"] + o for p, o in zip(parts, offsets[:-1], strict=True)])
    cut_to = np.concatenate([p["cut_to"] for p in parts]).astype(np.int64)
    if len(cut_from):
        reach = block.stitch_vox * scale.max()
        dist, target = cKDTree(voxel * scale).query(cut_to * scale, distance_upper_bound=reach)
        ok = np.isfinite(dist) & (target != cut_from)
        pairs = np.sort(np.stack([cut_from[ok], target[ok]], axis=1), axis=1)
        edges.append(np.unique(pairs, axis=0))
        log.info("Stitched %d of %d cut edges across blocks", int(ok.sum()), len(cut_from))
    edges = np.concatenate(edges).astype(np.int64)
    if len(edges):
        edges = np.unique(np.sort(edges, axis=1), axis=0)

    # pass 2: piece-level filters on the whole graph
    voxel, edges, filled = filter_pieces(voxel, edges, filled, scale, config)
    log.info("After piece filters: %d nodes · %.0f s", len(voxel), time.time() - t0)

    # pass 3: connect, block by block, with global pieces and a shared union-find
    if config.connect_um > 0 and len(voxel):
        n = len(voxel)
        graph = _adjacency(edges, n)
        _, piece = connected_components(graph, directed=False)
        degree = np.asarray(graph.sum(axis=1)).ravel()
        root = np.arange(piece.max() + 1)
        new_vox, new_edges, next_id = [], [], n
        order = np.lexsort(voxel.T[::-1])
        sorted_vox = voxel[order]
        for lo, hi, core_lo, core_hi in _blocks(shape, block):
            # nodes in the padded block (binary search on the leading axis, then mask)
            a = np.searchsorted(sorted_vox[:, 0], lo[0])
            b = np.searchsorted(sorted_vox[:, 0], hi[0])
            idx = order[a:b]
            idx = idx[np.all((voxel[idx] >= lo) & (voxel[idx] < hi), axis=1)]
            core = np.all((voxel[idx] >= core_lo) & (voxel[idx] < core_hi), axis=1)
            if not (core & (degree[idx] == 1)).any():
                continue
            local = np.full(n, -1, dtype=np.int64)
            local[idx] = np.arange(len(idx))
            e = local[edges]
            e = e[(e >= 0).all(axis=1)]
            probability = _read(source, lo, hi, mask_fn)
            v2, e2, _ = _join_pieces(
                voxel[idx] - lo, e, np.zeros(len(idx), bool), probability, scale,
                gap_um=config.connect_um, min_cos=config.connect_min_cos, mutual=False,
                min_p=config.connect_min_p, piece=piece[idx], root=root,
                tip_mask=core, degree=degree[idx],
            )  # fmt: skip
            if len(v2) > len(idx) or len(e2) > len(e):
                added = e2[len(e) :]
                ids = np.r_[idx, np.arange(next_id, next_id + len(v2) - len(idx))]
                new_vox.append(v2[len(idx) :] + lo)
                new_edges.append(ids[added])
                next_id += len(v2) - len(idx)
        if new_edges:
            voxel = np.concatenate([voxel, *new_vox])
            filled = np.r_[filled, np.ones(next_id - n, dtype=bool)]
            edges = np.concatenate([edges, *new_edges])
        log.info("Connected %d gaps across the volume · %.0f s", len(new_edges), time.time() - t0)

    origin = np.asarray(region.start) * scale + np.asarray(grid.translation)
    result = AxonSkeleton(np.zeros((1, 1, 1), bool), voxel * scale + origin, edges, config, filled)
    log.info("Blockwise skeleton: %d nodes, %.1f mm (%.1f mm gap-filled) · %.0f s", len(voxel),
             result.length_um / 1e3, result.filled_length_um / 1e3, time.time() - t0)  # fmt: skip
    return result
