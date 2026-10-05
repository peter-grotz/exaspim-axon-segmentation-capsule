"""Follow axons out of a region: segment cubes ahead of every axon leaving it, round after round.

Starting from a CCF region (grown as for ``run``), each round

1. finds the *exits*: end points of followed axon-candidate cable that point out of the area
   segmented so far,
2. predicts a cube of side ``cube_um`` ahead of each exit (one fixed whole-image output grid, so
   rounds only add tiles),
3. skeletonizes the new cubes (outside the area done before) and stitches their cable onto the
   skeleton where it meets the old cut ends,
4. labels the new cable (dendrites of detected somata vs axon candidates) and extends the
   followed set through axon-candidate cable connected to it.

An axon stops being followed when it reaches a detected soma (traced back to its cell), runs
out of segmented cable, or exceeds ``budget_um`` of path from the region. Caps on rounds, cubes
and minutes bound one run.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.config import InferenceConfig
from exaspim_axon_segmentation.inference.normalize import estimate_scaling
from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR
from exaspim_axon_segmentation.inference.predict import (
    TilePredictor,
    predict_tiles,
    resolve_device,
    resolve_precision,
)
from exaspim_axon_segmentation.inference.tiling import Tile
from exaspim_axon_segmentation.io import Box, CcfOntology, ExaspimAsset, RegionRoi
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.model import load_model
from exaspim_axon_segmentation.model.unet import CONTEXT
from exaspim_axon_segmentation.postprocess.dendrites import DendriteConfig, label_dendrites
from exaspim_axon_segmentation.postprocess.skeleton import (
    AxonSkeleton,
    SkeletonConfig,
    skeletonize_probability,
)
from exaspim_axon_segmentation.postprocess.somata import load_somata

log = logging.getLogger(__name__)

AXON, DENDRITE = 2, 3


@dataclass(frozen=True)
class ExpandConfig:
    """Settings of the axon-following loop."""

    cube_um: float = 800.0
    behind_um: float = 80.0  # a cube starts this far behind its exit, so cut ends overlap
    exit_um: float = 40.0  # an end point is an exit if this far ahead lies outside the done area
    join_um: float = 12.0  # stitch new cable to old cable this close
    budget_um: float = 100_000.0  # path length followed from the region, per axon
    soma_um: float = 50.0
    max_rounds: int = 40
    max_cubes: int = 800
    max_minutes: float = 600.0


@dataclass
class Graph:
    """The stitched skeleton: nodes (z, y, x µm), edges, SWC types, round each node was added."""

    coords: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    edges: NDArray[np.int64] = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    types: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    round: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n(self) -> int:
        """Node count."""
        return len(self.coords)

    def add(self, skeleton: AxonSkeleton, round_: int) -> NDArray[np.int64]:
        """Append a skeleton; returns the new node indices."""
        offset = self.n
        self.coords = np.concatenate([self.coords, skeleton.coordinates_um])
        self.edges = np.concatenate([self.edges, skeleton.edges + offset])
        types = skeleton.node_types if skeleton.node_types is not None else None
        types = np.full(len(skeleton.coordinates_um), AXON) if types is None else types
        self.types = np.concatenate([self.types, types])
        self.round = np.concatenate([self.round, np.full(len(types), round_)])
        return np.arange(offset, self.n)

    def adjacency(self, mask: NDArray[np.bool_] | None = None):
        """Symmetric edge-length adjacency, optionally of the nodes in ``mask`` only."""
        e = self.edges
        if mask is not None:
            e = e[mask[e[:, 0]] & mask[e[:, 1]]]
        w = np.maximum(np.linalg.norm(self.coords[e[:, 0]] - self.coords[e[:, 1]], axis=1), 1e-3)
        return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(self.n, self.n)).tocsr()  # fmt: skip


class DoneArea:
    """Union of the grown region and every predicted cube; answers "already segmented?"."""

    def __init__(self, roi: RegionRoi):
        self.roi = roi
        self.centres: list[NDArray] = []
        self.halves: list[float] = []

    def add_cube(self, centre: NDArray, side_um: float) -> None:
        """Record a predicted cube."""
        self.centres.append(np.asarray(centre, dtype=np.float64))
        self.halves.append(side_um / 2)

    def mask_on(self, grid, box: Box, *, n_cubes: int) -> NDArray[np.bool_]:
        """Done area on a box of a level grid (region mask plus the first ``n_cubes`` cubes)."""
        mask = self.roi.sample_on(grid, box)
        for centre, half in zip(self.centres[:n_cubes], self.halves[:n_cubes], strict=True):
            lo = np.ceil((centre - half - np.asarray(grid.translation)) / np.asarray(grid.scale))
            hi = np.floor((centre + half - np.asarray(grid.translation)) / np.asarray(grid.scale))
            a = np.clip(lo.astype(int) - np.asarray(box.start), 0, box.shape)
            b = np.clip(hi.astype(int) + 1 - np.asarray(box.start), 0, box.shape)
            if np.all(b > a):
                mask[tuple(slice(i, j) for i, j in zip(a, b, strict=True))] = True
        return mask

    def contains(self, points: NDArray, *, n_cubes: int | None = None) -> NDArray[np.bool_]:
        """Return where points lie in the region or the first ``n_cubes`` cubes (default all)."""
        points = np.atleast_2d(points)
        tf, box = self.roi.mask_transform, self.roi.mask_box
        idx = np.floor(tf.to_index(points) + 0.5).astype(int) - np.asarray(box.start)
        ok = np.all((idx >= 0) & (idx < self.roi.mask.shape), axis=1)
        inside = np.zeros(len(points), dtype=bool)
        inside[ok] = self.roi.mask[tuple(idx[ok].T)]
        k = len(self.centres) if n_cubes is None else n_cubes
        if k:
            centres = np.asarray(self.centres[:k])
            halves = np.asarray(self.halves[:k])
            d, j = cKDTree(centres).query(points, k=min(8, k), p=np.inf)
            d, j = d.reshape(len(points), -1), j.reshape(len(points), -1)
            inside |= (d <= halves[np.minimum(j, k - 1)]).any(axis=1)
        return inside


def follow_axons(
    asset_uri: str,
    regions: list[str],
    out_dir: Path,
    inference: InferenceConfig,
    config: ExpandConfig = ExpandConfig(),  # noqa: B008
    *,
    annotation_asset: str | None = None,
) -> dict:
    """Run the loop for one asset; writes skeletons, a per-round log and a summary to out_dir."""
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    asset = ExaspimAsset(asset_uri, annotation_root=annotation_asset)
    fused, level = asset.fused, inference.level
    grid = fused.transforms[level]
    full = Box.full(fused.shape(level))
    atlas = RegionRoi.from_annotation(asset.annotation, CcfOntology.load().resolve(regions))
    roi = atlas.grown(inference.region_growth)
    done = DoneArea(roi)
    try:
        somata = np.array([s.center_um for s in load_somata(asset)]).reshape(-1, 3)
    except FileNotFoundError:
        somata = np.zeros((0, 3))
    soma_tree = cKDTree(somata) if len(somata) else None
    log.info("%s: %.1f mm³ region, %d detected somata", asset.name, roi.volume_mm3, len(somata))

    # one probability output on the whole-image grid; rounds add tiles to it
    scaling = estimate_scaling(inference.normalization, fused, level, roi)
    predictor = TilePredictor(load_model(inference.weights), resolve_device(inference.device),
                              resolve_precision(inference.precision))  # fmt: skip
    predictor.self_check(inference.batch_size)
    prob_uri = str(out_dir / "probability.zarr")
    out = create_ngff_array(prob_uri, source=grid, region=full,
                            chunks=(inference.tile_output,) * 3, dtype="uint8",
                            name="axon_probability",
                            attrs={PROVENANCE_ATTR: {"asset": asset.root, "level": level,
                                                     "scaling": scaling.as_dict()}})  # fmt: skip
    group = zarr.open_group(prob_uri, mode="a")

    def predict(boxes: list[Box]) -> None:
        tiles = {t.chunk: t for b in boxes for t in _tiles_covering(b, full, inference.tile_output)}
        predict_tiles(fused, level, list(tiles.values()), full, scaling, predictor, out, group,
                      batch_size=inference.batch_size, io_threads=inference.io_threads)  # fmt: skip

    def skeletonize(box: Box, keep: NDArray[np.bool_] | None, relaxed: bool) -> AxonSkeleton:
        probability = np.asarray(out[box.slices])
        if keep is not None:
            probability[~keep] = 0
        cfg = SkeletonConfig(anchor_um=0, min_piece_um=30.0) if relaxed else SkeletonConfig()
        skeleton = skeletonize_probability(probability, grid, box, cfg)
        if len(skeleton.coordinates_um) and len(somata):
            dendrite = label_dendrites(skeleton.coordinates_um, skeleton.edges, somata,
                                       DendriteConfig())  # fmt: skip
            types = np.where(dendrite, DENDRITE, AXON)
        else:
            types = np.full(len(skeleton.coordinates_um), AXON)
        return AxonSkeleton(skeleton.voxels, skeleton.coordinates_um, skeleton.edges, cfg,
                            skeleton.filled, types)  # fmt: skip

    # round 0: the region itself
    region_box = roi.box_in(fused, level, pad_um=inference.mask_dilation_um)
    predict(_boxes_touching(roi, grid, region_box, inference.tile_output))
    graph = Graph()
    graph.add(skeletonize(region_box, roi.sample_on(grid, region_box), relaxed=False), 0)
    followed = graph.types == AXON
    reached: dict[int, dict] = {}
    rounds = []
    stop = "no exits"
    for k in range(1, config.max_rounds + 1):
        dist = _path_from_region(graph, followed, done, roi)
        exits = _exits(graph, followed, done, config, dist)
        if soma_tree is not None and len(exits):
            near = soma_tree.query(graph.coords[exits[:, 0].astype(int)])[0] < config.soma_um
            exits = exits[~near]
        if not len(exits):
            stop = "no exits"
            break
        if len(done.centres) >= config.max_cubes:
            stop = "cube cap"
            break
        if (time.time() - started) / 60 > config.max_minutes:
            stop = "time cap"
            break
        n_before = len(done.centres)
        cubes = []
        for row in exits[np.argsort(dist[exits[:, 0].astype(int)])]:
            tip, direction = graph.coords[int(row[0])], row[1:4]
            centre = tip + direction * (config.cube_um / 2 - config.behind_um)
            if len(cubes) and done.contains(tip + direction * config.cube_um / 4)[0]:
                continue  # an earlier cube of this round already covers this exit
            if len(done.centres) >= config.max_cubes:
                break
            done.add_cube(centre, config.cube_um)
            cubes.append(centre)
        half = config.cube_um / 2
        boxes = [grid.box_covering(c - half, c + half, fused.shape(level)) for c in cubes]
        predict(boxes)
        new_nodes = 0
        for i, box in enumerate(boxes):
            before = done.mask_on(grid, box, n_cubes=n_before + i)
            skeleton = skeletonize(box, ~before, relaxed=True)
            if not len(skeleton.coordinates_um):
                continue
            added = graph.add(skeleton, k)
            _stitch(graph, added, config.join_um)
            new_nodes += len(added)
        followed = _extend_followed(graph, followed)
        if soma_tree is not None:
            # a followed axon reaches a cell where it touches that cell's soma or the dendrite
            # cable labelled as connected to it
            e = graph.edges
            touch = np.r_[e[followed[e[:, 0]] & (graph.types[e[:, 1]] == DENDRITE), 1],
                          e[followed[e[:, 1]] & (graph.types[e[:, 0]] == DENDRITE), 0]]  # fmt: skip
            near = np.flatnonzero(followed)
            near = near[soma_tree.query(graph.coords[near])[0] < config.soma_um]
            for node in np.unique(np.r_[touch, near]).astype(int):
                d, soma = soma_tree.query(graph.coords[node])
                if d <= DendriteConfig().connected_um:
                    reached.setdefault(int(soma), {"soma_um": somata[soma].round(1).tolist(),
                                                   "round": k, "node": int(node),
                                                   "distance_um": round(float(d), 1)})  # fmt: skip
        mm = _length(graph, followed) / 1e3
        rounds.append({"round": k, "exits": len(exits), "cubes": len(cubes),
                       "new_nodes": int(new_nodes), "followed_mm": round(mm, 1),
                       "somata_reached": len(reached),
                       "minutes": round((time.time() - started) / 60, 1)})  # fmt: skip
        log.info("round %d: %d exits -> %d cubes, %.0f mm followed, %d somata reached",
                 k, len(exits), len(cubes), mm, len(reached))  # fmt: skip
        (out_dir / "rounds.json").write_text(json.dumps(rounds, indent=1))
    else:
        stop = "round cap"

    _write(graph, followed, out_dir, asset.subject_id)
    summary = {
        "asset": asset.root, "regions": regions, "config": asdict(config), "stop": stop,
        "rounds": rounds, "cubes": [c.round(1).tolist() for c in done.centres],
        "cube_um": config.cube_um, "followed_mm": round(_length(graph, followed) / 1e3, 1),
        "somata_reached": list(reached.values()), "detected_somata": len(somata),
        "minutes": round((time.time() - started) / 60, 1),
    }  # fmt: skip
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("stopped (%s) after %d rounds: %.0f mm followed, %d somata reached",
             stop, len(rounds), summary["followed_mm"], len(reached))  # fmt: skip
    return summary


def _tiles_covering(box: Box, full: Box, side: int) -> list[Tile]:
    """Tiles of the whole-image grid whose output overlaps box."""
    lo = np.maximum(np.asarray(box.start) - np.asarray(full.start), 0) // side
    hi = -(-np.minimum(np.asarray(box.stop), full.stop) // side)
    tiles = []
    for chunk in np.ndindex(*(hi - lo)):
        c = lo + np.asarray(chunk)
        start = np.asarray(full.start) + c * side
        out_box = Box.from_arrays(start, start + side)
        tiles.append(Tile(chunk=tuple(int(v) for v in c), out_box=out_box,
                          in_box=out_box.pad(CONTEXT)))  # fmt: skip
    return tiles


def _boxes_touching(roi: RegionRoi, grid, box: Box, side: int) -> list[Box]:
    """Tile-sized sub-boxes of box that touch the region (so round 0 predicts only those)."""
    boxes = []
    counts = [-(-s // side) for s in box.shape]
    for chunk in np.ndindex(*counts):
        start = np.asarray(box.start) + np.asarray(chunk) * side
        sub = Box.from_arrays(start, np.minimum(start + side, box.stop))
        if roi.fraction_inside(grid, sub, dilate_um=50.0) > 0:
            boxes.append(sub)
    return boxes


def _stitch(graph: Graph, added: NDArray[np.int64], join_um: float) -> None:
    """Join each new end point (and each old end point) to the nearest node across, if close."""
    new = np.zeros(graph.n, dtype=bool)
    new[added] = True
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
    joins = []
    for side, other in ((new, ~new), (~new, new)):
        tips = np.flatnonzero(side & (degree <= 1))
        targets = np.flatnonzero(other)
        if not len(tips) or not len(targets):
            continue
        d, j = cKDTree(graph.coords[targets]).query(
            graph.coords[tips], distance_upper_bound=join_um
        )
        ok = np.isfinite(d)
        joins += list(zip(tips[ok], targets[j[ok]], strict=True))
    if joins:
        pairs = np.unique(np.sort(np.asarray(joins), axis=1), axis=0)
        graph.edges = np.concatenate([graph.edges, pairs])


def _extend_followed(graph: Graph, followed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Followed = axon-candidate cable connected (through axon candidates) to followed cable."""
    axon = graph.types == AXON
    _, piece = connected_components(graph.adjacency(axon), directed=False)
    grow = np.zeros(piece.max() + 1, dtype=bool)
    old = np.zeros(graph.n, dtype=bool)
    old[: len(followed)] = followed
    grow[piece[old & axon]] = True
    return axon & grow[piece]


def _path_from_region(graph, followed, done, roi) -> NDArray[np.float64]:
    """Path length (µm) along followed cable from the region, per node (inf if unreachable)."""
    inside = np.zeros(graph.n, dtype=bool)
    tf, box = roi.mask_transform, roi.mask_box
    idx = np.floor(tf.to_index(graph.coords) + 0.5).astype(int) - np.asarray(box.start)
    ok = np.all((idx >= 0) & (idx < roi.mask.shape), axis=1)
    inside[ok] = roi.mask[tuple(idx[ok].T)]
    seeds = np.flatnonzero(inside & followed)
    if not len(seeds):
        return np.full(graph.n, np.inf)
    return dijkstra(graph.adjacency(followed), directed=False, indices=seeds, min_only=True)


def _exits(graph, followed, done, config, dist) -> NDArray[np.float64]:
    """Rows (node, direction z, y, x) for followed end points pointing out of the done area."""
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
    tips = np.flatnonzero(followed & (degree == 1) & (dist < config.budget_um))
    if not len(tips):
        return np.zeros((0, 4))
    adj = graph.adjacency(followed)
    rows = []
    for tip in tips:
        prev, cur = -1, tip
        for _ in range(4):  # walk ~12 µm back along the cable
            nbrs = [v for v in adj.indices[adj.indptr[cur] : adj.indptr[cur + 1]] if v != prev]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
        v = graph.coords[tip] - graph.coords[cur]
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        direction = v / norm
        if not done.contains(graph.coords[tip] + direction * config.exit_um)[0]:
            rows.append([tip, *direction])
    return np.asarray(rows).reshape(-1, 4)


def _length(graph: Graph, mask: NDArray[np.bool_]) -> float:
    e = graph.edges[mask[graph.edges[:, 0]] & mask[graph.edges[:, 1]]]
    return float(np.linalg.norm(graph.coords[e[:, 0]] - graph.coords[e[:, 1]], axis=1).sum())


def _write(graph: Graph, followed: NDArray[np.bool_], out_dir: Path, subject: str) -> None:
    """Whole stitched skeleton, and the followed axons (one SWC per connected piece)."""
    blank = np.zeros(graph.n, dtype=bool)
    skeleton = AxonSkeleton(np.zeros((1, 1, 1), dtype=bool), graph.coords, graph.edges,
                            SkeletonConfig(), blank, graph.types)  # fmt: skip
    skeleton.write_swc(out_dir / "skeleton.swc")
    axons = skeleton.subset(followed)
    axons.write_swc(out_dir / "followed_axons.swc")
    axons.write_pieces(out_dir / "followed_swc", f"{subject}_followed")
    np.savez_compressed(out_dir / "graph.npz", coords=graph.coords, edges=graph.edges,
                        types=graph.types, round=graph.round, followed=followed)  # fmt: skip
