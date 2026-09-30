"""Skeletonize a probability map into an axon graph with physical lengths.

Pipeline: threshold → hysteresis (keep components with a confident core) → drop specks →
per-component 3D thinning → branch graph (``skan``) on the anisotropic grid → prune short spurs
(thinning artefacts) → bridge aligned gaps → keep long, non-isolated cable → connect the cable
along the probability map → SWC forest + summary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cc3d
import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import cKDTree
from skan import Skeleton, summarize
from skimage.morphology import skeletonize

from exaspim_axon_segmentation import __version__
from exaspim_axon_segmentation.io import Box, LevelTransform
from exaspim_axon_segmentation.train.swc import Swc, write_swc

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkeletonConfig:
    """Skeletonization settings.

    Attributes
    ----------
    threshold : float
        Probability defining the neurite mask. 0.2 maximized trace recall on 822178 MD (held-out
        slab: 0.79 at 0.88 precision near traces, vs 0.69 / 0.91 at 0.3).
    core : float
        A component is kept only if some voxel exceeds this probability.
    min_volume_um3 : float
        Components smaller than this are dropped before thinning.
    spur_um : float
        Terminal branches shorter than this are pruned (repeated ``prune_rounds`` times).
    prune_rounds : int
        Pruning passes; each pass can expose new short terminals.
    bridge_um : float
        Before length filtering, reconnect skeleton tips from different pieces that are within
        this distance and point toward each other (both tip directions within
        ``arccos(bridge_min_cos)`` of the gap), so a real axon broken where it dims is measured as
        one cable. Set 0 to disable.
    bridge_min_cos : float
        Alignment required for a bridge (0.7 ≈ within 45°).
    min_piece_um : float
        Minimum cable length: connected pieces (after bridging) shorter than this are dropped.
        On 822178 MD's traced lobe, 30 µm bridging + 100 µm minimum cut pieces 27.5k → 107 and
        off-trace skeleton 200 → 29 mm, at held-out trace recall 0.92 (0.93 unfiltered) and
        precision 0.92 near traces.
    anchor_um : float
        Isolation filter: pieces at least this long are *anchor* cable and always kept; shorter
        pieces (still ≥ ``min_piece_um``) are kept only within ``attach_um`` of an anchor. Broken
        branches of a real arbor sit next to it, noise fragments sit alone: on 822178 MD's traced
        lobe every on-trace piece was ≤ 32 µm from an anchor, while off-trace pieces had a median
        distance of 320 µm. Set 0 to disable.
    attach_um : float
        How close a shorter piece must come to anchor cable to be kept.
    connect_um : float
        Final gap filling on the kept cable: each tip joins the nearest node of another piece
        within this distance and within ``arccos(connect_min_cos)`` of the tip's direction (end to
        end, or end onto the side of a branch), along the brightest path through the probability
        map. Set 0 to disable.
    connect_min_cos : float
        Direction tolerance of a connection (0.5 ≈ within 60° of the tip's direction).
    connect_min_p : float
        A connection is rejected if any voxel along its path is below this probability, i.e. if it
        crosses background rather than a dim stretch of neurite. On 822178 MD, joins that
        disagreed with the traces mostly crossed p < 0.05 or detoured (> 1.3 × the gap).
    speckle_max_um : float
        Speckle removal: connected pieces confined to a single z-plane *and* shorter than this are
        dropped. In-plane speckle (median 4 µm; 80k of 146k pieces on 822178 MD, none ≥ 60 µm) is
        as bright and as confident as real axons but does not cross planes. Long pieces are kept
        even when single-plane, because an axon running parallel to the image plane thins to a
        one-plane centerline.
    """

    threshold: float = 0.2
    core: float = 0.5
    min_volume_um3: float = 100.0
    spur_um: float = 10.0
    prune_rounds: int = 2
    bridge_um: float = 30.0
    bridge_min_cos: float = 0.7
    min_piece_um: float = 100.0
    anchor_um: float = 500.0
    attach_um: float = 50.0
    connect_um: float = 50.0
    connect_min_cos: float = 0.5
    connect_min_p: float = 0.05
    speckle_max_um: float = 60.0


@dataclass(frozen=True, eq=False)
class AxonSkeleton:
    """Skeleton voxels and graph of a probability map.

    Attributes
    ----------
    voxels : numpy.ndarray of bool
        Skeleton voxels on the probability map's grid (region-local).
    coordinates_um : numpy.ndarray, shape (n, 3)
        Physical (z, y, x) position of every skeleton node.
    edges : numpy.ndarray, shape (m, 2)
        Node index pairs of the skeleton graph.
    config : SkeletonConfig
        Settings used.
    filled : numpy.ndarray of bool, shape (n,)
        Nodes added by gap bridging / connection rather than thinning.
    """

    voxels: NDArray[np.bool_]
    coordinates_um: NDArray[np.float64]
    edges: NDArray[np.int64]
    config: SkeletonConfig
    filled: NDArray[np.bool_]

    @property
    def filled_length_um(self) -> float:
        """Length of edges touching gap-filled nodes."""
        touches = self.filled[self.edges[:, 0]] | self.filled[self.edges[:, 1]]
        return float(self.edge_lengths_um[touches].sum())

    @property
    def edge_lengths_um(self) -> NDArray[np.float64]:
        """Physical length of every edge."""
        a, b = self.coordinates_um[self.edges[:, 0]], self.coordinates_um[self.edges[:, 1]]
        return np.linalg.norm(a - b, axis=1)

    @property
    def length_um(self) -> float:
        """Total skeleton length."""
        return float(self.edge_lengths_um.sum())

    def length_inside_um(
        self, inside: NDArray[np.bool_], grid: LevelTransform, region: Box
    ) -> float:
        """Length of edges whose both endpoints fall inside a mask on the same grid."""
        index = np.floor(
            (self.coordinates_um - np.asarray(grid.translation)) / np.asarray(grid.scale) + 0.5
        ).astype(int) - np.asarray(region.start)
        index = np.clip(index, 0, np.asarray(inside.shape) - 1)
        node_in = inside[tuple(index.T)]
        keep = node_in[self.edges[:, 0]] & node_in[self.edges[:, 1]]
        return float(self.edge_lengths_um[keep].sum())

    def pieces(self) -> list[tuple[NDArray[np.int64], float]]:
        """Node indices and length of every connected piece, longest first."""
        n = len(self.coordinates_um)
        if n == 0:
            return []
        _, part = connected_components(_adjacency(self.edges, n), directed=False)
        n_parts = part.max() + 1
        length = np.bincount(
            part[self.edges[:, 0]], weights=self.edge_lengths_um, minlength=n_parts
        )
        members = np.split(np.argsort(part, kind="stable"), np.cumsum(np.bincount(part))[:-1])
        return [(members[p], float(length[p])) for p in np.argsort(-length, kind="stable")]

    def to_swc(self, nodes: NDArray[np.int64] | None = None) -> Swc:
        """SWC forest of the given nodes (default: all), one root per connected piece.

        Nodes are ordered breadth-first from an end point of each piece, so every parent precedes
        its children and ids run 1..n. Node type is 0 (undefined) and radius 1.0, as Horta writes
        them; the network does not distinguish axon from dendrite yet.
        """
        nodes = np.arange(len(self.coordinates_um)) if nodes is None else np.asarray(nodes)
        local = np.full(len(self.coordinates_um), -1, dtype=np.int64)
        local[nodes] = np.arange(len(nodes))
        keep = (local[self.edges] >= 0).all(axis=1)
        edges = local[self.edges[keep]]
        graph = _adjacency(edges, len(nodes))
        n_parts, part = connected_components(graph, directed=False)
        parent = np.full(len(nodes), -1, dtype=np.int64)
        order = []
        degree = np.asarray(graph.sum(axis=1)).ravel()
        for p in range(n_parts):
            members = np.flatnonzero(part == p)
            root = members[np.argmin(degree[members])]  # an end point where there is one
            visit, predecessors = breadth_first_order(graph, root, directed=False)
            order.extend(visit.tolist())
            parent[visit] = predecessors[visit]
        order = np.asarray(order, dtype=np.int64)
        ids = np.empty(len(nodes), dtype=np.int64)
        ids[order] = np.arange(1, len(nodes) + 1)
        parents = np.where(parent[order] >= 0, ids[np.maximum(parent[order], 0)], -1)
        return Swc(
            ids=ids[order],
            types=np.zeros(len(nodes), dtype=np.int64),
            xyz_um=self.coordinates_um[nodes[order]][:, ::-1],
            radius=np.ones(len(nodes)),
            parents=parents,
            header=(f"# ORIGINAL_SOURCE exaspim-axon-segmentation {__version__}",),
        )

    def write_swc(self, path: Path) -> None:
        """Write the whole skeleton as one SWC forest (Horta format, see :meth:`write_pieces`)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        swc = self.to_swc()
        write_swc(swc, path, offset_um=np.zeros(3) if not len(swc.ids) else None)

    def write_pieces(self, directory: Path, prefix: str) -> list[dict]:
        """Write one SWC per connected piece, longest first: ``<prefix>_0001.swc``, ...

        Files follow the Janelia Workstation / HortaCloud export: ``# ORIGINAL_SOURCE``,
        ``# OFFSET x y z`` (the piece's first node, µm) and ``# COLOR r,g,b`` headers, then
        ``id type x y z radius parent`` rows in micrometers relative to the offset, in the physical
        frame of ``fused.zarr``. Returns one record (file, nodes, length) per piece.
        """
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for k, (nodes, length) in enumerate(self.pieces(), start=1):
            name = f"{prefix}_{k:04d}.swc"
            write_swc(
                self.to_swc(nodes),
                directory / name,
                color=PIECE_COLORS[(k - 1) % len(PIECE_COLORS)],
            )
            records.append({"file": name, "nodes": len(nodes), "length_um": length})
        return records


PIECE_COLORS = (  # colour of each piece's SWC (and QC figure), cycling by length rank
    (0.122, 0.467, 0.706), (1.0, 0.498, 0.055), (0.173, 0.627, 0.173), (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741), (0.549, 0.337, 0.294), (0.890, 0.467, 0.761), (0.090, 0.745, 0.812),
    (0.737, 0.741, 0.133), (1.0, 1.0, 1.0),
)  # fmt: skip


def skeletonize_probability(
    probability: NDArray[np.uint8], grid: LevelTransform, region: Box, config: SkeletonConfig
) -> AxonSkeleton:
    """Skeletonize a uint8 (p × 255) probability map covering ``region`` of ``grid``."""
    scale = np.asarray(grid.scale)
    mask = probability > round(config.threshold * 255)
    labels, n = cc3d.connected_components(mask, connectivity=26, return_N=True, out_dtype=np.uint32)
    stats = cc3d.statistics(labels)
    core = np.zeros(n + 1, dtype=bool)
    core[np.unique(labels[probability > round(config.core * 255)])] = True
    min_voxels = config.min_volume_um3 / float(np.prod(scale))

    voxels = np.zeros(mask.shape, dtype=bool)
    kept = 0
    for k in range(1, n + 1):
        if not core[k] or stats["voxel_counts"][k] < min_voxels:
            continue
        bbox = stats["bounding_boxes"][k]
        voxels[bbox] |= skeletonize(labels[bbox] == k)
        kept += 1
    log.info("Skeletonized %d of %d components (p > %.2f)", kept, n, config.threshold)

    skeleton = Skeleton(voxels, spacing=scale)
    for _ in range(config.prune_rounds):
        branches = summarize(skeleton, separator="_")
        spurs = branches.index[
            (branches["branch_type"] == 1) & (branches["branch_distance"] < config.spur_um)
        ].to_numpy()
        if spurs.size == 0:
            break
        skeleton = skeleton.prune_paths(spurs)
    origin = np.asarray(region.start) * scale + np.asarray(grid.translation)
    voxel = np.round(skeleton.coordinates).astype(np.int64)
    graph = skeleton.graph.tocoo()
    upper = graph.row < graph.col
    edges = np.stack([graph.row[upper], graph.col[upper]], axis=1).astype(np.int64)
    filled = np.zeros(len(voxel), dtype=bool)

    # Bridge aligned tip-to-tip gaps so an axon broken where it dims is measured as one cable.
    voxel, edges, filled = _join_pieces(
        voxel, edges, filled, probability, scale,
        gap_um=config.bridge_um, min_cos=config.bridge_min_cos, mutual=True, min_p=0.0,
    )  # fmt: skip
    _, edges, keep = _drop_small_pieces(
        voxel * scale, edges, config.min_piece_um, config.speckle_max_um, float(scale[0])
    )
    voxel, filled = voxel[keep], filled[keep]
    _, edges, keep = _drop_isolated_pieces(voxel * scale, edges, config.anchor_um, config.attach_um)
    voxel, filled = voxel[keep], filled[keep]
    # Connect what is left: ends onto the continuing cable, along the probability map.
    voxel, edges, filled = _join_pieces(
        voxel, edges, filled, probability, scale, gap_um=config.connect_um,
        min_cos=config.connect_min_cos, mutual=False, min_p=config.connect_min_p,
    )  # fmt: skip

    final = np.zeros(mask.shape, dtype=bool)
    final[tuple(voxel.T)] = True
    result = AxonSkeleton(final, voxel * scale + origin, edges, config, filled)
    log.info(
        "Skeleton: %d nodes, %.1f mm total length (%.1f mm gap-filled)",
        len(voxel), result.length_um / 1e3, result.filled_length_um / 1e3,
    )  # fmt: skip
    return result


def _adjacency(edges: NDArray[np.int64], n: int):
    from scipy.sparse import coo_matrix

    data = np.ones(len(edges) * 2, dtype=np.int8)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _drop_small_pieces(
    coordinates: NDArray[np.float64],
    edges: NDArray[np.int64],
    min_um: float,
    speckle_max_um: float,
    z_step_um: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Remove short pieces (``min_um``) and short single-plane speckle (``speckle_max_um``).

    Returns kept coordinates, re-indexed edges, and the kept node indices.
    """
    n = len(coordinates)
    if n == 0 or (min_um <= 0 and speckle_max_um <= 0):
        return coordinates, edges, np.arange(n)
    _, piece = connected_components(_adjacency(edges, n), directed=False)
    m = piece.max() + 1
    lengths = np.linalg.norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1)
    piece_length = np.bincount(piece[edges[:, 0]], weights=lengths, minlength=m)
    z_lo, z_hi = np.full(m, np.inf), np.full(m, -np.inf)
    np.minimum.at(z_lo, piece, coordinates[:, 0])
    np.maximum.at(z_hi, piece, coordinates[:, 0])
    planes = np.round((z_hi - z_lo) / z_step_um).astype(int) + 1
    speckle = (planes == 1) & (piece_length < speckle_max_um)
    good = (piece_length >= min_um) & ~speckle
    keep = np.flatnonzero(good[piece])
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    kept_edges = remap[edges]
    return coordinates[keep], kept_edges[(kept_edges >= 0).all(axis=1)], keep


def _drop_isolated_pieces(
    coordinates: NDArray[np.float64], edges: NDArray[np.int64], anchor_um: float, attach_um: float
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Keep anchor pieces (≥ ``anchor_um``) and pieces within ``attach_um`` of one.

    Returns kept coordinates, re-indexed edges, and the kept node indices.
    """
    n = len(coordinates)
    if n == 0 or anchor_um <= 0:
        return coordinates, edges, np.arange(n)
    _, piece = connected_components(_adjacency(edges, n), directed=False)
    lengths = np.linalg.norm(coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1)
    piece_length = np.bincount(piece[edges[:, 0]], weights=lengths, minlength=piece.max() + 1)
    anchor = piece_length >= anchor_um
    good = anchor.copy()
    if anchor.any():
        near, _ = cKDTree(coordinates[anchor[piece]]).query(
            coordinates, distance_upper_bound=attach_um
        )
        np.logical_or.at(good, piece, np.isfinite(near))
    log.info(
        "Kept %d anchor pieces (≥ %.0f µm) and %d attached; dropped %d isolated",
        anchor.sum(), anchor_um, (good & ~anchor).sum(), (~good).sum(),
    )  # fmt: skip
    keep = np.flatnonzero(good[piece])
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    kept_edges = remap[edges]
    return coordinates[keep], kept_edges[(kept_edges >= 0).all(axis=1)], keep


def _join_pieces(
    voxel: NDArray[np.int64],
    edges: NDArray[np.int64],
    filled: NDArray[np.bool_],
    probability: NDArray[np.uint8],
    scale: NDArray[np.float64],
    *,
    gap_um: float,
    min_cos: float,
    mutual: bool,
    min_p: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    """Join piece tips to nearby pieces along the brightest path through the probability map.

    A tip joins the nearest node of another piece within ``gap_um`` lying within
    ``arccos(min_cos)`` of the tip's direction; with ``mutual`` that node must itself be a tip
    pointing back. Candidates are taken shortest first, each tip joins at most once, and never
    into its own (possibly already joined) piece, so no cycles form. The join follows the
    minimal-cost path through ``1 / (p + 0.05)``; it is rejected if it detours (longer than
    1.3 × the gap plus one voxel) or passes a voxel below ``min_p``. Path voxels become new nodes.
    """
    n = len(voxel)
    if gap_um <= 0 or n == 0:
        return voxel, edges, filled
    um = voxel * scale
    graph = _adjacency(edges, n)
    _, piece = connected_components(graph, directed=False)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    tips = np.flatnonzero(degree == 1)
    if len(tips) == 0:
        return voxel, edges, filled
    direction = np.zeros((n, 3))
    direction[tips] = [_tip_direction(graph, um, tip) for tip in tips]

    neighbors = cKDTree(um).query_ball_point(um[tips], gap_um)
    counts = np.fromiter(map(len, neighbors), dtype=np.int64, count=len(tips))
    a = np.repeat(tips, counts)
    b = np.fromiter((j for near in neighbors for j in near), dtype=np.int64, count=counts.sum())
    a, b = a[piece[a] != piece[b]], b[piece[a] != piece[b]]
    gap = um[b] - um[a]
    distance = np.linalg.norm(gap, axis=1)
    unit = gap / np.maximum(distance, 1e-9)[:, None]
    ok = np.einsum("ij,ij->i", direction[a], unit) >= min_cos
    if mutual:
        ok &= (degree[b] == 1) & (np.einsum("ij,ij->i", direction[b], -unit) >= min_cos)
    a, b, distance = a[ok], b[ok], distance[ok]
    # the nearest node of each other piece, per tip; then all candidates shortest first
    order = np.lexsort((distance, piece[b], a))
    a, b, distance = a[order], b[order], distance[order]
    first = np.ones(len(a), dtype=bool)
    first[1:] = (a[1:] != a[:-1]) | (piece[b[1:]] != piece[b[:-1]])
    a, b, distance = a[first], b[first], distance[first]

    root = np.arange(piece.max() + 1)

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    used = np.zeros(n, dtype=bool)
    new_voxels, new_edges = [], []
    next_id, rejected = n, 0
    for k in np.argsort(distance, kind="stable"):
        s, t = a[k], b[k]
        if used[s] or used[t]:
            continue
        ps, pt = find(piece[s]), find(piece[t])
        if ps == pt:
            continue
        path = _route(probability, voxel[s], voxel[t], scale)
        length = np.linalg.norm(np.diff(path * scale, axis=0), axis=1).sum()
        interior = path[1:-1]
        weak = len(interior) and probability[tuple(interior.T)].min() < min_p * 255
        if length > 1.3 * distance[k] + scale.max() or weak:
            rejected += 1
            continue
        chain = np.concatenate([[s], np.arange(next_id, next_id + len(interior)), [t]])
        next_id += len(interior)
        new_voxels.append(interior)
        new_edges.append(np.stack([chain[:-1], chain[1:]], axis=1))
        root[ps] = pt
        used[s] = True
        used[t] = degree[t] == 1
    log.info(
        "Joined %d gaps (≤ %.0f µm%s); rejected %d off-signal or detouring",
        len(new_edges), gap_um, ", tip to tip" if mutual else "", rejected,
    )  # fmt: skip
    if not new_edges:
        return voxel, edges, filled
    return (
        np.concatenate([voxel, *new_voxels]),
        np.concatenate([edges, *new_edges]),
        np.concatenate([filled, np.ones(next_id - n, dtype=bool)]),
    )


def _route(
    probability: NDArray[np.uint8], a: NDArray[np.int64], b: NDArray[np.int64], scale, pad: int = 4
) -> NDArray[np.int64]:
    """Voxel path from ``a`` to ``b`` of minimal ``1 / (p + 0.05)`` cost (physical distances)."""
    from skimage.graph import MCP_Geometric

    lo = np.maximum(np.minimum(a, b) - pad, 0)
    hi = np.minimum(np.maximum(a, b) + pad + 1, probability.shape)
    block = probability[tuple(slice(i, j) for i, j in zip(lo, hi, strict=True))]
    mcp = MCP_Geometric(1.0 / (block / 255.0 + 0.05), sampling=tuple(scale))
    mcp.find_costs([tuple(a - lo)], [tuple(b - lo)])
    return np.asarray(mcp.traceback(tuple(b - lo)), dtype=np.int64) + lo


def _tip_direction(graph, coordinates: NDArray[np.float64], tip: int, steps: int = 3) -> NDArray:
    """Return the unit vector out of a tip, from the node ``steps`` along the cable into it."""
    previous, current = -1, tip
    for _ in range(steps):
        neighbors = [k for k in graph.indices[graph.indptr[current] : graph.indptr[current + 1]]
                     if k != previous]  # fmt: skip
        if not neighbors:
            break
        previous, current = current, neighbors[0]
    vector = coordinates[tip] - coordinates[current]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
