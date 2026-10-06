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

With a Robo-LS tracer checkpoint (``tracer_checkpoint``), every followed end point is first
traced on the image: a trace that reaches another piece the network calls the same cell links
it (P(same cell) >= ``trace_accept``); one that runs on along signal adds its path and is traced
again (``trace_passes`` per round); one that loses the signal, reaches a terminal or turns back
ends that axon there. Only ends still open after tracing that point out of the done area get
cubes. Geometric stitching is then only the seam join of new cubes onto old cut ends.

An axon stops being followed when it reaches a detected soma (traced back to its cell), runs
out of segmented cable (or the tracer ends it), or exceeds ``budget_um`` of path from the
region. Caps on rounds, cubes and minutes bound one run.
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
SEGMENTED, TRACED = 0, 1


@dataclass(frozen=True)
class ExpandConfig:
    """Settings of the axon-following loop."""

    cube_um: float = 800.0
    behind_um: float = 80.0  # a cube starts this far behind its exit, so cut ends overlap
    exit_um: float = 40.0  # an end point is an exit if this far ahead lies outside the done area
    overlap_um: float = 30.0  # new cubes are skeletonized this far into the done area
    duplicate_um: float = 6.0  # new cable this close to old cable (in the overlap) is dropped
    join_um: float = 20.0  # stitch new and old end points this close, ahead of the end
    join_any_um: float = 8.0  # ... or this close in any direction
    budget_um: float = 100_000.0  # path length followed from the region, per axon
    soma_um: float = 50.0
    max_rounds: int = 40
    max_cubes: int = 800
    max_minutes: float = 600.0
    tracer_checkpoint: str | None = None  # Robo-LS v2 best.pt; None = geometric only
    trace_passes: int = 3  # traces chained per end point per round
    trace_steps: int = 60  # 5 µm steps per trace
    trace_accept: float = 0.5  # P(same cell) to link a piece
    trace_back_um: float = 100.0  # path the tracer sees behind an end point


@dataclass
class Graph:
    """The stitched skeleton: nodes (z, y, x µm), edges, SWC types, round each node was added."""

    coords: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 3)))
    edges: NDArray[np.int64] = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    types: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    round: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    source: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

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
        self.source = np.concatenate([self.source, np.full(len(types), SEGMENTED)])
        return np.arange(offset, self.n)

    def add_path(self, start: int, points: NDArray, round_: int) -> NDArray[np.int64]:
        """Append a traced polyline (z, y, x µm) hanging off node ``start``; returns new nodes."""
        offset, m = self.n, len(points)
        if not m:
            return np.zeros(0, dtype=np.int64)
        chain = np.r_[start, np.arange(offset, offset + m)]
        self.coords = np.concatenate([self.coords, points])
        self.edges = np.concatenate([self.edges, np.stack([chain[:-1], chain[1:]], axis=1)])
        self.types = np.concatenate([self.types, np.full(m, AXON)])
        self.round = np.concatenate([self.round, np.full(m, round_)])
        self.source = np.concatenate([self.source, np.full(m, TRACED)])
        return np.arange(offset, offset + m)

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

    def mask_on(self, grid, box: Box, *, n_cubes: int, shrink_um: float = 0.0) -> NDArray[np.bool_]:
        """Done area on a box of a level grid (region plus the first ``n_cubes`` cubes).

        ``shrink_um`` erodes it, so new cubes are skeletonized a little into the done area and
        their cable overlaps the old cut ends.
        """
        mask = self.roi.sample_on(grid, box)
        if shrink_um > 0 and mask.any():
            from scipy.ndimage import minimum_filter

            r = np.round(shrink_um / np.asarray(grid.scale)).astype(int)
            mask = minimum_filter(mask, size=tuple(2 * r + 1), mode="constant", cval=True)
        for centre, half in zip(self.centres[:n_cubes], self.halves[:n_cubes], strict=True):
            half = half - shrink_um
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
    tracer, ended = None, np.zeros(0, dtype=bool)
    if config.tracer_checkpoint:
        from exaspim_axon_segmentation.robo.image import Level1Image
        from exaspim_axon_segmentation.robo.tracer import RoboTracer

        tracer = RoboTracer(config.tracer_checkpoint, Level1Image(asset),
                            device=str(resolve_device(inference.device)),
                            accept=config.trace_accept, max_steps=config.trace_steps)  # fmt: skip
    for k in range(1, config.max_rounds + 1):
        traced = {}
        if tracer is not None:
            # the tracer runs from the open ends of the cable added last round (round 0: MD)
            followed, ended, traced = _trace_ends(graph, followed, ended, done, roi, tracer,
                                                  config, soma_tree, k - 1)  # fmt: skip
        dist = _path_from_region(graph, followed, done, roi)
        exits = _exits(graph, followed, done, config, dist)
        if len(exits) and len(ended):
            exits = exits[~ended[exits[:, 0].astype(int)]]
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
            before = done.mask_on(grid, box, n_cubes=n_before + i, shrink_um=config.overlap_um)
            skeleton = skeletonize(box, ~before, relaxed=True)
            if not len(skeleton.coordinates_um):
                continue
            if graph.n:  # drop the new cable that re-traces old cable in the overlap
                d, _ = cKDTree(graph.coords).query(
                    skeleton.coordinates_um, distance_upper_bound=config.duplicate_um
                )
                skeleton = skeleton.subset(~np.isfinite(d))  # fmt: skip
                if not len(skeleton.coordinates_um):
                    continue
            added = graph.add(skeleton, k)
            _stitch(graph, added, config.join_um, config.join_any_um)
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
        rounds.append({"round": k, "exits": len(exits), "cubes": len(cubes), "traced": traced,
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


def _stitch(graph: Graph, added: NDArray[np.int64], join_um: float, any_um: float) -> None:
    """Join new end points to old cable and old end points to new cable.

    An end joins the nearest node across within ``any_um``, or within ``join_um`` if that node
    lies ahead of it (within 45° of the direction the cable arrives in).
    """
    new = np.zeros(graph.n, dtype=bool)
    new[added] = True
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
    adj = graph.adjacency()
    joins = []
    for side, other in ((new, ~new), (~new, new)):
        tips = np.flatnonzero(side & (degree <= 1))
        targets = np.flatnonzero(other)
        if not len(tips) or not len(targets):
            continue
        tree = cKDTree(graph.coords[targets])
        for tip, near in zip(tips, tree.query_ball_point(graph.coords[tips], join_um), strict=True):
            if not near:
                continue
            cand = targets[near]
            gap = graph.coords[cand] - graph.coords[tip]
            dist = np.linalg.norm(gap, axis=1)
            ok = dist <= any_um
            heading = _tip_heading(adj, graph.coords, tip)
            if heading is not None:
                ok |= (gap @ heading) >= 0.7 * dist  # within 45° ahead
            if ok.any():
                joins.append((tip, cand[ok][np.argmin(dist[ok])]))
    if joins:
        pairs = np.unique(np.sort(np.asarray(joins), axis=1), axis=0)
        graph.edges = np.concatenate([graph.edges, pairs])


def _tip_heading(adj, coords: NDArray, tip: int, steps: int = 4) -> NDArray | None:
    """Return the direction the cable arrives at ``tip`` in (``steps`` nodes back), if any."""
    prev, cur = -1, tip
    for _ in range(steps):
        nbrs = [v for v in adj.indices[adj.indptr[cur] : adj.indptr[cur + 1]] if v != prev]
        if not nbrs:
            break
        prev, cur = cur, nbrs[0]
    v = coords[tip] - coords[cur]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


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


def _path_back(adj, coords: NDArray, tip: int, length_um: float) -> NDArray[np.int64]:
    """Nodes from ~``length_um`` behind ``tip`` up to it, along the cable (forward order)."""
    path, prev, cur, run = [tip], -1, tip, 0.0
    while run < length_um:
        nbrs = [v for v in adj.indices[adj.indptr[cur] : adj.indptr[cur + 1]] if v != prev]
        if not nbrs:
            break
        run += float(np.linalg.norm(coords[nbrs[0]] - coords[cur]))
        prev, cur = cur, nbrs[0]
        path.append(cur)
    return np.asarray(path[::-1])


def _trace_ends(graph, followed, ended, done, roi, tracer, config, soma_tree, round_):
    """Trace every open followed end point, adding the traced cable and links to the graph.

    Returns the extended followed mask, the mask of ends the tracer closed, and outcome counts.
    """
    from exaspim_axon_segmentation.robo.tracer import Fragments, Start

    counts: dict[str, int] = {}
    for _ in range(config.trace_passes):
        e = np.zeros(graph.n, dtype=bool)
        e[: len(ended)] = ended
        ended = e
        degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
        dist = _path_from_region(graph, followed, done, roi)
        tips = np.flatnonzero(followed & (degree == 1) & ~ended & (dist < config.budget_um))
        if soma_tree is not None and len(tips):
            tips = tips[soma_tree.query(graph.coords[tips])[0] >= config.soma_um]
        if not len(tips):
            break
        _, piece = connected_components(graph.adjacency(), directed=False)
        xyz = graph.coords[:, ::-1]
        frags = Fragments(xyz, piece.astype(np.int64))
        adj = graph.adjacency(followed)
        starts = [Start(xyz[_path_back(adj, graph.coords, t, config.trace_back_um)], int(piece[t]))
                  for t in tips]  # fmt: skip
        results = tracer.trace(starts, frags)
        reopened, bridged = False, set()
        for tip, r in zip(tips, results, strict=True):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            path = r["trace"][1:, ::-1]  # drop the start (the tip itself); to z, y, x
            ended[tip] = True  # this end is dealt with; a continuing trace has a new end
            if r["status"] == "returned_own" or (r["status"] != "linked" and len(path) < 2):
                continue
            if r["status"] == "linked":
                pair = tuple(sorted((int(piece[tip]), int(r["link"]))))
                if pair in bridged:
                    continue  # the gap was traced from its other end already
                bridged.add(pair)
            new = graph.add_path(tip, path, round_)
            ended = np.r_[ended, np.zeros(len(new), dtype=bool)]
            if r["status"] == "linked":
                members = np.flatnonzero(piece == r["link"])
                j = members[np.argmin(np.linalg.norm(graph.coords[members] - path[-1], axis=1))]
                graph.edges = np.concatenate([graph.edges, [[new[-1], j]]])
                ended[new[-1]] = True
            elif r["status"] in ("lost", "terminal"):
                ended[new[-1]] = True
            else:
                reopened = True  # maxlen / oob: the trace end is open, trace on next pass
        followed = _extend_followed(graph, followed)
        if not reopened:
            break
    e = np.zeros(graph.n, dtype=bool)
    e[: len(ended)] = ended
    log.info("traced: %s", counts)
    return followed, e, counts


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
                        types=graph.types, round=graph.round, source=graph.source,
                        followed=followed)  # fmt: skip
