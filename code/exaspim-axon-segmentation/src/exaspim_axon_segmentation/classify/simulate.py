"""Turn complete traces into skeletons that look like the segmentation pipeline's output.

Predicted skeletons differ from traces in known ways, which are imitated here so a classifier
trained on traces transfers: nodes on the level-2 voxel grid, gaps where the signal fades, short
spurs, merges wherever two neurites touch (including with untraced neighbours, imitated by
"distractor" neurons from other brains placed around each traced soma), pieces shorter than the
pipeline's minimum dropped, and soma positions from a detector (offset by ~14 µm, some missed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from exaspim_axon_segmentation.classify.graph import SkeletonGraph
from exaspim_axon_segmentation.classify.traces import TracedNeuron


@dataclass(frozen=True)
class SimulationConfig:
    """How traces are degraded (rates per µm of cable)."""

    grid_um: tuple[float, float, float] = (4.0, 2.992, 2.992)
    step_um: float = 3.0
    side_um: float = 1200.0
    distractors_per_neuron: int = 3
    distractor_distance_um: tuple[float, float] = (100.0, 500.0)
    distractor_reach_um: float = 700.0
    gap_rate_axon: float = 1 / 250
    gap_rate_dendrite: float = 1 / 500
    gap_um: tuple[float, float] = (4.0, 25.0)
    spur_rate: float = 1 / 200
    spur_um: tuple[float, float] = (10.0, 25.0)
    merge_um: float = 4.0
    merge_spacing_um: float = 15.0
    min_piece_um: float = 100.0
    soma_jitter_um: float = 8.0
    soma_miss_rate: float = 0.03


@dataclass(frozen=True, eq=False)
class SimulatedScene:
    """A degraded skeleton with ground truth, and the somata a detector would report.

    ``traced`` lists the neuron IDs of the brain's own (complete) neurons; distractors have other
    IDs. ``somata_um`` holds detected (jittered) somata and ``soma_neuron`` their neuron.
    """

    graph: SkeletonGraph
    somata_um: NDArray[np.float64]
    soma_neuron: NDArray[np.int64]
    traced: NDArray[np.int64]


def simulate_scene(
    neurons: Sequence[TracedNeuron],
    pool: Sequence[TracedNeuron],
    rng: np.random.Generator,
    config: SimulationConfig | None = None,
) -> SimulatedScene:
    """Simulate the pipeline's skeleton around every soma of ``neurons`` (one brain).

    Distractors are drawn from ``pool`` (neurons of *other* brains), rotated randomly about their
    soma and placed ``distractor_distance_um`` from a traced soma.
    """
    config = config or SimulationConfig()
    centers = np.array([n.soma_um for n in neurons])
    half = config.side_um / 2

    cube_tree = cKDTree(centers)

    def in_cubes(points: NDArray) -> NDArray:
        # inside some cube <=> Chebyshev distance to the nearest centre <= half
        return cube_tree.query(points, p=np.inf)[0] <= half if len(points) else np.zeros(0, bool)

    graphs, somata, ids = [], [], []
    for i, neuron in enumerate(neurons):
        graphs.append(neuron.graph(config.step_um, neuron_id=i, keep=in_cubes))
        somata.append(neuron.soma_um)
        ids.append(i)
    next_id = 10_000
    for neuron in neurons:
        for _ in range(config.distractors_per_neuron):
            other = pool[rng.integers(len(pool))]
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            soma = neuron.soma_um + direction * rng.uniform(*config.distractor_distance_um)
            rotation = Rotation.random(random_state=rng)

            def placed(points: NDArray, other=other, soma=soma, rotation=rotation) -> NDArray:
                return rotation.apply(points - other.soma_um) + soma

            def keep(points: NDArray, other=other, placed=placed) -> NDArray:
                near = np.linalg.norm(points - other.soma_um, axis=1) < config.distractor_reach_um
                inside = np.zeros(len(points), dtype=bool)
                inside[near] = in_cubes(placed(points[near]))
                return inside

            g = other.graph(config.step_um, neuron_id=next_id, keep=keep)
            g.coordinates_um = placed(g.coordinates_um)
            graphs.append(g)
            somata.append(soma)
            ids.append(next_id)
            next_id += 1

    graph = _concatenate(graphs)
    grid = np.asarray(config.grid_um)
    graph.coordinates_um = np.round(graph.coordinates_um / grid) * grid
    graph = _add_gaps(graph, rng, config)
    graph = _add_spurs(graph, rng, config)
    graph = _add_merges(graph, config)
    graph = _drop_short_pieces(graph, config.min_piece_um)

    somata = np.asarray(somata) + rng.normal(scale=config.soma_jitter_um, size=(len(somata), 3))
    detected = rng.random(len(somata)) >= config.soma_miss_rate
    return SimulatedScene(
        graph, somata[detected], np.asarray(ids)[detected], np.arange(len(neurons))
    )


def _concatenate(graphs: Sequence[SkeletonGraph]) -> SkeletonGraph:
    offsets = np.cumsum([0] + [g.n for g in graphs[:-1]])
    return SkeletonGraph(
        np.concatenate([g.coordinates_um for g in graphs]),
        np.concatenate([g.edges + o for g, o in zip(graphs, offsets, strict=True)]),
        np.concatenate([g.neuron for g in graphs]),
        np.concatenate([g.compartment for g in graphs]),
    )


def _add_gaps(graph: SkeletonGraph, rng, config: SimulationConfig) -> SkeletonGraph:
    """Remove the nodes within a random radius of random points along the cable."""
    rate = np.where(graph.compartment == 1, config.gap_rate_axon, config.gap_rate_dendrite)
    centres = np.flatnonzero(rng.random(graph.n) < rate * config.step_um)
    if not len(centres):
        return graph
    radius = rng.uniform(*config.gap_um, size=len(centres)) / 2
    hits = cKDTree(graph.coordinates_um).query_ball_point(graph.coordinates_um[centres], radius)
    drop = np.zeros(graph.n, dtype=bool)
    drop[np.fromiter((j for h in hits for j in h), dtype=np.int64)] = True
    return graph.subgraph(~drop)


def _add_spurs(graph: SkeletonGraph, rng, config: SimulationConfig) -> SkeletonGraph:
    """Attach short straight side branches (thinning artefacts that survive pruning)."""
    roots = np.flatnonzero(rng.random(graph.n) < config.spur_rate * config.step_um)
    coords, edges = [graph.coordinates_um], [graph.edges]
    neuron, compartment = [graph.neuron], [graph.compartment]
    next_id = graph.n
    for root in roots:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        steps = np.arange(1, int(rng.uniform(*config.spur_um) / config.step_um) + 1)
        points = graph.coordinates_um[root] + np.outer(steps * config.step_um, direction)
        chain = np.r_[root, np.arange(next_id, next_id + len(steps))]
        coords.append(points)
        edges.append(np.stack([chain[:-1], chain[1:]], axis=1))
        neuron.append(np.full(len(steps), graph.neuron[root]))
        compartment.append(np.full(len(steps), graph.compartment[root]))
        next_id += len(steps)
    return SkeletonGraph(
        np.concatenate(coords), np.concatenate(edges), np.concatenate(neuron),
        np.concatenate(compartment),
    )  # fmt: skip


def _add_merges(graph: SkeletonGraph, config: SimulationConfig) -> SkeletonGraph:
    """Join different neurons wherever they pass within ``merge_um``, one join per contact."""
    pairs = cKDTree(graph.coordinates_um).query_pairs(config.merge_um, output_type="ndarray")
    pairs = pairs[graph.neuron[pairs[:, 0]] != graph.neuron[pairs[:, 1]]]
    if not len(pairs):
        return graph
    c = graph.coordinates_um
    pairs = pairs[np.argsort(np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1))]
    # one join per contact: the closest pair in each merge_spacing_um cell
    cells = np.floor((c[pairs[:, 0]] + c[pairs[:, 1]]) / 2 / config.merge_spacing_um).astype(
        np.int64
    )
    _, first = np.unique(cells, axis=0, return_index=True)
    taken = pairs[np.sort(first)]
    return SkeletonGraph(
        graph.coordinates_um, np.concatenate([graph.edges, taken]), graph.neuron,
        graph.compartment,
    )  # fmt: skip


def _drop_short_pieces(graph: SkeletonGraph, min_um: float) -> SkeletonGraph:
    _, piece = connected_components(graph.adjacency(), directed=False)
    length = np.bincount(piece[graph.edges[:, 0]], weights=graph.edge_lengths_um,
                         minlength=piece.max() + 1)  # fmt: skip
    return graph.subgraph(length[piece] >= min_um)
