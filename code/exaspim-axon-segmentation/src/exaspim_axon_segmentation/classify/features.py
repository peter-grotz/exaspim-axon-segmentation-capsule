"""Shape features of skeleton branches relative to detected somata (no image data)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.classify.graph import (
    AXON,
    DENDRITE,
    Branches,
    SkeletonGraph,
    branches,
)

FEATURES = (
    "length", "tortuosity", "tips", "junction_degree",
    "geo_min", "geo_mean", "euc_min", "euc_mean", "geo_over_euc", "reachable",
    "order", "downstream_um", "downstream_tips", "radial_cos",
    "density_30", "density_60", "density_120", "tips_60",
    "second_soma_um", "somata_300", "piece_um", "piece_somata",
)  # fmt: skip

SOMA_SNAP_UM = 15.0
MAX_GEODESIC_UM = 3000.0


@dataclass(frozen=True, eq=False)
class BranchTable:
    """Features (one row per branch) plus ground truth where the graph carries it."""

    branches: Branches
    features: NDArray[np.float64]
    length_um: NDArray[np.float64]
    compartment: NDArray[np.int64]
    neuron: NDArray[np.int64]
    soma_index: NDArray[np.int64]


def branch_table(graph: SkeletonGraph, somata_um: NDArray) -> BranchTable:
    """Compute :data:`FEATURES` for every branch of ``graph`` given detected somata."""
    br = branches(graph)
    c = graph.coordinates_um
    lengths = graph.edge_lengths_um
    adj = graph.adjacency(np.maximum(lengths, 1e-3))
    degree = br.degree
    tips = degree == 1

    # geodesic distance to the nearest soma, and the shortest-path tree it defines
    somata = np.atleast_2d(np.asarray(somata_um, dtype=np.float64)).reshape(-1, 3)
    tree = cKDTree(c)
    seeds, seed_soma = [], []
    for s, near in enumerate(tree.query_ball_point(somata, SOMA_SNAP_UM) if len(somata) else []):
        seeds += near
        seed_soma += [s] * len(near)
    if seeds:
        geo, pred, src = dijkstra(adj, directed=False, indices=np.asarray(seeds), min_only=True,
                                  return_predecessors=True, limit=MAX_GEODESIC_UM)  # fmt: skip
        seed_to_soma = dict(zip(seeds, seed_soma, strict=True))
        soma_index = np.array([seed_to_soma.get(int(s), -1) for s in src])
    else:
        geo = np.full(graph.n, np.inf)
        pred, soma_index = np.full(graph.n, -9999), np.full(graph.n, -1)
    reach = np.isfinite(geo)
    order = np.zeros(graph.n)
    down = np.zeros(graph.n)
    down_tips = tips.astype(np.float64)
    visit = np.flatnonzero(reach)[np.argsort(geo[reach])]
    for v in visit:  # parents are visited first
        p = pred[v]
        if p >= 0:
            order[v] = order[p] + (degree[p] >= 3)
    for v in visit[::-1]:
        p = pred[v]
        if p >= 0:
            down[p] += down[v] + np.linalg.norm(c[v] - c[p])
            down_tips[p] += down_tips[v]

    if len(somata):
        k = min(2, len(somata))
        d, _ = cKDTree(somata).query(c, k=k)
        d = d.reshape(graph.n, -1)
        euc, second = d[:, 0], (d[:, 1] if k > 1 else np.full(graph.n, 5000.0))
        n300 = np.asarray(cKDTree(somata).query_ball_point(c, 300.0, return_length=True))
    else:
        euc, second, n300 = np.full(graph.n, 5000.0), np.full(graph.n, 5000.0), np.zeros(graph.n)
    step = float(np.median(lengths)) if len(lengths) else 3.0
    density = {r: _box_sum(c, np.full(graph.n, step), r) for r in (30, 60, 120)}
    tips60 = _box_sum(c, tips.astype(np.float64), 60)

    _, piece = connected_components(adj, directed=False)
    piece_um = np.bincount(piece[graph.edges[:, 0]], weights=lengths, minlength=piece.max() + 1)
    piece_somata = np.zeros(piece.max() + 1)
    if seeds:
        for s in range(len(somata)):
            hit = np.flatnonzero(np.asarray(seed_soma) == s)
            if len(hit):
                piece_somata[piece[seeds[hit[0]]]] += 1

    rows, blen, comp, neur, bsoma = [], [], [], [], []
    for nodes in br.nodes:
        p = c[nodes]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        length = float(seg.sum())
        chord = float(np.linalg.norm(p[-1] - p[0]))
        g = geo[nodes]
        near_end, far_end = (nodes[0], nodes[-1]) if g[0] <= g[-1] else (nodes[-1], nodes[0])
        radial = np.zeros(3)
        if soma_index[near_end] >= 0:
            radial = c[near_end] - somata[soma_index[near_end]]
        direction = c[far_end] - c[near_end]
        norms = np.linalg.norm(radial) * np.linalg.norm(direction)
        mid = nodes[len(nodes) // 2]
        finite = np.isfinite(g)
        rows.append([
            length, length / max(chord, 1e-3), float(tips[nodes[[0, -1]]].sum()),
            float(max(degree[nodes[0]], degree[nodes[-1]])),
            float(g[finite].min()) if finite.any() else MAX_GEODESIC_UM * 2,
            float(g[finite].mean()) if finite.any() else MAX_GEODESIC_UM * 2,
            float(euc[nodes].min()), float(euc[nodes].mean()),
            float(geo[mid] / max(euc[mid], 1.0)) if np.isfinite(geo[mid]) else 10.0,
            float(finite.any()),
            float(order[near_end]), float(down[near_end]), float(down_tips[near_end]),
            float(radial @ direction / norms) if norms > 0 else 0.0,
            float(density[30][nodes].mean()), float(density[60][nodes].mean()),
            float(density[120][nodes].mean()), float(tips60[nodes].mean()),
            float(second[nodes].mean()), float(n300[nodes].mean()),
            float(piece_um[piece[mid]]), float(piece_somata[piece[mid]]),
        ])  # fmt: skip
        blen.append(length)
        labels = graph.compartment[nodes]
        known = labels[labels > 0]
        comp.append(int(np.bincount(known).argmax()) if len(known) else 0)
        neur.append(int(np.bincount(graph.neuron[nodes] - graph.neuron.min()).argmax()
                        + graph.neuron.min()))  # fmt: skip
        bsoma.append(int(soma_index[mid]))
    return BranchTable(
        br, np.asarray(rows, dtype=np.float64).reshape(-1, len(FEATURES)), np.asarray(blen),
        np.asarray(comp), np.asarray(neur), np.asarray(bsoma),
    )  # fmt: skip


def _box_sum(
    points: NDArray, weights: NDArray, side_um: float, *, cell_um: float = 10.0,
    block_um: float = 1000.0,
) -> NDArray:  # fmt: skip
    """Sum of ``weights`` within a cube of ``side_um`` around each point (``cell_um`` grid).

    Computed block by block (``block_um`` plus a halo) so memory stays bounded however far apart
    the points are.
    """
    from scipy.ndimage import uniform_filter

    out = np.zeros(len(points))
    if not len(points):
        return out
    size = max(round(side_um / cell_um), 1)
    halo = size * cell_um
    block = np.floor(points / block_um).astype(np.int64)
    keys, owner = np.unique(block, axis=0, return_inverse=True)
    for k, key in enumerate(keys):
        lo = key * block_um - halo
        hi = (key + 1) * block_um + halo
        src = np.all((points >= lo) & (points < hi), axis=1)
        dst = np.flatnonzero(owner.ravel() == k)
        cell = np.floor((points[src] - lo) / cell_um).astype(np.int64)
        grid = np.zeros(np.ceil((hi - lo) / cell_um).astype(int) + 1, dtype=np.float32)
        np.add.at(grid, tuple(cell.T), weights[src])
        summed = uniform_filter(grid, size=size, mode="constant") * size**3
        out[dst] = summed[tuple(np.floor((points[dst] - lo) / cell_um).astype(np.int64).T)]
    return out


__all__ = ["AXON", "DENDRITE", "FEATURES", "BranchTable", "branch_table"]
