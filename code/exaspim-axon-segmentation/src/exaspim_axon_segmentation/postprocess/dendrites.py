"""Separate the dendrites of detected somata from the rest of a skeleton (axon candidates).

The network marks every neurite; in regions with labelled cell bodies most of that is their
dendrites. Cable is called dendrite, using only the skeleton and the soma detections, if

1. **connected**: its branch lies within ``connected_um`` along the skeleton of a detected soma
   (a soma counts as touching the skeleton where a piece comes within ``soma_snap_um`` of its
   centre; the skeleton stops at the cell body's edge, 20–50 µm out);
2. **in a dendritic field**: it is on a piece, not connected, with at least ``field_fraction`` of
   its length within ``field_um`` of a detected soma (broken-off dendrites; cells whose detection
   misses the skeleton). An axon crossing a field has most of its length outside it;
3. **agglomerated**: the remaining pieces are grouped wherever a piece end comes within
   ``join_um`` of another piece; a group with an end within ``join_um`` of dendrite cable, or
   within ``soma_snap_um`` of a soma, is dendrite as a whole.

Everything else is an axon candidate. On 851474 MD (1,328 mm of skeleton, mostly dendrites of MD
neurons) this leaves 20 mm of axon candidates, mostly long fibres crossing the region; with
``join_um`` = 100 µm those fibres are absorbed as well (4 mm left).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.classify.graph import SkeletonGraph, branches


@dataclass(frozen=True)
class DendriteConfig:
    """Thresholds of the dendrite rules (see module docstring)."""

    soma_snap_um: float = 50.0
    connected_um: float = 600.0
    field_um: float = 400.0
    field_fraction: float = 0.8
    join_um: float = 50.0


def label_dendrites(
    coordinates_um: NDArray, edges: NDArray, somata_um: NDArray, config: DendriteConfig
) -> NDArray[np.bool_]:
    """Return a per-node mask: True for dendrite of a detected soma, False for axon candidate."""
    graph = SkeletonGraph(np.asarray(coordinates_um, dtype=np.float64), edges)
    n, c, e = graph.n, graph.coordinates_um, graph.edges
    dendrite = np.zeros(n, dtype=bool)
    somata = np.asarray(somata_um, dtype=np.float64).reshape(-1, 3)
    if n == 0 or not len(somata):
        return dendrite
    lengths = graph.edge_lengths_um
    tree = cKDTree(c)

    # 1. connected: branches within connected_um along the skeleton of a touched soma
    seeds = np.unique(np.concatenate(
        [np.asarray(s, dtype=np.int64) for s in tree.query_ball_point(somata, config.soma_snap_um)]
        + [np.zeros(0, dtype=np.int64)]
    ))  # fmt: skip
    edge_dendrite = np.zeros(len(e), dtype=bool)
    if len(seeds):
        geo = dijkstra(graph.adjacency(np.maximum(lengths, 1e-3)), directed=False,
                       indices=seeds, min_only=True, limit=config.connected_um)  # fmt: skip
        br = branches(graph)
        near = np.array([np.isfinite(geo[nodes]).any() for nodes in br.nodes])
        edge_dendrite = near[br.edge_branch]
    dendrite[e[edge_dendrite].ravel()] = True

    # the rest, as pieces
    rest = e[~edge_dendrite]
    rest_len = lengths[~edge_dendrite]
    if not len(rest):
        return dendrite
    _, piece = connected_components(
        coo_matrix((np.ones(len(rest)), tuple(rest.T)), shape=(n, n)), directed=False
    )
    edge_piece = piece[rest[:, 0]]
    soma_dist = cKDTree(somata).query(c)[0]

    # 2. dendritic field
    inside = (soma_dist[rest[:, 0]] <= config.field_um) & (soma_dist[rest[:, 1]] <= config.field_um)
    total = np.bincount(edge_piece, weights=rest_len, minlength=n)
    within = np.bincount(edge_piece, weights=rest_len * inside, minlength=n)
    field = within >= config.field_fraction * np.maximum(total, 1e-9)
    field_edges = field[edge_piece]
    dendrite[rest[field_edges].ravel()] = True

    # 3. agglomerate the remaining pieces by their ends
    cand = rest[~field_edges]
    if not len(cand):
        return dendrite
    cand_nodes = np.unique(cand)
    degree = np.bincount(cand.ravel(), minlength=n)
    ends = cand_nodes[degree[cand_nodes] <= 1]
    root = np.arange(n)

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for end, near in zip(ends, cKDTree(c[cand_nodes]).query_ball_point(c[ends], config.join_um),
                         strict=True):  # fmt: skip
        for j in near:
            a, b = find(piece[end]), find(piece[cand_nodes[j]])
            if a != b:
                root[a] = b
    touches = soma_dist[ends] <= config.soma_snap_um
    if dendrite.any():
        d, _ = cKDTree(c[dendrite]).query(c[ends], distance_upper_bound=config.join_um)
        touches |= np.isfinite(d)
    absorbed = {find(piece[end]) for end in ends[touches]}
    group = np.array([find(p) for p in piece[cand_nodes]])
    dendrite[cand_nodes[np.isin(group, list(absorbed))]] = True
    return dendrite
