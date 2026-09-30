"""Skeleton graphs with optional ground truth, and their decomposition into branches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

UNKNOWN, AXON, DENDRITE = 0, 1, 2


@dataclass(eq=False)
class SkeletonGraph:
    """An undirected skeleton in physical (z, y, x) micrometers.

    Attributes
    ----------
    coordinates_um : numpy.ndarray, shape (n, 3)
        Node positions.
    edges : numpy.ndarray, shape (m, 2)
        Node index pairs.
    neuron : numpy.ndarray of int, shape (n,)
        Ground-truth neuron per node (-1 unknown).
    compartment : numpy.ndarray of int, shape (n,)
        Ground-truth ``AXON`` / ``DENDRITE`` per node (``UNKNOWN`` if not known).
    """

    coordinates_um: NDArray[np.float64]
    edges: NDArray[np.int64]
    neuron: NDArray[np.int64] = field(default=None)  # type: ignore[assignment]
    compartment: NDArray[np.int64] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.coordinates_um)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        if self.neuron is None:
            self.neuron = np.full(n, -1, dtype=np.int64)
        if self.compartment is None:
            self.compartment = np.zeros(n, dtype=np.int64)

    @property
    def n(self) -> int:
        """Node count."""
        return len(self.coordinates_um)

    @property
    def edge_lengths_um(self) -> NDArray[np.float64]:
        """Physical length of every edge."""
        c = self.coordinates_um
        return np.linalg.norm(c[self.edges[:, 0]] - c[self.edges[:, 1]], axis=1)

    def adjacency(self, weights: NDArray | None = None) -> csr_matrix:
        """Symmetric sparse adjacency (edge lengths by default)."""
        w = self.edge_lengths_um if weights is None else weights
        e = self.edges
        return coo_matrix(
            (np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(self.n,) * 2
        ).tocsr()

    def pieces(self) -> NDArray[np.int64]:
        """Return the connected piece index of every node."""
        return connected_components(self.adjacency(), directed=False)[1]

    def subgraph(self, keep: NDArray[np.bool_]) -> SkeletonGraph:
        """Nodes where ``keep`` is set, with the edges between them."""
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        e = remap[self.edges]
        return SkeletonGraph(
            self.coordinates_um[keep], e[(e >= 0).all(axis=1)],
            self.neuron[keep], self.compartment[keep],
        )  # fmt: skip


@dataclass(frozen=True, eq=False)
class Branches:
    """Paths between junctions and end points; every edge belongs to exactly one branch.

    Attributes
    ----------
    nodes : list of numpy.ndarray
        Node indices along each branch, end to end.
    edge_branch : numpy.ndarray of int, shape (m,)
        Branch of every graph edge.
    degree : numpy.ndarray of int, shape (n,)
        Node degree.
    """

    nodes: list[NDArray[np.int64]]
    edge_branch: NDArray[np.int64]
    degree: NDArray[np.int64]

    def __len__(self) -> int:
        return len(self.nodes)


def branches(graph: SkeletonGraph) -> Branches:
    """Split a skeleton into branches (runs of degree-2 nodes between other nodes)."""
    e = graph.edges
    adj = graph.adjacency(np.arange(1, len(e) + 1, dtype=np.float64))  # data = edge id + 1
    ip, ix, eid = adj.indptr, adj.indices, adj.data.astype(np.int64) - 1
    degree = np.diff(ip)
    edge_branch = np.full(len(e), -1, dtype=np.int64)
    out: list[NDArray[np.int64]] = []

    def walk(u: int, slot: int) -> None:
        path, cur, k = [u], ix[slot], eid[slot]
        while True:
            edge_branch[k] = len(out)
            path.append(cur)
            if degree[cur] != 2 or cur == u:
                break
            a = ip[cur]
            slot = a if eid[a] != k else a + 1  # the other edge of this degree-2 node
            cur, k = ix[slot], eid[slot]
            if edge_branch[k] >= 0:
                break
        out.append(np.asarray(path, dtype=np.int64))

    for u in np.flatnonzero(degree != 2):
        for slot in range(ip[u], ip[u + 1]):
            if edge_branch[eid[slot]] < 0:
                walk(int(u), slot)
    for k in np.flatnonzero(edge_branch < 0):  # isolated cycles of degree-2 nodes
        u = int(e[k, 0])
        slot = next(s for s in range(ip[u], ip[u + 1]) if eid[s] == k)
        walk(u, slot)
    return Branches(out, edge_branch, degree)
