"""Complete neuron reconstructions (separate axon and dendrite SWCs) as labelled graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.classify.graph import AXON, DENDRITE, SkeletonGraph
from exaspim_axon_segmentation.train.swc import Swc, read_swc


@dataclass(frozen=True, eq=False)
class TracedNeuron:
    """One complete reconstruction: axon and dendrite trees sharing a soma.

    Attributes
    ----------
    brain, name : str
        Subject ID and neuron name (e.g. ``N001-794491``).
    axon, dendrite : Swc
        The two traced trees, absolute physical coordinates.
    """

    brain: str
    name: str
    axon: Swc
    dendrite: Swc

    @property
    def soma_um(self) -> NDArray[np.float64]:
        """(z, y, x) of the dendrite root, which Horta traces start at the soma."""
        root = np.flatnonzero(self.dendrite.parents == -1)[0]
        return self.dendrite.zyx_um[root]

    def graph(
        self,
        step_um: float = 3.0,
        neuron_id: int = 0,
        *,
        keep: Callable[[NDArray], NDArray] | None = None,
    ) -> SkeletonGraph:
        """Both trees as one labelled graph, densified to ``step_um``; the axon root joins the soma.

        ``keep`` maps (z, y, x) trace nodes to a boolean mask; only kept nodes (and edges between
        them) are densified, so long axons leaving the area of interest cost nothing.
        """
        coords, edges, compartment, offset = [], [], [], 0
        for swc, label in ((self.dendrite, DENDRITE), (self.axon, AXON)):
            coords.append(swc.zyx_um)
            edges.append(swc.edges() + offset)
            compartment.append(np.full(len(swc.ids), label))
            offset += len(swc.ids)
        c, e, comp = np.concatenate(coords), np.concatenate(edges), np.concatenate(compartment)
        n_dendrite = len(self.dendrite.ids)
        soma = int(np.argmin(np.linalg.norm(c[:n_dendrite] - self.soma_um, axis=1)))
        # the axon leaves the soma; extra roots of multi-tree files stay detached
        roots = np.flatnonzero(self.axon.parents == -1) + n_dendrite
        root = roots[np.argmin(np.linalg.norm(c[roots] - c[soma], axis=1))]
        e = np.concatenate([e, [[soma, root]]])
        graph = SkeletonGraph(c, e, np.full(len(c), neuron_id), comp)
        if keep is not None:
            graph = graph.subgraph(keep(c))
        return densify(graph, step_um)


def densify(graph: SkeletonGraph, step_um: float) -> SkeletonGraph:
    """Insert nodes along every edge so none is longer than ``step_um`` (labels of the child)."""
    lengths = graph.edge_lengths_um
    pieces = np.maximum(np.ceil(lengths / step_um).astype(int), 1)
    extra = pieces - 1
    n_new = int(extra.sum())
    if n_new == 0:
        return graph
    owner = np.repeat(np.arange(len(graph.edges)), extra)
    k = np.arange(n_new) - np.repeat(np.cumsum(extra) - extra, extra) + 1  # 1..extra per edge
    a, b = graph.edges[owner, 0], graph.edges[owner, 1]
    t = (k / pieces[owner])[:, None]
    points = graph.coordinates_um[a] * (1 - t) + graph.coordinates_um[b] * t
    new = graph.n + np.arange(n_new)
    first = np.r_[0, np.cumsum(extra)[:-1]]
    chains = []
    for i in np.flatnonzero(extra == 0):
        chains.append(graph.edges[i : i + 1])
    has = np.flatnonzero(extra > 0)
    for i in has:  # chain a - new... - b
        ids = new[first[i] : first[i] + extra[i]]
        path = np.r_[graph.edges[i, 0], ids, graph.edges[i, 1]]
        chains.append(np.stack([path[:-1], path[1:]], axis=1))
    return SkeletonGraph(
        np.concatenate([graph.coordinates_um, points]),
        np.concatenate(chains) if chains else graph.edges,
        np.concatenate([graph.neuron, graph.neuron[b]]),
        np.concatenate([graph.compartment, graph.compartment[b]]),
    )


def load_traced_neurons(root: Path, *, complete_only: bool = True) -> list[TracedNeuron]:
    """Read ``exaSPIM_<brain>_zarr/{Complete,Incomplete}/<neuron>/*.swc`` reconstructions.

    File names vary (``-axon-``, `` axon-``, ``dendrtie``); the compartment is taken from the
    name. When a folder holds several tracings of a compartment the first (sorted) is used, and
    ``-denoised`` re-tracings of an already listed neuron are skipped.
    """
    neurons, seen = [], set()
    statuses = ("Complete",) if complete_only else ("Complete", "Incomplete")
    for folder in sorted(root.glob("exaSPIM_*_zarr/*/*")):
        if folder.parent.name not in statuses or not folder.is_dir():
            continue
        brain = folder.parts[-3].split("_")[1]
        name = folder.name.replace("-denoised", "")
        if name in seen:
            continue
        files = sorted(folder.glob("*.swc"))
        axon = [f for f in files if "axon" in f.name.lower()]
        dendrite = [f for f in files if "dend" in f.name.lower()]
        if not axon or not dendrite:
            continue
        seen.add(name)
        neurons.append(TracedNeuron(brain, name, read_swc(axon[0]), read_swc(dendrite[0])))
    return neurons
