"""Train the dendrite-vs-axon branch classifier on simulated skeletons of traced neurons."""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.classify.features import FEATURES, BranchTable, branch_table
from exaspim_axon_segmentation.classify.graph import DENDRITE
from exaspim_axon_segmentation.classify.simulate import SimulationConfig, simulate_scene
from exaspim_axon_segmentation.classify.traces import TracedNeuron

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompartmentScores:
    """Length-weighted agreement of predicted and true compartments."""

    accuracy: float
    dendrite_recall: float
    axon_recall: float
    dendrite_share: float

    def text(self) -> str:
        """One-line summary."""
        return (
            f"accuracy {self.accuracy:.3f} · dendrite recall {self.dendrite_recall:.3f} · "
            f"axon recall {self.axon_recall:.3f} (dendrite = {self.dendrite_share:.0%} of cable)"
        )


def simulate_tables(
    neurons: Sequence[TracedNeuron],
    brains: Sequence[str],
    *,
    repeats: int,
    exclude_from_pool: Sequence[str] = (),
    seed: int = 0,
    config: SimulationConfig | None = None,
) -> list[BranchTable]:
    """Branch tables of ``repeats`` independent simulations per brain.

    Distractor neurons come from the other brains, never from ``exclude_from_pool`` (held out).
    """
    rng = np.random.default_rng(seed)
    tables = []
    for brain in brains:
        own = [n for n in neurons if n.brain == brain]
        pool = [n for n in neurons if n.brain != brain and n.brain not in exclude_from_pool]
        for _ in range(repeats):
            start = time.time()
            scene = simulate_scene(own, pool, rng, config)
            tables.append(branch_table(scene.graph, scene.somata_um))
            log.info(
                "%s: %d neurons, %d nodes, %d branches (%.0f s)",
                brain, len(own), scene.graph.n, len(tables[-1].length_um), time.time() - start,
            )  # fmt: skip
    return tables


def stack(tables: Sequence[BranchTable]) -> tuple[NDArray, NDArray, NDArray]:
    """Features, dendrite labels and length weights of all labelled branches."""
    x = np.concatenate([t.features for t in tables])
    y = np.concatenate([t.compartment for t in tables])
    w = np.concatenate([t.length_um for t in tables])
    known = y > 0
    return x[known], (y[known] == DENDRITE).astype(int), w[known]


def fit(x: NDArray, y: NDArray, w: NDArray):
    """Gradient-boosted trees, length-weighted."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=48, l2_regularization=1.0,
        random_state=0,
    )  # fmt: skip
    return model.fit(x, y, sample_weight=w)


def score(model, x: NDArray, y: NDArray, w: NDArray) -> CompartmentScores:
    """Length-weighted scores at p > 0.5."""
    dendrite = model.predict_proba(x)[:, 1] > 0.5
    truth = y == 1
    return CompartmentScores(
        accuracy=float((w * (dendrite == truth)).sum() / w.sum()),
        dendrite_recall=float((w * (dendrite & truth)).sum() / max((w * truth).sum(), 1e-9)),
        axon_recall=float((w * (~dendrite & ~truth)).sum() / max((w * ~truth).sum(), 1e-9)),
        dendrite_share=float((w * truth).sum() / w.sum()),
    )


def feature_importance(model, x: NDArray, y: NDArray, w: NDArray, n: int = 20000) -> list:
    """Permutation importance (accuracy drop), most important first."""
    from sklearn.inspection import permutation_importance

    pick = np.random.default_rng(0).permutation(len(y))[:n]
    result = permutation_importance(
        model, x[pick], y[pick], n_repeats=3, random_state=0, sample_weight=w[pick]
    )
    order = np.argsort(-result.importances_mean)
    return [(FEATURES[i], float(result.importances_mean[i])) for i in order]


def save(model, path: Path) -> None:
    """Pickle the model with the feature names it expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"model": model, "features": FEATURES}))
