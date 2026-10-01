"""Score predictions against complete traces, and show them together.

Inside a completely traced region (e.g. one MD lobe) the traces are ground truth for every axon,
so both directions are measurable:

* trace recall — fraction of trace points (≈ traced length) with p > threshold within one voxel;
* precision — fraction of p > threshold voxels in the region lying within ``near_um`` of a trace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR
from exaspim_axon_segmentation.io import Box, ExaspimAsset, RegionRoi
from exaspim_axon_segmentation.train.labels import traced_lobe_box
from exaspim_axon_segmentation.train.swc import read_swc, resample

HIT, MISS = "#00A59B", "#CD0F55"


@dataclass(frozen=True)
class TraceScore:
    """Agreement between one prediction and complete traces over one part of the region."""

    part: str
    trace_points: int
    recall: float
    predicted_voxels: int
    precision: float

    def text(self) -> str:
        """One-line summary."""
        return (
            f"{self.part}: trace recall {self.recall:.2f} ({self.trace_points} pts) · precision "
            f"{self.precision:.2f} ({self.predicted_voxels} voxels > thr)"
        )


@dataclass(frozen=True, eq=False)
class TracedPrediction:
    """A prediction output with the traced lobe and trace points mapped onto its grid."""

    label: str
    probability: NDArray[np.uint8]
    region: Box
    lobe: NDArray[np.bool_]
    points_index: NDArray[np.int64]
    points_um: NDArray[np.float64]
    scale: NDArray[np.float64]
    translation: NDArray[np.float64]


def load(uri: str, label: str, swc_paths: Sequence[Path]) -> TracedPrediction:
    """Open a prediction and map traces + the traced lobe of its region onto its grid."""
    group = zarr.open_group(uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    level = provenance["level"]
    tf = asset.fused.transforms[level]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    roi = RegionRoi.from_annotation(asset.annotation, provenance["label_ids"])
    points = np.concatenate([resample(read_swc(p), 0.5 * min(tf.scale)).zyx_um for p in swc_paths])
    lobe_box, lobe_mask = traced_lobe_box(asset.fused, level, roi, points, pad_um=0.0)
    lobe = np.zeros(region.shape, dtype=bool)
    lo = np.maximum(lobe_box.start, region.start)
    hi = np.minimum(lobe_box.stop, region.stop)
    dst = tuple(slice(a - r, b - r) for a, b, r in zip(lo, hi, region.start, strict=True))
    src = tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, lobe_box.start, strict=True))
    lobe[dst] = lobe_mask[src]
    index = np.floor(tf.to_index(points) + 0.5).astype(int) - np.asarray(region.start)
    ok = np.all((index >= 0) & (index < region.shape), axis=1)
    index, points = index[ok], points[ok]
    keep = lobe[tuple(index.T)]
    return TracedPrediction(
        label=label, probability=np.asarray(group["0"][:]), region=region, lobe=lobe,
        points_index=index[keep], points_um=points[keep], scale=np.asarray(tf.scale),
        translation=np.asarray(tf.translation),
    )  # fmt: skip


def score(
    pred: TracedPrediction, x_split_um: float, *, threshold: float = 0.5, near_um: float = 5.0
) -> list[TraceScore]:
    """Trace recall and precision on each side of ``x_split_um`` (train | held-out)."""
    thr = round(threshold * 255)
    hit = _hits(pred.probability, pred.points_index, thr)

    positive = np.argwhere((pred.probability > thr) & pred.lobe)
    positive_um = (positive + np.asarray(pred.region.start)) * pred.scale + pred.translation
    distance, _ = cKDTree(pred.points_um).query(positive_um, distance_upper_bound=near_um)
    near = np.isfinite(distance)

    scores = []
    for part, pts, pos in (
        ("train side", pred.points_um[:, 2] < x_split_um, positive_um[:, 2] < x_split_um),
        ("held-out slab", pred.points_um[:, 2] >= x_split_um, positive_um[:, 2] >= x_split_um),
    ):
        precision = float(near[pos].mean()) if pos.any() else 0.0
        scores.append(TraceScore(
            part=part, trace_points=int(pts.sum()), recall=float(hit[pts].mean()),
            predicted_voxels=int(pos.sum()), precision=precision,
        ))  # fmt: skip
    return scores


def render_trace_comparison(
    uris: Sequence[str],
    labels: Sequence[str],
    swc_paths: Sequence[Path],
    out_path: Path,
    *,
    val_fraction: float = 0.2,
    window_um: float = 500.0,
    slab_um: float = 40.0,
    threshold: float = 0.5,
) -> list[list[TraceScore]]:
    """Figure + scores: raw with traces, then each prediction with trace points hit/missed.

    Rows: the densest traced window on the training side, and on the held-out slab (the last
    ``val_fraction`` of trace points along x, matching the fine-tuning split).
    """
    preds = [load(u, lab, swc_paths) for u, lab in zip(uris, labels, strict=True)]
    ref = preds[0]
    x_split = float(np.quantile(ref.points_um[:, 2], 1 - val_fraction))
    scores = [score(p, x_split, threshold=threshold) for p in preds]

    asset = ExaspimAsset.from_provenance(
        dict(zarr.open_group(uris[0], mode="r").attrs[PROVENANCE_ATTR])
    )
    level = zarr.open_group(uris[0], mode="r").attrs[PROVENANCE_ATTR]["level"]
    fig, axes = plt.subplots(2, 1 + len(preds), figsize=(5.2 * (1 + len(preds)), 10.5),
                             layout="constrained", squeeze=False)  # fmt: skip
    sides = (
        ("train side", ref.points_um[:, 2] < x_split),
        ("held-out slab", ref.points_um[:, 2] >= x_split),
    )
    for row, (part, side) in zip(axes, sides, strict=True):
        center = _densest(ref.points_um[side], window_um)
        half = np.array([slab_um / 2, window_um / 2, window_um / 2])
        lo_idx = np.floor((center - half - ref.translation) / ref.scale).astype(int)
        hi_idx = np.ceil((center + half - ref.translation) / ref.scale).astype(int)
        win = Box.from_arrays(lo_idx, hi_idx)
        local = Box.from_arrays(
            np.clip(np.subtract(win.start, ref.region.start), 0, ref.region.shape),
            np.clip(np.subtract(win.stop, ref.region.start), 0, ref.region.shape),
        )
        glob = Box.from_arrays(
            np.add(local.start, ref.region.start), np.add(local.stop, ref.region.start)
        )
        raw = asset.fused.read(level, glob).max(axis=0).astype(np.float32)
        lo, hi = np.percentile(raw, [1, 99.7])
        extent_lo = glob.start * ref.scale + ref.translation
        extent_hi = (np.asarray(glob.stop) - 1) * ref.scale + ref.translation
        extent = (extent_lo[2], extent_hi[2], extent_hi[1], extent_lo[1])
        in_win = np.all((ref.points_index >= local.start) & (ref.points_index < local.stop), axis=1)
        pts = ref.points_um[in_win]
        row[0].imshow(
            np.clip((raw - lo) / max(hi - lo, 1), 0, 1) ** 0.5, cmap="gray", extent=extent
        )
        row[0].scatter(pts[:, 2], pts[:, 1], s=0.6, c="#FF6E00", linewidths=0)
        row[0].set_title(f"raw + traces · {part} · {slab_um:.0f} um mip/", loc="left", fontsize=10)
        for ax, pred, sc in zip(row[1:], preds, scores, strict=True):
            prob = pred.probability[local.slices].max(axis=0) / 255
            ax.imshow(prob, cmap="gray", vmin=0, vmax=1, extent=extent)
            p_idx = pred.points_index[in_win]
            hit = _hits(pred.probability, p_idx, round(threshold * 255))
            ax.scatter(pts[hit, 2], pts[hit, 1], s=0.8, c=HIT, linewidths=0)
            ax.scatter(pts[~hit, 2], pts[~hit, 1], s=0.8, c=MISS, linewidths=0)
            s = next(x for x in sc if x.part == part)
            ax.set_title(f"{pred.label} · recall {s.recall:.2f} · precision {s.precision:.2f}/",
                         loc="left", fontsize=10)  # fmt: skip
        for ax in row:
            ax.tick_params(length=0, labelsize=7)
            ax.set_xlabel("x (um)", fontsize=8)
    fig.suptitle(
        f"{asset.name} · traced MD lobe · probability with trace points: teal = detected "
        f"(p > {threshold:.1f} within 1 voxel), red = missed/",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )  # fmt: skip
    lines = [
        f"{p.label} — " + " | ".join(s.text() for s in sc)
        for p, sc in zip(preds, scores, strict=True)
    ]
    notes = (
        f"split at x = {x_split:.0f} um (last {val_fraction:.0%} of trace points); precision = "
        "share of p > thr voxels in the lobe within 5 um of a trace"
    )
    fig.text(0.01, -0.01, "\n".join([*lines, notes]), fontsize=9, ha="left", va="top")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return scores


def _hits(probability: NDArray[np.uint8], index: NDArray[np.int64], thr: int) -> NDArray[np.bool_]:
    best = np.zeros(len(index), dtype=np.uint8)
    shape = np.asarray(probability.shape)
    for offset in np.ndindex(3, 3, 3):
        at = np.clip(index + np.asarray(offset) - 1, 0, shape - 1)
        best = np.maximum(best, probability[tuple(at.T)])
    return best > thr


def _densest(points_um: NDArray, window_um: float) -> NDArray[np.float64]:
    """Center of the window (in y/x) holding the most trace points, at their median z."""
    cells = np.floor(points_um[:, 1:] / window_um).astype(int)
    keys, counts = np.unique(cells, axis=0, return_counts=True)
    cell = keys[np.argmax(counts)]
    members = np.all(cells == cell, axis=1)
    sub = points_um[members]
    return np.array([np.median(sub[:, 0]), *((cell + 0.5) * window_um)])


def score_skeleton(
    coordinates_um: NDArray,
    edges: NDArray,
    traces_um: NDArray,
    *,
    near_um: float = 5.0,
    envelope_um: float | None = 50.0,
) -> dict[str, float]:
    """Length-based agreement between a skeleton and complete traces.

    Recall: share of trace points within ``near_um`` of a skeleton node (≈ share of traced
    length). Precision: share of skeleton length within ``near_um`` of a trace, counting only
    skeleton edges within ``envelope_um`` of the traces (tracing may be complete only there).
    """
    skeleton_tree = cKDTree(coordinates_um)
    d_trace, _ = skeleton_tree.query(traces_um, distance_upper_bound=near_um)
    midpoints = (coordinates_um[edges[:, 0]] + coordinates_um[edges[:, 1]]) / 2
    lengths = np.linalg.norm(coordinates_um[edges[:, 0]] - coordinates_um[edges[:, 1]], axis=1)
    d_skel, _ = cKDTree(traces_um).query(midpoints, distance_upper_bound=envelope_um or np.inf)
    counted = np.isfinite(d_skel)
    near = counted & (d_skel <= near_um)
    return {
        "recall": float(np.isfinite(d_trace).mean()),
        "precision": float(lengths[near].sum() / max(lengths[counted].sum(), 1e-9)),
        "skeleton_length_counted_um": float(lengths[counted].sum()),
        "trace_points": len(traces_um),
    }
