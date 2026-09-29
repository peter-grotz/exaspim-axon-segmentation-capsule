"""Visual QC for predictions: raw vs. probability maximum-intensity projections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.figure import Figure
from numpy.typing import NDArray

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR
from exaspim_axon_segmentation.inference.predict import COMPLETED_ATTR
from exaspim_axon_segmentation.io import Box, ExaspimAsset, RegionRoi
from exaspim_axon_segmentation.model.unet import CONTEXT

REGION_COLOR = "#CDEB05"


@dataclass(frozen=True)
class PredictionRun:
    """A prediction output opened for QC."""

    uri: str
    label: str
    probability: NDArray[np.uint8]
    provenance: dict
    provenance_completed: tuple[tuple[int, int, int], ...]

    @classmethod
    def open(cls, uri: str, label: str | None = None) -> PredictionRun:
        """Load a (small) prediction output fully into memory."""
        group = zarr.open_group(uri, mode="r")
        provenance = dict(group.attrs[PROVENANCE_ATTR])
        completed = tuple(tuple(c) for c in group.attrs.get(COMPLETED_ATTR, []))
        probability = np.asarray(group["0"][:])
        return cls(uri, label or Path(uri).stem, probability, provenance, completed)

    @property
    def region(self) -> Box:
        """Predicted region in fused-level indices."""
        box = self.provenance["region_box"]
        return Box.from_arrays(box["start"], box["stop"])

    @property
    def computed(self) -> NDArray[np.bool_]:
        """Voxels covered by completed tiles (uncovered voxels hold the fill value, not p = 0)."""
        side = self.provenance["config"]["tile_size"] - 2 * CONTEXT
        out = np.zeros(self.probability.shape, dtype=bool)
        for chunk in self.provenance_completed:
            out[tuple(slice(c * side, (c + 1) * side) for c in chunk)] = True
        return out

    def scaling_text(self) -> str:
        """Compact description of the intensity scaling used."""
        s = self.provenance["scaling"]
        return f"raw {s['offset']:.0f} -> 0, {s['offset'] + s['scale']:.0f} -> 1"


def render_prediction_comparison(
    uris: Sequence[str],
    out_path: Path,
    *,
    labels: Sequence[str] | None = None,
    slab_um: float = 48.0,
    n_slabs: int = 2,
) -> Figure:
    """Compare prediction runs over the same region against the raw image.

    Rows are z-slabs (maximum-intensity projections of ``slab_um`` thickness); the first column is
    the raw image with the region outline, then one column of probabilities per run.

    Parameters
    ----------
    uris : sequence of str
        Prediction outputs; all must share asset, level and region.
    out_path : Path
        Output PNG.
    labels : sequence of str, optional
        Column titles (default: output names).
    slab_um : float
        Slab thickness.
    n_slabs : int
        Number of evenly spaced slabs.
    """
    runs = [PredictionRun.open(u, lab) for u, lab in zip(uris, labels or [None] * len(uris))]
    ref = runs[0]
    for run in runs[1:]:
        if run.region != ref.region or run.provenance["asset"] != ref.provenance["asset"]:
            raise ValueError(f"{run.uri} covers a different region than {ref.uri}")
    asset = ExaspimAsset(ref.provenance["asset"])
    level = ref.provenance["level"]
    tf = asset.fused.transforms[level]
    raw = asset.fused.read(level, ref.region)
    roi = RegionRoi.from_annotation(asset.annotation, ref.provenance["label_ids"])
    mask = roi.sample_on(tf, ref.region)

    depth = max(1, round(slab_um / tf.scale[0]))
    centers = np.linspace(0, raw.shape[0], n_slabs + 2)[1:-1].astype(int)
    slabs = [slice(max(c - depth // 2, 0), c + depth - depth // 2) for c in centers]
    lo_um, hi_um = tf.physical_extent(ref.region)
    extent = (lo_um[2] / 1e3, hi_um[2] / 1e3, hi_um[1] / 1e3, lo_um[1] / 1e3)

    n_cols = 1 + len(runs)
    fig, axes = plt.subplots(
        n_slabs, n_cols, figsize=(3.6 * n_cols, 3.9 * n_slabs), layout="constrained", squeeze=False
    )
    for row, slab in zip(axes, slabs, strict=True):
        z_mm = tf.to_physical(np.array([ref.region.start[0] + slab.start, 0, 0]))[0] / 1e3
        raw_mip = raw[slab].max(axis=0).astype(np.float32)
        lo, hi = np.percentile(raw_mip, [1, 99.5])
        row[0].imshow(
            np.clip((raw_mip - lo) / max(hi - lo, 1), 0, 1) ** 0.5, cmap="gray", extent=extent
        )
        region_plane = mask[slab].any(axis=0)
        if region_plane.any():
            row[0].contour(
                region_plane, levels=[0.5], colors=[REGION_COLOR], linewidths=1.0,
                extent=extent, origin="upper",
            )  # fmt: skip
        row[0].set_title(f"raw · z {z_mm:.2f} mm + {slab_um:.0f} um mip/", loc="left", fontsize=10)
        for ax, run in zip(row[1:], runs, strict=True):
            im = ax.imshow(
                run.probability[slab].max(axis=0) / 255, cmap="magma", vmin=0, vmax=1, extent=extent
            )
            ax.set_title(f"{run.label}/", loc="left", fontsize=10)
        for ax in row:
            ax.tick_params(length=0, labelsize=7)
            ax.set_xlabel("x (mm)", fontsize=8)
            ax.set_ylabel("y (mm)", fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.colorbar(im, ax=axes[:, -1], shrink=0.6, label="axon probability")

    lines = []
    for run in runs:
        p = run.probability[run.computed & mask] / 255
        lines.append(
            f"{run.label}: {run.scaling_text()} · inside {''.join(ref.provenance['regions'])}: "
            f"mean p {p.mean():.3f}, p>0.5 {100 * (p > 0.5).mean():.2f}% of voxels"
        )
    fig.suptitle(
        f"{asset.name} · level {level} ({tf.scale[2]:.2f} um xy, {tf.scale[0]:.0f} um z) · "
        f"axon probability/",
        x=0.01, ha="left", fontsize=13, fontweight="bold",
    )  # fmt: skip
    fig.text(0.01, -0.01, "\n".join(lines), fontsize=9, ha="left", va="top")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
