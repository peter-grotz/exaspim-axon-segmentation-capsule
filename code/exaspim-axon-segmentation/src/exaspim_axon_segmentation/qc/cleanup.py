"""Visual QC for the cleanup: what was kept as axon and what was removed, by reason."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR, prediction_roi
from exaspim_axon_segmentation.io import Box, ExaspimAsset
from exaspim_axon_segmentation.postprocess.cleanup import Category
from exaspim_axon_segmentation.postprocess.pipeline import CLEANUP_ATTR

REGION_COLOR = "#CDEB05"
# Drawn in this order, so later categories win where projections overlap.
COLORS = {
    Category.NO_CONFIDENT_CORE: "#737373",
    Category.TOO_SMALL: "#6464FF",
    Category.SOMA_CONNECTED: "#FF6E00",
    Category.KEPT: "#FFFFFF",
}
LABELS = {
    Category.KEPT: "kept (axon)",
    Category.SOMA_CONNECTED: "soma-connected",
    Category.TOO_SMALL: "too small",
    Category.NO_CONFIDENT_CORE: "no confident core",
}


def render_cleanup(
    prediction_uri: str, categories_uri: str, out_path: Path, *, slab_um: float = 48.0
) -> Figure:
    """Write a figure of raw, probability, and cleanup categories for two z-slabs.

    Parameters
    ----------
    prediction_uri : str
        Probability OME-Zarr from ``predict``.
    categories_uri : str
        Category OME-Zarr from ``cleanup`` for the same prediction.
    out_path : Path
        Output PNG.
    slab_um : float
        Thickness of each maximum-intensity projection.
    """
    prob = zarr.open_group(prediction_uri, mode="r")
    cats = zarr.open_group(categories_uri, mode="r")
    provenance = dict(prob.attrs[PROVENANCE_ATTR])
    summary = dict(cats.attrs[CLEANUP_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    level = provenance["level"]
    tf = asset.fused.transforms[level]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    roi = prediction_roi(asset, provenance)

    depth = max(1, round(slab_um / tf.scale[0]))
    centers = np.linspace(0, region.shape[0], 4)[1:-1].astype(int)
    lo_um, hi_um = tf.physical_extent(region)
    extent = (lo_um[2] / 1e3, hi_um[2] / 1e3, hi_um[1] / 1e3, lo_um[1] / 1e3)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
    for row, center in zip(axes, centers, strict=True):
        z0, z1 = max(center - depth // 2, 0), min(center + depth - depth // 2, region.shape[0])
        slab = Box.from_arrays(
            (region.start[0] + z0, *region.start[1:]), (region.start[0] + z1, *region.stop[1:])
        )
        raw = asset.fused.read(level, slab).max(axis=0).astype(np.float32)
        lo, hi = np.percentile(raw, [1, 99.5])
        row[0].imshow(
            np.clip((raw - lo) / max(hi - lo, 1), 0, 1) ** 0.5, cmap="gray", extent=extent
        )
        outline = roi.sample_on(tf, slab).any(axis=0)
        if outline.any():
            row[0].contour(
                outline, levels=[0.5], colors=[REGION_COLOR], linewidths=1.0, extent=extent,
                origin="upper",
            )  # fmt: skip
        row[1].imshow(
            prob["0"][z0:z1].max(axis=0) / 255, cmap="magma", vmin=0, vmax=1, extent=extent
        )
        row[2].imshow(_category_mip(cats["0"][z0:z1]), extent=extent)
        z_mm = tf.to_physical(np.array([slab.start[0], 0, 0]))[0] / 1e3
        for ax, title in zip(row, ("raw", "probability", "cleanup"), strict=True):
            ax.set_title(
                f"{title} · z {z_mm:.2f} mm + {slab_um:.0f} um mip/", loc="left", fontsize=10
            )
            ax.set_facecolor("black")
            ax.tick_params(length=0, labelsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)

    handles = [Patch(color=COLORS[c], label=LABELS[c]) for c in reversed(COLORS)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    voxels = summary["voxels"]
    total = sum(v for k, v in voxels.items() if k != "background") or 1
    shares = " · ".join(f"{LABELS[c]} {100 * voxels[c.name.lower()] / total:.1f}%" for c in LABELS)
    fig.suptitle(
        f"{asset.name} · {''.join(provenance['regions'])} · level {level} · cleanup of p > "
        f"{summary['config']['low']:.2f} voxels: {shares}/",
        x=0.01, ha="left", fontsize=11, fontweight="bold",
    )  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _category_mip(categories: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*categories.shape[1:], 3), dtype=np.float32)
    for category, color in COLORS.items():
        present = (categories == category).any(axis=0)
        rgb[present] = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    return rgb
