"""Segment any OME-Zarr image (e.g. a raw 1x tile), skipping tissue without bright structure.

The processed-asset pipeline segments a CCF region of ``fusion/fused.zarr``. Raw tiles have no
fusion or atlas, so here the region is built from the image itself on a coarse level:

* tissue: smoothed intensity above the out-of-sample background,
* signal: tissue voxels whose local excess over a smooth background (a 3D difference of
  Gaussians) is ``signal_sigmas`` noise SDs above zero, dilated by ``dilate_um``.

Only output tiles touching the signal mask are predicted. A sample of skipped tissue tiles can be
predicted too, to measure what the skipping misses (``sample_skipped``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, gaussian_filter

from exaspim_axon_segmentation.io import Box, NgffImage, RegionRoi

log = logging.getLogger(__name__)


@dataclass
class ImageAsset:
    """Stand-in for :class:`~exaspim_axon_segmentation.io.ExaspimAsset` around one image."""

    root: str
    annotation_root: str | None = None

    def __post_init__(self) -> None:
        self.fused = NgffImage(self.root)

    @property
    def name(self) -> str:
        """Last path component."""
        return self.root.rstrip("/").split("/")[-1]


def signal_roi(
    image: NgffImage,
    mask_level: int,
    lo_um: NDArray,
    hi_um: NDArray,
    *,
    signal_sigmas: float = 6.0,
    dilate_um: float = 100.0,
) -> tuple[RegionRoi, dict]:
    """Region of a physical box holding tissue with bright structure, on ``mask_level``."""
    tf = image.transforms[mask_level]
    box = tf.box_covering(
        np.asarray(lo_um, float), np.asarray(hi_um, float), image.shape(mask_level)
    )
    v = image.read(mask_level, box).astype(np.float32)
    smooth = gaussian_filter(v, 1.0)
    outside = np.percentile(smooth, 5)
    mad = 1.4826 * np.median(np.abs(smooth - np.median(smooth)))
    tissue = smooth > outside + max(3.0 * mad, 1.0)
    dog = smooth - gaussian_filter(v, 6.0)
    noise = (
        1.4826 * float(np.median(np.abs(dog[tissue] - np.median(dog[tissue]))))
        if tissue.any()
        else 1.0
    )
    signal = tissue & (dog > signal_sigmas * max(noise, 1e-3))
    n = np.maximum(np.round(dilate_um / np.asarray(tf.scale)).astype(int), 1)
    grown = binary_dilation(signal, structure=np.ones(tuple(2 * n + 1), bool))
    grown &= binary_dilation(tissue, iterations=1)
    stats = {
        "mask_level": mask_level,
        "mask_voxel_um": list(tf.scale),
        "box_voxels": int(v.size),
        "tissue_fraction": float(tissue.mean()),
        "signal_fraction": float(grown.mean()),
        "signal_of_tissue": float(grown.sum() / max(tissue.sum(), 1)),
    }
    log.info(
        "Mask on level %d %s: tissue %.1f%%, kept %.1f%% (%.0f%% of tissue)",
        mask_level,
        v.shape,
        100 * stats["tissue_fraction"],
        100 * stats["signal_fraction"],
        100 * stats["signal_of_tissue"],
    )
    roi = RegionRoi(frozenset(), grown, Box.from_arrays(box.start, box.stop), tf)
    tissue_roi = RegionRoi(frozenset(), tissue, Box.from_arrays(box.start, box.stop), tf)
    return roi, stats | {"_tissue_roi": tissue_roi}


def predict_image(
    url: str,
    lo_um: NDArray,
    hi_um: NDArray,
    out_uri: str,
    config,
    *,
    mask_level: int = 3,
    signal_sigmas: float = 6.0,
    check_tiles: int = 0,
    sample_skipped: int = 0,
) -> dict:
    """Segment a physical box of an OME-Zarr image where it holds bright structure.

    ``check_tiles`` > 0 first compares ``config.precision`` against fp32 on that many tiles;
    ``sample_skipped`` > 0 afterwards predicts that many skipped tissue tiles to measure what the
    skipping missed. Returns a report (also stored in the output's provenance).
    """
    import torch

    from exaspim_axon_segmentation.inference.normalize import estimate_scaling
    from exaspim_axon_segmentation.inference.pipeline import _predict, plan_region
    from exaspim_axon_segmentation.inference.predict import (
        TilePredictor,
        resolve_device,
        resolve_precision,
    )
    from exaspim_axon_segmentation.model import load_model

    asset = ImageAsset(url)
    roi, stats = signal_roi(asset.fused, mask_level, lo_um, hi_um, signal_sigmas=signal_sigmas)
    tissue_roi = stats.pop("_tissue_roi")
    plan = plan_region(asset, roi, config)
    tissue_plan = plan_region(asset, tissue_roi, config)
    report = {
        "mask": stats,
        "tiles_signal": len(plan.tiles),
        "tiles_tissue": len(tissue_plan.tiles),
    }
    log.info("Tiles: %d with signal of %d in tissue", len(plan.tiles), len(tissue_plan.tiles))
    scaling = estimate_scaling(config.normalization, asset.fused, config.level, roi)
    rng = np.random.default_rng(0)
    device = resolve_device(config.device)

    def run(tiles, precision):
        p = TilePredictor(load_model(config.weights), device, precision)
        return [p(scaling(asset.fused.read_padded(config.level, t.in_box))[None])[0] for t in tiles]

    if check_tiles and resolve_precision(config.precision) != torch.float32:
        pick = [
            plan.tiles[i]
            for i in rng.choice(len(plan.tiles), min(check_tiles, len(plan.tiles)), replace=False)
        ]
        a, b = run(pick, torch.float32), run(pick, resolve_precision(config.precision))
        diff = [float(np.abs(x - y).max()) for x, y in zip(a, b, strict=True)]
        agree = [float(((x >= 0.5) == (y >= 0.5)).mean()) for x, y in zip(a, b, strict=True)]

        def dice(x, y):
            u, v = x >= 0.5, y >= 0.5
            return float(2 * (u & v).sum() / max(u.sum() + v.sum(), 1))

        dices = [dice(x, y) for x, y in zip(a, b, strict=True)]
        report["precision_check"] = {
            "tiles": len(pick),
            "max_abs_diff": max(diff),
            "median_max_abs_diff": float(np.median(diff)),
            "voxel_agreement_min": min(agree),
            "axon_dice_median": float(np.median(dices)),
        }
        log.info("Precision check %s vs fp32: %s", config.precision, report["precision_check"])
        del a, b
    t0 = time.time()
    run_stats = _predict(
        asset,
        roi,
        ["signal"],
        out_uri,
        config,
        {"image_mask": stats, "image_report": report},
        scaling=scaling,
    )
    report["predict_seconds"] = round(time.time() - t0, 1)
    report["voxels_per_second"] = run_stats.voxels_per_second
    if sample_skipped:
        kept = {t.chunk for t in plan.tiles}
        skipped = [t for t in tissue_plan.tiles if t.chunk not in kept]
        pick = (
            [
                skipped[i]
                for i in rng.choice(len(skipped), min(sample_skipped, len(skipped)), replace=False)
            ]
            if skipped
            else []
        )
        probs = run(pick, resolve_precision(config.precision))
        frac = [float((p >= 0.5).mean()) for p in probs]
        f = np.asarray(frac) if frac else np.zeros(1)
        report["skipped_sample"] = {
            "tiles": len(pick),
            "of_skipped": len(skipped),
            "axon_fraction_median": float(np.median(f)),
            "axon_fraction_p90": float(np.percentile(f, 90)),
            "axon_fraction_max": float(f.max()),
        }
        log.info("Skipped-tile sample: %s", report["skipped_sample"])
    return report
