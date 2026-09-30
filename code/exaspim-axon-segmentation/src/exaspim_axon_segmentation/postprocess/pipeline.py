"""Run the cleanup on a prediction output and write a category OME-Zarr beside it."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zarr

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR, prediction_roi
from exaspim_axon_segmentation.io import Box, ExaspimAsset
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.postprocess.cleanup import Category, CleanupConfig, cleanup
from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig, skeletonize_probability
from exaspim_axon_segmentation.postprocess.somata import load_somata

log = logging.getLogger(__name__)

CLEANUP_ATTR = "exaspim_axon_cleanup"


def cleanup_prediction(
    prediction_uri: str, out_uri: str, config: CleanupConfig, *, overwrite: bool = False
) -> dict[str, int]:
    """Classify a probability output voxel by voxel; returns voxel counts per category.

    The output has the prediction's grid, region and chunking; its attributes carry the cleanup
    config, the category legend, and the prediction's provenance.
    """
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset(provenance["asset"])
    grid = asset.fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    probability = np.asarray(group["0"][:])

    somata = load_somata(asset) if config.remove_soma_connected else None
    categories = cleanup(probability, grid, region, config, somata)
    counts = np.bincount(categories.ravel(), minlength=len(Category))
    summary = {c.name.lower(): int(counts[c]) for c in Category}

    attrs = {
        CLEANUP_ATTR: {
            "prediction": prediction_uri,
            "config": asdict(config),
            "legend": {int(c): c.name.lower() for c in Category},
            "voxels": summary,
        },
        PROVENANCE_ATTR: provenance,
    }
    out = create_ngff_array(
        out_uri, source=grid, region=region, chunks=group["0"].chunks, dtype="uint8",
        name="axon_categories", attrs=attrs, overwrite=overwrite,
    )  # fmt: skip
    out[:] = categories
    log.info("Wrote %s: %s", out_uri, summary)
    return summary


def skeletonize_prediction(
    prediction_uri: str,
    out_dir: Path,
    config: SkeletonConfig,
    *,
    crop: Box | None = None,
    swc_prefix: str | None = None,
    region_growth: float | None = None,
) -> dict:
    """Skeletonize a probability output within its (grown) region.

    Writes ``skeleton.zarr`` (skeleton voxels), ``skeleton.swc`` (all pieces as one forest),
    ``swc/<prefix>_NNNN.swc`` (one Horta-importable file per connected piece, longest first) and
    ``summary.json``.

    Parameters
    ----------
    prediction_uri : str
        Probability OME-Zarr from ``predict``.
    out_dir : Path
        Output directory.
    config : SkeletonConfig
        Thresholds and pruning.
    crop : Box, optional
        Sub-box of the prediction (region-local indices) to process instead of all of it.
    swc_prefix : str, optional
        File name prefix of the per-piece SWCs (default ``<subject>_<regions>``).
    region_growth : float, optional
        Region growth to apply instead of the one recorded with the prediction.
    """
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset(provenance["asset"])
    grid = asset.fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    local = crop or Box.full(region.shape)
    sub = Box.from_arrays(np.add(local.start, region.start), np.add(local.stop, region.start))
    probability = np.asarray(group["0"][local.slices])
    roi = prediction_roi(asset, provenance, growth=region_growth)
    inside = roi.sample_on(grid, sub)
    probability[~inside] = 0  # segment within the (grown) region only
    skeleton = skeletonize_probability(probability, grid, sub, config)

    summary = {
        "prediction": prediction_uri,
        "config": asdict(config),
        "box": {"start": list(sub.start), "stop": list(sub.stop)},
        "nodes": len(skeleton.coordinates_um),
        "length_um": skeleton.length_um,
        "length_in_region_um": skeleton.length_inside_um(inside, grid, sub),
        "gap_filled_length_um": skeleton.filled_length_um,
        "region_volume_mm3": float(inside.sum() * np.prod(grid.scale) * 1e-9),
        "region_growth": roi.growth,
    }
    summary["density_mm_per_mm3"] = (
        summary["length_in_region_um"] / 1e3 / summary["region_volume_mm3"]
        if summary["region_volume_mm3"]
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton.write_swc(out_dir / "skeleton.swc")
    prefix = swc_prefix or f"{asset.subject_id}_{'+'.join(provenance['regions'])}"
    summary["pieces"] = skeleton.write_pieces(out_dir / "swc", prefix)
    array = create_ngff_array(
        str(out_dir / "skeleton.zarr"), source=grid, region=sub, chunks=(128, 128, 128),
        dtype="uint8", name="axon_skeleton", attrs={"exaspim_axon_skeleton": summary},
        overwrite=True,
    )  # fmt: skip
    array[:] = skeleton.voxels.astype(np.uint8)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Skeleton: %.1f mm total, %.1f mm inside region (%.2f mm³ → %.1f mm/mm³)",
        summary["length_um"] / 1e3, summary["length_in_region_um"] / 1e3,
        summary["region_volume_mm3"], summary["density_mm_per_mm3"] or 0,
    )  # fmt: skip
    return summary
