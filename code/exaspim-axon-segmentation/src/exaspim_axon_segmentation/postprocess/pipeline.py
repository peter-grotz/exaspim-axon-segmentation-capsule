"""Run the cleanup on a prediction output and write a category OME-Zarr beside it."""

from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import zarr

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR
from exaspim_axon_segmentation.io import Box, ExaspimAsset
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.postprocess.cleanup import Category, CleanupConfig, cleanup
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
