"""End-to-end region prediction: resolve region → plan tiles → normalize → infer → write."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import zarr

from exaspim_axon_segmentation import __version__
from exaspim_axon_segmentation.config import InferenceConfig
from exaspim_axon_segmentation.inference.normalize import IntensityScaling, estimate_scaling
from exaspim_axon_segmentation.inference.predict import (
    RunStats,
    TilePredictor,
    predict_tiles,
    resolve_device,
    resolve_precision,
)
from exaspim_axon_segmentation.inference.tiling import Tile, plan_tiles
from exaspim_axon_segmentation.io import Box, CcfOntology, ExaspimAsset, RegionRoi
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.model import REGISTRY

log = logging.getLogger(__name__)

PROVENANCE_ATTR = "exaspim_axon_segmentation"


@dataclass(frozen=True)
class RegionPlan:
    """Everything decided before the first network call."""

    roi: RegionRoi
    region: Box
    tiles: list[Tile]
    n_tiles_in_box: int


def plan_region(
    asset: ExaspimAsset, roi: RegionRoi, config: InferenceConfig, *, crop_um: float | None = None
) -> RegionPlan:
    """Choose the predicted region on the fused level grid and the tiles that touch the ROI.

    Parameters
    ----------
    asset : ExaspimAsset
        Source asset.
    roi : RegionRoi
        Region to segment.
    config : InferenceConfig
        Inference settings.
    crop_um : float, optional
        Restrict to a cube of this side centered on the region's largest cross-sections
        (for quick tests).
    """
    fused, level = asset.fused, config.level
    tf = fused.transforms[level]
    region = roi.box_in(fused, level, pad_um=config.mask_dilation_um)
    if crop_um is not None:
        half = crop_um / 2
        cube = tf.box_covering(roi.peak_um - half, roi.peak_um + half, fused.shape(level))
        region = Box.from_arrays(
            np.maximum(region.start, cube.start), np.minimum(region.stop, cube.stop)
        )
    all_tiles = plan_tiles(region, config.tile_output)
    tiles = [
        t
        for t in all_tiles
        if roi.fraction_inside(tf, t.out_box, dilate_um=config.mask_dilation_um) > 0
    ]
    return RegionPlan(roi, region, tiles, len(all_tiles))


def predict_region(
    asset_uri: str,
    regions: Sequence[str | int],
    out_uri: str,
    config: InferenceConfig,
    *,
    crop_um: float | None = None,
    overwrite: bool = False,
) -> RunStats:
    """Segment axons inside CCF regions of one asset and write an OME-Zarr probability map.

    Output: uint8 probabilities (``p × 255``) on the chosen fused level's grid, covering the
    region's padded bounding box; voxels in tiles that miss the region stay 0. Provenance (config,
    weights, normalization, region) is stored in the group attributes.
    """
    asset = ExaspimAsset(asset_uri)
    ontology = CcfOntology.load()
    roi = RegionRoi.from_annotation(asset.annotation, ontology.resolve(regions))
    plan = plan_region(asset, roi, config, crop_um=crop_um)
    log.info(
        "Region %s: %.2f mm³; level %d box %s; %d/%d tiles touch the region",
        list(regions), roi.volume_mm3, config.level, plan.region.shape,
        len(plan.tiles), plan.n_tiles_in_box,
    )  # fmt: skip

    scaling = estimate_scaling(config.normalization, asset.fused, config.level, roi)
    log.info("Intensity scaling: offset %.1f, scale %.1f", scaling.offset, scaling.scale)

    device = resolve_device(config.device)
    precision = resolve_precision(config.precision)
    log.info("Device %s, precision %s", device, precision)
    model = REGISTRY[config.weights].load()
    predictor = TilePredictor(
        model, device, precision, compile=config.compile, allow_tf32=config.allow_tf32
    )
    predictor.self_check(config.batch_size)

    source = asset.fused.transforms[config.level]
    attrs = {PROVENANCE_ATTR: _provenance(asset, regions, roi, plan, config, scaling)}
    out = create_ngff_array(
        out_uri, source=source, region=plan.region, chunks=(config.tile_output,) * 3,
        dtype="uint8", name="axon_probability", attrs=attrs, overwrite=overwrite,
    )  # fmt: skip
    group = zarr.open_group(out_uri, mode="a")
    stats = predict_tiles(
        asset.fused, config.level, plan.tiles, plan.region, scaling, predictor, out, group,
        batch_size=config.batch_size, io_threads=config.io_threads,
    )  # fmt: skip
    log.info(
        "Predicted %d tiles (%d resumed) in %.0f s · %.2f Mvox/s",
        stats.n_predicted, stats.n_resumed, stats.seconds, stats.voxels_per_second / 1e6,
    )  # fmt: skip
    return stats


def _provenance(
    asset: ExaspimAsset,
    regions: Sequence[str | int],
    roi: RegionRoi,
    plan: RegionPlan,
    config: InferenceConfig,
    scaling: IntensityScaling,
) -> dict:
    weights = REGISTRY[config.weights]
    return {
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "asset": asset.root,
        "regions": [str(r) for r in regions],
        "label_ids": sorted(roi.label_ids),
        "level": config.level,
        "region_box": {"start": list(plan.region.start), "stop": list(plan.region.stop)},
        "config": config.model_dump(mode="json"),
        "weights": {
            "name": weights.name,
            "repo_id": weights.repo_id,
            "filename": weights.filename,
            "revision": weights.revision,
            "sha256": weights.sha256,
        },
        "scaling": scaling.as_dict(),
    }
