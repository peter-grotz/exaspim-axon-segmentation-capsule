"""End-to-end region prediction: resolve region → plan tiles → normalize → infer → write."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import zarr
from numpy.typing import NDArray

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
from exaspim_axon_segmentation.model import REGISTRY, load_model

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
    annotation_asset: str | None = None,
) -> RunStats:
    """Segment axons inside CCF regions of one asset and write an OME-Zarr probability map.

    Output: uint8 probabilities (``p × 255``) on the chosen fused level's grid, covering the
    region's padded bounding box; voxels in tiles that miss the region stay 0. Provenance (config,
    weights, normalization, region) is stored in the group attributes. ``annotation_asset`` is
    the asset holding the CCF alignment and soma detections, if not ``asset_uri`` itself.
    """
    asset = ExaspimAsset(asset_uri, annotation_root=annotation_asset)
    ontology = CcfOntology.load()
    atlas_roi = RegionRoi.from_annotation(asset.annotation, ontology.resolve(regions))
    roi = atlas_roi.grown(config.region_growth)
    log.info(
        "Region %s: %.2f mm³ (%.2f mm³ grown by %.0f%%)",
        list(regions), atlas_roi.volume_mm3, roi.volume_mm3, 100 * config.region_growth,
    )  # fmt: skip
    return _predict(asset, roi, [str(r) for r in regions], out_uri, config, {},
                    crop_um=crop_um, overwrite=overwrite)  # fmt: skip


def predict_points(
    asset_uri: str,
    centers_um: NDArray,
    side_um: float,
    out_uri: str,
    config: InferenceConfig,
    *,
    overwrite: bool = False,
) -> RunStats:
    """Segment cubes of side ``side_um`` around physical (z, y, x) points, e.g. traced somata.

    One output covers all cubes (their bounding box); only tiles touching a cube are predicted
    and stored. Needs only the fused image, no CCF annotation.
    """
    asset = ExaspimAsset(asset_uri)
    centers = np.atleast_2d(np.asarray(centers_um, dtype=np.float64))
    roi = RegionRoi.around_points(centers, side_um)
    log.info("%d cubes of %.0f µm: %.2f mm³", len(centers), side_um, roi.volume_mm3)
    points = {"centers_um": centers.round(3).tolist(), "side_um": side_um}
    return _predict(asset, roi, ["points"], out_uri, config, {"points": points},
                    overwrite=overwrite)  # fmt: skip


def predict_box(
    asset_uri: str,
    lo_um: NDArray,
    hi_um: NDArray,
    out_uri: str,
    config: InferenceConfig,
    *,
    overwrite: bool = False,
    scaling: IntensityScaling | None = None,
) -> RunStats:
    """Segment a physical (z, y, x) box, e.g. the volume spanned by a set of traced neurons.

    Only the fused image is needed (no CCF annotation). Skeletonize the output block by block
    (``skeletonize-blocks``) when it is too large for memory. Pass ``scaling`` to segment
    adjacent boxes (slabs of one volume) with one intensity normalization.
    """
    asset = ExaspimAsset(asset_uri)
    lo, hi = np.asarray(lo_um, dtype=np.float64), np.asarray(hi_um, dtype=np.float64)
    roi = RegionRoi.box(lo, hi)
    log.info("Box %s-%s µm: %.1f mm³", lo.round(0).tolist(), hi.round(0).tolist(), roi.volume_mm3)
    box = {"lo_um": lo.round(3).tolist(), "hi_um": hi.round(3).tolist()}
    return _predict(asset, roi, ["box"], out_uri, config, {"box": box}, overwrite=overwrite,
                    scaling=scaling)  # fmt: skip


def _predict(
    asset: ExaspimAsset,
    roi: RegionRoi,
    regions: list[str],
    out_uri: str,
    config: InferenceConfig,
    extra: dict,
    *,
    crop_um: float | None = None,
    overwrite: bool = False,
    scaling: IntensityScaling | None = None,
) -> RunStats:
    plan = plan_region(asset, roi, config, crop_um=crop_um)
    log.info(
        "Level %d box %s; %d/%d tiles touch the region",
        config.level, plan.region.shape, len(plan.tiles), plan.n_tiles_in_box,
    )  # fmt: skip

    if scaling is None:
        scaling = estimate_scaling(config.normalization, asset.fused, config.level, roi)
    log.info("Intensity scaling: offset %.1f, scale %.1f", scaling.offset, scaling.scale)

    device = resolve_device(config.device)
    precision = resolve_precision(config.precision)
    log.info("Device %s, precision %s", device, precision)
    model = load_model(config.weights)
    predictor = TilePredictor(
        model, device, precision, compile=config.compile, allow_tf32=config.allow_tf32
    )
    predictor.self_check(config.batch_size)

    source = asset.fused.transforms[config.level]
    provenance = _provenance(asset, regions, roi, plan, config, scaling) | extra
    out = create_ngff_array(
        out_uri, source=source, region=plan.region, chunks=(config.tile_output,) * 3,
        dtype="uint8", name="axon_probability", attrs={PROVENANCE_ATTR: provenance},
        overwrite=overwrite,
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


def prediction_roi(
    asset: ExaspimAsset, provenance: dict, *, growth: float | None = None
) -> RegionRoi:
    """Rebuild the (grown) region a prediction was made for from its provenance.

    ``growth`` overrides the recorded growth, e.g. to apply a margin to an older prediction. Its
    tiles cover only ``mask_dilation_um`` (50 µm) around the atlas region: enough for the ~41 µm
    shell of a 15% growth of MD, not for the ~58 µm of 20% (the outer voxels read as 0).
    """
    if "box" in provenance:
        roi = RegionRoi.box(
            np.asarray(provenance["box"]["lo_um"]), np.asarray(provenance["box"]["hi_um"])
        )
    elif "points" in provenance:
        points = provenance["points"]
        roi = RegionRoi.around_points(np.asarray(points["centers_um"]), points["side_um"])
    else:
        roi = RegionRoi.from_annotation(asset.annotation, provenance["label_ids"])
    return roi.grown(provenance.get("region_growth", 0.0) if growth is None else growth)


def _provenance(
    asset: ExaspimAsset,
    regions: Sequence[str | int],
    roi: RegionRoi,
    plan: RegionPlan,
    config: InferenceConfig,
    scaling: IntensityScaling,
) -> dict:
    if config.weights in REGISTRY:
        w = REGISTRY[config.weights]
        weights = {k: getattr(w, k) for k in ("name", "repo_id", "filename", "revision", "sha256")}
    else:
        weights = {"checkpoint": config.weights, "sha256": _sha256(config.weights)}
    return {
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "asset": asset.root,
        "annotation_asset": asset.annotation_root,
        "regions": list(regions),
        "label_ids": sorted(roi.label_ids),
        "region_growth": roi.growth,
        "region_volume_mm3": roi.volume_mm3,
        "level": config.level,
        "region_box": {"start": list(plan.region.start), "stop": list(plan.region.stop)},
        "config": config.model_dump(mode="json"),
        "weights": weights,
        "scaling": scaling.as_dict(),
    }


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()
