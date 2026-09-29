"""Training volumes: raw image + labels for the traced part of a region, stored as OME-Zarr."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

from exaspim_axon_segmentation import __version__
from exaspim_axon_segmentation.io import Box, CcfOntology, ExaspimAsset, LevelTransform, RegionRoi
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.train.labels import LabelConfig, build_labels, traced_lobe_box
from exaspim_axon_segmentation.train.swc import read_swc, resample

log = logging.getLogger(__name__)

VOLUME_ATTR = "exaspim_axon_training_volume"


def make_training_volume(
    asset_uri: str,
    regions: Sequence[str | int],
    swc_paths: Sequence[Path],
    out_dir: Path,
    *,
    level: int = 2,
    config: LabelConfig | None = None,
) -> dict:
    """Write ``raw.zarr`` and ``labels.zarr`` for the traced lobe of a region.

    Traces are resampled to a spacing below the level's voxel size before rasterization, so sparse
    SWCs still give connected centerlines (snap sparse traces to the image first).
    """
    config = config or LabelConfig()
    asset = ExaspimAsset(asset_uri)
    tf = asset.fused.transforms[level]
    roi = RegionRoi.from_annotation(asset.annotation, CcfOntology.load().resolve(regions))
    step = 0.5 * min(tf.scale)
    points = np.concatenate([resample(read_swc(p), step).zyx_um for p in swc_paths])
    pad = config.edge_radius_um + config.negative_clearance_um
    box, inside = traced_lobe_box(asset.fused, level, roi, points, pad_um=pad)
    labels, raw, stats = build_labels(asset.fused, level, box, inside, points, config)

    provenance = {
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "asset": asset.root,
        "regions": [str(r) for r in regions],
        "label_ids": sorted(roi.label_ids),
        "traces": [Path(p).name for p in swc_paths],
        "label_config": asdict(config),
        "legend": {
            "0": "ignore",
            "1": "axon",
            "2": "background",
            "3": "edge",
            "4": "hard_negative",
        },
        **stats,
    }
    write_training_volume(out_dir, raw, labels, tf, box, provenance)
    return provenance


def write_training_volume(
    out_dir: Path,
    raw: np.ndarray,
    labels: np.ndarray,
    grid: LevelTransform,
    box: Box,
    provenance: dict,
) -> None:
    """Write ``raw.zarr`` + ``labels.zarr`` (OME-NGFF, same grid) with shared provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = (64, 128, 128)
    for name, data in (("raw.zarr", raw), ("labels.zarr", labels)):
        array = create_ngff_array(
            str(out_dir / name), source=grid, region=box, chunks=chunks, dtype=data.dtype,
            name=name.removesuffix(".zarr"), attrs={VOLUME_ATTR: provenance}, overwrite=True,
        )  # fmt: skip
        array[:] = data
    log.info("Wrote %s (box %s): %s", out_dir, box.shape, provenance.get("voxels"))


def open_training_volume(out_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load raw, labels, and provenance of a training volume into memory."""
    raw = zarr.open_group(str(out_dir / "raw.zarr"), mode="r")
    labels = zarr.open_group(str(out_dir / "labels.zarr"), mode="r")
    return np.asarray(raw["0"][:]), np.asarray(labels["0"][:]), dict(labels.attrs[VOLUME_ATTR])
