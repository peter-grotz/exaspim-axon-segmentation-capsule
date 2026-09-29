"""Training volumes from traced exaSPIM blocks on GCS (``<root>/<variant>/block_NNN``).

Each block holds a 1024³ level-0 crop (``input.zarr``, multiscale), a per-neuron fill mask of the
traced axons (``Fill_Label_Mask.zarr``, same grid), ``metadata.json`` (source image and origin),
and complete axon traces in ``<root>/swcs/block_NNN/*.swc`` — SNT exports in block-local level-0
voxel coordinates, ordered (x, y, z).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import gcsfs
import numpy as np

from exaspim_axon_segmentation import __version__
from exaspim_axon_segmentation.io import Box, NgffImage
from exaspim_axon_segmentation.train.labels import LabelConfig, build_labels
from exaspim_axon_segmentation.train.swc import parse_swc, resample
from exaspim_axon_segmentation.train.volume import write_training_volume

log = logging.getLogger(__name__)


def block_trace_points(fs: gcsfs.GCSFileSystem, swc_dir: str, image: NgffImage, step_um: float):
    """Dense trace points (physical z, y, x µm, block frame) from all SWCs of one block."""
    points = []
    for path in sorted(p for p in fs.ls(swc_dir) if p.endswith(".swc")):
        swc = parse_swc(fs.cat(path).decode())
        # SNT stores block-local level-0 voxel indices as (x, y, z): convert to physical µm first.
        zyx_um = image.transforms[0].to_physical(swc.xyz_um[:, ::-1])
        points.append(resample(replace(swc, xyz_um=zyx_um[:, ::-1]), step_um).zyx_um)
    return np.concatenate(points) if points else np.empty((0, 3))


def make_block_volumes(
    root: str,
    out_dir: Path,
    *,
    token: dict | str,
    variant: str = "blocks",
    level: int = 2,
    blocks: Sequence[str] | None = None,
    config: LabelConfig | None = None,
) -> list[dict]:
    """Write one training volume per traced block; returns their provenance records.

    Blocks are completely traced, so the whole block is labelled (no ``IGNORE`` region) and bright
    untraced structure becomes hard negatives, exactly as for a traced region.
    """
    config = config or LabelConfig()
    root = root.removeprefix("gs://").rstrip("/")
    fs = gcsfs.GCSFileSystem(token=token)
    traced = {p.rstrip("/").split("/")[-1] for p in fs.ls(f"{root}/swcs")}
    names = sorted(blocks or (p.rstrip("/").split("/")[-1] for p in fs.ls(f"{root}/{variant}")))
    records = []
    for name in names:
        if name not in traced:
            log.warning("Skipping %s: no traces", name)
            continue
        base = f"{root}/{variant}/{name}"
        meta = json.loads(fs.cat(f"{base}/metadata.json"))
        image = NgffImage(f"gs://{base}/input.zarr", storage_options={"token": token})
        tf = image.transforms[level]
        points = block_trace_points(fs, f"{root}/swcs/{name}", image, 0.5 * min(tf.scale))
        box = Box.full(image.shape(level))
        fill = NgffImage(f"gs://{base}/Fill_Label_Mask.zarr", storage_options={"token": token})
        labels, raw, stats = build_labels(
            image, level, box, np.ones(box.shape, dtype=bool), points, config,
            traced_mask=fill.read(level) > 0,
        )  # fmt: skip
        record = {
            "version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": f"gs://{base}",
            "block": name,
            "brain": meta["image_url"].rstrip("/").split("/")[-2].split("_")[1],
            "image_url": meta["image_url"],
            "chunk_origin_xyz": meta.get("chunk_origin"),
            "label_config": asdict(config),
            "legend": {"0": "ignore", "1": "axon", "2": "background", "3": "edge",
                       "4": "hard_negative"},
            **stats,
        }  # fmt: skip
        write_training_volume(out_dir / name, raw, labels, tf, box, record)
        records.append(record)
    return records
