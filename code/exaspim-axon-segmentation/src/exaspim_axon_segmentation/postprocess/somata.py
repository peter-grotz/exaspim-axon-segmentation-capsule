"""Soma detections produced by the exaSPIM soma-detection capsule."""

from __future__ import annotations

import ast
import csv
import io

import fsspec
import numpy as np

from exaspim_axon_segmentation.io import ExaspimAsset
from exaspim_axon_segmentation.postprocess.cleanup import Soma

SOMA_CSV = "soma_detection/soma_locations.csv"


def load_somata(asset: ExaspimAsset) -> list[Soma]:
    """Read ``soma_locations.csv`` (``xyz_raw`` is (x, y, z) in micrometers; radii are averaged)."""
    uri = f"{asset.root.rstrip('/')}/{SOMA_CSV}"
    options = {"anon": asset.anon} if uri.startswith("s3://") else {}
    with fsspec.open(uri, "rb", **options) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
    radius_key = next(k for k in rows[0] if k.startswith("Radii"))
    somata = []
    for row in rows:
        x, y, z = ast.literal_eval(row["xyz_raw"])
        radii = ast.literal_eval(row[radius_key])
        somata.append(Soma(center_um=(z, y, x), radius_um=float(np.mean(radii))))
    return somata
