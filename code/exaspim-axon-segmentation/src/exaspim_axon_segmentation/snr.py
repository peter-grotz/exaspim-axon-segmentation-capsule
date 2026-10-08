"""Signal-to-noise of axons in raw OME-Zarr tiles, with TRAILMAP as the axon detector.

For comparing acquisitions of one brain (e.g. 1x and 5x) at the same places: blocks are chosen at
random inside the tissue (or read from a saved list of physical centres), segmented with the
network, natively and/or resampled to a target voxel grid, and scored per block:

* signal: median raw intensity of the detected axon voxels (p >= ``axon_p``),
* background: median raw intensity of the tissue around them (detections dilated by
  ``guard_vox`` voxels are excluded; the darkest ``outside_pct`` percent are taken as outside
  the tissue and excluded too),
* noise: robust SD (1.4826 x MAD) of that background,
* SNR = (signal - background) / noise.

Block centres are recorded in physical µm (the OME-Zarr scale and translation of level 0), so a
second acquisition can be scored at the same locations.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion, zoom

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnrConfig:
    """Settings of :func:`snr_blocks`."""

    block_um: tuple[float, float, float] = (1280.0, 1440.0, 1440.0)  # z, y, x
    resample_um: tuple[float, float, float] | None = (4.0, 2.992, 2.992)  # training grid
    axon_p: float = 0.5
    guard_vox: int = 2
    outside_pct: float = 10.0
    tile: int = 160


def _store(url: str):
    url = url.removeprefix("zarr://")
    options = {"anon": True} if url.startswith("s3://") else None
    return zarr.storage.FsspecStore.from_url(url, read_only=True, storage_options=options)


def open_ome(url: str, level: int):
    """Array (z, y, x view) and level transform (scale, translation, z y x µm) of an OME-Zarr."""
    root = zarr.open_group(_store(url), mode="r")
    ms = root.attrs["multiscales"][0]
    axes = [a["name"] for a in ms["axes"]]
    ds = ms["datasets"][level]
    tf = {t["type"]: t[t["type"]] for t in ds["coordinateTransformations"]}
    scale = np.asarray(tf["scale"], float)[-3:]
    translation = np.asarray(tf.get("translation", [0.0] * len(axes)), float)[-3:]
    return root[ds["path"]], len(axes) - 3, scale, translation


def _read(array, lead: int, lo, hi) -> NDArray[np.float32]:
    key = (0,) * lead + tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    return np.asarray(array[key]).astype(np.float32)


def random_centres(url: str, n: int, seed: int, overview_level: int | None = None) -> NDArray:
    """``n`` physical (z, y, x µm) points inside the tissue, from a low-resolution overview."""
    root = zarr.open_group(_store(url), mode="r")
    levels = len(root.attrs["multiscales"][0]["datasets"])
    level = levels - 1 if overview_level is None else overview_level
    array, lead, scale, translation = open_ome(url, level)
    ov = _read(array, lead, (0, 0, 0), array.shape[lead:])
    med = np.median(ov)
    mad = max(1.4826 * np.median(np.abs(ov - med)), 2.0)
    tissue = binary_erosion(ov > med + 4 * mad, iterations=3)
    cand = np.argwhere(tissue)
    pick = cand[
        np.random.default_rng(seed).choice(len(cand), size=min(n, len(cand)), replace=False)
    ]
    return pick * scale + translation + scale / 2


def _predict(predictor, vol: NDArray[np.float32], tile: int) -> NDArray[np.float32]:
    from exaspim_axon_segmentation.model.unet import CONTEXT

    out = tile - 2 * CONTEXT
    pad = [(CONTEXT, CONTEXT + (-s) % out) for s in vol.shape]
    v = np.pad(vol, pad, mode="reflect")
    prob = np.zeros(tuple(s + (-s) % out for s in vol.shape), np.float32)
    for z in range(0, prob.shape[0], out):
        for y in range(0, prob.shape[1], out):
            for x in range(0, prob.shape[2], out):
                t = v[z : z + tile, y : y + tile, x : x + tile]
                prob[z : z + out, y : y + out, x : x + out] = predictor(t[None])[0]
    return prob[: vol.shape[0], : vol.shape[1], : vol.shape[2]]


def score(raw: NDArray[np.float32], prob: NDArray[np.float32], config: SnrConfig) -> dict:
    """Signal, background, noise and SNR of the detected axons in one block."""
    axon = prob >= config.axon_p
    guard = binary_dilation(axon, iterations=config.guard_vox) if axon.any() else axon
    bg = ~guard & (raw > np.percentile(raw, config.outside_pct))
    out = {"axon_voxels": int(axon.sum()), "axon_fraction": float(axon.mean())}
    if axon.sum() < 50 or bg.sum() < 1000:
        return out | {"signal": None, "background": None, "noise_sd": None, "snr": None}
    s = float(np.median(raw[axon]))
    b = float(np.median(raw[bg]))
    sd = float(1.4826 * np.median(np.abs(raw[bg] - b)))
    return out | {"signal": s, "background": b, "noise_sd": sd, "snr": (s - b) / max(sd, 1e-6)}


def snr_blocks(
    url: str,
    centres_um: NDArray,
    out_dir: Path,
    weights: str,
    config: SnrConfig = SnrConfig(),  # noqa: B008
    *,
    level: int = 0,
    device: str = "auto",
) -> list[dict]:
    """Segment and score one block around each physical centre; writes JSON and preview MIPs."""
    import torch

    from exaspim_axon_segmentation.config import InferenceConfig
    from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array
    from exaspim_axon_segmentation.inference.predict import (
        TilePredictor,
        resolve_device,
        resolve_precision,
    )
    from exaspim_axon_segmentation.model import load_model

    out_dir.mkdir(parents=True, exist_ok=True)
    array, lead, scale, translation = open_ome(url, level)
    shape = np.asarray(array.shape[lead:])
    norm = InferenceConfig(weights=weights).normalization
    dev = resolve_device(device)
    precision = resolve_precision("auto") if dev.type == "cuda" else torch.float32
    predictor = TilePredictor(load_model(weights), dev, precision)
    half = np.asarray(config.block_um) / 2
    rows = []
    for k, centre in enumerate(np.atleast_2d(centres_um)):
        t0 = time.time()
        lo = np.clip(np.floor((centre - half - translation) / scale).astype(int), 0, shape)
        hi = np.clip(np.ceil((centre + half - translation) / scale).astype(int), 0, shape)
        raw = _read(array, lead, lo, hi)
        row = {"block": k, "centre_um": np.round(centre, 1).tolist(), "lo_vox": lo.tolist(),
               "shape": list(raw.shape), "voxel_um": scale.tolist()}  # fmt: skip
        variants = {"native": (raw, scale)}
        if config.resample_um is not None:
            factor = scale / np.asarray(config.resample_um)
            variants["resampled"] = (zoom(raw, factor, order=1).astype(np.float32),
                                     np.asarray(config.resample_um))  # fmt: skip
        for name, (vol, vox) in variants.items():
            if min(vol.shape) < 64:
                row[name] = {"skipped": f"block {vol.shape} smaller than the network input"}
                continue
            scaling = estimate_scaling_array(
                vol, vol > np.percentile(vol, config.outside_pct), norm
            )
            prob = _predict(predictor, scaling(vol), config.tile)
            extra = {
                "voxel_um": np.asarray(vox).tolist(),
                "scaling": [scaling.offset, scaling.scale],
            }
            row[name] = score(vol, prob, config) | extra
            np.savez_compressed(out_dir / f"block{k:03d}_{name}.npz", raw_mip=vol.max(0),
                                prob_mip=np.round(prob.max(0) * 255).astype(np.uint8))  # fmt: skip
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        log.info(
            "block %d: %s",
            k,
            {n: row[n].get("snr") for n in variants if isinstance(row.get(n), dict)},
        )
        summary = {"url": url, "level": level, "config": asdict(config), "blocks": rows}
        (out_dir / "snr_blocks.json").write_text(json.dumps(summary, indent=1))
    return rows
