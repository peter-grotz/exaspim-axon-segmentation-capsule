"""Signal-to-noise of axons in raw OME-Zarr tiles, with TRAILMAP as the axon detector.

For comparing acquisitions of one brain (e.g. 1x and 5x) at the same places: blocks are chosen at
random inside the tissue (or read from a saved list of physical centres), segmented with the
network, natively and/or resampled to a target voxel grid, and scored per block:

* signal: mean raw intensity of the detected axon voxels (p >= ``axon_p``),
* background: mean raw intensity of the tissue around them, sigma-clipped at 3 SD (detections
  dilated by ``guard_vox`` voxels are excluded; the darkest ``outside_pct`` percent are taken as
  outside the tissue and excluded too),
* noise: SD of that clipped background,
* SNR = (signal - background) / noise.

Means and a clipped SD, not medians and MADs: raw cameras quantize (e.g. 12-bit values stored in
steps of 16), and medians of quantized values only take a few steps. Medians are kept as well.

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
    resample_um: tuple[float, float, float] | None = (
        None  # e.g. the training grid (4, 2.992, 2.992)
    )
    axon_p: float = 0.5
    guard_vox: int = 2
    outside_pct: float = 10.0
    tile: int = 160
    qc: bool = True  # false-detection checks per block (qc/detection_qc.py)
    precision: str = "auto"  # "bf16" on GPUs that have it (checked against fp32: axon Dice 0.99)
    shell_vox: int = 6  # local background: a shell this thick around the (guarded) detections
    save_mips: bool = True
    qc_normalization: bool = True  # the QC's shared-scaling check: one more prediction per block


def _store(url: str):
    url = url.removeprefix("zarr://")
    options = {"anon": True} if url.startswith("s3://") else None
    return zarr.storage.FsspecStore.from_url(url, read_only=True, storage_options=options)


def _multiscales(root) -> list:
    attrs = root.attrs
    # OME-Zarr 0.4 keeps multiscales at the top level, 0.5 under "ome"
    return attrs["multiscales"] if "multiscales" in attrs else attrs["ome"]["multiscales"]


def open_ome(url: str, level: int):
    """Array (z, y, x view) and level transform (scale, translation, z y x µm) of an OME-Zarr."""
    root = zarr.open_group(_store(url), mode="r")
    ms = _multiscales(root)[0]
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
    levels = len(_multiscales(root)[0]["datasets"])
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


def tissue_block_centres(
    url: str,
    n: int,
    block_um,
    seed: int = 0,
    *,
    min_tissue: float = 0.9,
    mask_um: float = 16.0,
) -> tuple[NDArray, NDArray]:
    """Up to ``n`` non-overlapping blocks lying (>= ``min_tissue``) in tissue, physical centres.

    Tissue is the built-in mask of :mod:`exaspim_axon_segmentation.inference.image` (smoothed
    intensity above the out-of-sample background) on the first level with y/x voxels of at least
    ``mask_um``. Returns the centres (z, y, x µm) and each block's tissue fraction.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter

    root = zarr.open_group(_store(url), mode="r")
    n_levels = len(_multiscales(root)[0]["datasets"])
    level = next((lv for lv in range(n_levels) if open_ome(url, lv)[2][2] >= mask_um), n_levels - 1)
    array, lead, scale, translation = open_ome(url, level)
    v = _read(array, lead, (0, 0, 0), array.shape[lead:])
    smooth = gaussian_filter(v, 1.0)
    outside = np.percentile(smooth, 5)
    mad = 1.4826 * np.median(np.abs(smooth - np.median(smooth)))
    tissue = (smooth > outside + max(3.0 * mad, 1.0)).astype(np.float32)
    size = np.maximum(np.round(np.asarray(block_um) / scale).astype(int), 1)
    frac = uniform_filter(tissue, size=tuple(size), mode="constant")
    half = size // 2
    valid = np.zeros(frac.shape, bool)
    inner = tuple(slice(h, s - (k - h)) for h, k, s in zip(half, size, frac.shape, strict=True))
    valid[inner] = frac[inner] >= min_tissue
    cand = np.argwhere(valid)
    rng = np.random.default_rng(seed)
    chosen: list[NDArray] = []
    for c in cand[rng.permutation(len(cand))]:
        if all(np.any(np.abs(c - q) >= size) for q in chosen):  # boxes do not intersect
            chosen.append(c)
            if len(chosen) == n:
                break
    idx = np.array(chosen, int).reshape(-1, 3)
    log.info("Tissue blocks on level %d (%s um): %.1f%% tissue, %d candidates, %d chosen",
             level, scale.round(1).tolist(), 100 * tissue.mean(), len(cand), len(idx))  # fmt: skip
    return idx * scale + translation, frac[tuple(idx.T)] if len(idx) else np.zeros(0)


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
    values = raw[bg].astype(np.float64)
    for _ in range(5):  # sigma-clip the background
        m, sd = values.mean(), values.std()
        values = values[np.abs(values - m) <= 3 * sd]
    s, b, sd = float(raw[axon].mean()), float(values.mean()), float(values.std())
    sm, bm = float(np.median(raw[axon])), float(np.median(raw[bg]))
    sdm = float(1.4826 * np.median(np.abs(raw[bg] - bm)))
    step = int(np.gcd.reduce(np.unique(raw[::4, ::4, ::4].astype(np.int64))[:5000]))
    # local background: a thin shell around the detections, so gradients/stripes far away do not
    # count as noise
    shell = binary_dilation(guard, iterations=config.shell_vox) & ~guard
    shell &= raw > np.percentile(raw, config.outside_pct)
    local = {"local_background": None, "local_noise_sd": None, "local_snr": None}
    if shell.sum() >= 1000:
        lv = raw[shell].astype(np.float64)
        for _ in range(5):
            m, d = lv.mean(), lv.std()
            lv = lv[np.abs(lv - m) <= 3 * d]
        lb, lsd = float(lv.mean()), float(lv.std())
        local = {
            "local_background": lb,
            "local_noise_sd": lsd,
            "local_snr": (s - lb) / max(lsd, 1e-6),
        }
    whole = {"signal": s, "background": b, "noise_sd": sd, "snr": (s - b) / max(sd, 1e-6),
             "median_snr": (sm - bm) / max(sdm, 1e-6), "quantization_step": step}  # fmt: skip
    return out | whole | local


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
    precision = resolve_precision(config.precision) if dev.type == "cuda" else torch.float32
    predictor = TilePredictor(load_model(weights), dev, precision)
    from exaspim_axon_segmentation.inference.normalize import scaling_from_samples
    from exaspim_axon_segmentation.qc.detection_qc import DetectionQcConfig, block_qc

    half = np.asarray(config.block_um) / 2
    boxes = []
    for centre in np.atleast_2d(centres_um):
        lo = np.clip(np.floor((centre - half - translation) / scale).astype(int), 0, shape)
        hi = np.clip(np.ceil((centre + half - translation) / scale).astype(int), 0, shape)
        boxes.append((lo, hi))
    # one acquisition-wide normalization, from central 64^3 sub-crops pooled over all blocks
    # (blocks are then read one at a time, so memory stays at one block)
    pooled = []
    for lo, hi in boxes:
        c = (lo + hi) // 2
        pooled.append(_read(array, lead, np.maximum(c - 32, lo), np.minimum(c + 32, hi)).ravel())
    try:
        shared = scaling_from_samples(np.concatenate(pooled), norm) if pooled else None
    except ValueError:
        shared = None
    log.info("shared scaling: %s", None if shared is None else (shared.offset, shared.scale))
    qc_cfg = DetectionQcConfig()
    rows = []
    for k, centre in enumerate(np.atleast_2d(centres_um)):
        t0 = time.time()
        lo, hi = boxes[k]
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
            try:
                inside = vol > np.percentile(vol, config.outside_pct)
                scaling = estimate_scaling_array(vol, inside, norm)
            except ValueError as error:  # no foreground: nothing to scale by, so nothing to detect
                row[name] = {"skipped": f"no foreground: {error}", "verdict": "no_signal"}
                log.warning("block %d %s: %s; skipped", k, name, error)
                continue
            prob = _predict(predictor, scaling(vol), config.tile)
            extra = {
                "voxel_um": np.asarray(vox).tolist(),
                "scaling": [scaling.offset, scaling.scale],
            }
            row[name] = score(vol, prob, config) | extra
            if config.qc and name == "native":
                row[name]["qc"] = block_qc(
                    vol,
                    lambda v: _predict(predictor, v, config.tile),
                    scaling,
                    shared_scaling=shared if config.qc_normalization else None,
                    prob=prob,
                    config=qc_cfg,
                )
            if config.save_mips:
                mips = {
                    "raw_mip": vol.max(0),
                    "prob_mip": np.round(prob.max(0) * 255).astype(np.uint8),
                }
                np.savez_compressed(out_dir / f"block{k:03d}_{name}.npz", **mips)
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
