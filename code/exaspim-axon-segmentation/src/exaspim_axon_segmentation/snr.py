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
from scipy.ndimage import binary_erosion, zoom

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
    save_histograms: bool = True  # per block: counts of detected / background voxels per DN
    qc_normalization: bool = True  # the QC's shared-scaling check: one more prediction per block
    save_probability: bool = True  # per tile: probability.zarr on the level grid, blocks only
    batch: int = 4  # network tiles per GPU call (halved automatically if memory runs out)


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


_NETWORK_SECONDS = [0.0]  # time inside the network calls, for profiling


class _Laps:
    """Seconds per step of one block, for profiling."""

    def __init__(self, seconds: dict[str, float]):
        self.seconds = seconds
        self.tick = time.time()

    def lap(self, key: str) -> None:
        now = time.time()
        self.seconds[key] = self.seconds.get(key, 0.0) + now - self.tick
        self.tick = now


def _predict(predictor, vol: NDArray[np.float32], tile: int, batch: int = 1) -> NDArray[np.float32]:
    """Network probabilities for a whole volume, in ``tile``-sided tiles sent ``batch`` at a time.

    Halves the batch when the GPU runs out of memory.
    """
    import torch

    from exaspim_axon_segmentation.model.unet import CONTEXT

    out = tile - 2 * CONTEXT
    pad = [(CONTEXT, CONTEXT + (-s) % out) for s in vol.shape]
    v = np.pad(vol, pad, mode="reflect")
    prob = np.zeros(tuple(s + (-s) % out for s in vol.shape), np.float32)
    starts = list(np.ndindex(*(n // out for n in prob.shape)))
    i = 0
    while i < len(starts):
        group = [tuple(int(c) * out for c in s) for s in starts[i : i + batch]]
        tiles = np.stack([v[z : z + tile, y : y + tile, x : x + tile] for z, y, x in group])
        t0 = time.time()
        try:
            res = predictor(tiles)
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            torch.cuda.empty_cache()
            batch //= 2
            log.warning("GPU out of memory; batch now %d", batch)
            continue
        _NETWORK_SECONDS[0] += time.time() - t0
        for (z, y, x), r in zip(group, res, strict=True):
            prob[z : z + out, y : y + out, x : x + out] = r
        i += len(group)
    return prob[: vol.shape[0], : vol.shape[1], : vol.shape[2]]


def _sub(values: NDArray, n: int = 2_000_000) -> NDArray:
    """Return a fixed random subset of at most ``n`` values, for percentiles and medians."""
    flat = values.ravel()
    if flat.size <= n:
        return flat
    return flat[np.random.default_rng(0).integers(0, flat.size, n)]


def score(raw: NDArray[np.float32], prob: NDArray[np.float32], config: SnrConfig) -> dict:
    """Signal, background, noise and SNR of the detected axons in one block.

    Means and SDs use every voxel; percentiles and medians use a 2 M-voxel random subset.
    """
    return _score(raw, prob, config)[0]


def _score(raw, prob, config: SnrConfig) -> tuple[dict, NDArray[np.bool_]]:
    """:func:`score` and the guarded detections (reused by the QC)."""
    from exaspim_axon_segmentation.qc.detection_qc import _clipped, sparse_dilation

    axon = prob >= config.axon_p
    guard = sparse_dilation(axon, config.guard_vox)
    outside = np.percentile(_sub(raw), config.outside_pct)
    bg = ~guard & (raw > outside)
    out = {"axon_voxels": int(axon.sum()), "axon_fraction": float(axon.mean())}
    if axon.sum() < 50 or bg.sum() < 1000:
        return out | {"signal": None, "background": None, "noise_sd": None, "snr": None}, guard
    bgv_all = raw[bg]
    b, sd = _clipped(bgv_all)  # sigma-clipped background
    s = float(raw[axon].mean())
    bgv = _sub(bgv_all)
    sm, bm = float(np.median(_sub(raw[axon]))), float(np.median(bgv))
    sdm = float(1.4826 * np.median(np.abs(bgv - bm)))
    step = int(np.gcd.reduce(np.unique(raw[::4, ::4, ::4].astype(np.int64))[:5000]))
    # local background: a thin shell around the detections, so gradients/stripes far away do not
    # count as noise
    shell = sparse_dilation(guard, config.shell_vox) & ~guard
    shell &= raw > outside
    local = {"local_background": None, "local_noise_sd": None, "local_snr": None}
    if shell.sum() >= 1000:
        lb, lsd = _clipped(raw[shell])
        local = {
            "local_background": lb,
            "local_noise_sd": lsd,
            "local_snr": (s - lb) / max(lsd, 1e-6),
        }
    whole = {"signal": s, "background": b, "noise_sd": sd, "snr": (s - b) / max(sd, 1e-6),
             "median_snr": (sm - bm) / max(sdm, 1e-6), "quantization_step": step}  # fmt: skip
    return out | whole | local, guard


def _block_boxes(shape, scale, translation, centres_um, block_um) -> list[tuple[NDArray, NDArray]]:
    half = np.asarray(block_um) / 2
    boxes = []
    for centre in np.atleast_2d(centres_um):
        lo = np.clip(np.floor((centre - half - translation) / scale).astype(int), 0, shape)
        hi = np.clip(np.ceil((centre + half - translation) / scale).astype(int), 0, shape)
        boxes.append((lo, hi))
    return boxes


def _load_block(array, lead: int, lo, hi, scale, config: SnrConfig, norm) -> dict:
    """Read one block and normalize each variant (runs in the reader thread)."""
    from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array

    t0 = time.time()
    raw = _read(array, lead, lo, hi)
    t_read = time.time() - t0
    variants = {"native": (raw, scale)}
    if config.resample_um is not None:
        factor = scale / np.asarray(config.resample_um)
        variants["resampled"] = (zoom(raw, factor, order=1).astype(np.float32),
                                 np.asarray(config.resample_um))  # fmt: skip
    out = {}
    for name, (vol, vox) in variants.items():
        if min(vol.shape) < 64:
            out[name] = (vol, vox, None, f"block {vol.shape} smaller than the network input")
            continue
        try:
            inside = vol > np.percentile(vol, config.outside_pct)
            out[name] = (vol, vox, estimate_scaling_array(vol, inside, norm), None)
        except ValueError as error:  # no foreground: nothing to scale by, so nothing to detect
            out[name] = (vol, vox, None, error)
    return {"raw": raw, "variants": out, "read_s": t_read, "normalize_s": time.time() - t0 - t_read}


def prefetch_first_block(url: str, centres_um: NDArray, config: SnrConfig, weights: str, *,
                         level: int = 0) -> dict | None:  # fmt: skip
    """Read and normalize the first block of a tile ahead of :func:`snr_blocks` (``first=``)."""
    from exaspim_axon_segmentation.config import InferenceConfig

    if len(np.atleast_2d(centres_um)) == 0 or np.size(centres_um) == 0:
        return None
    array, lead, scale, translation = open_ome(url, level)
    lo, hi = _block_boxes(np.asarray(array.shape[lead:]), scale, translation, centres_um,
                          config.block_um)[0]  # fmt: skip
    norm = InferenceConfig(weights=weights).normalization
    return {"lo": lo, "hi": hi} | _load_block(array, lead, lo, hi, scale, config, norm)


_PREDICTORS: dict = {}  # one network per (weights, device, precision), reused across tiles


def snr_blocks(
    url: str,
    centres_um: NDArray,
    out_dir: Path,
    weights: str,
    config: SnrConfig = SnrConfig(),  # noqa: B008
    *,
    level: int = 0,
    device: str = "auto",
    first: dict | None = None,
) -> list[dict]:
    """Segment and score one block around each physical centre; writes JSON and preview MIPs.

    Three stages overlap: the next block is read and normalized in one thread, the network runs on
    the current block, and the previous block is scored and written in another thread. ``first``
    is the first block already loaded by :func:`prefetch_first_block`.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from exaspim_axon_segmentation.config import InferenceConfig
    from exaspim_axon_segmentation.inference.normalize import scaling_from_samples
    from exaspim_axon_segmentation.inference.predict import (
        TilePredictor,
        resolve_device,
        resolve_precision,
    )
    from exaspim_axon_segmentation.io import Box
    from exaspim_axon_segmentation.io.ngff import LevelTransform
    from exaspim_axon_segmentation.io.writer import create_ngff_array
    from exaspim_axon_segmentation.model import load_model
    from exaspim_axon_segmentation.qc.detection_qc import DetectionQcConfig, block_qc

    out_dir.mkdir(parents=True, exist_ok=True)
    array, lead, scale, translation = open_ome(url, level)
    shape = np.asarray(array.shape[lead:])
    norm = InferenceConfig(weights=weights).normalization
    dev = resolve_device(device)
    precision = resolve_precision(config.precision) if dev.type == "cuda" else torch.float32
    key = (weights, str(dev), str(precision))
    if key not in _PREDICTORS:
        _PREDICTORS[key] = TilePredictor(load_model(weights), dev, precision)
    gpu = threading.Lock()  # the network is called from the main and the scoring thread

    def network(batch):
        with gpu:
            return _PREDICTORS[key](batch)

    def predict(v):
        return _predict(network, v, config.tile, config.batch)

    boxes = _block_boxes(shape, scale, translation, centres_um, config.block_um)
    shared = None
    if config.qc and config.qc_normalization:
        # one acquisition-wide normalization, from central 64^3 sub-crops pooled over all blocks
        t_shared = time.time()
        pooled = []
        for lo, hi in boxes:
            c = (lo + hi) // 2
            pooled.append(
                _read(array, lead, np.maximum(c - 32, lo), np.minimum(c + 32, hi)).ravel()
            )
        try:
            shared = scaling_from_samples(np.concatenate(pooled), norm) if pooled else None
        except ValueError:
            shared = None
        log.info(
            "shared scaling: %s (%.1f s)",
            None if shared is None else (shared.offset, shared.scale),
            time.time() - t_shared,
        )
    probability = None
    if config.save_probability and boxes:
        probability = create_ngff_array(
            str(out_dir / "probability.zarr"),
            source=LevelTransform(tuple(scale.tolist()), tuple(translation.tolist())),
            region=Box.from_arrays((0, 0, 0), shape),
            chunks=(64, 128, 128),
            dtype="uint8",
            name="trailmap_probability",
            attrs={"source": url, "level": level, "weights": str(weights),
                   "note": "p x 255; only the sampled blocks are written, elsewhere 0"},
            overwrite=True,
        )  # fmt: skip
    qc_cfg = DetectionQcConfig()
    rows: list[dict] = []

    def finish(k, centre, loaded, probs, main_s):
        """Score, QC and write one block (scoring thread)."""
        laps = _Laps({})
        lo, hi = boxes[k]
        raw = loaded["raw"]
        row = {"block": k, "centre_um": np.round(centre, 1).tolist(), "lo_vox": lo.tolist(),
               "shape": list(raw.shape), "voxel_um": scale.tolist()}  # fmt: skip
        for name, (vol, vox, scaling, error) in loaded["variants"].items():
            if scaling is None:
                if isinstance(error, str):
                    row[name] = {"skipped": error}
                else:
                    row[name] = {"skipped": f"no foreground: {error}", "verdict": "no_signal"}
                    log.warning("block %d %s: %s; skipped", k, name, error)
                continue
            prob = probs[name]
            extra = {
                "voxel_um": np.asarray(vox).tolist(),
                "scaling": [scaling.offset, scaling.scale],
            }
            metrics, guard = _score(vol, prob, config)
            row[name] = metrics | extra
            laps.lap("score")
            if config.save_histograms:
                det = prob >= config.axon_p
                dn = np.clip(vol, 0, 65535).astype(np.uint16)
                fg = np.bincount(dn[det], minlength=65536)
                bg = (np.bincount(dn.ravel(), minlength=65536) - fg).astype(np.uint32)
                fg = fg.astype(np.uint32)
                np.savez_compressed(out_dir / f"block{k:03d}_{name}_hist.npz", foreground=fg,
                                    background=bg)  # fmt: skip
                laps.lap("histograms")
            if config.qc and name == "native":
                same = qc_cfg.axon_p == config.axon_p and qc_cfg.guard_vox == config.guard_vox
                row[name]["qc"] = block_qc(vol, predict, scaling, shared_scaling=shared,
                                           prob=prob, config=qc_cfg,
                                           guard=guard if same else None)  # fmt: skip
                laps.lap("qc")
            p8 = np.round(prob * 255).astype(np.uint8)
            if probability is not None and name == "native":
                probability[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))] = p8
                laps.lap("probability_write")
            if config.save_mips:
                np.savez_compressed(out_dir / f"block{k:03d}_{name}.npz", raw_mip=vol.max(0),
                                    prob_mip=p8.max(0))  # fmt: skip
                laps.lap("mips")
        row["seconds"] = round(main_s, 1)  # time this block held the pipeline (read wait + GPU)
        row["timing"] = {key: round(v, 2) for key, v in
                         (main_timing[k] | {"read": loaded["read_s"],
                                            "normalize": loaded["normalize_s"]}
                          | laps.seconds).items()}  # fmt: skip
        rows.append(row)
        log.info(
            "block %d: %s",
            k,
            {n: row[n].get("snr") for n in loaded["variants"] if isinstance(row.get(n), dict)},
        )
        summary = {"url": url, "level": level, "config": asdict(config), "blocks": rows}
        (out_dir / "snr_blocks.json").write_text(json.dumps(summary, indent=1))

    reader, scorer = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    main_timing: dict[int, dict] = {}
    try:
        if first is not None and boxes and np.array_equal(first["lo"], boxes[0][0]):
            nxt = reader.submit(lambda: first)
        else:
            nxt = (
                reader.submit(_load_block, array, lead, *boxes[0], scale, config, norm)
                if boxes
                else None
            )
        pending = None
        for k, centre in enumerate(np.atleast_2d(centres_um)[: len(boxes)]):
            t0 = time.time()
            loaded = nxt.result()
            read_wait = time.time() - t0
            if k + 1 < len(boxes):
                nxt = reader.submit(_load_block, array, lead, *boxes[k + 1], scale, config, norm)
            net0 = _NETWORK_SECONDS[0]
            t1 = time.time()
            probs = {
                name: predict(scaling(vol))
                for name, (vol, _, scaling, _) in loaded["variants"].items()
                if scaling is not None
            }
            t2 = time.time()
            if pending is not None:  # at most one block waits for scoring, so memory stays bounded
                pending.result()
            main_timing[k] = {"read_wait": read_wait, "predict": t2 - t1,
                              "network_calls": _NETWORK_SECONDS[0] - net0,
                              "score_wait": time.time() - t2}  # fmt: skip
            pending = scorer.submit(finish, k, centre, loaded, probs, time.time() - t0)
        if pending is not None:
            pending.result()
    finally:
        reader.shutdown()
        scorer.shutdown()
    return rows
