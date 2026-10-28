"""Is a block's axon detection trustworthy? Cheap checks to run per block during acquisition.

A *false-detection* block is one where the detector fires on noise or texture rather than on
neurites (typically a dim block whose own intensity normalization stretches the noise). Four
checks, none of which needs a skeleton or a second acquisition:

1. **signal present** (before any network): the block's bright tail must stand above its noise,
   ``(p99.9 - median) / noise >= signal_min``, noise = 1.4826 x MAD. A block that fails is
   skipped: any detection there would be false.
2. **detection contrast**: detected voxels (p >= ``axon_p``) must be brighter than the tissue
   around them, ``SNR >= snr_min`` (mean signal vs sigma-clipped background, see
   :func:`contrast`).
3. **noise control**: the detector run on Gaussian noise with the block's own background mean and
   SD (on a ``control_vox``-sided crop, normalized the same way) must detect far less than on the
   block, ``control rate / block rate <= control_ratio_max``.
4. **normalization sensitivity** (when an acquisition-wide intensity scaling is given): detections
   with the block's own scaling and with the shared one must agree within
   ``norm_ratio_range``; false detections depend on stretching one block's noise.

``block_qc`` returns every measurement, the failed checks and a verdict (``ok``, ``no_signal``,
``false_detection``). ``predict`` is any callable mapping a normalized float32 volume to a
same-shaped probability volume.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation


@dataclass(frozen=True)
class DetectionQcConfig:
    """Thresholds of :func:`block_qc`."""

    axon_p: float = 0.5
    signal_min: float = 10.0  # (p99.9 - median) / noise
    snr_min: float = 3.0
    control_ratio_max: float = 0.5
    control_vox: int = 96
    norm_ratio_range: tuple[float, float] = (0.5, 2.0)
    guard_vox: int = 2
    outside_pct: float = 10.0
    seed: int = 0


def signal_present(raw: NDArray, config: DetectionQcConfig | None = None) -> float:
    """Return ``(p99.9 - median) / (1.4826 MAD)`` of a block: its bright tail above the noise."""
    v = raw[:: max(1, raw.shape[0] // 32)].astype(np.float64).ravel()
    med = np.median(v)
    noise = max(1.4826 * np.median(np.abs(v - med)), 1e-6)
    return float((np.percentile(v, 99.9) - med) / noise)


def sparse_dilation(mask: NDArray[np.bool_], iterations: int, cell: int = 64) -> NDArray[np.bool_]:
    """``binary_dilation(mask, iterations=...)``, computed only in cells near True voxels.

    Same voxels as the full dilation (each iteration grows by one face neighbour, so a voxel is set
    when a True voxel lies within ``iterations`` steps); cells with nothing within reach are
    skipped, which makes sparse masks (axons fill < 1% of a block) much cheaper.
    """
    if iterations < 1 or not mask.any():
        return mask.copy()
    shape = np.asarray(mask.shape)
    grid = -(-shape // cell)
    pad = [(0, g * cell - n) for g, n in zip(grid, shape, strict=True)]
    occupied = np.pad(mask, pad).reshape(grid[0], cell, grid[1], cell, grid[2], cell).any((1, 3, 5))
    reach = -(-iterations // cell)  # cells a detection can grow into
    near = binary_dilation(occupied, np.ones((3, 3, 3), bool), iterations=reach)
    out = np.zeros_like(mask)
    for c in np.argwhere(near):
        lo, hi = c * cell, np.minimum((c + 1) * cell, shape)
        elo, ehi = np.maximum(lo - iterations, 0), np.minimum(hi + iterations, shape)
        sub = binary_dilation(
            mask[elo[0] : ehi[0], elo[1] : ehi[1], elo[2] : ehi[2]], iterations=iterations
        )
        a, b = lo - elo, hi - elo
        out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = sub[
            a[0] : b[0], a[1] : b[1], a[2] : b[2]
        ]
    return out


def clipped_counts(counts: NDArray) -> tuple[float, float]:
    """:func:`_clipped` of the integer values whose counts are ``counts`` (index = value)."""
    counts = np.asarray(counts).astype(np.float64)
    v = np.arange(counts.size, dtype=np.float64)
    keep = counts > 0
    for _ in range(5):
        c, x = counts[keep], v[keep]
        n = c.sum()
        m = (c * x).sum() / n
        sd = np.sqrt((c * (x - m) ** 2).sum() / n)
        keep &= np.abs(v - m) <= 3 * sd
    c, x = counts[keep], v[keep]
    n = c.sum()
    m = (c * x).sum() / n
    return float(m), float(np.sqrt((c * (x - m) ** 2).sum() / n))


def _clipped(values: NDArray) -> tuple[float, float]:
    """Mean and SD after 5 rounds of 3-SD clipping.

    Integer values (raw camera counts) are clipped on their histogram, which gives the same result
    as clipping the values themselves, in milliseconds.
    """
    values = np.asarray(values)
    if (
        values.size
        and values.min() >= 0
        and values.max() < 2**20
        and np.all(values == np.floor(values))
    ):
        return clipped_counts(np.bincount(values.astype(np.int64)))
    values = values.astype(np.float64)
    for _ in range(5):
        m, sd = values.mean(), values.std()
        values = values[np.abs(values - m) <= 3 * sd]
    return float(values.mean()), float(values.std())


def _background(raw, axon, config, guard=None):
    """Background voxels (outside the guarded detections and the darkest voxels) and their count."""
    if guard is None:
        guard = sparse_dilation(axon, config.guard_vox)
    sub = raw.ravel()[:: max(1, raw.size // 2_000_000)]
    bg = ~guard & (raw > np.percentile(sub, config.outside_pct))
    return bg, int(bg.sum())


def contrast(
    raw: NDArray,
    axon: NDArray[np.bool_],
    config: DetectionQcConfig,
    *,
    guard: NDArray[np.bool_] | None = None,
    _stats: tuple | None = None,
) -> dict:
    """Mean signal of the detections against the sigma-clipped background around them.

    ``guard`` (the detections dilated by ``config.guard_vox``) is reused when given.
    """
    if _stats is None:
        bg, n_bg = _background(raw, axon, config, guard)
        _stats = (_clipped(raw[bg]) if n_bg > 1000 else None, n_bg)
    clipped, n_bg = _stats
    if axon.sum() < 50 or n_bg < 1000:
        return {"signal": None, "background": None, "noise_sd": None, "snr": None}
    b, sd = clipped
    s = float(raw[axon].mean())
    return {"signal": s, "background": b, "noise_sd": sd, "snr": (s - b) / max(sd, 1e-6)}


def block_qc(
    raw: NDArray,
    predict: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    scaling: Callable[[NDArray], NDArray[np.float32]],
    *,
    shared_scaling: Callable[[NDArray], NDArray[np.float32]] | None = None,
    prob: NDArray[np.float32] | None = None,
    config: DetectionQcConfig = DetectionQcConfig(),  # noqa: B008
    guard: NDArray[np.bool_] | None = None,
) -> dict:
    """Run the four checks on one raw block; ``prob`` (with ``scaling``) is reused if given.

    ``guard``, the detections (p >= ``config.axon_p``) dilated by ``config.guard_vox``, is reused
    if given.
    """
    out: dict = {"signal_present": signal_present(raw, config)}
    if out["signal_present"] < config.signal_min:
        return out | {"failed": ["signal_present"], "verdict": "no_signal"}
    if prob is None:
        prob = predict(scaling(raw))
    axon = prob >= config.axon_p
    rate = float(axon.mean())
    if guard is None:
        guard = sparse_dilation(axon, config.guard_vox)
    # background shared by the contrast check and the noise control
    bg, n_bg = _background(raw, axon, config, guard)
    clipped = _clipped(raw[bg]) if n_bg > 1000 else None
    out |= {"detected_fraction": rate} | contrast(raw, axon, config, _stats=(clipped, n_bg))
    failed = []
    if out["snr"] is None or out["snr"] < config.snr_min:
        failed.append("contrast")

    # noise control: the block's own background statistics, no structure
    mean, sd = clipped if clipped is not None else (float(raw.mean()), float(raw.std()))
    side = [min(config.control_vox, s) for s in raw.shape]
    noise = np.random.default_rng(config.seed).normal(mean, max(sd, 1e-6), side).astype(np.float32)
    control_rate = float((predict(scaling(noise)) >= config.axon_p).mean())
    out["control_fraction"] = control_rate
    out["control_ratio"] = control_rate / max(rate, 1e-9)
    if out["control_ratio"] > config.control_ratio_max:
        failed.append("noise_control")

    if shared_scaling is not None:
        shared_rate = float((predict(shared_scaling(raw)) >= config.axon_p).mean())
        out["shared_fraction"] = shared_rate
        out["norm_ratio"] = rate / max(shared_rate, 1e-9)
        lo, hi = config.norm_ratio_range
        if not lo <= out["norm_ratio"] <= hi:
            failed.append("normalization")
    return out | {"failed": failed, "verdict": "false_detection" if failed else "ok"}
