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


def _clipped(values: NDArray) -> tuple[float, float]:
    values = values.astype(np.float64)
    for _ in range(5):
        m, sd = values.mean(), values.std()
        values = values[np.abs(values - m) <= 3 * sd]
    return float(values.mean()), float(values.std())


def contrast(raw: NDArray, axon: NDArray[np.bool_], config: DetectionQcConfig) -> dict:
    """Mean signal of the detections against the sigma-clipped background around them."""
    guard = binary_dilation(axon, iterations=config.guard_vox) if axon.any() else axon
    sub = raw.ravel()[:: max(1, raw.size // 2_000_000)]
    bg = ~guard & (raw > np.percentile(sub, config.outside_pct))
    if axon.sum() < 50 or bg.sum() < 1000:
        return {"signal": None, "background": None, "noise_sd": None, "snr": None}
    b, sd = _clipped(raw[bg])
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
) -> dict:
    """Run the four checks on one raw block; ``prob`` (with ``scaling``) is reused if given."""
    out: dict = {"signal_present": signal_present(raw, config)}
    if out["signal_present"] < config.signal_min:
        return out | {"failed": ["signal_present"], "verdict": "no_signal"}
    if prob is None:
        prob = predict(scaling(raw))
    axon = prob >= config.axon_p
    rate = float(axon.mean())
    out |= {"detected_fraction": rate} | contrast(raw, axon, config)
    failed = []
    if out["snr"] is None or out["snr"] < config.snr_min:
        failed.append("contrast")

    # noise control: the block's own background statistics, no structure
    guard = binary_dilation(axon, iterations=config.guard_vox) if axon.any() else axon
    sub = raw.ravel()[:: max(1, raw.size // 2_000_000)]
    bg = ~guard & (raw > np.percentile(sub, config.outside_pct))
    mean, sd = _clipped(raw[bg]) if bg.sum() > 1000 else (float(raw.mean()), float(raw.std()))
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
