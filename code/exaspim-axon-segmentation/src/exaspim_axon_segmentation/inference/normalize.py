"""Intensity scaling from raw exaSPIM counts to the network's input range."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.config import NormalizationConfig
from exaspim_axon_segmentation.io import Box, NgffImage, RegionRoi

UINT16_MAX = 65535.0

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntensityScaling:
    """Affine map ``clip((raw - offset) / scale, 0, 1)``."""

    offset: float
    scale: float

    def __call__(self, raw: NDArray) -> NDArray[np.float32]:
        """Scale a raw block to float32 in [0, 1]."""
        out = (raw.astype(np.float32) - np.float32(self.offset)) / np.float32(self.scale)
        return np.clip(out, 0.0, 1.0, out=out)

    def as_dict(self) -> dict[str, float]:
        """JSON-serializable form."""
        return asdict(self)


def estimate_scaling(
    config: NormalizationConfig, image: NgffImage, level: int, roi: RegionRoi
) -> IntensityScaling:
    """Build the run-wide intensity scaling.

    Parameters
    ----------
    config : NormalizationConfig
        Normalization settings.
    image : NgffImage
        Fused image.
    level : int
        Level being segmented.
    roi : RegionRoi
        Region the statistics are sampled from (``foreground`` mode only).
    """
    if config.mode == "gain":
        return IntensityScaling(offset=0.0, scale=UINT16_MAX / config.gain)
    if config.mode == "range":
        return IntensityScaling(offset=config.low, scale=config.high - config.low)
    return scaling_from_samples(_sample_region(image, level, roi, config), config)


def estimate_scaling_array(
    raw: NDArray, inside: NDArray[np.bool_], config: NormalizationConfig
) -> IntensityScaling:
    """Estimate :func:`estimate_scaling` from patches of an in-memory volume inside a mask."""
    rng = np.random.default_rng(config.seed)
    candidates = np.argwhere(inside[::4, ::4, ::4]) * 4
    picks = candidates[rng.choice(len(candidates), size=config.n_samples)]
    half = config.sample_size // 2
    blocks = [
        raw[tuple(slice(max(c - half, 0), c + half) for c in center)].ravel() for center in picks
    ]
    return scaling_from_samples(np.concatenate(blocks), config)


def scaling_from_samples(samples: NDArray, config: NormalizationConfig) -> IntensityScaling:
    """Foreground-mode scaling from sampled raw voxels (see :class:`NormalizationConfig`)."""
    samples = samples.astype(np.float64)
    background = float(np.median(samples))
    sigma = 1.4826 * float(np.median(np.abs(samples - background)))
    threshold = background + max(config.foreground_mads * sigma, config.foreground_min_counts)
    foreground = samples[samples > threshold]
    if foreground.size == 0:
        raise ValueError(f"No voxels above {threshold:.1f} counts in {config.n_samples} samples")
    offset = background
    if config.floor_percentile is not None:
        offset = max(background, float(np.percentile(samples, config.floor_percentile)))
    signal = float(np.median(foreground)) - offset
    if signal <= 0:
        raise ValueError(f"Noise floor {offset:.1f} is above the foreground median")
    log.info(
        "Background %.1f counts (sd %.1f), offset %.1f; %d foreground voxels, median %.1f counts",
        background, sigma, offset, foreground.size, signal + offset,
    )  # fmt: skip
    return IntensityScaling(offset=offset, scale=signal / config.foreground_level)


def _sample_region(
    image: NgffImage, level: int, roi: RegionRoi, config: NormalizationConfig
) -> NDArray:
    rng = np.random.default_rng(config.seed)
    inside = np.argwhere(roi.mask)
    picks = inside[rng.choice(len(inside), size=config.n_samples, replace=len(inside) < 1e6)]
    centers_um = roi.mask_transform.to_physical(picks + np.array(roi.mask_box.start))
    tf = image.transforms[level]
    half = config.sample_size // 2
    blocks = []
    for center in np.round(tf.to_index(centers_um)).astype(int):
        box = Box.from_arrays(center - half, center + half).clip(image.shape(level))
        blocks.append(image.read(level, box).ravel())
    return np.concatenate(blocks)
