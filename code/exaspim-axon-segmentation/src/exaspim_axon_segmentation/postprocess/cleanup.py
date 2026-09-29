"""Clean a probability map into an axon mask: hysteresis, speckle removal, soma-connected removal.

Every removed voxel keeps a reason code, so QC can show what was dropped and why instead of it
disappearing silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

import cc3d
import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.io import Box, LevelTransform

log = logging.getLogger(__name__)


class Category(IntEnum):
    """Per-voxel outcome of the cleanup, stored as uint8."""

    BACKGROUND = 0
    KEPT = 1
    NO_CONFIDENT_CORE = 2
    TOO_SMALL = 3
    SOMA_CONNECTED = 4


@dataclass(frozen=True)
class CleanupConfig:
    """Cleanup thresholds.

    Attributes
    ----------
    low, high : float
        Hysteresis thresholds on probability: a voxel above ``low`` is kept only if its connected
        component also contains a voxel above ``high``.
    min_volume_um3 : float
        Components smaller than this are dropped as speckle.
    remove_soma_connected : bool
        Drop every component that touches a detected soma (dendrites and local axons of labelled
        neurons in the region).
    soma_radius_scale : float
        Soma spheres use the detector's mean radius times this factor.
    """

    low: float = 0.5
    high: float = 0.9
    min_volume_um3: float = 2000.0
    remove_soma_connected: bool = False
    soma_radius_scale: float = 1.0


@dataclass(frozen=True)
class Soma:
    """A detected soma: physical center (z, y, x) and mean radius, in micrometers."""

    center_um: tuple[float, float, float]
    radius_um: float


def cleanup(
    probability: NDArray[np.uint8],
    grid: LevelTransform,
    region: Box,
    config: CleanupConfig,
    somata: list[Soma] | None = None,
) -> NDArray[np.uint8]:
    """Classify every voxel of a uint8 (p × 255) probability map.

    Parameters
    ----------
    probability : numpy.ndarray of uint8
        Probability map covering ``region``.
    grid : LevelTransform
        Transform of the level the map is on.
    region : Box
        Location of ``probability`` in level indices.
    config : CleanupConfig
        Thresholds.
    somata : list of Soma, optional
        Detections used when ``config.remove_soma_connected`` is set.

    Returns
    -------
    numpy.ndarray of uint8
        ``Category`` code per voxel.
    """
    low, high = round(config.low * 255), round(config.high * 255)
    labels, n = cc3d.connected_components(
        probability > low, connectivity=26, return_N=True, out_dtype=np.uint32
    )
    flat = labels.ravel()
    category = np.full(n + 1, Category.KEPT, dtype=np.uint8)
    category[0] = Category.BACKGROUND

    has_core = np.zeros(n + 1, dtype=bool)
    has_core[np.unique(flat[probability.ravel() > high])] = True
    category[~has_core] = Category.NO_CONFIDENT_CORE

    voxel_um3 = float(np.prod(grid.scale))
    too_small = np.bincount(flat, minlength=n + 1) * voxel_um3 < config.min_volume_um3
    category[too_small & has_core] = Category.TOO_SMALL

    if config.remove_soma_connected and somata:
        touched = _labels_touching_somata(labels, grid, region, somata, config.soma_radius_scale)
        category[touched[category[touched] == Category.KEPT]] = Category.SOMA_CONNECTED
    category[0] = Category.BACKGROUND

    out = category[labels]
    counts = np.bincount(out.ravel(), minlength=len(Category))
    log.info(
        "Cleanup: %d components; voxels %s",
        n, {c.name.lower(): int(counts[c]) for c in Category if c != Category.BACKGROUND},
    )  # fmt: skip
    return out


def _labels_touching_somata(
    labels: NDArray[np.uint32],
    grid: LevelTransform,
    region: Box,
    somata: list[Soma],
    radius_scale: float,
) -> NDArray[np.int64]:
    touched: set[int] = set()
    scale = np.asarray(grid.scale)
    for soma in somata:
        center = grid.to_index(np.asarray(soma.center_um)) - np.asarray(region.start)
        radius = soma.radius_um * radius_scale
        half = np.ceil(radius / scale).astype(int)
        lo = np.maximum(np.floor(center).astype(int) - half, 0)
        hi = np.minimum(np.floor(center).astype(int) + half + 1, labels.shape)
        if np.any(hi <= lo):
            continue
        axes = [(np.arange(a, b) - c) * s for a, b, c, s in zip(lo, hi, center, scale, strict=True)]
        dz, dy, dx = np.meshgrid(*axes, indexing="ij")
        sphere = dz**2 + dy**2 + dx**2 <= radius**2
        window = labels[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))]
        touched.update(np.unique(window[sphere]).tolist())
    touched.discard(0)
    return np.array(sorted(touched), dtype=np.int64)
