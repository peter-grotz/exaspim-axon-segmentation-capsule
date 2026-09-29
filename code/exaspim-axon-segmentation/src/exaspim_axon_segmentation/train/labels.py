"""Dense training labels from complete SWC traces within a region.

Inside the labelled region every axon is traced, so anything bright that is *not* near a trace is a
genuine negative (dendrites, somata, debris). Labels follow TRAILMAP's classes, with an explicit
``IGNORE`` for voxels outside the traced region (seen by the network as context, never scored).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import cc3d
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter

from exaspim_axon_segmentation.io import Box, NgffImage, RegionRoi

log = logging.getLogger(__name__)


class Label(IntEnum):
    """Training label per voxel (uint8)."""

    IGNORE = 0
    AXON = 1
    BACKGROUND = 2
    EDGE = 3
    HARD_NEGATIVE = 4


@dataclass(frozen=True)
class LabelConfig:
    """How traces become voxel labels.

    Attributes
    ----------
    axon_radius_um : float
        Voxels whose center is within this distance of a traced centerline are axon.
    edge_radius_um : float
        Voxels between ``axon_radius_um`` and this distance are edge (near-zero loss weight).
    negative_clearance_um : float
        Bright voxels at least this far from any trace become hard negatives.
    negative_fraction : float
        "Bright" means above ``background + negative_fraction × (axon − background)``, where axon
        brightness is the median over trace points of the brightest voxel within one voxel of the
        point (robust to traces sitting a voxel off-center) and background is the median voxel.
    smooth_um : float
        Light smoothing before the brightness test, so single noisy voxels are not negatives.
    """

    axon_radius_um: float = 2.0
    edge_radius_um: float = 5.0
    negative_clearance_um: float = 8.0
    negative_fraction: float = 0.5
    smooth_um: float = 1.5


def trace_distance_um(
    points_zyx_um: NDArray, grid_scale: Sequence[float], grid_translation: Sequence[float], box: Box
) -> NDArray[np.float32]:
    """Distance (µm) from every voxel center in ``box`` to the nearest trace point.

    Trace points must be dense (spacing below the voxel size) so the rasterized centerline is
    connected; distances are then exact to within half a voxel.
    """
    scale = np.asarray(grid_scale)
    index = np.floor((points_zyx_um - np.asarray(grid_translation)) / scale + 0.5).astype(int)
    local = index - np.asarray(box.start)
    inside = np.all((local >= 0) & (local < box.shape), axis=1)
    centerline = np.ones(box.shape, dtype=bool)
    centerline[tuple(local[inside].T)] = False
    if centerline.all():
        return np.full(box.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(centerline, sampling=scale).astype(np.float32)


def traced_lobe_box(
    image: NgffImage, level: int, roi: RegionRoi, traces_zyx_um: NDArray, *, pad_um: float
) -> tuple[Box, NDArray[np.bool_]]:
    """Box and mask (on ``level``) of the region's connected lobe(s) that contain trace points.

    Bilateral nuclei such as MD are two separate lobes in the annotation; only the traced one is
    completely annotated, so labels are restricted to it.
    """
    lobes, n = cc3d.connected_components(roi.mask, connectivity=26, return_N=True)
    index = np.floor(roi.mask_transform.to_index(traces_zyx_um) + 0.5).astype(int)
    local = index - np.asarray(roi.mask_box.start)
    ok = np.all((local >= 0) & (local < roi.mask.shape), axis=1)
    hit = np.bincount(lobes[tuple(local[ok].T)], minlength=n + 1)[1:]
    traced = np.flatnonzero(hit > 0) + 1
    if traced.size == 0:
        raise ValueError("No trace points fall inside the region")
    log.info("Region has %d lobe(s); trace points per lobe %s; using %s", n, hit.tolist(), traced)
    lobe = RegionRoi(roi.label_ids, np.isin(lobes, traced), roi.mask_box, roi.mask_transform)
    tight = np.argwhere(lobe.mask)
    lobe = RegionRoi(
        roi.label_ids,
        lobe.mask[tuple(slice(a, b + 1) for a, b in zip(tight.min(0), tight.max(0), strict=True))],
        Box.from_arrays(tight.min(0) + roi.mask_box.start, tight.max(0) + 1 + roi.mask_box.start),
        roi.mask_transform,
    )
    box = lobe.box_in(image, level, pad_um=pad_um)
    return box, lobe.sample_on(image.transforms[level], box)


def build_labels(
    image: NgffImage,
    level: int,
    box: Box,
    inside: NDArray[np.bool_],
    traces_zyx_um: NDArray,
    config: LabelConfig,
    *,
    traced_mask: NDArray[np.bool_] | None = None,
    chunk: tuple[int, int, int] = (128, 256, 256),
) -> tuple[NDArray[np.uint8], NDArray[np.uint16], dict]:
    """Label a box on one pyramid level, chunk by chunk.

    Parameters
    ----------
    image : NgffImage
        Fused image.
    level : int
        Level to label (use the inference level).
    box : Box
        Box on the level grid (e.g. from :func:`traced_lobe_box`).
    inside : numpy.ndarray of bool
        Completely traced voxels of ``box``; everything else is ``IGNORE``.
    traces_zyx_um : numpy.ndarray, shape (n, 3)
        Dense trace points (spacing ≤ voxel size), physical (z, y, x) µm.
    config : LabelConfig
        Label geometry.
    traced_mask : numpy.ndarray of bool, optional
        Voxels known to belong to traced neurons (e.g. a fill mask). They are never background or
        hard negatives; those beyond the centerline tube become ``EDGE`` (near-zero weight).
    chunk : tuple of int
        Processing chunk; a halo covering the largest distance is added automatically.

    Returns
    -------
    labels : numpy.ndarray of uint8
        ``Label`` per voxel of ``box``.
    raw : numpy.ndarray of uint16
        The image in ``box``.
    stats : dict
        Voxel counts per label and the brightness threshold used.
    """
    tf = image.transforms[level]
    scale = np.asarray(tf.scale)
    halo = np.ceil(
        max(config.negative_clearance_um, config.edge_radius_um, 3 * config.smooth_um) / scale
    ).astype(int)
    raw = image.read(level, box)
    smooth = np.zeros(box.shape, dtype=np.float32)
    ring = np.full(box.shape, 3, dtype=np.uint8)  # 0 axon, 1 edge, 2 too close for negative, 3 far

    for start in np.ndindex(*(-(-np.array(box.shape) // chunk))):
        core = Box.from_arrays(
            np.multiply(start, chunk), np.minimum(np.multiply(np.add(start, 1), chunk), box.shape)
        )
        outer = core.pad(halo).clip(box.shape)
        inner = tuple(
            slice(a - o, b - o) for a, b, o in zip(core.start, core.stop, outer.start, strict=True)
        )
        global_outer = Box.from_arrays(
            np.add(outer.start, box.start), np.add(outer.stop, box.start)
        )
        distance = trace_distance_um(traces_zyx_um, tf.scale, tf.translation, global_outer)[inner]
        block = raw[outer.slices].astype(np.float32)
        smooth[core.slices] = gaussian_filter(block, config.smooth_um / scale)[inner]
        r = ring[core.slices]
        r[distance < config.negative_clearance_um] = 2
        r[distance <= config.edge_radius_um] = 1
        r[distance <= config.axon_radius_um] = 0

    axon = (ring == 0) & inside
    if not axon.any():
        raise ValueError("No traced axon voxels inside the region; check coordinates")
    axon_level = _trace_brightness(raw, traces_zyx_um, tf.scale, tf.translation, box, inside)
    background = float(np.median(raw[inside][:: max(1, int(inside.sum()) // 5_000_000)]))
    bright = background + config.negative_fraction * (axon_level - background)

    labels = np.full(box.shape, Label.BACKGROUND, dtype=np.uint8)
    labels[ring == 1] = Label.EDGE
    labels[ring == 0] = Label.AXON
    labels[(ring == 3) & (smooth >= bright)] = Label.HARD_NEGATIVE
    if traced_mask is not None:
        labels[traced_mask & (ring != 0)] = Label.EDGE
    labels[~inside] = Label.IGNORE

    counts = np.bincount(labels.ravel(), minlength=len(Label))
    stats = {
        "voxels": {lab.name.lower(): int(counts[lab]) for lab in Label},
        "bright_threshold_counts": bright,
        "axon_brightness_counts": axon_level,
        "background_counts": background,
        "box": {"start": list(box.start), "stop": list(box.stop)},
        "level": level,
    }
    log.info(
        "Labels on level %d box %s: %s (bright ≥ %.1f)", level, box.shape, stats["voxels"], bright
    )
    return labels, raw, stats


def _trace_brightness(
    raw: NDArray,
    points_zyx_um: NDArray,
    scale: Sequence[float],
    translation: Sequence[float],
    box: Box,
    inside: NDArray[np.bool_],
) -> float:
    """Median over trace points (inside the region) of the max voxel in their 3×3×3 neighborhood."""
    index = np.floor((points_zyx_um - np.asarray(translation)) / np.asarray(scale) + 0.5).astype(
        int
    )
    local = np.unique(index - np.asarray(box.start), axis=0)
    local = local[np.all((local >= 0) & (local < box.shape), axis=1)]
    local = local[inside[tuple(local.T)]]
    best = np.zeros(len(local), dtype=raw.dtype)
    for offset in np.ndindex(3, 3, 3):
        at = np.clip(local + np.array(offset) - 1, 0, np.array(box.shape) - 1)
        best = np.maximum(best, raw[tuple(at.T)])
    return float(np.median(best))
