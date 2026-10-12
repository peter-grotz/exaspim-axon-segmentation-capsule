"""Coarse training volumes: 5x data averaged to a 1x-like voxel size, labelled from traces.

The image is a fused level (2 by default, 4 x 2.99 x 2.99 µm) mean-pooled in float by ``factor``
per axis (5 x 5 x 5 gives 20 x 14.96 x 14.96 µm, the 1x scan binned 4x). Stored pyramid levels are
not used: they are rounded to integers, and averaging thousands of raw voxels leaves noise below
one count.

Labels follow :mod:`exaspim_axon_segmentation.train.labels` on the coarse grid, with distances
scaled to the voxel size (an axon is one voxel wide at 15 µm). Volumes are written in the same
layout as :func:`~exaspim_axon_segmentation.train.volume.write_training_volume`, so the trainer
loads them unchanged; the 1x look (blur, contrast, noise) is added during training
(``TrainConfig.degrade``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from exaspim_axon_segmentation import __version__
from exaspim_axon_segmentation.io import Box, NgffImage
from exaspim_axon_segmentation.io.ngff import LevelTransform
from exaspim_axon_segmentation.train.labels import Label
from exaspim_axon_segmentation.train.volume import write_training_volume

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoarseLabelConfig:
    """Label geometry on the coarse grid (µm)."""

    axon_radius_um: float = 8.0
    edge_radius_um: float = 20.0
    negative_clearance_um: float = 30.0
    negative_fraction: float = 0.5
    smooth_um: float = 8.0
    envelope_um: float | None = 60.0
    hard_negatives: bool = True  # False: bright untraced voxels are ignored, not negatives


def coarse_grid(fine: LevelTransform, factor: tuple[int, int, int]) -> LevelTransform:
    """Grid of ``fine`` mean-pooled by ``factor``: voxel centres at the pooled cells' centres."""
    f = np.asarray(factor, float)
    scale = np.asarray(fine.scale) * f
    translation = np.asarray(fine.translation) + (f - 1) / 2 * np.asarray(fine.scale)
    return LevelTransform(tuple(scale.tolist()), tuple(translation.tolist()))


def pooled_read(image: NgffImage, level: int, box: Box, factor: tuple[int, int, int]) -> NDArray:
    """Read ``box`` (fine grid, a multiple of ``factor``) and mean-pool it in float, by slabs."""
    f = np.asarray(factor)
    shape = np.asarray(box.shape) // f
    out = np.zeros(tuple(shape), np.float32)
    step = max(1, 64 // f[0]) * f[0]  # fine planes per slab
    for z0 in range(0, int(shape[0] * f[0]), int(step)):
        z1 = min(z0 + step, int(shape[0] * f[0]))
        hi = [z1, shape[1] * f[1], shape[2] * f[2]]
        sub = Box.from_arrays(np.add(box.start, [z0, 0, 0]), np.add(box.start, hi))
        v = image.read(level, sub).astype(np.float32)
        n = (z1 - z0) // f[0]
        out[z0 // f[0] : z0 // f[0] + n] = v.reshape(n, f[0], shape[1], f[1], shape[2], f[2]).mean(
            (1, 3, 5)
        )
    return out


def coarse_labels(
    raw: NDArray[np.float32],
    grid: LevelTransform,
    origin: NDArray,
    inside: NDArray[np.bool_],
    axon_zyx_um: NDArray,
    config: CoarseLabelConfig,
    *,
    other_traced_zyx_um: NDArray | None = None,
) -> tuple[NDArray[np.uint8], dict]:
    """Label a coarse volume from dense axon trace points (spacing below the voxel size).

    ``origin`` is the coarse index of ``raw[0, 0, 0]``; ``other_traced_zyx_um`` (e.g. traced
    dendrites) are never background or hard negatives.
    """
    scale = np.asarray(grid.scale)

    def distance(points):
        idx = np.floor((points - np.asarray(grid.translation)) / scale + 0.5).astype(int) - origin
        ok = np.all((idx >= 0) & (idx < raw.shape), axis=1)
        line = np.ones(raw.shape, bool)
        line[tuple(idx[ok].T)] = False
        if line.all():
            return np.full(raw.shape, np.inf, np.float32)
        return distance_transform_edt(line, sampling=scale).astype(np.float32)

    dist = distance(axon_zyx_um)
    inside = inside.copy()
    if config.envelope_um is not None:
        inside &= dist <= config.envelope_um
    traced_other = None
    if other_traced_zyx_um is not None and len(other_traced_zyx_um):
        traced_other = distance(other_traced_zyx_um) <= config.edge_radius_um
    axon = (dist <= config.axon_radius_um) & inside
    if not axon.any():
        raise ValueError("No traced axon inside the volume")
    smooth = gaussian_filter(raw, config.smooth_um / scale)
    background = float(np.median(raw[inside]))
    axon_level = float(np.median(smooth[axon]))
    bright = background + config.negative_fraction * (axon_level - background)
    labels = np.full(raw.shape, Label.BACKGROUND, np.uint8)
    labels[(dist > config.negative_clearance_um) & (smooth >= bright)] = (
        Label.HARD_NEGATIVE if config.hard_negatives else Label.IGNORE
    )
    labels[dist <= config.edge_radius_um] = Label.EDGE
    labels[dist <= config.axon_radius_um] = Label.AXON
    if traced_other is not None:
        labels[traced_other & (labels != Label.AXON)] = Label.EDGE
    labels[~inside] = Label.IGNORE
    counts = np.bincount(labels.ravel(), minlength=len(Label))
    stats = {
        "voxels": {lab.name.lower(): int(counts[lab]) for lab in Label},
        "background": background,
        "axon_brightness": axon_level,
        "bright_threshold": bright,
    }
    return labels, stats


def make_coarse_volume(
    image: NgffImage,
    box: Box,
    inside_fine: NDArray[np.bool_] | None,
    axon_zyx_um: NDArray,
    out_dir: Path,
    provenance: dict,
    *,
    level: int = 2,
    factor: tuple[int, int, int] = (5, 5, 5),
    config: CoarseLabelConfig = CoarseLabelConfig(),  # noqa: B008
    other_traced_zyx_um: NDArray | None = None,
) -> dict:
    """Pool, label and write one coarse training volume for a fine-grid ``box``."""
    f = np.asarray(factor)
    stop = np.asarray(box.start) + (np.asarray(box.shape) // f) * f
    box = Box.from_arrays(np.asarray(box.start), stop)
    fine = image.transforms[level]
    grid = coarse_grid(fine, factor)
    raw = pooled_read(image, level, box, factor)
    if inside_fine is None:
        inside = np.ones(raw.shape, bool)
    else:
        sub = inside_fine[: raw.shape[0] * f[0], : raw.shape[1] * f[1], : raw.shape[2] * f[2]]
        sh = [(n, q) for n, q in zip(raw.shape, f, strict=True)]
        inside = sub.reshape(sh[0] + sh[1] + sh[2]).mean((1, 3, 5)) >= 0.5
    origin = np.asarray(box.start) // f
    labels, stats = coarse_labels(raw, grid, origin, inside, axon_zyx_um, config,
                                  other_traced_zyx_um=other_traced_zyx_um)  # fmt: skip
    cbox = Box.from_arrays(origin, origin + np.asarray(raw.shape))
    info = {
        "version": __version__,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "coarse": {"level": level, "factor": list(factor), "voxel_um": list(grid.scale)},
        "label_config": asdict(config),
        "legend": {
            "0": "ignore",
            "1": "axon",
            "2": "background",
            "3": "edge",
            "4": "hard_negative",
        },
        **provenance,
        **stats,
    }
    write_training_volume(out_dir, raw, labels, grid, cbox, info)
    log.info("Coarse volume %s: %s %s", out_dir.name, raw.shape, stats["voxels"])
    return info


def trace_cubes(
    axon_zyx_um: NDArray, cube_um: tuple[float, float, float], n_cubes: int, min_points: int
) -> list[NDArray]:
    """Centres of the ``n_cubes`` grid cubes holding the most trace points (each >= min_points)."""
    cube = np.asarray(cube_um)
    lo = axon_zyx_um.min(0)
    cell = np.floor((axon_zyx_um - lo) / cube).astype(int)
    keys, counts = np.unique(cell, axis=0, return_counts=True)
    order = np.argsort(-counts)
    keep = [keys[i] for i in order if counts[i] >= min_points][:n_cubes]
    return [lo + (np.asarray(k) + 0.5) * cube for k in keep]


def dilate_inside(mask: NDArray[np.bool_], iterations: int) -> NDArray[np.bool_]:
    """Grow a region mask (fine grid) so the coarse majority vote keeps its border."""
    return binary_dilation(mask, iterations=iterations) if iterations > 0 else mask
