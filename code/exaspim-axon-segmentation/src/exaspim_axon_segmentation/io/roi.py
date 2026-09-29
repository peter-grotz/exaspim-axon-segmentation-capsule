"""Regions of interest defined by CCF labels in sample space."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.io.box import Box
from exaspim_axon_segmentation.io.ngff import LevelTransform, NgffImage


@dataclass(frozen=True, eq=False)
class RegionRoi:
    """A set of CCF labels located in a sample-space annotation, with its full-resolution mask.

    Attributes
    ----------
    label_ids : frozenset of int
        Annotation values belonging to the region (the region and all its descendants).
    mask : numpy.ndarray of bool
        Region mask on the annotation's level-0 grid, cropped to ``mask_box``.
    mask_box : Box
        Location of ``mask`` in annotation level-0 indices (the region's tight bounding box).
    mask_transform : LevelTransform
        Annotation level-0 voxel-center transform.
    """

    label_ids: frozenset[int]
    mask: NDArray[np.bool_]
    mask_box: Box
    mask_transform: LevelTransform

    @classmethod
    def from_annotation(
        cls, annotation: NgffImage, label_ids: Iterable[int], *, search_level: int | None = None
    ) -> RegionRoi:
        """Locate a region: coarse search on a low-resolution level, refine at level 0.

        Parameters
        ----------
        annotation : NgffImage
            Sample-space CCF annotation.
        label_ids : iterable of int
            Structure IDs to include (typically ``CcfOntology.resolve(...)``).
        search_level : int, optional
            Level used for the coarse search; defaults to the second-coarsest.

        Raises
        ------
        ValueError
            If no voxel carries any of the labels.
        """
        ids = frozenset(int(i) for i in label_ids)
        search_level = max(annotation.n_levels - 2, 0) if search_level is None else search_level
        coarse = np.isin(annotation.read(search_level), list(ids))
        if not coarse.any():
            raise ValueError(f"None of labels {sorted(ids)} present in {annotation.uri}")

        # Pad by one coarse voxel: mode-downsampling can drop thin parts of the region.
        coarse_box = _mask_box(coarse).pad(1).clip(coarse.shape)
        lo, hi = annotation.transforms[search_level].physical_extent(coarse_box)
        fine_tf = annotation.transforms[0]
        search_box = fine_tf.box_covering(lo, hi, annotation.shape(0))
        fine = np.isin(annotation.read(0, search_box), list(ids))
        tight = _mask_box(fine)
        mask_box = Box.from_arrays(
            np.add(tight.start, search_box.start), np.add(tight.stop, search_box.start)
        )
        return cls(ids, fine[tight.slices], mask_box, fine_tf)

    @property
    def lo_um(self) -> NDArray[np.float64]:
        """Physical lower (z, y, x) bound in micrometers, at voxel edges."""
        return self.mask_transform.physical_extent(self.mask_box)[0]

    @property
    def hi_um(self) -> NDArray[np.float64]:
        """Physical upper (z, y, x) bound in micrometers, at voxel edges."""
        return self.mask_transform.physical_extent(self.mask_box)[1]

    @property
    def n_voxels(self) -> int:
        """Region voxel count at the annotation's full resolution."""
        return int(self.mask.sum())

    @property
    def volume_mm3(self) -> float:
        """Region volume in sample space (mm³)."""
        return self.n_voxels * float(np.prod(self.mask_transform.scale)) * 1e-9

    @property
    def peak_um(self) -> NDArray[np.float64]:
        """Per axis, the physical position of the region's largest cross-section.

        Unlike the bounding-box center this always intersects the region (a bilateral region
        leaves the center empty).
        """
        axes = range(3)
        areas = [self.mask.sum(axis=tuple(a for a in axes if a != axis)) for axis in axes]
        peak = np.add(self.mask_box.start, [int(np.argmax(area)) for area in areas])
        return self.mask_transform.to_physical(peak)

    def box_in(self, image: NgffImage, level: int, *, pad_um: float = 0.0) -> Box:
        """Index-space box of this ROI (optionally padded) on a level of another image."""
        return image.transforms[level].box_covering(
            self.lo_um - pad_um, self.hi_um + pad_um, image.shape(level)
        )

    def fraction_inside(self, grid: LevelTransform, box: Box, *, dilate_um: float = 0.0) -> float:
        """Fraction of mask voxels set within the physical extent of ``box`` on another grid.

        Parameters
        ----------
        grid : LevelTransform
            Transform of the grid ``box`` is defined on.
        box : Box
            Index box on ``grid``.
        dilate_um : float, default 0
            Grow the box by this margin before testing, so tiles near the boundary are kept.

        Returns
        -------
        float
            0 when the (grown) box misses the region entirely.
        """
        lo, hi = grid.physical_extent(box)
        local = self.mask_transform.box_covering(
            lo - dilate_um, hi + dilate_um, (np.iinfo(np.int32).max,) * 3
        )
        start = np.clip(np.subtract(local.start, self.mask_box.start), 0, self.mask.shape)
        stop = np.clip(np.subtract(local.stop, self.mask_box.start), start, self.mask.shape)
        window = self.mask[tuple(slice(a, b) for a, b in zip(start, stop, strict=True))]
        return float(window.mean()) if window.size else 0.0

    def sample_on(self, grid: LevelTransform, box: Box) -> NDArray[np.bool_]:
        """Region mask resampled (nearest neighbor) onto a box of another grid."""
        per_axis = []
        for ax in range(3):
            physical = (
                np.arange(box.start[ax], box.stop[ax]) * grid.scale[ax] + grid.translation[ax]
            )
            idx = np.floor(
                (physical - self.mask_transform.translation[ax]) / self.mask_transform.scale[ax]
                + 0.5
            ).astype(int)
            per_axis.append(idx - self.mask_box.start[ax])
        valid = [(i >= 0) & (i < n) for i, n in zip(per_axis, self.mask.shape, strict=True)]
        out = np.zeros(box.shape, dtype=bool)
        if all(v.any() for v in valid):
            local = [i[v] for i, v in zip(per_axis, valid, strict=True)]
            out[np.ix_(*valid)] = self.mask[np.ix_(*local)]
        return out


def _mask_box(mask: NDArray[np.bool_]) -> Box:
    idx = np.argwhere(mask)
    return Box.from_arrays(idx.min(axis=0), idx.max(axis=0) + 1)
