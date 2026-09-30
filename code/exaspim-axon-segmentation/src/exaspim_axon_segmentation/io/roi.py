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
    growth : float
        Fraction of its volume the region was grown by (see :meth:`grown`); 0 for the atlas region.
    """

    label_ids: frozenset[int]
    mask: NDArray[np.bool_]
    mask_box: Box
    mask_transform: LevelTransform
    growth: float = 0.0

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

    @classmethod
    def around_points(
        cls, centers_um: NDArray, side_um: float, *, spacing_um: float = 20.0
    ) -> RegionRoi:
        """Cubes of side ``side_um`` centred on physical (z, y, x) points, as one region.

        The mask lives on a synthetic isotropic ``spacing_um`` grid, so everything that consumes a
        CCF region (tile planning, normalization sampling, masking) works unchanged. Used to
        segment around traced somata instead of inside an atlas region.
        """
        centers = np.atleast_2d(np.asarray(centers_um, dtype=np.float64))
        tf = LevelTransform((spacing_um,) * 3, (0.0, 0.0, 0.0))
        half = side_um / 2
        lo = np.floor((centers.min(axis=0) - half) / spacing_um).astype(int)
        hi = np.ceil((centers.max(axis=0) + half) / spacing_um).astype(int) + 1
        mask = np.zeros(hi - lo, dtype=bool)
        for center in centers:
            a = np.ceil((center - half) / spacing_um).astype(int) - lo
            b = np.ceil((center + half) / spacing_um).astype(int) - lo  # centres in [lo, hi)
            mask[tuple(slice(i, j) for i, j in zip(a, b, strict=True))] = True
        tight = _mask_box(mask)
        start = lo + np.asarray(tight.start)
        return cls(frozenset(), mask[tight.slices], Box.from_arrays(start, start + tight.shape), tf)

    def grown(self, fraction: float) -> RegionRoi:
        """Return the region dilated uniformly (µm) until its volume is ``1 + fraction`` times.

        Registration of the atlas to the sample is not voxel-exact, so a margin keeps neurites at
        the region's edge. The shell is the set of outside voxels closest (Euclidean, physical
        units on the anisotropic annotation grid) to the region, cut at the distance that adds
        ``fraction`` of the original voxel count. On 851474, MD grown by 20% gains a ~58 µm shell
        (15%: ~41 µm).
        """
        if fraction <= 0:
            return self
        scale = np.asarray(self.mask_transform.scale)
        target = round(self.n_voxels * fraction)
        faces = sum(
            np.count_nonzero(np.diff(self.mask, axis=a)) * np.prod(np.delete(scale, a))
            for a in range(3)
        )
        # shell thickness if the surface were flat; the search radius doubles until it suffices
        reach_um = 2 * target * np.prod(scale) / max(faces, 1.0) + 2 * scale.max()
        while True:
            pad = np.ceil(reach_um / scale).astype(int) + 1
            padded = np.pad(self.mask, [(q, q) for q in pad])
            distance = _outside_distance(padded, scale, reach_um)
            shell = distance[np.isfinite(distance) & ~padded]
            if len(shell) >= target:
                break
            reach_um *= 2
        cutoff = np.partition(shell, target - 1)[target - 1]
        grown = padded | (distance <= cutoff)
        tight = _mask_box(grown)
        start = np.asarray(self.mask_box.start) - pad + np.asarray(tight.start)
        return RegionRoi(
            self.label_ids,
            grown[tight.slices],
            Box.from_arrays(start, start + np.asarray(tight.shape)),
            self.mask_transform,
            growth=self.growth + fraction * (1 + self.growth),
        )

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


def _outside_distance(
    mask: NDArray[np.bool_], scale: NDArray[np.float64], reach_um: float, block: int = 128
) -> NDArray[np.float32]:
    """Physical distance to the nearest mask voxel, exact up to ``reach_um`` (inf beyond).

    Computed block by block with a ``reach_um`` halo, so memory stays small for large masks.
    """
    from scipy.ndimage import distance_transform_edt

    halo = np.ceil(reach_um / scale).astype(int)
    out = np.full(mask.shape, np.inf, dtype=np.float32)
    for corner in np.ndindex(*(-(-np.asarray(mask.shape) // block))):
        lo = np.asarray(corner) * block
        hi = np.minimum(lo + block, mask.shape)
        a, b = np.maximum(lo - halo, 0), np.minimum(hi + halo, mask.shape)
        window = mask[tuple(slice(i, j) for i, j in zip(a, b, strict=True))]
        if not window.any():
            continue
        d = distance_transform_edt(~window, sampling=scale)
        core = tuple(slice(i - k, j - k) for i, j, k in zip(lo, hi, a, strict=True))
        out[tuple(slice(i, j) for i, j in zip(lo, hi, strict=True))] = d[core]
    out[out > reach_um] = np.inf
    return out


def _mask_box(mask: NDArray[np.bool_]) -> Box:
    idx = np.argwhere(mask)
    return Box.from_arrays(idx.min(axis=0), idx.max(axis=0) + 1)
