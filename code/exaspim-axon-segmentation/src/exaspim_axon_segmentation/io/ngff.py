"""Read-only access to OME-NGFF (v0.4) multiscale images with physical-space mapping.

Every array in an exaSPIM asset carries its own ``scale`` + ``translation`` per pyramid level, and
arrays that look like they share a grid often do not (the sample-space CCF annotation is offset by
half a voxel from ``fused.zarr`` level 3). All cross-array mapping therefore goes through physical
coordinates (micrometers, voxel centers) rather than index arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import zarr
from numpy.typing import NDArray

from exaspim_axon_segmentation.io.box import Box

SPATIAL_AXES = ("z", "y", "x")


@dataclass(frozen=True)
class LevelTransform:
    """Voxel-center affine of one pyramid level, restricted to the spatial (z, y, x) axes.

    Attributes
    ----------
    scale : tuple of float
        Voxel size in micrometers, (z, y, x).
    translation : tuple of float
        Physical position of voxel (0, 0, 0)'s center in micrometers, (z, y, x).
    """

    scale: tuple[float, float, float]
    translation: tuple[float, float, float]

    def to_physical(self, index: NDArray) -> NDArray[np.float64]:
        """Map voxel indices (..., 3) to physical coordinates in micrometers."""
        return np.asarray(index, dtype=np.float64) * self.scale + self.translation

    def to_index(self, physical: NDArray) -> NDArray[np.float64]:
        """Map physical coordinates (..., 3) to continuous (unrounded) voxel indices."""
        return (np.asarray(physical, dtype=np.float64) - self.translation) / self.scale

    def physical_extent(self, box: Box) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Physical bounds (voxel edges, not centers) covered by an index-space box."""
        half = np.asarray(self.scale) / 2
        return self.to_physical(box.start) - half, self.to_physical(np.asarray(box.stop) - 1) + half

    def box_covering(self, lo: NDArray, hi: NDArray, shape: tuple[int, ...]) -> Box:
        """Smallest index-space box whose voxels cover the physical range ``[lo, hi]``."""
        start = np.floor(self.to_index(lo) + 0.5).astype(int)
        stop = np.floor(self.to_index(hi) + 0.5).astype(int) + 1
        return Box.from_arrays(start, stop).clip(shape)


class NgffImage:
    """A multiscale OME-NGFF image exposed as spatial (z, y, x) arrays.

    Parameters
    ----------
    uri : str
        Local path or ``s3://`` URI of the multiscale group.
    anon : bool, default True
        Use anonymous S3 access (sufficient for ``aind-open-data``).
    storage_options : dict, optional
        fsspec options that override the default (e.g. ``{"token": ...}`` for ``gs://``).
    """

    def __init__(self, uri: str, *, anon: bool = True, storage_options: dict | None = None) -> None:
        self.uri = uri.rstrip("/")
        options = storage_options
        if options is None and self.uri.startswith("s3://"):
            options = {"anon": anon}
        self._group = zarr.open_group(self.uri, mode="r", storage_options=options)
        attrs = (
            self._group.attrs
        )  # OME-Zarr 0.4 keeps multiscales at the top level, 0.5 under "ome"
        multiscale = (
            attrs["multiscales"] if "multiscales" in attrs else attrs["ome"]["multiscales"]
        )[0]
        axes = [axis["name"] for axis in multiscale["axes"]]
        if tuple(axes[-3:]) != SPATIAL_AXES:
            raise ValueError(f"{uri}: expected trailing axes {SPATIAL_AXES}, got {axes}")
        self._n_leading = len(axes) - 3
        self._datasets = multiscale["datasets"]

    def __repr__(self) -> str:
        return f"NgffImage({self.uri!r}, levels={self.n_levels})"

    @property
    def n_levels(self) -> int:
        """Number of pyramid levels."""
        return len(self._datasets)

    @cached_property
    def transforms(self) -> tuple[LevelTransform, ...]:
        """Spatial voxel-center transform of every level."""
        return tuple(self._parse_transform(ds) for ds in self._datasets)

    def _parse_transform(self, dataset: dict) -> LevelTransform:
        scale, translation = np.ones(3), np.zeros(3)
        for transform in dataset["coordinateTransformations"]:
            if transform["type"] == "scale":
                scale = np.asarray(transform["scale"][-3:], dtype=float)
            elif transform["type"] == "translation":
                translation = np.asarray(transform["translation"][-3:], dtype=float)
        return LevelTransform(tuple(map(float, scale)), tuple(map(float, translation)))

    def array(self, level: int) -> zarr.Array:
        """Raw zarr array (including any leading t/c axes) of a level."""
        return self._group[self._datasets[level]["path"]]

    def shape(self, level: int) -> tuple[int, int, int]:
        """Spatial (z, y, x) shape of a level."""
        return tuple(self.array(level).shape[-3:])

    def dtype(self, level: int) -> np.dtype:
        """Return the data type of a level."""
        return self.array(level).dtype

    def read(self, level: int, box: Box | None = None) -> NDArray:
        """Read a spatial block (first time point and channel) into memory."""
        box = box or Box.full(self.shape(level))
        leading = (0,) * self._n_leading
        return np.asarray(self.array(level)[leading + box.slices])

    def read_padded(self, level: int, box: Box) -> NDArray:
        """Read a block that may extend past the volume, edge-padding the outside.

        Edge padding matches TRAILMAP's own handling of volume borders.
        """
        shape = self.shape(level)
        inner = box.clip(shape)
        block = self.read(level, inner)
        pad = [
            (a_in - a, b - b_in)
            for a, b, a_in, b_in in zip(box.start, box.stop, inner.start, inner.stop, strict=True)
        ]
        if not any(p for pair in pad for p in pair):
            return block
        if block.size == 0:
            return np.zeros(box.shape, dtype=block.dtype)
        return np.pad(block, pad, mode="edge")

    def level_for_spacing(self, target_um: float, *, axis: str = "x") -> int:
        """Index of the level whose voxel size along ``axis`` is closest to ``target_um``."""
        i = SPATIAL_AXES.index(axis)
        sizes = np.array([t.scale[i] for t in self.transforms])
        return int(np.argmin(np.abs(np.log(sizes / target_um))))

    def resample_nearest(
        self, level: int, grid: LevelTransform, box: Box, *, fill: int = 0
    ) -> NDArray:
        """Sample this image on another array's voxel grid with nearest-neighbor lookup.

        Parameters
        ----------
        level : int
            Source pyramid level of this image.
        grid : LevelTransform
            Transform of the destination grid (e.g. a level of ``fused.zarr``).
        box : Box
            Region of the destination grid to fill.
        fill : int, default 0
            Value for destination voxels that fall outside this image.

        Returns
        -------
        numpy.ndarray
            Array of shape ``box.shape`` on the destination grid.
        """
        src = self.transforms[level]
        shape = self.shape(level)
        # The grids are axis-aligned, so the lookup is separable: one index vector per axis.
        per_axis = []
        for ax in range(3):
            dst_index = np.arange(box.start[ax], box.stop[ax])
            physical = dst_index * grid.scale[ax] + grid.translation[ax]
            per_axis.append(np.floor((physical - src.translation[ax]) / src.scale[ax] + 0.5))
        per_axis = [idx.astype(int) for idx in per_axis]
        valid = [(idx >= 0) & (idx < n) for idx, n in zip(per_axis, shape, strict=True)]
        out = np.full(box.shape, fill, dtype=self.dtype(level))
        if not all(v.any() for v in valid):
            return out
        read_box = Box.from_arrays(
            [idx[v].min() for idx, v in zip(per_axis, valid, strict=True)],
            [idx[v].max() + 1 for idx, v in zip(per_axis, valid, strict=True)],
        )
        block = self.read(level, read_box)
        local = [idx[v] - s for idx, v, s in zip(per_axis, valid, read_box.start, strict=True)]
        out[np.ix_(*valid)] = block[np.ix_(*local)]
        return out
