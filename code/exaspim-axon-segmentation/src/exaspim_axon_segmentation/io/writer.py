"""Writing OME-NGFF (v0.4, zarr v2) outputs that overlay on their source image."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from exaspim_axon_segmentation.io.box import Box
from exaspim_axon_segmentation.io.ngff import LevelTransform


def create_ngff_array(
    uri: str,
    *,
    source: LevelTransform,
    region: Box,
    chunks: tuple[int, int, int],
    dtype: np.dtype | str,
    name: str,
    attrs: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> zarr.Array:
    """Create a single-level 3D OME-NGFF image covering ``region`` of a source grid.

    The output keeps the source voxel size, and its translation is shifted to ``region.start``, so
    it lands exactly on the source image in any NGFF-aware viewer.

    Parameters
    ----------
    uri : str
        Output group path (local or ``s3://``).
    source : LevelTransform
        Transform of the source level the output is aligned to.
    region : Box
        Covered region, in source-level indices.
    chunks : tuple of int
        Chunk shape (z, y, x).
    dtype : numpy dtype
        Output data type.
    name : str
        Image name stored in the multiscales metadata.
    attrs : dict, optional
        Extra group attributes (e.g. provenance).
    overwrite : bool, default False
        Replace an existing group; otherwise an existing compatible array is reopened so
        interrupted runs can resume.
    """
    group = zarr.open_group(uri, mode="w" if overwrite else "a", zarr_format=2)
    if "0" in group and not overwrite:
        array = group["0"]
        if array.shape != region.shape or array.chunks != chunks or array.dtype != dtype:
            raise ValueError(f"{uri}/0 exists with incompatible layout; pass overwrite=True")
        return array
    array = group.create_array(
        "0",
        shape=region.shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=0,
        chunk_key_encoding={"name": "v2", "separator": "/"},
        overwrite=overwrite,
    )
    translation = source.to_physical(np.array(region.start))
    group.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": name,
            "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": list(source.scale)},
                        {"type": "translation", "translation": translation.tolist()},
                    ],
                }
            ],
        }
    ]
    if attrs:
        group.attrs.update(attrs)
    return array
