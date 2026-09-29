"""Layout of a processed exaSPIM data asset."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from exaspim_axon_segmentation.io.ngff import NgffImage

FUSED_PATH = "fusion/fused.zarr"
ANNOTATION_PATH = "ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.zarr"


@dataclass(frozen=True)
class ExaspimAsset:
    """A processed exaSPIM asset, e.g. ``s3://aind-open-data/exaSPIM_<subject>_..._processed_...``.

    Parameters
    ----------
    root : str
        Asset root (local path or ``s3://`` URI).
    anon : bool, default True
        Anonymous S3 access.
    """

    root: str
    anon: bool = True

    @property
    def name(self) -> str:
        """Asset name (last path component)."""
        return self.root.rstrip("/").rsplit("/", 1)[-1]

    @property
    def subject_id(self) -> str:
        """Subject ID parsed from the asset name."""
        return self.name.split("_")[1]

    @cached_property
    def fused(self) -> NgffImage:
        """Fused light-sheet image (single channel)."""
        return NgffImage(f"{self.root.rstrip('/')}/{FUSED_PATH}", anon=self.anon)

    @cached_property
    def annotation(self) -> NgffImage:
        """CCF annotation (structure IDs) reverse-transformed into sample space."""
        return NgffImage(f"{self.root.rstrip('/')}/{ANNOTATION_PATH}", anon=self.anon)
