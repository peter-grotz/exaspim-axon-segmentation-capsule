"""Run configuration (validated, serialized into every output's provenance)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from exaspim_axon_segmentation.model.unet import CONTEXT, DIVISOR, MIN_INPUT


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class NormalizationConfig(_Frozen):
    """How raw uint16 intensities are mapped to the network's [0, 1] input range.

    All modes are one run-wide affine map ``clip((raw - offset) / scale, 0, 1)``; nothing is
    estimated per tile, which would create seams.

    ``foreground`` (default): ``offset`` is the background (median of voxels sampled inside the
    region) and ``scale`` puts the median *foreground* voxel (brighter than background by
    ``foreground_mads`` robust standard deviations) at ``foreground_level``. Axon signal is far too
    sparse (<0.1% of voxels) for whole-region percentiles to be stable.
    ``range``: explicit ``offset = low``, ``scale = high - low`` in raw counts.
    ``gain``: ``offset = 0``, ``scale = 65535 / gain``; ``gain = 1`` is TRAILMAP's own
    preprocessing.
    """

    mode: Literal["foreground", "range", "gain"] = "foreground"
    foreground_mads: float = Field(10.0, gt=0)
    foreground_min_counts: float = Field(10.0, ge=0)
    foreground_level: float = Field(0.25, gt=0, le=1)
    floor_percentile: float | None = Field(
        None,
        gt=50,
        lt=100,
        description="foreground mode: raise the offset to this percentile of sampled voxels, so "
        "quantized background texture maps to 0. Off by default: a global floor that removes "
        "texture (98th pct, ~10 counts on 851474 MD) also erases real fibres in dim tiles.",
    )
    low: float = Field(0.0, ge=0)
    high: float = Field(1000.0, gt=0)
    gain: float = Field(1.0, gt=0)
    n_samples: int = Field(64, ge=1)
    sample_size: int = Field(64, ge=8)
    seed: int = 0

    @model_validator(mode="after")
    def _ordered_range(self) -> NormalizationConfig:
        if self.high <= self.low:
            raise ValueError("high must exceed low")
        return self


class InferenceConfig(_Frozen):
    """Tiled inference settings."""

    weights: str = "trailmap-axon-210409"
    level: int = Field(2, ge=0, description="Pyramid level of fused.zarr to segment")
    tile_size: int = Field(160, description="Network input side; output side is tile_size - 28")
    batch_size: int = Field(2, ge=1)
    device: str = Field("auto", description="auto | cuda | cuda:N | mps | cpu")
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    compile: bool = False
    allow_tf32: bool = False
    region_growth: float = Field(
        0.2,
        ge=0,
        description="Grow the CCF region by this fraction of its volume before segmenting "
        "(uniform shell, ~58 µm for MD at 20%), for atlas-registration slack",
    )
    mask_dilation_um: float = Field(50.0, ge=0, description="Keep tiles this close to the region")
    io_threads: int = Field(8, ge=1)
    normalization: NormalizationConfig = NormalizationConfig()

    @field_validator("tile_size")
    @classmethod
    def _valid_tile(cls, value: int) -> int:
        if value % DIVISOR or value < MIN_INPUT:
            raise ValueError(f"tile_size must be a multiple of {DIVISOR} and >= {MIN_INPUT}")
        return value

    @property
    def tile_output(self) -> int:
        """Output side of one tile."""
        return self.tile_size - 2 * CONTEXT
