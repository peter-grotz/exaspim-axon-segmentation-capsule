"""Axis-aligned index-space boxes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Half-open index box ``[start, stop)`` in (z, y, x)."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(b < a for a, b in zip(self.start, self.stop, strict=True)):
            raise ValueError(f"Box stop {self.stop} precedes start {self.start}")

    @classmethod
    def from_arrays(cls, start: Sequence[int], stop: Sequence[int]) -> Box:
        """Build a box from any integer sequences."""
        return cls(tuple(int(v) for v in start), tuple(int(v) for v in stop))

    @classmethod
    def full(cls, shape: Sequence[int]) -> Box:
        """Box covering an entire array."""
        return cls.from_arrays((0, 0, 0), shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Extent along each axis."""
        return tuple(b - a for a, b in zip(self.start, self.stop, strict=True))

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        """Slices for indexing an array."""
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop, strict=True))

    @property
    def center(self) -> tuple[int, int, int]:
        """Integer center voxel."""
        return tuple((a + b) // 2 for a, b in zip(self.start, self.stop, strict=True))

    def pad(self, amount: int | Sequence[int]) -> Box:
        """Grow by ``amount`` voxels on every side (not clipped)."""
        pad = np.broadcast_to(amount, 3)
        return Box.from_arrays(np.subtract(self.start, pad), np.add(self.stop, pad))

    def clip(self, shape: Sequence[int]) -> Box:
        """Restrict to ``[0, shape)``."""
        start = np.clip(self.start, 0, shape)
        return Box.from_arrays(start, np.clip(self.stop, start, shape))
