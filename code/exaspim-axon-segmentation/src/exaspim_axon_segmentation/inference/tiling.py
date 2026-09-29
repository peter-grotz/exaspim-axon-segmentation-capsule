"""Overlap-tile plan: each tile's output is exactly one chunk of the output array."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from exaspim_axon_segmentation.io import Box
from exaspim_axon_segmentation.model.unet import CONTEXT


@dataclass(frozen=True)
class Tile:
    """One network evaluation.

    Attributes
    ----------
    chunk : tuple of int
        Chunk index in the output array.
    out_box : Box
        Voxels this tile predicts, in source-level indices (may extend past the region).
    in_box : Box
        Network input: ``out_box`` grown by the network context on every face.
    """

    chunk: tuple[int, int, int]
    out_box: Box
    in_box: Box

    def write_box(self, region: Box) -> tuple[Box, Box]:
        """Clip to ``region``: (box in output-array indices, matching box within the prediction)."""
        start = np.maximum(self.out_box.start, region.start)
        stop = np.minimum(self.out_box.stop, region.stop)
        dst = Box.from_arrays(start - np.array(region.start), stop - np.array(region.start))
        src = Box.from_arrays(
            start - np.array(self.out_box.start), stop - np.array(self.out_box.start)
        )
        return dst, src


def plan_tiles(region: Box, tile_output: int) -> list[Tile]:
    """Cover ``region`` with output tiles of side ``tile_output``, aligned to its start."""
    counts = [-(-side // tile_output) for side in region.shape]
    tiles = []
    for chunk in itertools.product(*(range(n) for n in counts)):
        start = np.add(region.start, np.multiply(chunk, tile_output))
        out_box = Box.from_arrays(start, start + tile_output)
        tiles.append(Tile(chunk=tuple(chunk), out_box=out_box, in_box=out_box.pad(CONTEXT)))
    return tiles
