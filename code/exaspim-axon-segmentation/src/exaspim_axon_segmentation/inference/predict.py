"""Batched, prefetching tile inference that streams probabilities into an OME-Zarr array."""

from __future__ import annotations

import contextlib
import copy
import logging
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import zarr
from numpy.typing import NDArray

from exaspim_axon_segmentation.inference.normalize import IntensityScaling
from exaspim_axon_segmentation.inference.tiling import Tile
from exaspim_axon_segmentation.io import Box, NgffImage
from exaspim_axon_segmentation.model import TrailmapUNet3D

log = logging.getLogger(__name__)

COMPLETED_ATTR = "completed_chunks"
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(name: str) -> torch.device:
    """Pick ``cuda`` if available, else ``cpu``, for ``"auto"``; otherwise use the named device.

    MPS is never chosen automatically: with torch 2.14 its Conv3d path returns wrong,
    batch- and size-dependent results for this network (differences up to 1.0 in probability).
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps":
        log.warning("MPS gives incorrect Conv3d results for this network; prefer cuda or cpu")
    return device


def resolve_precision(name: str) -> torch.dtype:
    """``"auto"`` means fp32 until reduced precision has been validated against the reference."""
    return torch.float32 if name == "auto" else _DTYPES[name]


class TilePredictor:
    """Run the network on batches of normalized tiles.

    Parameters
    ----------
    model : TrailmapUNet3D
        Network in eval mode.
    device : torch.device
        Compute device.
    precision : torch.dtype
        Autocast dtype (``torch.float32`` disables autocast).
    compile : bool, default False
        Wrap the model with ``torch.compile``.
    allow_tf32 : bool, default False
        Let CUDA use TF32 for fp32 convolutions (faster, ~1e-3 relative error).
    """

    def __init__(
        self,
        model: TrailmapUNet3D,
        device: torch.device,
        precision: torch.dtype,
        *,
        compile: bool = False,
        allow_tf32: bool = False,
    ) -> None:
        self.device = device
        self.precision = precision
        if device.type == "cuda":
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = True
        self.model = model.to(device).eval()
        self._forward = torch.compile(self.model) if compile else self.model

    @torch.inference_mode()
    def __call__(self, batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict ``(B, D, H, W)`` inputs → ``(B, D-28, H-28, W-28)`` probabilities."""
        x = torch.from_numpy(batch).unsqueeze(1).to(self.device)
        autocast = (
            torch.autocast(self.device.type, dtype=self.precision)
            if self.precision != torch.float32
            else contextlib.nullcontext()
        )
        with autocast:
            y = self._forward(x)
        return y.squeeze(1).float().cpu().numpy()

    def self_check(self, batch_size: int, *, side: int = 64, atol: float | None = None) -> float:
        """Compare this device against a CPU fp32 copy of the model; raise if they disagree.

        Cheap (``side``³ inputs) and catches broken device kernels before any tile is written.
        Returns the maximum absolute probability difference.
        """
        if self.device.type == "cpu" and self.precision == torch.float32:
            return 0.0
        atol = atol if atol is not None else (2e-3 if self.precision == torch.float32 else 5e-2)
        reference = copy.deepcopy(self.model).to("cpu").float().eval()
        generator = torch.Generator().manual_seed(0)
        x = torch.rand((batch_size, side, side, side), generator=generator) ** 8
        with torch.inference_mode():
            expected = reference(x.unsqueeze(1)).squeeze(1).numpy()
        diff = float(np.abs(self(x.numpy()) - expected).max())
        if diff > atol:
            raise RuntimeError(
                f"{self.device} ({self.precision}) disagrees with CPU by {diff:.3g} "
                f"(tolerance {atol}); refusing to predict"
            )
        log.info("Device self-check passed on %s: max |diff| %.2g", self.device, diff)
        return diff


@dataclass(frozen=True)
class RunStats:
    """What a prediction run did, and what bounded it."""

    n_planned: int
    n_resumed: int
    n_predicted: int
    seconds: float
    voxels_predicted: int

    @property
    def voxels_per_second(self) -> float:
        """Output voxels produced per wall-clock second."""
        return self.voxels_predicted / self.seconds if self.seconds else 0.0


def predict_tiles(
    image: NgffImage,
    level: int,
    tiles: Sequence[Tile],
    region: Box,
    scaling: IntensityScaling,
    predictor: TilePredictor,
    out: zarr.Array,
    group: zarr.Group,
    *,
    batch_size: int,
    io_threads: int,
) -> RunStats:
    """Predict every tile not yet recorded as complete and write it into ``out``.

    Reads are prefetched on ``io_threads`` threads while the device runs; writes happen on a
    background thread. Completed chunk indices are recorded in the group attributes after each
    batch, so an interrupted run resumes where it stopped.
    """
    done = {tuple(c) for c in group.attrs.get(COMPLETED_ATTR, [])}
    todo = [t for t in tiles if t.chunk not in done]
    log.info(
        "%d tiles planned, %d already complete, %d to predict", len(tiles), len(done), len(todo)
    )

    def load(tile: Tile) -> NDArray[np.float32]:
        return scaling(image.read_padded(level, tile.in_box))

    def write(tile: Tile, prob: NDArray[np.float32]) -> None:
        dst, src = tile.write_box(region)
        out[dst.slices] = np.round(prob[src.slices] * 255).astype(np.uint8)

    start, n_done, voxels = time.perf_counter(), 0, 0
    pending_writes: list[tuple[Tile, Future]] = []
    with ThreadPoolExecutor(io_threads) as readers, ThreadPoolExecutor(1) as writer:
        for batch in _batches(_prefetch(todo, load, readers, depth=2 * batch_size), batch_size):
            probs = predictor(np.stack([x for _, x in batch]))
            _drain(pending_writes, done, group)
            for (tile, _), prob in zip(batch, probs, strict=True):
                pending_writes.append((tile, writer.submit(write, tile, prob)))
                voxels += int(np.prod(tile.write_box(region)[0].shape))
            n_done += len(batch)
            _log_progress(n_done, len(todo), start)
        _drain(pending_writes, done, group)

    return RunStats(len(tiles), len(tiles) - len(todo), n_done, time.perf_counter() - start, voxels)


def _prefetch(
    tiles: Sequence[Tile], load, pool: ThreadPoolExecutor, depth: int
) -> Iterator[tuple[Tile, NDArray]]:
    queue: deque[tuple[Tile, Future]] = deque()
    source = iter(tiles)
    for tile in source:
        queue.append((tile, pool.submit(load, tile)))
        if len(queue) >= depth:
            break
    while queue:
        tile, future = queue.popleft()
        if (following := next(source, None)) is not None:
            queue.append((following, pool.submit(load, following)))
        yield tile, future.result()


def _batches(items: Iterator, size: int) -> Iterator[list]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _drain(pending: list[tuple[Tile, Future]], done: set, group: zarr.Group) -> None:
    if not pending:
        return
    for tile, future in pending:
        future.result()
        done.add(tile.chunk)
    pending.clear()
    group.attrs[COMPLETED_ATTR] = sorted(list(c) for c in done)


def _log_progress(n_done: int, n_total: int, start: float) -> None:
    if n_done % 25 and n_done != n_total:
        return
    elapsed = time.perf_counter() - start
    eta = elapsed / n_done * (n_total - n_done)
    log.info("%d/%d tiles · %.0f s elapsed · ETA %.0f s", n_done, n_total, elapsed, eta)
