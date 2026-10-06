"""Level-1 fused image over anonymous HTTPS with a byte-capped chunk cache (for the tracer).

Adapted from robo_ls/robo_lib.Zarr: the tracer reads many small blocks along its path, so chunks
are cached. Coordinates follow the tracer: world µm (x, y, z) in, voxel (z, y, x) internally.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np
import requests
from numpy.typing import NDArray

from exaspim_axon_segmentation.io import ExaspimAsset

BASE = "https://aind-open-data.s3.amazonaws.com/"


class Level1Image:
    """Level 1 of an asset's fused.zarr (uint16), read chunk by chunk with an LRU cache."""

    def __init__(self, asset: ExaspimAsset, cap_bytes: float = 2.0e9):
        root = asset.root.rstrip("/")
        if not root.startswith("s3://aind-open-data/"):
            raise ValueError(f"tracer image reader expects a public aind-open-data asset: {root}")
        self.url = f"{BASE}{root[len('s3://aind-open-data/') :]}/fusion/fused.zarr/1"
        meta = requests.get(f"{self.url}/.zarray", timeout=60).json()
        self.sep = meta.get("dimension_separator", ".")
        self.lead = len(meta["shape"]) - 3  # leading (t, c) axes of a 5-D array
        self.chunk = np.array(meta["chunks"][self.lead :])
        self.shape = np.array(meta["shape"][self.lead :])
        self.dtype = np.dtype(meta["dtype"])
        tf = asset.fused.transforms[1]
        self.scale = np.array(tf.scale, dtype=float)  # (z, y, x) µm
        self.translation = np.array(tf.translation, dtype=float)
        self.cap, self.cache, self.nbytes = cap_bytes, OrderedDict(), 0
        self.session = requests.Session()
        self.lock = threading.Lock()
        self.decoder = None

    def to_voxel(self, xyz: NDArray) -> NDArray[np.float64]:
        """World (x, y, z) µm -> fractional level-1 voxel (z, y, x)."""
        return (np.asarray(xyz, float)[..., ::-1] - self.translation) / self.scale

    def _download(self, ck: tuple[int, int, int]) -> NDArray | None:
        import numcodecs

        if self.decoder is None:
            self.decoder = numcodecs.Blosc()
        key = self.sep.join(["0"] * self.lead + [str(c) for c in ck])
        last = None
        for attempt in range(5):
            try:
                r = self.session.get(f"{self.url}/{key}", timeout=120)
                if r.status_code in (403, 404):
                    return None  # absent chunk: background
                r.raise_for_status()
                raw = self.decoder.decode(r.content)
                a = np.frombuffer(raw, dtype=self.dtype).astype(np.uint16)
                shp = np.minimum(self.chunk, self.shape - np.array(ck) * self.chunk)
                if a.size == self.chunk.prod():
                    a = a.reshape(self.chunk)[: shp[0], : shp[1], : shp[2]]
                else:
                    a = a.reshape(shp)
                return np.ascontiguousarray(a)
            except Exception as error:
                last = error
                time.sleep(2 + 3 * attempt)
        raise RuntimeError(f"chunk {ck} of {self.url} failed: {type(last).__name__}")

    def _fetch(self, ck: tuple[int, int, int]) -> NDArray | None:
        with self.lock:
            if ck in self.cache:
                self.cache.move_to_end(ck)
                return self.cache[ck]
        a = self._download(ck)
        with self.lock:
            self.cache[ck] = a
            self.nbytes += 0 if a is None else a.nbytes
            while self.nbytes > self.cap and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.nbytes -= 0 if old is None else old.nbytes
        return a

    def block(self, lo: NDArray, size: NDArray) -> NDArray[np.uint16]:
        """Voxels [lo, lo + size) (z, y, x); outside the image reads as 0."""
        lo, size = np.asarray(lo, int), np.asarray(size, int)
        hi = lo + size
        out = np.zeros(size, np.uint16)
        a0, a1 = np.maximum(lo, 0), np.minimum(hi, self.shape)
        if np.any(a1 <= a0):
            return out
        c0, c1 = a0 // self.chunk, (a1 - 1) // self.chunk
        for z in range(c0[0], c1[0] + 1):
            for y in range(c0[1], c1[1] + 1):
                for x in range(c0[2], c1[2] + 1):
                    a = self._fetch((z, y, x))
                    if a is None:
                        continue
                    o = np.array([z, y, x]) * self.chunk
                    s0, s1 = np.maximum(a0, o), np.minimum(a1, o + np.array(a.shape))
                    out[tuple(slice(s0[k] - lo[k], s1[k] - lo[k]) for k in range(3))] = a[
                        tuple(slice(s0[k] - o[k], s1[k] - o[k]) for k in range(3))
                    ]
        return out
