"""Coarse tissue fraction of a raw tile from its flat-field brain mask (sharded Zarr v3).

The masks (``<subject>_processed_*/flatfield_correction/mask/SPIM.ome.zarr/<tile>.ome.zarr``)
exist only at full resolution (0.748 x 0.748 x 1 um), as 512^3 shards of 128 x 256 x 256
blosc/zstd chunks. Most shards lie wholly inside or outside the brain and are byte-identical, so
each distinct shard is decoded once and its result reused: a tile costs one read of ~2 GB and a
few thousand decodes instead of ~21,000.

Returns the fraction of mask voxels set in every ``bin``^3 cell (default 32: 32 x 24 x 24 um).
"""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numcodecs
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def mask_fraction(
    url: str, *, bin: int = 32, threads: int = 32
) -> tuple[NDArray[np.float32], dict]:
    """Tissue fraction per ``bin``^3 voxels of a sharded uint8 mask (level 0)."""
    import fsspec

    fs = fsspec.filesystem("s3", anon=True)
    base = url.removeprefix("s3://").rstrip("/") + "/0"
    meta = json.loads(fs.cat(base + "/zarr.json"))
    shape = np.array(meta["shape"][-3:])
    shard = np.array(meta["chunk_grid"]["configuration"]["chunk_shape"][-3:])
    sh = meta["codecs"][0]["configuration"]
    inner = np.array(sh["chunk_shape"][-3:])
    blosc = next(c for c in sh["codecs"] if c["name"] == "blosc")
    codec = numcodecs.Blosc(cname=blosc["configuration"]["cname"])
    n_inner = shard // inner
    n_shards = -(-shape // shard)
    nb = -(-shape // bin)
    out = np.zeros(tuple(nb), np.float32)
    per = shard // bin
    cache: dict[bytes, NDArray] = {}
    keys = [
        (z, y, x)
        for z in range(n_shards[0])
        for y in range(n_shards[1])
        for x in range(n_shards[2])
    ]

    def fetch(k):
        try:
            buf = fs.cat(f"{base}/c/0/0/{k[0]}/{k[1]}/{k[2]}")
        except FileNotFoundError:
            return k, None, None
        h = hashlib.blake2b(buf, digest_size=16).digest()
        if h not in cache:  # a race may decode a shard twice; harmless
            cache[h] = decode(buf)
        return k, h, None

    def decode(buf: bytes) -> NDArray:
        n = int(np.prod(n_inner))
        index = np.frombuffer(buf[-(16 * n + 4) : -4], "<u8").reshape(n, 2)
        vol = np.zeros(tuple(shard), np.uint8)
        for j, (off, size) in enumerate(index):
            if off == 2**64 - 1:  # empty inner chunk: fill value
                continue
            iz, iy, ix = np.unravel_index(j, tuple(n_inner))
            c = np.frombuffer(codec.decode(buf[off : off + size]), np.uint8).reshape(tuple(inner))
            vol[iz * inner[0] : (iz + 1) * inner[0], iy * inner[1] : (iy + 1) * inner[1],
                ix * inner[2] : (ix + 1) * inner[2]] = c  # fmt: skip
        on = (vol > 0).astype(np.float32)
        return on.reshape(per[0], bin, per[1], bin, per[2], bin).mean((1, 3, 5))

    with ThreadPoolExecutor(threads) as ex:
        for k, h, _ in ex.map(fetch, keys):
            if h is None:
                continue
            frac = cache[h]
            lo = np.array(k) * per
            hi = np.minimum(lo + per, nb)
            n = hi - lo
            out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = frac[: n[0], : n[1], : n[2]]
    n_unique = len(cache)
    stats = {"shards": len(keys), "distinct_shards": n_unique, "bin": bin,
             "tissue_fraction": float(out.mean())}  # fmt: skip
    log.info("Mask %s: %d shards, %d distinct, tissue %.1f%%", url.rsplit("/", 1)[-1], len(keys),
             n_unique, 100 * out.mean())  # fmt: skip
    return out, stats


def blocks_from_fraction(
    frac: NDArray[np.float32],
    block_bins,
    n: int,
    *,
    min_tissue: float = 0.9,
    seed: int = 0,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Up to ``n`` non-overlapping blocks (in bins) with mean tissue >= ``min_tissue``.

    Blocks are laid on a regular grid; of 4 x 4 x 4 grid offsets the one fitting the most tissue
    blocks is kept, then ``n`` of its blocks are drawn at random. Returns block starts (bins) and
    their tissue fractions.
    """
    size = np.asarray(block_bins, int)
    best: tuple[int, NDArray, NDArray] = (-1, np.zeros((0, 3), int), np.zeros(0))
    for off in np.ndindex(4, 4, 4):
        o = (np.array(off) * size) // 4
        m = (np.array(frac.shape) - o) // size
        if np.any(m < 1):
            continue
        sub = frac[o[0] : o[0] + m[0] * size[0], o[1] : o[1] + m[1] * size[1],
                   o[2] : o[2] + m[2] * size[2]]  # fmt: skip
        means = sub.reshape(m[0], size[0], m[1], size[1], m[2], size[2]).mean((1, 3, 5))
        idx = np.argwhere(means >= min_tissue)
        if len(idx) > best[0]:
            best = (len(idx), o + idx * size, means[tuple(idx.T)])
    _, starts, fr = best
    if len(starts) > n:
        pick = np.random.default_rng(seed).choice(len(starts), n, replace=False)
        starts, fr = starts[pick], fr[pick]
    return starts, fr
