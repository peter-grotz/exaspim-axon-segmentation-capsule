"""Coarse tissue fraction of a raw tile from its flat-field brain mask (sharded Zarr v3).

The masks (``<subject>_processed_*/flatfield_correction/mask/SPIM.ome.zarr/<tile>.ome.zarr``)
exist at full resolution (0.748 x 0.748 x 1 um), as 512^3 shards of 128 x 256 x 256 blosc/zstd
chunks. Most shards lie wholly inside or outside the brain and are byte-identical: shards are
listed with their ETags and only the distinct ones (~1,500-3,700 of ~21,000) are fetched and
decoded. Some tiles also have an 8x-downsampled copy from the flat-field step, which is read
instead when available.

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


def _distinct(fs, prefix: str) -> tuple[dict[str, str], dict[str, str]]:
    """Object key -> ETag under ``prefix``, and one key per distinct ETag (identical bytes)."""
    found = fs.find(prefix, detail=True)
    keys = {k: v["ETag"] for k, v in found.items()}
    first: dict[str, str] = {}
    for k, e in keys.items():
        first.setdefault(e, k)
    return keys, first


def mask_fraction(
    url: str, *, bin: int = 32, threads: int = 8, low_res: str | None = None
) -> tuple[NDArray[np.float32], dict]:
    """Tissue fraction per ``bin``^3 voxels of a sharded uint8 mask (level 0).

    Shards are listed with their ETags (one listing call per 1000 objects), and only one shard per
    distinct ETag is fetched; inner chunks with identical bytes are decoded once. ``low_res`` is
    the flat-field step's 8x-downsampled copy of the same mask
    (``artifacts/<tile>/<tile>_mask_low_res.zarr``), read instead when given.
    """
    import fsspec

    fs = fsspec.filesystem("s3", anon=True)
    if low_res is not None:
        return _low_res_fraction(fs, low_res, bin, threads)
    base = url.removeprefix("s3://").rstrip("/") + "/0"
    if not fs.exists(base + "/zarr.json"):  # older masks: Zarr v2 with a full pyramid
        return _pyramid_fraction(url, bin)
    meta = json.loads(fs.cat(base + "/zarr.json"))
    shape = np.array(meta["shape"][-3:])
    shard = np.array(meta["chunk_grid"]["configuration"]["chunk_shape"][-3:])
    sh = meta["codecs"][0]["configuration"]
    inner = np.array(sh["chunk_shape"][-3:])
    blosc = next(c for c in sh["codecs"] if c["name"] == "blosc")
    codec = numcodecs.Blosc(cname=blosc["configuration"]["cname"])
    n_inner = shard // inner
    nb = -(-shape // bin)
    out = np.zeros(tuple(nb), np.float32)
    per, iper = shard // bin, inner // bin
    keys, first = _distinct(fs, base + "/c")
    inner_cache: dict[bytes, NDArray] = {}

    def decode(etag: str) -> tuple[str, NDArray]:
        buf = fs.cat(first[etag])
        n = int(np.prod(n_inner))
        index = np.frombuffer(buf[-(16 * n + 4) : -4], "<u8").reshape(n, 2)
        frac = np.zeros(tuple(per), np.float32)
        for j, (off, size) in enumerate(index):
            if off == 2**64 - 1:  # empty inner chunk: fill value
                continue
            raw = buf[off : off + size]
            h = hashlib.blake2b(raw, digest_size=16).digest()
            if h not in inner_cache:  # a race may decode a chunk twice; harmless
                c = np.frombuffer(codec.decode(raw), np.uint8).reshape(tuple(inner)) > 0
                inner_cache[h] = c.reshape(iper[0], bin, iper[1], bin, iper[2], bin).mean(
                    (1, 3, 5), dtype=np.float32
                )
            iz, iy, ix = np.unravel_index(j, tuple(n_inner))
            frac[iz * iper[0] : (iz + 1) * iper[0], iy * iper[1] : (iy + 1) * iper[1],
                 ix * iper[2] : (ix + 1) * iper[2]] = inner_cache[h]  # fmt: skip
        return etag, frac

    with ThreadPoolExecutor(threads) as ex:
        fracs = dict(ex.map(decode, first))
    for k, e in keys.items():
        z, y, x = (int(v) for v in k.rsplit("/", 3)[-3:])
        lo = np.array((z, y, x)) * per
        hi = np.minimum(lo + per, nb)
        n = hi - lo
        out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = fracs[e][: n[0], : n[1], : n[2]]
    stats = {"shards": len(keys), "distinct_shards": len(first), "bin": bin,
             "tissue_fraction": float(out.mean()), "mask_source": "full resolution"}  # fmt: skip
    log.info("Mask %s: %d shards, %d distinct, tissue %.1f%%", url.rsplit("/", 1)[-1], len(keys),
             len(first), 100 * out.mean())  # fmt: skip
    return out, stats


def _low_res_fraction(fs, url: str, bin: int, threads: int) -> tuple[NDArray[np.float32], dict]:
    """Tissue fraction from the 8x-downsampled mask copy (regular Zarr v3 chunks, zstd)."""
    base = url.removeprefix("s3://").rstrip("/")
    meta = json.loads(fs.cat(base + "/zarr.json"))
    shape = np.array(meta["shape"][-3:])
    chunk = np.array(meta["chunk_grid"]["configuration"]["chunk_shape"][-3:])
    if not any(c["name"] == "zstd" for c in meta["codecs"]):
        raise ValueError(f"{url}: unexpected codecs {meta['codecs']}")
    f = bin // 8  # low-res voxels per cell
    nb = -(-shape * 8 // bin)
    out = np.zeros(tuple(nb), np.float32)
    per = chunk // f
    keys, first = _distinct(fs, base + "/c")
    codec = numcodecs.Zstd()

    def decode(etag: str) -> tuple[str, NDArray]:
        c = np.frombuffer(codec.decode(fs.cat(first[etag])), np.uint8).reshape(tuple(chunk)) > 0
        return etag, c.reshape(per[0], f, per[1], f, per[2], f).mean((1, 3, 5), dtype=np.float32)

    with ThreadPoolExecutor(threads) as ex:
        fracs = dict(ex.map(decode, first))
    for k, e in keys.items():
        z, y, x = (int(v) for v in k.rsplit("/", 3)[-3:])
        lo = np.array((z, y, x)) * per
        hi = np.minimum(lo + per, nb)
        n = hi - lo
        out[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = fracs[e][: n[0], : n[1], : n[2]]
    stats = {
        "chunks": len(keys),
        "distinct_chunks": len(first),
        "bin": bin,
        "tissue_fraction": float(out.mean()),
        "mask_source": "low resolution (8x)",
    }
    log.info("Mask %s (low-res copy): %d chunks, %d distinct, tissue %.1f%%",
             url.rsplit("/", 1)[-1], len(keys), len(first), 100 * out.mean())  # fmt: skip
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


def ordered_blocks(
    frac: NDArray[np.float32], block_bins, n: int, *, min_tissue: float = 0.9, seed: int = 0
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Every qualifying block, in the order to try them.

    First the ``n`` that :func:`blocks_from_fraction` picks (same seed, same blocks), then the rest
    in a fixed random order, as replacements for blocks rejected later (e.g. by a tissue check).
    """
    starts, fr = blocks_from_fraction(frac, block_bins, 10**12, min_tissue=min_tissue)
    if len(starts) <= n:
        return starts, fr
    pick = np.random.default_rng(seed).choice(len(starts), n, replace=False)
    rest = np.setdiff1d(np.arange(len(starts)), pick)
    rest = rest[np.random.default_rng([seed, 1]).permutation(len(rest))]
    order = np.concatenate([pick, rest])
    return starts[order], fr[order]


def _pyramid_fraction(url: str, bin: int) -> tuple[NDArray[np.float32], dict]:
    """Tissue fraction from a mask pyramid level ``bin`` times coarser than level 0 (binary)."""
    from exaspim_axon_segmentation.io import NgffImage

    m = NgffImage(url)
    s0 = np.asarray(m.transforms[0].scale)
    level = next(lv for lv in range(m.n_levels)
                 if np.allclose(np.asarray(m.transforms[lv].scale) / s0, bin))  # fmt: skip
    v = (m.read(level) > 0).astype(np.float32)
    nb = -(-np.asarray(m.shape(0)) // bin)
    out = np.zeros(tuple(nb), np.float32)
    n = np.minimum(v.shape, nb)
    out[: n[0], : n[1], : n[2]] = v[: n[0], : n[1], : n[2]]
    log.info("Mask %s (pyramid level %d): tissue %.1f%%", url.rsplit("/", 1)[-1], level,
             100 * out.mean())  # fmt: skip
    return out, {"pyramid_level": level, "bin": bin, "tissue_fraction": float(out.mean())}
