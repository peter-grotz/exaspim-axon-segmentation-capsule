"""Per-block raw-intensity histograms by voxel class, so SNR can be recomputed any way offline.

For every scored snr-tiles block, the raw voxels (source image) and probabilities (the tile's
``probability.zarr``, p x 255) are read back at the recorded box, as in :mod:`rescore`. Each voxel
gets a class:

* ring 0: axon, stored probability >= ``stored_cut`` (230 = p >= 0.9),
* ring k (1..``N_RING``): k face-neighbour steps from the nearest axon voxel (the ring that
  ``binary_dilation(axon, iterations=k)`` adds; the current guard of 2 voxels = rings 1-2),
* ring ``N_RING`` + 1: further away,

and is inside or outside the tile's flat-field brain mask. The mask is read at ``MASK_BIN`` level-0
voxels per cell (16 = 4 x 4 x 4 level-2 voxels) and a cell is inside when >= half of it is mask;
tiles without a mask count as inside throughout (``mask_source`` "none"). Per block the output is
the count of voxels at every raw value (0..max) for each (inside/outside, ring) pair: ``hist`` of
shape (2, N_RING + 2, max + 1), index 0 = inside. Means, SDs, percentiles, cuts, guards and
clipping are all exact functions of these counts. Also stored: the block's current "outside" cut
(10th percentile of the 2 M-voxel subset, as in :func:`~exaspim_axon_segmentation.snr._score`), so
the current SNR can be reproduced from the histograms (:func:`current_metrics`).
"""

from __future__ import annotations

import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

N_RING = 10
MASK_BIN = 16
_STATE: dict = {}


def classify(stored: NDArray[np.uint8], cut: int) -> NDArray[np.uint8]:
    """Ring per voxel: 0 = axon (stored >= cut), k = k dilation steps away, N_RING + 1 beyond."""
    from exaspim_axon_segmentation.qc.detection_qc import sparse_dilation

    grown = stored >= cut
    ring = np.full(stored.shape, N_RING + 1, np.uint8)
    ring[grown] = 0
    for k in range(1, N_RING + 1):
        nxt = sparse_dilation(grown, 1)
        ring[nxt & ~grown] = k
        grown = nxt
    return ring


def histograms(raw: NDArray, ring: NDArray[np.uint8], inside: NDArray[np.bool_]) -> NDArray:
    """Count voxels per raw value and (inside/outside, ring): shape (2, N_RING + 2, max + 1)."""
    dn = np.clip(raw, 0, 65535).astype(np.int32)
    top = int(dn.max()) + 1
    idx = ring.astype(np.int32)
    idx[~inside] += N_RING + 2
    idx *= top
    idx += dn
    del dn
    flat = np.bincount(idx.ravel(), minlength=2 * (N_RING + 2) * top)
    return flat.reshape(2, N_RING + 2, top)


def current_metrics(h: NDArray, outside: float) -> dict:
    """Signal, background, noise and SNR of the current score, from the counts.

    As :func:`~exaspim_axon_segmentation.snr._score` computes them: guard 2 voxels, the voxels
    <= ``outside`` dropped, 3-SD clipping, the mask ignored.
    """
    from exaspim_axon_segmentation.qc.detection_qc import clipped_counts

    both = h.sum(0)
    v = np.arange(both.shape[1])
    axon = both[0]
    bg = both[3:].sum(0) * (v > outside)
    if axon.sum() < 50 or bg.sum() < 1000:
        return {"signal": None, "background": None, "noise_sd": None, "snr": None}
    s = float((axon * v).sum() / axon.sum())
    b, sd = clipped_counts(bg)
    return {"signal": s, "background": b, "noise_sd": sd, "snr": (s - b) / max(sd, 1e-6)}


def _init(url: str, level: int, prob_path: str, mask_path: str | None) -> None:
    import zarr

    from exaspim_axon_segmentation.snr import open_ome

    array, lead, _, _ = open_ome(url, level)
    mask = None if mask_path is None else np.load(mask_path, mmap_mode="r")
    _STATE.update(array=array, lead=lead, prob=zarr.open_group(prob_path, mode="r")["0"],
                  mask=mask, down=2**level)  # fmt: skip


def _block(row: dict, cut: int, out_dir: str) -> dict:
    from exaspim_axon_segmentation.snr import _read, _sub

    t0 = time.time()
    lo = np.asarray(row["lo_vox"])
    hi = lo + np.asarray(row["shape"])
    raw = _read(_STATE["array"], _STATE["lead"], lo, hi)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    stored = np.asarray(_STATE["prob"][box])
    m = _STATE["mask"]
    if m is None:
        inside = np.ones(raw.shape, bool)
    else:  # mask cells of MASK_BIN level-0 voxels -> this level's voxels
        per = MASK_BIN // _STATE["down"]
        idx = [np.minimum(np.arange(a, b) // per, n - 1)
               for a, b, n in zip(lo, hi, m.shape, strict=True)]  # fmt: skip
        inside = np.asarray(m[np.ix_(*idx)])
    ring = classify(stored, cut)
    h = histograms(raw, ring, inside)
    outside = float(np.percentile(_sub(raw), 10.0))
    name = f"block{row['block']:03d}_classhist.npz"
    np.savez_compressed(Path(out_dir) / name, hist=h.astype(np.uint32))
    return {"block": row["block"], "centre_um": row["centre_um"], "lo_vox": row["lo_vox"],
            "shape": row["shape"], "voxel_um": row["voxel_um"], "stored_cut": cut,
            "outside_p10": outside, "inside_fraction": float(inside.mean()),
            "check": current_metrics(h, outside),
            "seconds": round(time.time() - t0, 1)}  # fmt: skip


def blockhist_tile(tile_dir: Path, out_dir: Path, t: float = 0.9, workers: int = 0) -> dict:
    """Class histograms of every scored block of one tile; writes ``block_classhist.json``."""
    from exaspim_axon_segmentation.qc.tile_mask import mask_fraction
    from exaspim_axon_segmentation.rescore import _workers, stored_cut

    d = json.loads((tile_dir / "snr_blocks.json").read_text())
    rows = [b for b in d["blocks"] if "skipped" not in b.get("native", {})]
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = stored_cut(t)
    mask_path, mask_info = None, {"mask_source": "none"}
    if rows and d.get("selection") == "brain mask":
        frac, st = mask_fraction(d["mask"], bin=MASK_BIN, low_res=d.get("low_res_mask"))
        mask_path = str(out_dir / "mask_inside.npy")
        np.save(mask_path, frac >= 0.5)
        mask_info = {"mask_source": "flat-field mask pyramid"} | st | {
            "mask_bin": MASK_BIN, "inside_rule": "cell fraction >= 0.5"}  # fmt: skip
        del frac
    results = []
    if rows:
        init = (d["url"], d["level"], str(tile_dir / "probability.zarr"), mask_path)
        with ProcessPoolExecutor(_workers(workers), initializer=_init, initargs=init) as ex:
            futures = [ex.submit(_block, r, cut, str(out_dir)) for r in rows]
            results = [f.result() for f in futures]
    if mask_path:
        Path(mask_path).unlink()
    summary = {k: v for k, v in d.items() if k != "blocks"} | mask_info | {
        "threshold": t, "n_ring": N_RING, "source": str(tile_dir), "blocks": results,
        "complete": True}  # fmt: skip
    (out_dir / "block_classhist.json").write_text(json.dumps(summary, indent=1))
    log.info("%s: %d blocks, class histograms at p >= %g (mask: %s)", tile_dir.name, len(results),
             t, mask_info["mask_source"])  # fmt: skip
    return summary


def load(path: str | Path) -> NDArray[np.int64]:
    """One block's histogram array (2, N_RING + 2, max + 1); index 0 = inside the mask."""
    return np.load(path)["hist"].astype(np.int64)


def stats(counts: NDArray) -> tuple[float, float, int]:
    """Plain mean, SD (divided by N) and N of the voxels described by a 1-D count histogram."""
    v = np.arange(counts.size, dtype=np.float64)
    n = int(counts.sum())
    if n == 0:
        return math.nan, math.nan, 0
    m = float((counts * v).sum() / n)
    return m, float(np.sqrt((counts * (v - m) ** 2).sum() / n)), n
