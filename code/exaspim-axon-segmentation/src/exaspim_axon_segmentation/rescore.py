"""Rescore snr-tiles blocks at other probability thresholds from the saved probability maps.

No network: each block's probability is read back from the tile's ``probability.zarr`` (p x 255,
written by :func:`~exaspim_axon_segmentation.snr.snr_blocks`) and its raw voxels from the source
image, at the box recorded in ``snr_blocks.json``. At each threshold the block is scored exactly
as in the original run (:func:`~exaspim_axon_segmentation.snr._score`), with the QC's
signal-present and contrast checks. The QC noise control needs the network on synthetic noise,
which is not saved: its result at the original threshold is carried over (a detection rate on
noise can only fall at a higher threshold, so a block that passed it still passes).

Stored probabilities are 8-bit, so a threshold t means stored >= t * 255 rounded half up: p >= 0.5
is stored >= 128 (exactly as in the original run), p >= 0.9 is stored >= 230.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

_STATE: dict = {}


def stored_cut(t: float) -> int:
    """Smallest stored value (round(p * 255)) counted as p >= ``t``: t * 255 rounded half up.

    For t = 0.5 this is 128, exactly the voxels with p >= 0.5 in the original run (p * 255 =
    127.5 rounds to 128); for other t, voxels within half a step (1/510) below t also count.
    """
    return math.floor(t * 255 + 0.5)


def _init(url: str, level: int, prob_path: str) -> None:
    import zarr

    from exaspim_axon_segmentation.snr import open_ome

    array, lead, _, _ = open_ome(url, level)
    _STATE.update(array=array, lead=lead, prob=zarr.open_group(prob_path, mode="r")["0"])


def _rescore_block(row: dict, thresholds: tuple[float, ...], out_dir: str) -> dict:
    from exaspim_axon_segmentation.qc.detection_qc import (
        DetectionQcConfig,
        contrast,
        signal_present,
    )
    from exaspim_axon_segmentation.snr import SnrConfig, _read, _score

    t0 = time.time()
    lo = np.asarray(row["lo_vox"])
    hi = lo + np.asarray(row["shape"])
    raw = _read(_STATE["array"], _STATE["lead"], lo, hi)
    stored = np.asarray(
        _STATE["prob"][tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))]
    )
    prob = stored.astype(np.float32) / 255
    old = row.get("native", {})
    old_qc = old.get("qc", {})
    out = {"block": row["block"], "centre_um": row["centre_um"], "lo_vox": row["lo_vox"],
           "shape": row["shape"], "voxel_um": row["voxel_um"]}  # fmt: skip
    sp = signal_present(raw, DetectionQcConfig())
    dn = np.clip(raw, 0, 65535).astype(np.uint16)
    hist = {"total": np.bincount(dn.ravel(), minlength=65536).astype(np.uint32)}
    for t in thresholds:
        cut = stored_cut(t)
        cfg = SnrConfig(axon_p=(cut - 0.5) / 255)  # stored >= cut  <=>  stored / 255 >= this
        metrics, guard = _score(raw, prob, cfg)
        axon = stored >= cut
        qc_cfg = DetectionQcConfig(axon_p=cfg.axon_p)
        qc = {"signal_present": sp}
        if sp < qc_cfg.signal_min:
            qc |= {"failed": ["signal_present"], "verdict": "no_signal"}
        else:
            qc |= {"detected_fraction": float(axon.mean())} | contrast(
                raw, axon, qc_cfg, guard=guard
            )
            failed = [] if qc["snr"] is not None and qc["snr"] >= qc_cfg.snr_min else ["contrast"]
            # noise control: carried over from the original threshold (not recomputed)
            if "noise_control" in old_qc.get("failed", []):
                failed.append("noise_control")
            qc |= {"control_fraction_p0.5": old_qc.get("control_fraction"),
                   "control_ratio_p0.5": old_qc.get("control_ratio"),
                   "failed": failed, "verdict": "false_detection" if failed else "ok"}  # fmt: skip
        out[f"p{t:g}"] = metrics | {"stored_cut": cut, "qc": qc}
        hist[f"foreground_p{t:g}"] = np.bincount(dn[axon], minlength=65536).astype(np.uint32)
    np.savez_compressed(Path(out_dir) / f"block{row['block']:03d}_hist.npz", **hist)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _workers(requested: int) -> int:
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 4
    try:  # ~3 GB per worker (one block in float32 plus masks)
        with open("/proc/meminfo") as fh:
            mem_gb = int(next(ln for ln in fh if ln.startswith("MemTotal")).split()[1]) / 1e6
    except (OSError, StopIteration):
        mem_gb = 16
    return max(1, min(cpus, int(mem_gb // 3.5)))


def rescore_tile(
    tile_dir: Path, out_dir: Path, thresholds: tuple[float, ...], workers: int = 0
) -> dict:
    """Rescore every scored block of one tile; writes ``snr_rescore.json`` and histograms."""
    d = json.loads((tile_dir / "snr_blocks.json").read_text())
    rows = [b for b in d["blocks"] if "skipped" not in b.get("native", {})]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in d.items() if k != "blocks"} | {
        "thresholds": list(thresholds), "source": str(tile_dir),
        "skipped_blocks": len(d["blocks"]) - len(rows)}  # fmt: skip
    results = []
    if rows:
        n = _workers(workers)
        init = (d["url"], d["level"], str(tile_dir / "probability.zarr"))
        with ProcessPoolExecutor(n, initializer=_init, initargs=init) as ex:
            futures = [ex.submit(_rescore_block, r, thresholds, str(out_dir)) for r in rows]
            for f in futures:
                results.append(f.result())
    summary = meta | {"blocks": results, "complete": True}
    (out_dir / "snr_rescore.json").write_text(json.dumps(summary, indent=1))
    log.info("%s: %d blocks rescored at %s", tile_dir.name, len(results), list(thresholds))
    return summary
