"""SLAP2 fine-tuning helpers for the capsule: rebuild the 1 um training volume, and predict on it.

volume <REFERENCE.tif> <labels.npz> <out dir>
green channel (planes interleaved green/red) averaged over 4 x 4 px in xy -> 1 um voxels, checked against
the checksum stored with the labels (made by scripts/slap2_labels.py), written with the labels as a
training volume (split "all": every voxel trains and validates).
predict <volume dir> <out.npz> --weights A [--weights B ...]
probability on the whole volume per weights (same intensity scaling as the trainer), with voxel scores
against the labels: recall on positive voxels, positive rate on background and on ignored voxels.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from exaspim_axon_segmentation.config import NormalizationConfig
from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array
from exaspim_axon_segmentation.inference.predict import TilePredictor, resolve_device
from exaspim_axon_segmentation.io import Box
from exaspim_axon_segmentation.io.ngff import LevelTransform
from exaspim_axon_segmentation.model import load_model
from exaspim_axon_segmentation.train.labels import Label
from exaspim_axon_segmentation.train.volume import open_training_volume, write_training_volume

CROP = 14


def volume(a: argparse.Namespace) -> None:
    n = len(tifffile.TiffFile(a.tif).pages)
    img = tifffile.imread(a.tif, key=range(0, n, 2))
    img = (
        img.reshape(img.shape[0], img.shape[1] // 4, 4, img.shape[2] // 4, 4)
        .mean((2, 4))
        .astype(np.float32)
    )
    d = np.load(a.labels)
    lab = d["label"]
    if img.shape != lab.shape:
        raise SystemExit(f"image {img.shape} != labels {lab.shape}")
    want = d["image_checksum"]
    got = np.array([float(img.astype(np.float64).sum()), float(img[100].astype(np.float64).sum())])
    print("image checksum", got, "expected", want, flush=True)
    if not np.allclose(got, want, rtol=1e-6):
        raise SystemExit("image differs from the one the labels were made on")
    v = np.asarray(d["voxel_um"], float)
    prov = {"cube": "slap2-878030-dmd1", "brain": "878030", "split": "all", "modality": "slap2 in vivo 2p",
            "image": Path(a.tif).name, "voxel_um": v.tolist(), "label_stats": json.loads(str(d["stats"])),
            "voxels": {k.name.lower(): int((lab == k).sum()) for k in (Label.AXON, Label.BACKGROUND, Label.IGNORE)}}  # fmt: skip
    write_training_volume(Path(a.out), img, lab, LevelTransform(scale=tuple(v), translation=tuple(v / 2)),
                          Box.from_arrays(np.zeros(3, int), np.array(img.shape)), prov)  # fmt: skip
    print("wrote", a.out, prov["voxels"], flush=True)


def predict_volume(p: TilePredictor, x: np.ndarray, tile: int) -> np.ndarray:
    step = tile - 2 * CROP
    pad = [(CROP, CROP + (-s) % step) for s in x.shape]
    v = np.pad(x, pad, mode="reflect")
    out = np.zeros([s - 2 * CROP for s in v.shape], np.float32)
    for z in range(0, out.shape[0], step):
        for y in range(0, out.shape[1], step):
            for xx in range(0, out.shape[2], step):
                t = v[z : z + tile, y : y + tile, xx : xx + tile]
                t = np.pad(t, [(0, tile - s) for s in t.shape], mode="reflect")
                o = p(np.ascontiguousarray(t)[None])[0]
                sz = out[z : z + step, y : y + step, xx : xx + step].shape
                out[z : z + step, y : y + step, xx : xx + step] = o[: sz[0], : sz[1], : sz[2]]
    return out[: x.shape[0], : x.shape[1], : x.shape[2]]


def predict(a: argparse.Namespace) -> None:
    raw, lab, _ = open_training_volume(Path(a.volume))
    scaling = estimate_scaling_array(
        raw, lab != Label.IGNORE, NormalizationConfig()
    )  # as in build_volumes
    x = scaling(raw)
    device = resolve_device(a.device)
    rows, saved = [], {}
    for k, w in enumerate(a.weights):
        t0 = time.time()
        prob = predict_volume(TilePredictor(load_model(w), device, torch.float32), x, a.tile)
        hit = prob >= 0.5
        row = {"weights": w, "scaling": scaling.as_dict(), "recall_positive": float(hit[lab == Label.AXON].mean()),
               "positive_rate_background": float(hit[lab == Label.BACKGROUND].mean()),
               "positive_rate_ignore": float(hit[lab == Label.IGNORE].mean()),
               "positive_fraction": float(hit.mean()), "seconds": round(time.time() - t0, 1)}  # fmt: skip
        print("PREDICT", json.dumps(row), flush=True)
        rows.append(row)
        saved[f"prob_{k}"] = prob.astype(np.float16)
    np.savez_compressed(a.out, results=json.dumps(rows), **saved)
    print("saved", a.out, flush=True)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("volume")
for k in ("tif", "labels", "out"):
    s.add_argument(k)
s = sub.add_parser("predict")
s.add_argument("volume")
s.add_argument("out")
s.add_argument("--weights", action="append", required=True)
s.add_argument("--tile", type=int, default=128)
s.add_argument("--device", default="cuda")
args = ap.parse_args()
{"volume": volume, "predict": predict}[args.cmd](args)
