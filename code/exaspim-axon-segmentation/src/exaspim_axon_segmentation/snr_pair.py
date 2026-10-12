"""Axon SNR at the same places in two acquisitions of one brain (e.g. a 1x and a 5x scan).

The two volumes are related by a whole-brain affine, ``p_b = A p_a + t`` in physical µm (z, y, x),
fitted beforehand on low-resolution overviews. Each block centre in acquisition *a* is mapped into
*b* and refined locally, coarse then fine, by template matching of band-passed log intensity (1x
block on *b* resampled to *a*'s voxel grid); blocks whose fine match is weak are kept but
flagged. With ``"match": false`` in the affine, blocks are placed by the affine alone (e.g. one
built from metadata) and the cross metrics, which need the same tissue in both, are skipped.

Both blocks are segmented natively (each at its own, correct voxel size) and scored as in
:mod:`exaspim_axon_segmentation.snr`, with the acquisition QC of
:mod:`exaspim_axon_segmentation.qc.detection_qc`. Cross metrics use *b*'s detection (the higher
resolution one) mapped onto *a*'s grid:

* ``a_snr_on_b_mask``: SNR of *a* measured on the axons detected in *b*,
* ``recall_a`` / ``precision_a``: agreement of *a*'s detection with *b*'s, within ``tol_vox`` of
  *a*'s voxels,
* ``b.at_a_voxel``: *b* mean-pooled to *a*'s voxel size on its own grid and scored on its own
  (max-pooled) detection, which needs no alignment,
* ``b.down_to_a``: that mean-pooled *b* segmented again by the network and scored exactly as *a*
  (same voxel size, detector and scoring),
* ``a_matched`` / ``b_matched``: signal, background, noise and SNR of both acquisitions at the same
  voxel size (*b* averaged over each of *a*'s voxels) on the same mask (*b*'s axons).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    affine_transform,
    binary_dilation,
    gaussian_filter,
    maximum_filter,
    uniform_filter,
    zoom,
)

from exaspim_axon_segmentation.snr import SnrConfig, _predict, _read, open_ome, score

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairConfig:
    """Settings of :func:`snr_pair`."""

    block_um: tuple[float, float, float] = (1280.0, 1440.0, 1440.0)
    level_a: int = 0
    level_b: int = 2  # 5x fused level 2 = 2.992 x 2.992 x 4 um, the training grid
    coarse_a: int = 2
    coarse_b: int = 5
    fine_b: int = 3
    coarse_margin_um: float = 600.0
    fine_margin_um: float = 150.0
    good_ncc: float = 0.3
    axon_p: float = 0.5
    tol_vox: int = 1


def _feat(x: NDArray) -> NDArray:
    y = np.log1p(np.clip(x - np.percentile(x, 30), 0, None))
    return gaussian_filter(y, 1.0) - gaussian_filter(y, 4.0)


def _crop(lv, centre, half):
    array, lead, sc, tr = lv
    shape = np.asarray(array.shape[lead:])
    lo = np.clip(np.round((centre - half - tr) / sc).astype(int), 0, shape)
    hi = np.clip(np.round((centre + half - tr) / sc).astype(int), 0, shape)
    return _read(array, lead, lo, hi), tr + lo * sc


def _match(la, lb, aff, pa, pb, half, margin) -> tuple[NDArray, float, float] | None:
    """Where ``pa`` lands in *b*: template match of *a*'s crop in *b*'s, on *a*'s grid."""
    from skimage.feature import match_template

    s = float(np.abs(np.linalg.svd(aff)[1]).mean())
    a, oa = _crop(la, pa, half)
    b, ob = _crop(lb, pb, (half + margin) * s)
    rb = zoom(b, lb[2] / (s * la[2]), order=1)
    fa, fb = _feat(a), _feat(rb)
    if a.size == 0 or np.any(np.array(fb.shape) <= np.array(fa.shape)):
        return None
    m = match_template(fb, fa)
    idx = np.array(np.unravel_index(np.argmax(m), m.shape))
    second = m.copy()
    r = np.maximum(np.array(fa.shape) // 4, 2)
    second[tuple(slice(max(i - k, 0), i + k) for i, k in zip(idx, r, strict=True))] = -1
    return ob + idx * s * la[2] + (pa - oa) * s, float(m.max()), float(second.max())


def _to_a(vol_b, ob, scb, lin_map, shift, oa, sca, shape_a, order=1):
    """Resample from *b*'s grid onto *a*'s, for ``p_b = lin_map p_a + shift`` (physical µm)."""
    lin = (lin_map * sca[None, :]) / scb[:, None]
    off = (lin_map @ oa + shift - ob) / scb
    return affine_transform(vol_b, lin, off, output_shape=shape_a, order=order, cval=0)


class RawTiles:
    """Raw tiles placed by a BigStitcher XML; the fused volume's origin is the tiles' bounding box.

    Maps fused physical (z, y, x µm) to each tile's own physical frame, so blocks can be read from
    the raw camera data instead of the processed (denoised, flat-field corrected) fusion.
    """

    def __init__(self, root: str, xml: str, voxel0_zyx) -> None:
        import xml.etree.ElementTree as ET

        r = ET.parse(xml).getroot()
        self.names, self.sizes, self.world = {}, {}, {}
        for vs in r.iter("ViewSetup"):
            i = int(vs.find("id").text)
            self.names[i] = vs.find("name").text
            self.sizes[i] = np.array([int(x) for x in vs.find("size").text.split()])
        for vr in r.iter("ViewRegistration"):
            m = np.eye(4)
            for tr in vr.findall("ViewTransform"):
                a = np.eye(4)
                a[:3] = np.array([float(x) for x in tr.find("affine").text.split()]).reshape(3, 4)
                m = m @ a  # listed outermost first
            self.world[int(vr.get("setup"))] = m
        lo = np.full(3, np.inf)
        for i, m in self.world.items():
            sx, sy, sz = self.sizes[i]
            c = np.array([[x, y, z, 1] for x in (0, sx) for y in (0, sy) for z in (0, sz)]).T
            lo = np.minimum(lo, (m @ c)[:3].min(1))
        self.lo, self.root, self.v0 = lo, root.rstrip("/"), np.asarray(voxel0_zyx, float)

    def fused_to_voxel(self, i: int) -> NDArray:
        """4x4: fused physical z, y, x (µm) -> tile ``i`` level-0 voxel z, y, x."""
        perm = np.eye(4)[[2, 1, 0, 3]]
        f = np.diag([*(1 / self.v0[::-1]), 1.0])
        f[:3, 3] = self.lo
        return perm @ np.linalg.inv(self.world[i]) @ f @ perm

    def pick(self, corners_fused: NDArray) -> int:
        """Pick the tile that holds the block best (largest smallest margin, in voxels)."""
        best, score_ = -1, -np.inf
        for i in self.world:
            m = self.fused_to_voxel(i)
            v = (m[:3, :3] @ corners_fused.T).T + m[:3, 3]
            size = self.sizes[i][::-1]
            margin = min(v.min(), (size - v).min())
            if margin > score_:
                best, score_ = i, margin
        return best

    def url(self, i: int) -> str:
        """OME-Zarr URL of tile ``i``."""
        return f"{self.root}/{self.names[i]}.zarr"


def _corners(o, shape, sc):
    c = np.array([[z, y, x] for z in (0, 1) for y in (0, 1) for x in (0, 1)])
    return o + c * np.asarray(shape) * sc


def snr_pair(
    url_a: str,
    url_b: str,
    affine: dict,
    centres_um: NDArray,
    out_dir: Path,
    weights: str,
    config: PairConfig = PairConfig(),  # noqa: B008
    *,
    raw_tiles: RawTiles | None = None,
    weights_a: str | None = None,
    device: str = "auto",
) -> list[dict]:
    """Segment and score matched blocks in both acquisitions; writes JSON and preview MIPs.

    With ``raw_tiles``, *b* is read from its raw tiles (key ``b``) and also from the fused volume
    ``url_b`` (key ``b_fused``); without, from the fused volume only (key ``b``). ``weights_a``
    segments *a* with other weights than *b* (default: the same).
    """
    import torch

    from exaspim_axon_segmentation.config import InferenceConfig
    from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array
    from exaspim_axon_segmentation.inference.predict import (
        TilePredictor,
        resolve_device,
        resolve_precision,
    )
    from exaspim_axon_segmentation.model import load_model
    from exaspim_axon_segmentation.qc.detection_qc import DetectionQcConfig, block_qc

    out_dir.mkdir(parents=True, exist_ok=True)
    aff, t = np.asarray(affine["A"], float), np.asarray(affine["t"], float)
    la, lb = open_ome(url_a, config.level_a), open_ome(url_b, config.level_b)
    ca, cb = open_ome(url_a, config.coarse_a), open_ome(url_b, config.coarse_b)
    fb = open_ome(url_b, config.fine_b)
    norm = InferenceConfig(weights=weights).normalization
    dev = resolve_device(device)
    precision = resolve_precision("auto") if dev.type == "cuda" else torch.float32
    predictor = TilePredictor(load_model(weights), dev, precision)
    predict_b = lambda v: _predict(predictor, v, 160)  # noqa: E731
    predict_a = predict_b
    if weights_a is not None:  # e.g. a model fine-tuned for a's voxel size
        predictor_a = TilePredictor(load_model(weights_a), dev, precision)
        predict_a = lambda v: _predict(predictor_a, v, 160)  # noqa: E731
    sc = SnrConfig(axon_p=config.axon_p)
    qc_cfg = DetectionQcConfig(axon_p=config.axon_p)
    half = np.asarray(config.block_um) / 2
    tiles: dict = {}
    do_match = affine.get("match", True)  # False: place blocks by the affine alone
    rows = []
    for k, pa in enumerate(np.atleast_2d(centres_um)):
        t0 = time.time()
        row: dict = {"block": k, "centre_a_um": np.round(pa, 1).tolist()}
        pb0 = aff @ pa + t
        if do_match:
            coarse = _match(ca, cb, aff, pa, pb0, half, np.full(3, config.coarse_margin_um))
            fine = (
                None
                if coarse is None
                else _match(la, fb, aff, pa, coarse[0], half, np.full(3, config.fine_margin_um))
            )
            if fine is None:
                rows.append(row | {"skipped": "match crop outside a volume"})
                continue
            pb, ncc, ncc2 = fine
        else:  # metadata mapping only: no image matching
            pb, ncc, ncc2 = pb0, float("nan"), float("nan")
        d = pb - pb0
        row |= {
            "centre_b_um": np.round(pb, 1).tolist(),
            "refine_um": np.round(d, 1).tolist(),
            "match_ncc": ncc,
            "match_second": ncc2,
            "matched": bool(ncc >= config.good_ncc) if do_match else None,
        }
        raw_a, oa = _crop(la, pa, half)
        corners_a = _corners(oa, raw_a.shape, la[2])
        # each b variant: (pyramid level, map p_a -> p_b in that volume's physical frame)
        variants = {}
        fused_name = "b_fused" if raw_tiles is not None else "b"
        variants[fused_name] = (lb, aff, t + d)
        if raw_tiles is not None:
            i = raw_tiles.pick((aff @ corners_a.T).T + t + d)
            if i not in tiles:
                tiles[i] = (
                    open_ome(raw_tiles.url(i), 0),
                    open_ome(raw_tiles.url(i), config.level_b),
                )
            (_, _, sc0, tr0), lvl = tiles[i]
            m = np.diag([*sc0, 1.0]) @ raw_tiles.fused_to_voxel(i)
            m[:3, 3] += tr0
            variants["b"] = (lvl, m[:3, :3] @ aff, m[:3, :3] @ (t + d) + m[:3, 3])
            row["b_tile"] = raw_tiles.names[i]
        blocks = {"a": (raw_a, oa, la[2], None)}
        for name, (lv, lin_map, shift) in variants.items():
            pts = (lin_map @ corners_a.T).T + shift
            cen = (pts.min(0) + pts.max(0)) / 2
            raw, ob = _crop(lv, cen, (pts.max(0) - pts.min(0)) / 2 + 2 * lv[2])
            blocks[name] = (raw, ob, lv[2], (lin_map, shift))
        probs = {}
        for name, (raw, _, vox, _) in blocks.items():
            if min(raw.shape) < 64:
                row[name] = {"skipped": f"block {raw.shape} smaller than the network input"}
                continue
            try:
                scaling = estimate_scaling_array(raw, raw > np.percentile(raw, 10), norm)
            except ValueError as error:
                row[name] = {"skipped": f"no foreground: {error}", "verdict": "no_signal"}
                continue
            predict = predict_a if name == "a" else predict_b
            prob = predict(scaling(raw))
            row[name] = score(raw, prob, sc) | {
                "shape": list(raw.shape),
                "voxel_um": vox.tolist(),
                "scaling": [scaling.offset, scaling.scale],
            }
            row[name]["qc"] = block_qc(raw, predict, scaling, prob=prob, config=qc_cfg)
            if name != "a":  # b at a's voxel size on its own grid: mean-pool image, max-pool mask
                f = tuple(int(x) for x in np.maximum(np.round(la[2] / vox).astype(int), 1))
                cut = tuple(slice(0, (n // q) * q) for n, q in zip(raw.shape, f, strict=True))
                shp = [(n // q, q) for n, q in zip(raw.shape, f, strict=True)]
                pooled = raw[cut].reshape(shp[0] + shp[1] + shp[2]).mean((1, 3, 5))
                mask = prob[cut].reshape(shp[0] + shp[1] + shp[2]).max((1, 3, 5))
                keys = ("axon_fraction", "signal", "background", "noise_sd", "snr")
                at_a = score(pooled, mask, sc)
                row[name]["at_a_voxel"] = {key: at_a.get(key) for key in keys} | {
                    "voxel_um": (vox * np.array(f)).tolist(),
                    "detection": f"{name} detections, max-pooled",
                }
                # the same averaged image segmented again, exactly as a is: same voxel size,
                # same detector, same scoring
                down = {"voxel_um": (vox * np.array(f)).tolist(), "shape": list(pooled.shape)}
                if min(pooled.shape) < 64:
                    down["skipped"] = (
                        f"averaged block {pooled.shape} smaller than the network input"
                    )
                else:
                    try:
                        s_down = estimate_scaling_array(
                            pooled, pooled > np.percentile(pooled, 10), norm
                        )
                        p_down = predict(s_down(pooled))
                        down |= {key: score(pooled, p_down, sc).get(key) for key in keys}
                        down["scaling"] = [s_down.offset, s_down.scale]
                        np.savez_compressed(
                            out_dir / f"block{k:03d}_{name}_down.npz",
                            raw_mip=pooled.max(0),
                            prob_mip=np.round(p_down.max(0) * 255).astype(np.uint8),
                        )
                    except ValueError as error:
                        down["skipped"] = f"no foreground: {error}"
                row[name]["down_to_a"] = down
            np.savez_compressed(
                out_dir / f"block{k:03d}_{name}.npz",
                raw_mip=raw.max(0),
                prob_mip=np.round(prob.max(0) * 255).astype(np.uint8),
            )
            probs[name] = prob
        det_a = probs["a"] >= config.axon_p if "a" in probs else None
        for name in variants:
            raw, ob, vox, (lin_map, shift) = blocks[name]
            if raw.size:  # raw vs fused agreement, both on a's grid
                row.setdefault("_on_a", {})[name] = _to_a(
                    raw, ob, vox, lin_map, shift, oa, la[2], raw_a.shape
                )
            if not do_match or det_a is None or name not in probs:
                continue
            # b's axons on a's grid: max-pool to a's voxel size first so thin axons survive
            size = np.maximum(np.round(la[2] / vox).astype(int), 1)
            pooled = maximum_filter(probs[name], size=tuple(size))
            b_on_a = _to_a(pooled, ob, vox, lin_map, shift, oa, la[2], raw_a.shape)
            b_on_a = b_on_a >= config.axon_p
            np.savez_compressed(out_dir / f"block{k:03d}_{name}_on_a.npz", mip=b_on_a.max(0))
            tol = binary_dilation(b_on_a, iterations=config.tol_vox) if b_on_a.any() else b_on_a
            tol_a = binary_dilation(det_a, iterations=config.tol_vox) if det_a.any() else det_a
            mask = b_on_a.astype(np.float32)
            # b at a's voxel size: mean over each a voxel's footprint, sampled on a's grid, so
            # both acquisitions are scored with the same voxels and the same axon mask
            mean_b = uniform_filter(raw.astype(np.float32), size=tuple(size))
            b_at_a = _to_a(mean_b, ob, vox, lin_map, shift, oa, la[2], raw_a.shape)
            inside = _to_a(np.ones(raw.shape, np.float32), ob, vox, lin_map, shift, oa, la[2],
                           raw_a.shape, order=0)  # fmt: skip
            keys = ("signal", "background", "noise_sd", "snr")
            sa, sb = score(raw_a, mask, sc), score(b_at_a, mask, sc)
            row["cross" if name == "b" else f"cross_{name}"] = {
                "b_axon_fraction_on_a": float(b_on_a.mean()),
                "a_snr_on_b_mask": sa.get("snr"),
                "recall_a": float((b_on_a & tol_a).sum() / max(b_on_a.sum(), 1)),
                "precision_a": float((det_a & tol).sum() / max(det_a.sum(), 1)),
                "matched_voxel_um": la[2].tolist(),
                "inside_fraction": float((inside > 0).mean()),
                "a_matched": {key: sa.get(key) for key in keys},
                "b_matched": {key: sb.get(key) for key in keys},
            }
        on_a = row.pop("_on_a", {})
        if "b" in on_a and "b_fused" in on_a:
            x, y = _feat(on_a["b"]), _feat(on_a["b_fused"])
            x, y = x - x.mean(), y - y.mean()
            row["raw_vs_fused_ncc"] = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        log.info(
            "block %d: match %.2f, a SNR %s, b SNR %s, cross %s",
            k,
            ncc,
            row.get("a", {}).get("snr"),
            row.get("b", {}).get("snr"),
            row.get("cross"),
        )
        summary = {
            "url_a": url_a,
            "url_b": url_b,
            "raw_tiles": None if raw_tiles is None else raw_tiles.root,
            "affine": affine,
            "config": asdict(config),
            "blocks": rows,
        }
        (out_dir / "snr_pair.json").write_text(json.dumps(summary, indent=1))
    return rows
