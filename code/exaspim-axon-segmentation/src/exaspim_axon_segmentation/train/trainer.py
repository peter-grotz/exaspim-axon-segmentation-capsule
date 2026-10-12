"""Fine-tune the TRAILMAP U-Net on a training volume (raw + labels from complete traces)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file
from torch import nn

from exaspim_axon_segmentation.inference.normalize import IntensityScaling
from exaspim_axon_segmentation.model import TrailmapUNet3D
from exaspim_axon_segmentation.model.unet import CONTEXT
from exaspim_axon_segmentation.train.labels import Label

log = logging.getLogger(__name__)

# TRAILMAP's class weights (axon 1.5, background 0.2, artifact 0.8, edge 0.05); ignore is 0.
DEFAULT_WEIGHTS = {
    Label.AXON: 1.5,
    Label.BACKGROUND: 0.2,
    Label.HARD_NEGATIVE: 0.8,
    Label.EDGE: 0.05,
    Label.IGNORE: 0.0,
}


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning settings."""

    input_size: int = 96
    batch_size: int = 4
    steps: int = 4000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_every: int = 250
    val_patches: int = 64
    val_fraction: float = 0.2
    frac_axon: float = 0.5
    frac_negative: float = 0.25
    intensity_jitter: float = 0.35
    freeze_batchnorm: bool = True
    max_minutes: float | None = 40.0
    seed: int = 0
    # 1x look for coarse (averaged 5x) volumes, on normalized patches: Gaussian blur (sigma in
    # voxels, z and y/x), contrast gain, additive noise SD; each scaled per patch by a random
    # factor in [1 - degrade_jitter, 1 + degrade_jitter] (gain: drawn in [gain_lo, gain_hi]).
    degrade_blur: tuple[float, float] | None = None
    degrade_gain: tuple[float, float] = (1.0, 1.0)
    degrade_noise_sd: float = 0.0
    degrade_jitter: float = 0.5

    @property
    def output_size(self) -> int:
        """Label/prediction side of one patch."""
        return self.input_size - 2 * CONTEXT


class PatchSampler:
    """Draws (input, label) patch pairs from one side of a spatial split.

    Parameters
    ----------
    raw, labels : numpy.ndarray
        Training volume (same shape).
    x_range : tuple of int
        Patch centers are drawn with x in ``[x0, x1)``; the other split is never touched.
    config : TrainConfig
        Patch geometry and sampling mix.
    rng : numpy.random.Generator
        Randomness source.
    """

    def __init__(
        self,
        raw: NDArray[np.uint16],
        labels: NDArray[np.uint8],
        x_range: tuple[int, int],
        config: TrainConfig,
        rng: np.random.Generator,
    ) -> None:
        self.raw, self.labels, self.config, self.rng = raw, labels, config, rng
        self.x0, self.x1 = x_range
        half = config.output_size // 2
        sub = labels[:, :, self.x0 : self.x1]
        self._axon = self._centers(sub == Label.AXON)
        self._negative = self._centers(sub == Label.HARD_NEGATIVE)
        self._inside = self._centers(sub != Label.IGNORE, stride=8)
        self._half = half
        log.info(
            "Sampler x[%d:%d): %d axon, %d hard-negative, %d region centers",
            self.x0, self.x1, len(self._axon), len(self._negative), len(self._inside),
        )  # fmt: skip

    def _centers(self, mask: NDArray[np.bool_], stride: int = 1) -> NDArray[np.int64]:
        idx = np.argwhere(mask[::stride, ::stride, ::stride]) * stride
        idx[:, 2] += self.x0
        return idx

    def center(self) -> NDArray[np.int64]:
        """Draw a patch center from the configured mix of axon / negative / region voxels."""
        u = self.rng.random()
        pool = (
            self._axon
            if u < self.config.frac_axon
            else self._negative
            if u < self.config.frac_axon + self.config.frac_negative and len(self._negative)
            else self._inside
        )
        c = pool[self.rng.integers(len(pool))]
        jitter = self.rng.integers(-self._half // 2, self._half // 2 + 1, size=3)
        return c + jitter

    def patch(self, center: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """Return raw input (``input_size``³, edge-padded) and labels (``output_size``³)."""
        s_in, s_out = self.config.input_size, self.config.output_size
        raw = _crop(self.raw, center - s_in // 2, s_in, mode="edge").astype(np.float32)
        lab = _crop(self.labels, center - s_out // 2, s_out, mode="constant")
        # Voxels on the other side of the split are never scored.
        x_global = center[2] - s_out // 2 + np.arange(s_out)
        lab[:, :, (x_global < self.x0) | (x_global >= self.x1)] = Label.IGNORE
        return raw, lab


def _crop(volume: NDArray, start: NDArray, size: int, *, mode: str) -> NDArray:
    start = np.asarray(start)
    stop = start + size
    lo = np.clip(start, 0, volume.shape)
    hi = np.clip(stop, lo, volume.shape)
    block = volume[tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))]
    pad = [(a - s, e - b) for s, e, a, b in zip(start, stop, lo, hi, strict=True)]
    if not any(p for pair in pad for p in pair):
        return block.copy()
    if block.size == 0 or mode == "constant":
        return np.pad(block, pad, mode="constant", constant_values=0)
    return np.pad(block, pad, mode="edge")


def augment(
    raw: NDArray[np.float32], lab: NDArray[np.uint8], rng: np.random.Generator, jitter: float
) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """Random y/x flips, in-plane 90° rotation, and multiplicative intensity jitter.

    Z is never flipped into y/x: exaSPIM voxels are anisotropic (4 µm z vs 3 µm y/x at level 2).
    """
    for axis in (1, 2):
        if rng.random() < 0.5:
            raw, lab = np.flip(raw, axis), np.flip(lab, axis)
    if rng.random() < 0.5:
        raw, lab = raw.transpose(0, 2, 1), lab.transpose(0, 2, 1)
    gain = np.exp(rng.uniform(-jitter, jitter))
    return np.ascontiguousarray(raw * gain), np.ascontiguousarray(lab)


def degrade(x: NDArray[np.float32], rng: np.random.Generator, config: TrainConfig) -> NDArray:
    """Make a normalized averaged-5x patch look like a 1x one: blur, contrast, noise, clip."""
    if config.degrade_blur is None:
        return x
    from scipy.ndimage import gaussian_filter

    j = config.degrade_jitter
    sz, sxy = (s * rng.uniform(1 - j, 1 + j) for s in config.degrade_blur)
    x = gaussian_filter(x, (sz, sxy, sxy)) if max(sz, sxy) > 0 else x
    x = x * rng.uniform(*config.degrade_gain)
    sd = config.degrade_noise_sd * rng.uniform(1 - j, 1 + j)
    x = x + rng.normal(0.0, sd, x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def weighted_bce(
    logits: torch.Tensor, labels: torch.Tensor, weights: dict[Label, float]
) -> torch.Tensor:
    """TRAILMAP's weighted binary cross-entropy (axon = positive), normalized by total weight."""
    table = torch.zeros(len(Label), device=labels.device)
    for label, w in weights.items():
        table[int(label)] = w
    w = table[labels.long()]
    target = (labels == Label.AXON).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, weight=w, reduction="sum")
    return loss / w.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model: TrailmapUNet3D,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    autocast: torch.dtype | None,
) -> dict[str, float]:
    """Voxel metrics at p > 0.5 over labelled voxels (edge voxels excluded)."""
    model.eval()
    tp = fp = fn = neg_fp = neg_n = bg_fp = bg_n = 0
    for x, y in batches:
        with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
            p = model(x.to(device)).float().squeeze(1)
        y = y.to(device)
        pos = p > 0.5
        axon, neg, bg = y == Label.AXON, y == Label.HARD_NEGATIVE, y == Label.BACKGROUND
        tp += int((pos & axon).sum())
        fn += int((~pos & axon).sum())
        fp += int((pos & (neg | bg)).sum())
        neg_fp, neg_n = neg_fp + int((pos & neg).sum()), neg_n + int(neg.sum())
        bg_fp, bg_n = bg_fp + int((pos & bg).sum()), bg_n + int(bg.sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "hard_negative_fp_rate": neg_fp / max(neg_n, 1),
        "background_fp_rate": bg_fp / max(bg_n, 1),
    }


def split_by_axon_x(labels: NDArray[np.uint8], val_fraction: float) -> int:
    """X index such that ``val_fraction`` of axon voxels lie at or beyond it."""
    x = np.argwhere(labels == Label.AXON)[:, 2]
    return int(np.quantile(x, 1.0 - val_fraction))


@dataclass(eq=False)
class TrainingVolume:
    """One training volume with its own intensity scaling and split.

    Attributes
    ----------
    name : str
        Identifier used in logs and metrics.
    raw, labels : numpy.ndarray
        Image and labels on the same grid.
    scaling : IntensityScaling
        Normalization for this volume (estimated from the volume itself).
    train_x, val_x : tuple of int or None
        x ranges used for training / validation; ``None`` excludes the volume from that role.
    group : str
        Validation group the volume reports under (e.g. ``"822178"`` or ``"heldout-744783"``).
    weight : float
        Relative sampling weight for training patches.
    """

    name: str
    raw: NDArray[np.uint16]
    labels: NDArray[np.uint8]
    scaling: IntensityScaling
    train_x: tuple[int, int] | None
    val_x: tuple[int, int] | None
    group: str
    weight: float = 1.0


def finetune(
    model: TrailmapUNet3D,
    volumes: list[TrainingVolume],
    out_dir: Path,
    config: TrainConfig,
    *,
    device: torch.device,
    weights: dict[Label, float] | None = None,
) -> dict:
    """Fine-tune in place; writes ``best.safetensors``, ``last.safetensors``, ``metrics.jsonl``.

    Validation runs per group at step 0 (the starting weights, as baseline) and every
    ``val_every`` steps; the best checkpoint maximizes the mean F1 over groups. Every validation
    line is also logged (``METRICS {...}``) so results survive a run whose result sync fails.
    """
    weights = weights or DEFAULT_WEIGHTS
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    train = [
        (v, PatchSampler(v.raw, v.labels, v.train_x, config, rng)) for v in volumes if v.train_x
    ]
    if not train:
        raise ValueError("No volume has a training range")
    probs = np.array([v.weight for v, _ in train], dtype=np.float64)
    probs /= probs.sum()

    val_rng = np.random.default_rng(1234)
    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for v in volumes:
        if not v.val_x:
            continue
        sampler = PatchSampler(v.raw, v.labels, v.val_x, config, val_rng)
        for _ in range(max(1, config.val_patches // config.batch_size)):
            xs, ys = zip(*(sampler.patch(sampler.center()) for _ in range(config.batch_size)),
                         strict=True)  # fmt: skip
            batch = torch.stack(
                [torch.from_numpy(degrade(v.scaling(x), val_rng, config))[None] for x in xs]
            )
            groups.setdefault(v.group, []).append((batch, torch.from_numpy(np.stack(ys))))
    log.info("Training volumes %s; validation groups %s",
             {v.name: round(float(p), 3) for (v, _), p in zip(train, probs, strict=True)},
             {g: len(b) for g, b in groups.items()})  # fmt: skip

    model.to(device)
    # bf16 only where the GPU has it (A10G yes; T4 emulates it, ~15x slower): else fp32
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    autocast = torch.bfloat16 if use_bf16 else None
    log.info("Training on %s, %s", device, "bf16 autocast" if use_bf16 else "fp32")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    best: dict = {"mean_f1": -1.0}

    def validate(step: int, train_loss: float | None) -> None:
        nonlocal best
        per_group = {g: evaluate(model, b, device, autocast) for g, b in groups.items()}
        m = {
            "step": step,
            "train_loss": train_loss,
            "mean_f1": float(np.mean([r["f1"] for r in per_group.values()])),
            **{f"{g}/{k}": v for g, r in per_group.items() for k, v in r.items()},
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(m) + "\n")
        log.info("METRICS %s", json.dumps(m))
        if m["mean_f1"] > best["mean_f1"]:
            best = m
            save_file(model.state_dict(), str(out_dir / "best.safetensors"))

    validate(0, None)  # starting weights, on the same validation patches
    start, losses, step = time.perf_counter(), [], 0
    for step in range(1, config.steps + 1):
        model.train()
        if config.freeze_batchnorm:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
        inputs, targets = [], []
        for k in rng.choice(len(train), size=config.batch_size, p=probs):
            volume, sampler = train[k]
            r, lab = augment(*sampler.patch(sampler.center()), rng, config.intensity_jitter)
            inputs.append(torch.from_numpy(degrade(volume.scaling(r), rng, config))[None])
            targets.append(lab)
        x = torch.stack(inputs).to(device)
        y = torch.from_numpy(np.stack(targets)).to(device)
        with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
            logits = model.logits(x).float().squeeze(1)
        loss = weighted_bce(logits, y, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        losses.append(float(loss))
        elapsed = time.perf_counter() - start
        out_of_time = config.max_minutes is not None and elapsed > 60 * config.max_minutes
        if step % config.val_every == 0 or step == config.steps or out_of_time:
            recent = float(np.mean(losses[-config.val_every :]))
            log.info("step %d/%d · loss %.4f · %.2f s/step", step, config.steps, recent,
                     elapsed / step)  # fmt: skip
            validate(step, recent)
        if out_of_time:
            log.info("Stopped at step %d: %.0f min budget reached", step, config.max_minutes)
            break

    save_file(model.state_dict(), str(out_dir / "last.safetensors"))
    summary = {
        "best": best,
        "config": asdict(config),
        "weights": {k.name.lower(): v for k, v in weights.items()},
        "volumes": [
            {"name": v.name, "group": v.group, "train_x": v.train_x, "val_x": v.val_x,
             "weight": v.weight, "scaling": v.scaling.as_dict()}
            for v in volumes
        ],
        "steps_run": step,
        "seconds": time.perf_counter() - start,
    }  # fmt: skip
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Best validation: %s", json.dumps(best))
    return summary


def build_volumes(
    dirs: list[Path],
    config: TrainConfig,
    *,
    holdout_brains: list[str],
    region_share: float = 0.5,
) -> list[TrainingVolume]:
    """Load training volumes and assign roles.

    Region volumes (traced lobes) are split along x: the last ``config.val_fraction`` of their
    axon voxels is validation. Blocks train unless their brain is held out, in which case they are
    validation only. Region volumes together receive ``region_share`` of the training patches.
    """
    from exaspim_axon_segmentation.config import NormalizationConfig
    from exaspim_axon_segmentation.inference.normalize import estimate_scaling_array
    from exaspim_axon_segmentation.train.volume import open_training_volume

    volumes = []
    for d in dirs:
        raw, labels, info = open_training_volume(d)
        scaling = estimate_scaling_array(raw, labels != Label.IGNORE, NormalizationConfig())
        width = raw.shape[2]
        if "cube" in info:  # coarse trace cubes: split recorded per cube
            val = info.get("split") == "val"
            volumes.append(TrainingVolume(
                name=f"{info['brain']}/{info['cube']}", raw=raw, labels=labels, scaling=scaling,
                train_x=None if val else (0, width), val_x=(0, width) if val else None,
                group=f"heldout-{info['brain']}" if val else "blocks",
            ))  # fmt: skip
        elif "block" in info:
            held = info["brain"] in holdout_brains
            volumes.append(TrainingVolume(
                name=f"{info['brain']}/{info['block']}", raw=raw, labels=labels, scaling=scaling,
                train_x=None if held else (0, width), val_x=(0, width) if held else None,
                group=f"heldout-{info['brain']}" if held else "blocks",
            ))  # fmt: skip
        else:
            brain = info["asset"].rstrip("/").split("/")[-1].split("_")[1]
            split = split_by_axon_x(labels, config.val_fraction)
            volumes.append(TrainingVolume(
                name=f"{brain}/{'+'.join(info['regions'])}", raw=raw, labels=labels,
                scaling=scaling, train_x=(0, split), val_x=(split, width),
                group=f"{brain}-{'+'.join(info['regions'])}",
            ))  # fmt: skip
        log.info("Volume %s: scaling offset %.1f scale %.1f", volumes[-1].name, scaling.offset,
                 scaling.scale)  # fmt: skip
    regions = [v for v in volumes if v.train_x and v.group != "blocks"]
    blocks = [v for v in volumes if v.train_x and v.group == "blocks"]
    if regions and blocks:
        for v in regions:
            v.weight = region_share / len(regions)
        for v in blocks:
            v.weight = (1 - region_share) / len(blocks)
    return volumes
