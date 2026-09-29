"""Command-line interface: ``exaspim-axon <command>``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from exaspim_axon_segmentation.config import InferenceConfig, NormalizationConfig
from exaspim_axon_segmentation.io import CcfOntology, ExaspimAsset, RegionRoi

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("exaspim_axon_segmentation")

AssetArg = Annotated[str, typer.Argument(help="Asset root, e.g. s3://aind-open-data/exaSPIM_...")]
RegionOpt = Annotated[list[str], typer.Option("--region", "-r", help="CCF ID or acronym; repeat")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """TRAILMAP axon segmentation for exaSPIM."""
    # --verbose raises only this package's loggers; DEBUG on the root logger floods the log with
    # botocore/s3fs request traces (18 MB for a 1-minute run).
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log.setLevel(logging.DEBUG if verbose else logging.INFO)


@app.command("qc-region")
def qc_region(
    asset: AssetArg,
    region: RegionOpt,
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/region_overlay.png"),
    detail_um: Annotated[float, typer.Option(help="XY voxel size of zoomed rows")] = 6.0,
) -> None:
    """Draw the sample-space CCF region outline on the fused image for visual confirmation."""
    from exaspim_axon_segmentation.qc import render_region_overlay

    ontology = CcfOntology.load()
    data = ExaspimAsset(asset)
    label_ids = ontology.resolve(region)
    roi = RegionRoi.from_annotation(data.annotation, label_ids)
    log.info("ROI %s: %.3f mm³, bbox %s to %s um", region, roi.volume_mm3, roi.lo_um, roi.hi_um)
    render_region_overlay(data, roi, ontology[region[0]], out, detail_um=detail_um)
    log.info("Wrote %s", out)


@app.command()
def predict(
    asset: AssetArg,
    region: RegionOpt,
    out: Annotated[str, typer.Option(help="Output OME-Zarr (local path or s3://)")],
    level: Annotated[int, typer.Option(help="fused.zarr pyramid level")] = 2,
    tile_size: Annotated[int, typer.Option(help="Network input side (multiple of 8)")] = 160,
    batch_size: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option(help="auto | cuda | mps | cpu")] = "auto",
    norm: Annotated[str, typer.Option(help="foreground | range | gain")] = "foreground",
    foreground_level: Annotated[
        float, typer.Option(help="foreground mode: input value of the median foreground voxel")
    ] = 0.25,
    floor_percentile: Annotated[
        float, typer.Option(help="foreground mode: noise-floor percentile (e.g. 98); 0 = off")
    ] = 0.0,
    low: Annotated[float, typer.Option(help="range mode: raw counts mapped to 0")] = 0.0,
    high: Annotated[float, typer.Option(help="range mode: raw counts mapped to 1")] = 1000.0,
    gain: Annotated[float, typer.Option(help="gain mode: raw * gain / 65535")] = 1.0,
    precision: Annotated[str, typer.Option(help="auto (fp32) | fp32 | bf16 | fp16")] = "auto",
    io_threads: Annotated[int, typer.Option(help="Parallel S3 tile reads")] = 8,
    crop_um: Annotated[
        float | None, typer.Option(help="Only a cube of this side at the region's core")
    ] = None,
    weights: Annotated[
        str, typer.Option(help="Registered weights name or fine-tuned .safetensors path")
    ] = "trailmap-axon-210409",
    overwrite: Annotated[bool, typer.Option(help="Discard an existing output")] = False,
) -> None:
    """Segment axons within CCF regions and write a probability OME-Zarr."""
    from exaspim_axon_segmentation.inference import predict_region

    config = InferenceConfig(
        weights=weights,
        level=level,
        tile_size=tile_size,
        batch_size=batch_size,
        device=device,
        precision=precision,
        io_threads=io_threads,
        normalization=NormalizationConfig(
            mode=norm,
            foreground_level=foreground_level,
            floor_percentile=floor_percentile or None,
            low=low,
            high=high,
            gain=gain,
        ),
    )
    predict_region(asset, region, out, config, crop_um=crop_um, overwrite=overwrite)


@app.command("qc-predictions")
def qc_predictions(
    outputs: Annotated[list[str], typer.Argument(help="Prediction OME-Zarrs over the same region")],
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/predictions.png"),
    label: Annotated[
        list[str] | None, typer.Option("--label", "-l", help="Column title per output; repeat")
    ] = None,
    slab_um: Annotated[float, typer.Option(help="MIP slab thickness")] = 48.0,
) -> None:
    """Compare prediction runs side by side against the raw image."""
    from exaspim_axon_segmentation.qc import render_prediction_comparison

    render_prediction_comparison(outputs, out, labels=label, slab_um=slab_um)
    log.info("Wrote %s", out)


@app.command()
def cleanup(
    prediction: Annotated[str, typer.Argument(help="Probability OME-Zarr written by `predict`")],
    out: Annotated[str, typer.Option(help="Output category OME-Zarr")],
    low: Annotated[float, typer.Option(help="Hysteresis low threshold")] = 0.5,
    high: Annotated[float, typer.Option(help="Hysteresis high threshold")] = 0.9,
    min_volume_um3: Annotated[float, typer.Option(help="Drop smaller components")] = 2000.0,
    somata: Annotated[
        bool, typer.Option(help="Tag components touching detected somata as soma-connected")
    ] = True,
    overwrite: Annotated[bool, typer.Option(help="Discard an existing output")] = False,
) -> None:
    """Classify probability voxels into kept axon vs. removed (with the reason)."""
    from exaspim_axon_segmentation.postprocess import CleanupConfig
    from exaspim_axon_segmentation.postprocess.pipeline import cleanup_prediction

    config = CleanupConfig(
        low=low, high=high, min_volume_um3=min_volume_um3, remove_soma_connected=somata
    )
    cleanup_prediction(prediction, out, config, overwrite=overwrite)


@app.command("qc-cleanup")
def qc_cleanup(
    prediction: Annotated[str, typer.Argument(help="Probability OME-Zarr")],
    categories: Annotated[str, typer.Argument(help="Category OME-Zarr from `cleanup`")],
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/cleanup.png"),
    slab_um: Annotated[float, typer.Option(help="MIP slab thickness")] = 48.0,
) -> None:
    """Show what the cleanup kept and removed, by reason, against the raw image."""
    from exaspim_axon_segmentation.qc.cleanup import render_cleanup

    render_cleanup(prediction, categories, out, slab_um=slab_um)
    log.info("Wrote %s", out)


@app.command("make-labels")
def make_labels(
    asset: AssetArg,
    region: RegionOpt,
    swc: Annotated[
        list[Path], typer.Option("--swc", help="Complete traces for the region; repeat")
    ],
    out: Annotated[Path, typer.Option(help="Output directory (raw.zarr + labels.zarr)")],
    level: Annotated[int, typer.Option(help="fused.zarr pyramid level (match inference)")] = 2,
    envelope_um: Annotated[
        float | None, typer.Option(help="Label only within this distance of the traces")
    ] = None,
) -> None:
    """Build a training volume from complete SWC traces of a region's traced lobe."""
    from exaspim_axon_segmentation.train.labels import LabelConfig
    from exaspim_axon_segmentation.train.volume import make_training_volume

    config = LabelConfig(envelope_um=envelope_um)
    info = make_training_volume(asset, region, swc, out, level=level, config=config)
    log.info("Labels: %s", info["voxels"])


@app.command()
def finetune(
    volume: Annotated[
        list[Path], typer.Option("--volume", help="Training volume directory; repeat")
    ],
    out: Annotated[Path, typer.Option(help="Output directory for checkpoints and metrics")],
    holdout_brain: Annotated[
        list[str] | None, typer.Option(help="Brain ID whose blocks are validation only; repeat")
    ] = None,
    region_share: Annotated[
        float, typer.Option(help="Fraction of training patches from region volumes")
    ] = 0.5,
    weights: Annotated[str, typer.Option(help="Starting weights (name or .safetensors)")] = (
        "trailmap-axon-210409"
    ),
    steps: Annotated[int, typer.Option()] = 4000,
    batch_size: Annotated[int, typer.Option()] = 4,
    input_size: Annotated[int, typer.Option(help="Patch side (multiple of 8)")] = 96,
    lr: Annotated[float, typer.Option()] = 1e-4,
    max_minutes: Annotated[float, typer.Option(help="Stop training after this long")] = 40.0,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Fine-tune on region volumes (spatially split) and traced blocks (held out by brain)."""
    from exaspim_axon_segmentation.inference.predict import resolve_device
    from exaspim_axon_segmentation.model import load_model
    from exaspim_axon_segmentation.train.trainer import TrainConfig, build_volumes
    from exaspim_axon_segmentation.train.trainer import finetune as run_finetune

    config = TrainConfig(
        input_size=input_size, batch_size=batch_size, steps=steps, lr=lr, max_minutes=max_minutes
    )
    volumes = build_volumes(
        volume, config, holdout_brains=holdout_brain or [], region_share=region_share
    )
    run_finetune(load_model(weights), volumes, out, config, device=resolve_device(device))


@app.command("make-block-volumes")
def make_block_volumes_cmd(
    root: Annotated[str, typer.Argument(help="gs://.../training-data_<date> root")],
    out: Annotated[Path, typer.Option(help="Output directory (one volume per block)")],
    level: Annotated[int, typer.Option(help="Block pyramid level (match inference)")] = 2,
    key_env: Annotated[
        str, typer.Option(help="Environment variable holding the GCS service-account JSON")
    ] = "CUSTOM_KEY",
    key_file: Annotated[Path | None, typer.Option(help="Key file instead of --key-env")] = None,
) -> None:
    """Build training volumes from completely traced GCS blocks (SWC + fill masks)."""
    from exaspim_axon_segmentation.io.gcs import gcs_token
    from exaspim_axon_segmentation.train.blocks import make_block_volumes

    records = make_block_volumes(
        root, out, token=gcs_token(env=key_env, path=key_file), level=level
    )
    log.info("Built %d block volumes from %s", len(records), sorted({r["brain"] for r in records}))


@app.command("qc-traces")
def qc_traces(
    prediction: Annotated[list[str], typer.Argument(help="Probability OME-Zarrs (same region)")],
    swc: Annotated[list[Path], typer.Option("--swc", help="Complete traces; repeat")],
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/traces.png"),
    label: Annotated[list[str] | None, typer.Option("--label", "-l", help="Per prediction")] = None,
    val_fraction: Annotated[
        float, typer.Option(help="Held-out share of trace points (by x)")
    ] = 0.2,
) -> None:
    """Score predictions against complete traces (recall along traces, precision near traces)."""
    from exaspim_axon_segmentation.qc.traces import render_trace_comparison

    labels = label or [Path(p).stem for p in prediction]
    scores = render_trace_comparison(prediction, labels, swc, out, val_fraction=val_fraction)
    for name, sc in zip(labels, scores, strict=True):
        for s in sc:
            log.info("%s · %s", name, s.text())
    log.info("Wrote %s", out)
