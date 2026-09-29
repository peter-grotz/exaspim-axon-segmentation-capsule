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
    low: Annotated[float, typer.Option(help="range mode: raw counts mapped to 0")] = 0.0,
    high: Annotated[float, typer.Option(help="range mode: raw counts mapped to 1")] = 1000.0,
    gain: Annotated[float, typer.Option(help="gain mode: raw * gain / 65535")] = 1.0,
    precision: Annotated[str, typer.Option(help="auto (fp32) | fp32 | bf16 | fp16")] = "auto",
    io_threads: Annotated[int, typer.Option(help="Parallel S3 tile reads")] = 8,
    crop_um: Annotated[
        float | None, typer.Option(help="Only a cube of this side at the region's core")
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Discard an existing output")] = False,
) -> None:
    """Segment axons within CCF regions and write a probability OME-Zarr."""
    from exaspim_axon_segmentation.inference import predict_region

    config = InferenceConfig(
        level=level,
        tile_size=tile_size,
        batch_size=batch_size,
        device=device,
        precision=precision,
        io_threads=io_threads,
        normalization=NormalizationConfig(
            mode=norm, foreground_level=foreground_level, low=low, high=high, gain=gain
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
