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
    annotation_asset: Annotated[
        str | None,
        typer.Option(help="Asset with the CCF alignment, if not ASSET"),
    ] = None,
) -> None:
    """Draw the sample-space CCF region outline on the fused image for visual confirmation."""
    from exaspim_axon_segmentation.qc import render_region_overlay

    ontology = CcfOntology.load()
    data = ExaspimAsset(asset, annotation_root=annotation_asset)
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
    region_growth: Annotated[
        float, typer.Option(help="Grow the region by this fraction of its volume")
    ] = 0.2,
    overwrite: Annotated[bool, typer.Option(help="Discard an existing output")] = False,
) -> None:
    """Segment axons within CCF regions and write a probability OME-Zarr."""
    from exaspim_axon_segmentation.inference import predict_region

    config = InferenceConfig(
        weights=weights,
        region_growth=region_growth,
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


@app.command("make-coarse-volumes")
def make_coarse_volumes(
    asset: AssetArg,
    out: Annotated[Path, typer.Option(help="Output directory (one volume per lobe/cube)")],
    swc: Annotated[
        list[Path] | None, typer.Option("--swc", help="Lobe mode: complete traces; repeat")
    ] = None,
    region: Annotated[
        list[str] | None, typer.Option("--region", "-r", help="Lobe mode: CCF region")
    ] = None,
    points: Annotated[
        Path | None,
        typer.Option(help="Cube mode: npz with axon_zyx_um (and other_zyx_um) trace points"),
    ] = None,
    cube_um: Annotated[str, typer.Option(help="Cube mode: cube size z,y,x (um)")] = (
        "2000,2400,2400"
    ),
    n_cubes: Annotated[int, typer.Option(help="Cube mode: most-traced cubes to keep")] = 40,
    val_fraction: Annotated[float, typer.Option(help="Cube mode: held-out cubes (by x)")] = 0.2,
    factor: Annotated[str, typer.Option(help="Pooling of level 2 per axis z,y,x")] = "5,5,5",
    envelope_um: Annotated[float, typer.Option(help="Label only this close to traces")] = 60.0,
) -> None:
    """Training volumes at a 1x-like voxel size: level 2 mean-pooled, labelled from traces."""
    import numpy as np

    from exaspim_axon_segmentation.io import Box, CcfOntology, ExaspimAsset, RegionRoi
    from exaspim_axon_segmentation.train.coarse import (
        CoarseLabelConfig,
        make_coarse_volume,
        trace_cubes,
    )
    from exaspim_axon_segmentation.train.labels import traced_lobe_box
    from exaspim_axon_segmentation.train.swc import read_swc, resample

    fac = tuple(int(v) for v in factor.split(","))
    cfg = CoarseLabelConfig(envelope_um=envelope_um)
    a = ExaspimAsset(asset)
    fused, tf = a.fused, a.fused.transforms[2]
    brain = a.root.rstrip("/").split("/")[-1].split("_")[1]
    if swc:  # lobe mode, as make-labels
        roi = RegionRoi.from_annotation(a.annotation, CcfOntology.load().resolve(region or []))
        pts = np.concatenate([resample(read_swc(p), 0.5 * min(tf.scale)).zyx_um for p in swc])
        box, inside = traced_lobe_box(fused, 2, roi, pts, pad_um=cfg.edge_radius_um + 40)
        prov = {"asset": a.root, "regions": list(region or []), "traces": [p.name for p in swc]}
        make_coarse_volume(fused, box, inside, pts, out / f"{brain}_lobe", prov, factor=fac,
                           config=cfg)  # fmt: skip
        return
    if points is None:
        raise typer.BadParameter("give --swc (lobe mode) or --points (cube mode)")
    z = np.load(points)
    axon = z["axon_zyx_um"]
    other = z.get("other_zyx_um")
    size = np.array([float(v) for v in cube_um.split(",")])
    # only some neurons are traced: bright untraced fibres near them are ignored, not negatives
    cfg = CoarseLabelConfig(envelope_um=envelope_um, hard_negatives=False)
    centres = trace_cubes(axon, tuple(size), n_cubes, min_points=200)
    cut = np.quantile([c[2] for c in centres], 1 - val_fraction)
    shape = np.asarray(fused.shape(2))
    for k, c in enumerate(centres):
        lo = np.clip(np.floor(tf.to_index(c - size / 2)).astype(int), 0, shape)
        hi = np.clip(np.ceil(tf.to_index(c + size / 2)).astype(int), 0, shape)
        near = np.all(np.abs(axon - c) <= size / 2 + cfg.envelope_um, axis=1)
        prov = {"asset": a.root, "brain": brain, "cube": f"cube_{k:03d}",
                "split": "val" if c[2] >= cut else "train", "centre_um": c.tolist()}  # fmt: skip
        try:
            make_coarse_volume(fused, Box.from_arrays(lo, hi), None, axon[near],
                               out / f"cube_{k:03d}", prov, factor=fac, config=cfg,
                               other_traced_zyx_um=None if other is None else other[
                                   np.all(np.abs(other - c) <= size / 2 + 30, axis=1)])  # fmt: skip
        except ValueError as error:
            log.warning("cube %d skipped: %s", k, error)


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
    degradation: Annotated[
        Path | None,
        typer.Option(help="JSON from scripts/fit_degradation2.py: add the 1x look to patches"),
    ] = None,
    gain_max: Annotated[float, typer.Option(help="Degradation: largest contrast gain")] = 2.5,
) -> None:
    """Fine-tune on region volumes (spatially split) and traced blocks (held out by brain)."""
    from exaspim_axon_segmentation.inference.predict import resolve_device
    from exaspim_axon_segmentation.model import load_model
    from exaspim_axon_segmentation.train.trainer import TrainConfig, build_volumes
    from exaspim_axon_segmentation.train.trainer import finetune as run_finetune

    extra = {}
    if degradation is not None:
        import json as _json

        d = _json.loads(degradation.read_text())
        extra = {"degrade_blur": (d["blur_sigma_vox"]["z"], d["blur_sigma_vox"]["xy"]),
                 "degrade_gain": (1.0, gain_max),
                 "degrade_noise_sd": d["noise_floor_sd"]["extra"]}  # fmt: skip
    config = TrainConfig(
        input_size=input_size,
        batch_size=batch_size,
        steps=steps,
        lr=lr,
        max_minutes=max_minutes,
        **extra,
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


@app.command()
def skeletonize(
    prediction: Annotated[str, typer.Argument(help="Probability OME-Zarr from `predict`")],
    out: Annotated[Path, typer.Option(help="Output directory")],
    threshold: Annotated[float, typer.Option(help="Neurite mask threshold")] = 0.2,
    core: Annotated[float, typer.Option(help="Keep components with a voxel above this")] = 0.5,
    min_volume_um3: Annotated[float, typer.Option(help="Drop smaller components")] = 100.0,
    spur_um: Annotated[
        float, typer.Option(help="Prune terminal branches shorter than this")
    ] = 10.0,
    bridge_um: Annotated[
        float, typer.Option(help="Reconnect aligned piece tips across gaps up to this; 0 = off")
    ] = 30.0,
    min_piece_um: Annotated[
        float, typer.Option(help="Minimum cable length (after bridging)")
    ] = 100.0,
    anchor_um: Annotated[
        float, typer.Option(help="Pieces this long are anchor cable; 0 = no isolation filter")
    ] = 500.0,
    attach_um: Annotated[
        float, typer.Option(help="Keep shorter pieces only within this of anchor cable")
    ] = 50.0,
    connect_um: Annotated[
        float, typer.Option(help="Connect kept cable across gaps up to this; 0 = off")
    ] = 50.0,
    connect_min_p: Annotated[
        float, typer.Option(help="Reject connections passing a voxel below this probability")
    ] = 0.05,
    speckle_max_um: Annotated[
        float, typer.Option(help="Speckle removal: drop single-plane pieces shorter than this")
    ] = 60.0,
    swc: Annotated[
        list[Path] | None, typer.Option("--swc", help="Complete traces to score against; repeat")
    ] = None,
    envelope_um: Annotated[
        float, typer.Option(help="Score precision within this of traces")
    ] = 50.0,
    region_growth: Annotated[
        float | None,
        typer.Option(help="Region growth (fraction of volume); default: the prediction's"),
    ] = None,
) -> None:
    """Skeletonize a probability map within its region: debris removal, connection, SWCs."""
    import json

    from exaspim_axon_segmentation.postprocess.pipeline import skeletonize_prediction
    from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig

    config = SkeletonConfig(
        threshold=threshold,
        core=core,
        min_volume_um3=min_volume_um3,
        spur_um=spur_um,
        bridge_um=bridge_um,
        min_piece_um=min_piece_um,
        anchor_um=anchor_um,
        attach_um=attach_um,
        connect_um=connect_um,
        connect_min_p=connect_min_p,
        speckle_max_um=speckle_max_um,
    )
    summary = skeletonize_prediction(prediction, out, config, region_growth=region_growth)
    if swc:
        import numpy as np

        from exaspim_axon_segmentation.qc.traces import score_skeleton
        from exaspim_axon_segmentation.train import read_swc

        skel = read_swc(out / "skeleton.swc")
        index = {int(i): k for k, i in enumerate(skel.ids)}
        edges = np.array(
            [(k, index[int(p)]) for k, p in enumerate(skel.parents) if int(p) in index]
        )
        traces = np.concatenate([read_swc(p).zyx_um for p in swc])
        roi_box = summary["box"]
        # only trace points inside the processed box can be found
        lo = np.min(skel.zyx_um, axis=0) - envelope_um
        hi = np.max(skel.zyx_um, axis=0) + envelope_um
        traces = traces[np.all((traces >= lo) & (traces <= hi), axis=1)]
        scores = score_skeleton(skel.zyx_um, edges, traces, envelope_um=envelope_um)
        summary["trace_scores"] = scores
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Skeleton vs traces (box %s): %s", roi_box, scores)


@app.command("predict-box")
def predict_box_cmd(
    asset: AssetArg,
    lo: Annotated[str, typer.Option(help="Box corner x,y,z (µm, as in SWCs)")],
    hi: Annotated[str, typer.Option(help="Opposite corner x,y,z (µm)")],
    out: Annotated[Path, typer.Option(help="Output probability OME-Zarr")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    level: Annotated[int, typer.Option(help="fused.zarr pyramid level")] = 2,
    batch_size: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    scaling: Annotated[
        str | None, typer.Option(help="Fixed intensity scaling offset,scale (one volume in slabs)")
    ] = None,
) -> None:
    """Segment an arbitrary box of an asset (resumable); then use skeletonize-blocks."""
    import numpy as np

    from exaspim_axon_segmentation.inference.normalize import IntensityScaling
    from exaspim_axon_segmentation.inference.pipeline import predict_box

    a = np.array([float(v) for v in lo.split(",")])[::-1]
    b = np.array([float(v) for v in hi.split(",")])[::-1]
    config = InferenceConfig(weights=weights, level=level, batch_size=batch_size, device=device,
                             region_growth=0.0)  # fmt: skip
    fixed = None
    if scaling:
        offset, scale = (float(v) for v in scaling.split(","))
        fixed = IntensityScaling(offset, scale)
    predict_box(asset, np.minimum(a, b), np.maximum(a, b), str(out), config, scaling=fixed)


@app.command("predict-image")
def predict_image_cmd(
    url: Annotated[str, typer.Argument(help="Any OME-Zarr image, e.g. a raw 1x tile")],
    lo: Annotated[str, typer.Argument(help="Box corner x,y,z (um)")],
    hi: Annotated[str, typer.Argument(help="Opposite corner x,y,z (um)")],
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    level: Annotated[int, typer.Option(help="Level to segment")] = 0,
    mask_level: Annotated[int, typer.Option(help="Coarse level for the tissue/signal mask")] = 3,
    signal_sigmas: Annotated[float, typer.Option(help="Signal threshold (noise SDs)")] = 6.0,
    tile_size: Annotated[
        int, typer.Option(help="Network input side (256 OOMs in fp32 on 24 GB)")
    ] = 192,
    batch_size: Annotated[int, typer.Option()] = 1,
    precision: Annotated[str, typer.Option(help="fp32 | bf16 | fp16 | auto")] = "bf16",
    check_tiles: Annotated[int, typer.Option(help="Compare precision vs fp32 on N tiles")] = 0,
    sample_skipped: Annotated[int, typer.Option(help="Also predict N skipped tiles")] = 0,
    skip_no_signal: Annotated[
        bool, typer.Option(help="Skip tissue without bright structure (else all tissue)")
    ] = True,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Segment a box of any OME-Zarr image where it holds bright structure (no atlas needed)."""
    import json as _json

    import numpy as np

    from exaspim_axon_segmentation.inference.image import predict_image

    a = np.array([float(v) for v in lo.split(",")])[::-1]
    b = np.array([float(v) for v in hi.split(",")])[::-1]
    config = InferenceConfig(weights=weights, level=level, tile_size=tile_size,
                             batch_size=batch_size, precision=precision, device=device,
                             region_growth=0.0)  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    report = predict_image(url, np.minimum(a, b), np.maximum(a, b), str(out / "probability.zarr"),
                           config, mask_level=mask_level, signal_sigmas=signal_sigmas,
                           check_tiles=check_tiles, sample_skipped=sample_skipped,
                           skip_no_signal=skip_no_signal)  # fmt: skip
    (out / "report.json").write_text(_json.dumps(report, indent=1, default=float))
    log.info("Report: %s", report)


@app.command("snr-tiles")
def snr_tiles_cmd(
    root: Annotated[str, typer.Argument(help="Raw tile container, e.g. s3://.../SPIM/")],
    out: Annotated[Path, typer.Option(help="Output directory (one folder per tile)")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    channel: Annotated[str, typer.Option(help="Channel suffix of the tiles")] = "561",
    n_blocks: Annotated[int, typer.Option(help="Tissue blocks per tile")] = 40,
    min_tissue: Annotated[float, typer.Option(help="Least tissue fraction of a block")] = 0.9,
    level: Annotated[int, typer.Option(help="Level to segment (5x: 2 = training grid)")] = 2,
    block_um: Annotated[str, typer.Option(help="Block size z,y,x (um)")] = "1280,1440,1440",
    precision: Annotated[str, typer.Option(help="auto | fp32 | bf16")] = "bf16",
    tile: Annotated[int, typer.Option(help="Network input side")] = 192,
    save_mips: Annotated[bool, typer.Option(help="Write per-block preview MIPs")] = True,
    mask_root: Annotated[
        str | None,
        typer.Option(help="Flat-field brain masks (.../flatfield_correction/mask/SPIM.ome.zarr)"),
    ] = None,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Axon SNR in ``n_blocks`` non-overlapping tissue blocks of every tile of an acquisition."""
    import json as _json
    from concurrent.futures import ThreadPoolExecutor

    import fsspec
    import numpy as np

    from exaspim_axon_segmentation.qc.tile_mask import blocks_from_fraction, mask_fraction
    from exaspim_axon_segmentation.snr import SnrConfig, open_ome, snr_blocks, tissue_block_centres

    fs = (
        fsspec.filesystem("s3", anon=True)
        if root.startswith("s3://")
        else fsspec.filesystem("file")
    )
    base = root.rstrip("/")
    names = sorted(p.rstrip("/").split("/")[-1] for p in fs.ls(base.removeprefix("s3://"))
                   if f"_ch_{channel}" in p.rsplit("/", 1)[-1])  # fmt: skip
    log.info("%d tiles in %s", len(names), root)
    size = tuple(float(v) for v in block_um.split(","))
    cfg = SnrConfig(block_um=size, precision=precision, tile=tile, save_mips=save_mips,
                    qc_normalization=False)  # fmt: skip
    stem = lambda name: name.split(".")[0]  # noqa: E731
    masks = {}
    if mask_root:
        mbase = mask_root.rstrip("/")
        for p in fs.ls(mbase.removeprefix("s3://")):
            masks[stem(p.rstrip("/").split("/")[-1])] = "s3://" + p.rstrip("/")
        log.info("%d tile masks in %s", len(masks), mask_root)
    bin_ = 32

    def prepare(name):
        """Block centres for one tile: from its brain mask, else from bright structure."""
        url = f"{base}/{name}"
        if stem(name) in masks:
            frac, st = mask_fraction(masks[stem(name)], bin=bin_)
            _, _, sc0, tr0 = open_ome(url, 0)
            bins = np.maximum(np.round(np.asarray(size) / (sc0 * bin_)).astype(int), 1)
            starts, fr = blocks_from_fraction(frac, bins, n_blocks, min_tissue=min_tissue,
                                              seed=names.index(name))  # fmt: skip
            centres = tr0 + ((starts + bins / 2) * bin_) * sc0
            return centres, fr, {"selection": "brain mask", "mask": masks[stem(name)]} | st
        centres, fr = tissue_block_centres(url, n_blocks, size, seed=names.index(name),
                                           min_tissue=min_tissue)  # fmt: skip
        return centres, fr, {"selection": "intensity (no brain mask for this tile)"}

    todo = []
    for name in names:
        done = out / stem(name) / "snr_blocks.json"
        if done.exists() and _json.loads(done.read_text()).get("complete"):
            log.info("%s: done, skipping", name)
            continue
        todo.append(name)
    with ThreadPoolExecutor(1) as pool:  # the next tile's mask is read while this one predicts
        futures = {todo[0]: pool.submit(prepare, todo[0])} if todo else {}
        for i, name in enumerate(todo):
            if i + 1 < len(todo):
                futures[todo[i + 1]] = pool.submit(prepare, todo[i + 1])
            url = f"{base}/{name}"
            tdir = out / stem(name)
            done = tdir / "snr_blocks.json"
            try:
                centres, frac, sel = futures.pop(name).result()
            except Exception as error:
                log.warning("%s: block selection failed (%s); skipped", name, error)
                continue
            tdir.mkdir(parents=True, exist_ok=True)
            meta = {"tile": name, "url": url, "n_requested": n_blocks,
                    "n_tissue_blocks": len(centres),
                    "tissue_fraction": np.round(frac, 3).tolist()} | sel  # fmt: skip
            if len(centres):
                snr_blocks(url, centres, tdir, weights, cfg, level=level, device=device)
            summary = _json.loads(done.read_text()) if done.exists() else {"blocks": []}
            done.write_text(_json.dumps(summary | meta | {"complete": True}, indent=1))
            log.info("%s: %d tissue blocks scored (%s)", name, len(centres), sel["selection"])


@app.command("snr-blocks")
def snr_blocks_cmd(
    url: Annotated[
        str, typer.Argument(help="OME-Zarr tile (s3:// or https://), e.g. a raw SPIM tile")
    ],
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    n_blocks: Annotated[
        int, typer.Option(help="Random tissue blocks (ignored with --centres)")
    ] = 20,
    centres: Annotated[
        Path | None, typer.Option(help="snr_blocks.json of an earlier run: score the same places")
    ] = None,
    level: Annotated[int, typer.Option(help="Pyramid level to segment")] = 0,
    block_um: Annotated[str, typer.Option(help="Block size z,y,x (um)")] = "1280,1440,1440",
    resample_um: Annotated[
        str, typer.Option(help="Also run resampled to this grid z,y,x (um); 'none' = native only")
    ] = "none",
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Axon SNR in random tissue blocks of an OME-Zarr tile, with TRAILMAP as the detector."""
    import json as _json

    import numpy as np

    from exaspim_axon_segmentation.snr import SnrConfig, random_centres, snr_blocks

    if centres is not None:
        pts = np.array([b["centre_um"] for b in _json.loads(centres.read_text())["blocks"]])
    else:
        pts = random_centres(url, n_blocks, seed)
    rs = None if resample_um == "none" else tuple(float(v) for v in resample_um.split(","))
    cfg = SnrConfig(block_um=tuple(float(v) for v in block_um.split(",")), resample_um=rs)
    snr_blocks(url, pts, out, weights, cfg, level=level, device=device)


@app.command("snr-pair")
def snr_pair_cmd(
    url_a: Annotated[str, typer.Argument(help="Acquisition a (e.g. the 1x tile), OME-Zarr")],
    url_b: Annotated[str, typer.Argument(help="Acquisition b (e.g. the 5x fused), OME-Zarr")],
    affine: Annotated[Path, typer.Option(help="JSON with A, t: p_b = A p_a + t (um, z y x)")],
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    n_blocks: Annotated[int, typer.Option(help="Random tissue blocks in a")] = 20,
    seed: Annotated[int, typer.Option()] = 0,
    level_a: Annotated[int, typer.Option(help="Level of a to segment")] = 0,
    level_b: Annotated[int, typer.Option(help="Level of b to segment")] = 2,
    raw_tiles: Annotated[
        str | None, typer.Option(help="Raw tiles of b (SPIM.ome.zarr root): score raw, not fused")
    ] = None,
    stitching: Annotated[
        Path | None, typer.Option(help="BigStitcher XML placing the raw tiles in the fusion")
    ] = None,
    raw_voxel_um: Annotated[str, typer.Option(help="Raw tile level-0 voxel z,y,x (um)")] = (
        "1.0,0.748,0.748"
    ),
    weights_a: Annotated[
        str | None, typer.Option(help="Other weights for a (e.g. a 1x model); default --weights")
    ] = None,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Axon SNR at the same places in two acquisitions of one brain (e.g. 1x vs 5x)."""
    import json as _json

    from exaspim_axon_segmentation.snr import random_centres
    from exaspim_axon_segmentation.snr_pair import PairConfig, RawTiles, snr_pair

    pts = random_centres(url_a, n_blocks, seed)
    cfg = PairConfig(level_a=level_a, level_b=level_b)
    tiles = None
    if raw_tiles is not None:
        if stitching is None:
            raise typer.BadParameter("--raw-tiles needs --stitching")
        vox = tuple(float(v) for v in raw_voxel_um.split(","))
        tiles = RawTiles(raw_tiles, str(stitching), vox)
    snr_pair(url_a, url_b, _json.loads(affine.read_text()), pts, out, weights, cfg,
             raw_tiles=tiles, weights_a=weights_a, device=device)  # fmt: skip


@app.command("skeletonize-blocks")
def skeletonize_blocks(
    prediction: Annotated[str, typer.Argument(help="Probability OME-Zarr from `predict`")],
    out: Annotated[Path, typer.Option(help="Output directory")],
    relaxed: Annotated[
        bool, typer.Option(help="Block settings: no anchor rule, pieces >= 30 um (as in expand)")
    ] = False,
    block_vox: Annotated[int, typer.Option(help="Block core side (voxels)")] = 384,
    halo_vox: Annotated[int, typer.Option(help="Context read around each block (voxels)")] = 32,
    workers: Annotated[int, typer.Option(help="Processes for the local pass")] = 4,
    min_swc_um: Annotated[float, typer.Option(help="Write per-piece SWCs from this length")] = 0.0,
    mask_region: Annotated[
        bool, typer.Option(help="Zero the prediction outside its region")
    ] = True,
) -> None:
    """Skeletonize a probability map block by block, for volumes too large for memory."""
    from exaspim_axon_segmentation.postprocess.blockwise import BlockConfig
    from exaspim_axon_segmentation.postprocess.pipeline import skeletonize_prediction_blockwise
    from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig

    config = SkeletonConfig(anchor_um=0, min_piece_um=30.0) if relaxed else SkeletonConfig()
    block = BlockConfig(block_vox=block_vox, halo_vox=halo_vox, workers=workers)
    import resource
    import time

    t0 = time.time()
    skeletonize_prediction_blockwise(prediction, out, config, block, mask_region=mask_region,
                                     min_swc_um=min_swc_um)  # fmt: skip
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)  # fmt: skip
    log.info("skeletonize-blocks: %.0f s, peak memory of one process %.1f GB (Linux kB units)",
             time.time() - t0, peak / 1e6)  # fmt: skip


@app.command("skeletonize-slabs")
def skeletonize_slabs_cmd(
    predictions: Annotated[
        list[str], typer.Argument(help="Probability OME-Zarrs of adjacent slabs")
    ],
    out: Annotated[Path, typer.Option(help="Output directory")],
    relaxed: Annotated[bool, typer.Option(help="Relaxed cleanup (no anchor rule)")] = False,
    block_vox: Annotated[int, typer.Option(help="Block core side (voxels)")] = 384,
    halo_vox: Annotated[int, typer.Option(help="Context around each block (voxels)")] = 32,
    workers: Annotated[int, typer.Option(help="Processes for the local pass")] = 4,
    min_swc_um: Annotated[
        float, typer.Option(help="Write per-piece SWCs from this length")
    ] = 100.0,
) -> None:
    """Skeletonize adjacent slab predictions (e.g. parallel predict-box runs) as one volume."""
    import resource
    import time

    from exaspim_axon_segmentation.postprocess.blockwise import BlockConfig
    from exaspim_axon_segmentation.postprocess.pipeline import skeletonize_slabs
    from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig

    t0 = time.time()
    config = SkeletonConfig(anchor_um=0, min_piece_um=30.0) if relaxed else SkeletonConfig()
    block = BlockConfig(block_vox=block_vox, halo_vox=halo_vox, workers=workers)
    skeletonize_slabs(list(predictions), out, config, block, min_swc_um=min_swc_um)
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)  # fmt: skip
    log.info("skeletonize-slabs: %.0f s, peak memory of one process %.1f GB", time.time() - t0,
             peak / 1e6)  # fmt: skip


@app.command("qc-skeleton")
def qc_skeleton(
    prediction: Annotated[str, typer.Argument(help="Probability OME-Zarr from predict")],
    skeleton_dir: Annotated[Path, typer.Argument(help="Output directory of skeletonize")],
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/skeleton.png"),
) -> None:
    """Draw the (grown) region with every skeleton piece, plus close-ups of the densest cable."""
    from exaspim_axon_segmentation.qc.skeleton import render_skeleton_overview

    render_skeleton_overview(prediction, skeleton_dir, out)
    log.info("Wrote %s", out)


@app.command()
def run(
    asset: AssetArg,
    region: RegionOpt,
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[
        str,
        typer.Option(help="Fine-tuned .safetensors (e.g. trailmap_finetune_v2/best.safetensors)"),
    ],
    level: Annotated[int, typer.Option(help="fused.zarr pyramid level")] = 2,
    tile_size: Annotated[int, typer.Option(help="Network input side (multiple of 8)")] = 160,
    batch_size: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    region_growth: Annotated[
        float, typer.Option(help="Grow the region by this fraction of its volume")
    ] = 0.2,
    crop_um: Annotated[
        float | None, typer.Option(help="Only a cube of this side at the region's core")
    ] = None,
    annotation_asset: Annotated[
        str | None,
        typer.Option(help="Asset with the CCF alignment and soma detections, if not ASSET"),
    ] = None,
) -> None:
    """Full pipeline: segment → skeletonize, remove debris, connect → per-piece SWCs → QC figure.

    Writes into OUT: ``<tag>_axon_probability.zarr``, ``<tag>_skeleton/`` (``skeleton.swc``,
    ``swc/<tag>_NNNN.swc`` one Horta-importable file per piece, ``skeleton.zarr``,
    ``summary.json``) and ``qc/<tag>_skeleton.png``, with ``<tag> = <subject>_<regions>_L<level>``.
    An interrupted run resumes the prediction where it stopped.
    """
    from exaspim_axon_segmentation.inference import predict_region
    from exaspim_axon_segmentation.postprocess.pipeline import skeletonize_prediction
    from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig
    from exaspim_axon_segmentation.qc.skeleton import render_skeleton_overview

    tag = f"{ExaspimAsset(asset).subject_id}_{'+'.join(region)}_L{level}"
    if crop_um is not None:
        tag += f"_crop{crop_um:.0f}"
    probability = str(out / f"{tag}_axon_probability.zarr")
    skeleton_dir = out / f"{tag}_skeleton"
    config = InferenceConfig(
        weights=weights, level=level, tile_size=tile_size, batch_size=batch_size, device=device,
        region_growth=region_growth,
    )  # fmt: skip
    predict_region(asset, region, probability, config, crop_um=crop_um,
                   annotation_asset=annotation_asset)  # fmt: skip
    summary = skeletonize_prediction(probability, skeleton_dir, SkeletonConfig(), swc_prefix=tag)
    render_skeleton_overview(probability, skeleton_dir, out / "qc" / f"{tag}_skeleton.png")
    from exaspim_axon_segmentation.qc.skeleton import render_skeleton_map

    render_skeleton_map(
        skeleton_dir, ExaspimAsset(asset, annotation_root=annotation_asset),
        CcfOntology.load().resolve(region), out / "qc" / f"{tag}_skeleton_map.png",
    )  # fmt: skip
    log.info(
        "Done: %.0f mm of skeleton in %d pieces → %s",
        summary["length_in_region_um"] / 1e3, len(summary["pieces"]), skeleton_dir / "swc",
    )  # fmt: skip


@app.command("predict-somata")
def predict_somata(
    asset: AssetArg,
    somata: Annotated[
        Path, typer.Option(help="CSV with columns name, x_um, y_um, z_um (physical, as in SWCs)")
    ],
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    side_um: Annotated[float, typer.Option(help="Cube side around each soma")] = 1200.0,
    level: Annotated[int, typer.Option(help="fused.zarr pyramid level")] = 2,
    tile_size: Annotated[int, typer.Option(help="Network input side (multiple of 8)")] = 160,
    batch_size: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
) -> None:
    """Segment and skeletonize a cube around each listed soma.

    Skeletons (debris removed, pieces connected, per-piece SWCs) go to
    ``<subject>_somata_skeletons/<name>/``. For building skeleton-level training data from traced
    neurons: one prediction covers all cubes of the asset; an interrupted run resumes.
    """
    import csv

    import numpy as np

    from exaspim_axon_segmentation.inference import predict_points
    from exaspim_axon_segmentation.postprocess.pipeline import skeletonize_around
    from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig

    with somata.open() as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    centers = np.array([[float(r["z_um"]), float(r["y_um"]), float(r["x_um"])] for r in rows])
    subject = ExaspimAsset(asset).subject_id
    probability = str(out / f"{subject}_somata_L{level}_axon_probability.zarr")
    config = InferenceConfig(
        weights=weights, level=level, tile_size=tile_size, batch_size=batch_size, device=device,
        region_growth=0.0,
    )  # fmt: skip
    predict_points(asset, centers, side_um, probability, config)
    index = []
    for name, center in zip(names, centers, strict=True):
        summary = skeletonize_around(
            probability, center, side_um, out / f"{subject}_somata_skeletons" / name,
            SkeletonConfig(), swc_prefix=name,
        )  # fmt: skip
        index.append({"name": name, "length_mm": round(summary["length_um"] / 1e3, 3),
                      "pieces": len(summary["pieces"])})  # fmt: skip
        log.info("%s: %.1f mm in %d pieces", name, index[-1]["length_mm"], index[-1]["pieces"])
    with (out / f"{subject}_somata_skeletons" / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "length_mm", "pieces"])
        writer.writeheader()
        writer.writerows(index)


@app.command("qc-skeleton-map")
def qc_skeleton_map(
    skeleton_dir: Annotated[Path, typer.Argument(help="Output directory of skeletonize")],
    asset: Annotated[str, typer.Option(help="Asset the skeleton came from")],
    region: RegionOpt,
    out: Annotated[Path, typer.Option(help="Output PNG")] = Path("qc/skeleton_map.png"),
    annotation_asset: Annotated[
        str | None, typer.Option(help="Asset with the CCF alignment, if not ASSET")
    ] = None,
) -> None:
    """Draw the skeleton on the region's footprint, coloured by piece and by dendrite/axon."""
    from exaspim_axon_segmentation.qc.skeleton import render_skeleton_map

    data = ExaspimAsset(asset, annotation_root=annotation_asset)
    render_skeleton_map(skeleton_dir, data, CcfOntology.load().resolve(region), out)
    log.info("Wrote %s", out)


@app.command()
def expand(
    asset: AssetArg,
    region: RegionOpt,
    out: Annotated[Path, typer.Option(help="Output directory")],
    weights: Annotated[str, typer.Option(help="Fine-tuned .safetensors or registered name")],
    budget_mm: Annotated[
        float, typer.Option(help="Path followed per axon from the region")
    ] = 100.0,
    cube_um: Annotated[float, typer.Option(help="Side of each cube predicted ahead")] = 800.0,
    max_rounds: Annotated[int, typer.Option()] = 40,
    max_cubes: Annotated[int, typer.Option()] = 800,
    max_minutes: Annotated[float, typer.Option(help="Stop starting new rounds after this")] = 600.0,
    region_growth: Annotated[
        float, typer.Option(help="Grow the region by this volume share")
    ] = 0.2,
    batch_size: Annotated[int, typer.Option()] = 2,
    device: Annotated[str, typer.Option(help="auto | cuda | cpu")] = "auto",
    annotation_asset: Annotated[
        str | None, typer.Option(help="Asset with the CCF alignment and somata, if not ASSET")
    ] = None,
    tracer: Annotated[
        str | None, typer.Option(help="Robo-LS v2 checkpoint (best.pt): trace ends on the image")
    ] = None,
    trace_accept: Annotated[float, typer.Option(help="P(same cell) to link a piece")] = 0.5,
) -> None:
    """Follow axon candidates out of a region, predicting cubes ahead of them round by round."""
    from exaspim_axon_segmentation.expand import ExpandConfig, follow_axons

    inference = InferenceConfig(weights=weights, batch_size=batch_size, device=device,
                                region_growth=region_growth)  # fmt: skip
    config = ExpandConfig(cube_um=cube_um, budget_um=budget_mm * 1000, max_rounds=max_rounds,
                          max_cubes=max_cubes, max_minutes=max_minutes,
                          tracer_checkpoint=tracer, trace_accept=trace_accept)  # fmt: skip
    follow_axons(asset, list(region), out, inference, config, annotation_asset=annotation_asset)
