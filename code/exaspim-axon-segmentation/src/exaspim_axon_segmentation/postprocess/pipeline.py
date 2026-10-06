"""Run the cleanup on a prediction output and write a category OME-Zarr beside it."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import zarr
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR, prediction_roi
from exaspim_axon_segmentation.io import Box, ExaspimAsset
from exaspim_axon_segmentation.io.writer import create_ngff_array
from exaspim_axon_segmentation.postprocess.cleanup import Category, CleanupConfig, cleanup
from exaspim_axon_segmentation.postprocess.dendrites import DendriteConfig, label_dendrites
from exaspim_axon_segmentation.postprocess.skeleton import SkeletonConfig, skeletonize_probability
from exaspim_axon_segmentation.postprocess.somata import load_somata

log = logging.getLogger(__name__)

CLEANUP_ATTR = "exaspim_axon_cleanup"


def cleanup_prediction(
    prediction_uri: str, out_uri: str, config: CleanupConfig, *, overwrite: bool = False
) -> dict[str, int]:
    """Classify a probability output voxel by voxel; returns voxel counts per category.

    The output has the prediction's grid, region and chunking; its attributes carry the cleanup
    config, the category legend, and the prediction's provenance.
    """
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    grid = asset.fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    probability = np.asarray(group["0"][:])

    somata = load_somata(asset) if config.remove_soma_connected else None
    categories = cleanup(probability, grid, region, config, somata)
    counts = np.bincount(categories.ravel(), minlength=len(Category))
    summary = {c.name.lower(): int(counts[c]) for c in Category}

    attrs = {
        CLEANUP_ATTR: {
            "prediction": prediction_uri,
            "config": asdict(config),
            "legend": {int(c): c.name.lower() for c in Category},
            "voxels": summary,
        },
        PROVENANCE_ATTR: provenance,
    }
    out = create_ngff_array(
        out_uri, source=grid, region=region, chunks=group["0"].chunks, dtype="uint8",
        name="axon_categories", attrs=attrs, overwrite=overwrite,
    )  # fmt: skip
    out[:] = categories
    log.info("Wrote %s: %s", out_uri, summary)
    return summary


def skeletonize_prediction(
    prediction_uri: str,
    out_dir: Path,
    config: SkeletonConfig,
    *,
    crop: Box | None = None,
    swc_prefix: str | None = None,
    region_growth: float | None = None,
    dendrites: DendriteConfig | None = None,
) -> dict:
    """Skeletonize a probability output within its (grown) region.

    Writes ``skeleton.zarr`` (skeleton voxels), ``skeleton.swc`` (all pieces as one forest),
    ``swc/<prefix>_NNNN.swc`` (one Horta-importable file per connected piece, longest first) and
    ``summary.json``.

    Parameters
    ----------
    prediction_uri : str
        Probability OME-Zarr from ``predict``.
    out_dir : Path
        Output directory.
    config : SkeletonConfig
        Thresholds and pruning.
    crop : Box, optional
        Sub-box of the prediction (region-local indices) to process instead of all of it.
    swc_prefix : str, optional
        File name prefix of the per-piece SWCs (default ``<subject>_<regions>``).
    region_growth : float, optional
        Region growth to apply instead of the one recorded with the prediction.
    dendrites : DendriteConfig, optional
        Rules separating dendrites of detected somata from axon candidates. When the asset has
        soma detections, nodes get SWC type 3 (dendrite) or 2 (axon candidate) and the axon
        candidates are also written on their own (``axon_candidates.swc``, ``axon_swc/``).
    """
    dendrites = dendrites or DendriteConfig()
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    grid = asset.fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    local = crop or Box.full(region.shape)
    sub = Box.from_arrays(np.add(local.start, region.start), np.add(local.stop, region.start))
    probability = np.asarray(group["0"][local.slices])
    roi = prediction_roi(asset, provenance, growth=region_growth)
    inside = roi.sample_on(grid, sub)
    probability[~inside] = 0  # segment within the (grown) region only
    skeleton = skeletonize_probability(probability, grid, sub, config)

    summary = {
        "prediction": prediction_uri,
        "config": asdict(config),
        "box": {"start": list(sub.start), "stop": list(sub.stop)},
        "nodes": len(skeleton.coordinates_um),
        "length_um": skeleton.length_um,
        "length_in_region_um": skeleton.length_inside_um(inside, grid, sub),
        "gap_filled_length_um": skeleton.filled_length_um,
        "region_volume_mm3": float(inside.sum() * np.prod(grid.scale) * 1e-9),
        "region_growth": roi.growth,
    }
    summary["density_mm_per_mm3"] = (
        summary["length_in_region_um"] / 1e3 / summary["region_volume_mm3"]
        if summary["region_volume_mm3"]
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = swc_prefix or f"{asset.subject_id}_{'+'.join(provenance['regions'])}"
    somata = _somata_um(asset)
    if somata is not None and len(skeleton.coordinates_um):
        # only somata whose dendritic field reaches the skeleton matter (and are reported)
        near = cKDTree(skeleton.coordinates_um).query(somata)[0] <= dendrites.field_um
        somata = somata[near]
    if somata is not None:
        dendrite = label_dendrites(skeleton.coordinates_um, skeleton.edges, somata, dendrites)
        skeleton = replace(skeleton, node_types=np.where(dendrite, SWC_DENDRITE, SWC_AXON))
        axon = skeleton.subset(~dendrite)
        summary["dendrites"] = asdict(dendrites) | {
            "somata": len(somata),
            "dendrite_um": skeleton.length_um - axon.length_um - _cut_um(skeleton, dendrite),
            "axon_candidate_um": axon.length_um,
        }
        axon.write_swc(out_dir / "axon_candidates.swc")
        summary["axon_candidate_pieces"] = axon.write_pieces(out_dir / "axon_swc", f"{prefix}_axon")
    skeleton.write_swc(out_dir / "skeleton.swc")
    summary["pieces"] = skeleton.write_pieces(out_dir / "swc", prefix)
    array = create_ngff_array(
        str(out_dir / "skeleton.zarr"), source=grid, region=sub, chunks=(128, 128, 128),
        dtype="uint8", name="axon_skeleton", attrs={"exaspim_axon_skeleton": summary},
        overwrite=True,
    )  # fmt: skip
    array[:] = skeleton.voxels.astype(np.uint8)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "Skeleton: %.1f mm total, %.1f mm inside region (%.2f mm³ → %.1f mm/mm³)",
        summary["length_um"] / 1e3, summary["length_in_region_um"] / 1e3,
        summary["region_volume_mm3"], summary["density_mm_per_mm3"] or 0,
    )  # fmt: skip
    return summary


class _RegionMask:
    """Picklable ``mask_fn`` for block-wise skeletonization: the (grown) region on a block."""

    def __init__(self, roi, grid, region_start):
        self.roi, self.grid, self.start = roi, grid, np.asarray(region_start)

    def __call__(self, lo, hi):
        return self.roi.sample_on(self.grid, Box.from_arrays(self.start + lo, self.start + hi))


def skeletonize_prediction_blockwise(
    prediction_uri: str,
    out_dir: Path,
    config: SkeletonConfig,
    block=None,
    *,
    swc_prefix: str | None = None,
    region_growth: float | None = None,
    dendrites: DendriteConfig | None = None,
    mask_region: bool = True,
    min_swc_um: float = 0.0,
) -> dict:
    """Skeletonize a probability output block by block (for volumes too large for memory).

    Same outputs as :func:`skeletonize_prediction` except ``skeleton.zarr``; per-piece SWCs are
    written for pieces of at least ``min_swc_um``. See :mod:`.blockwise` for the method.
    """
    from exaspim_axon_segmentation.postprocess.blockwise import BlockConfig, skeletonize_blockwise

    block = block or BlockConfig()
    dendrites = dendrites or DendriteConfig()
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    grid = asset.fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    mask_fn = None
    if mask_region:
        mask_fn = _RegionMask(prediction_roi(asset, provenance, growth=region_growth), grid,
                              region.start)  # fmt: skip
    skeleton = skeletonize_blockwise(prediction_uri, grid, region, config, block, mask_fn=mask_fn)
    summary = {
        "prediction": prediction_uri, "config": asdict(config), "block": asdict(block),
        "box": {"start": list(region.start), "stop": list(region.stop)},
        "nodes": len(skeleton.coordinates_um), "length_um": skeleton.length_um,
        "gap_filled_length_um": skeleton.filled_length_um, "blockwise": True,
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = swc_prefix or f"{asset.subject_id}_{'+'.join(provenance['regions'])}"
    somata = _somata_um(asset)
    if somata is not None and len(skeleton.coordinates_um):
        near = cKDTree(skeleton.coordinates_um).query(somata)[0] <= dendrites.field_um
        somata = somata[near]
    if somata is not None and len(skeleton.coordinates_um):
        dendrite = label_dendrites(skeleton.coordinates_um, skeleton.edges, somata, dendrites)
        skeleton = replace(skeleton, node_types=np.where(dendrite, SWC_DENDRITE, SWC_AXON))
        axon = skeleton.subset(~dendrite)
        summary["dendrites"] = asdict(dendrites) | {
            "somata": len(somata),
            "dendrite_um": skeleton.length_um - axon.length_um - _cut_um(skeleton, dendrite),
            "axon_candidate_um": axon.length_um,
        }
        axon.write_swc(out_dir / "axon_candidates.swc")
    skeleton.write_swc(out_dir / "skeleton.swc")
    if min_swc_um > 0 and len(skeleton.coordinates_um):
        keep = np.zeros(len(skeleton.coordinates_um), dtype=bool)
        for nodes, length in skeleton.pieces():
            if length >= min_swc_um:
                keep[nodes] = True
        skeleton = skeleton.subset(keep)
    summary["pieces"] = skeleton.write_pieces(out_dir / "swc", prefix)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Blockwise skeleton: %.1f mm in %d SWC pieces -> %s", summary["length_um"] / 1e3,
             len(summary["pieces"]), out_dir)  # fmt: skip
    return summary


def skeletonize_slabs(
    prediction_uris: list[str],
    out_dir: Path,
    config: SkeletonConfig,
    block=None,
    *,
    swc_prefix: str | None = None,
    dendrites: DendriteConfig | None = None,
    min_swc_um: float = 100.0,
) -> dict:
    """Skeletonize several adjacent probability outputs (slabs of one volume) as one volume."""
    from exaspim_axon_segmentation.postprocess.blockwise import (
        BlockConfig,
        SlabSource,
        skeletonize_blockwise,
    )

    block = block or BlockConfig()
    dendrites = dendrites or DendriteConfig()
    slabs, asset, level = [], None, None
    for uri in prediction_uris:
        provenance = dict(zarr.open_group(uri, mode="r").attrs[PROVENANCE_ATTR])
        box = provenance["region_box"]
        slabs.append((uri, tuple(box["start"]), tuple(box["stop"])))
        asset = asset or ExaspimAsset.from_provenance(provenance)
        if level is not None and provenance["level"] != level:
            raise ValueError("slabs come from different pyramid levels")
        level = provenance["level"]
    source = SlabSource(slabs)
    grid = asset.fused.transforms[level]
    region = Box.from_arrays(source.start, source.stop)
    log.info("%d slabs -> one volume of %s voxels", len(slabs), list(source.shape))
    skeleton = skeletonize_blockwise(source, grid, region, config, block)
    summary = {
        "predictions": prediction_uris, "config": asdict(config), "block": asdict(block),
        "box": {"start": list(region.start), "stop": list(region.stop)},
        "nodes": len(skeleton.coordinates_um), "length_um": skeleton.length_um,
        "gap_filled_length_um": skeleton.filled_length_um, "blockwise": True,
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = swc_prefix or f"{asset.subject_id}_box"
    somata = _somata_um(asset)
    if somata is not None and len(skeleton.coordinates_um):
        near = cKDTree(skeleton.coordinates_um).query(somata)[0] <= dendrites.field_um
        somata = somata[near]
        if len(somata):
            dendrite = label_dendrites(skeleton.coordinates_um, skeleton.edges, somata, dendrites)
            skeleton = replace(skeleton, node_types=np.where(dendrite, SWC_DENDRITE, SWC_AXON))
            summary["dendrites"] = {"somata": len(somata)}
    skeleton.write_swc(out_dir / "skeleton.swc")
    np.savez_compressed(out_dir / "graph.npz", coords=skeleton.coordinates_um,
                        edges=skeleton.edges, filled=skeleton.filled)  # fmt: skip
    if len(skeleton.coordinates_um):
        keep = np.zeros(len(skeleton.coordinates_um), dtype=bool)
        for nodes, length in skeleton.pieces():
            if length >= min_swc_um:
                keep[nodes] = True
        summary["pieces"] = skeleton.subset(keep).write_pieces(out_dir / "swc", prefix)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Slab skeleton: %.1f mm -> %s", summary["length_um"] / 1e3, out_dir)
    return summary


def skeletonize_around(
    prediction_uri: str,
    center_um: NDArray,
    side_um: float,
    out_dir: Path,
    config: SkeletonConfig,
    *,
    swc_prefix: str,
) -> dict:
    """Skeletonize the cube of side ``side_um`` around one (z, y, x) point of a prediction."""
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    grid = ExaspimAsset.from_provenance(provenance).fused.transforms[provenance["level"]]
    box = provenance["region_box"]
    region = Box.from_arrays(box["start"], box["stop"])
    half = side_um / 2
    cube = grid.box_covering(np.subtract(center_um, half), np.add(center_um, half), region.stop)
    start = np.clip(np.subtract(cube.start, region.start), 0, region.shape)
    stop = np.clip(np.subtract(cube.stop, region.start), start, region.shape)
    return skeletonize_prediction(
        prediction_uri, out_dir, config, crop=Box.from_arrays(start, stop), swc_prefix=swc_prefix
    )


SWC_AXON, SWC_DENDRITE = 2, 3


def _somata_um(asset: ExaspimAsset) -> NDArray | None:
    """Return detected soma centres (z, y, x µm), or None when the asset has none."""
    try:
        return np.array([s.center_um for s in load_somata(asset)]).reshape(-1, 3)
    except (FileNotFoundError, OSError, KeyError, ValueError) as error:
        log.warning("No soma detections (%s); nodes stay type 0", error)
        return None


def _cut_um(skeleton, dendrite: NDArray[np.bool_]) -> float:
    """Length of edges joining a dendrite node to an axon-candidate node."""
    e = skeleton.edges
    mixed = dendrite[e[:, 0]] != dendrite[e[:, 1]]
    return float(skeleton.edge_lengths_um[mixed].sum())
