"""Tiled TRAILMAP inference over CCF regions."""

from exaspim_axon_segmentation.inference.normalize import IntensityScaling, estimate_scaling
from exaspim_axon_segmentation.inference.pipeline import plan_region, predict_region
from exaspim_axon_segmentation.inference.predict import RunStats, TilePredictor
from exaspim_axon_segmentation.inference.tiling import Tile, plan_tiles

__all__ = [
    "IntensityScaling",
    "RunStats",
    "Tile",
    "TilePredictor",
    "estimate_scaling",
    "plan_region",
    "plan_tiles",
    "predict_region",
]
