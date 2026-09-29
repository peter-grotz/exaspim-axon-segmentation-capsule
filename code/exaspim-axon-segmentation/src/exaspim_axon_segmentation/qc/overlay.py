"""Visual QC: sample-space CCF region outlines drawn on the fused image.

Both layers are placed on the image's own voxel grid via physical coordinates, so a correct
overlay here validates exactly the mapping the inference ROI uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from exaspim_axon_segmentation.io import Box, ExaspimAsset, RegionRoi, Structure

AXIS_NAMES = ("z", "y", "x")
REGION_COLOR = "#CDEB05"
BOUNDARY_COLOR = (1.0, 1.0, 1.0, 0.28)
DETAIL_COLOR = "#FF6E00"


@dataclass(frozen=True)
class PlaneView:
    """One orthogonal image plane with the region mask and label boundaries on the same grid."""

    image: NDArray
    region: NDArray[np.bool_]
    boundaries: NDArray[np.bool_]
    extent_mm: tuple[float, float, float, float]
    normal: int
    position_um: float

    @property
    def axis_labels(self) -> tuple[str, str]:
        """(horizontal, vertical) axis names."""
        rows, cols = (a for a in range(3) if a != self.normal)
        return AXIS_NAMES[cols], AXIS_NAMES[rows]


def load_plane(
    asset: ExaspimAsset, roi: RegionRoi, level: int, normal: int, position_um: float, box: Box
) -> PlaneView:
    """Read an image plane at ``position_um`` along ``normal``, clipped to ``box`` in-plane."""
    image = asset.fused
    tf = image.transforms[level]
    index = round((position_um - tf.translation[normal]) / tf.scale[normal])
    start, stop = list(box.start), list(box.stop)
    start[normal], stop[normal] = index, index + 1
    plane_box = Box.from_arrays(start, stop).clip(image.shape(level))

    pixels = np.squeeze(image.read(level, plane_box), axis=normal)
    labels = np.squeeze(asset.annotation.resample_nearest(0, tf, plane_box), axis=normal)
    lo, hi = tf.physical_extent(plane_box)
    rows, cols = (a for a in range(3) if a != normal)
    extent = (lo[cols] / 1e3, hi[cols] / 1e3, hi[rows] / 1e3, lo[rows] / 1e3)
    return PlaneView(
        image=pixels,
        region=np.isin(labels, list(roi.label_ids)),
        boundaries=_label_boundaries(labels),
        extent_mm=extent,
        normal=normal,
        position_um=tf.to_physical(np.array(plane_box.start))[normal],
    )


def _label_boundaries(labels: NDArray) -> NDArray[np.bool_]:
    edges = np.zeros(labels.shape, dtype=bool)
    edges[1:, :] |= labels[1:, :] != labels[:-1, :]
    edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    return edges


def _display_range(pixels: NDArray) -> tuple[float, float]:
    foreground = pixels[pixels > 0]
    if foreground.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(foreground, [1.0, 99.7])
    return float(lo), float(max(hi, lo + 1))


def draw_plane(ax: plt.Axes, view: PlaneView, *, title: str, highlight: RegionRoi | None) -> None:
    """Render one plane: gamma-scaled image, all label boundaries, and the region outline."""
    lo, hi = _display_range(view.image)
    scaled = np.clip((view.image.astype(np.float32) - lo) / (hi - lo), 0, 1) ** 0.5
    ax.imshow(scaled, cmap="gray", extent=view.extent_mm, interpolation="none")
    overlay = np.zeros((*view.boundaries.shape, 4), dtype=np.float32)
    overlay[view.boundaries] = BOUNDARY_COLOR
    ax.imshow(overlay, extent=view.extent_mm, interpolation="none")
    if view.region.any():
        ax.contour(
            view.region.astype(np.uint8),
            levels=[0.5],
            colors=[REGION_COLOR],
            linewidths=1.4,
            extent=view.extent_mm,
            origin="upper",
        )
    if highlight is not None:
        rows, cols = (a for a in range(3) if a != view.normal)
        lo_mm, hi_mm = highlight.lo_um / 1e3, highlight.hi_um / 1e3
        ax.add_patch(
            Rectangle(
                (lo_mm[cols], lo_mm[rows]),
                hi_mm[cols] - lo_mm[cols],
                hi_mm[rows] - lo_mm[rows],
                fill=False,
                edgecolor=DETAIL_COLOR,
                linewidth=1.2,
                linestyle="--",
            )
        )
    horizontal, vertical = view.axis_labels
    ax.set_xlabel(f"{horizontal} (mm)")
    ax.set_ylabel(f"{vertical} (mm)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("black")


def _row(
    subfig: SubFigure,
    views: list[PlaneView],
    titles: list[str],
    highlight: RegionRoi | None,
    heading: str,
) -> None:
    axes = subfig.subplots(1, len(views))
    for ax, view, title in zip(np.atleast_1d(axes), views, titles, strict=True):
        draw_plane(ax, view, title=title, highlight=highlight)
    subfig.suptitle(heading, x=0.01, ha="left", fontsize=14, fontweight="bold")


def render_region_overlay(
    asset: ExaspimAsset,
    roi: RegionRoi,
    structure: Structure,
    out_path: Path,
    *,
    overview_um: float = 24.0,
    detail_um: float = 6.0,
    pad_um: float = 400.0,
    n_sections: int = 4,
) -> Figure:
    """Write a three-row QC figure of a region's outline on the fused image.

    Rows: (1) whole-brain orthogonal planes through the region's largest cross-sections;
    (2) the same planes zoomed to the ROI; (3) ``n_sections`` z-planes evenly spaced through it.

    Parameters
    ----------
    asset : ExaspimAsset
        Asset providing the fused image and sample-space annotation.
    roi : RegionRoi
        Region to outline.
    structure : Structure
        Ontology entry of the region (for labelling).
    out_path : Path
        Output image path (PNG recommended).
    overview_um, detail_um : float
        Target XY voxel size of the overview and zoomed rows; the closest pyramid level is used.
    pad_um : float
        Margin around the ROI in the zoomed rows.
    n_sections : int
        Number of z-sections in the last row.
    """
    fused = asset.fused
    overview_level = fused.level_for_spacing(overview_um)
    detail_level = fused.level_for_spacing(detail_um)
    overview_box = Box.full(fused.shape(overview_level))
    detail_box = roi.box_in(fused, detail_level, pad_um=pad_um)
    center = roi.peak_um

    overview = [
        load_plane(asset, roi, overview_level, n, c, overview_box) for n, c in enumerate(center)
    ]
    zoom = [load_plane(asset, roi, detail_level, n, c, detail_box) for n, c in enumerate(center)]
    z_positions = np.linspace(roi.lo_um[0], roi.hi_um[0], n_sections + 2)[1:-1]
    sections = [load_plane(asset, roi, detail_level, 0, z, detail_box) for z in z_positions]

    def plane_title(view: PlaneView) -> str:
        return f"{AXIS_NAMES[view.normal]} = {view.position_um / 1e3:.2f} mm/"

    fig = plt.figure(figsize=(18, 17), layout="constrained")
    rows = fig.subfigures(3, 1, height_ratios=[1.0, 1.0, 0.9])
    xy_um = fused.transforms[overview_level].scale[2]
    _row(
        rows[0],
        overview,
        [plane_title(v) for v in overview],
        roi,
        f"whole brain · level {overview_level} ({xy_um:.1f} um xy)/",
    )
    xy_um = fused.transforms[detail_level].scale[2]
    _row(
        rows[1],
        zoom,
        [plane_title(v) for v in zoom],
        None,
        f"region zoom · level {detail_level} ({xy_um:.1f} um xy)/",
    )
    _row(
        rows[2],
        sections,
        [plane_title(v) for v in sections],
        None,
        f"z-sections through {structure.acronym}/",
    )

    fig.suptitle(
        f"{asset.name}\n{structure.acronym} ({structure.id}) · {structure.name.lower()} · "
        f"reverse anno on fused image/",
        x=0.01,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    extent = (roi.hi_um - roi.lo_um) / 1e3
    fig.text(
        0.01,
        -0.005,
        f"region volume {roi.volume_mm3:.2f} mm³ (sample space) · bbox (z, y, x) "
        f"{extent[0]:.2f} × {extent[1]:.2f} × {extent[2]:.2f} mm · labels {sorted(roi.label_ids)}"
        f" · lime = {structure.acronym}, white = all CCF boundaries, dashed = region bounding box",
        fontsize=10,
        ha="left",
        va="top",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
