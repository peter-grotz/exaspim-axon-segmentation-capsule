"""Overview of a skeletonized region: the (grown) CCF compartment with every traced piece."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.collections import LineCollection
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from exaspim_axon_segmentation.inference.pipeline import PROVENANCE_ATTR
from exaspim_axon_segmentation.io import ExaspimAsset, RegionRoi
from exaspim_axon_segmentation.postprocess.skeleton import PIECE_COLORS
from exaspim_axon_segmentation.train.swc import read_swc

ATLAS, GROWN = "#FFFFFF", "#FFC400"
DENDRITE, AXON = (0.12, 0.47, 0.71), (1.0, 0.43, 0.0)


def render_skeleton_overview(
    prediction_uri: str,
    skeleton_dir: Path,
    out_path: Path,
    *,
    overview_level: int = 4,
    window_um: float = 300.0,
    slab_um: float = 40.0,
    n_windows: int = 2,
) -> None:
    """Draw the region with its skeleton: whole-compartment projections plus close-ups.

    Top row: z-projection of the fused image over the grown region, with the atlas outline
    (white) and the grown outline (yellow, dashed); beside it the skeleton on the same view, one
    colour per connected piece, matching the ``# COLOR`` of that piece's SWC (the palette cycles
    by length rank). Lower rows: the densest skeleton windows at the prediction's resolution, raw
    next to overlay.
    """
    group = zarr.open_group(prediction_uri, mode="r")
    provenance = dict(group.attrs[PROVENANCE_ATTR])
    asset = ExaspimAsset.from_provenance(provenance)
    summary = json.loads((skeleton_dir / "summary.json").read_text())
    atlas = RegionRoi.from_annotation(asset.annotation, provenance["label_ids"])
    grown = atlas.grown(summary.get("region_growth", 0.0))

    swc = read_swc(skeleton_dir / "skeleton.swc")
    zyx, edges = swc.zyx_um, swc.edges()
    piece_colour = _piece_colours(zyx, edges)

    axon_path = skeleton_dir / "axon_candidates.swc"
    typed = axon_path.exists() and "dendrites" in summary
    fig = plt.figure(figsize=(27 if typed else 18, 8.5 + 6.2 * n_windows), layout="constrained")
    top, *rows = fig.subfigures(1 + n_windows, 1, height_ratios=[1.4] + [1.0] * n_windows)
    top_axes = top.subplots(1, 3 if typed else 2)
    ax_raw, ax_sk = top_axes[0], top_axes[1]

    level = min(overview_level, asset.fused.n_levels - 1)
    tf = asset.fused.transforms[level]
    box = grown.box_in(asset.fused, level, pad_um=60.0)
    image = _display(asset.fused.read(level, box).max(axis=0))
    lo, hi = tf.physical_extent(box)
    extent = (lo[2], hi[2], hi[1], lo[1])
    ax_raw.imshow(image, cmap="gray", extent=extent)
    ax_sk.imshow(image * 0.35, cmap="gray", vmin=0, vmax=1, extent=extent)
    _draw_skeleton(ax_sk, zyx, edges, piece_colour, width=0.5)
    if typed:
        ax_axon = top_axes[2]
        ax_axon.imshow(image * 0.35, cmap="gray", vmin=0, vmax=1, extent=extent)
        dendrite = swc.types == 3
        both = dendrite[edges[:, 0]] & dendrite[edges[:, 1]]
        _draw_skeleton(ax_axon, zyx, edges[both], np.tile(DENDRITE, (len(zyx), 1)), width=0.3)
        axon = read_swc(axon_path)
        _draw_skeleton(ax_axon, axon.zyx_um, axon.edges(), np.tile(AXON, (len(axon.ids), 1)),
                       width=0.9)  # fmt: skip
        d = summary["dendrites"]
        ax_axon.set_title(
            f"axon candidates (orange) · {d['axon_candidate_um'] / 1e3:.0f} mm in "
            f"{len(summary['axon_candidate_pieces'])} SWCs · faint blue = dendrites of "
            f"{d['somata']} detected somata ({d['dendrite_um'] / 1e3:.0f} mm)/",
            loc="left", fontsize=10,
        )  # fmt: skip
    for ax in top_axes:
        _outline(ax, atlas, ATLAS, "-")
        _outline(ax, grown, GROWN, "--")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x (µm)", fontsize=8)
        ax.set_ylabel("y (µm)", fontsize=8)
        ax.tick_params(length=0, labelsize=7)
    regions = "+".join(provenance["regions"])
    ax_raw.set_title(
        f"fused image · z-projection · {regions} atlas outline (white) and grown by "
        f"{grown.growth:.0%} of its volume (yellow)/",
        loc="left", fontsize=10,
    )  # fmt: skip
    ax_sk.set_title(
        f"skeleton · {summary['length_in_region_um'] / 1e3:.0f} mm in {len(summary['pieces'])} "
        f"pieces · colour = piece (as in its SWC)/",
        loc="left", fontsize=10,
    )  # fmt: skip

    fine = asset.fused.transforms[provenance["level"]]
    for sub, centre in zip(rows, _dense_windows(zyx, window_um, n_windows), strict=False):
        half = np.array([slab_um / 2, window_um / 2, window_um / 2])
        wbox = fine.box_covering(
            centre - half, centre + half, asset.fused.shape(provenance["level"])
        )
        crop = _display(asset.fused.read(provenance["level"], wbox).max(axis=0))
        wlo, whi = fine.physical_extent(wbox)
        wext = (wlo[2], whi[2], whi[1], wlo[1])
        raw_ax, over_ax = sub.subplots(1, 2)
        raw_ax.imshow(crop, cmap="gray", extent=wext)
        over_ax.imshow(crop * 0.5, cmap="gray", vmin=0, vmax=1, extent=wext)
        inside = np.all((zyx >= wlo) & (zyx <= whi), axis=1)
        keep = inside[edges[:, 0]] & inside[edges[:, 1]]
        _draw_skeleton(over_ax, zyx, edges[keep], piece_colour, width=1.4)
        for ax in (raw_ax, over_ax):
            _outline(ax, atlas, ATLAS, "-")
            _outline(ax, grown, GROWN, "--")
            ax.set_xlim(wext[0], wext[1])
            ax.set_ylim(wext[2], wext[3])
            ax.tick_params(length=0, labelsize=7)
        where = f"z {centre[0]:.0f} · y {centre[1]:.0f} · x {centre[2]:.0f} µm"
        raw_ax.set_title(f"densest skeleton · {where} · raw · {slab_um:.0f} µm projection/",
                         loc="left", fontsize=10)  # fmt: skip
        over_ax.set_title("skeleton on raw · colour = piece/", loc="left", fontsize=10)

    checkpoint = Path(provenance["weights"].get("checkpoint", ""))
    weights = provenance["weights"].get("name") or f"{checkpoint.parent.name}/{checkpoint.name}"
    fig.suptitle(
        f"{asset.name} · {regions} · weights {weights} · p > {summary['config']['threshold']} · "
        f"debris removed · connected ≤ {summary['config']['connect_um']:.0f} µm/",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _piece_colours(zyx: NDArray, edges: NDArray) -> NDArray:
    """RGB per node: the palette colour of its piece, ranked by length like the SWC files."""
    n = len(zyx)
    if n == 0:
        return np.zeros((0, 3))
    graph = coo_matrix((np.ones(len(edges)), tuple(edges.T)), shape=(n, n))
    _, piece = connected_components(graph, directed=False)
    lengths = np.linalg.norm(zyx[edges[:, 0]] - zyx[edges[:, 1]], axis=1)
    per_piece = np.bincount(piece[edges[:, 0]], weights=lengths, minlength=piece.max() + 1)
    rank = np.empty_like(per_piece, dtype=np.int64)
    rank[np.argsort(-per_piece, kind="stable")] = np.arange(len(per_piece))
    return np.asarray(PIECE_COLORS)[rank[piece] % len(PIECE_COLORS)]


def _draw_skeleton(ax, zyx: NDArray, edges: NDArray, colour: NDArray, *, width: float) -> None:
    segments = np.stack([zyx[edges[:, 0]][:, [2, 1]], zyx[edges[:, 1]][:, [2, 1]]], axis=1)
    ax.add_collection(LineCollection(segments, colors=colour[edges[:, 0]], linewidths=width))


def _outline(ax, roi: RegionRoi, colour: str, style: str) -> None:
    """Contour of the region's z-projection in physical (x, y)."""
    footprint = np.pad(roi.mask.any(axis=0), 1).astype(float)  # pad so the contour closes
    scale, shift = roi.mask_transform.scale, roi.mask_transform.translation
    y = (np.arange(footprint.shape[0]) - 1 + roi.mask_box.start[1]) * scale[1] + shift[1]
    x = (np.arange(footprint.shape[1]) - 1 + roi.mask_box.start[2]) * scale[2] + shift[2]
    ax.contour(x, y, footprint, levels=[0.5], colors=colour, linestyles=style, linewidths=1.0)


def _dense_windows(zyx: NDArray, window_um: float, n: int) -> list[NDArray]:
    """Centres of the ``n`` windows holding the most skeleton nodes, at least 2 windows apart."""
    if not len(zyx):
        return []
    cells = np.floor(zyx[:, 1:] / window_um).astype(int)
    keys, counts = np.unique(cells, axis=0, return_counts=True)
    chosen: list[NDArray] = []
    for k in np.argsort(-counts):
        centre = (keys[k] + 0.5) * window_um
        if all(np.linalg.norm(centre - c[1:]) >= 2 * window_um for c in chosen):
            members = np.all(cells == keys[k], axis=1)
            chosen.append(np.array([np.median(zyx[members, 0]), *centre]))
        if len(chosen) == n:
            break
    return chosen


def _display(mip: NDArray) -> NDArray[np.float64]:
    image = mip.astype(np.float32)
    lo, hi = np.percentile(image, [1, 99.7])
    return np.clip((image - lo) / max(hi - lo, 1.0), 0, 1) ** 0.5


MAP_PALETTE = {
    "axon": "#FF6E00",  # axon candidates (Allen brand orange)
    "dendrite": "#B9B9FF",  # dendrites of detected somata: pale tint of Allen brand blue
    "footprint": "#F3F0E8",  # region z-footprint (Allen warm page)
    "outline": "#AAA39F",  # grown region, dashed (Allen warm gray)
    "soma": "#000000",
    "spine": "#AAA39F",
}


def render_skeleton_map(
    skeleton_dir: Path,
    asset: ExaspimAsset,
    label_ids,
    out_path: Path,
    *,
    region_growth: float | None = None,
    title: str | None = None,
    palette: dict | None = None,
    head_font: str | None = None,
    cache: Path | None = None,
) -> None:
    """Draw the skeleton as a clean map: no image, the region's z-footprint as a light fill.

    Left: one colour per connected piece (as in the SWCs). Right, when the skeleton is typed:
    dendrites of detected somata (pale, thin), axon candidates (bright, thick) and the somata
    (rings). Needs only the skeleton folder and the CCF annotation, so it can be drawn after the
    fact. Text uses the current matplotlib font (``head_font`` for titles); ``cache`` (an
    ``.npz``) stores the region footprints so a re-render skips the annotation entirely.
    """
    colours = MAP_PALETTE | (palette or {})
    summary = json.loads((skeleton_dir / "summary.json").read_text())
    growth = summary.get("region_growth", 0.0) if region_growth is None else region_growth
    if cache is not None and cache.exists():
        z = np.load(cache)
        footprint, grown_fp, lo, cell = z["footprint"], z["grown"], z["lo"], z["cell"]
    else:
        # the footprint and outline need no full resolution: level 1 is 8x smaller
        level = min(1, asset.annotation.n_levels - 1)
        atlas = RegionRoi.from_annotation(asset.annotation, label_ids, level=level)
        grown = atlas.grown(growth)
        cell = np.asarray(atlas.mask_transform.scale)[1:]
        shift = np.asarray(atlas.mask_transform.translation)[1:]
        lo = np.asarray(atlas.mask_box.start[1:]) * cell + shift - cell / 2
        footprint = atlas.mask.any(axis=0)
        # grown footprint on the atlas footprint's grid, padded to the grown extent
        g_lo = np.asarray(grown.mask_box.start[1:]) * cell + shift - cell / 2
        offset = np.round((g_lo - lo) / cell).astype(int)
        grown_fp = grown.mask.any(axis=0)
        pad = np.maximum(-offset, 0)
        lo = lo - pad * cell
        size = np.maximum(np.asarray(footprint.shape) + pad, offset + pad + grown_fp.shape)
        footprint = _place(footprint, pad, size)
        grown_fp = _place(grown_fp, offset + pad, size)
        if cache is not None:
            np.savez_compressed(cache, footprint=footprint, grown=grown_fp, lo=lo, cell=cell)
    hi = lo + np.multiply(footprint.shape, cell)
    extent = (lo[1], hi[1], hi[0], lo[0])
    margin = 0.04 * max(hi - lo)

    swc = read_swc(skeleton_dir / "skeleton.swc")
    zyx, edges = swc.zyx_um, swc.edges()
    typed = "dendrites" in summary and (swc.types > 0).any()
    panels = 2 if typed else 1
    aspect = (hi[0] - lo[0]) / (hi[1] - lo[1])
    fig, axes = plt.subplots(1, panels, figsize=(9.5 * panels, 9.5 * aspect + 1.2),
                             layout="constrained", squeeze=False)  # fmt: skip
    axes = axes[0]
    fill = np.zeros((*footprint.shape, 4))
    fill[footprint] = (*matplotlib.colors.to_rgb(colours["footprint"]), 1.0)
    yy = lo[0] + (np.arange(footprint.shape[0]) + 0.5) * cell[0]
    xx = lo[1] + (np.arange(footprint.shape[1]) + 0.5) * cell[1]
    title_font = {"fontfamily": head_font} if head_font else {}
    for ax in axes:
        ax.imshow(fill, extent=extent, interpolation="nearest")
        ax.contour(xx, yy, np.pad(grown_fp, 0).astype(float), levels=[0.5],
                   colors=colours["outline"], linestyles="--", linewidths=1.0)  # fmt: skip
        ax.set_xlim(lo[1] - margin, hi[1] + margin)
        ax.set_ylim(hi[0] + margin, lo[0] - margin)
        ax.set_aspect("equal")
        ax.tick_params(length=0, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(colours["spine"])
        ax.set_xlabel("x (um)", fontsize=9)
        ax.set_ylabel("y (um)", fontsize=9)
    n_pieces = len(summary.get("pieces", []))
    _draw_skeleton(axes[0], zyx, edges, _piece_colours(zyx, edges), width=0.9)
    axes[0].set_title(f"{n_pieces} connected pieces · colour per piece (as in its SWC)/",
                      loc="left", fontsize=14, **title_font)  # fmt: skip
    if typed:
        dendrite = swc.types == 3
        both = dendrite[edges[:, 0]] & dendrite[edges[:, 1]]
        pale = np.tile(matplotlib.colors.to_rgb(colours["dendrite"]), (len(zyx), 1))
        _draw_skeleton(axes[1], zyx, edges[both], pale, width=0.5)
        bright = np.tile(matplotlib.colors.to_rgb(colours["axon"]), (len(zyx), 1))
        _draw_skeleton(axes[1], zyx, edges[~both], bright, width=2.2)
        d = summary["dendrites"]
        try:
            from exaspim_axon_segmentation.postprocess.somata import load_somata

            somata = np.array([x.center_um for x in load_somata(asset)]).reshape(-1, 3)
            inside = (somata[:, 1] >= lo[0]) & (somata[:, 1] <= hi[0])
            inside &= (somata[:, 2] >= lo[1]) & (somata[:, 2] <= hi[1])
            axes[1].scatter(somata[inside, 2], somata[inside, 1], s=28, facecolors="none",
                            edgecolors=colours["soma"], linewidths=0.8)  # fmt: skip
        except FileNotFoundError:
            pass
        axes[1].set_title(
            f"axon candidates {d['axon_candidate_um'] / 1e3:.0f} mm · dendrites of detected "
            f"somata {d['dendrite_um'] / 1e3:.0f} mm/",
            loc="left", fontsize=14, **title_font,
        )  # fmt: skip
    fig.suptitle(
        title or f"{asset.subject_id} · md · {summary['length_in_region_um'] / 1e3:.0f} mm of "
        f"skeleton · region grown by {growth:.0%} (dashed)/",
        x=0.01, ha="left", fontsize=18, fontweight="bold", **title_font,
    )  # fmt: skip
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _place(mask: NDArray[np.bool_], corner: NDArray, size: NDArray) -> NDArray[np.bool_]:
    """Copy a 2-D mask into a zero canvas of ``size`` at ``corner``."""
    canvas = np.zeros(tuple(size), dtype=bool)
    canvas[corner[0] : corner[0] + mask.shape[0], corner[1] : corner[1] + mask.shape[1]] = mask
    return canvas
