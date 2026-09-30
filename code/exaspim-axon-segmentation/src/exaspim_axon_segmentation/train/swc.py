"""SWC traces (Janelia Workstation / Horta exports): read, resample, write.

Horta writes node coordinates relative to a ``# OFFSET x y z`` header, in micrometers; the absolute
physical position is ``node + offset``, in the same frame as ``fused.zarr`` (voxel index =
physical / scale, translation 0 at level 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from exaspim_axon_segmentation.io import NgffImage


@dataclass(frozen=True, eq=False)
class Swc:
    """A traced tree (or forest) in absolute physical coordinates.

    Attributes
    ----------
    ids, types, parents : numpy.ndarray of int
        SWC node id, compartment type, and parent id (-1 for roots).
    xyz_um : numpy.ndarray of float, shape (n, 3)
        Absolute (x, y, z) positions in micrometers.
    radius : numpy.ndarray of float
        Node radius as stored (Horta writes 1.0).
    header : tuple of str
        Original comment lines (``OFFSET`` is rewritten on output).
    """

    ids: NDArray[np.int64]
    types: NDArray[np.int64]
    xyz_um: NDArray[np.float64]
    radius: NDArray[np.float64]
    parents: NDArray[np.int64]
    header: tuple[str, ...] = ()

    @property
    def zyx_um(self) -> NDArray[np.float64]:
        """Positions in (z, y, x) order, the package's axis convention."""
        return self.xyz_um[:, ::-1]

    def edges(self) -> NDArray[np.int64]:
        """(child, parent) row-index pairs for every non-root node."""
        index = {int(i): k for k, i in enumerate(self.ids)}
        pairs = [(k, index[int(p)]) for k, p in enumerate(self.parents) if int(p) in index]
        return np.array(pairs, dtype=np.int64).reshape(-1, 2)

    def path_length_um(self) -> float:
        """Total length of all edges."""
        e = self.edges()
        return float(np.linalg.norm(self.xyz_um[e[:, 0]] - self.xyz_um[e[:, 1]], axis=1).sum())


def read_swc(path: Path | str) -> Swc:
    """Read an SWC file, applying any ``# OFFSET`` header."""
    return parse_swc(Path(path).read_text())


def parse_swc(text: str) -> Swc:
    """Parse SWC text, applying any ``# OFFSET`` header."""
    offset = np.zeros(3)
    header, rows = [], []
    for line in text.splitlines():
        if line.startswith("#"):
            header.append(line)
            if line.upper().startswith("# OFFSET"):
                offset = np.array([float(v) for v in line.split()[2:5]])
        elif line.strip():
            rows.append(line.split()[:7])
    a = np.array(rows, dtype=np.float64).reshape(-1, 7)
    return Swc(
        ids=a[:, 0].astype(np.int64),
        types=a[:, 1].astype(np.int64),
        xyz_um=a[:, 2:5] + offset,
        radius=a[:, 5],
        parents=a[:, 6].astype(np.int64),
        header=tuple(h for h in header if not h.upper().startswith("# OFFSET")),
    )


def resample(swc: Swc, step_um: float) -> Swc:
    """Insert nodes along every edge so no two connected nodes are more than ``step_um`` apart.

    Topology, types, and radii are preserved; inserted nodes inherit the child's type and radius.
    Positions are interpolated linearly, so curved axons between sparse nodes become chords.
    """
    index = {int(i): k for k, i in enumerate(swc.ids)}
    new_id = {int(i): n + 1 for n, i in enumerate(swc.ids)}
    ids, types, xyz, radius, parents = [], [], [], [], []
    next_id = len(swc.ids) + 1

    def add(node_id: int, k: int, position: NDArray, parent_id: int) -> None:
        ids.append(node_id)
        types.append(swc.types[k])
        xyz.append(position)
        radius.append(swc.radius[k])
        parents.append(parent_id)

    for k, node in enumerate(swc.ids):
        parent = int(swc.parents[k])
        if parent not in index:
            add(new_id[int(node)], k, swc.xyz_um[k], -1)
            continue
        start, end = swc.xyz_um[index[parent]], swc.xyz_um[k]
        n_steps = int(np.ceil(np.linalg.norm(end - start) / step_um))
        previous = new_id[parent]
        for t in np.arange(1, n_steps) / n_steps:
            add(next_id, k, start + t * (end - start), previous)
            previous, next_id = next_id, next_id + 1
        add(new_id[int(node)], k, swc.xyz_um[k], previous)
    # Renumber in insertion order: every parent then precedes its children, as SWC readers expect.
    renumber = {old: new for new, old in enumerate(ids, start=1)}
    return Swc(
        ids=np.arange(1, len(ids) + 1),
        types=np.asarray(types),
        xyz_um=np.asarray(xyz),
        radius=np.asarray(radius),
        parents=np.array([renumber.get(p, -1) for p in parents]),
        header=swc.header,
    )


def write_swc(
    swc: Swc,
    path: Path | str,
    *,
    offset_um: NDArray | None = None,
    color: tuple[float, float, float] | None = None,
) -> None:
    """Write an SWC file in Horta's layout; positions are relative to ``offset_um``.

    ``offset_um`` defaults to the first node. Header order follows the Janelia Workstation export:
    the original comment lines (e.g. ``# ORIGINAL_SOURCE``), ``# OFFSET``, then ``# COLOR``.
    """
    offset = np.asarray(offset_um if offset_um is not None else swc.xyz_um[0], dtype=np.float64)
    header = [h for h in swc.header if not h.upper().startswith("# COLOR")]
    lines = [*header, f"# OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}"]
    if color is not None:
        lines.append("# COLOR " + ",".join(f"{c:.6f}" for c in color))
    rel = swc.xyz_um - offset
    for i, t, (x, y, z), r, p in zip(swc.ids, swc.types, rel, swc.radius, swc.parents, strict=True):
        lines.append(f"{i} {t} {x:.6f} {y:.6f} {z:.6f} {r:.6f} {p}")
    Path(path).write_text("\n".join(lines) + "\n")


def snap_to_signal(
    swc: Swc,
    image: NgffImage,
    level: int,
    *,
    step_um: float,
    tube_um: float = 8.0,
    smooth_um: float = 1.5,
) -> Swc:
    """Replace every edge by the brightest path between its endpoints, then resample to ``step_um``.

    For sparse traces (nodes tens of micrometers apart) a straight chord misses curved axons by
    several micrometers. Each edge is re-routed as the minimal-cost path through
    ``1 / (smoothed intensity)`` inside a box around the chord grown by ``tube_um``; endpoints are
    first moved to the brightest voxel within ``tube_um / 2``.

    Parameters
    ----------
    swc : Swc
        Sparse trace in absolute physical coordinates.
    image : NgffImage
        Fused image the trace was made on.
    level : int
        Pyramid level used for routing (level 1, 1.5 µm, is a good default).
    step_um : float
        Node spacing of the output.
    tube_um : float
        Search margin around each chord.
    smooth_um : float
        Gaussian smoothing before computing the cost, to route along the axon, not noise.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.graph import route_through_array

    tf = image.transforms[level]
    scale = np.asarray(tf.scale)
    zyx = swc.zyx_um
    snapped = np.array([_brightest_near(image, level, p, tube_um / 2) for p in zyx])

    index = {int(i): k for k, i in enumerate(swc.ids)}
    paths: dict[int, NDArray] = {}
    for k, p in enumerate(swc.parents):
        if int(p) not in index:
            continue
        a, b = snapped[index[int(p)]], snapped[k]
        box = tf.box_covering(
            np.minimum(a, b) - tube_um, np.maximum(a, b) + tube_um, image.shape(level)
        )
        block = gaussian_filter(image.read(level, box).astype(np.float32), smooth_um / scale)
        cost = 1.0 / (block - block.min() + 1.0)
        start = np.clip(
            np.round(tf.to_index(a)).astype(int) - box.start, 0, np.array(box.shape) - 1
        )
        end = np.clip(np.round(tf.to_index(b)).astype(int) - box.start, 0, np.array(box.shape) - 1)
        route, _ = route_through_array(cost, tuple(start), tuple(end), geometric=True)
        paths[k] = tf.to_physical(np.asarray(route) + np.asarray(box.start))

    # Rebuild a dense tree: each routed path replaces its edge, then resample to step_um.
    ids, types, xyz, radius, parents = [], [], [], [], []
    new_id: dict[int, int] = {}
    for k, node in enumerate(swc.ids):
        parent_row = index.get(int(swc.parents[k]))
        previous = -1 if parent_row is None else new_id[int(swc.ids[parent_row])]
        interior = paths[k][1:-1] if k in paths else np.empty((0, 3))
        for point in interior:
            ids.append(len(ids) + 1)
            types.append(swc.types[k]), xyz.append(point[::-1]), radius.append(swc.radius[k])
            parents.append(previous)
            previous = ids[-1]
        ids.append(len(ids) + 1)
        types.append(swc.types[k]), xyz.append(snapped[k][::-1]), radius.append(swc.radius[k])
        parents.append(previous)
        new_id[int(node)] = ids[-1]
    dense = Swc(
        ids=np.asarray(ids), types=np.asarray(types), xyz_um=np.asarray(xyz),
        radius=np.asarray(radius), parents=np.asarray(parents), header=swc.header,
    )  # fmt: skip
    return resample(dense, step_um)


def _brightest_near(image: NgffImage, level: int, zyx_um: NDArray, radius_um: float) -> NDArray:
    tf = image.transforms[level]
    box = tf.box_covering(zyx_um - radius_um, zyx_um + radius_um, image.shape(level))
    block = image.read(level, box)
    k = np.unravel_index(np.argmax(block), block.shape)
    return tf.to_physical(np.asarray(k) + np.asarray(box.start))
