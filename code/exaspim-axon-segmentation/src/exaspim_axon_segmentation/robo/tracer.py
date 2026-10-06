"""Batched Robo-LS v2 tracer: trace forward from piece ends, and score arms at junctions.

Follows robo_ls/robo_trace2.py and robo_nodes2.py (skeleton_proofreading): at every 5 µm step the
network sees a 48³ heading-aligned crop at 1.5 µm (image from fused level 1, normalised to the
neurite's own brightness), a mask of all fragment points in view, and a mask of its own path,
plus momentum scalars; it predicts the next heading, an event (continue / branch / terminal) and,
for query points, P(same cell as my path). Positions are world µm (x, y, z).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.spatial import cKDTree

from exaspim_axon_segmentation.robo.image import Level1Image

log = logging.getLogger(__name__)

N, VOX, HALF = 48, 1.5, 36.0
STEP_UM = 5.0
HIST = 5
PROBE_S = np.arange(14) * 1.5  # last ~20 µm of path, for the brightness probe


def unit(v: NDArray, axis: int = -1) -> NDArray:
    """Normalise v to unit length along ``axis``."""
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-9)


def frame(h: NDArray) -> NDArray:
    """Orthonormal rows (a, u, v) for heading h (x, y, z), u horizontal."""
    h = unit(np.asarray(h, float))
    ref = np.array([0.0, 0.0, 1.0]) if abs(h[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(h, ref))
    return np.stack([h, u, np.cross(h, u)])


def polyline_back(points: NDArray, s: NDArray) -> NDArray:
    """Points at arclength ``s`` (µm) behind the end of a polyline (forward order)."""
    p = np.asarray(points, float)[::-1]
    g = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s = np.minimum(np.asarray(s, float), g[-1])
    return np.stack([np.interp(s, g, p[:, k]) for k in range(3)], 1)


def scalars(f: NDArray, hist: NDArray, soma_vec: NDArray) -> NDArray[np.float32]:
    """Momentum (4 previous headings), soma direction and log distance, in the local frame."""
    rel = (hist[1:HIST] @ f.T).reshape(-1)
    d = np.linalg.norm(soma_vec)
    sd = f @ (soma_vec / max(d, 1e-6))
    return np.concatenate([rel, sd, [np.log1p(d / 100.0)]]).astype(np.float32)


def crop(
    image: Level1Image, p: NDArray, f: NDArray, probe: NDArray
) -> tuple[NDArray[np.float32], float, float]:
    """Return the normalised heading-aligned 48³ crop at 1.5 µm and its centre contrast."""
    g1 = (np.arange(N) - (N - 1) / 2) * VOX
    grid = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
    pts = np.concatenate([p[None] + grid @ f, probe], 0)
    v = image.to_voxel(pts)
    lo = np.floor(v.min(0)).astype(int) - 2
    hi = np.ceil(v.max(0)).astype(int) + 3
    blk = image.block(lo, hi - lo).astype(np.float32)
    vv = (v - lo).T
    c = map_coordinates(blk, vv[:, : N**3], order=1, mode="nearest").reshape(N, N, N)
    if not np.any(c):
        return np.zeros((N, N, N), np.float32), 0.0
    bv = map_coordinates(maximum_filter(blk, size=3), vv[:, N**3 :], order=1, mode="nearest")
    bg = np.percentile(c, 20)
    b = np.median(np.sort(bv)[len(bv) // 2 :])
    return ((c - bg) / max(b - bg, 8.0)).astype(np.float32), contrast(c)


def contrast(c: NDArray) -> float:
    """Return the signal-to-noise at the centre of a crop.

    Brightest smoothed voxel within ~2 µm, above the crop's median, in units of its robust noise
    (MAD); robust to the few-count data of dim brains.
    """
    m = N // 2
    core = gaussian_filter(c[m - 8 : m + 8, m - 8 : m + 8, m - 8 : m + 8], 1.0)
    med = np.median(c)
    noise = max(1.4826 * np.median(np.abs(c - med)), 0.5)
    return float((core[6:10, 6:10, 6:10].max() - med) / noise)


def cone_ahead(
    image: Level1Image,
    p: NDArray,
    h: NDArray,
    snr_min: float,
    pick: tuple = (8.0, 20.0),
    max_hop: float = 7.0,
) -> tuple[NDArray, NDArray] | None:
    """Brightest point ``pick`` µm ahead of p within 30° of heading h, if the cone is bright.

    Bright means smoothed signal at least ``snr_min`` x the robust noise (MAD of raw counts) in
    every 8 µm band from 8 to 40 µm ahead, and the brightest points of successive 4 µm bands
    (the ridge ahead) lie within ``max_hop`` µm of each other but once, i.e. form a line; otherwise
    None. Also returns that ridge, for an identity check.
    """
    f = frame(h)
    centre = p + 24.0 * f[0]
    g1 = (np.arange(N) - (N - 1) / 2) * VOX
    grid = np.stack(np.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
    v = image.to_voxel(centre[None] + grid @ f)
    lo = np.floor(v.min(0)).astype(int) - 2
    hi = np.ceil(v.max(0)).astype(int) + 3
    blk = image.block(lo, hi - lo).astype(np.float32)
    im = map_coordinates(blk, (v - lo).T, order=1, mode="nearest").reshape(N, N, N)
    sm = gaussian_filter(im, 1.0)
    a, u, w = np.meshgrid(g1 + 24.0, g1, g1, indexing="ij")
    cone = (a >= 8) & (a <= 40) & (np.hypot(u, w) <= np.tan(np.radians(30)) * a + 2)
    med = np.median(im)
    noise = max(1.4826 * np.median(np.abs(im - med)), 0.5)
    for d in (8, 16, 24, 32):
        band = cone & (a >= d) & (a < d + 8)
        if (sm[band].max() - med) / noise < snr_min:
            return None

    def brightest(mask):
        k = np.unravel_index(np.argmax(np.where(mask, sm, -np.inf)), sm.shape)
        return p + a[k] * f[0] + u[k] * f[1] + w[k] * f[2]

    target = brightest(cone & (a >= pick[0]) & (a <= pick[1]))
    ridge = np.array([brightest(cone & (a >= d) & (a < d + 4)) for d in range(8, 33, 4)])
    if (np.linalg.norm(np.diff(ridge, axis=0), axis=1) > max_hop).sum() > 1:
        return None  # scattered noise maxima, not a fibre (one jump allowed: bends, beads)
    return target, ridge


def recentre(image: Level1Image, p: NDArray, d: NDArray, r: float = 1.0) -> NDArray:
    """Move p within ±r µm perpendicular to d onto the brightness ridge."""
    f = frame(d)
    offs = np.linspace(-r, r, 5)
    pts = np.array([[a, b, c] for a in (-1.5, 0.0, 1.5) for b in offs for c in offs])
    v = image.to_voxel(p[None] + pts @ f)
    lo = np.floor(v.min(0)).astype(int) - 4
    hi = np.ceil(v.max(0)).astype(int) + 5
    blk = gaussian_filter(image.block(lo, hi - lo).astype(np.float32), 0.8)
    val = map_coordinates(blk, (v - lo).T, order=1, mode="nearest").reshape(3, 5, 5).mean(0)
    k = np.unravel_index(np.argmax(val), val.shape)
    if val[k] <= 1.1 * val[2, 2]:
        return p
    return p + offs[k[0]] * f[1] + offs[k[1]] * f[2]


@dataclass
class Fragments:
    """All skeleton points in view of the tracer, with their piece (-1: not part of a piece)."""

    points: NDArray[np.float64]  # (n, 3) x, y, z µm
    piece: NDArray[np.int64]
    tree: cKDTree = field(init=False)

    def __post_init__(self) -> None:
        self.tree = cKDTree(self.points)


@dataclass
class Start:
    """Where a trace starts: the end of a piece, arriving along ``path`` (forward order)."""

    path: NDArray[np.float64]  # (m, 3) x, y, z, ending at the start position
    piece: int
    soma: NDArray | None = None


@dataclass
class _State:
    start: Start
    pos: NDArray
    hist: NDArray
    path: NDArray
    trace: list = field(default_factory=list)
    rejected: set = field(default_factory=set)
    steps: int = 0
    s: float = 0.0
    term_run: int = 0
    dark_run: int = 0
    signal: list = field(default_factory=list)  # centre contrast (signal / noise), per step
    restarts: int = 0
    bridge_p: list = field(default_factory=list)  # identity P of every gap the trace considered
    last_bridge: int = -100  # step of the last bridge
    visited: list = field(default_factory=list)  # every position stepped to (diagnostics)
    alive: bool = True
    status: str = "maxlen"
    link: int = -1
    p_link: float = float("nan")


def _initial(start: Start) -> _State:
    fwd = np.asarray(start.path, float)
    pb = polyline_back(fwd, np.arange(0, 25.1, 5.0))
    ch = pb[:-1] - pb[1:]
    ok = np.linalg.norm(ch, axis=1) > 1e-3
    hist = unit(ch)
    for k in range(1, len(hist)):
        if not ok[k]:
            hist[k] = hist[k - 1]
    path = polyline_back(fwd, np.arange(0, 100.1, 2.0))[::-1].copy()
    return _State(start, fwd[-1].copy(), hist, path, trace=[fwd[-1].copy()])


class RoboTracer:
    """The Robo-LS v2 network with the decoding of robo_trace2 (soft-argmax, ridge re-centring)."""

    def __init__(
        self,
        checkpoint: str,
        image: Level1Image,
        *,
        device: str = "cpu",
        accept: float = 0.5,
        max_steps: int = 60,
        batch: int = 32,
        threads: int = 8,
        lost_snr: float = 3.0,
        restart_snr: float = 3.0,
        max_restarts: int = 4,
        bridge_accept: float = 0.8,
        bridge_every: int = 4,
    ):
        import torch

        from exaspim_axon_segmentation.robo.model import RoboNet2, grid_dirs2

        self.torch = torch
        self.model = RoboNet2(width=32)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model.to(device).eval()
        self.device = device
        self.dirs = grid_dirs2().to(device).view(-1, 3)
        self.image, self.accept, self.max_steps = image, accept, max_steps
        self.batch, self.pool = batch, ThreadPoolExecutor(threads)
        self.lost_snr = lost_snr  # stop after 2 steps with centre contrast below this
        # gap bridging: a lost / terminal stop with bright fibre in the cone ahead that the identity
        # head calls the same cell (P >= accept) resumes from the brightest point 8-20 µm ahead
        self.restart_snr, self.max_restarts = restart_snr, max_restarts
        self.bridge_accept = bridge_accept  # identity P needed to bridge
        self.bridge_every = bridge_every  # steps on signal needed between two bridges

    # ------------------------------------------------------------------ observation and forward
    def _observe(self, st: _State, frags: Fragments, queries: list[int] | None):
        f = frame(st.hist[0])
        image, centre = crop(self.image, st.pos, f, polyline_back(st.path, PROBE_S))
        jj = np.asarray(frags.tree.query_ball_point(st.pos, HALF * 1.75), int)
        loc = (frags.points[jj] - st.pos) @ f.T if len(jj) else np.zeros((0, 3))
        inb = np.all(np.abs(loc) <= HALF, 1) if len(loc) else np.zeros(0, bool)
        loc_in, pc_in = loc[inb], frags.piece[jj[inb]] if len(jj) else np.zeros(0, int)
        fr = np.full((1024, 3), np.nan, np.float32)
        sel = np.arange(len(loc_in))
        if len(sel) > 1024:
            sel = np.random.default_rng(st.steps).choice(len(loc_in), 1024, replace=False)
        fr[: len(sel)] = loc_in[sel]
        own = (st.path[-50:] - st.pos) @ f.T
        ownp = np.full((64, 3), np.nan, np.float32)
        ownp[: min(64, len(own))] = own[-64:]
        if queries is None:  # trace: pieces just ahead
            ahead = np.zeros(len(loc_in), bool)
            if len(loc_in):
                ahead = (loc_in[:, 0] >= -2) & (loc_in[:, 0] <= 15)
                ahead &= np.linalg.norm(loc_in[:, 1:], axis=1) <= 6
            queries = [c for c in np.unique(pc_in[ahead])
                       if c >= 0 and c != st.start.piece and c not in st.rejected][:6]  # fmt: skip
            pool_pts, pool_pc = loc_in[ahead], pc_in[ahead]
        else:  # junction: arms 4-30 µm from the node
            dist = np.linalg.norm(loc_in, axis=1) if len(loc_in) else np.zeros(0)
            near = (dist >= 4) & (dist <= 30)
            pool_pts, pool_pc = loc_in[near], pc_in[near]
        q, qmap = [], []
        for c in queries:
            pts_c = pool_pts[pool_pc == c]
            if not len(pts_c):
                continue
            pick = pts_c[np.linspace(0, len(pts_c) - 1, min(8, len(pts_c))).astype(int)]
            qmap.append((c, len(q), len(q) + len(pick)))
            q += list(pick)
        soma = st.start.soma
        sv = (soma - st.pos) if soma is not None else np.zeros(3)
        return dict(f=f, image=image, centre=centre, fr=fr, own=ownp, scal=scalars(f, st.hist, sv),
                    q=np.asarray(q, np.float32).reshape(-1, 3), qmap=qmap)  # fmt: skip

    def _forward(self, obs: list[dict]):
        torch = self.torch
        from torch.nn import functional as F  # noqa: N812

        def rasterise(p):
            ok = torch.isfinite(p).all(-1)
            idx = torch.round(torch.nan_to_num(p) / VOX + (N - 1) / 2).long()
            ok &= ((idx >= 0) & (idx < N)).all(-1)
            lin = (idx[..., 0] * N + idx[..., 1]) * N + idx[..., 2]
            m = torch.zeros(p.shape[0], N**3, device=p.device)
            m.scatter_(1, torch.where(ok, lin, torch.zeros_like(lin)), ok.float())
            return F.max_pool3d(m.view(-1, 1, N, N, N), 3, 1, 1)[:, 0]

        nq = max(1, max(len(o["q"]) for o in obs))
        with torch.no_grad():
            img = torch.from_numpy(np.stack([o["image"] for o in obs])).to(self.device)
            fr = torch.from_numpy(np.stack([o["fr"] for o in obs])).to(self.device)
            own = torch.from_numpy(np.stack([o["own"] for o in obs])).to(self.device)
            x = torch.stack([img, rasterise(fr), rasterise(own)], 1)
            sc = torch.from_numpy(np.stack([o["scal"] for o in obs])).to(self.device)
            qq = torch.zeros(len(obs), nq, 3, device=self.device)
            for i, o in enumerate(obs):
                if len(o["q"]):
                    qq[i, : len(o["q"])] = torch.from_numpy(o["q"]).to(self.device)
            o5, _, ev, idl = self.model(x, sc, qq)
            p5 = torch.softmax(o5.flatten(1).float(), 1)
            events = torch.softmax(ev.float(), 1).cpu().numpy()
            ident = torch.sigmoid(idl.float()).cpu().numpy()
            k0 = p5.argmax(1)
            nb = round(p5.shape[1] ** 0.5)
            iu, iv = k0 // nb, k0 % nb
            ar = torch.arange(nb, device=self.device)
            win = ((ar[None, :] - iu[:, None]).abs() <= 2).float()[:, :, None] * (
                (ar[None, :] - iv[:, None]).abs() <= 2
            ).float()[:, None, :]
            wgt = (p5.view(-1, nb, nb) * win).flatten(1)
            dl = (wgt[..., None] * self.dirs[None]).sum(1)
            dl = (dl / dl.norm(dim=1, keepdim=True).clamp(min=1e-9)).cpu().numpy()
        return dl, events, ident

    def _run(self, states: list[_State], frags: Fragments, queries):
        obs = list(self.pool.map(lambda a: self._observe(a[0], frags, a[1]),
                                 zip(states, queries, strict=True)))  # fmt: skip
        out = []
        for b in range(0, len(obs), self.batch):
            dl, ev, ident = self._forward(obs[b : b + self.batch])
            for i, o in enumerate(obs[b : b + self.batch]):
                pid = {c: float(ident[i, a:z].mean()) for c, a, z in o["qmap"]}
                out.append((unit(o["f"].T @ dl[i]), ev[i], pid, o["centre"]))
        return out

    # ------------------------------------------------------------------ public
    def trace(
        self, starts: list[Start], frags: Fragments, max_steps: int | None = None
    ) -> list[dict]:
        """Trace from every start until it stops.

        A trace stops when it links to another piece (P(own) >= accept), returns to its own piece,
        reaches a terminal, leaves the image or takes ``max_steps`` steps.
        """
        states = [_initial(s) for s in starts]
        for _ in range(self.max_steps if max_steps is None else max_steps):
            act = [s for s in states if s.alive]
            if not act:
                break
            results = self._run(act, frags, [None] * len(act))

            def step(a):
                return recentre(self.image, a[0].pos + STEP_UM * a[1][0], a[1][0])

            nexts = list(self.pool.map(step, zip(act, results, strict=True)))
            for st, (_, ev, pid, centre), pn in zip(act, results, nexts, strict=True):
                st.steps += 1
                st.visited.append(pn.copy())
                st.term_run = st.term_run + 1 if ev[2] > 0.8 else 0
                st.signal.append(round(centre, 2))
                st.dark_run = st.dark_run + 1 if centre < self.lost_snr else 0
                linked = hit_own = False
                for qpt in (0.5 * (st.pos + pn), pn):
                    dd, jj = frags.tree.query(qpt, k=12, distance_upper_bound=3.0)
                    for d_, j_ in zip(np.atleast_1d(dd), np.atleast_1d(jj), strict=True):
                        if not np.isfinite(d_):
                            continue
                        c = int(frags.piece[j_])
                        if c < 0:
                            continue
                        if c == st.start.piece:
                            hit_own = hit_own or st.steps >= 3
                            continue
                        if c in st.rejected:
                            continue
                        pr = pid.get(c, float("nan"))
                        if np.isfinite(pr) and pr >= self.accept:
                            st.status, st.link, st.p_link, linked = "linked", c, pr, True
                            break
                        st.rejected.add(c)
                    if linked or hit_own:
                        break
                if linked:
                    st.alive = False
                    st.trace.append(pn.copy())
                    continue
                if hit_own:
                    st.alive, st.status = False, "returned_own"
                    continue
                h = unit(pn - st.pos)
                st.hist = np.concatenate([h[None], st.hist[:-1]])
                st.path = np.concatenate([st.path, pn[None]])[-60:]
                st.s += float(np.linalg.norm(pn - st.pos))
                st.pos = pn
                st.trace.append(pn.copy())
                v = self.image.to_voxel(pn)
                if np.any(v < 0) or np.any(v >= self.image.shape):
                    st.alive, st.status = False, "oob"
                elif st.dark_run >= 2:
                    st.alive, st.status = False, "lost"
                    st.trace = st.trace[: max(1, len(st.trace) - 2)]  # drop the dark steps
                    self._restart(st, frags)
                elif st.term_run >= 2:
                    st.alive, st.status = False, "terminal"
                    self._restart(st, frags)
        return [
            dict(
                status=s.status,
                link=s.link,
                p_link=s.p_link,
                steps=s.steps,
                path_um=s.s,
                trace=np.asarray(s.trace),
                rejected=sorted(s.rejected),
                signal=np.asarray(s.signal),
                restarts=s.restarts,
                bridge_p=np.asarray(s.bridge_p),
                visited=np.asarray(s.visited).reshape(-1, 3),
            )
            for s in states
        ]

    def _identity(self, st: _State, frags: Fragments, points: NDArray) -> float:
        """P(same cell as the trace so far) of world points, seen from the trace's position."""
        obs = self._observe(st, frags, [])
        obs["q"] = ((points - st.pos) @ obs["f"].T).astype(np.float32)
        _, _, ident = self._forward([obs])
        return float(ident[0, : len(points)].mean())

    def _restart(self, st: _State, frags: Fragments) -> None:
        """Bridge a dim gap: resume from bright fibre ahead if the identity head calls it ours."""
        if st.restarts >= self.max_restarts or st.steps - st.last_bridge < self.bridge_every + 2:
            return
        base = st.trace[-1]
        h = unit(base - st.trace[-2]) if len(st.trace) >= 2 else st.hist[0]
        found = cone_ahead(self.image, base, h, self.restart_snr)
        if found is None:
            return
        q, ridge = found
        st.pos = base.copy()  # judge from the last bright step
        p_same = self._identity(st, frags, ridge)
        st.bridge_p.append(round(p_same, 3))
        if p_same < self.bridge_accept:
            return
        st.last_bridge = st.steps
        q = recentre(self.image, q, unit(q - base))
        st.restarts += 1
        st.alive, st.status, st.dark_run, st.term_run = True, "maxlen", 0, 0
        h = unit(q - base)
        st.hist = np.concatenate([h[None], st.hist[:-1]])
        st.path = np.concatenate([st.path, q[None]])[-60:]
        st.s += float(np.linalg.norm(q - st.pos))
        st.pos = q
        st.trace.append(q.copy())

    def score_arms(
        self, starts: list[Start], arms: list[list[int]], frags: Fragments
    ) -> list[dict]:
        """P(same cell) of each candidate arm (piece) at the end of each start's piece."""
        states = [_initial(s) for s in starts]
        return [r[2] for r in self._run(states, frags, arms)]  # (one observation each)
