"""Robo-LS v2 tracer network (fragment-aware, with a piece-identity head).

Copied from robo_ls/robo_model2.py (skeleton_proofreading, capsule 9440746) so the segmentation
pipeline can load the trained checkpoint; the architecture must stay identical to the checkpoint.
"""

# ruff: noqa
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

NB2 = 15
ANG2_DEG = torch.linspace(-84, 84, NB2)
NSCAL = 16
N2, VOX2 = 48, 1.5


def grid_dirs2():
    a = torch.deg2rad(ANG2_DEG)
    au, av = torch.meshgrid(a, a, indexing="ij")
    d = torch.stack([torch.ones_like(au), torch.tan(au), torch.tan(av)], -1)
    return d / d.norm(dim=-1, keepdim=True)  # (NB2, NB2, 3) local (a, u, v)


def cbr(cin, cout, k=3, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, stride, pad, bias=False), nn.GroupNorm(8, cout), nn.SiLU()
    )


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = (
            cbr(c, c),
            nn.Sequential(nn.Conv3d(c, c, 3, 1, 1, bias=False), nn.GroupNorm(8, c)),
        )

    def forward(self, x):
        return F.silu(x + self.b(self.a(x)))


class RoboNet2(nn.Module):
    def __init__(self, width=32, nscal=NSCAL, drop=0.15):
        super().__init__()
        w = width
        self.stem = nn.Sequential(cbr(3, w, 3, 2), Res(w))  # 24^3
        self.s2 = nn.Sequential(cbr(w, 2 * w, 3, 2), Res(2 * w))  # 12^3
        self.s3 = nn.Sequential(cbr(2 * w, 3 * w, 3, 2), Res(3 * w))  # 6^3
        self.s4 = nn.Sequential(cbr(3 * w, 4 * w, 3, 2), Res(4 * w))  # 3^3
        self.sc = nn.Sequential(nn.Linear(nscal, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.trunk = nn.Sequential(nn.Linear(27 * 4 * w + 64, 384), nn.SiLU(), nn.Dropout(drop))
        self.d5, self.d10, self.ev = (
            nn.Linear(384, NB2 * NB2),
            nn.Linear(384, NB2 * NB2),
            nn.Linear(384, 3),
        )
        cf = w + 2 * w + 3 * w
        self.idh = nn.Sequential(
            nn.Linear(cf + 384 + 3, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, s, q=None):
        """x (B,3,48,48,48); s (B,16); q (B,M,3) query points in the crop frame (um, a/u/v) or None.
        Returns d5 (B,NB2,NB2), d10, ev (B,3), id_logit (B,M) or None."""
        f1 = self.stem(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        h = self.trunk(torch.cat([f4.flatten(1), self.sc(s)], 1))
        out_id = None
        if q is not None:
            half = 0.5 * N2 * VOX2
            # grid_sample: grid (x, y, z) indexes (W, H, D) = (v, u, a) of the (a, u, v) crop; normalised to [-1, 1]
            g = torch.stack([q[..., 2], q[..., 1], q[..., 0]], -1) / half
            g = g.clamp(-1, 1)[:, :, None, None, :]  # (B,M,1,1,3)
            feats = [
                F.grid_sample(f, g.to(f.dtype), align_corners=False)
                .squeeze(-1)
                .squeeze(-1)
                .transpose(1, 2)
                for f in (f1, f2, f3)
            ]
            z = torch.cat(
                feats + [h[:, None, :].expand(-1, q.shape[1], -1), (q / half).to(h.dtype)], -1
            )
            out_id = self.idh(z).squeeze(-1)
        return self.d5(h).view(-1, NB2, NB2), self.d10(h).view(-1, NB2, NB2), self.ev(h), out_id
