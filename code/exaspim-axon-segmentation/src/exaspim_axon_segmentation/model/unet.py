"""PyTorch port of the TRAILMAP 3D U-Net (Friedmann et al., PNAS 2020).

The layer layout, op order, and numerics follow the released Keras model exactly so the published
weights load without retraining:

* every convolution is followed by ReLU and *then* BatchNorm (eps = 1e-3, the Keras default);
* the encoder uses ``same`` padding, the decoder ``valid`` padding, and the two upper skip
  connections are center-cropped (by 4 and 12 voxels) to match;
* the decoder concatenates ``[upsampled, skip]`` in that order.

The network is fully convolutional: any cubic or non-cubic input whose sides are multiples of 8
(and at least 64) yields an output 28 voxels smaller per side (14 of context on each face).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

CONTEXT = 14
"""Voxels of context consumed on each face: output side = input side - 2 * CONTEXT."""

DIVISOR = 8
"""Input sides must be multiples of this (three 2× poolings)."""

MIN_INPUT = 64
"""Smallest input side for which every decoder stage has positive size."""

BN_EPS = 1e-3


class ConvReluBn(nn.Sequential):
    """3×3×3 convolution → ReLU → BatchNorm, the unit block of the TRAILMAP network."""

    def __init__(self, in_channels: int, out_channels: int, *, padding: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels, eps=BN_EPS),
        )


class DoubleConv(nn.Sequential):
    """Two :class:`ConvReluBn` blocks."""

    def __init__(self, in_channels: int, mid: int, out_channels: int, *, padding: int) -> None:
        super().__init__(
            ConvReluBn(in_channels, mid, padding=padding),
            ConvReluBn(mid, out_channels, padding=padding),
        )


class Up(nn.Module):
    """2× transposed-convolution upsampling (with ReLU), skip crop + concat, then two convs."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, crop: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.crop = crop
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, out_channels, padding=0)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample ``x``, concatenate the center-cropped ``skip``, and convolve."""
        x = torch.relu(self.up(x))
        if self.crop:
            c = self.crop
            skip = skip[..., c:-c, c:-c, c:-c]
        return self.conv(torch.cat([x, skip], dim=1))


class TrailmapUNet3D(nn.Module):
    """TRAILMAP 3D U-Net: single-channel volume in, axon probability out.

    Input ``(N, 1, D, H, W)`` with intensities scaled to ``[0, 1]``; output
    ``(N, 1, D - 28, H - 28, W - 28)`` of sigmoid probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = DoubleConv(1, 32, 64, padding=1)
        self.enc2 = DoubleConv(64, 64, 128, padding=1)
        self.enc3 = DoubleConv(128, 128, 256, padding=1)
        self.bottleneck = DoubleConv(256, 256, 512, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.up3 = Up(512, 256, 256, crop=0)
        self.up2 = Up(256, 128, 128, crop=4)
        self.up1 = Up(128, 64, 64, crop=12)
        self.head = nn.Conv3d(64, 1, kernel_size=1)

    @staticmethod
    def output_shape(input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Spatial output shape for a spatial input shape (validates divisibility)."""
        for side in input_shape:
            if side % DIVISOR or side < MIN_INPUT:
                raise ValueError(
                    f"Input sides must be multiples of {DIVISOR} and >= {MIN_INPUT}, got "
                    f"{input_shape}"
                )
        return tuple(side - 2 * CONTEXT for side in input_shape)

    def logits(self, x: Tensor) -> Tensor:
        """Pre-sigmoid output; use for training with ``BCEWithLogits``-style losses."""
        self.output_shape(tuple(x.shape[2:]))
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)

    def forward(self, x: Tensor) -> Tensor:
        """Axon probability for each output voxel."""
        return torch.sigmoid(self.logits(x))
