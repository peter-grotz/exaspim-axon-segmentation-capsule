"""Convert released TRAILMAP Keras HDF5 weights into a :class:`TrailmapUNet3D` state dict.

Only ``h5py`` is needed — TensorFlow is never imported. Keras stores convolution kernels
channels-last as ``(kD, kH, kW, in, out)`` and transposed-convolution kernels as
``(kD, kH, kW, out, in)``; PyTorch wants ``(out, in, kD, kH, kW)`` and ``(in, out, kD, kH, kW)``
respectively, so both are the same axis permutation.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from exaspim_axon_segmentation.model.unet import TrailmapUNet3D

# Keras auto-generated layer names in creation order, paired with the PyTorch module prefix.
_CONV_BN_BLOCKS = [
    "enc1.0", "enc1.1", "enc2.0", "enc2.1", "enc3.0", "enc3.1", "bottleneck.0", "bottleneck.1",
    "up3.conv.0", "up3.conv.1", "up2.conv.0", "up2.conv.1", "up1.conv.0", "up1.conv.1",
]  # fmt: skip
_TRANSPOSED = ["up3.up", "up2.up", "up1.up"]


def _suffix(i: int) -> str:
    return "" if i == 0 else f"_{i}"


def keras_layer_map() -> dict[str, str]:
    """Keras layer name → PyTorch module prefix for every weighted layer."""
    mapping = {}
    for i, block in enumerate(_CONV_BN_BLOCKS):
        mapping[f"conv3d{_suffix(i)}"] = f"{block}.0"
        mapping[f"batch_normalization{_suffix(i)}"] = f"{block}.2"
    for i, module in enumerate(_TRANSPOSED):
        mapping[f"conv3d_transpose{_suffix(i)}"] = module
    mapping["conv3d_14"] = "head"
    return mapping


def _keras_to_torch(keras_param: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    name = {
        "kernel": "weight", "bias": "bias", "gamma": "weight", "beta": "bias",
        "moving_mean": "running_mean", "moving_variance": "running_var",
    }[keras_param]  # fmt: skip
    if keras_param == "kernel":
        value = np.transpose(value, (4, 3, 0, 1, 2))
    return name, value


def state_dict_from_keras(path: Path) -> dict[str, torch.Tensor]:
    """Read a TRAILMAP Keras ``.hdf5`` file into a PyTorch state dict.

    Parameters
    ----------
    path : Path
        Keras model or weights file (``model_weights`` group is used when present).

    Returns
    -------
    dict of str to torch.Tensor
        Complete state dict for :class:`TrailmapUNet3D` (strictly validated against it).
    """
    layer_map = keras_layer_map()
    state: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as f:
        weights = f.get("model_weights", f)
        for layer, prefix in layer_map.items():
            group = weights[layer][layer]
            for key in group:
                name, value = _keras_to_torch(key.split(":")[0], group[key][()])
                state[f"{prefix}.{name}"] = torch.from_numpy(np.ascontiguousarray(value))
    _validate(state)
    for key in TrailmapUNet3D().state_dict():
        if key.endswith("num_batches_tracked"):
            state[key] = torch.tensor(0, dtype=torch.long)
    return state


def _validate(state: dict[str, torch.Tensor]) -> None:
    reference = TrailmapUNet3D().state_dict()
    expected = {k for k in reference if not k.endswith("num_batches_tracked")}
    if missing := expected - state.keys():
        raise ValueError(f"Keras file lacks weights for {sorted(missing)}")
    if extra := state.keys() - expected:
        raise ValueError(f"Unmapped weights {sorted(extra)}")
    for key, value in state.items():
        if value.shape != reference[key].shape:
            raise ValueError(f"{key}: shape {tuple(value.shape)} != {tuple(reference[key].shape)}")


def load_keras_weights(path: Path) -> TrailmapUNet3D:
    """Build a :class:`TrailmapUNet3D` in eval mode with weights from a Keras file."""
    model = TrailmapUNet3D()
    model.load_state_dict(state_dict_from_keras(path))
    return model.eval()
