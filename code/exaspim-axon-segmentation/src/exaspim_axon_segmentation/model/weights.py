"""Named, pinned pretrained weights."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from exaspim_axon_segmentation.model.convert import load_keras_weights
from exaspim_axon_segmentation.model.unet import TrailmapUNet3D


@dataclass(frozen=True)
class PretrainedWeights:
    """A Keras TRAILMAP checkpoint pinned to an exact Hugging Face revision and checksum."""

    name: str
    repo_id: str
    filename: str
    revision: str
    sha256: str
    description: str

    def fetch(self, cache_dir: Path | None = None) -> Path:
        """Download (or reuse the cached copy of) the checkpoint and verify its checksum."""
        path = Path(
            hf_hub_download(
                self.repo_id, self.filename, revision=self.revision, cache_dir=cache_dir
            )
        )
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != self.sha256:
            raise ValueError(f"{path}: sha256 {digest.hexdigest()} != expected {self.sha256}")
        return path

    def load(self, cache_dir: Path | None = None) -> TrailmapUNet3D:
        """Fetch and convert into an eval-mode :class:`TrailmapUNet3D`."""
        return load_keras_weights(self.fetch(cache_dir))


_HF_REPO = "dfriedma/trailmap"
_HF_REVISION = "f8c8ed37a3dfb81260a1d026ef0c7bcc5f1e05ff"

REGISTRY: dict[str, PretrainedWeights] = {
    w.name: w
    for w in (
        PretrainedWeights(
            name="trailmap-axon-210409",
            repo_id=_HF_REPO,
            filename="AXON_model_210409.hdf5",
            revision=_HF_REVISION,
            sha256="8d181a8ada8348ce57c9efd5672ee3911da610566b8a200bfcc6cac363a1dc59",
            description="TRAILMAP axon model (released 2021-04-09)",
        ),
        PretrainedWeights(
            name="trailmap-fos-210815",
            repo_id=_HF_REPO,
            filename="FOS_model_210815.hdf5",
            revision=_HF_REVISION,
            sha256="f62161127b7a8ff690b646dceb94061ca5e55ad3be749f25126ba591af7eae7f",
            description="TRAILMAP-architecture nuclear (FOS/H2B) model (2021-08-15)",
        ),
    )
}


def pretrained(name: str = "trailmap-axon-210409", cache_dir: Path | None = None) -> TrailmapUNet3D:
    """Load a registered pretrained model by name."""
    try:
        weights = REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown weights {name!r}; available: {sorted(REGISTRY)}") from None
    return weights.load(cache_dir)
