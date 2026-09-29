# exaspim-axon-segmentation

![support](https://img.shields.io/badge/support-unsupported-red)

Axon segmentation and skeletonization for exaSPIM light-sheet data, restricted to CCF regions, using
a PyTorch port of the pretrained [TRAILMAP](https://github.com/albert597/TRAILMAP) 3D U-Net
(Friedmann et al., *PNAS* 2020; weights: [dfriedma/trailmap](https://huggingface.co/dfriedma/trailmap)).

Status: early development. See [PLAN.md](PLAN.md) for the roadmap.

## Install

```bash
uv sync
```

## Usage

Check that a sample-space CCF region lines up with the fused image before running anything on it:

```bash
uv run exaspim-axon qc-region \
    s3://aind-open-data/exaSPIM_851474_2026-07-10_15-23-22_processed_2026-07-21_13-59-36 \
    --region MD --out qc/851474_MD_overlay.png
```

`--region` takes a CCF structure ID or acronym and can be repeated. Descendant structures are
included automatically. Public `aind-open-data` assets are read anonymously.

## Data expectations

A processed exaSPIM asset with:

| path | content |
|---|---|
| `fusion/fused.zarr` | fused image, OME-NGFF v0.4 multiscale |
| `ccf_alignment/ccf_anno_to_sample/ccf_anno_in_sample_space.zarr` | CCF annotation in sample space |

The two are on different voxel grids (the annotation is offset by half a voxel from the matching
fused level), so every mapping between them goes through physical coordinates.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## License

MIT
