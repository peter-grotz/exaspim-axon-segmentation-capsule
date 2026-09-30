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

### Full pipeline

```bash
uv run exaspim-axon run \
    s3://aind-open-data/exaSPIM_851474_2026-07-10_15-23-22_processed_2026-07-21_13-59-36 \
    --region MD --weights trailmap_finetune_v2/best.safetensors --out results/851474_MD
```

1. **Region.** The CCF region is located in the sample-space annotation and grown by 15% of its
   volume (`--region-growth`), a uniform shell (~40 µm for MD) that absorbs atlas-registration
   error.
2. **Segment.** The fine-tuned TRAILMAP U-Net predicts axon probability over the grown region on
   fused level 2 (4 × 2.99 × 2.99 µm). A GPU is expected for whole regions (MD: ~6 min on an A10G);
   interrupted runs resume.
3. **Skeletonize and remove debris.** Within the grown region: threshold p > 0.2 with a p > 0.5
   core, thin, prune spurs, drop single-plane speckle, bridge aligned gaps ≤ 30 µm, keep cable
   ≥ 100 µm, and drop pieces shorter than 500 µm lying more than 50 µm from longer cable.
4. **Connect.** Loose ends join the nearest aligned cable ≤ 50 µm away along the brightest path of
   the probability map; joins that detour or cross background are rejected.
5. **Export.** One SWC per connected piece, plus the QC figure.

Outputs (`<tag>` = `<subject>_<regions>_L<level>`):

| path | content |
|---|---|
| `<tag>_axon_probability.zarr` | uint8 axon probability (p × 255), OME-NGFF on the fused level grid |
| `<tag>_skeleton/swc/<tag>_NNNN.swc` | one SWC per connected piece, longest first |
| `<tag>_skeleton/skeleton.swc` | all pieces as one SWC forest |
| `<tag>_skeleton/skeleton.zarr` | skeleton voxels, OME-NGFF |
| `<tag>_skeleton/summary.json` | settings, lengths, and the per-piece file list |
| `qc/<tag>_skeleton.png` | the region (atlas and grown outlines) with every piece, and close-ups |

The SWCs follow the Janelia Workstation / HortaCloud export, so they import into Horta, SNT or
Vaa3D next to manual traces: `# ORIGINAL_SOURCE`, `# OFFSET x y z` and `# COLOR r,g,b` headers,
then `id type x y z radius parent` rows in micrometers relative to the offset, in the physical
frame of `fused.zarr` (voxel = x / 0.748, y / 0.748, z / 1.0 at level 0). Ids run 1..n with parents
before children; type is 0 (undefined) and radius 1.0, as in Horta exports. Each piece's `# COLOR`
matches its colour in the QC figure.

The steps also run separately: `predict`, `skeletonize` (with `--region-growth` to apply a margin
to an older prediction), `qc-skeleton`.

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
