# exaspim-axon-segmentation

![support](https://img.shields.io/badge/support-unsupported-red)

Axon segmentation and skeletonization for exaSPIM light-sheet data, restricted to CCF regions, using
a PyTorch port of the pretrained [TRAILMAP](https://github.com/albert597/TRAILMAP) 3D U-Net
(Friedmann et al., *PNAS* 2020; weights: [dfriedma/trailmap](https://huggingface.co/dfriedma/trailmap)).

Status: early development.

![exaSPIM image and TRAILMAP axon probability](docs/images/709220_image_vs_probability.png)
*709220, a 40 µm slab of the fused exaSPIM image (level 2, 3 µm voxels) and the fine-tuned
network's axon probability: fibres are picked up, bright puncta and debris are not.*

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

1. **Region.** The CCF region is located in the sample-space annotation and grown by 20% of its
   volume (`--region-growth`), a uniform shell (~58 µm for MD) that absorbs atlas-registration
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

## Results

**Following axons out of a region (709220).** Starting from MD, `expand` segments 800 µm cubes
ahead of every open axon end, round after round; neuron N006 (red, 11.4 mm) was followed back to
within 105 µm of its soma. Top: the predictions (MD, the cubes, N006 in red, other followed axons
in blue, predicted but not followed in grey); bottom: the same on the sample-space image, and the
soma.

![Axon followed to its soma](docs/images/709220_N006_followed_axon.png)

## Fine-tuning

The network is the TRAILMAP 3D U-Net with its pretrained weights as the starting point, fine-tuned
on exaSPIM data at fused level 2 (4 × 2.99 × 2.99 µm, the grid every prediction runs on). The
released weights are `trailmap_finetune_v2/best.safetensors`.

**Training data.** Volumes where every axon is traced, so anything bright that is not near a trace
is a genuine negative:

- 822178 MD, a traced lobe (two neurons, ~32 mm of axon). Its tracing is complete only near the
  traced cells, so voxels farther than 50 µm from a trace are ignored; the last 20% of the lobe
  along x is held out for validation.
- 17 completely traced 256³ blocks from four other brains (754611, 754612, 730223, 721828).
- Brain 744783 (3 blocks) is held out entirely, as an unseen-brain test.

**Labels**, from the SWC traces on the level-2 grid:

| class | voxels | loss weight |
|---|---|---|
| axon | within 2 µm of a traced centreline | 1.5 |
| edge | 2–5 µm from a centreline (uncertain axon boundary) | 0.05 |
| hard negative | bright (above background + half the axon contrast, after 1.5 µm smoothing) and ≥ 8 µm from any trace: dendrites, somata, debris | 0.8 |
| background | everything else in the traced region | 0.2 |
| ignore | outside the traced region or envelope (context only) | 0 |

The weights are TRAILMAP's own class weights; the loss is weighted binary cross-entropy,
normalized by the summed weight.

**Regime.**

- 96³ input patches, 68³ scored output (the U-Net's 14-voxel context on each side), batch 4.
- Patch centres: 50% on axon, 25% on hard negatives, 25% anywhere in the labelled region; the 822178
  lobe is drawn half the time, the blocks the other half.
- Augmentation: y/x flips, 90° rotations in-plane, and ±35% multiplicative intensity jitter. Z is
  never swapped with y/x because the voxels are anisotropic.
- Intensities are normalized per volume the same way as at inference: the background median maps
  to 0 and the median foreground voxel to 0.25.
- AdamW, learning rate 1e-4 with cosine decay, weight decay 1e-5; BatchNorm statistics frozen
  (small batches); bf16 autocast on GPU.
- 4,000 steps (≤ 40 min on an A10G); validation every 250 steps on 64 axon-enriched patches per
  group; the checkpoint with the best mean F1 is kept (step 3,000).

**Validation** (voxels at p > 0.5 on held-out patches):

| | precision | recall | F1 | background called axon |
|---|---|---|---|---|
| 822178 MD, held-out slab · pretrained | 0.03 | 0.34 | 0.05 | 3.0% |
| 822178 MD, held-out slab · fine-tuned | 0.66 | 0.86 | 0.75 | 0.10% |
| brain 744783 (unseen) · pretrained | 0.01 | 0.55 | 0.02 | 5.5% |
| brain 744783 (unseen) · fine-tuned | 0.38 | 0.62 | 0.47 | 0.07% |

Known limitation: dendrites of labelled neurons are still often called axon (on 822178, 64% of
hard-negative voxels are above 0.5), so the skeletonizer separates soma-connected cable afterwards.

Train with the capsule's `finetune` mode (`capsule/code/run finetune [STEPS] [MAX_MINUTES]
[HOLDOUT_BRAIN] [ENVELOPE_UM]`).

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
