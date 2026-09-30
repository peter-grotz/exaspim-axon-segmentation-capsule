# exaSPIM TRAILMAP axon segmentation (Code Ocean capsule)

Thin Code Ocean wrapper around
[exaspim-axon-segmentation](https://github.com/peter-grotz/exaspim-axon-segmentation). The
package is vendored under `code/exaspim-axon-segmentation/` at the commit in its `SOURCE_COMMIT`;
do not edit it here — change the package, then re-run `scripts/sync_capsule.py`.

Machine: dedicated `g5.2xlarge` (NVIDIA A10G, 24 GB). Input data is read anonymously from
`s3://aind-open-data`; weights (Hugging Face `dfriedma/trailmap`, pinned revision) and the CCF
ontology are baked into the image at build time.

## Parameters

`code/run [ASSET] [REGIONS] [LEVEL] [CROP_UM] [TILE_SIZE] [BATCH_SIZE] [WEIGHTS] [REGION_GROWTH]`
— see the header of `code/run` for defaults. The default weights are the fine-tuned model asset
`52a5d51b-5ba7-4e5a-a923-65e4dd04c359`, which must be attached at mount `trailmap_finetune_v2`.
Example quick diagnostic: `… MD 2 600`.

The run segments the region grown by 20% of its volume, skeletonizes, removes debris, connects the
pieces, and writes one Horta-importable SWC per piece (see the package README).

## Outputs

| file | content |
|---|---|
| `<tag>_axon_probability.zarr` | uint8 axon probability (p × 255), OME-NGFF, on the fused level grid |
| `<tag>_skeleton/swc/<tag>_NNNN.swc` | one SWC per connected piece (µm, `# OFFSET` / `# COLOR` headers) |
| `<tag>_skeleton/skeleton.swc`, `skeleton.zarr`, `summary.json` | whole skeleton, voxels, lengths |
| `qc/<tag>_skeleton.png` | region (atlas + grown outline) with every piece |
| `qc/<tag>_region_overlay.png` | atlas region outline on the fused image |
| `run.log`, `pip_list.txt`, `nvidia_smi.txt`, `SOURCE_COMMIT` | provenance |
