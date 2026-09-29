# exaSPIM TRAILMAP axon segmentation (Code Ocean capsule)

Thin Code Ocean wrapper around
[exaspim-axon-segmentation](https://github.com/peter-grotz/exaspim-axon-segmentation). The
package is vendored under `code/exaspim-axon-segmentation/` at the commit in its `SOURCE_COMMIT`;
do not edit it here — change the package, then re-run `scripts/sync_capsule.py`.

Machine: dedicated `g5.2xlarge` (NVIDIA A10G, 24 GB). Input data is read anonymously from
`s3://aind-open-data`; weights (Hugging Face `dfriedma/trailmap`, pinned revision) and the CCF
ontology are baked into the image at build time.

## Parameters

`code/run [ASSET] [REGIONS] [LEVEL] [CROP_UM] [TILE_SIZE] [BATCH_SIZE]` — see the header of
`code/run` for defaults. Example quick diagnostic: `… MD 2 600`.

## Outputs

| file | content |
|---|---|
| `<subject>_<regions>_L<level>_axon_probability.zarr` | uint8 axon probability (p × 255), OME-NGFF, on the fused level grid |
| `qc/*_region_overlay.png` | region outline on the fused image |
| `qc/*_prediction.png` | raw vs probability MIPs |
| `predict.log`, `pip_list.txt`, `nvidia_smi.txt`, `SOURCE_COMMIT` | provenance |
