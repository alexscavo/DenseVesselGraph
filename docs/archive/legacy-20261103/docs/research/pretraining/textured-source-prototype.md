# Simple textured-background plant prototype

Date: 2026-10-16. Appearance prototype only; no dataset or training settings changed.

## Input provenance

Verified `Image-to-Graph-Vessel/3d/data/generate_plants_data.py`, lines 311–325: the saved raw input is generated from the segmentation with four 3×3 all-ones convolutions, saturation at 1 and conversion to 8-bit intensity. It is not the original RGB photograph. `PlantsDataset` subsequently resizes it, embeds it into a five-slice slab, scales to [-0.5,0.5], and pads the volume.

The generated input is mask-like but not strictly binary after resizing/warping: the two planar prototypes have 29 and 30 distinct intensities. Therefore use `alpha=clip(raw+0.5,0,1)` as the existing soft foreground coverage rather than discarding its interpolation profile or replacing it with the narrower segmentation. The segmentation and graph annotations remain unchanged.

Reuse both samples from the nonplanar prototype, each in planar and nonplanar form. This yields four cases with three appearance variants each. No new geometries are sampled. Appearance is shown before rotation to make the original slab orientation easy to inspect; a later training integration must rotate the completed image and annotations together.

## Minimal construction

Seed 364506. In normalized [0,1] intensity units:

- Generate independent Gaussian fields: standardized slow variation (Gaussian smoothing sigma 8 voxels), standardized correlated texture (sigma 1.2 voxels), and unsmoothed unit noise.
- Background B = clip(0.22 + 0.045 slow + 0.035 texture + 0.015 noise, 0, 1).
- Foreground F = clip(0.72 + 0.04 slow + 0.025 texture + 0.015 noise, 0, 1).
- Textured variant: I = (1-alpha) B + alpha F.
- Softened variant: same B and F, but smooth alpha in 3D with Gaussian sigma 0.65 voxels before compositing.

The same texture realization is used for the two textured variants within each case. Textures differ between planar/nonplanar cases; use within-case columns for controlled appearance comparisons. Every zero-alpha voxel in the unsoftened variant receives exactly B, including background pixels within the slab and original padding. There is no slice-sized pasted background rectangle. Texture is also present in foreground intensities rather than leaving saturated flat white branches.

These constants are illustrative, not fitted to MRI or claimed as a physical MRI noise model. No real MRI image is pasted into the background, avoiding accidental unlabeled vessels. Background textures can still create distracting bright regions. The Gaussian texture lacks real anatomical tissue structure.

## Verification and observations

Four cases completed with finite intensities in [0,1]; the exact zero-alpha replacement assertion passed. Nodes, edges, masks and dense guides are copied unchanged into NPZ outputs. The original foreground already extends beyond the segmentation due to generation/resizing; this discrepancy is not repaired. Boundary softening further alters apparent thickness and contrast, without changing annotations.

Inspected PNGs for sample_000001 nonplanar and sample_000002 planar: background is continuous through and outside the embedded structure; foreground remains visible; soft boundaries remove abrupt extrusion edges. The planar geometry still looks like an extruded sheet. This appearance change does not solve that geometric limitation and provides no evidence of improved transfer yet.

Across the four cases, mean intensity inside the mask is 0.68099–0.72669 for the textured variant and 0.64235–0.68360 for the softened version. Outside-mask mean is approximately 0.224 (includes generator foreground halos). Full values: `experiments/source_augmentation/texture-report.json`.

## Artifacts and reproduction

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/source_texture_20261016/`.

- `textured-plants.html`: standalone, offline slice viewer (~5.4 MB); download just this file. Select sample/geometry, move three orthogonal slice sliders, optionally overlay mask outlines and nearby nodes. Shared [0,1] display limits across columns.
- Four PNGs: orthogonal slices and a maximum intensity projection.
- Four NPZs: three input variants, background, mask, nodes, paths and edges.
- `report.json`: measured intensity summaries.

```bash
source cluster/jean_zay/env.sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/source_augmentation/prototype_textured_background.py --output /path/to/new/output
```

The command requires a new output folder, preserving previous prototypes. Python syntax and numerical checks passed; PNGs were visually inspected. The HTML was not exercised in a browser on the cluster.

## Requested SyntheticMRI + IXI pretraining: already completed

Checked the existing completion markers and held-out test summaries rather than submitting duplicate jobs. Configuration: `configs/ixi_transfer.yaml`; results under `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/`.

Mixed pretraining starts from scratch: 25,000 SyntheticMRI source patches + 10,000 IXI target patches, 50 epochs, 50,000 replacement draws/epoch, balanced expected source/target sampling (target weight 2.5), batch 32, domain loss weight 1.0. Fine-tuning: 100 epochs on the same IXI training population, domain adaptation disabled. All arms share IXI validation/test membership; test has 2,178 patches. Best checkpoint is selected on validation, not test. Geometry uses K8 residual splines with detached geometry gradients, not fully coupled geometry optimization. These runs use the original source rendering, not this prototype.

| Pretraining source | Representation | Best fine-tuning epoch | Test node F1 | Test edge F1 | Test curve F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| IXI only | Straight | 100 | 0.47404 | 0.15985 | 0.11644 |
| Plants + IXI | Straight | 80 | 0.44480 | 0.12084 | 0.09022 |
| SyntheticMRI + IXI | Straight | 95 | 0.44676 | 0.11688 | 0.08831 |
| IXI only | Spline throughout | 100 | 0.47369 | 0.16395 | 0.12215 |
| Plants + IXI | Spline throughout | 100 | 0.45038 | 0.12090 | 0.09441 |
| SyntheticMRI + IXI | Spline throughout | 100 | 0.44171 | 0.11126 | 0.09214 |
| SyntheticMRI + IXI | Spline only during fine-tuning | 85 | 0.44243 | 0.11373 | 0.08963 |

All finished the 100-epoch fine-tuning budget; best epoch can be earlier. Test summaries are `<arm>/held-out-test/summary.json`; relevant arms are `synthetic_straight`, `synthetic_spline`, `synthetic_delayed_spline`. In this single-seed setup synthetic vessels did not outperform plants or the IXI-only baseline on edge/curve F1. This does not prove synthetic pretraining cannot help under other settings. The IXI-only control matches update count, not IXI exposure: all its draws are IXI. Existing leaf comparisons target SyntheticMRI, so they cannot be inserted into this IXI table as a matched comparison.
