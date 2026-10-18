# Plant/leaf augmentation integration

Implemented 2026-10-16; completed verification and status review 2026-10-17.

## Status

Available as an opt-in training policy in the shared `PlantsDataset`, which also loads the leaf and combined biological datasets. Implementation verification initially involved no training. On 2026-10-18 the matched SyntheticMRI training chains were submitted; see [the launch record](augmented-source-comparison.md). Visual plausibility and geometric consistency have been checked; a learning/generalization improvement has not been demonstrated.

## Policy

1. Load the generator-produced mask-like raw image, preserving its soft foreground profile. Embed and pad using existing conventions.
2. Sample independent node depth shifts (Gaussian sigma 5 voxels), subtract their mean, fit a smooth Gaussian RBF field (length 10 voxels, ridge 0.05). Limit maximum depth shift to 10 voxels, slope to 1, and available volume clearance. One field warps the image, mask, nodes and branch paths. Raw uses linear interpolation; mask uses nearest neighbour. No topology changes or node removal.
3. For continuous supervision, densify the original stored annotation polylines to 257 samples before warping; refit endpoint-constrained cubic B-splines with the configured control count (K8 in these plans). Resample guides along the deformed paths. Do not directly warp old control positions. Distances use the existing virtual voxel spacing; it is not measured plant anatomy.
4. Render correlated background throughout the padded cube. Reuse prototype slow-field sigma8 and correlated-field sigma1.2; background mean varies uniformly 0.18–0.26, foreground mean 0.62–0.82, noise strength 0.75–1.25 times the prototype, boundary smoothing sigma0.5–0.8 voxels. Texture and parameters are drawn afresh per retrieval. These ranges are mild extensions of the preferred visual prototype, not calibrated MRI statistics.
5. Apply existing quarter-turns/flips consistently to images, nodes and targets. Update the geometry metric for the rigid transform. Do not min/max normalize or add empty padding after rendering, because that would undo the intended background/intensity policy.

Intensity remains [-0.5,0.5]. Masks remain binary labels. Foreground softening changes apparent thickness, not labels. Geometry is still a warped sheet rather than arbitrary volumetric anatomy. The policy is applied only when training augmentation and its own enabled flag are true. Validation/test and old configurations follow the previous path. NumPy receives a per-item seed from the existing worker-seeded Python RNG; equal seed and retrieval order reproduce outputs.

## Configuration and files

Plan option (copied by the training configuration builder to `augmentation.plants.source_augmentation`):

```yaml
source_augmentation:
  enabled: true
  nonplanar: true
  textured_soft: true
```

The last two flags can be disabled independently for controlled ablations. Separate configurations preserve the old plans and use distinct output roots/tracking groups:

- `configs/plant_generalization_da_b32_source_aug.yaml`
- `configs/leaf_generalization_da_b32_source_aug.yaml`
- `configs/combined_generalization_da_b32_source_aug.yaml`
- `configs/ixi_transfer_source_aug.yaml`

The source-transfer stage policy is unchanged: the SyntheticMRI plans pretrain graph-only and introduce splines in fine-tuning; the IXI runner supports both-stage or delayed spline variants. Enabling augmentation does not silently change that scientific choice, sampling, losses, batch size or epochs. MRI-only arms are unaffected because they have no biological source loader.

Implementation: `data/source_augmentation.py`, `data/loaders/plants.py`, `data/loaders/mixed.py`, `scripts/train_plant_generalization.py`. The IXI runner reuses that configuration builder.

## Checks

- All four configuration builders validate with augmentation enabled, including IXI K8 pretraining with the recorded virtual spacing 0.5601755961.
- Four focused tests passed: new source-warp/refit test and existing spline projection/rotation tests.
- Actual loader checked on two plant and two leaf training samples, each with K8 targets. Endpoints match nodes; edges unchanged; arrays finite and nodes inside the cube. Equal seed gives identical input. Validation with the option enabled matches old validation output exactly, including geometry.
- Four samples collate through the existing training collator; graph-only source-loading route also passed.
- Mean refit error: 0.00930/0.00698 voxels for plants; 0.02508/0.02259 for leaves. Maximum sampled error: 0.201/0.151/0.491/0.445 voxels. These compare arc-resampled fitted curves with warped guides; they are not learned-model errors.
- Guide-mask support: plants 98.96→98.18% and 93.08→93.08%; leaves 84.31→84.44% and 87.32→87.32%. Existing leaf annotation/mask discrepancies remain. Small differences reflect warp voxelization and guide resampling.
- Observed per-item loader time 0.126–0.360 seconds in the four-sample CPU check; not a full DataLoader throughput or GPU utilization benchmark.
- Plant and leaf preview PNGs visually inspected. Different projected crossings in a MIP do not imply changed 3D connectivity.

Numeric record: `experiments/source_augmentation/training-loader-check.json`.
Previews: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/source_augmentation_loader_20261016/`.
Reproduction script: `experiments/source_augmentation/check_training_augmentation.py --output /path/to/new/folder` in the fixed project environment.

## Scheduler/results review — 2026-10-17

Scheduler queue empty at review. Previously cancelled/failed job entries are historical and have successful replacement chains (see the disk-recovery and overfit reports); no pending dependency chain found. Source-transfer, IXI full-100-epoch continuations and their held-out evaluations are complete. No new augmentation-training results exist.

The previous SyntheticMRI comparison remains led by MRI-only spline: edge F1 0.7258 and curve F1 0.5097; plants spline 0.6879/0.4581; leaves spline 0.6905/0.4607; combined spline 0.6809/0.4484. IXI spline throughout: IXI-only 0.16395/0.12215; plants+IXI 0.12090/0.09441; SyntheticMRI+IXI 0.11126/0.09214. Single seed, different target datasets: do not compare the numeric scales across tables as a controlled dataset effect.

## Next decision

Run a short model forward/backward smoke with the new resolved configuration before substantial training, then compare augmented versus existing source rendering under identical membership, budgets and stage policy. Existing training comparisons are the baseline, not evidence that the new augmentation improves transfer. No long runs were launched during implementation.

## Follow-up submission — 2026-10-18

[Three source pretrainings, six matched fine-tunings and dependent held-out tests](augmented-source-comparison.md) now queued behind running GPU smoke gates. This supersedes the earlier no-training status; results are still pending.
