# Compact adaptive degree-two scratch baseline

## Decision, 2026-10-06

The user requested the third representation arm and selected the **compact
adaptive** version previously used for IXI. The first two runs remain unchanged.
All three use the same 10,000 training, 1,952 validation and 13,071 test images,
original subject splits, seed, MRI preprocessing and 300-epoch scratch schedule.
No plants, pretrained weights, extra samples, or new exclusion filtering.

The third arm adds degree-two nodes to approximate each branch with straight
segments. It retains exactly the shared junction/termination/boundary endpoints
from the spline target caches. No skeleton re-extraction or node snapping is
performed. Original full generator guides remain uncertain references.

## Target construction

Reuse `optimal_keep_mask` and the prior adaptive physical-error profile:
`max(0.75 * local vessel radius, 0.1 mm)`, with a 95% mask-containment requirement
for shortcuts. Radius comes from the existing patch mask in physical units.
These SyntheticMRI guides are already smooth, so no additional smoothing is
applied. This derivation operates on the shared **cropped generator guides**;
it is not an exact rerun of IXI's full-volume skeleton extraction/smoothing.
That deliberate difference preserves the common endpoint topology and crop
boundary supervision of the current comparison.

A fixed 36-patch audit showed strict containment infeasible on three original
guides. Existing adjacent guide segments are therefore allowed even when mask
and guide disagree; all newly introduced shortcuts still require 95% containment.
These inherited violations are counted in the manifest. No guide is repaired,
branch deleted, or patch excluded because of mask disagreement. The revised
audit passed all 36 patches, added 240 degree-two nodes, found a maximum of 45
nodes and recorded five inherited segments below the containment threshold.

The optimal solver searches predecessor costs in ascending order, stopping at
the first feasible arc. This preserves its minimum-segment solution and historical
smallest-index tie break while reducing expensive chord checks. A targeted check
confirmed identical retained samples on straight/bending curves at two tolerances;
a separate check verified preservation of uncertain adjacent segments.
The new `preserve_adjacent` option defaults to false for all existing callers.

## Token budget and training

Keep 120 queries if **all training and validation targets fit**. If necessary,
raise the third arm to the smallest multiple of 32 covering their measured
maximum, with a safety guard at 512 queries. Test labels do not determine this
budget. Keep all test patches and report any test targets exceeding the budget.
Any increase is recorded explicitly and makes this an accuracy/capacity comparison,
not a comparison at equal token count. The two earlier 120-token arms are unchanged.
Same seed does not imply identical whole-model initialization if query dimensions
change. The immutable generated `adaptive/training-config.yaml` records the budget.

The architecture is the existing graph model with the edge existence head and
no spline head/loss. Other settings inherit the straight-edge scratch config:
batch 4; AdamW 7e-5, decay 1e-4; one warmup epoch; polynomial power 0.9;
no augmentation; validation every five epochs; shared five graph-loss weights
3/4/0.8/2/6. The inherited validation positive-edge cap is 40 per graph.
Best checkpoint selection uses the five graph losses, and final/latest is retained.
The same four validation examples show the degree-two graph predictions.

Native graph losses and node/edge metrics should not be compared directly with
junction-only targets, because there are different numbers of supervised nodes
and edges. Evaluate all arms against common endpoint/branch topology after
contracting degree-two chains, full-guide curve geometry, tortuosity and branch
coverage. Report compute and token budgets too.

## Execution and provenance

- `scripts/prepare_adaptive_comparison.py`: derive all targets on CPU, record
  node counts, inherited mask disagreements, shared manifest checksum and budget.
- `scripts/check_adaptive_comparison.py`: require exact split IDs, real batch,
  degree-two contraction agreement, capacity, model and loss construction.
- `cluster/jean_zay/prepare_adaptive_comparison.slurm`: full preparation and check.
- Training uses `COMPARISON_ARM=adaptive` and depends on successful preparation.
- Targets/config/provenance live in SCRATCH under
  `spline_learning/scratch_comparison_10k_300ep/adaptive`; MRI/masks are linked.
- Both jobs use the same verified package archive and protected interpreter.
  W&B compute logging is offline and must be synced from the login node.

Submission IDs will be recorded after submission. Full GT checks are still
pending until the preparation job completes; the GPU training cannot start first.

## Submissions

- CPU preparation and full preflight: **709919**.
- Compact adaptive scratch GPU training: **709924**, dependency `afterok:709919`.
- Stable implementation commit: `d0ccbef` (`representation`).
- Training parameters and split memberships match the first two arms; its actual
  query budget will be recorded after full preparation, before GPU training.
- Job provenance is saved alongside the original two IDs in
  `experiments/spline_learning/metadata/scratch-comparison-jobs.json`.

## Clarification about existing adaptive targets

The earlier IXI 300-epoch run (`ixi_q256_types_300ep_20260924`) already used
pre-extracted compact adaptive degree-two targets from
`IXI/vascular_patches/optimal_radius_0p75x_c095_vedo_original`. Those are IXI
images and cannot serve as targets for the current SyntheticMRI patch IDs.
The older SyntheticMRI 300-epoch run (`finetune_vessels_full_300ep`) used
`syntheticMRI/new_patches_boundary`, the native generator graphs with sparse
straight-edge targets; it did not use the radius-based adaptive representation.
An audit of the 91 available SyntheticMRI source graphs found 126 native
degree-two nodes, versus 181,034 terminal and 185,839 degree-three nodes.
Thus native degree-two nodes can exist, but these are not systematic adaptive
subdivision nodes used to capture branch geometry. The current CPU job reuses
existing SyntheticMRI guides and creates their compact adaptive targets.

## 2026-10-07: CPU preflight recovery

Target extraction completed successfully for all 25,023 patches in job 709919.
It produced 175,683 additional degree-two nodes; maximum nodes per patch is 67
(training/validation maximum 60). All targets fit the unchanged 120-query budget,
including test. There are 4,213 inherited guide segments below mask containment;
these were retained and counted under the documented uncertain-guide policy.

The job failed afterwards during `check_adaptive_comparison.py`: the cluster's
PyTorch CPU import attempted `cudaSetDevice` without a GPU. Its diagnostic
explicitly requires `IDR_DEBUG=WARN` for intended CPU-only execution. The
preparation script now sets that flag. Verification-only job **721618** reuses
the complete extracted targets; no graph regeneration is performed. Existing
training job **709924** now depends on `afterok:721618`. Verification outcome
and scheduler state will be recorded after the check completes.

Recovery verified: job **721618 COMPLETED (exit 0)**. Exact memberships, a real
validation batch, contraction back to shared endpoint topology, 120-query
capacity, model and criterion construction all passed on the CPU allocation.
The original adaptive training **709924 is RUNNING**, released automatically
by the corrected successful dependency. No targets were regenerated.

## Completed training and queued held-out evaluation

Training **709924 completed all 300 epochs**, exit 0, elapsed 13:51:03.
The validation-loss-selected checkpoint is epoch 25; the final checkpoint is
epoch 300. Test job **750145** evaluates both on the same 13,071 held-out IDs.
It saves native predictions/metrics and applies `shared-guides-v2` topology,
whole-vessel geometry and object-count scoring from saved predictions.
No test-set tuning or model/dataset changes are involved. Results will be
saved to `scratch_adaptive_degree2_10k_300ep/test/{best,final}` under the shared
SCRATCH run root, with W&B logging offline as for the other two arms.
