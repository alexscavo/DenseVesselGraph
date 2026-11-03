# Connectivity informed by predicted vessel curves — 2026-10-08

## Objective and change

Recover connectivity and tortuous paths jointly. A frozen graph with curved
edges cannot repair missing or incorrect connections. The user authorized
implementing connectivity informed by image evidence along candidate curves.

The new optional path predicts a candidate eight-control spline **before edge
selection**, samples learned MRI features along that path, and adds a learned
correction to the original relation logits. Node positions, existence scores,
relation tokens, curve controls, and image features all remain trainable.
Both positive and negative candidate pairs receive edge classification loss.
Positive curves retain the existing fitted-target supervision.

### Implementation

- `models/curve_evidence.py`: a small full-resolution 3D MRI encoder (eight
  learned channels), raw MRI intensity, and an all-ones support channel.
  No target segmentation, centerline, or fitted controls enter this encoder.
- Differentiable trilinear sampling at 32 positions on each predicted spline.
  Coordinates follow voxel-index/axis-size DHW, with the required half-voxel
  adjustment for `grid_sample(align_corners=False)` and conversion to XYZ.
- A 1D convolution processes the sampled sequence; mean/max/min pooling and
  physical arc/chord lengths feed a two-logit residual scorer. Average both
  curve directions for undirected scoring. Initialize the final correction
  layer to zero, preserving original inference scores initially.
- One shared candidate-scoring function is used for matched training pairs,
  optional unmatched hard negatives, and inference. It uses canonical endpoint
  orientation and chunks pair computation at 256 candidates.
- Classification gradients pass through sampled coordinates into spline
  controls and predicted endpoints. Curves are not detached for edge scoring.
  Endpoint anchoring and existing relation/classification heads are retained.
- The feature tensor is carried through model predictions and never written
  into prediction JSON. Physical metric metadata comes from the existing crop
  affine convention; GT curve coordinates do not enter inference.

The default disabled path preserves existing models/checkpoint keys. The first
enabled implementation requires MRI input, Hungarian matching, and disabled
Betti losses; unsupported combinations fail validation rather than silently
scoring different graph heads. Existing masks/guides remain uncertain targets.

## Separate localization and connectivity errors

`scripts/diagnose_graph_connectivity.py` analyzes saved validation predictions:
role-agnostic, one-to-one raw-node matching in physical coordinates, then:

- Fraction of GT nodes recovered.
- Fraction of GT edges for which both endpoints are recovered.
- Connection precision/recall conditioned on those recovered endpoints.

Use 1 mm plus a diagnostic 2 mm sensitivity check. These diagnostics do not
replace the fixed shared-guides-v2 benchmark or tune its thresholds. They do
not contract predicted degree-2 chains and must not be reported as common
branch F1. A connection head cannot recover an endpoint removed by node
filtering; joint localization remains necessary.

## Paired pilot

Reuse the fixed 512 training / 128 validation patches and epoch-35 graph
checkpoint from the [staged pilot](spline-staged-learning.md). Both arms:

- Same architecture, initial tensors, sampling/batch order, 5,000 updates,
  fresh AdamW moments, original LR trajectory starting at update 87,500.
- Same spline weight ramp, 0 to 1.474 over 2,500 updates; all modules trainable.
- Same symmetric relation scoring in training and inference. This differs
  from the historical baseline's randomly oriented training pairs, so compare
  the new paired arms rather than attributing differences from old runs to
  image evidence alone.
- Same native/common validation metrics, same-endpoint chord comparison,
  node/link diagnostic, fixed projections, and offline W&B tracking.

Only the evidence contribution differs: **token_control scale 0** versus
**image_evidence scale 1**. Both contain the same initialized image module;
the control receives zero gradient through this module's scoring contribution.
This is a controlled pilot, not a claim that connectivity is already solved.
No full training or test evaluation is included in this pilot.

## Verification and files

Focused tests cover non-cubic voxel-center/axis alignment, fractional sampling,
zero-correction compatibility, undirected scoring, equality of training and
inference candidate scores, and edge-loss gradients into images, endpoints,
and spline controls. Existing spline and inference tests also run. A real
three-update GPU smoke exercises the complete model/loss/evaluation path.

Config: `configs/spline_curve_evidence_pilot.yaml`.
Slurm script: `cluster/jean_zay/spline_curve_evidence.slurm`.
Verification job: **778675**.
Outputs: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/curve_evidence_pilot_20261008/`.
Compact evidence: `experiments/spline_learning/metadata/curve-evidence/`.
Documentation is in this folder; code and generated run outputs retain their
separate scopes. Prior checkpoints, targets, and results are preserved.

## Execution and verification

All 13 focused CPU tests passed. GPU verification job **778675** completed
successfully in 1m43s, including those tests and three actual optimizer updates
with initial/final validation and rendering. These establish implementation
execution, not an improvement in prediction quality.

Paired 5,000-update jobs submitted as **779095_0** (token control) and
**779095_1** (image evidence). Submission metadata records source SHA256 hashes.
No training completion wait is required; compare both finished summaries later.

The preceding staged pilot's raw-node diagnostic recovered 11.86% / 9.72% of
nodes within 1 mm (graph control / staged spline), leaving only 20 / 12 of 2,059
reference edges with both endpoints recovered. Conditional connection recall
was 65% / 75%, but on these very small eligible subsets. At 2 mm, endpoint
recoverability rises to 179 / 141 edges. Localization is therefore a major
limitation under these reference tolerances; edge classification alone cannot
resolve it. These are uncertain graph-reference diagnostics, not new benchmark
scores or evidence that every reference node is anatomically exact.

### Pretraining decision

Pretraining followed by joint spline fine-tuning remains a sensible strategy:
learn useful image/node features first, then adapt connectivity and curved
geometry with the whole network trainable and gradual geometry supervision.
The earlier successful fine-tuning supports feasibility; the recent staged
pilot avoided collapse but did not improve geometry meaningfully. The new
paired experiment tests image-informed connectivity on the same pretrained
initialization. It does not establish the best pretraining objective or a
superiority over scratch training at equal total compute. Any later comparison
must account for pretraining data exposure and total optimization budget.
