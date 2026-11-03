# Local image information for geometry — 2026-10-10

## Why this experiment

The user authorized a short geometry-specific local-feature comparison and
then pointed out that SyntheticMRI contains mostly small bends. This is a
material limitation: predictions close to chords can reflect the data's
geometry, not necessarily missing image information or a failed representation.
Keep the successful frozen SyntheticMRI detector for an interpretable matched
comparison, but stratify geometry evaluation by target tortuosity. IXI is the
next useful strongly curved-data study; its existing one-case oracle confirms
geometric capacity, but is not a population training benchmark.

The current geometry MLP receives object/relation tokens and predicted endpoint
geometry. These tokens summarize image information and may omit local details
along a vessel. The new question is whether early CNN features sampled on the
predicted path provide useful additional shape information. This is different
from the earlier [curve-informed connectivity](curve-informed-connectivity.md)
experiment, which sampled an added image encoder to change edge scores.
Here connectivity and detector state are fixed; only curve geometry learns.

## Dataset curvature check and corrected interpretation

Quick audit of the existing **128 SyntheticMRI validation patches**, using
cached eight-control fitted target curves and retaining branches >=5 mm:

- 1,509 eligible branches; median fitted tortuosity **1.0159**.
- 90th/95th/99th percentiles **1.0576 / 1.0821 / 1.1584**.
- **12.8%** exceed tortuosity 1.05; **2.5%** exceed 1.10.
- 567 long branches have maximum fitted-curve deviation from the chord >1 mm;
  low length/chord ratio does not imply a globally straight long branch.

In the existing illustrative single cases, 6/12 IXI branches exceed extracted
guide tortuosity 1.05, with maximum **1.8745**; 0/19 SyntheticMRI branches do,
with maximum **1.0308**. These cases were selected before fitting but are not
representative dataset samples. Guides/fits remain uncertain references.
Do not use these figures to claim a population difference without a broader
IXI audit, or use the small SyntheticMRI average gains alone to diagnose failure.

The available IXI checkpoint has reported validation Branch F1 around 0.0068
under its existing adaptive-graph protocol. A direct switch to that detector
would confound geometry with its detection/connectivity quality and a different
representation. No IXI model, dataset or target cache was changed in this task.
The strongly curved follow-up needs a separately specified IXI geometry study
and reliable endpoints/connectivity (or an explicitly labeled geometry oracle).

## Three matched arms

| Arm / task | Existing geometry MLP | New local correction | New CNN values |
|---|---|---|---|
| `token_control` / 871744_0 | Learns | Multiplied by zero | Sampled but inactive |
| `local_features` / 871744_1 | Learns | Learns | Actual local feature map |
| `zero_features` / 871744_2 | Learns | Learns | Zeroed before sampling |

All instantiate identical learned tensors/parameter counts. The correction's
final weights/bias start at zero, preserving the source curves in all arms.
The control's inactive branch gives the original token-head training behavior.
The zero-feature arm uses the same extra architecture to distinguish local
information from additional capacity. It retains physical predicted-endpoint
features and valid-support samples; it is not image-independent, since endpoints
and original tokens are image-predicted. Feature masking changes input statistics
as well as information, so it is a controlled ablation, not a perfect causal
decomposition of all capacity effects.

### What the head does

1. The existing model predicts an eight-control B-spline from the same frozen
   nodes and object/relation tokens as before.
2. Reuse the frozen CNN's first activation **before max pooling**: 64 channels
   at stride two (32³ positions for a 64³ MRI patch). Capturing this tensor
   adds no image encoder, detector computation or trainable CNN parameters.
3. Sample 32 uniformly selected parameter positions along that current base
   predicted curve. No target guides, controls, nodes or segmentation enter
   sampling. Sampled coordinates and CNN values are detached.
4. A small 1D convolution plus MLP reads this sequence, valid-support values,
   and the same ten physical endpoint/chord features. It predicts **six × three
   millimetre offsets** for the existing internal controls. Add them to the
   base controls. Endpoints remain exact copies of the detector's nodes.

This is one refinement per forward pass, not an iterative tracker. New module
size: **37,778 parameters**. The representation, cubic degree, knots, number
of controls and existing residual supervision remain unchanged. There is no
additional loss term or image-alignment objective.

Coordinate checks use the actual SENet first convolution: kernel 7, stride 2,
padding 3. Its feature centers correspond to raw voxel indices 0,2,4,... .
Convert raw normalized DHW coordinates to the feature grid before applying
the existing half-voxel `align_corners=False` trilinear sampler. Tests cover
fractional coordinates and odd/non-cubic input sizes. Paths are detached to
avoid an additional geometry feedback path through sampling locations.

## Fixed training and evaluation protocol

- Start from the successful frozen residual head after 15k updates, source
  `frozen_followup_20261009/residual_continue/model.pt`.
- Same 10,000 SyntheticMRI training patches; existing ordered 128 validation
  patches; fixed 512 training-monitoring patches. No new train/val membership,
  augmentation, oversampling of curved branches or test evaluation.
- **5,000 additional updates per arm**, batch four, seed 364505, fresh AdamW,
  constant LR **7e-5**, weight decay **1e-4**, spline weight **1.4739989613**.
- Train existing spline MLP plus the optional local module only. CNN,
  transformer, queries, node/edge heads, matching and all BN buffers stay fixed.
  Existing targets and the residual-control loss remain unchanged.
- Three-update GPU smoke per arm in a separate folder, then a fresh actual
  run from the identical source state. Checkpoint loading permits only the
  explicitly new local-module keys to be absent; all source weights load.
- Existing monitoring at 0/1k/5k: decoded graph metrics, same-endpoint chords,
  GT-matched positive-branch geometry, endpoint-neutral shape, endpoint-localized
  subset, fixed projections, losses, sensitivity and offline W&B.
- Additional **diagnostic-only** matched-positive strata: fitted GT length
  >=5 mm and tortuosity >1.05 (`curved_long`); other long branches
  (`near_straight_long`); length <5 mm (`short`). They partition evaluated
  positives and do not alter targets, losses, sampling, or primary metrics.
- Detector digests, assignment hashes and source initial decoded metrics are
  checked. Initial aggregate tolerance is 1e-5, following the previously
  measured numerical roundoff. Save actual initial tensors and batch-order
  hashes for a final cross-arm pairing audit. First gradients must be finite
  and active/inactive on the added branch according to its scale.

Initial CPU checks passed 11 tests across the new sampler/head routing, existing
frozen geometry and curve-evidence behavior. Eight relevant tests passed again
after adding strata, and the actual source checkpoint/model construction passed
after correcting the pre-pool activation hook name (`relu1`). An initial local
model check needed escalation because MPI initialization uses local sockets;
no scientific configuration was changed to resolve that environment restriction.
Python/Slurm syntax and diff checks passed. GPU smoke is still required.

## Interpretation and stopping point

- Local features outperform both controls on validation **curved** branches:
  supports additional usable local information under this sampling/feature recipe.
- Local and zero-feature arms both outperform the inactive token control:
  extra geometry capacity/endpoint conditioning is a plausible contributor;
  do not attribute the whole gain to image information.
- Training gain without validation gain: generalization remains the issue.
- No benefit on curved branches: this setup does not establish a useful gain;
  it does not prove images are unnecessary. Rare curved training examples,
  image/guide uncertainty, receptive-field coverage, feature choice and budget
  remain limitations. Sampling near a base curve can miss vessels with large
  displacements; this experiment does not test all image-aware representations.

Connectivity should remain exactly unchanged. The next result review should
keep geometry improvements distinct from detection and examine fixed visuals.
No automatic unfreezing, 300-epoch training or dataset switch follows these jobs.
An IXI curved-data study requires its own justified protocol.

## Files and jobs

Submitted **871744_0/1/2**: token control / local features / zero features.
One H100 each, one-hour ceiling, independently smoke-gated. Actual parallel
execution depends on scheduler availability. No results claimed at submission.

Code: `models/local_spline_geometry.py`; optional head construction and shared
feature plumbing in model, criterion, inference and matched evaluator;
`tests/test_local_spline_geometry.py`; reusable pilot runner.
Config: `configs/spline_local_geometry_pilot.yaml`.
Submission: `cluster/jean_zay/spline_local_geometry.slurm`.

Run root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/local_geometry_pilot_20261010/`.
Contains each named arm and `smoke-<arm>` folders. Original files are preserved.
Logs: `experiments/spline_learning/logs/local-geometry-871744_<task>.{out,err}`.
Compact evidence/curvature audit:
`experiments/spline_learning/metadata/local-geometry/` in the code repository.

GPU startup confirmed: all three `smoke-<arm>/summary.json` files report **3 updates completed** and **detector unchanged**. No traceback/assertion markers were found in their logs at the check. All three Slurm tasks were running on separate one-GPU allocations (two may share a multi-GPU node). Full trial results remain pending; no performance conclusion from smoke runs.

## Completed results: local image information improves geometry

All three tasks completed successfully (14:53 / 15:28 / 14:48), each 5,000 additional updates. Initial state, detector state and batch-order hashes match across arms; validation assignments match, and every final detector digest is unchanged.

Validation: 128 patches, 2,059 matched positive branches, including 193 curved long branches. Train monitoring uses 512 patches.

| Arm | Train decoded curve F1 | Validation decoded curve F1 | Matched guide error (mm) | Endpoint-neutral error (mm) | Curved-long neutral error (mm) |
|---|---:|---:|---:|---:|---:|
| token_control | 0.7110 | 0.6905 | 0.8941 | 0.4509 | 1.2927 |
| local_features | 0.8455 | 0.8454 | 0.7399 | 0.3496 | 0.8985 |
| zero_features | 0.7108 | 0.6905 | 0.8945 | 0.4510 | 1.2940 |

Same-endpoint validation chords: decoded curve F1 0.6554, endpoint-neutral error 0.5014 mm overall and 1.6712 mm on curved long branches. Local features reduce endpoint-neutral error versus chords by 30.3% overall and 46.2% on curved long branches. Native node F1 0.9075 and edge F1 0.8586 are identical across arms.

The local-feature arm improves both training and validation; the zero-feature capacity control remains almost identical to the token control. This supports useful local image information for geometry under this protocol. It does not establish the initiating cause of the original scratch collapse or solve connectivity, which was frozen. Single seed, reused diagnostic validation and imperfect guide targets remain limitations. Short branches still worsen on endpoint-neutral error (local 0.1016 mm versus chord 0.0669 mm), so gains are not universal.

Decision point: validate the promising geometry result on the full existing validation split and inspect fixed predictions before another controlled joint-training comparison. No further jobs submitted during this result review. Per user clarification, remain on SyntheticMRI; defer IXI, where continuous branch targets and a reliable detector have not been established.

Compact evidence: `experiments/spline_learning/metadata/local-geometry/results.json`.
