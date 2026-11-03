# Scratch geometry with detector gradients isolated

Authorized after scratch run 885337 showed learning in the graph-only control
but near-zero image sensitivity and node F1 in the local-spline arm. Initial
weights and actual target-batch hashes matched. Those runs used geometry
gradients through tokens, with predicted endpoints already detached.

## Single intervention

Use the existing `graph_only` gradient route in the spline criterion: detach
object/relation tokens as well as predicted endpoint coordinates when forming
the geometry loss. CNN image values and sampling coordinates were already
detached by the local geometry module. Graph losses still train every active
CNN/transformer/node/edge parameter; spline losses still train both geometry
MLPs. This is **not a frozen detector** and loads **no pretrained weights**.

Reuse the completed `scratch_local_pilot_20261010_attempt2/graph_control`
reference. Only the positive spline loss with isolated gradients differs from
that control; versus the failed local-spline arm, only the token gradient route
changes. There is no reason to repeat either completed reference run.

## Matched conditions

- Same random seed 364505, eight controls, image sampler, residual targets,
  Hungarian matcher, 10k training patches and batch 4.
- Same AdamW LR 7e-5, weight decay 1e-4, 2,500-update warmup and original
  polynomial schedule horizon. Hard stop at 25,000 updates, approximately 10
  data passes; this is not a full 300-epoch run.
- Spline weight 1.4739989612791453; both spline modules train from step one.
- Normal CNN/BN training; saved BN statistics for evaluation, no recalibration.
- Same 512 training-monitor and 128 validation cases, monitored at 5k/15k/25k;
  losses and fixed-probe sensitivity every 1k. Geometry metrics remain separate
  from decoded graph metrics and are compared with identical-endpoint chords.
- Intermediate 5k/15k and final25k checkpoints; offline W&B and projections.
- Before starting: compare reference initial-state hash, configs (excluding spline
  loss weight), data/evaluation membership, update budget and schedule metadata.
  After training: assert actual target-batch hash matches the completed control.
- A targeted real train-step test verifies that three isolated-geometry updates
  change the head but leave detector parameters/BN buffers identical to the
  graph-only control. Existing local gradient-routing checks are reused.
- Independent three-update GPU smoke before the full run, starting afresh from
  random weights afterward; no smoke-trained weights are reused.

## Interpretation

Detector learning should track the graph-only reference closely. If it avoids
collapse, this supports the remaining token-mediated geometry gradient route
as the destabilizing intervention in the earlier experiment. It does not explain
why those gradients harm optimization internally, or prove every coupled setup
fails. If detector behavior differs strongly, first verify routing and pairing.

If curvature also improves, isolated concurrent learning becomes a candidate
scratch recipe. If curvature fails while detector learning is healthy, poor
localization and changing early image features remain likely difficulties;
a later graph-first warm-up would then be a justified next question. Common
branch recovery was still poor in the control at 25k, so modest geometry progress
would not establish a usable graph extractor. No additional follow-up auto-runs.

## Reproduction

Config: `configs/spline_scratch_detached_pilot.yaml`.
Submission: `cluster/jean_zay/spline_scratch_detached.slurm`.
Existing runner `scripts/train_spline_weight_pilot.py` now accepts
`scratch_gradient_route` (default retains `tokens_only`) and optional
`scratch_reference_run` pairing checks. No matcher or architecture changes.

Outputs:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_detached_pilot_20261011/`
with `detached_spline` and `smoke-detached_spline` directories.

Submitted **903323**. Five targeted CPU tests passed, including detector/BN equivalence while the isolated head learns. Syntax/diff checks passed. GPU smoke pending at submission. Expected runtime around 45–50 minutes after allocation, based on matched 25k runs; three-hour ceiling. Logs: `experiments/spline_learning/logs/scratch-detached-903323.{out,err}`.

## Completed result and narrow follow-up (2026-10-12)

903323 completed 25,000 updates in 45:11 (exit 0:0). Initialization and batch-order
hashes match the graph-only reference. Final diagnostic validation (128 cases):

| Arm | Node F1 | Edge F1 | Curve F1 | Memory sensitivity |
|---|---:|---:|---:|---:|
| Graph-only control | 0.23642 | 0.19574 | 0.05641 | 0.89433 |
| Geometry through tokens | 0.00000 | 0.00828 | 0.00400 | 0.00000338 |
| Geometry detached | 0.20566 | 0.19278 | 0.05605 | 0.85032 |

Isolation avoids the observed collapse, but does not produce useful branch
recovery yet: validation branch F1 remains zero. Endpoint-neutral guide error
0.49984 mm versus chords 0.50138 mm gives only 0.31% relative improvement. This
is not evidence that scratch geometry is solved. The detached detector does
not exactly match the graph-only reference despite paired inputs; numerical
variation is plausible but has not been established as the explanation.

User authorized needed tests. Submitted short GPU diagnostic **948360**:

1. At the coupled scratch 5k checkpoint, compare geometry gradients into all
   active detector parameters under token vs detached routes; assert detached
   gradients are zero on the real GPU model.
2. Restore identical model/optimizer/scheduler/RNG state and take three real
   updates for two repeated graph controls and one detached-geometry condition.
   Compare detector/BN changes against the repeated-control numerical variation.
   Controls use the same 5k optimizer state, including spline momentum; geometry
   head movement in these short zero-loss controls is not a detector-gradient leak.
3. Probe all three final checkpoints with saved BN statistics versus current
   batch statistics using the original fixed four images. Restore buffers after
   each probe. This measures image sensitivity only, not full dataset accuracy.

These are bounded checks, not another 25k run. No checkpoint is overwritten.
The initial 25k outcomes support a role for geometry feedback but do not prove
an internal mechanism or exclude BN-mediated changes. Resolve these checks
before choosing a longer graph-first/warm-up experiment.

Code: `scripts/diagnose_scratch_isolation.py` and
`cluster/jean_zay/spline_scratch_isolation_checks.slurm`.
Output: `runs/scratch_isolation_checks_20261012/summary.json` under the same
scratch experiment root. Logs: `experiments/spline_learning/logs/isolation-checks-948360.{out,err}`.
Python/Slurm syntax checked; the submitted small GPU diagnostic is itself the
real-data correctness test, with a 20-minute ceiling.

## Isolation/BN diagnostic results (2026-10-12)

Job **948360** completed successfully in **1:07**.

- Real-model detached geometry gradient into the detector: exactly zero across
  all active tensors. Token-coupled geometry has nonzero gradients in 569 tensors.
- Same-start three-update detector differences: repeated graph control relative
  L2 1.153e-6 versus detached/control 1.253e-6. This supports numerical variation
  as an explanation for short-run differences; it does not quantitatively explain
  every difference accumulated across 25k updates or establish seed robustness.
- The collapsed model's memory sensitivity stays near 3.4e-6 with either saved
  BN statistics or current batch statistics (3.379e-6 versus 3.451e-6).
  Graph control remains about 0.894 under both; detached remains 0.850/0.828.
  Thus changing inference BN statistics does not restore image sensitivity on
  the fixed four-image probe. It does not exclude a role for BN during training.
- At the 5k checkpoint on this one batch, weighted geometry gradient norm is
  0.002369 versus graph 15.582 (ratio 0.000152). No evidence here for a simply
  overwhelming geometry gradient; this late probe cannot describe earlier
  gradients, layer-specific alignment or nonlinear optimization trajectories.

Together with the matched scratch runs, the evidence points to harmful early
geometry feedback through shared tokens under this setup. The exact internal
mechanism remains unresolved. Correct detachment avoids collapse but curves
remain near chords while scratch node/branch localization is still poor.

Next proposed research decision: extend graph-only and detached scratch training
with matched budgets to see whether stronger localization allows curvature
learning without external pretraining. Do not select a coupled scratch recipe
or assume eventual success from this short test. No new jobs submitted in this
result review. Evidence copy: `metadata/local-geometry/scratch-isolation-results.json`.
