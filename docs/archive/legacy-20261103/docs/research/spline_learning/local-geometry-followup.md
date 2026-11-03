# Local geometry: full validation and controlled unfreezing

User authorized both experiments in parallel after the local-feature pilot.
Remain on SyntheticMRI; defer IXI until the representation learns reliably and
an appropriate continuous-branch target protocol is established.

## Evidence and purpose

The 5k frozen-detector pilot improved diagnostic validation curve F1 from 0.6905
to 0.8454 with local CNN features. Zeroing those features gave 0.6905. Detector
weights, assignments, initialization and batch order matched. This supports
useful image information for geometry, with a single seed and 128 validation
patches; it does not establish the cause of the original scratch collapse.

1. Full validation evaluates all three heads on exactly 1,952 existing validation
   patches. It compares decoded graphs, identical-endpoint chords and matched
   geometry separately, including the predeclared curvature strata. Six fixed
   cases (same seed/selection as earlier full-validation comparison) provide a
   portable HTML gallery of 2D projections of 3D predictions. No model updates or
   BN recalibration; these remain validation/model-selection results.
2. Controlled unfreezing starts all arms from `local_features/model.pt` after
   20,000 cumulative geometry updates. It asks whether graph learning can
   continue while preserving the newly improved geometry.

## Matched training protocol

- 10,000 existing SyntheticMRI training patches, batch 4, 5,000 updates.
- Same 128 validation and 512 training monitoring cases; no new sampling.
- Seed 364505; fresh AdamW; constant detector LR 7e-6, spline-head LR 7e-5.
- Residual supervision and spline weight 1.4739989612791453 unchanged.
- Unfreeze transformer encoder/decoder, queries, input projections and graph
  heads. The CNN, all normalization buffers and dropout mode remain fixed.
- Both the original spline MLP and local correction module train in every arm.
- `graph_only`: geometry inputs detached, so detector gradients come only from
  graph losses; geometry adapts to the changing detector.
- `tokens_only`: geometry also updates shared tokens, with predicted endpoint
  coordinates detached from the geometry loss.
- `joint`: geometry can also update predicted endpoint coordinates.
- Local feature sampling coordinates and CNN features remain detached in all
  arms, as in the successful pilot; no additional sampling-coordinate gradient
  is introduced. Thus `joint` names the existing token/endpoint routes, not a
  newly differentiable image sampler.
- Geometry monitoring 0/1k/5k, loss/sensitivity every 500. Gradient probes at
  1/1k/5k, assignment changes, frozen CNN/buffer digests and initial-source
  metric agreement are checked. Each run performs a fresh 3-update GPU smoke
  before its full trial. Evaluation jobs smoke on two samples first.

The new local-module gradient-routing unit test checks identical forward losses,
active head gradients, intended token/endpoint routes and no CNN-feature gradients.
Existing local sampling/inference tests are reused. Changes are limited to
configurable source update count/tracking groups, optional curvature summaries,
config selection in existing submission scripts and two new configs.

## Interpretation and stopping point

A joint arm should retain the frozen local geometry gain and improve detection
relative to the graph-only continuation. Compare topology as well as edge F1;
previous unfreezing improved some topology metrics and worsened others. Track
curved and short branches separately. Degradation relative to the matched
control implicates optimization coupling under this protocol; it is not evidence
against spline representability. No automatic scratch/300-epoch run follows.
Review full-validation images/results and all three continuations together.

## Reproducibility

Configs: `configs/spline_local_full_validation.yaml` and
`configs/spline_local_controlled_unfreeze.yaml` in the representation code repo.
Use existing Slurm scripts with `DIAGNOSTIC_CONFIG` exported via `sbatch`.

Output roots under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/`:

- `local_full_validation_20261010/`: per-arm evaluation and, when all finish,
  `comparison.html`, `.csv`, `.md`, `.json`. Download just `comparison.html`.
- `local_controlled_unfreeze_20261010/`: `graph_only`, `tokens_only`, `joint`,
  plus separate smoke folders; configs, provenance, metrics, checkpoint and
  offline W&B records. No original output is overwritten.

Submission IDs and checks are recorded below after submission.

Submitted evaluation array **875855_0/1/2** (token/local/zero) and controlled-unfreezing array **875856_0/1/2** (graph-only/token-route/joint-route). Six targeted CPU tests passed, plus syntax/diff checks. GPU smokes are pending at submission. One H100 per task, one-hour ceiling; concurrency depends on scheduler availability. Logs: `local-full-val-875855_<task>.{out,err}` and `local-unfreeze-875856_<task>.{out,err}` under `experiments/spline_learning/logs/`.

## Completed results — 2026-10-10

All six jobs COMPLETED (exit 0:0). Full validation took 10:08–10:22 per arm; unfreezing took 17:04–17:33. All unfreezing arms completed 5,000 updates with identical initial-state and batch-order hashes, and frozen CNN/buffers unchanged.

Full validation (1,952 patches, 31,934 matched branches):

| Head | Curve F1 | Endpoint-neutral error mm |
|---|---:|---:|
| token_control | 0.68411 | 0.46036 |
| local_features | 0.84012 | 0.35986 |
| zero_features | 0.68475 | 0.46009 |

Local features reduce endpoint-neutral error versus same-endpoint chords by 30.1% overall and 47.2% on 3,116 curved long branches. Native node F1 0.91041, edge F1 0.86363 and branch F1 0.27625 remain identical across frozen arms.

Unfreezing results on the original **128 diagnostic validation patches**, not the full split:

| Route | Node F1 | Edge F1 | Branch F1 | Curve F1 | Endpoint-neutral error mm |
|---|---:|---:|---:|---:|---:|
| graph_only | 0.90590 | 0.86392 | 0.27125 | 0.84159 | 0.34037 |
| tokens_only | 0.90599 | 0.86786 | 0.27160 | 0.84678 | 0.34016 |
| joint | 0.90716 | 0.86679 | 0.28253 | 0.84290 | 0.34261 |

Frozen local source on these same 128 patches: node F1 0.90753, edge F1 0.85858, branch F1 0.25497, curve F1 0.84542, endpoint-neutral error 0.34958 mm. All continuations retain the large geometry gain. Token-route has the highest edge/curve F1; joint-route has the highest branch F1. No decisive overall winner or statistical significance established. Geometry improvement relative to graph-only continuation is small; the larger gain remains local image conditioning. Transformer memory sensitivity stays near 0.92 and GT assignment changes are similar (6.64–6.72%); no collapse detected.

Topology errors (graph-only / tokens / joint): Betti0 1.01563 / 0.97656 / 1.02344; Betti1 1.11719 / 1.00781 / 1.03125. These are diagnostic subset scores; avoid comparing directly with full-validation scores.

Gallery generated: `local_full_validation_20261010/comparison.html` (9,818,757 bytes). Inspected token/local projections for sample_000075_0287 and local sample_000014_0919. Visible residual localization/connectivity errors remain; individual branches are still nearly straight. The gallery is illustrative, not evidence of universal improvement. Original records contain distinct learned controls and 64-point sampled curves; the images plot sampled curves.

Next decision: compare the three unfreezing checkpoints on the full validation split before selecting a continuation. Remain on SyntheticMRI. No additional jobs launched during this result review.
