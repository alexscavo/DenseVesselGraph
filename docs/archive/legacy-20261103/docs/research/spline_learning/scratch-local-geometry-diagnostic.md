# Scratch learning with residual local-image geometry

User authorized proceeding after the pretrained stability phase. This is a
bounded paired diagnostic, not another full 300-epoch training.

## Why investigate overfitting and scratch learning separately?

The 25k pretrained continuation's fixed 512-case training monitor showed branch
F1 rising from 0.32094 to 0.39205 and edge F1 from 0.87553 to 0.90534. On the
fixed 128-case validation monitor, branch F1 went from 0.27125 to 0.26933
(peaking at 0.28466 after 5k), while edge F1 went from 0.86392 to 0.85943.
Together with full-validation branch decline, this supports overfitting in
connectivity; it does not prove a unique cause. Geometry continued improving.
The earlier full-validation-selected checkpoint remains preserved.

Pretraining supplies graph localization/connectivity already. Successful local
geometry with that detector does not establish scratch learning. The next test
removes all pretrained weights while holding graph training conditions matched.

## Two arms

- `graph_control`: spline weight 0; graph losses alone train the detector.
- `local_spline`: spline weight 1.4739989612791453; graph losses plus residual
  control supervision train from update one.

Both construct the same eight-control B-spline MLP and local-image correction
module, initialized identically alongside the detector. Only the geometry loss
weight differs. The control's zero-initialized output layers receive no geometry
updates, so its curves remain exact chords; this invariant is checked after
training. This retains paired parameter construction while reproducing a
straight-edge graph-learning control.

The spline arm uses the existing `tokens_only` gradient route: predicted node
coordinates are detached for geometry supervision; geometry gradients can
propagate through object/relation tokens into the transformer and CNN. The
local sampler retains its existing detached CNN values/sampling coordinates.
The CNN still learns from graph losses and token-mediated geometry gradients.
No GT coordinates enter the head or sampling path. Matcher and targets unchanged.

This selects endpoint detachment as the first minimal scratch recipe because
it retained geometry/edge performance in the prior controlled experiment. It
is not a claim that the alternative route is invalid. This two-arm experiment
cannot isolate the separate contributions of residual targets, local features
and detachment; it tests whether this recipe interferes with graph learning.

## Fixed protocol

- All model modules, including CNN, randomly initialized; no checkpoint loaded.
- Exactly the existing 10,000 SyntheticMRI train patches, batch 4, seed 364505.
- 25,000 updates per arm (10 passes), one H100 per arm, independent parallel jobs.
- Original scratch AdamW LR 7e-5, WD 1e-4, 2,500-update warmup from 1e-9.
  Preserve the original polynomial LR horizon of 750k updates (300 epochs),
  but the job **stops at 25k**; the horizon is not authorization to train 300 epochs.
- Ordinary train mode, including BN updates, in both arms. Saved BN statistics
  used for evaluation; no recalibration. Unlike frozen/pretrained diagnostics,
  the entire detector and CNN now learn.
- No augmentation, target, sampler, graph/node-budget or split changes.
- Reuse the same 512 train-monitor and 128 disjoint validation cases. The source
  run supplies sample IDs only, never weights or predictions.
- Loss/image sensitivity every 1k; decoded graph/chord metrics, separate
  GT-matched geometry, curvature strata and fixed projections at 5k/15k/25k.
- Initial-weight and actual batch-order hashes saved for paired verification.
- Model+optimizer+scheduler snapshots at 5k, 15k, final25k. Offline W&B retained.
- Each task passes a fresh 3-update GPU smoke before its full run. Check finite
  gradients, active CNN gradients, intended active/zero spline gradients and
  control's exact-chord preservation. Targeted CPU checks use the real training
  step to verify BN updates and the supervision toggle, plus existing routing tests.

## Interpretation

- Both learn detection, spline gains curvature without substantial graph harm:
  evidence the improved representation can start learning without pretraining;
  longer scratch evaluation is still needed.
- Graph-only learns but spline loses detection/image sensitivity: evidence of
  geometry coupling under this recipe. A same-run graph-first warm-up is then
  a justified follow-up, not automatically launched now.
- Both fail/learn slowly: initialization, shared training settings or short
  budget remain explanations; cannot attribute failure specifically to splines.
- Training improves while validation stalls: inspect generalization instead of
  interpreting lower training loss as reconstruction success.

Original 300-epoch collapse may occur later than this diagnostic. Absence of
collapse at 25k does not rule it out. No test-set inference, IXI switch or further
training is queued automatically.

## Files and outputs

Config: `configs/spline_scratch_local_pilot.yaml`.
Runner: existing `scripts/train_spline_weight_pilot.py` with explicit opt-in
`scratch_geometry`; defaults for earlier frozen pilots remain unchanged.
Submission: `cluster/jean_zay/spline_scratch_local.slurm`.

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_local_pilot_20261010/`
with `graph_control`, `local_spline` and separate `smoke-*` directories.

Submitted **878748_0** (graph-only control) and **878748_1** (local-image residual spline). Four targeted CPU tests passed, including real train-step BN and zero/active geometry supervision checks. Syntax/diff checks passed. GPU smokes pending at submission. Each task has a three-hour ceiling. Logs: `experiments/spline_learning/logs/scratch-local-878748_<task>.{out,err}`.

## First smoke failure and correction

Both 878748 tasks stopped before training during model setup (48–50 seconds). The diagnostic asserted every stored parameter was trainable, but the real decoder deliberately keeps `cross_attn2` and `norm12` frozen because the legacy relation cross-attention block is not called by forward. The miniature CPU fixture did not contain that block. This was a diagnostic guard error, not evidence of learning collapse.

The guard now requires exactly those known inactive tensors to be frozen and rejects freezing any active parameter. Model architecture, forward computation, optimizer and scientific settings are unchanged. A real decoder regression check and full actual-model construction verify the correction. The first local construction attempt was blocked by sandbox MPI sockets and required execution outside the sandbox; no experiment setting changed.

Resubmission uses a new `scratch_local_pilot_20261010_attempt2` output root to preserve failed smoke artifacts. The original array started no full 25k training, and no results from it should be interpreted scientifically.

Actual full model construction passed the revised guard: exactly 24 intentionally inactive tensors; an injected CNN freeze was correctly rejected. Both targeted tests passed, including the real decoder regression and the existing three-step scratch supervision test. Resubmitted **885337_0/1**, unchanged settings, separate attempt2 output directory. GPU smokes pending at submission.

## 2026-10-11 — Completed scratch comparison

Jobs 885337_0/1 completed successfully, including smokes, in 47:08 / 44:12. Each completed 25,000 updates. Initial-state and actual batch-order hashes are identical; no checkpoint was loaded.

Validation results on the fixed 128 cases (not the full validation split):

| Arm | Node F1 | Edge F1 | Curve F1 | Memory image sensitivity |
|---|---:|---:|---:|---:|
| graph_control | 0.236421 | 0.195736 | 0.056406 | 0.89432681 |
| local_spline | 0.000000 | 0.008279 | 0.004004 | 3.3786341e-06 |

Graph-control edge F1 rises 0.00665 → 0.14435 → 0.19574 at 5k/15k/25k; node F1 reaches 0.23642. It is learning but still far from a usable graph: common branch F1 remains zero. Local-spline final train-monitor node F1 is also zero, so its failure is not explained as ordinary train-good/validation-bad overfitting.

Spline transformer-memory sensitivity drops from 0.8491 at 4k to 0.001600 at 9k and 0.000003379 at 25k; graph control reaches 0.8943. These are fixed-probe eval-mode measurements, not proof that every representation has collapsed on every image. CNN projected-feature sensitivity remains 0.1381 for spline versus 2.7208 for control. Final spline endpoint-neutral guide error 0.50264 mm is slightly worse than the chord reference 0.50138 mm. Curvature did not improve.

This matched experiment supports destabilization associated with geometry supervision from initialization, despite residual targets, local image features and endpoint detachment. It does not isolate the exact internal mechanism, exclude BN-mediated effects, or prove all scratch recipes fail. The only intended arm difference is spline loss weight; head sampling and targets are unchanged. The successful pretrained geometry experiments therefore do not yet transfer to scratch learning.

Next discriminative proposal: train the spline head from scratch while detaching both tokens and endpoints from its loss, allowing the detector to learn only from graph losses. This directly tests the remaining geometry-to-detector gradient route, before adding a graph-first warm-up or changing architecture. The completed graph-only run supplies the matched reference if initialization, batch order and settings are preserved. No new experiment launched during this review.
