# Staged graph-to-spline learning — 2026-10-08

## Why this experiment

The earlier four-patch result was not the only successful spline experiment.
The completed 50-epoch fine-tuning run on 9,709 SyntheticMRI training patches
reported test node F1 **0.884115**, edge F1 **0.815718**, and native spline
branch F1 **0.169838** over 13,071 test patches, with BN recalibration zero.
Source: `runs/spline_v0_k8_10k_val2k_50ep/test/summary.json` and `metadata.json`.
Those native metrics are not substituted for the newer common-guide metrics.
They establish that spline learning already succeeded beyond four samples.

The unresolved problem is optimization from scratch under the current matched
10k protocol. The [short weight comparison](spline-weight-pilot.md) did not
resolve this and was too early to establish a mature graph-learning control.
The user authorized proceeding with the work needed to solve the failure.

## Controlled staged pilot

- Initialize both arms from `scratch_old_graph_10k_300ep/models/best_checkpoint.pt`,
  **epoch 35 / 87,500 updates**, chosen by that run's original validation-loss
  rule. This graph model learned on the same 10,000 training patches; it adds
  no external training population. It is not the final epoch-300 checkpoint.
- Add the same newly initialized eight-control spline head to both copies.
  Strict loading permits missing spline tensors only. Initial head residuals
  are zero, so both begin with the original straight-edge geometry.
- Reuse the weight pilot's exact 512 training / 128 validation patches,
  covering all 64 training / 13 validation subjects, seed 364505, batch four,
  and 5,000-update budget. Preserve boundary nodes, targets, and metrics.
- **Graph control:** spline loss weight zero; the zero-residual spline head
  stays a chord. Ordinary graph losses train the entire original graph model.
- **Staged spline:** ramp the spline loss linearly from zero to **1.474** over
  2,500 updates, then keep it fixed for the remaining 2,500. Jointly train the
  original graph model and geometry head, using predicted endpoints.
- Both arms deliberately start with **fresh AdamW moments**. The LR scheduler
  is advanced to update 87,500 on the original 750,000-update trajectory.
  This is a matched optimizer restart, not an exact resume of the old run.
- Ordinary training BN and saved statistics for validation; no recalibration.
  No core model/loss formula changes or test-set tuning.

This tests one intervention between the two arms: introducing ramped spline
supervision after graph learning. It is not a rerun of the scratch weight
comparison and cannot isolate the separate benefits of warmup and ramping.

## Checks and outcomes to assess

The reused runner records exact membership, initial-state and batch-order
hashes. Each GPU job first runs three updates and evaluates four validation
patches, then restarts the real trial from the original checkpoint. Original
checkpoints and prior experiments are never overwritten.

Monitor losses and input sensitivity every 500 updates; evaluate native and
common full-guide topology/geometry at updates 0, 2,500, and 5,000. Compare
each decoded curved graph with **its own straight chords using identical
predicted nodes and edges**, so shape improvement can be separated from graph
changes. Save two fixed projections and offline W&B records.

Evidence of a useful solution requires maintaining graph recovery and learning
geometry, rather than merely lowering a differently weighted total loss.
If spline supervision damages graph prediction while the control remains
healthy, isolate geometry gradients in a subsequent targeted experiment.
If both regress, investigate continuation settings before blaming splines.
If the staged arm remains healthy and improves geometry, scale the staged
protocol to the full training population before declaring the failure solved.

## Submission and files

Slurm array **775025**: task 0 `graph_control`, task 1 `staged_spline`, one H100
each, 30-minute task limit. Code/config:
`scripts/train_spline_weight_pilot.py`, `configs/spline_staged_pilot.yaml`,
`cluster/jean_zay/spline_staged_pilot.slurm` in the representation repository.

Outputs:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/staged_pilot_20261008/`.
Compact provenance/results will be kept under
`experiments/spline_learning/metadata/staged-pilot/` in the code repository.

### Startup status

The staged-spline task **775025_1** passed its three-update GPU smoke and
started the real trial. The graph-control task stopped before creating run
outputs because configuration validation required a strictly positive spline
weight. Updated that check to permit zero (for graph-only controls/curricula)
while rejecting negative weights. Targeted verification passed for both cases.
Only the control task was resubmitted as **775308_0**, using the same protocol.

At the user's request, active monitoring stopped after submission. No completed
trial result is claimed here; inspect both summaries and run the paired-results
verifier before drawing conclusions. The validator change does not alter the
already running positive-weight arm's training behavior.

## Completed pilot

Both real trials completed 5,000 updates successfully: graph control
**775308_0, 10m45s**; staged spline **775025_1, 10m26s**. Pairing verification
passed for initial model state, membership, actual batch order, source
checkpoint, schedule, and ramp configuration. Both initial validation scores
were identical. No test-set evaluation was performed.

| Final validation metric | Graph control | Staged spline |
| --- | ---: | ---: |
| Native node F1 | 0.56421 | 0.56469 |
| Native edge F1 | 0.45498 | 0.46081 |
| Common anchor F1 | 0.07841 | 0.06258 |
| Common branch F1 | 0.001836 | 0.000607 |
| Common curve F1 | 0.15118 | 0.15400 |
| Same-endpoint straight-chord curve F1 | 0.15118 | 0.15448 |
| Matched-positive fitted-curve error (mm) | 3.6101 | 3.6242 |

Both maintain input sensitivity and graph detection, unlike the failed
300-epoch scratch spline model. However, both slightly regress from the
initial node F1 0.58495 and curve F1 0.16009. The spline's curve score is
slightly worse than its own chords, so its small advantage over the control
does not establish a learned-shape benefit. Common topology recovery remains
weak; matched-branch geometric errors apply to very few recovered branches.

Conclusion: staged initialization avoids collapse over this short interval,
but does not yet solve geometry learning or justify a long run. A targeted
next diagnostic would freeze the working graph predictor and train only its
spline head, then measure improvement against identical-endpoint chords.
That would isolate geometry learnability from graph drift. No further training
was submitted while recording this result.

Verified tables, raw histories, summaries, and completion records are saved
under `experiments/spline_learning/metadata/staged-pilot/` in the code repo.
