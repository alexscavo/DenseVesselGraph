# Paired spline-weight pilot — 2026-10-08

## Question and protocol

Does reducing spline supervision prevent the scratch model from losing image
dependence? This follows the [collapse diagnosis](spline-collapse-diagnosis.md).
The user authorized implementation and testing of the proposed weight comparison.

| Setting | Both arms |
| --- | --- |
| Training population | 512 deterministic seeded patches from the original 10,000, covering all 64 training subjects |
| Validation population | 128 deterministic seeded patches from the original 1,952, covering all 13 validation subjects |
| Test population | Not evaluated or used for choosing settings |
| Initialization / batch | Scratch, seed 364505 / batch 4 |
| Budget | 5,000 updates, approximately 39.1 passes over the subset |
| Optimizer and LR | Original AdamW, peak 7e-5; original 2,500-update LR warmup and 750,000-update polynomial schedule |
| Geometry | Same eight-control cubic splines, predicted endpoints, original full-path targets |
| Augmentation / BN | Original augmentation disabled; normal training BN, saved statistics for validation, no recalibration |
| Only experimental difference | Spline weight 1.4739989612791453 versus 0.14739989612791453 |

The reduced value is a diagnostic candidate, not an established optimum.
The scheduler keeps the original **update-count** trajectory; setting a short
epoch budget in the original scheduler would also change learning-rate decay.
`resolved-config.yaml` retains that original 300-epoch scheduler configuration;
`pilot-config.json`, `provenance.json`, and `summary.json` record the actual
5,000-update stopping budget. These are short optimization trials, not new
300-epoch runs. The first 2,500 updates are the original LR warmup.

No core model, loss formula, dataset, or completed benchmark is modified.
The runner calls the existing training step, optimizer, scheduler, validation
loss, inference, and common metric functions. Dataset caches are read directly;
no new copy of MRI volumes or target files is needed.

## Monitoring and verification

- Static syntax/Slurm checks; each arm runs a separate three-update GPU smoke
  with four validation cases before restarting from fresh initialization for
  its bounded trial. The smoke exercises loss/backward/update, graph evaluation,
  full-guide scoring, and projection rendering.
- Record exact membership, initial state checksum, and actual training-batch
  order checksum. Compare these between arms before interpreting results.
- Every 500 updates: training losses over that interval, validation losses,
  and input sensitivity at CNN features, transformer encoder memory, decoder
  tokens, and predicted node coordinates. The fixed sensitivity probe includes
  different validation subjects. Monitoring preserves training RNG state.
- At updates 2,500 and 5,000: native graph diagnostics and shared-guides-v2
  geometry/topology on all 128 validation patches, plus two fixed projections.
  These use the original uncertain guides, not fitted splines alone.
- Offline W&B records training, validation, input sensitivity, common metrics,
  and projections. Validation epoch axes count completed subset passes; update
  counts are also recorded. Local JSON history remains authoritative.
- Retain only the final model per arm. Refuse to overwrite an existing run.

Expected comparisons: does ordinary graph loss fall, is image dependence
preserved, and does decoded validation recovery improve? Weighted total loss
alone is unsuitable because its definition differs between arms. If both arms
work on the subset, expand the training population before claiming a solution.
If reducing the weight fails, the next proposed intervention is delayed/ramped
spline supervision, tested separately rather than combining changes now.

## Files

Code: `DenseVesselGraph/scripts/train_spline_weight_pilot.py`.
Protocol: `DenseVesselGraph/configs/spline_weight_pilot.yaml`.
Submission: `DenseVesselGraph/cluster/jean_zay/spline_weight_pilot.slurm`.

Run root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/weight_pilot_20261008/`.
`original/` and `reduced/` contain actual trials; `smoke-original/` and
`smoke-reduced/` contain explicitly separate startup checks.

## Submission

Slurm array **763233**, task 0 = original weight, task 1 = reduced weight.
Each task requests one H100, with a 30-minute maximum, and performs its own
three-update smoke before fresh initialization for the 5,000-update trial.
The two tasks can run concurrently on separate GPUs. Submission/source hashes
are recorded under `experiments/spline_learning/metadata/weight-pilot/` in the
code repository.

## Completed results

Both tasks completed successfully: original **11m42s**, reduced **11m19s**,
including runtime staging, smoke, training, evaluation, and W&B finalization.
Actual trial execution was approximately 8m09s and 7m48s, respectively.
The checksum audit confirmed identical initial weights, patch membership,
actual training-target batch order, and learning-rate schedule. Both completed
5,000 updates. Final projections for the same fixed validation sample were
inspected and confirm poor vessel recovery in both arms.

| Metric at 5,000 updates | Weight 1.474 | Weight 0.1474 |
| --- | ---: | ---: |
| Training node loss, final 500-update mean | 0.22354 | 0.22354 |
| Validation node loss | 0.23806 | 0.21980 |
| Matched-positive fitted-curve error (mm) | 8.6153 | 8.1064 |
| Native node F1 | 0 | 0 |
| Common anchor F1 | 0 | 0 |
| Common branch F1 | 0 | 0 |
| Common whole-curve F1 | 0.001450 | 0.000808 |
| Fixed-probe node coordinate difference | 0.19927 | 0.37115 |
| Fixed-probe transformer-memory difference | 0.55432 | 0.66949 |

The reduced weight slightly improves validation node loss and matched-positive
curve error, but does not produce useful decoded reconstruction. Its common
whole-curve F1 is lower. Neither model has recovered any branches, so matched
decoded-branch guide/tortuosity errors are undefined. The reported fitted-curve
error uses GT-matched positive pairs and does not imply successful edge detection.
Both models remain input-sensitive; neither reproduces the near-constant
predictions observed after 300 epochs.

### Interpretation and next decision

This is a negative result for an immediate rescue by a tenfold weight reduction,
and an **inconclusive test of the cause of the eventual collapse**. The reduced
weight is not promoted as a validated fix.

The 5,000-update budget equals only the first two epochs of the original full
training schedule, including 2,500 warmup updates. The original straight-edge
run's epoch-2 training node loss was also high (0.2214). Repeating the small
subset about 39 times does not make this equivalent to a mature full training.
One seed, limited updates, and a different training population prevent a claim
that weighting can never help or that the spline representation is infeasible.

Recommended next controlled experiment: establish graph learning with spline
weight zero on the same population, extending the budget until this control
learns; then compare continuing graph-only against introducing/ramping spline
supervision from the same learned graph state. That tests whether adding shape
disrupts already working node/edge prediction. This is proposed, not launched.

### Result locations

Compact verified CSV, JSON, histories and provenance:
`DenseVesselGraph/experiments/spline_learning/metadata/weight-pilot/`.
Each run directory retains `model.pt`, `history.jsonl`, offline `wandb/`, and
`validation-005000/` containing predictions, native/common scores and PNGs.
No original checkpoint, dataset, or test result was overwritten.
