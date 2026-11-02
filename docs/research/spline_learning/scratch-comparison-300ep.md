# SyntheticMRI: paired scratch comparison, 300 epochs

## Decision (2026-10-06)

The user requested two scratch runs, each on exactly 10,000 vessel patches for
300 epochs. They clarified that the old representation should also use only
junctions, terminations, and boundary nodes, without intermediate degree-two
nodes. Consequently this experiment isolates **straight versus spline edge
geometry**, rather than testing degree-two node removal. A degree-two baseline
is a useful later experiment and has not been submitted as a third job.

## Shared population and supervision

- SyntheticMRI only; no plants; original subject-disjoint splits preserved.
- 10,000 train patches from the available training subjects: retain the existing
  9,709 compatible patches and add the next 291 eligible patches in the same
  seed-364505 hash ranking. Closed/parallel branches remain excluded; no quality
  or model-score filtering. Never fill training from validation/test.
- Keep the existing 1,952 validation and 13,071 test patches exactly unchanged.
- Both arms consume identical MRI, masks, VTP endpoint nodes, and edge pairs.
  The straight arm decodes each edge as a chord. The spline arm additionally
  consumes eight fitted controls per branch (six internal predicted controls).
- Both preserve existing boundary supervision: voxel-cell crop bounds, snap
  to the first/last existing inside guide samples within each component, with
  the recorded exact-intersection fallback for undersampled components.
- Original full generator guides are uncertain supervision, not perfect anatomy.
- Common exclusions mean these results apply to the eligible population;
  closed/parallel branches are not evaluated by this experiment.

## Training protocol

- Entire model initialized from scratch: no pretrained checkpoint and no resume.
- Shared seed 364505; verify all common initial model weights are identical.
- Same 120 object queries and two relation tokens; same encoder/decoder and
  original edge existence head. Extra spline MLP enabled only in spline arm.
- Native 64³ MRI, batch 4, validation batch 4, no augmentation in either arm.
- AdamW: learning rate 7e-5, weight decay 1e-4, betas (0.9,0.999), epsilon 1e-8.
  One warmup epoch; polynomial schedule power 0.9; 300 epochs, no early stopping.
  Exactly 2,500 batches per epoch, 750,000 optimizer steps per arm.
- Shared loss weights: boxes 3, class 4, cardinality 0.8, nodes 2, edges 6.
  Same edge focal classification and candidate/balancing policy.
- Spline arm adds control loss weight **1.4739989612791453**, retaining the
  previously recorded training-only calibration. It has not been tuned on the
  new scratch results or held-out test set. Scratch-specific tuning is a later
  sensitivity study if necessary, not a hidden difference in this first pair.
- Validation every five epochs; select best checkpoint using the weighted sum
  of the **five common graph losses**, excluding spline loss in both arms.
  Keep the final/latest full-state checkpoint too. Full validation totals have
  different meanings between arms and must not be compared as one metric.
- Four fixed validation subjects/examples use the same MRI/mask/full-guide
  projections. Old predictions display chords, new predictions display splines.
- W&B compute logging is offline, with distinct runs in a common comparison
  group; sync from the login node, avoiding the previous compute connectivity
  failure. Metrics and validation images remain saved locally before upload.

## Storage and scripts

Targets and memberships:
`$SCRATCH/experiments/gnbm/spline_learning/scratch_comparison_10k_300ep/`.
`comparison-manifest.json` records exact IDs, source metadata and extra exclusions.
Each arm has its own dataset root; shared volumes/graphs are symlinked, not copied.

Code on the `representation` branch:
- `scripts/prepare_scratch_comparison.py`: extend eligible train membership and pair roots.
- `scripts/check_scratch_comparison.py`: config, exact IDs, real validation batch,
  control shapes, common initialization and projection checks.
- `configs/scratch_comparison_{old,spline}_300ep.yaml`.
- `cluster/jean_zay/train_scratch_comparison.slurm` selected with `COMPARISON_ARM`.

The immutable verified runtime archive is staged job-locally for both jobs.
Each requests one H100 and 20 hours. Previous spline throughput suggests roughly
17–19 hours per run, excluding queue time; this is an estimate, not a guarantee.
If interrupted by the time limit, resume the full latest state to complete the
same 300-epoch protocol; never restart silently with a changed schedule.

## Interpretation and next steps

10,000 patches are sufficient for an informative first paired experiment;
they are correlated crops from 64 training subjects, not 10,000 independent
observations. A single seed does not establish statistical significance.
Evaluate both arms on the unchanged held-out subjects with the same decoding
thresholds, endpoint/branch connectivity, physical curve distance and tortuosity.
Always report unmatched branches/coverage along with matched curve errors.
Compare both best-by-common-validation and final-epoch checkpoints explicitly.
Use subject-level intervals and repeated seeds before paper-level conclusions.
Then consider the degree-two-node baseline, augmentation and spline-loss
sensitivity without retroactively changing this pair's protocol.

Submission IDs and preflight outcome will be recorded after submission.

## Preparation checks

Preparation finished with the requested 10,000 / 1,952 / 13,071 membership.
The next 323 ranked training candidates yielded 311 compatible and 12 excluded;
the first 291 compatible IDs were retained. Endpoint graphs have at most 54
nodes, fitting the common 120-query budget. All MRI/mask/graph triplets, exact
arm memberships, one real validation batch per arm, cache-control shapes,
shared initial weights and projection generation were checked.

An older temporary package copy contained an incomplete SymPy installation.
Fresh archive staging resolved it; staging now also uses the protected cluster
module interpreter, keeping mutable venv fallback packages out of import search.
The two jobs use the same checksum-verified archive and interpreter selection.

## Submitted jobs

- Straight-edge scratch arm: **708757** (`scratch_old_graph_10k_300ep`).
- Eight-control spline scratch arm: **708759** (`scratch_spline_k8_10k_300ep`).
- Stable implementation commit: `dacb858` on `representation`.
- Final CPU preflight passed with the protected interpreter and fresh archive;
  full training entrypoint imports also passed.
- Job IDs, manifest checksum and verification are recorded in
  `DenseVesselGraph/experiments/spline_learning/metadata/scratch-comparison-jobs.json`.
- Both are independent Slurm jobs, with no dependency on the other's completion.
  Compute W&B logs are offline and need login-node syncing for cloud visibility.

Submission check: both jobs are **PENDING (Priority)**; neither has started yet.

## Third arm requested

The user subsequently requested the intermediate-node baseline and selected the
compact adaptive version. This supersedes deferring that variant to a follow-up.
See [the degree-two protocol](adaptive-degree2-scratch-comparison.md). The first
two jobs and their targets/configurations are unchanged.

## 2026-10-07: spline filesystem interruption and recovery

Spline job 708759 stopped after epoch 171 during a training DataLoader file
stat, with filesystem error 108 (`Cannot send after transport endpoint shutdown`)
on `sample_000128_0705_graph.vtp`. A subsequent real VTP read succeeded. This
evidence supports a storage-access failure; the underlying storage cause is
not known. No target file was changed, removed or replaced.

`latest_checkpoint.pt` was fully read and verified at epoch 171 / iteration
427,500, with model, optimizer, scheduler, RNG and trainer state present.
Recovery job **721673** uses `--resume` to continue at epoch 172 under the same
300-epoch config, data, loss weights and schedule. The original baseline job
708757 remains running (observed epoch 223 during diagnosis). Stable recovery
implementation commit: `edb30bc`. This restores a scratch-trained run; it does
not initialize from an unrelated pretrained model.

Recovery startup verified: **721673 is RUNNING**. The offline tracker started
with the original run ID after strict model/optimizer/scheduler/RNG restoration
(the tracker is constructed only after the resume load succeeds). It continues
the same scratch experiment from epoch 172. Adaptive **709924 is also RUNNING**
after verification job 721618 completed successfully.

## Held-out tests and common representation-independent evaluation

Straight-edge job 708757 completed all 300 epochs (13:29:14, exit 0).
Test job **730103** evaluates both the validation-selected best checkpoint
(epoch 35, common validation graph loss 2.629155270050692) and the final
300-epoch checkpoint on the exact same 13,071 held-out IDs, saving predictions.
No test-based threshold tuning or checkpoint selection is performed.

Native box-based node/edge metrics remain diagnostics against each arm's own
native targets. They are not the primary comparison with adaptive degree-two
nodes. The common protocol (`shared-guides-v2`) reads saved predictions and uses
identical full generator guides, target endpoints, and physical units for every
arm:

- Contract all degree-two chains in predictions and targets while retaining
  their geometry. Straight graphs retain chords; adaptive graphs retain their
  piecewise-linear paths; splines retain sampled curves reconstructed from the
  predicted controls. No fitted GT controls are substituted into predictions.
- Common anchor F1 and Branch F1: one-to-one role-aware physical matching
  within 1 mm. Report branch coverage, false/missing branches, component (β0)
  and cycle (β1) errors and topological-anchor count error.
- Whole-vessel curve precision/recall/F1: all predicted and reference branches,
  fixed 1 mm tolerance, uniform 0.25 mm arc-length sampling. This includes
  missing and extra geometry without requiring a successful branch match.
- Symmetric guide distance and tortuosity error on topology-matched branches;
  always report their branch coverage alongside conditional shape errors.
- Report raw predicted nodes, predicted degree-two nodes, topological anchors,
  junctions/endpoints/isolates, and internal spline controls separately. Crop
  boundary endpoints are included in endpoint counts. Controls are additional
  geometric parameters, not additional anatomical objects or object tokens.

Both native and common results are retained. Common v2 results use separate
`common-v2-summary.json` / `common-v2-per-patch.csv` files so earlier metrics
remain available. Source guides are shared, uncertain references; agreement
with them is not independent anatomical validation. Common metrics preserve
subdivision invariance and were checked on straight subdivision, curved spline
geometry, missing vessels, and the existing branch-connectivity regression tests.

The common scorer is representation-independent and uses only saved prediction
JSON and existing caches. It needs CPU scoring, not another GPU prediction run.
Per-patch source IDs remain available for later subject-level uncertainty estimates.

GPU test job 730103 **COMPLETED (exit 0)**: best and final each evaluated all
13,071 held-out patches. CPU-only job **733084** scores both saved exports with
the finalized `shared-guides-v2` protocol, preserving earlier result files.
No new GPU inference is required. Stable common-protocol commit: `9980965`;
14 focused branch/geometry regression tests passed.

## Straight-edge held-out results (13,071 patches)

CPU scoring 733084 completed successfully. Common protocol uses fixed 1 mm
anchor/curve tolerances and 0.25 mm arc-length sampling.

| Metric | Validation-selected epoch 35 | Final epoch 300 |
|---|---:|---:|
| Native node F1 | 0.5779 | 0.6534 |
| Native edge F1 | 0.4499 | 0.5449 |
| Common anchor F1 | 0.0811 | 0.1353 |
| Common Branch F1 | 0.0033 | 0.0114 |
| Common branch coverage | 0.27% | 0.94% |
| Whole-vessel Curve F1 | 0.1650 | 0.2305 |
| Mean predicted topological anchors | 15.79 | 15.49 |

Reference topological anchors average 21.39 per patch. The final model predicts
21.35 raw nodes, but only 15.49 remain topological anchors after degree-two
contraction; raw count agreement does not establish correct topology.
Matched-branch guide errors are 0.643/0.671 mm, applying only to the very small
0.27%/0.94% recovered subset, not the complete vessel graphs.
These independent physical scores indicate weak topology/geometry recovery for
this baseline. Native box-based F1 alone would overstate its reconstruction
quality. The final checkpoint scores higher on these tests despite higher
validation loss; the validation-selected checkpoint remains epoch 35. Do not
change checkpoint selection based on held-out test results. Spline/adaptive
conclusions await their completed training and identical held-out scoring.

## Spline scratch training completed; held-out tests submitted

Recovery 721673 completed successfully, finishing epoch 300 / iteration 750,000.
The original scratch run and its recovery together complete the unchanged
300-epoch protocol. The common validation-loss-selected checkpoint is epoch 170.
GPU test job **737021** evaluates epoch 170 and the final epoch-300 checkpoint
on the identical 13,071 held-out IDs. Saved predictions are scored with the same
`shared-guides-v2` topology, whole-curve geometry and object-count protocol as
the straight baseline. No checkpoint/threshold selection uses test metrics.
Actual spline geometry uses predicted endpoints and predicted internal controls,
never target controls. Native and common summaries are saved under
`scratch_spline_k8_10k_300ep/test/{best,final}`.
Stable generic evaluation update: `120419e`.
