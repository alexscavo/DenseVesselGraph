# Learning continuous vessel branches: implementation and experiment log

Date: 2026-10-05. Status: implementation and initial experiments complete. Slurm job **612351** finished successfully in **6 min 35 sec**, exit code **0:0**.

## Working copy and source plan

Worktree: `Graph-Native-Betti-Matching-Representation`, branch **representation**, based on commit `2306709`. Existing uncommitted tracked and untracked source changes were copied and verified before representation changes began. The original working tree remains separate. This is a Git worktree sharing repository history, not a second independent repository.

Design source: [representation-implementation.md](representation-implementation.md). Documentation lives in this folder; experiment code, targets, logs and outputs live in the representation worktree.

## Decisions agreed with the user

| Decision | Initial implementation |
| --- | --- |
| Dataset | SyntheticMRI only |
| Representation | Fixed clamped cubic B-spline, 8 total controls, 64 parameter samples |
| Topological nodes | Junctions, terminations, crop boundaries; retain isolated original nodes |
| Predicted geometry | Six internal 3D controls: 18 scalar coordinates per branch |
| Endpoints | Matched predicted node coordinates during learning and inference |
| Head | Parallel flat MLP, existing relation classifier retained |
| Parameterization | Physical residuals in mm; branch-relative scaling configurable for later comparison |
| Orientation | Lexicographic physical endpoint position; target reversal follows matched endpoint identity |
| V0 | SmoothL1 on fitted internal control coordinates in mm |
| V1 | SmoothL1 on sampled fitted curves in mm, including predicted endpoints |
| Matching / existence | Hungarian matching; focal edge loss on the existing classifier architecture |
| Initialization | Jointly fine-tune existing SyntheticMRI weights; matched scratch run as diagnostic |
| Initial experiment | Four training patches, no augmentation, 500 updates per initialization, batch size 4 |
| Complex topology | Detect and report unsupported closed/parallel branches; no silent branch dropping |
| Reference uncertainty | Original full generator paths are supervision guides, not perfect anatomical truth |

The pretrained network was trained with cross-entropy; this experiment uses focal edge classification as specified in the design. Existing classifier weights and architecture are retained. No Betti loss, node-type head, or new image-attention decoder is introduced.

## Existing SyntheticMRI weights were found

Checkpoint: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep/models/best_edge_f1_checkpoint.pt`.

Checkpoint inspection and adjacent saved configuration/manifest confirmed epoch 300, iteration 855000, scratch-initialized SyntheticMRI-only training on the boundary-corrected dataset, 91,200 training / 19,200 validation patches. Architecture: 120 object queries, two relation tokens, hidden width 552. It has not learned spline geometry. This is an existing SyntheticMRI initialization, not original GrEVE pretraining lineage.

Only missing `spline_head.*` tensors are allowed when initializing from old weights. Other incompatibilities remain errors, and ordinary training-state resume remains strict. The newly added head starts with zero residual output. Greville abscissae place initial controls so the curve equals the endpoint chord at every parameter value, avoiding the nonuniform parameter speed of equally spaced controls with clamped knots.

## Targets prepared before training

Data source: `/lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary` and full generator `graphs/{patient}/graph.vvg`. Source data is read-only.

The ordinary training VTP files contain chords, not full vessel paths. The new preparation script reconstructs endpoint-aware full paths, reproduces the stored version 3 closed-center crop, retains ordered path samples, and contracts degree-two chains. It checks reconstructed sparse coordinates against the stored VTP before fitting. Source affine and five-voxel padding are respected; identity patch affines are not used as physical source coordinates.

Selection before fitting: first training index rows with 20–40 stored graph nodes and more than 1,500 foreground voxels, subject to topology/target compatibility checks. Four tangent-contact candidates were rejected and recorded. No selected patch contained unsupported loops or parallel branches. All four selected patches are from subject 1; this intentionally small, related subset is for overfitting only.

| Training patch | Anchors | Branches |
| --- | ---: | ---: |
| sample_000001_0006 | 20 | 16 |
| sample_000001_0021 | 28 | 23 |
| sample_000001_0023 | 22 | 18 |
| sample_000001_0024 | 28 | 22 |

There are 79 complete branches. Their 8-control oracle fits have **0.0242 mm mean symmetric distance** to the original supplied paths, averaged across branches. This assesses fitting capacity on these guides, not anatomical correctness. All retained guide vertices fall inside the existing masks; this is a vertex check, not an independent anatomical validation or a densely sampled curve-support claim.

The fitted controls and sampled fitted curves provide V0/V1 targets. Full original guide samples are also retained for inspection and future geometry evaluation. The target cache stores physical metric metadata and checks graph/control alignment on loading.

**The `val` folder in this diagnostic cache mirrors the same four training patches. It is an overfit monitor, not held-out validation.** Original dataset splits were not changed.

Artifacts:

- [Target manifest](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/targets_k8/manifest.json).
- [Oracle metrics](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/targets_k8/oracle.csv).
- [Offline interactive target viewer](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/targets_k8/targets.html).
- [MRI target overlays](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/targets_k8/target_overview.png).

## What was implemented

- `models/spline.py`: fixed differentiable cubic basis, canonical orientation, parallel residual head. Internal controls are not clipped or passed through a sigmoid.
- `models/relationformer.py`: optional spline head; classifier preserved, existing shared initialization unchanged when disabled.
- `data/spline_targets.py` and `scripts/prepare_spline_targets.py`: full-path crop, topology contraction, endpoint-constrained least-squares fits, target cache and oracle/provenance records.
- SyntheticMRI loader / engine: pass geometry metadata through the existing six-part batch interface. Cached native-size targets are checked against the graph. Geometry augmentation is explicitly rejected until correctly implemented.
- `training/losses/criterion.py`: positive matched relation geometry supervision independent of classification sampling/caps; V0/V1 selectable; physical loss and separate control/curve/endpoint error logs. Empty geometry batches retain a valid zero-gradient path.
- Inference / export: predict splines only on retained edges, return ordered controls and curves alongside the graph. Preserve endpoint anchoring.
- `configs/overfit_spline_v0.yaml` and `overfit_spline_v1.yaml`: distinguish loss variants. The full model remains trainable.
- `scripts/overfit_spline.py`: bounded calibration, smoke, training and offline prediction exports, with both matched-positive diagnostic and actually decoded graph views.
- `cluster/jean_zay/spline_overfit.slurm`: one H100 allocation, ordered smoke/overfit stages, conditional V1.

The existing generic graph metrics still use chord geometry. New matched-positive curve/endpoint diagnostics are reported separately; generic SMD must not be interpreted as evaluating the learned splines.

## Spline loss weight

The total includes the existing weighted graph losses plus `lambda_spline * L_spline`. SmoothL1 uses beta = 1 mm; V0 averages coordinate components and internal controls over positive branches, V1 averages coordinate components and 64 curve samples over positive branches.

Before optimization, a training-only calibration measures the gradient norms of the existing weighted graph loss and unweighted spline loss at shared decoder tokens for both initializations. The common starting weight is the geometric mean of their graph/spline ratios, clipped to [0.001, 100]. It is fixed thereafter and reused by both V0 runs. Raw losses, gradient norms, unclipped ratio and final weight are saved. Calibration uses no held-out data, performs no optimizer updates, and is a scale heuristic, not evidence of an optimal weight. V1 uses its own recorded calibration because its loss differs.

## Verification before submission

- Fixed Torch spline basis agrees with SciPy; zero residual gives an exact chord; endpoints remain anchored.
- Swapped matched query identities and reversed physical orientation give equivalent supervision.
- Both V0/V1 losses backpropagate into predicted endpoints and the spline head; empty positive cases are handled.
- Existing baseline training-step test passes. An initially discovered projection-slot compatibility issue was fixed by accepting only explicit spline metadata dictionaries.
- Actual loader batch verified: `[4,1,64,64,64]`, four anchor graphs, 79 branch targets with `[edges,8,3]` controls.
- MRI target overlays inspected directly; targets checked before any GPU launch. Browser WebGL rendering is not tested on the cluster.

## Launch protocol and interpretation

One H100, development QoS, 45-minute bound. First calibrate V0, then three optimizer-step smokes for pretrained and scratch. Both smokes must pass before either 500-update run starts. Initializations use the same four patches, seed, architecture, loss weight and optimization schedule. Outputs are separated under `experiments/spline_learning/runs/job-JOBID/`.

V1 is conditional: a V0 run must reduce mean corresponding fitted-curve point error by at least 50% and reach at most 0.5 mm on these training patches. Prefer pretrained if it passes; otherwise scratch if it passes. This is an explicit exploratory learning gate, not an anatomical or generalization threshold. V1 starts from the same **original** initialization, not V0-trained weights, for an interpretable loss comparison. If neither passes, V1 is deferred and failures investigated.

Prediction exports include actual decoded node/edge decisions and a separately labelled matched-positive geometry view. Final `model.pt` files are model-only initialization/export weights, not complete optimizer-state resume checkpoints.

Generated measured results and next-step notes will appear in [spline-learning-results.md](spline-learning-results.md). They are written from completed run summaries, never inferred from queued jobs or oracle fits.

## How to interpret poor performance

Good oracle fits but failed learning point toward supervision, matching, optimization or architecture problems. Scratch success with fine-tuning failure implicates initialization/optimization; it does not show the representation is invalid. Both failing on four patches is still insufficient to reject the approach. Endpoint error is logged separately so shape learning is not confused with node localization. Fine-tuning success is not evidence of held-out generalization.

## Next steps

1. Inspect smoke results and both V0 curves/topologies; compare initialization effects.
2. Inspect the conditional V1 loss comparison if its gate passes.
3. Check held-out subjects and more tortuous branches with original guides treated as uncertain references.
4. Compare 6/8/10 controls and physical versus relative residual scaling under a fixed evaluation protocol.
5. Implement correct geometry augmentation, then loops/parallel branches before broader training.
6. Consider the per-control shared MLP only if the flat head remains a demonstrated bottleneck; image-conditioned attention follows later.

## Reproduction

From the representation worktree, after sourcing the cluster environment:

```bash
python scripts/prepare_spline_targets.py
python scripts/render_spline_targets.py
python -m unittest -v tests.test_spline tests.test_training.TrainingTests.test_training_step_updates_model_and_scheduler
mkdir -p experiments/spline_learning/logs
sbatch --output=experiments/spline_learning/logs/%j.out --error=experiments/spline_learning/logs/%j.err cluster/jean_zay/spline_overfit.slurm
```

The Slurm script supplies the required data/output environment variables and records the calibrated weights and resolved run settings. Source hashes and selected source rows are stored with the target cache.

## Submitted job

Slurm job **612351**, submitted on 2026-10-05. Log files: `DenseVesselGraph/experiments/spline_learning/logs/612351.out` and `.err`. Source hashes and base commit are recorded in `experiments/spline_learning/submission.json`. Results will be under `experiments/spline_learning/runs/job-612351/`. No training results are claimed at submission.

## GPU calibration and smoke checks

Job 612351 completed both three-update GPU smokes. Losses and spline gradients were finite and nonzero. The fixed common V0 spline weight is **1.4739989613**, obtained by the documented calibration. Both initializations are reset before their actual 500-update runs. Calibration details: `experiments/spline_learning/runs/job-612351/calibration_v0.json`.

The target oracle maximum branch mean distance is 0.1784 mm and maximum guide tortuosity is 1.1969; the 0.0242 mm aggregate should not conceal branch variation. Source code used for submission is archived in `experiments/spline_learning/submission-source.tar.gz`.

## Completed first experiment

Both GPU smokes and all three 500-update runs completed successfully. V0 fine-tuning / V0 scratch / V1 fine-tuning yielded mean matched fitted-curve point deviations of **0.1093 / 0.3811 / 0.1311 mm**, respectively. V1 weight: **1.4785307154**. All three decoded training graphs recovered the supplied topology under spatial node assignment, and static projections were inspected. The same-predicted-endpoint chord errors are approximately **0.48 mm**, so improvement includes learned shape rather than just improved endpoints.

These results support proceeding beyond the initial sanity check. They do not validate anatomy or generalization. Fine-tuning is stronger than scratch at this update budget; V1 is not better than V0 here. The next experiment should use a distinct set of training patches and genuinely held-out subjects, with exact full-path guide metrics and topology metrics evaluated separately. No larger run was launched.

[Measured results, comparison viewer, static previews and next steps](spline-learning-results.md). Additional inspection/viewer scripts: `scripts/inspect_spline_predictions.py` and `scripts/build_spline_comparison.py`; these analyze completed outputs and do not modify training.

## Visualization follow-up: vessel meshes

At the user’s request, existing mask surfaces were added to the combined comparison, individual prediction viewers, and target viewer. Marching cubes at segmentation level 0.5 produces each mesh; vertices use voxel-index/axis-size normalization and the same source-derived physical metric as the curves. Meshes are not predictions or independent anatomical references. Visibility/opacity controls keep nodes and curves inspectable. The combined viewer embeds each of the four patch meshes once and remains offline (~9.8 MB). Target viewer also provides a mesh opacity slider and legend toggling.

No models were retrained and completed prediction JSON/metrics were unchanged. Updated visualization/export code makes future exports include the mask surface. Browser/WebGL rendering remains untested on the cluster; mesh indices and coordinate bounds were checked.

Mesh viewer verification: visibility toggle, opacity changes, physical coordinate transformation, and run/patch switching passed a JavaScript logic check with mocked DOM/Plotly. All individual prediction viewers and the combined viewer were regenerated from completed outputs without retraining.

## Saved-model evaluation and training recap (2026-10-05)

At the user's request, Slurm job **613399** ran `evaluate.py` on all three final `model.pt` files. Completed in **1 minute 14 seconds**, exit `0:0`. Submission script: `DenseVesselGraph/cluster/jean_zay/spline_evaluate_four.slurm`. Each call uses the saved resolved config, `--dataset synthetic_mri --split val --max-samples 4 --batch-size 4 --workers 0 --device cuda`. The cache's `val` is the same four training patches; this evaluation is explicitly an overfit test, not held-out testing. No BN recalibration or threshold tuning was performed.

All three runs achieved node/edge F1 and node/edge mAP **1.00**, with 98 true positive nodes and 79 true positive edges, zero false positives/negatives, and zero absolute Betti-0/Betti-1 error. Protocol: F1 IoU threshold 0.5; AP IoU thresholds 0.50–0.95, target node size 0.2 and edge half-width 0.1 in normalized coordinates. These box-based detection metrics do not measure spline tortuosity. Standard SMD samples straight graph edges in normalized coordinates, not the spline curves: V0 fine-tuned 0.0001615496, V0 scratch 0.0001617138, V1 fine-tuned 0.0001640488.

Outputs are in each run's `evaluation_four_patches/`: `summary.json`, `metadata.json`, `per-patch-metrics.csv`, `predictions.json`, and `resolved-config.yaml`. A comparison with the earlier viewer exports found exactly identical decoded edges, nodes and spline curves (maximum absolute numerical difference zero); recorded in `viewer_consistency.json`.

Training recap: each run used **500 updates**, batch size four, the same four MRI patches every update (500 passes over this tiny dataset), without augmentation. All parameters were trained jointly using AdamW, base LR 7e-5, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8; five warmup updates from LR 1e-9, then polynomial decay with power 0.9. Seed 364505, 120 object queries, two relation tokens. Each 500-update loop took about **85 seconds**; the full training job including calibration, smokes and exports took **6 minutes 35 seconds**.

The two fine-tuning arms started independently from the existing SyntheticMRI-only 300-epoch graph checkpoint (855,000 previous updates), with a new zero-initialized spline output head; scratch V0 started from random model weights. V1 did not continue from V0. Loss weights: boxes 3, node class 4, cardinality 0.8, node coordinates 2, edges 6; spline **1.4739989613** for V0 and **1.4785307154** for V1. Spline weights were calibrated once from shared-token gradient norms before training and then fixed. Node class uses weighted CE [0.25, 0.75]; edge class uses focal loss gamma 2, class weights [1, 1], without balancing. V0 spline SmoothL1 supervises internal control components; V1 supervises sampled curve components, both in mm with beta 1 mm. No Betti, degree, node-type or domain loss was enabled.

The comparison viewer's cyan curves in **Decoded graph** mode are fully decoded predictions. **Matched positive pairs** mode uses predicted nodes and geometry but supplies target edge identities to isolate shape learning. Gray fitted targets, gold original guides and the segmentation mesh are references; target-only/oracle viewers do not display learned model predictions.

## Approach selected after verification

**V0 control supervision with joint fine-tuning** is selected for the next main experiment; V1 remains a loss ablation. Additional checks against the original full-path guides favor V0 (symmetric distance 0.1010 versus V1 0.1214 mm; tortuosity absolute error 0.01013 versus 0.01128). Scratch V0 has guide distance 0.3498 mm and tortuosity error 0.02459, slightly worse on the latter than a chord (0.02286). These are four-patch training diagnostics, not held-out results. [Verification, rationale and requirements for a paper](spline-approach-selection.md). No larger training run was launched.

## Expanded experiment requested

The user authorized training on SyntheticMRI with at most 10,000 training patches, then evaluation on the separate test subjects, and requested validation projections in W&B. The initialization audit found no overlap with test subjects. Boundary supervision is being aligned to the requested last-inside-centerline-sample policy, using voxel-cell bounds and a recorded sparse-component fallback to preserve crossing segments. [Full protocol, boundary explanation, implementation and launch log](spline-10k-training.md). This is a separate target cache; four-patch evidence is preserved.

## Main-run resubmission (2026-10-05)

Authorized graph-complete subjects only; source changed from 109 to 27 complete subjects, so preserved those graphs locally. Training cap 10,000, fixed validation cap 2,000 every five epochs, uncapped available test (4,800 candidates / five subjects). Accepted counts await GT compatibility filtering. Repeated GPU smoke omitted at the user's request; 615708 passed earlier. Replacement preparation → training → test jobs: **625282 → 625284 → 625312**, with success dependencies. [Latest source population, cancellations and protocol](spline-10k-training.md). No held-out results yet.

## Dataset archive recovery

Located today's 44 GB dataset archive in the other account's STORE. The user requested replacing their incomplete SyntheticMRI folder with the archive contents. Restore job **627540** extracts into staging, verifies the boundary-patch population and original volume transforms, then replaces the folder while retaining the old directory as a backup. [Restoration procedure and verification](synthetic-mri-restoration.md).

Dataset restoration **627540 succeeded**. All 130,560 boundary triplets and 136 full segmentation headers verified; 91 complete archived graph subjects. The working SyntheticMRI folder was replaced and the previous folder retained. Verified full-volume transforms are copied to WORK.

## Restored-data run submitted

User authorized resumption. **646175 → 646177 → 646195**: GT preparation → joint training → held-out test, gated on successful preceding stages. Uses all 91 graph-complete restored subjects (64 train / 13 val / 14 test), preserving original splits; max 10,000 training, fixed 2,000 validation and uncapped available test candidates. Zero initialization exposure to test subjects reverified. 50 epochs, validation/projections every five epochs, no repeated GPU smoke. [Current protocol and submission log](spline-10k-training.md).

## Preparation fixes and storage cleanup (2026-10-06)

Real patch sample_000001_0272 exposed a tiny closed guide A→B→A (source edge 806, node 2488), which generic simple-graph contraction rejected as duplicate edges. Whole-patch closed/parallel eligibility is now checked before contraction; targeted real and synthetic reproductions pass. The next preparation failure was WORK's 500,000-file quota; 88 completed patient caches were verified and relocated to SCRATCH. An incomplete PyVista installation was repaired at the same 0.46.3 version. Current resumed chain: **676631 → 676650 → 676692**. Stable implementation/restoration commits: **24c2605 / 98eb6a9**. Full details and test populations remain in the protocol report.

Main GT preparation **676631 completed**: 9,709 train, 1,952 val, 13,071 test targets. Whole-patch closed/parallel exclusions are recorded; available-test eligibility is 97.25%. Training **676650** is normally queued for GPU priority, test **676692** waits for successful training. Exact manifests/membership copied to WORK; targets/runs on SCRATCH. No held-out model results yet.

Archived historical pilots/snapshots/prototype runs in STORE and simplified the active layout to current dataset, runs, logs and metadata. Historical files remain available for reproduction without cluttering projects. Training 676650 started; W&B is recording offline as pnbftfvi after compute-node connection failure.

Training 676650 completed nine epochs, then failed on missing `medpy.io.header`. MedPy 0.5.2 repaired without dependency changes; preserved runtime packages outside projects and staged them locally per job. Real MRI/checkpoint verification passed. Strict resume **681009** from epoch 9 / iteration 21,852; test **681013** waits for success. No optimization/data/loss settings changed. W&B ID pnbftfvi retained offline. Commit **81b21ec**.

Completed required-library audit: **86/86 imports passed** in a repaired preserved private runtime. Repairs: MedPy 0.5.2, Matplotlib 3.8.4, POT 0.9.7, VTK 9.5.2, timm 1.0.22, W&B 0.28.1. Actual active module versions/paths recorded, including cluster-module dependencies. Train/evaluate imports, real MRI, PNG/W&B Image and strict epoch-9 model checkpoint loading passed. Final fixed archive checksummed; resume **687805** queued, test **687813** waits for success. No repeated GPU smoke. Patch-count arithmetic/exclusion reasons explained in the training report.

Training completed all 50 epochs (687805, exit 0:0); both offline W&B segments uploaded and cloud epoch-50/ten validation events verified. [Finished training in W&B](https://wandb.ai/alessandrobenvenuti2002-politecnico-di-torino/focal-loss/runs/pnbftfvi). Test 687813 was waiting for GPU priority at the latest scheduler check.

Held-out evaluation completed all **13,071** eligible patches with the validation-selected checkpoint. Job 687813 failed only afterward during W&B initialization; saved population/counts verified and logging recovered online from login, without rerunning GPU inference. Standard box-based node/edge F1: **0.8841 / 0.8157**. Separate strict 1-mm branch correspondence coverage is **17.41%**; matched-only curve/guide errors (0.7370/0.6396 mm) must be read alongside that coverage. Full results and definitions are in the training protocol report. Future compute-node test logging defaults offline (commit f1725bb).

Created two offline 3D viewers from the saved fully decoded best-checkpoint test outputs: subjects 20/24, patches 0470/0478. Fixed selection independent of performance; vessel mesh, references, predicted nodes/splines and optional control polygons. Exported geometry preserved, target transforms/endpoints verified, MRI projection alignment inspected, and viewer control logic checked. [Viewers and provenance](synthetic-mri-test-predictions.md).

## 2026-10-06: new scratch comparison

The user requested two 300-epoch scratch runs with exactly 10k shared training
patches and identical validation. They clarified that both arms should retain
only junctions, terminations and boundary nodes, with straight edges versus
8-control spline edges. No degree-two variant is submitted in this pair.
See [the complete protocol](scratch-comparison-300ep.md) for membership,
initialization, losses, common validation checkpoint selection and next steps.
This comparison tests the additional spline geometry while keeping endpoint
and connectivity supervision identical; it does not test degree-two removal.

### Additional compact adaptive scratch arm

The user requested a third, compact degree-two-node arm with the same data and
300-epoch schedule. A small audit required preserving uncertain original guide
segments when strict mask containment is infeasible; no samples/endpoints are
removed. Full preparation and capacity/data checks gate its GPU training.
Protocol and decisions: [adaptive comparison](adaptive-degree2-scratch-comparison.md).

## 2026-10-07: Catmull–Rom single-case alternative

The user requested an interpolating curve alternative in the original IXI and
SyntheticMRI single-case simulations. Both standalone viewers now include
centripetal Catmull–Rom with 6/8/10/12/16 total points, an equal-budget B-spline
overlay and editable on-curve interpolation points. Original embedded image,
mask, mesh, guides, anchors and every non-Catmull fit were verified unchanged.
These are fitting demonstrations, with no training/dataset/checkpoint changes.
See [the Catmull–Rom report and download links](catmull-rom-single-cases.md)
for construction differences, measured case metrics and next steps.

## 2026-10-08: all three held-out comparisons complete

All three trainings and their best/final checkpoint tests completed. Created
[comparison tables](scratch-comparison-results.md) and a full-precision CSV
with source checksums. All six evaluations use the exact same 13,071 test IDs
and `shared-guides-v2` thresholds. The report separates native diagnostics
from common topology/whole-curve geometry and object/geometry counts.
The current spline run has essentially input-independent saved predictions
and no recovered branches; this failure requires diagnosis before interpreting
representation feasibility or launching another expensive spline experiment.
All models' physical topology recovery is weak at the fixed 1 mm tolerance.
Final adaptive curve F1 is higher than straight edges, while straight-edge
anchor/Branch F1 is higher; this single seed does not establish significance.
No checkpoint selection was changed based on these test results.

## 2026-10-08: scratch spline collapse diagnosis

[Diagnosis, bounded checks, and proposed remedies](spline-collapse-diagnosis.md).
Eight-patch GPU check 762225 found no recovery when switching from saved BN
statistics to batch statistics; all three checkpoints load correctly.
The spline encoder remains input-sensitive but decoder tokens are effectively
constant. Poor training losses predate the interrupted/resumed job. One
initial training batch shows a 7.49-fold spline/graph gradient ratio at node
coordinates. Four-patch attention probe 762449 localizes loss of image
dependence inside the transformer encoder. Restoring supported decoder
sampling in memory does not rescue predictions; poor attention coverage is
also present in the straight baseline, so it is not a sufficient explanation.
Proposed next step: a short controlled comparison of the current versus
tenfold-reduced spline weight, then consider warmup or endpoint-gradient
isolation if needed. No long training or full re-evaluation was launched.

## 2026-10-08: implement the paired spline-weight trial

The user authorized the recommended next step. Implemented a bounded runner
reusing existing training/evaluation code, with weights 1.474 versus 0.1474.
Both use 512 training patches from all 64 training subjects, 128 validation
patches from all 13 validation subjects, matched random initialization and
5,000 updates. The original LR schedule is preserved by update count.
[Protocol, implementation, checks, and results](spline-weight-pilot.md).

Both tasks of array 763233 completed successfully. Exact initialization,
membership, batch-order and schedule pairing was verified. At 5,000 updates,
both have native node F1 and common branch F1 zero on 128 validation patches.
Reducing the spline weight improves validation node loss slightly but does not
rescue decoded reconstruction (common curve F1 0.00145 versus 0.00081).
Both remain input-sensitive, so the eventual 300-epoch collapse is not
reproduced. This short budget is only two original full-data epochs; the old
graph run also had high training node loss at that point. Do not promote the
lower weight as a fix. Next proposal is a graph-only learning control followed
by delayed/ramped spline supervision once graph prediction actually learns.

## 2026-10-08: staged continuation from learned graph weights

The user correctly recalled successful pretrained spline learning. Rechecked
the earlier 50-epoch full fine-tuning result: native test node/edge F1
0.8841/0.8157, so the evidence extends beyond the original four patches.
The unresolved issue is the current scratch optimization. Submitted paired
array 775025: start from the same-data graph checkpoint at epoch 35, compare
graph-only continuation against a 2,500-update ramp of spline supervision,
for 5,000 updates on the fixed diverse subset. Both restart AdamW moments
while using the LR schedule at update 87,500. Add same-endpoint chord scoring.
[Protocol and measured results](spline-staged-learning.md).

Staged pilot completed: graph control 775308_0 (10m45s), staged spline
775025_1 (10m26s). Pairing checks passed. Both retain graph detection
(native node F1 about 0.564) and input sensitivity. Common curve F1 is
0.15118/control versus 0.15400/spline, but the spline's own chords score
0.15448: no demonstrated learned-shape advantage. Both regress slightly from
initial validation scores. Do not treat this as a solved geometry-learning
problem or launch a long run on these results alone. Next targeted option:
freeze graph prediction and assess spline-head learning in isolation.

## 2026-10-08: redirect toward joint connectivity and geometry

The user clarified that a frozen wrong graph does not meet the objective.
Implemented optional MRI evidence sampled along predicted candidate splines
before deciding connectivity. Keep all node, relation, and geometry parameters
trainable. Added a role-agnostic endpoint/link diagnostic, focused gradient and
coordinate tests, and a paired image-evidence on/off pilot with otherwise
identical settings. Verification job 778675 precedes pilot submission.
[Design, protocol, verification, and results](curve-informed-connectivity.md).

Curve-evidence implementation verified: 13 tests and GPU smoke job 778675 passed. Paired joint fine-tuning pilot submitted as 779095_0/1 (5,000 updates each); see [protocol and rationale](curve-informed-connectivity.md). No claim of prediction improvement yet.

## 2026-10-09: Catmull–Rom scratch comparison

Implemented eight-point centripetal Catmull–Rom with the same head size, original splits, 300-epoch scratch schedule, and control loss weight. New targets use equal physical arc-length guide points; original graphs and caches are preserved. Preparation 805038 produced all caches; GPU check 805044 gates training 805047. Local tests passed. See [implementation, protocol, jobs, and next steps](catmull-rom-learning.md).

## 2026-10-09: evidence review and frozen-head diagnostic prepared

Reviewed original/fine-tuned/scratch, BN, attention, weight, staged, and curve-evidence results. Encoder collapse is a verified final-state symptom; its initiating cause remains unproven. The frozen-detector experiment has not previously been performed. Prepared one 5,000-update head-only trial from the original successful straight checkpoint, fixed 512/128 membership, unchanged absolute-control objective, paired chord and separate GT-matched/decoded evaluation. Verified CPU tests and real checkpoint compatibility; no GPU job submitted, awaiting the user’s explicit approval. See [review, ranked hypotheses, protocol, files, and decision rules](frozen-spline-investigation.md).

Frozen-head diagnostic authorized and submitted as **812623** (GPU smoke followed by fresh 5,000-update trial; one H100, 30-minute limit). Results pending. See [protocol and submission](frozen-spline-investigation.md).

Frozen-head 812623 failed before updates because of a train-split evaluation-loader restriction. Fixed the diagnostic loader and resubmitted as **813146**, preserving the failed attempt under its original path; new output root ends `_attempt2`. Separate real-sample check skipped at user request; existing GPU smoke retained.

Frozen-head **813146 completed** (14m17s, 5,000 updates). Exact detector and assignment invariance verified across 512 train/128 validation patches. Matched-guide error improves 40.5% on training but worsens 8.3% on validation; decoded curve F1 0.8340/train and 0.5857/validation versus own chords 0.6481/0.6554. Slight early validation gain precedes overfitting. Endpoint-neutral results suggest absolute-control fitting partly compensates endpoint error. Recommend a matched frozen residual-objective diagnostic next; not implemented or submitted. [Full results and interpretation](frozen-spline-investigation.md).

## 2026-10-09: parallel frozen residual/token diagnostics

Submitted **815195_0/1/2**: residual/full tokens, absolute/endpoints only, residual/endpoints only. Reuse completed absolute/full control 813146. Same seed, checkpoint, 512/128 membership, head initialization, 5,000 updates, optimizer and monitoring. Added an optional replacement residual-control objective and token masking only at the geometry head; defaults unchanged. Eleven CPU tests passed; each task is gated by a separate GPU smoke. Stop for interpretation after the frozen comparison; no unfreezing or 300-epoch job launched. [Full protocol](residual-token-diagnostics.md).

Frozen factorial **815195_0/1/2 completed** (about 14 minutes each). Pairing/freeze checks passed. Residual/full learns endpoint-neutral training shape (50.7% better than chords) but still fails validation (curve F1 0.6159 vs chord 0.6554). Endpoint-only arms stay near chords (0.6558/absolute, 0.6496/residual validation F1), removing both large training fit and substantial validation harm. Residual loss reduces localization compensation but does not solve transfer; original scratch-collapse cause remains unproven. Broader-data frozen residual pilot proposed for discussion; no further training submitted. [Tables and interpretation](residual-token-diagnostics.md).

## 2026-10-09: broader-data frozen residual diagnostic

User authorized the follow-up; submitted **816293** (one H100, smoke then 5,000 updates). Expand training from 512 to the existing 10,000 samples; preserve 128 validation cases and evaluate training geometry on the original 512. Reuse completed residual/full 815195_0 as control. Same initialization, objective, LR, optimizer and update budget; detector/BN frozen. Runtime asserts fixed conditions and evaluation assignments, while explicitly allowing changed training membership/order. Five CPU tests and membership checks passed. Results pending; stop afterwards for interpretation. This tests data breadth, not the initiating cause of original scratch encoder collapse. [Protocol](residual-token-diagnostics.md).

**816293 completed** in 14m19s. Fixed conditions, evaluation membership and detector invariance verified. With 10k patches, validation curve F1 is **0.67865**, versus same-endpoint chord **0.65540** and 512-patch residual control **0.61590**. Raw guide error improves 3.66%, endpoint-neutral geometry 6.07% versus chords. Training-subset curve F1 is 0.67275; the large prior train/validation gap is reduced. This supports modest transferable geometry from frozen features with broader patch coverage; connectivity stays unchanged. Single seed/128 validation cases, no test scoring or significance claim. The cause of original scratch collapse remains unproven. Recommend discussing a bounded frozen-head continuation before unfreezing; no new job launched. [Detailed results](residual-token-diagnostics.md).

## 2026-10-09: parallel frozen continuation and objective comparison

User authorized another bounded run plus a parallel diagnostic; submitted **817904_0/1**. Task 0 continues the successful 10k residual head for 10,000 additional updates (15k cumulative), explicitly restarting AdamW because source checkpoint lacks optimizer state. Task 1 trains a fresh absolute-control head for 5,000 updates against the completed residual 816293 control with the same initial tensors and batch sequence. Both keep the successful pretrained detector/BN frozen, 10k train/128 validation and original 512 training monitoring cases. Five CPU tests and checkpoint/configuration checks passed; each task has a GPU smoke gate. No new scratch or 300-epoch training; compare absolute versus residual at the matched 5k budget, separately assess continued residual learning. [Protocol and limitations](residual-token-diagnostics.md).

**817904_0/1 completed**, 17m51s/13m58s, exit 0:0. Matched absolute/residual 5k validation curve F1: **0.67461 / 0.67865**; endpoint-neutral shape improvement versus chords: **−3.64% / +6.07%**. Residual continuation reaches **0.68995** validation curve F1 and **10.27%** endpoint-neutral improvement at 15k cumulative updates (raw guide error 5.97% better than chords). Detector, BN, matching and connectivity remain unchanged; continuation starts exactly at source final metrics. Evidence supports the residual pretrained/frozen route, but does not establish the cause of scratch collapse. No new run launched; discuss controlled unfreezing versus graph-only continuation if moving to connectivity/joint optimization. [Tables and interpretation](residual-token-diagnostics.md).

## 2026-10-09: controlled unfreezing authorized and submitted

Submitted **823645_0/1/2**, 5,000 updates each from the successful residual 15k checkpoint: graph-only detector gradients with detached head inputs, joint token gradients with detached endpoints, and fully joint geometry gradients. Same 10k/128 membership and 512 training monitoring cases; unfreeze transformer/projection/queries/graph heads at LR 7e-6, geometry MLP at 7e-5. CNN and BN remain fixed; dropout disabled in every arm. All arms get fresh AdamW. Seven CPU tests passed; each GPU task has a smoke gate. Save geometry versus same-endpoint chords, full decoded graphs, matching changes, image sensitivity and sparse gradient alignment probes. No scratch/full-network/300-epoch training. [Protocol, hypotheses, implementation and interpretation](controlled-unfreezing.md).

Controlled unfreezing **823645_0/1/2 stopped before training**: all GPU smokes passed, but the initial training curve-F1 equality assertion rejected a 1.138e-6 numerical difference. Saved predictions have identical query IDs/edges; normalized coordinate/curve drift is <=5.96e-7. Raised only diagnostic metric tolerance from 1e-6 to 1e-5, verified against saved evaluations, and resubmitted **824964_0/1/2** to a separate `_attempt2` output root. No trained results yet; original artifacts preserved. [Failure evidence and fix](controlled-unfreezing.md).

**824964_0/1/2 completed**, 16–18 minutes each. Pairing and frozen-state audits passed. Validation edge F1: graph-only detector **0.87126**, token-coupled **0.86380**, fully joint **0.86197** (initial 0.85858). Curve F1: **0.68420 / 0.68857 / 0.68195** (initial 0.68995). Transformer image sensitivity stays near 0.920: no collapse. Endpoint-neutral geometry stays near its initial value; direct endpoint coupling has no demonstrated advantage. Conservative working direction: graph-only detector updates plus residual head on detached inputs, retaining initial frozen checkpoint as the better curvature reference. Single-seed validation trade-off, not a combined topology/shape win or an explanation of scratch-collapse onset. No new jobs launched. [Full comparison and decision](controlled-unfreezing.md).

## 2026-10-09: full-validation comparison submitted

User authorized evaluating the three candidate checkpoints on all **1,952 validation patches**. Submitted **826121_0/1/2**: frozen residual 15k head, graph-only detector plus detached head, joint tokens with detached endpoints. Identical ordered IDs/protocols, no BN recalibration or training, no test-set evaluation. Store native/common connectivity and geometry, same-endpoint chords, GT-matched geometry and six fixed cross-model visualization cases. The last task automatically creates CSV/Markdown comparison tables and a portable offline HTML gallery with curves/chords toggle. Checkpoint/configuration/membership/import/syntax checks passed; each GPU task has a two-case smoke gate. [Protocol, jobs and download location](full-validation-comparison.md).

**826121_0/1/2 completed**, about 10 minutes each. Full 1,952-case paired audit passed; 31,934 GT-positive branches, unchanged model states, no test scoring. Frozen / detached / endpoints-detached validation edge F1: **0.86363 / 0.87198 / 0.87096**; curve F1: **0.68441 / 0.67955 / 0.68296**; common branch F1: **0.27625 / 0.29183 / 0.29236**. Endpoints-detached is a tentative compromise, not an overall winner: frozen has better curve/node F1 and component-count error. All retain roughly 10.5% endpoint-neutral shape improvement vs chords, but inspected cases still miss substantial bends. The portable 9.8 MB HTML gallery and comparison tables are ready; all 36 embedded images verified. No new training submitted. [Full results and download path](full-validation-comparison.md).

## 2026-10-10: local geometry information and dataset-curvature caveat

User authorized testing local image/CNN information for the geometry head and highlighted SyntheticMRI's limited bends. Audit of long fitted targets in the 128-case validation subset: median tortuosity **1.0159**, only **2.5% >1.10**. The existing IXI case has much stronger bends; it is illustrative, not population evidence. Small SyntheticMRI gains alone should not diagnose missing information. Keep the matched SyntheticMRI pilot with curvature-stratified geometry metrics; IXI needs a separate protocol that avoids confounding poor graph detection/domain shift.

Submitted **871744_0/1/2**, 5,000 updates each from the successful frozen residual source: inactive token control, early CNN feature correction, identical correction with zeroed local image values. Added 37,778 optional geometry parameters; zero output initialization preserves curves. Sample 32 positions on current predicted curves; detector and BN fixed, no GT inputs to sampling, unchanged residual loss/data/batch sequence. Eleven initial CPU tests passed; eight relevant checks passed after strata; real model/checkpoint construction passed. Each task has a GPU smoke gate. No dataset extraction, scratch/full training or test evaluation. [Why, implementation, evidence, constraints and interpretation](local-geometry-diagnostic.md).

Local-geometry **871744_0/1/2 GPU smokes passed**, three real updates each with detector unchanged; jobs running. SyntheticMRI curvature caveat and separate long-curved-branch reporting are recorded. Full results pending; no IXI training launched.

## 2026-10-10 — Local-image geometry result and parallel follow-up

Local-image frozen geometry reached validation curve F1 0.8454 versus token/zero-feature controls 0.6905/0.6905 on 128 diagnostic patches; detector and assignments unchanged. Full evidence: [local geometry diagnostic](local-geometry-diagnostic.md). User authorized full validation and the next controlled unfreezing concurrently. Submitted arrays 875855 (three heads, 1,952 validation patches and fixed gallery) and 875856 (three matched gradient routes, 5k updates from the successful local-feature checkpoint). Six targeted tests passed; independent GPU smoke gates precede all full tasks. Stay on SyntheticMRI and defer IXI per user. Protocol and interpretation: [local geometry follow-up](local-geometry-followup.md).

## 2026-10-10 — Selected longer-stability continuation queued

User authorized selection and extension together. Full-validation candidate array 876609 precedes deterministic selection and a 25k-update continuation (876621), preserving optimizer state, with periodic diagnostic validation and snapshots. Final full validation 876622_1 produces before/after tables and gallery using the existing selected baseline. Selection favors branch F1 subject to declared geometry/node preservation guards. Five targeted tests passed. Complete protocol: [longer stability](local-geometry-stability.md). No new test-set evaluation or IXI work.

## 2026-10-10 — First improved scratch comparison

The pretrained continuation train/validation trends support connectivity overfitting (train branch F1 0.321→0.392; diagnostic val 0.271→0.269), rather than collapse. User authorized proceeding to scratch. Submitted 878748_0/1: graph-only versus residual local-image spline, no pretrained weights, same 10k patches and random initialization seed, 25k updates each. Full CNN/BN training restored in both; geometry endpoint inputs detached, tokens trainable. Four targeted tests passed; per-arm GPU smokes gate full runs. [Protocol and interpretation](scratch-local-geometry-diagnostic.md). No 300-epoch or test-set run launched.

### Scratch comparison startup correction

878748_0/1 failed before training: my blanket trainability assertion rejected intentionally frozen, unused relation-attention tensors. Corrected the guard without changing model computation or training settings; full actual-model construction and regression checks passed. Resubmitted **885337_0/1** into `scratch_local_pilot_20261010_attempt2`. Original failed artifacts preserved.

## 2026-10-11 — Scratch gradient-isolation follow-up

Completed 885337 paired runs: graph-only detection improved; spline supervision through tokens produced near-zero image sensitivity by 15k, while train and validation node F1 remained zero. Initial and batch hashes matched. User authorized the next isolated-route experiment. Submitted **903323**, same 25k scratch protocol and positive geometry loss, detaching both tokens and endpoints from geometry; reuse completed graph control. Five targeted tests passed. [Protocol and interpretation](scratch-detached-geometry.md).

## 2026-10-12 — Detached scratch result

903323 completed. Detached geometry preserved image sensitivity and graph learning (val edge F1 0.1928 versus graph-control 0.1957), but endpoint-neutral curvature gain was only 0.31%; branch F1 remains zero. Detector trajectories differ despite paired seed/batches, so avoid claiming exact replication. Submitted narrow GPU checks 948360 for real-model gradient isolation, repeated-control numerical variation and saved-vs-batch BN image sensitivity. [Details](scratch-detached-geometry.md). No new long training submitted.

### Gradient isolation and BN checks completed

948360 completed in 1:07: detached detector geometry gradients exactly zero; three-update detached/control differences comparable to repeated-control numerical variation; switching saved to batch BN statistics does not restore collapsed image sensitivity on the fixed probe. One 5k-batch geometry/graph norm ratio was only 0.000152, so a universally excessive loss magnitude is not established. [Evidence and limits](scratch-detached-geometry.md). No new job submitted.

## 2026-10-12 — Authorized scratch extension to 300 total epochs

Submitted **949063_0/1** (epoch 10→150), **949065_0/1** (150→300, matching dependencies), and **949071_0–3** (full validation of both best/final checkpoints, two concurrent). Extend graph-only and detached B-spline; weights, AdamW and original LR schedule restored. Same 10k train/128 monitored validation patches. Two targeted continuation tests passed; each training task has a real GPU resume smoke gate. Initial environment-held submissions were cancelled and replaced using positional arguments. Estimated 20–24 h/model, parallel, plus queues. [Protocol, RNG caveat, checkpoint selection, commands and output paths](scratch-extension-300.md).

Added automatic paired **epoch-100 and epoch-200 controls** at user request. Existing detailed monitoring supplies all tests; a lightweight background service collates detection, chord/curve geometry, train/validation matched shape and image-sensitivity collapse screening, with portable visualizations. No GPU restart or extra inference. Report generation verified on real epoch-20 data; the known failed coupled run triggers the screen. [Details and report locations](scratch-extension-300.md#added-epoch-100-and-epoch-200-controls).

## 2026-10-13 — Both detached scratch runs and full validation completed

Both 300epoch trainings and all four 1952-case evaluations completed successfully before maintenance. Final control/detached node F1 **0.63383/0.64995**, edge F1 **0.51851/0.54195**, curve F1 **0.20201/0.34261**; detached same-endpoint chords **0.22633**. No encoder collapse (memory sensitivity 0.95624/0.86923). Shape overfits: full-validation endpoint-neutral gain final **-4.32%** versus selected epoch 170 **+2.37%**; final training gain 57.64%. Complete-branch F1 remains very low 0.00793/0.01179. On 128 monitored val cases only, curved long branches improve 22%, near-straight/short branches worsen. No new GPU tests required to review termination. Scheduled 07:00 documentation failed on incompatible CLI flags; fixed and fulfilled during the active review. [Full tables, provenance, limitations and next decisions](scratch-extension-300.md#final-results--2026-10-13).
