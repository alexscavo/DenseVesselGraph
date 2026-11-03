# V0 SyntheticMRI training with up to 10,000 patches

Date: 2026-10-06. Status: training completed all 50 epochs; all 13,071 compatible test patches evaluated. Test job 687813 failed only during W&B initialization; saved results verified and logging recovered from the login node.

## Protocol fixed before training

- Main method: cubic B-spline, eight controls (six free internal 3D controls), fixed knots, 64 samples, V0 internal-control SmoothL1 in mm. Predicted graph nodes anchor endpoints; entire model fine-tuned jointly.
- Initialization: original SyntheticMRI graph checkpoint `finetune_vessels_full_300ep/models/best_edge_f1_checkpoint.pt`, not the four-patch overfit checkpoint. The supplied training manifest contains 91,200 training and 19,200 validation samples; neither sample IDs nor patient IDs overlap the test population.
- Original dataset population: 91,200 train patches / 95 subjects; 19,200 validation patches / 20 subjects; 20,160 test patches / 21 subjects. Subject-disjoint splits are preserved.
- Training membership: rank source training IDs by SHA256 of seed, split and sample ID; select at most 10,000 before target compatibility checks. Seed 364505. No foreground, node-count, fit-quality or tortuosity filtering. Validation uses a fixed, seeded subset of at most 2,000 candidates before compatibility checks; test has no cap. Membership is restricted to subjects with preserved complete source graphs, as authorized by the user. Any unsupported patches are listed by split and reason; test metrics must report the eligible fraction of the complete test population.
- Batch four, one H100, 50 epochs, augmentation off until geometric transforms are implemented correctly. AdamW LR 7e-5, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8; one epoch of warmup followed by polynomial decay (power 0.9). This is a bounded first generalization experiment, not a claim that 50 epochs establishes convergence.
- Retain V0 graph loss weights (boxes 3, class 4, cardinality 0.8, nodes 2, edges 6), and spline weight 1.4739989613 from the training-only prototype calibration. No tuning on test data; no topology/domain/node-type losses.
- Validate every five epochs on the fixed compatible validation subset (up to 2,000 patches). Select the checkpoint with the lowest combined validation loss (`best_checkpoint.pt`), including spline loss; do not select by test scores. Keep a complete latest training-state checkpoint each epoch. No early stopping for this fixed-budget experiment.
- Final evaluation runs once after successful training, using the validation-selected checkpoint and the full compatible test split. No BN recalibration or threshold search on test data.

## Boundary correction requested by the user

The source prepared MRI patches were generated with version-3 center-coordinate crop bounds `[start, start+size-1]`. The current graph-extraction repository also contains a newer exact voxel-cell policy and the earlier last-inside-centerline-sample policy. For this spline run, the user requested the latter behavior: preserve a boundary endpoint and its incident branch instead of dropping a boundary-crossing vessel.

New targets clip full paths in voxel coordinates to the complete image voxel cells, `[start-0.5, start+size-0.5]`. Where a clipped component contains at least two distinct original samples, an interpolated boundary endpoint is moved to its nearest existing sample inside that same ordered component. This prevents selecting a point from another branch or another re-entry of the same vessel. True source nodes remain shared across incident branches.

If fewer than two usable existing samples remain, retain the positive-length clipped component with exact intersections as a recorded fallback rather than silently lose its vessel segment. A genuinely singleton contact with the full voxel-cell box is retained as an isolated boundary node, with node supervision and no fictitious zero-length spline. Counts of snapped endpoints, remaining singleton contacts and sparse-component fallbacks are recorded per split. This differs from the earlier four-patch experiment; its cache and results remain separate and unchanged.

Source index counts: 40,447 train, 8,626 validation and 9,023 test patches contained a singleton contact under the **old center-coordinate bounds** (~44%). This is not a measured failure rate of the corrected voxel-cell targets. The distinction is necessary because a boundary voxel can contain a short vessel segment even when the center-coordinate box sees only a point.

## Closed and parallel branches

A closed branch returns to the same topological anchor, or forms a pure degree-two cycle. A parallel pair consists of two distinct vessel paths connecting the same anchors. Splines themselves can describe those shapes, but the current decoder predicts one spline for each pair of distinct nodes: no self-edge output and no multiple-edge output for a pair. Preparation detects and reports these whole-patch incompatibilities instead of merging or silently dropping branches. Their prevalence is not assumed; preparation must measure it. Results on an eligible test subset must state coverage and exclusions.

## Validation visualization and W&B

Four fixed validation samples from different subjects are chosen deterministically, without predicted scores or visual quality selection. At each validation event, save a two-row, three-column PNG: original guides above, decoded predictions below, in projections along D/H/W. Background is the MRI maximum-intensity projection with translucent mask support; anchor nodes are visible. A crossing in a projection is not necessarily a junction in 3D.

Images are logged as `validation/projections` against `validation/epoch` in the existing W&B entity/project; PNGs also remain under `runs/spline_v0_k8_10k_val2k_50ep/validation-projections/epoch-XXX/`. No test examples are used for training monitoring. Validation loss scalars are retained locally in `validation-losses.jsonl` and in W&B.

## Test measurements

Use existing box-based node/edge detection and Betti metrics, plus separate physical spline metrics. Detection `max_detections` is 120 for this run so a 40-detection cap does not artificially truncate larger anchor graphs; other box/SMD settings remain unchanged. Standard SMD still uses chords in normalized coordinates.

Additional branch correspondence uses one-to-one spatial node assignment accepted only within 1 mm, then endpoint identities. Report matched, missing and extra branches, micro branch F1, and recovered target-branch coverage. Shape errors apply only to recovered branches and must always accompany coverage. Report equal-branch mean curve error against fitted targets, symmetric distance to original full guides, tortuosity absolute error, and complementary existing-mask support. Guide comparison uses 257 parameter samples and 0.25 mm arc-length query spacing with exact point-to-polyline distance. Both guides and masks are uncertain references, not independent anatomical truth. Export per-patch metrics and graph/spline predictions for later subject-level analysis.

## Implementation and launch gates

Representation worktree additions: `scripts/prepare_spline_dataset.py`, `scripts/smoke_spline_dataset.py`, `training/evaluation/spline_metrics.py`, `training/evaluation/spline_visualization.py`, `scripts/log_spline_test.py`, and `configs/finetune_spline_v0_10k.yaml`. Generic training optionally saves/logs validation projections; evaluation exports geometry and its coverage separately from standard graph metrics.

Launch stages:

1. Pilot preparation, boundary/basis/tracking tests, three real-batch GPU updates, validation loss, decoded geometry metrics, and real W&B image logging.
2. CPU preparation of capped training (10,000), fixed validation (2,000), and uncapped available test targets.
3. After successful GT preparation and manifest checks, start the 50-epoch joint fine-tuning run. The user requested skipping another GPU smoke because job 615708 already passed.
4. After training succeeds, evaluate the validation-selected checkpoint on test and attach metrics/population coverage to the same W&B run.

First pilot job 614887 failed before training because original full raw MRI volumes are no longer present for most subjects. The prepared MRI patches and full segmentations remain available. The generator required raw/seg affine equality; preparation now uses the retained full segmentation header and verifies it against every stored source VTP before constructing new targets. No affine is guessed. Corrected pilot job: **615011**. Long training has not been submitted at this point.

### Source availability problem discovered before long-run submission

Job 615011 exposed missing full graph files (not a training failure). Of the 136 graph directories, 27 do not contain the required original `nodes.csv`, `edges.csv`, and `graph.vvg`:

- Train: 2, 6, 7, 11, 17, 21, 23, 31, 33, 50, 62, 63, 65, 67, 81, 87, 89, 95, 99, 112, 124.
- Validation: 32, 77, 102.
- Test: 45, 64, 107.

The user's own SyntheticMRI directory contains empty graph stubs and is not a backup. No complete backup was found in the inspected project/store locations. The user was asked for a restoration/archive path. Original curved paths cannot be recovered from the chord-only patch VTPs without changing the supervision assumptions.

Preparation now checks full graph availability before an expensive run and refuses to silently change the complete experiment population. A separately labelled, source-complete pilot subset (`--pilot-available-subjects`) can still verify implementation and W&B images; pilot job **615230**. It must not be used for the main experiment. The training launcher checks that its target manifest is not a pilot and that all 20,160 source test candidates were considered. **Long-run submission waits for complete GT extraction, as requested by the user.**

### Dataset health and conditional relocation request

The user asked to move the upz73jr source to their own dataset folder and replace the old version **if healthy**. Inventory against the source index found all 130,560 prepared image/mask/VTP triplets (91,200 train / 19,200 val / 20,160 test), but only one of 136 full raw MRI volumes, all 136 full segmentation volumes, and 109 of 136 complete full graphs. Prepared patch completeness is a filename/readability inventory, not exhaustive content validation. [Full health inventory and missing subject IDs](archive/synthetic-mri-source-health-2026-10-05.json).

The complete source is therefore not healthy. Both dataset directories were left unchanged; no files were moved, replaced or deleted. Full raw MRI volumes are not needed to train on the existing prepared images once the missing original graphs are restored: retained full segmentation headers supply the verified affine. Complete dataset relocation would additionally require restoring the missing full MRI volumes.

### Completed boundary and visualization smoke

Job **615708** completed in **1 minute 26 seconds**, exit **0:0**. Ten focused boundary/basis/loss/tracking tests passed. Three GPU updates on real pilot training patches had finite losses and nonzero spline-head gradients. Four validation projections were saved and visually inspected; decoded graph and physical geometry evaluation also completed on four pilot validation patches. These are smoke checks, not generalization results.

The separately labelled pilot prepared 31 training, 16 validation and 15 test-compatible caches from 64 requested pilot candidates, excluding one training and one test patch with parallel branches. No model was evaluated on pilot test patches. There were 593 snapped endpoints and 38 sparse-component exact fallbacks across the compatible pilot caches; two genuine singleton voxel-cell contacts were preserved as isolated nodes. These counts are pilot-specific and do not establish the full-data prevalence.

Online W&B initialization on the compute node timed out (job 615230). Tracking now records offline rather than lose images when online initialization raises a communication error. Job 615708 used offline W&B deliberately, then its metrics and four PNGs were successfully synced from the login node. [Actual smoke run in W&B](https://wandb.ai/alessandrobenvenuti2002-politecnico-di-torino/focal-loss/runs/8gzgsw6k). Inspect `validation/projections`. This establishes actual W&B image logging and upload; live streaming from compute nodes is not established.

Example [validation projection PNG](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/runs/dataset-smoke-615708/validation-projections/epoch-000/sample_000013_0603.png). Full smoke outputs: `runs/dataset-smoke-615708/`, including `smoke.json`, `validation-evaluation/`, `wandb-run.json` and `wandb-sync.json`.

Next required input: a backup/restoration location for the original full graphs of the 27 listed subjects. Then resume complete target extraction, inspect population/exclusion counts, smoke the completed cache, submit the long run, and arrange the final test evaluation after successful training. No permission to change the test population or reconstruct missing paths from straight-edge labels was inferred.

## Latest launch update: available subjects, smaller validation, and resubmission

The user authorized using only subjects with complete original full graphs and reducing validation to 2,000 candidates. The original split membership remains unchanged. During resubmission, the source inventory changed from 109 complete subjects to 27; the cause has not been established. The 27 remaining graph triplets were copied to `experiments/spline_learning/source_snapshot/graphs/` (936 MB), with subject membership in `source_snapshot/snapshot.json`. Preparation now uses this preserved graph root and records SHA256 hashes of all three graph files per subject in its plan and manifest. MRI/mask patches and segmentation headers still come from the original source; the complete dataset was not relocated.

Available snapshot population before compatibility filtering:

| Split | Subjects | Available patches | Requested candidates |
| --- | ---: | ---: | ---: |
| Train | 17 | 16,320 | 10,000 |
| Validation | 5 | 4,800 | 2,000 |
| Test | 5 | 4,800 | 4,800 |

Whole-patch topology/capacity exclusions can reduce these counts; final accepted counts must be read from the completed manifest. Held-out results will describe the five available test subjects and the eligibility fraction, not claim coverage of all 21 original test subjects. The graph checkpoint was previously trained on the broader original training population and selected on original validation subjects; this initialization is recorded, and no test subject exposure was found.

I cancelled jobs 624790/624917/624924 to reduce validation, then 625009/625029 after the source availability changed. Neither cancelled training job started. The user interruption occurred before replacement submission. The user then explicitly requested skipping repeated smoke tests; the passed real-data GPU smoke 615708 is retained as evidence.

Replacement chain submitted on 2026-10-05:

- **625282**: CPU GT preparation from the preserved graphs, train cap 10,000 and validation cap 2,000.
- **625284**: 50-epoch joint V0 fine-tuning, `afterok:625282`; no repeated GPU smoke. Manifest guards require the authorized source subset, validation cap and uncapped available test membership.
- **625312**: final held-out evaluation of the validation-selected checkpoint, `afterok:625284`.

Training cannot start unless preparation succeeds; evaluation cannot start unless training succeeds. These are submitted jobs, not completed results. Configuration/run name: `spline_v0_k8_10k_val2k_50ep`; cache: `synthetic_mri_k8_10k_val2k`. Four fixed validation projections remain enabled every five epochs, with W&B online initialization and recorded offline fallback if compute-node communication fails.

### Replacement preparation failed: more source metadata missing

Job **625282** failed after 26 seconds (exit 1:0): source `new_patches_boundary/generation_config.json` disappeared before preparation could read it. The next read-only check found all 27 required full segmentation volumes missing too; those headers supplied the verified full-volume affine. Sample MRI/mask/VTP patch files still existed for one checked subject per split, which is not a fresh complete inventory. No model training started. Dependent jobs **625284/625312** were cancelled because their success dependency cannot be satisfied.

The exact generation configuration retained from the successful pilot was preserved alongside the current patch index after confirming its SHA256 matches the pilot preparation plan. Previously verified patient affines were also preserved from the pilot manifest. These cover only part of the graph snapshot; missing verified affines are subjects **5, 27, 28, 42, 43, 46, 49, 71, 88, 93, 133, 16, 24, 125**. Original segmentation/MRI headers or equivalent verified affine metadata for those subjects must be restored before the approved population can be prepared. Patch NIfTI affines are identity by construction and cannot replace original volume transforms. The source-change cause remains unknown. No affine values or missing curved targets were guessed.

### Recovery archive located

A read-only scan confirmed `/lustre/fsstor/projects/rech/vnc/upz73jr/Graph-Native-Betti-Matching/2026-10-05/datasets.tar` (44 GB) contains the original SyntheticMRI generation configuration, **136 full segmentation volumes**, and **91 full graph.vvg files**, plus prepared patch data. This provides a recovery route for the missing volume transforms; extraction and alignment verification have not yet been performed. Original graph coverage still requires checking against the 27 preserved graph subjects.

### User-requested dataset replacement

The user requested restoring the archive to `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI`, replacing their current incomplete dataset. Restoration uses a separate staging directory and retains the old folder under `syntheticMRI.before-restore-JOBID` after verification. Only SyntheticMRI is extracted, with fresh filesystem timestamps. Verification checks the exact preserved configuration/index, all indexed triplets, all full segmentation headers, pilot affine consistency, and one image/mask payload per split. Job 627391 failed before extraction because the STORE archive was not visible on the allocated CPU node. The archive remains accessible on the login node and is being copied to WORK before retry.

Restoration job **627540** succeeded: all indexed boundary-patch triplets and 136 full segmentation headers verified, 91 complete archived graph subjects available. The old dataset folder was retained as a backup. [Restoration report](synthetic-mri-restoration.md). The selected 27-subject graph snapshot and experimental splits remain unchanged; restoration itself did not restart training.

## Resumption using the restored dataset

The user authorized resuming extraction and training after the verified dataset replacement. The preparation script now reads both patches and original full graphs from `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI`; it no longer restricts membership to the earlier emergency 27-subject snapshot. This follows the requested use of all available graph-complete subjects. Restored availability is **64 train / 13 validation / 14 test subjects**, with **61,440 / 12,480 / 13,440** candidates respectively. Original subject-disjoint split membership is preserved. Initialization training and validation exposure was rechecked: **zero overlap with the 14 test subjects**.

Fixed protocol: seeded selection of up to 10,000 training and 2,000 validation candidates before topology/capacity compatibility checks; all 13,440 available test candidates considered. Unsupported whole patches are recorded rather than partially relabelled; final eligible counts await completed extraction. Validation every five epochs, four fixed projection examples logged to W&B, 50 epochs, joint eight-control V0 fine-tuning, fixed spline weight 1.4739989613. Model initialization, losses, optimizer and checkpoint selection remain as specified above. No repeated GPU smoke, per the user's instruction.

Submission chain:

- **646175** — CPU spline GT preparation from restored data.
- **646177** — joint training, starts only after preparation succeeds (`afterok:646175`).
- **646195** — held-out test using the validation-selected checkpoint, starts only after training succeeds (`afterok:646177`).

Syntax checks passed for the preparation script and all three Slurm scripts; the restored dataset was already verified separately. No prior long-run cache or output directory exists, so no old population is reused. Preparation records source-graph hashes, selected membership, boundary handling and exclusions. Logs: `experiments/spline_learning/logs/{prepare,train,test}-JOBID.{out,err}`. These are submitted jobs; completed GT or training results are not asserted.

## Failure diagnosis and resumption (2026-10-06)

Job 646175 failed on duplicate undirected edges before training. Reproduced on real patch **sample_000001_0272**. Source CSV edge **806** has endpoints **2488 → 2488**, recorded length **1.28088 mm**, and one interior guide sample. Its dense crop is **A → B → A**; generic simple-graph contraction rejects the two opposite segments as duplicate edges before it can classify the closed branch. This is a closed source-guide component, not a learned prediction or evidence of two anatomical vessels.

`crop_dense_paths` now records closed components. `contract_spline_graph` excludes such whole patches as `closed_branch` before contraction, and rejects duplicate direct connections as `parallel_branches` without deduplicating source records. Existing longer closed/parallel branches and object capacity checks remain. Regression coverage includes a two-segment source loop, short parallel branches and a valid one-edge graph. Actual real-crop replay passes. This applies the already specified eligibility policy; no branches are silently removed from retained patches.

Job 675623 progressed to **88 completed patient caches**, then failed creating a link because WORK reached its **500,000-file quota**. The cache was moved, preserving those completed results, to `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/synthetic_mri_k8_10k_val2k`; its repository path is now a link. All accepted target NPZ files in those 88 patient results were confirmed retained. Approximately **97,700 file entries** were freed initially. Resume reuses complete patient results and reprocesses unfinished patients with unchanged selected membership.

Job 676478 failed at startup because the installed PyVista package lacked `typing/mypy_plugin.py`. Reinstalled **exactly PyVista 0.46.3**, without changing dependencies. A real relocated VTP was successfully read and checked against its target NPZ. No repeated model/GPU smoke was run. The cause of missing environment files is not established.

Current jobs: **676631** resumed GT, **676650** training after successful GT, **676692** test after successful training. Scheduler-held jobs 676486/676532 were replaced after the scheduler refused dependency update/release. Previous failed chains are historical, not active. Configuration, sampling caps, original split membership, losses and initialization are unchanged.

Stable Git commits on representation: **24c2605** (spline training/evaluation/validation projections plus topology eligibility) and **98eb6a9** (verified staged dataset restoration). External documentation stays in graph-representation/docs; the code repository includes a short documentation index.

### Completed main GT population

Job **676631 completed in 3 min 15 s, exit 0:0**, resuming the three unfinished patients. Final target cache:

| Split | Requested candidates | Accepted | Closed-branch exclusions | Parallel-branch exclusions |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,000 | 9,709 | 133 | 158 |
| Validation | 2,000 | 1,952 | 22 | 26 |
| Test | 13,440 | 13,071 | 156 | 213 |

No object-capacity exclusions occurred. Compatibility coverage on the available test candidates is **97.25%** (13,071/13,440). Availability itself covers 14 of the original 21 test subjects; eligible metrics must not imply complete original-test coverage. Actual evaluated counts will be checked against 13,071 after training. Membership is selected before compatibility checks, with no replacement/top-up based on shape quality.

The completed manifest, preparation plan and exact patch index were copied back to WORK as `experiments/spline_learning/source_snapshot/current-{manifest.json,preparation-plan.json,patch_index.csv}`. Generated targets remain on SCRATCH. Main source graphs are the restored 91-subject population, not the obsolete 27-subject emergency copy. Training 676650 is pending GPU priority normally; test 676692 is pending its success dependency. No model/GPU smoke was repeated.

### Active layout clarified

Historical pilots, four-patch targets/model runs, old snapshot records, temporary diagnosis artifacts and failed job logs have been archived in STORE; see the cleanup record. Current immutable manifests/configuration are now under `experiments/spline_learning/metadata/`. The main dataset/run links and active logs remain. Training **676650 is running**; it initialized the new spline head from the graph checkpoint successfully. Compute-node W&B initialization fell back to offline recording (run ID **pnbftfvi**), retaining scalars/images locally for later upload. No validation/generalization result is asserted at this point.

## Epoch-9 interruption and strict resume

Training **676650 failed after 33 min 27 s** inside the epoch-10 DataLoader, because the shared environment no longer contained `medpy.io.header`. Nine complete training epochs and an epoch-5 validation event had succeeded. Recorded combined validation loss at epoch 5: **2.092977**. This is an optimization diagnostic, not held-out test performance.

The atomically saved latest checkpoint was read and verified: **epoch 9 / iteration 21,852**, with model, optimizer, scheduler and runtime state. Existing strict checkpoint resume restores these states, plus prior checkpoint-selection state. Partial epoch-10 work after that checkpoint is replayed; no completed epochs are discarded and no optimizer reset is performed.

Reinstalled **exactly MedPy 0.5.2**, with no dependency changes. A targeted RECORD audit found other missing package entries, including stale duplicate-version metadata and unused assets/tests; it is not evidence that every missing entry is an active runtime fault. Given the earlier missing PyVista file and the later MedPy file disappearance, preserved the current repaired site-packages tree outside projects at `/lustre/fswork/projects/rech/vnc/upz25mj/experiments/gnbm/spline-runtime-20261006/site-packages.tar` (1.4 GB). Both train and test now extract this fixed package copy into job-local storage and prepend it to PYTHONPATH. System PyTorch/CUDA remain supplied by the same cluster module. The cause of the shared-file deletions is still unknown.

Targeted verification using that job-local copy passed: actual 64³ MRI loaded with finite values through the real MedPy reader, real checkpoint read, and MedPy/PyVista imports confirmed from the local copy. No repeated GPU/model smoke was run. The source package snapshot is kept outside projects.

Resume job **681009** uses `--resume latest_checkpoint.pt` and continues epochs **10–50**. Tracking uses the existing W&B ID **pnbftfvi** in offline mode, avoiding the known compute-node timeout. The old segment and resumed segment must both be synced from a login node to the same run. Test job **681013** depends on successful resume. Replaced blocked test 676692. Model/loss/data/scheduler configuration and all exact patch memberships are unchanged.

Stable commit: **81b21ec**, adding job-local package staging and optional checkpoint-resume submission support. Active job IDs are recorded in `experiments/spline_learning/metadata/current-jobs.json`.

### Complete library availability audit and final runtime repair

Resume 681009 failed during startup on Matplotlib's missing `tri.tricontour`, before modifying the epoch-9 checkpoint. Reinstalled **Matplotlib 3.8.4**, matching the version actually imported before failure; conflicting newer distribution metadata had been present. The shared environment then showed additional missing files during successive checks. Recopying it captured new holes, so the final runtime was built from the earlier preserved local copy and repaired directly there rather than from the changing shared installation.

Audited all **81 declared requirements plus torch, torchvision, torchaudio, W&B and the custom CUDA extension: 86 imports**, each in its own process. MPI-enabled probes required access outside the restricted login sandbox. Fixed the audit's Markdown package alias and imported PyTorch before the custom extension, as required for its shared libraries. Missing package files were confirmed in POT, VTK, timm and W&B; repaired directly in the private runtime at **POT 0.9.7 / VTK 9.5.2 / timm 1.0.22 / W&B 0.28.1**. Other packages in the preserved copy remained unchanged. All **86 import probes passed** afterward.

The actual active versions/paths are recorded in `metadata/library-availability.json`; some dependencies are supplied by the protected cluster module at versions different from the old captured requirements file. This audit verifies availability, not exact agreement with every historical pin or exhaustive testing of every optional API. System PyTorch remains **2.3.1**, with the same cluster CUDA stack. No package upgrades to that stack were performed.

Additional CPU checks passed: train/evaluate entry points, real 64³ MRI loading, PNG generation, W&B Image creation, full model construction and strict loading of the saved epoch-9 parameter state. No new GPU/model smoke run was used. Existing optional Hugging Face/torchvision warnings during CPU imports did not prevent the tested main model construction or image pipeline.

Saved the validated private package tree as the fixed archive outside projects and recorded its SHA256; staging verifies that checksum before local extraction. The cause of shared package-file disappearances remains unconfirmed. Current run resumes from the full **epoch 9 / iteration 21,852** checkpoint with optimizer/scheduler/RNG and checkpoint selection state retained. Jobs **687805 → 687813** are resume → held-out evaluation; 681009 failed and its blocked test 681013 was cancelled. Tracking keeps W&B ID pnbftfvi offline.

Reusable audit: `scripts/check_spline_libraries.py`, after sourcing cluster env and fixed runtime. Stable audit/checksum commit **26d1630**. Do not claim GPU execution from the CPU import audit: actual resumed epochs remain to be observed.

### Why fewer than the requested caps?

Caps apply to the seeded candidate selection **before** graph compatibility checks. From 10,000 training candidates, 133 closed-branch and 158 parallel-branch patches are excluded whole, leaving 9,709. Validation: 2,000 − 22 − 26 = 1,952. Test: 13,440 − 156 − 213 = 13,071. There are no capacity exclusions. The originally selected set is retained without topping up after inspection. This respects the user's at-most-10k budget and makes the exclusions/population coverage explicit. Changing to an exactly-10k eligible selection would be a different declared sampling protocol; it has not been done mid-run.

## Completed training and W&B upload

Training **687805 completed**, exit **0:0**, resumed job elapsed **2 h 18 min 53 s**. This completed epochs 10–50 after the saved first nine epochs. Two offline segments with W&B ID **pnbftfvi** were uploaded from the login node, using append for the resumed segment. The cloud API confirms `train/epoch = 50` and validation epochs **5, 10, 15, 20, 25, 30, 35, 40, 45, 50**. [Training dashboard](https://wandb.ai/alessandrobenvenuti2002-politecnico-di-torino/focal-loss/runs/pnbftfvi). Validation projections are logged under `validation/projections`; scalar curves remain in the same run. Local sync provenance is saved as `wandb-sync.json`. No final held-out results are asserted before test completes.

## Completed held-out evaluation and logging recovery

Test job **687813** ran **40 min 11 s**. GPU evaluation completed and wrote summary, metadata, all per-patch metrics and decoded predictions, then the tracking step failed (`ServerResponseError: run ID pnbftfvi is in use`) after compute-node online initialization/fallback. Slurm therefore records **FAILED / exit 1:0**, although the evaluation outputs are complete. This status is not rewritten as a successful Slurm job.

Verified **13,071** consistently in evaluation metadata, summary, target manifest and per-patch CSV row count. Used the validation-selected `best_checkpoint.pt`, not the epoch-50 checkpoint. No BN recalibration, test subset selection or threshold tuning. Re-ran **only** `scripts/log_spline_test.py` from the login node in online mode; it successfully attached the saved metrics and population coverage to the existing W&B run. No GPU re-evaluation occurred.

Measured results:

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Standard node F1 | 0.884115 | Existing box-based detection protocol |
| Standard edge F1 | 0.815718 | Existing graph detection protocol |
| Physical branch F1 | 0.169838 | Separate one-to-one endpoint correspondence within 1 mm |
| Physical target-branch coverage | 0.174068 | Only 17.41% of target branches recovered under that strict criterion |
| Matched-branch fitted-curve error | 0.736972 mm | Applies to recovered branches only |
| Matched-branch original-guide distance | 0.639648 mm | Applies to recovered branches only; guides uncertain |

The two F1 protocols are distinct. Matched-branch geometry must accompany its low correspondence coverage and must not be presented as whole-test vessel reconstruction accuracy. The held-out subject population and topology exclusions remain as recorded above. Full outputs: `runs/spline_v0_k8_10k_val2k_50ep/test/`.

Future test submissions default to offline compute-node tracking, then login-node upload; stable commit **f1725bb**. [W&B run with test metrics](https://wandb.ai/alessandrobenvenuti2002-politecnico-di-torino/focal-loss/runs/pnbftfvi).
