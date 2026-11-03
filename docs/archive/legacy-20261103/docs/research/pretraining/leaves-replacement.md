# HALVS replacement of the plant source domain — 2026-10-15

## Question and controlled comparison

Does a leaf-vein source improve SyntheticMRI graph generalization relative to plants?
Reuse the completed MRI-only control from `plant_generalization_20261014_da_b32_w1_restart`; it is identical and does not need another training. Run a new mixed HALVS/MRI pretraining from random initialization, followed by separate straight-edge and detached residual B-spline MRI fine-tunings. Do not mix plants and leaves yet.

## Plant field of view

The requested native64 crop was a separate inspection, not the training patching policy. The recovered plant generator downsamples by6 before cropping118 content pixels with5-pixel padding to128. This covers about708 original pixels, then the loader resizes to54 with padding into64³ and projects a five-slice slab.

`plants_3d2cut/prova/field_of_view/comparison.png` compares source crops64,256,708,1024 at the same original junction, displayed at64 pixels. They contain1,3,12,24 annotated points respectively (including degree-2). Wider crops capture more structure at the expense of fine-branch resolution. This illustration uses the photograph and occupancy-resized mask; actual training uses smoothed mask-derived images. Existing data were not overwritten. A larger plant FOV is feasible, but not yet applied to a new training population.

## HALVS extraction choices

- Dataset: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET`.
- Original parent-leaf split:144 train,6 validation. No cross-leaf resplitting; no patch-level random split.
- Reuse `leaf-veins-extractor` full-resolution skeleton tracing, snapshotted under `experiments/leaf_pretraining/reference/` with hashes.
- Primary+secondary veins (red+green). Tertiary veins excluded from BOTH image mask and graph; this experiment is not supervision of all leaf venation.
- Downscale5.42, effective source crop about640pixels,128px padded output, same plant loader and64³ slab projection.
- Keep extractor defaults: prune short terminal branches8pixels, merge junctions6pixels, curvature approximation tolerance6pixels. The segmentation still contains small traces pruned from the graph: extraction is an approximate annotation, not exact ground truth.
- Overlap87.5% to provide a pool for deterministic selection of25,000 accepted training patches and480 source validation patches. These are spatially correlated crops of144 leaves, not25,000 independent biological samples. Full extraction is uncapped so early filenames do not consume the budget.
- Existing extractor filters graphs with more than80 pre-contraction nodes. These exclusions are logged; source population is conditional on these filters.
- Preserve intermediate curve points during extraction; reuse our existing degree-2 contraction afterward, preserving boundary anchors at5/128 and122/128. Keep full paths in the spline cache. Existing unsupported closed/parallel branches are excluded and recorded, never silently collapsed into a different graph.
- Require exactly25,000/480 accepted patches; insufficient population fails preparation and blocks all training.
- Raw network inputs are smoothed/dilated selected-vein masks, as for the recovered plants recipe, NOT original leaf photographs. This tests morphology transfer under the existing pipeline.
- Pretraining supervises nodes/connectivity only. Stored source spline targets are in PNG coordinates/pixel units and are not used as MRI geometry targets.

## Training unchanged

Config: `configs/leaf_generalization_da_b32.yaml`; same training implementation as plants.

-50 pretraining epochs, batch32, seed364505, AdamW LR7e-5, weight decay1e-4, same polynomial schedule and one-epoch warmup.
-25k HALVS source + identical10k SyntheticMRI target.50k replacement draws/epoch, target upsampling2.5 for expected50:50 source/target exposure.
-Domain adaptation enabled with user-selected weight1.0 (image, instance and consistency components).
-Same approved augmentations: source quarter-turns/flips; MRI quarter-turns and Gaussian noise(probability.35,max std.015); no zoom; spline-safe coordinate transforms.
-Two100-epoch maximum vessel-only continuations from the best leaf mixed pretraining checkpoint: straight edges and8-control detached residual B-splines. Same graph losses and spline weight1.4739989612791453. Same early-stopping policy.
-Identical MRI training membership and1952 validation patches, common branch-F1 checkpoint selection, same final/best validation metrics. Held-out test remains untouched by this submission.
-W&B offline logs and validation projections retained. Internal arm/loader names retain `mixed`/`plants` for compatibility; this config's source is HALVS.

## Execution and artifacts

Output root: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/leaf_generalization_20261015`.

CPU extraction smoke → full extraction/contraction/audit → GPU smoke checks → pretraining25+25 → parallel straight/spline fine-tunings50+50. Slurm `afterok`/`aftercorr` dependencies and invalid-dependency cancellation prevent training after failed preparation/checks.

Submission record: `experiments/leaf_pretraining/submission-20261015.json`.
Reproduce submission once: `python cluster/jean_zay/submit_leaf_generalization.py --smoke-job <CPU_SMOKE_JOB>`; script refuses to overwrite an existing record.

After preparation: `source-audit.json`, `leaf_graph_preview.png`, `extracted/patches_manifest.csv`, `leaves-k8/manifest.json`. Preview cyan chords/red anchors show contracted GT, not model predictions. Continuous guides are retained separately. Run outputs: `runs/pretrain_mixed`, `runs/mixed_straight`, `runs/mixed_spline`.

## Pending interpretation

CPU smoke1093471 succeeded:2 train+2 val leaves,209/288 raw patches at80% overlap, and2/2 contracted targets. This small pool suggested80% overlap may not supply25k patches; full extraction uses87.5% overlap instead (same FOV, denser crop grid). No leaf performance result yet. Compare final and best MRI validation with the completed plants and MRI-only controls, separating geometry from graph detection. Check source graph density and coverage before proposing a larger plant FOV or joint plant+leaf pretraining. Overlapping leaf crops and approximate extracted targets limit any attribution to dataset diversity alone.

### Submission status

- CPU extraction smoke1093471: completed successfully; two contracted training overlays visually inspected (`previews/halvs-smoke.png`). Paths lie within vein inputs and crop endpoints are retained.
- Full preparation1094726: running at submission check.
- GPU verification1094729: pending preparation.
- Mixed leaf/MRI pretraining1094731 →1094733: epochs1–25 then26–50, pending successful verification.
- Parallel straight/spline fine-tunings1094737_[0-1] →1094739_[0-1]: to50 then100, pending pretraining. Index0=straight,1=B-spline.
- All dependencies confirmed in Slurm. No production training has started at this check. Failure of preparation or verification prevents dependent training; no automatic fallback to fewer samples.

Static Python compilation and shell syntax checks passed. GPU correctness and full-population checks are queued, not yet claimed successful.

### Dataset placement update

Full extraction completed:80,079training and3,953validation candidates; selected25,000/480accepted patches,391compatibility exclusions encountered during selection. GPU verification also completed.

Dataset entries are now available at `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/HALVS_DATASET/patches_3d_transfer_20261015/`. They currently link to the experiment storage to avoid interrupting active readers. The attempted atomic directory exchange was unsupported by Lustre and made no data changes. CPU job1127211 waits for leaf fine-tuning array1094739, then moves the actual directories into the dataset folder and leaves reverse aliases at the old experiment paths. Configuration now uses the dataset path; existing resolved configs remain valid through aliases. Check `relocation.json` for physical move completion.

## Completed training and held-out comparison submission

Leaf pretraining 1094733 and both continuation stages 1094737/1094739 completed successfully. Existing full-validation results are collected in [source-transfer-results/validation-comparison.md](source-transfer-results/validation-comparison.md) and the adjacent CSV. These use `best.pt` selected by validation, not final checkpoints. This distinction can explain differences from previously reported final-checkpoint metrics.

Matched comparison includes leaves+MRI, plants+MRI and the equal-update MRI-only control, each with straight edges and detached residual B-splines. All six used graph-only pretraining; geometry is introduced only in spline fine-tuning. The fine-tuning data, training settings, model architecture and evaluation configuration were checked for equality. The source-domain difference is intentional, including its rendering/extraction and graph distributions; it is not an isolated test of biological species. Single-seed results do not establish statistical significance.

Validation on 1,952 patches: leaf spline edge F1 0.68793, plant spline 0.68524, MRI-only spline 0.72243; curve F1 0.45700, 0.45580 and 0.50357 respectively. Leaf spline node mAP 0.43951 versus plant 0.44261. Leaves currently look similar to plants and do not close the MRI-only gap. This is provisional validation evidence, not a held-out test conclusion.

New held-out evaluation uses all 13,071 original test IDs, verified against the existing comparison manifest with a recorded ordered-ID hash. All six validation-selected checkpoints use identical native node/edge mAP/mAR/F1, common curve/branch metrics, same-endpoint chord comparison and the existing core geometry/topology/Betti/APLS/TOPO diagnostics. No augmentation, BatchNorm recalibration, checkpoint reselection, threshold tuning or retraining. Saved running statistics are used.

- GPU smoke checks on two test patches for each leaf model: array **1145697**.
- Six full test evaluations, gated by smoke success: array **1145698**, at most three concurrent GPUs.
- Automatic report/CSV generation after all tests succeed: CPU job **1145699**.
- Plan and submission records: `experiments/leaf_pretraining/evaluation/`.
- Raw test output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/source_transfer_test_20261015/`.
- Future final tables: `docs/research/pretraining/source-transfer-results/test-comparison.md` and `test-comparison.csv` (created only after all tests succeed).

Static checks passed; all six configuration checks and full test membership checks passed. GPU evaluations were submitted and results are pending. Existing validation outputs were reused without repeated inference.

### Held-out tests completed (2026-10-16)

All six evaluations 1145698 and comparison job 1145699 completed successfully on the same 13,071 test patches. Full results: [test table](source-transfer-results/test-comparison.md), [CSV](source-transfer-results/test-comparison.csv). Spline edge F1: leaves 0.6905, plants 0.6879, MRI-only 0.7258; curve F1: 0.4607, 0.4581, 0.5097. Leaves and plants remain close, without a consistent overall winner; MRI-only is stronger on detection and curve recovery. These are one-seed comparisons, not significance tests. Combined-source results remain pending.
