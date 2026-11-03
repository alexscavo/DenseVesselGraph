# Superseded experiment folders (archived 3 Nov 2026)

This file lists the 49 study folders that were removed from `experiments/` on 3 Nov 2026.
For each one it says what the study tested and why it was superseded or not pursued. The
consolidated findings, with all caveats, are in
[`../docs/research/HISTORY.md`](../docs/research/HISTORY.md); "→ §x" below points there.
Numbers are copied from HISTORY or from the folder's own report.

**Where the folders are now**

- Archive: `$STORE/DenseVesselGraph-experiments-archive-20261103/<folder>.tar.zst`, with
  `STORE=/lustre/fsstor/projects/rech/vnc/upz25mj`. Restore one folder from the repository root:
  `zstd -dc $STORE/DenseVesselGraph-experiments-archive-20261103/<folder>.tar.zst | tar -xf - -C experiments/`
- Git: the tracked report files (JSON, CSV, Markdown, logs) are in history up to the commit before
  the removal. Checkpoints and other large files were never in git; they exist only in the archive.
- Symlinks: some folders contained symlinks to `$SCRATCH/experiments/gnbm/...` (for example
  `spline_learning/runs`, `brava_long_20261027/large`, `brava_augmentation_20261026/runs`,
  `r06_*/large`, `r07_*/large`, `native2d_20261027/runtime`). The archive keeps the links, not the
  targets, and SCRATCH may already have purged them.

**Still read by current code.** Some current scripts read files inside these folders. Restore
the folder before rerunning these scripts:

| Folder | Read by |
|---|---|
| `native2d_mixed_20261028/plan.json` | `scripts/train_native2d_long.py` reads it on import (it gives `DATA` = `native2d_long_20261028`), so `dense_graph_2d_20261030.py` reads it too, also for `QUERIES`; `cluster/jean_zay/dense_graph_2d.slurm` uses its `mpl/` as `MPLCONFIGDIR` |
| `local_refinement_long/` (`92741/runs/B0/model-12000.pt`, `plan.json`) | `b0_straight_dense_20261101.py` (SyntheticMRI B0 checkpoint), `straight_lineage_20261101.py`; `ixi_long_20261030/plan.json` records this checkpoint as its initialiser |
| `brava_long_20261027/` (`plan.json`, `batch-selection.json`, `large/runs/synthetic/...`) | `anchored_relationformer_20261030.py`, `ixi_long_experiment_20261030.py`, `straight_lineage_20261101.py`, `b0_straight_dense_20261101.py` (BRAVA D08 epoch-300 checkpoint, behind a SCRATCH symlink) |
| `fine_feature_architecture/plan.json` | `straight_lineage_20261101.py` |
| `brava_graph_study/plan.json` | `tests/test_brava_graph_study.py` |

Labels follow HISTORY: DONE, DONE — negative, PROPOSED (never run), RUNNING (no final result was
ever written). Unless stated otherwise, results are single-seed and validation-only.

## 1. 3D RelationFormer on IXI: diagnostic series 1–11 (18–20 Oct) → §4.2

Shared setup: the frozen IXI-only delayed-spline parent (step 28170) and its derivatives, mostly
256 train / 128 validation patches, positional edge F1, one seed, no test data. The dense
traced model replaced RelationFormer and addresses each mechanism found here (HISTORY §9).

- **`ixi_supervision_audit`** (18 Oct): read-only check of the references and predictions against the vessel mask.
  DONE. GT is consistent with the mask (99.52% of curve samples inside), which does not show anatomical correctness. Of 51,937 validation GT edges, 69.38% miss at least one endpoint.
- **`ixi_features_nodes`** (18 Oct): linear foreground probes on frozen detector features.
  DONE. Validation AP is 0.6551 for early 32³ CNN features and 0.0246 for the 2³ transformer memory. The 2×2×2 grid cannot represent fine vessels, which led to the full-resolution U-Net heatmap.
- **`ixi_node_refinement`** (18 Oct): early-CNN local refinement of the node heads (1,000 updates), with a zero-image control.
  DONE. Native node F1 rose 0.4857 → 0.5196 and relation recall did not change. This was the first evidence for A-xyz. The folder holds only the smoke outputs; the full run was on SCRATCH.
- **`ixi_relation_isolation`** (19 Oct): relation scoring isolated on a frozen detector. Two parts: a refit of the relation head on matched pairs, and image+geometry vs geometry-only classifiers on GT endpoints.
  DONE. On recovered-endpoint pairs the existing head has AP 0.4702 but recall 0.1257 at 0.5. In `results.md` the refit raised training AP but lowered validation AP.
- **`ixi_threshold_calibration`** (19 Oct): relation thresholds chosen on training data.
  DONE — negative. The train-max-F1 threshold raised edge F1 0.0737 → 0.1608, but β₁ MAE went 1.04 → 19.52. Threshold sweeps were then banned as a fix.
- **`ixi_query_diagnostics`** (19 Oct): attribution of full queries, duplicates and false edges. The folder holds only the submission record.
  DONE. 91.1% of false edges touch an unmatched node. Fixed NMS lowers positional edge F1 (0.0737 → 0.0626 at 0.5 mm) because it removes real nearby nodes.
- **`ixi_mechanisms`** (19 Oct): mechanism batch A–D: raw-query selection, duplicate separability, the refinement residuals and relation-only continuations.
  DONE — negative. Using all 256 raw queries recovers 83.62% of nodes but only 162 vs 142 correct edges. The relation-only continuations ranked validation pairs worse (conditional AP 0.3659–0.4153 against 0.4702).
- **`ixi_relation_matching_calibration`** (19 Oct): CPU re-analysis of whether Hungarian-selected queries suit the relation head better.
  DONE — negative. Conditional AP was 0.4658 with Hungarian selection at inference and 0.4702 with normal decoding, and the interval includes zero.
- **`ixi_image_relation_learning`** (20 Oct): image-aware residual relation adapters (endpoint, chord and corridor samples, plus a token control) on 2,048 / 512 patches for 65,600 updates.
  DONE — negative. Every adapter had overfit by the final update: validation AP 0.31–0.32 against 0.4723. The dense model's edge scorer, which reads features along the path, replaced the idea.
- **`ixi_node_focal_ablation`** (20 Oct): node loss: weighted CE vs equal-weight CE vs equal-weight focal.
  DONE — negative. Edge F1 was 0.1189 / 0.1165 / 0.0767, so the colleague's SyntheticMRI focal benefit did not reproduce. The edge loss changed at the same moment, which confounds the comparison.
- **`ixi_segmentation_input`** (20 Oct): the binary vessel mask as input instead of raw MRA. This is a reconstruction oracle, because the mask generated the references.
  DONE. Positional edge F1 rose 0.0415 → 0.0904 and Branch F1 0.0061 → 0.0243. Even with this mask, edge F1 stays at 0.09, so much of the failure is in graph decoding.
- **`cheap_diagnostics_20261026`** (26 Oct): CPU-only follow-up. It re-analysed the saved adapter checkpoints from `ixi_image_relation_learning`, ran BN inference calibrated on training data only, and audited the archived August evaluation logs.
  DONE. At 2k updates the chord and corridor adapters reached AP 0.50156 / 0.50191 against 0.47660 for the token control, but those checkpoints were chosen after seeing the curves. 30 of 32 August evaluations had calibrated BN on the validation loader (→ §1).

## 2. RelationFormer node and relation follow-ups (23–28 Oct) → §4.3, §6.3

Superseded by the 30 Oct switch to the dense model. Most of these runs fine-tune from S0, the
clean SyntheticMRI B0 run.

- **`fine_feature_architecture`** (24–25 Oct): S0 training (100 epochs, epoch 95 selected), then 3,000-update pilots of local refinement (A) and multiscale attention (C).
  DONE. S0 validation node / edge F1 @2 mm was 0.7882 / 0.5881 (train 0.7910 / 0.5894), so SyntheticMRI fits. C used its fine level but showed no gain. S0 `runs/baseline/best.pt` is the source checkpoint of D07, and `plan.json` defines the recipe that the straight lineage's stage s1 reruns.
- **`local_refinement_long`** (25 Oct): arms B0, A, A-xyz and A-control, each with two fine-tuning seeds, 12,000 updates.
  DONE. A-xyz − B0 edge F1 was +0.0393 [+0.0313, +0.0467]. The confidence correction adds nothing, and cycles and duplicates did not improve. B0 seed 92741 `model-12000.pt` became the SyntheticMRI initialiser for D08 and for `ixi_long_20261030`, and it is the SyntheticMRI B0 in `b0_straight_dense_20261101`. The folder is about 44 GB of checkpoints.
- **`node_verifier_feasibility`** (23 Oct): learned rescoring of decoder tokens (V0–V3) on IXI, SyntheticMRI and BRAVA.
  DONE — negative. At matched node counts, node F1 fell [−0.0810, −0.0563] on IXI. The 25 Oct cleanup zipped its prediction NPZs (`ARCHIVED_PREDICTIONS.md`).
- **`graph_error_attribution_20261026`** (26 Oct): R02, attribution of A-xyz graph errors on SyntheticMRI.
  DONE. 82.4–83.2% of false edges touch an unmatched node. 90.7–90.8% of missed edges lack a recovered endpoint; only 9.2–9.3% are pure relation misses.
- **`oracle_representatives_20261026`** (26 Oct): N06, choosing among nearby queries with GT (a representative oracle).
  DONE. Picking the confidence-preferred query raised edge F1 from 0.720/0.729 to 0.834/0.838, so spatially equivalent queries carry different relation content. The oracle is not deployable.
- **`gt_free_query_selection_20261026`** (26–27 Oct): N08, GT-free versions of the N06 selection, chosen on training data.
  DONE — negative. The intervals include zero, and the search was stopped.
- **`r06_relation_tokens_20261027`** (27 Oct): R06, relation tokens enriched with CNN image features.
  DONE — negative. Edge F1 −0.003678 / −0.002978 with intervals below zero. The predeclared gate for an unfrozen confirmation did not pass.
- **`r07_matched_supervision_20261027`** (27 Oct): R07, reliable nearby nonadjacent pairs as negatives in relation-only training.
  DONE — negative. Only 18 of 28,494 training negatives qualify, and the gain stays below the original checkpoint.
- **`ixi_local_components_20261026`** (26–28 Oct): N07, six arms from S0 on IXI (4,096 / 512 patches, 5,500 updates, one seed).
  DONE. Positional edge F1: B0 0.0879, A-xyz 0.0943, A-cls 0.1015. A-cls − B0 was +0.013610 [0.008265, 0.019401], but duplicates rose 313 → 535 and 93.02% of missed edges lack endpoints. The first attempt failed with `AF_UNIX path too long`; the 28 Oct recovery completed it.
- **`node_first_curriculum_20261028`** (28 Oct): L08, a curriculum that trains nodes first and ramps in the edge loss. SyntheticMRI from random initialisation, 20-epoch pilot.
  DONE — inconclusive. Edge F1 0.12783 vs 0.10357 (+0.02426 [+0.01047, +0.03784]), but the Branch and Curve intervals include zero. The confirmation run was never launched.

## 3. Representation and curved edges → §2, §5

- **`spline_learning`** (5–13 Oct): every SyntheticMRI curved-edge study. It covers the V0/V1 overfit, the 300-epoch scratch runs (straight vs K8 vs adaptive degree-2), the collapse diagnosis, the gradient-route tests, residual targets, the local corrector, curve-informed connectivity and Catmull–Rom.
  DONE; the 300-epoch comparison used the test split. The scratch K8 arm collapsed (test Branch F1 0.0000). The spline head does not help the detector: the routes gave edge F1 0.87198 vs 0.87096. The residual targets, the local corrector and `graph_only` carried into B0. `runs/` and the dataset are SCRATCH symlinks.
- **`spline_target_capacity`** (22 Oct): GT-only error of the cached K8 targets on IXI and BRAVA (train + validation).
  DONE. BRAVA validation P95 maximum error is 1.011 mm, with 5.01% of branches > 1 mm; IXI P95 is 0.149 mm. Only BRAVA's long branches need more than 8 controls.
- **`continuous_representation`** (22 Oct): BRAVA GT fits with fixed K, adaptive K and piecewise curves, plus K8/K16/K24 heads trained on BRAVA over a frozen IXI detector.
  DONE. Branches within 0.5 mm: K8 77.19%, K16 94.02%, K24 98.34%. Learned K24 − K8 Curve F1 was only +0.000363 [+0.000145, +0.000611] at 23.47% node recall: while detection is this weak, more controls bring no gain. 8,000 updates, validation only.

## 4. Pretraining and transfer (14–24 Oct) → §7

The dense model trains from scratch, which superseded this line. Each run was compared with a
target-only control given the same updates. Checkpoints were selected on validation and then
**evaluated once on the test split**. One seed.

- **`plant_pretraining`** (14 Oct): mixed 3D2Cut plant + SyntheticMRI pretraining with adversarial DA.
  DONE — negative. Test edge F1: Plants + MRI 0.6879, MRI only 0.7258. The MRI-only arms of this run (on SCRATCH, `plant_generalization_20261014_da_b32_w1_restart`) were reused later as SyntheticMRI initialisers.
- **`leaf_pretraining`** (15 Oct): the same protocol with HALVS leaf veins.
  DONE — negative. Test edge F1: Leaves + MRI 0.6905, MRI only 0.7258.
- **`combined_pretraining`** (15 Oct): plants + leaves + MRI. The folder holds only the submission and evaluation plan.
  DONE — negative. Test edge F1 0.6809.
- **`source_augmentation`** (16–18 Oct): rendering augmentation for the plant source (nonplanar warp, textured background).
  DONE — negative. It helped plants slightly (test edge F1 0.6935) but never reached MRI only.
- **`ixi_transfer`** (15–16 Oct): IXI-only vs plants → IXI vs SyntheticMRI → IXI, each straight and spline.
  DONE — negative. IXI only was best (straight edge F1 0.15985); plants + IXI reached 0.12084 and SyntheticMRI + IXI 0.11688. Its `ixi_delayed_spline` arm is the frozen parent of series §1.
- **`ixi_overfit`** (16 Oct): overfit test on four IXI patches.
  DONE. Train edge F1 1.0: the detector can memorise real IXI, so the failure is in fitting the full set.
- **`limited_botanical_transfer`** (≈21 Oct): 200 MRI patches, leaves + vessels vs vessels only, DA off.
  DONE — positive, but single seed and test split. Test node F1 0.3855 vs 0.3246 and edge F1 0.2979 vs 0.2588, while component error got worse. This is the only positive botanical result. It was never replicated or tested with the dense model, so it remains an **open idea**, not a disproven one.
- **`multiscale_pretrained`** (23–24 Oct): B1, a 4³ attention level added to a pretrained checkpoint.
  DONE — negative. Edge F1 −0.0034 [−0.0170, +0.0089]. The fine level was essentially unused (attention mass 0.0014–0.0065%). The U-Net later addressed the coarse grid directly.

## 5. BRAVA with RelationFormer (21–28 Oct) → §6.1

Errors are measured against incomplete SWC traces. The test split was never used.

- **`brava_graph_study`** (21 Oct): Stage 1. GT-representation audit, zero-shot evaluation of earlier models and two short scratch arms. It also holds the prepared Stage 2 plan.
  DONE — negative. The IXI spline model zero-shot was the best start (node recall 0.2309, edge F1 0.0269), and the scratch arms recovered 4 and 1 correct edges. Stage 2 (`stage2-plan.json`, `training_allowed: false`) is PROPOSED and was never run.
- **`brava_representation_20261026`** (26 Oct): G05, K24, C1 and explicit degree-2-split heads against K8.
  DONE — negative. Every arm recovered zero complete branches. K24 and C1 moved learned Curve F1 by only +0.0001 / +0.0002. Splitting gave +0.0424 [0.0348, 0.0494], at the cost of more edges (435 → 1,147).
- **`brava_learning_diagnosis_20261026`** (26 Oct): D06, learnability on a 160-patch training monitor, plus BN recalibrated on training data only.
  DONE. Train node / edge F1 0.1325 / 0.0114 against validation 0.1183 / 0.0089. The BN recalibration did not help, so the cause is neither overfitting nor BN.
- **`brava_augmentation_20261026`** (26–27 Oct): D07, augmentation on vs off over 4,000 updates.
  DONE — mixed. Train edge F1 +0.1551 [+0.1294, +0.1785]; validation +0.0062 [−0.0009, +0.0125]. Turning augmentation off only buys memorisation. `runs/` is a SCRATCH symlink.
- **`brava_long_20261027`** (27–28 Oct): D08, B0 on the full cohort for 300 epochs (82,200 updates), from scratch vs SyntheticMRI initialisation.
  DONE. Validation edge F1 0.00110 (scratch) vs 0.06794 (SyntheticMRI init); 96.69% of missed edges lack a recovered endpoint pair. The detector underfits its own training cohort. The 30 Oct node-heatmap probe later reached node F1 0.539 against D08's 0.356, from scratch (§6.4). `large/` is a SCRATCH symlink.
- **`d08_operating_points_20261028`** (28 Oct): D08 thresholds chosen on training data (CPU, saved predictions).
  DONE — negative. A node cut of 0.45 gave node F1 +0.01190 but doubled duplicates. The train-max-F1 edge cut gave edge F1 0.06892 with 2,890 extra false positives. The 0.5/0.5 thresholds were kept.

## 6. Native 2D RelationFormer (27–30 Oct) → §8

The 2D dense model (`dense_graph_2d_20261030`) superseded this line. Every Native2D-02 to -05 run
used a **broken backbone**: `init_weights: null` left a random, un-normalised ResNet101. Validation
only, one seed. The live folders `native2d_20261028`, `native2d_20261028_targetfix` and
`native2d_long_20261028` hold Native2D-02/-03 and the prepared datasets.

- **`native2d_20261027`** (27 Oct): Native2D-01, GPU correctness and tiny memorisation smoke.
  DONE. A tiny model fits two AV-WIDE crops to F1 1.0, so the implementation is correct. This measures memorisation, not validation. `runtime/` is a SCRATCH symlink.
- **`native2d_mixed_20261028`** (28 Oct): Native2D-04, 50 epochs of mixed plant/vessel pretraining, then 100 vessel-only epochs (B0 and A-xyz) on AV-WIDE, REAVER and OCTA-Synth.
  DONE, confounded. Validation node / edge F1 for OCTA-Synth B0 was 0.8038 / 0.5950, against 0.903 / 0.782 for the 2D dense model. Pretraining adds about 1.29 million target-crop exposures, the backbone was broken, and all six scratch100 models predicted empty graphs. **Still read by current code** (see the table at the top).
- **`native2d_straight_20261029`** (29 Oct): Native2D-05, straight-only OCTA-Synth B0, otherwise identical.
  DONE. Edge F1 0.5903 vs 0.5950 for the curved model. OCTA edges are nearly straight, so 2D cannot show a spline benefit.
- **`native2d_imagenet_20261030`** (30 Oct): WS1 (workstream 1), the backbone fix with ImageNet initialisation, trained from scratch on AV-WIDE and REAVER.
  DONE — negative. AV-WIDE node / edge F1 0.020 / 0.001, REAVER 0 / 0 (from a run json, not a report). The fix did not rescue scratch training.

## 7. Audits, reviews, coordination and cleanup records

- **`q0_ixi_audit_20261028`** (28 Oct): Q0 query-capacity audit. Contracted K8 targets peak at 153 (train) / 141 (val), so 120 queries are unsafe for IXI and the RelationFormer comparisons keep 256 (→ §2). Its `decision.json` lists N07 as "NOT RUN", which was true at 09:40 on 28 Oct and was overtaken by the later run.
- **`n07_status_audit_20261028`** (28 Oct): status audit that found N07 not yet run (smoke blocked by `AF_UNIX path too long`); it updated the registry and roadmap. Overtaken by `ixi_local_components_20261026`.
- **`afternoon_job_review_20261028`** (28 Oct): status snapshot of the jobs that finished on the afternoon of 28 Oct (2D B0 timeout, data-preparation failures, L08 smoke). It holds no results.
- **`completed_result_review_20261028`** (28 Oct): verification review of the final D08 and Native2D report sections (per-subject CSV, milestones).
- **`coordination`** (26 Oct): one H100 reservation record that the BRAVA representation study left for parallel agents.
- **`storage_cleanup`** (16, 25, 27 Oct): manifests of the three cleanups.
  - 25 Oct: zipped the prediction NPZs of `fine_feature_architecture`, `multiscale_pretrained` and `node_verifier_feasibility`.
  - 27 Oct: packed old launchers and audit/viewer files into `20261027/secondary-artifacts.zip`. `20261027/restore.py` restores them and refuses to overwrite differing files.
