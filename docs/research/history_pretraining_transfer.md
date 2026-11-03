# Pretraining and transfer for the 3D RelationFormer — consolidated history (to 29 Oct 2026)

**Status.** This file covers source-domain pretraining (plants, HALVS leaves, SyntheticMRI), source
rendering augmentation, transfer to SyntheticMRI and IXI, the IXI overfit diagnostic, the pretrained
multiscale pilot and the TopBrain time estimate. It consolidates archived documents and was written
on 3 Nov 2026. **Superseded by the dense traced model**
([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §1), which trains **from scratch,
with no pretraining**. That model beats every RelationFormer variant on all three 3D datasets.

## Context

The question was whether pretraining on a large, diverse *source* graph population improves 3D
vessel-graph extraction on a *target* dataset. It was tested with RelationFormer, a DETR-style
image-to-graph transformer. The targets were SyntheticMRI (synthetic MRA patches) and IXI (real MRA).
Candidate sources:
- plant branching (3D2Cut grapevine) and HALVS leaf veins, both 2D images embedded as thin slabs in a 64³ volume;
- SyntheticMRI as an auxiliary vascular source for IXI.

The standard protocol had two stages:
1. mixed source + target pretraining (graph losses on both domains, usually with adversarial domain
   adaptation);
2. target-only fine-tuning with straight edges or curved K8 edges.

Every run was compared with a target-only control given the same number of optimizer updates.
Checkpoints were chosen on validation. The September–October source comparisons then **evaluated
the chosen checkpoint once on the held-out test split**, so many numbers below are test numbers.
Every comparison is single seed.

## Shorthand

- **B0**: the baseline RelationFormer recipe (no node refinement, K8 spline head). See the glossary of
  [network_proposals_20261030.md](network_proposals_20261030.md).
- **B1**: B0 plus a second, finer CNN feature level (multiscale pilot only).
- **K8 / spline arm**: curved edges as clamped cubic B-splines with 8 controls, predicted as a
  *detached residual* on the chord. The geometry loss does not reach the detector (the `graph_only`
  route).
- **Straight arm**: the same detector with a frozen zero-residual head, so its edges are straight
  chords.
- **Delayed spline**: graph-only pretraining, with the spline head added only at fine-tuning.
  **Spline throughout**: spline loss in both stages.
- **MRI-only / IXI-only control**: the same two-stage schedule and update count, but with target
  data only. It therefore sees *more* target exposures than the mixed arm.
- **DA**: adversarial domain adaptation. Image- and instance-level discriminators with gradient
  reversal, plus a consistency term ([../experiments/domain_adaptation.md](../experiments/domain_adaptation.md)).
- **HALVS**: the leaf-vein dataset `HALVS_DATASET` (144 training / 6 validation leaves).
- **Branch F1 / Curve F1**: the project's common physical metrics. Branch F1 counts correctly
  reconstructed branches; Curve F1 counts centreline samples within 1 mm. **SMD** is the mean
  symmetric distance.
- **Q0**: the 28 Oct audit of IXI contracted-target capacity (object count per patch).

## 1. How the source domains were built [DONE]

- **Plants** were recovered from the sibling Image-to-Graph-Vessel repository
  ([plants-recovery](../archive/legacy-20261103/docs/research/pretraining/plants-recovery.md)):
  - 25,923 train / 6,421 val / 6,737 test patches in `plants_3d2cut/patches_3d`;
  - the network input is **not the photograph**, but a mask smoothed by four 3×3 all-ones
    convolutions ([textured-source-prototype](../archive/legacy-20261103/docs/research/pretraining/textured-source-prototype.md));
  - each patch is resized to 54², repeated over five slices and padded into an empty 64³ cube, so
    the source anatomy is a flat slab.
- **Plant K8 targets** ([generalization-training](../archive/legacy-20261103/docs/research/pretraining/generalization-training.md)):
  - 25,000 train / 480 val patches;
  - 18,177 intermediate degree-2 nodes contracted, with crop-boundary anchors kept;
  - plant coordinates stay in PNG pixel units, so pretraining on plants supervised the graph only.
- **HALVS leaves** ([leaves-replacement](../archive/legacy-20261103/docs/research/pretraining/leaves-replacement.md)):
  - split by parent leaf (144/6);
  - primary and secondary veins only;
  - 87.5% crop overlap gave 80,079 candidates, from which 25,000 train / 480 val patches were
    selected.
  - These are spatially correlated crops of 144 leaves, and the extracted graphs are approximate
    annotations, not exact ground truth.
- **Combined source** ([combined-sources](../archive/legacy-20261103/docs/research/pretraining/combined-sources.md)):
  - 12,500 plant + 12,500 leaf patches, keeping the 25,000-patch source budget;
  - both carry one "source" domain label.
- **Domain adaptation history.** In the legacy repository the domain loss was computed and logged,
  but `domain` was missing from the optimized loss list, so it never trained anything
  ([../experiments/domain_adaptation.md](../experiments/domain_adaptation.md)). The October runs
  re-enabled DA properly. The user chose weight 1.0 instead of the legacy 0.1, and the first
  batch-4 / weight-0.1 chain was cancelled and restarted from scratch at batch 32
  ([generalization-training](../archive/legacy-20261103/docs/research/pretraining/generalization-training.md)).

## 2. Full-data transfer to SyntheticMRI: plants, leaves, combined vs MRI-only [DONE — negative]

**Protocol** ([generalization-training](../archive/legacy-20261103/docs/research/pretraining/generalization-training.md),
[leaves-replacement](../archive/legacy-20261103/docs/research/pretraining/leaves-replacement.md)):
- **Pretraining:** 50 epochs of graph-only pretraining, 50,000 replacement draws per epoch
  (about 50:50 source/target), batch 32, DA weight 1.0 in the mixed arms.
- **Fine-tuning:** 100 epochs on 10,000 SyntheticMRI patches, as straight or K8 spline. Patience
  stopping was disabled on 16 Oct, so every run used the full 100 epochs.
- **Evaluation:** checkpoints selected by validation Branch F1 on 1,952 patches, then evaluated on
  all **13,071 held-out test patches**.

**Held-out test, spline arms** ([source-transfer test table](../archive/legacy-20261103/docs/research/pretraining/source-transfer-results/test-comparison.md),
[augmented test table](../archive/legacy-20261103/docs/research/pretraining/augmented-transfer-results/test-comparison.md)):

| Pretraining source | Node mAP | Edge F1 | Curve F1 |
|---|---:|---:|---:|
| MRI only (control) | **0.4788** | **0.7258** | **0.5097** |
| Plants + MRI | 0.4448 | 0.6879 | 0.4581 |
| Leaves + MRI | 0.4418 | 0.6905 | 0.4607 |
| Plants + leaves + MRI | 0.4281 | 0.6809 | 0.4484 |
| Plants (augmented, §3) + MRI | 0.4555 | 0.6935 | 0.4733 |
| Leaves (augmented) + MRI | 0.4302 | 0.6785 | 0.4473 |
| Plants + leaves (augmented) + MRI | 0.4428 | 0.6892 | 0.4627 |

- **MRI-only wins on every detection and curve metric.** Validation shows the same ranking: MRI-only
  spline edge F1 0.7224 against plants 0.6852 and leaves 0.6879
  ([validation table](../archive/legacy-20261103/docs/research/pretraining/augmented-transfer-results/validation-comparison.md)).
- **Leaves and plants are indistinguishable.** Mixing them is slightly worse than either alone.
- **Splines changed curve quality, not graph quality.** In every source, the spline arm raises
  Curve F1 over the straight arm at nearly equal edge F1. For example, MRI-only straight scores
  edge F1 0.7219 / Curve F1 0.4729, against 0.7258 / 0.5097 for the spline arm. The current doc
  reaches the same conclusion: training with or without splines gives identical graphs
  ([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §1).
- **Caveats:**
  - single seed;
  - the mixed arms differ from the control in source data *and* in the DA objective;
  - the control sees about twice as many MRI exposures;
  - geometry is measured against extracted guides.

## 3. Source rendering augmentation: nonplanar warp + textured background [DONE — no gain over MRI-only]

**Motivation.** The IXI overfit work (§5) showed that source inputs are five-voxel slabs in empty
backgrounds, while IXI has textured tissue
([ixi-overfit](../archive/legacy-20261103/docs/research/pretraining/ixi-overfit.md)). That gap was
proposed as a possible cause, explicitly a hypothesis.

- **Nonplanar prototype**, two samples
  ([nonplanar-source-prototype](../archive/legacy-20261103/docs/research/pretraining/nonplanar-source-prototype.md)):
  - one smooth height-field warp moves image, mask, nodes and paths together;
  - realized RMS distance from the best-fit plane was 1.80 and 2.23 voxels;
  - guide-in-mask support stayed unchanged.
  - It is still a bent sheet, not volumetric anatomy. This showed feasibility only.
- **Textured prototype**
  ([textured-source-prototype](../archive/legacy-20261103/docs/research/pretraining/textured-source-prototype.md)):
  - correlated Gaussian background and foreground texture, with soft boundaries;
  - the constants are illustrative and not fitted to MRI.
- **Integrated training policy**
  ([source-augmentation-training](../archive/legacy-20261103/docs/research/pretraining/source-augmentation-training.md)):
  - fresh warp and texture on every retrieval;
  - K8 targets refit to the warped paths, with mean refit error 0.00930 / 0.00698 voxels (plants)
    and 0.02508 / 0.02259 (leaves);
  - opt-in flags `source_augmentation.{enabled,nonplanar,textured_soft}`;
  - validation and test inputs unchanged.
- **Matched training comparison**: three sources × straight/spline, same protocol as §2
  ([augmented-source-comparison](../archive/legacy-20261103/docs/research/pretraining/augmented-source-comparison.md)).
  Test results are in the §2 table:
  - augmented plants improve slightly on plain plants (spline edge F1 0.6879 → 0.6935, Curve F1
    0.4581 → 0.4733);
  - augmented leaves get worse (0.6905 → 0.6785);
  - the augmented mixture improves (0.6809 → 0.6892);
  - **no augmented arm reaches MRI-only.**
  - This tests the combined policy, not warp and texture separately, with one seed. No
    interpretation document was written after the collector tables appeared. The launch document
    still says "results pending".

## 4. Limited-data transfer: 200 MRI patches, leaves vs vessel-only [DONE — positive, single seed]

**Question.** Does leaf pretraining help when only 200 annotated vessel patches exist (2% of the
full set, 62 training volumes)?
([limited-data-botanical-transfer](../archive/legacy-20261103/docs/research/pretraining/limited-data-botanical-transfer.md))

**Protocol:**
- arm A uses the 200 MRI patches only;
- arm B pretrains on the same 200 patches plus 25,000 leaf patches, then fine-tunes on the 200;
- **DA is off in both arms**, so the loss definitions are identical;
- both arms get the same updates (78,150 pretraining + 31,300 fine-tuning) and the same 32-example
  batches;
- target exposure is *not* matched: arm B saw 1,249,796 MRI pretraining presentations against
  2,500,800 for arm A.

| Held-out test (13,071 patches) | Node F1 | Edge F1 | Branch F1 | Curve F1 | Component MAE | Cycle MAE |
|---|---:|---:|---:|---:|---:|---:|
| Vessel only | 0.3246 | 0.2588 | 0.0004 | 0.1009 | 2.4808 | 3.5468 |
| Leaves + vessels | **0.3855** | **0.2979** | 0.0005 | **0.1206** | 2.6998 | **3.3283** |

- **Leaf pretraining helped:** +0.0390 edge F1 and +0.0197 Curve F1 on test, with similar gains on
  validation.
- **Not every metric improved:** component error got worse and Branch F1 stayed about 0 in both
  arms.
- **Both arms memorized their 200 patches** (final train edge F1 0.9997).
- This is the **only positive botanical result**. It contrasts with the full-data result in §2. It
  was never replicated with a second seed and never tested with the dense model.

## 5. Transfer to IXI, and why real data was hard

### 5.1 Plants / SyntheticMRI → IXI [DONE — negative]

**Design** ([ixi-transfer](../archive/legacy-20261103/docs/research/pretraining/ixi-transfer.md)):
- six arms: IXI-only, plants+IXI and SyntheticMRI+IXI, each straight or spline throughout;
- three delayed-spline arms added later, reusing the straight pretraining checkpoints;
- data and model: 10,000 IXI training patches, 25,000 source patches, 256 object queries
  (SyntheticMRI used 120), DA weight 1.0 in the mixed pretraining;
- fine-tuning ran the full 100 epochs; three early-stopped arms were resumed to 100.

Held-out IXI test, 2,178 patches
([textured-source-prototype](../archive/legacy-20261103/docs/research/pretraining/textured-source-prototype.md),
the latest source with the final 100-epoch numbers):

| Pretraining | Representation | Node F1 | Edge F1 | Curve F1 |
|---|---|---:|---:|---:|
| IXI only | straight | **0.47404** | 0.15985 | 0.11644 |
| Plants + IXI | straight | 0.44480 | 0.12084 | 0.09022 |
| SyntheticMRI + IXI | straight | 0.44676 | 0.11688 | 0.08831 |
| IXI only | spline throughout | 0.47369 | **0.16395** | **0.12215** |
| Plants + IXI | spline throughout | 0.45038 | 0.12090 | 0.09441 |
| SyntheticMRI + IXI | spline throughout | 0.44171 | 0.11126 | 0.09214 |
| SyntheticMRI + IXI | delayed spline | 0.44243 | 0.11373 | 0.08963 |

- Neither plants nor SyntheticMRI helped on IXI. The IXI-only control was best.
- The control matches updates, not IXI exposure: all of its draws are IXI.
- The test results of the IXI-only and plants delayed-spline arms are not written up in any
  document.
- All IXI scores were low, which prompted §5.2.
- **Q0 audit (28 Oct, read-only)** ([ixi-transfer](../archive/legacy-20261103/docs/research/pretraining/ixi-transfer.md), last section):
  - 10,000 train / 2,649 validation contracted targets;
  - maximum 153 / 141 anchors per patch; five train and five validation patches exceed 120;
  - so **256 queries are capacity-safe for IXI and 120 are not**.

### 5.2 IXI four-patch overfit diagnostic [DONE]

**Design** ([ixi-overfit](../archive/legacy-20261103/docs/research/pretraining/ixi-overfit.md)):
- start from the IXI-only straight pretraining checkpoint;
- 1,000 updates on four fixed training patches from four subjects;
- score four disjoint validation patches.

| Arm | Train loss | Val loss | Train node / edge F1 | Train Curve F1 | Same-endpoint chord Curve F1 | Val edge F1 |
|---|---|---|---|---:|---:|---:|
| Straight | 2.4292 → 0.0292 | 3.4769 → 8.4953 | 1.0000 / 1.0000 | 0.5952 | 0.5952 | 0.0988 |
| Spline | 2.8199 → 0.1122 | 3.9079 → 9.4349 | 0.9688 / 0.9000 | 0.8959 | 0.5323 | 0.1370 |

- The pretrained detector **can memorize real IXI inputs**, and learned splines follow curved
  references much better than chords. Validation loss rises, which is plain overfitting on a tiny
  set.
- This rules out a blanket inability to fit, nothing more. Later work showed the real failure:
  RelationFormer does not fit the *full* IXI training set. Train node/edge F1 is about 0.39/0.09,
  which the later work traces to the stride-32 2×2×2 feature grid, the train/inference mismatch of
  Hungarian matching, and edge scoring that never looks at the image
  ([network_proposals_20261030.md](network_proposals_20261030.md) §0;
  [dense_traced_model_20261102.md](dense_traced_model_20261102.md) §2.1).

### 5.3 Earlier plants → IXI DA run, September [RUNNING when last documented; no final result in the docs]

**Plan** ([evidence/domain_ixi_pretraining/README.md](evidence/domain_ixi_pretraining/README.md)):
- 50-epoch pretraining on 25,900 plants + 8,000 IXI, with DA and a node-type head;
- then 100 IXI-only epochs.

The smoke runs passed, with a peak of 10.28 GiB at batch 32. No result for this run or its matched
no-node-type control appears in any document.

The test reports in the same folder are a different model. They evaluate an **IXI-only model
trained from random initialization for 300 epochs** ([evidence/T10_node_type_pilot.md](evidence/T10_node_type_pilot.md)),
not a plants-pretrained one, on 2,384 test patches:
- plain decoder: node F1 0.5103, edge F1 0.3894, predicted β₁ 82.82 against target 0.77, so the
  graph is strongly over-connected
  ([evidence/domain_ixi_pretraining/test_types300.md](evidence/domain_ixi_pretraining/test_types300.md));
- soft degree penalty 0.5 at inference: edge F1 0.3894 → 0.4847, β₁ error 82.05 → 28.92
  ([evidence/domain_ixi_pretraining/test_decoder_comparison.md](evidence/domain_ixi_pretraining/test_decoder_comparison.md)).

These IXI numbers use an earlier target set and protocol, so do not compare them with §5.1.

## 6. Multiscale attention on a pretrained checkpoint (B1) [DONE — negative]

**Design** ([multiscale_pretrained_feasibility](../archive/legacy-20261103/docs/research/multiscale_pretrained_feasibility.md)):
- start from the MRI-only K8 checkpoint of §2;
- B1 adds the `layer3` 4³ CNN level next to B0's 2³ level, with migration that keeps the coarse
  path intact;
- 2,000 matched updates per arm on SyntheticMRI;
- validation only: 480 patches from 13 subjects, subject-paired bootstrap CIs, one seed.

**Results:**
- **Node F1 @2 mm:** B1 − B0 = +0.0125 [+0.0061, +0.0185].
- **Edge F1 @2 mm:** −0.0034 [−0.0170, +0.0089]; conditional relation recall −0.0200
  [−0.0306, −0.0102].
- **The fine level was essentially unused.**
  - Final fine-attention mass was 0.0014–0.0065%, below its initial 0.0100%.
  - Blocking it changed node/edge F1 by +0.0003.
  - Its gradient fell about 85-fold.
- **Both arms degraded from the starting checkpoint.** On a fixed 64-patch monitor, B0 node/edge
  F1 fell from 0.7476 / 0.4460 to 0.6033 / 0.2729. This points to the loss change and optimizer
  restart, not to the architecture.
- **Not done:** the proposed from-scratch B0/B1 comparison was never run. The full-resolution
  U-Net of the dense model later addressed the coarse feature grid directly
  ([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §2).
- Evidence: [evidence/multiscale_pretrained/pilot.png](evidence/multiscale_pretrained/pilot.png).

## 7. TopBrain mixed-pretraining time estimate [PROPOSED, never run in these documents]

[evidence/topbrain_training_time_estimate/README.md](evidence/topbrain_training_time_estimate/README.md):
- **Estimate:** about **9.5 h** of compute (range 7.3–13.0 h) for one model, with 50 mixed
  epochs on one H100 and 100 fine-tuning epochs on four H100s.
- **Basis:** an estimated 28,843 TopBrain training patches. It is an extrapolation from an 81.30 ms
  IXI-only batch-8 step, not a timing run.
- **Single-GPU alternative:** the 1+1 GPU layout is estimated at about 13.1 h. That is the layout
  allowed by today's one-GPU-per-job policy.
- No TopBrain pretraining run appears in the assigned sources.

## What carried forward / what was dropped

**Carried forward:**
- **Target-only training at full data.** With 10,000 target patches, biological or synthetic
  pretraining never beat a matched target-only control (SyntheticMRI and IXI). The dense traced
  model trains from scratch and still wins
  ([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §3).
- **SyntheticMRI initialization for RelationFormer baselines.** The current RelationFormer
  comparison arms are still SyntheticMRI-pretrained, then fine-tuned
  ([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §3). On BRAVA, that
  initialization clearly beat scratch for RelationFormer: validation edge F1 0.06794 against 0.00110
  ([decisions](../archive/legacy-20261103/research/decisions.md), 28 Oct).
- **Splines improve curves, not graphs.** This held in every transfer pair and matches the current
  doc.
- **IXI query capacity.** 256 queries are needed for IXI; 120 are not enough (Q0).

**Open:**
- The one positive result, leaves at 200 patches, is single seed and untested with the dense
  model. Low-label botanical pretraining is still an open idea, not a dropped one.

**Dropped:**
- adversarial DA;
- the plant and leaf source pipelines and their augmentation (the code remains opt-in);
- SyntheticMRI → IXI transfer;
- multiscale attention on RelationFormer;
- the TopBrain mixed-pretraining plan.

## Sources

Archived originals (paths after the move):
- [pretraining/plants-recovery.md](../archive/legacy-20261103/docs/research/pretraining/plants-recovery.md): recovery of the legacy plant loader, generator and dataset layout.
- [pretraining/generalization-training.md](../archive/legacy-20261103/docs/research/pretraining/generalization-training.md): plant vs MRI-only protocol, spline augmentation, DA weight-1.0 restart.
- [pretraining/leaves-replacement.md](../archive/legacy-20261103/docs/research/pretraining/leaves-replacement.md): HALVS extraction, leaf pretraining, leaf/plant/MRI-only validation and test.
- [pretraining/combined-sources.md](../archive/legacy-20261103/docs/research/pretraining/combined-sources.md): plants + leaves 12.5k + 12.5k protocol.
- [pretraining/source-transfer-results/](../archive/legacy-20261103/docs/research/pretraining/source-transfer-results/test-comparison.md): six-model test and validation tables (`.md` + `.csv`).
- [pretraining/combined-transfer-results/](../archive/legacy-20261103/docs/research/pretraining/combined-transfer-results/test-comparison.md): eight-model test table (`.md` + `.csv`).
- [pretraining/augmented-transfer-results/](../archive/legacy-20261103/docs/research/pretraining/augmented-transfer-results/test-comparison.md): fourteen-model test and validation tables (`.md` + `.csv`).
- [pretraining/nonplanar-source-prototype.md](../archive/legacy-20261103/docs/research/pretraining/nonplanar-source-prototype.md): two-sample nonplanar warp prototype.
- [pretraining/textured-source-prototype.md](../archive/legacy-20261103/docs/research/pretraining/textured-source-prototype.md): textured background prototype and the final IXI transfer test table.
- [pretraining/source-augmentation-training.md](../archive/legacy-20261103/docs/research/pretraining/source-augmentation-training.md): integrated augmentation policy and loader checks.
- [pretraining/augmented-source-comparison.md](../archive/legacy-20261103/docs/research/pretraining/augmented-source-comparison.md): matched augmented-source training protocol.
- [pretraining/limited-data-botanical-transfer.md](../archive/legacy-20261103/docs/research/pretraining/limited-data-botanical-transfer.md): 200-patch leaves vs vessel-only study and results.
- [pretraining/ixi-transfer.md](../archive/legacy-20261103/docs/research/pretraining/ixi-transfer.md): IXI six + three arm design, recoveries, Q0 capacity audit.
- [pretraining/ixi-overfit.md](../archive/legacy-20261103/docs/research/pretraining/ixi-overfit.md): IXI four-patch overfit diagnostic and source-embedding proposal.
- [pretraining/previews/](../archive/legacy-20261103/docs/research/pretraining/previews/halvs-smoke.png) and [pretraining/ixi-overfit-previews/](../archive/legacy-20261103/docs/research/pretraining/ixi-overfit-previews/source-embedding.png): QC images for HALVS targets, source embedding and IXI references.
- [multiscale_pretrained_feasibility.md](../archive/legacy-20261103/docs/research/multiscale_pretrained_feasibility.md): pretrained B0 vs B1 pilot.

Evidence and reference docs that stay in place:
- [evidence/domain_ixi_pretraining/README.md](evidence/domain_ixi_pretraining/README.md), [test_types300.md](evidence/domain_ixi_pretraining/test_types300.md), [test_decoder_comparison.md](evidence/domain_ixi_pretraining/test_decoder_comparison.md): September plants → IXI plan and IXI node-type test results.
- [evidence/multiscale_pretrained/](evidence/multiscale_pretrained/pilot.png): pilot figure.
- [evidence/topbrain_training_time_estimate/README.md](evidence/topbrain_training_time_estimate/README.md): TopBrain time estimate.
- [../experiments/domain_adaptation.md](../experiments/domain_adaptation.md): DA objective and the legacy inert-loss bug.
- Companion: [history_native2d.md](history_native2d.md), which covers 2D mixed plant pretraining and the legacy Image-to-Graph-Vessel source ablations.
