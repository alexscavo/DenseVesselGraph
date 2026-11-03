# Multi-dataset and BRAVA RelationFormer work, 21–29 Oct 2026 — consolidated history (to 29 Oct 2026)

**Status.** This document covers the late-October RelationFormer work on SyntheticMRI, IXI and BRAVA: BRAVA training from zero-shot to the long D08 run, learned continuous-geometry heads, node-localization and node-selection studies, relation-supervision tweaks, and the planning and decision documents of that period. It was written on 3 Nov 2026 by consolidating archived documents (listed under Sources).
**Superseded by** the two-stage dense model with traced connectivity, chosen on 30 Oct and described in [`dense_traced_model_20261102.md`](dense_traced_model_20261102.md). Its design, glossary and comparisons are in [`network_proposals_20261030.md`](network_proposals_20261030.md). The section "How this work led to the 30 Oct diagnosis" below explains how the two connect.

Related histories: earlier IXI diagnostics (series 1–11, T09, T10) are in [`history_relationformer_3d.md`](history_relationformer_3d.md). Spline-head learning is in [`history_continuous_edges.md`](history_continuous_edges.md); GT representation and spline-target capacity audits are in [`history_representation.md`](history_representation.md). Extraction and datasets are in [`history_data.md`](history_data.md); metric definitions in [`history_metrics.md`](history_metrics.md). Pretraining and botanical transfer are in [`history_pretraining_transfer.md`](history_pretraining_transfer.md). Native 2D work (Native2D-01..05, AV-WIDE, REAVER, OCTA) is in [`history_native2d.md`](history_native2d.md).

## Context

The questions in this period were:
- whether the RelationFormer (DETR-style, object queries plus Hungarian matching) graph detector with K8 spline edges could be made to work on real MRA (IXI, and BRAVA's manual SWC traces), not only on SyntheticMRI;
- whether richer edge geometry (more controls, adaptive K, piecewise C0/C1, degree-2 subdivision) improved reconstruction.

Protocol, common to almost everything below:
- **Data.** 64³ patches. The BRAVA split is 39 train / 8 validation / 9 test subjects, with 8,768 / 1,796 / 2,043 patches ([`BRAVA_preparation.md`](BRAVA_preparation.md)). IXI development uses 123 / 26 subjects. SyntheticMRI uses 64 / 13 volumes, with 480 validation patches.
- **Evaluation.**
  - Physical one-to-one node matching at 1 and 2 mm.
  - Edge F1 counts an edge as correct when both endpoints are matched.
  - Branch and Curve F1 use the common-v2 protocol.
  - Uncertainty comes from subject-paired bootstrap intervals, which do not measure seed variance.
- **Test splits** were never evaluated, except where stated.
- **BRAVA caveats.**
  - BRAVA `seg/` files are brain masks, not vessel labels.
  - SWC traces are incomplete, so every BRAVA false positive, false negative and Betti error is *relative to the annotation*.
- Almost every result is **single-seed and validation-only**, and the same validation subjects were reused across many studies.

## Shorthand

Codes follow the glossary of [`network_proposals_20261030.md`](network_proposals_20261030.md) where it defines them. The stable IDs (N0x, R0x, D0x, G0x, L0x, Q0) come from the experiment registry.

- **B0.** Baseline RelationFormer recipe: SE-ResNet (stride 32) plus a deformable transformer, K8 spline head, no node refinement. It uses node softmax focal γ=2 with equal weights, matched-pair edge CE with ratio upsampling 0.15, and the `graph_only` route.
- **S0.** The clean SyntheticMRI reference checkpoint: B0 trained 100 epochs from scratch, with epoch 95 selected (120 queries). It is the parent of most later runs.
- **A / A-xyz / A-cls / A-control.** Local refinement heads that sample early high-resolution CNN features around each node.
  - A corrects coordinates and confidence.
  - A-xyz corrects coordinates only.
  - A-cls corrects confidence only (IXI N07).
  - A-control has the same head with zeroed image input.
- **C.** A two-level (fine plus coarse) deformable-attention variant.
- **K8 / K16 / K24.** Clamped cubic B-spline edges with 8 / 16 / 24 controls.
  - **C0 / C1:** four-section piecewise splines with shared joins or tangent continuity.
  - **Adaptive K:** a bank of K-specific output projections plus a separately trained selector.
  - **split_K8:** explicit degree-2 subdivision of the graph.
- **`graph_only`.** The spline loss trains only the geometry head; no gradient reaches the detector.
- **Raw vs filtered recall.** Raw recall is node recall over all queries (an oracle bound on proposals). Filtered recall is recall after the 0.5 confidence cut.
- **Conditional relation recall / AP.** Relation quality restricted to GT edges whose two endpoints were recovered.
- **D1 / D2 / D3.** BRAVA recovery arms of 23 Oct, initialized from IXI, from SyntheticMRI, and from scratch. They are *not* the registry IDs below.
- **D06.** BRAVA K8 representative-cohort learnability/BN diagnosis (also called the "K8 learning diagnostic").
- **D07.** BRAVA augmentation on/off ablation.
- **D08.** Full-cohort 300-epoch BRAVA run, scratch vs SyntheticMRI initialization.
- **G05.** BRAVA representation follow-up: K8 vs K24/C1 heads, and split_K8.
- **V0–V3.** Node-verifier arms:
  - V0: original confidence;
  - V1: training-selected threshold plus NMS;
  - V2: independent learned rescoring;
  - V3: contextual (5-NN) learned rescoring.
- **N07.** Modern IXI B0 / A-xyz / A-cls transfer from S0 (the registry also files it as D04).
- **N08.** GT-free relation-aware query selection.
- **R06.** Relation-specific CNN token fusion.
- **R07.** Reliable nearby-negative relation supervision.
- **R01 / R02.** False-edge and missed-edge attribution studies (legacy IXI and modern SyntheticMRI).
- **L08.** Node-first edge curriculum from scratch.
- **Q0 (J00/J02).** IXI status and contracted-target capacity audit.
- **J01.** The D08 cached operating-point audit.
- **M01 / N09 / N10 / T01.** Proposed journal-plan experiments:
  - M01: spatially gated Hungarian;
  - N09: query denoising;
  - N10: image-conditioned node proposals;
  - T01: controlled botanical transfer.
- **`monitor_train` / representative monitor.** A fixed subset of training patches used to select thresholds and track fit. In the BRAVA D06–D08 studies this is 160 patches covering all 39 training subjects.

## Findings

### 1. BRAVA: from zero-shot to the long D08 run

**1.1 Stage 1 zero-shot and short scratch training (21–22 Oct) [DONE — negative].**
- **Zero-shot.** Existing models were evaluated on all 1,796 BRAVA validation patches without retraining. The IXI spline model was the best starting point: node recall 0.2309, edge F1 0.0269, Curve F1 0.0580. SyntheticMRI models reached node recall of only about 0.035.
- **Scratch arms.** Two scratch BRAVA arms, standard and coordinate refinement, ran 23,500 updates each. They recovered almost nothing: 4 and 1 correct edges, with edge F1 0.0004 and 0.0001.
- **Refinement.** Coordinate refinement raised all-query coverage (32.19% → 34.68%) but lowered filtered recall (10.78% → 7.95%). The paired interval for its node-recall change was [−0.0306, −0.0256].
- **Geometry.** With so few correct edges, the geometry comparison was uninformative. On the same IXI zero-shot graph, spline minus chord Curve F1 was +0.0026.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [evidence](evidence/brava_stage1_20261021/results.md), [completion review](evidence/brava_stage1_20261021/completion-review.md))

**1.2 Recovery arms D1–D3 (22–23 Oct) [DONE — negative, gate failed].**
- **Setup.** Fully trainable detectors, 30,000 updates, cosine LR schedule with a 30% floor. The recipe used weighted node CE [0.25, 0.75] and edge focal.
- **Results at the default 0.5 threshold:**

  | Arm | Node recall | Raw recall | Edge F1 |
  |---|---:|---:|---:|
  | D1 (IXI) | 0.341 | 0.563 | 0.0210 |
  | D2 (SyntheticMRI) | 0.273 | 0.485 | 0.0134 |
  | D3 (scratch) | 0.106 | 0.313 | 0.0004 |

- **Gate.** The predeclared gate (edge F1 > 0.04 and node recall > 0.25) failed.
- **Training-selected threshold.** D1's train-selected edge threshold gave 0.03696 edge F1, but with 21,236 false edges and a cycle MAE of 2.211.
- **Scratch fit.** Scratch was equally poor on monitored *training* patches (node recall 0.1159, edge F1 0), so poor generalization alone could not explain it.
- **Unmatched queries.** 3,862 raw pairs that were spatially GT-positive involved a Hungarian-unassigned query. So "unmatched" is not a safe negative label.

([source](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261023.md), [protocol](../archive/legacy-20261103/docs/research/multidataset_continuous_graph_experiments.md), [evidence](evidence/multidataset_continuous/report.md))

**1.3 BRAVA tiny-set learnability (23 Oct) [DONE].**
- **Setup.** Eight BRAVA training patches, 512 updates, edge CE, no augmentation.
- **Result.** Both initializations memorized them. Eval-mode edge F1 was 0.8393 from scratch and 0.9818 from IXI. Localization gradients were finite and nonzero.
- **Interpretation.** The implementation can learn BRAVA targets. The failure at cohort scale lies elsewhere.

([source](../archive/legacy-20261103/docs/research/node_verifier_feasibility.md))

**1.4 G05 representation follow-up (26 Oct) [DONE — negative/mixed].**
- **Setup.**
  - Bounded cohort: 1,248 training and 256 validation patches.
  - Initialization from S0; 4,000 updates.
  - K8, K24 and C1 heads trained on one shared detector, plus a separately trained split_K8 detector.
- **Graph quality.** Every arm recovered **zero** complete common-v2 branches. Original K8 reached node F1 0.1183, edge F1 0.0089, Curve F1 0.0241.
- **K24 and C1.** They fit GT much better: on 141 difficult branches, mean maximum error was 1.571 → 0.397 → 0.360 mm. Learned Curve F1, however, changed by only +0.0001 / +0.0002.
- **split_K8.**
  - Curve F1 rose by +0.0424 [0.0348, 0.0494].
  - The cost was more predicted nodes (1,271 → 2,154) and edges (435 → 1,147), and a higher cycle MAE (+0.6094).
- **Proposals and confidence.**
  - Raw 2-mm recall was 0.307, but only 0.0767 survived the confidence threshold.
  - The confidence score separated correct nodes from duplicates at near-chance level (AUC 0.518).
- **Decision.** Keep K8 as the controlled baseline. Do not promote K24, C1 or subdivision while detection is weak.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [artifacts](../../experiments/brava_representation_20261026/analysis.json))

**1.5 D06 K8 learnability and BN diagnosis (26 Oct) [DONE].**
- **Setup.** Inference-only analysis on a representative 160-patch training monitor that spans complexity strata across all 39 subjects. The earlier 39-patch monitor had been unrepresentative (8.67 vs 16.70 nodes per patch).
- **Result at update 4,000:**
  - Train node/edge F1 0.1325 / 0.0114, against validation 0.1183 / 0.0089: the cohort is not fitted at all.
  - Training-only BN recalibration did not help (validation 0.1108 / 0.0078); batch statistics made it worse.
  - Augmentation alignment passed all 256 checks.
- **Interpretation.** The failure is not held-out overfitting and not a BN artefact.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [artifacts](../../experiments/brava_learning_diagnosis_20261026/))

**1.6 D07 augmentation on/off (27 Oct) [DONE — mixed].**
- **Setup.** Same S0 source, 1,248 patches, seed, patch order and 4,000 updates. The only change was augmentation off (quarter-turns plus Gaussian noise) vs on.
- **Effect of turning augmentation off** (off − on, at 4k):

  | Population | Node F1 | Edge F1 |
  |---|---|---|
  | Train | +0.2908 [+0.2658, +0.3146] | +0.1551 [+0.1294, +0.1785] |
  | Validation | +0.0285 [+0.0189, +0.0379] | +0.0062 [−0.0009, +0.0125] |

- **Signs of memorization.**
  - Validation raw recall fell (0.3215 → 0.3063).
  - Validation loss rose (5.469 → 6.098).
  - Cycle MAE rose (0.246 → 1.480).
- **Conclusion.** The augmentation regime has an optimization cost, but turning it off is not a recovered detector.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [artifacts](../../experiments/brava_augmentation_20261026/analysis.json))

**1.7 D08 full-cohort long run (27–28 Oct) [DONE].**
- **Setup.**
  - B0 with 256 queries and K8; 8,768 patches.
  - 300 epochs = 82,200 updates at batch 32.
  - Scratch vs initialization from a SyntheticMRI B0 checkpoint (local-refinement study, seed 92741, update 12,000).
  - Identical augmented data streams for both arms; about 8 H100-hours per arm.
- **Final validation results at 2 mm, threshold 0.5:**

  | Metric | Scratch | SyntheticMRI init |
  |---|---:|---:|
  | Node F1 | 0.02544 | 0.35572 |
  | Edge F1 | 0.00110 | 0.06794 |
  | Raw / filtered node recall | 0.48288 / 0.01342 | 0.65530 / 0.28643 |
  | Branch / Curve F1 | 0.00000 / 0.00302 | 0.00647 / 0.12174 |
  | Train-monitor node / edge F1 | 0.02282 / 0.00000 | 0.39449 / 0.08989 |

- **Initialization.** Pretrained minus scratch: node F1 +0.33027 [0.32262, 0.33879], edge F1 +0.06684 [0.06245, 0.07032].
- **More epochs barely helped.** Pretrained edge F1 moved 0.06542 → 0.06794 between epochs 200 and 300.
- **The errors are node-driven.** In the pretrained arm:
  - 17,839 of 18,450 missed edges (96.69%) lack a recovered endpoint pair;
  - 89.40% of false edges touch a spatially unmatched node;
  - conditional relation AP is 0.605.
- **Conclusion.** The detector underfits its own training cohort even with a long budget. Scratch nearly fails.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [final report](../../experiments/brava_long_20261027/report.md), [review tables](../../experiments/completed_result_review_20261028/review.json))

**1.8 D08 training-only operating-point audit (28 Oct) [DONE — negative].** The audit ran on CPU from cached predictions. Thresholds were selected on the 160 training patches only.
- **Node threshold.** Lowering the pretrained arm's node cut to 0.45 moved validation node F1 0.35572 → 0.36761, +0.01190 [0.01009, 0.01359]. Recall rose and precision fell, and duplicates doubled (621 → 1,201).
- **Edge threshold.** The train-max-F1 edge cut moved edge F1 0.06794 → 0.06892. It added 118 TP and 2,890 FP, lowered Branch F1 to 0.00576, and raised cycle MAE to 1.82461. Ranking (AP) is unchanged by construction.
- **Not evaluated.** A combined node-plus-edge graph was not evaluated: the caches lacked relation scores for newly admitted queries.
- **Decision.** Keep the 0.5/0.5 decoder. The threshold is not the fix.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md), [decision](../archive/legacy-20261103/research/decisions.md), [artifacts](../../experiments/d08_operating_points_20261028/))

**1.9 The planned BRAVA Stage 2 matrix [PROPOSED, never run].** The 21 Oct plan listed these comparisons:
- independently trained straight vs spline models;
- BRAVA-only vs SyntheticMRI vs botanical transfer;
- 25% vs 100% annotations (a 10-subject, 2,243-patch subset was prepared but marked not-for-training);
- object-query budgets;
- a second seed.

Only the SyntheticMRI-transfer question was later answered, by D08 under a different recipe. The rest were never launched.

([source](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md))

### 2. Learned continuous geometry on frozen detectors (22–23 Oct) [DONE — negative/mixed]

**Setup.** Six geometry heads were trained for 20,000 updates on frozen IXI and SyntheticMRI detectors: K8, K16, K24, adaptive K, piece-C0 and piece-C1. Every head saw identical nodes and edges, so only the edge shape differs.

**Curve F1:**
- IXI: 0.1211–0.1222 across all heads;
- SyntheticMRI: 0.5817–0.5842.

**Paired differences against K8** were tiny and inconsistent:
- IXI: K24 +0.00056 and C1 +0.00067, both with intervals above zero;
- SyntheticMRI: adaptive −0.0022 and C1 −0.0011, both negative.

**Adaptive K selector.**
- It averaged 4.31 / 4.54 controls per edge.
- Its accuracy (91.02% on IXI) was below the always-K4 baseline (91.58%).
- It rarely chose complex classes: K8/K12 recall on IXI was 36.5% / 15.0%.

**Decision.** Keep K8 as the learned control; no representation was selected.

**Relation to the GT audit.** The GT capacity audit of the same period found K8 adequate for IXI. It was *not* adequate for long BRAVA branches: 88.37% of 50–100 mm BRAVA validation branches exceed 1 mm error. Details are in [`history_continuous_edges.md`](history_continuous_edges.md).

([source](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261023.md), [capacity summary](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261022.md), [evidence](evidence/multidataset_continuous/report.md))

### 3. Node localization with fine image features (SyntheticMRI, IXI)

**3.1 S0 baseline and fine-feature pilots (24–25 Oct) [DONE].**
- **S0.** A fresh 100-epoch scratch SyntheticMRI B0, with node focal plus edge CE.
  - Validation node/edge F1 at 2 mm: 0.7882 / 0.5881, against 0.7670 / 0.4701 for the older historical checkpoint.
  - Its training scores (0.7910 / 0.5894) were close to validation, so SyntheticMRI fits.
- **3,000-update fully unfrozen pilots from S0:**
  - **A beat B0** by +0.0182 [0.0142, 0.0220] edge F1 at 2 mm, and by +0.0345 node F1 at 1 mm.
  - Zeroing A's image input cost −0.0328 edge F1, so the head really uses image information.
  - **C (multiscale attention)** used its fine level but showed no gain (edge −0.0034 [−0.0109, +0.0027]). It was deferred, not rejected.

([source](../archive/legacy-20261103/docs/research/fine_feature_architecture_study.md), [figures](evidence/fine_feature_architecture/pilots.png))

**3.2 Long local-refinement study (25 Oct) [DONE].**
- **Setup.** Two seeds, 12,000 updates, B0 vs A vs A-xyz vs A-control, all from S0.
- **Two-seed means:**

  | Arm | Node F1 @1 mm | Edge F1 @2 mm |
  |---|---:|---:|
  | B0 | 0.4886 | 0.5946 |
  | A | 0.5695 | 0.6368 |
  | A-xyz | 0.5701 | 0.6339 |
  | A-control | — | 0.5947 |

- **A-xyz minus B0.** Edge F1 +0.0393 [+0.0313, +0.0467]. Hungarian error fell from about 1.344 to 1.213 mm.
- **Confidence correction is not needed.** A-xyz minus A: −0.0029 [−0.0087, +0.0035]. A-xyz was therefore preferred for simplicity.
- **Limits.** The gain is localization. Cycle error (~1.5 per patch) and duplicates did not improve.
- **Recorded in the 30 Oct glossary.** The 2-mm node F1 of this study (B0 0.793, A-xyz 0.817) is the RelationFormer reference that the 30 Oct glossary quotes.

([source](../archive/legacy-20261103/docs/research/local_refinement_long_study.md), [artifacts](../../experiments/local_refinement_long/analysis.json))

**3.3 N07 modern IXI transfer (28 Oct) [DONE].**
- **Setup.** Six arms from S0 with 256 queries; 5,500 updates; 4,096 training and 512 validation patches; one seed. It first failed twice on an over-long multiprocessing socket path and was recovered the same day.
- **Edge F1 at 2 mm:**

  | Arm | Edge F1 |
  |---|---:|
  | B0 | 0.0879 |
  | A-xyz (fresh) | 0.0943 |
  | A-xyz (transferred) | 0.0946 |
  | A-cls | 0.1015 |
  | A | 0.0917 |
  | A-control | 0.0878 |

- **A-cls minus B0.** +0.013610 [0.008265, 0.019401], and still +0.010862 at matched node counts.
- **Costs.** Duplicates rose 313 → 535 and cycle MAE 1.998 → 2.580.
- **Node-driven errors.** 93.02% of missed edges lack recovered endpoints.
- **Conclusion.** IXI graphs remain weak.

([source](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md), [example overlay](evidence/ixi_local_components_20261028/sample_000001_0009-comparison.png))

### 4. Node selection and relation fixes within RelationFormer (all [DONE — negative])

- **Node verifier V0–V3 (23 Oct).**
  - Setup: learned rescoring of frozen decoder tokens (58,129 parameters).
  - V3 vs V1 at matched node counts: node F1 [−0.0810, −0.0563] on IXI and [−0.0819, −0.0629] on SyntheticMRI.
  - The verifiers fit training data (AUC near 1), but validation AP fell below the original scores.
  - Training selected *no* NMS for the original scores.
  - Decision: keep V1 as control; no BRAVA transfer. ([source](../archive/legacy-20261103/docs/research/node_verifier_feasibility.md))
- **N08 GT-free query selection (26 Oct).**
  - Setup: CPU-only, two SyntheticMRI A-xyz seeds.
  - Result: the local relation-profile rule tied count-matched NMS (edge F1 0.637638 vs 0.637675 in seed 1; both intervals include zero).
  - A "guarded support" rule added 106 TP but 283 FP.
  - Decision: stop the heuristic search. The GT-assisted oracle (edge F1 about 0.834) is not deployable. ([source](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md))
- **R06 relation-specific CNN token fusion (27 Oct).**
  - Setup: frozen A-xyz, two seeds.
  - R2 conditional AP was 0.97357 / 0.97482, below R0 (original) at 0.97763 / 0.97797. The gate failed. (same source)
- **R07 reliable nearby-negative supervision (27 Oct).**
  - All matched negatives were already in training.
  - Fourfold reliable-negative exposure changed edge F1 by only +0.000755 / +0.000684 against the continued control, and stayed below the original A-xyz. All-retained AP worsened.
  - The gate failed, and R08 (unmatched-query labels) was never run. (same source)
- **Attribution findings carried into later work.**
  - R02 (modern SyntheticMRI): 82–83% of A-xyz false edges touch spatially unmatched nodes, and about 90.7% of missed edges lack a recovered endpoint pair.
  - R01 (legacy IXI): 91.1% of false edges touched unmatched nodes. (same source; details in [`history_relationformer_3d.md`](history_relationformer_3d.md))

### 5. L08 node-first curriculum from scratch (28 Oct) [DONE — inconclusive]

**Setup.** Scratch SyntheticMRI B0 with 120 queries; 15,640 updates (20 epochs); one seed. Edge CE was switched off for the first 10% of updates, then ramped back in.

**Result:**
- Edge F1 at 2 mm: 0.12783 vs 0.10357 for the joint control, a difference of +0.02426 [+0.01047, +0.03784].
- Branch and Curve F1 intervals included zero.
- The AP gain on identical GT pairs (+0.01266) had an interval including zero.

**Decision.** "Inconclusive, do not auto-expand". The 100-epoch, two-seed confirmation was never launched.

([source](../archive/legacy-20261103/docs/research/fine_feature_architecture_study.md), [plan](../../experiments/node_first_curriculum_20261028/plan.json))

### 6. Q0 IXI capacity and status audit (28 Oct) [DONE]

**Capacity.** Post-contraction IXI targets reach a maximum of 153 nodes (train) and 141 (validation), with 5 / 5 patches above 120 nodes. No degree-2 object nodes remain. The older "239 nodes" figure came from a pre-contraction population. So 120 queries are unsafe for IXI: keep 256, and the 120-vs-256 study (Q120) was deferred. BRAVA targets peak at 64 / 69 nodes.

**Status.** Q0 also reported that no modern IXI result existed yet. That was overtaken the same day by N07 (§3.3).

([source](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md), [artifacts](../../experiments/q0_ixi_audit_20261028/))

### 7. Planning documents and decisions [SUPERSEDED by the 30 Oct switch]

- **Journal goal (roadmap, 23 Oct).** Compact continuous graph reconstruction across BRAVA, IXI and SyntheticMRI. The candidate contributions were adaptive edge geometry and a controlled study of graph complexity against geometry complexity. ([source](../archive/legacy-20261103/research/roadmap.md))
- **Method-selection gate (roadmap).** An approach would be preferred only if all of the following held:
  - learned shape improves against the dense paths, including difficult branches;
  - the gain survives K8, larger-K and same-graph chord controls;
  - geometric coverage is not bought with false edges or cycles;
  - controls, queries, parameters and runtime are reported;
  - uncertainty is clear.

  No geometry variant passed it. As of 23 Oct, piecewise geometry was only a provisional candidate and K8 remained the control.
- **Decisions log** ([source](../archive/legacy-20261103/research/decisions.md)):
  - **Accepted:**
    - original dense paths as the geometric reference;
    - subject-disjoint splits with locked tests;
    - physical multi-metric evaluation;
    - no extra joint losses by default;
    - BRAVA SWC alignment corrections kept explicit.
  - **Measured:** K8 is insufficient for some BRAVA branches. Within a 0.5 mm bound K8 covers 77.19% and K24 98.34%.
  - **Not adopted:**
    - naive NMS;
    - hard-negative and image-aware relation continuations;
    - node focal on IXI, based on synthetic evidence alone;
    - V2/V3 verifiers.
  - **BRAVA:** use SyntheticMRI-initialized K8 as the reference; keep the 0.5/0.5 decoder after the D08 audit.
- **Eight-week journal closure plan (registry §10–11, 28 Oct) [SUPERSEDED].**
  - Executed:
    - J00/J02 (Q0);
    - I01 (N07);
    - 2D data QC and the AV-WIDE B0 pilot (see [`history_native2d.md`](history_native2d.md));
    - J01, done as the D08 audit (§1.8).
  - Never run: M01, N09, N10 and T01. Q120 was deferred.
  - The plan's "stop-adding-ideas" rule (pick at most one of N09/N10 by week 5) became moot when the architecture was replaced on 30 Oct. N10's idea, a training-time node heatmap from early CNN features, is the closest antecedent of the 30 Oct node-heatmap probe. ([source](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md))

## How this work led to the 30 Oct diagnosis

The "Read this first" section and §0 of [`network_proposals_20261030.md`](network_proposals_20261030.md) build directly on the following results.

**Evidence that RelationFormer underfits real MRA even when trained long enough.**
- §0 and the glossary use D08 as the "trained long enough" evidence for BRAVA: 82k updates, about 2.6M exposures, train 0.394 / 0.090, validation 0.356 / 0.068.
- D06 (representative training fit is as poor as validation), D07 (removing augmentation only buys memorization) and the tiny-set learnability tests (the implementation can memorize) had already ruled out held-out overfitting, BN and implementation bugs.
- The D08 operating-point audit ruled out thresholds.

**Evidence that the errors are node-driven.**
- §0 states that about 90% of missed edges lack an endpoint pair and that 83–91% of false edges touch an unmatched node.
- This matches D08 (96.69% / 89.40%), N07 (93.02%), R02 (90.7%, 82–83%) and R01 (91.1%).

**Fixes within RelationFormer that had failed.**
- The node verifier, N08, R06 and R07 did not help. Neither did adaptive or larger-K geometry on frozen detectors (§2), or K24/C1/split_K8 on BRAVA (§1.4).
- These failures left the architecture itself as the remaining suspect. §0 then names the bottleneck: the stride-32 encoder gives a 2×2×2 feature grid.

**A hint that resolution matters.**
- A-xyz on SyntheticMRI (§3.2) and A-cls on IXI (§3.3) improved results by reading higher-resolution early CNN features.
- The later relation-adapter result (endpoint, chord and corridor features helped early) prefigured the dense model's corridor edge scorer. That result is in [`history_relationformer_3d.md`](history_relationformer_3d.md).

**Reused definitions and cohorts.**
- B0's recipe and the D08 initialization come from the local-refinement study's plan.
- The SyntheticMRI B0 / A-xyz node-F1 references (0.793 / 0.817) come from §3.2.
- The 30 Oct comparison cohorts reuse `experiments/multidataset_continuous/plan.json` ([link](../../experiments/multidataset_continuous/plan.json)).

**Later reversals.**
- The 30 Oct node-heatmap probe reached validation node F1 0.539 on BRAVA, against 0.356 for D08, from scratch in about 7 minutes. So the poor BRAVA nodes were a detector limitation, not unlearnable data.
- The dense model trains from scratch, so D08's "scratch nearly fails" result is specific to RelationFormer.
- The literature-scale IXI control lifted B0 to 0.508 / 0.156 node/edge F1 but did not close the gap ([`network_proposals_20261030.md`](network_proposals_20261030.md) §6.3).

## What carried forward / what was dropped

**Carried forward:**
- the BRAVA split, preparation and annotation-relative caveats;
- subject-disjoint cohorts and train-only (`monitor_train`) threshold selection;
- physical 1/2 mm matching plus common-v2 Branch/Curve F1;
- D08 as the BRAVA RelationFormer reference;
- the "node recall is the main limit" diagnosis, which is still the main open problem for the dense model;
- the finding that long BRAVA branches need more than 8 controls, which the dense model now handles with per-edge K8/K16/K24;
- 256-query capacity for IXI in RelationFormer comparisons.

**Dropped:**
- K24/C1/adaptive-K/piecewise heads and split_K8 on RelationFormer;
- node verifiers, GT-free query selection, R06/R07/R08 relation tweaks and the C multiscale attention;
- the L08 confirmation run and the BRAVA Stage 2 matrix;
- M01/N09/N10/T01 and the eight-week schedule;
- the method-selection gate as formulated around geometry variants.

All were dropped because the 30 Oct diagnosis moved the work to a different architecture, not because each was individually disproven.

## Sources

Archived originals (post-move paths):
- [docs/research/EXPERIMENT_REVIEW_20261021.md](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261021.md). Review of 20 Oct jobs: IXI series 8–11 (covered in `history_relationformer_3d.md`), BRAVA preparation, the failed botanical smoke, and the appended BRAVA Stage 1 recovery.
- [docs/research/EXPERIMENT_REVIEW_20261022.md](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261022.md). K8 capacity audit summary, recovered BRAVA Stage 1, and completed limited-data botanical transfer.
- [docs/research/EXPERIMENT_REVIEW_20261023.md](../archive/legacy-20261103/docs/research/EXPERIMENT_REVIEW_20261023.md). BRAVA recovery arms D1–D3, learned geometry heads on IXI/SyntheticMRI, and the unmatched-query analysis.
- [docs/research/BRAVA_experiment_plan.md](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md). BRAVA plan and log (21–28 Oct): Stage 1, Stage 2 plan (never run), G05, D06, D07, D08 and the D08 operating-point audit.
- [docs/research/multidataset_continuous_graph_experiments.md](../archive/legacy-20261103/docs/research/multidataset_continuous_graph_experiments.md). Protocol and completed interpretation of the 15-arm multi-dataset batch.
- [docs/research/node_verifier_feasibility.md](../archive/legacy-20261103/docs/research/node_verifier_feasibility.md). V0–V3 verifier study and BRAVA eight-patch learnability.
- [docs/research/fine_feature_architecture_study.md](../archive/legacy-20261103/docs/research/fine_feature_architecture_study.md). S0 baseline, A/A-control/C pilots, and the L08 curriculum protocol and pilot.
- [docs/research/local_refinement_long_study.md](../archive/legacy-20261103/docs/research/local_refinement_long_study.md). Two-seed 12k B0/A/A-xyz/A-control study.
- [research/roadmap.md](../archive/legacy-20261103/research/roadmap.md). Journal goal, milestones, method-selection gate and dated priorities (23–28 Oct).
- [research/decisions.md](../archive/legacy-20261103/research/decisions.md). Accepted, provisional and rejected decisions (23–28 Oct).
- [research/experiments.md](../archive/legacy-20261103/research/experiments.md). Experiment register with the "Complete source catalog" (skimmed); also the 28 Oct afternoon N07 table.
- [research/architecture.md](../archive/legacy-20261103/research/architecture.md). Repository/model map of the RelationFormer pipeline as of 23–28 Oct.
- [research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md). Stable-ID registry (N/R/L/G/D series), the eight-week plan, Q0, N07, R06/R07, N08 and L08 outcomes.
- [research/evidence/experiment-registry/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.received.md](../archive/legacy-20261103/research/evidence/experiment-registry/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.received.md). The byte-identical registry as received on 26 Oct (290 lines). It differs from the working copy in three ways:
  - N08 is "NOT RUN — NEXT";
  - R06/R07 are pending;
  - it has none of the D07/D08/Q0/N07/L08 results, the eight-week plan or §11.
- [research/evidence/experiment-registry/import-20261026.json](../archive/legacy-20261103/research/evidence/experiment-registry/import-20261026.json). Import record: hashes of the received and working copies, preserved IDs, and the N08 reconciliation.
- [research/evidence/image-to-graph-vessel/archive-index.csv](../archive/legacy-20261103/research/evidence/image-to-graph-vessel/archive-index.csv) and [historical-evaluation-metrics.csv](../archive/legacy-20261103/research/evidence/image-to-graph-vessel/historical-evaluation-metrics.csv). Image-to-Graph-Vessel archive index and legacy metrics (listed only; see `history_native2d.md`).

Evidence that stays in place:
- [evidence/brava_stage1_20261021/results.md](evidence/brava_stage1_20261021/results.md), [completion-review.md](evidence/brava_stage1_20261021/completion-review.md). BRAVA Stage 1 tables, bootstrap and learning curves.
- [evidence/multidataset_continuous/report.md](evidence/multidataset_continuous/report.md). All 15 arms of the 22–23 Oct batch, with CSVs and 3D viewers.
- [evidence/node_verifier_feasibility/](evidence/node_verifier_feasibility/feasibility.png), [fine_feature_architecture/](evidence/fine_feature_architecture/baseline.png), [local_refinement_long/](evidence/local_refinement_long/trajectories.png). Figures for §3–4.
- [evidence/ixi_local_components_20261028/](evidence/ixi_local_components_20261028/sample_000001_0009-comparison.png). N07 A-cls vs A-xyz overlays on one IXI validation patch.
- [evidence/brava_supervision_20261027/](evidence/brava_supervision_20261027/brava_clean_augmented_overlay.png). Clean vs augmented BRAVA training target overlay (alignment check for D07).
- Experiment folders: [brava_long_20261027](../../experiments/brava_long_20261027/report.md), [d08_operating_points_20261028](../../experiments/d08_operating_points_20261028/), [brava_augmentation_20261026](../../experiments/brava_augmentation_20261026/analysis.json), [brava_representation_20261026](../../experiments/brava_representation_20261026/analysis.json), [local_refinement_long](../../experiments/local_refinement_long/analysis.json), [fine_feature_architecture](../../experiments/fine_feature_architecture/analysis.json), [node_first_curriculum_20261028](../../experiments/node_first_curriculum_20261028/plan.json), [q0_ixi_audit_20261028](../../experiments/q0_ixi_audit_20261028/).
