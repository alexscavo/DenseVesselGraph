# Research history, Sept – 29 Oct 2026: from RelationFormer to the dense traced model

This document condenses the research logs written before the switch to the current model. For
each line of work it records what was tried, the key result, what was concluded and what
superseded it. The current model and its results are in
[`dense_traced_model_20261102.md`](dense_traced_model_20261102.md). The full design, glossary and
the 30 Oct – 1 Nov comparisons are in [`network_proposals_20261030.md`](network_proposals_20261030.md).
The original documents are archived; [`../archive/README.md`](../archive/README.md) explains how to
restore them.

Labels: **[DONE]**, **[DONE — negative]**, **[SUPERSEDED by …]**, **[PROPOSED, never run]**,
**[RUNNING when last documented]** (no final result was ever written).

## Contents

0. [Summary](#0-summary)
1. [Shorthand and reading caveats](#1-shorthand-and-reading-caveats)
2. [Ground-truth graphs, datasets and representation](#2-ground-truth-graphs-datasets-and-representation)
3. [Metric design](#3-metric-design)
4. [RelationFormer on SyntheticMRI and IXI (Sept – 20 Oct)](#4-relationformer-on-syntheticmri-and-ixi-sept--20-oct)
5. [Learning curved edges (5 – 13 Oct)](#5-learning-curved-edges-5--13-oct)
6. [Multi-dataset and BRAVA work (21 – 29 Oct)](#6-multi-dataset-and-brava-work-21--29-oct)
7. [Pretraining and transfer](#7-pretraining-and-transfer)
8. [Native 2D](#8-native-2d)
9. [What carried forward into the dense model](#9-what-carried-forward-into-the-dense-model)

## 0. Summary

- **Ground truth.** The IXI and TopBrain graphs come from full-volume extraction with the
  original Vedo centreline plus mask repair. They are simplified with an exact minimum-node
  solver (deviation ≤ 0.75 × local radius, ≥ 95% chord containment) and then cropped into 64³
  patches with exact boundary nodes. These datasets are still in use.
- **Representation.** Each vessel is stored as "topology + curved edge": junctions, ends and
  border crossings are nodes, and every edge carries a B-spline. Only long or tortuous branches
  (mostly in BRAVA) need more than 8 controls.
- **RelationFormer** (DETR-style: object queries, Hungarian matching, stride-32 encoder) never
  produced good graphs on real MRA, even with long training (BRAVA D08: validation edge F1
  0.068). The diagnostics traced the failure to three causes:
  - nodes are regressed from a 2×2×2 feature grid;
  - Hungarian matching at training does not match decoding at inference;
  - edges are scored without looking at the image between the nodes.

  About 90% of missed edges lack a recovered endpoint.
- **Curved edges** improve curve quality but not the graph. Training with or without the spline
  head gives the same nodes and edges.
- **Pretraining** on plants, leaves or SyntheticMRI never beat a matched target-only control at
  full data. The only positive result is single-seed and uses 200 patches.
- **These findings led to the 30 Oct switch** to a two-stage dense model: a full-resolution
  U-Net node heatmap, then edges scored from image features along the candidate path. It trains
  from scratch and beats every RelationFormer variant.

## 1. Shorthand and reading caveats

**Models and recipes**
- **B0**: baseline RelationFormer recipe:
  - SE-ResNet (stride 32) plus deformable transformer;
  - K8 spline head, no node refinement;
  - node softmax focal γ=2;
  - matched-pair edge CE with ratio upsampling 0.15;
  - the `graph_only` route.
- **S0**: clean SyntheticMRI B0, 100 epochs from scratch (epoch 95, 120 queries). It is the parent
  of most late-October runs.
- **A / A-xyz / A-cls / A-control**: local refinement heads that sample early high-resolution CNN
  features around each predicted node:
  - A corrects coordinates and confidence;
  - A-xyz corrects coordinates only;
  - A-cls corrects confidence only;
  - A-control has zeroed image input.
- **K8 / K16 / K24**: clamped cubic B-spline edge with 8 / 16 / 24 controls. The two end controls
  are the nodes. Interior controls start on the chord.
- **`graph_only`**: the spline loss trains only the geometry head and never reaches the detector.
  Alternatives: `tokens_only` (gradients reach the shared tokens) and `joint`.
- **Residual target**: supervise only the curve's deviation from the chord, not absolute control
  positions.
- **Local corrector**: samples the first CNN activation (64 channels, 32³) along the curve and
  adds a mm correction to the interior controls. Part of B0.

**Codes**
- **D01–D08 (tracker)**: September representation and data decisions (§2).
- **D06 / D07 / D08 (registry)**: BRAVA learnability diagnosis, augmentation on/off ablation, and
  the 300-epoch full-cohort run (§6).
- **D1–D3**: BRAVA recovery arms of 23 Oct.
- **N06 / N08**: GT-assisted and GT-free query-selection studies.
- **N07**: modern IXI B0 / A-xyz / A-cls transfer.
- **R06 / R07**: relation-token fusion and reliable-negative supervision.
- **L08**: node-first curriculum.
- **Q0**: the IXI capacity audit.
- **T05–T10**: September tracker tasks:
  - T05/T06 representation study;
  - T07/T08 metric checks;
  - T09 the 300-epoch IXI run;
  - T10 node types and the query benchmark.
- **`monitor_train`**: fixed training patches used to choose thresholds. Validation never selects
  anything.

**Metrics** (definitions in [`../EVALUATION.md`](../EVALUATION.md))
- **Node F1 @2 mm**: one-to-one matching within 2 mm.
- **Edge F1 (positional)**: both endpoints matched and adjacent in GT.
- **MM / MU / UU**: false edges by how many endpoints are matched.
- **common-v2 Branch F1**: branches after degree-2 contraction, anchors at 1 mm.
- **common-v2 Curve F1**: centreline samples within 1 mm.
- **β₀ / β₁**: components / cycles.
- **Conditional relation recall / AP**: relation quality on pairs whose two endpoints were
  recovered.

**Caveats that change how results read**
- **Native vs positional edge F1.** Edge F1 values before 15 Oct, and every "native" value, are
  RelationFormer's box-based metric. It is not endpoint-correct adjacency and is much higher: the
  same 28 Oct IXI model scores native 0.288 against positional 0.10. Do not compare the two.
- **Test-split use.** Several September – mid-October studies evaluated selected checkpoints once on
  the held-out **test** split: T09, the focal ablation, the 10k spline fine-tune, the 300-epoch
  scratch comparison and all pretraining source comparisons. Those test splits are no longer
  untouched; a paper claim needs a new holdout.
- **Reference quality.** IXI/TopBrain references are extracted from segmentations, not anatomical
  truth. BRAVA SWC traces are incomplete, so BRAVA errors are relative to the annotation.
  SyntheticMRI guides are generator centrelines with mild curvature (median tortuosity 1.0159 on
  branches ≥ 5 mm).
- **BN-adapted August evaluations.** 30 of 32 archived August evaluation logs calibrated
  batch-norm on the *validation* loader (`evaluate.py --bn-calibration-batches` chooses it). So
  those historical numbers are BN-adapted, not frozen references. Later plans use
  `bn_calibration_batches=0`.
- **Seeds and validation.** Almost every result is single-seed, and validation subjects were
  reused across many studies. Subject-paired bootstrap intervals do not cover seed variance.

## 2. Ground-truth graphs, datasets and representation

**Data.** 220 paired image/segmentation volumes:
- 170 IXI MRA;
- 25 TopBrain MR;
- 18 TopBrain CT training and 7 labelled TopBrain CT test volumes.

The patient split is 154 / 33 / 33. Dataset-specific details are in
[`../IXI_DATASET.md`](../IXI_DATASET.md), [`BRAVA_preparation.md`](BRAVA_preparation.md) and the
SyntheticMRI reference docs.

**Extraction pipeline [DONE, still in use].**
- **Full volume first, crop afterwards.** Each segmentation is skeletonised once into an immutable
  dense centreline. Three products are derived from it: dense, adaptive and junction-only. Only
  then is the saved graph cropped.
- **No patch-level re-simplification,** so overlapping patches share the same graph.
- **Exact boundary crossings.** Crops clip at voxel-cell faces and insert a flagged boundary node
  at each crossing.
- **Representation mechanics:**
  - a pure degree-2 ring becomes 3 nodes and 3 arcs (D01);
  - "dense" means before smoothing and simplification (D04);
  - extraction rules use physical or locally normalised units (D03).

**Centreline backend.**
- **Re-implemented Vedo-style backend (18 Sept) [ABANDONED].** It made the IXI002 crop worse:
  244 nodes / 276 edges / β₁ = 37 became 363 / 411 / β₁ = 49.
- **Original Vedo code vs the legacy extractor (CT-001, 22 Sept) [DONE].**
  - Original Vedo had fewer cycles (β₁ 45 vs 73), but 78 dense nodes outside the mask (legacy:
    0) and took 190.3 s vs 5.6 s.
  - A repair wrapper moved 78 points into the mask (median 0.41 mm), rerouted 52 edges and dropped
    5.
  - The report states that the fewer cycles could mean false loops removed, true loops removed,
    or both. One case, no centreline ground truth.
- **Production (22–23 Sept)** used `vedo_original` plus repair for all 220 volumes. The 18 Sept
  "rejected" entry refers to the re-implementation, not this code. No written rationale exists
  beyond CT-001.

**Simplification policy [DONE].**
- **Junction-only is not a geometric target.** On the densest IXI crop, junction-only straight
  edges reach a minimum chord containment of 0.106.
- **Greedy RDP [SUPERSEDED by the exact solver].** Fixed 2-voxel RDP still left chords outside
  the lumen. Radius-1.5× greedy RDP was chosen first.
- **Exact solver (D05).** Then replaced by `optimal_radius_0p75x_c095`: an exact constrained
  shortest path (fewest samples with deviation ≤ 0.75 × local radius and ≥ 95% containment). On
  14 real volumes against greedy 1.5×:
  - full-volume degree-2 nodes 9,682 → 8,962;
  - patches > 120 nodes 57 → 45;
  - worst error / local radius 1.496 → 0.750;
  - 0 edges below 95% containment.
- **Caveats.** TopBrain MR did not improve. The rule is a promising dataset-agnostic policy, not a
  proven one. Simplification fixes geometry but never changed β₁.

**Junction-only vs adaptive vs dense (T05/T06, 30 volumes) [DONE].**
- Adaptive beats junction-only on branch-paired 0.5-mm Curve F1 in 30/30 volumes (stratum means
  0.757–0.893 vs 0.227–0.306).
- It uses fewer nodes than dense in 30/30 (550–1,352 vs 3,943–6,760 per volume).
- Branch F1 is invariant to degree-2 subdivision (T07: adaptive vs dense = 1.0).
- Comparisons are against the extracted dense centreline, not manual GT.

**BRAVA full-volume GT representations (21 Oct) [DONE].** Validation means per volume, against
the original dense development centrelines:

| Representation | Nodes | Curve F1 | Guide distance |
|---|---:|---:|---:|
| dense | 3143.375 | 1.0000 | — |
| adaptive | 2471.375 | 0.9991 | 0.0492 mm |
| junction-only (straight) | 203.0 | 0.1866 | 3.1478 mm |
| junction-only + K8 spline | 203.0 | 0.8413 | 0.2903 mm |

Branch F1 is 1.0 in every row. The spline rows are oracle fits, not predictions. The scores are
relative to the manual annotations.

**How many controls a curved edge needs (22 Oct, GT only, train + validation, no test)
[DONE].**
- **Cached K8 targets.**
  - IXI validation: P95 maximum error 0.149 mm; 0.18% of branches definitely > 1 mm.
  - BRAVA validation: P95 maximum error 1.011 mm; 5.01% of branches > 1 mm; 35.86% of patches
    have at least one such branch.
- **Long branches.** Errors concentrate on long branches: 50–100 mm BRAVA validation branches
  exceed 1 mm in 88.37% of cases, IXI in 58.76%.
- **Fixed K, BRAVA validation, branches within 0.5 mm:** K8 77.19%, K16 94.02%, K24 98.34%.
  Piecewise K8 inside an edge reaches 100% at 10.61 controls per branch. Explicit degree-2
  splitting gives the same geometry but +20.6% nodes.
- **IXI and SyntheticMRI** meet 0.5 mm with K8 on 99.14% / 99.32% of branches. The
  representation bottleneck is BRAVA's long branches.
- **Caveat.** Counts are overlapping patch-edge occurrences, not independent vessels.

**Object-query capacity [DONE].**
- **Cause of overflow.** Overflow comes from degree-2 geometry nodes: topological plus boundary
  nodes never exceed 81 on the 14-volume subset.
- **192 queries rejected.** 192 queries were recommended from 7 volumes (max 177 nodes; +2% step
  time). The full legacy grid then had 35 of 50,613 patches above 192 (max 239), so 192 was
  rejected.
- **256 adopted.** 256 costs +7.51% step time against 192.
- **After contraction (Q0, 28 Oct).** Contracted K8 targets peak at 153 (train) / 141 (val), so
  120 queries are unsafe for IXI. RelationFormer IXI comparisons keep 256.
- **Overflow handling.** Training fails loudly if a target exceeds the query count.

**Patch eligibility [DONE].**
- **Grid.** Deterministic 64³ grid, stride ≤ 40.
- **Graph-positive filter.** The Vedo production restricts every split to graph-positive patches
  (foreground, ≥ 1 node, ≥ 1 edge):

  | Dataset | Active train / val / test |
  |---|---|
  | IXI | 13,742 / 2,961 / 2,384 |
  | TopBrain | 7,391 / 1,650 / 2,685 |

  An earlier rule (21 Sept) filtered training only.
- **Caveat.** Patch metrics are conditional on a vessel being present.

**False loops [PROPOSED, never run].**
- Dense cycle tangles in IXI come from the segmentation (fused vessels, digital contacts), not
  from simplification.
- A raw-image bridge-scoring repair with a labelled validation sample was proposed, and training
  data generation was meant to wait for it. Production went ahead without it.
- False segmentation loops therefore remain unmeasured label noise in IXI/TopBrain β₁.

**Storage [DONE].** Retention rules adopted on 16 Oct are still the policy:
- keep config, data IDs, metrics and interpretation for every comparison, negative ones included;
- keep best and final checkpoints;
- large arrays go to SCRATCH, which is purged and is not a backup; compact reports go in Git.

Sources: archived under `docs/archive` (see its README). The originals are
`IMPLEMENTATION_TRACKER.md`, `VEDO_PRODUCTION_RUN.md`, `vascular_graph_representation.md`,
`vascular_graph_roadmap.md`, `loop_failure_mitigation.md`, `continuous_representation_comparison.md`,
`storage/` and the evidence reports `T01`–`T05`, `adaptive_tuning`, `vedo_original_ct001`,
`ixi_capacity_192_256`, `spline_target_capacity_20261022` and `brava_stage1_20261021`.

## 3. Metric design

The inherited RelationFormer/GrEVe evaluator scores nodes and edges as boxes (COCO-style mAP),
plus the street mover distance (SMD). Neither checks whether an edge joins the right two
vessel events. The September metric plan and its implementation set the rules still used today.

**The plan (Sept) [DONE as a plan].**
- **Three evaluation classes:**
  - representation quality: target vs dense;
  - model quality: prediction vs its own target;
  - end-to-end fidelity: prediction vs dense, always shown next to the representation oracle.
- **Topology and geometry are scored separately.**
- **Degree-2 contraction** before topology scoring, so subdivision does not change the score.
- **Branch Connectivity F1** as the core topology metric, which became common-v2 Branch F1.
- **Betti numbers** as diagnostics only.
- **Tolerances** in physical mm, never adapted to density.
- **SMD demoted to legacy.** It ignores branch identity, and redundant edges can lower it.

**Controlled metric checks (T07, T08) [DONE].**
- On two real IXI patches, removing one GT edge drops Branch F1 to 0.857 / 0.783.
- A false cross-link between two close vessels takes Branch F1 1.000 → 0.000, while SMD barely
  moves (0 → 0.000031) and node mAP is unchanged.
- A bent path with correct anchors leaves Branch F1 at 1.000 but drops edge mAP to 0.100.
- β₁ error is 0 in both cases.
- Conclusion: Branch F1 and curve metrics are complementary.
- T08 also warned that the 40-detection AP cap is unsuitable for dense vessel patches. The current
  box mAP in `network_proposals` §6.5 still uses it.

**vessel-core v1 (13 Oct) [DONE; extras unused since].**
- **What it is.** A superset of common-v2: its Curve F1 @1 mm and Branch F1 reproduce common-v2
  exactly, and it adds ACD, HD95, APLS-3D, TOPO-3D and payload size. It is not a renaming of
  common-v2.
- **Straight vs detached B-spline scratch models** (1,952 SyntheticMRI validation patches):
  - Curve F1 @1 mm 0.2020 vs 0.3426;
  - Branch F1 0.0079 vs 0.0118.
- **Curvature, not adjacency.** The spline model's own chords score 0.2263, so its gain is overlap
  from curvature.
- **Payload.** The spline payload is about 4.07× larger.
- **Never implemented or never used:**
  - Graph GOSPA;
  - whole-volume stitching;
  - clDice;
  - branch-angle error;
  - curve-aware SMD.

**Branch-length diagnostic (13 Oct) [DONE].**
- **Long relations are missed.** With both endpoints found at 2 mm, conditional direct-edge recall
  is 92.9–93.8% in the shortest GT arc-length quartile and 38.8–42.8% in the longest.
- **Nearby non-adjacent pairs are joined.** 69.0–71.9% of matched non-adjacent pairs closer than
  4.90 mm are connected.
- **Localization dominates.** Only 13.7% / 16.9% of GT branches had both endpoints found.
- **Later explanation.** The relation MLP never looks at the image along the path, and the dense
  model's traced candidates address this.
- **A 5× near-negative weighting pilot [DONE — negative]** lowered the near false-positive rate
  (73.7% → 66.1%) but beat the starting checkpoint on nothing.

**Generalization audit (14 Oct) [DONE].**
- **Not late collapse.** The validation node-recall gap (about 45% @2 mm for these October scratch
  models) was already present at the best checkpoints.
- **No split shift or leakage.** No coarse split shift and no leakage were found.
- **Missing augmentation.** The spline loader had augmentation disabled, which was flagged as a
  missing regularizer.
- **Not comparable with current numbers.** These recalls do not compare with the 30 Oct B0 numbers
  (0.794 node F1), which use another recipe and cohort.

Sources: archived under `docs/archive` (see its README). The originals are
`vascular_graph_metrics.md`, `metrics/` and the evidence reports `T07` and `T08`.

## 4. RelationFormer on SyntheticMRI and IXI (Sept – 20 Oct)

**SUPERSEDED by the dense traced model.** The question was why RelationFormer, which fits
SyntheticMRI reasonably well, gives poor graphs on IXI.

### 4.1 Training the detector on IXI

All edge F1 values in this subsection are native (box-based).

- **SyntheticMRI loss-ablation matrix [PROPOSED, never run].** It was designed as baseline /
  focal / Betti / focal + Betti, with configs in `configs/experiments/finetune_mri/`. No results
  exist.
- **T09: IXI 300-epoch adaptive-GT run [DONE; test split used].**
  - Setup: 8,000 IXI patches, random initialization, 256 queries; epoch 270 selected on validation
    Branch F1.
  - Test, 2,384 patches: node F1 0.4086, native edge F1 0.3067, Branch F1 0.00535.
  - Graphs were massively over-connected: 179.76 predicted edges per patch against 50.39 in GT, and
    109.27 cycles against 0.77.
  - More epochs did not fix extra edges.
- **T10: node-type head and soft degree decoding [DONE, validation].**
  - A four-class type head alone changed nothing (edge F1 0.2673 → 0.2684), and type accuracy was
    59.63%.
  - A soft degree penalty of 0.5 at decoding gave edge F1 0.3611 and Branch F1 0.00658, against
    0.3421 / 0.00536 for plain thresholding.
  - Stronger penalties raised edge F1 but destroyed branch continuity.
  - An earlier hard-negative comparison also lowered the matched-pair limit from 9,999 to about
    1,000, so it cannot isolate hard-negative mining.
- **Focal-loss 300-epoch ablation [DONE; exploratory test-split evaluation].**
  - Focal losses raised node F1 (0.5103 → 0.5417 / 0.5474) but added edges and cycles.
  - The CE model with degree penalty 0.5 reached native edge F1 0.4847.
  - Branch F1 stayed around 0.01 everywhere.
- **Degree-consistency training [RUNNING when last documented].**
  - Interim epoch-220 numbers: fewer edges and cycles, native edge F1 0.446 vs 0.369, but much more
    fragmented (12.3 components vs 2.5).
  - No final result was written, and three factors changed at once.
- **Conclusion.** Edge count and β₁ alone are unsafe selection criteria.

### 4.2 The IXI diagnostic series 1–11 (18–20 Oct): what fails

**Setup.**
- The frozen IXI-only "delayed-spline" parent checkpoint and its derivatives.
- Mostly 256 train / 128 validation patches, 26 validation subjects, positional edge F1, one seed.
- The parent's actual edge loss was focal without upsampling. Earlier documents called it
  regularized CE.

**Labels and edge budget [DONE].**
- **Labels.** GT is consistent with the mask (99.52% of curve samples inside). This does not show
  anatomical correctness.
- **Missed edges.** Of 51,937 validation GT edges, 69.38% miss at least one endpoint, 26.45% have
  both endpoints but no relation, and 4.17% are correct.

**Node localisation [DONE].**
- **Foreground probes** (validation AP at about 0.86% prevalence):
  - early 32³ CNN features: 0.6551;
  - transformer memory at 2³: 0.0246;
  - even the GT mask pooled to 2³: only 0.0665.

  The 2×2×2 grid itself cannot represent fine vessels.
- **Local-image refinement (1,000 updates).**
  - Native node F1 0.4857 → 0.5196; junction recall 49.37% → 58.22%; relation recall unchanged.
  - At inference the coordinate residual alone carries the gain: positional edge F1 0.0581 →
    0.0730, paired interval [+0.0087, +0.0207]. This was the first evidence for A-xyz.
- **Node class weighting and focal [DONE — negative].**
  - Weighted CE: edge F1 0.1189.
  - Equal-weight CE: edge F1 0.1165, with fewer cycles.
  - Equal-weight focal: edge F1 0.0767.

  IXI did not reproduce a colleague's SyntheticMRI focal benefit. The edge loss changed at the
  same moment, which confounds the comparison.

**Duplicates [DONE].**
- **Composition.** Of 3,827 retained nodes, 584 are duplicates, 672 are foreground extras and 592
  are off-vessel.
- **Confidence does not separate duplicates.** Matched-vs-duplicate AUC is 0.5495, and 70.4% of
  duplicates have confidence ≥ 0.9.
- **Fixed NMS lowers positional edge F1** (0.0737 → 0.0626 at 0.5 mm) because it removes real
  nearby nodes.

**Train/inference mismatch [DONE].**
- **False edges.** 91.1% of false edges touch an unmatched node. Relation training covers only
  Hungarian-matched pairs, while inference scores every retained pair.
- **Oracles do not rescue edges.** Deleting all unmatched nodes (a GT oracle) leaves edge recall at
  4.77% (edge F1 0.0892). Using all 256 raw queries recovers 83.62% of nodes but only 162 vs 142
  correct edges.
- **Hungarian selection at inference is not the fix.** Its conditional AP is 0.4658 against 0.4702
  normal, with the interval including zero.

**Relations ignore the image [DONE — negative for every fix].**
- **The existing head.** It ranks recovered-endpoint pairs with AP 0.4702 but recall 0.1257 at
  0.5. Recall falls to 3.80% in the longest arc quartile.
- **Lower thresholds.** The train-max-F1 threshold raises edge F1 0.0737 → 0.1608, but β₁ MAE
  goes 1.04 → 19.52. Threshold sweeps were banned as a fix.
- **Relation-only continuations.**
  - The arms were focal, regularized CE, CE + unmatched hard negatives and CE + matched hard
    negatives.
  - Each fit training better and ranked validation pairs worse: conditional AP 0.3659–0.4153
    against 0.4702.
  - Their edge F1 gains came with false cycles.
- **Image-aware residual adapters** (endpoint, chord and corridor samples, 65,600 updates).
  - All overfit by the final update: validation AP 0.31–0.32 against 0.4723.
  - Re-analysis of saved 2k checkpoints showed a real image signal: chord and corridor reached AP
    0.50156 / 0.50191, against 0.47660 for the token control.
  - Those checkpoints were chosen after seeing the curves, so no stopping rule was validated.
  - At the final update, chord and corridor kept positive all-retained AP intervals over the
    control, while their conditional-AP intervals included zero.

**Segmentation-input oracle [DONE].**
- Training on the binary mask instead of raw MRA raised positional edge F1 0.0415 → 0.0904 and
  Branch F1 0.0061 → 0.0243.
- Even with the mask that generated the GT, edge F1 stays at 0.09, so much of the failure is in
  graph decoding.
- The mask generated the references, so this is a reconstruction oracle.

**Mechanisms identified.**
1. Nodes are regressed from 8 blended vectors (2³ memory).
2. There is no usable uniqueness signal: duplicates are as confident as true nodes.
3. Train/inference mismatch: 91% of IXI false edges touch unmatched nodes.
4. Edge recall is bounded by node recall.
5. The relation MLP never looks between the nodes.
6. The training data are underfitted (training node recovery 63%), not just poorly generalized.

### 4.3 Follow-ups appended 26–28 Oct [DONE]

- **SyntheticMRI A-xyz error attribution (R02).**
  - 82.4–83.2% of false edges touch a spatially unmatched node.
  - 90.7–90.8% of missed edges lack a recovered endpoint.
  - Only 9.2–9.3% are pure relation misses. On the old IXI model, 34.85% of missed edges were
    relation misses.
- **N06 representative oracle.** Choosing the confidence-preferred nearby query instead of the
  nearest raises edge F1 from 0.720/0.729 to 0.834/0.838. Spatially equivalent queries carry
  different relation content. The oracle is not deployable.
- **Negative fixes:**
  - **N08** GT-free selectors: intervals include zero, search stopped.
  - **R06** CNN-conditioned relation tokens: edge F1 −0.003678 / −0.002978, intervals below zero.
  - **R07** reliable nearby negatives: only 18 of 28,494 training negatives qualify; the gain stays
    below the original checkpoint.
- **IXI six-arm local-head study (N07).** See §6.3.

Sources: archived under `docs/archive` (see its README). The originals are `metrics/1-…` to
`metrics/11-…`, `PROJECT_REVIEW_20261019.md`, `IXI_graph_extraction_diagnostic_roadmap.md`,
`docs/EXPERIMENTS.md` and the evidence reports `T09`, `T09_long`, `T10`, `focal_ixi_300ep`,
`degree_consistency`, `ixi_*` and `job_review_20261020`.

## 5. Learning curved edges (5 – 13 Oct)

**SUPERSEDED as main model by the dense traced model.** Its end-state became B0's spline head.
Almost all experiments ran on SyntheticMRI with RelationFormer (120 queries).

- **A curved edge can be learned [DONE].**
  - On a four-patch overfit, V0 (smooth-L1 on control coordinates), fine-tuned, reached 0.1010 mm
    from the guides against 0.4812 mm for chords. V1 (a sampled-curve loss) was kept as an
    ablation and never revisited.
  - A 50-epoch V0 fine-tune evaluated on 13,071 **test** patches had native node / edge F1
    0.8841 / 0.8157, but strict 1-mm Branch F1 of only 0.1698. Box-based detection greatly
    overstates physical branch recovery.
- **Scratch 300 epochs: straight vs K8 vs adaptive degree-2 [DONE; test split].**

  | Final epoch 300 (test) | Straight | K8 spline | Adaptive degree-2 |
  |---|---:|---:|---:|
  | Branch F1 | 0.0114 | 0.0000 | 0.0050 |
  | Curve F1 | 0.2305 | 0.0014 | 0.2471 |

  - **The K8 arm collapsed** (absolute target, joint route): its outputs did not depend on the
    input (the same 38 nodes and 3 edges on every patch).
  - **Checkpoint selection misled.** Checkpoints selected by validation graph loss scored worse
    than the final ones.
- **Why the scratch spline collapsed [DONE, mechanism partly isolated].**
  - Image sensitivity vanished inside the transformer encoder. BN statistics were ruled out.
  - In paired 25k-update scratch tests, geometry gradients through the shared tokens
    (`tokens_only`) collapsed the detector: node F1 0.0000. Detached geometry (`graph_only`)
    did not: 0.2057, against 0.2364 for the no-spline control.
  - The internal mechanism was never identified.
- **The spline head does not help the detector [DONE; later confirmed].**
  - On a pretrained detector, `graph_only`, `tokens_only` and `joint` routes gave the same graphs:
    edge F1 0.87198 vs 0.87096.
  - Rule adopted: train the detector on graph losses only.
  - **Confirmed on 2 Nov.** A straight-edge RelationFormer with B0's recipe matched B0 within
    ±0.005 node, edge and Branch F1 on all three 3D datasets. The spline added only Curve F1
    (+0.015 to +0.048) ([`network_proposals_20261030.md`](network_proposals_20261030.md) §6.7).
- **Residual targets, data breadth [DONE].**
  - Absolute control targets learned to compensate node errors and did not transfer: validation
    Curve F1 0.58567 against 0.65540 for chords.
  - Residual targets on 10k patches beat chords: 0.67865, and 0.68995 at 15k updates.
- **Local image information helps geometry [DONE; later confirmed].** On a frozen detector, the
  local corrector raised full-validation Curve F1 from 0.68411 to 0.84012 and cut endpoint-neutral
  error 0.46036 → 0.35986 mm. A zero-feature control stayed at 0.68475, so the gain is
  information, not capacity.
  - Short branches got worse.
  - SyntheticMRI's mild curvature limited the conclusion.
  - The 30 Oct oracle-geometry study (true graph given) confirmed it on real data: corridor-fed
    K8 cut mean distance to the centreline on IXI from 0.537 to 0.186 mm and on BRAVA from 1.479
    to 0.615 mm.
- **Scratch vs pretrained detector [DONE, confounded].**
  - Scratch detached K8 at epoch 300: Curve F1 0.343, Branch F1 0.0118.
  - Pretrained detector + local head: 0.840 / 0.276.
  - The pretrained detector had far more prior training, and the validation subjects were in its
    own validation population.
  - Superseded: the dense model trains from scratch.
- **Curve-informed connectivity [SUPERSEDED].** Sampling MRI features along candidate curves to
  correct relation logits raised native edge F1 0.46077 → 0.50654. Only 9.7–11.9% of GT nodes
  were within 1 mm, so localization was the bottleneck. The dense model's path-feature edge
  scorer replaced the idea.
- **Catmull–Rom vs B-spline.**
  - Fitting [DONE]: on 12 IXI branches the B-spline fit better (0.03391 vs 0.05460 mm at 8
    points); on SyntheticMRI the two were about equal.
  - A scratch Catmull–Rom training run finished but was never evaluated. Its logged validation
    node loss (0.203) matches the collapsed B-spline arm (from a run log, not a report).
  - Superseded: the oracle study found the curve family matters little (polyline ≈ K8).
- **BRAVA-trained geometry heads on a frozen IXI detector (22 Oct) [DONE].**
  - Validation common-v2 Curve F1: original IXI K8 0.05782; BRAVA-trained K8 0.06050, K16 0.06064,
    K24 0.06087.
  - K24−K8 was +0.000363 [+0.000145, +0.000611]; the endpoint-neutral interval includes zero.
  - Only 81 of 5,445 Hungarian-positive pairs had both endpoints within 1 mm, and node recall was
    23.47%.
  - More controls give almost no learned gain while detection is this weak.
  - 2,048 / 512 patches, 8,000 updates, validation only.

Sources: archived under `docs/archive` (see its README). The originals are `spline_learning/` and
the evidence report `continuous_representation`.

## 6. Multi-dataset and BRAVA work (21 – 29 Oct)

**SUPERSEDED by the 30 Oct switch.**

**Setup.**
- BRAVA: 39 / 8 / 9 subjects, with 8,768 / 1,796 / 2,043 patches.
- Physical matching at 1 and 2 mm, common-v2 Branch/Curve F1, subject-paired intervals.
- Test splits unused.

### 6.1 BRAVA: from zero-shot to D08

- **Zero-shot and short scratch runs (21–22 Oct) [DONE — negative].**
  - The IXI spline model zero-shot was the best start: node recall 0.2309, edge F1 0.0269.
  - Two scratch arms (23,500 updates) recovered 4 and 1 correct edges.
- **Recovery arms D1–D3 (30,000 updates) [DONE — gate failed].**

  | Initialization | Node recall | Edge F1 |
  |---|---:|---:|
  | D1 (IXI) | 0.341 | 0.0210 |
  | D2 (SyntheticMRI) | 0.273 | 0.0134 |
  | D3 (scratch) | 0.106 | 0.0004 |

  The predeclared gate was edge F1 > 0.04. Scratch was equally poor on training patches. 3,862
  spatially positive pairs involved a Hungarian-unassigned query, so "unmatched" is not a safe
  negative label.
- **Tiny-set learnability [DONE].** Both initializations memorized 8 BRAVA patches (edge F1 0.8393
  from scratch, 0.9818 from IXI). The implementation can learn BRAVA targets.
- **G05 representation follow-up [DONE — negative].**
  - Every arm recovered zero complete branches.
  - K24 and C1 heads fit GT far better (mean maximum error 1.571 → 0.397 / 0.360 mm), but learned
    Curve F1 moved only +0.0001 / +0.0002.
  - Explicit degree-2 splitting raised Curve F1 by +0.0424 [0.0348, 0.0494], at the cost of
    more predicted edges (435 → 1,147) and a higher cycle MAE (+0.6094).
  - Raw recall was 0.307, and only 0.0767 survived the confidence cut.
- **D06 learnability and BN [DONE].** On a representative 160-patch training monitor, train node /
  edge F1 was 0.1325 / 0.0114, against validation 0.1183 / 0.0089. Training-only BN recalibration
  did not help. So the cause is neither overfitting nor BN.
- **D07 augmentation off vs on [DONE — mixed].**
  - Train edge F1 +0.1551 [+0.1294, +0.1785].
  - Validation edge F1 +0.0062 [−0.0009, +0.0125].
  - Cycle MAE 0.246 → 1.480.
  - Augmentation slows fitting, but turning it off only buys memorization.
- **D08 full-cohort long run (27–28 Oct) [DONE].** B0, 256 queries, 300 epochs (82,200 updates),
  scratch vs SyntheticMRI initialization. Validation at 2 mm:

  | Metric | Scratch | SyntheticMRI init |
  |---|---:|---:|
  | Node F1 | 0.02544 | 0.35572 |
  | Edge F1 | 0.00110 | 0.06794 |
  | Branch / Curve F1 | 0.00000 / 0.00302 | 0.00647 / 0.12174 |
  | Train-monitor node / edge F1 | 0.02282 / 0.00000 | 0.39449 / 0.08989 |

  - Edge F1 moved only 0.06542 → 0.06794 between epochs 200 and 300.
  - 96.69% of missed edges lack a recovered endpoint pair, and 89.40% of false edges touch an
    unmatched node.
  - The detector underfits its own training cohort.
- **D08 operating-point audit (CPU, train-selected thresholds) [DONE — negative].**
  - Node cut 0.45: node F1 +0.01190, but duplicates doubled.
  - Train-max-F1 edge cut: edge F1 0.06892 with 2,890 extra false positives.
  - Kept 0.5/0.5.
- **BRAVA Stage 2 matrix [PROPOSED, never run].** It planned straight vs spline, source transfer,
  25% vs 100% annotations, query budgets and a second seed.

### 6.2 Learned geometry on frozen IXI/SyntheticMRI detectors (22–23 Oct) [DONE — negative]

- Six heads (K8, K16, K24, adaptive K, piecewise C0/C1; 20,000 updates) on identical graphs. Curve
  F1 spanned 0.1211–0.1222 on IXI and 0.5817–0.5842 on SyntheticMRI.
- The adaptive-K selector (91.02% accuracy) was below an always-K4 baseline (91.58%).
- K8 was kept as the control.

### 6.3 Node localization with fine features [DONE]

- **S0.** Validation node / edge F1 @2 mm 0.7882 / 0.5881, with training close to that
  (0.7910 / 0.5894). SyntheticMRI fits.
- **Long local-refinement study** (two seeds, 12,000 updates):

  | Arm | Node F1 @1 mm | Edge F1 @2 mm |
  |---|---:|---:|
  | B0 | 0.4886 | 0.5946 |
  | A | 0.5695 | 0.6368 |
  | A-xyz | 0.5701 | 0.6339 |
  | A-control | — | 0.5947 |

  - A-xyz − B0 edge F1: +0.0393 [+0.0313, +0.0467].
  - The confidence correction adds nothing, and cycles and duplicates did not improve.
  - The 2-mm node F1 of this study (B0 0.793, A-xyz 0.817) is the reference quoted in the 30 Oct
    glossary.
  - Multiscale attention C used its fine level but showed no gain.
- **N07, modern IXI transfer** (six arms from S0, 4,096 / 512 patches, 5,500 updates, one seed).
  - Positional edge F1: B0 0.0879, A-xyz 0.0943, A-cls 0.1015.
  - A-cls − B0: +0.013610 [0.008265, 0.019401].
  - A-cls raised duplicates 313 → 535.
  - 93.02% of missed edges lack endpoints. Native edge F1 of the same model is 0.288.
- **Node verifiers V0–V3 [DONE — negative].** Learned rescoring of decoder tokens. At matched node
  counts, node F1 fell: [−0.0810, −0.0563] on IXI.
- **L08 node-first curriculum [DONE — inconclusive].**
  - Edge F1 0.12783 vs 0.10357 (+0.02426 [+0.01047, +0.03784]).
  - Branch/Curve intervals included zero.
  - The confirmation run was never launched.

### 6.4 Plans and decisions [SUPERSEDED]

- **Method-selection gate.** The roadmap's gate required a geometry gain that survives K8,
  larger-K and chord controls without adding false edges. No variant passed it.
- **Eight-week journal plan.** Partly executed (Q0, N07, 2D pilots, the D08 audit); M01, N09, N10
  and T01 were never run. N10's idea, a node heatmap from early CNN features, is the antecedent of
  the 30 Oct node-heatmap probe.
- **Later reversal.** The probe reached BRAVA validation node F1 0.539 against D08's 0.356, from
  scratch in about 7 min. So poor BRAVA nodes were a detector limitation, not unlearnable data.
- **Source disagreements.** The registry lists J01 (the D08 audit) and N07 as not run, while later
  documents record them as done; this history follows the later documents. One review says the
  limited-data botanical transfer never ran; a later review says it completed (§7).

Sources: archived under `docs/archive` (see its README). The originals are
`EXPERIMENT_REVIEW_2026102{1,2,3}.md`, `BRAVA_experiment_plan.md`,
`multidataset_continuous_graph_experiments.md`, `node_verifier_feasibility.md`,
`fine_feature_architecture_study.md`, `local_refinement_long_study.md`, the former top-level
`research/` overview with its experiment registry, and the evidence reports `brava_stage1_20261021`
and `multidataset_continuous`. Run artifacts are under `experiments/`.

## 7. Pretraining and transfer

**SUPERSEDED: the dense model trains from scratch.**

**Protocol.**
- Stage 1: mixed source + target graph pretraining, usually with adversarial domain adaptation
  (DA) at weight 1.0.
- Stage 2: target-only fine-tuning, straight or K8.
- Every run was compared with a target-only control given the same number of updates; that
  control sees more target data.
- **Checkpoints were chosen on validation, then evaluated once on the test split.**
- One seed.

**Sources.**
- Plant branching (3D2Cut grapevine) and HALVS leaf veins.
- Both are 2D masks smoothed and embedded as five-slice slabs in an empty 64³ volume, not the
  photographs.
- In the legacy Image-to-Graph-Vessel code the DA loss was computed but never optimized. The
  October runs enabled it properly.

**Full-data transfer to SyntheticMRI [DONE — negative; test split].** Held-out test on 13,071
patches, spline arms:

| Pretraining | Node mAP | Edge F1 | Curve F1 |
|---|---:|---:|---:|
| MRI only (control) | **0.4788** | **0.7258** | **0.5097** |
| Plants + MRI | 0.4448 | 0.6879 | 0.4581 |
| Leaves + MRI | 0.4418 | 0.6905 | 0.4607 |
| Plants + leaves + MRI | 0.4281 | 0.6809 | 0.4484 |
| Plants (augmented) + MRI | 0.4555 | 0.6935 | 0.4733 |

- MRI-only won every metric, and validation showed the same ranking.
- **Rendering augmentation** (nonplanar warp + textured background) helped plants slightly but
  never reached MRI-only.
- **Splines changed curve quality, not graphs.** MRI-only straight scored edge F1 0.7219 / Curve
  F1 0.4729.

**Limited data, 200 MRI patches [DONE — positive, single seed; test split].** DA off in both arms,
same updates; target exposure not matched.
- Leaves + vessels against vessels only: test node F1 0.3855 vs 0.3246, edge F1 0.2979 vs 0.2588,
  Curve F1 0.1206 vs 0.1009.
- Component error got worse, and Branch F1 stayed about 0.
- This is the only positive botanical result. It was never replicated and never tested with the
  dense model, so it remains an open idea.

**Transfer to IXI [DONE — negative; test split].** Held-out IXI test, 2,178 patches.
- IXI-only was best: node F1 0.47404, edge F1 0.15985 straight; spline 0.47369 / 0.16395.
- Plants + IXI reached edge F1 0.12084, and SyntheticMRI + IXI 0.11688 (straight).
- A four-patch IXI overfit showed the detector *can* memorize real IXI: train edge F1 1.0. The
  failure is not fitting a tiny set; it is fitting the full set.

**Earlier plants → IXI DA run (Sept) [RUNNING when last documented].** No result was written.
The test reports in its evidence folder actually evaluate an IXI-only 300-epoch model (§4.1).

**Multiscale attention on a pretrained checkpoint (B1) [DONE — negative].**
- Adding a 4³ level: node F1 +0.0125 [+0.0061, +0.0185], edge F1 −0.0034 [−0.0170, +0.0089].
- The fine level was essentially unused: attention mass 0.0014–0.0065%.
- The full-resolution U-Net later addressed the coarse grid directly.

**TopBrain mixed-pretraining estimate [PROPOSED, never run].** About 9.5 h of compute, extrapolated
from timing, not measured.

**Carried forward.** SyntheticMRI initialization is still used for the RelationFormer comparison
arms. On BRAVA D08 it clearly beat scratch for RelationFormer (edge F1 0.06794 vs 0.00110).

Sources: archived under `docs/archive` (see its README). The originals are `pretraining/`,
`multiscale_pretrained_feasibility.md` and the evidence reports `domain_ixi_pretraining` and
`topbrain_training_time_estimate`.

## 8. Native 2D

**SUPERSEDED by the native 2D dense model.** The 2D work used the original GrEVe-era 2D
RelationFormer (ResNet101 + deformable DETR) with this project's losses, A-xyz and a K8 head.

**Setup.**
- Datasets:
  - AV-WIDE: 17 / 6 images;
  - REAVER diaphragm60x: 4 / 2 images, in microns;
  - OCTA-Synth: official 88 / 40 split.
- 128² crops, validation only, cutoff 0.5, one seed.

**Companion repository.** The Image-to-Graph-Vessel inventory is kept as provenance only:
- its DA loss was inert;
- config names are unreliable;
- no 2D checkpoint was recoverable;
- its August "biological beats roads" ranking held in 2D but flipped between 3D variants.

**Native2D-01 to -05 [DONE].**
- **Implementation is correct.** A tiny model fits two AV-WIDE crops to F1 1.0.
- **A target-fit defect** (a K8 control at −56,890 px) invalidated Native2D-02 geometry. The fix
  was refitting from uniform arc-length samples.
- **Scratch training never works.** On AV-WIDE, B0 and A-xyz from scratch predicted zero nodes at
  cutoff 0.5. All six scratch100 models in Native2D-04 predicted empty graphs.
- **Mixed plant/vessel pretraining (50 + 100 epochs)** produced graphs (validation node / edge
  F1):
  - OCTA-Synth B0 0.8038 / 0.5950;
  - REAVER B0 0.7455 / 0.4085;
  - AV-WIDE B0 0.2045 / 0.0123.
- **This benefit is confounded.**
  - Pretraining also adds about 1.29 million target-crop exposures.
  - Validation sets are tiny.
  - **Every Native2D-02 to -05 run used a broken backbone:** `init_weights: null` left a random,
    un-normalized ResNet101 trained at the backbone learning rate meant for ImageNet weights.
- **The ImageNet fix (WS1, 30 Oct) did not rescue scratch training.** AV-WIDE node / edge F1 was
  0.020 / 0.001 and REAVER 0 / 0 (taken from a run json, not a report).
- **A-xyz is unstable in 2D.** It improves OCTA 1-px localization but collapses on REAVER.
- **Splines in 2D.** Same-graph K8 minus chord Curve F1 was −0.0014 to +0.0165. A trained
  straight-only OCTA model matched the curved one (edge F1 0.5903 vs 0.5950). OCTA edges are
  nearly straight, so 2D cannot show a spline benefit.

**Superseded by the 2D dense model** (16-NN edges, from scratch, 15k updates). Validation node /
edge F1, from [`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) §5:

| Dataset | B0 (mixed) | 2D dense |
|---|---|---|
| AV-WIDE | 0.204 / 0.012 | 0.454 / 0.133 |
| REAVER | 0.745 / 0.408 | 0.806 / 0.589 |
| OCTA-Synth | 0.804 / 0.595 | 0.903 / 0.782 |

Single seed. REAVER has 2 validation images. **[PROPOSED]** Porting traced connectivity to 2D.

Sources: archived under `docs/archive` (see its README). The originals are `native2d_*.md` and the
former `research/image-to-graph-vessel.md`. Run reports are under `experiments/`.

## 9. What carried forward into the dense model

Each diagnosed failure and the dense model's answer (design rationale in
[`network_proposals_20261030.md`](network_proposals_20261030.md) §1B.6 and §3B.7):

| Diagnosed in RelationFormer | Fixes that failed | Dense traced model |
|---|---|---|
| Nodes regressed from a 2³ grid | Local refinement (partial), multiscale C/B1 | Full-resolution 3D U-Net heatmap |
| Duplicates as confident as true nodes | NMS, thresholds, verifiers, N08 | Heatmap focal loss + 3³ NMS; unmatched peaks are trained as negatives |
| Hungarian train/inference mismatch | Oracle deletion, Hungarian selection, hard negatives | No Hungarian; training candidates are the model's own peaks |
| Relations blind to the image between nodes | Adapters, R06, R07 | Edge scorer reads U-Net features along the traced centreline |
| Edge-threshold trade-off | Threshold sweeps (false cycles) | Thresholds chosen on `monitor_train` only |
| Edge recall bounded by node recall | Not solved | Still the main limit: about 80% of missed BRAVA long edges lack an endpoint |

**Kept from this era:**
- the datasets, splits and graph-positive patches;
- the dense centreline as geometric GT;
- physical metrics with degree-2 contraction;
- "topology + curved edge", with per-edge K8/K16/K24 for long or tortuous branches;
- the residual and local-image geometry ideas;
- train-only threshold selection;
- reporting node recovery and relation quality separately.

**Dropped:**
- adaptive degree-2 nodes as targets;
- query-capacity tuning;
- hard-negative and relation-only fixes;
- node-type and degree heads;
- verifiers;
- botanical and SyntheticMRI pretraining;
- adversarial DA;
- multiscale attention on RelationFormer;
- automatic loop repair.

Most were dropped because the architecture changed, not because each was individually
disproven.
