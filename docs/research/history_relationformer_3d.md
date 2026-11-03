# 3D RelationFormer on SyntheticMRI and IXI — consolidated history (to 29 Oct 2026)

**Status.** This document covers the 3D RelationFormer detector on SyntheticMRI and IXI, from the
first IXI integration (Sept 2026) through the numbered IXI diagnostic series 1–11 (18–20 Oct),
plus the follow-ups (26–28 Oct) that were appended to those reports. It consolidates archived
documents and was written on 3 Nov 2026. **[SUPERSEDED by the dense traced model]**
([`dense_traced_model_20261102.md`](dense_traced_model_20261102.md)), whose design answers the
failure mechanisms identified here (see §5).

## Context

The question was why RelationFormer, which fits SyntheticMRI reasonably well, produces such poor
graphs on real IXI MRA. RelationFormer is a DETR-style image-to-graph transformer. A 3D
SE-ResNet encoder (stride 32) feeds a deformable transformer. Each object query predicts node
presence and coordinates, and an MLP scores query pairs for edges
([`../MODEL.md`](../MODEL.md), [`../LOSSES.md`](../LOSSES.md)). In training, Hungarian matching
decides which query represents which GT node. At inference there is no matcher: every pair of
confidence-retained queries is scored.

- **Datasets.** SyntheticMRI, and IXI with segmentation-derived reference graphs. Two IXI
  extractions were used:
  - an older adaptive Vedo-original extraction (13,742 / 2,961 / 2,384 active
    train/val/test patches), used for the September–early October runs;
  - the current `transfer_splines_k8_20261015` cache (10,000 / 2,649 / 2,178 patches;
    123 / 26 / test subjects), used from 15 Oct on. Splits are by subject.
- **Protocol of the diagnostic series.** Every model in reports 1–9 is frozen or derived from the
  same parent checkpoint (see "delayed-spline parent" below). Most use a fixed 256-train /
  128-validation patch set (26 validation subjects). Validation was reused for development;
  nearly everything is single-seed.
- Metric definitions are in [`history_metrics.md`](history_metrics.md). The later multi-dataset,
  BRAVA, node-verifier, K8 and D07/D08 work (21–29 Oct) is in
  [`history_multidataset_brava.md`](history_multidataset_brava.md).

## Shorthand

- **Delayed-spline parent.** The IXI-only RelationFormer checkpoint `ixi_delayed_spline`,
  step 28170: straight-edge pretraining, then spline fine-tuning. It was the frozen starting
  point for reports 1–10. Its actual losses were weighted-CE node classification [0.25, 0.75]
  and **focal** edge loss without ratio upsampling ([report 10][r10]). The roadmap described
  its baseline as regularized CE; [report 7][r7] found the discrepancy.
- **Local-image refinement.** A small MLP that samples the early 32³ CNN map (`layer0.relu1`,
  64 channels) at 27 points (offsets −4/0/+4 input voxels per axis) around each predicted
  node. It outputs a bounded coordinate correction (±4 voxels per axis) and a class-logit
  correction ([report 3][r3]). Later called **A** (both outputs), **A-xyz** (coordinates
  only) and **A-cls** (classification only).
- **B0.** Baseline RelationFormer recipe: no node refinement, K8 spline head (8-control
  B-spline edges).
- **M / U; MM / MU / UU.** A predicted node is matched (M) if it is one-to-one positionally
  paired with a GT node within 2 mm, else unmatched (U). A false edge is MM, MU or UU by its
  endpoints.
- **Positional edge F1.** An edge counts as correct only if both endpoints are matched within
  2 mm and their GT nodes are adjacent. **Native edge F1** is the older box-based benchmark
  metric: it is much higher and is not endpoint-correct adjacency
  ([project review][rev]). They must not be compared.
- **Conditional relation recall / AP.** Computed only on pairs whose two endpoints were both
  recovered. **All-retained AP** includes every retained pair.
- **β₀ / β₁ MAE.** Mean absolute per-patch error in component count / cycle count.
- **Train-max-F1 / train-preserve-precision.** Edge thresholds chosen on training patches
  (maximise F1, or maximise recall at the baseline training precision), then applied unchanged
  to validation.
- **Raw-query oracle.** A GT-assisted, non-deployable selection of the closest query to each GT
  node among all 256 queries, including confidence-rejected ones.
- **P0 / P1 / P2.** Priorities in the 19 Oct roadmap: P0 read-only diagnostics, P1 small
  controlled training, P2 architecture redesign ([roadmap][road]).
- **N06 / N08 / R06 / R07 / Q0.** Codes from the 26–28 Oct registry, used in §4:
  - N06: GT-assisted representative-query oracle;
  - N08: GT-free relation-aware query selection;
  - R06: CNN-conditioned relation-token residual;
  - R07: reliable matched-negative resampling, with arms S0 (unchanged) and S1 (resampled);
  - Q0: IXI status and target-capacity audit.
- **`monitor_train`.** A fixed set of training patches used to choose thresholds (current
  convention; see the glossary of [`network_proposals_20261030.md`](network_proposals_20261030.md)).

---

## 1. Before the diagnostics: training the detector on IXI (Sept – early Oct)

All edge F1 values in this section are the native box-based benchmark metric (old IXI
extraction). They are not comparable with the positional edge F1 used from §2 on.

### 1.1 SyntheticMRI loss-ablation matrix — [PROPOSED, never run (no results in the docs)]

A four-arm SyntheticMRI fine-tuning matrix was designed ([EXPERIMENTS.md][exp]):
- `baseline`: weighted-CE nodes, CE + ratio upsampling for edges;
- `focal`: focal nodes and edges, plus unmatched edge candidates;
- `betti`: baseline plus H0/H1 Betti losses, coefficient 0.1 each, after 2 spatial-only epochs
  and a 5-epoch ramp;
- `focal_betti`: both.

The design fixed one initialization (plants + SyntheticMRI pretraining) and 10 epochs on at most
256/32 patches. It required an eight-sample overfit check first. The Betti coefficients were
explicitly experimental, not baseline settings. No results from this matrix appear in any
assigned document.

The closest evidence is a colleague's externally reported SyntheticMRI study, recorded in
[report 10][r10] and not reproduced here. Node focal with edge CE + upsampling raised node AP
0.6422→0.6540 and lowered cycle MAE 1.360→0.456. Adding edge focal reduced cycles further but
increased fragmentation, and hard unmatched supervision often suppressed needed connections.

### 1.2 Object-query capacity — [DONE]

- **120 vs 192 queries** ([benchmark][qb]). On seven IXI volumes (884 patches), 59 patches
  (6.67%) had more than 120 nodes; the maximum was 177. Going to 192 queries cost about
  +2% step time and +0.10 GiB peak memory. The benchmark recommended 192.
- The IXI runs actually used **256 queries**. The 28 Oct Q0 audit of the current contracted K8
  targets found a maximum of 153 nodes (train) and 141 (val), with zero degree-2 object
  anchors. So 120 is not capacity-safe for IXI, and 256 was kept ([report 3, Q0 section][r3]).
  A historical 239-node patch is pre-contraction evidence from the older extraction.

### 1.3 IXI 256-query integration smoke (T09) — [DONE]

This was an integration test only: 32 train / 16 val patches, one epoch
([evidence/T09](evidence/T09/README.md)). It found and fixed one real bug. Exact patch-face
nodes have normalized coordinates −0.5/64 or 63.5/64, which the augmentation validator rejected;
`allow_patch_faces=True` now admits that half-voxel extension. The evaluator now also finds the
source manifest via `generation_config.json`. The one-epoch model predicted 30 isolated nodes and
no edges, which is expected at that stage.

### 1.4 IXI 300-epoch adaptive-GT run and held-out test (T09_long) — [DONE; test split used]

Setup ([evidence/T09_long](evidence/T09_long/README.md)):
- 8,000 IXI training patches, random initialization, 256 queries, two H100s;
- 100 epochs, then a 200-epoch continuation with the learning rate restarted at 7e-6;
- after 100 epochs: validation node F1 0.374, native edge F1 0.296, Branch F1 0.0057;
- epoch 270 selected by validation Branch F1 (0.0067563).

It was then evaluated once on **all 2,384 held-out test patches** (21 patients), with no
threshold tuning ([test results](evidence/T09_long/test_results.md)):

| Test metric | Nodes | Edges (native) | Branches (1 mm) |
|---|---:|---:|---:|
| Precision / recall | 0.3570 / 0.4778 | 0.1963 / 0.7003 | 0.00298 / 0.02649 |
| F1 | 0.4086 | 0.3067 | 0.00535 |

The graphs were massively overconnected:
- predicted edges per patch: 179.76, against 50.39 in GT;
- predicted cycles per patch: 109.27, against 0.77;
- Branch F1 was zero on 75.8% of test patches.

Conclusion: more epochs did not fix the extra-edge problem. This test split has since been
inspected, so any later method needs a new untouched holdout for a paper claim
([roadmap][road] §4).

Sources disagree on one point. The roadmap lists "node F1 0.409 … edge F1 0.307" and "179.76
vs 50.39 edges" as coming from different protocols. `test_results.md` shows that both come
from this same epoch-270 test evaluation.

### 1.5 Node-type head and soft degree decoding (T10) — [DONE, validation only]

([evidence/T10](evidence/T10_node_type_pilot.md))

- **Old target composition.** In 100 sampled patches of the old extraction: degree-2 57.24%,
  terminals 29.40%, junctions 13.36%. The current K8 targets have no degree-2 anchors (Q0).
- **Type head alone does nothing.** A four-class type head was trained for 10 epochs from the
  epoch-270 model. Edge F1 went 0.2673→0.2684 and predicted edges 214.78→211.08 (target
  52.43). The epoch-300 initializer gave the same picture.
- **Types are poorly predicted.** Type accuracy among matched nodes was 59.63%. Junctions were
  7.9% of predictions against 15.0% of GT.
- **Soft degree penalty at decoding.** A soft penalty from predicted type probabilities helped
  more than plain thresholding at similar edge counts:
  - plain threshold 0.65: edge F1 0.3421, Branch F1 0.00536;
  - penalty 0.5: edge F1 0.3611, Branch F1 0.00658;
  - stronger penalties raised edge F1 (0.4101 at penalty 2) but destroyed branch continuity
    (Branch F1 0.00067).
- Conclusion: edge count and β₁ alone are unsafe selection criteria.

### 1.6 Focal-loss 300-epoch ablation — [DONE; exploratory test-split evaluation]

Three IXI-only 300-epoch models were compared on the 2,384-patch test split: cross-entropy,
object focal, and object + type focal, all with the type head
([evidence/focal_ixi_300ep](evidence/focal_ixi_300ep/README.md)).
- **Focal losses.** They raised node F1 (0.5103 → 0.5417 / 0.5474) but predicted more edges
  and cycles. Native edge F1 went 0.3894 → 0.3679 / 0.3771.
- **Degree penalty 0.5.** It cut edges to 89.60 per patch for CE, which reached native edge
  F1 0.4847. No setting came near the GT cycle count.
- **Branch F1** stayed around 0.01 everywhere.

### 1.7 Degree-consistency training — [RUNNING when last documented; no final result in the docs]

This run combined three changes at once: an expected-degree Smooth-L1 loss, unmatched hard
negatives, and a higher object-classification weight
([evidence/degree_consistency](evidence/degree_consistency/README.md)).

At epoch 220 (interim, validation), compared with the baseline at the same epoch:
- predicted edges 77 vs 158, and cycles 23 vs 92 (GT 52 and 0.83);
- native edge F1 0.446 vs 0.369;
- but more fragmented (12.3 components per patch vs 2.5; GT 4.7), with Branch F1 0.0057 vs
  0.0093 and node F1 0.466 vs 0.487.

The planned test evaluation has no recorded result. Because three factors changed together, the
outcome cannot be attributed to hard negatives ([roadmap][road] §2.3).

---

## 2. The IXI diagnostic series (18–20 Oct): what fails

The 19 Oct [project review][rev] and [roadmap][road] reframed the work. The rule became "no new
loss sweep until saved predictions show what fails". The series below follows that plan: P0
attribution from frozen caches, then P1 small controlled training. P2 (multiscale attention) was
**[PROPOSED, never run]**.

### 2.1 Labels are consistent with the mask; most missed edges lack an endpoint — [DONE]

([report 1][r1]; delayed-spline parent, full validation)
- **Labels.** In a 16-patch audit, all GT nodes lie on the segmentation and 99.52% of GT curve
  samples lie inside it. This shows consistency with the mask the graphs were extracted from,
  **not anatomical correctness**.
- **Predictions are image-guided but poorly placed.** 34.25% of predicted nodes are inside the
  mask (random points: 0.94%), and 65.57% are within 1 mm of foreground.
- **Edge budget at 2 mm**, over 51,937 GT edges in 2,649 patches:
  - 69.38% miss at least one endpoint;
  - 26.45% have both endpoints but no predicted relation;
  - 4.17% are correct.
- Node recall is 50.82%, and conditional relation recall 13.62% (6.57% in the longest
  arc-length quartile, against 14.45% in the shortest).

So node errors already block most edges, and relations are weak even when endpoints exist.

### 2.2 Node localisation: the 2³ feature grid cannot place nodes — [DONE]

**Frozen-feature probes** ([report 2][r2]; [evidence](evidence/ixi_features_nodes_20261018/results.md)).
Linear voxel-foreground probes, validation AP at about 0.86% prevalence:
- early 32³ CNN features: 0.6551;
- raw intensity: 0.6802;
- projected CNN features at 2³: 0.0271;
- transformer encoder memory at 2³: 0.0246.

A resolution control explains this: even the **GT mask pooled to 2³** reaches only AP 0.0665
(0.9512 at 32³). The transformer does not destroy information relative to its input. The
**2×2×2 grid itself cannot represent fine vessels**, so every node must be regressed from a
blend of at most 8 vectors.

Node recovery at 2 mm was 63.43% on training vs 51.12% on validation (junctions 67.62% vs
47.99%; interior terminations 48.54% vs 38.13%). This is a generalization gap **plus**
substantial training error.

**Local-image refinement pilot** ([report 3][r3]; [comparison](evidence/ixi_node_refinement_20261018/comparison.md)).
Only the node heads were trained, for 1,000 updates, and validation used 128 patches, one seed:

| Arm | Node F1 (native) | Native edge F1 | Curve F1 |
|---|---:|---:|---:|
| Node-head continuation | 0.4857 | 0.1553 | 0.1171 |
| Local-image refinement | 0.5196 | 0.1673 | 0.1676 |
| Zero-image control | 0.4829 | 0.1538 | 0.1164 |

- Endpoint-pair recovery rose 30.20%→37.96%, and junction recall 49.37%→58.22%.
- Conditional relation recall did not move (12.35%→12.48%).
- Straight chords between the refined endpoints gained almost as much Curve F1 (0.1134→0.1641)
  as the frozen splines. So the geometry gain came from better endpoints.

**Coordinate vs classification residual** ([report 7][r7] C). At inference, the coordinate
residual alone carries nearly all the gain: positional edge F1 0.0730, against 0.0581 with
neither residual (paired interval [+0.0087, +0.0207]). The classification residual alone gives
0.0582. This was the first evidence for what later became A-xyz.

**Node-loss weighting / focal** ([report 10][r10]; [evidence](evidence/ixi_node_focal_20261020/results.md)) — [DONE — negative].
Setup: full-network continuation from the parent plus refinement, 2,048 / 512 patches,
6,900 updates ([job review](evidence/job_review_20261020/review.md)). Validation results:
- weighted CE [0.25, 0.75]: node P/R 0.4365 / 0.5522, positional edge F1 0.1189, cycle MAE
  13.72;
- equal-weight CE [1, 1]: 0.5474 / 0.4225, edge F1 0.1165, cycle MAE 3.92;
- equal-weight focal (γ = 2): 0.5293 / 0.3071, edge F1 0.0767, cycle MAE 2.69.

Equal weights trade recall for precision and fewer cycles. Focal modulation lowered recall
and edge F1 further. The IXI data do not reproduce the colleague's SyntheticMRI focal benefit.

One confound applies to all three arms: they switched from the parent's focal edge loss to
CE + ratio at the same moment. The report records the table without a final written decision.

### 2.3 Duplicates and extra nodes: confidence cannot tell them apart — [DONE]

From the frozen query attribution ([report 6][r6]; [evidence](evidence/ixi_query_attribution_20261019/README.md))
and mechanism batch B ([report 7][r7]), local-image model, 128 validation patches:

- **Composition of retained nodes.** Of 3,827 retained nodes, 584 are duplicate candidates,
  672 are foreground-supported extras and 592 are off-vessel.
- **Confidence does not separate duplicates.** Matched-vs-duplicate confidence AUC is 0.5495,
  and 70.4% of duplicates have confidence ≥0.9 (medians 0.9521 vs 0.9632).
- **Thresholding does not help.** Lowering the node cutoff 0.5→0.1 raises recall only
  56.27%→61.27%, while precision falls 51.71%→45.38%. All 256 raw queries together would
  recover 83.62% of GT nodes.
- **Fixed NMS hurts** [DONE — negative]. Positional edge F1 falls 0.0737 → 0.0626 at 0.5 mm
  and 0.0508 at 1 mm, because true nearby nodes are lost.
- **Decoder tokens.** A linear probe on decoder tokens separates node categories somewhat better
  than confidence (balanced accuracy 0.4836 vs 0.3550). Neither is a usable selector.

### 2.4 Train/inference mismatch: the relation head is never trained on what it sees — [DONE]

([report 6][r6], [report 7][r7] A, [report 8][r8])

- **False edges.** 91.1% of false edges (668/733) touch an unmatched node, but only 35.7% touch
  a duplicate. Relation supervision in training covers matched pairs only; at inference every
  retained pair is scored.

| Policy (validation, 2 mm) | Node recall | Edge TP / FP | Positional edge F1 | Conditional recall |
|---|---:|---:|---:|---:|
| Normal decoding | 0.5627 | 142 / 733 | 0.0737 | 0.1257 |
| Oracle: delete unmatched nodes | 0.5627 | 142 / 65 | 0.0892 | 0.1257 |
| Raw-query oracle (all 256) | 0.8362 | 162 / 103 | 0.0999 | 0.0722 |
| Training-style Hungarian selection | 0.6267 | — | 0.1121 | 0.1619 |

- **Even perfect selection does not rescue edges.**
  - Deleting every unmatched node (a GT oracle) leaves edge recall at 4.77%.
  - The raw-query oracle doubles the recoverable GT edges (1,130→2,243), but correct edges
    rise only 142→162. Only 42 correct edges are shared between the two selections.
- **Hungarian selection is not the fix.** Selecting the queries that the training matcher
  would pick gives a slightly better operating point. Its conditional AP (0.4658) is no better
  than normal (0.4702): the paired interval includes zero.
- Conclusion: the mismatch exists, but fixing node selection alone leaves the relation scorer
  as a second bottleneck.

### 2.5 Relations do not look at the image, and relation fixes failed — [DONE — negative]

**Isolation** ([report 4][r4]; [evidence](evidence/ixi_relation_isolation_20261019/results.md)).
Results on recovered-endpoint pairs (24,054 validation pairs, 4.70% positive):
- the existing relation head has validation AP 0.4702 but recall only 0.1257 at 0.5;
- refitting it on fixed tokens raised training AP (0.5976) and lowered validation AP (0.3774);
- a geometry-only probe reaches 0.4043, and a straight-chord image probe 0.4261;
- recall falls from 15.62% in the shortest arc quartile to 3.80% in the longest.

**Threshold calibration** ([report 5][r5]; [evidence](evidence/ixi_threshold_calibration_20261019/results.md)).
Lowering the edge threshold to the train-max-F1 value 0.382:
- raised positional edge F1 0.0737→0.1608;
- cut β₀ MAE 19.29→2.32;
- but raised β₁ MAE 1.04→19.52, i.e. many false cycles.

The precision-preserving threshold (0.4977) barely helped (0.0780). The roadmap therefore
banned repeating edge-threshold sweeps as a solution.

**Relation-only continuations** ([report 7][r7] D; [report 8][r8] B). Five heads were compared
on frozen nodes, each trained 1,000 updates on 256 cached patches:

| Relation head | Edge TP / FP | Positional edge F1 | Cycle MAE | Val conditional AP |
|---|---:|---:|---:|---:|
| Original (frozen) | 142 / 733 | 0.0737 | 1.04 | **0.4702** |
| Focal continuation | 324 / 2244 | 0.1169 | 5.70 | 0.4098 |
| Regularized CE (ratio 0.15) | 554 / 4325 | 0.1410 | 16.51 | 0.4153 |
| CE + unmatched hard negatives | 446 / 3037 | 0.1381 | 10.73 | 0.3659 |
| CE + matched hard negatives | 419 / 3121 | 0.1286 | 9.71 | 0.3996 |

- Every trained head fits training better and ranks validation pairs **worse**. The AP
  intervals against the original head are below zero on both pair populations.
- Their fixed-cutoff edge F1 gains come from moving the operating point, paid for in false
  cycles.
- Unmatched mining removed 32.9% of U-involved false edges but lost 108 of 554 true edges.
- When every head is set to a common training-precision target, edge and Curve F1 rise, but
  none of them avoids a larger cycle error. The original scorer was retained.

**Image-aware relation adapters** ([report 9][r9]; [evidence](evidence/ixi_image_relation_learning_20261020/results.md)).
Four residual adapters were trained on top of the frozen scorer (2,048 / 512 patches,
65,600 updates):
- token + geometry only (control);
- plus 27 samples around each endpoint;
- plus samples along the straight chord;
- plus a 3×3 cross-section corridor along the chord.

By the final update all of them had overfit. Validation conditional AP was 0.31–0.32, against
0.4723 for the original, while training AP was 0.60–0.65.

A retrospective re-analysis of the saved 1k/2k checkpoints did find a real image signal:
- at 2k updates, chord and corridor adapters reached validation AP 0.50156 / 0.50191, vs
  0.47660 for the token control;
- chord minus control: +0.02496 [+0.02153, +0.02802];
- under train-selected precision-preserving thresholds, edge F1 was 0.13147 / 0.13849 vs
  0.08047 for the original.

These checkpoints were examined after seeing the learning curves, so no early-stopping rule was
validated, and Branch/Curve F1 were not recomputed. At the final update, the endpoint adapter's
fixed-cutoff gains came with 12,397 vs 2,821 false edges
([job review](evidence/job_review_20261020/review.md)).

Conclusion: image evidence between nodes helps ranking. However, bolting it onto frozen
RelationFormer tokens with a small adapter is unstable.

### 2.6 Appearance vs graph reasoning: segmentation-input oracle — [DONE]

([report 11][r11]; [evidence](evidence/ixi_segmentation_input_20261020/results.md))

Setup: the same architecture was trained from a pre-IXI SyntheticMRI initializer on raw MRA vs
the supplied binary vessel mask (2,048 / 511 patches, matched budget). Validation, 2 mm:
- node precision / recall: 0.4383 / 0.5488 with raw MRA, 0.5871 / 0.6580 with the mask;
- positional edge F1: 0.0415 → 0.0904;
- conditional recall: 0.0635 → 0.1180;
- Branch F1: 0.0061 → 0.0243.

Removing appearance helps, but **even with the mask that generated the GT, edge F1 is 0.09**. A
large part of the failure is therefore in graph decoding, not in seeing vessels.

Caveats:
- the masks generated the references, so this is a reconstruction oracle, not anatomy;
- the spline head was frozen;
- the run is short and single-seed.

---

## 3. Mechanisms identified (summary)

1. **Node localisation from a 2³ memory.** Nodes are regressed from 8 blended vectors. Early
   high-resolution CNN features help (coordinate refinement), but only locally (§2.2).
2. **No usable uniqueness signal.** Duplicates are as confident as true nodes, and fixed NMS
   removes real nodes (§2.3).
3. **Train/inference mismatch.** Training supervises relations only on Hungarian-matched pairs.
   At inference, 91% of IXI false edges touch unmatched nodes (§2.4), and 82–86% on
   SyntheticMRI (§4).
4. **Edge recall is bounded by node recall.** 62–69% of IXI GT edges lack a recovered endpoint
   (§2.1, §2.4). In the later models, about 90–94% of *missed* edges lack a recovered endpoint
   pair (§4).
5. **Relations never look between the nodes.** The relation MLP sees two summary tokens and two
   global tokens. Its ranking does not generalize, and retraining it on fixed tokens overfits
   (§2.5).
6. **Underfitting of real training data**, not only a generalization gap. Training node
   recovery was 63%, and training positional edge F1 stayed around 0.1 (§2.2, report 5 train
   F1 0.1037).

---

## 4. Follow-ups appended to these reports (26–28 Oct) — [DONE]

These used the stronger SyntheticMRI B0/A-xyz models (two fine-tuning seeds, 12,000 updates)
and, for IXI, a new matched six-arm study. They refine §3 rather than reverse it.

- **Modern error attribution** ([report 6][r6]). For SyntheticMRI A-xyz:
  - 82.4–83.2% of false edges touch a spatially unmatched node, but only about 35% touch a
    Hungarian-unassigned query;
  - 90.7–90.8% of missed edges lack a recovered endpoint;
  - only 9.2–9.3% are pure relation misses.

  The old IXI model differs: 34.85% of its missed edges are relation misses, with both
  endpoints recovered.
- **N06, representative oracle** ([report 6][r6]). For the same recovered GT vertices, choosing
  the confidence-preferred nearby query instead of the nearest one raises edge F1 from
  0.720/0.729 to 0.834/0.838 (normal decoding: 0.634). Spatially equivalent queries therefore
  carry different relation content. On old IXI, the same oracle reaches only 0.158.
- **N08, GT-free selectors** [DONE — negative] ([report 6][r6]). None of the predeclared
  confidence / relation-support rules consistently beat count-matched controls (for example
  +0.000159 and +0.002071, intervals including zero). The search was stopped.
- **R06, CNN-conditioned relation tokens** [DONE — negative] ([report 9][r9]). The image branch
  is used (shuffling hurts), but edge F1 drops versus the original: −0.003678 / −0.002978, both
  intervals below zero.
- **R07, reliable nearby negatives** [DONE — negative] ([report 8][r8]):
  - every matched negative already enters the loss, and only 18 of 28,494 training negatives
    (0.063%) are reliable nearby ones;
  - resampling them (S1) gives +0.000755 / +0.000684 edge F1 over S0, but stays below the
    original checkpoint.
- **IXI six-arm local-head study** ([report 3][r3]). Setup: 4,096 / 512 patches, 5,500 updates,
  all arms from the SyntheticMRI B0 checkpoint, 256 queries, one seed.
  - A-cls raised positional edge F1 0.087926→0.101536 (+0.013610 [+0.008265, +0.019401]) and
    Curve F1 0.214892→0.224389. It also raised duplicates (313→535) and cycle MAE
    (1.998→2.580).
  - A-xyz improved 1-mm node F1 (0.162→0.193/0.205) without a clear edge gain.
  - The combined head A was not better than either.
  - 93.02% of A-cls missed edges still lacked an endpoint pair, and full edge recall was 0.076.

  Its native edge F1 (0.288) illustrates how far the native metric overstates positional
  accuracy.
- **BN provenance audit and L06 training-only BN recalibration** [DONE — partial]
  ([spline-collapse-diagnosis.md, appended 26–28 Oct](../archive/legacy-20261103/docs/research/spline_learning/spline-collapse-diagnosis.md)):
  - 30 of 32 archived August evaluation logs ran batch-norm calibration on the *validation*
    loader (`evaluate.py --bn-calibration-batches` chooses it), so those historical metrics are
    BN-adapted evaluations, not frozen references. Modern plans use `bn_calibration_batches=0`.
  - Training-only recalibration of a SyntheticMRI K8 checkpoint (256 training patches): node F1
    @2 mm 0.752528 → 0.756058, edge F1 @2 mm 0.445495 → 0.439856. Descriptive only, not adopted.
    The IXI half failed for an implementation reason and was never recovered.

---

## 5. How the dense traced model addresses these mechanisms

The design rationale is written out in
[`network_proposals_20261030.md` §1B.6](network_proposals_20261030.md#1b6-where-the-failure-modes-come-from)
(failure modes) and
[§3B.7](network_proposals_20261030.md#3b7-why-it-addresses-each-diagnosed-failure)
(the response).

| Mechanism (this history) | Fixes tried here and their outcome | Dense traced model |
|---|---|---|
| 2³ node localisation | Local CNN refinement: helps locally (§2.2) | Full-resolution 3D U-Net heatmap; IXI val node F1 0.639 (probe) vs about 0.37 |
| Duplicates, uninformative confidence | NMS, thresholds, N08 selectors: fail (§2.3, §4) | Heatmap focal loss + 3³ NMS; surviving duplicates are unmatched and trained as negatives |
| Hungarian train/inference mismatch | Oracle deletion, Hungarian selection, unmatched mining: no deployable fix (§2.4, §2.5) | No Hungarian; training candidates come from the model's own peaks with the evaluation matcher; unmatched peaks are supervised |
| Relations blind to the image between nodes | Chord/corridor adapters, R06: overfit or no gain (§2.5, §4) | Edge scorer samples U-Net features along the traced centreline / corridor, trained end-to-end |
| Edge threshold trade-off | Lower thresholds: false cycles (§2.5) | Thresholds chosen on `monitor_train` only |
| Edge recall bounded by node recall | Not solved | Still the main limit: about 80% of missed BRAVA long edges lack an endpoint |

Outcome on IXI validation: the dense traced model reaches positional edge F1 0.313 and Curve F1
0.770, against 0.156 / 0.344 for RelationFormer B0 at literature scale
([dense model §4.1](dense_traced_model_20261102.md)).

## 6. What carried forward / what was dropped

**Carried forward**
- High-resolution image features are needed for nodes. This came from the probes and the
  coordinate refinement, and became the full-resolution U-Net.
- Train and inference must see the same candidates, and unmatched candidates must be supervised.
- Edge decisions must read the image between the two nodes.
- Thresholds are chosen on training data only. Edge count, β₀ or β₁ alone never select a model.
- Positional (endpoint-correct) edge F1, Branch F1 and Curve F1 replace native box edge F1.
- Report node recovery and conditional relation quality separately.

**Dropped**
- Edge-threshold sweeps and fixed NMS as fixes.
- Hard-negative mining (matched, unmatched, reliable-nearby) and relation-only continuations.
- Node-type heads, degree-consistency losses and soft degree decoding. They reduced edge
  counts but never produced sound topology.
- Node focal / class-weight tuning for RelationFormer.
- GT-free query selectors (N08) and token residuals (R06/R07).
- The multiscale deformable transformer (P2-A), which was proposed but never built. The dense
  model made it moot.
- The SyntheticMRI Betti / focal ablation matrix, which was never run.

## Sources

Archived originals (post-move locations):
- [EXPERIMENTS.md][exp] — design of the 4-arm SyntheticMRI loss-ablation matrix (baseline / focal / Betti); protocol only.
- [PROJECT_REVIEW_20261019.md][rev] — 19 Oct review of what earlier experiments support; launched the diagnostic series.
- [IXI_graph_extraction_diagnostic_roadmap.md][road] — 19–20 Oct P0/P1/P2 diagnostic plan and summary of prior IXI evidence.
- [1-ixi-supervision-audit.md][r1] — label/foreground support audit and endpoint-vs-relation edge budget.
- [2-ixi-feature-node-diagnostics.md][r2] — frozen-feature foreground probes, resolution control, node subtypes.
- [3-ixi-node-refinement-pilot.md][r3] — local CNN node refinement pilot; 26–28 Oct IXI six-arm A-xyz/A-cls study; Q0 capacity audit.
- [4-ixi-relation-isolation.md][r4] — relation head on recovered/GT endpoints, refit and chord-image probes.
- [5-ixi-threshold-calibration.md][r5] — train-calibrated edge thresholds and the cycle trade-off.
- [6-ixi-query-attribution.md][r6] — all-query attribution, MM/MU/UU false edges, deletion/NMS controls; modern attribution, N06, N08.
- [7-ixi-mechanism-batch.md][r7] — raw-query oracle, confidence/token probes, residual ablation, relation-loss continuations.
- [8-ixi-relation-matching-calibration.md][r8] — Hungarian selection, calibrated operating points; R07.
- [9-ixi-image-aware-relation-learning.md][r9] — endpoint/chord/corridor relation adapters, early-checkpoint re-analysis; R06.
- [10-ixi-node-focal-ablation.md][r10] — node class-weight / focal continuation on IXI.
- [11-ixi-segmentation-input-oracle.md][r11] — raw MRA vs vessel-mask input.

Evidence kept in place:
- [evidence/T09/README.md](evidence/T09/README.md) — 256-query integration smoke.
- [evidence/T09_long/README.md](evidence/T09_long/README.md), [test_results.md](evidence/T09_long/test_results.md) — 300-epoch IXI run and held-out test.
- [evidence/T10_node_type_pilot.md](evidence/T10_node_type_pilot.md) — node-type head and soft degree decoding.
- [evidence/focal_ixi_300ep/README.md](evidence/focal_ixi_300ep/README.md) — 300-epoch focal ablation, test split.
- [evidence/degree_consistency/README.md](evidence/degree_consistency/README.md) — degree-consistency run (interim).
- [evidence/ixi_query_benchmark/README.md](evidence/ixi_query_benchmark/README.md), [RESULTS.md](evidence/ixi_query_benchmark/RESULTS.md) — 120 vs 192 query cost.
- Diagnostic evidence folders:
  [ixi_features_nodes_20261018](evidence/ixi_features_nodes_20261018/results.md),
  [ixi_node_refinement_20261018](evidence/ixi_node_refinement_20261018/comparison.md),
  [ixi_relation_isolation_20261019](evidence/ixi_relation_isolation_20261019/results.md),
  [ixi_threshold_calibration_20261019](evidence/ixi_threshold_calibration_20261019/results.md),
  [ixi_query_attribution_20261019](evidence/ixi_query_attribution_20261019/README.md),
  [ixi_mechanisms_20261019](evidence/ixi_mechanisms_20261019/README.md) ([results](evidence/ixi_mechanisms_20261019/results.md)),
  [ixi_relation_matching_calibration_20261019](evidence/ixi_relation_matching_calibration_20261019/results.md),
  [ixi_image_relation_learning_20261020](evidence/ixi_image_relation_learning_20261020/results.md),
  [ixi_node_focal_20261020](evidence/ixi_node_focal_20261020/results.md),
  [ixi_segmentation_input_20261020](evidence/ixi_segmentation_input_20261020/results.md),
  [job_review_20261020](evidence/job_review_20261020/review.md) (20 Oct job/status snapshot).

Reference docs (not duplicated here): [`../MODEL.md`](../MODEL.md), [`../TRAINING.md`](../TRAINING.md),
[`../LOSSES.md`](../LOSSES.md), [`../EVALUATION.md`](../EVALUATION.md).

[exp]: ../archive/legacy-20261103/docs/EXPERIMENTS.md
[rev]: ../archive/legacy-20261103/docs/research/PROJECT_REVIEW_20261019.md
[road]: ../archive/legacy-20261103/docs/research/IXI_graph_extraction_diagnostic_roadmap.md
[qb]: evidence/ixi_query_benchmark/RESULTS.md
[r1]: ../archive/legacy-20261103/docs/research/metrics/1-ixi-supervision-audit.md
[r2]: ../archive/legacy-20261103/docs/research/metrics/2-ixi-feature-node-diagnostics.md
[r3]: ../archive/legacy-20261103/docs/research/metrics/3-ixi-node-refinement-pilot.md
[r4]: ../archive/legacy-20261103/docs/research/metrics/4-ixi-relation-isolation.md
[r5]: ../archive/legacy-20261103/docs/research/metrics/5-ixi-threshold-calibration.md
[r6]: ../archive/legacy-20261103/docs/research/metrics/6-ixi-query-attribution.md
[r7]: ../archive/legacy-20261103/docs/research/metrics/7-ixi-mechanism-batch.md
[r8]: ../archive/legacy-20261103/docs/research/metrics/8-ixi-relation-matching-calibration.md
[r9]: ../archive/legacy-20261103/docs/research/metrics/9-ixi-image-aware-relation-learning.md
[r10]: ../archive/legacy-20261103/docs/research/metrics/10-ixi-node-focal-ablation.md
[r11]: ../archive/legacy-20261103/docs/research/metrics/11-ixi-segmentation-input-oracle.md
