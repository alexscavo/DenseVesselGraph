# Evaluation metrics — consolidated history (to 29 Oct 2026)

> **Status.** Part 1 defines the metrics that the current documents report. Part 2 covers how
> the metric suite was designed, validated and used from Sept to mid-Oct 2026. Written on
> 3 Nov 2026 as a consolidation of archived documents (listed under Sources). The models these
> metrics were first applied to are superseded by the
> [dense traced model](dense_traced_model_20261102.md). The metric definitions in Part 1 are
> still current.

## Context

The question was how to score a predicted vessel graph so that three things are measured
separately: topology (which vessel events are connected), geometry (where the vessels run) and
representation size. The RelationFormer/GrEVe evaluator that the project inherited scores
nodes and edges as boxes (COCO-style mAP), plus the street mover distance (SMD). Neither
checks whether an edge connects the right two anatomical events. In September 2026 a metric
plan was written for 2D and 3D, at patch level and whole-volume level. It was implemented in
stages: a physical Branch F1 (T07), synthetic failure checks (T08), the `shared-guides-v2`
common protocol, and on 13 Oct the broader vessel-core v1 suite. Most of the Oct results here
come from 1,952 SyntheticMRI validation patches, scored with saved predictions of two
scratch-trained RelationFormer models (straight edges vs detached B-spline edges). Since
30 Oct, every network has been scored by the single evaluator of the dense-model scripts
(Part 1).

## Shorthand

- **GrEVe.** The earlier RelationFormer-based vessel-graph work that this project extends.
  Its evaluator defines the **native** (legacy) metrics: box mAP/mAR, box-based F1, SMD, β.
- **Physical metrics.** Scores computed in millimetres after mapping normalized patch
  coordinates through the cached per-patch `metric` matrix.
- **common-v2.** The `shared-guides-v2` protocol in
  [`metrics/common_vessel_metrics.py`](../../metrics/common_vessel_metrics.py). It gives
  Branch F1, Curve F1, anchor F1 and guide distance against the original dense centrelines
  ("guides").
- **vessel-core v1.** The broader suite of 13 Oct 2026
  ([`metrics/vessel_suite.py`](../../metrics/vessel_suite.py),
  [`scripts/evaluate_vessel_core.py`](../../scripts/evaluate_vessel_core.py)). It adds
  Curve F1 at three tolerances, ACD, HD95, APLS-3D, TOPO-3D, role F1 and payload size.
- **T05–T08.** September tracker tasks: T05 representation geometry metrics, T06
  rate-distortion study, T07 degree-2 contraction and physical Branch F1, T08 controlled
  metric-failure checks.
- **Anchor.** A topological node, i.e. one whose degree is not 2: an endpoint (degree 1), a
  junction (degree ≥ 3) or an isolate (degree 0).
- **Degree-2 contraction.** Merging every maximal chain of degree-2 nodes into one branch
  between anchors. The chain's ordered points are kept as the branch polyline.
- **ACD.** Average centreline distance: the mean of the two directed mean distances between
  sampled curves, in mm.
- **HD95 / APLS / TOPO / GOSPA.** 95th-percentile Hausdorff distance; average path length
  similarity (SpaceNet); reachable-neighbourhood topology score (Sat2Graph); graph generalized
  optimal sub-pattern assignment metric.
- **MM / MU / UU.** A false edge between two matched nodes / one matched and one unmatched
  node / two unmatched nodes.
- **`monitor_train`.** The fixed 256 training patches on which current thresholds are
  selected.
- **K8.** A cubic B-spline edge with 8 controls (6 interior).
- **B0.** The baseline RelationFormer recipe of the current docs (no node refinement, K8
  spline head).
- **Straight / detached B-spline (Oct).** The two final epoch-300 scratch RelationFormer
  models, `straight_final` and `bspline_detached_final`. In the detached model, geometry
  gradients do not reach the detector.

---

# Part 1 — Metrics actually in use

These are the definitions behind the tables in
[`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) and
[`network_proposals_20261030.md`](network_proposals_20261030.md) (glossary, lines 41-153).
Every network, including converted RelationFormer predictions, goes through the same
`evaluate` function of the dense scripts. I checked tolerances against the code cited for each
metric.

### Node and edge detection (physical, 2 mm)

- **Node F1 @2 mm.** Predicted and GT nodes are matched one-to-one within 2 mm by
  `match_nodes` ([`scripts/diagnose_branch_relations.py`](../../scripts/diagnose_branch_relations.py)):
  a Hungarian assignment in which pairs beyond 2 mm get a prohibitive cost, so it maximizes
  the number of matches first, then minimizes distance. TP/FP/FN are summed over patches.
- **Edge F1.** Computed by `edge_accounting`
  ([`scripts/ixi_query_diagnostics.py`](../../scripts/ixi_query_diagnostics.py)):
  a retained predicted edge is TP iff both endpoints are matched at 2 mm and the two matched
  GT nodes share a GT edge; FN = GT edges − TP. It also reports the fraction of missed GT
  edges that lack a matched endpoint pair ("missing endpoint"). It needs a simple GT graph
  (closed or parallel GT edges raise an error).
- **MM / MU / UU.** The three classes of false edge, by how many of their endpoints are
  matched. The class shows whether a false edge is a relation error (MM) or comes from a
  detection error (MU, UU).
- **Thresholds.** Node then edge confidence thresholds maximize node F1 and then edge F1 on
  `monitor_train`. Validation is also reported at a fixed 0.5/0.5. Validation never selects
  anything.

### common-v2 Branch F1 and Curve F1

These are computed by `evaluate_common_vessels` / `aggregate_common_vessels`, protocol string
`shared-guides-v2`.

- **Preparation.** Prediction and GT are contracted at degree 2, keeping the geometry
  (chords for straight graphs, 257 samples per spline). The reference geometry is the original
  dense paths (`full_paths`), not the fitted spline targets.
- **Anchor matching.** One-to-one within **1 mm**.
- **Branch F1.** A predicted branch is TP when both its anchors are matched and GT has an
  unused branch between the two matched anchors (multiplicity counts). Branch geometry is
  deliberately ignored and never chooses the correspondence. Isolated predicted nodes are not
  branches. Dataset F1 = 2TP/(2TP+FP+FN), pooled over patches; two empty graphs score 1.
- **Curve F1.** All predicted and GT branches are sampled every **0.25 mm** of arc length. A
  sample is supported if it lies within **1 mm** of the other graph's continuous polylines.
  Precision and recall are pooled sample fractions. Missing and extra geometry both count; no
  branch match is needed.
- **Anchor F1.** Anchors, isolates included, are matched at 1 mm.
- **Guide distance and tortuosity error.** Computed only on topology-matched branches (the
  guide distance is the symmetric mean point-to-polyline distance), so report them with
  branch coverage, TP/(TP+FN).
- The glossary short form, "Curve F1 counts centreline samples within 1 mm of the other
  graph", matches the code.

### β₀ / β₁ errors and loops

- β₀ = connected components, β₁ = independent cycles = E − V + β₀.
- These are computed on the **decoded native graph** (all retained nodes, isolates included)
  by `betti` in [`scripts/ixi_query_diagnostics.py`](../../scripts/ixi_query_diagnostics.py),
  as used by [`scripts/topology_metrics_20261102.py`](../../scripts/topology_metrics_20261102.py).
- The reported value is the per-patch mean absolute error against GT. "Loops" is the mean
  predicted β₁ per patch (GT value in brackets).
- Unconnected detected nodes therefore raise the β₀ error. That is why the BRAVA dense models
  show large β₀ errors.
- Equal Betti numbers do not imply correct adjacency (T08 below).

### SMD (street mover distance)

- Native RelationFormer protocol (`street_mover_distance` in
  [`training/evaluation/metrics.py`](../../training/evaluation/metrics.py)):
  100 points per graph, Sinkhorn ε = 1e-7 with 100 iterations, **normalized patch
  coordinates** (not mm), and **straight segments between nodes**, so predicted curves are
  ignored.
- It is undefined, and skipped, when either graph has no edges. The number of skipped patches
  differs between arms (BRAVA 78–236), so BRAVA SMD needs care
  ([source](dense_traced_model_20261102.md)).
- It is kept only for comparability with GrEVe/RelationFormer (see §2.1).

### Mean symmetric distance (oracle geometry)

- This is the per-edge geometry score of the oracle-topology study
  ([`scripts/oracle_geometry_20261030.py`](../../scripts/oracle_geometry_20261030.py)).
- Both the predicted curve and the dense path are densely sampled (256 points). The score is
  the mean of the two directed mean nearest-point distances, in mm. Hausdorff and coverage
  at 0.5/1 mm are also computed.
- The GT endpoints and edges are given, so it measures edge shape only. Compare it with the
  conditional common-v2 guide distance, not with whole-graph Curve F1.

### Average precision

The definitions follow
[`scripts/dense_ap_20261030.py`](../../scripts/dense_ap_20261030.py) and
[network_proposals §6.5](network_proposals_20261030.md).

- **Node AP @2 mm.** All proposals are ranked by score and matched greedily to GT within
  2 mm. Proposals are every peak ≥ 0.01 for the dense model, and all 256 queries for
  RelationFormer.
- **Edge ranking AP.** AP over candidate pairs among the nodes retained at the selected
  threshold. It is reported for all retained pairs, and for pairs with both endpoints matched.
- **Box node/edge mAP/mAR.** The GrEVe/RelationFormer metric: IoU 0.50:0.05:0.95, GT node
  boxes of side 0.2, edge boxes spanning the end nodes with half-width 0.1, at most
  **40 detections** (as coded in `dense_ap_20261030.py`). Arms without a box head use fixed
  side-0.2 boxes.
- AP was not computed for the traced model.

### Long-edge recovery

- The fraction of GT edges recovered (Edge F1 rule), by GT arc-length bin (0–10, 20–40,
  ≥ 40 mm) ([dense model §4.4](dense_traced_model_20261102.md)).
- It descends from the October branch-length diagnostic (§2.5).

### Aggregation and uncertainty

- Dataset F1 values pool TP/FP/FN (or curve samples) over patches. They never average
  per-patch F1.
- Differences are reported with subject-paired bootstrap 95% CIs over validation subjects.
  These do not cover training-seed variation. All current results are single-seed
  validation results.

### Name mapping (verified)

| Earlier name | What it is now |
|---|---|
| "common" / "common-vessel" metrics, `shared-guides-v2`, `common-v2-summary.json` | common-v2, unchanged. It was introduced for the 300-epoch scratch straight-vs-adaptive comparison and is still used for Branch/Curve F1 |
| Branch Connectivity F1 (Sept plan), physical Branch F1 (T07) | common-v2 Branch F1. The rule is the same; T07's native-evaluator version used `evaluation.protocol.branch_threshold_mm` (1.0 mm on IXI) |
| Curve P/R/F1 (plan), "whole-vessel curve F1" | common-v2 Curve F1 at 1 mm |
| vessel-core v1 | **Not** a renaming of common-v2. It is a later superset whose Curve F1 @1 mm and Branch F1 reproduce common-v2 exactly ([source](../archive/legacy-20261103/docs/research/metrics/core-metrics-results.md)). Its extra metrics (APLS-3D, TOPO-3D, ACD, HD95, payload) are not used in the current docs |
| "physical node/edge F1 at 1/2 mm" (Oct pilots) | Node F1 / Edge F1 @2 mm (the 1 mm variant is no longer reported) |
| Native box node/edge F1 (`evaluate.py`) | Not current Edge F1. It measures edge-box overlap, not endpoint-correct adjacency |

### Relation to `docs/EVALUATION.md`

[`../EVALUATION.md`](../EVALUATION.md) documents only the native `evaluate.py` metrics: box
mAP/mAR (40-detection cap), box-based node/edge F1, SMD and β counts. It does not describe
the physical 2 mm Node/Edge F1, MM/MU/UU, common-v2, Node AP @2 mm or the dense-model
evaluator. It is therefore incomplete for the metrics the current docs report. This document is the
pointer for those definitions.

---

# Part 2 — Metric design history and key findings

## 2.1 The September metric plan [DONE as a plan; partly implemented]

The plan was written before any new metric existed
([source](../archive/legacy-20261103/docs/research/vascular_graph_metrics.md)). Its main
decisions:

- **Three evaluation classes.**
  1. Representation quality: the adaptive target vs the dense reference, with no network.
  2. Model quality: the prediction vs the exact target it was trained on. This is the main
     model score; a perfect model can reach 1.
  3. End-to-end fidelity: the prediction vs the dense reference, always shown next to the
     class-1 oracle, because final error ≈ representation error + prediction error.
- **Topology and geometry stay separate.** A graph can be topologically right and
  geometrically wrong, or the reverse. Neither family substitutes for the other.
- **Node roles come from degree.** No node-type head is needed. Degree-2 nodes are geometric
  control points, not anatomical events.
- **Degree-2 contraction for topology,** keeping the polyline for geometry. Topology scores
  become invariant to arbitrary subdivision.
- **Spatially grounded topology is primary.** Anchors must first match within a physical
  tolerance. A structural diagnostic, in which a misplaced node with correct abstract
  connectivity still counts, was left as secondary (a candidate use for Graph GOSPA).
- **Branch Connectivity F1** is the core topology metric. A GT branch A–B is TP when the
  matched anchors A′–B′ are joined by a contracted predicted branch. It deliberately does not
  require geometric overlap.
- **Betti errors** are global diagnostics only, because very different graphs can share
  Betti numbers.
- **3D TOPO deferred.** The common implementation is 2D-specific.
- **Geometry.** Core: Curve P/R/F1 at several physical tolerances, and ACD, with distances to
  continuous segments so compact graphs are not penalized for having fewer vertices. HD95 is
  a tail-error diagnostic.
- **Tolerance policy.** Physical mm, not voxels. Never adapt the tolerance to foreground
  density (stratify by density instead). A local-radius-normalized score is a possible future
  complement, but never a dataset-mean radius.
- **Morphology and size.**
  - Branch-wise length, tortuosity and branch-angle errors are diagnostics.
  - Total node count |V| is the complexity measure, optionally per mm of vessel. |E| is
    descriptive.
  - Representation quality is framed as rate-distortion: |V| vs ACD or Curve F1, for
    junction-only, adaptive and dense targets.
- **SMD is demoted to legacy.** Subdivision barely changes it, but it ignores branch identity.
  Redundant nearby or crossing edges can lower SMD, so lower SMD does not mean a better graph.
- **Legacy node/edge box mAP/mAR** are kept for GrEVe/RelationFormer continuity, but only as
  detection metrics.
- **APLS** is useful at patch level, as local path consistency truncated by the crop, but
  more important at whole-volume level.
- **Graph GOSPA** is experimental. It would decompose error into localization, missed/false
  nodes and edge mismatch, and would be kept only if it adds information beyond Branch F1.
- **Metrics are not losses.** They are non-differentiable. First change the target
  representation under a fixed loss, then add surrogates only for a diagnosed failure.
- **Implementation phases:** (1) representation metrics, (2) contraction and Branch F1,
  (3) APLS / GOSPA / 3D TOPO, (4) whole-volume evaluation after stitching.
- The same file ends with an FGW-matcher conclusion and proposals for a degree- or role-aware
  Hungarian matcher. Those are matcher topics, not metrics, and are not summarised
  here.

## 2.2 What was implemented, and what was not

| Plan item | Outcome |
|---|---|
| Degree-2 contraction, anchors, Branch F1 | **[DONE]** T07, then common-v2 and vessel-core v1 |
| Curve P/R/F1 | **[DONE]** common-v2 at 1 mm; vessel-core v1 at 0.5/1/2 mm |
| ACD (unsquared Chamfer/ASSD-style centreline distance) | **[DONE]** in vessel-core v1. Not reported now; the current docs use oracle mean symmetric distance and the common-v2 guide distance instead |
| HD95 | **[DONE]** in vessel-core v1 (mean per-patch). Not reported now |
| Betti errors | **[DONE]** and still reported |
| APLS-3D, TOPO-3D | **[DONE]** in vessel-core v1 as explicit 3D adaptations (below). Used only in the Oct comparisons |
| Junction / endpoint / anchor F1 | **[DONE]** in vessel-core v1; common-v2 anchor F1 |
| Representation geometry, morphology, complexity; rate-distortion | **[DONE]** as T05/T06 (30-volume study; covered in other history docs) |
| Native payload size (bytes, internal controls) | **[DONE]** in vessel-core v1 |
| Graph GOSPA | **[ABANDONED: never implemented]**. No code exists; the plan made it conditional on the simpler suite leaving a gap, and no later doc took it up |
| Branch-angle error | **[PROPOSED, never run]** |
| Curve-aware SMD | **[ABANDONED: deferred]**. SMD kept only as labelled chord/normalized-space legacy |
| clDice / foreground-mask support | **[PROPOSED, never run]**. Needs a declared curve-to-mask/radius convention |
| Curvature error, surface ASSD | **[PROPOSED, never run]**. Guides are noisy; surfaces need radii |
| Graph edit distance, CRR | **[PROPOSED, never run]** |
| Inference time, peak memory | **[PROPOSED, never run]**. Needs timed inference, not saved predictions |
| VTK/SWC/raster skeleton import, external centreline tools | **[PROPOSED, never run]**. The array/polyline interface is ready |
| Tolerance / radius / sampling-budget calibration | **[PROPOSED, never run]**. Fixed diagnostic defaults are used |
| Whole-volume evaluation | **[PROPOSED, never run]** in these sources |

Sources: [implementation status](../archive/legacy-20261103/docs/research/metrics/implementation-status.md),
[protocol](../archive/legacy-20261103/docs/research/metrics/vessel-core-v1.md).

A separate literature shortlist
([source](../archive/legacy-20261103/docs/research/metrics/metrics_ideas.md)) ranked
candidate metrics for the straight-vs-spline journal extension:

- Priority list: Curve F1, Chamfer, TOPO, APLS, edge P/R, junction matching, object count
  and representation size.
- Supporting metrics: SMD, clDice, Betti and tortuosity.

It fed vessel-core v1. Its rule "same physical geometry and matching for all
representations, native objects counted separately" became a design principle.

## 2.3 Controlled metric-failure checks (T07, T08) [DONE]

These test the metrics, not any model or extractor.

- **T07, Branch F1** ([evidence/T07/README.md](evidence/T07/README.md)).
  - On two real saved full-volume graph families, adaptive vs dense scores Branch F1 = 1.0.
    Deleting one real edge lowers it, only slightly at whole-volume scale.
  - On two real IXI validation patches, self-comparison gives 1.000. Removing one GT edge
    gives **0.857** and **0.783**.
  - Rules fixed here:
    - inputs are in world mm, reconstructed from the source affine, because the patch NIfTI
      headers are identity and must not be used for spacing;
    - micro aggregation, because two empty graphs score 1 per patch;
    - isolated predicted nodes do not affect Branch F1, so they must be checked by
      node/count metrics;
    - direct self-loops are rejected.
- **T08, failure matrix** ([evidence/T08/README.md](evidence/T08/README.md)).
  - A false cross-link between two close separate vessels: Branch F1 1.000 → 0.000, edge mAP
    1.000 → 0.835, node mAP unchanged, SMD 0 → 0.000031 (barely moves).
  - A bent interior path with the same two anchors: Branch F1 unchanged at 1.000, edge mAP
    1.000 → 0.100, SMD 0 → 0.002276.
  - SMD here is in normalized patch coordinates, not mm.
  - Node mAP detects neither failure. β₁ error is 0 in both cases, and other tests show that
    equal Betti numbers can hide a wrong branch pairing.
  - Conclusion: topology (Branch F1) and geometry (curve/edge metrics) are complementary, as
    the plan required.
  - The same note warns that AP/AR depend on the detection cap. The IXI 256-query recipe used
    a 256 cap, and the old 40 cap must not be a headline score on dense vessel patches.

## 2.4 vessel-core v1 validation comparison (13 Oct) [DONE; superseded by the dense traced model]

The suite is a common physical interface for straight graphs, curves and centreline graphs
([protocol](../archive/legacy-20261103/docs/research/metrics/vessel-core-v1.md)).

- **Fixed parameters.**
  - Geometry sampling 0.25 mm; curve tolerances 0.5/1/2 mm; anchor tolerance 1 mm.
  - Route sampling 0.5 mm, snapping 1 mm.
  - TOPO radius 10 mm of graph distance, at most 32 seeds per direction; at most 64 APLS
    controls.
  - These were chosen before evaluation. They are diagnostic defaults, not calibrated.
- **APLS-3D.**
  - Shortest paths are weighted by arc length, and controls are snapped to the other graph
    within 1 mm.
  - Each pair scores min(1, |Δlength|/length). Missing routes score error 1.
  - Both directions are scored and combined by harmonic mean.
- **TOPO-3D.**
  - Nodes reachable within 10 mm of graph distance are matched one-to-one at 1 mm.
  - There is no angle gate.
  - Both APLS-3D and TOPO-3D are explicit adaptations, not reproductions of the 2D road
    benchmarks.
- Geometry-only inputs report topology as unavailable, never as perfect.

Results on all 1,952 SyntheticMRI validation patches, from saved epoch-300 predictions. This
is validation, single-seed, scored against uncertain guides
([source](../archive/legacy-20261103/docs/research/metrics/core-metrics-results.md),
[CSV](../archive/legacy-20261103/docs/research/metrics/core-metrics-results.csv)):

| Metric | Straight | Detached B-spline |
|---|---:|---:|
| Curve F1 0.5 / 1 / 2 mm | 0.0642 / 0.2020 / 0.4490 | 0.1533 / 0.3426 / 0.5539 |
| ACD / HD95 (mm) | 4.092 / 17.069 | 3.581 / 16.776 |
| TOPO-3D F1 / APLS-3D | 0.0910 / 0.0390 | 0.1860 / 0.1042 |
| Junction / Branch F1 | 0.0220 / 0.0079 | 0.0294 / 0.0118 |
| β₀ / β₁ error | 2.661 / 2.268 | 2.702 / 2.133 |
| Payload bytes | 386.25 | 1571.07 |

- **What the comparison showed.**
  - Splines improved overlap and the adapted route scores.
  - Exact branch and junction recovery stayed very low for both models.
- **Curvature, not adjacency.** Using chords with the spline model's own endpoints gives Curve
  F1 @1 mm **0.2263**, against **0.3426** with its curves. So the gain is overlap from
  curvature, not better adjacency.
- **Different populations.** The global ACD (4.09 vs 3.58 mm) covers the whole decoded graph.
  The earlier common-v2 matched-branch distances (about 0.660 vs 0.460 mm) cover only matched
  branches.
- **Payload.** The spline payload is about 4.07× the straight payload at similar node/edge
  counts. No efficiency advantage was shown.
- This finding (splines change curve quality, not the graph) is consistent with the later
  RelationFormer results in the current docs: "Training RelationFormer with or without
  splines gives identical graphs; only curve quality differs".

## 2.5 Branch fragmentation and relation-distance diagnostic (13 Oct) [DONE]

The user had seen branches split into short edges while nearby different vessels were joined.
The diagnostic used the same two models and 1,952 patches (31,934 GT branches), with nodes
matched one-to-one at 1 and 2 mm
([source](../archive/legacy-20261103/docs/research/metrics/branch-connectivity-diagnostic.md)).

- **Long relations are missed even when both endpoints are found.** Conditional direct-edge
  recall at 2 mm:
  - shortest GT arc-length quartile (< 4.97 mm): 92.9% straight, 93.8% spline;
  - longest quartile (≥ 17.71 mm): 38.8% straight, 42.8% spline.
- **Nearby non-adjacent pairs are wrongly connected.** Among matched non-adjacent pairs closer
  than 4.90 mm, 71.9% (spline) and 69.0% (straight) are connected. At ≥ 17.19 mm apart, only
  0.60% and 0.57% are.
- **Localization dominates.**
  - Only 13.7% / 16.9% of GT branches have both endpoints found at 2 mm.
  - For the spline model, 1,131 of the 1,725 missed direct branches have no connecting
    predicted path at all.
  - Degree-2 chains explain only 244 of them, so contracting chains cannot fix most misses.
- **Total object counts are close to GT** (spline 42,388 nodes vs 41,646 GT). Reducing edge or
  node counts uniformly would not help.
- **Conclusion at the time.** There is a short-relation bias in the decoded output, not a
  proven cause. Length is confounded with chord distance and image difficulty.
  - The relation MLP sees no image evidence along the path.
  - A blanket length reward, a forced spanning tree or a threshold change was judged
    unjustified.
- **Later explanation.** The current docs name "edges are scored without ever looking at the
  image between the two nodes" as one RelationFormer failure. The dense model scores edges from
  image features along the path, and its traced candidates raise IXI recovery of ≥ 40 mm GT
  edges from 0.01 (dense 16-NN) to 0.15 ([source](dense_traced_model_20261102.md)).

## 2.6 Relation-distance pilot (13–14 Oct) [DONE — negative]

Four 1,000-update continuations of the final detached B-spline checkpoint
([source](../archive/legacy-20261103/docs/research/metrics/relation-distance-pilot.md)):

- **Arms.** Relation-head-only, or relation + node classification/coordinate heads. Each was
  run with original weights or with a 5× weight on non-edges whose GT endpoints are < 5 mm
  apart (negative mass renormalized).
- **Frozen parts.** Backbone, transformer, BN statistics and spline head.
- **Evaluation.** Full validation, compared with the saved starting checkpoint.

| Model | Curve F1 1 mm | TOPO-3D | APLS-3D | Long-branch recall (2 mm) | Near non-edge FPR (2 mm) |
|---|---:|---:|---:|---:|---:|
| starting checkpoint | 0.3426 | 0.1860 | 0.1042 | 42.8% | 71.9% |
| relation_control | 0.3336 | 0.1806 | 0.0990 | 39.3% | 73.7% |
| relation_near | 0.3351 | 0.1826 | 0.1007 | 41.1% | 66.1% |

- **Near-negative weighting has a modest specific effect.** Near FPR falls from 73.7% to 66.1%
  at fixed nodes. But no arm beats the starting checkpoint on Curve F1, TOPO or APLS.
- **β trade-off.** Weighting lowers β₁ error but raises β₀ error (3.32 vs 3.02): it removes
  links and leaves fragments.
- **Node-head adaptation barely moves recall:** 45.27% → 45.37% at 2 mm.
- **Large generalization gap.** On 512 training patches, node recall is 97.81% / 98.77% at
  1 / 2 mm, against 18.75% / 45.27% on validation. With weighting, training near-non-edge FPR falls from 0.66%
  to 0.22%, while validation stays at 66.1%.
- **Decision.** Do not extend these pilots; investigate transfer instead.
- **Caveats.** Single seed. The 5 mm and 5× settings were validation-informed choices, not
  tuned.

## 2.7 Generalization audit (14 Oct) [DONE]

([source](../archive/legacy-20261103/docs/research/metrics/generalization-audit.md))

- **The gap is not late collapse.** Re-scoring the earlier best checkpoints (straight epoch
  255, spline epoch 170) shows the gap was already there. Spline validation node recall at
  2 mm is 44.54% at the best checkpoint and 45.27% at the final one. Overfitting before epoch
  170 is not excluded.
- **No coarse split shift.** On 512 patches per split, intensity, foreground fraction, node
  and edge counts, boundary fraction, branch length and tortuosity are all near-identical.
  For example, the foreground fraction is 0.01275 (train) vs 0.01301 (validation).
- **No leakage found.** There are no overlapping source IDs and no identical sampled images.
  This was not an exhaustive leakage check.
- **Scale-check bug.** A first scale check falsely reported scale errors because it compared
  the signed source affine with the identity patch header. The corrected v2 audit finds zero
  scale error.
- **Missing augmentation.** The spline loader then disabled augmentation, which was flagged as
  a concrete missing regularizer, not a proven cause. The follow-up (plant pretraining
  recovery) belongs to the pretraining history.
- **Not comparable to the current numbers.** The current docs report much higher SyntheticMRI
  validation node F1 for RelationFormer (B0 0.794 @2 mm) under the 30 Oct cohort and recipe.
  So the ~45% recall above is specific to these Oct scratch models, and the two are not
  directly comparable.

## 2.8 Metric ideas still open [PROPOSED, never run]

- A structural, non-spatial topology diagnostic, e.g. Graph GOSPA, to separate a misplaced
  junction from wrong connectivity.
- Whole-volume Branch F1, Curve F1, APLS and Betti after stitching patches, with crop-boundary
  subdivisions removed.
- Local-radius-normalized tolerances, stratification by density, and sensitivity of the fixed
  tolerances, TOPO radius and APLS/TOPO seed budgets.
- clDice, curvature and branch-angle errors, graph edit distance, CRR, and curve-aware SMD.
- External centreline-tool baselines through the polyline interface (needs file importers).
- A held-out test evaluation, a second seed and seed-level intervals for any metric claim.

## What carried forward / what was dropped

**Carried forward**

- Physical-mm one-to-one node matching, and endpoint-correct Edge F1.
- Degree-2 contraction with geometry kept, as common-v2 Branch F1 and Curve F1 @1 mm.
- Topology and geometry kept separate: report Branch F1 and Curve F1 together.
- Pooled micro-F1 and subject-paired CIs.
- β errors as diagnostics, and SMD only as a legacy score, with its skipped-patch caveat.
- The October finding that splines improve overlap but not adjacency, and the length-binned
  edge recovery diagnostic. Both reappear in the current dense-model docs.

**Dropped or unused**

- APLS-3D, TOPO-3D, HD95, ACD and payload bytes. They were implemented in vessel-core v1, but
  no current table uses them.
- Graph GOSPA and the 3D structural diagnostic were never built.
- Representation-class headline tables (Table A/B/C of the plan) were never produced as such
  for the current model.
- Near-negative relation weighting as an intervention.
- The 40-detection box AP cap was flagged as unsuitable for dense patches (T08), but the
  current box mAP still uses it.

## Sources

Archived (post-move locations):

- [vascular_graph_metrics.md](../archive/legacy-20261103/docs/research/vascular_graph_metrics.md):
  the September metric plan (three classes, contraction, Branch F1, Betti, TOPO, Curve F1,
  ACD, HD95, APLS, GOSPA, SMD as legacy), plus FGW and structure-aware matcher notes.
- [metrics/README.md](../archive/legacy-20261103/docs/research/metrics/README.md): index of
  the metric reports and of the IXI diagnostic series 1–11 (covered in
  `history_relationformer_3d.md`).
- [metrics/metrics_ideas.md](../archive/legacy-20261103/docs/research/metrics/metrics_ideas.md):
  literature shortlist of geometry, topology, efficiency and detection metrics, with
  relevance scores.
- [metrics/vessel-core-v1.md](../archive/legacy-20261103/docs/research/metrics/vessel-core-v1.md):
  vessel-core v1 protocol, its interface and its fixed parameters.
- [metrics/implementation-status.md](../archive/legacy-20261103/docs/research/metrics/implementation-status.md):
  implemented, deferred and pending metric items as of 14 Oct.
- [metrics/core-metrics-results.md](../archive/legacy-20261103/docs/research/metrics/core-metrics-results.md)
  and [core-metrics-results.csv](../archive/legacy-20261103/docs/research/metrics/core-metrics-results.csv):
  paired 1,952-patch straight vs detached B-spline comparison.
- [metrics/branch-connectivity-diagnostic.md](../archive/legacy-20261103/docs/research/metrics/branch-connectivity-diagnostic.md):
  relation recall by branch length, nearby false links, fragmentation.
- [metrics/relation-distance-pilot.md](../archive/legacy-20261103/docs/research/metrics/relation-distance-pilot.md):
  four-arm near-negative weighting / node-head pilot.
- [metrics/generalization-audit.md](../archive/legacy-20261103/docs/research/metrics/generalization-audit.md):
  best vs final checkpoints and the train/validation split audit.

In place:

- [evidence/T07/README.md](evidence/T07/README.md): Branch F1 real-graph and IXI-patch
  edge-deletion checks.
- [evidence/T08/README.md](evidence/T08/README.md): controlled metric-failure matrix.
- [../EVALUATION.md](../EVALUATION.md): reference for the native `evaluate.py` metrics.
- [dense_traced_model_20261102.md](dense_traced_model_20261102.md) and
  [network_proposals_20261030.md](network_proposals_20261030.md): the current results and
  their metric glossary.
