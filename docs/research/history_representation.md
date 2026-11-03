# Graph representation choices — consolidated history (to 29 Oct 2026)

> **Status.** Covers what graph the model should be asked to predict: junction-only vs adaptive
> vs dense graphs, the adaptive simplification policy, off-vessel chords, object-query capacity,
> and how much spline capacity a curved-edge target needs (Sept – 22 Oct 2026). Consolidated on
> 3 Nov 2026 from documents now archived under `docs/archive/legacy-20261103/`. The RelationFormer
> setting it was written for is superseded by the
> [dense traced model](dense_traced_model_20261102.md), which uses the "topology + curved edge"
> form that these studies argued for. Extraction and dataset production are in
> [`history_data.md`](history_data.md). Learned spline-head experiments are in
> [`history_continuous_edges.md`](history_continuous_edges.md).

## Context

RelationFormer predicts a graph from a fixed set of object queries, one per node. If every
centreline sample is a node, a 64³ patch needs hundreds of queries. If only junctions and ends are
nodes, straight edges cut across curved vessels and leave the lumen. The guiding idea
([representation note](../archive/legacy-20261103/docs/research/vascular_graph_representation.md))
was that **a vascular graph = topology + edge geometry**:
- junctions, endpoints and border crossings are topological events;
- degree-2 points are only geometric control points.

Two parameterisations of the same embedded graph were considered:
- **Approach A:** keep RelationFormer and add only the degree-2 nodes needed to stay within an
  error bound;
- **Approach B:** keep only topological nodes and give each edge a continuous curve
  (Bézier/B-spline), possibly later with a radius function r(t).

Data: 220 IXI/TopBrain full volumes and their 64³ patches; BRAVA, IXI and SyntheticMRI patches
for the spline studies. Distances are in physical mm. Every comparison here is against the
*extracted* dense centreline, not anatomical ground truth.

## Shorthand

- **Dense.** The ordered centreline after topology cleanup but before smoothing and
  simplification, at voxel centres (D04).
- **Adaptive.** Endpoints, junctions and only the degree-2 points needed by the simplification
  policy; edges are straight. This was the training target.
- **Junction-only.** Endpoints and junctions only, one straight edge per branch.
- **D01–D08.** Decision-log entries of the implementation tracker (listed in §1).
- **T01–T10.** Tracker tasks:
  - T01 rings, T03 three products, T04 dense reference;
  - T05/T06 representation study, T07 Branch F1;
  - T09 300-epoch IXI run, T10 query benchmark.
- **RDP.** Ramer–Douglas–Peucker polyline simplification (greedy).
- **Chord containment.** Fraction of a straight edge's samples that lie inside the segmentation.
- **Branch-paired Curve F1 / ACD / HD95.** Curve F1 at a mm tolerance, average centreline
  distance and 95th-percentile Hausdorff distance, with each branch compared only to its
  counterpart. This differs from the network-wide common-v2 Curve F1 of the
  [glossary](network_proposals_20261030.md).
- **Branch F1.** Branch connectivity F1 after degree-2 contraction (glossary).
- **K4…K24.** Endpoint-constrained clamped cubic B-spline with K controls. The two end controls
  are the nodes. K8 was the default spline target.
- **Oracle adaptive K.** The smallest K that meets a tolerance on the GT path.
- **Piecewise inside edge / split graph.** The same error-driven subsegments, stored either as
  internal joins of one edge or as extra degree-2 graph nodes.
- **β₀ / β₁.** Number of components / independent cycles.

## 1. Decision log D01–D08 (representation view)

| ID | Date | Decision | Status at last record |
|---|---|---|---|
| D01 | 18 Sept | A pure degree-2 ring becomes 3 deterministic nodes and 3 arcs (a self-loop or two parallel edges do not fit a simple adjacency matrix) | DONE (T01) |
| D02 | 18 Sept | Emit `junction_only`, `adaptive`, `dense` as separate products; patches default to adaptive | DONE (T03) |
| D03 | 18 Sept | Extraction rules must use physical or locally normalised units; voxel constants are provisional | Active constraint; tested on sampled strata only |
| D04 | 18 Sept | Dense = before smoothing and simplification, so smoothing distortion stays visible | DONE (T04) |
| D05 | 19 Sept | Exact constrained minimum-node simplification per full-volume branch: deviation ≤ 0.75 × local radius, ≥ 95% chord containment; crop afterwards, never re-simplify patches | DONE; used in production |
| D06 | 19–21 Sept | One global representation and a uniform query count; fail loudly on overflow | 192 rejected for full IXI; 256 adopted |
| D07 | 21 Sept | Exhaustive grid plus opt-in graph-positive selection | See [`history_data.md`](history_data.md) §5 |
| D08 | 21 Sept | Use CaravelMetrics definitions (length, arc/chord tortuosity, counts, β) to *evaluate* the saved graphs, not as an extractor or repair | Length/branch geometry validated on a sample; spline curvature, fractal and biomarker claims deferred |

Source: [tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md). D08
caveats worth keeping: β₁ is a cycle rank and cannot by itself show that a loop is real; a node
degree > 3 is a QC flag, not a pathology; arc/chord tortuosity is undefined for closed branches.
Segmentation-derived radius is a GT-only diagnostic, not a free inference-time feature.

**[DONE] Mechanics checks.**
- T01: a square-ring mask with 72 skeleton voxels used to export 0 nodes and β = (0, 0); after the
  fix it gives 3 nodes and (1, 1) ([evidence/T01](evidence/T01/ring_regression.json)).
- T03: a synthetic build gave 2 / 4 / 39 nodes for junction-only / adaptive / dense with the same
  β = (1, 0) ([evidence/T03](evidence/T03/representation_counts.json)).
- T04: the dense reference stays on integer voxel centres (144 samples, none sub-voxel). Adaptive
  samples are smoothed (78 samples, 38 non-integer) ([evidence/T04](evidence/T04/reference_policy_check.csv)).

These checks validate the mechanics only, not the choice of representation.

## 2. Off-vessel chords: containment as a hard constraint

**[DONE] Diagnosis on real IXI crops (18 Sept).** User 3D review found straight edges that cross
outside the lumen, and separately dense cycle tangles
([tracker E07/E08](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md),
[loop mitigation](../archive/legacy-20261103/docs/research/loop_failure_mitigation.md)).
- Fixed 2-voxel RDP left 4, 6 and 3 edges with < 50% chord containment in three crops.
  Radius-normalised RDP (0.6 × radius) cut this to 0, 2 and 3. **β₁ did not change under any
  policy.** Simplification can fix geometry, not segmentation-derived topology. The loop question
  is in [`history_data.md`](history_data.md) §6.
- The fix adopted: smoothing that rejects moves leaving the mask, then subdivision until every
  straight training edge stays inside the lumen. On the densest IXI002 crop (β₀/β₁ = 5/37 in
  every representation):

  | Representation | Nodes / edges | Minimum chord containment |
  |---|---:|---:|
  | Junction-only | 98 / 130 | 0.106 |
  | Adaptive | 244 / 276 | 1.0 |
  | Dense | 983 / 1,015 | — |

- Raw voxel-centre paths turn by a median 45.0° between samples; constrained smoothing brings this
  to 6.6°. 28.4% of adaptive nodes are sub-voxel in that crop (46.5% in the whole volume).
- Conclusion: junction-only stays a topology-preserving ablation, not a geometric target.

## 3. Adaptive simplification policy

**[SUPERSEDED by the exact solver] Fixed and radius-aware greedy RDP sweep (19 Sept).** The
first production extraction was cancelled before it started, because the strict 2-voxel baseline
produced too many degree-2 nodes. Eight policies, then a second stage (fixed 4/5/6 voxels;
radius-aware 1/1.5/2×; all with ≥ 95% containment), were derived from one dense extraction per
volume on 14 real volumes (6 IXI, 4 TopBrain CT, 4 TopBrain MR; the 72-Mvoxel `topcow_ct_017`
deferred). All preserved β₀/β₁ on 14/14. Fixed tolerances up to 4 voxels were too weak.
Radius-1.5× was chosen over 2× (P95 maximum error 0.634 → 0.603 mm, worst error / local radius
1.77 → 1.50, for about 1% more edges). An independent check found 0 of 13,527 edges below 95%
containment ([adaptive tuning](evidence/adaptive_tuning/README.md)).

**[DONE] Exact constrained shortest path, radius 0.75× (19 Sept).** For each dense branch, sample
indices form a DAG. An arc is allowed only if the chord stays within the minimum local tolerance
over the skipped interval and has ≥ 95% containment. The shortest path keeps the fewest samples
possible under these constraints, is deterministic, and its node count cannot rise as tolerance is
relaxed. On the same 14 volumes ([comparison](evidence/adaptive_tuning/selected_policy_comparison/README.md)):

| Metric | Greedy RDP 1.5× | Optimal 0.75× |
|---|---:|---:|
| Full-volume degree-2 nodes | 9,682 | 8,962 (−7.4%) |
| P95 nodes per non-empty 64³ patch | 100.5 | 95 |
| Patches > 120 nodes | 57 | 45 |
| P95 / worst edge maximum error | 0.603 / 2.736 mm | 0.585 / 1.913 mm |
| Worst error / local radius | 1.496 | 0.750 |
| Length-weighted shortening | 1.80% | 1.94% |
| Edges < 95% containment | 0 | 0 |

- Accepted cost: slightly more average shortening and slightly lower mean containment
  (99.82% → 99.65%).
- On the IXI068 calibration volume, 0.5× improved fidelity but produced more edges than the RDP
  baseline. 1× and 1.25× allowed more error than wanted.
- Per-stratum caveat: IXI MRA and TopBrain CT improve, TopBrain MR is flat (P95 45 → 46, patches
  > 70 nodes 4 → 7, max 115). The rule is therefore promising as a dataset-agnostic policy but not
  a proven one.

## 4. Junction-only vs adaptive vs dense (T05/T06)

**[DONE — scoped study, 21 Sept] 30 real full volumes.** Six each from IXI Guys, HH and IOP, and
TopBrain CT and MR. Saved full-volume products were compared with fixed branch identities and
straight edges. Distances were sampled every 0.5 mm against each branch's own counterpart
([evidence/T05](evidence/T05/README.md),
[rate–distortion plot](evidence/T05/stratified_30/rate_distortion.png)).
- Adaptive beats junction-only on branch-paired 0.5-mm Curve F1 and ACD in **30/30** volumes,
  uses fewer nodes than dense in 30/30, and keeps the extracted β₀/β₁ in 30/30.
- 0.5-mm Curve F1:

  | Representation | Range of stratum means |
  |---|---:|
  | Adaptive | 0.757 (TopBrain CT) – 0.893 (IXI Guys) |
  | Junction-only | 0.227 – 0.306 |

  At 1 mm adaptive reaches 0.966–0.998.
- Mean ACD: adaptive 0.271–0.366 mm; junction-only 2.261–4.303 mm.
- Mean nodes per volume: adaptive 550–1,352; dense 3,943–6,760.
- The apparent 14.7–16.1% length loss from raw dense to adaptive is mostly smoothing the voxel
  staircase (13.4–14.5 percentage points). Only 1.4–2.2 points come from straight-edge
  simplification.
- Halving the sample spacing moved 0.5-mm F1 by < 0.006 on the two volumes checked.

Caveats kept from the source:
- identical topology is expected, since all three products come from one extraction, and says
  nothing about segmentation quality;
- dense is not manual ground truth;
- not an exhaustive 220-volume audit (an earlier unbounded 220-volume run died without saving and
  is not evidence);
- IXI spacing is nearly constant within a site.

Conclusion: adaptive was selected as a **provisional compact target**.

**[DONE] Branch F1 is invariant to subdivision (T07).** On the real IXI122 and TopBrain CT-001
full graphs, adaptive vs dense scores Branch F1 = 1.0 despite different node counts; deleting one
edge lowers it to 0.9977 and 0.9899. These are controlled perturbations, not model predictions
([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)).

**[PROPOSED; partly run elsewhere] Model trained on adaptive vs junction-only vs dense.** The
roadmap's Phase 5 asked for the unchanged model retrained on adaptive GT, then junction-only and
denser ablations ([roadmap](../archive/legacy-20261103/docs/research/vascular_graph_roadmap.md)).
The adaptive run (T09) was done; its results are outside this topic. A learned comparison of
explicit degree-2 subdivision on BRAVA (26 Oct) is summarised in
[`BRAVA_preparation.md`](BRAVA_preparation.md). That summary reports that subdivision improves
partial curve coverage but increases cycle errors.

## 5. Object-query capacity

**[DONE] Overflow is caused by geometry nodes, not topology.** With D05 on the 14-volume subset,
45 of 2,326 non-empty patches exceed 120 nodes and none exceeds 192. Topological plus
exact-boundary nodes never exceed 81, so degree-2 geometry nodes cause every overflow
([adaptive tuning](evidence/adaptive_tuning/README.md)). Simplifying overflowing patches on their
own was rejected, because the target would then depend on crop placement. Training now raises an
error when a target exceeds the query count, instead of letting Hungarian matching silently drop
GT nodes.

**[DONE] 120 → 192 cost (T10, 20 Sept).** On 884 patches from seven IXI volumes (59 > 120 nodes,
max 177), 192 queries cost +2.02% mean step time and +0.101 GiB peak memory; the 177-node patch
trained ([tracker](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md),
[benchmark](evidence/ixi_query_benchmark/RESULTS.md)).

**[DONE] 192 rejected for full IXI; 256 adopted (21 Sept).** The complete legacy-backend grid has
35 of 50,613 patches above 192 nodes (all IXI, max 239), while TopBrain peaks at 133. This
overturned the 14-volume and 7-volume inference that 192 covers everything. A paired H100 test on
real IXI crops ([evidence](evidence/ixi_capacity_192_256/README.md)):

| | 192 queries | 256 queries |
|---|---:|---:|
| Mean step | 92.22 ms | 99.14 ms (+7.51%) |
| Peak allocated memory | 3.286 GiB | 3.371 GiB (+0.085 GiB) |

A real 239-node / 261-edge crop completed a 256-query step. This settles compute and capacity
only, not convergence or accuracy. The current Vedo-backend active sets have no patch above 256
nodes ([`history_data.md`](history_data.md) §5).

## 6. Continuous (spline) targets: how much capacity does an edge need?

The roadmap placed "function-valued edges" last (Phase 13), after the adaptive-node approach was
understood. In practice the project adopted K8 spline edges on contracted graphs in October; see
[`history_continuous_edges.md`](history_continuous_edges.md). Two GT-only studies measured whether
the targets themselves were adequate.

**[DONE] Adequacy of the cached K8 targets (22 Oct).** Errors are measured against each edge's
own dense path in mm, with a conservative bound on the continuous maximum. All training and
validation patches; no test data
([report](evidence/spline_target_capacity_20261022/report.md); the same report was filed as
[spline_learning/ixi-brava-control-capacity.md](../archive/legacy-20261103/docs/research/spline_learning/ixi-brava-control-capacity.md)).

| Dataset / split | Branches | P95 maximum error | Definitely > 1 mm | Patches with any > 1-mm branch |
|---|---:|---:|---:|---:|
| BRAVA val | 19,446 | 1.011 mm | 5.01% | 35.86% |
| IXI val | 51,937 | 0.149 mm | 0.18% | 3.32% |

- Errors concentrate in long branches: BRAVA validation branches of 50–100 mm exceed 1 mm in
  88.37% of cases, IXI in 58.76%.
- On an enriched difficult subset, K12/K16 refits cut the BRAVA validation P95 maximum from 3.658
  to 2.356/1.397 mm. This shows capacity helps on those branches; it is not a prevalence estimate
  and not a learned result.
- Counts are patch-edge occurrences from overlapping crops, not independent vessels.

**[DONE] Fixed vs adaptive vs piecewise GT comparison (22 Oct).**
- Data: all BRAVA validation patches (1,796 patches, 8 subjects, 19,446 branches) plus
  subject-balanced 480-patch validation subsets of IXI and SyntheticMRI.
- Rules: tolerances fixed in advance; greedy error-driven splitting with a 2-mm minimum child arc.
  This is a bounded heuristic, not a proof of minimal segmentation
  ([comparison](../archive/legacy-20261103/docs/research/continuous_representation_comparison.md),
  [GT report](evidence/continuous_representation/gt-report.md),
  [summary](evidence/continuous_representation/results-summary.md)).

BRAVA validation, against original dense paths:

| Representation | Mean error | P95 branch max | Branches within 0.5 mm | Controls / branch |
|---|---:|---:|---:|---:|
| Fixed K8 | 0.0898 mm | 1.0110 mm | 77.19% | 8 |
| Fixed K16 | 0.0266 mm | 0.4732 mm | 94.02% | 16 |
| Fixed K24 | 0.0131 mm | 0.3011 mm | 98.34% | 24 |
| Oracle adaptive K, 0.5 mm | 0.0690 mm | 0.4271 mm | 98.35% | 7.79 |
| Piecewise K8 inside edge, 0.5 mm | 0.0460 mm | 0.4131 mm | 100% | 10.61 |
| Same pieces as split graph | 0.0460 mm | 0.4131 mm | 100% | 10.61 |

- Oracle adaptive K spends the tolerance rather than minimising error. It still fails 0.5 mm on
  1.65% of branches, because no available K passes.
- Piecewise fitting passes 0.5 mm on every BRAVA validation branch, with 22.81% of branches
  split. At 0.25 mm it passes 99.84%.
- Joins are C0 only (tangent continuity not enforced).
- Explicit graph splitting gives identical geometry but raises mean nodes 17.14 → 20.67 (+20.6%),
  maximum nodes 69 → 97, and candidate node pairs +48.2%. All patches still fit 256 queries.
  Extra graph nodes therefore bring **no GT geometry benefit**; their learning cost was not
  tested here.
- IXI and SyntheticMRI barely need more than K8: fixed K8 meets 0.5 mm on 99.14% / 99.32% of
  branches, and oracle adaptive K meets it on 100% with 4.38 / 4.58 controls. The representation
  bottleneck is BRAVA and rare complex branches. It does not explain the IXI connectivity
  failures.

Recommendation recorded at the time:
- piecewise geometry inside each existing edge was the most promising representation candidate;
- fixed K16/K24 were to be kept as simple controls;
- explicit degree-2 splitting only if its extra supervision showed a measurable benefit.

The source states this is a representation result, not a learned one. The frozen-detector
learned heads from the same batch are covered in
[`history_multidataset_brava.md`](history_multidataset_brava.md).

## 7. Evaluation principle that came with the representation

**[DONE as protocol]** The representation note fixed a three-block evaluation that the metrics
work then implemented
([representation note](../archive/legacy-20261103/docs/research/vascular_graph_representation.md)
§16–19):
- **representation quality:** adaptive vs dense, a property of the target, not of the network;
- **model quality:** prediction vs adaptive, where a perfect score is reachable;
- **end-to-end fidelity:** prediction vs dense, always shown next to the adaptive-vs-dense oracle
  so that deliberate compression error is not counted as model error.

It also asked that topology be scored after degree-2 contraction, so that the number and
placement of geometry nodes does not dominate the score.

**[PROPOSED, never run]**
- a radius function r(t) per edge for lumen, morphometry and resistance estimates;
- degree-2 density ρ₂ per mm reported as an explicit compactness axis;
- APLS, whole-volume stitching and structure-aware Hungarian matching. These are roadmap
  Phases 8–12 ([roadmap](../archive/legacy-20261103/docs/research/vascular_graph_roadmap.md));
  none was executed in the sources assigned here.

## What carried forward / what was dropped

Carried forward to the current dense model:
- **Topology + curved edge.** The dense model's nodes are junctions, ends and border crossings
  only. Degree-2 points are merged into edges ([glossary](network_proposals_20261030.md)), and
  each edge is a clamped cubic B-spline (Approach B).
- **Rule-based capacity per edge.** It chooses K8/K16/K24 per edge by a fixed rule on the traced
  path's arc length and tortuosity ([dense model](dense_traced_model_20261102.md) §2). This is in
  the spirit of the GT finding that only long or tortuous branches need more than 8 controls.
- **Dense centreline as geometric GT,** branch-paired distances, and degree-2-invariant Branch F1.
- **Full-volume simplification with exact boundary crops** and the containment constraint, which
  still define the stored adaptive graphs.

Dropped:
- adaptive degree-2 nodes as detection targets;
- query-count capacity as a design driver (the dense model takes heatmap peaks, up to 200 per
  patch during training);
- explicit graph splitting;
- greedy RDP and voxel-fixed tolerances;
- junction-only with straight edges as a target (kept only as an ablation).

## Sources

Archived originals (post-move paths):
- [IMPLEMENTATION_TRACKER.md](../archive/legacy-20261103/docs/research/IMPLEMENTATION_TRACKER.md)
  — decision log D01–D08, T03–T07/T10 results, policy selection, capacity follow-ups.
- [vascular_graph_representation.md](../archive/legacy-20261103/docs/research/vascular_graph_representation.md)
  — design note: topology vs geometry, adaptive degree-2 vs function-valued edges, radius
  extension, three-block evaluation.
- [vascular_graph_roadmap.md](../archive/legacy-20261103/docs/research/vascular_graph_roadmap.md)
  — ordered Phase 0–13 plan (representation, metrics, retraining, matcher, whole-volume,
  function-valued edges).
- [continuous_representation_comparison.md](../archive/legacy-20261103/docs/research/continuous_representation_comparison.md)
  — fixed/adaptive/piecewise/split spline GT study and frozen-head pilot (22 Oct).
- [loop_failure_mitigation.md](../archive/legacy-20261103/docs/research/loop_failure_mitigation.md)
  — chord-containment correction and the false-loop plan (detailed in `history_data.md`).
- [SPLINE_REPRESENTATION.md](../archive/legacy-20261103/docs/SPLINE_REPRESENTATION.md) — K8
  endpoint-constrained spline target (V0 controls / V1 sampled curves).

Evidence kept in place: [adaptive_tuning](evidence/adaptive_tuning/README.md),
[selected_policy_comparison](evidence/adaptive_tuning/selected_policy_comparison/README.md),
[T05](evidence/T05/README.md), [T01](evidence/T01/), [T03](evidence/T03/), [T04](evidence/T04/),
[ixi_capacity_192_256](evidence/ixi_capacity_192_256/README.md),
[spline_target_capacity_20261022](evidence/spline_target_capacity_20261022/report.md),
[continuous_representation](evidence/continuous_representation/gt-report.md)
([learned report](evidence/continuous_representation/learned-report.md),
[summary](evidence/continuous_representation/results-summary.md)).
