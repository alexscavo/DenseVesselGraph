# Vascular graph roadmap implementation tracker

> **Research consolidation, 23 October 2026:** [Consolidated decisions and alternatives](../../research/decisions.md). This document retains its detailed and dated historical context.

## Purpose

This is the living execution record for
[`vascular_graph_roadmap.md`](vascular_graph_roadmap.md). Update it whenever a
task changes state, a decision is made, or validation produces evidence.

Status vocabulary: `not started`, `in progress`, `blocked`, `complete`.

A task is not complete until it has automated checks and inspectable evidence.
For spatial or quantitative work, save both machine-readable data and a plot or
screenshot under `evidence/<task-id>/` so results can be independently reviewed.

## Current position

- Current phase: Phase 5 — IXI-only model evaluation. The
  focused Phase 2 study is complete on 30 real stratified volumes (scope and
  caveats: [`evidence/T05/`](evidence/T05/)). Phase 3 Branch F1 is integrated
  in the evaluator in physical coordinates. Phase 4 controlled metric checks,
  including SMD and node/edge detection contrasts, are complete; see
  [`evidence/T08/`](evidence/T08/). The 220-volume Vedo-original production
  extraction and graph-positive patch sets are validated in
  [`VEDO_PRODUCTION_RUN.md`](VEDO_PRODUCTION_RUN.md). The 300-epoch IXI run
  and full 2,384-patch held-out test are complete. Test node F1 is 0.409,
  edge F1 0.307, and physical Branch F1 0.00535; predictions contain far
  too many edges and cycles. See [`evidence/T09_long/test_results.md`](evidence/T09_long/test_results.md)
  and its linked-camera real-patch views.
- Next task: diagnose edge overconnection using validation-only confidence
  calibration and topology constraints before another test evaluation. The
  original Work quota failure, epoch-100 recovery, 300-epoch extension, and
  its test are recorded in [`evidence/T09_long/`](evidence/T09_long/).
  False fused-vessel loops in input segmentation remain independent QC work.

## Ordered task list

| ID | Task | Roadmap phase | Status |
|---|---|---:|---|
| T01 | Preserve pure degree-2 rings and add topology regression tests | 1 | complete |
| T02 | Enforce the IXI638/IXI661 annotation overrides and record provenance | 1 | complete |
| T03 | Emit junction-only, adaptive, and dense comparison representations | 1 | complete |
| T04 | Preserve and name an immutable dense reference before simplification | 1 | complete |
| T05 | Implement representation geometry, morphology, and complexity metrics | 2 | complete (30-volume scoped study) |
| T06 | Run the rate-distortion study across sites and acquisition resolutions | 2 | complete (five sampled strata; transfer caveats) |
| T07 | Implement degree-2 contraction, physical anchor matching, and Branch F1 | 3 | complete (evaluator + 2 real IXI patch ablations) |
| T08 | Validate metrics on the prescribed handcrafted failure cases | 4 | complete (metric behavior only; not extractor/model quality) |
| T09 | Train the 256-query IXI model through epoch 300 and evaluate the held-out test split | 5 | complete (job 113928; test job 138721; poor edge topology documented) |
| T10 | Benchmark 120 versus 192 and 192 versus 256 object queries on paired real IXI patches | 5 | complete |
| T11 | Diagnose predicted edge overconnection using validation-only decisions | 5 | not started |

### 2026-09-22 — physical Branch F1 integration and real patch checks

The evaluator reports per-patch `branch_{tp,fp,fn,precision,recall,f1}`
and micro-aggregated `branch_f1` if a physical-mm threshold is configured.
The IXI experiment config enables 1.0 mm; other experiments keep the previous
metric protocol. Full source-affine coordinates are reconstructed from the
dataset patch index and source manifest, **not** the identity-affine patch
NIfTI. [Two real IXI validation patch ablations](evidence/T07/ixi_patches/ixi_patch_edge_ablation.png)
and [machine-readable scores](evidence/T07/ixi_patches/ixi_patch_edge_ablation.json)
show 1.000 -> 0.857 and 1.000 -> 0.783 after removing one actual graph edge;
no model weights or anatomical truth are involved. Focused evaluator,
physical-provenance, and branch tests passed. The single-edge changes test
the metric's sensitivity, **not** whether real model predictions improve.
The evaluator's computed Branch F1 was checked against the standalone score
on both of these saved IXI patches. Focused Phase 4 unit cases also show that
a small real-world-coordinate break or an extra close bridge can score
Curve F1=1 at the chosen tolerance while Branch F1 drops; conversely,
altering only the interior degree-2 path can preserve Branch F1 while Curve
F1/ACD/HD95 worsen. Wrong branch pairings can leave both Betti numbers
unchanged. These are *metric-behavior synthetic controls*, not evidence of
extractor anatomical quality; those claims still rely on real-image checks.
Training is unnecessary for T07/T08 metric validation, but is necessary for
Phases 5–7 to establish learned graph quality and convergence.

## Decision log

### Evidence curation (2026-09-23)

The [research index](README.md#evidence-to-show-the-decisions) now points to
the evidence needed to explain each current choice. The superseded five-volume
T05 pilot, a duplicated IXI Vedo smoke graph, CT-001 extraction intermediates,
redundant diagnostic viewers, and a generated Python cache were removed from
`evidence/` (about 25 MB). The 30-volume study, final CT-001 method comparison,
production checks, capacity benchmarks, metric checks, and live IXI training
evidence remain. Relative Markdown links in `docs/research` were verified.

### D01 — Representation of an anchorless ring

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: represent each pure degree-2 ring with three deterministic
  degree-2 nodes and three ordered arcs.
- Reason: one node plus a self-loop and two nodes plus parallel edges are not
  reliably representable by the downstream simple adjacency matrix. Three
  nodes form a simple cycle, preserve beta-0/beta-1, and remain compatible with
  adaptive RDP subdivision.

### D02 — Three explicit source representations

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: emit `graphs/junction_only`, `graphs/adaptive`, and `graphs/dense`
  as separate graph products. Patch generation defaults to `adaptive` when this
  layout is detected.
- Reason: representation quality must be measured before selecting a training
  target. Keeping the products separate makes comparisons reproducible and
  prevents dense-reference geometry from being confused with model ground truth.

### D03 — Dataset-agnostic extraction constraint

- Date: 2026-09-18
- Status: active design constraint
- Decision: extractor policies should use physical units or local normalized
  quantities. Dataset-specific constants are provisional and must be tested
  across acquisition resolutions, sites, vessel calibres, and geometries.
- Consequence: the current fixed voxel-based RDP and spur defaults are baselines,
  not accepted final policies. T06 must explicitly test their transferability.

### D04 — Dense means pre-smoothing reference

- Date: 2026-09-18
- Status: accepted for implementation
- Decision: `dense` is the ordered centerline after topology cleanup but before
  smoothing and RDP. It is one of the three representations, not a fourth
  auxiliary graph.
- Reason: smoothing is an intentional geometric transformation. Its distortion
  must be visible in the representation study instead of being silently folded
  into the reference.

### D05 — Global constrained-optimal adaptive simplification

- Date: 2026-09-19
- Status: accepted for full-dataset validation
- Decision: derive the adaptive graph once per full volume with the exact
  minimum-segment path through each dense branch. A candidate chord is valid
  only when its physical deviation is no greater than `0.75 ×` the minimum
  local lumen radius on the interval and at least 95% of its samples lie in the
  segmentation. Crop this fixed graph afterward using exact boundary
  intersections; never re-simplify individual patches.
- Reason: the shortest path on the ordered dense samples is globally optimal
  for the stated constraints and has monotonic complexity as tolerance is
  relaxed. On 14 real volumes it uses fewer degree-2 nodes than greedy RDP while
  also reducing P95 and worst-case centerline error.

### D06 — Handle graph-token overflow with uniform capacity

- Date: 2026-09-19
- Status: 192-query limit rejected for full IXI on 2026-09-21; uniform-capacity
  principle retained pending a new capacity measurement
- Decision: retain the single global graph representation and use 192 object
  queries for the initial seven-volume real-data experiment, but do not use
  192 for the complete IXI dataset without resolving its 35 overflow cases.
  The data pipeline and loss must fail explicitly when a target exceeds the
  configured capacity.
- Reason: 45 of 2,326 nonempty 64³ patches exceed 120 nodes under D05, but none
  exceeds 192. Topological plus exact-boundary nodes have a maximum of 81;
  geometry nodes explain every observed overflow. Patch-specific simplification
  would make the target representation depend on crop placement.
- Full-data result: 35/50,613 patches exceed 192 nodes, all from IXI MRA;
  maximum 239. The 192-query setting is sufficient for all 28,611 TopBrain
  patches (maximum 133) but not for all 22,002 IXI patches. These overflow
  patches are preserved in the dataset and must fail explicitly if fed into a
  192-query model. Measure a uniform 240/256-query alternative or an explicit
  overflow policy before full-IXI training; do not silently discard targets.
- Follow-up (2026-09-21): 256 queries cover the observed IXI maximum of 239
  nodes. The paired real-IXI H100 benchmark found +7.51% mean step time and
  +0.085 GiB peak allocation relative to 192; a 239-node patch completed a
  training step. This resolves *capacity*, not convergence, graph correctness,
  or the treatment of negative validation/test crops. Evidence:
  [`evidence/ixi_capacity_192_256/`](evidence/ixi_capacity_192_256/).

### D07 — Direct dataset output and explicit patch eligibility

- Date: 2026-09-21
- Status: implemented; training policy pending representation study
- Decision: default future production full-volume extraction and patch
  generation to the corresponding IXI/TopBrain dataset folders, with distinct
  `vascular_graphs/optimal_radius_0p75x_c095/` and
  `vascular_patches/optimal_radius_0p75x_c095/` outputs. Keep the complete
  deterministic 64³ grid for reproducibility and inspectable negative examples.
  The MRI loader can opt into `patch_selection: graph_positive` using the
  per-dataset `patch_index.csv`; this excludes masks without foreground and
  masks without both graph nodes and edges **before** applying a sample cap.
  The default remains `all`; no training configuration has been changed.
- Reason: discarding source data at extraction time prevents studying
  foreground-negative examples, while including them unintentionally in
  graph-learning runs distorts the target distribution. Graph-free but
  mask-positive border slabs require separate QC, not silent relabeling.

### D08 — Reuse CaravelMetrics as a graph-evaluation reference, not an extractor

- Date: 2026-09-21
- Status: length/branch geometry comparisons validated on a stratified sample;
  optional spline/fractal or anatomical-biomarker claims remain unvalidated
- Source: [CaravelMetrics feature definitions](https://github.com/i-vesseg/CaravelMetrics/blob/master/docs/FEATURES.md).
- Decision: first reuse *definitions and comparison ideas* for total physical
  vessel length, branch-wise arc/chord tortuosity, endpoint/junction counts and
  density, component count, and cycle rank. Compute these on the **three saved
  graph representations** or on their ordered branch polylines; the model does
  not need to output a centerline. Our `representation_geometry.py` already
  has physical polyline length, arc/chord tortuosity, curve F1, ACD, HD95, and
  graph complexity; `branch_evaluation.py` has paired branch length/tortuosity
  errors and a discrete turning proxy. Extend/report the missing summaries and
  junction-angle error before introducing new curvature estimators.
- Caveats: a cycle *rank* (β₁) does not count all simple loops and cannot alone
  establish that a cycle is anatomically real; node degree >3 is a **QC flag**,
  not automatically pathological. Define bifurcation density on the same
  physical reference length across representations, and exclude or label
  artificial patch-boundary nodes. For closed branches and near-zero chords,
  arc/chord tortuosity is undefined/infinite and must be reported separately.
- Defer spline curvature and fitting RMSE until an explicit sampling/smoothing
  sensitivity study; optional box-counting fractal dimension/lacunarity must
  use a common physical field of view and scales to avoid confusing crop size,
  resolution, or graph sampling density with anatomy. Segmentation-derived
  radius/diameter/volume are useful **GT-only diagnostics**, not free
  inference-time features of a model predicting only nodes and edges.
- Do not adopt CaravelMetrics' skeletonization, orphan-linking, Laplacian
  smoothing, mesh geodesic, or longest-edge triangle deletion as production
  graph repair. Their effects on real IXI/TopBrain topology and lumen geometry
  need independent, visually reviewed validation. False fused-vessel loops
  remain an open QC concern (see `loop_failure_mitigation.md`).

## Immediate ordered work (updated 2026-09-21)

1. **T05 [complete, scoped study]:** run the saved full-volume three-representation comparison in
   physical coordinates. Include node/edge and degree-2 budgets, curve
   F1/ACD/HD95, matched branch length and tortuosity, junction-angle error,
   β₀/β₁, and explicit boundary/closed-branch rules. Write CSV/JSON and
   representative graph-only screenshots/interactive HTML. Validate any new
   metric with tests; do not claim CaravelMetrics' results transfer here.
2. **T06 [complete, sampled site/spacing study] + QC [ongoing]:** summarize per-subject and by IXI site/voxel spacing versus
   TopBrain MR/CT, plus radius/curvature/density strata. Inspect suspected
   false loops and graph-free foreground on real segmentations/images; report
   failures and annotation uncertainty, without silent graph repair. Preserve
   the complete validation/test grids for unbiased evaluation.
3. **T07/T08 [in progress]:** implement deterministic degree-2 contraction, spatial anchor
   matching and Branch F1; test subdivision invariance, false links, breaks,
   shifted geometry, cycles and shared-border effects on small *metric tests*.
   Real-volume screenshots remain necessary for extractor claims.
4. **Phase 0 + T09:** freeze a comparable model baseline/checkpoint and
   evaluate the fixed adaptive target. Confirm the one-dataset choice and
   negative-crop evaluation protocol before launching training; for IXI, use
   the tested uniform 256-query capacity (not a per-patch re-simplification).
   Record runtime, convergence and model metrics against adaptive GT **and**
   the dense-reference oracle. Only then consider junction-only/dense ablations.

### T05/T06 outcome — focused full-volume representation study

Thirty real full volumes (six per IXI Guys/HH/IOP and TopBrain CT/MR stratum)
were evaluated with fixed branch identities and physical-space straight graph
edges. Adaptive beats junction-only in branch-paired 0.5-mm Curve F1 and ACD
on **30/30**, uses fewer nodes than dense on **30/30**, and preserves extracted
β₀/β₁ on **30/30**. Across strata, adaptive mean F1 at 0.5 mm is 0.757–0.893,
versus 0.227–0.306 for junction-only; at 1 mm adaptive reaches 0.966–0.998.
Its mean ACD is 0.271–0.366 mm. The apparent 14.7–16.1% raw-dense length
shortening is predominantly 13.4–14.5 percentage points of smoothing, plus
1.4–2.2 of adaptive straight-edge simplification. Twice-denser sampling on
one IXI and one TopBrain case moved 0.5-mm F1 by <0.006. Existing unit tests
verify subdivision invariance, directionality, bend sensitivity, parallel
branch matching and explicit degenerate-cycle treatment. Summary/CSV/PNG and
graph-only interactive 3D views: [`evidence/T05/`](evidence/T05/).

This is sufficient to select adaptive as a *provisional compact target* and
move to Phase 3. It is not a claim of anatomically valid raw dense paths or
correct loops: these three representations inherit the same extracted
topology. Source IXI spacing is nearly constant within each site and this
study is not an exhaustive 220-volume or external-dataset audit. The
previous unbounded local audit processed 79 subjects but was terminated
before saving complete results; no unsaved measurements are included.

### T07 early validation on real saved graph families

`metrics/branch_connectivity.py` contracts degree-2 chains while preserving
ordered branch polylines, assigns nearby physical-space anchors one-to-one by
role, and scores branch connectivity separately from geometric overlap.
Unanchored pure rings receive one virtual closed branch with a length-weighted
spatial centroid; parallel branches retain multiplicity. Tests cover
subdivision, displacement, wrong branch pairing with equal Betti numbers,
missing/extra branches, nonconnecting line crossings, pure rings and empty
graphs. On the real IXI122 and TopBrain CT-001 full graphs, adaptive vs dense
scores Branch F1=1.0 despite different geometry-node counts; deleting one
edge of each saved adaptive graph lowers F1 to 0.9977 and 0.9899 respectively.
These are **controlled real-graph perturbations**, not model predictions.
Machine-readable scores and plot are in [`evidence/T07/`](evidence/T07/).
The evaluator must convert normalized patch coordinates to world millimetres
before this metric is used for model checkpoints; this has not been wired yet.

## Evidence log

### E01 — Pre-fix ring failure

A handcrafted square-ring mask produced 72 skeleton voxels and `ring_count=1`,
but exported zero nodes and zero edges, so `VesselGraph.betti()` returned
`(0, 0)` instead of `(1, 1)`. This establishes the T01 regression target.

### E02 — Ring regression result

The focused test suite verifies both the basic three-node ring and its adaptive
RDP subdivision. Both return `(beta_0, beta_1) = (1, 1)`; 2 tests passed. Data
and visualization: [`evidence/T01/`](evidence/T01/).

### E03 — Annotation override provenance

The 397-subject planning run selects `nii(1).gz` for IXI638 and IXI661 and
records both rows as `preferred_duplicate`; all focused extraction tests pass.
Data and visualization: [`evidence/T02/`](evidence/T02/).

### E04 — Three-representation source build

An end-to-end synthetic NIfTI regression build emitted nine graph files: the three
Voreen files for each of `junction_only`, `adaptive`, and `dense`. The example
retained identical topology `(beta_0, beta_1) = (1, 0)` while using 2, 4, and 39
nodes respectively. This validates mechanics only; it is not scientific evidence
for choosing a representation. Data and visualization: [`evidence/T03/`](evidence/T03/).

### E05 — Dense-reference immutability

The regression check verifies that dense-reference samples remain on the raw
voxel-centre centerline while adaptive path samples reflect smoothing. The
example contains 144 concatenated dense samples, all integer-valued, versus 78
adaptive path samples with sub-voxel coordinates. Data and visualization:
[`evidence/T04/`](evidence/T04/).

### E06 — Real IXI three-representation 3D QC

Three real IXI crops cover Guys, HH, and IOP plus both 0.469-mm and 0.264-mm
in-plane acquisition resolutions. Interactive HTML reports overlay all three
representations on the real vessel surface. In every crop, beta-0 and beta-1 are
identical across representations. Adaptive versus dense node counts are 141 vs
924, 110 vs 807, and 314 vs 2141. This supports compression without topology
change, but does not yet establish geometric fidelity or resolution invariance.
Data, regeneration commands, and overview visualization:
superseded exploratory evidence (removed).

### E07 — Real IXI bad-loop and chord diagnosis

User 3D inspection exposed dense cycle tangles and off-vessel straight chords.
Across the three real crops, fixed 2-voxel RDP produced 4, 6, and 3 edges with
less than 50% chord containment. Radius-normalized RDP reduced these to 0, 2,
and 3 at radius fraction 0.6, but beta-1 was unchanged for every policy. Thus
radius adaptation improves geometry but cannot repair segmentation-derived
topology. Data and visualization:
superseded exploratory evidence (removed).

Training-data generation remains paused until the containment contract and
raw-MRA-supported false-bridge validation in
[`loop_failure_mitigation.md`](loop_failure_mitigation.md) are implemented and
reviewed.

### E08 — Real IXI coordinate and patch-budget verification

On a real IXI002 crop, raw voxel-centre paths have a 45.0° median sample-to-sample
turn angle; constrained smoothing reduces this to 6.6°. Adaptive graph nodes are
not voxel restricted (28.4% sub-voxel in the crop; 46.5% in the whole-volume
audit), and every training VTP edge is encoded as a straight two-point line.
Superseded exploratory evidence (removed).

The preliminary whole-volume IXI002 audit finds mean/p95/p99 patch edge counts of
21.7/63.0/78.6. Five of 175 non-empty patches exceed the preferred 70-edge
ceiling and one reaches 139, although no patch exceeds 120 nodes. This means the
typical patch is acceptable but the complexity tail requires topology cleaning
and/or a budget-aware representation policy. CPU Slurm array job `2141634` is
pending for the three-representation, three-subject audit. Evidence:
superseded exploratory evidence (removed).

## Dense intermediate and lumen-safe compact geometry (2026-09-18)

The extractor now skeletonizes each segmentation once into an immutable dense
centerline intermediate and deterministically derives junction-only, adaptive,
and dense graphs from it. Junction representatives retain their real skeleton
routes through non-convex clusters; smoothing rejects moves whose adjacent
segments leave the mask; adaptive RDP subdivision then adds samples until every
straight training edge remains inside the segmented lumen.

On the real IXI002 densest crop, all representations retain β₀/β₁ = 5/37. The
adaptive graph has 244 nodes / 276 edges versus 983 / 1015 dense, with minimum
edge-chord lumen containment 1.0. Junction-only has 98 / 130 but minimum
containment 0.106, so it remains a topology-preserving ablation rather than the
recommended geometric target. See
superseded exploratory evidence (removed).

The reusable evaluator `scripts/compare_dense_centerlines.py` emits CSV, JSON,
PNG and interactive HTML, and `metrics/representation_geometry.py` supplies
continuous-polyline distance, curve precision/recall/F1, ACD, HD95, length,
tortuosity and normalized graph-complexity metrics.

The oversized Slurm audit job 2141634 was cancelled before execution. The audit
request is now one CPU and 30 minutes; no replacement job has been submitted
while local real-crop validation is sufficient.

## Rejected Vedo dense-centerline backend (2026-09-18)

The historical Vedo extractor was reinterpreted correctly: its dense NetworkX
sample graph is a centerline intermediate, followed by our compact graph
extraction. A reusable backend implemented surface-geodesic validation, orphan
handling, constrained sub-voxel smoothing, local digital-cycle cleanup and the
usual junction/branch reduction.

Real IXI evidence rejected it. On the IXI002 crop, adaptive complexity worsened
from 244 nodes / 276 edges / β₁=37 to 363 / 411 / β₁=49, and visual tangles
remained. The legacy extractor was restored as the default; Vedo is available
only by explicit selection for reproducibility. See
superseded exploratory evidence (removed).

## New IXI and TopBrain restart (2026-09-18)

The rejected Vedo evidence was removed at the user's request, and evaluation
restarted on newly supplied data using the better pre-Vedo (`legacy`) backend.
The first subjects are improved-segmentation IXI122 and multiclass TopBrain
MRA subject 001. Fresh interactive 3-D reports, static 3-D views, measurements
and densest-64³ patch counts are in
superseded exploratory evidence (removed).

Both adaptive representations preserve dense β₀/β₁ and have complete lumen
containment. Densest 64³ adaptive patches contain 99 IXI edges and 79 TopBrain
edges: below the 120-token capacity but above the preferred 60–70 range. The
next dataset-level test must measure percentiles across subjects rather than
optimizing against these two worst-density patches.

Those initial counts were produced by cropping the segmentation before graph
extraction. The corrected evaluation now extracts all three representations on
the full volume and only then crops the saved adaptive graph. Cropping is done
in voxel coordinates (important for TopBrain's oblique affine) and uses
`SourceGraph.crop_inherited()`: a crossing edge terminates at its last existing
centerline sample inside the patch, without an interpolated box-face node.
Results and full-volume visualizations are in
superseded exploratory evidence (removed).

## Exact boundary intersections selected (2026-09-18)

Exact voxel-cell-face clipping is now the production patch policy. Full graphs
are transformed into voxel coordinates and clipped against
`[start - 0.5, start + crop_size - 0.5]`, which fixes world-axis-box errors for
oblique affines and gives neighboring non-overlapping patches the same shared
face. Synthetic intersection nodes retain `source_edge_index` and an explicit
boundary flag in VTP, yielding deterministic cross-patch correspondence while
remaining compatible with the existing model reader.

The last-inside-sample policy remains an ablation. On real densest IXI122 and
TopBrain-MR-001 patches it produced identical topology and token counts, but
terminated crossing edges approximately 0.42 voxel inside the true face. Exact
clipping therefore strictly improves boundary geometry without increasing the
graph budget in these tests. Metrics, JSON, PNG, and interactive comparisons
are superseded by the complete production inventory.

The complete extraction inventory contains 220 paired volumes: 170 IXI MRA,
25 TopBrain MR, 18 TopBrain CT training, and 7 labeled TopBrain CT test cases.
The resumable CPU-array extractor skeletonizes every full segmentation once and
writes separate junction-only, adaptive, and dense graph products. Adaptive is
the selected training representation; the other two are retained for controlled
comparison. Each volume has an atomic, configuration-fingerprinted completion
marker.

The Jean Zay launcher uses one CPU and partition-default memory per volume,
a 20-minute walltime, and an array concurrency cap of 12. Explicit `--mem` is
not valid on Jean Zay and is intentionally omitted.

Full-dataset extraction was submitted as Slurm array job `2161189` with tasks
`0-219%12`. Outputs are written below
`dataset-local vascular_graphs outputs/`; rerunning the launcher safely
skips volumes whose configuration-fingerprinted completion marker and nine
graph files are present.

Job `2161189` was cancelled while still pending, before any cluster task ran,
because the strict baseline produced too many degree-2 nodes in the inspected
patches. A deterministic 15-volume, eight-policy tuning sweep now precedes the
production extraction. Its design and subset evidence are in
[`evidence/adaptive_tuning/`](evidence/adaptive_tuning/).

The tuning sweep was submitted as Slurm array job `2161680` (`0-14%4`, one CPU,
30 minutes per selected volume). It derives all eight policies from one dense
extraction per volume and writes exact-crop patch-budget statistics. Production
extraction remains intentionally paused until these results select one global
policy.

Model-only full-volume adaptive views (straight edges plus every termination,
degree-2 subdivision, and junction; no centerline polylines) are recorded in
current full-dataset sample views.

## Initial greedy-RDP policy selection, now superseded (2026-09-19)

The pending Slurm sweep was cancelled and reproduced locally, sequentially, on
14 real volumes (six IXI, four TopBrain CT, four TopBrain MR). The 72-million-
voxel `topcow_ct_017` outlier is intentionally deferred. All policies were
derived from saved full-volume dense graphs, then audited with exact 64³,
stride-40 clipping; no crop-first skeletonization was used.

The first sweep showed that fixed tolerances up to four voxels were not strong
enough. A second sweep tested fixed 4/5/6-voxel and radius-aware 1/1.5/2×
tolerances, all with a 95% minimum straight-chord lumen-containment constraint.
Every policy preserved full-volume β₀/β₁ on 14/14 subjects. A follow-up geometry
audit selected radius-1.5× rather than radius-2×: it lowers P95 maximum error
from 0.634 to 0.603 mm, length-weighted shortening from 1.89% to 1.80%, and the
worst error normalized by local radius from 1.77× to 1.50×. The cost is about
1% more full-volume edges; patch P95 changes from 99 to 100.5 nodes and the
number of patches above 120 changes from 59 to 57. Radius-1.5× was therefore the
initial global dataset-agnostic policy before the exact solver below.

An independent audit found zero radius-1.5× edges below the configured 95%
containment floor (13,527 edges total; mean containment 99.82%). Budget data and plots are in
[retained compact policy comparison],
geometry data and plots in
[retained compact policy comparison],
and graph-only PNG/interactive HTML views for IXI425 and TopBrain-MR-002 in
[retained compact policy comparison].

Patch overflow is not caused by unavoidable topology. For radius-1.5×, the 57
patches above 120 contain on average 32.2 topological nodes, 20.6 exact boundary
nodes, and 91.3 degree-2 geometry nodes; topology plus boundary has a maximum of
81 over all 2,326 nonempty patches. Without changing or locally simplifying the
representation, 192 object queries cover every observed patch (maximum 186),
whereas 160 still misses 14. The selected strategy is therefore a single
full-volume representation plus a 192-query real-data model configuration,
subject to confirmation on the complete dataset and a GPU memory benchmark.
Training now raises an explicit error when a target exceeds query capacity,
instead of letting Hungarian matching silently omit unmatched ground-truth
nodes. Evidence is in
[retained compact overflow summary].

The audit also exposed and fixed a serialization defect: six-decimal world
coordinates in `nodes.csv` could move a junction by approximately 1e-7 voxel
across a voxel-cell face after reloading. Node coordinates now use 12 decimal
places, legacy tuning inputs recover full-precision nodes from `graph.vvg`, and
a regression test checks preservation of voxel-cell side. Forced tuning runs
now invalidate stale completion markers before overwriting artifacts, so an
interruption cannot masquerade as a completed regeneration.

## Global optimal simplification selected (2026-09-19)

Greedy RDP followed by containment repair was replaced by a constrained
shortest-path solver over each full-volume dense branch. The solver retains the
fewest dense samples whose chords satisfy both the local physical-error profile
and the lumen-containment requirement. This is performed before cropping, so
all overlapping or neighboring patches inherit the same graph and matching
face-intersection nodes.

The final 14-volume comparison selected optimal radius-0.75× at 95% minimum
chord containment over the previous greedy radius-1.5× baseline. Both preserve
topology on 14/14 volumes and neither has an audited edge below 95%
containment. The selected policy changes:

- full-volume degree-2 nodes: 9,682 → 8,962 (-7.4%);
- P95 patch nodes: 100.5 → 95;
- patches above 120 nodes: 57 → 45 (-21.1%);
- P95 edge maximum error: 0.603 → 0.585 mm;
- worst edge maximum error: 2.736 → 1.913 mm (-30.1%);
- length-weighted shortening: 1.80% → 1.94%.

The last change is the measured cost: the exact optimizer uses its allowed
deviation more consistently, producing slightly more average shortening even
though its P95 and maximum errors improve. On the IXI068 calibration volume,
optimal radius-0.5× further improves geometry but exceeds the RDP baseline edge
count; radius-1× and 1.25× are more aggressive than desired. Data, a direct
comparison plot, overflow
decomposition, and interactive IXI/TopBrain graph viewers are in
[`evidence/adaptive_tuning/`](evidence/adaptive_tuning/).

Stratified patch counts improve for IXI MRA and TopBrain CT. TopBrain MR is
approximately neutral: mean nodes remain 16.1, P95 changes from 45 to 46, and
patches above 70 change from 4 to 7, with a maximum of 115. The policy is thus a
promising dataset-agnostic rule, not yet a completed transferability claim.

## Full-dataset production extraction (2026-09-20)

The selected D05 policy is now being applied to the complete deterministic
inventory: 170 IXI MRA, 25 TopBrain MR, 18 TopBrain CT training, and 7 labeled
TopBrain CT test volumes. CPU array job `2192366` has one volume per task, one
CPU per task, a 20-minute limit, and at most 12 concurrent tasks. Extraction is
resumable and each result is accepted only when its configuration fingerprint
and all graph files match constrained-optimal simplification at radius-0.75x
with 95% minimum chord containment.

Dependent job `2192776` will run only after all 220 extraction tasks succeed.
It validates and stages the selected full-volume `adaptive` products, then
crops 64-cubed patches at maximum stride 40 with exact graph/patch-face
intersections and no patch-level re-simplification. The seven official
TopBrain CT test cases remain in test; the other subjects are deterministically
stratified, yielding 154/33/33 train/validation/test patients overall.

Generated patches are written to
`/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/VesselGraph_patches_optimal_0p75_c095`.
The post-run audit will validate every full graph and patch graph and write JSON,
CSV, overflow tables, and `production_graph_and_patch_qc.png` below
[`evidence/full_dataset_production/`](evidence/full_dataset_production/).

### Production run and dataset placement

All 220 tasks of array `2192366` completed successfully, including
`topcow_ct_017` (1 minute 35 seconds); all 220 selected-policy full-volume
markers were validated by the staging job; final dataset placement will
checksum all nine graph files per subject before removing any originals.
An independent inventory verified all nine files on all 220 subjects and
identical full-volume β₀/β₁ across junction-only, adaptive, and dense graphs
for every subject. This confirms representation-internal topology preservation,
but does not establish that every segmentation has correct vascular topology.
The first patch run `2192776` completed 218 patients, but failed on IXI371:
the legacy normalizer first casts floating-point MRA values to integers, which
made its median/MAD threshold zero although the image contains real signal.
The fallback uses the finite floating-point 99.5th percentile **only** when
the legacy threshold is invalid, preserving the 218 already completed patch
normalizations. The first resume `14070` additionally exposed a `2.94e-7`
image/label affine rounding difference on TopBrain MR-002; patch geometry now
uses the same `1e-4` absolute tolerance as full-volume extraction. Both
failures were caught by the source/patch validation, not silently accepted.
Focused generator, split, and relocation tests pass (9/9).

The second resumable patch job `14146` was canceled while pending to avoid
unnecessary queue delay. The sole remaining TopBrain MR patient was completed
locally with one CPU, and the independent audit verified all 220 full graphs
and all 50,613 patch VTP files. Relocation then copies and checksums
**all three** (`junction_only`, `adaptive`, `dense`) full-volume representations
into `IXI/vascular_graphs/optimal_radius_0p75x_c095/` and
`TopBrain_Data_Release_Batches1n2_081425/vascular_graphs/optimal_radius_0p75x_c095/`.
It moves corresponding exact-boundary patch triplets to each dataset's
`vascular_patches/optimal_radius_0p75x_c095/` folder with dataset-specific
indices, original subject provenance, and the preserved patient split. Only
after every relocated file has been checked are the redundant production
outputs/staging entries removed from `docs/research/artifacts/` and the mixed
temporary patch directory. Research comparisons and their real-data visual
evidence remain available in `docs/research/evidence/`.

The dependent relocation job `14170` was canceled while pending to avoid
waiting in a compute queue for file placement. The verified and resumable
relocation instead runs locally after patch QC; its one-hour Slurm request was
not an estimate of actual copy time.

The complete inventory found 50,613 exact-boundary 64³ patches: node median 7,
P95 100, P99 142, maximum 239. There are 1,271 above 120 nodes and **35 above
192** (all IXI); TopBrain's 28,611 patches have maximum 133. Edge P95 is 101,
maximum 261. This overturns the seven-volume IXI inference that 192 queries
cover *all* cases while confirming that 192 suffices for TopBrain. Machine-
readable inventory, per-modality/split breakdown, every overflowing sample,
and a full-tail distribution plot are in
[`evidence/full_dataset_production/`](evidence/full_dataset_production/).

Two additional production volumes outside the original 14-volume selection
subset have graph-only full-volume screenshots and interactive HTML, colored by
termination, degree-2 subdivision, and junction: IXI033 (1,198 nodes / 1,305
edges) and TopBrain CT-010 (287 nodes / 290 edges). These are qualitative QC,
not proof that segmentation-derived false connections are absent. Views are in
[`evidence/full_dataset_production/sample_views/`](evidence/full_dataset_production/sample_views/).

### E11 — Exhaustive patch-grid eligibility on the relocated real datasets

The patch generator chooses per-axis endpoints with `ceil((length-64)/40)`
intervals and rounded evenly spaced starts, then takes the Cartesian product;
it does **not** choose random positions or preselect foreground. The source
volumes are graphed first and then cropped; 64³ crops overlap by at least 24
voxels per axis where there are consecutive starts. An independent scan of the
relocated `patch_index.csv` files identified 30,778 graph-positive patches,
18,903 masks with zero foreground, and 932 with foreground but no graph edge
(928 without nodes, four more with nodes but no edges). All 932 corresponding
real segmentation crops were reopened and checked against the stored foreground
counts. In 922/932, no segmented foreground lies ≥8 voxels from all patch
faces; the other ten need closer review. This *suggests* a boundary-cropping
explanation for many cases, not proof that the ten or any other case has a
correct full-volume graph. Per-dataset counts, individual exception rows with
bounding boxes/depth, and the rendered comparison are in
[`evidence/full_dataset_production/patch_eligibility/`](evidence/full_dataset_production/patch_eligibility/).
The same folder includes real segmentation projections of a two-voxel
boundary remnant and a 3,550-voxel graph-free slab, to help decide whether
mask-only negatives deserve their own treatment.

## Change log

- 2026-09-18: Created the tracker and organized the research documents under
  `docs/research/`.
- 2026-09-18: Completed T01 by exporting pure rings as simple three-edge cycles
  and adding topology regression coverage.
- 2026-09-18: Completed T02 by enforcing the two documented annotation
  overrides and recording the selected source and variant in both provenance
  outputs.
- 2026-09-18: Completed T03 by emitting three explicit representation products,
  retaining adaptive as the automatically selected patch-training target, and
  validating the layout end to end.
- 2026-09-18: Completed T04 by defining `dense` as the immutable centerline
  before smoothing and RDP, recording the policy in provenance, and bumping the
  representation schema so old completion markers cannot skip regeneration.
- 2026-09-19: Replaced greedy RDP plus repair with exact constrained branch
  simplification, selected radius-0.75× on 14 real volumes, and retained uniform
  192-query handling rather than patch-specific target changes.
- 2026-09-20: Prepared the IXI-only 120-versus-192 query benchmark. Seven
  patient-split real volumes yielded 884 64-cubed patches; 59 exceed 120 nodes,
  none exceeds 192, and the maximum is 177. Added a paired full-training-step
  benchmark, overflow smoke test, selection tables, and visualization under
  [`evidence/ixi_query_benchmark/`](evidence/ixi_query_benchmark/).
- 2026-09-20: Completed T10 on one H100 with 100 full training steps per case.
  Moving from 120 to 192 queries increased mean step time by 2.02%, the 5%-
  trimmed mean by 1.53%, and absolute peak allocation by 0.101 GiB (3.33%). The
  maximum real IXI patch (177 nodes) completed successfully. Raw steps, GPU
  telemetry, logs, and plots are recorded in
  [`evidence/ixi_query_benchmark/`](evidence/ixi_query_benchmark/).
- 2026-09-20: Estimated the proposed 50-epoch Plants+TopBrain pretraining and
  100-epoch TopBrain specialization schedule from the loader/sampler semantics,
  manifest-derived patch inventory, and measured H100 step cost. The current
  one-to-four-H100 launcher layout is approximately 9.5 compute hours, with a
  7.3–13.0-hour planning range before queue time. Assumptions and visualization
  are in [`evidence/topbrain_training_time_estimate/`](evidence/topbrain_training_time_estimate/).
- 2026-09-20: Submitted selected-policy extraction for all 220 IXI/TopBrain
  volumes as job `2192366` and dependent exact-boundary patch generation plus
  production QC as job `2192776`.
- 2026-09-21: Finished dataset-folder relocation locally and completed the
  independent 50,613-patch foreground/graph audit. Future Slurm wrappers
  default to dataset-local production output; the loader exposes an optional
  index-validated graph-positive patch selection, with no training config
  altered. No relocation or additional extraction job was submitted.
- 2026-09-21: Cleared superseded staging artifacts and exploratory visual
  dumps from `docs/research/`. Retained only foundational checks, compact
  adaptive-policy comparisons, the query benchmark final report, and the
  full-dataset production/QC evidence. The current source of production data
  is the dataset-local `vascular_graphs/` and `vascular_patches/` layout.
- 2026-09-21: Curated **training only** in both dataset-local patch roots.
  IXI retains 13,736 graph-positive foreground training crops and archives
  2,092 (1,966 empty mask; 126 foreground without graph); TopBrain retains
  7,379 and archives 11,124 (10,649 empty mask; 475 foreground without
  graph). Each rejected raw/seg/VTP triplet is recoverable under
  `excluded_train/`; `patch_index.csv` still describes the original exhaustive
  grid and `training_view.json`, `active_train_index.csv`, and
  `excluded_train_index.csv` document the active view. The original val/test
  grids were not filtered, so negatives remain represented in evaluation.
  The existing grid is 64³ with maximum stride 40, giving at least 24/64
  (37.5%) overlap on each axis. No resampling is needed for 5–10k training
  patches: TopBrain already supplies 7,379; IXI can take a reproducible 8,000
  of its 13,736 train patches without changing the volume-level split. The
  IXI-only training config `configs/finetune_ixi_vessels_q256.yaml` makes the
  latter choice with 256 uniform object queries, preserving all val/test.
  See [`evidence/ixi_capacity_192_256/`](evidence/ixi_capacity_192_256/) for
  the training-count plot and the 192-versus-256 IXI-only capacity experiment.
- 2026-09-21: Job `22067` completed the IXI-only paired 192/256-query test
  (60 full steps per capacity, batch eight). At 256 queries mean step time was
  +7.51% and peak allocated GPU memory +0.085 GiB; a real 239-node IXI crop
  completed a full 256-query training step. This verifies compute/capacity,
  not convergence. Exact samples, visualization and timings are in
  [`evidence/ixi_capacity_192_256/`](evidence/ixi_capacity_192_256/).
