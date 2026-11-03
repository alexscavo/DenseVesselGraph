# Ordered Research and Implementation Roadmap

> **Research consolidation, 23 October 2026:** [Consolidated current roadmap](../../research/roadmap.md). This document retains its detailed and dated historical context.

## Goal

This file lists the recommended order of work for the next stage after GrEVE.

The guiding principle is to change **one conceptual layer at a time** so that improvements remain attributable.

---

# Phase 0 — Freeze the current baseline

Before changing anything:

1. Keep the current RelationFormer/GrEVE architecture.
2. Keep the current Hungarian matcher.
3. Keep the current training losses, including the current focal-loss modification used to reduce overprediction.
4. Record baseline metrics and qualitative examples on the existing datasets.
5. Preserve the current checkpoints for direct comparison.

Do not add new loss terms yet.

---

# Phase 1 — Finalize the adaptive GT representation

## 1.1 Build the dense-reference representation

For every branch, retain the original dense centerline/polyline before simplification.

Store:

- topological endpoint/junction identities;
- ordered dense centerline points;
- physical voxel spacing / world coordinates where available.

This dense representation is used for representation-quality and end-to-end anatomical evaluation.

## 1.2 Build the adaptive representation

Generate the compact training target by retaining:

- endpoints;
- junctions;
- only the degree-2 control points required by the adaptive simplification policy.

The goal is not perfect centerline reproduction at any cost.

The goal is a useful compromise between:

\[
\text{geometry fidelity}
\leftrightarrow
\text{graph complexity}.
\]

## 1.3 Create comparison representations

At minimum:

1. junction/endpoints only;
2. adaptive representation;
3. dense/all-degree-2 reference where computationally feasible.

---

# Phase 2 — Implement representation-quality metrics first

Before retraining any model, evaluate the target representations themselves.

Implement:

1. total node count \(|V|\);
2. total edge count \(|E|\);
3. optional nodes per physical vessel length;
4. Curve Precision / Recall / F1 at several physical distance thresholds;
5. Average Centerline Distance (ACD);
6. HD95;
7. branch-wise length error;
8. branch-wise tortuosity error;
9. branch-angle error with caution.

## Deliverable

A representation study showing:

\[
\text{compactness}
\quad\text{vs}\quad
\text{geometric/morphological fidelity}.
\]

Do not proceed to architectural modifications until the adaptive representation itself is shown to be sensible.

---

# Phase 3 — Implement topology preprocessing and Branch F1

## 3.1 Degree-2 contraction

Implement a deterministic graph transformation that:

- identifies topological anchors with \(\deg(v)\neq2\);
- contracts every maximal degree-2 chain;
- preserves the ordered branch polyline separately.

## 3.2 Spatial anchor matching

Match predicted and GT topological anchors within a physical distance threshold.

Use physical units rather than voxel thresholds.

## 3.3 Branch Connectivity Precision / Recall / F1

A branch is topologically correct when matched topological anchors are connected by the corresponding contracted branch.

Do not require the branch trajectory to be geometrically accurate for this metric.

## 3.4 Keep Betti metrics

Continue computing:

- \(\beta_0\) error;
- \(\beta_1\) error.

Treat them as global topology diagnostics, not sufficient topology evaluation by themselves.

---

# Phase 4 — Validate the metrics on handcrafted failure cases

Before trusting any metric on trained models, create synthetic graph examples.

Test:

1. same curve, different degree-2 subdivision;
2. same topology, shifted geometry;
3. missing branch;
4. extra branch;
5. redundant nearby/crossing edges;
6. small break;
7. wrong connection between nearby vessels;
8. badly misplaced but structurally similar junction;
9. same Betti numbers but wrong branch pairing;
10. unnecessary degree-2 overprediction.

Verify expected behavior of:

- Branch F1;
- Curve F1;
- ACD;
- HD95;
- Betti;
- SMD;
- node/edge mAP where relevant;
- complexity.

This phase is essential.

---

# Phase 5 — Retrain with adaptive GT, keeping optimization unchanged

Run the clean experiment:

\[
\boxed{
\text{same model}
+
\text{same matcher}
+
\text{same losses}
+
\text{adaptive GT}
}
\]

Do not change:

- matcher;
- relation formulation;
- node losses;
- DA losses;
- architecture.

This isolates the effect of the representation.

Compare against the existing baseline and, if feasible, junction-only and denser GT variants.

---

# Phase 6 — Evaluate model quality against adaptive GT

Primary comparison:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{adaptive GT}}.
\]

Headline metrics:

1. Branch Connectivity F1;
2. Curve F1;
3. ACD.

Legacy/diagnostic metrics:

- node mAP/mAR;
- edge mAP/mAR;
- SMD;
- Betti;
- 2D TOPO where available.

Interpret failures by category:

- poor Branch F1 + good geometry → connectivity problem;
- good Branch F1 + poor ACD/Curve F1 → geometry/localization problem;
- many extra nodes/edges → overprediction/compactness problem;
- both topology and geometry poor → more fundamental detection/matching issue.

---

# Phase 7 — Evaluate end-to-end anatomical fidelity

Compare:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{dense/reference}}.
\]

Always show the adaptive oracle:

\[
G_{\text{adaptive GT}}
\quad\text{vs}\quad
G_{\text{dense/reference}}.
\]

Report:

- Curve F1;
- ACD;
- HD95;
- morphology diagnostics;
- Branch F1 where topology is consistently defined.

Report the oracle and prediction side by side rather than inventing a normalized aggregate score initially.

---

# Phase 8 — Add APLS/path evaluation

Implement patch-level APLS once the simpler metric stack is stable.

Interpret it as:

\[
\text{local path consistency within the cropped graph}.
\]

Later, after whole-volume reconstruction:

\[
\text{whole-volume APLS}
\]

becomes the more important global connectivity metric.

Do not use APLS to replace Branch F1; use it to detect the downstream consequences of local connectivity mistakes.

---

# Phase 9 — Only then decide whether optimization must change

Use the new metrics to identify the real bottleneck.

Do not add new losses simply because new metrics exist.

## If geometry is the bottleneck

Investigate replacing/reformulating node-coordinate supervision for degree-2 control nodes with a more geometry-aware objective.

Possible future direction:

- retain strong coordinate supervision for endpoints/junctions;
- use curve-aware supervision for degree-2 geometric control points.

## If edge/connectivity prediction is the bottleneck

Investigate the current edge/relation objective first.

Possible directions:

- hard-negative relation mining;
- better handling of false shortcut edges;
- relation-loss balancing;
- focal-loss tuning.

## If matching is the bottleneck

Proceed to Phase 10 below.

## If long-range paths are the bottleneck

Investigate a path/global structural objective only after local topology is already good.

---

# Phase 10 — Matcher experiments, in order

Do not return directly to FGW.

## 10.1 Degree-aware Hungarian

Add:

\[
\lambda_d |\hat d_i-d_a^{GT}|
\]

to the Hungarian matching cost.

Evaluate whether nearby ambiguous nodes are matched more consistently.

## 10.2 Role-aware Hungarian

Use coarse roles:

- endpoint;
- intermediate;
- junction.

No node-classification head is required.

## 10.3 Spatially gated structural tie-breaking

Allow structure to influence matching only among spatially plausible/ambiguous candidates.

Use a physical gating radius.

## 10.4 Local structure-aware matching

Apply structural reasoning only to ambiguous local groups rather than globally.

## 10.5 Multi-hop/geodesic descriptors

If degree is insufficient, try:

- shortest-path distance to topological anchors;
- local multi-hop signatures;
- graph geodesic descriptors.

## 10.6 Reconsider FGW only afterward

FGW becomes worth revisiting only if simpler structure-aware Hungarian variants still fail and the new metrics demonstrate a persistent matching problem.

---

# Phase 11 — Experimental diagnostics

After the core metric suite is stable, optionally test:

- Graph GOSPA;
- 3D TOPO;
- radius-normalized geometry metrics;
- density-stratified evaluation.

Keep these secondary until they prove they add distinct information.

---

# Phase 12 — Whole-volume inference

Once patch-level performance is stable:

1. tile the volume deterministically;
2. predict each patch graph;
3. map all patch predictions into global physical coordinates;
4. identify nodes near shared patch faces;
5. match boundary nodes across neighboring patches;
6. merge matched nodes/edges;
7. recompute final global node degrees;
8. remove artificial patch-boundary subdivisions where appropriate;
9. rerun adaptive simplification globally if needed.

Then evaluate the full graph using:

- Branch F1;
- Curve F1;
- ACD;
- Betti;
- APLS;
- total graph size.

Whole-volume evaluation should eventually carry more weight than patch-only evaluation.

---

# Phase 13 — Future architecture: function-valued edges

Only after the adaptive-node approach is well understood should the more ambitious representation be attempted.

Target formulation:

\[
G=(V_T,E_\Gamma)
\]

where:

- \(V_T\) contains only endpoints/junctions;
- every edge stores a continuous branch geometry \(\Gamma_{ij}(t)\).

Possible implementation:

- cubic Bézier;
- B-spline;
- piecewise spline;
- edge-geometry decoder attending to image features.

Later extend each edge with radius:

\[
e_{ij}(t)
=
[x(t),y(t),z(t),r(t)].
\]

This is a separate architectural project and should not be mixed into the first adaptive-GT experiments.

---

# Priority summary

## Do first

1. Freeze baseline.
2. Store dense reference centerlines.
3. Generate adaptive GT.
4. Implement representation metrics.
5. Implement degree-2 contraction + Branch F1.
6. Validate metrics on synthetic failure cases.
7. Retrain unchanged model on adaptive GT.
8. Evaluate the three metric classes.

## Do next, only if justified by results

9. Patch APLS.
10. Whole-volume graph stitching.
11. Whole-volume APLS.
12. Structure-aware Hungarian if matching is shown to be a bottleneck.

## Do later / exploratory

13. Graph GOSPA.
14. 3D TOPO.
15. radius-aware normalized metrics.
16. loss redesign.
17. function-valued edge architecture.
18. edge radius function prediction.

---

# Rule for every later modification

Before introducing a new loss, matcher, or architecture component, answer:

> Which failure mode in the new evaluation suite is this modification intended to fix?

If that answer is not clear, do not add the component yet.
