# Evaluation and Metrics Plan for Adaptive Vascular Graph Extraction

## Purpose

This document consolidates the design decisions reached for the next stage after **GrEVE**.

The goal is to evaluate a vascular graph representation that is:

1. **Topologically meaningful**
2. **Geometrically faithful enough for the intended downstream use**
3. **Compact**, avoiding unnecessary centerline nodes
4. Compatible with the current direct **raw image/volume → graph** RelationFormer-style pipeline
5. Applicable in both **2D and 3D**
6. Suitable both for **patch-level** evaluation now and **whole-volume** evaluation later

The central scientific claim is not only that the model predicts better, but that the **adaptive representation itself is useful** and provides a better trade-off between topology, geometry, and complexity.

---

# 1. Main evaluation philosophy

Evaluation should be separated into three classes.

## Class 1 — Representation quality

No neural network is involved.

Compare the chosen target representation against a denser anatomical reference.

Primary comparison:

\[
G_{\text{adaptive}} \quad \text{vs} \quad G_{\text{dense/reference}}
\]

Questions answered:

- How much geometry is intentionally lost by compression?
- How many graph elements are required?
- Are morphometric quantities preserved?
- Is the adaptive representation a useful middle point between junction-only and dense centerline representations?

This class measures the **quality of the target representation itself**.

---

## Class 2 — Model quality

Compare the model output against the exact representation it was trained to predict.

Primary comparison:

\[
G_{\text{pred}} \quad \text{vs} \quad G_{\text{adaptive GT}}
\]

Questions answered:

- Did the network learn the target representation?
- Are the correct topological events detected?
- Are the correct branches connected?
- Does the predicted embedded graph follow the adaptive target geometry?

A perfect model should in principle be able to obtain a perfect score here.

This class should be the **main model-performance evaluation**.

---

## Class 3 — End-to-end anatomical fidelity

Compare the final predicted graph against the dense anatomical reference.

Primary comparison:

\[
G_{\text{pred}} \quad \text{vs} \quad G_{\text{dense/reference}}
\]

Questions answered:

- How close is the final output to the underlying anatomy?
- How much total error remains after both representation compression and model prediction?
- Are long-range paths preserved?

This class must always be interpreted together with the Class-1 oracle:

\[
G_{\text{adaptive GT}} \quad \text{vs} \quad G_{\text{dense/reference}}
\]

because the adaptive target intentionally discards some fine-scale geometry.

---

# 2. Representation error versus prediction error

The final discrepancy should be interpreted as having two components:

\[
\text{final anatomical error}
\approx
\text{representation error}
+
\text{prediction error}
\]

For example, if:

\[
D(G_{\text{adaptive}},G_{\text{dense}})=0.6\,\text{mm}
\]

and

\[
D(G_{\text{pred}},G_{\text{dense}})=0.9\,\text{mm},
\]

then the additional model-induced discrepancy is approximately:

\[
\Delta_{\text{model}} = 0.3\,\text{mm}.
\]

Do not force this gap into a new headline normalized score initially. Report oracle and prediction side by side.

---

# 3. Topology and geometry must be separable

A graph can be:

- topologically correct but geometrically inaccurate;
- geometrically close but topologically wrong;
- wrong in both ways.

The evaluation should expose these cases separately.

Example:

```text
GT:
A ~~~~~~~~~~~~~ B

Prediction:
A' ----------- B'
```

If matched \(A'\) and \(B'\) are connected correctly, then:

- branch topology may be correct;
- branch geometry may be poor.

The reverse can also happen: an edge distribution may lie close to the vessel but connect the wrong anatomical events.

Therefore topology metrics must not be replaced by geometry metrics, and vice versa.

---

# 4. Topological node roles

Topological node roles can be inferred from graph degree after graph construction:

\[
\deg(v)=1 \Rightarrow \text{endpoint}
\]

\[
\deg(v)=2 \Rightarrow \text{intermediate/control node}
\]

\[
\deg(v)\ge3 \Rightarrow \text{junction}
\]

For the current work, no additional node-type classification head is required.

Important distinction:

- endpoint/junction nodes are anatomically meaningful;
- degree-2 nodes are primarily geometric control points.

Legacy node mAP/mAR can still evaluate all predicted nodes for comparability, but the new topology metrics should focus on the topological graph after degree-2 contraction.

---

# 5. Degree-2 contraction for topology evaluation

For topology evaluation, contract every maximal degree-2 chain.

Example:

\[
A-p_1-p_2-p_3-B
\]

becomes:

\[
A-B
\]

for the **topological graph**.

However, preserve the ordered coordinates:

\[
[A,p_1,p_2,p_3,B]
\]

as the **branch polyline** for geometry evaluation.

This gives two simultaneous representations:

### Contracted topology

Used for:

- branch connectivity;
- Betti numbers;
- Graph GOSPA experiments;
- APLS/path metrics.

### Embedded branch polyline

Used for:

- Curve F1;
- Average Centerline Distance;
- HD95;
- branch length;
- tortuosity;
- angle-related morphology.

The contraction therefore removes arbitrary subdivision from topology evaluation without discarding geometry.

---

# 6. Spatially grounded topology versus structural topology

Two topology notions should be distinguished.

## 6.1 Primary: spatially grounded topology

A predicted topological node must first be matched spatially to a GT topological node within a physical distance tolerance.

Then connectivity is evaluated.

If a predicted junction has correct graph degree and correct abstract neighbors but lies too far from the true anatomical junction, it fails the primary spatial match.

This remains the headline topology notion.

---

## 6.2 Secondary: structural topology diagnostic

A badly misplaced node may still reveal that the network recovered the correct abstract connectivity pattern.

A secondary structural diagnostic may therefore attempt to separate:

- node localization error;
- missed/false nodes;
- edge/connectivity error.

**Graph GOSPA** is a candidate for this role because it explicitly decomposes graph discrepancy into localization, missed/false nodes, and edge mismatch.

It should be investigated experimentally after the simpler metrics are validated.

It should not initially replace spatially grounded Branch F1.

---

# 7. Proposed core topology metric: Branch Connectivity F1

After contracting degree-2 chains:

1. Extract topological anchors:
   \[
   \deg(v)\neq2
   \]

2. Spatially match predicted anchors to GT anchors using a physical distance threshold.

3. Each maximal degree-2 chain becomes one branch between topological anchors.

4. A GT branch \(A-B\) is a true positive when matched anchors \(A'-B'\) are connected by the corresponding contracted predicted branch.

5. Missing GT branches are false negatives.

6. Extra predicted branches are false positives.

Then compute:

\[
P_{\text{branch}},
\quad
R_{\text{branch}},
\quad
F1_{\text{branch}}.
\]

## Important rule

Branch Connectivity F1 should **not require geometric branch overlap**.

If anchors are correctly matched and connectivity is correct, the branch is topologically correct even if its trajectory is geometrically inaccurate.

Geometry is evaluated separately.

A stricter spatial branch F1 can optionally be reported later as a combined end-to-end metric.

---

# 8. Betti metrics

Keep:

\[
\beta_0 = \text{number of connected components}
\]

\[
\beta_1 = \text{number of independent cycles}
\]

and report errors such as:

\[
|\Delta\beta_0|,
\qquad
|\Delta\beta_1|.
\]

These are:

- dimension-independent;
- useful global sanity checks;
- purely/comparatively topological.

However, they are weak by themselves because very different graphs can share the same Betti numbers.

They should remain diagnostic/global metrics rather than headline topology metrics.

---

# 9. TOPO score

The classic TOPO concept is not fundamentally restricted to 2D, but the commonly used implementation is strongly 2D-specific.

A 3D version would require replacing:

- 2D polyline operations;
- 2D spatial matching;
- circle-based neighborhoods;

with:

- 3D polyline sampling;
- 3D KD-tree search;
- Euclidean 3D matching;
- spherical/3D local neighborhoods.

Decision:

- do **not** prioritize 3D TOPO now;
- first validate Branch F1 + geometric metrics + Betti + path metrics;
- consider 3D TOPO later if it adds genuinely distinct information.

---

# 10. Geometry metrics

The geometry metrics should evaluate the **embedded graph**, not whether individual degree-2 control points land at identical coordinates.

## 10.1 Curve Precision / Recall / F1

Represent GT and prediction as unions of piecewise-linear edge/branch curves.

Using physical distance tolerance \(\tau\):

\[
P_{\text{curve}}(\tau)
=
\frac{
|\{p\in C_P:d(p,C_G)\le\tau\}|
}{
|C_P|
}
\]

\[
R_{\text{curve}}(\tau)
=
\frac{
|\{g\in C_G:d(g,C_P)\le\tau\}|
}{
|C_G|
}
\]

\[
F1_{\text{curve}}(\tau)
=
\frac{2PR}{P+R}.
\]

Distances should be computed to the continuous polyline/segments rather than only to graph vertices.

This avoids penalizing a compact representation simply because the dense reference has many more sampled centerline points.

---

## 10.2 Average Centerline Distance (ACD)

Use a symmetric curve distance:

\[
ACD =
\frac{1}{2}
\left[
\frac{1}{|C_G|}\sum_{g\in C_G} d(g,C_P)
+
\frac{1}{|C_P|}\sum_{p\in C_P} d(p,C_G)
\right].
\]

Use physical units, preferably mm.

This is the main continuous geometric error metric.

---

## 10.3 HD95

Use a 95th-percentile Hausdorff-style distance rather than the raw maximum.

Purpose:

- detect severe local deviations;
- remain less sensitive to one extreme outlier than full Hausdorff distance.

Keep as supplementary/diagnostic rather than headline.

---

# 11. Geometric tolerance policy

Initial decision:

Use **physical distance thresholds**, not voxel thresholds.

For Curve F1, initially report several thresholds:

\[
\tau_1,\tau_2,\tau_3
\]

instead of selecting one arbitrary value immediately.

Example concept:

- strict tolerance;
- medium tolerance;
- relaxed tolerance.

Exact values should be selected after inspecting physical vessel scales across datasets.

## Radius-normalized tolerance: future option

A future supplementary metric may normalize spatial error by vessel radius:

\[
d_{\text{norm}} = \frac{d}{r}
\]

or use a local radius-aware threshold.

This is scientifically plausible because a 1 mm displacement has different meaning for a 0.5 mm vessel versus a 5 mm vessel.

However, avoid using **dataset-average radius as the only tolerance** initially because:

- it reduces direct cross-dataset comparability;
- different branches within one dataset can have very different radii;
- a thick-vessel dataset would receive a systematically more permissive metric.

If radius-aware evaluation is added later, prefer **local branch/vessel radius** and report it as a complementary normalized metric alongside the raw physical-mm metric.

## Foreground/background ratio

Do **not** change the spatial tolerance based on foreground/background density.

Dense patches may be harder, but density does not change what counts as an anatomically correct spatial localization.

Instead, if density is important:

- stratify results by graph/foreground density;
- analyze performance as a function of density.

Do not make the metric more or less permissive depending on patch density.

---

# 12. Morphology preservation metrics

Representation quality should measure whether compression preserves useful branch morphology, not only centerline proximity.

Compute morphology **branch-wise first**, then aggregate.

Recommended branch-level quantities:

## Relative branch-length error

\[
\epsilon_L =
\frac{|L_{\text{rep}}-L_{\text{dense}}|}
{L_{\text{dense}}}
\]

## Tortuosity error

Use the same tortuosity definition consistently across representations.

For a simple path-length/chord formulation:

\[
T = \frac{L_{\text{path}}}{\|A-B\|}
\]

then compare representation versus dense reference.

## Branch-angle error

At matched junctions, compare branch directions/angles using a physically meaningful local tangent estimate.

Use with caution because angle estimation depends on how much local centerline context is used.

Decision:

- ACD + Curve F1 = core geometric representation metrics;
- branch length, tortuosity, and angle = morphology diagnostics;
- HD95 = tail-error diagnostic.

---

# 13. Representation complexity

Primary complexity metric:

\[
\boxed{|V|}
\]

the total number of graph nodes.

Why total nodes rather than only degree-2 nodes:

- junction-only methods should not get a trivially special complexity definition;
- dense centerline methods can be compared directly;
- RelationFormer computational burden is strongly related to the number of object queries/nodes;
- the final question is how many elements are required to represent the vascular structure.

Also report:

\[
|E|
\]

descriptively.

For cross-dataset comparison, consider normalized total complexity:

\[
\frac{|V|}{L_{\text{reference}}}
\]

in nodes per physical vessel length.

Degree-2 node count may remain a diagnostic but is not the headline complexity metric.

---

# 14. Representation quality as a rate-distortion problem

The representation objective is not:

\[
\min \text{geometric error}
\]

because the trivial solution is to keep almost every dense centerline point.

Instead the problem is:

\[
\text{geometry/morphology fidelity}
\quad\leftrightarrow\quad
\text{representation complexity}.
\]

Compare at least:

1. junction/endpoints only;
2. adaptive representation;
3. dense/all-degree-2 representation where feasible.

Visualize the Pareto trade-off:

\[
|V|
\quad \text{vs} \quad
ACD
\]

and/or:

\[
|V|
\quad \text{vs} \quad
F1_{\text{curve}}.
\]

Morphology errors can be shown as additional diagnostics.

---

# 15. SMD: keep for legacy comparison, not as a core metric

Street Mover Distance compares spatial edge distributions through sampled points and optimal transport.

Useful property:

- subdividing the same curve into more degree-2 nodes should have little effect.

Problem:

- SMD does not explicitly understand branch identity, graph topology, or representation redundancy.

A pathological prediction can overpredict several nearby/crossing edges and sometimes obtain a lower SMD than a simpler single edge that is slightly displaced.

Therefore:

\[
\text{lower SMD} \not\Rightarrow \text{better graph}
\]

SMD should be interpreted as a geometric distribution similarity score.

Decision:

- **short term:** keep SMD for GrEVE/RelationFormer comparability;
- **medium term:** demote it to legacy/diagnostic status;
- **long term:** potentially remove it once the new metric suite is validated.

The new evaluation suite must independently penalize:

- wrong topology;
- false edges;
- redundant nodes;
- geometric error.

---

# 16. Legacy RelationFormer metrics

Continue reporting initially:

- Node mAP
- Node mAR
- Edge mAP
- Edge mAR
- SMD
- 2D TOPO where available

Purpose:

- continuity with GrEVE;
- direct comparison with RelationFormer/Berger;
- monitoring whether improvements come at the cost of conventional detection performance.

Interpretation should change:

- node mAP/mAR are meaningful for topological anchors;
- exact degree-2 placement is less scientifically important;
- edge bounding-box mAP/mAR are legacy detection metrics rather than the final definition of geometric correctness.

---

# 17. Overprediction and redundancy

A prediction can have:

- correct geometry;
- correct topology;
- too many unnecessary nodes.

This should not contaminate geometry or topology scores.

Instead:

- legacy node detection against adaptive GT naturally penalizes unmatched extra nodes;
- total node count explicitly captures representation inefficiency;
- false branches/edges affect Branch Precision;
- focal loss may reduce overprediction during training, but evaluation must detect it independently.

Current project-specific note:

The present implementation uses **focal loss** to reduce prediction overproduction rather than relying only on standard cross-entropy for the relevant classification component.

This is a training design choice, not a replacement for explicit evaluation of redundancy.

---

# 18. Long-range connectivity and APLS

APLS-style path similarity is useful because a small local break may produce a large functional/global graph error.

Example:

```text
GT:
A -------- B -------- C

Prediction:
A -------- B    ----- C
```

The geometric discrepancy may be small, but the path \(A\rightarrow C\) is destroyed.

## Use at patch level?

Yes, it can still be useful.

Within a patch there may be long-range relations and path structure.

### Downsides at patch level

- paths are artificially truncated by crop boundaries;
- maximum path length is limited by patch size;
- a path that should continue outside the patch cannot be assessed globally;
- results may depend on the patch tiling.

### Decision

Use APLS in both settings if implementation effort is reasonable, but distinguish interpretation:

- **Patch APLS:** local/path consistency inside a cropped graph;
- **Whole-volume APLS:** global connectivity/path preservation.

Whole-volume APLS is the more important final use.

For patch APLS, both prediction and GT must use the same clipped patch graph convention.

---

# 19. Graph GOSPA

Graph GOSPA is an experimental candidate for structural error decomposition.

Potentially useful decomposition:

- localization of matched nodes;
- false nodes;
- missed nodes;
- edge mismatch.

This directly addresses the question:

> Is a junction wrong because it is spatially misplaced, or because its connectivity is wrong?

Implementation decision:

- do not make it a headline metric initially;
- test it after degree-2 contraction;
- validate behavior on synthetic failure cases;
- retain only if it adds clear information beyond Branch F1 + node localization metrics.

---

# 20. Primary metric architecture

The current recommended hierarchy is:

## Class 1 — Representation quality

### Headline
- Total nodes \(|V|\)
- ACD
- Curve F1 at multiple physical tolerances

### Diagnostic
- \(|E|\)
- HD95
- branch-length error
- tortuosity error
- branch-angle error
- optional nodes per physical vessel length

---

## Class 2 — Model quality

Comparison:

\[
G_{\text{pred}} \text{ vs } G_{\text{adaptive GT}}
\]

### Headline
- Branch Connectivity F1
- Curve F1
- ACD

### Legacy / diagnostic
- Node mAP/mAR
- Edge mAP/mAR
- SMD
- Betti errors
- 2D TOPO
- experimental Graph GOSPA
- patch APLS

---

## Class 3 — End-to-end anatomical fidelity

Comparison:

\[
G_{\text{pred}} \text{ vs } G_{\text{dense/reference}}
\]

Always show the adaptive oracle:

\[
G_{\text{adaptive GT}} \text{ vs } G_{\text{dense/reference}}
\]

### Headline
- Curve F1
- ACD
- Branch Connectivity F1 where dense topology is defined consistently
- APLS, especially at whole-volume level

### Diagnostic
- HD95
- Betti errors
- morphology preservation
- model-to-oracle performance gap

---

# 21. Training losses versus evaluation metrics

This is a critical distinction.

## 21.1 What is a training loss?

A training loss is part of the optimization objective.

It must provide a useful gradient to update network parameters.

In the RelationFormer/Berger formulation, the training objective contains terms such as:

- object/node localization regression;
- box/generalized-IoU terms;
- object classification;
- relation/edge classification;
- regularized relation/edge sampling;
- domain-adaptation losses in the cross-domain framework.

Berger et al. combine:

\[
L_{\text{reg}}
+
L_{\text{gIoU}}
+
L_{\text{cls}}
+
L_{\text{Reslt}}
+
L_{\text{DA}}
\]

with the DA component containing image-level, graph-level, and consistency terms.

Hungarian matching associates predictions to GT objects before the matched-object losses are evaluated.

The current project additionally uses focal loss in the relevant classification setting to reduce overprediction.

---

## 21.2 What is an evaluation metric?

An evaluation metric does **not** automatically enter the loss.

It is computed during validation/test to answer whether the final prediction is good according to a meaningful criterion.

The following proposed metrics should initially be **evaluation-only**:

- Branch Connectivity F1
- Curve Precision/Recall/F1
- ACD
- HD95
- Betti errors
- APLS
- Graph GOSPA
- total node count
- morphology preservation errors
- SMD
- TOPO

Most are:

- non-differentiable;
- threshold-based;
- discrete;
- dependent on graph matching;
- dependent on shortest paths/global connectivity;
- unsuitable as direct gradient objectives.

Therefore they should **not simply be inserted into the loss function**.

---

## 21.3 Could some become losses later?

Yes, but only through carefully designed differentiable surrogates.

Examples:

- a differentiable point-to-segment / Chamfer-style geometry loss could encourage edge geometry;
- soft topology/connectivity losses could be investigated;
- differentiable persistence/topological losses could target connectivity;
- path-consistency surrogates could potentially be designed.

Do not add these initially.

First:

1. keep training as stable as possible;
2. change the target representation;
3. evaluate with the new metric suite;
4. identify the dominant failure mode;
5. only then consider a new loss that specifically targets that failure.

This avoids making the already multi-term training objective harder to balance without evidence that another loss is needed.

---

# 22. Why the new metrics should remain out of the loss initially

The current scientific question is:

> Does a better graph representation make the existing direct image-to-graph framework produce a more useful vascular graph?

If target representation, model architecture, loss, and evaluation all change simultaneously, improvements become difficult to attribute.

Preferred first experiment:

\[
\text{same architecture}
+
\text{same training strategy/loss}
+
\text{new adaptive GT}
+
\text{new evaluation}
\]

This isolates the representation effect.

Only after that should optimization changes be considered.

---

# 23. Validation protocol for the metrics themselves

Before evaluating trained models, create synthetic graph perturbations with known failure types.

## Test cases

1. Same embedded curve, different degree-2 subdivision
2. Same topology, shifted geometry
3. Missing branch
4. Extra branch
5. Redundant nearby/crossing edges
6. Small break causing disconnection
7. Wrong connection between nearby vessels
8. Correct connectivity with badly misplaced junction
9. Same Betti numbers but wrong branch pairing
10. Overprediction of unnecessary degree-2 nodes

## Expected behavior

### Same subdivision only
- Branch F1: unchanged
- Curve F1: unchanged
- ACD: unchanged
- SMD: approximately unchanged
- node mAP: may change
- complexity: changes

### Wrong curvature / shifted geometry
- Branch F1: may remain correct
- Curve F1: decreases
- ACD: worsens
- Betti: unchanged

### Missing branch
- Branch recall: decreases
- Curve recall: decreases
- APLS: worsens
- Betti may change

### Extra branch
- Branch precision: decreases
- Curve precision may decrease
- complexity increases

### Small break
- Curve metrics may change only mildly
- Branch connectivity worsens
- APLS should worsen strongly
- \(\beta_0\) may change

### Wrong connection near correct vessel geometry
- SMD may remain deceptively good
- Curve metrics may remain reasonably good
- Branch F1 must decrease
- APLS should decrease

### Redundant nearby/crossing edges
- SMD may improve or remain misleadingly low
- Branch precision should worsen if extra branches survive contraction
- total graph complexity should increase
- node/edge legacy precision may worsen

This synthetic validation is essential before trusting the new evaluation suite.

---

# 24. Implementation order

## Phase 1 — representation evaluation

Implement:

- dense reference branch/polyline storage;
- adaptive target construction;
- total node count;
- ACD;
- Curve F1 at multiple physical thresholds;
- HD95;
- branch-wise length/tortuosity/angle errors.

Compare:

- junction-only;
- adaptive;
- dense/all-degree-2.

---

## Phase 2 — topology-aware model evaluation

Implement:

- degree-2 contraction;
- topological anchor extraction;
- physical-distance anchor matching;
- Branch Precision/Recall/F1;
- existing Betti metrics;
- synthetic metric unit tests.

---

## Phase 3 — path/global metrics

Implement:

- patch APLS;
- later whole-volume APLS;
- optionally Graph GOSPA;
- optionally 3D TOPO only if the simpler suite leaves a real gap.

---

## Phase 4 — whole-volume evaluation

After global graph stitching is available:

- recompute global topology;
- remove artificial patch-boundary subdivisions where appropriate;
- evaluate full-volume Branch F1;
- whole-volume Curve F1 / ACD;
- Betti;
- APLS;
- total graph size.

Whole-volume metrics should ultimately carry more weight than patch metrics for the final downstream claim.

---

# 25. Recommended reporting structure

Avoid one giant table where every metric has equal status.

Use three blocks.

## Table A — Representation study

| Representation | Nodes ↓ | ACD ↓ | Curve F1 ↑ | Length error ↓ | Tortuosity error ↓ |
|---|---:|---:|---:|---:|---:|
| Junction-only | | | | | |
| Adaptive | | | | | |
| Dense | | | | | |

Optional:
- HD95
- angle error
- nodes/mm

---

## Table B — Model quality against adaptive GT

| Model | Branch F1 ↑ | Curve F1 ↑ | ACD ↓ | Node mAP ↑ | Edge mAP ↑ | SMD ↓ | Betti error ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|

Legacy metrics remain visible but are not the main scientific definition of graph quality.

---

## Table C — End-to-end anatomical fidelity

| Model / Oracle | Curve F1 ↑ | ACD ↓ | HD95 ↓ | Branch F1 ↑ | APLS ↑ |
|---|---:|---:|---:|---:|---:|
| Adaptive GT oracle | | | | | |
| Prediction | | | | | |

This directly exposes the representation ceiling and the additional model-induced loss.

---

# 26. Current decisions summary

The following design decisions are considered settled for the first implementation:

- Scientific goal: prove both representation quality and model benefit.
- Use three metric classes: representation, model, end-to-end.
- Keep topology and geometry conceptually separate.
- Use spatially grounded topology as the primary topology notion.
- Keep structural-topology analysis as a secondary diagnostic.
- Contract degree-2 chains for topology evaluation.
- Preserve their polyline geometry for geometric evaluation.
- Branch Connectivity F1 is the preferred new core topology metric.
- Betti metrics remain global diagnostics.
- 3D TOPO is deferred.
- Curve F1 + ACD are the preferred new core geometry metrics.
- HD95 is supplementary.
- Use physical distance thresholds.
- Evaluate Curve F1 at multiple thresholds initially.
- Consider local radius-normalized evaluation later, but do not replace raw physical-mm metrics.
- Do not adapt tolerance to patch foreground density.
- Morphology is evaluated branch-wise first.
- Main morphology diagnostics: length, tortuosity, branch angle.
- Total node count is the main complexity measure.
- Edge count is reported descriptively.
- SMD remains legacy in the short term and may eventually be removed.
- Legacy node/edge mAP/mAR remain for comparability.
- Overprediction must be detected independently of the focal-loss training choice.
- APLS is useful at both patch and whole-volume scale, with different interpretation.
- Graph GOSPA is experimental, not headline.
- New metrics are evaluation metrics, not automatically training losses.
- Keep the current training objective unchanged initially unless evaluation reveals a specific failure that motivates a new differentiable loss.

---

# 27. Literature anchors to keep in mind

## RelationFormer
Useful for:
- current node/edge detection setup;
- Hungarian matching;
- SMD / TOPO / node-edge mAP/mAR legacy metrics;
- baseline object/relation loss formulation.

## Berger et al., Cross-Domain and Cross-Dimension Learning for Image-to-Graph Transformers
Useful for:
- regularized edge sampling;
- domain-adaptation loss terms;
- current transfer-learning optimization structure.

## StreetMover / Image-Conditioned Graph Generation
Useful for:
- understanding SMD;
- understanding why SMD alone cannot measure graph plausibility or compactness.

## Sat2Graph / APLS literature
Useful for:
- path-based connectivity evaluation;
- TOPO and APLS interpretation.

## Trexplorer Super
Useful for:
- branch-level graph evaluation;
- Betti metrics;
- 3D centerline/tree evaluation.

## Graph GOSPA
Useful for:
- decomposing graph discrepancy into localization, false/missed nodes, and edge mismatch.

## Centerline/vector vessel representation papers
Useful for:
- Centerline F1;
- centerline distance metrics;
- branch-level geometric fidelity.

---

# 28. First Codex implementation target

The first coding milestone should **not** modify the training loss.

Implement an evaluation package with approximately these components:

```text
evaluation/
    geometry.py
        point_to_polyline_distance
        sample_polyline_by_arclength
        curve_precision_recall_f1
        average_centerline_distance
        hd95

    topology.py
        node_degrees
        contract_degree2_chains
        extract_topological_anchors
        match_anchors_physical
        branch_precision_recall_f1
        betti_numbers

    morphology.py
        branch_length
        tortuosity
        branch_angles
        morphology_errors

    complexity.py
        total_nodes
        total_edges
        nodes_per_reference_length

    paths.py
        apls_patch
        # later: apls_volume

    experimental/
        graph_gospa
        topo3d
```

Add unit tests for all synthetic failure cases before applying the metrics to real model predictions.

---

# 29. Matcher experiments: current conclusion

A separate experiment investigated replacing the standard Hungarian matcher with a **Fused Gromov-Wasserstein (FGW)** matcher that uses both node-level information and predicted graph structure.

The motivation was valid: ordinary Hungarian matching is mainly driven by node-level costs, while FGW can prefer correspondences that are more consistent with the predicted adjacency structure.

The tested FGW formulation combined:

\[
\mathcal L_{\mathrm{FGW}}
=
(1-\alpha)\mathcal L_{\mathrm{feature}}
+
\alpha\mathcal L_{\mathrm{structure}}.
\]

The feature term used node position/object evidence, while the structural term compared the predicted edge structure to the GT adjacency under a soft transport plan. The resulting transport plan was then hardened to preserve one-to-one DETR-style supervision.

## Empirical conclusion

The experiments showed:

- FGW is not inactive;
- on frozen predictions, increasing structural weight can alter correspondences in structurally sensible ways;
- it can reduce structural mismatch and improve edge/non-edge separation;
- however, after longer training, those early gains did not persist reliably;
- the epoch-100 comparison did not justify replacing Hungarian with FGW in the current pipeline.

Therefore, the current recommendation is:

\[
\boxed{\text{keep Hungarian as the baseline matcher for the adaptive-representation experiments}}
\]

This preserves a clean experiment:

\[
\text{same training/matching}
+
\text{new representation}
+
\text{new evaluation}.
\]

## Why full FGW may be difficult here

There is a circular dependency:

\[
\text{predicted edges}
\rightarrow
\text{structure-aware matching}
\rightarrow
\text{supervision for those same predicted edges}.
\]

Late in training, predicted edges are reliable but Hungarian has already made most matches unambiguous.

Early in training, structural matching could matter more, but predicted edges are not yet trustworthy enough to safely guide correspondence.

This creates a chicken-and-egg problem.

There is also a representation-specific issue for the adaptive graph: many internal degree-2 control points have very similar one-hop structure, so a global adjacency-based structural term may provide little extra information for distinguishing neighboring control points.

---

# 30. Preferred future matcher direction: structure-aware Hungarian

If the new evaluation reveals that matching itself is a meaningful bottleneck, the preferred next step is **not** to return immediately to full FGW.

Instead, first make Hungarian only **slightly structure-aware** while preserving its simple one-to-one assignment formulation.

## 30.1 Degree-aware Hungarian cost

For every predicted node \(i\), compute an expected degree from relation probabilities:

\[
\hat d_i
=
\sum_j p^{\mathrm{edge}}_{ij}.
\]

For every GT node \(a\), compute its true graph degree:

\[
d^{GT}_a.
\]

Add a structural compatibility term to the ordinary Hungarian cost:

\[
C_{ia}
=
\lambda_x C_{\mathrm{position}}
+
\lambda_o C_{\mathrm{object}}
+
\lambda_d
\left|
\hat d_i-d^{GT}_a
\right|.
\]

This encourages:

- endpoint-like predictions to match GT endpoints;
- degree-2 predictions to match GT intermediate/control nodes;
- junction-like predictions to match GT junctions.

The final assignment is still obtained with ordinary Hungarian matching.

This has several advantages:

- no non-convex FGW solver;
- no soft transport/hardening stage;
- no full pairwise graph-matching objective;
- preserves the current DETR-style training flow;
- introduces only a limited and interpretable structural bias.

---

## 30.2 Role-aware Hungarian as a coarser alternative

Instead of exact degree, define a coarse GT/predicted role:

\[
r(v)=
\begin{cases}
\text{endpoint} & d=1,\\
\text{intermediate} & d=2,\\
\text{junction} & d\ge3.
\end{cases}
\]

Then add a role-mismatch penalty to the matching cost.

This may be more robust than exact degree if predicted edge probabilities are noisy.

No separate node-type classification head is required: predicted role can be estimated from the predicted relation probabilities.

---

## 30.3 Spatially gated structural tie-breaking

An even more conservative option is to use structure **only when the spatial/node matching is ambiguous**.

Example:

\[
C_{1A}=0.10,\qquad C_{1B}=0.11
\]

is a genuinely ambiguous unary match, so structure may be useful to break the tie.

By contrast:

\[
C_{1A}=0.05,\qquad C_{1B}=0.80
\]

should not be overturned by uncertain relation predictions.

A physical gating rule can be used:

\[
C_{ia}=\infty
\qquad
\text{if}
\qquad
\|x_i-x_a\|>\tau_{\mathrm{match}}.
\]

Then structure is only allowed to disambiguate anatomically plausible candidates.

This directly addresses the concern of accidentally identifying a very distant structural lookalike as the same node.

---

## 30.4 Why this is preferable to immediate FGW reuse

The intended purpose of structure-aware matching is not full graph isomorphism.

It is to solve local ambiguities such as:

- two nearby nodes with similar coordinate costs;
- endpoint versus junction confusion;
- neighboring control-point swaps;
- locally plausible correspondences with different connectivity roles.

Therefore the preferred progression is:

1. standard Hungarian;
2. degree/role-aware Hungarian;
3. spatially gated structural tie-breaking;
4. local structure-aware matching for ambiguous groups;
5. only then reconsider full FGW.

---

## 30.5 Possible later improvement: multi-hop/geodesic structural context

If one-hop degree information is insufficient, a richer structural descriptor may be used.

This is particularly relevant for chains such as:

\[
J-p_1-p_2-p_3-K.
\]

All \(p_i\) have degree 2, so one-hop adjacency does not distinguish them well.

A future matcher could use:

- shortest-path distance to nearby topological anchors;
- multi-hop neighborhood signatures;
- distances to endpoints/junctions;
- local graph geodesic descriptors.

This may provide more useful structural identity than direct adjacency alone.

---

## 30.6 When to revisit the matcher

Do **not** change the matcher preemptively.

First run the adaptive-representation experiments with the new metric suite.

Matcher work becomes justified if the evaluation shows a pattern such as:

- good geometry but poor Branch F1;
- good node localization but wrong branch pairing;
- qualitative evidence of nearby node swaps;
- structurally wrong correspondences despite reasonable spatial detections.

Only then should structure-aware Hungarian be tested.

This keeps the development path evidence-driven and avoids adding optimization complexity without a demonstrated need.
