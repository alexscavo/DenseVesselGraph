# Vascular Graph Representation: Current and Future Directions

## Scope

This note summarizes the conclusions reached while discussing two complementary directions for direct vascular graph extraction from raw 2D/3D images:

1. **Topology + adaptive degree-2 control points**
2. **Topology + function-valued / curve-valued edges**

The common goal is to preserve the key advantage of the RelationFormer-style pipeline: **direct image/volume-to-graph prediction without requiring segmentation masks as training targets or as an inference-time intermediate representation.**

---

# 1. Core modeling principle

The main conceptual distinction is:

- **Topological nodes** represent anatomically meaningful events:
  - endpoints,
  - bifurcations,
  - higher-order junctions.

- **Geometric information** represents the shape of the vessel between those events.

A degree-2 point is therefore not necessarily an anatomical object. It can instead be interpreted as a **geometric control point** used to approximate the vessel trajectory.

This motivates the decomposition

\[
\text{vascular graph} = \text{topology} + \text{edge geometry}.
\]

That decomposition should guide both representation and evaluation.

---

# 2. Approach A: Junctions/endpoints + adaptive degree-2 control points

## Representation

Keep the existing RelationFormer graph formulation, but preprocess the ground-truth centerline so that degree-2 points are retained **only where necessary to approximate the original centerline within a chosen geometric error**.

Thus a straight vessel may be represented as

\[
A \;-\; B
\]

while a curved vessel may be represented as

\[
A \;-\; p_1 \;-\; p_2 \;-\; B,
\]

where \(A,B\) are topological nodes and \(p_1,p_2\) are degree-2 geometric control points.

The control points should be selected from the dense centerline according to reconstruction error, not by a simple local angle threshold.

## Why this is attractive

- Minimal architectural changes.
- Compatible with existing RelationFormer training.
- Preserves raw-volume-to-graph prediction.
- Reduces unnecessary object-token usage compared with retaining all degree-2 nodes.
- Provides a direct test of the hypothesis:
  - geometry matters,
  - but redundant geometry does not.

## Main limitation

The topology and geometry are still mixed in the same node set:

\[
V = V_{\text{topology}} \cup V_{\text{geometry}}.
\]

A highly tortuous vessel can therefore require many more object tokens than a straight vessel even when both have the same topology.

---

# 3. Contracting degree-2 vertices while preserving geometry

For evaluation, degree-2 vertices can be removed from the **topological graph** without discarding the shape they encode.

Example:

\[
A - p_1 - p_2 - p_3 - B
\]

becomes topologically

\[
A \longleftrightarrow B
\]

but the edge stores the ordered polyline

\[
\gamma_{AB} = [A,p_1,p_2,p_3,B].
\]

So:

- topology sees one edge \(A \leftrightarrow B\),
- geometry still sees the complete vessel path.

This makes topology evaluation invariant to arbitrary subdivision of an otherwise identical vessel.

Two predictions with different numbers of degree-2 points can therefore be considered topologically equivalent and compared geometrically through their reconstructed curves.

---

# 4. Approach B: Junctions/endpoints + function-valued edges

## Representation

The cleaner long-term formulation is

\[
G = (V_T, E_\Gamma),
\]

where \(V_T\) contains only topological nodes and every edge has an associated spatial curve

\[
e_{ij} = (v_i, v_j, \Gamma_{ij}),
\]

with

\[
\Gamma_{ij}:[0,1] \rightarrow \mathbb{R}^3.
\]

Instead of representing curvature by inserting graph nodes,

\[
A - p_1 - p_2 - p_3 - B,
\]

the model predicts

\[
A \overset{\Gamma_{AB}}{\longleftrightarrow} B.
\]

The edge itself contains the geometry.

## Why this representation is conceptually stronger

It separates two distinct concepts:

\[
\text{node} = \text{anatomical/topological event}
\]

\[
\text{edge geometry} = \text{vessel trajectory}.
\]

This prevents curvature from increasing the number of graph vertices and object tokens.

It also removes the dependence of graph complexity on vessel tortuosity.

---

# 5. Possible function representations

A "function-valued edge" should not initially mean an arbitrary neural function. A compact parametric curve is more practical.

## Cubic Bézier

A first implementation could use

\[
\Gamma(t)
=
(1-t)^3P_0
+
3(1-t)^2tP_1
+
3(1-t)t^2P_2
+
t^3P_3.
\]

The endpoints are fixed by the predicted nodes:

\[
P_0 = x_i,\qquad P_3 = x_j.
\]

The network only predicts the internal control points

\[
P_1,\;P_2.
\]

In 3D, this requires only six geometric regression values per positive edge.

## B-spline / piecewise spline

For more tortuous vessels, a B-spline or piecewise Bézier representation can be used:

\[
\Gamma(t)=\sum_k B_k(t)P_k.
\]

Internal spline points are still **edge parameters**, not graph vertices.

This preserves a compact topological graph while allowing arbitrarily detailed geometry.

---

# 6. How the function-based approach could fit RelationFormer

Current RelationFormer logic:

\[
(o_i,r,o_j)
\rightarrow
p_{ij},
\]

where \(p_{ij}\) is the edge/relation probability.

A future geometry-aware relation head could predict

\[
(o_i,r,o_j)
\rightarrow
\begin{cases}
p_{ij} & \text{edge existence},\\
\theta_{ij} & \text{curve parameters}.
\end{cases}
\]

Here \(\theta_{ij}\) may be Bézier or spline control parameters.

A stronger version should allow the edge-geometry decoder to attend again to the image/volume features, because the endpoints alone are insufficient to determine the path taken by the vessel.

Conceptually:

\[
I
\rightarrow
F_I
\rightarrow
V_T
\rightarrow
(E,\{\Gamma_e\}),
\]

where \(F_I\) is the encoded raw-image feature field.

No segmentation is required.

---

# 7. Supervision for function-valued edges

The dense centerline available in the graph annotation can supervise the curve directly.

For a ground-truth branch between nodes \(A\) and \(B\),

\[
C_{AB} = \{c_1,\dots,c_M\}
\]

is retained as geometric supervision.

The model predicts

\[
\Gamma_{AB}(t).
\]

Then sample the predicted curve:

\[
\hat C_{AB}
=
\{\Gamma(t_1),\dots,\Gamma(t_K)\}.
\]

The preferred strategy is to supervise the **resulting geometry**, not the exact control-point parameters.

A symmetric geometric loss can be used:

\[
L_{\text{geom}}
=
\frac{1}{|\hat C|}
\sum_{\hat c\in\hat C}
d(\hat c,C)
+
\frac{1}{|C|}
\sum_{c\in C}
d(c,\hat C).
\]

Possible additional terms:

### Length consistency

\[
L_{\text{length}}
=
\frac{
|\ell(\Gamma)-\ell(C)|
}{
\ell(C)
}.
\]

### Weak smoothness regularization

\[
L_{\text{smooth}}
=
\int_0^1
\|\Gamma''(t)\|^2dt.
\]

This should remain weak so that real tortuosity is not suppressed.

### Combined edge loss

\[
L_{\text{edge}}
=
L_{\text{existence}}
+
\mathbf{1}_{\text{GT edge}}
\left(
\lambda_{\text{geom}}L_{\text{geom}}
+
\lambda_{\text{length}}L_{\text{length}}
+
\lambda_{\text{smooth}}L_{\text{smooth}}
\right).
\]

---

# 8. Matching predicted edges to ground truth

RelationFormer already performs Hungarian matching between predicted object tokens and ground-truth nodes.

If

\[
\hat v_i \leftrightarrow v_A
\]

and

\[
\hat v_j \leftrightarrow v_B,
\]

then:

- if \(A\) and \(B\) are connected in the GT graph, the pair \((\hat v_i,\hat v_j)\) receives:
  - positive relation supervision,
  - geometric supervision from the dense branch centerline \(C_{AB}\);

- otherwise it receives negative relation supervision only.

For undirected vessels, the geometric loss should be orientation invariant.

---

# 9. Rendering a function-valued edge

The continuous curve is only the compact mathematical representation.

To draw or render it, sample the curve densely:

\[
t_k = \frac{k}{K-1},
\qquad
x_k = \Gamma(t_k),
\]

for \(k=0,\dots,K-1\).

This gives a standard polyline:

\[
x_0 - x_1 - x_2 - \dots - x_{K-1}.
\]

## 2D rendering

Connect consecutive samples with line segments.

## 3D rendering

Several options are possible:

### Centerline rendering

Draw consecutive 3D line segments between the sampled points.

### Tube rendering

Construct a tube around the sampled centerline.

If only the trajectory is predicted, use a fixed visualization radius.

If radius is predicted later as a function

\[
r(t),
\]

the edge can be rendered as a variable-radius tube using

\[
(\Gamma(t),r(t)).
\]

## Rasterized voxel rendering

For comparison with voxel data, rasterize every polyline segment into the voxel grid.

Thus the function-valued representation can always be translated back into an ordinary discrete centerline whenever needed.

---

# 10. Evaluation philosophy

The evaluation should separate:

\[
\boxed{\text{Topology}}
\qquad+\qquad
\boxed{\text{Geometry}}
\qquad+\qquad
\boxed{\text{Representation efficiency}}.
\]

The exact placement and number of degree-2 control points should not dominate the score.

---

# 11. Topology evaluation

First contract degree-2 vertices while preserving their polyline geometry.

Then evaluate the topological graph using metrics such as:

- endpoint precision / recall / F1,
- junction precision / recall / F1,
- branch-level precision / recall / F1,
- path-based connectivity / APLS-style metric,
- Betti-0 / Betti-1 as supplementary global checks.

A branch-level metric should answer:

> Are the correct topological endpoints/junctions connected by a valid predicted vessel path?

---

# 12. Geometry evaluation

For either a piecewise-linear current prediction or a future spline prediction, represent each branch as a continuous/spatial curve and sample it by arc length.

## Curve precision

\[
P_{\text{curve}}(\tau)
=
\frac{
|\{p\in C_P:d(p,C_G)<\tau\}|
}{
|C_P|
}.
\]

Interpretation:

> How much of the predicted vessel lies sufficiently close to the ground-truth vessel?

## Curve recall

\[
R_{\text{curve}}(\tau)
=
\frac{
|\{g\in C_G:d(g,C_P)<\tau\}|
}{
|C_G|
}.
\]

Interpretation:

> How much of the true vessel was recovered?

## Curve F1

\[
F1_{\text{curve}}
=
\frac{
2P_{\text{curve}}R_{\text{curve}}
}{
P_{\text{curve}}+R_{\text{curve}}
}.
\]

Useful additional metrics:

- symmetric mean curve distance,
- 95th-percentile Hausdorff distance,
- SMD,
- physical-unit errors in mm rather than only voxels.

Edge bounding-box IoU can remain for legacy comparison but should not be a primary metric for curved edges.

---

# 13. Representation efficiency

For the adaptive degree-2 approach, report compactness explicitly:

\[
\rho_2
=
\frac{
N_{\text{degree-2}}
}{
\text{total centerline length}
}.
\]

This can be plotted against geometric error or curve F1.

The desired representation lies on the favorable part of the geometry-complexity Pareto frontier:

- fewer control points,
- high geometric fidelity.

For the spline/function approach, an analogous complexity quantity could be:

- number of spline control points per mm,
- number of curve segments per branch,
- total geometry parameters per mm.

---

# 14. Recommended near-term and long-term strategy

## Near term

Use:

\[
\boxed{
\text{junctions/endpoints}
+
\text{minimum error-adaptive degree-2 control points}
}
\]

because:

- it is already implemented,
- it preserves the current RelationFormer architecture,
- it has relatively low research risk,
- it directly tests whether selective geometric detail improves real-data performance.

At the same time, change the evaluation so that degree-2 nodes are treated primarily as geometric samples rather than anatomical detection targets.

## Long term

Develop:

\[
\boxed{
\text{topological object tokens}
+
\text{curve-valued relational edges}
}
\]

where:

- object tokens predict only endpoints and junctions,
- relations predict both connectivity and vessel geometry,
- curve geometry can query image features,
- no segmentation is required,
- no degree-2 graph vertices are required.

---

# 15. Unifying view

The two approaches are not fundamentally different graph types.

They are two parameterizations of the same embedded vascular graph.

## Current adaptive representation

\[
A - p_1 - p_2 - \dots - B
\]

defines a piecewise-linear curve

\[
\gamma_{\text{PL}}.
\]

## Future function-valued representation

\[
A \overset{\gamma_{\text{spline}}}{\longleftrightarrow} B
\]

defines a continuous spline curve.

Both can therefore be evaluated in the same space:

\[
\boxed{
\text{topological correctness}
+
\text{geometric correctness}
}
\]

This is useful because the evaluation framework can be implemented now and reused unchanged if the model later moves from adaptive degree-2 nodes to explicit curve-valued edges.

---

# Final recommendation

For the current project, prioritize the adaptive centerline-error representation.

For future architectural work, the stronger formulation is:

\[
\boxed{
I
\rightarrow
\left(
V_{\text{topology}},
E,
\{\Gamma_e\}_{e\in E}
\right)
}
\]

directly from the raw image/volume.

The model should recover:

1. **where the anatomical graph events are,**
2. **which events are connected,**
3. **the geometric trajectory of each vessel between them.**

That separates topology from geometry while preserving the core goal of segmentation-free image-to-graph prediction.

---

# 16. Future extension: radius as an edge function

A major advantage of the function-valued edge representation is that an edge does not need to encode only the vessel centreline trajectory. It can also carry local vessel attributes defined continuously along the branch.

Instead of representing an edge only by

\[
\Gamma_e(t)=
\begin{bmatrix}
x(t)\\
y(t)\\
z(t)
\end{bmatrix},
\qquad t\in[0,1],
\]

we can augment it with a radius function

\[
r_e(t).
\]

The edge then becomes

\[
e=
\left(
v_i,
v_j,
\Gamma_e(t),
r_e(t)
\right),
\qquad t\in[0,1].
\]

This means the graph stores both:

- the spatial trajectory of the vessel centreline;
- how the vessel radius varies along the branch.

This is more informative than attaching a single scalar radius to the whole edge, because real vessels can taper or locally change caliber.

## Practical parameterization

A simple implementation could use the same spline parameter \(t\) for both geometry and radius.

For example, if the centreline is represented as

\[
\Gamma(t)=\sum_k B_k(t)P_k,
\]

then the radius could be represented as

\[
r(t)=\sum_k B_k(t)\rho_k,
\]

where:

- \(P_k\) are geometric control points;
- \(\rho_k\) are radius control values;
- \(B_k(t)\) are the spline basis functions.

The network would therefore predict both spatial and radius parameters for every positive edge.

## Rendering with radius

At render time, sample both functions at the same positions:

\[
x_k=\Gamma(t_k),
\qquad
r_k=r(t_k).
\]

This gives a sequence of centreline samples with associated local radii:

\[
(x_0,r_0),
(x_1,r_1),
\dots,
(x_{K-1},r_{K-1}).
\]

A variable-radius tube can then be swept along the sampled centreline. Therefore, the function-valued graph can reconstruct not only the branch path but also an approximation of the vessel lumen geometry.

If no radius is available, the edge can still be rendered as a centreline or with a fixed visualization radius.

## Potential downstream benefits

Adding \(r(t)\) would make the graph substantially more useful for downstream vascular analysis, including:

- branch-wise radius statistics;
- tapering measurements;
- vessel volume and surface estimates;
- morphometry;
- more realistic hemodynamic simulations;
- resistance-related quantities.

For example, in an idealized Poiseuille-style setting, hydraulic resistance depends strongly on radius:

\[
R \propto \int \frac{ds}{r(s)^4}.
\]

Therefore, knowing radius continuously along a branch is much more informative than knowing only connectivity and centreline geometry.

## Supervision requirements

The raw image/volume can still remain the only network input, so the approach remains segmentation-free at inference time.

However, supervising \(r(t)\) requires some form of radius target in the training annotations. This may come from:

- segmentation-derived local vessel radius;
- explicit diameter annotations;
- precomputed radius estimates attached to the centreline.

The important distinction is that segmentation may be used only to generate training labels for radius, while the model can still learn the mapping

\[
\text{raw volume}
\rightarrow
\text{graph + geometry + radius}
\]

directly.

## Long-term edge representation

A compact long-term representation could therefore be written as

\[
\boxed{
e(t)=
\left[
x(t),y(t),z(t),r(t)
\right]
}
\]

with the possibility of adding further continuous edge attributes later.

This strengthens the motivation for separating topology from edge geometry: once the edge is treated as a structured object rather than a binary relation, it can naturally carry both shape and local vessel properties without increasing the number of graph nodes.


---

# 16. Important refinement: separate representation error from prediction error

For highly curved real vessels, the adaptive graph is intentionally **not** expected to reproduce every small oscillation of the dense centerline. Otherwise, the graph would again become saturated with degree-2 nodes.

Therefore, direct comparison of the model prediction against the dense centerline must be interpreted carefully.

The total discrepancy can be decomposed conceptually into:

\[
\text{final geometric error}
=
\text{representation error}
+
\text{prediction error}.
\]

The adaptive graph introduces a deliberate approximation error in exchange for compactness. That error should not be counted as model failure.

## 16.1 Representation quality

First evaluate the adaptive target representation itself against the original dense centerline:

\[
G_{\text{adaptive}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

This is **not a model-performance evaluation**. It measures how much vessel geometry is intentionally sacrificed in order to obtain a compact graph.

Useful quantities include:

- number of total nodes,
- number of degree-2 nodes,
- degree-2 nodes per unit centerline length,
- average centerline approximation distance,
- 95th-percentile centerline distance / HD95,
- centerline coverage at one or more distance tolerances.

This defines the geometry-complexity trade-off of the representation itself.

A highly curved branch may therefore have non-zero approximation error by design.

## 16.2 Model quality

The main model evaluation should compare:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{adaptive}}.
\]

This asks:

> Did the model correctly learn the graph representation it was trained to predict?

Here, a perfect score should in principle be achievable.

Recommended metrics include:

- SMD,
- legacy node mAP / mAR,
- legacy edge mAP / mAR,
- branch precision / recall / F1 after degree-2 contraction,
- curve/polyline precision / recall / F1 against the adaptive GT,
- average curve distance,
- Betti-0 / Betti-1 errors.

For GrEVE compatibility, the legacy RelationFormer metrics should remain, but the new topology- and geometry-aware metrics should carry more weight in the interpretation.

## 16.3 End-to-end anatomical fidelity

A second, complementary evaluation can compare:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

This measures the geometric fidelity of the final predicted graph with respect to the original vessel centerline.

However, these numbers should always be interpreted together with the corresponding **oracle adaptive target**:

\[
G_{\text{adaptive}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

For example:

\[
D(G_{\text{adaptive}},G_{\text{dense}})=1.20
\]

and

\[
D(G_{\text{pred}},G_{\text{dense}})=1.47.
\]

The additional discrepancy is then approximately:

\[
\Delta_{\text{geom}}
=
D(G_{\text{pred}},G_{\text{dense}})
-
D(G_{\text{adaptive}},G_{\text{dense}})
=
0.27.
\]

This separates:

- error deliberately introduced through graph compression,
- extra error caused by imperfect model prediction.

This distinction is especially important on highly tortuous real vessels.

# 17. The adaptive representation is a rate-distortion problem

The target is not:

\[
\min \text{geometric error}
\]

without any constraint.

That trivial objective would simply retain almost every centerline point.

The true objective is closer to:

\[
\min
\left[
\text{geometric distortion}
+
\lambda\,\text{representation complexity}
\right].
\]

Alternatively, avoid choosing a single arbitrary weight \(\lambda\) and explicitly analyze the Pareto frontier:

\[
\text{graph complexity}
\quad\leftrightarrow\quad
\text{geometric fidelity}.
\]

For example, compare:

- junction/endpoints only,
- all degree-2 points,
- adaptive degree-2 points.

The adaptive representation should ideally achieve much of the geometric fidelity of a dense graph with substantially fewer nodes.

This is particularly relevant for real datasets, where vessels can be highly tortuous and following every local curvature variation would otherwise cause a large increase in graph size.

A more accurate description of the target is therefore:

> Retain enough degree-2 control points to preserve meaningful vessel geometry while enforcing a practical graph-complexity budget.

# 18. Revised evaluation structure

The evaluation should therefore be organized into three distinct blocks.

## A. Representation quality

Compare:

\[
G_{\text{adaptive}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

Purpose:

> How much geometric information is deliberately lost through compression?

Recommended quantities:

- number of nodes,
- number of degree-2 nodes,
- degree-2 nodes per mm / voxel-length unit,
- average centerline distance,
- HD95,
- centerline coverage / curve F1 against the dense centerline.

This is a property of the target representation, not of the network.

## B. Model quality

Compare:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{adaptive}}.
\]

Purpose:

> How well does the model predict the chosen compact graph representation?

Recommended quantities:

- Node mAP / mAR,
- Edge mAP / mAR,
- SMD,
- branch F1,
- curve/polyline F1 against adaptive GT,
- Betti-0 / Betti-1,
- average curve distance if useful.

This should be the primary model-performance evaluation.

## C. End-to-end anatomical fidelity

Compare:

\[
G_{\text{pred}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

Purpose:

> How close is the final predicted vascular graph to the original vessel geometry?

Recommended quantities:

- average centerline distance,
- HD95,
- centerline coverage / F1.

These should always be reported together with the oracle:

\[
G_{\text{adaptive}}
\quad\text{vs}\quad
G_{\text{dense}}.
\]

The gap between the two gives an estimate of the extra geometric degradation introduced by the model rather than by the chosen representation.

# 19. Important consequence for interpreting centerline metrics

The adaptive target should **not** be expected to obtain 100% geometric fidelity against the dense centerline when the chosen representation deliberately ignores very fine curvature.

Therefore:

- dense-centerline metrics should not be used alone as the main model score;
- they should be used to characterize the representation and the final end-to-end output;
- model quality should primarily be measured against the adaptive target representation.

This avoids penalizing the network for geometric detail that the target representation itself intentionally discards.
