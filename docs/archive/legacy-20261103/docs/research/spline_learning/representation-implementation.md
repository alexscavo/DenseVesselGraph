Given our current representation, I would **not replace the relation head outright**. I would reinterpret an edge as having two outputs:

\[
e_{ij}=
\left(
p_{ij}^{\text{edge}},
\;P_1,\ldots,P_8
\right)
\]

because

\[
P_0=x_i,\qquad P_9=x_j
\]

are already fixed by the two graph nodes. With our fixed cubic degree and fixed knot vector, the network only has to predict **16 values in 2D or 24 in 3D**.

The current RelationFormer relation mechanism is already based on a pair representation built from the two object tokens and the shared relation token; the natural modification is therefore to make that pair representation produce geometry as well as existence.

## The approaches I would seriously consider

| Approach | Edge existence | 8 internal controls | Complexity | My view |
|---|---|---|---|---|
| **A. Parallel flat spline head** | existing relation head | one MLP → 16/24 values | Low | **Start here** |
| **B. Shared multi-task edge head** | shared pair trunk → classifier | same trunk → geometry | Low | Also very reasonable |
| **C. Per-control shared head** | relation classifier | same small head called for \(k=1...8\) | Medium | **Best structured alternative** |
| **D. Image-conditioned control decoder** | relation classifier | 8 control queries attend to image features | High | Likely strongest long-term |
| E. Autoregressive controls | relation classifier | predict \(P_1\to P_2\to...\) | Medium/high | I would avoid initially |

The difference among A–D is mostly **where the information for the spline comes from and how much structure you impose on the 8 controls**.

---

# A. Parallel spline head — the cleanest first experiment

Keep our current relation classifier:

\[
z_{ij} = f_{\rm pair}(o_i,r,o_j)
\]

\[
p_{ij}^{edge}
=
h_{\rm cls}(z_{ij})
\]

and simply add:

\[
\Delta_{ij}
=
h_{\rm spline}(z_{ij},x_i,x_j)
\in\mathbb R^{8d}.
\]

So in 3D:

\[
h_{\rm spline}\rightarrow24
\]

and in 2D:

\[
h_{\rm spline}\rightarrow16.
\]

I would explicitly include endpoint geometry in this head, perhaps something like

\[
[z_{ij},x_i,x_j,x_j-x_i,\|x_j-x_i\|].
\]

The object embeddings already contain spatial information, but making the actual chord explicit makes the regression problem easier.

### Don't predict absolute controls

Initialize them along the straight chord:

\[
B_k
=
(1-t_k)x_i+t_kx_j,
\qquad
t_k=\frac{k}{9}.
\]

Then predict:

\[
P_k=B_k+\Delta P_k,
\qquad k=1,\ldots,8.
\]

So a straight vessel corresponds approximately to

\[
\Delta P_k=0.
\]

This is the version I would implement first.

```text
                         ┌── current classifier ──> edge / no edge
[o_i, r, o_j] ── pair ──┤
                         └── spline MLP ─────────> ΔP1 ... ΔP8
                                      ↑
                                  x_i , x_j
```

The major advantage is that you can initialize the existing network from our pretrained GrEVE checkpoint and **randomly initialize only the spline head**.

---

# B. A genuinely shared edge head

Slightly different architecture:

\[
[o_i,r,o_j,\text{geometry}]
\rightarrow
z_{ij}^{edge}
\]

and then:

\[
z_{ij}^{edge}
\rightarrow
\begin{cases}
p_{ij}^{edge}\\
\Delta P_{1:8}
\end{cases}
\]

The edge-existence and edge-shape tasks therefore share more computation.

This is conceptually attractive because:

> “Does a vessel connect these points?” and “What path does that vessel take?” are related questions.

For example, the shared representation may learn:

```text
these two nodes belong to the same vessel
                  +
the vessel bends upward between them
```

rather than learning those things independently.

But this modifies the currently trained relation head more substantially.

For our first experiment, I prefer **A** because you preserve the edge classifier exactly. If A works, B becomes an interesting ablation.

---

# C. Per-control shared prediction

I actually like this one quite a lot as a second experiment.

Instead of one MLP outputting all 24 coordinates at once,

\[
z_{ij}\rightarrow
[\Delta P_1,\Delta P_2,\ldots,\Delta P_8],
\]

introduce a small embedding identifying each internal control position:

\[
q_k=[z_{ij},e_k,t_k,B_k].
\]

Then use the **same MLP** for each control:

\[
\Delta P_k=h_{\rm ctrl}(q_k).
\]

So:

```text
                            k=1 ──> ΔP1
                            k=2 ──> ΔP2
                              ...
pair representation ------ k=8 ──> ΔP8
                    same control MLP
```

The network knows that \(P_2\) is near the beginning and \(P_7\) is near the end because it gets \(t_k\).

There are several nice properties:

- weight sharing across controls;
- explicit ordering along the branch;
- fewer arbitrary output dimensions;
- potentially easier to move later from 10 controls to another \(K\);
- natural starting point for image-conditioned control queries.

It's essentially saying:

> “Given this vessel relation, tell me where the control at relative branch position \(t_k\) should move.”

I think this is more elegant than a flat `Linear(hidden_dim, 24)` head, although the flat head is the correct baseline.

---

# D. Image-conditioned control decoder

This is the one I think could eventually be strongest for real data.

The problem with A–C is that after RelationFormer creates \(o_i,o_j,r\), the spline head has no new opportunity to look closely at the **image along the vessel between the endpoints**.

For a highly curved branch:

```text
i
 \
  \
   \____
        \____ j
```

the exact trajectory is encoded in the raw volume.

So initialize our eight control references:

\[
B_1,\ldots,B_8
\]

along the straight chord.

Create eight control queries conditioned on the relation:

\[
q_{ij}^{k}
=
f(z_{ij},e_k).
\]

Then let them cross-attend to image features around \(B_k\).

Conceptually:

```text
                  image feature volume
                       ↑ ↑ ↑ ↑
                       │ │ │ │
i --- B1 --- B2 --- B3 --- ... --- B8 --- j
      ↓      ↓      ↓             ↓
     q1     q2     q3            q8
      │      │      │             │
     ΔP1    ΔP2    ΔP3           ΔP8
```

That is very compatible with the deformable-attention philosophy already used in RelationFormer.

Each control effectively asks:

> “I currently think the vessel should pass around here. Where does the image evidence tell me to move?”

I would consider this the **proper architectural extension** if the simple MLP cannot recover tortuous branches.

But not the first implementation.

---

# One detail I would change from our current description: controls in mm are good, but consider residual scaling

You said:

> Spatial controls and segment joins are edge parameters in mm.

I agree that the final representation should be physical.

But raw regression of a displacement such as

\[
\Delta P_k=(+5.2,-3.4,+1.7)\text{ mm}
\]

may have a very different scale across tiny and very long branches.

I would test two parameterizations.

### Physical residual

\[
P_k=B_k+\Delta P_k^{mm}.
\]

Very interpretable.

### Relative residual

\[
P_k
=
B_k+
L_{ij}\Delta \tilde P_k,
\]

where

\[
L_{ij}=\|x_j-x_i\|.
\]

Then the network predicts a dimensionless relative deformation.

At rendering/output time, everything still becomes mm.

I wouldn't decide theoretically which is better. This is a cheap ablation.

---

# The undirected-edge problem needs to be solved explicitly

This becomes more important now.

For vessel graphs,

\[
i\leftrightarrow j
\]

is the same edge as

\[
j\leftrightarrow i.
\]

But:

\[
[P_0,P_1,\ldots,P_9]
\]

becomes

\[
[P_9,P_8,\ldots,P_0]
\]

when traversed backward.

our current binary relation head can be order invariant. A spline-control head **cannot simply be invariant**, because control order has meaning.

I would choose a deterministic orientation before passing a pair through the spline head.

For example:

\[
i\rightarrow j
\]

according to lexicographic physical position:

1. lower \(x\);
2. then lower \(y\);
3. then lower \(z\).

Then GT splines are stored/reversed to follow the same convention.

That way there is always one canonical sequence:

\[
P_0\rightarrow P_9.
\]

Alternatively, use an orientation-invariant loss:

\[
L=
\min
\left[
L(P,P^*),
L(P,\operatorname{reverse}(P^*))
\right].
\]

I slightly prefer **canonical orientation** because it also gives the network a deterministic learning target.

---

# What I would use for the loss

With our fixed basis, one very nice thing happens.

For the fixed cubic B-spline:

\[
C(t)=
\sum_{i=0}^{9}
P_i N_{i,3}(t),
\]

the basis functions are always the same.

Choose, say, \(M=32\) or \(64\) values of \(t\), and precompute

\[
B_{mi}=N_{i,3}(t_m).
\]

Then:

\[
X=BP
\]

gives the whole sampled curve.

This operation is differentiable and almost free.

So I would compare two losses.

### Baseline

Direct control regression:

\[
L_{\rm ctrl}
=
\frac18
\sum_{k=1}^{8}
\operatorname{SmoothL1}
(P_k,P^*_k).
\]

### More representation-aligned version

Curve regression:

\[
L_{\rm curve}
=
\frac1M
\sum_m
\operatorname{SmoothL1}
(
\hat C(t_m),C^*(t_m)
).
\]

I suspect the second may be better because the **controls themselves are not what you care about**; the curve is.

And critically, I would start with:

\[
L_{\rm edge}
=
L_{\rm focal}
+
\lambda_{\rm geom}L_{\rm curve},
\]

only for GT-positive edges.

Not five spline losses.

---

# There's also a fifth possibility, but I wouldn't prioritize it

You could predict a set of sampled centerline points first:

\[
\hat X_1,\ldots,\hat X_M
\]

and then solve for the ten B-spline controls through a fixed least-squares mapping.

Since knots and basis are fixed:

\[
P \approx B^\dagger X.
\]

This could even remain differentiable.

But this effectively asks the network to predict many centerline samples just to compress them afterward, which starts moving back toward the dense-centerline methods you are trying to avoid.

So I see no reason to start there.

---

## My experimental tree would therefore be

**V0 — sanity check**

\[
[o_i,r,o_j,x_i,x_j]
\rightarrow
24
\]

flat residual controls, direct control SmoothL1.

If this doesn't even overfit a small dataset, something is wrong upstream.

**V1 — actual baseline**

Same flat spline head, but train through the sampled B-spline curve:

\[
L_{\rm curve}.
\]

**V2 — structured controls**

Shared per-control MLP:

\[
(z_{ij},t_k,B_k)
\rightarrow
\Delta P_k.
\]

Compare to V1.

**V3 — only if geometry remains the bottleneck**

Per-control queries with image/deformable cross-attention.

And throughout V0–V3:

- keep the existing edge classifier;
- keep endpoints fixed;
- keep \(K=10\);
- keep degree and knots fixed;
- keep Hungarian initially;
- keep focal edge classification;
- supervise spline geometry only for positive matched relations.

That progression would tell you something meaningful at every step instead of jumping immediately to a completely new edge decoder.