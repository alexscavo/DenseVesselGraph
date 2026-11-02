# Dense two-stage model with traced connectivity — description and main results (2 Nov 2026)

Short companion to [`network_proposals_20261030.md`](network_proposals_20261030.md), which has
the full design, the glossary and every intermediate experiment. All numbers below are
**validation** results, single seed. Node and edge thresholds are selected on 256 training
patches (`monitor_train`) only. No test data has been used.

## 1. Summary

- **New model.** A full-resolution 3D U-Net detects nodes as heatmap peaks. Edges are proposed
  by tracing the predicted vessel centreline between nodes (plus a 16-nearest-neighbour
  fallback), then scored and shaped from image features sampled *along the traced path*.
  Edge shapes are B-splines with 8, 16 or 24 controls, chosen per edge.
- **It beats every RelationFormer variant on all three 3D datasets.** On IXI, edge F1 is 0.31
  vs 0.16, Curve F1 0.77 vs 0.34, and SMD 0.023 vs 0.039. It trains from scratch, with no
  pretraining.
- **Traced connectivity vs plain 16-NN** (same model otherwise):
  - Curve F1 +0.10 on IXI and BRAVA;
  - Branch F1 +0.01–0.02;
  - fewer spurious components on IXI (β₀ error 4.0 → 1.7);
  - better recovery of long edges;
  - edge F1 unchanged.
- **Curved edges, separately from the full model.** With the true graph given, learned curved
  edges are 2–3× closer to the real centreline than straight ones. Training RelationFormer
  with or without splines gives identical graphs; only curve quality differs.
- **Main remaining limit: node recall on real data.** About 80% of missed BRAVA long edges
  lack a detected endpoint.

## 2. The new approach: two-stage dense model with traced connectivity

### 2.1 The idea

RelationFormer predicts a graph with a fixed set of learned "object queries". It reads the
image through a 2×2×2 feature grid and pairs queries with ground truth by Hungarian matching,
which exists only during training. Our diagnostics showed this is what fails on real data:
- nodes are not localised;
- the training-time matching does not match what happens at inference;
- edges are scored without ever looking at the image between the two nodes.

The new model replaces this with two simple, image-grounded stages:

1. **Stage 1, where are the nodes?** A full-resolution 3D U-Net predicts, for every voxel, how
   likely it is to be a graph node (a heatmap), plus where the vessels' centrelines run. Nodes
   are the peaks of the heatmap.
2. **Stage 2, which nodes are connected, and how does the vessel run between them?**
   - Starting from every node, we *follow the predicted centreline* to find which other nodes
     are reachable along a vessel. Each reachable pair becomes a candidate edge, at any distance.
   - For each candidate, a small network looks at the image features *along the traced path*.
     It decides whether the edge exists, and draws its curved shape as a B-spline whose
     number of control points depends on how long and tortuous the path is.

No Hungarian matching is involved. Training builds its edge candidates from the model's *own*
predicted nodes, exactly as inference does.

### 2.2 Overview

```
 64³ MRA patch (intensity − foreground mean)
   │
   ▼  Stage 1: MONAI BasicUNet, full resolution, 37 output channels at 64³
   ├─ ch0      node heatmap (logit) ─► 3×3×3 peak NMS + threshold ─► nodes
   ├─ ch1–3    sub-voxel offsets ───────────────────────────────────┘ (refine peak position)
   ├─ ch4      centreline probability ─► mask (p ≥ 0.5)
   └─ ch5–36   32 feature channels F
   │
   ▼  Stage 2
   nodes + centreline mask ─► geodesic Voronoi tracing ─► candidate pairs + traced paths
                                   (∪ 16 nearest neighbours, straight paths, as fallback)
   per candidate pair (a, b):
     features F sampled in a 24×11×11 tube that follows the path ─► corridor CNN ─► 6,912-d
     + chord length/direction + F at a and b + heat at a and b + path arc length ─► 6,983-d
       ├─► edge head ─────────────────────────────► P(a–b is an edge)
       └─► geometry head K8 / K16 / K24 (fixed rule on the path) ─► B-spline from a to b
```

### 2.3 Stage 1: dense node and centreline maps

**Network.** MONAI `BasicUNet` (3D) with feature widths (32, 32, 64, 128, 256, 32), used as-is:
- four 2× down-sampling levels and back up to the full 64³ resolution, with skip connections;
- instance normalisation, which behaves identically in training and evaluation;
- LeakyReLU, about 9.6M parameters.

Its 37 output channels are read as follows:

| Channel(s) | Meaning | Training target and loss |
|---|---|---|
| 0 | node heatmap logit | One Gaussian blob per GT node (σ = 1.5 voxels, peak 1 at the node's rounded voxel). **CenterNet penalty-reduced focal loss** (α = 2, β = 4), normalised by the number of nodes. The output bias starts at −4.6 (probability ≈ 0.01) |
| 1–3 | sub-voxel offset of a node from its voxel centre | L1, only at GT node voxels |
| 4 | centreline logit | The GT dense centreline of every edge, rasterised (each edge's 64-point path upsampled to 256 points, then dilated by one voxel). Binary cross-entropy with positive weight 5. Auxiliary: it makes the features vessel-aware and is used for tracing |
| 5–36 | 32 general feature channels F | no direct target; shaped by the edge and geometry losses of stage 2 |

**Node extraction.**
1. Take σ(heatmap).
2. Keep voxels that are the maximum of their 3×3×3 neighbourhood and above a threshold.
3. Position = voxel index + predicted offset, clamped to the patch.

Peaks are sorted by score. During training, peaks ≥ 0.2 are used (at most 200 per patch). At
inference the threshold is the one selected on training patches (§2.6).

### 2.4 Stage 2: from nodes to edges

**(a) Tracing the centreline: geodesic Voronoi regions.**
- **Domain.** Voxels with centreline probability ≥ 0.5, plus a 2-voxel ball around every node,
  so each node connects to its nearby centreline.
- **Propagation.**
  - Starting from all nodes simultaneously, a geodesic distance is propagated over the
    26-neighbour voxel grid on the GPU, by Bellman–Ford relaxation (up to 320 sweeps, stopping
    early once stable).
  - Entering a voxel costs (step length) × 1/(p + 0.05): cheap where the vessel is confident,
    expensive where it is uncertain.
  - Each voxel ends up *owned* by the node that reaches it most cheaply: its geodesic Voronoi
    region.
- **Candidates.**
  - Two nodes form a candidate pair when their regions *touch* somewhere inside the domain;
    the cheapest touching point is kept.
  - Properties: there is no distance limit. A node lying between two others on the same
    vessel separates their regions, so **"skip" pairs (a–c when the vessel runs a–b–c) are
    never proposed**.
  - The pairs among each node's **16 nearest neighbours** (in mm) are added as a fallback
    with straight paths, so tracing can only add candidates, never remove the old ones.
- **Path.**
  - From the touching point, the path is read back by steepest descent of the distance field
    to each of the two nodes, then the two halves are joined.
  - It is resampled to 64 points by arc length, smoothed (moving average of 5) and pinned to
    the exact sub-voxel node positions.

**(b) Corridor features.**
- The path is resampled to 24 points. At each point a local frame is built: the tangent t,
  plus two perpendicular directions u and w.
- The 32 feature channels F are sampled by trilinear interpolation on an 11×11 cross-section
  spanning ±12 voxels in the (u, w) plane, at each of the 24 points.
- The result is a "tube" of features that **follows the vessel**, of shape 32×24×11×11. The
  first, 16-NN-only version of the model sampled a straight tube around the chord instead.
- A small 3D CNN encodes it to a 6,912-d vector: three 3×3×3 convolutions, 32→64→64→64
  channels, with strides 1, 2 and (1, 2, 2), and GELU activations.
- It is concatenated with:
  - chord length and direction (4);
  - F at both nodes (64);
  - heatmap value at both nodes (2);
  - path arc length (1).

  That gives a 6,983-d pair vector.

**(c) Edge head.** A two-layer MLP (6,983 → 256 → 1, GELU) outputs P(a and b are directly
connected).

**(d) Geometry head with adaptive K.**
- The curve is a clamped cubic B-spline that starts at node a and ends exactly at node b.
- Its interior control points start *on the traced path*, at the B-spline's standard
  parameter positions (Greville abscissae). A head predicts small 3D residuals for them in the
  local path frame, scaled by 4 voxels. For fallback chord pairs the scale is 12 voxels.
- The last layer is zero-initialised, so the initial curve follows the traced path.
- There are three heads, chosen per edge by a fixed rule on the traced path (no learned
  selector):

  | Edge | Controls | Interior points |
  |---|---|---|
  | default, and all fallback pairs | **K8** | 6 |
  | traced, arc ≥ 15 mm or arc/chord ≥ 1.15 | **K16** | 14 |
  | traced, arc ≥ 35 mm or arc/chord ≥ 1.5 | **K24** | 22 |

  Each head is 6,983 → 256 → 3 × (interior points).
- The curve is sampled at 64 points.

### 2.5 Training

**Labels without Hungarian matching.** Each training patch contributes two sets of candidate
pairs:

| Source | Nodes | Node → GT mapping | Pair label |
|---|---|---|---|
| Predicted | the model's own peaks (≥ 0.2, at most 200) | one-to-one matching to GT nodes within 2 mm, the *same* rule the evaluation uses | positive iff both peaks are matched and their GT nodes share an edge; every pair touching an unmatched (false or duplicate) peak is a negative |
| Jittered GT | GT nodes + 0.5 mm Gaussian jitter | identity | positive iff a GT edge |

The jittered-GT source bootstraps the edge and geometry heads before the heatmap is good. In
the from-scratch dense model it is used alone for the first 2,000 updates. All positives are
kept; negatives are randomly subsampled to at most 256 per patch and source.

**Losses** (all weights 1):
- **L_node:** heatmap focal;
- **L_offset:** L1;
- **L_centreline:** BCE;
- **L_edge:** BCE with positive weight min(#neg/#pos, 20);
- **L_geometry:** symmetric chamfer distance in mm between the predicted curve and the GT dense
  centreline (the original traced path, not a fitted spline), on positive pairs.

**Gradient flow.**
- The edge and geometry losses train the corridor CNN, the heads and, through the sampling,
  the U-Net's shared feature channels.
- Peak positions, offsets, endpoint heat values and tracing are computed without gradient.
  So node placement stays a pure detection problem, trained only by L_node and L_offset.

**Recipe.**
- **Optimizer:** AdamW, LR 1e-3, OneCycle schedule (5% warm-up), weight decay 1e-4, gradient
  clipping at norm 5, batch 8, bf16 for the U-Net.
- **Augmentation:** random 90° rotations in all three planes, and Gaussian noise (p 0.35,
  std ≤ 0.015). No zoom.
- **Dense 16-NN model:** 15,000 updates from scratch.
- **Traced model:** continued from that checkpoint for 7,500 updates at LR 3e-4. The U-Net and
  corridor CNN are inherited; the edge and geometry heads are new, because their input grew by
  the arc-length feature.
- **Speed:** about 0.2 s/update (16-NN) and 0.76 s/update (traced, because of the tracing) on
  one H100.

### 2.6 Inference and threshold selection

1. **Nodes:** run the U-Net, then keep peaks ≥ the node threshold.
2. **Candidates:** trace on the predicted centreline; candidates = touching regions ∪ 16-NN
   pairs among the kept nodes.
3. **Edges:** keep pairs with P(edge) ≥ the edge threshold; draw each kept edge with its
   adaptive-K curve.
4. **Thresholds**, chosen on 256 *training* patches only:
   - first the node threshold maximising node F1 (grid 0.10–0.70);
   - then the edge threshold maximising edge F1 (grid 0.10–0.90).

   Validation is never used for any choice.

### 2.7 Building blocks and their origin

| Block | Origin |
|---|---|
| 3D U-Net (`monai.networks.nets.BasicUNet`) | third-party, used as-is (U-Net: Ronneberger et al., 2015) |
| Heatmap target, penalty-reduced focal loss, offset head, peak NMS | formulation of CenterNet (Zhou et al., 2019), re-implemented here |
| Trilinear feature sampling (`torch.nn.functional.grid_sample`) | PyTorch, as-is |
| One-to-one node matching (`scipy.optimize.linear_sum_assignment`) | SciPy, as-is |
| Cubic B-spline basis (`models/spline.py`) | earlier in this project (used by the RelationFormer K8 head) |
| Geodesic-Voronoi tracing, path read-back, path-following corridor, corridor CNN, edge head, adaptive-K geometry heads, label construction from predicted peaks | new in this project. Tracing follows the classic minimal-path idea for vessel tracking (Cohen & Kimmel, 1997) |

### 2.8 How it differs from RelationFormer

| | RelationFormer (B0) | Dense + traced |
|---|---|---|
| Resolution the decisions see | 2×2×2 feature grid (stride 32) | full 64³ |
| Nodes | fixed set of learned queries; coordinates regressed | heatmap peaks + sub-voxel offsets |
| Training assignment | Hungarian on moving predictions; training only | same one-to-one 2 mm matching as the evaluation; train = inference |
| Edge candidates | all pairs of kept queries | nodes connected along the predicted centreline (+ 16-NN) |
| Edge evidence | two node tokens + global relation tokens; never the image between the nodes | image features sampled along the path between the nodes |
| Edge shape | K8 spline from tokens; the loss never reaches the detector | adaptive K8/K16/K24 spline around the traced path; the loss shapes the shared features |
| Pretraining | SyntheticMRI, then fine-tuning | none (from scratch on the target) |

Code:
- [`scripts/dense_graph_20261030.py`](../../scripts/dense_graph_20261030.py): stage 1, 16-NN stage 2, training, evaluation;
- [`scripts/dense_graph_traced_20261030.py`](../../scripts/dense_graph_traced_20261030.py): tracing, path corridor, adaptive K;
- [`scripts/dense_graph_2d_20261030.py`](../../scripts/dense_graph_2d_20261030.py): the 2D port (16-NN version).

## 3. How the models were trained

| Model | Initialization | Training |
|---|---|---|
| Dense 16-NN | **from scratch** | 15k updates × batch 8 on the target dataset |
| Dense + traced | the dense 16-NN checkpoint | +7.5k updates × 8, same dataset, LR 3e-4 |
| Dense 16-NN continued (control for traced) | the same checkpoint | the same +7.5k updates |
| RelationFormer B0 / D08 / literature scale / anchored | **SyntheticMRI-pretrained** checkpoint | fine-tuned on the target: IXI 31k × 32, BRAVA 82k × 32, anchored ~15k × 32 |
| RelationFormer trained without spline (straight) | same as B0 | B0's exact recipe, spline loss weight 0 |

Training cohorts (subject-disjoint, fixed splits):

| Dataset | Train patches | Val patches |
|---|---|---|
| SyntheticMRI | 4,096 | 480 |
| IXI | 4,096 (literature-scale RelationFormer: all 10,000) | 512 |
| BRAVA | 8,768 | 1,796 |

The dense models therefore win with **no pretraining** and **8–20× fewer patch exposures**.

## 4. Main results (3D, validation)

### 4.1 Graph detection: node / edge / Branch / Curve F1

| Model | SyntheticMRI | IXI | BRAVA |
|---|---|---|---|
| RelationFormer, straight (trained) | 0.794 / 0.608 / 0.197 / 0.567 | 0.504 / 0.154 / 0.022 / 0.314 | 0.364 / 0.069 / 0.005 / 0.146 |
| RelationFormer B0 (K8 spline) | 0.794 / 0.607 / 0.197 / 0.615 | 0.508 / 0.156 / 0.024 / 0.344 | 0.368 / 0.073 / 0.005 / 0.161 |
| Heatmap-anchored RelationFormer | — | 0.640 / 0.263 / 0.123 / 0.472 | 0.547 / 0.154 / 0.035 / 0.229 |
| Dense 16-NN | 0.890 / 0.792 / 0.429 / 0.936 | 0.648 / 0.315 / 0.163 / 0.677 | 0.543 / 0.217 / 0.054 / 0.365 |
| **Dense + traced** | **0.894 / 0.797 / 0.441 / 0.945** | 0.649 / 0.313 / **0.179 / 0.770** | **0.546** / 0.206 / **0.062 / 0.395** |

- RelationFormer IXI is the literature-scale run (100 epochs); BRAVA is D08 (300 epochs).
- Paired 95% CIs over validation subjects:
  - dense − RelationFormer: every metric's CI excludes zero;
  - traced − 16-NN continued: Curve F1 +0.097 [+0.085, +0.112] (IXI) and +0.104 [+0.096, +0.113]
    (BRAVA); Branch F1 +0.01–0.02; edge F1 within ±0.006.
- The 16-NN continued control's own BRAVA scores are 0.545 / 0.202 / 0.050 / 0.291.

### 4.2 Average precision (dense 16-NN)

| | Node AP @2 mm | Box node mAP | Box edge mAP |
|---|---|---|---|
| SyntheticMRI | 0.860 | 0.638 | 0.622 |
| IXI | 0.613 | 0.337 | 0.215 |
| BRAVA | 0.502 | 0.220 | 0.089 |
| RelationFormer literature scale, IXI | 0.446 | 0.126 | 0.066 |

- Box mAP is the Relationformer/GrEVe metric (IoU 0.50:0.95). The dense model uses fixed
  GT-size boxes; RelationFormer uses its regressed boxes.
- AP was not computed for the traced model.

### 4.3 Topology (per patch; GT in brackets)

| Model | β₀ error | β₁ error | Loops | SMD ×10³ |
|---|---|---|---|---|
| SyntheticMRI: RelationFormer straight / B0 | 1.77 / 1.73 | 0.56 / 0.65 | 0.56 / 0.64 (0.04) | 3.01 / 2.99 |
| SyntheticMRI: dense | 1.01 | 0.32 | 0.29 | 1.84 |
| SyntheticMRI: **traced** | 1.02 | **0.22** | **0.20** | **1.81** |
| IXI: RelationFormer straight / B0 | 2.97 / 2.98 | 8.33 / 8.65 | 9.0 / 9.3 (0.71) | 43.0 / 38.6 |
| IXI: anchored | 1.93 | 3.63 | 4.29 | 29.5 |
| IXI: dense | 4.02 | **3.05** | 3.64 | 27.0 |
| IXI: **traced** | **1.68** | 3.50 | 4.10 | **23.2** |
| BRAVA: RelationFormer straight / B0 | **2.22** / 2.59 | 3.32 / 2.52 | 3.3 / 2.5 (0.00) | 62.6 / 62.5 |
| BRAVA: anchored | 3.28 | 4.73 | 4.73 | **57.7** |
| BRAVA: dense | 7.15 | 1.50 | 1.50 | 61.0 |
| BRAVA: **traced** | 4.08 | **0.62** | **0.62** | 63.8 |

Notes:
- **Definitions.** β₀ = connected components; β₁ = independent loops. SMD is the street mover
  distance, native protocol: 100 points, normalized coordinates, straight segments.
- **Loops.** The dense models create far fewer spurious loops than RelationFormer (IXI 3.6–4.1
  vs 9.0–9.3 per patch; BRAVA traced 0.6 vs 2.5–3.3).
- **Components on BRAVA.** The dense models still fragment the graph (9.6–13.3 components vs
  6.3 GT): many detected nodes end up without an accepted edge. Tracing halves the excess,
  and RelationFormer is closer here.
- **SMD.** It ignores patches where a graph has no edges. The number of such patches differs
  between arms (BRAVA 78–236), so BRAVA SMD should be read with care.

Script: [`scripts/topology_metrics_20261102.py`](../../scripts/topology_metrics_20261102.py).
Output: `experiments/dense_graph_20261030/topology-validation.json`.

### 4.4 Long edges: fraction of GT edges recovered, by arc length

| Arc length | IXI: 16-NN | IXI: traced | BRAVA: 16-NN | BRAVA: traced |
|---|---|---|---|---|
| 0–10 mm | 0.27 | 0.26 | 0.19 | 0.18 |
| 20–40 mm | 0.17 | **0.30** | 0.08 | **0.11** |
| ≥ 40 mm | 0.01 | **0.15** | 0.02 | **0.05** |

The remaining BRAVA misses at ≥ 20 mm are mostly missing endpoints (78–82%), not rejected
edges.

### 4.5 Curved vs straight edges

- **Learnability, given the true graph.** Mean distance to the real centreline:
  - BRAVA: straight 1.48 mm → K8 0.62 → K24 0.49;
  - IXI: 0.54 → 0.19;
  - SyntheticMRI: 0.52 → 0.15.

  At equal output count, a learned polyline (points inside the edge) performs like the spline.
- **Within RelationFormer.** Training with vs without the spline loss gives the same detector:
  node, edge and Branch F1 differ by ≤ 0.005, with CIs spanning about zero. The spline adds only
  Curve F1, +0.015–0.048 over chords. In this recipe the geometry loss cannot reach the
  detector, so the edge representation does not change detection.
- **Within the dense model.** Curve F1 is 0.68–0.95, against 0.31–0.62 for RelationFormer.

## 5. 2D (validation; node / edge F1 @2 units)

| Dataset | RelationFormer B0 (mixed pretraining) | Dense 16-NN, from scratch |
|---|---|---|
| AV-WIDE | 0.204 / 0.012 | **0.454 / 0.133** |
| REAVER | 0.745 / 0.408 | **0.806 / 0.589** |
| OCTA-Synth | 0.804 / 0.595 | **0.903 / 0.782** |

- Traced connectivity is not yet ported to 2D.
- AV-WIDE is node-limited.
- REAVER overfits and has 2 validation images.

Report: `DenseVesselGraph/experiments/dense_graph_2d_20261030/report.md`.

## 6. Limitations and next steps

- **Limitations:**
  - single seed;
  - validation only;
  - BRAVA has 8 validation subjects;
  - tracing and adaptive K changed together, so their contributions are not separated;
  - the traced model was continued from the dense checkpoint rather than trained from scratch;
  - the dense model needs train-calibrated thresholds (at a fixed 0.5 it underperforms on
    BRAVA).
- **Next steps:**
  1. **Node recall on real data.** It limits both edges and long-vessel recovery. Candidates:
     reading boundary nodes off the centreline where it crosses the patch face, and junction
     detection.
  2. **BRAVA fragmentation.** Remove or reconnect detected nodes with no accepted edge.
  3. **Ablation.** Tracing with K8 only, to separate tracing from adaptive K.
  4. **2D.** Port tracing to 2D.
  5. **Before any paper claim.** Seeds, then one locked test evaluation.
