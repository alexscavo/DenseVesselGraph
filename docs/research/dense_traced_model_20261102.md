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
- **Variants tested 3–4 Nov (§4.7–4.9; single seed, validation only).** Neighbourhood offsets lower node error
  slightly. The pass-through decision adds Edge F1 +0.003–0.010 on a fixed model, without improving long edges.
  U-KAN does not beat the U-Net or its MLP control at this budget. On BRAVA the choice of node threshold
  (0.25 vs 0.30) moves Edge F1 by about 0.03, more than any of these changes.
- **Curve shape and feature width (9 Nov, §4.12–4.13; single seed, validation only).** Short straight edges were
  drawn as sub-millimetre coils (arc/chord 2–4, invisible to Curve F1). An arc-length penalty in the geometry loss
  removes them with no metric loss and is now the training default; compressing each output curve to the fewest
  controls within 0.5 voxel cuts the representation from about 13 to 8 controls per edge at no cost. The 32 feature
  channels are kept: stage 2 uses about 12–16 directions, and a 16-channel model loses Curve F1 on TopBrain CTA.

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

Stage-1 outputs of `traced_adaptive` on one validation patch per dataset (detections at the train-selected node
threshold, 2 mm matching):

![Stage-1 U-Net outputs: image MIP with GT graph, node heatmap, detected nodes, sub-voxel offset zoom, centreline probability](figures/unet_outputs_20261102/unet_outputs_3d.png)

A second set of patches: [`unet_outputs_3d_example2.png`](figures/unet_outputs_20261102/unet_outputs_3d_example2.png).
Interactive 3D views of the same patches, prediction vs GT:
[SyntheticMRI](figures/interactive_3d_20261103/prediction_3d_SyntheticMRI.html),
[IXI](figures/interactive_3d_20261103/prediction_3d_IXI.html),
[BRAVA](figures/interactive_3d_20261103/prediction_3d_BRAVA.html). GitHub shows HTML as source, so download the
file and open it in a browser.

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
- K is never learned: it is a fixed rule on the predicted traced path, or (option, 9 Nov) a compression of the
  predicted curve. Options in `dense_graph_traced_20261030.py`, all recorded in each run's `final.json` (§4.12):
  - `--k-rule ratio` (default): the table above; `--k-min-arc 8 20` (training default since 9 Nov) lets the arc/chord
    ratio raise K only once the traced arc reaches 8 mm (K16) or 20 mm (K24);
  - `--k-rule fit`: the fewest controls whose least-squares fit of the traced path (smoothed at 1 mm) is within a
    tolerance; tested and not adopted;
  - `--k-rule fixed24`: a single K24 head for every pair;
  - output compression (`MODE['compress_tol']`, evaluation only): each predicted curve is replaced by the
    fewest-control (K8, then K16) endpoint-pinned least-squares spline that reproduces it within 0.5 voxel
    (Hausdorff), never more controls than it had.

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
  - The reference is the GT dense path resampled to 64 points by arc length (cache key `guides`, within about
    0.1 mm of the path). The K8 fit stored in the caches (`curves`, drawn as "GT edge (B-spline)" in the 3D viewers)
    is never used by the dense model; on long tortuous edges it leaves the vessel (§4.12).
  - **Arc-length penalty (training default since 9 Nov 2026, weight λ = 0.1):** each positive pair adds
    λ · max(0, L_pred − 1.02 · L_GT − 0.1 mm), where L_pred and L_GT are the arc lengths of the predicted curve and of
    the reference (both from their 64 samples). The chamfer averages point distances, so a curve that coils inside the
    vessel costs almost nothing; a coil is pure extra length, which this term charges. It never fires on a curve no
    longer than the GT (+2% + 0.1 mm slack for sampling), so smooth bends and sideways shifts are not penalised.
    `--arc-penalty 0` restores the original loss; runs before `traced_arcpen_r1109` used 0. Stage 1
    (`dense_graph_20261030.py`) is unchanged.

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

### 4.6 Soft tracing and gap features (DONE, 3 Nov 2026)

Two variants of the traced model (Dense + traced, called `traced_adaptive` here), with the same
recipe and budget: warm start from `spline6`, 7500 updates, lr 3e-4.
- `traced_gap`: the edge and geometry heads also read the centreline probability along the path.
- `traced_soft`: `traced_gap`, plus no hard centreline mask (ink may cross any voxel at cost
  1/(p+0.05)), plus candidates from geodesic contacts only, with no 16-NN fallback pairs.

**Main result.** Neither variant improves long-edge recovery on IXI or BRAVA. Gap features alone
change little. Soft tracing helps BRAVA Curve F1, hurts IXI, and removes most spurious IXI loops
at the cost of edge recall.

Paired differences in F1, with 95% CIs over validation subjects:

| Dataset | Comparison | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|
| SyntheticMRI | gap − adaptive | −0.001 [−0.003, +0.002] | 0.000 [−0.004, +0.005] | −0.004 [−0.012, +0.004] | +0.002 [−0.001, +0.006] |
| SyntheticMRI | soft − gap | +0.001 [−0.001, +0.002] | +0.001 [−0.004, +0.005] | −0.002 [−0.008, +0.004] | **−0.008 [−0.011, −0.006]** |
| IXI | gap − adaptive | +0.004 [−0.003, +0.011] | **+0.010 [+0.004, +0.018]** | **+0.010 [+0.002, +0.017]** | **+0.012 [+0.004, +0.022]** |
| IXI | soft − gap | −0.005 [−0.012, +0.001] | **−0.019 [−0.026, −0.012]** | **−0.017 [−0.027, −0.005]** | **−0.084 [−0.093, −0.075]** |
| BRAVA | gap − adaptive | −0.002 [−0.004, +0.000] | 0.000 [−0.005, +0.005] | −0.003 [−0.008, +0.003] | +0.002 [−0.001, +0.006] |
| BRAVA | soft − gap | **+0.003 [+0.001, +0.005]** | **−0.003 [−0.005, −0.001]** | **+0.009 [+0.004, +0.013]** | **+0.025 [+0.019, +0.032]** |

Bold marks a CI that excludes zero. Absolute node / edge / Branch / Curve F1:

| Dataset | `traced_adaptive` | `traced_gap` | `traced_soft` |
|---|---|---|---|
| SyntheticMRI | 0.894 / 0.797 / 0.441 / 0.945 | 0.893 / 0.797 / 0.437 / 0.948 | 0.893 / 0.798 / 0.435 / 0.939 |
| IXI | 0.649 / 0.313 / 0.179 / 0.770 | 0.653 / 0.323 / 0.189 / 0.783 | 0.648 / 0.305 / 0.173 / 0.699 |
| BRAVA | 0.546 / 0.206 / 0.062 / 0.395 | 0.544 / 0.206 / 0.059 / 0.397 | 0.547 / 0.203 / 0.068 / 0.423 |

Fraction of long GT edges recovered:

| Dataset | Arc length (GT edges) | adaptive | gap | soft | gap − adaptive | soft − gap |
|---|---|---|---|---|---|---|
| SyntheticMRI | ≥ 20 mm (1556) | 0.809 | 0.814 | 0.797 | +0.004 [−0.001, +0.011] | **−0.017 [−0.027, −0.008]** |
| SyntheticMRI | ≥ 40 mm (145) | 0.710 | 0.717 | 0.697 | +0.007 [−0.037, +0.044] | −0.021 [−0.060, +0.025] |
| IXI | ≥ 20 mm (648) | 0.279 | 0.279 | 0.219 | 0.000 [−0.024, +0.023] | **−0.060 [−0.089, −0.031]** |
| IXI | ≥ 40 mm (83) | 0.145 | 0.108 | 0.120 | −0.036 [−0.086, +0.014] | +0.012 [−0.052, +0.071] |
| BRAVA | ≥ 20 mm (5042) | 0.092 | 0.092 | 0.091 | −0.001 [−0.008, +0.007] | −0.001 [−0.007, +0.005] |
| BRAVA | ≥ 40 mm (1251) | 0.052 | 0.050 | 0.058 | −0.002 [−0.011, +0.010] | +0.008 [−0.003, +0.020] |

Topology per patch, in the format of §4.3 (GT loops in brackets):

| Dataset: arm | β₀ error | β₁ error | Loops | SMD ×10³ | Patches without SMD |
|---|---|---|---|---|---|
| SyntheticMRI: `traced_adaptive` | 1.02 | 0.22 | 0.20 (0.04) | 1.81 | 0 |
| SyntheticMRI: `traced_gap` | 0.99 | 0.26 | 0.24 (0.04) | 1.69 | 0 |
| SyntheticMRI: `traced_soft` | 1.17 | 0.12 | 0.09 (0.04) | 1.96 | 0 |
| IXI: `traced_adaptive` | 1.68 | 3.50 | 4.10 (0.71) | 23.24 | 42 |
| IXI: `traced_gap` | 3.00 | 2.69 | 3.25 (0.71) | 21.35 | 32 |
| IXI: `traced_soft` | 3.17 | 0.74 | 0.86 (0.71) | 23.19 | 50 |
| BRAVA: `traced_adaptive` | 4.08 | 0.62 | 0.62 (0.00) | 63.79 | 225 |
| BRAVA: `traced_gap` | 3.96 | 0.72 | 0.72 (0.00) | 62.45 | 226 |
| BRAVA: `traced_soft` | 3.85 | 0.60 | 0.60 (0.00) | 58.63 | 190 |

Reading the results:
- **Traced pair fraction.** On validation candidates: `traced_adaptive` and `traced_gap` 0.10–0.13,
  `traced_soft` 1.0. In the last training log: 0.09–0.23 and 1.0.
- **Why long edges do not improve.** Most misses on BRAVA are missing endpoints: 79–80% of edges
  ≥ 20 mm in all three arms, which no change to stage 2 can fix. Soft tracing halves rejections there
  (537 → 274 against `traced_gap`), but edges that are not candidates rise from 41 to 320, so
  recovery is unchanged. These two effects come from the soft domain and the dropped fallback
  together and cannot be separated here.
- **IXI loops.** `traced_soft` comes close to the GT loop count (0.86 vs 0.71), but keeps 10.9 edges
  per patch against 16.9 for `traced_gap`: edge precision rises from 0.366 to 0.451 and recall
  falls from 0.289 to 0.230.

Caveats:
- Single seed, validation only, all arms warm-started from the same `spline6`. The CIs cover
  subject sampling, not training variation, and are not corrected for multiple comparisons, so
  small effects such as the IXI gain from `traced_gap` need more seeds.
- Thresholds differ between arms, each selected on `monitor_train`. On IXI the node threshold is
  0.30 for `traced_gap` and 0.35 for the other two (21.3 vs 17.8 nodes per patch). This probably
  explains part of its worse β₀ error (3.00 vs 1.68) and fewer missing endpoints, and it confounds
  both IXI comparisons.
- SMD skips patches with no predicted edges, and their number differs between arms.
- `traced_adaptive` was trained on 30 Oct, before the flags were added. The new code was not rerun
  with the flags off to confirm it reproduces that run.

Scripts: `scripts/compare_traced_knn_20261030.py <dataset> <arm_a> <arm_b>`,
`scripts/compare_arc_length_20261102.py` and `scripts/topology_metrics_20261102.py`.
Outputs in `experiments/dense_graph_20261030/`: `paired-traced_*-vs-*.json`,
`paired-arc-traced_*-vs-*.json` and `topology-validation.json`.

### 4.7 Neighbourhood offsets (DONE, 3–4 Nov 2026)

`traced_offset` is `traced_adaptive` with one training change: the node offset is supervised on every
voxel within 1.75 voxels of a GT node, not only on the rounded GT voxel (`--offset-radius 1.75`, now the
default). Same warm start from `spline6`, 7500 updates, lr 3e-4.

**Main result.** Offsets lower node error slightly and, where thresholds are equal, raise Branch F1
sharply on SyntheticMRI. The large BRAVA gains come mostly from a lower selected node threshold, and
they cost fragmentation.

Paired differences, `traced_offset` − `traced_adaptive`, with 95% CIs over validation subjects. Bold
marks a CI that excludes zero (some bounds round to 0.000):

| Dataset | Thresholds, offset vs adaptive (node, edge) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|
| SyntheticMRI | 0.45, 0.9 vs 0.45, 0.9 | +0.003 [0.000, +0.006] | +0.001 [−0.005, +0.006] | **+0.086 [+0.067, +0.106]** | **+0.008 [+0.003, +0.012]** |
| IXI | 0.35, 0.8 vs 0.35, 0.75 | −0.001 [−0.004, +0.002] | +0.006 [−0.002, +0.012] | **+0.007 [0.000, +0.014]** | **−0.009 [−0.015, −0.002]** |
| BRAVA | 0.25, 0.75 vs 0.3, 0.75 | **+0.007 [+0.002, +0.013]** | **+0.031 [+0.025, +0.038]** | **+0.013 [+0.008, +0.017]** | **+0.071 [+0.064, +0.079]** |

The same comparison with `traced_offset` decoded at `traced_adaptive`'s thresholds, to remove the shift:

| Dataset | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|
| IXI (exact) | −0.001 [−0.004, +0.002] | +0.003 [−0.004, +0.009] | +0.006 [−0.002, +0.013] | **+0.009 [+0.002, +0.016]** |
| BRAVA (approximate) | **+0.004 [+0.002, +0.006]** | +0.003 [0.000, +0.007] | **+0.007 [+0.003, +0.012]** | **−0.048 [−0.052, −0.043]** |

- **IXI** is exact, because an edge threshold only filters scores. The Curve F1 loss above reverses into a
  gain, so that loss was the threshold shift.
- **BRAVA** is approximate. Saved rows were traced among nodes ≥ 0.25, and filtering them to ≥ 0.30 does
  not repeat tracing among the retained nodes. Candidates that ran through a removed node are lost, which
  penalises `traced_offset`, so its Curve F1 here is not a fair estimate. An exact answer needs a GPU
  re-evaluation with the node threshold fixed; this was not run.

Matched-node error in mm, median / 90th percentile (each arm's own matched nodes, no CI):

| Dataset | `traced_adaptive` | `traced_offset` |
|---|---|---|
| SyntheticMRI | 0.65 / 1.39 | 0.57 / 1.16 |
| IXI | 0.46 / 1.05 | 0.43 / 1.04 |
| BRAVA | 0.74 / 1.44 | 0.73 / 1.41 |

Reading the results:
- **Fragmentation on BRAVA.** The lower threshold keeps 20.4 nodes per patch against 15.1, and β₀ error rises
  from 4.08 to 6.28 and loops from 0.62 to 1.28 (table in §4.8). Missing endpoints among edges ≥ 20 mm fall
  from 4024 to 3553, which accounts for the +0.030 in long-edge recovery.
- **Branch F1 on SyntheticMRI.** This is the largest effect at equal thresholds and it comes with the lower
  node error. Why it is confined to SyntheticMRI was not investigated.

### 4.8 Pass-through decision (DONE, 3–4 Nov 2026)

`traced_passthrough` is `traced_offset` plus two-hop candidates and a pass-through head (`--passthrough`).
The head scores each traced chain A–C–B, where C is a detected node between A and B. At inference, chains
are visited by decreasing score q. If q ≥ τ and C has no accepted edge other than to A or B, then C, A–C
and C–B are replaced by A–B. τ is selected on `monitor_train` after the node and edge thresholds. Two-hop
rows are never ordinary edges, so A–B can be accepted only through this decision.

**Main result.** On a fixed model the decision helps a little on all three datasets: Edge F1 rises by
0.003–0.010, because it removes more false edges than true ones. It does not improve long-edge recovery,
and on real data the arm recovers fewer blocked edges than `traced_offset`. Its clearest gain is fewer
spurious loops, and that comes from training with the two-hop rows rather than from the decision.

**How the head works.** Scores for 2 mm-labelled chains on validation (log scale, because real nodes
outnumber correct chains 19–43×):

![Pass-through score per chain category](figures/passthrough_20261103/q_distributions.png)

The head separates correct chains well on SyntheticMRI and only partly on IXI and BRAVA. On all three, most
chains above τ are still not correct (100 of 277, 76 of 242, 232 of 1096 are), because the other categories are
so much larger. Examples, before and after the decision:

![Pass-through decisions, before and after](figures/passthrough_20261103/examples.png)

**The decision alone:** the same model with contraction on vs off, at the same node and edge thresholds.
This is the only comparison here free of threshold shifts and of training differences:

| Dataset | τ | Node F1 | Edge F1 | Branch F1 | Curve F1 | True edges | False edges |
|---|---|---|---|---|---|---|---|
| SyntheticMRI | 0.75 | **+0.002 [0.000, +0.003]** | **+0.003 [+0.001, +0.006]** | **+0.007 [+0.004, +0.010]** | **+0.005 [+0.003, +0.007]** | +22 | −29 |
| IXI | 0.85 | **+0.001 [0.000, +0.003]** | **+0.005 [+0.002, +0.008]** | **+0.002 [0.000, +0.005]** | **+0.005 [+0.003, +0.007]** | +34 | −107 |
| BRAVA | 0.65 | **+0.002 [+0.001, +0.003]** | **+0.010 [+0.006, +0.013]** | 0.000 [−0.001, +0.001] | **+0.002 [+0.001, +0.002]** | +129 | −235 |

**What each contraction did:**

| Dataset | Contractions (per patch) | A–B correct, strict | A–B correct, final graph | C within 2 mm of A or B | Sides accepted before: both / one / none | True edges removed | Correct chains contracted |
|---|---|---|---|---|---|---|---|
| SyntheticMRI | 165 (0.34) | 0.39 | 0.59 | 0.45 | 0.21 / 0.62 / 0.17 | 79 | 0.41 |
| IXI | 170 (0.33) | 0.39 | 0.51 | 0.43 | 0.49 / 0.44 / 0.06 | 55 | 0.21 |
| BRAVA | 743 (0.41) | 0.24 | 0.44 | 0.87 | 0.19 / 0.76 / 0.05 | 212 | 0.26 |

- *Strict* uses the node matching before contraction: C spurious and A–B a GT edge. *Final graph* re-matches
  the kept nodes after contraction. The two differ because the matching is one-to-one: with C and an endpoint
  both near one GT node, removing C lets that endpoint take the match.
- *Sides accepted* counts how many of A–C and C–B the edge head had accepted. The rule does not require either.
- *Correct chains contracted* is the share of chains with the strict "correct" label that were contracted.

**Blocked edges:** GT edges with both endpoints matched and a spurious detection within 1.5 mm of their
interior. The set is fixed before contraction and recovery is counted in the final graph:

| Dataset | `traced_adaptive` | `traced_offset` | Pass-through, off | Pass-through, on | On − off | Pass-through − offset |
|---|---|---|---|---|---|---|
| SyntheticMRI | 34/100 (0.34) | 26/94 (0.28) | 13/110 (0.12) | 45/110 (0.41) | **+0.291 [+0.204, +0.378]** | +0.132 [−0.007, +0.265] |
| IXI | 126/318 (0.40) | 98/313 (0.31) | 22/306 (0.07) | 68/306 (0.22) | **+0.150 [+0.105, +0.200]** | **−0.091 [−0.135, −0.044]** |
| BRAVA | 82/295 (0.28) | 165/758 (0.22) | 25/308 (0.08) | 51/308 (0.17) | **+0.084 [+0.058, +0.108]** | **−0.052 [−0.101, −0.016]** |

How a blocked edge arises, one example per dataset (`traced_adaptive`): a detection C on the vessel between A and B
splits the geodesic regions, so tracing proposes A–C and C–B, and A–B is left to the straight 16-NN fallback.

![In-between detection C blocks the traced candidate A–B](figures/unet_outputs_20261102/tracing_blocking_examples.png)

The arms' blocked sets differ, so the last column compares rates. On BRAVA `traced_offset` runs at a lower node
threshold; against `traced_adaptive`, at the same thresholds, the difference is **−0.112 [−0.165, −0.073]**.

**Against `traced_offset`,** each arm at its own thresholds:

| Dataset | Thresholds, pass-through vs offset (node, edge) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|
| SyntheticMRI | 0.45, 0.9 vs 0.45, 0.9 | 0.000 [−0.002, +0.003] | +0.003 [−0.003, +0.009] | −0.004 [−0.013, +0.005] | **+0.005 [+0.001, +0.009]** |
| IXI | 0.35, 0.8 vs 0.35, 0.8 | +0.004 [0.000, +0.008] | +0.001 [−0.007, +0.009] | +0.006 [0.000, +0.013] | **−0.009 [−0.016, −0.002]** |
| BRAVA | 0.3, 0.75 vs 0.25, 0.75 | −0.003 [−0.009, +0.002] | **−0.030 [−0.038, −0.023]** | +0.007 [−0.001, +0.014] | **−0.077 [−0.086, −0.070]** |

The BRAVA row is confounded by the node threshold. With `traced_offset` filtered to 0.30 (approximate, see
§4.7), the Edge F1 difference is −0.002 [−0.006, +0.002] in pass-through's direction. So there is no
detectable Edge F1 change, and the −0.030 should not be read as the head's effect.

Absolute node / edge / Branch / Curve F1:

| Dataset | `traced_adaptive` | `traced_offset` | `traced_passthrough` |
|---|---|---|---|
| SyntheticMRI | 0.894 / 0.797 / 0.441 / 0.945 | 0.897 / 0.798 / 0.527 / 0.953 | 0.897 / 0.801 / 0.523 / 0.958 |
| IXI | 0.649 / 0.313 / 0.179 / 0.770 | 0.648 / 0.319 / 0.187 / 0.762 | 0.651 / 0.320 / 0.193 / 0.753 |
| BRAVA | 0.546 / 0.206 / 0.062 / 0.395 | 0.553 / 0.237 / 0.075 / 0.466 | 0.549 / 0.207 / 0.082 / 0.389 |

Fraction of long GT edges recovered:

| Dataset | Arc length (GT edges) | adaptive | offset | pass-through | offset − adaptive | pass-through − offset |
|---|---|---|---|---|---|---|
| SyntheticMRI | ≥ 20 mm (1556) | 0.809 | 0.815 | 0.812 | +0.006 [−0.004, +0.017] | −0.003 [−0.018, +0.013] |
| SyntheticMRI | ≥ 40 mm (145) | 0.710 | 0.731 | 0.752 | +0.021 [−0.014, +0.056] | +0.021 [−0.014, +0.063] |
| IXI | ≥ 20 mm (648) | 0.279 | 0.276 | 0.265 | −0.003 [−0.029, +0.023] | −0.011 [−0.032, +0.011] |
| IXI | ≥ 40 mm (83) | 0.145 | 0.157 | 0.169 | +0.012 [−0.053, +0.074] | +0.012 [−0.027, +0.053] |
| BRAVA | ≥ 20 mm (5042) | 0.092 | 0.122 | 0.089 | **+0.030 [+0.016, +0.042]** | **−0.034 [−0.046, −0.020]** |
| BRAVA | ≥ 40 mm (1251) | 0.052 | 0.058 | 0.054 | +0.006 [−0.002, +0.016] | −0.004 [−0.019, +0.011] |

Both BRAVA ≥ 20 mm differences follow the node threshold: missing endpoints are 4024 / 3553 / 4014.

Topology per patch, in the format of §4.3:

| Dataset: arm | β₀ error | β₁ error | Loops | SMD ×10³ | Patches without SMD | Nodes | Edges |
|---|---|---|---|---|---|---|---|
| SyntheticMRI: `traced_adaptive` | 1.02 | 0.22 | 0.20 (0.04) | 1.81 | 0 | 19.3 | 13.7 |
| SyntheticMRI: `traced_offset` | 0.94 | 0.28 | 0.25 (0.04) | 1.72 | 0 | 19.3 | 13.8 |
| SyntheticMRI: `traced_passthrough` | 0.83 | 0.16 | 0.13 (0.04) | 1.38 | 0 | 19.1 | 13.6 |
| IXI: `traced_adaptive` | 1.68 | 3.50 | 4.10 (0.71) | 23.24 | 42 | 17.8 | 16.4 |
| IXI: `traced_offset` | 1.81 | 2.25 | 2.82 (0.71) | 21.18 | 42 | 17.8 | 14.6 |
| IXI: `traced_passthrough` | 2.02 | 1.04 | 1.44 (0.71) | 21.21 | 42 | 17.5 | 12.6 |
| BRAVA: `traced_adaptive` | 4.08 | 0.62 | 0.62 (0.00) | 63.79 | 225 | 15.1 | 6.1 |
| BRAVA: `traced_offset` | 6.28 | 1.28 | 1.28 (0.00) | 49.67 | 172 | 20.4 | 9.3 |
| BRAVA: `traced_passthrough` | 4.41 | 0.30 | 0.30 (0.00) | 68.37 | 252 | 14.8 | 5.1 |

Reading the results:
- **Why Edge F1 rises although most contractions are not clean.** Only 0.44–0.59 of the new A–B edges are
  correct in the final graph, but what they replace is mostly wrong too: false edges fall by more than true
  edges are lost.
- **On BRAVA the head mostly removes near-duplicates.** In 87% of contractions C is within the 2 mm matching
  radius of an endpoint (43–45% on the other two). It is not mainly doing what it was designed for there,
  which is removing a spurious node in the middle of an edge.
- **Blocked edges.** The decision raises recovery by 0.08–0.29, but with two-hop rows barred from being ordinary
  edges the arm starts from only 0.07–0.12. `traced_offset` reaches 0.22–0.31 by having the ordinary edge
  head accept A–B as a 16-NN candidate, which can leave A–C and C–B accepted too. So on IXI and BRAVA the
  arm trades blocked-edge recovery for fewer loops.
- **Loops come from training, not from the decision.** Within `traced_passthrough`, turning contraction on
  barely changes β₁ error (0.158 → 0.160, 1.029 → 1.037, 0.286 → 0.300). The fall against `traced_offset` on
  IXI (2.25 → 1.04) comes with fewer edges (14.6 → 12.6 per patch) and precision 0.393 → 0.431, recall
  0.268 → 0.254. Contraction does improve β₀ error on all three datasets (1.06 → 0.83, 2.14 → 2.02,
  4.69 → 4.41).
- **Curves.** A contracted A–B uses the joined A–C–B path, which can wiggle at C (fifth row of the examples).
- **The guard.** It blocks some correct chains above τ (35, 10 and 51). It also lets through chains where neither
  side was accepted (5–17% of contractions), so a contraction can add an edge the edge head had rejected
  on both sides. Requiring accepted sides is an untested option.

Caveats:
- Single seed, validation only, all arms warm-started from the same `spline6`. The CIs cover subject sampling,
  not training variation, and are not corrected for multiple comparisons.
- Each arm has its own train-selected thresholds. `traced_offset` vs `traced_adaptive` is confounded on IXI
  (edge) and BRAVA (node), and `traced_passthrough` vs `traced_offset` on BRAVA (node).
- `traced_passthrough` differs from `traced_offset` in both two-hop training and the head, and those two were
  not separated. Only the on/off comparison isolates the inference decision.
- τ was selected on `monitor_train`, where contraction also helps (Edge F1 0.803 → 0.809, 0.343 → 0.346,
  0.189 → 0.200). The curve over τ is shallow, so τ itself is not sharply determined.
- `node-quality.json` has a `blocked_edges` field computed after contraction. It is biased low for
  `traced_passthrough` because successful contractions leave the denominator; use `passthrough-effect.json`.
- IXI and BRAVA `traced_passthrough` crashed in final evaluation after completing training, on patches with fewer
  than two detections (fixed in `87c37b4`). They were re-evaluated from the saved `model.pt`, not retrained. The
  rows the crashed jobs had already saved are byte-identical to the re-evaluation's. For all three datasets the
  paired comparison and arc-length counts reproduce the numbers in each run's `final.json`, which confirms that
  the scripts apply contraction as the model did.

Scripts: `scripts/passthrough_effect_20261104.py`, `scripts/passthrough_figures_20261103.py`,
`scripts/node_quality_20261103.py`, and the three in §4.6. Outputs in `experiments/dense_graph_20261030/`:
`passthrough-effect.json`, `passthrough-decisions.json`, `node-quality.json`, `paired-traced_offset-vs-*.json`,
`paired-traced_passthrough-vs-*.json`, the matching `paired-arc-*` files, and `topology-validation.json`.

### 4.9 U-KAN backbone (DONE, 3–4 Nov 2026)

The 3D U-Net is replaced following U-KAN (arXiv 2406.02918): its two deepest encoder stages and first two
decoder stages become tokenized KAN blocks (grid 5, cubic), with concatenated skips and nothing else changed.
Three backbones run through the full two-stage pipeline with the same optimiser, lr, schedule and budget:

| Name | Backbone | Stage 1 (15k updates, lr 1e-3) | Stage 2 (7500 updates, lr 3e-4, warm start) |
|---|---|---|---|
| U-Net | BasicUNet, the reference | `spline6` | `traced_adaptive` |
| U-KAN | KAN blocks in those four stages | `spline6_ukan` | `traced_adaptive_ukan` |
| MLP control | the same layout with `nn.Linear` in place of KAN layers | `spline6_ukan_mlp` | `traced_adaptive_ukan_mlp` |

**Main result.** U-KAN does not improve on the U-Net at this budget, and the KAN layers do not improve on the
MLP control. Edge F1 is lower on SyntheticMRI and IXI. On BRAVA the large gain comes from a lower selected
node threshold, and at shared thresholds it shrinks to about +0.004. Node F1 and node error are essentially
unchanged at both stages. Training was stable, so neither fallback (lower lr for KAN, `--backbone ukan_add`) applied.

Paired differences for the full model (stage 2), each arm at its own thresholds. Bold marks a CI that excludes
zero (some bounds round to 0.000):

| Dataset | Comparison | Thresholds (node, edge) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|---|
| SyntheticMRI | U-KAN − U-Net | 0.45, 0.9 vs 0.45, 0.9 | −0.004 [−0.009, 0.000] | **−0.010 [−0.016, −0.003]** | −0.006 [−0.023, +0.009] | +0.001 [−0.004, +0.006] |
| IXI | U-KAN − U-Net | 0.35, 0.8 vs 0.35, 0.75 | −0.004 [−0.010, +0.002] | −0.009 [−0.019, 0.000] | −0.004 [−0.013, +0.006] | **−0.015 [−0.023, −0.006]** |
| BRAVA | U-KAN − U-Net | 0.25, 0.75 vs 0.3, 0.75 | +0.006 [−0.001, +0.013] | **+0.035 [+0.030, +0.041]** | **+0.005 [+0.001, +0.009]** | **+0.082 [+0.072, +0.093]** |
| SyntheticMRI | U-KAN − MLP | 0.45, 0.9 vs 0.4, 0.9 | −0.001 [−0.006, +0.003] | **−0.005 [−0.011, 0.000]** | 0.000 [−0.012, +0.012] | −0.001 [−0.005, +0.003] |
| IXI | U-KAN − MLP | 0.35, 0.8 vs 0.35, 0.85 | −0.006 [−0.013, +0.001] | **−0.013 [−0.024, −0.002]** | −0.004 [−0.013, +0.005] | **+0.026 [+0.018, +0.034]** |
| BRAVA | U-KAN − MLP | 0.25, 0.75 vs 0.3, 0.7 | +0.004 [−0.002, +0.010] | **+0.035 [+0.030, +0.040]** | **+0.009 [+0.003, +0.014]** | **+0.070 [+0.061, +0.076]** |

The same comparisons with both arms decoded at shared thresholds (the higher node threshold, the second arm's
edge threshold). SyntheticMRI U-KAN − U-Net already shares its thresholds:

| Dataset | Comparison | Shared (node, edge) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|---|
| IXI (exact) | U-KAN − U-Net | 0.35, 0.75 | −0.004 [−0.010, +0.002] | **−0.012 [−0.022, −0.002]** | −0.002 [−0.011, +0.006] | +0.002 [−0.005, +0.010] |
| IXI (exact) | U-KAN − MLP | 0.35, 0.85 | −0.006 [−0.013, +0.001] | **−0.014 [−0.025, −0.003]** | −0.006 [−0.013, +0.002] | −0.004 [−0.014, +0.006] |
| SyntheticMRI (approximate) | U-KAN − MLP | 0.45, 0.9 | −0.002 [−0.006, +0.002] | −0.004 [−0.010, +0.001] | +0.001 [−0.011, +0.013] | +0.004 [−0.002, +0.009] |
| BRAVA (approximate) | U-KAN − U-Net | 0.3, 0.75 | **+0.004 [+0.001, +0.007]** | **+0.004 [0.000, +0.009]** | 0.000 [−0.004, +0.003] | **−0.039 [−0.046, −0.029]** |
| BRAVA (approximate) | U-KAN − MLP | 0.3, 0.7 | +0.002 [−0.001, +0.004] | +0.004 [−0.002, +0.011] | −0.002 [−0.007, +0.003] | **−0.032 [−0.038, −0.024]** |

*Exact* means the arms differ only in edge threshold. *Approximate* means one arm's rows were filtered to a higher
node threshold, as in §4.7, which penalises that arm's Curve F1, so those Curve F1 values are not fair estimates.

Stage 1 alone, against the U-Net `spline6`, each arm at its own thresholds:

| Dataset | Comparison | Thresholds (node, edge) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|---|
| SyntheticMRI | U-KAN − U-Net | 0.4, 0.9 vs 0.45, 0.85 | −0.001 [−0.003, +0.002] | −0.003 [−0.010, +0.005] | −0.009 [−0.022, +0.004] | **−0.009 [−0.012, −0.005]** |
| IXI | U-KAN − U-Net | 0.3, 0.85 vs 0.3, 0.8 | 0.000 [−0.005, +0.004] | **−0.013 [−0.020, −0.005]** | −0.010 [−0.019, +0.001] | **−0.044 [−0.052, −0.036]** |
| BRAVA | U-KAN − U-Net | 0.25, 0.8 vs 0.25, 0.75 | **+0.004 [+0.001, +0.007]** | −0.001 [−0.007, +0.005] | +0.002 [−0.003, +0.006] | **−0.035 [−0.045, −0.025]** |
| SyntheticMRI | MLP − U-Net | 0.4, 0.9 vs 0.45, 0.85 | 0.000 [−0.003, +0.003] | −0.003 [−0.008, +0.001] | **−0.013 [−0.022, −0.003]** | **−0.012 [−0.016, −0.008]** |
| IXI | MLP − U-Net | 0.35, 0.85 vs 0.3, 0.8 | −0.004 [−0.010, +0.003] | **−0.017 [−0.026, −0.009]** | **−0.011 [−0.019, −0.002]** | **−0.073 [−0.084, −0.063]** |
| BRAVA | MLP − U-Net | 0.3, 0.7 vs 0.25, 0.75 | −0.001 [−0.006, +0.005] | **−0.032 [−0.040, −0.023]** | **−0.006 [−0.010, −0.001]** | **−0.078 [−0.082, −0.072]** |

All of these stage-1 rows have threshold shifts, and no shared-threshold check was run for stage 1. Node F1 is
the least affected. Curve F1 falls when the edge threshold rises, so its size here is partly a threshold effect.

Absolute node / edge / Branch / Curve F1:

| Dataset | Stage | U-Net | U-KAN | MLP control |
|---|---|---|---|---|
| SyntheticMRI | 1 | 0.890 / 0.792 / 0.429 / 0.936 | 0.890 / 0.789 / 0.421 / 0.928 | 0.890 / 0.788 / 0.416 / 0.924 |
| SyntheticMRI | 2 | 0.894 / 0.797 / 0.441 / 0.945 | 0.890 / 0.788 / 0.434 / 0.946 | 0.891 / 0.793 / 0.435 / 0.947 |
| IXI | 1 | 0.648 / 0.315 / 0.163 / 0.677 | 0.647 / 0.303 / 0.153 / 0.633 | 0.644 / 0.299 / 0.152 / 0.604 |
| IXI | 2 | 0.649 / 0.313 / 0.179 / 0.770 | 0.645 / 0.304 / 0.176 / 0.756 | 0.651 / 0.317 / 0.180 / 0.729 |
| BRAVA | 1 | 0.543 / 0.217 / 0.054 / 0.365 | 0.547 / 0.216 / 0.055 / 0.329 | 0.542 / 0.185 / 0.048 / 0.287 |
| BRAVA | 2 | 0.546 / 0.206 / 0.062 / 0.395 | 0.552 / 0.240 / 0.067 / 0.477 | 0.548 / 0.206 / 0.058 / 0.407 |

Matched-node error in mm, median / 90th percentile (each arm's own matched nodes, no CI):

| Dataset | Stage 1: U-Net | U-KAN | MLP | Stage 2: U-Net | U-KAN | MLP |
|---|---|---|---|---|---|---|
| SyntheticMRI | 0.67 / 1.39 | 0.65 / 1.39 | 0.66 / 1.39 | 0.65 / 1.39 | 0.65 / 1.39 | 0.65 / 1.39 |
| IXI | 0.47 / 1.13 | 0.47 / 1.11 | 0.46 / 1.06 | 0.46 / 1.05 | 0.46 / 1.04 | 0.46 / 1.06 |
| BRAVA | 0.75 / 1.44 | 0.76 / 1.44 | 0.75 / 1.44 | 0.74 / 1.44 | 0.76 / 1.43 | 0.75 / 1.44 |

Fraction of long GT edges recovered (stage 2, own thresholds):

| Dataset | Arc length (GT edges) | U-Net | U-KAN | MLP | U-KAN − U-Net | U-KAN − MLP |
|---|---|---|---|---|---|---|
| SyntheticMRI | ≥ 20 mm (1556) | 0.809 | 0.804 | 0.812 | −0.005 [−0.019, +0.008] | −0.008 [−0.018, +0.003] |
| SyntheticMRI | ≥ 40 mm (145) | 0.710 | 0.690 | 0.710 | −0.021 [−0.095, +0.050] | −0.021 [−0.094, +0.040] |
| IXI | ≥ 20 mm (648) | 0.279 | 0.258 | 0.258 | −0.022 [−0.050, +0.009] | 0.000 [−0.026, +0.023] |
| IXI | ≥ 40 mm (83) | 0.145 | 0.133 | 0.157 | −0.012 [−0.092, +0.066] | −0.024 [−0.085, +0.031] |
| BRAVA | ≥ 20 mm (5042) | 0.092 | 0.121 | 0.102 | **+0.029 [+0.023, +0.035]** | **+0.019 [+0.014, +0.024]** |
| BRAVA | ≥ 40 mm (1251) | 0.052 | 0.066 | 0.067 | **+0.014 [+0.006, +0.026]** | −0.002 [−0.012, +0.008] |

The BRAVA U-KAN gains at ≥ 20 mm track the node threshold: missing endpoints are 4024 (U-Net), 3563 (U-KAN), 4030 (MLP).

Topology per patch, in the format of §4.3:

| Dataset: arm | β₀ error | β₁ error | Loops | SMD ×10³ | Patches without SMD | Nodes | Edges |
|---|---|---|---|---|---|---|---|
| SyntheticMRI: `traced_adaptive` | 1.02 | 0.22 | 0.20 (0.04) | 1.81 | 0 | 19.3 | 13.7 |
| SyntheticMRI: `traced_adaptive_ukan` | 1.03 | 0.29 | 0.28 (0.04) | 1.76 | 0 | 19.3 | 13.8 |
| SyntheticMRI: `traced_adaptive_ukan_mlp` | 1.17 | 0.30 | 0.30 (0.04) | 1.69 | 0 | 19.6 | 13.9 |
| IXI: `traced_adaptive` | 1.68 | 3.50 | 4.10 (0.71) | 23.24 | 42 | 17.8 | 16.4 |
| IXI: `traced_adaptive_ukan` | 1.94 | 2.05 | 2.62 (0.71) | 21.44 | 41 | 17.6 | 14.3 |
| IXI: `traced_adaptive_ukan_mlp` | 2.82 | 1.12 | 1.54 (0.71) | 22.22 | 41 | 17.6 | 12.0 |
| BRAVA: `traced_adaptive` | 4.08 | 0.62 | 0.62 (0.00) | 63.79 | 225 | 15.1 | 6.1 |
| BRAVA: `traced_adaptive_ukan` | 6.01 | 1.41 | 1.41 (0.00) | 50.39 | 170 | 20.5 | 9.9 |
| BRAVA: `traced_adaptive_ukan_mlp` | 3.42 | 0.98 | 0.98 (0.00) | 58.92 | 189 | 15.1 | 7.2 |
| SyntheticMRI: `spline6` | 1.01 | 0.32 | 0.29 (0.04) | 1.84 | 0 | 19.3 | 13.8 |
| SyntheticMRI: `spline6_ukan` | 1.37 | 0.29 | 0.28 (0.04) | 2.29 | 0 | 19.6 | 13.6 |
| SyntheticMRI: `spline6_ukan_mlp` | 1.39 | 0.32 | 0.30 (0.04) | 2.49 | 0 | 19.6 | 13.6 |
| IXI: `spline6` | 4.02 | 3.05 | 3.64 (0.71) | 27.02 | 39 | 21.0 | 16.0 |
| IXI: `spline6_ukan` | 5.51 | 1.55 | 1.97 (0.71) | 28.34 | 48 | 20.6 | 12.4 |
| IXI: `spline6_ukan_mlp` | 3.60 | 1.40 | 1.74 (0.71) | 30.07 | 55 | 17.3 | 10.9 |
| BRAVA: `spline6` | 7.15 | 1.50 | 1.50 (0.00) | 61.03 | 191 | 20.6 | 8.8 |
| BRAVA: `spline6_ukan` | 7.93 | 0.96 | 0.96 (0.00) | 63.96 | 255 | 20.4 | 7.3 |
| BRAVA: `spline6_ukan_mlp` | 4.20 | 0.89 | 0.89 (0.00) | 72.57 | 270 | 14.8 | 6.0 |

Reading the results:
- **The node threshold dominates BRAVA across all arms.** Full-model arms at node threshold 0.25 reach Edge F1
  0.237–0.240 (`traced_offset`, `traced_adaptive_ukan`). Those at 0.30 reach 0.206–0.207 (`traced_adaptive`,
  `traced_passthrough`, `traced_adaptive_ukan_mlp`, whatever their other differences). The selection falls either side
  of a grid step, and this gap is larger than any Edge F1 effect of a model change in §4.7–4.9. The lower threshold
  also brings more nodes per patch (about 20 vs 15) and worse β₀ error.
- **KAN vs MLP.** At shared thresholds U-KAN is below the control on IXI and no CI favours it on Edge F1. The +0.026
  Curve F1 on IXI disappears at the shared edge threshold (−0.004 [−0.014, +0.006]), so it was a threshold effect.
- **IXI loops.** Both replaced backbones have fewer loops than the U-Net, with fewer edges per patch. The MLP control
  has the fewest (1.54 vs 2.62), so this is not specific to KAN layers, and its higher edge threshold (0.85)
  contributes too.
- **Spurious nodes at stage 1.** At the same node threshold, U-KAN changes spurious detections little: IXI 3039 → 2905
  and BRAVA 18600 → 18175, with matched nodes 7728 → 7657 and 18402 → 18433.
- **Stability.** No non-finite values in any training log, and loss levels and maxima similar to the U-Net's, so lr was
  not lowered for the KAN arm and `ukan_add` was not run. Stage 1 took about 15–20% longer than the 30 Oct U-Net
  (3845 vs 3181 s on IXI), on possibly different nodes.

Caveats:
- Single seed, validation only. The CIs cover subject sampling, not training variation, and are not corrected for
  multiple comparisons.
- This tests U-KAN at this project's budget and optimiser, as decided for this comparison, not at the longer schedule
  of the U-KAN paper. It is not evidence against U-KAN at a literature-scale budget, which was not tried.
- Each arm has its own train-selected thresholds; the deltas that depend on a threshold shift are called out above.
- `spline6` and `traced_adaptive` were trained on 30 Oct with earlier code. The U-KAN arms ran on the 3 Nov code with
  `--offset-radius 0`. The U-Net was not rerun on the new code to confirm that it reproduces.
- The backbones warm-start stage 2 from their own stage 1, so a stage-1 difference carries into stage 2.

Scripts: `scripts/compare_shared_threshold_20261104.py`, `scripts/node_quality_20261103.py`, and the three in §4.6.
Outputs in `experiments/dense_graph_20261030/`: `shared-thresholds.json`, `node-quality.json`,
`paired-traced_adaptive_ukan-vs-*.json`, `paired-spline6_ukan*-vs-spline6-*.json`, the matching `paired-arc-*` files, and
`topology-validation.json`.

Summary figure of 4 Nov, which also includes U-KAN with additive skips and DSCNet (K = 3). That DSCNet arm has
2.8x the U-Net's backbone parameters; §4.11 replaces the comparison with size-matched arms:

![Backbone comparison of 4 Nov: U-KAN concat, U-KAN add, DSCNet K=3 minus U-Net](figures/backbones_20261104/backbone_comparison.png)

### 4.10 Loop patches in training and evaluation (DONE, 5 Nov 2026)

**Problem.** The adjacency target stores one edge per node pair and no self-loop, so patches whose contracted
graph had a parallel branch, a self-loop or an anchor-free ring were dropped from training *and* from
validation/test (IXI 10.2%, TopBrain CTA 8.8%, TopBrain MRA 3.3%, SyntheticMRI 2.9%, BRAVA 0%). A
GT-dependent filter on the evaluation set is not defensible for a paper.

**Fix (DONE).**
- *Training encoding:* such branches get degree-2 "breaking point" anchors (midpoint of a repeated pair; 1/3
  and 2/3 of a self-loop; three anchors on an anchor-free ring); no patch is excluded. Geometry, beta_0 and
  beta_1 are checked unchanged on every patch (`data/spline_targets.subdivide_loops`,
  `prepare_transfer_splines.py`, default since 5 Nov; `--no-subdivide-loops` reproduces older caches).
- *Evaluation:* inserted anchors are "don't care" nodes; edges are scored on the original branches as a
  multiset (`metrics/loop_aware.py`). Loop-free patches score exactly as before. 2D crops are now contracted
  again after cropping, with the same rule and explicit `inserted` masks (`native2d/targets.crop_graph`).
- Validation (and later test) sets are complete. Old results remain comparisons on loop-free patches.

**A/B (single seed, U-Net, traced_adaptive recipe).** Same recipe, trained without loop patches (A) or with
them (B); both scored on the same complete validation list. Final model, B − A, 95% subject-bootstrap CI:

| Dataset (val patches) | Thresholds | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|
| IXI (2,961) | own | +0.002 [−0.005, +0.009] | +0.006 [−0.001, +0.011] | +0.002 [−0.003, +0.006] | +0.030 [+0.024, +0.036] |
| TopBrain MRA (1,148) | same (exact) | +0.006 [−0.005, +0.019] | +0.008 [−0.006, +0.022] | +0.008 [−0.000, +0.014] | +0.007 [−0.014, +0.022] |
| ↳ loop patches (49) | exact | +0.004 | +0.038 [+0.015, +0.060] | +0.028 [−0.002, +0.079] | +0.007 |
| TopBrain CTA (803) | own | −0.011 [−0.020, −0.000] | +0.003 | +0.005 | +0.069 [+0.046, +0.089] |
| SyntheticMRI (480; 8 with loops) | own | +0.001 | +0.002 | −0.006 | −0.002 |

- On IXI, CTA and SyntheticMRI the B models selected a lower node threshold, so own-threshold gains are partly
  a threshold effect. The shared-threshold approximation (higher node threshold, rows filtered without
  re-tracing) penalises B and cannot settle it (IXI shared: Curve −0.027, Branch −0.011).
- The only exact comparison (TopBrain MRA) shows B recovering more edges on loop patches and no change elsewhere.
- **Decision (user, 5 Nov 2026): train with loop patches by default.** No harm under the real protocol, no
  data dropped, training matches the complete evaluation sets. Not claimed as a result: effects are within
  single-seed variation.
- PROPOSED: exact re-scoring of the B models at A's node thresholds (re-traced, 4 GPU evaluations) and 2-3
  seeds per arm on IXI and TopBrain CTA.
- Files: `experiments/loop_subdivision_20261105/` (`loop-ab.json`, `ab_plan_summary.json`), plan entries
  `<dataset>-loops` / `<dataset>-noloops`, `scripts/compare_loop_ab_20261105.py`.

### 4.11 Size-matched backbones: U-Net, U-Net wide, DSC3 slim, DSC3 (DONE, 9 Nov 2026)

**Question.** DSC3 beat the U-Net on IXI and BRAVA (§4.9 follow-up, 4 Nov) but has 2.8x the backbone parameters.
Is the gain from the snake convolutions or from capacity? 2x2 design on the current recipe (loop patches, offset
radius 1.75), one seed, IXI / BRAVA / TopBrain CTA:

| | ~5.8M backbone | ~15.7M backbone |
|---|---|---|
| U-Net | `unet` 5.75M (existing runs) | `unet_wide` 15.57M (widths 56-56-104-208-424-56) |
| DSC3 | `dsc3_slim` 5.78M (widths 20-20-40-76-152-20) | `dsc3` 15.89M |

**Protocol (review-safe).** Node AP @2 mm (threshold-free) and F1 / topology at the frozen global decoding setting of
the evaluation protocol (V0: node 0.30 / edge 0.75), the same for every arm; subject-paired bootstrap vs U-Net.
`scripts/backbone_compare_20261109.py`, results `experiments/backbone_compare_20261109/<dataset>.json`.

| Arm − U-Net (V0 global) | IXI edge F1 | BRAVA edge F1 | CTA edge F1 | Node AP IXI / BRAVA / CTA |
|---|---|---|---|---|
| U-Net (absolute) | 0.323 | 0.208 | 0.212 | 0.611 / 0.519 / 0.447 |
| U-Net wide | **+0.016** [+0.012, +0.021] | **+0.027** [+0.021, +0.032] | +0.010 [−0.002, +0.020] | 0.624 / 0.526 / 0.461 |
| DSC3 slim | **+0.012** [+0.008, +0.017] | **+0.011** [+0.005, +0.019] | **+0.024** [+0.012, +0.040] | 0.625 / 0.521 / 0.471 |
| DSC3 | **+0.021** [+0.017, +0.027] | −0.003 [−0.009, +0.006] | +0.015 [−0.004, +0.034] | 0.638 / 0.520 / 0.478 |

All four F1 measures and the |β₁| error, arm − U-Net with 95% subject-paired CIs (`scripts/backbone_figure_20261109.py`).
For |β₁| error, right of zero is worse:

![Size-matched backbones minus U-Net at the frozen global decoding setting, IXI / BRAVA / TopBrain CTA](figures/backbones_20261109/backbone_size_matched.png)


- **Snake convolutions at equal small size help:** DSC3 slim beats the U-Net on all three datasets (edge, node and
  Curve F1; Branch F1 on IXI and CTA).
- **Capacity alone helps too:** U-Net wide beats the U-Net on IXI and BRAVA (Curve F1 +0.046 on BRAVA).
- **At equal large size, DSC3 is not consistently better than U-Net wide:** better on IXI (+0.006 edge F1), worse on
  BRAVA (−0.030), similar on CTA. Most of DSC3's earlier gain over the U-Net was capacity.
- **Topology (|β1| error) moves both ways:** the DSC arms add loops on IXI (+0.52 / +0.55 per patch), U-Net wide
  removes them (−0.22); on BRAVA all three larger/snake arms add loops (+0.19 to +0.42).
- **Cost:** GPU-h per dataset (both stages): U-Net ~2.7, U-Net wide 2.4–3.1, DSC3 slim 4.0–4.4, DSC3 4.3–4.9.
- **Single seed;** gaps between the three challengers (0.005–0.03) are of the size of seed variation seen in §4.10.
- PROPOSED: U-Net wide as the default backbone (best or near-best on IXI and BRAVA, no extra training time, a
  standard architecture), with DSC3 slim as the alternative; confirm with 2 more seeds of U-Net, U-Net wide and DSC3
  slim on the three datasets (~45 GPU-h) before the journal runs.

### 4.12 Curve shape: coiled short edges, the arc-length penalty and the number of controls (DONE, 9 Nov 2026)

**Problem.** On short, nearly straight vessels the predicted curves coiled inside the vessel. Accepted true edges with
chord < 5 mm, reference models, medians (GT in brackets):

| Dataset | arc / chord | total turning at 1 mm steps | share turning > 45° more than the GT |
|---|---|---|---|
| IXI-loops (`traced_adaptive`) | 2.42 (1.00) | 227° (9°) | 0.79 |
| BRAVA (`traced_offset`) | 3.64 (1.04) | 1179° (31°) | 0.89 |
| TopBrain CTA (`traced_adaptive`) | 2.09 (1.00) | 191° (10°) | 0.77 |

Distance metrics did not show it: the coil stays within a fraction of a millimetre of the vessel (mean distance to the
GT about 0.35 mm IXI, 0.6 mm BRAVA, as for straight curves), so Curve F1 and the chamfer loss are blind to it.

**Causes.**
- Each interior control may move about 4 voxels from the traced path whatever the edge length; on a 3 mm edge the
  controls are 0.1–0.5 mm apart, so small offsets cross neighbours. BRAVA, chord < 5 mm: arc/chord 1.38 with K8,
  5.11 with K16, 17.7 with K24.
- The K rule gave K16/K24 to most short edges: the arc/chord ratio is read on the voxel-staircase trace, where noise of
  fixed size inflates it on short paths (IXI chords < 5 mm: traced ratio 1.22 vs GT 1.00). On the training monitors
  (`scripts/k_selection_check_20261109.py`), the fewest controls that fit the smoothed GT within 0.5 voxel ("needed K")
  is K8 for 99–100% of traced true pairs (BRAVA 86%), and the original rule agrees with it on only 31–49%.

**Arms** (stage 2 only: 7.5k updates, lr 3e-4, warm start from each dataset's stage-1 `spline6`, the reference recipe;
one change each; IXI-loops / BRAVA / TopBrain CTA; scored at the frozen global decoding, node 0.30 / edge 0.75,
subject-paired bootstrap vs the reference; `scripts/compare_arms_20261109.py`, `experiments/arm_compare_20261109/`):
- `traced_arcpen_r1109`: arc-length penalty λ = 0.1 (§2.5), original K rule;
- `traced_kfit_r1109`: fit K rule, tolerance 0.75 voxel (chosen on the training monitors);
- `traced_k24_r1109`: one K24 head for every pair + arc penalty;
- `@compressed`: the same model with output compression at 0.5 voxel (§2.4 d), evaluation only.

| Dataset: arm | Edge F1 Δ | Curve F1 Δ | short arc/chord | > 45° extra: < 5 / 10–20 mm | controls per output edge |
|---|---|---|---|---|---|
| IXI: reference | 0.323 | 0.812 | 2.42 | 0.79 / 0.32 | 13.1 |
| IXI: arc penalty | −0.001 [−0.003, +0.001] | −0.003 [−0.005, −0.002] | 1.09 | 0.16 / 0.12 | 13.2 |
| IXI: arc penalty, compressed | −0.001 [−0.003, +0.001] | −0.004 [−0.005, −0.002] | 1.07 | 0.15 / 0.03 | **8.1** |
| IXI: fit rule | −0.002 [−0.004, −0.000] | −0.006 [−0.008, −0.004] | 1.24 | 0.43 / 0.06 | — |
| IXI: K24 + penalty, compressed | −0.002 [−0.004, +0.000] | −0.005 [−0.006, −0.003] | 1.08 | 0.16 / 0.04 | **8.1** |
| BRAVA: reference | 0.208 | 0.391 | 3.64 | 0.89 / 0.33 | 12.9 |
| BRAVA: arc penalty | +0.002 [−0.002, +0.007] | +0.000 [−0.004, +0.004] | 1.11 | 0.18 / 0.05 | 13.0 |
| BRAVA: arc penalty, compressed | +0.002 [−0.002, +0.007] | −0.000 [−0.004, +0.004] | 1.09 | 0.15 / 0.02 | **8.4** |
| BRAVA: fit rule | −0.000 [−0.003, +0.004] | −0.007 [−0.012, −0.003] | 1.62 | 0.76 / 0.09 | — |
| BRAVA: K24 + penalty, compressed | −0.001 [−0.004, +0.002] | −0.004 [−0.009, +0.001] | 1.10 | 0.16 / 0.03 | **8.6** |
| CTA: reference | 0.212 | 0.582 | 2.09 | 0.77 / 0.47 | 13.5 |
| CTA: arc penalty | +0.004 [+0.000, +0.010] | −0.000 [−0.004, +0.007] | 1.11 | 0.20 / 0.13 | 13.4 |
| CTA: arc penalty, compressed | +0.004 [+0.000, +0.010] | −0.000 [−0.004, +0.007] | 1.08 | 0.20 / 0.05 | **8.1** |
| CTA: fit rule | +0.004 [−0.001, +0.009] | −0.010 [−0.020, −0.001] | 1.26 | 0.52 / 0.05 | — |
| CTA: K24 + penalty, compressed | +0.005 [+0.001, +0.009] | −0.000 [−0.010, +0.012] | 1.09 | 0.22 / 0.05 | **8.1** |

Reference rows give absolute values; Δ rows are arm − reference with 95% CIs. "Controls per output edge" is the mean K
over accepted edges at the frozen decoding. Node F1, Branch F1 and node AP change by ≤ 0.005 in every arm.

Real validation examples (a random sample, seed 0, not hand-picked; `scripts/curve_shape_figures_20261109.py`):

![Random short true edges: reference curves coil, arc-penalty curves follow the GT](figures/curve_shape_20261109/curve_examples.png)

![Whole validation patches: reference vs arc penalty with output compression](figures/curve_shape_20261109/patch_overlay.png)

- **The arc penalty removes the coils** on all three datasets (short-edge arc/chord 2.1–3.6 → 1.09–1.11) with no loss
  beyond the protocol's −0.01 F1 margin, and it improves the worst long edges (90th-percentile distance for GT arcs
  ≥ 40 mm: IXI 0.77 → 0.47 mm, BRAVA 1.23 → 0.93 mm).
- **Compression nearly halves the representation for free:** 13 → 8.1–8.6 controls per output edge, at most 0.001
  Curve F1, and it removes the residual turning on mid-length edges (10–20 mm: 0.12 → 0.03 on IXI).
- **Option (a) K24 + penalty + compression and option (b) penalty + current K rule + compression are tied** within
  single-seed noise on every measure; (b) is marginally better on Curve F1 (IXI −0.004 vs −0.005, BRAVA −0.000 vs
  −0.004) and on IXI long edges (p90 0.48 vs 0.55 mm), (a) is simpler (one head, no rule). Decision pending.
- **The fit rule is not adopted:** it removes fewer coils (BRAVA 1.62) and is the only arm with a Curve F1 loss on all
  three datasets.
- **Decision (user, 9 Nov 2026): the arc penalty λ = 0.1 is the stage-2 training default.**

Caveats: single seed; validation only; the arc-penalty arm used the original K rule, while the training default since
9 Nov is the minimum-length rule (`--k-min-arc 8 20`), a combination not yet trained; the 3D viewers still draw the cache
K8 fit as "GT edge" (on TubeTK edges of 40–80 mm only 69% of that fit lies inside the vessel, against ≥ 99% for the real
path and for the training target).

### 4.13 Width of the shared feature channels F (DONE, 9 Nov 2026)

**Question.** Are the 32 feature channels read by stage 2 too many or too few?

**Structure.** F = `final_conv(u1)` is a 1×1×1 *linear* map of the last decoder output u1 (32 channels for the U-Net),
and the corridor CNN's first layer is linear too. So F carries at most 32 dimensions, more F channels alone add no
information (the decoder width would have to grow), and a native channel is an arbitrary basis direction. Only stage 2
reads F; nodes and tracing are unaffected by any change to it.

**Test-time ablation** (`scripts/feature_ablation_20261109.py`, `experiments/feature_ablation_20261109/`): on the trained
reference models, F is replaced by its projection onto the top-k principal directions of the corridor samples (fitted
on the training monitor; removed directions set to their mean). Edge F1 change at the frozen global decoding:

| Dataset (full Edge F1) | k = 1 | 2 | 4 | 8 | 12 | 16 | 24 | all removed |
|---|---|---|---|---|---|---|---|---|
| SyntheticMRI (0.787) | −0.111 | −0.044 | −0.029 | −0.004 | −0.004 | −0.001 | 0.000 | −0.787 |
| IXI (0.323) | −0.066 | −0.032 | −0.022 | −0.004 | −0.003 | 0.000 | 0.000 | −0.244 |
| BRAVA (0.208) | −0.052 | −0.078 | −0.009 | −0.002 | 0.000 | 0.000 | 0.000 | −0.169 |
| TopBrain MRA (0.268) | −0.078 | −0.041 | −0.018 | −0.005 | 0.000 | 0.000 | 0.000 | −0.188 |
| TopBrain CTA (0.212) | −0.018 | −0.016 | −0.008 | −0.009 | −0.002 | +0.002 | 0.000 | −0.156 |

![Edge F1 change vs number of principal feature directions kept](figures/curve_shape_20261109/feature_ablation.png)

- 12–16 directions keep everything (also Curve F1 within 0.002 at k = 16); the top 8 hold 80–83% of the variance,
  participation ratio 5.7–8.3.
- Single native channels matter little (worst: −0.026 Edge F1, channel 30 on CTA), but removing the 16 or 24 least
  important native channels together costs up to −0.061 / −0.133: the information is spread across channels.
- Removing all of F collapses stage 2 (−0.16 to −0.79 Edge F1): edges depend on the image features, not on geometry
  alone.

**Training with 16 channels** (`spline6_fc16_r1109` → `traced_fc16_r1109`, both stages from scratch, decoder unchanged,
original K rule, arm_compare protocol), 16 − 32:

| Dataset | Edge F1 | Branch F1 | Curve F1 | node AP |
|---|---|---|---|---|
| IXI | +0.003 [−0.001, +0.008] | +0.003 [−0.002, +0.008] | −0.000 [−0.004, +0.003] | 0.612 vs 0.611 |
| BRAVA | +0.001 [−0.002, +0.005] | −0.004 [−0.008, +0.000] | +0.009 [−0.000, +0.020] | 0.517 vs 0.519 |
| TopBrain CTA | −0.007 [−0.023, +0.007] | −0.010 [−0.019, +0.002] | **−0.024 [−0.036, −0.010]** | 0.445 vs 0.447 |

- 16 channels match 32 on IXI and BRAVA but lose Curve F1 on TopBrain CTA, which the test-time ablation did not
  predict. Stage 1 is only ~10% faster with 16.
- **Decision (user, 9 Nov 2026): keep 32 feature channels.** 32 is not a bottleneck either; testing "more" would mean
  widening the decoder (`--feature-channels` exists for both stages).
- Single seed; validation only.

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
  6. **BRAVA at a fixed node threshold (PROPOSED).** Re-evaluate `traced_adaptive`, `traced_offset`, `traced_passthrough`
     and the U-KAN arms at one node threshold, repeating tracing on GPU. §4.7–4.9 could only filter saved rows, so
     their BRAVA shared-threshold comparisons are approximate.
  7. **Pass-through rule (PROPOSED).** Require A–C and C–B to be accepted before contracting, and separate two-hop
     training from the head (§4.8).
  8. **Number of controls (PROPOSED, decision pending).** Choose between (a) one K24 head + compression and (b) the
     current K rule + compression (§4.12, tied on one seed), align the K-rule training default with it, make output
     compression the evaluation default, and confirm with 3 seeds together with the backbone choice (§4.11).
  9. **3D viewers (PROPOSED).** Label the blue "GT edge (B-spline)" as the cache's K8 display fit, or draw the dense
     GT path, since the K8 fit leaves long tortuous vessels.
