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

## 2. The network

```
64³ MRA patch
  │
  ▼  MONAI BasicUNet (3D, full resolution, 37 output channels)
  ├─ ch0      node heatmap ──► 3×3×3 peak NMS + threshold ──► nodes (+ ch1–3 sub-voxel offsets)
  ├─ ch4      centreline probability
  └─ ch5–36   32 feature channels F
        │
  nodes + centreline ──► geodesic Voronoi tracing ──► candidate pairs (∪ 16-NN fallback) + paths
        │
  per pair: features of F sampled in a 24×11×11 tube that follows the path ──► small 3D CNN
        ├─► edge head ──► P(edge)
        └─► geometry head (K8 / K16 / K24 by path length and tortuosity) ──► curved edge
```

**Stage 1: nodes.** A standard 3D U-Net (MONAI `BasicUNet`, used as-is) predicts:
- a CenterNet-style node heatmap: a Gaussian blob per node, penalty-reduced focal loss;
- sub-voxel offsets;
- a vessel-centreline map (an auxiliary loss);
- 32 feature channels.

Nodes are the local maxima of the heatmap.

**Stage 2: edges.**
1. **Tracing.** From all nodes at once, geodesic distance is propagated through the predicted
   centreline map (on GPU; cheap where the vessel is confident). Every vessel voxel is owned by
   its nearest node along the vessel.
   - Two nodes are a candidate pair when their regions touch, at any distance.
   - A node lying in between blocks "skip" pairs.
   - The connecting path is read back from the distance field.
   - Pairs from the 16 nearest neighbours are added as a fallback, with straight paths.
2. **Evidence.** For each pair, the U-Net features are sampled in a tube around the traced path
   and encoded by a small 3D CNN. The endpoint features and heat are added.
3. **Edge head.** It outputs the probability that the two nodes are directly connected.
4. **Geometry head.** It outputs residual control points around the traced path, decoded as a
   clamped cubic B-spline that always ends exactly on the two nodes. The number of controls is
   set by a fixed rule on the traced path: K8 by default, K16 when the arc is ≥ 15 mm or
   arc/chord ≥ 1.15, K24 when the arc is ≥ 35 mm or arc/chord ≥ 1.5.

**Training labels (no Hungarian matching).**
- Candidate pairs are built from the model's *own* predicted peaks. Each peak is matched
  one-to-one to a GT node within 2 mm, by the same rule the evaluation uses.
- A pair is positive iff its matched GT nodes share an edge, so training sees the same
  candidates as inference.
- Pairs built from slightly jittered GT nodes are also used, to bootstrap the edge head.

**Losses.** All weights are 1:
- heatmap focal;
- offset L1;
- centreline BCE;
- edge BCE;
- geometry chamfer (mm) to the dense GT centreline.

Code:
- [`scripts/dense_graph_20261030.py`](../../scripts/dense_graph_20261030.py) (16-NN model);
- [`scripts/dense_graph_traced_20261030.py`](../../scripts/dense_graph_traced_20261030.py)
  (traced + adaptive K).

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

Report: `GNBM/gnbm-native-2d-20261027/experiments/dense_graph_2d_20261030/report.md`.

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
