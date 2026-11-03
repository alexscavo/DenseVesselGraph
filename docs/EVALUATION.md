# Evaluation metrics and how to run them

This page has two parts:

1. **The physical metrics reported in the current results**
   ([`research/dense_traced_model_20261102.md`](research/dense_traced_model_20261102.md) and
   [`research/network_proposals_20261030.md`](research/network_proposals_20261030.md)). Every
   network, including converted RelationFormer predictions, is scored by the same `evaluate`
   function of the dense-model scripts.
2. **The native `evaluate.py` metrics** of the RelationFormer pipeline (box mAP/mAR, box F1,
   SMD), how to run that evaluator, and why its numbers are not the ones reported now.

How the metric suite was designed and validated is summarised in
[`research/HISTORY.md`](research/HISTORY.md).

Shorthand used below:

- **GT**: ground truth.
- **Physical**: computed in millimetres, after mapping normalized patch coordinates through the
  per-patch `metric` matrix stored in the target cache.
- **Anchor**: a node whose degree is not 2 (endpoint, junction or isolated node).
- **Degree-2 contraction**: merging every maximal chain of degree-2 nodes into one branch between
  anchors, keeping the chain's points as the branch polyline.
- **common-v2**: the `shared-guides-v2` protocol of
  [`metrics/common_vessel_metrics.py`](../metrics/common_vessel_metrics.py), which gives
  Branch F1, Curve F1, anchor F1 and guide distance against the original dense centrelines
  ("guides", stored as `full_paths` in the target caches).
- **`monitor_train`**: the fixed 256 training patches on which decision thresholds are chosen.

## 1. Physical metrics in use

### 1.1 Where they are computed

| Metric | Code |
|---|---|
| Node F1, Edge F1, MM/MU/UU, missing-endpoint fraction, β₀/β₁ error, Branch F1, Curve F1 | `evaluate()` in [`scripts/dense_graph_20261030.py`](../scripts/dense_graph_20261030.py), which calls `match_nodes` ([`scripts/diagnose_branch_relations.py`](../scripts/diagnose_branch_relations.py)), `edge_accounting` ([`scripts/ixi_query_diagnostics.py`](../scripts/ixi_query_diagnostics.py)) and `evaluate_common_vessels` / `aggregate_common_vessels` ([`metrics/common_vessel_metrics.py`](../metrics/common_vessel_metrics.py)) |
| Per-patch topology table (β₀, β₁, loops, SMD) | [`scripts/topology_metrics_20261102.py`](../scripts/topology_metrics_20261102.py) (`betti` from `ixi_query_diagnostics.py`, `street_mover_distance` from [`training/evaluation/metrics.py`](../training/evaluation/metrics.py)) |
| Node AP @2 mm, edge ranking AP, box node/edge mAP | [`scripts/dense_ap_20261030.py`](../scripts/dense_ap_20261030.py) |
| Paired differences with subject-level CIs | [`scripts/compare_traced_knn_20261030.py`](../scripts/compare_traced_knn_20261030.py) |
| Long-edge recovery by arc length | [`scripts/compare_arc_length_20261102.py`](../scripts/compare_arc_length_20261102.py) |
| Oracle-topology edge geometry (mean symmetric distance) | [`scripts/oracle_geometry_20261030.py`](../scripts/oracle_geometry_20261030.py) |

Outputs of the current comparison are in `experiments/dense_graph_20261030/`.

### 1.2 Node and edge detection (2 mm)

- **Node F1 @2 mm.** Predicted and GT nodes are matched one-to-one within 2 mm by `match_nodes`:
  a Hungarian assignment in which pairs farther than 2 mm get a prohibitive cost, so it first
  maximizes the number of matches and then minimizes distance. TP/FP/FN are summed over patches.
- **Edge F1.** Computed by `edge_accounting`. A retained predicted edge is TP if and only if both
  its endpoints are matched at 2 mm and the two matched GT nodes share a GT edge.
  FN = GT edges − TP. The GT graph must be simple: closed or parallel GT edges raise an error.
- **Missing endpoint.** The fraction of missed GT edges for which at least one endpoint has no
  matched prediction. It separates detection failures from relation failures.
- **MM / MU / UU.** False edges between two matched nodes / one matched and one unmatched node /
  two unmatched nodes. MM is a relation error; MU and UU come from node detection errors.
- **Thresholds.** The node threshold is chosen to maximize node F1, then the edge threshold to
  maximize edge F1, both on `monitor_train`. Validation never selects anything.

### 1.3 Branch F1 and Curve F1 (common-v2)

- **Preparation.** Prediction and GT are degree-2 contracted with their geometry kept: chords for
  straight edges, 257 samples per spline edge. The reference geometry is the original dense
  centreline paths, not the fitted spline targets.
- **Anchor matching.** One-to-one within **1 mm**.
- **Branch F1.** A predicted branch is TP when both its anchors are matched and the GT has an
  unused branch between the two matched anchors (multiplicity counts). Branch shape is ignored
  and never chooses the correspondence. Isolated predicted nodes are not branches. Dataset
  F1 = 2TP / (2TP + FP + FN), pooled over patches; two empty graphs score 1.
- **Curve F1.** All predicted and GT branches are sampled every **0.25 mm** of arc length. A
  sample is supported if it lies within **1 mm** of the other graph's polylines. Precision and
  recall are pooled sample fractions, so missing and extra geometry both count. No branch match
  is needed.
- **Anchor F1.** Anchors, isolated nodes included, matched at 1 mm.
- **Guide distance and tortuosity error.** Computed only on topology-matched branches (symmetric
  mean point-to-polyline distance), so they must be read together with branch coverage,
  TP / (TP + FN).

### 1.4 Topology: β₀, β₁, loops

- β₀ = number of connected components; β₁ = number of independent cycles = E − V + β₀.
- Computed on the decoded graph with all retained nodes, isolated nodes included. Detected nodes
  without an accepted edge therefore raise the β₀ error.
- Reported as the per-patch mean absolute error against GT. "Loops" is the mean predicted β₁ per
  patch, with the GT value in brackets.
- Equal Betti numbers do not imply correct adjacency.

### 1.5 SMD (street mover distance)

- The native RelationFormer protocol (`street_mover_distance`): 100 points per graph, Sinkhorn
  ε = 1e-7 with 100 iterations, **normalized patch coordinates** (not mm) and **straight
  segments between nodes**, so predicted curves are ignored.
- Undefined, and skipped, when either graph has no edges. The number of skipped patches differs
  between arms (on BRAVA 78–236), so BRAVA SMD must be read with care.
- Kept only for comparability with GrEVe/RelationFormer (GrEVe is the earlier
  RelationFormer-based vessel-graph work whose evaluator defines the native metrics).

### 1.6 Average precision

- **Node AP @2 mm.** All proposals ranked by score and matched greedily to GT within 2 mm.
  Proposals are every heatmap peak ≥ 0.01 for the dense model, and all 256 queries for
  RelationFormer.
- **Edge ranking AP.** AP over candidate pairs among the nodes kept at the selected threshold,
  reported for all pairs and for pairs with both endpoints matched. It ranks the proposed
  candidates only; GT edges that were never proposed are not counted.
- **Box node/edge mAP.** The GrEVe/RelationFormer metric (§2.2) with the `configs/base.yaml`
  protocol: IoU 0.50:0.05:0.95, GT node boxes of side 0.2, edge boxes spanning the two end nodes
  with half-width 0.1, at most 40 detections. Models without a box head get fixed side-0.2 boxes
  centred on their predicted nodes.

### 1.7 Other geometry scores

- **Long-edge recovery.** Fraction of GT edges recovered under the Edge F1 rule, binned by GT arc
  length (0–10, 20–40, ≥ 40 mm).
- **Mean symmetric distance (oracle geometry).** With the GT endpoints and edges given, the
  predicted curve and the dense path are each sampled at 256 points; the score is the mean of
  the two directed mean nearest-point distances, in mm (Hausdorff and coverage at 0.5/1 mm are
  also computed). It measures edge shape only: compare it with the common-v2 guide distance, not
  with Curve F1.

### 1.8 Aggregation and uncertainty

- Dataset F1 values pool TP/FP/FN (or curve samples) over patches; per-patch F1 is never
  averaged.
- Differences are reported with subject-paired bootstrap 95% CIs over validation subjects. These
  do not cover training-seed variation. All current results are single-seed and validation-only.

## 2. Native `evaluate.py` (RelationFormer pipeline)

`evaluate.py` evaluates a RelationFormer checkpoint built from a YAML config: it runs inference,
turns retained object queries into an undirected graph, computes the native metrics, optionally
recalibrates BatchNorm statistics, exports predictions as JSON and writes PNG comparisons. It
does **not** compute the physical metrics of §1, and the dense model is not evaluated through it.

### 2.1 Inference

- Object queries are kept by foreground-class argmax unless `evaluation.node_threshold` is set.
- Every unordered pair of kept nodes is a candidate edge. Relation logits are evaluated in both
  endpoint orders and averaged before softmax.
- Edges are kept by class argmax unless `evaluation.edge_threshold` is set.
- No non-maximum suppression.

Each exported prediction contains normalized D/H/W nodes, all six predicted centre-size box
values, node scores, local undirected edges and edge scores.

### 2.2 Native metrics

- Node and edge mAP/mAR over IoU 0.50:0.05:0.95, with at most **40 detections** per patch.
  Detections are pooled by confidence within each of 5 contiguous folds before AP/AR is
  computed; the reported value is the mean over folds.
- Node and edge micro precision/recall/F1 at the configured confidence thresholds and
  `protocol.f1_iou_threshold` (default IoU 0.5), with no detection cap. Matching is greedy,
  score-ordered and one-to-one on boxes; duplicates count as false positives, including on
  patches with no GT; zero denominators give 0.
- SMD (§1.5), absolute β₀/β₁ errors, and node/edge counts with their absolute errors.
- Optional: a physical Branch F1 when `evaluation.protocol.branch_threshold_mm` is set, and
  spline-branch geometry metrics when the model has a spline head.

Undefined values (for example edge AP on a patch without GT edges, or SMD when a graph has no
non-zero-length edge) are written as JSON `null` and left out of means. The protocol constants
(node size 0.2, edge half-width 0.1, IoU thresholds, SMD settings, 5 folds, 40-detection cap)
are under `evaluation.protocol` in `configs/base.yaml`.

**Why these are not the reported metrics.**

- **Box edge F1 overstates positional accuracy.** A node box has side 0.2 of the patch (12.8
  voxels in a 64³ patch) and an edge box spans its two end nodes with half-width 0.1, so an edge
  can count as correct at IoU 0.5 although its endpoints are several millimetres from the true
  nodes, or although it connects the wrong pair of nearby nodes. It measures spatial edge-box
  overlap, not endpoint-correct adjacency. The physical Edge F1 of §1.2 requires both endpoints
  within 2 mm of the two GT nodes of a real GT edge.
- **The 40-detection cap** truncates AP/AR on patches with more than 40 GT nodes, which occur in
  the real datasets (the IXI sets need up to 256 queries); F1 has no cap.
- **SMD** uses normalized coordinates and straight segments, and skips empty graphs.

The evaluator also fixes four logic errors of the old report script (node centre-size boxes are
converted to corners exactly once; every sample is included when the dataset size is not
divisible by five; SMD samples the endpoints named by each edge; plots use the declared D/H/W
order). Its numbers must therefore not be presented as identical to CSV files from the old
evaluator.

### 2.3 Running it

On Jean Zay, run inside a GPU job (see [`RUNNING_THE_MODEL.md`](RUNNING_THE_MODEL.md) §7 for a
complete Slurm example and every flag). A small validation run with four plots:

```bash
python evaluate.py \
  --config configs/finetune_synthetic_mri.yaml \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --output-dir "$GNBM_OUTPUT_DIR/evaluation_mri" \
  --dataset synthetic_mri \
  --split val \
  --max-samples 32 \
  --batch-size 4 \
  --visualizations 4
```

To export predictions for chosen source patches only, repeat `--sample-id` or pass
`--sample-list` (one `source_sample_id` per line). `--sample-id`, `--sample-list` and
`--max-samples` are mutually exclusive. Explicit selection keeps the manifest order and fails if
an ID is missing from the split, so several models can be evaluated on exactly the same patches.

The dataset root must contain the requested `val/` or `test/` split. Evaluation never applies
random augmentation, and it sorts source sample IDs before applying `--max-samples`.

Outputs:

- `resolved-config.yaml`: exact evaluation configuration;
- `metadata.json`: checkpoint path and size, split, sample count, calibration;
- `summary.json`: aggregate means and fold standard deviations;
- `predictions.json`: graph predictions, source IDs and per-sample metrics (skip with
  `--no-export-predictions`);
- `per-patch-metrics.csv`: one row per source patch;
- `plots/sample_*.png`: optional comparisons.

For interactive 3D inspection of `predictions.json`, see
[`GRAPH_PREDICTION_3D.md`](GRAPH_PREDICTION_3D.md).

**BatchNorm calibration** is off by default. Set `--bn-calibration-batches N` (or
`evaluation.bn_calibration_batches`) only for a checkpoint whose protocol requires it. It
updates BatchNorm running statistics on the configured dataset's validation split, without
changing learned parameters, and never on the test split.

**Parity with the original repository** (model-to-graph only, independent of the corrected
metrics):

```bash
python scripts/compare_model_forward.py \
  --legacy-root /absolute/path/to/original/3d \
  --checkpoint "$GNBM_MRI_CHECKPOINT" \
  --compare-inference
```

### 2.4 Tests

```bash
python -m unittest -v tests.test_inference tests.test_evaluation_metrics tests.test_evaluator
```

With a real dataset and checkpoint, inside a GPU job:

```bash
export SYNTHETIC_MRI_DATASET=/path/to/syntheticMRI/new_patches_boundary
export GNBM_MRI_CHECKPOINT=/absolute/path/to/checkpoint.pt
python -m unittest -v tests.test_evaluation_cluster
```

This exercises real loading, CUDA inference, graph metrics, JSON export and PNG rendering.
