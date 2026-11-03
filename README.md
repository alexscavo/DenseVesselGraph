# DenseVesselGraph

This repository extracts vessel graphs from images. Nodes are vessel junctions, vessel ends and
patch-border crossings. Each edge carries a curve.
- **3D:** MRA and CT from SyntheticMRI, IXI, BRAVA and TopBrain.
- **2D:** AV-WIDE, REAVER and OCTA-Synth.

The chosen model is a **two-stage dense model with traced connectivity**:
1. A full-resolution 3D U-Net detects nodes as heatmap peaks.
2. Edges are proposed by tracing the predicted centreline between nodes. Each edge is scored
   and shaped from image features sampled along the traced path, as a B-spline with 8, 16 or
   24 controls.

It trains from scratch and beats every RelationFormer baseline on all three 3D datasets.
Validation node / edge / Branch / Curve F1:

| Model | SyntheticMRI | IXI | BRAVA |
|---|---|---|---|
| RelationFormer B0 | 0.794 / 0.607 / 0.197 / 0.615 | 0.508 / 0.156 / 0.024 / 0.344 | 0.368 / 0.073 / 0.005 / 0.161 |
| Dense + traced | 0.894 / 0.797 / 0.441 / 0.945 | 0.649 / 0.313 / 0.179 / 0.770 | 0.546 / 0.206 / 0.062 / 0.395 |

These are validation results from a single seed. Thresholds were chosen on training patches only;
the test splits are unused.

## Where to read

| Document | What it is |
|---|---|
| [`docs/research/dense_traced_model_20261102.md`](docs/research/dense_traced_model_20261102.md) | **Start here.** Current model, main results, limitations. |
| [`docs/research/network_proposals_20261030.md`](docs/research/network_proposals_20261030.md) | Full design, glossary of project shorthand, and the 30 Oct – 1 Nov comparisons (RelationFormer B0, anchored RelationFormer, dense models, AP, traced vs 16-NN). |
| [`docs/research/HISTORY.md`](docs/research/HISTORY.md) | Condensed history of the earlier RelationFormer work (Sept – 29 Oct 2026): what was tried, the key numbers, and why it was superseded. |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Metric definitions (physical node/edge F1, Branch/Curve F1, β errors, AP, SMD) and which script computes each. |
| [`docs/research/BRAVA_preparation.md`](docs/research/BRAVA_preparation.md), [`docs/IXI_DATASET.md`](docs/IXI_DATASET.md), [`docs/SYNTHETIC_MRI_DATASET_REPORT.md`](docs/SYNTHETIC_MRI_DATASET_REPORT.md), [`docs/DATA_GENERATION.md`](docs/DATA_GENERATION.md) | Dataset preparation and provenance. |
| [`docs/MODEL.md`](docs/MODEL.md), [`docs/TRAINING.md`](docs/TRAINING.md), [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md), [`docs/RUNNING_THE_MODEL.md`](docs/RUNNING_THE_MODEL.md), [`docs/GRAPH_PREDICTION_3D.md`](docs/GRAPH_PREDICTION_3D.md) | The RelationFormer baseline pipeline (`train.py`, `evaluate.py`): model, losses, data loading and augmentation, configuration, and the 3D viewer. |
| [`configs/README.md`](configs/README.md), [`scripts/README.md`](scripts/README.md), [`cluster/jean_zay/README.md`](cluster/jean_zay/README.md), [`experiments/README.md`](experiments/README.md) | Configurations, data-extraction scripts, Jean Zay setup and launchers, and per-study experiment folders. |
| [`docs/archive/README.md`](docs/archive/README.md) | What was archived on 3 Nov 2026 (old research logs and evidence reports) and how to restore it. |

## Code map

- **Dense model** (current), in `scripts/`:
  - `dense_graph_20261030.py`: stage 1, 16-NN stage 2, training and evaluation;
  - `dense_graph_traced_20261030.py`: tracing, path corridor, adaptive K;
  - `dense_graph_2d_20261030.py`: 2D port.
- **Dense model launchers:** `cluster/jean_zay/dense_graph*.slurm`.
- **RelationFormer baseline:**
  - `train.py`, `evaluate.py`;
  - `models/` (SE-ResNet, deformable transformer, spline and refinement heads; CUDA ops source
    in `models/ops/src/`);
  - `training/` (losses, evaluation);
  - `data/` (loaders, preprocessing);
  - `configs/`.
- **Native 2D RelationFormer:** `native2d/`.
- **Metrics:** `metrics/` (common-v2 Branch/Curve F1, vessel suite), `boxes/`.
- **Tests:** `tests/`.
- **Data extraction and analysis scripts:** `scripts/`.

The deformable-attention CUDA extension is compiled on Jean Zay. Build output is never committed.
Setup is described in [`cluster/jean_zay/README.md`](cluster/jean_zay/README.md).

## Provenance

This repository was formerly `Graph-Native-Betti-Matching-Representation`, a fork of Alessandro
Benvenuti's Graph-Native-Betti-Matching. On 2 Nov 2026 all experiment worktrees were
consolidated here. Branches `research/*`, `brava-representation-20261026` and
`graph-extraction` keep each worktree's final state; see `../archive/refactor-20261102/`.
