# DenseVesselGraph

Vessel-graph extraction from 3D MRA (SyntheticMRI, IXI, BRAVA) and 2D images (AV-WIDE, REAVER,
OCTA-Synth). The repository holds the chosen **two-stage dense model with traced
connectivity**, together with every RelationFormer baseline, diagnostic and experiment
record of this research line.

- Main model and results: [`docs/research/dense_traced_model_20261102.md`](docs/research/dense_traced_model_20261102.md)
- Full design, glossary and comparisons: [`docs/research/network_proposals_20261030.md`](docs/research/network_proposals_20261030.md)
- Code: `scripts/dense_graph_20261030.py`, `scripts/dense_graph_traced_20261030.py`, `scripts/dense_graph_2d_20261030.py`
- Research history and document index: [`docs/research/README.md`](docs/research/README.md) (consolidated histories of the earlier RelationFormer work; originals archived under [`docs/archive/legacy-20261103/`](docs/archive/legacy-20261103/))

This repository was formerly `Graph-Native-Betti-Matching-Representation`, a fork of Alessandro
Benvenuti's Graph-Native-Betti-Matching. On 2 Nov 2026 all experiment worktrees were
consolidated here. Branches `research/*`, `brava-representation-20261026` and
`graph-extraction` keep each worktree's final state; see
`../archive/refactor-20261102/`.

---

# Graph-Native Betti Matching

## Research overview

The research documentation index is [`docs/research/README.md`](docs/research/README.md):
it lists what to read first (the dense traced model), the consolidated topic histories of
the earlier work, and the evidence folders. The former top-level `research/` overview
(roadmap, architecture, decisions, experiment register) and the older dated reports were
moved, unchanged apart from link targets, to
[`docs/archive/legacy-20261103/`](docs/archive/legacy-20261103/) on 3 Nov 2026.


Clean research implementation of a 3D graph-extraction pipeline with graph-native topology objectives.

This repository is being reconstructed from an experimental codebase. It
contains the reviewed augmentation and loading pipeline, validated
configuration schema, checkpoint-compatible 3D RelationFormer, the
baseline/Betti/focal training stack, graph inference, corrected baseline
evaluation metrics, plotting/export, and loss-based validation/checkpoint
selection.

## Intended scope

The first supported workflow will focus on:

- 3D training and checkpoint evaluation;
- mixed plants and synthetic-MRI pretraining;
- synthetic-MRI finetuning;
- RelationFormer-style node and relation prediction;
- focal hard-negative mining;
- focal relation objectives;
- graph-native Betti H0/H1 objectives;
- baseline-compatible graph inference and loss-based model selection.

Diameter-smoothing losses, metrics, profiling tools, and experiment configurations are intentionally out of scope.

Road, OCTA, MoCo, 2D pretraining, and general-transformer adapter paths will not be migrated unless they are explicitly restored as supported requirements.

## Planned structure

```text
.
├── research/                Consolidated roadmap, architecture, decisions and experiments
├── configs/                 Reproducible training and evaluation configurations
├── data/
│   ├── loaders/             Dataset loading and batching
│   └── preprocessing/       Dataset creation, splitting, and normalization
├── models/
│   └── ops/                 Custom deformable-attention extension
│       ├── functions/       Python autograd bindings
│       ├── modules/         PyTorch modules
│       └── src/
│           ├── cpu/         C++ CPU implementation
│           └── cuda/        CUDA implementation
├── training/
│   ├── losses/              Detection, relation, and topology losses
│   └── evaluation/          Inference, metrics, and BN calibration
├── metrics/                 Reusable evaluation metrics
├── boxes/                   Box and non-maximum-suppression operations
├── utils/                   Small shared utilities
├── tests/                   Unit and integration tests
├── scripts/                 Reproducible workflow entry points
└── docs/                    Training, configuration, and evaluation documentation
```

## CUDA extension policy

The custom CUDA/C++ source code required by deformable attention belongs in `models/ops/src/` and will be version-controlled.

Precompiled shared libraries, object files, and `build/` directories will not be committed. They are tied to the operating system, CPU architecture, Python version, PyTorch version, CUDA toolkit, compiler, and C++ ABI of the machine that built them. The extension must therefore be compiled in the Jean Zay environment used for training.

The future build documentation will record the tested Jean Zay modules and provide a reproducible build or installation command. Build output should be generated locally on the cluster and can be cached outside Git when useful.

## Status

The augmentation and data-loading contracts, merged/validated configuration
loading, reproducible batching, optional real-dataset checks, minimal 3D
RelationFormer, PyTorch deformable-attention fallback, CUDA extension source,
Hungarian matching, modular graph losses, optimizer, scheduler, strict resume,
training loop, inference, corrected SMD/AP/topology evaluation, graph export,
plotting, and baseline loss-based model selection are in place. See
`docs/MODEL.md`, `docs/LOSSES.md`, `docs/TRAINING.md`, and
`docs/EVALUATION.md`. The read-only SyntheticMRI source/grid audit is documented
in `docs/DATA_AUDIT.md`. The first controlled ablation matrix is documented in
`docs/archive/legacy-20261103/docs/EXPERIMENTS.md` (archived; never run). The completed SyntheticMRI dataset findings and proposed
decisions are recorded in `docs/SYNTHETIC_MRI_DATASET_REPORT.md`.
The safe, resumable v2 split and patch generator is documented in
`docs/DATA_GENERATION.md`.

Jean Zay H100 environment setup, the bounded H100 development test, production
submission, and resume instructions are documented in
[`cluster/jean_zay/README.md`](cluster/jean_zay/README.md).
