# Research documents — index

Vessel-graph extraction from 3D MRA (SyntheticMRI, IXI, BRAVA, TopBrain) and 2D images
(AV-WIDE, REAVER, OCTA-Synth). Reorganised on 3 Nov 2026: the older dated reports were
condensed into the topic histories below. The originals are in
[`../archive/legacy-20261103/`](../archive/legacy-20261103/README.md) and every history cites
the originals it summarises.

Status labels used throughout: **DONE**, **RUNNING** (when last documented), **PROPOSED**
(never run), **SUPERSEDED by …**, **ABANDONED**. Project shorthand (B0, K8, A-xyz, D08,
`monitor_train`, Branch/Curve F1, MM/MU/UU …) is defined in the glossary of
[`network_proposals_20261030.md`](network_proposals_20261030.md#glossary) and again in
each history's "Shorthand" list.

## 1. Read first: the current model

| Document | What it is |
|---|---|
| [`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) | The chosen model: a full-resolution 3D U-Net that detects nodes as heatmap peaks, traces edges along the predicted centreline, and gives each edge a K8/K16/K24 B-spline. It has the main results. All are validation results from a single seed, with thresholds chosen on `monitor_train`. |
| [`network_proposals_20261030.md`](network_proposals_20261030.md) | Full design, glossary and the 30 Oct – 1 Nov comparisons: RelationFormer B0, heatmap-anchored RelationFormer, the dense model, average precision, traced vs 16-NN connectivity, and the straight-edge control. |

## 2. Consolidated histories of the earlier work (to 29 Oct 2026)

Almost all of this is RelationFormer work that the dense traced model superseded. Suggested
reading order:

| Document | Covers | Main takeaway |
|---|---|---|
| [`history_relationformer_3d.md`](history_relationformer_3d.md) | 3D RelationFormer on SyntheticMRI and IXI, Sept – 20 Oct: the T09/T10 runs, focal and degree pilots, query capacity, and IXI diagnostic series 1–11 with their late-October follow-ups | Diagnosed the failure mechanisms: nodes are not localised by the stride-32 grid; matching at training does not match inference; relations ignore the image between nodes. §5 maps each one to the dense-model fix. |
| [`history_multidataset_brava.md`](history_multidataset_brava.md) | 21–29 Oct: multi-dataset continuous-graph batch, BRAVA stage 1 to the D08 long run, the K8 learnability (D06) and augmentation (D07) diagnostics, node verifier, fine-feature and local-refinement (A-xyz/A-cls), N07/N08, R06/R07, Q0, L08, the method gate and the 8-week plan | Long, well-fitted BRAVA training still left node recall as the limit. This led to the 30 Oct diagnosis and the switch of architecture. |
| [`history_continuous_edges.md`](history_continuous_edges.md) | Learning curved edges with RelationFormer, 5–13 Oct: spline heads, the scratch collapse, gradient routes, local-image geometry, Catmull–Rom, curve-informed connectivity | Splines improve curve quality but not the graph. This end-state became B0's spline head and `graph_only` route. The oracle-geometry study confirmed it later. |
| [`history_representation.md`](history_representation.md) | What graph to predict: junction-only vs adaptive vs dense, the simplification policy, off-vessel chords, query capacity, and how many spline controls a GT edge needs | "Topology + curved edge" was the chosen form. Only long or tortuous branches (mostly in BRAVA) need more than 8 controls. |
| [`history_data.md`](history_data.md) | GT graph extraction and dataset production for IXI and TopBrain: Vedo centrelines, production runs, patch grid and eligibility, false loops, storage and retention | These datasets are still in use. False segmentation loops remain unmeasured label noise. |
| [`history_metrics.md`](history_metrics.md) | Part 1 defines the metrics in use (node/edge F1 @2 mm, MM/MU/UU, common-v2 Branch/Curve F1, β errors, SMD, AP). Part 2 covers the metric plan, the T07/T08 metric validation, and vessel-core v1 | Branch F1 and Curve F1 are reported together. Box-based "native" edge F1 overstates positional accuracy. Several planned metrics were never used. |
| [`history_pretraining_transfer.md`](history_pretraining_transfer.md) | Botanical (plants, HALVS leaves) and SyntheticMRI pretraining, source augmentation, IXI transfer and overfit, domain adaptation, the multiscale pilot, and the TopBrain estimate | At full data, pretraining never beat a matched target-only control. Many of these numbers come from the test split. The dense model trains from scratch. |
| [`history_native2d.md`](history_native2d.md) | Native 2D RelationFormer (Native2D-01..05: AV-WIDE, REAVER, OCTA-Synth) and the Image-to-Graph-Vessel inventory | Superseded by the 2D dense model. Plant-pretraining benefits were confounded and were measured with a broken 2D backbone. |

## 3. Reference documents (how to run things)

These are kept up to date separately from the research log:
- [`../MODEL.md`](../MODEL.md), [`../TRAINING.md`](../TRAINING.md),
  [`../EVALUATION.md`](../EVALUATION.md), [`../RUNNING_THE_MODEL.md`](../RUNNING_THE_MODEL.md),
  [`../LOSSES.md`](../LOSSES.md), [`../AUGMENTATION.md`](../AUGMENTATION.md) and
  [`../DATA_LOADING.md`](../DATA_LOADING.md): the RelationFormer code path.
- [`../GRAPH_PREDICTION_3D.md`](../GRAPH_PREDICTION_3D.md): the Plotly viewer.
- [`../experiments/domain_adaptation.md`](../experiments/domain_adaptation.md): the opt-in
  domain-adaptation switch.
- Data: [`../IXI_DATASET.md`](../IXI_DATASET.md),
  [`../DATA_GENERATION.md`](../DATA_GENERATION.md),
  [`../SYNTHETIC_MRI_DATASET_REPORT.md`](../SYNTHETIC_MRI_DATASET_REPORT.md),
  [`../DATA_AUDIT.md`](../DATA_AUDIT.md) and
  [`BRAVA_preparation.md`](BRAVA_preparation.md).

Known stale points as of 3 Nov 2026:
- `EVALUATION.md` describes only the native `evaluate.py` metrics. These are box mAP/mAR with a
  40-detection cap and box-based F1. It does not cover the physical metrics in the current
  docs; see [`history_metrics.md`](history_metrics.md) Part 1. It still has commands for the
  old Magnolia cluster.
- `IXI_DATASET.md` §8 still lists the simplification tolerance as undecided and quotes 120
  object queries. Both were settled later, as the radius-0.75× solver and 256 queries; see
  [`history_representation.md`](history_representation.md).

## 4. Evidence

[`evidence/`](evidence/) holds the compact per-study reports, JSON/CSV tables, plots and 3D
viewers that the histories cite. It was not moved. Main folders by topic:
- **Extraction and representation:** `T01`–`T05`, `adaptive_tuning/`, `vedo_original_ct001/`,
  `vedo_production/`, `full_dataset_production/`, `spline_target_capacity_20261022/`,
  `continuous_representation/`.
- **Metrics:** `T07/` and `T08/`.
- **IXI RelationFormer:** `T09/`, `T09_long/`, `T10_*`, `focal_ixi_300ep/`,
  `degree_consistency/`, `ixi_query_benchmark/`, `ixi_capacity_192_256/`, and the dated
  `ixi_*_202610*` folders for diagnostic series 1–11.
- **BRAVA and multi-dataset:** `brava_preparation/`, `brava_stage1_20261021/`,
  `brava_supervision_20261027/`, `multidataset_continuous/`, `node_verifier_feasibility/`,
  `fine_feature_architecture/`, `local_refinement_long/`, `ixi_local_components_20261028/`.
- **Pretraining:** `domain_ixi_pretraining/`, `multiscale_pretrained/` and
  `topbrain_training_time_estimate/`.

Figures for the current model are in [`figures/`](figures/). Large run outputs live on
`$SCRATCH/experiments/gnbm/`. Per-study plans, manifests and small results are in the
repository's [`experiments/`](../../experiments/) folder.

## 5. Archive

[`../archive/legacy-20261103/`](../archive/legacy-20261103/README.md) mirrors the repository
paths of everything moved on 3 Nov 2026. It holds:
- the former top-level `research/` overview (roadmap, architecture, decisions, experiment
  register, Image-to-Graph-Vessel inventory, 26 Oct experiment registry);
- `docs/research/` reports, reviews and plans (`IMPLEMENTATION_TRACKER`, `BRAVA_experiment_plan`,
  `EXPERIMENT_REVIEW_*`, `vascular_graph_*`, native2d …);
- the `metrics/`, `spline_learning/`, `pretraining/` and `storage/` folders;
- `docs/EXPERIMENTS.md`, `docs/SPLINE_REPRESENTATION.md` and `docs/repo-graph.md`;
- `ALL_EXPERIMENTS_AND_RESULTS.md`, a verbatim 19 Oct compilation of the reports of that time.

The archive index lists which history summarises each file.
