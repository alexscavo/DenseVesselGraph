# Experiment records

This folder keeps code-adjacent provenance for experiments: plans, compact JSON results,
reports, logs and small analysis files. Large volumes, checkpoints and predictions live on
SCRATCH under `$SCRATCH/experiments/gnbm/` (some subfolders here are symlinks to it).
[`../docs/research/HISTORY.md`](../docs/research/HISTORY.md) summarises what each study found.

## Live folders

**Current model (from 30 Oct 2026).** The full design and every intermediate experiment of this
phase are in [`../docs/research/network_proposals_20261030.md`](../docs/research/network_proposals_20261030.md).

| Folder | Role |
| --- | --- |
| `dense_graph_20261030/` | Dense two-stage model and traced connectivity on SyntheticMRI, IXI and BRAVA: training logs, validation results, paired comparisons (`paired-*.json`), AP and topology tables. Described in [`../docs/research/dense_traced_model_20261102.md`](../docs/research/dense_traced_model_20261102.md) |
| `dense_graph_2d_20261030/` | 2D port of the dense 16-NN model (`report.md`) |
| `b0_straight_dense_20261101/`, `straight_lineage_20261101/` | RelationFormer B0 and the trained straight-edge RelationFormer, scored with the dense-model evaluator |
| `anchored_rf_20261030/` | Heatmap-anchored RelationFormer |
| `node_heatmap_probe_20261030/`, `oracle_geometry_20261030/`, `ixi_long_20261030/` | Node-heatmap probe, oracle-topology edge-geometry study, literature-scale IXI RelationFormer control |

**Plans and prepared data that current code reads.**

| Folder | Role |
| --- | --- |
| `multidataset_continuous/` | Train/validation split and evaluation plan (`plan.json`) used by all current 3D code |
| `native2d_long_20261028/` | Prepared native 2D datasets (AV-WIDE, REAVER, OCTA-Synth, plants: manifests and QC), read by the 2D scripts through `native2d_mixed_20261028/plan.json` |
| `native2d_20261028/`, `native2d_20261028_targetfix/` | Native2D-02 and -03 AV-WIDE studies (original and corrected K8 targets). The AV-WIDE and REAVER manifests in `native2d_long_20261028/` point to the prepared crops and targets in these folders |
| `brava_preparation/` | BRAVA SWC preparation and orientation/alignment probes |

## Superseded studies

The 49 earlier study folders (RelationFormer, transfer, representation, BRAVA, native-2D,
audits and cleanup records, up to 30 Oct 2026) were removed on 3 Nov 2026.
[`SUPERSEDED.md`](SUPERSEDED.md) has one entry per folder (what it tested and why it was
superseded), lists the current scripts that still read files inside some of them, and gives
restore instructions. Each folder is archived as
`$STORE/DenseVesselGraph-experiments-archive-20261103/<folder>.tar.zst`
(`STORE=/lustre/fsstor/projects/rech/vnc/upz25mj`). The original reports and plans of the studies
up to 29 Oct 2026 are archived separately ([`../docs/archive/README.md`](../docs/archive/README.md)).

## SCRATCH backup of current runs

SCRATCH purges untouched files. The current run outputs under `$SCRATCH/experiments/gnbm/` (model
checkpoints, `final.json`, validation rows) are backed up in
`$STORE/DenseVesselGraph-experiments-archive-20261103/current-dense-graph-runs-scratch-20261103.tar.zst`.

## Retention rule

Keep validation-selected best and final/resume checkpoints for completed scientific comparisons;
keep source pretraining checkpoints, population manifests, configs, logs, predictions used by
metrics and representative visualizations. A failed approach can still be important evidence.
Smoke checkpoints are disposable once startup is verified. When a study is superseded, pack its
folder into a verified zstd archive on STORE, add an entry to [`SUPERSEDED.md`](SUPERSEDED.md)
(what it tested, why it was superseded, archive path), and only then remove it from the
repository. Never delete a study merely because a newer experiment exists.

No automatic age-based deletion. Datasets and derived graph/spline targets stay in their existing
paths because configurations and repeat evaluations depend on them. SCRATCH purges untouched
files, so anything irreplaceable belongs on WORK or STORE. Keep W&B offline records until
synchronized or deliberately archived with restore instructions.
