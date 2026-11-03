# Experiment records

This folder keeps code-adjacent provenance for experiments: plans, compact JSON results,
reports, logs and small analysis files. Large volumes, checkpoints and predictions live on
SCRATCH under `$SCRATCH/experiments/gnbm/` (some subfolders here are symlinks to it).
What each study found is summarised in [`../docs/research/HISTORY.md`](../docs/research/HISTORY.md);
the original reports and plans of the studies up to 29 Oct 2026 are archived
([`../docs/archive/README.md`](../docs/archive/README.md)).

## Current model (from 30 Oct 2026)

| Folder | Role |
| --- | --- |
| `dense_graph_20261030/` | Dense two-stage model and traced connectivity on SyntheticMRI, IXI and BRAVA: training logs, validation results, paired comparisons (`paired-*.json`), AP and topology tables. Described in [`../docs/research/dense_traced_model_20261102.md`](../docs/research/dense_traced_model_20261102.md) |
| `dense_graph_2d_20261030/` | 2D port of the dense 16-NN model (`report.md`) |
| `b0_straight_dense_20261101/`, `straight_lineage_20261101/` | RelationFormer B0 and the trained straight-edge RelationFormer, scored with the dense-model evaluator |
| `anchored_rf_20261030/` | Heatmap-anchored RelationFormer |
| `node_heatmap_probe_20261030/`, `oracle_geometry_20261030/`, `ixi_long_20261030/` | Node-heatmap probe, oracle-topology edge-geometry study, literature-scale IXI RelationFormer control |

The full design and every intermediate experiment of this phase are in
[`../docs/research/network_proposals_20261030.md`](../docs/research/network_proposals_20261030.md).

## Earlier studies (to 29 Oct 2026)

All other folders are records of earlier RelationFormer, transfer, representation, BRAVA and
native-2D studies. Their status words (queued, running) describe the time they were written;
`sacct` and the research history are the reference. Two non-experiment folders:

- `storage_cleanup/`: audited removal and archive manifests of the 16, 25 and 27 Oct 2026
  cleanups. `storage_cleanup/20261027/secondary-artifacts.zip` holds old launchers and
  secondary audit/viewer files; restore them with `storage_cleanup/20261027/restore.py`, which
  refuses to overwrite differing files. Some loose prediction NPZ files of three October studies
  were zipped by the 25 Oct cleanup and need selective restoration before reuse.
- `coordination/`: small coordination records between parallel studies.

## Retention rule

Keep validation-selected best and final/resume checkpoints for completed scientific comparisons;
keep source pretraining checkpoints, population manifests, configs, logs, predictions used by
metrics and representative visualizations. A failed approach can still be important evidence.
Smoke checkpoints are disposable once startup is verified. Abandoned attempts may be packed into
a verified archive; do not delete them merely because a newer experiment exists.

No automatic age-based deletion. Datasets and derived graph/spline targets stay in their existing
paths because configurations and repeat evaluations depend on them. SCRATCH purges untouched
files, so anything irreplaceable belongs on WORK or STORE. Keep W&B offline records until
synchronized or deliberately archived with restore instructions.
