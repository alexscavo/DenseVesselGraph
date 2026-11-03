# Archived documentation (3 Nov 2026)

To keep the repository readable, the research logs and evidence reports written before the
switch to the dense traced model were removed from the working tree on 3 Nov 2026. Their
important findings, with numbers and caveats, are condensed in
[`../research/HISTORY.md`](../research/HISTORY.md).

## What was archived

- `docs/archive/legacy-20261103/` holds the original research logs, reviews, plans and overviews
  up to 29 Oct 2026 (124 files):
  - the former top-level `research/` folder (roadmap, architecture, decisions, experiment
    registry);
  - `IMPLEMENTATION_TRACKER.md`, `BRAVA_experiment_plan.md`, `EXPERIMENT_REVIEW_*`,
    `vascular_graph_*`, `native2d_*`;
  - the `metrics/`, `spline_learning/`, `pretraining/` and `storage/` folders;
  - `ALL_EXPERIMENTS_AND_RESULTS.md`;
  - `docs/EXPERIMENTS.md`, `docs/SPLINE_REPRESENTATION.md` and `docs/repo-graph.md`.
- `docs/research/evidence/` held per-study evidence reports, JSON/CSV tables, plots and HTML
  viewers. Two items stay because code reads them:
  - `brava_preparation/`, read by `scripts/prepare_brava.py` and `cluster/jean_zay/submit_brava.py`;
  - `ixi_mechanisms_20261019/D/focal_control/resolved-config.json`, read by
    `tests/test_ixi_node_focal_ablation.py`.
- `docs/research/history_*.md` (eight topic histories) and `docs/research/README.md` were merged
  into `HISTORY.md`.
- Merged reference docs:
  - `docs/AUGMENTATION.md` and `docs/DATA_LOADING.md` went into `docs/DATA_PIPELINE.md`;
  - `docs/LOSSES.md` and `docs/experiments/domain_adaptation.md` went into `docs/TRAINING.md`;
  - `docs/DATA_AUDIT.md` went into `docs/SYNTHETIC_MRI_DATASET_REPORT.md`.

## Where it is

- **Tarball (708 files, about 102 MB):**
  `/lustre/fsstor/projects/rech/vnc/upz25mj/DenseVesselGraph-docs-archive-20261103.tar.gz`, on
  STORE. It also contains the git-ignored files that were never committed: empty collector
  `*.lock` files, `continuous_representation/control-selector.joblib` and
  `ixi_features_nodes_20261018/probes.pt`.
- **Git history.** Commit `8fe22bc` is the last commit with all of these files in the tree.

## Restore

From the repository root:

```bash
# one file or folder, from git
git checkout 8fe22bc -- docs/archive/legacy-20261103/research/roadmap.md
git checkout 8fe22bc -- docs/research/evidence/T05

# everything, including git-ignored files, from the tarball
tar -xzf /lustre/fsstor/projects/rech/vnc/upz25mj/DenseVesselGraph-docs-archive-20261103.tar.gz
# or a single path
tar -xzf /lustre/fsstor/projects/rech/vnc/upz25mj/DenseVesselGraph-docs-archive-20261103.tar.gz docs/research/evidence/T05
```

Inside the archive, paths under `docs/archive/legacy-20261103/` mirror the repository root.
Their links were rewritten on 3 Nov 2026 to resolve from that location.

Several report collectors in `scripts/` still write to `docs/research/evidence/<study>/` by
default. Re-running one recreates that folder.
