# Experiment navigation

Keep code and compact provenance here; large volumes, checkpoints and predictions live on SCRATCH. The consolidated overview is in [`research/experiments.md`](../research/experiments.md); detailed scientific protocols and evidence remain in `docs/research/`.

Latest: [completed longer local refinement](../docs/research/local_refinement_long_study.md); [25 October quota audit and verified prediction archives](../docs/research/storage/cleanup-20261025.md). Older loose NPZ paths in three completed studies now require selective archive restoration.

## Current work and paper comparisons

| Folder / report | Role |
| --- | --- |
| `source_augmentation/` | Nonplanar/textured prototypes and opt-in training integration; loader checks passed; matched augmented-source chains submitted 2026-10-18 |
| `plant_pretraining/`, `leaf_pretraining/`, `combined_pretraining/` | SyntheticMRI source-transfer launch records; final comparison in `docs/research/pretraining/combined-transfer-results/` |
| `ixi_transfer/` | IXI-only / plants+IXI / SyntheticMRI+IXI comparison; straight, spline throughout and delayed spline |
| `ixi_overfit/` | Four-patch IXI learnability diagnostic |
| `spline_learning/metadata/` | Exact membership, diagnostic results, source hashes and submission provenance |
| `spline_learning/runs/` | Link to historical and main scratch/geometry run outputs; use the retained register below |
| `storage_cleanup/20261016/` | Audited removal manifest, verified archive manifest and scripts; not a training experiment |

Start with [the current research/storage map](../docs/research/storage/cleanup-20261016.md) and [retained experiment register](../docs/research/storage/retained-experiments.md).

## Retention rule

Keep validation-selected best and final/resume checkpoints for completed scientific comparisons; keep source pretraining checkpoints, population manifests, configs, logs, predictions used by metrics and representative visualizations. A failed approach can still be important evidence. Smoke checkpoints are disposable once startup is verified. Abandoned attempt evidence may be packed into a verified archive; do not delete it merely because a newer experiment exists.

No automatic age-based deletion. Datasets and derived graph/spline targets remain in their existing paths because configurations and repeat evaluations depend on them. Keep W&B offline records until synchronized or deliberately archived with restore instructions.

## Archived secondary files — 27 October 2026

Completed historical launchers and secondary audit/viewer artifacts are in `storage_cleanup/20261027/secondary-artifacts.zip`. See the [cleanup and selective restore instructions](../docs/research/storage/cleanup-20261025.md#repository-decluttering--27-october-2026). Reports, compact results, code, tests and checkpoints remain available; restore archived wrappers before repeating old SLURM commands.
