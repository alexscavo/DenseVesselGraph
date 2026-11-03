# Projects cleanup — 2026-10-06

## Outcome

`projects` decreased from approximately **53 GB to 630 MB** (about 99%). WORK file-quota use decreased from **500,000/500,000 to about 399,800/500,000**, restoring roughly 100,000 free file entries. Generated spline targets and future run outputs now reside on SCRATCH; code, compact manifests and useful research documentation remain on WORK.

## Artifact locations

| Material | Action / current location |
| --- | --- |
| 44 GB dataset recovery archive | Moved to `/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/source-datasets-2026-10-05.tar` |
| Main spline target cache | Moved to `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/synthetic_mri_k8_10k_val2k`; repository path is a link |
| Spline prototype results and future main outputs | Moved to `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs`; repository path is a link |
| Earlier IXI graph run artifacts (5.2 GB) | Moved to `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/projects-archive-20261006/gnbm-baseline-runs`; original run path is a link |
| August `job-runs` output (569 MB) | Archived as `/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/job-runs-2026-08.tar.gz` (56 MB); 271 files verified byte for byte before removal. The old folder now contains an archive-location README. |
| Redundant emergency copies of 27 full graphs (about 936 MB) | Removed after SHA256 equality checks against restored graphs. The permanent recovery archive contains the same graph subjects. Snapshot metadata retained. |
| Old source-health inventory | Moved to `docs/archive/synthetic-mri-source-health-2026-10-05.json`; current dataset availability comes from restored verification/GT manifests. |
| Current manifests and selection | Copied to WORK under `experiments/spline_learning/metadata/` before main training |

Useful prototype documentation and measured comparison evidence are retained. Existing unrelated source edits are preserved. Reports now point to the current completed GT population and active job chain, with earlier failures labelled as history. `.gitignore` excludes generated spline artifacts and caches.

## Current experiment

GT job **676631 completed successfully**. Accepted populations: **9,709 train / 1,952 validation / 13,071 test**. Training **676650** queues after preparation; test **676692** waits for successful training. No repeat model smoke was run.

Stable commits: **24c2605** (spline implementation and eligibility fix), **98eb6a9** (verified restoration). The temporary screenshot is kept for the user's inspection at `DenseVesselGraph/experiments/spline_learning/temporary-debug/duplicate-edge.png`; it is generated and excluded from Git. Delete this temporary directory after inspection. The source patch ID and precise failure diagnosis remain in the training report.

## Reuse of the recovery archive

STORE is not accessible from the CPU compute node used during restoration. Stage a copy from STORE onto WORK/SCRATCH on a login node, then set `SYNTHETIC_MRI_RESTORE_ARCHIVE` when submitting `restore_synthetic_mri.slurm`. The recovery archive is kept permanently in the user's STORE rather than in projects.

## Follow-up: simplify the visible experiment folders

The user pointed out that many pilot/snapshot folders remained confusing despite their small size. Archived **both pilot caches, the four-patch targets and saved model runs, successful/failed smoke outputs, old recovery snapshots, submission bundles, temporary diagnostic images and superseded job logs** to:

`/lustre/fsstor/projects/rech/vnc/upz25mj/archives/2026-10-06/spline-prototypes-and-recovery-history.tar.gz`

Archive content was verified against the original files and links before their removal from the active workspace. These are useful historical evidence for reproducing the V0/V1 choice and the failure diagnosis; they are not inputs to main training. The old cache symlinks may require repointing to restored images if those experiments are extracted for reproduction.

The visible active folder now has one current dataset link, one current run-output link, current logs, a clearly named `metadata` directory, and a README explaining them. Current metadata includes exact accepted membership, preparation plan, source graph hashes, source crop configuration, restoration checks and active job IDs. Restore tools use this metadata rather than an ambiguous `source_snapshot` directory.

The initial 2D/3D, IXI and SyntheticMRI interactive representation examples remain under graph-representation/experiments as reference examples; their role is explained in the README. They are distinct from current model-training outputs.

Runtime recovery update: the fixed 1.4 GB package archive is stored outside projects at `/lustre/fswork/projects/rech/vnc/upz25mj/experiments/gnbm/spline-runtime-20261006/site-packages.tar`. Job-local extraction uses node temporary storage. It prevents the training job from repeatedly importing packages from the changing shared installation; it is required for the current resumed run, not an obsolete experiment folder.
