# SyntheticMRI restoration — 2026-10-05

## Requested action

Restore SyntheticMRI from today's dataset archive to `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI`, replacing the existing incomplete folder. The old folder is retained as `syntheticMRI.before-restore-627540` after successful verification. No changes are made to the other account's dataset or its archive.

## Source and execution

Archive: `/lustre/fsstor/projects/rech/vnc/upz73jr/Graph-Native-Betti-Matching/2026-10-05/datasets.tar` (44 GB). A read-only archive scan found all 136 full segmentation volumes, the original generation configuration, 91 `graph.vvg` files and prepared SyntheticMRI patch data. This is a recovery archive, not evidence of complete full-graph coverage.

First job 627391 could not access STORE from its compute node and failed before extraction. The archive was copied to WORK as `DenseVesselGraph/experiments/spline_learning/datasets-restoration.tar`; retry job **627540** extracts only SyntheticMRI to a staging folder. Fresh filesystem timestamps are used while preserving file contents.

## Checks before replacement

- Generation configuration and patch-index contents match the copies retained from the successful pilot.
- All 130,560 indexed boundary patches have nonempty MRI, segmentation and VTP files.
- All 136 segmentation headers have valid invertible transforms and mm units.
- Previously verified pilot transforms agree with restored headers.
- One actual MRI/mask payload per split loads with finite values and the expected 64³ shape.

These checks do not validate every voxel in every volume. Full-graph availability is recorded separately in `restore-verification.json`.

## Status

Job **627540 completed restoration and verification successfully**. The target folder has been replaced. The previous folder is retained at `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI.before-restore-627540`. Verified 391,680 boundary-patch files (91,200 train / 19,200 val / 20,160 test triplets), all 136 segmentation headers, and 91 complete graph subjects. No experimental split membership or graph snapshot was changed. GT preparation now points to the restored dataset path. Long training has not been restarted by this restoration.

Verification report: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI/restore-verification.json`; a copy including all 136 verified affines is retained in WORK at `experiments/spline_learning/source_snapshot/restore-verification.json`. The full archive copy is retained in WORK at `experiments/spline_learning/datasets-restoration.tar`.
