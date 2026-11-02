## Native2D-04 — longer fixed-epoch training and pure plant transfer, 28 October 2026

User authorized longer runs with the original node/edge confidence thresholds0.5, plants50 epochs, then vessel-only finetuning. This supersedes the previous17.3-epoch pilot stopping budget; earlier experiments remain preserved. No threshold calibration or architecture/loss/matcher change.

Protocol: [native2d_long_20261028.md](../docs/research/native2d_long_20261028.md). Queued/running states are not completed experimental results.
GPU gate job `1670808`; conditional training launcher `1670809`. The launcher submits training only after all dataset/GPU gates pass.
Full native ResNet101/512D,3+3 layers,5 relation tokens, common96 object queries from contracted training GT distributions. Each vessel stage:100 exact passes, batch4, identical exposure/augmentation/optimizer schedule within each B0/A-xyz pair.
Plant pretraining:50 exact passes, batch32. One common B0 plant final checkpoint initializes both finetuned variants; A-xyz starts with zero coordinate residuals. Fresh optimizer/scheduler at finetuning; no plant samples are replayed.

| Dataset | Train / validation crops | Train maximum nodes | Units |
|---|---:|---:|---|
| avwide | 1385 / 506 | 76 | pixel |
| reaver | 256 / 128 | 16 | micron |
| plants | 25867 / 6398 | 28 | pixel |
| octasynth | 10648 / 4840 | 52 | pixel |

| Run | Slurm | State | Epoch | Node mAP | Edge mAP | Node F1 @2 units | Edge F1 @2 units |
|---|---|---|---:|---:|---:|---:|---:|
| plants_50 | — | not submitted | — | — | — | — | — |
| avwide_scratch_B0 | — | not submitted | — | — | — | — | — |
| avwide_scratch_Axyz | — | not submitted | — | — | — | — | — |
| avwide_plants_B0 | — | not submitted | — | — | — | — | — |
| avwide_plants_Axyz | — | not submitted | — | — | — | — | — |
| reaver_scratch_B0 | — | not submitted | — | — | — | — | — |
| reaver_scratch_Axyz | — | not submitted | — | — | — | — | — |
| reaver_plants_B0 | — | not submitted | — | — | — | — | — |
| reaver_plants_Axyz | — | not submitted | — | — | — | — | — |
| octasynth_scratch_B0 | — | not submitted | — | — | — | — | — |
| octasynth_scratch_Axyz | — | not submitted | — | — | — | — | — |
| octasynth_plants_B0 | — | not submitted | — | — | — | — | — |
| octasynth_plants_Axyz | — | not submitted | — | — | — | — | — |

Primary comparisons use the final fixed-epoch checkpoints on complete validation populations; selected checkpoints are secondary. Selection uses fixed interim source-image groups, excludes initialization and resolves trained ties toward later epochs. Pixel and micron metrics are reported separately. Raw versus filtered recall, conditional relation populations/AP, duplicates, components/cycles and same-graph K8/chord geometry remain in per-run validation JSON. Plant pretraining reports detection/graph metrics on its sparse annotations; its geometry is not an independently traced dense reference.

No test arrays or predictions are used. AV-WIDE remains17 training /6 validation images with unknown patient independence; image07 excluded. REAVER is the separately verified diaphragm60x4/2-image cohort in microns. OCTA uses official source-image membership and regenerated128² crops with coordinate/crop-boundary QC. Plant parent IDs are disjoint across official train/val/test membership.

Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_long_20261028`. Source/config hashes: `source.zip`, `code-identities.json`; dataset manifests/audits: `<dataset>/`; epoch losses and full state checkpoints: `runs/<name>/`; final metrics: `final-heldout.json`; plots: `curves-<dataset>.png`. Stop at50 plant /100 vessel epochs; exceptions stop jobs without automatic retries, new sweeps or budget extensions.
