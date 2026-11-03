# Plants + leaves + SyntheticMRI: matched source mixture

## Decision

User requested the combined pretraining in addition to the separate plant and leaf studies. Use **12,500 existing plant patches + 12,500 existing leaf patches** as one 25,000-patch source pool. This preserves the original source budget and 50,000 replacement draws per epoch; it does not test doubling the source budget. The same 10,000 SyntheticMRI training patches, 1,952 validation patches and 13,071 held-out test patches are retained.

Source patches are selected by a fixed seed-dependent filename hash within each original training split; 240 validation patches per source give 480 source-validation patches. Original parent-image splits are preserved. Both source types receive source-domain label 0; MRI retains target label 1. Thus domain adaptation distinguishes biological source versus MRI, not three separate domains. Balanced source/target sampling yields approximately 25% plants, 25% leaves, 50% MRI draws, with the existing target upsampling. Domain loss weight remains **1.0**.

No images or graphs are regenerated: reuse the exact cached targets from the earlier plant and leaf runs, including degree-2 contraction and their existing crop/resize policies. A new dataset view under `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/biological_mixture_20261015/` uses source-prefixed IDs and symlinks to those targets. It copies no image payloads and does not modify either source. The normal PlantsDataset embeds both types into 64³ slabs with the established rotations/flips. MRI uses the approved 90-degree rotations and Gaussian-noise augmentation; zoom stays disabled.

## Training and evaluation

- Random-initialized graph-only pretraining: 50 epochs, batch 32, 50,000 draws/epoch, 120 object queries, existing optimizer/schedule and learning rate 7e-5.
- Two MRI-only fine-tunings from the same validation-selected pretraining checkpoint: straight edges and detached residual 8-control B-splines. Maximum 100 epochs with the unchanged early-stopping policy. Spline head initialized fresh at fine-tuning, spline weight 1.4739989612791453.
- Identical validation selection: common branch F1, native edge F1 as tie-breaker. BatchNorm saved statistics during evaluation; no recalibration.
- Fine-tuning best checkpoints evaluated on all 13,071 test IDs, with native AP/AR/F1, common geometry and topology, same-endpoint chords, Betti and core diagnostics.
- Automatic eight-model comparison against the separate leaves, plants and equal-update MRI-only arms. One seed; results cannot establish statistical superiority.

Configuration: `configs/combined_generalization_da_b32.yaml`.
Reproduction: `python3 cluster/jean_zay/submit_combined_generalization.py` (refuses duplicate submissions).
Submission record: `experiments/combined_pretraining/submission.json`.
Evaluation plan: `experiments/combined_pretraining/evaluation-plan.json`.
Training output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/combined_generalization_20261015/runs/`.
Test output shares `source_transfer_test_20261015/`, under unique `combined_straight` and `combined_spline` names. Automatic tables will appear in `docs/research/pretraining/combined-transfer-results/` after all eight tests succeed.

## Checks and current status

All three training configurations validated. Each original source has 25,000 training and 480 validation cached samples available; example raw/seg/VTP/spline paths checked. Python/shell syntax checks passed. CPU preparation validates every selected target before writing READY; a GPU smoke gate checks loading, training, domain gradients and both fine-tuning variants before full training. No production results yet. Earlier jobs/results are preserved.

| Stage | Job |
|---|---|
| prepare | 1147566 |
| verify | 1147567 |
| pretrain25 | 1147568 |
| pretrain50 | 1147569 |
| finetune50 | 1147570 |
| finetune100 | 1147571 |
| test | 1147572 |
| comparison | 1147573 |
