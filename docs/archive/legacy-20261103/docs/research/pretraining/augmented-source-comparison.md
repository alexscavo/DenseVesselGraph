# Augmented biological-source comparison — 2026-10-18

## Research question and scope

Does the selected nonplanar + textured-soft rendering policy improve transfer from plants, leaves or their mixture to SyntheticMRI, compared with the completed original-rendering experiments?

This is a matched **SyntheticMRI-target** comparison. IXI is not included in this submission. Three independent pretrainings start from scratch; each initializes two fine-tunings (straight edges and detached K8 residual B-splines). Splines are introduced only in fine-tuning, matching the previous SyntheticMRI experiment. Changing to spline supervision during pretraining would confound this augmentation comparison.

## Fixed comparison

| Source arm | Unique source patches | Target training | New models |
| --- | ---: | ---: | --- |
| Plants | 25,000 | Same 10,000 SyntheticMRI | Straight, spline |
| Leaves | 25,000 | Same 10,000 SyntheticMRI | Straight, spline |
| Plants + leaves | 12,500 + 12,500 | Same 10,000 SyntheticMRI | Straight, spline |

- Pretraining: 50 epochs, 50,000 replacement draws per epoch, expected 50/50 source/target balance, batch32, domain-adaptation weight1.0.
- Fine-tuning: 100 epochs, SyntheticMRI only, no domain loss; no early stopping. LR, optimizer, loss weights, seed364505, architecture and geometry gradient routing match the original plans.
- Same 1,952 validation and 13,071 held-out test patches. Same fixed 128-case validation monitoring and 512-case training monitor. Validation selects best by common branch F1, native edge F1 tie-breaker; no test-based selection.
- Source changes only: shared depth warp, refitted continuous targets when applicable, fresh correlated texture and mild soft boundaries. MRI augmentation and all evaluation inputs are unchanged. Input policy and four-sample checks are in `source-augmentation-training.md`.
- Single seed. This evaluates the **combined** augmentation policy, not independent contributions of warping and texture; it does not establish statistical superiority on its own.

## Training and dependencies

The existing datasets and all eight completed baseline results are reused without regeneration or repeated held-out evaluation. Preflight verified source/smoke READY markers, target membership, separate output roots, fixed epochs/batch, and baseline test-ID hashes. The comparison collector was exercised using all eight existing summaries, preserving their values.

Each arm has a GPU gate: focused geometry tests, existing domain tests, and three-update checks for mixed/MRI-only pretraining plus straight/spline fine-tuning. Main training starts only after success. The full pretraining is split at epoch25 and fine-tuning at epoch50 for resumability, preserving optimizer/scheduler/RNG between stages.

Each task uses one H100 and six CPUs; sources and fine-tuning arms may run concurrently on separate GPUs subject to scheduler allocation. Requests: smoke1h, each pretraining stage12h, each fine-tuning stage6h, each full test3h. These are limits, not measured completion estimates. Initial inspection confirmed all three gates running and every downstream dependency correct.

| Source | GPU gate | Pretrain 25 → 50 | Fine-tune 50 → 100 (straight/spline array) | Tests |
| --- | --- | --- | --- | --- |
| plants | 1255158 | 1255159 → 1255160 | 1255161 → 1255162 | 1255163 |
| leaves | 1255164 | 1255165 → 1255166 | 1255167 → 1255168 | 1255169 |
| combined | 1255170 | 1255171 → 1255172 | 1255173 → 1255174 | 1255175 |

Collector job **1255176** waits for all six tests, then writes validation and held-out test comparison tables (CSV + Markdown) to `docs/research/pretraining/augmented-transfer-results/`. Fourteen rows: six augmented models plus six original biological-source models and two MRI-only controls. Missing/new results are never silently omitted.

Evaluation retains native node/edge mAP/mAR/F1, common branch/curve geometry, and existing vessel-core topology/Betti/APLS/TOPO diagnostics. Saved BatchNorm statistics are used; no recalibration. Curves can also be compared with identical-endpoint chords. Full metrics and prediction outputs stay on SCRATCH.

## Artifacts and reproducibility

- Configurations: `configs/{plant,leaf,combined}_generalization_da_b32_source_aug.yaml`.
- Submission script: `cluster/jean_zay/submit_augmented_sources.py` (default dry-run/preflight; `--submit` records each accepted job; refuses duplicate submission).
- Exact commands and IDs: `experiments/source_augmentation/training_20261018/submission.json`.
- Scheduler verification: `experiments/source_augmentation/training_20261018/scheduler-check.json`.
- Evaluation membership/protocol/run mapping: `experiments/source_augmentation/training_20261018/evaluation-plan.json`.
- New test output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/augmented_source_tests_20261018/`.
- Training roots are distinct from baselines and end `_source_aug_20261016`, following the implementation configuration date.

W&B remains offline as in the baseline recipe, including its validation visualizations. At successful smoke completion only disposable best/latest smoke weights are removed; summaries/configs/provenance remain. Main checkpoints are retained. Job-local runtime extraction is removed on shell exit to avoid the prior `/tmp` accumulation problem.

## Interpretation after completion

Compare each augmented model with its own unaugmented source/representation control, then with MRI-only. Check graph detection/connectivity and geometry separately, and compare training/validation behavior. Better image appearance is not evidence of improved learning. Full comparison results are pending; no gain is claimed at submission.
