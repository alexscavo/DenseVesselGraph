# Native 2D two-stage dense model vs RelationFormer B0 — results (31 Oct 2026) [DONE]

Code and protocol: [`README.md`](README.md); design: GNBM-R `docs/research/network_proposals_20261030.md` §3–4.

## Runs

| Dataset | Job | Elapsed | Train crops | Monitor crops | Validation crops |
|---|---|---|---|---|---|
| AV-WIDE | 1761237 | 22m07 | 1,385 | 64 | 506 |
| REAVER | 1761239 | 20m13 | 256 | 16 | 128 |
| OCTA-Synth | 1761240 | 28m34 | 10,648 | 64 | 4,840 |

All three runs:
- `--run-name spline6`, 15k updates × batch 8 (120k exposures), LR 1e-3, 2k GT-only warm-up;
- from scratch (no pretraining), one H100 each;
- preceded by GPU smoke 1745322 (avwide, 200 updates, COMPLETED);
- exited 0:0 at step 15000 with a FINAL line and no Traceback.

Outputs:
- `results/<dataset>-spline6-final.json` (small, here);
- `$SCRATCH/experiments/gnbm/dense_graph_2d_20261030/<dataset>/spline6/` (model, training log,
  validation rows).

## Validation results (node / edge F1 at 2 native units)

The RelationFormer `final-heldout.json` runs use **fixed 0.5 / 0.5 thresholds**, so the matched
comparison is the dense model's `validation_fixed_0p5` column. The dense model's own policy
selects thresholds on the training monitor; it is shown alongside.

| Dataset | RelationFormer B0 (mixed init) | Dense, fixed 0.5 / 0.5 | Dense, train-selected (node / edge thr.) |
|---|---|---|---|
| AV-WIDE | 0.204 / 0.012 | **0.393 / 0.045** | **0.454 / 0.133** (0.30 / 0.85) |
| REAVER | 0.745 / 0.408 | **0.784 / 0.517** | **0.806 / 0.589** (0.10 / 0.65) |
| OCTA-Synth | 0.804 / 0.595 | **0.901 / 0.758** | **0.903 / 0.782** (0.45 / 0.85) |

Other reference points:
- AV-WIDE RelationFormer with ImageNet init: 0.020 / 0.001.
- REAVER RelationFormer from scratch: 0 / 0.

### Branch / Curve F1 (K8 spline, crop-macro, same `aggregate` as RelationFormer)

| Dataset | RelationFormer B0 (mixed) | Dense, fixed 0.5 | Dense, train-selected |
|---|---|---|---|
| AV-WIDE | 0.018 / 0.098 | 0.033 / 0.108 | 0.034 / 0.224 |
| REAVER | 0.350 / 0.615 | 0.504 / 0.677 | 0.541 / 0.780 |
| OCTA-Synth | 0.350 / 0.685 | 0.711 / 0.933 | 0.708 / 0.905 |

### Node and edge F1 at other tolerances, and mAP (validation)

The dense columns use train-selected thresholds.

| Dataset | Model | @1 node / edge | @4 node / edge | node mAP | edge mAP |
|---|---|---|---|---|---|
| AV-WIDE | RF B0 mixed | 0.108 / 0.003 | 0.331 / 0.048 | 0.276 | 0.077 |
| AV-WIDE | Dense | 0.274 / 0.034 | 0.567 / 0.234 | 0.495 | 0.304 |
| REAVER | RF B0 mixed | 0.593 / 0.285 | 0.780 / 0.435 | 0.474 | 0.119 |
| REAVER | Dense | 0.709 / 0.485 | 0.839 / 0.619 | 0.640 | 0.633 |
| OCTA-Synth | RF B0 mixed | 0.506 / 0.221 | 0.886 / 0.706 | 0.805 | 0.445 |
| OCTA-Synth | Dense | 0.855 / 0.705 | 0.916 / 0.797 | 0.899 | 0.839 |

mAP is not exactly like for like: the dense model scores fixed-width GT-size boxes over its
16-NN candidates, while RelationFormer uses regressed boxes over all query pairs (README
deviation 8).

### Training monitor vs validation (dense, train-selected thresholds, node / edge @2)

| Dataset | Train monitor | Validation |
|---|---|---|
| AV-WIDE | 0.458 / 0.231 | 0.454 / 0.133 |
| REAVER | 0.982 / 0.938 | 0.806 / 0.589 |
| OCTA-Synth | 0.907 / 0.796 | 0.903 / 0.782 |

## Reading

1. **The dense model beats RelationFormer B0 on every dataset and every headline metric**,
   at matched fixed 0.5 thresholds and at its own train-selected thresholds. It does so from
   scratch, without the mixed pretraining that RelationFormer needs to train at all
   (RelationFormer from scratch scores 0 on all three datasets).
2. **AV-WIDE is still hard.**
   - Edge F1 rises from 0.012 to 0.133 (0.045 at fixed 0.5).
   - Only 15% of GT edges have both endpoints among the candidates (`gt_edge_candidate_coverage_2`
     = 0.152). Node recall@2 is 0.41, so edges are node-limited, as in BRAVA (3D).
   - The train / validation edge gap (0.231 vs 0.133) shows some overfitting even at about 87
     passes.
3. **REAVER fits training almost perfectly** (0.98 / 0.94) and generalizes to 0.81 / 0.59.
   The 120k-exposure budget is about 469 passes over 256 crops, so heavy repetition is
   expected.
   - The node threshold was selected at **0.10, the lowest grid value**, from only 16 monitor
     crops, so it is poorly determined.
   - The fixed 0.5 result (0.784 / 0.517) is the more conservative comparison and still beats
     RelationFormer.
4. **OCTA-Synth:** +0.10 node, +0.16–0.19 edge, about +0.36 Branch and +0.22–0.25 Curve F1 over
   RelationFormer. This uses about 9× fewer exposures than RelationFormer's 1.06M fine-tuning
   (which also had mixed pretraining).

## Caveats

- Single seed per arm.
- Validation only: no test data was loaded.
- REAVER has 128 validation crops and 16 monitor crops.
- Budgets differ. RelationFormer B0 used 100 epochs at batch 4: 138.5k exposures (AV-WIDE),
  25.6k (REAVER), 1.06M (OCTA-Synth), plus mixed pretraining. The dense model used 120k
  exposures from scratch.
- No subject-paired bootstrap was run for 2D. The differences above are point estimates.
- Thresholds: RelationFormer is at fixed 0.5 / 0.5. The dense model is reported at both fixed
  0.5 and monitor-selected thresholds; validation never selected anything.

## Sources

- Dense: `results/{avwide,reaver,octasynth}-spline6-final.json`.
- RelationFormer: `../native2d_mixed_20261028/runs/<ds>_mixed_B0/final-heldout.json` (+ `protocol.json`),
  `../native2d_imagenet_20261030/runs/avwide_imagenet_B0/final-heldout.json`,
  `../native2d_mixed_20261028/runs/reaver_scratch_B0/final-heldout.json`.
