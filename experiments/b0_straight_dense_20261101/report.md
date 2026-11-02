# Dense model vs RelationFormer B0 vs straight-edge RelationFormer — 3D validation (1 Nov 2026) [DONE]

Script: [`../../scripts/b0_straight_dense_20261101.py`](../../scripts/b0_straight_dense_20261101.py).
Launchers: `cluster/jean_zay/b0_straight_dense_{gpu,cpu}_20261101.slurm`.
Jobs (all COMPLETED 0:0, no Traceback): inference 1803938 (SyntheticMRI) and 1803939 (BRAVA);
scoring 1803937 (IXI), 1804050 (SyntheticMRI) and 1804051 (BRAVA). No training was run.

## Why the straight-edge arm needs no training

In the B0 recipe the spline head cannot change the detector:
- the `graph_only` route detaches the tokens and node coordinates in the spline loss
  (`scripts/controlled_spline_unfreeze.py::RoutedCriterion`);
- the local corrector samples a detached feature map (`models/local_spline_geometry.py`);
- no 3D update clips gradients, and AdamW updates each parameter separately;
- the spline head is built after every detector module, so the detector's initialization is
  unchanged;
- the RNG is reseeded at every step.

A straight-only RelationFormer trained with the B0 recipe therefore decodes B0's graph, up to
GPU nondeterminism, and draws each edge as its endpoint chord. Each B0 graph is scored twice:
- **B0**, with its K8 splines;
- **straight**, with endpoint chords (`evaluate_common_vessels` without curves).

The user chose this option over training a ~19 GPU-h straight lineage. In 2D the B0 recipe
does clip gradients globally (0.1), so the spline and detector are coupled there. Even so,
the separately trained straight OCTA-Synth model scored within 0.011 of B0 on node and edge
F1 (`GNBM/gnbm-native-2d-20261027/experiments/native2d_straight_20261029/report.md`).

## What was evaluated

| Dataset | B0 checkpoint (fixed final update) | Validation | Train monitor |
|---|---|---|---|
| SyntheticMRI | `local_refinement_long/92741/runs/B0/model-12000.pt` (source of the IXI and BRAVA runs; 120 queries) | 480 patches / 13 subjects | 256 |
| IXI | literature-scale run, `ixi_long_20261030` epoch 100 (31.2k × 32); saved predictions reused | 512 / 26 | 256 |
| BRAVA | D08 synthetic arm, epoch 300 (82.2k × 32) | 1,796 / 8 | 256 |

- **Populations:** the same as the dense model, from `experiments/multidataset_continuous/plan.json`.
- **Evaluator:** B0 decoding and scoring use `ixi_long_experiment_20261030.snapshot` unchanged.
  - Policies: fixed 0.5 / 0.5, and train-selected (node by node F1 @2 mm, then edge).
  - Comparisons are strict `>`.
- **Dense model:** `dense_graph_20261030/<dataset>/spline6/final-v2*`, decoded with its own `>=`
  rule and train-selected thresholds.
- **CIs:** subject-paired bootstrap (2000 resamples, seed 261030) on the train-selected
  contrasts.

Parity:
- Every B0 value recomputed here equals B0's own summary exactly (difference 0).
- Dense train-selected values equal `final-v2.json` exactly.
- Dense **fixed-0.5** values are quoted from `final-v2.json` rather than recomputed. Its saved
  rows were decoded at the train-selected node threshold, so the 16-NN candidates differ at
  0.5 (Curve F1 off by up to 0.008).

## Validation results (train-selected thresholds; node / edge F1 @2 mm, common-v2 Branch / Curve F1)

| Dataset | Dense two-stage | RelationFormer B0 (K8 spline) | Straight-edge RelationFormer |
|---|---|---|---|
| SyntheticMRI | **0.890 / 0.792 / 0.429 / 0.936** | 0.794 / 0.607 / 0.197 / 0.615 | 0.794 / 0.607 / 0.197 / 0.564 |
| IXI | **0.648 / 0.315 / 0.163 / 0.677** | 0.508 / 0.156 / 0.024 / 0.344 | 0.508 / 0.156 / 0.024 / 0.321 |
| BRAVA | **0.543 / 0.217 / 0.054 / 0.365** | 0.368 / 0.073 / 0.005 / 0.161 | 0.368 / 0.073 / 0.005 / 0.136 |

Thresholds (node / edge):
- dense: 0.45 / 0.85 on SyntheticMRI, 0.30 / 0.80 on IXI, 0.25 / 0.75 on BRAVA;
- B0: 0.50 / 0.75, 0.45 / 0.50 and 0.45 / 0.50.

### Paired differences (95% subject CI)

| Dataset | Contrast | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|---|
| SyntheticMRI | dense − B0 | +0.097 [+0.087, +0.106] | +0.184 [+0.167, +0.199] | +0.232 [+0.210, +0.252] | +0.321 [+0.310, +0.332] |
| SyntheticMRI | dense − straight | +0.097 [+0.087, +0.106] | +0.184 [+0.167, +0.199] | +0.232 [+0.210, +0.252] | +0.372 [+0.360, +0.385] |
| SyntheticMRI | B0 − straight | 0 | 0 | 0 | +0.051 [+0.046, +0.056] |
| IXI | dense − B0 | +0.139 [+0.125, +0.153] | +0.159 [+0.143, +0.176] | +0.138 [+0.124, +0.154] | +0.333 [+0.317, +0.347] |
| IXI | dense − straight | +0.139 [+0.125, +0.153] | +0.159 [+0.143, +0.176] | +0.138 [+0.124, +0.154] | +0.356 [+0.341, +0.369] |
| IXI | B0 − straight | 0 | 0 | 0 | +0.023 [+0.021, +0.026] |
| BRAVA | dense − B0 | +0.175 [+0.163, +0.186] | +0.144 [+0.131, +0.154] | +0.048 [+0.045, +0.050] | +0.204 [+0.173, +0.228] |
| BRAVA | dense − straight | +0.175 [+0.163, +0.186] | +0.144 [+0.131, +0.154] | +0.048 [+0.045, +0.050] | +0.228 [+0.197, +0.252] |
| BRAVA | B0 − straight | 0 | 0 | 0 | +0.024 [+0.020, +0.027] |

### Fixed 0.5 / 0.5 thresholds (validation)

| Dataset | Dense (reported) | B0 | Straight |
|---|---|---|---|
| SyntheticMRI | 0.890 / 0.782 / 0.425 / 0.949 | 0.794 / 0.597 / 0.199 / 0.623 | 0.794 / 0.597 / 0.199 / 0.569 |
| IXI | 0.534 / 0.215 / 0.110 / 0.536 | 0.493 / 0.147 / 0.022 / 0.290 | 0.493 / 0.147 / 0.022 / 0.270 |
| BRAVA | **0.130 / 0.012 / 0.005 / 0.024** | 0.356 / 0.068 / 0.006 / 0.122 | 0.356 / 0.068 / 0.006 / 0.103 |

On BRAVA at a fixed 0.5 cut-off the dense model is **worse** than B0. Its BRAVA heatmap peaks are
mostly below 0.5; its own policy selects 0.25 on training patches. The dense model needs a
calibrated threshold, whereas B0's scores happen to sit near 0.5.

### Train monitor (train-selected; node / edge / Branch / Curve)

| Dataset | Dense | B0 | Straight |
|---|---|---|---|
| SyntheticMRI | 0.898 / 0.812 / 0.449 / 0.943 | 0.800 / 0.624 / 0.210 / 0.606 | 0.800 / 0.624 / 0.210 / 0.558 |
| IXI | 0.662 / 0.362 / 0.197 / 0.703 | 0.528 / 0.169 / 0.026 / 0.347 | 0.528 / 0.169 / 0.026 / 0.322 |
| BRAVA | 0.553 / 0.229 / 0.056 / 0.392 | 0.419 / 0.094 / 0.009 / 0.192 | 0.419 / 0.094 / 0.009 / 0.163 |

### Detection AP (validation, train-selected graph)

| Dataset | Node AP @2 mm (dense / B0) | Box node mAP / mAR (dense / B0) | Box edge mAP / mAR (dense / B0) |
|---|---|---|---|
| SyntheticMRI | 0.860 / 0.787 | 0.638 / 0.701 vs 0.483 / 0.572 | 0.622 / 0.681 vs 0.474 / 0.551 |
| IXI | 0.613 / 0.446 | 0.337 / 0.409 vs 0.126 / 0.228 | 0.215 / 0.298 vs 0.066 / 0.186 |
| BRAVA | 0.502 / 0.254 | 0.220 / 0.367 vs 0.071 / 0.154 | 0.089 / 0.188 vs 0.026 / 0.088 |

Straight equals B0 on every AP (same graph and scores). Dense values are from
`../dense_graph_20261030/average-precision-validation.json` (§6.5). Box mAP is not like for
like: the dense model has no box head and uses fixed GT-size boxes, while B0 uses regressed boxes.

## Reading

1. **The dense model beats both RelationFormer variants on every dataset and every metric**
   at each model's own train-selected thresholds. Every 95% subject CI excludes zero.
   - Node F1: +0.10 to +0.18.
   - Edge F1: +0.14 to +0.18.
   - Curve F1: +0.20 to +0.37.
2. **Straight vs spline RelationFormer.** The spline head adds only Curve F1, by +0.02 to +0.05.
   Nodes, edges and Branch F1 are identical by construction. The gap to the dense model comes
   from the detector, not the edge geometry.
3. **The dense model also fits training data better.** Its train-monitor scores are 0.10 to
   0.19 higher in node and edge F1, and validation tracks training for all three models.
4. **Exception: fixed 0.5 on BRAVA.** Without threshold calibration the dense model loses to
   B0 on BRAVA (0.130 / 0.012 vs 0.356 / 0.068). Selecting thresholds on training patches is part
   of the dense model's method and should be stated as such.

## Caveats

- One training seed per model; validation only, and no test data was read.
- The straight arm is inferred, not trained. The argument above shows it equals B0's detector
  up to GPU nondeterminism. A separately trained run would also carry run-to-run noise, which
  we have not measured for any arm.
- **Budgets favour RelationFormer:**
  - SyntheticMRI B0 trained on 25k patches (78.2k + 12k updates at batch 32), against the
    dense model's 15k × 8 on 4,096 patches.
  - The IXI and BRAVA runs add SyntheticMRI pretraining and up to 82.2k × 32 updates.
- **SyntheticMRI B0 is slightly optimistic:** its stage-1 checkpoint (`fine_feature_architecture`
  `best.pt`) was selected on this SyntheticMRI validation set.
- BRAVA has only 8 validation subjects.

## Outputs

- `SyntheticMRI.json`, `IXI.json`, `BRAVA.json` (this folder): point values, train monitor,
  paired CIs, AP and parity.
- B0 predictions and summaries: `$SCRATCH/experiments/gnbm/b0_straight_dense_20261101/predictions/{SyntheticMRI,BRAVA}/`,
  and for IXI `$SCRATCH/experiments/gnbm/ixi_long_20261030/predictions/epoch-100/`.
