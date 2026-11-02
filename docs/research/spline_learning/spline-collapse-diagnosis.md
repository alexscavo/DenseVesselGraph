# Scratch B-spline collapse diagnosis — 2026-10-08

## Scope and established result

Diagnose the completed three-arm, 300-epoch SyntheticMRI comparison before
changing the representation or launching another long run. The final models
have common whole-curve F1 of 0.23051 (straight), 0.00135 (B-spline), and
0.24711 (adaptive); see [full comparison](scratch-comparison-results.md).

The spline run failed to learn the training population. At epoch 300 its
training/validation node losses are 0.20171/0.20324, versus 0.00731/0.07341
for straight edges and 0.00758/0.08366 for adaptive. Spline node loss remains
around 0.21–0.24 from early training onward. This predates the epoch-171
filesystem interruption and resume. Both best and final checkpoints fail.
This differs from the baselines' large training/validation gap.

## Checks performed

1. Replayed saved predictions for the first two test patches. Spline graphs
   have identical edge indices and 38 nodes, with maximum normalized node
   coordinate change 1.79e-7. Both baselines change substantially. This is
   input-independent collapse, rather than a subtle curve-fitting error.
2. GPU job **762225**, eight fixed validation patches, batch size four,
   all three final checkpoints: compared saved BN running statistics with
   current batch statistics, keeping dropout and the rest of the model in
   evaluation mode. No weights/checkpoints were saved or changed.
3. One initial training batch: measured weighted spline versus ordinary graph
   gradient norms at shared tokens, node coordinates, and image features.
   No optimization step was performed.
4. GPU job **762449**, four fixed validation patches, spline and straight final
   checkpoints: recorded transformer attention support and feature dependence.
   For the spline model only, zeroed cross-attention offsets in disposable
   memory to sample inside the feature grid. This was an intervention to
   localize the failure, not a new evaluated model or a checkpoint repair.

| Model | Node loss, saved BN | Node loss, batch BN | Largest between-input node change, saved / batch BN |
| --- | ---: | ---: | ---: |
| Straight | 0.06868 | 0.07287 | 0.7461 / 0.8065 |
| B-spline | 0.18983 | 0.18983 | 1.79e-7 / 1.79e-7 |
| Adaptive | 0.08500 | 0.08566 | 0.7852 / 0.7802 |

All checkpoints load compatibly with no missing active tensors. Each has 89
BN layers and 750,000 recorded BN updates. Input differences are identical
across arms. The spline encoder still produces different image features,
but decoder token differences are only about 7e-8. BN mode does not restore
input sensitivity. These checks strongly disfavor stale inference BN
statistics as the explanation; they do not prove that normalization never
affected optimization during training.

### Transformer localization

For the first pair of validation patches, the spline model's projected CNN
feature mean absolute difference is **0.85446**. Across transformer encoder
layers, it falls to 0.00424, 0.000427, 1.72e-5, 3.12e-7, 4.23e-8, and finally
**3.42e-8** at encoder output. The straight model preserves a difference of
**0.87522** at encoder output. Thus the CNN remains input-sensitive, while
the spline model's transformer encoder nearly removes that dependence.

Most deformable-attention samples lack interpolation support. Spline decoder
cross-attention supported weight is approximately 1e-7, 0, 0, and 0.07436
across its four layers. However, the straight baseline also has poor sampling
coverage, including zero supported encoder-attention samples on this batch.
Its residual/FFN paths still preserve image dependence. Coverage alone is
therefore insufficient to identify the cause of the spline-specific collapse.

Zeroing spline cross-attention offsets restores supported weight to 1 in
every decoder layer but leaves node differences at **1.19e-7**. The incoming
transformer memory has already collapsed. This intervention does not repair
the prediction, and no modified weights were saved. No further diagnostic
tests are needed before choosing a controlled short optimization experiment.

The initial weighted spline/graph gradient ratios were **1.35 at tokens,
7.49 at node coordinates, and 1.24 at image features**. The coordinate
gradient cosine was +0.779, so this batch demonstrates scale imbalance,
not directly opposing coordinate gradients. It is one diagnostic batch,
not a population estimate or proof of the initiating cause.

## Likely optimization issue

The spline loss uses millimetres and backpropagates through predicted endpoint
coordinates as well as shared features. Early Hungarian assignments and
endpoints are inaccurate; much of this loss initially concerns localization,
not tortuosity. The fixed weight 1.474 was calibrated at shared tokens using
the original four training patches (both scratch and pretrained
initializations, in evaluation mode). That calibration did not constrain
the coordinate-gradient ratio during full-population training.

Four-patch overfitting demonstrated that the parameterization is learnable.
It does not establish that the same optimization settings work across
10,000 varied patches. Identifying the exact initiating cause would require
a controlled short training comparison; no causal claim from loss magnitude
alone is justified.

## Proposed next experiment

Use fixed training membership, seed, and disjoint validation. Keep the
existing held-out test set untouched during solution selection:

- First compare the current spline weight (1.474) with a tenfold smaller
  weight (0.1474) on a modest diverse training subset. Change only this
  variable; the smaller value is a diagnostic candidate, not an established
  optimum. Monitor graph learning and transformer input sensitivity alongside
  curve error. If both settings work on the small subset, expand the training
  population before drawing conclusions about the original collapse.
- If necessary, next test graph-learning warmup followed by a gradual
  spline-weight ramp. Calibrate on several training batches in training mode,
  checking coordinate and shared-feature gradients separately.
- If endpoint coupling remains unstable, compare temporarily detaching
  predicted endpoints from the spline loss while ordinary node losses still
  train them; retain predicted endpoints at inference.
- Gate a long run on improving training graph loss, input-sensitive decoded
  outputs, and validation topology/geometry. A short successful import or
  finite-gradient smoke alone cannot detect this failure.

These are proposals; the training objective and completed benchmark results
have not been changed. No new full training or dataset-wide test was launched.

## Artifacts

Reproducer: `Graph-Native-Betti-Matching-Representation/scripts/diagnose_spline_collapse.py`.
Raw BN/gradient measurements:
`Graph-Native-Betti-Matching-Representation/experiments/spline_learning/metadata/diagnosis/bn-differential.json`.
Attention measurements: the adjacent `attention-differential.json`.

[One actual epoch-300 prediction](../experiments/synthetic_mri_scratch_spline_prediction/results/sample_000020_0470.html)
and [its static projections](../experiments/synthetic_mri_scratch_spline_prediction/results/preview.png).
Download the HTML alone and open it locally. This fixed sample contains 38
predicted nodes/3 edges versus 22 reference nodes/17 branches. It was also
used in earlier viewers and was not selected for prediction quality.

Training/native spline diagnostics use eight-control fitted targets. Common
cross-representation metrics use the original full generator guide polylines
and shared endpoint topology. Both guides and segmentation are uncertain
references. All six original benchmark tests used saved BN statistics with
zero recalibration batches; the diagnostic above is separate.

<!-- BEGIN CHEAP DIAGNOSTICS bn-protocol 20261026 -->
## BN provenance audit and training-only comparison protocol —26 October2026

**Audit completed; new training-only comparison pending the isolated inference jobs.** Worktree `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-cheap-diagnostics-20261026`, branch `research/cheap-diagnostics-20261026`. No shared model/loader/evaluator code or other agents’ checkpoints/reports were modified. This section concerns two compatible stable references, not the active BRAVA or IXI node-refinement arms.

The existing `training/evaluation/evaluator.py::calibrate_batch_norm` switches the model to training mode, disables dropout, forwards inputs without gradients, and returns to eval mode. It retains saved running statistics/momenta rather than resetting them. The generic `evaluate.py --bn-calibration-batches` path **chooses the validation loader**; that is evaluation-domain adaptation and is not a training-only protocol. The legacy sibling evaluator similarly calibrated on validation images. Neither path is silently relabeled here.

The archived August job logs show actual BN-calibration progress in **30 of32 evaluation logs**, generally requested100 batches but consumed at most the32-batch loader. This includes baseline697088, plants695050, HALVS695049, roads708435 and multiple SupCL/weight variants. Exact log member names, observed progress, saved-result markers and SHA256 are in isolated `experiments/cheap_diagnostics_20261026/bn-provenance-audit.json`. Thus those historical metrics must be described as BN-adapted evaluations, not frozen zero-shot references. Names/options alone were not used as completion evidence.

In contrast, inspected modern multiscale, fine-feature and longer-refinement resolved plans have `bn_calibration_batches=0`; they train BN normally and evaluate saved statistics. The earlier collapse job762225 compared saved versus current **batch statistics**, and BRAVA tiny-set train/eval monitors tested BN mode: neither was post-hoc training-only recalibration. This clarifies experimental provenance without revising their measured results.

New paired diagnostic: original IXI delayed-spline checkpoint SHA256 `1cd9caf35a3683ef643c541ea4c4d06b247c490a4215d4a250f4519992d68eb3`, and SyntheticMRI vascular K8 SHA256 `06513f58b7b032c9eb4f7be7820312941e7bac08fb97915f263d48f1b894facb`. Reuse the exact multi-dataset256 training/128 validation monitor IDs and prior source/split audit. IXI has123/26 train/validation subjects; SyntheticMRI64/13 volumes. Calibration uses all256 training patches, no augmentation, batch8 for exactly32 forwards. Model weights have gradients disabled, optimizer absent, GT not passed to the network, and original node/relation cutoffs0.5 retained.

Saved-BN raw-query predictions already exist: reuse them, with fresh three-patch coordinate/confidence/relation score and pair-order parity checks before proceeding. Validation inference is consistently batch1 for recalibrated predictions. Check parameter and non-BN-buffer checksums; only running means/variances/counters may change. Save isolated BN-buffer tensors and compressed numeric predictions under `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/cheap_diagnostics_20261026`, never over an original checkpoint.

Report physical1/2-mm raw/filtered recall, localization/Hungarian distances, node/edge F1, duplicate counts, mean-patch conditional relation AP/recall, and component/cycle errors, with2,000-draw paired subject intervals. No test data, threshold tuning, training or new curve metric. A node gain is not sufficient if connectivity degrades. The two30-minute GPU tasks are serialized(%1) and submitted behind all currently queued same-user GPU work; a CPU collector appends results here automatically if this file remains unchanged, otherwise stages a JSON integration block rather than overwriting a concurrent edit.

Plan, data/checkpoint identities, job records, tests and provenance: isolated worktree `experiments/cheap_diagnostics_20261026/`. Three focused tests passed: strict tied-score threshold semantics, bootstrap AP versus sklearn, and the existing BN-update/weight-invariance test. The real cache-parity check gates the inference diagnostic. Detailed BN conclusions remain pending; no improvement is claimed at submission.
<!-- END CHEAP DIAGNOSTICS bn-protocol 20261026 -->

<!-- BEGIN CHEAP DIAGNOSTICS bn-jobs 20261026 -->
## Accepted training-only BN diagnostic jobs —26 October2026

- GPU inference array **1584593_0–1**, serialized `%1`: IXI then SyntheticMRI, 30-minute ceiling each; no optimizer/training. Dependency `afterany:1583591:1583592:1583593` waits for the currently active/queued IXI GPU work. The latest submission audit found no remaining BRAVA GPU allocations; none were modified.
- CPU paired collector **1584594**, `afterok:1584593`, 30-minute ceiling. It appends results to this existing report only if its audited hash still agrees, otherwise leaves an explicit JSON integration block.
- Maximum additional GPU concurrency: one H100; all predecessors are retained. Both submission IDs were accepted. Scientific BN results are pending.

Retrieve saved relation results, BN status/results and deferred integration blocks from the parent `projects` folder:

```bash
python3 gnbm-cheap-diagnostics-20261026/scripts/show_cheap_diagnostics.py
```

No new relation training, model architecture, loss, test-set evaluation or representation experiment was launched.
<!-- END CHEAP DIAGNOSTICS bn-jobs 20261026 -->

<!-- BEGIN Q0 BN PARTIAL 20261028 -->
## Q0: verified partial training-only BN result — 28 October 2026

**L06 remains PARTIAL / RECOVERY.** Live/accounting/artifact audit confirms IXI **1584593_0 FAILED** after49s; SyntheticMRI **1584593_1 COMPLETED** after1m37s; paired collector **1584594 CANCELLED**. No later completed IXI recovery was found. IXI failed in saved-cache pair accounting: an empty eligibility list became a float array, which NumPy cannot use as a Boolean index. There is no valid IXI before/after result; this is a diagnostic implementation failure, not evidence against recalibration.

The SyntheticMRI half is complete:256 training-only calibration patches, batch8 ×32 forwards, no augmentation;128 validation patches at batch1; fixed node/edge cutoffs0.5. Archived artifacts verify three-case baseline-cache parity and unchanged parameters/non-BN buffers. Existing checkpoint: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/plant_generalization_20261014_da_b32_w1_restart/runs/vessels_spline/best.pt`, SHA256 `06513f58b7b032c9eb4f7be7820312941e7bac08fb97915f263d48f1b894facb` (file hash rechecked in Q0).

| Tolerance | Node F1 saved → recalibrated | Edge F1 saved → recalibrated | Raw recall saved → recalibrated |
|---|---:|---:|---:|
| 1.0 mm | 0.423984 → 0.430084 | 0.133365 → 0.133493 | 0.536792 → 0.541575 |
| 2.0 mm | 0.752528 → 0.756058 | 0.445495 → 0.439856 | 0.870493 → 0.870125 |

At2mm, TP edges fall942→916 while predicted edges fall2166→2102; the small node-F1 gain accompanies lower edge F1. These are descriptive saved measurements; Q0 computes no new significance claim and does not adopt recalibration. Mean-patch conditional AP is a separate statistic, not pooled relation AP. The intended paired two-domain collection is incomplete.

IXI reference checkpoint remains `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/ixi_delayed_spline/best.pt`, SHA256 `1cd9caf35a3683ef643c541ea4c4d06b247c490a4215d4a250f4519992d68eb3` (also rechecked). Reuse the completed SyntheticMRI half if recovering IXI later; do not repeat it. Q0's latest executable scope forbids GPU submission/training, so it records the narrow empty-mask correction needed for a future IXI-only diagnostic recovery without launching it.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-q0-ixi-audit-20261028/experiments/q0_ixi_audit_20261028/bn-partial.json`; original completed artifacts remain under `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/cheap_diagnostics_20261026/SyntheticMRI/`. Earlier historical validation-image BN adaptation and old collapse results remain separate and unchanged.
<!-- END Q0 BN PARTIAL 20261028 -->
