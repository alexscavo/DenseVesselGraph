# 5. IXI training-calibrated relation thresholds

## Question

Can the existing relation scorer's useful ranking produce better validation connectivity by selecting a different threshold on training data? Continue from the local-image node refinement model used in [relation isolation](4-ixi-relation-isolation.md).

## Method

Freeze model parameters, node decoding and BN statistics. Score every unordered pair of decoded nodes with the original symmetric relation head; cache full pair lists and probabilities. Reuse the exact256 training/128 validation patches from relation isolation, with disjoint subject IDs. No model updates.

Match decoded nodes to GT within2mm using maximum-cardinality positional assignment. A predicted pair is a TP only if both endpoints are matched and their GT nodes are adjacent. Every other selected pair is an FP, including unmatched endpoints. Recall denominator includes all GT edges, including those whose endpoints were not recovered. This is a full decoded-graph **positional** edge diagnostic, separate from native benchmark matching and the previous recovered-endpoint-only precision/recall.

Select thresholds from training predictions only:

1. Baseline: probability >0.5.
2. **Train-max-F1:** maximize aggregate positional edge F1.
3. **Train-preserve-precision:** maximize aggregate recall subject to precision at least the baseline training precision.

Evaluate the exact same thresholds on validation. Equal-score groups are indivisible under the strict `>` inference comparator; ties in the objective choose the highest threshold. The precision constraint is a training constraint, not a guarantee on validation. Selection is in-sample for the frozen detector; validation estimates transfer, with no test-set involvement.

## Graph/topology inspection

For each patch and threshold, construct the full decoded graph with the same predicted nodes, including isolated nodes. Compute connected components β0 using union-find and cycle rank β1=E−V+β0. Report mean absolute differences from GT β0/β1 alongside edge TP/FP/FN, precision, recall and F1. This reveals whether adding edges reduces fragmentation while creating cycles. Count errors can cancel and cannot certify anatomically correct routes; selected-edge correctness remains separately measured.

Inspect `comparison.csv`, per-patch counts and cached pair graphs together. This experiment changes connectivity selection while keeping node locations fixed. Exact probabilities and graph endpoints are preserved for subsequent examination of patches with large topology changes. Specific observed examples/results will be added after inspection.

## Implementation and checks

`scripts/calibrate_ixi_relations.py`; wrapper `cluster/jean_zay/ixi_threshold_calibration.slurm`. Tiny correctness checks passed for isolated/connected/cyclic graphs, tied probabilities, train-F1 selection and precision-constrained selection. Frozen-state hash checked across model inference. The evaluator asserts GT has no closed/parallel branches before applying simple-graph scoring.

Smoke uses four train/four validation patches. Full job is gated on smoke success. Output directories are `experiments/ixi_threshold_calibration/{smoke,full}-JOBID/` with `summary.json`, `comparison.csv`, `per-patch.csv`, `pair-graphs.json.gz` and `results.md`. No new detector checkpoints are stored.

## Submission

Smoke1279545 completed in57seconds with exit0. Full job1279574 follows with afterok dependency. Submission metadata: `experiments/ixi_threshold_calibration/submission.json`.
