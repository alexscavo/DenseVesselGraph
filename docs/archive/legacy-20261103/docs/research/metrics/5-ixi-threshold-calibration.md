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


## Completed results — 2026-10-19

Full1279574 completed in40seconds, exit0.256 train/128 validation patches, model frozen. Both selected thresholds were fixed from training before evaluating validation. No unsupported closed/parallel GT was encountered.

| Validation policy | Threshold | Precision | Recall | Positional edge F1 | β0 MAE | β1 MAE |
|---|---:|---:|---:|---:|---:|---:|
| Baseline |0.5000|0.1623|0.0477|0.0737|19.2891|1.0391|
| Train-max-F1 |0.3820|0.1226|0.2338|0.1608|2.3203|19.5234|
| Train-preserve-precision |0.4977|0.1654|0.0511|0.0780|19.0234|1.0859|

These are full-graph positional edge metrics: all predicted pairs can contribute FPs, including unmatched endpoints. They are not comparable numerically to the recovered-endpoint precision0.686 from report4 or native edge F1 from report3, whose matching protocol differs.

### Interpretation

The lower threshold recovers additional GT edges but introduces many false edges under the declared GT matching. Component count error drops while cycle count error increases strongly. Merely making the graph less fragmented does not establish correct topology. The conservative threshold gives a small validation gain and a small increase in cycle error; it is not a substantial solution. The threshold tradeoff exists on training too: train-max-F1 raises cycle MAE0.9805→16.3555.

Keep the model's production/evaluation default unchanged for now. Focus the next research step on discriminating wrong relations, particularly nearby nonadjacent candidates, and on endpoint errors contributing to false predicted edges. Threshold calibration alone does not justify adopting the F1-maximizing graph. No further jobs launched in this step.

### Inspection method and concrete examples

Inspected [the validation comparison chart](../../../../../research/evidence/ixi_threshold_calibration_20261019/overview.png), generated directly from the saved aggregate metrics. Left bars compare precision/recall/F1; right bars compare component/cycle errors under each fixed threshold. Its visible pattern is rising recall alongside sharply rising cycle error under train-max-F1.

Also inspected the per-patch graph counts for the same two fixed image examples used previously. `sample_000040_0078`: baseline TP0/FP8 becomes TP0/FP17 with train-max-F1; component error falls4→0 while cycle error rises2→7. This illustrates how matching the GT component count can coincide with incorrect extra edges. `sample_000166_0036`: TP remains0, FP rises5→6 and cycle error2→3. The conservative policy leaves these two examples unchanged. These are numerical connectivity audits using the fixed2mm matching convention; anatomical judgments still depend on the image/reference review described in earlier reports.

Tables, per-patch CSV and summary are preserved under `docs/research/evidence/ixi_threshold_calibration_20261019/`. The compressed complete scored pair graphs remain in `experiments/ixi_threshold_calibration/full-1279574/pair-graphs.json.gz` for reinspection without model inference.
