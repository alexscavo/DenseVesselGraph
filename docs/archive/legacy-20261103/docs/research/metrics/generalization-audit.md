# Generalization diagnostics — 2026-10-14

Authorized: compare earlier/best vs final saved predictions and audit train/validation inputs. No new model inference or changes to datasets, matching or model thresholds.

## Jobs and protocol

-1049134: completed,1minute. Existing branch-relations-v1 on both best checkpoints, same1952validation IDs andGT quartiles as final checkpoints. Straight bestepoch255; spline bestepoch170. CPU rescoring; `--checkpoints best` added to existing script.
-1049135: completed,1m17s.512deterministically sampled patches/split, seed364505, full10k train membership and1952val. Compare intensity, foreground fraction, graph counts, boundary fraction, physical branch lengths/tortuosity, coordinate range, node-to-mask distance, source membership and sampled identical-image hashes.
-1049189: completed,1m09s; repeats the short split audit with orientation-invariant metric scale check. The first version compared signed source affine against identity patch header, incorrectly naming orientation differences scale errors. Source metricdiag(-64,-64,64) vs patchdiag(64,64,64) preserves distances. Original output retained; v2 is authoritative.

Outputs under `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012/`: `branch_relations_best_v1/summary.json`, `split_audit_v2/{summary.json,per-patch.csv}`. Existing final diagnostic remains `branch_relations_v1/summary.json`.

## Results

| Model | Node recall1mm best→final | Node recall2mm best→final | Long conditional relation recall2mm best→final |
|---|---|---|---|
| Straight |16.86%→17.70%|42.50%→43.03%|37.42%→38.78%|
| Detached B-spline |17.96%→18.75%|44.54%→45.27%|42.73%→42.84%|

The severe gap is already present at the earlier selected checkpoints. These results do not support late localization collapse as the main explanation. They do not rule out overfitting established before epoch170, nor establish what early checkpoints would show. “Best” was selected by the previous metric, not this diagnostic.

512patch descriptive means:

| Statistic | Train | Validation |
|---|---:|---:|
| Raw intensity mean |.33268|.33289|
| Raw intensity std |.28841|.28865|
| Foreground fraction |.01275|.01301|
| Nodes |21.49|21.67|
| Edges |16.43|16.63|
| Boundary-node fraction |.50582|.50085|
| Mean branch length mm |13.011|12.855|
| Tortuosity |1.02376|1.02403|
| Approximate node-to-mask distance mm |.02543|.02495|

No overlapping patch/source IDs under the filename-derived source rule, and no identical images across the sampled subsets. These are not exhaustive leakage checks. The coarse statistics show no large shift; finer anatomical/texture correlations and subject-specific memorization remain possible. Original full-path guides are uncertain.

The loader applies the same fixed intensity subtraction and normalized coordinates to both splits. Spline loading explicitly prevents augmentation until geometry transforms are implemented. This is a concrete missing regularization mechanism, not proof of causality. See [plant recovery and next experiment](../pretraining/plants-recovery.md).

Small correctness checks:2cases/model best-checkpoint rescoring and2patches/split audit passed before submission. No test-set evaluation or hypothesis selection on test data.

Corrected v2 scale error is zero in all sampled patches in both splits. Compact summaries and CSV are preserved in `experiments/spline_learning/metadata/metrics/generalization-audit/`.
