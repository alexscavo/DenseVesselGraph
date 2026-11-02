# Job review — 20 October 2026

Snapshot from Slurm accounting, current queue and saved artifacts. No training or evaluation was rerun. Single-seed results; subject bootstrap and final collection are incomplete.

## Image-aware relation study

All four arms completed 65,600 updates on 2,048 train / 512 validation patches. Same frozen nodes and candidate populations. Conditional AP below uses 2 mm spatial matching (75,573 validation candidate pairs, 3,774 positives).

|Scorer|Validation AP at 1,000|At 2,000|At final 65,600|Final training AP|
|---|---:|---:|---:|---:|
|original|—|—|0.4723|0.4856|
|token_geometry|0.4753|0.4766|0.3058|0.6044|
|endpoints|0.4898|0.4907|0.3113|0.6394|
|chord|0.4955|0.5016|0.3216|0.6468|
|corridor|0.4944|0.5019|0.3211|0.6445|

Early endpoint/chord/corridor adapters improve validation ranking over the original and token-only control; final adapters fit training better but generalize substantially worse. This is an exploratory image-information signal, not a confirmed graph/topology improvement. No validation-selected replacement model is declared. The 2,000-update checkpoint was predefined, but interpreting it after examining the learning curve requires confirmation/uncertainty.

At final update/fixed 0.5, endpoint image arm versus original: edge F1 0.1060 vs 0.0780 and Curve F1 0.3284 vs 0.1732, but false edges 12,397 vs 2,821 and cycle MAE 9.7832 vs 0.9941. Training-max-F1 calibrated validation edge F1 instead favors original, 0.1533 vs 0.1236. Therefore fixed-cutoff recall/Curve F1 gains do not establish improved reconstruction. Early-checkpoint full graph comparisons remain outstanding.

Three CPU jobs failed after saving ranking, per-patch/common graph metrics and summaries: original 1331643, token control 1331645_0, endpoint 1331645_1. Cause: degree histogram aggregation adds an empty Python list (NumPy float64) into an int64 accumulator. Final collector 1331646 was cancelled. Chord/corridor CPU evaluations remain active. Saved metric files are usable, but completed_evaluation in their summary predates the tail failure and is not proof of Slurm success. Repair/recovery should reuse saved tables rather than rerun inference/training.

## Other batches

- Node-loss: all three arms completed 6,900 updates; prediction array 1332029 is waiting for GPU scheduling, so no new validation comparison is available. CPU evaluation 1332030 and collector 1332031 remain dependent.
- Segmentation-input: full audit 1339156 completed, retaining 2,048 train / 511 validation patches, 123/26 subjects. GPU smoke and subsequent training await node prediction completion. No raw-versus-mask training results yet.
- BRAVA: audit 1339525 and actual graph/patch/spline smoke 1339526 completed. All 56 subjects / 111 valid SWC variants passed alignment. Snapshot: 31 subject-array tasks completed and 32 completed volume markers including smoke reuse; remaining array tasks pending. Staging, full patches and full spline targets have not yet run. No new BRAVA failure at this check.

Source image-study output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_image_relation_learning_20261020-113444`. Job records are in experiments/*/submission.json.
