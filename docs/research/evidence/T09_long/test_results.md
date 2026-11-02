# IXI held-out test: validation-selected epoch-270 model

The 300-epoch training job `113928` completed successfully (5h53m41s,
exit 0). The checkpoint was selected using **validation Branch F1**:
`best_metric_checkpoint.pt`, epoch 270, validation Branch F1 0.0067563.
Job `138721` evaluated that frozen checkpoint on **all 2,384 active IXI test
patches** (21 source patients), with the configured argmax node/edge decisions,
no test threshold tuning, and no batch-norm calibration. It completed in
1m41s (exit 0). The full machine-readable outputs are under
`$SCRATCH/experiments/gnbm/ixi_q256_extend200_20260923/evaluation/test_best_branch_f1_epoch270/`:
`summary.json`, `per-patch-metrics.csv`, `predictions.json`, and `metadata.json`.

## Dataset-level results

| Metric | Nodes | Edges | Branches (1 mm physical matching) |
|---|---:|---:|---:|
| Precision | 0.3570 | 0.1963 | 0.00298 |
| Recall | 0.4778 | 0.7003 | 0.02649 |
| F1 | 0.4086 | 0.3067 | 0.00535 |
| mAP | 0.05865 | 0.05974 | — |
| mAR | 0.17262 | 0.24242 | — |

| Count / geometry metric (mean per patch) | Ground truth | Prediction | Mean absolute error |
|---|---:|---:|---:|
| Nodes | 54.35 | 72.75 | 19.46 |
| Edges | 50.39 | 179.76 | 129.43 |
| Components β₀ | 4.73 | 2.25 | 2.81 |
| Cycles β₁ | 0.77 | 109.27 | 108.50 |

Street mover distance (SMD) was **0.02531** over the 2,365 patches where it
was defined. The other 19 patches had no predicted edges, for which this
implementation leaves SMD undefined. Edge count exceeded GT in 2,350/2,384
patches; predicted cycle count exceeded GT in 2,359/2,384. Branch F1 was zero
on 1,807/2,384 patches (75.8%). The histogram and count comparisons are in
[test_metrics.png](test_metrics.png). The full training/validation trajectory
is in [progress.png](progress.png).

Under the current default decoder decisions, the model has learned some node
placement, but edge precision, physical connectivity, and cycle counts are
poor. The predicted graphs are commonly overconnected even where nodes follow
the segmented vessels. SMD alone does
not reveal this topology failure. These results describe the predictor's
output; they do not establish a failure of the GT graph extractor.

## Paired real-patch views

These four samples were fixed **before reading test predictions**, using seed
364505, distinct source patients, at least 500 foreground voxels, and GT node
count bins [10,25), [35,55), [70,90), and [120,150). No sample was selected
by model accuracy. Each HTML contains segmentation + GT at left and
segmentation + prediction at right. Their cameras are linked: rotating,
panning, or zooming either pane updates the other; the checkbox lets you turn
this off and on. The PNGs capture the same
initial camera for quick review.

| Test patch | GT nodes / edges | Predicted nodes / edges | Interactive | Still image |
|---|---:|---:|---|---|
| `sample_000052_0015` | 21 / 18 | 27 / 51 | [HTML](test_samples/sample_000052_0015__sample_000683.html) | [PNG](test_samples/sample_000052_0015__sample_000683.png) |
| `sample_000066_0063` | 48 / 40 | 65 / 117 | [HTML](test_samples/sample_000066_0063__sample_001059.html) | [PNG](test_samples/sample_000066_0063__sample_001059.png) |
| `sample_000137_0007` | 89 / 85 | 89 / 212 | [HTML](test_samples/sample_000137_0007__sample_002143.html) | [PNG](test_samples/sample_000137_0007__sample_002143.png) |
| `sample_000104_0117` | 146 / 141 | 213 / 901 | [HTML](test_samples/sample_000104_0117__sample_001683.html) | [PNG](test_samples/sample_000104_0117__sample_001683.png) |

The linked-camera event handler and both 3D scenes were checked in the
self-contained HTML, and all four static views were inspected. A browser was
not available on the cluster login node to exercise the drag interaction, so
camera synchronization should also be checked when opening an HTML file.

## Interpretation and next decision

More epochs did not resolve the extra-edge problem. The most direct next
diagnostic is a **validation-only** sweep of edge confidence decisions and
edge topology constraints while keeping the model weights fixed, then a
visual review of the resulting graphs. The test split should not be used to
pick a threshold. Any method revised after inspecting this test result needs
an independent final holdout or an explicit exploratory-test designation for
a paper.
