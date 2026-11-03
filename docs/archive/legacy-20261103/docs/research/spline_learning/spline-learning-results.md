# Spline learning: generated run results

Run directory: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/runs/job-612351`

**Four training patches only. These are overfit diagnostics, not held-out validation results.**

| Run | Updates | Fixed spline weight | Initial curve error (mm) | Final curve error (mm) | Final endpoint error (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0_pretrained | 500 | 1.474 | 1.1218 | 0.1093 | 0.0453 |
| v0_scratch | 500 | 1.474 | 26.1840 | 0.3811 | 0.0475 |
| v1_pretrained | 500 | 1.4785 | 1.1218 | 0.1311 | 0.0466 |

Curve error means mean Euclidean deviation at corresponding spline parameter samples against the fitted target, over matched positive edges. It is not closest-centerline distance or anatomical accuracy.

V0 gate results: `{'v0_pretrained': True, 'v0_scratch': True}`. Selected V1 initialization: `pretrained`.
Gate: at least 50% reduction and final mean curve error ≤0.5 mm. This is an exploratory learning check, not a benchmark threshold.

## Interactive outputs

- [v0_pretrained: targets versus predictions](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/runs/job-612351/v0_pretrained/predictions.html). Switch matched-pair geometry and decoded-graph modes.
- [v0_scratch: targets versus predictions](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/runs/job-612351/v0_scratch/predictions.html). Switch matched-pair geometry and decoded-graph modes.
- [v1_pretrained: targets versus predictions](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/runs/job-612351/v1_pretrained/predictions.html). Switch matched-pair geometry and decoded-graph modes.

## Next steps

- Inspect decoded topology separately from matched-positive geometry.
- If V0 fails, inspect endpoint errors, target alignment and optimization; do not conclude the representation is invalid.
- If V0 succeeds, compare V1 from the same original initialization and inspect its predicted curves.
- Then evaluate held-out subjects and more tortuous branches; compare control counts and residual scaling.
- Generalize augmentation and support for loops/parallel branches before dataset-wide experiments.

## Decoded predictions and shape check

All three completed runs recovered the supplied node/edge topology on all four training patches: 98 nodes and 79 edges, with no extra or missing edges under minimum-distance node assignment. This is a diagnostic correspondence check, not benchmark F1. Decoded curves were plotted and inspected.

| Run | Learned curve point error (mm) | Chord with identical predicted endpoints (mm) | Branches improved / 79 |
| --- | ---: | ---: | ---: |
| v0_pretrained | 0.1093 | 0.4812 | 68 |
| v0_scratch | 0.3811 | 0.4803 | 50 |
| v1_pretrained | 0.1311 | 0.4821 | 68 |

The same-endpoint chord comparison isolates the shape contribution from endpoint localization. Fine-tuned V0 improves over its chord baseline by about 77%; scratch V0 improves by about 21%. Scratch still misses some bends visible in the overlays. Both can learn geometry here, with stronger fine-tuning results at this fixed update budget. V1 did not outperform V0 in this one small experiment. No conclusion about held-out subjects or the best general loss follows.

[One-file interactive comparison](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/runs/job-612351/comparison.html) starts in decoded-graph mode and switches runs and patches. Download and open locally; no internet or separate assets required.

- [v0_pretrained: static decoded overview](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/runs/job-612351/v0_pretrained/decoded_overview.png).
- [v0_scratch: static decoded overview](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/runs/job-612351/v0_scratch/decoded_overview.png).
- [v1_pretrained: static decoded overview](../../Graph-Native-Betti-Matching-Representation/experiments/spline_learning/runs/job-612351/v1_pretrained/decoded_overview.png).

The interactive comparison and individual viewers now include the existing vessel-mask mesh, enabled at 12% opacity by default. Toggle **Vessel-mask mesh** or adjust **Mesh opacity**. The surface comes from the existing segmentation, not a predicted segmentation.

## Test script on the four training patches

Slurm **613399** completed `evaluate.py` for all three saved checkpoints in 1 minute 14 seconds (exit 0:0), performing fresh MRI inference. The evaluation cache's `val` contains the same four training patches: these are overfit results, not held-out performance.

| Run | Node F1 | Edge F1 | Node mAP | Edge mAP | Standard chord SMD (normalized coordinates) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v0_pretrained | 1.00 | 1.00 | 1.00 | 1.00 | 0.0001615496 |
| v0_scratch | 1.00 | 1.00 | 1.00 | 1.00 | 0.0001617138 |
| v1_pretrained | 1.00 | 1.00 | 1.00 | 1.00 | 0.0001640488 |

Each recovered 98 nodes and 79 edges, zero detection false positives/negatives and zero Betti errors. Standard graph detection uses boxes; its SMD uses straight edges, so these metrics do not evaluate learned tortuosity. Use the separate fitted-target spline errors above for shape. Fresh decoded predictions exactly match the viewer exports (nodes, edges and spline curves), confirming that the displayed decoded curves come from the saved trained models.

Each run's `evaluation_four_patches/` contains summary, per-patch CSV, fresh prediction JSON, evaluation metadata/config and `viewer_consistency.json`. [Evaluation protocol and training recap](spline-learning-report.md).
