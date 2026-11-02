# Three scratch trainings: completed held-out comparison

All three train from scratch for 300 epochs on the same 10,000 vessel patches, with 1,952 validation and 13,071 held-out test patches. All use 120 object queries, batch 4, the same seed and shared optimizer/schedule. The spline arm adds its head and fixed geometry loss weight 1.4739989612791453.

The source guide paths and masks remain uncertain references; agreement with them does not establish anatomical accuracy.

## Checkpoint selection

| Representation | Validation-loss-selected epoch | Final epoch |
|---|---:|---:|
| Straight edges | 35 | 300 |
| B-spline, 8 controls | 170 | 300 |
| Adaptive degree-2 | 25 | 300 |

“Best” means the lowest weighted validation graph loss (boxes/class/cardinality/nodes/edges, weights 3/4/0.8/2/6), not highest reconstruction F1. No checkpoint or threshold was selected using test scores. Final and selected checkpoints are both reported; these measurements do not justify retrospective test-based model selection.

## Shared evaluation protocol

Degree-two chains are contracted while retaining their full predicted geometry. All arms use the same target endpoint topology and full guides, physical matching at 1 mm, and curve sampling every 0.25 mm. Anchor matching is role-aware (junction, endpoint, isolate); virtual pure-cycle anchors are used only for topology. Whole-curve precision/recall/F1 include missing and extra geometry. Matched guide distance and tortuosity apply only to recovered branches and must be read with coverage. Dataset-level anchor/branch F1 are micro counts; curve precision/recall use pooled arc-length samples. Count/error entries are patch means.

## Final checkpoints, epoch 300

| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |
|---|---:|---:|---:|
| Anchor F1 ↑ | 0.135344 | 0.000084 | 0.103053 |
| Branch F1 ↑ | 0.011416 | 0.000000 | 0.005017 |
| Branch coverage ↑ | 0.940% | 0.000% | 0.491% |
| Whole-curve precision ↑ | 0.268873 | 0.003784 | 0.276281 |
| Whole-curve recall ↑ | 0.201725 | 0.000825 | 0.223516 |
| Whole-curve F1 ↑ | 0.230508 | 0.001355 | 0.247114 |
| Matched guide distance, mm ↓ | 0.67129 | N/A | 0.57151 |
| Matched tortuosity error ↓ | 0.02085 | N/A | 0.01900 |
| Absolute component-count error (β₀) ↓ | 2.58297 | 29.95004 | 3.50409 |
| Absolute loop-count error (β₁) ↓ | 2.13909 | 0.02930 | 4.35827 |
| Topological-node count error ↓ | 5.98187 | 15.63408 | 3.52903 |

## Validation-loss-selected checkpoints

| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |
|---|---:|---:|---:|
| Anchor F1 ↑ | 0.081095 | 0.000053 | 0.060244 |
| Branch F1 ↑ | 0.003307 | 0.000000 | 0.001926 |
| Branch coverage ↑ | 0.270% | 0.000% | 0.173% |
| Whole-curve precision ↑ | 0.235393 | 0.003950 | 0.222295 |
| Whole-curve recall ↑ | 0.127004 | 0.001234 | 0.113864 |
| Whole-curve F1 ↑ | 0.164989 | 0.001881 | 0.150592 |
| Matched guide distance, mm ↓ | 0.64255 | N/A | 0.56546 |
| Matched tortuosity error ↓ | 0.01830 | N/A | 0.01679 |
| Absolute component-count error (β₀) ↓ | 4.35590 | 28.95004 | 7.28085 |
| Absolute loop-count error (β₁) ↓ | 2.85923 | 0.02930 | 3.17298 |
| Topological-node count error ↓ | 5.75977 | 17.62107 | 3.28070 |

## Predicted object and geometry counts: final checkpoints

| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |
|---|---:|---:|---:|
| Raw predicted nodes | 21.34840 | 38.00000 | 28.85456 |
| Predicted degree-2 nodes | 5.85403 | 1.00000 | 9.59406 |
| Topological nodes (includes isolates) | 15.49438 | 37.00000 | 19.26050 |
| Junctions | 3.14513 | 0.00000 | 5.78930 |
| Endpoints (includes crop boundaries) | 10.11032 | 4.00000 | 11.45643 |
| Isolated nodes | 2.23893 | 33.00000 | 2.01477 |
| Contracted branches | 10.58251 | 2.00000 | 15.65267 |
| Internal spline-control points | 0.00000 | 18.00000 | 0.00000 |

Shared reference means: 21.39255 topological anchors, 5.65710 junctions, 15.70056 endpoints and 16.37189 branches per patch. Spline control points are geometry parameters, not object tokens or anatomical junctions. A predicted degree-two node is contracted regardless of the intended native node role; false degree-two structure remains visible in raw counts and native diagnostics. Count agreement alone does not imply correct connectivity.

## Native diagnostics: final checkpoints

| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |
|---|---:|---:|---:|
| Native node F1 ↑ | 0.653409 | 0.000000 | 0.567310 |
| Native edge F1 ↑ | 0.544920 | 0.004376 | 0.511229 |
| Native target nodes / patch | 21.39255 | 21.39255 | 28.41213 |
| Native target edges / patch | 16.37189 | 16.37189 | 23.39148 |

## Native diagnostics: validation-loss-selected checkpoints

| Metric | Straight edges | B-spline, 8 controls | Adaptive degree-2 |
|---|---:|---:|---:|
| Native node F1 ↑ | 0.577861 | 0.000000 | 0.489399 |
| Native edge F1 ↑ | 0.449901 | 0.005019 | 0.422493 |

Native node/edge metrics use each arm’s own targets. The adaptive arm has extra degree-two target nodes/edges; native scores are therefore diagnostics, not the representation-independent headline comparison.

## Interpretation and limitations

- All three current runs show weak physical topology recovery under the fixed 1 mm protocol. Small conditional shape errors describe very few recovered branches.

- The final adaptive model has higher whole-curve F1 (0.24711 versus 0.23051 for straight edges), while straight edges have higher anchor and Branch F1. No single representation wins every relevant quantity, and no statistical-significance claim is made.

- The current spline run is degenerate: node/branch counts are constant across all 13,071 patches and no branches are recovered. Saved prediction checks across two distinct test subjects showed identical edge indices and coordinate differences of only approximately 7.6e-6 mm. Coordinates are not bit-identical. This is a warning about this training/evaluation outcome, not proof that spline representation learning is infeasible.

- Spline guide/tortuosity errors are N/A because zero branches match, not because the geometry is perfect. Its small loop-count error is also trivial: predicting essentially no cycles does not imply correct vessels.

- Final checkpoints outperform loss-selected checkpoints on several measures. The current validation-loss selection is poorly aligned with reconstruction metrics; a future validation-only topology/geometry selection protocol must be predefined.

- This is a single-seed comparison using correlated patches from 14 test subjects and the common eligible subset. Subject-level uncertainty estimates and repeated seeds remain necessary for paper-level claims.

## Next steps

Diagnose the near-input-independent spline predictions before launching another large spline run or deciding between spline families. Check the input dependence, shared-head gradients, and scratch-versus-finetuning geometry-loss calibration. Keep these current results as the reproducible baseline. Validate any revised checkpoint-selection policy on validation subjects only, then repeat controlled runs and common held-out scoring.

## Files and reproducibility

Full-precision CSV: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/metadata/comparison/comparison.csv`. Source checksums: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/spline_learning/metadata/comparison/provenance.json`.

Underlying results: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_{old_graph,spline_k8,adaptive_degree2}_10k_300ep/test/{best,final}/`.

Regenerate from the representation repository: `python scripts/build_scratch_comparison_tables.py`. This reads existing results and changes no dataset, checkpoint or metric source files.
