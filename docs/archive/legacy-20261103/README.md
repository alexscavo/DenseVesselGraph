# Legacy research documents archived on 3 Nov 2026

These are the original research logs, reviews, plans and overviews written up to 29 Oct 2026,
moved here unchanged except for link targets (rewritten so they still resolve from the new
location). Paths below this folder mirror the repository root: `docs/research/x.md` was at
`docs/research/x.md`, and `research/x.md` was the former top-level `research/` overview folder.

Read the consolidated histories in [`docs/research/`](../../research/README.md) first; they keep
the important findings and cite these originals inline. Evidence reports under
`docs/research/evidence/` and figures under `docs/research/figures/` were not moved.

Notes:
- Status words inside these files (queued, running, pending) describe the time they were written,
  not the scheduler. `sacct` and the consolidated histories are the reference.
- Several collectors in `scripts/` still write to the pre-move paths. Re-running one would
  recreate a file at the old location rather than update the copy here.
- Non-markdown companions (csv tables, preview pngs, the registry import json, the
  Image-to-Graph-Vessel csv indexes) moved with their folders. Empty `*.lock` files from the old
  report collectors also moved with their documents; they are git-ignored.

| Archived file | Summarised in |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/SPLINE_REPRESENTATION.md](docs/SPLINE_REPRESENTATION.md) | [history_data.md](../../research/history_data.md), [history_representation.md](../../research/history_representation.md) |
| [docs/repo-graph.md](docs/repo-graph.md) | [history_data.md](../../research/history_data.md) |
| [docs/research/ALL_EXPERIMENTS_AND_RESULTS.md](docs/research/ALL_EXPERIMENTS_AND_RESULTS.md) | verbatim compilation (19 Oct) of reports 1-107 listed in its contents; no separate summary |
| [docs/research/BRAVA_experiment_plan.md](docs/research/BRAVA_experiment_plan.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/EXPERIMENT_REVIEW_20261021.md](docs/research/EXPERIMENT_REVIEW_20261021.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/EXPERIMENT_REVIEW_20261022.md](docs/research/EXPERIMENT_REVIEW_20261022.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/EXPERIMENT_REVIEW_20261023.md](docs/research/EXPERIMENT_REVIEW_20261023.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/IMPLEMENTATION_TRACKER.md](docs/research/IMPLEMENTATION_TRACKER.md) | [history_data.md](../../research/history_data.md), [history_representation.md](../../research/history_representation.md) |
| [docs/research/IXI_graph_extraction_diagnostic_roadmap.md](docs/research/IXI_graph_extraction_diagnostic_roadmap.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/PROJECT_REVIEW_20261019.md](docs/research/PROJECT_REVIEW_20261019.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/README.md](docs/research/README.md) | former index of docs/research; replaced by [docs/research/README.md](../../research/README.md) |
| [docs/research/VEDO_PRODUCTION_RUN.md](docs/research/VEDO_PRODUCTION_RUN.md) | [history_data.md](../../research/history_data.md) |
| [docs/research/continuous_representation_comparison.md](docs/research/continuous_representation_comparison.md) | [history_representation.md](../../research/history_representation.md) |
| [docs/research/fine_feature_architecture_study.md](docs/research/fine_feature_architecture_study.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/local_refinement_long_study.md](docs/research/local_refinement_long_study.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/loop_failure_mitigation.md](docs/research/loop_failure_mitigation.md) | [history_data.md](../../research/history_data.md), [history_representation.md](../../research/history_representation.md) |
| [docs/research/metrics/1-ixi-supervision-audit.md](docs/research/metrics/1-ixi-supervision-audit.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/10-ixi-node-focal-ablation.md](docs/research/metrics/10-ixi-node-focal-ablation.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/11-ixi-segmentation-input-oracle.md](docs/research/metrics/11-ixi-segmentation-input-oracle.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/2-ixi-feature-node-diagnostics.md](docs/research/metrics/2-ixi-feature-node-diagnostics.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/3-ixi-node-refinement-pilot.md](docs/research/metrics/3-ixi-node-refinement-pilot.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/4-ixi-relation-isolation.md](docs/research/metrics/4-ixi-relation-isolation.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/5-ixi-threshold-calibration.md](docs/research/metrics/5-ixi-threshold-calibration.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/6-ixi-query-attribution.md](docs/research/metrics/6-ixi-query-attribution.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/7-ixi-mechanism-batch.md](docs/research/metrics/7-ixi-mechanism-batch.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/8-ixi-relation-matching-calibration.md](docs/research/metrics/8-ixi-relation-matching-calibration.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/9-ixi-image-aware-relation-learning.md](docs/research/metrics/9-ixi-image-aware-relation-learning.md) | [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/metrics/README.md](docs/research/metrics/README.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/branch-connectivity-diagnostic.md](docs/research/metrics/branch-connectivity-diagnostic.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/core-metrics-results.md](docs/research/metrics/core-metrics-results.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/generalization-audit.md](docs/research/metrics/generalization-audit.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/implementation-status.md](docs/research/metrics/implementation-status.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/metrics_ideas.md](docs/research/metrics/metrics_ideas.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/relation-distance-pilot.md](docs/research/metrics/relation-distance-pilot.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/metrics/vessel-core-v1.md](docs/research/metrics/vessel-core-v1.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/multidataset_continuous_graph_experiments.md](docs/research/multidataset_continuous_graph_experiments.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/multiscale_pretrained_feasibility.md](docs/research/multiscale_pretrained_feasibility.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/native2d_heldout_20261028.md](docs/research/native2d_heldout_20261028.md) | [history_native2d.md](../../research/history_native2d.md) |
| [docs/research/native2d_long_20261028.md](docs/research/native2d_long_20261028.md) | [history_native2d.md](../../research/history_native2d.md) |
| [docs/research/native2d_results_20261029.md](docs/research/native2d_results_20261029.md) | [history_native2d.md](../../research/history_native2d.md) |
| [docs/research/native2d_straight_20261029.md](docs/research/native2d_straight_20261029.md) | [history_native2d.md](../../research/history_native2d.md) |
| [docs/research/node_verifier_feasibility.md](docs/research/node_verifier_feasibility.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [docs/research/pretraining/augmented-source-comparison.md](docs/research/pretraining/augmented-source-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/augmented-transfer-results/test-comparison.md](docs/research/pretraining/augmented-transfer-results/test-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/augmented-transfer-results/validation-comparison.md](docs/research/pretraining/augmented-transfer-results/validation-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/combined-sources.md](docs/research/pretraining/combined-sources.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/combined-transfer-results/test-comparison.md](docs/research/pretraining/combined-transfer-results/test-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/generalization-training.md](docs/research/pretraining/generalization-training.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/ixi-overfit.md](docs/research/pretraining/ixi-overfit.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/ixi-transfer.md](docs/research/pretraining/ixi-transfer.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/leaves-replacement.md](docs/research/pretraining/leaves-replacement.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/limited-data-botanical-transfer.md](docs/research/pretraining/limited-data-botanical-transfer.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/nonplanar-source-prototype.md](docs/research/pretraining/nonplanar-source-prototype.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/plants-recovery.md](docs/research/pretraining/plants-recovery.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/source-augmentation-training.md](docs/research/pretraining/source-augmentation-training.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/source-transfer-results/test-comparison.md](docs/research/pretraining/source-transfer-results/test-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/pretraining/source-transfer-results/validation-comparison.md](docs/research/pretraining/source-transfer-results/validation-comparison.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) (results-table folder) |
| [docs/research/pretraining/textured-source-prototype.md](docs/research/pretraining/textured-source-prototype.md) | [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [docs/research/spline_learning/README.md](docs/research/spline_learning/README.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/adaptive-degree2-scratch-comparison.md](docs/research/spline_learning/adaptive-degree2-scratch-comparison.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/catmull-rom-learning.md](docs/research/spline_learning/catmull-rom-learning.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/catmull-rom-single-cases.md](docs/research/spline_learning/catmull-rom-single-cases.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/controlled-unfreezing.md](docs/research/spline_learning/controlled-unfreezing.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/curve-informed-connectivity.md](docs/research/spline_learning/curve-informed-connectivity.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/frozen-spline-investigation.md](docs/research/spline_learning/frozen-spline-investigation.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/full-validation-comparison.md](docs/research/spline_learning/full-validation-comparison.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/ixi-brava-control-capacity.md](docs/research/spline_learning/ixi-brava-control-capacity.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md), [history_representation.md](../../research/history_representation.md) |
| [docs/research/spline_learning/ixi_patch.md](docs/research/spline_learning/ixi_patch.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/local-geometry-diagnostic.md](docs/research/spline_learning/local-geometry-diagnostic.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/local-geometry-followup.md](docs/research/spline_learning/local-geometry-followup.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/local-geometry-stability.md](docs/research/spline_learning/local-geometry-stability.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/projects-cleanup-2026-10-06.md](docs/research/spline_learning/projects-cleanup-2026-10-06.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/representation-implementation.md](docs/research/spline_learning/representation-implementation.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/residual-token-diagnostics.md](docs/research/spline_learning/residual-token-diagnostics.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/scratch-comparison-300ep.md](docs/research/spline_learning/scratch-comparison-300ep.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/scratch-comparison-results.md](docs/research/spline_learning/scratch-comparison-results.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/scratch-detached-geometry.md](docs/research/spline_learning/scratch-detached-geometry.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/scratch-extension-300.md](docs/research/spline_learning/scratch-extension-300.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/scratch-local-geometry-diagnostic.md](docs/research/spline_learning/scratch-local-geometry-diagnostic.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-10k-training.md](docs/research/spline_learning/spline-10k-training.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-approach-selection.md](docs/research/spline_learning/spline-approach-selection.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-collapse-diagnosis.md](docs/research/spline_learning/spline-collapse-diagnosis.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md), [history_relationformer_3d.md](../../research/history_relationformer_3d.md) |
| [docs/research/spline_learning/spline-learning-report.md](docs/research/spline_learning/spline-learning-report.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-learning-results.md](docs/research/spline_learning/spline-learning-results.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-staged-learning.md](docs/research/spline_learning/spline-staged-learning.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/spline-weight-pilot.md](docs/research/spline_learning/spline-weight-pilot.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/synthetic-mri-restoration.md](docs/research/spline_learning/synthetic-mri-restoration.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/synthetic-mri-test-predictions.md](docs/research/spline_learning/synthetic-mri-test-predictions.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/spline_learning/synthetic_mri_patch.md](docs/research/spline_learning/synthetic_mri_patch.md) | [history_continuous_edges.md](../../research/history_continuous_edges.md) |
| [docs/research/storage/cleanup-20261016.md](docs/research/storage/cleanup-20261016.md) | [history_data.md](../../research/history_data.md) |
| [docs/research/storage/cleanup-20261025.md](docs/research/storage/cleanup-20261025.md) | [history_data.md](../../research/history_data.md) |
| [docs/research/storage/retained-experiments.md](docs/research/storage/retained-experiments.md) | [history_data.md](../../research/history_data.md) |
| [docs/research/vascular_graph_metrics.md](docs/research/vascular_graph_metrics.md) | [history_metrics.md](../../research/history_metrics.md) |
| [docs/research/vascular_graph_representation.md](docs/research/vascular_graph_representation.md) | [history_representation.md](../../research/history_representation.md) |
| [docs/research/vascular_graph_roadmap.md](docs/research/vascular_graph_roadmap.md) | [history_representation.md](../../research/history_representation.md) |
| [research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md](research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md), [history_native2d.md](../../research/history_native2d.md) |
| [research/architecture.md](research/architecture.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md), [history_native2d.md](../../research/history_native2d.md) |
| [research/decisions.md](research/decisions.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md), [history_native2d.md](../../research/history_native2d.md), [history_pretraining_transfer.md](../../research/history_pretraining_transfer.md) |
| [research/evidence/experiment-registry/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.received.md](research/evidence/experiment-registry/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.received.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md) |
| [research/experiments.md](research/experiments.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md), [history_native2d.md](../../research/history_native2d.md) |
| [research/image-to-graph-vessel.md](research/image-to-graph-vessel.md) | [history_native2d.md](../../research/history_native2d.md) |
| [research/roadmap.md](research/roadmap.md) | [history_multidataset_brava.md](../../research/history_multidataset_brava.md), [history_native2d.md](../../research/history_native2d.md) |
