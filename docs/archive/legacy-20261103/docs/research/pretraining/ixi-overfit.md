# IXI four-patch fitting diagnostic (2026-10-16)

## Question and design

Low training and validation reconstruction scores suggest the real-data difficulty is not explained by generalization alone. Test whether the existing detector can fit four real training patches. Two arms (straight and detached residual K8 B-spline) initialize the detector from the same completed `pretrain_ixi_straight/best.pt`. Geometry is initialized fresh with the fixed seed. This is a pretrained fitting-capacity diagnostic, not a demonstration of training from scratch.

Four training patches and four validation patches come from the existing fixed monitoring manifests, taking the first patch of each of four distinct subjects in each split. No test samples and no selection by visual quality or model score. Exact IDs are in `experiments/ixi_overfit/plan.json`. The tiny validation group is descriptive, not a population-level generalization estimate.

- 1,000 optimizer updates on the same four-patch batch, no early stopping or augmentation.
- Same architecture, graph matcher, graph losses, spline weight and residual objective as IXI transfer. Domain adaptation disabled for target-only fitting.
- Full detector trained through graph loss; geometry gradients remain detached from shared detector features/endpoints as in the main runs.
- Batch 4, AdamW LR 7e-5, existing polynomial scheduler over 1,000 updates, 10-update warmup. Training-mode BN during optimizer updates; saved statistics during evaluation.
- Every 100 updates: train and validation losses computed with the same criterion in evaluation mode and no augmentation, plus decoded graph/curve metrics and same-endpoint chord metrics. W&B offline tracking retained. No extra losses.
- Initial/final reference/prediction overlays for ALL four fitting patches; existing monitor projections also retained. Original extracted guides remain uncertain references.

A decreasing training loss alone does not establish success: require improved decoded node/edge and curve recovery, verified in the renders. Successful fitting with weak validation suggests a data coverage/generalization problem. Failure to fit directs inspection toward input/reference consistency or optimization. A straight/spline discrepancy implicates geometry integration; both failing would indicate a broader detection problem. No automatic longer run or rendering ablation is authorized by this diagnostic.

## Source embedding proposal

Current `PlantsDataset` repeats the 2D image/mask across five adjacent slices, centered in an otherwise empty cube, then applies rotations/flips and padding. It already has thickness, but remains a constant-thickness planar extrusion. Training input is `image`; altering image noise would affect what the model receives.

A future controlled rendering ablation could replace slab extrusion with a vessel-shaped volumetric cross-section around the existing centerline, preserving graph targets, and add background noise throughout the cube, including background pixels in the original image plane. Keep segmentation and graph targets noise-free. Use target TRAINING images only to choose contrast/noise ranges; arbitrary high noise could instead conceal thin branches. Intensity blending should avoid a visible rectangular plane boundary. Thickness alone does not create genuinely nonplanar branching geometry. Any deformation must transform the graph and spline targets consistently.

This is a hypothesis about source/target appearance mismatch, not an established explanation of poor IXI results. Do not change the active comparisons. Actual unaugmented loader renderings and four IXI reference audits are saved under `ixi-overfit-previews/`.

## Execution

- Paired GPU smoke array: **1164010** (two updates per arm).
- Paired 1,000-update diagnostic: **1164011**, corresponding-task smoke-success dependency.
- Node jzxh163 excluded because an earlier job exhausted its temporary disk.
- Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_overfit_20261016/{straight,spline}/`.
- Initial/final predictions: each arm's `previews/`; losses/metrics: `history.jsonl`; checkpoint: `final.pt`.
- Code: `scripts/overfit_ixi_diagnostic.py`; reproduction: `cluster/jean_zay/submit_ixi_overfit.py`.

Static syntax checks passed. CPU rendering first encountered the already-known incomplete SimpleITK in the mutable environment; it is being run using the fixed package archive also used by GPU jobs. GPU results pending at submission.

## Visual inspection and startup correction

All four IXI reference overlays and the actual unaugmented plant/leaf loader outputs were rendered and inspected. Source side projections show five-voxel slabs in empty backgrounds; the full source graph is visible only along the plane normal. IXI inputs have textured tissue and much less uniform contrast. In `sample_000107_0017`, the reference has a short segment near the crop boundary while a longer bright vessel-like structure is visible. This is a candidate annotation/coverage issue, not proof from a projection alone. Other selected patches include both brighter labelled branches and faint branches; projections alone cannot establish 3D crossings/connectivity. Keep all four fixed cases instead of selecting replacements based on appearance.

The initial pretrained spline prediction was also inspected for `sample_000100_0033`: decoded paths are sparse and fail to follow much of the labelled network. This is before overfitting, not a diagnostic outcome.

Initial two-update GPU smoke tests 1164010 passed for both arms. Production launch 1164011 failed before updates because the new caller omitted `build_tracker(..., resume=False)`. Fixed the required argument and now exercise W&B during smoke too. Preserved initial directories under `initial-attempt-*`; no learned checkpoint overwritten. Replacement smoke array **1164572**, paired overfit array **1164573** (success-gated). Recovery commands/IDs are recorded in `experiments/ixi_overfit/recovery.json`.

CPU previews succeeded with the fixed runtime after restricting source discovery to the two displayed records; no need to scan/load all source patches. See [source embedding](ixi-overfit-previews/source-embedding.png) and the four reference PNGs in the same folder. The input-audit panels are explicitly labelled image/mask-only, not predictions.

## Completed diagnostic — 2026-10-16

Both replacement smoke tests and both 1,000-update jobs completed successfully (straight 2m55s, spline 3m30s). These are four fitting patches and four disjoint monitoring patches, with pretrained initialization.

| Arm | Train total loss, initial → final | Val total loss, initial → final | Final train node F1 | Final train edge F1 | Train curve F1 | Same-endpoint chord F1 | Final val edge F1 |
|---|---|---|---:|---:|---:|---:|---:|
|Straight|2.4292 → 0.0292|3.4769 → 8.4953|1.0000|1.0000|0.5952|0.5952|0.0988|
|Spline|2.8199 → 0.1122|3.9079 → 9.4349|0.9688|0.9000|0.8959|0.5323|0.1370|

Loss magnitudes across arms include different objectives and should not be interpreted as a direct ranking. Within each arm, train loss falls while validation loss rises: clear memorization/overfitting on this intentionally tiny set. The pretrained network can fit these real inputs and reference graphs; the spline can recover curved geometry much better than same-endpoint chords. This rules out a blanket inability to fit these examples, not dataset-wide annotation problems, scratch-training difficulties, or optimization issues on larger populations. It does not establish that the full-data runs' weakness has exactly the same cause.

Final straight and spline prediction renders for `sample_000100_0033` were directly inspected: straight edges cut across curved reference paths; learned splines follow them substantially better, with some residual graph errors. Full initial/final render sets are in each run's `previews/` folder. Validation scores on four examples are illustrative only.
