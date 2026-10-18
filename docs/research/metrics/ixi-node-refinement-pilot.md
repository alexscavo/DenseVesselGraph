# IXI higher-resolution node refinement pilot — 2026-10-18

## Question and design

Follow-up to [frozen feature/node diagnostics](ixi-feature-node-diagnostics.md): can early 32³ CNN features improve node localization compared with the existing node heads fed by transformer tokens derived from a 2³ image grid?

Three matched arms from **IXI-only delayed B-spline best checkpoint, step 28170**:

| Array task | Arm | Node heads updated | Added refinement |
|---|---|---|---|
|0|baseline|Existing class/coordinate MLPs|None|
|1|local_image|Existing heads + refinement|Early CNN image samples, node token, proposed coordinates|
|2|zero_image|Existing heads + refinement|Same architecture, image samples set to zero|

Backbone, transformer, projection, query embeddings, relation head and spline head remain frozen. All modules stay in eval mode; saved BN statistics and dropout behavior are fixed. This is a **node-head continuation pilot**, not full-network retraining. The zero-image arm controls for adding a token/coordinate-conditioned MLP; image-input weights are inactive when inputs are zero.

Same full IXI training membership, deterministic loader seed, batch size **32**, constant AdamW LR **7e-5**, **1,000 updates**. Weight decay/betas/epsilon inherited from checkpoint configuration. No augmentation for this isolation pilot; this is an explicit choice shared by all arms, not a proposed final training policy. Existing classification/node objectives and Hungarian matcher are preserved. Spline loss weight is zero because its head/inputs were detached and frozen; this avoids computing a constant geometry objective. Other existing losses remain configured; frozen token/relation parameters cannot update.

128 training / 128 disjoint-subject validation patches selected by the diagnostic's deterministic subject-round-robin rule, independent of results. Monitor at updates 0, 500, 1000; training-set evaluation at 0/1000. No test set, early stopping, or best-step cherry-picking. Compare the fixed final step, with intermediate trajectory available.

## Minimal model change

Optional `model.node_refinement` in RelationFormer; disabled by default. Checkpoint-compatible for existing configurations.

- Reuses the early `layer0.relu1` feature map, before max-pooling, 64 channels at 32³.
- Samples 27 positions on a 3×3×3 lattice around each predicted node, offsets −4, 0, +4 **input voxels** per axis. This radius is an exploratory engineering choice, not a physical-mm threshold.
- Sampling follows the actual early convolution grid: feature index `i` is centred at input voxel `2*i` for kernel7/stride2/padding3. Uses W,H,D order for `grid_sample`; node arrays are D,H,W with `index/shape` normalization.
- Detached sampling centres and frozen feature map avoid changing upstream localization through spatial-sampling gradients.
- Concatenates samples, object token and detached proposed coordinates; one 128-wide hidden layer emits three coordinate corrections and two class-logit corrections.
- Coordinate correction bounded by tanh to ±4 voxels **per axis**, then clamped to existing normalized output range [0,1]. Box-size coordinates are unchanged. Total displacement can exceed four voxels in Euclidean norm; existing coordinate heads can also move base proposals further.
- Last layer initialized to zero: exact identity for node positions and logits at initialization, verified on the real model in smoke.

This cannot guarantee recovering missing junctions or endpoints beyond the local sampling region. Good results would support access to local image evidence; bad results would not rule out a higher-resolution or multiscale transformer.

## Correctness and reporting

CPU check passed: identity initialization, trainable-head gradients, per-axis displacement bounds, zero-image invariance. Syntax and Slurm shell checks passed.

GPU smoke runs all three arms sequentially, three updates each, with initial/final graph metrics and plots. Checks finite gradients, new-head gradient presence, absent gradients in frozen parameters, unchanged frozen-state hashes, unchanged transformer tokens, and identity predictions before updates. Full array starts only after all smoke arms succeed.

Full metrics: existing node/edge F1; common curve/branch metrics; same-endpoint chords; 1/2 mm node precision/recall, endpoint-pair recovery and conditional relation recall; boundary/junction/termination localization counts. Changing node confidence affects selected edges even though relation weights stay fixed. This is **not a relation-head-learning experiment**.

Collector verifies identical initial validation metrics and training-batch hashes before emitting comparison tables. Inspect node precision and duplicates alongside recall: accepting more nodes alone is not sufficient. Attribute an image-evidence benefit only if `local_image` improves over both baseline and zero-image on validation, with training/validation behavior and geometry considered.

## Jobs and artifacts

- Smoke **1264051**, 30-minute limit.
- Pilot array **1264054**, tasks 0–2 above, `afterok:1264051`; each requests one H100 independently, 2-hour limit.
- CPU collector **1264058**, `afterok:1264054`.

Configuration: `configs/ixi_node_refinement.yaml`.

Full output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018/{baseline,local_image,zero_image}`. Includes predictions, node subtype summaries, fixed-example PNGs, scalar histories and W&B offline runs. Only updated parameters are saved as `trained-heads.pt`; reconstruct by loading the recorded parent checkpoint then overlaying these named parameters. No duplicate full detector checkpoints are stored.

Comparison: `docs/research/evidence/ixi_node_refinement_20261018/{comparison.md,comparison.csv,summary.json}`, generated after completion. Full validation metrics here use the fixed 128-patch diagnostic subset, not all IXI validation patches.

Logs: `experiments/spline_learning/logs/ixi-node-refine-*` and `ixi-node-table-*`. Submission manifest: `experiments/ixi_node_refinement/submission.json`.

Implementations: `models/node_refinement.py`, opt-in integration in `models/relationformer.py`, `scripts/train_ixi_node_refinement.py`, `scripts/collect_ixi_node_refinement.py` and two Slurm wrappers. Shared node-accounting helper now tolerates empty decoded predictions (no nearest-query distance available).


## Smoke completion

All three arms completed their three GPU updates, metrics and image export successfully. The collector passed on the smoke artifacts: batch-order hashes and initial validation scores agree exactly. Frozen parameters/buffers remained unchanged, active heads had finite gradients, and the new head began as the identity. A local-image smoke validation projection was visually inspected. These checks establish execution correctness only; three-update scores are not evidence of benefit. Compact checks: `experiments/ixi_node_refinement/smoke-checks.json`.
