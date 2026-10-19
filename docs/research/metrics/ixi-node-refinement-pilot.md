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


## Completed results — reviewed 2026-10-19

All jobs exited 0: baseline 6m36s, local_image 6m40s, zero_image 6m10s; collector 2s. Same initial scores and training-batch hashes verified; frozen parameters/buffers unchanged. Results below are on the fixed 128 validation patches, one seed, final update 1000 (not the held-out test set).

| Arm | Node recall 2mm | Node precision 2mm | Native node F1 | Native edge F1 | Common curve F1 |
|---|---:|---:|---:|---:|---:|
| Baseline continuation |0.5112|0.4701|0.4857|0.1553|0.1171|
| Local image refinement |**0.5627**|**0.5171**|**0.5196**|**0.1673**|**0.1676**|
| Zero-image control |0.5115|0.4702|0.4829|0.1538|0.1164|

Native F1 uses its original benchmark matching; do not derive it directly from the separate position-only 2mm precision/recall columns.

### Interpretation

Local image evidence helps beyond simply adding an MLP: node precision and recall both improve, while the zero-image control stays near baseline. This is evidence for the proposed direction in this pilot, not multi-seed/test-set proof or proof that the 2³ grid alone caused the original problem. The intervention changes coordinate and confidence prediction jointly; their individual contributions remain unisolated.

Relative to baseline, local refinement adds 0.0339 node F1, 0.0120 edge F1, and 0.0505 curve F1. Validation curve F1 rises from 0.1169 initially to 0.1622 at 500 updates and 0.1676 at 1000. Training curve F1 rises from 0.1366 to 0.1882; improvement is not restricted to training patches.

At 2mm, recoverable endpoint pairs increase from 30.20% (baseline final) to 37.96% (local final), but conditional direct relation recall is essentially unchanged: 12.35% versus 12.48%. This supports improved localization/endpoint recovery, not improved relation-head learning (that head was frozen). Conditional populations differ between arms, so this is not a causal relation comparison.

Junction recall improves from 592/1199 (49.37%) to 698/1199 (58.22%); boundary recall from 884/1400 (63.14%) to 946/1400 (67.57%). Interior termination recall remains low: 321/916 (35.04%) versus 334/916 (36.46%). No claim that all node types are solved.

Same-endpoint straight-chord curve F1 improves from 0.1134 to 0.1641, close to the spline gain (0.1171→0.1676). Most geometric improvement therefore appears attributable to corrected endpoints, not extra learned curvature; the spline head was frozen.

### Visual inspection and evidence

Inspected the fixed sample `sample_000040_0078` baseline/local final projection panels directly. Node positions move toward some visible branches, but misplaced points, missing branches and incorrect connections remain. Projection crossings cannot establish anatomical topology. Panels copied to the evidence folder; no favorable example search was performed.

[Comparison table and CSV](../evidence/ixi_node_refinement_20261018/comparison.md), `comparison.csv`, and all-arm `summary.json` preserve results. Fixed-example images: `baseline-sample_000040_0078.png`, `local_image-sample_000040_0078.png` in the same folder. Original predictions and other fixed examples remain in the run directories.

### Next decision (not launched)

Retain local image refinement as a promising node-localization option. Next isolate relations using recovered/GT endpoint pairs, because connectivity remains poor even when endpoints are available. A controlled longer node-refinement continuation can assess stability, but should preserve both controls and validation-based decisions. Do not claim improved curvature or schedule a full training based only on this pilot.


## Image inspection: method and observed examples

Use the same fixed validation patch and final update for each arm. The six-panel renderer shows maximum-intensity projections along D/H/W: the top row overlays original GT paths and nodes in orange, and the bottom overlays decoded predicted curves/nodes in cyan. This permits direct visual comparison of displacement, omitted structures and spurious connections on the same input. Projections preserve bright structures across depth, so apparent crossings require slice-level confirmation.

For `sample_000040_0078` at update1000, the local-image variant shifts several nodes/segments toward the bright vessel compared with the baseline; isolated nodes and incorrect small connections remain. The annotated images are preserved as `baseline-sample_000040_0078.png` and `local_image-sample_000040_0078.png` beside the comparison table. Aggregate positional matching shows improved node precision/recall, while conditional relation recall stays nearly constant. Thus the image comparison and metrics support better endpoint placement while identifying connectivity as unresolved.
