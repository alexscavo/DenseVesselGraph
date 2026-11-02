# 3. IXI higher-resolution node refinement pilot — 2026-10-18

## Question and design

Follow-up to [frozen feature/node diagnostics](2-ixi-feature-node-diagnostics.md): can early 32³ CNN features improve node localization compared with the existing node heads fed by transformer tokens derived from a 2³ image grid?

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


## Fully unfrozen IXI component/transfer study —26 October2026

**Status: submitted; scientific results pending.** This dated extension preserves the October18 node-head-only pilot above. It follows the completed [SyntheticMRI 12,000-update/two-seed local refinement study](../local_refinement_long_study.md), which supports coordinate refinement while showing no clear added confidence benefit. The later KNN [node verifier](../node_verifier_feasibility.md) failed its gate; **A-cls here is the existing 27-sample early-CNN module, not that verifier**.

### Isolation, data and spatial contracts

Work is isolated on branch `research/ixi-local-components-20261026` in `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph`. The original working tree was already heavily modified; required working sources were copied into this worktree with SHA256 provenance, without modifying shared models, configs or canonical research files. This report is the only intended shared scientific-document update; the collector checks for concurrent changes before replacing its own bounded results section.

Use the existing IXI `transfer_splines_k8_20261015` population and exact multi-dataset4096/512 manifest: **4,096 training patches/123 subjects, 512 validation patches/26 subjects**. Train/validation subjects remain disjoint, test data untouched. The full prepared population remains10,000/2,649; no new extraction, sampling representation or labels are generated. K8 endpoint-constrained cubic B-splines retain original dense paths and boundary provenance; graph anchors/connectivity are unchanged.

Prepared64³ MRA patches use `legacy_mad_clip_v1` and the existing loader's0.33335259556770325 centering. Physical positions use the per-patch `metric` matrix, not patch headers or an assumed1-mm voxel. The GPU gate checks one cache per represented subject, reports voxel-axis spacing ranges as `norm(metric columns)/64`, verifies target endpoints, and checks1-versus4 inference-batch drift. Graph and spline targets receive the existing quarter-turn augmentation; Gaussian noise probability0.35/std≤0.015 is shared. Zoom is disabled. No segmentation input is used.

The local head samples `layer0.relu1` (64 channels,32³) at27 offsets on a3³ lattice, radius±4 **input voxels** per axis. Early-convolution feature center `i` corresponds to input voxel `2*i`; D/H/W nodes normalize by axis size, and sampling converts to W/H/D with `align_corners=False`. Physical radius therefore varies with native spacing; it is not a4-mm radius. Sampling centers stay detached, but sampled CNN features receive gradients in the fully unfrozen fits. Coordinates are bounded per axis/clamped; sizes remain unchanged.

### Zero-shot and matched fine-tuning matrix

Zero-shot evaluates the matched SyntheticMRI seed364505 final-update12000 checkpoints **B0, A-xyz and A**, preserving their native120 object queries. Report Stage1 versus Stage2 predictions within each refined checkpoint, plus zeroed/spatially shifted image-feature interventions. These checkpoints have co-adapted shared weights: their native comparison does not isolate head transfer. No equivalent A-cls source checkpoint exists, so no classifier-only pretrained result is invented.

For IXI fine-tuning, all arms instead use **exactly the same B0 source trunk/heads** from `experiments/local_refinement_long/364505/runs/B0/model-12000.pt` in the original repository. New object-query rows are identically seeded in every arm to expand120→256 (IXI target max in the source audit153); learned120 object rows and the two relation rows are preserved in their correct positions. This shared expansion changes capacity relative to native zero-shot and is reported separately.

| Arm | Added local head / initialization | Question |
|---|---|---|
| B0 | None; common B0 checkpoint | Matched IXI baseline |
| A-xyz-fresh | Fresh128-wide/3-output head, zero residual | Coordinate-refinement benefit |
| A-xyz-transfer | Only the learned A-xyz local-head parameters transplanted into the same common B0 checkpoint | Head transfer versus fresh learning, without changing the shared initializer |
| A-cls | Fresh original5-output head, zero residual; return Stage1 coordinates exactly | Local-CNN classification contribution with a matched identity initialization |
| A | Fresh original5-output coordinate+classification head | Combined effect/interactions |
| A-control | Fresh5-output head with zero image-feature slots | Added token/coordinate head versus actual local image evidence |

The three unused coordinate output rows in A-cls receive no coordinate correction gradient. A-xyz preserves Stage1 logits exactly at inference. All active backbone, transformer, query, node, relation and K8 geometry parameters train; the already-dormant legacy relation-attention sub-block stays inactive, consistently with the source recipe. End-to-end arms can co-adapt upstream predictions: “coordinate-only”/“classification-only” describes the **direct local-head output**, not frozen upstream coordinates/confidence.

Losses remain the source's node softmax focal gamma2 with weights[1,1], matched-query edge **CE**, weights[1,1], ratio upsampling0.15; Hungarian class/node cost2/5 and existing coordinate/box/cardinality weights. K8 residual-control loss and established `graph_only` gradient route are retained (weight1.4739989612791453). No edge focal, unmatched mining, multiscale attention, verifier, representation change or added detector loss.

### Budget, metrics and decision rules

One controlled fine-tuning seed261026; batch32, AdamW inherited betas/epsilon/weight decay1e-4, shared clip5. Warmup100 updates to7e-6, polynomial power0.9 to1e-6 at the common final step. **Cap6,000 updates**, checkpoint at1,500/3,000/final (6,000 if throughput permits); the real batch32 smoke benchmarks all arms and chooses the largest shared500-update multiple fitting a conservative3,600-second training estimate. A budget below1,500 aborts downstream work. The scheduler horizon follows the chosen budget. It is a bounded learning-curve pilot, not a convergence claim.

GPU gate tests exact shared initial weights, fresh-head identity, coordinate/classification routing, finite gradients across active detector components, local-image gradients, physical-coordinate/target consistency, and a tiny cached-evaluation plus spline-endpoint collection path. It reuses existing evaluation utilities and records source hashes. Full image/target hashes must match across every training arm.

Primary result: common final update on512 validation patches, not a selected checkpoint. Secondary checkpoint selection uses the same128 validation-monitor patches at each saved step; it does not substitute for matched-final inference. Training comparisons use128 fixed monitoring patches. Raw-query analysis uses batch1 consistently; common-v2/native reconstruction retains the established batch4 and separately reports measured numerical drift.

Evaluate physical1/2-mm localization, raw/filtered coverage, node precision/recall/F1, duplicate fraction, confidence canonical AP/correct–duplicate AUROC/reliable calibration, Hungarian distances, conditional relation AP/recall, edge precision/recall/F1, components/cycles, native metrics and common-v2 Branch/Curve F1. Use paired2,000-draw subject intervals, with explicit single-seed limitations. Full caches permit equal per-patch **B0 predicted-count** comparisons; no GT count is supplied. Operating thresholds/precision targets0.5/0.7/0.8/0.9 and max-node-F1 selection use training monitoring data only, applied unchanged to the128 validation-monitor population. PR curves are saved separately from deployable calibration.

Final local-head interventions keep raw relation probabilities unchanged; check query-pair correspondence, coordinate/logit routing, image-feature dependence, valid graph indices, and spline/curve endpoints matching refined nodes. A calibrated recall gain bought by duplicates/false connections/cycles does not establish useful reconstruction. Coordinate improvements can change confidence-label categories; do not interpret AP/AUC differences alone as causal classification improvement.

Choose a supported arm using default and count-controlled graph/localization gains, subject intervals, Curve/Branch and topology, then distinguish fresh versus transferred xyz. The collector reports these comparisons and trajectories; no extension is submitted automatically. Longer training requires continuing learning-curve benefit and acceptable generalization/topology, not merely improved training fit. No new IXI efficacy conclusion is available at submission.

### Execution and retrieval

All GPU work in this batch is serialized to **one H100** (array `%1`) to leave one shared slot for concurrent agents. The submitter audits all same-user GPU allocations/pending jobs under a coordination lock and chains behind existing GPU work. A shared claim is recorded under `/lustre/fswork/projects/rech/vnc/upz25mj/projects/archive/coordination/ixi-local-components-20261026.json`; other agents must check this claim and the actual queue to maintain the global limit of two. No other agent's jobs are cancelled or changed.

Dependency chain: correctness/benchmark → native zero-shot → six-arm fine-tuning array → CPU collector. Invalid prerequisites block downstream jobs. GPU wall allocations: smoke30min, zero-shot75min, each training arm3h (including evaluation/buffer); actual update budget is recorded in `smoke.json`. Large checkpoints/caches/runtime staging use SCRATCH; numerical raw caches are ZIP archives to avoid the earlier file-count-quota failure. Only this batch's per-job temporary runtime is removed on exit.

Exact plan, source checksums, source snapshot and submitted job IDs: worktree `experiments/ixi_local_components_20261026/{plan.json,working-source-snapshot.json,jobs.json}`. Checkpoints and numeric results: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026`. After collection, `analysis.json` retains trajectories, operating policies, paired results, endpoint checks and image interventions; this existing report receives the bounded results section below.

From the parent `projects` folder, read saved status/results without running inference:

```bash
python3 gnbm-ixi-local-components-20261026/scripts/show_ixi_local_components.py
```

<!-- BEGIN IXI LOCAL COMPONENTS 20261026 -->
### Results pending

CPU routing, physical sampling-grid and query-expansion tests passed. GPU correctness, throughput and scientific comparisons are dependency-gated; no results are claimed before they finish.
<!-- END IXI LOCAL COMPONENTS 20261026 -->

### Accepted submissions —26 October2026

| Stage | Job | Dependency / allocation |
|---|---|---|
| GPU correctness and benchmark | 1569366 | 30 min; no prior same-user GPU jobs at submission |
| Matched native zero-shot | 1569367 | afterok:1569366; 75 min |
| Six-arm fine-tuning | 1569368_0–5 | afterok:1569367; array%1; 3 h per arm |
| CPU metrics/paired collection | 1569369 | afterok:1569368; 2 h |

All four submission IDs were accepted. The actual shared update count will be fixed by the GPU throughput gate (cap6,000); dependent training cannot launch if the gate or zero-shot evaluation fails. No scientific result or IXI method winner is claimed at submission.


### Batching gate interruption and correction —26 October2026

Original GPU smoke1569366 failed after1m46s because batch1-versus-batch4 coordinates exceeded the strict1e-4 normalized-coordinate drift gate. All three CPU checks passed. Native zero-shot1569367, training1569368 and collector1569369 were automatically cancelled before execution, preserving the correctness gate; no IXI fine-tuning results exist from that chain.

The isolated worktree now uses **batch1 for both common/native reconstruction and raw-query analysis**, replacing the mixed-batching plan above. The smoke records cross-batch drift rather than assuming its magnitude, and separately requires stable repeated batch1 inference. This addresses evaluation comparability; it does not establish the cause of the model's batch sensitivity. Targets, source weights, losses, train batch32 and budgets are unchanged. Exact recovery/source hashes and preserved original submissions are in the isolated experiment metadata. Replacement correctness remains dependency-gated; no scientific benefit is claimed.

Replacement accepted chain: **1583591** smoke → **1583592** zero-shot → **1583593_0–5** training(%1) → **1583594** collector. To respect shared allocations, smoke waits with `afterany:1583544:1583545:1583546:1583547` for the concurrent BRAVA GPU batch. No BRAVA jobs were modified. The original failure and cancelled IDs remain documented above.
