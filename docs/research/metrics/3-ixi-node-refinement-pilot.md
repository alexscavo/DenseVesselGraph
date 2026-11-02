# 3. IXI higher-resolution node refinement pilot — 2026-10-18

> Latest modern IXI results: [reviewed six-arm completion,28 October2026](#reviewed-final-modern-ixi-results--28-october-2026). A-cls improves physical edge/Curve F1, with more duplicates and cycle error; one fine-tuning seed.

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

**Status: DONE — measured; recovered six-arm study and collector completed 28 October2026. See the final reviewed results below.** This dated extension preserves the October18 node-head-only pilot above. It follows the completed [SyntheticMRI 12,000-update/two-seed local refinement study](../local_refinement_long_study.md), which supports coordinate refinement while showing no clear added confidence benefit. The later KNN [node verifier](../node_verifier_feasibility.md) failed its gate; **A-cls here is the existing 27-sample early-CNN module, not that verifier**.

### Isolation, data and spatial contracts

Work is isolated on branch `research/ixi-local-components-20261026` in `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026`. The original working tree was already heavily modified; required working sources were copied into this worktree with SHA256 provenance, without modifying shared models, configs or canonical research files. This report is the only intended shared scientific-document update; the collector checks for concurrent changes before replacing its own bounded results section.

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

All GPU work in this batch is serialized to **one H100** (array `%1`) to leave one shared slot for concurrent agents. The submitter audits all same-user GPU allocations/pending jobs under a coordination lock and chains behind existing GPU work. A shared claim is recorded under `/lustre/fswork/projects/rech/vnc/upz25mj/projects/experiment_gpu_coordination/ixi-local-components-20261026.json`; other agents must check this claim and the actual queue to maintain the global limit of two. No other agent's jobs are cancelled or changed.

Dependency chain: correctness/benchmark → native zero-shot → six-arm fine-tuning array → CPU collector. Invalid prerequisites block downstream jobs. GPU wall allocations: smoke30min, zero-shot75min, each training arm3h (including evaluation/buffer); actual update budget is recorded in `smoke.json`. Large checkpoints/caches/runtime staging use SCRATCH; numerical raw caches are ZIP archives to avoid the earlier file-count-quota failure. Only this batch's per-job temporary runtime is removed on exit.

Exact plan, source checksums, source snapshot and submitted job IDs: worktree `experiments/ixi_local_components_20261026/{plan.json,working-source-snapshot.json,jobs.json}`. Checkpoints and numeric results: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026`. After collection, `analysis.json` retains trajectories, operating policies, paired results, endpoint checks and image interventions; this existing report receives the bounded results section below.

From the parent `projects` folder, read saved status/results without running inference:

```bash
python3 gnbm-ixi-local-components-20261026/scripts/show_ixi_local_components.py
```

<!-- BEGIN IXI LOCAL COMPONENTS 20261026 -->
### Completed IXI component results —28 October2026

Completed bounded fixed-budget development comparison. Full reproducible numeric results: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026/analysis.json`. One fine-tuning seed; uncertainty resamples validation subjects, not source initializations.

| Arm | Node F1 1 mm | Raw /filtered recall 2 mm | Node F1 2 mm | Edge F1 2 mm | Branch F1 | Curve F1 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.16196 | 0.69127 /0.28629 | 0.36487 | 0.08793 | 0.01026 | 0.21489 |
| A-xyz-fresh | 0.19326 | 0.74355 /0.30560 | 0.39142 | 0.09431 | 0.00932 | 0.21775 |
| A-xyz-transfer | 0.20546 | 0.74782 /0.30056 | 0.38880 | 0.09464 | 0.00893 | 0.21359 |
| A-cls | 0.17772 | 0.68661 /0.30217 | 0.38658 | 0.10154 | 0.01119 | 0.22439 |
| A | 0.19627 | 0.74050 /0.29812 | 0.38466 | 0.09175 | 0.00937 | 0.21524 |
| A-control | 0.16410 | 0.69005 /0.28454 | 0.36392 | 0.08783 | 0.01044 | 0.21122 |

| Comparison | Edge F1 2-mm delta [95% subject interval] | Curve F1 delta [95% subject interval] |
|---|---:|---:|
| A-xyz-fresh-minus-B0 | +0.00639 [-0.00085, +0.01425] | +0.00286 [-0.00328, +0.00903] |
| A-xyz-transfer-minus-B0 | +0.00672 [-0.00010, +0.01418] | -0.00130 [-0.01055, +0.00762] |
| A-cls-minus-B0 | +0.01361 [+0.00826, +0.01940] | +0.00950 [+0.00234, +0.01738] |
| A-minus-B0 | +0.00382 [-0.00252, +0.01120] | +0.00035 [-0.00649, +0.00726] |
| A-control-minus-B0 | -0.00009 [-0.00618, +0.00583] | -0.00367 [-0.00883, +0.00185] |
| A-xyz-transfer-minus-A-xyz-fresh | +0.00033 [-0.00547, +0.00589] | -0.00416 [-0.01044, +0.00227] |
| A-minus-A-xyz-fresh | -0.00257 [-0.00915, +0.00362] | -0.00252 [-0.00822, +0.00333] |
| A-minus-A-cls | -0.00979 [-0.01588, -0.00301] | -0.00915 [-0.01725, -0.00123] |
| A-minus-A-control | +0.00391 [-0.00281, +0.01130] | +0.00401 [-0.00383, +0.01203] |

The descriptive final edge-F1 leader is **A-cls**. This does not alone select the method: consult matched-count/calibrated policies, confidence ranking, duplicate/topology errors, Stage1 ablations and paired intervals in `analysis.json`. Longer training is not automatically authorized. Attribution: fresh xyz versus B0 tests localization; fresh cls versus B0 tests classification; combined versus each single component tests interaction; transferred versus fresh xyz isolates head transfer on an identical shared initializer. Native zero-shot checkpoints have co-adapted trunks and are a separate comparison.

Paired edge improvement at both default and matched counts, without a negative Curve-F1 point difference, is supported for: **A-cls**. These are pilot criteria, not proof of topology improvement. Classification is only an added supported benefit if its fresh arm beats B0 and combined A improves over coordinate-only under the same controls. Transfer is assessed separately by transferred-minus-fresh xyz. Learning trajectories on the same128 validation patches are saved; any longer-training decision must consider plateau, training/validation gap and topology rather than select a favorable checkpoint after the fact. No extension is automatically submitted.

All arms share original IXI targets/splits, node-focal gamma2 plus edge-CE ratio0.15, K8 geometry and a fully trainable backbone/transformer/prediction stack. Fine-tuning uses a shared 256-query expansion; zero-shot retains native120. All reconstruction and raw-query evaluation uses batch1 consistently; cross-batch drift and same-batch repeat checks are recorded by the smoke gate. Graph metrics retain their original physical/common-v2 definitions. No test data, new representation or edge focal was introduced. Exact protocol and submissions remain in the worktree experiment directory.

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


<!-- N07 STATUS AUDIT 20261028 -->
## N07/D04 execution and current-target capacity audit — 28 October 2026

**N07 is prepared but scientifically untested (PARTIAL / RECOVERY).** The modern standalone image-conditioned A-cls comparison did not complete zero-shot evaluation or any planned IXI fine-tuning arm. This is not a negative result for classification refinement. The historical October18 continuation and SyntheticMRI combined-A component ablations do not supply that missing comparison.

### Actual execution, rather than accepted submission

| Chain | Correctness smoke | Zero-shot | Six-arm training | Collector |
|---|---|---|---|---|
| Initial | 1569366 FAILED after 1m46s (cross-batch coordinate drift gate) | 1569367 CANCELLED | 1569368 CANCELLED | 1569369 CANCELLED |
| Replacement | 1583591 TIMEOUT after 30m04s | 1583592 CANCELLED | 1583593 CANCELLED | 1583594 CANCELLED |

All downstream jobs have zero execution time. The replacement smoke ran October26 15:14:48–15:44:52 Paris time. Live `sacct` confirms the states; `squeue` contained no same-user jobs at audit. No planned-arm checkpoint, zero-shot summary or collected result exists in the experiment root. The 0-byte `runs/smoke-B0/tiny/caches.zip` is an interrupted smoke artifact, not reusable inference.

The replacement log contains multiprocessing **`OSError: AF_UNIX path too long`** before the timeout. The job wrapper uses a long experiment-runtime path as TMPDIR; a corresponding multiprocessing listener address is 133 bytes and reproduces that exception in a CPU socket check. The saved partial smoke records `completed=false`, no completed arms, batch1 repeat drift **0**, batch1-versus4 normalized coordinate drift **0.00109038**, and physical-spacing checks for149 IXI subjects (approximately0.46–0.4883,0.46–0.4883,0.80mm). It reached the B0 tiny evaluation, but did not validate the six-arm GPU gate or determine a common training budget. Increasing wall time alone is not a fix.

**Required recovery:** separate a short SCRATCH-backed multiprocessing TMPDIR from the long runtime/cache directory; preserve the same batch1 evaluation policy, original plan and old failure artifacts; pass the full correctness/benchmark gate before launching scientific work. No runtime/model code was changed and no GPU job was submitted in this audit, following registry §11's CPU-first/no-training instruction. The stale coordination claim was corrected to zero reserved/running H100s.

### Current serialized K8 targets (J02)

Read every retained development NPZ's actual `nodes`, `edges` and `boundary`, checking K8 control shapes and valid graph indices. Membership matches both dataset manifests, all counts match `patch_index.csv` anchor/branch counts, and eight representative VTP headers agree. Train/validation subjects are disjoint. No image processing, target regeneration, filtering, inference or test-array access was performed.

| Population | Patches / subjects | p95 / p99 nodes | Max nodes | >120 / >192 / >256 |
|---|---:|---:|---:|---:|
| IXI train | 10,000 / 123 | 62.00 / 86.00 | 153 | 5 / 0 / 0 |
| IXI val | 2,649 / 26 | 66.00 / 87.52 | 141 | 5 / 0 / 0 |
| BRAVA train | 8,768 / 39 | 38.00 / 47.00 | 64 | 0 / 0 / 0 |
| BRAVA val | 1,796 / 8 | 38.00 / 51.00 | 69 | 0 / 0 / 0 |

There are **zero degree-2 object anchors** in either audited current target population. Isolated anchors are preserved (IXI train/val76/36; BRAVA2610/584), as are recorded boundary anchors (IXI95940/25691; BRAVA102426/21613). Dense paths are separate spline guides, not extra detector objects. The current generator calls `contract_spline_graph(...max_anchors=256,preserve_nodes=boundary)`; existing IXI development exclusions comprise parallel branches (train1207/val253) and closed branches (217/59), not object-capacity exclusions. BRAVA records no exclusions. Counts describe the existing retained population, not all possible source crops.

**120 is not capacity-safe for unchanged IXI labels.** The fixed N07 population also contains one153-node training patch (`sample_000150_0066`) and one124-node validation patch (`sample_000058_0085`). Its4096/512 patches cover123/26 subjects. A 120-query native zero-shot evaluation may be reported as a source-capacity diagnostic, but it must retain overflow patches/GT in the denominator; it is not a capacity-matched fine-tuning arm. Keep256 for N07 recovery.192 would fit current counts but is an additional untested configuration. BRAVA's current retained development targets fit120; that says nothing by itself about model accuracy.

Historical239-node provenance is independently located in `docs/research/evidence/ixi_capacity_192_256/{selection,summary}.json`: older root `IXI/vascular_patches/optimal_radius_0p75x_c095`, sample`sample_000021_0117`,239nodes/261edges. That older root is no longer available at its recorded path, so its degree histogram could not be independently recounted. It must not be substituted for the **current contracted** maximum153. Current zero-degree2 counts and actual overflows are measured directly here.

### Identities, related BN status and decision

The immutable N07 plan/manifests and all three source checkpoint SHA256 values were verified. Source seed364505, step12000: common B0 `91ac46e091bb3f74c5748758235c26dc7c8814a232adda03dd49def0c1c8a3de`; A-xyz `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9`; combined A `ebb47360f7799801f1ab364f3ef9e60a1b877038d39962b4861d33d1944b9385`. Exact paths and byte sizes are in `checkpoint-identities.json`. The fine-tuning plan copies120 learned object rows, seeds136 additional object rows identically across arms, and moves both learned relation-token rows to256/257. Only A-xyz-transfer receives the learned local-refinement head; the shared trunk is B0 for every arm. A-cls is fresh zero-residual classification refinement using the established27-sample module, not the KNN verifier.

L06 was also checked: IXI1584593_0 FAILED after49s with an empty-pair indexing-dtype error; SyntheticMRI1584593_1 COMPLETED after1m37s and has saved `complete.json`, rows and recalibrated caches; collector1584594 CANCELLED. It remains **partial**, and none of those artifacts establishes modern IXI A-cls efficacy. No BN recovery was launched.

**Decision:** no modern IXI winner or training-duration recommendation can be made. The next permitted bounded recovery is the existing256-query B0/A-xyz/A-cls component study after its socket-path smoke fix, using the same initialization, splits and losses. Its final update budget is still unset (planned cap6000; minimum1500), because the gate did not finish. Do not launch a full-IXI120/256 comparison with silent overflow exclusions. No new architecture or loss experiment is justified by an infrastructure timeout.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/n07_status_audit_20261028` (`status.json`, `target-capacity.json`, `per-patch.csv`, `per-subject.json`, `overflow-above120.csv`, `cross-checks.json`, copied smoke/logs, source identities). Target audit took146.3s on CPU. Reproduce counts with:

```bash
/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python /lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/n07_status_audit_20261028/audit_targets.py
```

Read the confirmed state without inference or training:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/scripts/show_ixi_local_components.py
```

<!-- BEGIN Q0 IXI AUDIT 20261028 -->
## Q0: IXI status and contracted-target capacity audit — 28 October 2026

**DONE — measured (Q0 / J00 / J02 administration and CPU target audit).** Current IXI is **not 120-query safe**, and there are **no completed modern IXI B0/A-xyz/A-cls fine-tuning results** from the recorded component study. This is an infrastructure/capacity conclusion, not a negative result for local CNN refinement. No GPU, training or test evaluation was submitted during Q0.

### Verified execution state

Original correctness job **1569366 FAILED** on the cross-batch coordinate-drift gate. Replacement **1583591 TIMEOUT** after 30m04s; its stderr records `OSError: AF_UNIX path too long` from multiprocessing resource sharing. Partial batch-one repeat drift was zero, but `smoke.json` remains `completed=false` with no completed arm entries. Native zero-shot **1583592**, training array **1583593_0–5**, and collector **1583594** were cancelled before execution. The large output root contains only `runs/smoke-B0/tiny`, with no production-arm checkpoints or collected analysis. Live queue and all IXI/BN job names since 26 October were checked; no later successful recovery was found. D04 is **PARTIAL / RECOVERY**; standalone A-cls N07 is **NOT RUN**, cancelled before training.

BN L06 is independently **PARTIAL / RECOVERY**: IXI **1584593_0 FAILED**, SyntheticMRI **1584593_1 COMPLETED**, CPU collector **1584594 CANCELLED**. The IXI error is an empty Boolean eligibility mask inferred as floating point in `multiscale_pretrained.py::rows_for`, causing `IndexError` while indexing pairs. SyntheticMRI's completed half and its unchanged-weight/non-BN-buffer checks are preserved; see the dated Q0 section in the existing spline-collapse report. Stale GPU reservations were reconciled to zero only after terminal-job verification.

### Exact current object targets

CPU-only Slurm **1653086 COMPLETED, exit 0:0, 2m48s**. All **23,213** train/validation VTPs were read with the loader's PyVista point/line semantics and float32 conversion; every graph matched its saved K8 NPZ nodes/edges and eight-control dimensions. Original preparation/modern-study/D08 manifest hashes match. No test target/image was opened, no patch filtered, and no target regenerated.

| Population | Patches / subjects | Maximum nodes | p95 / p99 | Above 120 | Above 192 / 256 |
|---|---:|---:|---:|---:|---:|
| IXI full train | 10,000 / 123 | 153 | 62.00 / 86.00 | 5 | 0 / 0 |
| IXI modern-study train | 4,096 / 123 | 153 | 61.00 / 85.05 | 1 | 0 / 0 |
| IXI full val | 2,649 / 26 | 141 | 66.00 / 87.52 | 5 | 0 / 0 |
| IXI modern-study val | 512 / 26 | 124 | 66.00 / 83.00 | 1 | 0 / 0 |
| BRAVA full train | 8,768 / 39 | 64 | 38.00 / 47.00 | 0 | 0 / 0 |
| BRAVA full val | 1,796 / 8 | 69 | 38.00 / 51.00 | 0 | 0 / 0 |

All audited populations have **zero degree-2 object vertices**. Degree-2 path samples remain in the original dense paths/guides, outside the object-query target. Boundary provenance is retained in the caches; IXI train/val contain 95,940/25,691 boundary anchors and 76/36 isolated anchors, which were counted without deletion. BRAVA train/val retain 2,610/584 isolates. Thus the current 153-node maximum is already post-contraction. Per-subject distributions, role counts and every cache/VTP hash are in `targets.json` / `targets.csv`.

The modern 4,096-patch training set includes **sample_000150_0066 (153 anchors)**; its 512-patch validation set includes **sample_000058_0085 (124)**. Full-cache overflow identities are retained in `targets.json`: five train and five val, across the reported 123/26 disjoint subjects. The training maximum requires at least 153 object queries. Although 192 fits these audited train/val targets, no 192-versus-256 accuracy result was established; keep the existing **256-query** reference. The proposed 120/256 IXI study is **DEFERRED because its capacity prerequisite failed**, including on the modern subsets. Do not discard the overflowing examples.

Preparation version: input `format_version=4`, `voxel_cell_box_endpoint_aware_with_boundary_provenance_v4`, followed by `prepare_transfer_splines.py` → `contract_spline_graph` → `contract_degree_two`, preserving boundary anchors. The fixed preparation cap was 256. IXI train/val compatibility exclusions were 1,207/253 parallel-branch and 217/59 closed-branch cases; **zero capacity exclusions** were recorded. The final train cache is the existing seeded 10,000-compatible-patch selection, not the entire earlier raw candidate pool.

The archived **239-node** case is **PRE-CONTRACTION historical evidence**, from job **22067**, `sample_000021_0117` (261 edges), under the older `optimal_radius_0p75x_c095` source in `ixi_capacity_192_256/selection.json` and `summary.json`. Its original directory is no longer available for an independent VTP/degree recount. The archived measurement and that limitation are preserved; 239 is not claimed as a freshly measured current K8 maximum. Current decisions use the complete serialized-target audit above.

### Source and query-expansion provenance

The six-arm IXI plan uses the same SyntheticMRI seed364505 **B0 model-12000** trunk for all fine-tuning arms; no target optimizer/checkpoint exists to resume. Planned target seed261026, focal node gamma2/equal weights, matched edge CE/ratio0.15, K8 `graph_only`, train batch32 and batch-one evaluation remain unchanged. The original smoke must first determine the shared fixed update budget (cap6,000); Q0 does not set a new schedule.

Source files below are under `experiments/local_refinement_long/364505/runs/` in the main repository; SHA256 was rechecked against each file:

| Source role | Relative checkpoint | SHA256 |
|---|---|---|
| B0 | `B0/model-12000.pt` | `91ac46e091bb3f74c5748758235c26dc7c8814a232adda03dd49def0c1c8a3de` |
| A-xyz | `A-xyz/model-12000.pt` | `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9` |
| A | `A/model-12000.pt` | `ebb47360f7799801f1ab364f3ef9e60a1b877038d39962b4861d33d1944b9385` |

`expand_queries` preserves source object rows 0:120 exactly and moves all source relation/tail rows after target object row 256. Added object rows 120:256 come from `torch.randn(256,552)` using a dedicated generator seeded **262017 = 261026+991**, identically across the six arms. This is the documented IXI initializer, distinct from D08's retained fresh scratch rows. Native zero-shot preserves 120 queries and cannot be treated as a capacity-matched IXI continuation.

**One permitted next GPU run:** recover the original **256-query IXI correctness/benchmark smoke only**, with a short task-specific `TMPDIR` to resolve the socket-path failure. Preserve source/subjects/targets/losses and the batch-one evaluation correction. Submit downstream fine-tuning only after that smoke's parity/gradient/throughput gates pass. No duplicate or altered-capacity study is needed. The next independent queue task is **2D-QC**. Q0 launched neither GPU recovery nor training, as required by the latest executable request.

Isolated branch/worktree: `research/q0-ixi-audit-20261028`, `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-q0-ixi-audit-20261028`. Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-q0-ixi-audit-20261028/experiments/q0_ixi_audit_20261028`; [main-repository pointer](../../../experiments/q0_ixi_audit_20261028/worktree.json). `status.json`, `slurm-status.psv`, `checkpoint-identities.json`, `representation-provenance.json`, `targets.csv`, `targets.json`, `bn-partial.json`, preserved failure logs and `decision.json` retain exact identities, populations and limitations.
<!-- END Q0 IXI AUDIT 20261028 -->


<!-- N07 SOCKET RECOVERY 20261028 -->
### User-authorized socket-path recovery — 28 October 2026

**Status: recovered GPU correctness completed; zero-shot/fine-tuning/collection tracked in the verified snapshot below.** After the completed CPU status/capacity audit, the user explicitly requested recovery of the same experiment. This supersedes the audit-only/no-submission scope for this recovery; the failed attempts above remain preserved.

The isolated wrappers now source `cluster/jean_zay/ixi_local_tmp.sh` **after** `env.sh` (which resets TMPDIR). Multiprocessing uses short `$SCRATCH/tmp/icl-$SLURM_JOB_ID`; staged packages stay under the original experiment runtime directory. A path-length guard fails early if the short path is still unsafe. Both temporary locations are cleaned by the GPU wrapper; the collector also uses a short temporary path. No model, checkpoint, data, target, matcher, loss, threshold, batching or update-budget setting changed.

**Five focused CPU tests passed**, including actual multiprocessing FD sharing (old long path fails, new path succeeds), existing coordinate/classification routing, physical sampling and deterministic query expansion. Shell syntax and Python compile checks passed. The full six-arm GPU smoke/benchmark remains a prerequisite; CPU checks do not replace it.

| Stage | Accepted job ID | Dependency / wall allocation |
|---|---|---|
| Correctness and common-budget benchmark | **1653858** | afterany:1653554 (concurrent native2D GPU gate);30min |
| Native120-query zero-shot | **1653859** | afterok:1653858;75min |
| Six-arm256-query fine-tuning | **1653860_0–5** | afterok:1653859;array%1;3h per arm |
| CPU paired metrics/report collector | **1653861** | afterok:1653860;2h |

All IDs were accepted and dependencies checked once. The array remains limited to **one H100**; global maximum remains two across agents. The same4096 training/512 validation patches (123/26 disjoint subjects), six arms, common source initialization, node-focal/edge-CE recipe and batch1 inference are retained. The common update count is selected by the successful smoke (minimum1500,cap6000,conservative3600-second training estimate), not claimed fixed before it runs. No inference or training results are claimed at submission.

Previous `jobs.json` is archived as `jobs.initial-1583591.json`; original smoke/error evidence remains in `experiments/n07_status_audit_20261028`. Recovery checksums, source snapshot, immutable-plan copy, test verification and accepted IDs: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/ixi_local_components_20261026/socket-recovery-20261028`. Outputs remain `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026`. The existing CPU collector updates this report's bounded results section automatically.

Read saved status/results from any directory, without inference or training:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/scripts/show_ixi_local_components.py
```

<!-- BEGIN IXI RECOVERY VERIFIED SMOKE 20261028 -->
### Recovered smoke passed; existing jobs verified — 28 October 2026

Verified at `2026-10-28T10:05:01.360009+00:00`. **No additional submission was made:** the intended recovered chain was already accepted, and was inspected instead of duplicated. Correctness/benchmark **1653858 COMPLETED, exit0:0, 2m54s**. All six arms passed the existing checkpoint/query-expansion, shared-initializer, routing, finite forward/backward, active-module gradient, local-image gradient/control and tiny reconstruction gates. Batch-one repeat coordinate drift is **0**; the previously documented cross-batch drift remains **0.00109038** and evaluation stays batch1. Refined spline endpoint error was0 on31 reconstructed edges. Five startup unit checks also passed. No tolerance or scientific condition was weakened.

The **unchanged** budget rule selected **5,500 updates per arm**, checkpoints **1,500 / 3,000 / 5,500**, from the worst post-warmup batch32 timing **0.54789048 s/update** with the original1.15 safety factor and3,600-second training estimate. Smoke optimizer updates are discarded: every production arm calls `setup` again, reloads its original B0 source and gets a fresh target optimizer. Warmup100, peak7e-6, floor1e-6, polynomial0.9, augmentation, graph losses, K8 and `graph_only` remain as preregistered.

Live/accounting snapshot: native zero-shot **1653859 — RUNNING**; six-arm fine-tuning **1653860 — PENDING**, gated by `afterok:1653859` and serialized `%1`; collector **1653861 — PENDING**, gated by successful completion of the training array. These jobs already existed before this verification. The256-query fine-tuning cohort remains4,096/512 patches across123/26 disjoint subjects, with no target-capacity overflow, changed membership or test evaluation. Original native120-query zero-shot is retained as a separate diagnostic; its capacity differs from256-query fine-tuning and is not a matched-capacity efficacy comparison.

The immutable plan SHA256 (`b9554511dcbf0596f6615394bff91404c036ac6e6d4f0620531535532830598a`), all nine recovery source hashes, three source checkpoint hashes and four membership hashes were checked again. B0 source SHA256 remains `91ac46e091bb3f74c5748758235c26dc7c8814a232adda03dd49def0c1c8a3de`; A-xyz donor `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9`; combined-A native source `ebb47360f7799801f1ab364f3ef9e60a1b877038d39962b4861d33d1944b9385`. Source files and the deterministic added-query seed262017 remain unchanged. The accepted study has one H100 maximum; native2D B0/Axyz are serialized separately, permitting at most one2D plus oneIXI allocation. No2D jobs were altered by this verification.

**Scientific results remain pending.** Final full-stream parity across the six production arms and paired efficacy/Branch/Curve/topology conclusions can only be checked after training/collection. The existing collector writes only its bounded results block and preserves prior/Q0/recovery evidence. Snapshot, source identities and duplicate-job checks: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/ixi_local_components_20261026/socket-recovery-20261028/existing-chain-audit-20261028.json` and adjacent accounting/queue files.


**Zero-shot prerequisite verified at `2026-10-28T10:07:41.427171+00:00`:** **1653859 COMPLETED0:0 (7m55s)**; all three native-source completion artifacts (B0/A-xyz/A) are present and valid. The already submitted256-query training array1653860 is now eligible (pending scheduler priority at this check), followed by collector1653861. No new job or scientific change was made.
<!-- END IXI RECOVERY VERIFIED SMOKE 20261028 -->

<!-- BEGIN IXI FINAL REVIEW 20261028 -->
## Reviewed final modern IXI results — 28 October 2026

**DONE — measured. All six fully trainable arms completed5,500 updates; paired collection completed.** Smoke1653858, zero-shot1653859, every task1653860_0–5 and collector1653861 exited0:0. Same4,096 training patches/123 subjects and512 validation patches/26 disjoint subjects, target seed261026,256 object queries, original K8/`graph_only`, gamma2 node focal and matched edge CE/ratio0.15, shared augmentation, optimizer and fixed schedule. Primary comparison is the common final5,500 checkpoint, not each arm's selected checkpoint. Exactly176,000 training-patch exposures per arm. Test data remain untouched.

The complete augmented image/target stream SHA256 is identical in all six arms: `e206417d13a7e7aece56211278259da9fdb9724e7a93a50b898895975c9a5816`. The six final checkpoint files were independently hashed; canonical array IDs match recorded raw Slurm task IDs. Completion is supported by saved outputs and the successful collector, not just scheduler state.

### Main findings and controls

**Standalone A-cls has the clearest physical-edge and Curve-F1 gain, with a topology cost.** At2mm, node F1 rises0.364866→0.386583, edge F1 **0.087926→0.101536** and common Curve F1 **0.214892→0.224389**. The paired26-subject edge-F1 difference is **+0.013610 [+0.008265, +0.019401]**, and Curve-F1 difference **+0.009497 [+0.002337, +0.017381]**. With exactly the B0 retained-node count imposed separately on every validation patch, A-cls still reaches edge F1 **0.098788** versus0.087926; difference **+0.010862 [+0.005577, +0.016087]**. Thus the gain is not explained simply by accepting more nodes or changing a scalar threshold.

| Full512 validation population,2mm | B0 | A-cls |
|---|---:|---:|
| Predicted nodes | 7,459 | 7,380 |
| Node TP | 3,751 | 3,959 |
| Predicted edges | 5,091 | 5,440 |
| Edge TP | 706 | 833 |
| Edge FP | 4,385 | 4,607 |
| Edge FN | 10,262 | 10,135 |
| Duplicate nodes | 313 | 535 |
| Component-count MAE /patch | 3.287109 | 3.283203 |
| Cycle-count MAE /patch | 1.998047 | 2.580078 |

A-cls adds127 TP edges **and222 FP edges**; duplicate count increases313→535 and cycle MAE1.998047→2.580078 (difference **+0.582031 [+0.334645, +0.849651]**). Component MAE is nearly unchanged. Branch F1 remains very low,0.010263→0.011187. This is a measured graph-accuracy/geometry gain with worse duplication and cycle counts, **not** a general topology solution. Signed node/edge count bias remains strongly negative: B0-11.021/-11.479 versus A-cls-11.176/-10.797 per patch; net underprediction does not eliminate false nodes/connections.

**A-xyz improves localization/raw coverage but does not establish a clear final2mm edge gain.** Fresh/transferred xyz increase1mm node F1 from0.161957 to0.193265/0.205460, and raw2mm recall from0.691269 to0.743551/0.747825. Their2mm edge F1 is0.094314/0.094642; both default-threshold differences versus B0 have intervals including zero. Transferred-minus-fresh edge F1 is+0.000328 [−0.005469,+0.005891], and transferred Curve F1 is lower by0.004162 on the point estimate. Head-transfer superiority is not established. The coordinate arms lower cycle-count MAE (~1.51–1.52) but increase component MAE (~3.67–3.79), another trade-off.

**Combined A does not improve over its coordinate-only or classifier-only controls.** Final edge F1 is0.091748; A-minus-A-cls is−0.009788 [−0.015881,−0.003011]. A-minus-fresh-xyz is−0.002566 with an interval including zero. The zero-image A-control stays near B0 (edge F1 0.087834). Do not infer an added benefit from combining xyz and classification, or promote the transferred head from its small point advantage.

### Confidence, actual image use and remaining coverage failures

A-cls's saved canonical node-confidence AP increases0.270608→0.316830 and correct-versus-duplicate AUROC0.613616→0.626530; default retained-node recall rises0.286292→0.302168 even as raw recall changes0.691269→0.686613. These are node-confidence diagnostics, not relation AP. They support improved confidence selection in this pilot while duplicates still worsen.

In the predefined128-validation-patch intervention monitor, zeroing or spatially rolling A-cls image features reduces edge F1 by0.041345/0.068823; disabling its residual (Stage1) reduces it by0.020622. Node coordinates and corresponding raw relation probabilities stay identical under these within-model classification interventions. The trained head therefore uses actual local CNN evidence; these perturbations alone are not a parameter-control efficacy estimate. Across separately trained fully unfrozen B0/A-cls models, upstream coordinates may co-adapt and are not required to be identical.

Only1,540/10,968 GT edges have recovered retained endpoint pairs for A-cls. **93.02% of its missed edges lack such a pair** (B094.46%). Conditional relation recall is0.553725→0.540909; mean-patch conditional AP is0.672768→0.676601. These conditional populations differ, and mean-patch AP is not pooled relation AP. Full2mm edge recall is still only0.064369/0.075948 for B0/A-cls. Missing/rejected endpoints and weak full-graph reconstruction remain major limitations.

### Trajectories, uncertainty and metric scope

The table below uses the **same128 validation-monitor patches at every checkpoint**; it must not be mixed with the primary full512 results above.

| Arm | Edge F1 at1,500 | At3,000 | At5,500 |
|---|---:|---:|---:|
| B0 | 0.072426 | 0.087604 | 0.092973 |
| A-xyz-fresh | 0.072324 | 0.080781 | 0.093285 |
| A-xyz-transfer | 0.071094 | 0.086631 | 0.100649 |
| A-cls | 0.079597 | 0.091306 | 0.095548 |
| A | 0.074685 | 0.092043 | 0.087434 |
| A-control | 0.073464 | 0.083167 | 0.085069 |

A-cls's monitor improvement persists to the fixed endpoint; combined A peaks at3,000 on this monitor and declines later. Training-monitor node/edge F1 is0.390417/0.094132 for B0 and0.416388/0.123086 for A-cls; primary validation is0.364866/0.087926 and0.386583/0.101536. Both optimized-on training and validation graphs remain weak. This bounded pilot does not establish convergence, a purely held-out generalization failure, or that additional epochs suffice.

All intervals use2,000 paired whole-subject resamples of26 reused validation subjects; they do not measure training-seed uncertainty. This is one target seed and an exploratory multi-arm development study, not independent-seed/test confirmation. IXI references are segmentation-derived uncertain graph targets. Node/edge tables here use **physical1/2mm correspondence**; Branch/Curve retain their distinct existing common-v2 protocol. Legacy native edge F1 (B0 0.280715;A-cls0.288030) is saved separately and must not be substituted for physical2mm edge F1.

Native120-query source zero-shot B0/A-xyz/A edge F1 is0.031461/0.041274/0.040634 on the same validation population. Fine-tuning uses256 queries, so this is a separate source-capacity/co-adaptation diagnostic, not a controlled estimate of head transfer. All GT remains in evaluation; the124-node validation overflow was not filtered.

### Completion, costs and exact checkpoint provenance

Measured allocation: six training arms **3.6139 H100-hours**; including recovered smoke and zero-shot **3.7942 H100-hours**. The two earlier failed correctness attempts consumed another0.5306, giving **4.3247 H100-hours** for the recorded study/recovery history. Source pretraining and CPU collection are excluded. Collector1653861 took12m29s CPU wall time. No new GPU inference, training, threshold search or extension was run during this review.

All final checkpoints below are `model-5500.pt` under `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_local_components_20261026/runs/<arm>/`:

| Arm | Canonical Slurm task / raw ID | SHA256 |
|---|---|---|
| B0 | 1653860_0 / 1654596 | `682c4cb19e3150758a7ac04149931d76aa7da3ecc5916f8eb96e70007390b667` |
| A-xyz-fresh | 1653860_1 / 1655824 | `f02d205bf738c3f696be275379269acd5ff331c0bc4822fd2cbecaeea3c75f6f` |
| A-xyz-transfer | 1653860_2 / 1656912 | `096a0d1f61c061cd00570df8fd62f2f401d6f49f62b11309bb67387c67e64e50` |
| A-cls | 1653860_3 / 1657934 | `ee2cf41fe67eca90030cb1243f84fa9965dee28f8ecebab5d074bd55acd34fd2` |
| A | 1653860_4 / 1659185 | `1f9eb3e51717f7992e504d025658841bafffb7c2e100c7b1a5fc4bdf4c8e4f3a` |
| A-control | 1653860_5 / 1653860 | `6a433683b4025c58791a96d4a545d897ddde0d47c6e149cd14fe38762ac8c4c4` |

Raw IDs differ from canonical array IDs by normal Slurm array-task allocation; both were verified against accounting. Exact paths/sizes, plan/source checkpoint SHA256, saved-analysis hash, matching input-stream proof, cost accounting and completion checks: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/ixi_local_components_20261026/completion-review-20261028/verification.json` and `accounting.psv`. Original checkpoints, failed-job evidence and machine-generated results are preserved.

**Decision:** record A-cls as a promising standalone IXI accuracy candidate with a clear duplication/cycle cost; preserve B0 and the localization-specific A-xyz findings. The study does not support a universal combined-head improvement or a topology claim. Independent-seed confirmation and resolving endpoint/confidence/assignment errors would be needed before stronger promotion. **No follow-up experiment is launched automatically.** L06's separate incomplete BN control is not marked complete by these results.
<!-- END IXI FINAL REVIEW 20261028 -->

<!-- BEGIN IXI FINAL PREDICTION PREVIEW 20261028 -->
### Fixed validation prediction overlays —28 October2026

Saved final5,500-update predictions for the first existing fixed validation visualization, `sample_000001_0009` / `IXI_000001`, with no score-based example selection. [Shared raw-MRI/GT/A-xyz-fresh/A-cls comparison](../evidence/ixi_local_components_20261028/sample_000001_0009-comparison.png), [A-xyz-fresh alone](../evidence/ixi_local_components_20261028/sample_000001_0009-A-xyz-fresh.png), [A-cls alone](../evidence/ixi_local_components_20261028/sample_000001_0009-A-cls.png). Orange is the GT graph/dense reference; cyan is saved predicted nodes/K8 curves on the actual mean-centered MRI input. Each row shows D/H/W projections; projected crossings do not establish3D junctions. GT has44 nodes/42 edges; A-xyz-fresh predicts25/5 (15 nodeTP,0 edgeTP at2mm), A-cls24/17 (15 nodeTP,3 edgeTP). This is one fixed example, not an aggregate estimate.

These previews correct the generic `update0` title emitted by standalone evaluation exports: source arrays were read from `step-5500-reconstruction` and checked against final raw-query caches, query IDs, node scores and exact spline endpoints. No new inference/training, segmentation overlay or changed thresholds. [Source/provenance](../evidence/ixi_local_components_20261028/sample_000001_0009-provenance.json).
<!-- END IXI FINAL PREDICTION PREVIEW 20261028 -->

<!-- BEGIN IXI 3D PREDICTION PREVIEW 20261028 -->
### Interactive 3D prediction views —28 October2026

Final5,500-update fixed validation sample `sample_000001_0009`: [GT / A-xyz-fresh / A-cls, three adjacent views](../evidence/ixi_local_components_20261028/sample_000001_0009-3d-comparison.html); [GT beside A-xyz-fresh](../evidence/ixi_local_components_20261028/sample_000001_0009-A-xyz-fresh-3d.html); [GT beside A-cls](../evidence/ixi_local_components_20261028/sample_000001_0009-A-cls-3d.html). All HTMLs are self-contained/offline, with linked rotation/zoom, optional raw-MRI slices, optional GT overlay on predictions and common physical axis ranges. Coordinates use the cached physical metric with a shared patch-centre translation; predicted K8 curves and GT dense paths are unchanged. No segmentation overlay, new inference or training. Counts remain44/42 GT nodes/edges,25/5 A-xyz-fresh and24/17 A-cls. [3D source and verification record](../evidence/ixi_local_components_20261028/sample_000001_0009-3d-provenance.json).
<!-- END IXI 3D PREDICTION PREVIEW 20261028 -->
