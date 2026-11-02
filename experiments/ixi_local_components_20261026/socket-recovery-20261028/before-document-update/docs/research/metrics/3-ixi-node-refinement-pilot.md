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

**Status checked 28 October2026: PARTIAL / RECOVERY; zero-shot and planned fine-tuning NOT RUN.** This dated extension preserves the October18 node-head-only pilot above. It follows the completed [SyntheticMRI 12,000-update/two-seed local refinement study](../local_refinement_long_study.md), which supports coordinate refinement while showing no clear added confidence benefit. The later KNN [node verifier](../node_verifier_feasibility.md) failed its gate; **A-cls here is the existing 27-sample early-CNN module, not that verifier**.

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
### Scientific results unavailable — execution audited 28 October2026

The replacement GPU smoke timed out after multiprocessing socket-path failures; all downstream jobs were cancelled before starting. CPU checks passed, but the full GPU gate never completed. See the dated execution/capacity audit below. No A-cls/B0/A-xyz efficacy result is available from this batch.
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
