# IXI source-domain comparison, representation trained in both stages

Date: 2026-10-15. User authorized three data strategies, each with straight edges and B-splines during BOTH pretraining and fine-tuning. This differs from the preceding plant/leaf→SyntheticMRI experiments, which pretrained graph detection only and introduced spline loss during fine-tuning. Those existing runs are preserved.

## Six arms

| Array index | Arm | First stage | Second stage |
|---|---|---|---|
|0|ixi_straight|IXI, straight|IXI, straight|
|1|ixi_spline|IXI, B-spline|IXI, B-spline|
|2|plants_straight|Plants + IXI, straight|IXI, straight|
|3|plants_spline|Plants + IXI, B-spline|IXI, B-spline|
|4|synthetic_straight|SyntheticMRI + IXI, straight|IXI, straight|
|5|synthetic_spline|SyntheticMRI + IXI, B-spline|IXI, B-spline|

All first stages start from random weights; the base YAML supplies configuration, not pretrained parameters. Each second stage loads its own validation-selected first-stage checkpoint, INCLUDING learned spline parameters, with fresh optimizer/scheduler. The IXI-only second stage is continued training on the same population, not a domain transfer.

## Controlled training recipe

- First stage50epochs,50,000 replacement draws/epoch, batch32 (1563updates/epoch).
- Same10,000 compatible IXI training patches across all six arms, selected by seeded hash ranking; subject-disjoint original splits preserved.
- Mixed sources25,000 plants or25,000 SyntheticMRI patches. Equal expected source/target draws via target weight2.5. IXI-only uses50,000 IXI draws. This matches updates, not IXI exposure or exact GPU compute. Edge-geometry loss averages matched positives, so sample balance does not imply equal per-domain geometry gradient magnitude.
- Second stage up to100epochs on the same IXI training population; existing patience30/minimum30 policy and monitoring every5epochs retained.
- AdamW LR7e-5, decay1e-4, one-epoch warmup and existing polynomial schedule. Graph losses unchanged; spline residual-control weight1.4739989612791453. This weight is reused from SyntheticMRI, not newly calibrated to IXI.
- Mixed pretraining uses domain adaptation weight1.0, image/instance adversarial losses and consistency. IXI-only and all second stages disable domain adaptation.
- All arms use256object queries, following existing IXI configurations; original SyntheticMRI study used120. Node-type prediction remains disabled. All other architecture choices follow the prior controlled setup.
- Cubic B-splines have8total controls (6internal). Retain the tested detached gradient route: residual geometry learns from relation/object tokens and local image features without sending geometry gradients into the shared detector or predicted endpoints. Graph losses still train the detector. Thus “B-splines throughout” means geometry supervision in both stages, not fully coupled optimization.
- Straight controls retain a frozen zero-residual spline head for common geometry inference. Their curves are straight chords.
- Same approved quarter-turn/noise MRI augmentation; no zoom. Plants retain quarter-turns/flips and five-slice projection into padded64³ volumes.
- BN training statistics during updates, saved running statistics during evaluation. Offline W&B and fixed validation projections retained.

## Data and geometric supervision

New derived vascular caches are under `/datasets`, not the experiment output directory:

- IXI: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015`.
- Auxiliary SyntheticMRI: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/syntheticMRI/transfer_source_k8_25k_20261015`.
- Plants reuse the existing25k contracted source and retained branch-path cache from the previous experiment.

IXI uses the existing `optimal_radius_0p75x_c095_vedo_original` sources/patches. Trace full stored paths before fitting, preserve junctions/endpoints and boundary anchors using the established closest-inside-centerline-sample policy, with exact intersection fallback for sparse clipped components. Straight and spline arms use identical nodes, edges, membership and full-path evaluation references. Intermediate degree-2 samples remain in curve guides, not object targets.

Unsupported parallel/closed branches and capacity overflow exclude the entire patch for EVERY arm; reasons are recorded. Original IXI active pool has13,742train/2,961val/2,384test patches, before this common compatibility check. All compatible validation/test patches are retained; final counts are only established by preparation. Failure to supply10k IXI or25k SyntheticMRI training patches blocks training. Source SyntheticMRI is limited to original training subjects with available full graphs; missing subjects are reported. Original source validation/test subjects are not imported as training data.

IXI graph paths are segmentation-derived uncertain references, not perfect anatomical truth. MRI metrics use original full-volume affines, never identity patch headers.

### Plant spline targets: previously missing integration

`PlantsDataset` optionally reads cached controls/curves/guides, verifies node/edge correspondence, swaps PNG axes, embeds curves at z=.5, applies the SAME sampled rotations/flips as the graph, and maps them through final padding. Controls are never clipped. Its returned geometry dictionary is supervised by the existing residual-control loss.

Plants have no physical millimetres. Define virtual isotropic voxel spacing as the median geometric-mean spacing across selected IXI TRAINING subjects. This scales the projected source geometry to the target loss's physical range; it is an explicit rendering assumption, not measured plant anatomy. Maintain a metric matrix through augmentation/padding so distances and canonical edge orientation remain consistent. The exact value is recorded in `monitoring.json` and each run's provenance. The source cache stays in original PNG units; only loading transforms it.

## Evaluation and execution gates

- Fixed hash-selected128validation and512training monitors, identical across arms; length-bin cutoffs computed on IXI training guides only.
- Checkpoint selection: validation common branch-F1, then native edge-F1. Full compatible validation evaluated for final and selected checkpoints, including common geometry and topology suite.
- After fine-tuning, evaluate the validation-selected checkpoint once on the fixed compatible IXI test set. Test outcomes never select checkpoints or adjust the setup.
- GPU smoke checks run three updates per stage/arm. They assert finite gradients, active domain classifiers in mixed runs, spline gradients in spline runs, and nonzero geometry loss gradients separately for source and target domains. Shared initialization hashes must agree across all six before production training starts.
- CPU smoke1125311 failed because the IXI source path required `graphs/adaptive/`; corrected without changing data. Replacement1125734 passed, accepting2train/2val/2test targets;3parallel-branch exclusions among24attempted candidates. No performance conclusion from this extraction check.
- Targeted coordinate tests:3passed (plant embedding and existing MRI spline rotation). All12stage configurations validated. GPU checks remain pending until full target preparation completes.

## Submitted jobs

Record: `experiments/ixi_transfer/submission-20261015.json`.

| Stage | Job |
|---|---|
|IXI target preparation|1126571|
|SyntheticMRI source preparation|1126575|
|GPU verification, six arms|1126576|
|Pretraining to25|1126577|
|Pretraining to50|1126578|
|Fine-tuning to50|1126579|
|Fine-tuning to100|1126580|
|Validation-selected held-out tests|1126581|

GPU arrays use indices0–5 with at most3concurrent tasks per array. Each task uses oneH100. `afterok` gates preparation/verification; `aftercorr` links the corresponding arm's later stages. Failed prerequisites cancel dependent jobs instead of launching an incomplete experiment.

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/`.
Config: `configs/ixi_transfer.yaml`. Runner: `scripts/train_ixi_transfer.py`; it reuses the existing loss, optimizer, matcher, augmentation, monitoring and metrics infrastructure. `source_mri` is an additional loader name for the existing vascular loader; `synthetic_mri` remains the internal target key but points to IXI here.

## Next research decision

First inspect full-population exclusions and GPU checks; then compare learning curves and final held-out metrics. This experiment tests auxiliary biological versus synthetic vascular sources under a fixed update budget. It does not isolate a single causal “biological alignment” mechanism or establish source-independent superiority. No new sweep or loss terms are authorized by this setup.

Submission check: both full preparation jobs are running; all GPU stages and tests are registered with the intended dependencies. The two training smoke targets were visually inspected in `previews/ixi-target-smoke.png`: retained paths align with the masks, but source-derived graphs do not explain every foreground fragment. This is a limitation of the existing extracted references, not evidence of anatomical completeness.

## Added matched protocol: graph-only pretraining → spline fine-tuning (2026-10-15)

All original jobs above are preserved. The original six-arm study includes spline supervision in both stages; earlier SyntheticMRI experiments introduced spline supervision only during fine-tuning. To distinguish these protocols, three additional fine-tunings reuse the existing graph-only pretraining checkpoints:

| Source strategy | Existing straight control | Added delayed spline | Existing both-stage spline ablation |
|---|---|---|---|
|IXI only|`ixi_straight`|`ixi_delayed_spline`|`ixi_spline`|
|Plants + IXI|`plants_straight`|`plants_delayed_spline`|`plants_spline`|
|SyntheticMRI + IXI|`synthetic_straight`|`synthetic_delayed_spline`|`synthetic_spline`|

Each delayed arm loads `pretrain_<strategy>_straight/best.pt`, excludes `spline_head.*` and domain-classifier weights, and initializes fresh spline geometry with the fixed seed. It uses the same detached residual geometry objective, fine-tuning budget, augmentation, batch size, IXI membership, validation selection and held-out evaluation as the existing arms. No new pretraining is needed. Existing straight controls are shared, yielding nine final models. This matches the earlier SyntheticMRI *stage protocol*; IXI still uses its own dataset and 256 queries, versus 120 in that earlier study.

Lightweight verification passed for all three configurations (target-only, domain adaptation disabled, spline weight unchanged) and checkpoint transfer (detector restored, delayed geometry left fresh; existing both-stage geometry restored). Syntax checks passed. GPU verification is queued and gates each corresponding training; training results are not yet available.

| Added stage | Job array |
|---|---|
|Delayed-head GPU smoke|1128482|
|Fine-tuning through epoch 50|1128483|
|Continuation through epoch 100|1128484|
|Validation-selected held-out test|1128485|

Arrays use indices 0,2,4, preserving correspondence with original straight-pretraining tasks. Smoke depends on corresponding original smoke 1126576; fine-tuning depends on both its smoke and original pretraining 1126578. Subsequent stages use corresponding-task success dependencies. The existing early-stopping rule remains active; 100 epochs is the maximum.

Submission record: `experiments/ixi_transfer/delayed-submission-20261015.json`.
Reproduction: `python3 cluster/jean_zay/submit_ixi_delayed_spline.py` (refuses an existing submission record to prevent duplicates).
Output uses the existing runs root with separate `<strategy>_delayed_spline` directories. Compare delayed splines against the shared straight controls first, then against both-stage splines to assess whether geometry pretraining helps. Do not pool these two spline protocols under a single label.

### Recovery after GPU diagnostic failure

Both dataset preparations completed. Original GPU checks 1126576 tasks 0,1,2,4 and all delayed checks 1128482 passed. Tasks 3 and 5 failed inside `check_domain_geometry`: its one-sample forward reached the domain discriminator's training-mode BatchNorm (`[1,2048]`). This was a diagnostic failure before production pretraining, not evidence that optimization failed. Slurm automatically cancelled all downstream production chains; no running training was manually cancelled.

Minimal correction: run this standalone gradient probe in evaluation mode with autograd enabled, then restore each module's original mode in `finally`. BatchNorm running statistics are preserved. Actual training updates still use training mode, batch 32 and active domain adaptation. A CPU check reproduced the failure with the actual domain discriminator, then exercised the corrected probe for both domains: positive spline gradients, unchanged BN running means and restored training modes. The real GPU checks remain the production gate.

Restored submissions (completed prerequisites verified through accounting; their old IDs had expired from Slurm's live dependency records):

| Stage | Replacement job |
|---|---|
|Retry only failed GPU checks, indices 3,5|1145367|
|Six pretrainings through epoch 25|1145368|
|Six pretrainings through epoch 50|1145369|
|Original six fine-tunings through epoch 50|1145370|
|Original six fine-tunings through epoch 100|1145371|
|Original six held-out tests|1145372|
|Three delayed-spline fine-tunings through epoch 50|1145373|
|Three delayed-spline fine-tunings through epoch 100|1145374|
|Three delayed-spline held-out tests|1145375|

All production chains wait on the corrected GPU checks. Original records/results remain intact; these replacements retain all nine comparison arms. Authoritative recovery record: `experiments/ixi_transfer/probe-recovery-20261015.json`; reproducible submission script: `cluster/jean_zay/resubmit_ixi_after_probe_fix.py`.

### Status 2026-10-16

All six pretrainings completed through epoch 50. Four fine-tuned models completed so far: IXI straight, plants straight, IXI delayed spline, plants delayed spline. Three corresponding tests completed. Test 1145372_0 (IXI straight) failed before Python inference: runtime archive extraction exhausted `/tmp` on jzxh163. Resubmitted only this test as **1160089_0**, excluding that node. Training/checkpoint results are intact. Other IXI fine-tunings are running or waiting normally; no additional failures observed in this check.

### Fixed 100-epoch budget — user decision 2026-10-16

Disable patience-based early stopping for all current source-transfer comparison configurations (IXI, plants, leaves, combined and MRI-only controls). Pretraining remains 50 epochs. Fine-tuning now always runs through epoch 100; validation still selects `best.pt` using the unchanged branch-F1/edge-F1 rule. This changes stopping, not checkpoint selection, data, loss weights or learning-rate schedule.

Three IXI arms previously completed early and are resumed from `latest.pt`, restoring optimizer, scheduler and RNG:

| Arm | Previous stopping epoch | Continuation/test task |
|---|---:|---|
|plants_straight|45|1165050_0|
|plants_spline|35|1165050_1|
|plants_delayed_spline|65|1165050_2|

Each continuation archives the old final/best validation outputs, test outputs, completion marker, summary and provenance under `early-stopped-epoch<N>/`. A hard link preserves the earlier best checkpoint without duplicating its payload; training saves replace checkpoints atomically. Histories remain append-only. The run then completes epoch 100 and tests the validation-selected checkpoint again. Earlier reported test results remain historical early-stopping results until replacement tests complete.

All SyntheticMRI leaf/plant/MRI-only runs already reached 100. Combined straight is finishing its epoch-100 evaluation; combined spline is complete. Active IXI spline was at epoch95 with best70 and active SyntheticMRI→IXI straight at epoch90 with best85: neither can stop before epoch100 under its already-loaded policy. Pending IXI continuation jobs will read the updated disabled-stop configuration at launch. Thus no running training needed cancellation.

Verification: Python/shell syntax checks and direct execution of each runner's stopping expression passed (disabled policy cannot trigger; enabled legacy behavior still works). Existing successful training smoke tests cover the unchanged forward/backward/continuation path. Submission record: `experiments/ixi_transfer/full-budget-20261016.json`.

### Disk recovery and completion check (2026-10-16 12:55 Paris)

Two jobs remained nominally RUNNING after their DataLoader workers failed creating `/tmp/pymp-*` on jzxh163. The node was already administratively DRAINed for a full directory. Cancelled only these blocked jobs (1145371_1 and combined 1147571_0), then resumed their saved states on other nodes. IXI recovery 1165756 and test 1165758 completed; combined recovery 1165760 completed and test 1165761 is running. Replacement combined report job 1165762 waits on that test. Exact commands are preserved in `experiments/ixi_transfer/disk-recovery-20261016.json`.

Eight IXI arms are at epoch100; plants_spline continuation 1165050_1 is at epoch90. Six updated IXI held-out tests are complete; SyntheticMRI-source spline/delayed tests 1145372_5 and 1145375_4 are pending. Reduced those pending tests' requested walltimes from20h to30min based on observed4–6min evaluation runtimes; model/evaluation settings unchanged. Current scheduler estimate is Oct17 08:44 but may move after the shorter reservations are considered.

<!-- BEGIN Q0 CAPACITY 20261028 -->
## Q0: current contracted object counts — 28 October 2026

A read-only all-file audit (CPU Slurm1653086, completed exit0) verified the existing K8 VTP/NPZ target parity: **10,000 train/2,649 validation patches**,123/26 subjects. Max anchors **153/141**, p95 **62/66**, p99 **86/87.52**; **five train/five val exceed120**, none exceed192 or256. Every audited object target has zero degree-2 vertices; intermediate samples remain curve guides. The modern4096/512 subsets also contain one overflow each (153/124 anchors). Existing256-query training is capacity-safe;120 is not. No targets/splits/exclusions changed and no test target was read. See the [full Q0 audit](../metrics/3-ixi-node-refinement-pilot.md#q0-ixi-status-and-contracted-target-capacity-audit--28-october-2026) for exact membership, boundary/isolate counts, historical239-node provenance, source hashes, and why modern IXI training remains unexecuted after infrastructure failure. No new training was launched.
<!-- END Q0 CAPACITY 20261028 -->
