# Research roadmap

> Active native 2D follow-up: [fixed 100-epoch vessel comparisons and 50-epoch mixed plant/vessel pretraining](../docs/research/native2d_long_20261028.md), authorized after the completed pilot and clarified by the user before long training. Keep confidence cutoffs at0.5; include restored OCTA-Synth after its data/GPU gates. Live job/results state is recorded under Native2D-04 in the registry.

<!-- NATIVE2D CURRENT POINTER 20261028 -->
> Latest native2D result: **Native2D-03 corrected-target pilot completed** (B0/A-xyz6,000 updates each; no filtered graph recovery at0.5). [Final comparison, dataset readiness and stopping decision](../docs/research/native2d_heldout_20261028.md#completed-corrected-target-comparison). Native2D-02 remains preserved with its unstable-target and post-training-timeout caveats.


Latest priorities: [completed BRAVA and native 2D review](#reviewed-priorities-after-d08-and-native-2d-completion--28-october-2026). Earlier priorities remain dated research history.

Start new experiment planning with the [stable-ID vascular experiment registry](VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md). It records N08’s completed negative result and preserves the remaining conditional decision gates; dated entries below retain their historical context.

Consolidated 23 October 2026. Start here for goals and priorities; see [architecture](architecture.md), [decisions](decisions.md) and [experiment register](experiments.md) for implementation, reasoning and evidence. Dated status below is a research snapshot, not a live Slurm queue.

## Goal and journal scope

Extend the accepted GrEVe work toward **compact, continuous vascular graph reconstruction across BRAVA, IXI and SyntheticMRI**. Establish whether learned adaptive geometry improves anatomical shape and complete graph reconstruction without unnecessary graph nodes, model complexity or detector losses.

The candidate contributions are continuous/adaptive edge geometry, a controlled graph-versus-geometry complexity study, and evaluation that distinguishes localization, connectivity and anatomical shape. Botanical transfer remains an independently tested hypothesis; the journal must include strong vascular-pretraining controls rather than assume the original pretraining claim generalizes.

## Local-refinement update —25 October2026

The [completed longer study](../docs/research/local_refinement_long_study.md) reproduces image-conditioned localization and Branch/Curve gains in two fine-tuning seeds. Coordinate-only A retains most of the benefit; confidence correction is not shown necessary. The next supported experiment is a bounded, fully unfrozen IXI transfer comparison against B0, with domain-shift and topology checks. This is a recommendation only: no new transfer, multiscale or combined A+C training has been launched. The dated priorities below remain context for the broader journal program.

## Current priorities

1. **Use the completed multi-dataset results to narrow the next work.** All15 training/prediction and evaluation tasks completed. Reporting-only export was recovered from saved evaluations. The [completed review](../docs/research/EXPERIMENT_REVIEW_20261023.md) shows that BRAVA node recovery improves but connectivity still fails the practical gate; IXI/SyntheticMRI geometry gains are small and mixed. Do not launch another full representation matrix by default.
2. **Diagnose BRAVA training/selection before more long runs.** IXI initialization is the best recovery arm, but default edge F1 is0.02099 and train-calibrated0.03696; the gate remains unmet. Scratch fails on monitored training data too. The [completed learnability checks](../docs/research/node_verifier_feasibility.md#brava-learnability-protocol) now show that scratch and IXI initialization can memorize eight BRAVA training patches (512 updates; edge CE, no augmentation). Next isolate population/augmentation, supervision and BN effects under controlled settings; neither an inherent inability to learn BRAVA nor a single cause of the historical failure is established.
3. **Keep K8 and audit adaptive selection/complex branches.** The completed frozen comparisons show only tiny positive IXI K24/C1 Curve F1 gains, with small negative adaptive/C1 differences on SyntheticMRI. The adaptive bank reduces output controls but under-recovers rare complexity classes. Only2 IXI and9 SyntheticMRI validation occurrences combine K8-difficult geometry with correctly localized endpoints; a strong accuracy claim is premature.
4. **Validate explicit degree-2 splitting before training it.** The target adapter exists. Original-branch loss weighting, oracle-node geometry learning, raw versus contracted graph evaluation and matching ambiguity remain prerequisites.
5. **Select a journal direction and stop expanding the matrix.** Use the decision gate below. Refine one supported approach and its necessary controls rather than start unrelated architectures, losses or source mixtures.

Current execution: [multi-dataset protocol](../docs/research/multidataset_continuous_graph_experiments.md), [submission record](../experiments/multidataset_continuous/submission.json).

```bash
# From the repository root; reads saved artifacts only.
python3 scripts/show_multidataset_graph.py
```

## Node-verifier feasibility update —23 October2026

The [completed verifier study](../docs/research/node_verifier_feasibility.md) tests frozen decoder-feature rescoring, separately from coordinate refinement. **Retain V1 (training-selected threshold/NMS policy) as the operational control; revise supervision/training before any end-to-end verifier experiment.** V3 contextual rescoring loses node and edge F1 against V1 on both IXI and SyntheticMRI at equal retained-node counts and descriptively matched precision. More context capacity or longer training is not the supported next step; inspect ambiguity/ignored labels, coverage-compatible representative selection and generalization first. Early validation discrimination gains show that the embeddings are not simply devoid of useful signal.

BRAVA V2/V3 transfer/training was not launched because the verifier gate failed. Completed eight-patch detector learnability checks do not pass the full-validation detector-recovery gate, justify a longer scratch run, or isolate CE versus focal as the cause. No new geometry or joint detector loss was introduced.

## Milestones

| Milestone | Evidence/status | Exit criterion |
|---|---|---|
| M1: reproducible references and physical metrics | Completed within documented scopes: IXI extraction/QC; BRAVA SWC preparation; dense references; common-v2 and metric failure controls | Preserve source provenance, boundaries, subject splits and original guides |
| M2: establish historical geometry/transfer controls | Completed: collapse, staged/frozen/local geometry, detached scratch, vascular/botanical transfer and metric comparisons | Retain successful and failed arms; identify the appropriate baseline for each new comparison |
| M3: diagnose IXI nodes and relations | Completed pilots 1–11: attribution, selection oracles, matching/calibration, image-aware relations, node loss and segmentation input | Separate endpoint loss, node selection, relation ranking and operating-point effects |
| M4: measure spline representation capacity | Completed: full IXI/BRAVA K8 audit; paired fixed/adaptive/piecewise fits; focused SyntheticMRI extension | Compare maximum error against geometric/control and graph complexity; never label oracle fitting as learned accuracy |
| M5: recover detectors and learn actual adaptive geometry | Initial15-arm batch completed; BRAVA detector-quality exit criterion remains unmet | Strong unchanged K8 controls, identical frozen-graph comparisons, source-disjoint validation and uncertainty |
| M6: train selected representations end to end | Pending M5 review | Valid recovered detector; verified target/head/loss contracts; matched optimization and final/best checkpoint reporting |
| M7: isolate node/refinement and relation sampling choices | Earlier refinement pilots and frozen verifier feasibility completed (negative verifier gate); BRAVA tiny-set learnability completed; broader matched refinement/relation stage pending | Separate coordinate/confidence effects; replace a controlled fraction of negatives without discarding positives |
| M8: lock journal method and final evaluation | Pending method decision | Fixed final comparison set, predefined tests, computational/annotation limitations and repeatability evidence |

## Method-selection gate

Prefer an approach only when evidence supports the following together:

- Learned shape improves against original dense paths, including difficult branches and correctly localized endpoints.
- Improvements survive comparison with K8, a larger fixed-K control and the same-graph chord baseline.
- Full reconstruction does not buy geometric coverage through excessive false edges, cycles or duplicate nodes.
- Control count, graph queries, parameters, runtime and stability are reported; internal geometric joins and detector nodes are counted separately.
- Validation-subject uncertainty and training-seed limitations are clear. A GT-fit improvement, an oracle or a selected visual example alone is insufficient.

**Provisional candidate:** internal piecewise geometry remains supported at GT level, but the completed frozen C1/C0 pilots do not establish a consistent cross-dataset learned benefit. Adaptive K offers fewer output controls with a larger/slower geometry model and selector limitations. K8 remains the strong learned control. Explicit graph splitting remains an experiment, not a rejected approach.

## Final journal comparison to prepare after selection

- A strong straight-edge RelationFormer/GrEVe control and a strong unchanged K8 model per dataset.
- The chosen continuous approach, fixed larger-K and adaptive/explicit-splitting controls needed to identify its contribution.
- Same predicted nodes/edges for geometry-only comparisons, followed by matched end-to-end training.
- Full-data BRAVA/IXI/SyntheticMRI development comparisons; source and low-label ablations only where justified by the paper claim.
- Native detection metrics plus physical node/edge, Branch/Curve, geometry, component/cycle, duplication and efficiency metrics.
- Subject-level uncertainty; a focused seed confirmation if feasible; final test evaluation only after the method/protocol is fixed.

Previously authorized historical test results remain valid evidence for their original comparisons. They must not guide new hyperparameter selection or be pooled with different target/metric populations.

## Deferred directions and dependencies

Multiscale transformer attention, new heatmaps or verifier expansion, mask+MRI concatenation, new joint losses, broad negative-mining sweeps and extra botanical source variants are not default next steps. Reconsider them only when a specific measured failure and a controlled experiment justify them. The IXI mask-input study already tests a distinct mask-to-graph diagnostic and must not be mistaken for deployable MRA performance.

Maintain these four consolidated files as the overview. Keep exact protocols, figures, CSV/JSON, configs, checkpoints and detailed historical reports at their stable linked locations. New results should update the [register](experiments.md) and this priority order without erasing earlier conclusions.

Historical roadmaps and scope details: [original ordered phases](../docs/research/vascular_graph_roadmap.md), [IXI diagnostic roadmap](../docs/research/IXI_graph_extraction_diagnostic_roadmap.md), [original implementation tracker](../docs/research/IMPLEMENTATION_TRACKER.md).

## Companion-project history and paper controls

[Image-to-Graph-Vessel](image-to-graph-vessel.md) is now included in this research collection: earlier 2D/3D transfer pipelines, source-ablation configurations and archived evaluation observations. Before using these as final paper controls, reconcile exact launch/config/checkpoint and split identities; preserve the distinction between botanical transfer, vascular-domain optimization and continuous geometry. This documentation adds no training priority or automatic sweep.

<!-- BEGIN CHEAP DIAGNOSTICS roadmap 20261026 -->
## Scoped priorities and evidence reconciliation —26 October2026

This update distinguishes completed evidence, the **other agents’ active work**, and ideas that remain untested. It adds no training campaign. Exact new cache-only relation evidence is appended to [image-aware relation study](../docs/research/metrics/9-ixi-image-aware-relation-learning.md); the [existing BN diagnosis](../docs/research/spline_learning/spline-collapse-diagnosis.md) now audits calibration provenance and specifies a training-only weight-frozen comparison. Isolated code/results: sibling worktree `gnbm-cheap-diagnostics-20261026`, branch `research/cheap-diagnostics-20261026`.

| Direction | Completed evidence | Status and next discriminative question |
|---|---|---|
| Local CNN A-xyz /A-cls /A | SyntheticMRI supports coordinate refinement; modern IXI standalone A-cls remains scientifically untested | **Audited28Oct:**1583591 smoke TIMEOUT (AF_UNIX path too long),1583592–1583594 CANCELLED; no completed planned-arm result. Repair short SCRATCH socket TMPDIR and revalidate gate before bounded recovery. Keep256: current contracted IXI max153, ten development patches exceed120. See Report3 dated audit |
| BN statistics | Modern inspected protocols use saved statistics; old collapse batch-stat intervention did not restore sensitivity. **30/32 archived legacy evaluations used validation-image adaptation** | New independent inference-only comparison uses existing IXI/SyntheticMRI references and **training data only**, unchanged cutoffs/validation IDs. Results pending; no BRAVA/refinement checkpoints touched. Validation BN adaptation must be labeled separately |
| Image-conditioned relations | Final65,600-update residuals overfit. New **cache-only** reanalysis shows genuine early2,000-update conditional ranking gains beyond both original and same-step token control (chord AP0.50156 vs original0.47235). Paired early edge gains also survive train-selected operating-point comparison | Strongest inexpensive unresolved **relation** question is whether a predefined short schedule or train-only stopping rule reproduces on an independent subject confirmation. Do not launch another long residual fit or validation-select an early winner from the reused cohort. New full-graph topology/Curve benefit at early checkpoints is not yet established |
| Multiscale C | Earlier pretrained two-level migration was functionally weak; the later layer2 C pathway receives image information but its3,000-update continuation did not improve reconstruction | Small matched **B0-versus-C from-scratch** remains untested; a B0 scratch run alone is not that comparison. Only consider a bounded matched diagnostic after selecting its question/initialization; no broad transformer sweep. See [pretrained pilot](../docs/research/multiscale_pretrained_feasibility.md) and [fine-feature study](../docs/research/fine_feature_architecture_study.md) |
| Combined A+C | No controlled combined model result | Defer until C independently demonstrates a useful/complementary signal beyond A. Different local feature scales are motivation, not evidence of additive benefit |
| Genuine adaptive K | GT complexity fits and a learned K-specific projection bank/separate selector were evaluated on frozen IXI/SyntheticMRI detectors; rare complex classes were underselected, with no consistent learned accuracy gain | Distinguish oracle GT K, compression of a predicted curve, and actual learned K selection. Rare-complexity/selector diagnostics and model-cost accounting remain useful; no new detector loss by default |
| Piecewise C1 | GT piecewise capacity is promising; four **uniform** neural C0/C1 sections were trained in frozen-head pilots, with tiny mixed dataset gains | Enforced C1 continuity is not the same as learned/error-adaptive breakpoints. **Adaptive piecewise geometry remains unvalidated as a learned reconstruction method**. Use dense-reference difficult branches and a working detector before an end-to-end claim |
| Explicit degree-2 graph representation | Error-bounded split-target machinery/GT fits exist and increase node/query/pair costs | **Owned by ongoing BRAVA representation agent** (1583544–1583553). Wait for existing original/split controls. Evaluate original-branch weighting, query budget and raw-versus-contracted graph quality; do not launch duplicate splitting/representation runs |
| Matching and relation supervision | Spatial/Hungarian oracles and calibration studies completed; matching mismatch alone did not explain poor relations. Raw spatially positive relations can involve Hungarian-unassigned queries | Preserve positive supervision. If revisited, first audit annotation ambiguity/duplicates; compare replacing a fixed10–25% negative budget with random/plausible unmatched pairs, independently of architecture/node-loss changes. This separate sampling comparison remains pending, not an endorsed cure |
| Metrics and topology | Physical1/2-mm matching, common-v2 Branch/Curve, controlled failure-matrix/core graph tests and source conventions are implemented | Keep native SMD for historical comparability **alongside** overprediction counts/edge precision, Branch/Curve, component/cycle errors and original dense references. Existing tests show detection/geometric coverage cannot replace connectivity. A dedicated SMD redundant-edge/nearby-false-edge stress study is not established by those tests; it remains a cheap optional diagnostic. Do not infer complete anatomy from incomplete BRAVA forests |

**Priority order:** finish the owned BRAVA/IXI batches without duplication; inspect training-only BN paired results; use the early relation ranking/generalization finding to decide whether a narrowly predefined independent confirmation is warranted; then choose one supported representation/architecture direction. C-from-scratch, A+C, adaptive geometry variants and further negative mining are not simultaneously authorized next campaigns. Method selection must separate GT fitting, learned geometry, full graph recovery and compute/graph complexity.

No held-out tests were evaluated or used to choose checkpoints/tolerances here. Shared-roadmap integration was applied only after checking that this file was unchanged since the ownership audit; no automatic queued job will rewrite this roadmap. If concurrent edits prevent future report appends, staged JSON integration blocks identify the remaining merge work.
<!-- END CHEAP DIAGNOSTICS roadmap 20261026 -->

<!-- BRAVA LEARNING PREFLIGHT 20261026 roadmap -->
## BRAVA K8 learning diagnostic launched — 26 October 2026

The completed BRAVA representation run is now followed by a **detector-only diagnosis**, retaining original K8. CPU audit exposed a sparse/boundary-heavy old training monitor; the new 160-patch monitor covers all 39 training subjects and complexity strata. Validation stays fixed at 256 patches/eight subjects; tiny learnability remains a separate reference. Saved weights, matched train/eval batching, training-only BN statistics, confidence filtering and actual augmentation alignment are being compared by diagnostic inference job **1598674**, with automatic collector **1598675**. No training or geometry sweep was launched. See the [existing BRAVA report](../docs/research/BRAVA_experiment_plan.md); do not choose a dominant cause or promote K24/C1/subdivision before these measurements arrive.


**Diagnostic smoke recovery:** job 1598674 stopped before prediction export because float32 `cdist` produced a nonzero self-distance in the checker; collector 1598675 was automatically cancelled. The checker now uses direct float64 `pdist` with a tighter 0.0001-mm tolerance. The actual loader passed all 256 rotation checks on CPU after this fix. The unchanged frozen-K8 inference is resubmitted as **1598848**, with automatic CPU collection **1598849** (`afterok:1598848`). No model, data, target or training recipe was changed. Earlier failed-job logs remain archived.

<!-- BEGIN BRAVA LEARNING PRIORITY 20261026 -->
## BRAVA detector diagnosis update — 26 October 2026

Completed the original-K8 representative learning/BN/augmentation diagnostic; see [BRAVA report](../docs/research/BRAVA_experiment_plan.md). New monitor: 160 training patches/39 subjects with complexity strata and 256 validation patches/eight subjects, plus the unchanged tiny reference. Measured finding: **representative cohort fitting remains weak**. Saved-BN batch-eight train/validation node F1: 0.1325/0.1183; validation edge F1 0.0089. Train-only recalibration node/edge F1: 0.1108/0.0078. No new training was launched. Keep K8 as the reference; geometry-capacity and subdivision claims remain gated on functional detection. Follow the report’s single-variable next-experiment recommendation; do not infer anatomical absence from incomplete SWC annotations or use brain masks as vessel labels.

<!-- END BRAVA LEARNING PRIORITY 20261026 -->

<!-- BEGIN N08 ROADMAP 20261026 -->
## N08 stopping decision — 26 October 2026

Completed the bounded, CPU-only GT-free query-selection diagnostic; see [N08](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026) and [experiment register](experiments.md#n08--gt-free-representative-selection--completed-26-october-2026). Local relation-profile selection is neutral versus count-matched confidence NMS across two SyntheticMRI A-xyz seeds; global support increases false edges/cycles. No additional heuristic sweep, learned selector or architecture was launched. Keep original inference as the reference; do not adopt NMS solely for its small Edge F1 gain, because short-branch and nearby-junction recovery decline. The oracle selection gap remains diagnostic rather than deployable. Prioritize understanding spatial node identity, query assignment and relation-label consistency before replacing relation tokens or adding unmatched negatives; N08 alone does not establish a causal training fix.

<!-- END N08 ROADMAP 20261026 -->


<!-- NATIVE2D IMPLEMENTATION 20261027 -->
## Native 2D extension, bounded implementation gate — 27 October 2026

User-authorized native2D correctness work is isolated from ongoing BRAVA training. The [existing 2D report](image-to-graph-vessel.md#native-2d-journal-extension--27-october-2026) records passed CPU checks, paired retinal/microscopy overlays, modern objective reuse and a pending short GPU/tiny-set test. This supplies implementation evidence only. Full training remains **DEFERRED**: recover missing OCTA-Synth images, establish patient/specimen-disjoint manifests and scale, audit projected crossings and segmentation artifacts, then predeclare B0/A-xyz and same-graph chord/K8 controls. Respect the global two-H100 ceiling; the smoke is serialized behind already submitted BRAVA chains. No new relation, matcher or architecture sweep and no full2D training submitted.


Native2D-01 accepted smoke `1637307` → `afterany` CPU collector `1637308`; existing BRAVA GPU chain gates execution. Full2D training remains deferred.


<!-- COMPLETED RESULTS REVIEW 20261028 -->
## Reviewed priorities after D08 and native 2D completion — 28 October 2026

**Completed evidence:** [D08 final review](../docs/research/BRAVA_experiment_plan.md#reviewed-final-d08-results--28-october-2026) compares 300 epochs /82,200 updates on the full BRAVA training split. SyntheticMRI initialization improves final validation node/edge F1 at 2 mm from **0.02544 /0.00110** to **0.35572 /0.06794**, but common-v2 Branch/Curve F1 remains **0.00647 /0.12174**. Scratch is weak on representative training data too. Do not interpret this as a successful recovered graph benchmark or an isolated spline-capacity result.

1. Keep the transferred K8 arm as the stronger measured BRAVA reference under this recipe. Next low-cost recommendation: use saved training caches to calibrate node/edge operating points, apply unchanged to validation, and separate raw proposal coverage, filtering and endpoint-correct adjacency. No new job or threshold search was run during this review. Do not extend scratch/geometry training automatically.
2. Preserve the node-selection bottleneck: raw 2 mm coverage65.53% becomes28.64% filtered recall;96.69% of missed GT edges lack a recovered endpoint pair. More spline capacity does not directly repair those missing endpoints. Relation ranking and cycle creation remain relevant, with annotation-relative interpretation.
3. [Native2D-01](image-to-graph-vessel.md#native-2d-completion-review--28-october-2026) passed CPU/CUDA, original-width architecture backward and a 1,000-update two-crop memorization test. This implementation gate is **complete**. Full 2D training remains **DEFERRED**, pending source/patient grouping, units/crossing QC and a fixed train/validation protocol; no held-out 2D result exists. Prepare B0/A-xyz and identical-graph chord/K8 controls only after those gates.
4. Retain the completed N08/R06/R07 stopping decisions. Neither D08 nor the2D smoke authorizes a new selector, relation head, mining sweep, multiscale architecture or automatic full training.

All 24 D08 production GPU tasks, six milestone collectors,2D smoke 1637307 and collector 1637308 completed successfully. Historical submitted/pending sections above are preserved as dated launch records. No training or inference was repeated for this reporting update.


<!-- D08 CALIBRATION ROADMAP 20261028 -->
## D08 operating-point gate measured — 28 October 2026

D08 CPU calibration completed: trained node cut0.45 gives validation node F1 0.36761; edge-only train-F1 cut0.425374478 gives edge F1 0.06892 on unchanged>0.5-node candidates. Ranking/AP is unchanged. Joint lower-node Edge/Curve F1 remains unavailable because rejected-query relations/new-edge controls were not cached. No GPU, inference or training was run. See the [existing BRAVA report](../docs/research/BRAVA_experiment_plan.md#d08-training-only-operating-point-audit--28-october-2026) and `experiments/d08_operating_points_20261028/`. No architecture/training sweep is authorized; review precision/recall and cycles before deciding on the minimal missing inference replay.

<!-- N07 STATUS AUDIT 20261028 -->
## Modern IXI execution/capacity gate — 28 October 2026

N07/D04 status is now **PARTIAL / RECOVERY**, not an active/unknown completed study. Both accepted chains stopped at smoke; the replacement failed on an overlong multiprocessing socket path and timed out. Scientific zero-shot/fine-tuning/collector jobs were cancelled before execution; no A-cls winner or efficacy conclusion is available. L06 remains partial, independently.

J00/J02 CPU audit is complete: current contracted IXI train/val10000/2649 targets max153/141, with five/five above120 and none above192. The fixed4096/512 N07 cohort also contains overflow targets. Zero degree2 objects were found; old239-node history is not the current capacity evidence. BRAVA8768/1796 targets max64/69 fit120. Preserve all targets and keep256 for N07 recovery; do not start the conditional full-IXI120/256 comparison on an invalid120 arm.

Next bounded action is the short SCRATCH TMPDIR repair and full correctness-gate recheck before existing matched component training. **No GPU jobs, architecture changes or new losses in this audit.** [Existing node-refinement report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026); numerical artifacts `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/n07_status_audit_20261028`.


<!-- BEGIN Q0 STATUS CORRECTION 20261028 -->
## Q0 execution/capacity correction —28 October2026

The dated October26 “ongoing IXI agent / pending BN” statements above are superseded by the [Q0 live/artifact audit](../docs/research/metrics/3-ixi-node-refinement-pilot.md#q0-ixi-status-and-contracted-target-capacity-audit--28-october-2026). D04 smoke1583591 timed out on AF_UNIX path length; all modern IXI production arms and collector cancelled. No B0/A-xyz/A-cls IXI fine-tuning result exists. L06 preserves completed SyntheticMRI BN inference, but IXI failed and paired collection remains incomplete. All corresponding old GPU jobs are terminal; reservations reconciled to zero.

Q0/J02 counted actual contracted targets: IXI full max153/141 and five/five train/val patches over120; modern subsets153/124 with one/one overflow, no degree-2 object nodes. Retain256 and defer Q120; BRAVA max69 is capacity-safe but its weak D08 graph recovery remains unchanged. Next independent queue task is2D-QC. An eventual original IXI smoke recovery must fix TMPDIR and pass gates before production training; no GPU or training was submitted in Q0.
<!-- END Q0 STATUS CORRECTION 20261028 -->


<!-- NATIVE2D HELDOUT AUTHORIZED 20261028 -->
## Native 2D held-out continuation — 28 October 2026

User-authorized controlled B0/A-xyz training supersedes the earlier full-training deferral once the gates pass. [Protocol and source audit](../docs/research/native2d_heldout_20261028.md): 17 AV-WIDE training images, 6 validation, 6 reserved test; image07 excluded for a self-loop; image-disjoint only. Full original-width 128² optimizer/overfit and PyTorch attention throughput gate precedes bounded training. No 3D architecture change, new matching, denoising or proposals. Measured status and accepted jobs are recorded in the existing registry’s Native2D-02 section.


<!-- N07 SOCKET RECOVERY 20261028 -->
### N07 recovery submitted —28October2026

The user authorized recovery after the CPU audit. Short SCRATCH multiprocessing TMPDIR repair passes five focused checks; existing chain **1653858→1653859→1653860_0–5(%1)→1653861** accepted with automatic evaluation/collection. Same4096/512 IXI cohort,256-query fine-tuning, losses and bounded budget; one H100 for this chain, two globally. Results remain pending the full GPU gate. Earlier no-training statements describe the prior audit, not this authorized submission. [Existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#user-authorized-socket-path-recovery--28-october-2026).


<!-- NATIVE2D HELDOUT FINAL 20261028 -->
The bounded original-width Native2D-02 pilot now has genuine final6,000-update results on506 image-disjoint AV-WIDE validation crops. Both arms retain no predictions at the fixed0.5 threshold: node/edge F1=0; no A-xyz or spline benefit established. Full-width learning/gradient/throughput gates passed. Stop without extending training. B0 saved all training/final evaluation before a redundant initialization-checkpoint evaluation timed out; A-xyz completed normally. [Completed interpretation and dataset readiness](../docs/research/native2d_heldout_20261028.md#completed-bounded-result-and-stopping-decision). REAVER diaphragm4/2-image targets are prepared separately with micron calibration; no additional training launched.


<!-- L08 PROTOCOL 20261028 -->
## L08 scratch curriculum —28October2026

New controlled question: can node-first supervision improve subsequent connectivity? Preserve modern120query B0 and all objectives except edge multiplier; both arms start from identical **random** weights, with fresh matched joint control. Run20epoch/15640update pilot first, then100epoch/two-seed confirmation only after predefined graph/AP/uncertainty gate. No new architecture or loss term. [Existing scratch-study protocol](../docs/research/fine_feature_architecture_study.md#l08--node-first-to-joint-scratch-b0-curriculum-28-october-2026); artifacts `/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/gnbm-node-first-curriculum-20261028/experiments/node_first_curriculum_20261028`. Current implementation/CPU checks do not constitute scientific efficacy results.
