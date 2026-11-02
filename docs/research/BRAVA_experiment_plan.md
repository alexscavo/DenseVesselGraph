# BRAVA continuous vascular graph reconstruction — staged experiments

21 October 2026. Stage 1 implementation and submission record; no model-performance results are claimed before collection. Prior dataset construction: [BRAVA_preparation.md](BRAVA_preparation.md). The nine test subjects remain closed to model evaluation, threshold selection and checkpoint selection.

## Questions and scope

1. How do existing SyntheticMRI and IXI models generalize to manually traced BRAVA anatomy without weight updates?
2. On exactly the same predicted graph, does learned continuous geometry recover annotated centerlines better than straight chords?
3. Can a stable scratch-trained BRAVA continuous model learn useful connectivity, and does coordinate-only local CNN refinement help?
4. What geometric accuracy/node-count trade-offs exist in the supplied annotations, before making model-level query-efficiency claims?

Stage 1 submits these bounded comparisons and a CPU GT representation analysis. Independently trained straight controls, source-transfer comparisons, scarce-annotation training and a second seed remain **Stage 2, unsubmitted**.

## Dataset and annotation audit

Model root: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057/transfer_splines_k8`.

|Split|Subjects|Patches|Permitted use now|
|---|---:|---:|---|
|Train|39|8,768|Optimization; fixed 96-patch monitoring subset|
|Validation|8|1,796|Zero-shot evaluation, development and checkpoint selection|
|Test|9|2,043|Membership metadata only; no new target/image reads or model evaluation|

`READY`, `manifest.json`, `brava_semantics.json`, `generation_config.json`, source mappings, extraction QC and per-subject completion markers establish provenance. All 56 paired subjects/111 valid trace variants passed the repaired preparation audit. Primary original traces are used once per subject; BH0027 uses its valid ColorCoded counterpart because the original lacks a parent field. Other complete variants are exported but are not extra independent training subjects. No trace variant crosses the subject split. Six unpaired SWC subjects are not part of the image benchmark.

Coordinates: SWC uses the fixed0.62-unit tracer lattice; map to native voxels `[x/.62, Ny-1-z/.62, Nz-1+y/.62]`, with explicit x reflections for BG0014/BG0019–BG0022. Source-file hashes pin those corrections. The NIfTI affine supplies physical geometry. Targets use normalized native DHW with `metric = affine[:3,:3] @ diag(64,64,64)`. Native voxel-cell clipping and the existing boundary policy remain unchanged.

**`seg/` contains brain masks, never vessel labels.** The loader reads this required legacy slot, but the predictor receives only `image`; the criterion uses nodes/edges/spline targets, and the new evaluator never reads masks or invokes vessel-support/Dice categories. Masks are not concatenated with MRA or used as vessel supervision.

The validated target dataset is not regenerated. The read-only development audit checks split integrity, every train/validation cache's finite values/shapes/capacity, original path endpoints, source affine metrics and file readiness. On the47 development subjects, dense-source coordinates are checked against transformed SWC coordinates; junction/adaptive guide segments must belong to original SWC segments, and component/cycle counts must match the dense graph. Source graph hashes must agree with the target manifest. Actual loader samples exercise VTP/cache correspondence, MRI normalization and spline-aware rotation. Test membership/QC metadata are inspected without loading test images or targets.

### Target representation convention

Dense exports preserve every SWC vertex/parent link. Junction exports contract degree-two chains while preserving their original polylines. Curvature-adaptive exports add subdivisions under the established radius-scaled simplifier, retaining unsmoothed original paths; no vessel-lumen criterion is available.

**The supplied model-compatible continuous cache contracts these guides to shared junction/end/crop anchors.** Intermediate adaptive degree-two samples are geometry, not necessarily object-query targets. Each edge has8 endpoint-constrained cubic B-spline controls and its original `full_paths`/`path_offsets`. Stage 1 trains those existing caches. Calling this a dense-versus-adaptive object-node experiment would be incorrect. Any alternative node sampling requires a separately named Stage 2 comparison.

## Checks completed before submission

Four focused tests passed: paired configuration identity, zero-initialized coordinate-only refinement, unchanged connectivity in a crafted curved graph, and reproducible full-population ordering. Syntax and Slurm shell checks passed. All three source checkpoints loaded strictly, and both fresh training arms had identical shared parameter hashes.

The local read-only preflight passed: all47 development subjects ×3 graph representations preserved source SWC segments/topology and physical transforms;12 target patches passed cache/endpoint/metric checks. Actual native loader reads passed for eight validation patches (one per subject) and two augmented training samples. No test targets were read. Evidence: `docs/research/evidence/brava_stage1_20261021/local-preflight.json`. Optional local reader packages were incomplete; using the fixed cluster archive resolved the import problems without changing data or dependencies in the shared environment.

The submitted full-cache audit, real GPU forward/backward, tiny end-to-end prediction/evaluation/collection and throughput benchmark are **pending gates**, not claimed successful checks. Training cannot start unless these gates succeed. The exact common update count will be chosen automatically by the benchmark rule below.

## Stage 1A — zero-shot models

Exact paths/hashes: `experiments/brava_graph_study/plan.json`; CPU strict-load inventory: `checkpoint-inventory.json` in the same directory.

|Label|Candidate checkpoint|Queries|Interpretation|
|---|---|---:|---|
|synthetic_spline|`plant_generalization_20261014_da_b32_w1_restart/runs/vessels_spline/best.pt`|120|SyntheticMRI-only graph pretraining and spline fine-tuning|
|synthetic_straight|Same source study, `runs/vessels_straight/best.pt`|120|Current compatible straight-edge control, not a verified exact original GrEVe checkpoint|
|ixi_spline|Original delayed-spline IXI parent identified by the full-query pilot manifest|256|IXI-trained continuous model, no node-refinement overlay|

Source study root is `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/`. Models must load strictly from their own saved configurations, retain their original query structure and decoded node/edge decisions, and use compatible one-channel 3D images and8-control B-splines. Only evaluation dataset roots, batching and augmentation are replaced. No relation adapter or local-refinement overlay is grafted onto zero-shot checkpoints. Cross-model results are descriptive: source histories and query capacities differ.

Use source MRI normalization/foreground centering, the existing prepared MRA intensity scale, frozen saved BatchNorm state, augmentation off and inference batch1. There is no BN recalibration, GT query selection or threshold tuning in the primary comparison. Source node/edge thresholds (including argmax semantics) are preserved in prediction metadata. Any later BRAVA-validation calibration must be reported separately and must not use test data.

Full validation predictions are cached once. Besides decoded curves/scores, retain all raw node coordinates/confidences and all relation scores among normally retained nodes, with query IDs/pair ordering. Normal-decoder edge-score parity and evaluation-state invariance are asserted. These caches support metrics and same-graph comparisons without another model pass; no enormous all-pair image-feature tensor is stored.

## Stage 1B — same-predicted-graph chords versus curves

For every zero-shot model and every saved BRAVA baseline checkpoint, evaluate:

- Spline geometry from the learned checkpoint.
- Straight chords after removing curve controls from the **same nodes and edges**.

Node/edge counts, positional TP/FP/FN, graph topology and common-v2 Branch connectivity must remain equal. Curve F1, geometric distance and tortuosity may differ. The evaluator explicitly checks connectivity parity.

For direct predicted edges whose endpoints recover a GT adjacency at1/2mm, compare symmetric arc-sampled point-to-polyline distance of both chord and learned curve against the exact corresponding original annotated path. Both geometry versions use the identical recovered-connection set and predicted endpoints. GT never supplies coordinates or edges to the prediction. Save per-connection, per-patch, per-subject and aggregate tables; zero eligible connections are NA, not zero error. Native SMD is retained with its straight-edge point-cloud convention and is not relabeled as continuous-curve SMD.

Fixed examples from the first three validation subjects show annotated full paths, predicted curves, same-graph chords and reference-relative matched/unmatched edges. Examples are selected in manifest order, without outcome selection.

## Stage 1C — two matched BRAVA-only continuous baselines

|Arm|Node prediction|Image/backbone/transformer/relation|Spline head|
|---|---|---|---|
|standard|Existing classifier/coordinate predictor|Trainable|Trainable|
|coordinate_refinement|Same plus early-layer0 CNN coordinate residual|Trainable|Trainable|

Both start **from the same fresh random seed211039**, with no external source checkpoint. Shared initial parameter hashes are checked, and zero-initialized refinement must leave initial outputs identical. This supplies the BRAVA-only Stage 2 reference without conflating it with vascular pretraining.

Architecture: stable single-channel SE-ResNet/3D RelationFormer,256 object queries/two relation tokens, existing residual B-spline head with8 controls and local CNN geometry features. The local node module samples the established3×3×3 layer0 neighborhood,4-voxel radius. Its new `refine_classification: false` option leaves classification logits untouched; default `true` preserves all prior experiments. Its existing5-output layer is retained for compatibility, but the two classification residual outputs are unused in this arm. Added local-module parameters are counted explicitly. No verification head, heatmap, new relation adapter or transformer modification is introduced.

Stable graph recipe: weighted node CE `[.25,.75]`, existing node/box/cardinality coefficients, edge focal gamma2 with matched-only candidate population and unchanged balancing/caps. No node focal, unmatched negatives or new hard-mining objective. Actual Hungarian matching uses the final coordinates and unchanged classification outputs. Shared spline weight1.4739989612791453 and residual-control supervision reuse the established stable geometry recipe. Spline gradients detach detector tokens/endpoints and sampled CNN features (`graph_only` route), while graph objectives train the full detector. Geometry-head parameters receive finite nonzero gradients. No image/segmentation loss is introduced.

Training uses every one of the8,768 training patches in common seeded permutation order, batch8/workers0, exact quarter-turn graph/spline-aware rotations, existing Gaussian image noise and no zoom. Both arms share AdamW7e-5, inherited betas/epsilon/weight decay, one full-pass linear warmup followed by polynomial power0.9 over the common budget. No early stopping or validation-based hyperparameter tuning.

### Throughput-selected bounded budget

Benchmark3 warm-up plus32 timed full-network updates per arm, discarded afterward. Let `P=ceil(8768/8)=1096` and `s` be the slower measured seconds/update. Select:

`updates = min(40P, max(20P, floor(10800/(1.2s)/100)*100))`.

Thus the common budget is **21,920–43,840 updates**, covering20–40 complete-data passes in draw count. The target is roughly3 hours with a safety factor; if the minimum budget's conservative estimate exceeds4.5 hours, block training rather than silently reduce subjects/updates. Actual throughput, parameter counts and chosen budget are written automatically to `benchmark/`. Each training arm requests **6 hours**; the internal deadline saves an interrupted checkpoint and fails rather than evaluating mismatched incomplete runs. This is a learning-curve baseline, not proof of convergence.

Checkpoints:1k,5k,10k,20k and common final, deduplicated; recovery snapshot every1k. All predefined checkpoints receive full validation evaluation and the same fixed96-training-patch monitor. No per-step validation metric selection affects optimization. Primary comparison is matched final updates. Secondary checkpoint selection per arm uses full-eight-subject validation Branch F1, then Curve F1, positional edge F1 at2mm, then earlier update on ties. Learning curves retain both training and validation populations separately.

## Evaluation definitions and incomplete-annotation limitations

Use unchanged one-to-one physical1/2mm positional matching and common-v2 definitions (1mm branch-anchor/curve support,0.25mm curve sampling). Report node precision/recall/F1, matched localization mean/p95, raw-query recovery, duplicates per recovered GT node, high-confidence duplicates at0.9, node-count bias/MAE, edge TP/FP/FN/F1, endpoints-recovered GT edges, conditional direct relation recall, relation AP within each patch's retained/conditional populations, Branch/Curve F1, component/cycle errors and native detection AP/AR/SMD. Save raw counts and candidate prevalence; per-patch AP must not be confused with pooled AP. Native detection caps remain checkpoint-specific for zero-shot models; positional/common metrics use all decoded predictions.

**All false-positive/false-negative terminology is relative to available annotations.** A predicted vessel absent from SWC is unconfirmed, not proven anatomically wrong. Do not prune it merely because it lies outside traced structures, or use the brain mask to label it as vessel-supported/off-vessel. SWC parent-link forests do not prove the actual vasculature has no additional connections or cycles. Betti errors measure disagreement with those annotations. Manual centerlines provide geometric references but do not certify complete coverage or exact registration. Graph-positive patch selection is not a whole-volume deployment evaluation.

Eight validation subjects are the independent development units. Use1,000 paired subject-bootstrap resamples, seed211039, for coordinate-refinement differences and same-graph Curve-F1 differences. These are exploratory intervals from one training seed, with few subjects and no multiple-comparison correction. Per-connection geometry errors and source-specific candidate populations remain explicit. Compare geometry with connectivity and cycle errors; geometric coverage alone does not establish faithful graph reconstruction.

## CPU-only GT representation analysis

On the47 development subjects, compare dense straight graphs, junction-only chords, adaptive chords and8-control least-squares splines on junction anchors against original dense SWC polylines. This is annotation approximation, not learned-model output. Report object-node/edge counts, internal control counts, Branch/Curve F1, geometric guide distance and component/cycle preservation. Original full guide segments are verified to belong to dense SWC edges. Full-volume reference arrays created for this analysis are evaluation artifacts on SCRATCH; the validated dataset is unchanged. Model-target patch count statistics are reported separately from full-volume representation counts.

Fewer annotation anchors do **not** demonstrate lower learned-query requirements. Stage 2 must measure reconstruction accuracy against controlled object-query budgets on a common population, preserving capacity-limited cases rather than silently dropping them.

## Stage 0 gates, resources and automation

Local focused tests cover paired configurations/order, zero/coordinate-only refinement behavior and same-graph geometry against a crafted curved annotation without any mask. CPU checkpoint checks load source states strictly and verify common fresh detector weights. Local actual-reader checks use the same MedPy/SimpleITK readers as jobs. The protected module alone lacks MedPy; local checks use the existing installed readers, while jobs stage the existing fixed dependency archive. No package upgrade or loader substitution is introduced.

Full CPU dataset audit gates GPU smoke. GPU smoke performs two real training updates in both arms, finite-gradient/spline-gradient checks, identical fresh shared-state/output checks and tiny prediction passes through all three source checkpoints and both trained arms. Tiny CPU evaluation and collection must succeed before benchmarking or full jobs. Predictions verify batch-one cache/decoder parity and unchanged frozen state.

GPU sequence: smoke → benchmark → zero-shot array (`0-2%2`) → paired training (`0-1%2`) → trained-checkpoint prediction (`0-1%2`). CPU zero-shot evaluation runs alongside training; CPU GT analysis is independent after the dataset audit. CPU trained evaluations follow corresponding prediction tasks; final collection depends on all CPU evaluations and GT analysis. Failed prerequisites cancel/block dependent work. This serialization caps this work at two H100s. Existing limited-botanical GPU jobs are respected through an additional initial dependency on all active/pending H100 jobs, even though it may delay BRAVA execution.

Large outputs/runtime staging live on SCRATCH, with short job-specific temporary paths to avoid the previous AF_UNIX limit. Source data, old checkpoints and experiments are preserved. No unsupported `--mem` option, full300-epoch run or Stage 2 training sweep is submitted.

## Stage 2 — prepared matrix, no training submitted

|Comparison|Locked controls|Remaining choice after Stage 1|
|---|---|---|
|Independent straight versus spline|Same shared anchors/connectivity, detector initialization, image input, splits/order/updates and graph losses|Straight has frozen zero-residual geometry; spline has trainable curve parameters and explicit geometry loss; report trainable/allocated counts and gradient route|
|A: BRAVA-only|Current fresh Stage 1 arm, chosen refinement policy|Confirm convergence/common target budget before treating it as final reference|
|B: SyntheticMRI transfer|Compatible source checkpoint, identical BRAVA fine-tuning IDs/budget|Preserve source query count or document common query expansion before all arms; list pretraining and target updates separately|
|C: botanical transfer|Same BRAVA fine-tuning controls|A current plants+SyntheticMRI mixed checkpoint is available, but is not botanical-only; verify a pure botanical-compatible initializer or label the mixed-source comparison honestly|
|25% versus100% annotations|Deterministic10-of39 training subjects versus all39; same8 validation subjects|Record actual patch counts, equal target exposure versus equal total updates, and coverage of sites where metadata supports it|
|Object-query efficiency|Same annotated reference paths, image population and evaluation; no capacity-based outcome filtering|Choose query budgets after measuring target maxima; vary graph sampling separately from geometry modeling|
|Second seed|Identical selected Stage 1 protocol|Confirm only the promising comparison|

No new source dataset is created. Repeated BRAVA training samples across stages represent extra in-domain optimization, not independent source data. Current biological mixed-source pretraining also changes vascular exposure and possibly its auxiliary domain objective; these must not be hidden in a data-only transfer claim. Equal fine-tuning budgets do not imply equal overall training compute. Pure botanical checkpoint compatibility and exact legacy GrEVe compatibility remain unverified, so neither is silently substituted into Stage 1.

Prepared scarce-annotation membership: `experiments/brava_graph_study/stage2-plan.json` contains a deterministic10-subject subset with **2,243 training patches**, the unchanged validation IDs and an explicit `training_allowed: false` flag. For a future matched transfer study, prefer the source study's graph-only pretraining checkpoints (`pretrain_vessels` / accurately labeled `pretrain_mixed`) rather than silently comparing different amounts of subsequent SyntheticMRI fine-tuning. Their pretraining budgets must be read from saved summaries before launching. A valid detector-only transfer can retain120 source object embeddings, add136 common seeded embeddings, move both relation embeddings to256–257 and initialize the spline head/refinement identically and freshly across A/B/C; strict compatible shared detector tensors can transfer. This policy and any domain-loss/source-exposure confound require review before Stage 2. No source training or BRAVA transfer job is submitted now.

## Outputs and retrieval

Implementation: `scripts/{brava_graph_study,audit_brava_study,evaluate_brava_study,collect_brava_study}.py`, focused tests, existing model refinement with a backward-compatible coordinate-only flag, and `cluster/jean_zay/brava_study_{cpu,gpu}.slurm`.

Manifests: `experiments/brava_graph_study/{plan,submission,checkpoint-inventory}.json`; output root is recorded after submission. Large outputs: `audit/`, `representations/`, `benchmark/`, `runs/<arm>/`, `predictions/<model>/<checkpoint>/`, `evaluation/<model>/<checkpoint>/`, `results/`. Per-subject/per-patch and same-connection geometry CSVs are retained. Compact comparative results, bootstrap intervals, checkpoint selection and learning curves are copied to `docs/research/evidence/brava_stage1_20261021/`; the collector appends results here.

```bash
# Read completed results immediately; no inference, training or waiting.
python3 scripts/collect_brava_study.py --show
# Rebuild summaries only if needed after all evaluations have finished:
python3 scripts/collect_brava_study.py --root <submitted-output-root>
```

No scientific model result is asserted by dataset readiness or job acceptance. Failure of a source checkpoint or QC gate blocks affected downstream work rather than introducing an arbitrary transform, target change or silent checkpoint substitution.

## Accepted Stage 1 jobs

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_graph_study_20261021-102438`.

|Stage|Job|Dependency|Wall time|
|---|---|---|---|
|dataset_audit|1374050|None|02:00:00|
|GT_representation_analysis|1374051|afterok:1374050|04:00:00|
|gpu_smoke|1374052|afterok:1374050,afterany:1372129:1372131:1372132:1372133:1372134:1373095|00:45:00|
|cpu_smoke|1374053|afterok:1374052|00:45:00|
|throughput_benchmark|1374054|afterok:1374053|00:30:00|
|zero_shot_prediction|1374055|afterok:1374054|01:30:00|
|zero_shot_evaluation|1374056|aftercorr:1374055|04:00:00|
|training|1374057|afterok:1374055|06:00:00|
|trained_prediction|1374058|afterok:1374057|03:00:00|
|trained_evaluation|1374059|aftercorr:1374058|08:00:00|
|collection|1374060|afterok:1374056:1374059:1374051|02:00:00|

GPU stages wait for all existing H100 jobs: 1372129, 1372131, 1372132, 1372133, 1372134, 1373095. No Stage 2 training was submitted.

## Completion review — 22 October 2026

Both training arms completed23,500 updates each (batch8; approximately21.44 full-population passes), in2h31m and2h26m including job overhead. All three zero-shot evaluations and both five-checkpoint trained evaluations completed on1,796 validation patches/eight subjects. Full development audit passed10,564 patches and141 volume-representation checks; no test targets were evaluated.

All table values are percentages except predicted/correct edge counts. Trained rows use the common final checkpoint, not separately selected best checkpoints. Branch/Curve metrics use the existing common-v2 convention (1 mm geometry tolerance); positional node/edge metrics below use2 mm.

|Model|Node recall|Edge F1|Branch F1|Curve F1 spline|Curve F1 same-graph chord|Predicted edges|Correct edges|
|---|---:|---:|---:|---:|---:|---:|---:|
|synthetic_spline|3.446|0.067|0.000|1.441|1.398|10526|10|
|synthetic_straight|3.567|0.076|0.000|1.667|1.667|30349|19|
|ixi_spline|23.095|2.687|0.243|5.802|5.538|8536|376|
|standard|10.784|0.041|0.000|0.034|0.032|230|4|
|coordinate_refinement|7.955|0.010|0.000|0.013|0.013|69|1|

### Interpretation and outstanding work

The IXI zero-shot checkpoint is the strongest measured starting point here, although source histories and query capacities differ. Both scratch-trained models predict very few edges; the final standard model predicts230 edges across validation (4 correct), and coordinate refinement69 (1 correct). Their geometry comparison therefore cannot meaningfully establish the value of continuous representation. These bounded scratch runs do not establish convergence or rule out refinement.

Coordinate refinement increases all-query2-mm coverage from32.19% to34.68%, while post-filter node recall drops from10.78% to7.95%. Thus the measured coverage change did not yield better decoded graphs. These are single-seed, annotation-relative measurements, without the planned paired-bootstrap collector intervals.

GT representation analysis job1374051 failed with `ValueError: invalid edge polyline` in common-v2 degree-two contraction. Final collector1374060 was automatically cancelled because that prerequisite failed. The underlying model prediction/evaluation caches and checkpoints are intact. No training rerun is required to recover collection; the malformed full-volume reference polyline needs diagnosis. No recovery jobs or Stage2 training were submitted in this review.

Output root: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_graph_study_20261021-102438`.


## BRAVA Stage 1 recovery — 22 October 2026

The one-point full-volume reference path was traced to a coincident SWC parent/child pair. The evaluation adapter now writes both coincident endpoints while retaining the original link and geometry. The unchanged common-v2 metric accepts it; a focused regression test and replay of the failing BH0009 edge pass. No source data, model predictions, checkpoints or training were changed.

CPU recovery jobs: 1421288 (GT representation analysis,4h allocation) → 1421289 (collector,2h allocation; afterok dependency). Results are pending. The collector will replace this section with model/representation tables and paired subject-bootstrap intervals in both complete review files.

Retrieve without rerunning computation: `python3 scripts/collect_brava_study.py --show`.
## Completed Stage 1 results

All scores are relative to available manual annotations. Reference-unmatched predictions are not confirmed anatomical false positives. Test subjects were not evaluated. Primary trained comparison uses matched final updates; secondary checkpoints are selected on the full eight-subject validation split.

|Model|Queries|Node recall 2 mm|Edge F1 2 mm|Branch F1|Curve F1 spline|Curve F1 same-graph chord|Component MAE|Cycle MAE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|synthetic_spline|120|0.0345|0.0007|0.0000|0.0144|0.0140|106.3252|0.5741|
|synthetic_straight|120|0.0357|0.0008|0.0000|0.0167|0.0167|97.9722|3.1609|
|ixi_spline|256|0.2309|0.0269|0.0024|0.0580|0.0554|7.6464|0.4343|
|standard|256|0.1078|0.0004|0.0000|0.0003|0.0003|17.1804|0.0011|
|coordinate_refinement|256|0.0795|0.0001|0.0000|0.0001|0.0001|12.4360|0.0000|

|Model|Recovered connections at 2 mm|Chord error mm|Spline error mm|
|---|---:|---:|---:|
|synthetic_spline|10|0.8771|0.8694|
|synthetic_straight|19|0.8518|0.8518|
|ixi_spline|376|0.9874|0.9780|
|standard|4|0.9683|0.9744|
|coordinate_refinement|1|0.8298|0.8401|

### GT representation analysis (not learned predictions)

All geometry is evaluated against original dense development centerlines. Spline rows are oracle fits, not model predictions; fewer anchors alone do not demonstrate learned query efficiency.

|Split|Representation|Mean nodes|Mean edges|Branch F1|Curve F1|Guide distance mm|Component error|Cycle error|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|train|dense|3121.1538|3120.1538|1.0000|1.0000|0.0000|0.0000|0.0000|
|train|junction_only|203.4359|202.4359|1.0000|0.1900|3.0951|0.0000|0.0000|
|train|adaptive|2512.1026|2511.1026|1.0000|0.9992|0.0462|0.0000|0.0000|
|train|junction_spline_k8|203.4359|202.4359|1.0000|0.8385|0.2911|0.0000|0.0000|
|val|dense|3143.3750|3142.3750|1.0000|1.0000|0.0000|0.0000|0.0000|
|val|junction_only|203.0000|202.0000|1.0000|0.1866|3.1478|0.0000|0.0000|
|val|adaptive|2471.3750|2470.3750|1.0000|0.9991|0.0492|0.0000|0.0000|
|val|junction_spline_k8|203.0000|202.0000|1.0000|0.8413|0.2903|0.0000|0.0000|

### Paired subject-bootstrap uncertainty

95% percentile intervals;1,000 resamples of eight subjects, seed211039. Values are differences on the0–1 scale. These intervals do not measure training-seed uncertainty.

|Comparison|Metric (2 mm positional protocol)|95% interval|
|---|---|---|
|coordinate_refinement minus standard|node_precision|[-0.010484, +0.001837]|
|coordinate_refinement minus standard|node_recall|[-0.030636, -0.025552]|
|coordinate_refinement minus standard|f1|[-0.000789, +0.000196]|
|coordinate_refinement minus standard|conditional_recall|[-0.042105, +0.017504]|
|coordinate_refinement minus standard|common_branch_f1|[+0.000000, +0.000000]|
|coordinate_refinement minus standard|common_curve_f1|[-0.000374, -0.000019]|
|coordinate_refinement minus standard|beta0_mae|[-6.263949, -2.617365]|
|coordinate_refinement minus standard|beta1_mae|[-0.002703, +0.000000]|
|coordinate_refinement minus standard|duplicates_per_recovered_gt|[+0.020069, +0.033597]|
|synthetic_spline: same-graph spline minus chord|common_curve_f1|[+0.000266, +0.000606]|
|synthetic_straight: same-graph spline minus chord|common_curve_f1|[+0.000000, +0.000000]|
|ixi_spline: same-graph spline minus chord|common_curve_f1|[+0.001993, +0.003249]|
|standard: same-graph spline minus chord|common_curve_f1|[+0.000006, +0.000031]|
|coordinate_refinement: same-graph spline minus chord|common_curve_f1|[-0.000024, +0.000007]|

### Recovery and interpretation

The full-volume analysis originally failed because the source reader deduplicated a coincident SWC parent/child pair to a one-point path. The analysis now explicitly writes both coincident endpoints, retaining the annotated edge without inventing spatial extent, changing common-v2 metrics, or modifying training data. Completed subject analyses are checkpointed for safe CPU-only resumption. Original failure1374051 and cancelled collector1374060 remain in accounting; recovery job IDs are in `experiments/brava_graph_study/recovery.json`.
Both scratch arms completed23,500 updates on8,768 training patches, with all1,796 validation patches evaluated. IXI zero-shot is the strongest measured starting point; its differing pretraining history prevents a controlled causal claim about initialization. Scratch training yields too few recovered connections to establish whether learned spline geometry or refinement is beneficial. Refinement increases raw-query coverage but reduces filtered node recall in this pilot. These runs do not establish convergence. No new inference, training or Stage2 experiment was needed for reporting recovery.

Recovery jobs: 1421288 (CPU GT analysis) → 1421289 (CPU collection, afterok). No training or inference rerun.

### Interpretation boundaries

Same-graph geometry comparisons share identical nodes, edges and scores; differences in Curve F1/guide distance isolate geometry. Branch connectivity must be unchanged. Recovered-connection distance tables quantify spline/chord error on exactly the same direct GT connections, without GT input features or GT endpoint replacement.
The two scratch-trained arms isolate the enabled coordinate refinement module, including its added parameters; logits receive no local residual. Both use the full same training population and matched updates. Their initial shared weights and outputs are checked, but these single-seed development results do not establish convergence or robustness across seeds.
Zero-shot models differ in source training history and may have different query capacities; they are starting points, not an isolated causal comparison of source domains. Source straight-versus-spline checkpoints are independently trained current-compatible controls, not a verified rerun of the exact original GrEVe checkpoint.
Inspect per-subject errors, duplicates, conditional recall, native detection and same-connection geometry alongside coverage. Better SMD/Curve F1 or fewer cycles alone cannot establish correct anatomy. Native SMD samples straight edges; continuous geometry is measured separately. No source-transfer, separate straight-training or query-budget arm was automatically launched.

Output root: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/brava_graph_study_20261021-102438`.


## Fully trainable representation follow-up — 26 October 2026

**Completed.** Both diverse-subject learnability tests, both fully trainable 4,000-update pilots and all six evaluations finished successfully. Degree-2 subdivision improves partial curve coverage and GT-conditioned geometry, but does **not** establish useful BRAVA detector reconstruction: all arms recover zero complete common-reference branches at the established 1-mm protocol. K24 and fixed-section C1 offer substantially better GT fitting capacity but no convincing inference gain over K8 in this pilot.

**Recommendation:** retain the original graph with K8 as the controlled baseline for diagnosing BRAVA detector learning and subsequent refinement. Retain `split_K8` as the promising alternative for a targeted follow-up once detection is functional; do not promote it solely on Curve F1, because it increases cycle errors and predicted graph size. Defer K24/C1 capacity expansion and adaptive-K selection until learned geometry can be evaluated on reliably recovered branches. No longer training, new architecture, or additional sweep was launched.

This is a new bounded follow-up, not completion of the older unsubmitted Stage 2 matrix. Implementation is isolated in `../Graph-Native-Betti-Matching-Representation-BRAVA-20261026`, branch `brava-representation-20261026`, based on commit `5b3274d` plus the hash-recorded source snapshot. The primary repository's model/training files and other agents' uncommitted changes were not edited. Numeric artifacts are accessible through [experiments/brava_representation_20261026](../../experiments/brava_representation_20261026/), a link to the isolated experiment directory.

### Protocol and target audit

The existing [BRAVA preparation audit](BRAVA_preparation.md) is reused. SWC coordinates use the validated 0.62-unit tracer lattice transform, not direct NIfTI-world interpretation; the five explicit subject reflections and input hashes remain unchanged. The original 39/8/9 subject split is preserved. No test targets or images were opened. The brain mask is a legacy loader input, not a vessel label. Incomplete annotations make precision and topology errors annotation-relative.

The pilot contains 32 hash-selected patches per development subject: 1,248 training patches from 39 subjects and 256 validation patches from eight subjects. This is a subject-balanced bounded cohort, not full-population performance. Audit job 1584757 verified all 1,504 targets:

- 24,863 original nodes become 30,446 nodes after meaningful degree-2 subdivision; 1,159 patches change. Maximum nodes increase from 62 to 69, below the unchanged 120-query budget.
- All 15,718 original paths and graph components/cycles are preserved; added nodes have degree two. The existing error-driven K8 splitter uses 0.5-mm tolerance, minimum 2-mm arc length and at most 32 sections, with adjacent merging. No unresolved branches remain in this cohort.
- `boundary` records clipping provenance, not every geometric face point. There are 7,595 original nodes outside the sigmoid voxel-center range, including 270 without this flag. All remain valid within the existing voxel-cell bounds. Maximum physical clamp displacement is 0.4402 mm. Targets were neither clamped nor relabeled; this limitation cannot explain multi-millimetre assignment errors.

GT fitting bounds are independent of learned detector performance. Values below are sampled symmetric maximum curve errors, not the mean-distance metric used in the learned-geometry table.

| Validation fitting diagnostic | K8 | K24 | Fixed-section C1 |
|---|---:|---:|---:|
| Median / p95 maximum error, all 2,698 branches (mm) | 0.1763 / 1.0183 | 0.0479 / 0.3048 | 0.0431 / 0.2791 |
| Mean maximum error, 141 difficult branches (mm) | 1.5710 | 0.3971 | 0.3604 |

Full audit and split-specific fitting summaries: `target-audit.json`, `target-summary.json`. The earlier full development audit covered 10,564 patches; the new subdivision checks cover only the bounded cohort above.

### Controlled training and correctness

All shared weights start from the clean SyntheticMRI epoch-95 checkpoint:
`experiments/fine_feature_architecture/runs/baseline/best.pt`, SHA256
`d4bc78ad3eb7c1c29c9be8ec3e2927e88f2f45e6f34a3cb8be9ade5fd9d2431c`.
The recipe is node **softmax focal gamma 2**, weights `[1, 1]`, no focal curriculum; edge **cross-entropy**, weights `[1, 1]`, established ratio-upsample balancing. The Hungarian costs remain class 2 and normalized L1 node 5. The relation architecture, targets outside the explicit subdivision change, graph losses and existing `graph_only` spline route remain fixed. No edge focal or node refinement is added.

There are two detector trajectories: original graph and subdivided graph. Backbone, transformer and active prediction heads are trainable throughout. For the original graph, K8, K24 and C1 heads train independently on the same evolving detector features. This is valid because spline loss sends exactly zero gradient to the detector under `graph_only`; there is no global gradient clipping to couple the heads. These are not frozen-detector head experiments. Shared detector metrics are therefore identical across K8/K24/C1 **by design**, not evidence that independently coupled architectures converge identically.

C1 uses four fixed parameter sections and 26 controls, with continuity enforced by knot multiplicity. It is not a GT-error-adaptive piecewise model. K24/C1 copy compatible hidden layers and project token/local-image residual output layers into the new bases. The original-graph pilot's maximum initial coordinate difference was 0.00002124 mm for K24 and 0.00000708 mm for C1. CPU tests verify migration, C1 derivative continuity, rotated target correspondence, real boundary cases and common-reference reconstruction. GPU tests verify finite losses/gradients, nonzero gradients in every active detector group and geometry head, zero geometry-to-detector gradients, and exact gradient equality at decoder tokens/prediction tensors with one versus three geometry objectives.

The custom CUDA backward uses atomic accumulation: full parameter gradients are not bitwise repeatable. In the original pilot, repeating the same objective gave relative-L2 variation 0.0001952; the multiplex objective differed by 0.0001883. The corrected test requires the multiplex difference to remain within three times observed repeat variation, with a 1e-7 relative-L2 floor. No optimizer, matcher or loss was changed to pass this check.

Both pilots use seed 261039, batch 8, 4,000 updates, fresh AdamW, detector/head LR 7e-5 and backbone LR 3.5e-5, 100-update warmup and cosine decay to 30% of peak. Quarter-turn/noise augmentation is matched. The complete image-stream SHA256 is identical across trajectories: `3b66787ad35b1a1058ad76ef1a20f569e4aa33f4c3892d8a6c317672bddeb3a9`. Geometry loss gives each original parent branch total weight one, with inverse subedge-count weighting for split targets; graph-loss normalization stays unchanged. Higher-capacity heads retain residual-control Smooth L1 supervision, rather than introducing another loss variant.

Primary results use the fixed 4,000-update budget. Secondary selection was predefined as validation common Branch F1, then Curve F1, then earlier checkpoint. All arms select update 4,000. Node/edge thresholds are 0.5; physical node/edge metrics use 1/2 mm. Common-vessel metrics use the unchanged `shared-guides-v2` protocol: original anatomical reference, 1-mm anchors and curve support, 0.25-mm sampling. Paired uncertainty resamples eight validation subjects 1,000 times. There is one training seed; intervals do not establish seed robustness.

### Diverse-subject learnability and monitoring limitations

Both 512-update checks use the same 16 patches from 16 training subjects, augmentation off, starting from the clean source. The predefined eval-mode gate was node F1 >= 0.8 and edge F1 >= 0.65 at 2 mm. Both passed and completed the downstream prediction/evaluation smoke test.

| Representation | Tiny eval node F1 | Tiny eval edge F1 | Hungarian distance (mm) | Train-mode, batch-1 node / edge F1 |
|---|---:|---:|---:|---:|
| Original | 0.9900 | 0.9685 | 0.3059 | 0.7989 / 0.6893 |
| Subdivided | 0.9919 | 0.9770 | 0.3672 | 0.7570 / 0.6431 |

The lower train-mode result is a **batch-1 diagnostic**, whereas optimization uses batch 8; buffers were restored afterward. It shows sensitivity to batch statistics/composition, not proof that evaluation BN caused the larger-cohort failure. The pilot does not contain a matched batch-8 train/eval intervention.

The pilot's training monitor is also limited: one lexicographically first selected patch per training subject (39 patches), with 8.67 original nodes per patch versus 16.70 on validation. Its lower filtered scores must not be interpreted as a representative train-versus-validation generalization gap. Both training monitors nevertheless remain weak, so the current evidence does not support a simple “fits training but overfits validation” explanation. A representative same-population, matched-batch monitor is needed before attributing failure to BN, augmentation, or subject transfer.

<!-- BRAVA REPRESENTATION 20261026 RESULTS -->
### Completed bounded measurements — 26 October 2026

Primary: fixed 4,000 updates, one training seed and eight paired validation subjects. Native positional metrics use each arm’s own node/edge targets; subdivision changes those denominators. Use original-anchor recall and common Branch/Curve F1 for graph-representation comparisons. Intervals are paired subject bootstrap intervals, not seed uncertainty.

| Arm | Raw R1/R2, native | Filtered R1/R2, native | Node F1, 2 mm | Edge F1, 2 mm | Original-anchor R2 | Branch F1 | Curve F1 |
|---|---|---|---:|---:|---:|---:|---:|
| K8 | 0.0833/0.3070 | 0.0225/0.0767 | 0.1183 | 0.0089 | 0.0767 | 0.0000 | 0.0241 |
| K24 | 0.0833/0.3070 | 0.0225/0.0767 | 0.1183 | 0.0089 | 0.0767 | 0.0000 | 0.0242 |
| C1 | 0.0833/0.3070 | 0.0225/0.0767 | 0.1183 | 0.0089 | 0.0767 | 0.0000 | 0.0243 |
| split_K8 | 0.0782/0.2772 | 0.0262/0.0919 | 0.1297 | 0.0080 | 0.1022 | 0.0000 | 0.0665 |

| Comparison | Branch F1 delta [95% interval] | Curve F1 delta [95% interval] |
|---|---:|---:|
| K24-minus-K8 | +0.0000 [+0.0000, +0.0000] | +0.0001 [-0.0002, +0.0003] |
| C1-minus-K8 | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0001, +0.0005] |
| split_K8-minus-K8 | +0.0000 [+0.0000, +0.0000] | +0.0424 [+0.0348, +0.0494] |

| Arm | Train/val node F1, 2 mm | Train/val edge F1, 2 mm | Hungarian mm | Components MAE | Cycles MAE | Duplicates |
|---|---|---|---:|---:|---:|---:|
| K8 | 0.0335/0.1183 | 0.0000/0.0089 | 3.603 | 4.441 | 0.176 | 0.0220 |
| K24 | 0.0335/0.1183 | 0.0000/0.0089 | 3.603 | 4.441 | 0.176 | 0.0220 |
| C1 | 0.0335/0.1183 | 0.0000/0.0089 | 3.603 | 4.441 | 0.176 | 0.0220 |
| split_K8 | 0.0385/0.1297 | 0.0000/0.0080 | 3.663 | 4.344 | 0.785 | 0.0149 |

All geometry rows below are **GT-edge-conditioned diagnostics**: Hungarian associations select annotated positive edges regardless of predicted confidence or relation scores. They are not geometry metrics on the inferred graph. Subedges are joined per original parent branch. Endpoint-neutral geometry linearly corrects the two original endpoints; oracle-endpoint geometry reevaluates the head with GT endpoints (including every added subdivision anchor) and predicted query features. Difficult means original K8 GT-fit sampled maximum error > 1 mm; localized means both original endpoints within 2 mm. Distances are symmetric mean point-to-polyline distances with 0.1-mm sampling. Zero eligible branches are NA.

| Arm | Stratum | Branches | GT-conditioned predicted-endpoint mean mm | Endpoint-neutral mean mm | Oracle-endpoint mean mm |
|---|---|---:|---:|---:|---:|
| K8 | all | 2698 | 3.0220 | 1.4738 | 1.4641 |
| K8 | difficult | 141 | 5.5233 | 4.7876 | 4.7850 |
| K8 | localized | 180 | 1.6361 | 1.2318 | 1.2295 |
| K8 | difficult_localized | 5 | 4.5663 | 4.5457 | 4.5394 |
| K24 | all | 2698 | 3.0195 | 1.4737 | 1.4640 |
| K24 | difficult | 141 | 5.5126 | 4.7797 | 4.7796 |
| K24 | localized | 180 | 1.6298 | 1.2254 | 1.2239 |
| K24 | difficult_localized | 5 | 4.5159 | 4.4989 | 4.5020 |
| C1 | all | 2698 | 3.0183 | 1.4725 | 1.4629 |
| C1 | difficult | 141 | 5.5080 | 4.7757 | 4.7758 |
| C1 | localized | 180 | 1.6285 | 1.2243 | 1.2220 |
| C1 | difficult_localized | 5 | 4.4968 | 4.4746 | 4.4747 |
| split_K8 | all | 2698 | 2.6444 | 1.2672 | 1.0523 |
| split_K8 | difficult | 141 | 3.0284 | 3.1404 | 1.9668 |
| split_K8 | localized | 216 | 1.4008 | 1.0700 | 0.9462 |
| split_K8 | difficult_localized | 3 | 2.3625 | 2.3381 | 2.1044 |

| Arm | Primary update | Secondary validation-selected update |
|---|---:|---:|
| K8 | 4000 | 4000 |
| K24 | 4000 | 4000 |
| C1 | 4000 | 4000 |
| split_K8 | 4000 | 4000 |

Full numeric results, assignment/confidence/BN monitors, train/validation trajectories and target audit: `experiments/brava_representation_20261026/analysis.json`. Negative results remain in the same register.
<!-- END BRAVA REPRESENTATION 20261026 RESULTS -->

### Bottleneck, confidence and interpretation

**Both before and after filtering remain limiting.** Original-graph validation raw recall at 2 mm rises from 0.0550 at initialization to 0.3070 at update 4,000, and mean Hungarian distance falls from 5.727 to 3.603 mm. This is real adaptation, but 69.3% of annotated nodes remain unrecoverable even with one-to-one oracle selection. Thresholding then reduces recall to 0.0767: 984 of the 1,312 recoverable GT nodes are lost. Subdivision has almost identical raw recovery of the **original** anchors, 0.3081 versus 0.3070 (five additional anchors), while filtered original-anchor recall rises to 0.1022. Its apparent native-node recall uses a different GT denominator.

| Validation confidence diagnostic, native targets at 2 mm | Original | Subdivided |
|---|---:|---:|
| Correct versus false-candidate AUC | 0.7779 | 0.7732 |
| Correct versus duplicate AUC | 0.5180 | 0.5421 |
| Mean confidence: correct / duplicate / false candidate | 0.4092 / 0.4023 / 0.2848 | 0.4366 / 0.4153 / 0.3056 |
| Reliable-subset Brier / ECE | 0.1023 / 0.2461 | 0.1150 / 0.2620 |
| Train-monitor correct-versus-false / duplicate AUC | 0.8205 / 0.5426 | 0.7930 / 0.4606 |

“Correct,” “duplicate” and “false” are spatial annotation-relative labels, not anatomical certainty. Calibration uses the existing reliable-subset rules, ignoring ambiguous matches. Confidence is useful against far false candidates but barely distinguishes duplicate alternatives; the mean correct-node score is below the default 0.5 threshold. No threshold was tuned on validation, and no claim of optimal achievable inference is made.

Oracle selection below retains one-to-one spatially associated candidates and then applies the **unchanged learned relation head** at threshold 0.5. Only node recall is a proposal-recovery bound. The resulting edge score is conditional on that particular spatial selection, **not** a maximum over graph-aware selections.

| Validation diagnostic, 2 mm | Original | Subdivided |
|---|---:|---:|
| Native oracle node recall / F1 | 0.3070 / 0.4697 | 0.2772 / 0.4340 |
| Native edge F1 after oracle node selection | 0.0655 | 0.0630 |
| Recovered / eligible GT edges after oracle selection | 96 / 248 | 123 / 267 |
| Default retained-node conditional relation recall | 14 / 36 = 0.3889 | 19 / 59 = 0.3220 |

These oracle edge results remain poor. The failure cannot be attributed only to the classification threshold, and spline capacity cannot repair it under the deliberately isolated `graph_only` route. Assignment identities also remain unstable between updates 2,000 and 4,000: aggregate churn is 0.608 for original and 0.550 for split. This is a long-interval diagnostic over monitored train/validation patches, not per-update instability.

**Subdivision has a partial-coverage benefit with a topology cost.** The paired common Curve F1 gain is +0.04235 [0.03477, 0.04944], and original-anchor recall gain is +0.02550 [0.01458, 0.03776]. However, predicted nodes increase from 1,271 to 2,154 and edges from 435 to 1,147. Common curve precision barely changes (0.2440 to 0.2495), while coverage recall rises from 0.01268 to 0.03834. Cycle MAE increases by +0.6094 [0.5195, 0.7031]; component MAE changes by -0.0977 [-0.3751, 0.0938]. No complete common-reference branch is recovered by any arm. The [0, 0] Branch F1 interval is a floor effect, not evidence of general equivalence. Matched retained-count/precision comparisons were not run, so increased output size is an unresolved contributor to the Curve F1 gain.

**GT-conditioned geometry improves more than detector recovery.** On the same 141 difficult parent branches, subdivision reduces predicted-endpoint mean distance by 2.4949 mm; paired subject interval for split minus K8 is [-2.9340, -2.0890] mm. Endpoint-neutral and oracle-endpoint errors also improve, but oracle subdivision supplies additional exact internal GT anchors and therefore gives a more informative oracle input than the original two-endpoint graph. This is not evidence that the learned head alone accounts for the improvement. The localized difficult strata contain only five original versus three split branches and are not identical subsets; they cannot establish robust practical geometry superiority.

**Higher spline capacity remains unproven in learned reconstruction.** K24/C1 have excellent fitting bounds, but their common Curve F1 intervals overlap zero. Even with GT endpoints, difficult-branch mean distance stays near 4.78 mm for K8/K24/C1, far from the fitting capability. This points toward deficient learned residuals/features and supervision conditions rather than an exhausted K8 basis alone. The pilot does not separate those causes, prove convergence, or test GT-error-adaptive trainable C1. K24 and fixed C1 must not be described as rejected representations.

The next informative step is a representative, matched-batch check of detector learning, confidence, augmentation alignment and BN behavior across the larger training cohort, retaining the original K8 control. The tiny-set success rules out a universal inability to optimize this implementation but does not explain the larger-cohort failure. Do not increase duration or begin a long BRAVA refinement run merely on the strength of the tiny-set gate. Previous 23,500/30,000-update failures used different initialization/recipes, including edge focal, and different populations; historical scores are context, not a controlled causal comparison to this pilot.

### Artifacts, commands and resources

The dedicated experiment directory contains:

- `plan.json`: exact data IDs, subjects, seed, source, loss/matcher configuration, schedules and selection rules. `runs/{original,split}/config.json` records instantiated model configuration; the plan defines the study-specific optimizer schedule.
- `analysis.json`: final results, checkpoint trajectories, paired subject intervals, confidence/oracle diagnostics and monitors. `per-subject-results.json` and `geometry-paired-uncertainty.json` provide additional paired results. The latter uses 1,000 Python-RNG subject resamples with seed 261039, separately from the primary NumPy bootstrap.
- `runs/original/model-004000.pt`: selected shared detector and K8/K24/C1 states. `runs/split/model-004000.pt`: selected subdivided detector and K8 state. Updates 1,000 and 2,000 and both tiny-set checkpoints are preserved. Exported `geometry-*-final.pt` files identify their required shared checkpoint; they are not standalone detectors.
- `checkpoint-manifest.json`: exact paths, byte sizes and SHA256 hashes for every pilot checkpoint and exported geometry head. `source-snapshot.json`, initial Git status/patch, and `source-<jobid>.zip` preserve source provenance. Cached predictions are bundled as `predictions/{original,split}/{001000,002000,004000}.zip` with matching record manifests; evaluation rows, geometry and summaries are in the corresponding `evaluation/` directories. Archives need not be unpacked wholesale.
- `jobs.json`, `slurm-accounting.psv`, `resource-summary.json`, runtime/target recovery records and preserved failed-attempt manifests record execution and failures. Existing compressed historical predictions remain in their original archives.

Successful jobs (all exit 0):

| Stage | Job IDs | Actual elapsed |
|---|---|---|
| CPU target preparation | 1584757 | 5m42s |
| Original / split tiny tests | 1586043 / 1586044 | 2m26s / 2m09s |
| Original / split pilots | 1586045 / 1586046 | 17m14s / 14m22s |
| Original CPU evaluations, 1k / 2k / 4k | 1586047 / 1586048 / 1586049 | 5m35s / 5m20s / 5m55s |
| Split CPU evaluations, 1k / 2k / 4k | 1586050 / 1586051 / 1586052 | 3m40s / 3m38s / 3m53s |

Total H100 allocation was **0.6214 GPU-hours**, including the 66-second failed correctness check; successful GPU jobs used 0.6031 hours. Summed CPU-job elapsed time, including preparation failures, was 0.7400 hours. Peak measured PyTorch GPU allocation was 3.144 GiB. BRAVA used one serial H100; other agents' IXI/diagnostics jobs shared the other slot through verified scheduler dependencies. The BRAVA reservation is now released.

Preserved failures: CPU 1569387 exhausted node-local `/tmp` while staging the runtime; CPU 1583536 exposed the invalid assumption that every face vertex has a clipping flag; GPU 1584776 stopped before an optimizer update because of the overly strict bitwise gradient test. Their dependent jobs were cancelled automatically. The fixes reused the SHA-verified runtime on SCRATCH, retained valid targets, and measured CUDA repeat variation explicitly. No failed training checkpoint or prior experiment was overwritten.

Commands executed from the isolated worktree (the existing plan/preparation commands intentionally refuse overwrites):

```bash
python scripts/brava_representation_study.py prepare
python -m unittest tests.test_brava_representation_study tests.test_brava_representation_evaluation -v
sbatch cluster/jean_zay/brava_representation_study_cpu.slurm targets
python cluster/jean_zay/submit_brava_representation_study.py 1584757
# The submitter executes these modes through the recorded wrappers/dependencies:
python scripts/brava_representation_study.py tiny --group original
python scripts/brava_representation_study.py tiny --group split
python scripts/brava_representation_study.py train --group original
python scripts/brava_representation_study.py train --group split
# evaluate --group original|split --step 1000|2000|4000 (six CPU jobs)
python scripts/report_brava_representation_study.py
python experiments/brava_representation_20261026/collect_paired_geometry.py
```

Wrappers source `cluster/jean_zay/env.sh` (PyTorch 2.3.1), use the protected base Python, and set `PYTHONPATH` to the verified runtime at `$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536/ixi-protected` and its parent. Exact accepted SLURM arguments and per-job source are retained in the artifacts. Collection and report publication required no additional GPU run.
