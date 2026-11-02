# Multi-dataset continuous vascular graph experiments

## Status and research sequence

22 October2026. Initial bounded batch: BRAVA detector recovery plus independent IXI/SyntheticMRI frozen geometry learning. All15 training and evaluation arms completed; numerical results and reporting recovery are documented below. Existing full GT audits, BRAVA zero-shot/scratch evaluations and the completed first frozen geometry study are reused. Test sets are excluded. No representation end-to-end pilot is automatically launched in this batch.

Sources: [review](EXPERIMENT_REVIEW_20261022.md), [representation results](continuous_representation_comparison.md), [BRAVA training](BRAVA_experiment_plan.md), [preparation](BRAVA_preparation.md), [capacity](spline_learning/ixi-brava-control-capacity.md), node-refinement reports3/6/7, relation isolation4 and matching/calibration8. Previous negative-mining experiments do not justify treating every unmatched query as a safe negative.

## A. BRAVA detector diagnosis

Completed previous evidence: scratch standard/refinement trained23,500 updates with full8,768-patch exposure and a polynomial LR ending at zero. They produced only4/1 correct edges on1,796 validation patches. The IXI zero-shot model is stronger but still weak. On the512-patch frozen-head cohort it recovers23.47% of nodes at2 mm; only454/5,445 GT edges have both endpoints recovered, and106 are predicted. Geometry adaptation cannot repair these missing endpoints or relations.

### Findings and unresolved explanations

- Native64³ MRA patches and graph paths preserve the established physical metric and source alignment corrections. Previous56-subject BRAVA orientation audit and complete prepared-target audit passed. New focused checks validate16 patches per dataset, full-path endpoint correspondence, finite images, physical metrics and train/validation separation. They do not establish perfect anatomical registration or complete annotation coverage.
- The inspected16-patch intensity sample has mean/std BRAVA0.391/0.217, IXI0.284/0.137, SyntheticMRI0.333/0.289 **before** the unchanged loader subtraction0.3333526. This small selected sample demonstrates a domain difference; it is not a causal explanation or population histogram estimate. No new normalization is silently introduced.
- Selected graph-positive populations contain no zero-edge patches. Background sampling/selection may influence object calibration; this batch preserves the established population rather than silently adding new background negatives.
- Original matcher is2×negative object probability plus5×normalized-DHW L1 distance. Hungarian assignment need not mean physical recovery. Raw query coverage, physical1/2-mm recovery, Hungarian distance and confidence/duplicate statistics are monitored separately.
- Static gradient routes show the old `graph_only` criterion detaches tokens and predicted endpoints; local CNN sampling also detaches its feature map. Thus direct spline-loss interference with the detector is not supported by the implementation. The GPU smoke explicitly tests zero geometry gradient to all shared detector parameters. Indirect initialization/optimization and BN/domain effects remain plausible.
- Earlier logs decrease total loss13.22→4.61 while final logged minibatch endpoint error remains~4.93 mm. Different minibatches are not a fixed-population estimate. The new monitor uses fixed256 train/128 validation patches, covering all39/8 subjects, at the same update counts in every recovery arm.
- Incomplete SWC coverage and forest encoding limit negatives and topology interpretation. Brain masks in `seg/` are neither vessel-support labels nor network inputs. Existing matched-only relation supervision is preserved initially.

### Recovery matrix

|Arm|Initialization|Detector/head training|Queries|
|---|---|---|---:|
|D0|Existing IXI and SyntheticMRI zero-shot caches; new step0 monitor for actual transferred initialization|None|Prior native counts; transferred comparison256|
|D1|IXI delayed-spline best|Entire detector/backbone and existing K8 head|256|
|D2|SyntheticMRI vessels-spline best|Same modules, losses and schedule as D1|256|
|D3|Fresh seeded BRAVA-only model|Same modules, losses and schedule|256|

IXI checkpoint SHA256 `1cd9caf35a3683ef643c541ea4c4d06b247c490a4215d4a250f4519992d68eb3`, saved stage step28,170. SyntheticMRI checkpoint SHA256 `06513f58b7b032c9eb4f7be7820312941e7bac08fb97915f263d48f1b894facb`, saved stage step31,300. Saved stage steps are **not total pretraining histories**; full prior recipes remain in their original reports. Exact paths/hashes are in the plan/audit.

**Compatibility caveat:** the SyntheticMRI source has120 object queries, IXI256. D2 copies all120 source object-query embeddings and both relation-query rows into the corresponding256-query model; additional object rows retain the common fresh seeded initialization. All non-query parameters load strictly. Step0 evaluates this actual expanded initialization. The source query/history difference prevents interpreting D1/D2 as a pure source-data-only causal comparison.

Full BRAVA training **8,768 patches /39 subjects**, final evaluation **1,796 patches /8 subjects**, train monitor256, validation monitor128. Identical trainable parameter sets, augmentation, coordinate/box losses and matcher across recovery arms. No local node refinement, verifier or heatmap in the initial matrix.

**Budget:**30,000 batch8 updates (~27.4 full-population passes), checkpoints2,500/10,000/20,000/30,000 plus step0 monitoring. AdamW existing betas/weight decay; detector/head LR7e-5 and backbone LR3.5e-5.500-update warmup followed by cosine decay **to30% of base LR**, not zero. This common new recipe changes both LR grouping and schedule from the old scratch recipe; it tests recovery, not an isolated attribution of the earlier failure. It does not merely extend a finished zero-LR schedule. GPU benchmark blocks runs predicted to exceed4 training hours;5h allocations provide monitor/checkpoint buffer. A finite pilot does not prove convergence or an architecture limit.

Primary comparisons use matched final updates. A separate best checkpoint is selected using fixed BRAVA validation-monitor edge F1@2 mm, then evaluated on all validation patches. Both best and final are retained. Training data alone choose relation thresholds. No test-based selection.

## B. Actual trainable representations

### Shared experimental setup

IXI source: the compatible IXI delayed-spline detector. SyntheticMRI source: the compatible vascular-only SyntheticMRI spline detector, preserving its native120 queries. The latter's original source-training subjects are checked against current validation volumes using the archived comparison manifest; no overlap is allowed. Native query counts remain identical across geometry variants within each dataset.

Initial geometry populations: **4,096 train /512 validation IXI patches (123/26 subjects)** and **4,096 train /480 validation SyntheticMRI patches (64/13 volumes)**. These are deterministic subject-balanced feasibility subsets of the established splits, not final full-data comparisons. Target node maxima153/50 fit the respective256/120-query budgets. Every arm uses the same256 training-monitor patches per dataset. Full-data extensions remain pending results.

Freeze detector parameters **and BN statistics**. Copy existing token/CNN geometry trunks; reset native geometry output projections to zero in every arm, starting at identical chords. Train20,000 batch4 updates (~19.5 subset passes), same seed223039/order/AdamW and warmup/floor schedule. Checkpoints1,000/10,000/20,000; compare final matched updates. The actual parameter counts, peak GPU memory and timings are saved by the mandatory GPU benchmark. Inference batch1 and exact graph/candidate parity are checked. Dense-reference geometry remains distinct from the target spline fit.

|Arm per dataset|Trainable representation|Native geometry parameters|
|---|---|---|
|K8/K16/K24|Existing local-CNN-conditioned cubic B-spline head|Fixed K|
|Adaptive K|Shared token trunk and local-CNN trunk, five independent native output projections|K∈{4,8,12,16,24}|
|Piecewise C0|Four K8 spline sections, shared joins through knot multiplicity3|29 global control vectors|
|Piecewise C1|Four K8 spline sections, tangent continuity through knot multiplicity2|26 global control vectors|

### Genuine adaptive K

`models/adaptive_geometry.py::AdaptiveSplineBank` shares hidden geometry layers and local CNN feature processing; the output projections have different correct B-spline bases. During geometry training all five projections receive the corresponding independently fitted original-path controls. Losses average across projections **per original branch**, then across branches, preserving loss scale and positive population.

The selector is trained **after** geometry learning:2,000 small-classifier updates on detached shared geometry-trunk embeddings from256 training patches. Labels are the smallest K meeting the existing0.5-mm GT fit tolerance; impossible cases use the lowest-error available fit and are counted. Classifier parameters are excluded from the detector/geometry optimizer. Inference receives predicted features only, then selects an actual native projection—this is not K24 compression. Validation confusion matrix/control counts and oracle-selection error of the learned projections are saved separately.

### Learned internal piecewise geometry

`PiecewiseSplineHead` predicts a cubic spline with four fixed parameter sections. Each section has four interior simple knots (the K8 local basis); join knots are repeated3 times forC0 or2 times forC1. The resulting29/26 global coefficients enforce shared joins and C1 tangent continuity **for every network output**, without a penalty or additional task. Internal points are never object queries. GT fitting uses these same knot vectors and endpoint constraints on original dense paths, with257 arc-length fitting samples.

Endpoints stay at predicted detector nodes. The existing physical residual-control Smooth L1 objective is reused in each representation's native basis and averaged across control coordinates then original branches. This changes the geometric parameterization; it adds no joint detector loss. C0/C1 have different free coefficient counts and are reported accordingly.

At inference, optionally refit the **predicted** curve with fewer1–3 native sections within0.5-mm numerical compression tolerance, preserving its continuity family; fallback retains the original four-section curve. No GT breakpoint/count reaches inference. This simplification reduces output geometry, not initial neural-output computation. Four sections are an initial bounded parameterization, not the earlier error-adaptive GT partition or a universal tolerance guarantee.

New-basis readiness check on four previously identified long BRAVA **training** branches: every C0/C1 fit improves mean distance versus K8. For the82.1-mm example, K8 mean~0.932 mm versusC0/C1~0.084/0.089 mm; sampled maxima~0.417/0.416 mm. C0 join angles reach~27° on that branch (up to94° on another); C1 angles are numerical zero. These are new-basis GT checks only, not learned results or a repeated population audit. One checked branch still has maximum~0.524 mm atC1, so four sections are not claimed to satisfy0.5 mm everywhere.

### Explicit degree-2 splitting: implementation ready, training pending

`data/adaptive_split_targets.py` converts an existing cache directly using the established0.5-mm error-driven splitter, preserving original anchors, boundaries, dense path intervals and connectivity. It bypasses the contracting target exporter. Each subedge retains its original branch ID and inverse subedge-count geometry weight; tests check preserved endpoints/topology and total weight1 per original branch. It does not regenerate the validated dataset.

Actual split-graph training is **not submitted** yet. The existing criterion must consume branch weights, and a matched oracle-node learning/inference adapter plus anatomical-chain contraction comparison must pass before a full detector run. Added nodes change matching/relation supervision; geometry weighting alone does not make all detector loss populations identical. Existing capacity results (BRAVA mean17.14→20.67 nodes and~48% more candidate pairs at0.5 mm) are reused, not recomputed.

## C. Evaluation and automatic collection

- Same predicted nodes/query IDs/candidates for frozen geometry variants; collector checks exact populations and node/edge counts.
- Node precision/recall/F1 at1/2 mm, raw-query recall, localization, correct-versus-duplicate confidence AUC, duplicates/high-confidence duplicates and counts; native detection mAP/mAR and original SMD convention remain separately reported.
- Positional edges, conditional relation AP/recall and all-retained-pair AP; training-only max-edge-F1 threshold on a predefined0.05–0.95 grid, applied unchanged to validation. Node confidence policy remains the checkpoint default; arbitrary validation node thresholds are not selected.
- Common-v2 Branch/Curve and components/cycles use original dense references. A validated explicit-polyline adapter handles nonuniform and variable-K curves without changing common-v2 definitions. Native SMD still uses its established graph/chord convention; continuous geometry is evaluated independently rather than relabelling that SMD.
- Learned geometry on all Hungarian GT-positive pairs, correctly localized-endpoint subsets and endpoint-neutral diagnostics. Stratify arc length, chord distance/tortuosity, turning and K8 fit difficulty. Oracle selection is explicitly diagnostic.
- Recovery caches include **all raw-query pair scores**, Hungarian identities, raw/retained confidence and geometry. CPU analysis separately counts Hungarian-unmatched versus spatially unmatched false relations with population denominators. No vessel-mask support is inferred from BRAVA brain masks.
- Per-patch/per-subject/aggregate files,1000 subject-bootstrap intervals and paired geometry comparisons, training curves and representative overlays are collected automatically. Intervals capture subject sampling, not seed uncertainty. Trace-relative false edges and topology cannot establish complete anatomical falsehood on BRAVA.

## D. Node/refinement and relation stages

Local coordinate/confidence refinement exists and was inspected; its classifier correction can be disabled separately. No context verifier or auxiliary heatmap is implemented in the current model/configuration search. Initial recovery arms keep refinement off; subsequent matched detector ablations should separate coordinates and confidence on each dataset. Earlier IXI refinement results are not assumed to transfer to BRAVA.

A separate relation stage is designed after recovery: baseline versus20% random or plausible Hungarian-unmatched-involving negatives, **replacing** existing negatives at fixed total budget, keeping every positive, identical scorer/loss/detector. Plausibility uses predicted confidence/proximity; ambiguity near alternate GT associations must be identified before labelling such pairs negative. Incomplete annotations and duplicates make blind unmatched-negative labelling questionable. This stage is not launched alongside the representation changes.

## E. Stage gates and pending work

Initial batch submits detector recovery and genuine adaptive/piecewise frozen learning on the two independent datasets. Recovered-BRAVA geometry comparisons wait for a validation-selected detector. The collector records a predefined feasibility gate (node recall2mm>0.25 and edge F1>0.04), which is a practical gate rather than proof of superiority; source-paired comparisons and learning curves still require review.

Then: recovered-BRAVA fixed/adaptive/piecewise heads; oracle-node explicit-splitting geometry learning; modified graph training with verified branch weighting; full-data end-to-end K8/best-larger/adaptive/piecewise/split comparisons; isolated node/refinement and relation sampling studies. No full representation end-to-end run is submitted before these prerequisites. No automatic large sweep or additional seed.

## F. Execution, paths and retrieval

Source/config/data identities and exact fixed manifests: `experiments/multidataset_continuous/plan.json`. Local audit evidence: `experiments/multidataset_continuous/local-audit/audit.json`. Initial source/output hashes are captured at submission; jobs repeat the audit.

Large outputs: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/multidataset_continuous_20261022`. Compact results: `docs/research/evidence/multidataset_continuous/`. Job IDs/dependencies are appended after submission. A single array capped at two H100s runs training then prediction within each allocation, so training and inference together cannot exceed the two-GPU cap. CPU evaluation uses matching-array aftercorr dependencies and can start as each arm finishes; final collection follows all evaluations. No unsupported memory directives.

Retrieve saved results without inference or training:

```bash
python3 scripts/show_multidataset_graph.py
```

## Accepted initial batch

|Stage|Job|Dependency|Array|Wall allocation|
|---|---|---|---|---|
|Data and checkpoint audit|1451641|none|none|01:00:00|
|GPU forward backward and collector smoke|1451642|afterok:1451641|none|01:00:00|
|Detector recovery and geometry training plus predictions|1451643|afterok:1451642|0-14%2|05:00:00|
|Per-arm CPU evaluation|1451644|aftercorr:1451643|0-14%4|08:00:00|
|Aggregate report|1451645|afterok:1451644|none|01:00:00|

Training-array mapping:

|Task|Arm|Dataset|Updates|
|---|---|---|---:|
|0|D1_IXI|BRAVA|30000|
|1|D2_SyntheticMRI|BRAVA|30000|
|2|D3_scratch|BRAVA|30000|
|3|IXI_K8|IXI|20000|
|4|IXI_K16|IXI|20000|
|5|IXI_K24|IXI|20000|
|6|IXI_adaptive|IXI|20000|
|7|IXI_piece_C0|IXI|20000|
|8|IXI_piece_C1|IXI|20000|
|9|SyntheticMRI_K8|SyntheticMRI|20000|
|10|SyntheticMRI_K16|SyntheticMRI|20000|
|11|SyntheticMRI_K24|SyntheticMRI|20000|
|12|SyntheticMRI_adaptive|SyntheticMRI|20000|
|13|SyntheticMRI_piece_C0|SyntheticMRI|20000|
|14|SyntheticMRI_piece_C1|SyntheticMRI|20000|

GPU smoke checks and actual throughputs are pending their scheduled job. Training starts only after all smoke checks pass. No further model stages are automatically submitted.

## Smoke recovery —22 October2026

Cluster audit1451641 completed. GPU smoke1451642 passed the seven original tests and the first detector forward/backward updates, then failed because snapshot lacked its `canonical_reverse` import. Main training never started; downstream jobs were cancelled by their dependencies. The missing import was added and the actual snapshot-global orientation lookup was verified on CPU. A regression now exercises that dependency. Retry smoke folders include the job ID, preserving the failed evidence. Data, objectives and budgets are unchanged.

|Replacement stage|Job|Dependency|
|---|---|---|
|GPU forward backward and collector smoke|1460518|none|
|Detector recovery and geometry training plus predictions|1460519|afterok:1460518|
|Per-arm CPU evaluation|1460520|aftercorr:1460519|
|Aggregate report|1460521|afterok:1460520|

<!-- BEGIN COMPLETED INTERPRETATION -->
## Completed interpretation —23 October2026

## Completion and reporting recovery

All15 training/prediction tasks1460519 and all15 CPU evaluation tasks1460520 completed successfully. BRAVA arms completed30,000 updates on8,768 training patches; each independent IXI/SyntheticMRI head completed20,000 updates on4,096 training patches. No test evaluation or new end-to-end representation training was performed.

Collector1460521 failed **after numerical aggregation**, while loading missing Plotly assets from the archived training runtime. Collection was rerun from saved results using the complete protected plotting environment. All tables, figures, three offline3D viewers and the study report are now published. No inference/retraining was required. The Slurm failure remains in accounting; successful reporting recovery is recorded by `results/collection-complete.json`.

The retrieval-command error was a working-directory issue. From the parent `projects/` directory use `python3 Graph-Native-Betti-Matching-Representation/scripts/show_multidataset_graph.py`; from the repository root use `python3 scripts/show_multidataset_graph.py`.

## 1. BRAVA: node recovery improves, graph connectivity remains inadequate

All1,796 validation patches /8 subjects, default edge threshold0.5 and2-mm positional matching; common-v2 Branch/Curve retain their own1-mm convention. Native graph-query population and GT are the same across these final K8 recovery arms.

|Initializer|Node recall|Raw-query recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|
|---|---|---|---|---|---|---|---|
|D1_IXI|0.34093|0.56318|0.02099|0.00243|0.02220|14.33797|0.12416|
|D2_SyntheticMRI|0.27331|0.48499|0.01340|0.00128|0.01094|13.37862|0.08073|
|D3_scratch|0.10557|0.31264|0.00040|0.00010|0.00100|17.04510|0.00334|

**IXI initialization is best among the recovery arms**, but it is not a recovered high-quality graph detector. Its default edge F1 is0.02099 (95% subject-bootstrap interval[0.01949,0.02260]); the predefined gate required edge F1>0.04 and node recall>0.25. The gate fails.

The same-split historical IXI zero-shot reference has node recall0.23095, edge F10.02687 and Curve F10.05802. Fine-tuning improves node recall to0.34093 but lowers default edge/curve performance to0.02099/0.02220. It predicts249 correct and4,028 incorrect edges versus the reference376/8,160. Fewer false edges therefore coexist with fewer correct edges and greater fragmentation (component MAE7.65→14.34). This descriptive comparison uses the preserved reference, not a new tuned test.

Training-calibrated thresholds do not resolve connectivity: D1's train-selected0.425 gives validation precision0.03482, recall0.03939, edge F10.03696 and Curve F10.09217, with766 correct/21,236 incorrect edges. Branch F1 decreases0.00243→0.00158; cycle MAE rises0.124→2.211. These are recall/calibration trade-offs, not a demonstrated topology improvement.

D2's train-selected0.4 gives edge F10.02495, Curve F10.09674 and cycle MAE3.818. D3's0.375 gives edge F10.00257 and cycle MAE9.854. The threshold grid and selection used training data only.

### What this says about scratch failure

Scratch remains poor **on monitored training patches too**: final training node recall0.1159 and edge F10, versus full-validation node recall0.1056 and edge F10.00040 (4 correct/641 incorrect edges). Poor validation generalization alone cannot explain this. Losses decrease and the new LR floor remains positive; simply appending updates to an old zero-LR run is not the explanation tested here.

The static detach route and GPU gradient check exclude direct spline-loss gradients into the detector in this recipe. They do not identify a single cause of weak localization/relation learning. Intensity/domain differences, optimization/assignment behavior, confidence selection, graph-anchor distributions and incomplete annotations remain competing explanations. A small BRAVA training-patch learnability/BN/assignment diagnostic is justified before another long scratch run.

On the fixed128-patch validation monitor D1's node recall rises0.2271→0.3246 and mean Hungarian distance falls4.137→3.382 mm, while default edge F1 drops0.0292→0.0156. This directly separates better localization from worse default connectivity. D1's final training/validation edge F1 is0.0228/0.0210 on the reported populations; source-population/prevalence differences prevent attributing every train–validation AP difference to generalization.

### Selection and comparison limitations

D1's128-patch monitor selected step20,000 by a very small edge-F1 margin, but full validation scores step30,000 higher (0.02099 versus0.01910). The primary matched-final comparison remains30,000; the full-validation winner is retained by the collector. Any further selection must acknowledge this two-level validation process. D2 expands120 source queries to256; source history/query initialization prevent a pure source-data-only causal interpretation.

## 2. Actual learned geometry on IXI and SyntheticMRI

Every arm uses the same frozen dataset-specific detector, nodes, query population and relation predictions. These are genuine trained geometry heads, not GT-only fits. Adaptive inference selects a native K-specific projection; piecewise heads predict four shared-join sections with C0/C1 constraints. Their uniform parameter sections are not the same as the earlier error-adaptive GT partitions.

Validation:512 IXI patches /26 subjects and480 SyntheticMRI patches /13 volumes. “Mean error” includes incorrectly localized Hungarian endpoints; the endpoint-neutral diagnostic is evaluation-only.

|Dataset|Head|Curve F1|Mean error mm|Endpoint-neutral mm|Controls/branch|
|---|---|---|---|---|---|
|IXI|K8|0.12149|1.88304|0.51484|8.00|
|IXI|K16|0.12177|1.88311|0.51565|16.00|
|IXI|K24|0.12205|1.88373|0.51682|24.00|
|IXI|adaptive|0.12108|1.88701|0.51119|4.31|
|IXI|piece_C0|0.12170|1.88336|0.51692|29.00|
|IXI|piece_C1|0.12216|1.88329|0.51646|26.00|
|SyntheticMRI|K8|0.58386|0.97150|0.40446|8.00|
|SyntheticMRI|K16|0.58324|0.97338|0.40907|16.00|
|SyntheticMRI|K24|0.58377|0.97167|0.40750|24.00|
|SyntheticMRI|adaptive|0.58170|0.97383|0.40112|4.54|
|SyntheticMRI|piece_C0|0.58415|0.97225|0.40850|29.00|
|SyntheticMRI|piece_C1|0.58275|0.97359|0.40901|26.00|

Complete graph counts and positional node/edge outcomes are identical across frozen geometry variants by construction. IXI's positional edge F1 is0.06074, conditional relation AP0.45437 and Branch F10.00528. SyntheticMRI's corresponding values are0.47013,0.89837 and0.10250. Candidate prevalence/populations differ between datasets, so AP values are not a controlled cross-domain ranking comparison.

### Paired uncertainty

|Dataset|Arm vs K8|Curve F1 difference|95% subject interval|
|---|---|---|---|
|IXI|K16|0.000280|[-0.000074, 0.000650]|
|IXI|K24|0.000558|[0.000227, 0.000910]|
|IXI|adaptive|-0.000406|[-0.001080, 0.000415]|
|IXI|piece_C0|0.000211|[-0.000205, 0.000657]|
|IXI|piece_C1|0.000666|[0.000078, 0.001435]|
|SyntheticMRI|K16|-0.000618|[-0.002251, 0.000992]|
|SyntheticMRI|K24|-0.000095|[-0.001491, 0.001289]|
|SyntheticMRI|adaptive|-0.002158|[-0.003313, -0.000770]|
|SyntheticMRI|piece_C0|0.000286|[-0.000828, 0.001342]|
|SyntheticMRI|piece_C1|-0.001113|[-0.002604, -0.000028]|

Intervals use1,000 paired subject resamples and do not include seed uncertainty. IXI K24 and C1 have small positive Curve F1 differences; SyntheticMRI adaptive K and C1 have small negative differences. There is no substantial, consistent reconstruction advantage across datasets. Endpoint-neutral errors also do not show a broad larger-capacity advantage.

### Difficult branches and correctly localized endpoints

IXI has88 validation branch occurrences with K8 sampled maximum fit error>0.5 mm; only2 also have both predicted endpoints within1 mm. SyntheticMRI has35 difficult occurrences,9 localized. Conclusions about difficult correctly localized branches are therefore weakly supported.

On the difficult IXI subset, mean error K8/K24/C1 is4.571/4.551/4.549 mm. On difficult SyntheticMRI it is2.808/2.812/2.859 mm. This does not reveal a large hidden benefit masked by the short-branch majority. Long-branch (≥25 mm), endpoint-neutral and localization strata are preserved in `geometry-stratified.csv` and `interpretation-details.json`.

Among all correctly localized pairs, mean errors K8/C1 are0.5978/0.5973 mm for IXI (688 pairs) and0.5530/0.5599 mm for SyntheticMRI (1,333 pairs). These conditional cohorts do not imply the same accuracy for the complete decoded graph.

## 3. Adaptive efficiency and selector limitations

Actual native adaptive heads average4.31 controls on IXI and4.54 on SyntheticMRI, versus8 for the fixed baseline. This is real variable-output prediction; it is distinct from the earlier K24 output-compression classifier. Curve F1 is unchanged within the IXI interval and slightly lower on SyntheticMRI.

Selector validation accuracy is91.02% on IXI, but always choosingK4 would score91.58%. On SyntheticMRI it is87.19% versus86.22% for alwaysK4. Complex-class recall is low: K8/K12 recall36.5%/15.0% on IXI and49.8%/10.2% on SyntheticMRI; no K16 case is classified correctly. Overall accuracy alone is misleading under these imbalanced labels.

Using **GT-fit K labels to choose among the trained projections**, as a diagnostic, yields mean errors1.8863 versus actual selection1.8870 mm on IXI and0.9722 versus0.9738 mm on SyntheticMRI. This is not an oracle choosing the best learned curve by its measured error. It shows that perfect reproduction of the current GT-complexity labels alone would not establish a large gain in this trained bank.

|Head|Geometry-trained parameters|IXI training min|SyntheticMRI training min|
|---|---|---|---|
|K8|1,577,876|47.7|44.9|
|K16|1,592,708|46.6|46.4|
|K24|1,607,540|50.2|44.6|
|adaptive|1,666,868|65.2|60.8|
|piece_C0|1,616,810|49.0|43.6|
|piece_C1|1,611,248|48.1|46.7|

The bank's separately trained selector adds35,717 parameters to its1,666,868 geometry-trained parameters. Reduced output control count is therefore not a smaller model; bank training is also slower than K8. Runtimes are descriptive single-run wall times, not isolated inference benchmarks. Four-section C1 uses26 global control vectors, C0 uses29; C1 continuity is enforced by the basis, without another loss.

## 4. Unmatched-query populations: measured separately, no new mining trained

D1's raw-query analysis includes58.62 million candidate pairs across validation, **including confidence-rejected queries**. Their predicted false-edge counts must not be confused with the4,028 false edges in the normally decoded graph.

False-positive rates among labelled negative raw pairs are0.722% for Hungarian-matched/matched versus1.073% for Hungarian-unmatched-involving pairs; spatially matched/matched gives0.168% versus1.073% for spatially unmatched-involving pairs. Conditioning on population size matters.

There are also3,862 physically GT-positive raw pairs involving at least one Hungarian-unassigned query, versus1,412 among Hungarian-assigned pairs. Blanket labelling of every Hungarian-unmatched-involving pair as negative could therefore suppress spatially valid alternative query choices. This does not invalidate Hungarian training, but it reinforces the need to handle duplicates/assignment ambiguity before another negative-sampling experiment. These are annotation-relative2-mm associations, not independent anatomical truth.

## 5. Decisions and remaining experiments

1. **Keep BRAVA end-to-end representation expansion gated.** IXI initialization is the strongest recovery arm, but neither default nor training-calibrated connectivity meets the intended practical standard. Investigate training-patch learnability and assignment/confidence/BN behavior before another large scratch run or a loss sweep.
2. **Keep K8 as the strong geometry control on IXI/SyntheticMRI.** Their GT K8 fits are already very accurate; extra capacity gives small, mixed learned changes. C1 is a valid, trainable parameterization, not yet the selected journal method.
3. **Treat adaptive K as an efficiency result with selector limitations.** Examine rare-complexity classes and native-projection supervision before claiming an accuracy contribution. Add no joint selector loss by default.
4. **Do not discard piecewise geometry from these pilots.** They train four fixed parameter sections on frozen detectors; they do not test a recovered BRAVA model, the exact GT adaptive partition, or full end-to-end geometry learning.
5. Explicit degree-2 training/oracle-node geometry, recovered-BRAVA head comparisons, full end-to-end representation runs and isolated node/refinement/relation-sampling ablations remain unlaunched. No test data or new training was used for this review.

The overarching question remains open: this batch establishes trainable implementations and efficiency trade-offs, but does not establish a robust cross-dataset reconstruction advantage for adaptive geometry.

## Files and reproduction

- Detailed protocol and generated tables: [multi-dataset report](multidataset_continuous_graph_experiments.md).
- Compact evidence: [tables and figures](evidence/multidataset_continuous/report.md), [paired intervals](evidence/multidataset_continuous/paired-representations.csv), [analysis details](evidence/multidataset_continuous/interpretation-details.json).
- Held-out-development long-branch viewers: [BRAVA](evidence/multidataset_continuous/BRAVA-example.html), [IXI](evidence/multidataset_continuous/IXI-example.html), [SyntheticMRI](evidence/multidataset_continuous/SyntheticMRI-example.html). These show Hungarian-positive diagnostics, which need not be decoded edges.
- Summary command from the repository root: `python3 scripts/show_multidataset_graph.py`.
- Reproduce this interpretation from saved tables: `python3 scripts/report_multidataset_results.py`.

<!-- END COMPLETED INTERPRETATION -->
<!-- INITIAL RESULTS -->
## Completed initial batch

Matched-final development results; no test evaluation. Frozen geometry arms have identical node/relation results by design. BRAVA recovery arms train the full detector.

| dataset      | arm                   |   step |   node_recall |   raw_query_recall |      f1 |   conditional_relation_ap |   conditional_recall |   common_branch_f1 |   common_curve_f1 |   beta0_mae |   beta1_mae |
|:-------------|:----------------------|-------:|--------------:|-------------------:|--------:|--------------------------:|---------------------:|-------------------:|------------------:|------------:|------------:|
| BRAVA        | D1_IXI                |  30000 |       0.34093 |            0.56318 | 0.02099 |                   0.43697 |              0.12413 |            0.00243 |           0.0222  |    14.338   |     0.12416 |
| BRAVA        | D2_SyntheticMRI       |  30000 |       0.27331 |            0.48499 | 0.0134  |                   0.39735 |              0.11524 |            0.00128 |           0.01094 |    13.3786  |     0.08073 |
| BRAVA        | D3_scratch            |  30000 |       0.10557 |            0.31264 | 0.0004  |                   0.32605 |              0.02424 |            0.0001  |           0.001   |    17.0451  |     0.00334 |
| IXI          | IXI_K8                |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12149 |    17.0469  |     0.95898 |
| IXI          | IXI_K16               |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12177 |    17.0469  |     0.95898 |
| IXI          | IXI_K24               |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12205 |    17.0469  |     0.95898 |
| IXI          | IXI_adaptive          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12108 |    17.0469  |     0.95898 |
| IXI          | IXI_piece_C0          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.1217  |    17.0469  |     0.95898 |
| IXI          | IXI_piece_C1          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12216 |    17.0469  |     0.95898 |
| SyntheticMRI | SyntheticMRI_K8       |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58386 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_K16      |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58324 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_K24      |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58377 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.5817  |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58415 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58275 |     2.50833 |     1.51875 |

### Learned geometry

| dataset      | arm                   |   step | split   |   branches |   mean_mm |   max_p95_mm |   neutral_mm |   controls |
|:-------------|:----------------------|-------:|:--------|-----------:|----------:|-------------:|-------------:|-----------:|
| BRAVA        | D1_IXI                |  20000 | train   |       2677 |   2.8055  |     11.3429  |      1.37327 |    8       |
| BRAVA        | D1_IXI                |  20000 | val     |      19446 |   2.9451  |     11.8893  |      1.41613 |    8       |
| BRAVA        | D1_IXI                |  30000 | train   |       2677 |   2.72847 |     11.1136  |      1.3796  |    8       |
| BRAVA        | D1_IXI                |  30000 | val     |      19446 |   2.85363 |     11.7828  |      1.41333 |    8       |
| BRAVA        | D2_SyntheticMRI       |  30000 | train   |       2677 |   2.96554 |     11.8727  |      1.39862 |    8       |
| BRAVA        | D2_SyntheticMRI       |  30000 | val     |      19446 |   3.06706 |     12.1897  |      1.42832 |    8       |
| BRAVA        | D3_scratch            |  30000 | train   |       2677 |   3.59396 |     11.7253  |      1.40795 |    8       |
| BRAVA        | D3_scratch            |  30000 | val     |      19446 |   3.67252 |     12.2193  |      1.43489 |    8       |
| IXI          | IXI_K16               |  20000 | train   |       4244 |   1.54392 |      7.72345 |      0.49838 |   16       |
| IXI          | IXI_K16               |  20000 | val     |      10968 |   1.88311 |      9.03705 |      0.51565 |   16       |
| IXI          | IXI_K24               |  20000 | train   |       4244 |   1.54347 |      7.72788 |      0.49773 |   24       |
| IXI          | IXI_K24               |  20000 | val     |      10968 |   1.88373 |      9.04837 |      0.51682 |   24       |
| IXI          | IXI_K8                |  20000 | train   |       4244 |   1.54723 |      7.72345 |      0.49971 |    8       |
| IXI          | IXI_K8                |  20000 | val     |      10968 |   1.88304 |      9.05911 |      0.51484 |    8       |
| IXI          | IXI_adaptive          |  20000 | train   |       4244 |   1.55349 |      7.72345 |      0.50104 |    4.46843 |
| IXI          | IXI_adaptive          |  20000 | val     |      10968 |   1.88701 |      9.04175 |      0.51119 |    4.31182 |
| IXI          | IXI_piece_C0          |  20000 | train   |       4244 |   1.54354 |      7.72096 |      0.49838 |   29       |
| IXI          | IXI_piece_C0          |  20000 | val     |      10968 |   1.88336 |      9.04946 |      0.51692 |   29       |
| IXI          | IXI_piece_C1          |  20000 | train   |       4244 |   1.54459 |      7.76445 |      0.49872 |   26       |
| IXI          | IXI_piece_C1          |  20000 | val     |      10968 |   1.88329 |      9.03714 |      0.51646 |   26       |
| SyntheticMRI | SyntheticMRI_K16      |  20000 | train   |       4227 |   0.97893 |      5.98441 |      0.36578 |   16       |
| SyntheticMRI | SyntheticMRI_K16      |  20000 | val     |       7679 |   0.97338 |      5.61535 |      0.40907 |   16       |
| SyntheticMRI | SyntheticMRI_K24      |  20000 | train   |       4227 |   0.97859 |      5.99769 |      0.36489 |   24       |
| SyntheticMRI | SyntheticMRI_K24      |  20000 | val     |       7679 |   0.97167 |      5.59466 |      0.4075  |   24       |
| SyntheticMRI | SyntheticMRI_K8       |  20000 | train   |       4227 |   0.98129 |      6.03457 |      0.36406 |    8       |
| SyntheticMRI | SyntheticMRI_K8       |  20000 | val     |       7679 |   0.9715  |      5.6095  |      0.40446 |    8       |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 | train   |       4227 |   0.98254 |      6.02332 |      0.362   |    4.55642 |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 | val     |       7679 |   0.97383 |      5.6092  |      0.40112 |    4.53965 |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 | train   |       4227 |   0.97838 |      5.98931 |      0.36557 |   29       |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 | val     |       7679 |   0.97225 |      5.62123 |      0.4085  |   29       |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 | train   |       4227 |   0.98066 |      6.03668 |      0.36712 |   26       |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 | val     |       7679 |   0.97359 |      5.61535 |      0.40901 |   26       |

### Paired subject Curve F1 differences

| dataset      | arm                   | reference       |   curve_f1_difference |      low |     high |
|:-------------|:----------------------|:----------------|----------------------:|---------:|---------:|
| IXI          | IXI_K8                | IXI_K8          |               0       |  0       |  0       |
| IXI          | IXI_K16               | IXI_K8          |               0.00028 | -7e-05   |  0.00065 |
| IXI          | IXI_K24               | IXI_K8          |               0.00056 |  0.00023 |  0.00091 |
| IXI          | IXI_adaptive          | IXI_K8          |              -0.00041 | -0.00108 |  0.00041 |
| IXI          | IXI_piece_C0          | IXI_K8          |               0.00021 | -0.00021 |  0.00066 |
| IXI          | IXI_piece_C1          | IXI_K8          |               0.00067 |  8e-05   |  0.00144 |
| SyntheticMRI | SyntheticMRI_K8       | SyntheticMRI_K8 |               0       |  0       |  0       |
| SyntheticMRI | SyntheticMRI_K16      | SyntheticMRI_K8 |              -0.00062 | -0.00225 |  0.00099 |
| SyntheticMRI | SyntheticMRI_K24      | SyntheticMRI_K8 |              -9e-05   | -0.00149 |  0.00129 |
| SyntheticMRI | SyntheticMRI_adaptive | SyntheticMRI_K8 |              -0.00216 | -0.00331 | -0.00077 |
| SyntheticMRI | SyntheticMRI_piece_C0 | SyntheticMRI_K8 |               0.00029 | -0.00083 |  0.00134 |
| SyntheticMRI | SyntheticMRI_piece_C1 | SyntheticMRI_K8 |              -0.00111 | -0.0026  | -3e-05   |

### BRAVA recovery selection

{
  "dataset": "BRAVA",
  "arm": "D1_IXI",
  "step": 30000,
  "validation_edge_f1": 0.020992285967204823,
  "validation_node_recall": 0.34093419086597804,
  "representation_stage_ready": false,
  "gate": "Predefined feasibility gate: edge F1>0.04 and node recall2mm>0.25; review paired comparisons and annotation limitations before further submission.",
  "automatic_followup_training": false
}

A feasibility gate is not a scientific claim of superiority. Inspect best-versus-final learning curves, independent seed limitations, the selector confusion matrices and long-branch errors. Explicit splitting, recovered-BRAVA geometry learning, full end-to-end representations and node/relation ablations remain separate pending stages.