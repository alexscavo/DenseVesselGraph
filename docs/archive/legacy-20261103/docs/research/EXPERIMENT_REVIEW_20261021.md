# Experiment review — jobs run on 20 October 2026

Reviewed on 21 October against the live queue, Slurm accounting, submission records and collected evidence. The queue is empty. Five logical batches completed; the separate limited-data botanical batch failed its GPU smoke and its dependent training/evaluation jobs were cancelled. No new jobs, inference or training were launched for this review.

## 1. Execution inventory

|Batch|Jobs / successful collector|Outcome|
|---|---|---|
|Matching, generalization and calibration|1316107 analyses; 1316108 collector|Completed, cached 256 train / 128 validation patches|
|Image-aware relation adapters|1331642 training; 1344394 reporting recovery; 1344395 collector|Completed, 2,048 train / 512 validation patches|
|Node classification weights/focal|1332028 training; 1332029 prediction; 1332030 evaluation; 1332031 collector|All three arms completed, 2,048 / 512 patches|
|Raw MRA versus vessel-mask input|1339163 training; 1339164 prediction; 1339165 evaluation; 1339166 collector|Both arms completed, 2,048 / 511 paired patches|
|BRAVA preparation|1339525–1339532, subject array 1339528|All stages completed; model-ready targets available|
|Limited-data botanical transfer|1344027 CPU check; 1344474 GPU smoke; 1344475 and 1344484–1344488 downstream|CPU check passed; GPU smoke timed out; downstream jobs cancelled|

Earlier failures are preserved: image evaluation jobs 1331643 and all four 1331645 tasks failed during degree-histogram reporting after saving primary metrics. Integer conversion fixed this; CPU recovery and collection succeeded without inference or retraining. BRAVA audits 1335951/1336449 failed on tracer scale/orientation respectively; the repaired all-subject audit and final chain succeeded. Limited-data CPU attempt 1342759 failed during runtime staging, followed by successful 1344027; its later GPU smoke remains unresolved.

The IXI development batches use 123 training and 26 disjoint validation subjects. Unless otherwise specified, tables below use 2 mm positional matching; common-v2 Branch/Curve metrics retain their own unchanged geometric definitions. Native box AP and positional F1 measure different quantities. Runs have one optimization seed; paired subject bootstraps quantify subject variability, not seed robustness. Do not directly compare scores between batches with different initializers, losses, populations or budgets.

## 2. Segmentation input: appearance matters, connectivity remains weak

Source: [report 11](metrics/11-ixi-segmentation-input-oracle.md). Common pre-IXI SyntheticMRI initialization, identical architecture and stable losses, 10,000 updates per arm. Train all detector components with the existing spline head frozen. Both arms received identical paired IDs and seeded ordering. Actual optimization times were approximately 55.7 minutes raw / 35.2 minutes mask; update counts, rather than elapsed times, were matched.

Masks are supplied labels used to extract the graph targets: **provenance category 2, mask-to-graph reconstruction**, with unknown upstream annotation method. They are not graph rasterizations and are not independently annotated evaluation labels. One unsupported validation patch was excluded from both arms; 2,048 train / 511 validation patches remained. Binary input bypasses MRI intensity processing.

### Matched final-update validation

|Metric|Raw MRA|Vessel mask|
|---|---:|---:|
|Native node mAP|0.1347|0.2210|
|Native node mAR|0.2633|0.3444|
|Native edge mAP|0.0097|0.0276|
|Native edge mAR|0.0243|0.0532|
|Positional node precision|0.4383|0.5871|
|Positional node recall|0.5488|0.6580|
|Positional edge F1|0.0415|0.0904|
|Conditional relation recall|0.0635|0.1180|
|Common Branch F1|0.0061|0.0243|
|Common Curve F1|0.0931|0.1586|
|Duplicates / recovered GT node|0.3902|0.2969|
|Node-count MAE|8.8297|5.4560|
|Component-count MAE|21.9041|17.9041|
|Cycle-count MAE|0.7065|0.8630|
|SMD, native normalized convention|0.1089|0.0610|

Mask-minus-raw paired 95% intervals: node recall **[+0.0822,+0.1378]**, node precision **[+0.1311,+0.1676]**, edge F1 **[+0.0409,+0.0570]**, Curve F1 **[+0.0586,+0.0733]**, duplicates/recovered node **[−0.1262,−0.0593]**. Cycle MAE increases: **[+0.0859,+0.2335]**. These use 1,000 resamples of the same 26 validation subjects.

Raw-query recall before confidence filtering increases 0.7708→0.8353. Mean localization error among 2-mm matched predictions decreases 0.9989→0.8961 mm; these are different matched populations, not paired errors on identical nodes.

**What it shows:** removing much of the appearance-learning problem improves detection, uniqueness and geometric recovery. Appearance is a contributing bottleneck under this protocol. It is not sufficient: only 549 of 9,931 GT edges are recovered at the standard operating point, despite 4,652 GT edges having both endpoints recovered. Raw input recovers 240 edges with 3,780 eligible endpoint pairs. Predicted false edges rise 1,397→1,660 even as F1 improves. This is partial graph improvement, not successful topology reconstruction.

Calibration matters: training-max-F1 thresholds give validation edge F1 0.1130 raw / 0.1548 mask, Curve F1 0.3902 / 0.4019 and cycle MAE 16.9100 / 9.2133. Conditional recall at these separately calibrated policies is 0.4511 / 0.3867. Consequently, the fixed-cutoff conditional recall gap does not establish better relation ranking by itself.

Learning curves are not monotonic. Both predeclared validation-selection rules select update 4,000; there, edge F1 is 0.0533 raw / 0.1245 mask and Curve F1 0.1416 / 0.2215. At update 1,000, Curve F1 was 0.1881 / 0.2393. More training improved node recovery without consistently improving relations. Do not infer an architectural ceiling or convergence from this short experiment. Synthetic raw-image pretraining remains an initialization bias, and the frozen spline head constrains geometry.

## 3. Node-loss study: class weights trade precision for recovery; focal worsens this continuation

Source: [report 10](metrics/10-ixi-node-focal-ablation.md). Three common-initializer, full-detector arms, 6,900 updates, identical edge CE plus ratio upsampling 0.15. Mature IXI initialization previously used focal edges, so all three share an explicit edge-loss transition and optimizer restart. This isolates node settings conditional on that continuation; it is not a from-scratch focal-loss verdict.

|Validation metric|A: CE [.25,.75]|B: CE [1,1]|C: focal gamma 2 [1,1]|
|---|---:|---:|---:|
|Node precision|0.4365|0.5474|0.5293|
|Node recall|0.5522|0.4225|0.3071|
|Edge F1|0.1189|0.1165|0.0767|
|Branch F1|0.0216|0.0181|0.0063|
|Curve F1|0.3604|0.2528|0.1589|
|Duplicates / recovered GT|0.3992|0.2220|0.2827|
|High-confidence duplicates (score ≥.9), total|1,432|433|0|
|Node-count bias, nodes/patch|+6.2383|−5.3730|−9.8867|
|Node-count MAE|8.4102|7.3965|11.0000|
|Cycle MAE|13.7246|3.9160|2.6875|
|Raw-query node recall before filtering|0.8093|0.8089|0.7228|

**A→B, weights alone:** precision improves and duplicates decline, but node recall loses 12.96 percentage points (95% interval **[−14.32,−11.52] pp**). Edge-F1 difference is inconclusive (**[−0.0120,+0.0079]**), while Curve F1 decreases (**[−0.1190,−0.0961]**). This suppresses legitimate nodes as well as duplicates.

**B→C, focal modulation at equal weights:** node recall, edge F1 and Curve F1 all decrease. Edge-F1 difference **[−0.0510,−0.0301]**; Curve-F1 difference **[−0.1079,−0.0810]**. Duplicate multiplicity actually increases versus equal-weight CE (**[+0.0200,+0.1019]**).

Zero high-confidence duplicates in C is not successful discrimination: confidence shifts and many legitimate nodes disappear. Pooled correct-versus-duplicate confidence AUC remains weak (A 0.6094; B 0.6009; C 0.5980). Before filtering, C also loses GT-node coverage, so the observed degradation is not solely an inference threshold effect. C's training recall is only 0.3383 versus validation 0.3071; this is not simply successful training fit that fails to generalize. C degrades especially between updates 2,000 and 6,900 (validation edge F1 0.1207→0.0767).

**Decision supported:** no reason to adopt node focal under this tested IXI continuation. Equal weights are a precision/fragmentation trade-off, not an overall reconstruction win. Fewer cycles alone are insufficient evidence of better topology. The synthetic-MRI colleague result does not automatically transfer to IXI; initialization, edge transition, optimization and calibration remain relevant limitations.

## 4. Image-aware relation adapters: early positive signal, severe late overfitting

Source: [report 9](metrics/9-ixi-image-aware-relation-learning.md). Shared frozen local-refinement detector/cache, equal-capacity zero-initialized residual adapters, identical candidate pairs/objective/ordering. 65,600 common updates; actual arm training times approximately 24.1–51.4 minutes. Standard scores below are conditional relation AP at 2 mm on the same 512 validation patches.

|Scorer|Validation AP, 1k|Validation AP, 2k|Validation AP, final|Training AP, final|Final validation all-retained AP|
|---|---:|---:|---:|---:|---:|
|Original frozen head|0.4723|0.4723|0.4723|0.4856|0.1253|
|A: token/geometry|0.4753|0.4766|0.3058|0.6044|0.0777|
|B: endpoint CNN|0.4898|0.4907|0.3113|0.6394|0.0833|
|C: endpoint + chord|0.4955|0.5016|0.3216|0.6468|0.0882|
|D: endpoint + corridor|0.4944|0.5019|0.3211|0.6445|0.0889|

At 2k, C/D outperform original and A by about 0.030 / 0.025 conditional AP respectively. This is a real descriptive ranking difference on the saved populations, not a threshold effect. It is promising evidence to examine, but early-checkpoint paired intervals and full calibrated graph metrics were not produced by the final-checkpoint collector. No validation-selected replacement is declared here.

At final update, all adapters rank worse than original. For D minus original, the paired 95% conditional-AP interval is **[−0.1681,−0.1319]**. Conditional-AP differences between the final image arms and A have intervals including zero; some all-retained AP differences versus A are positive. Additional image evidence therefore should not be declared useless, but the tested long-training recipe does not yield a winning final head.

Training-max-F1 calibrated final graph comparison:

|Scorer|Validation edge F1|Curve F1|Cycle MAE|
|---|---:|---:|---:|
|Original|0.1533|0.3874|13.2949|
|A|0.1237|0.3705|26.0703|
|B|0.1236|0.3710|19.7402|
|C|0.1335|0.4044|20.2598|
|D|0.1261|0.3911|13.7051|

C's higher Curve F1 comes with lower edge F1 and more cycle error. At fixed 0.5, B raises Curve F1 0.1732→0.3284 but false edges rise 2,821→12,397 and cycle MAE 0.9941→9.7832. Thus more connections and higher geometric coverage do not establish better connectivity.

**Decision supported:** inspect/confirm the early signal from existing scores before another architectural experiment. More optimization is demonstrably not an automatic remedy. The final-head train/validation divergence is substantial.

## 5. Hungarian matching and calibrated relation-loss comparisons

Source: [report 8](metrics/8-ixi-relation-matching-calibration.md), 256 train / 128 validation patches, cache-only analysis with original scores and existing trained relation heads. No new training.

|Original head's query population|Node recall|Validation conditional AP|Conditional recall at fixed cutoff|
|---|---:|---:|---:|
|Normal confidence selection|0.5627|0.4702|0.1257|
|Spatial oracle, all queries|0.8362|0.3685|0.0722|
|Training-style Hungarian|0.6267|0.4658|0.1619|

Hungarian-minus-normal conditional-AP interval **[−0.0331,+0.0268]** includes zero. It improves some endpoint selection/counts but does not materially fix conditional relation ranking. Training Hungarian conditional AP is 0.5100 versus validation 0.4658. A Hungarian assignment can be distant; all recovery results above enforce physical tolerance. Query populations/prevalence differ, so raw AP differences between selections are not a clean causal matching effect.

|Relation head|Train conditional AP|Validation conditional AP|Validation edge F1, training-max-F1 threshold|
|---|---:|---:|---:|
|Original|0.4848|0.4702|0.1608|
|Focal continuation|0.6624|0.4098|0.1374|
|Regularized CE|0.5874|0.4153|0.1446|
|CE + unmatched hard negatives|0.5417|0.3659|0.1343|
|CE + matched hard negatives|0.6268|0.3996|0.1360|

All retrained heads have negative paired conditional-AP intervals versus original. Equal *training* precision constraints do not guarantee equal *validation* precision. Some retrained heads improve F1/Curve F1 under that calibration policy, but with worse cycle errors; none establishes an overall ranking/topology win.

**Decision supported:** training–inference matching mismatch is not the sole explanation, and the tested extra relation losses/mining do not justify adoption. Threshold changes matter but cannot repair ranking. Original calibrated F1 itself comes at a major cycle penalty: fixed 0.5 edge F1 0.0737 / cycle MAE 1.0391, versus training-max-F1 0.1608 / 19.5234.

## 6. BRAVA is prepared; this is not yet a model benchmark

Source: [preparation report](../../../../research/BRAVA_preparation.md). All 56 paired image subjects and 111 valid subject/trace variants pass the repaired audit. Explicit hash-pinned x reflections apply to BG0014/BG0019–BG0022. Original images/masks/SWC files are preserved; graph links/topology are not changed by the coordinate correction. Skull-stripped images and dense/junction/adaptive graph exports are complete.

Subject-disjoint split: 39 train / 8 validation / 9 test subjects. Model-ready patches: **8,768 / 1,796 / 2,043**, with no recorded target exclusions. Six additional trace subjects have no paired brain image and are inventoried as orphans rather than inserted into supervised training. Target `READY` exists.

Model root: `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057/transfer_splines_k8`.

No vessel segmentation was generated. `seg/` contains supplied **brain masks** for loader compatibility; these must not be used as vessel input or vessel-support labels. SWC parent forests and incomplete trace coverage do not certify complete anatomical topology or absence of cycles. No BRAVA model accuracy is available from this preparation batch. Preparing test targets is not test-set model selection.

## 7. Limited-data botanical transfer did not run

Source: [study definition](pretraining/limited-data-botanical-transfer.md). Intended comparison: 200 SyntheticMRI training patches (62 training volumes) alone versus those same 200 plus 25,000 leaf patches, same scratch initializer and optimizer-update budgets. This is the appropriate distinct question of botanical transfer under scarce vascular annotations; the full-data source comparison does not answer it.

CPU correctness job 1344027 passed actual small MRI/mixed/spline forward/backward checks. GPU smoke 1344474 timed out after 30m29s. Its error log contains **`OSError: AF_UNIX path too long`** from the DataLoader multiprocessing resource sharer and additional dependency-import warnings. The precise verified outcome is a failed smoke, not a negative transfer result. Its dependent pretraining, fine-tuning, test and collection jobs were cancelled before training started.

No performance numbers, winning model or conclusion about low-data plant/leaf transfer can be drawn. This review does not repair/resubmit it or introduce another training run.

## 8. Combined interpretation and next decisions

1. **Appearance contributes to IXI failure:** paired mask input improves nodes and graph recovery, supported by subject intervals. Category-2 label dependence limits the claim to mask-to-graph reconstruction.
2. **Visibility alone is insufficient:** even with mask input, endpoint recovery and conditional relation decisions remain weak. Fixed-cutoff weakness must be separated from calibration, optimization and ranking; it is not proof of missing anatomical information in the network.
3. **Node uniqueness is a precision/recall problem:** stronger suppression through class weights/focal removes legitimate nodes. The tested focal recipe is not a justified default for IXI.
4. **Image information may help relations early:** endpoint/chord/corridor AP gains at predefined early updates warrant cached-score uncertainty/graph analysis. The final adapters overfit, and no larger transformer is justified solely by these results.
5. **Matching/mining are not demonstrated remedies:** actual Hungarian evaluation and fairer calibration fail to establish a superior retrained relation head.
6. **The strongest current baselines remain important:** these findings do not overturn the earlier full-data SyntheticMRI result favoring vessel-only training over plants/leaves. That study used different data/targets from the new IXI diagnostics; do not combine its scores into a causal dataset comparison.
7. **Bound the next work:** first use cached early adapter scores for paired AP uncertainty and calibrated graph metrics; review the mask/node learning curves and retained confidence behavior. Treat restarting the failed low-data botanical study as a separate infrastructure repair/authorized continuation. Do not start an architecture/loss sweep from this review.

## Retrieval without rerunning experiments

```bash
python3 scripts/collect_ixi_relation_matching_calibration.py --show
python3 scripts/collect_ixi_image_relations.py --show
python3 scripts/collect_ixi_node_focal.py --show
python3 scripts/collect_ixi_segmentation_oracle.py --show
python3 scripts/prepare_brava.py show --output /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava/prepared_swc_20261020-132057
```

Canonical experiment reports preserve exact checkpoints, manifests, seeds, metric definitions, per-subject/per-patch results, curves and bootstrap files. Their original results have not been overwritten by this synthesis.

## BRAVA Stage 1 recovery — 22 October 2026

This dated update supersedes the preliminary BRAVA completion review and earlier preparation-only status. Training and inference were reused; recovered CPU representation analysis and final aggregation completed.

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

