# Completed experiment review —23 October2026

## Completion and reporting recovery

All15 training/prediction tasks1460519 and all15 CPU evaluation tasks1460520 completed successfully. BRAVA arms completed30,000 updates on8,768 training patches; each independent IXI/SyntheticMRI head completed20,000 updates on4,096 training patches. No test evaluation or new end-to-end representation training was performed.

Collector1460521 failed **after numerical aggregation**, while loading missing Plotly assets from the archived training runtime. Collection was rerun from saved results using the complete protected plotting environment. All tables, figures, three offline3D viewers and the study report are now published. No inference/retraining was required. The Slurm failure remains in accounting; successful reporting recovery is recorded by `results/collection-complete.json`.

The retrieval-command error was a working-directory issue. From the parent `projects/` directory use `python3 DenseVesselGraph/scripts/show_multidataset_graph.py`; from the repository root use `python3 scripts/show_multidataset_graph.py`.

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
- Compact evidence: [tables and figures](../../../../research/evidence/multidataset_continuous/report.md), [paired intervals](../../../../research/evidence/multidataset_continuous/paired-representations.csv), [analysis details](../../../../research/evidence/multidataset_continuous/interpretation-details.json).
- Held-out-development long-branch viewers: [BRAVA](../../../../research/evidence/multidataset_continuous/BRAVA-example.html), [IXI](../../../../research/evidence/multidataset_continuous/IXI-example.html), [SyntheticMRI](../../../../research/evidence/multidataset_continuous/SyntheticMRI-example.html). These show Hungarian-positive diagnostics, which need not be decoded edges.
- Summary command from the repository root: `python3 scripts/show_multidataset_graph.py`.
- Reproduce this interpretation from saved tables: `python3 scripts/report_multidataset_results.py`.
