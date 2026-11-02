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

