# Completed experiment review — 22 October 2026

> Later evidence: [23 October completed multi-dataset review](EXPERIMENT_REVIEW_20261023.md). The results and decisions below retain their22 October scope.

Slurm accounting confirms successful completion of BRAVA recovery1421288/1421289, all eight spline-capacity tasks1425071 and collector1425072, and the earlier limited-data botanical chain through collector1372135. The queue was empty when checked. No new jobs were launched for this review.

## 1. Are eight controls sufficient?

Source: [full capacity report](spline_learning/ixi-brava-control-capacity.md). Actual cached8-control targets were compared with their own original dense branch paths on23,213 development patches /349,357 patch-edge occurrences. No test data, masks, training or GPU inference were used. Distances are physical and bidirectional; continuous maximum-error bounds are sampled maximum−0.01 to sampled maximum+0.06 mm.

### Validation populations

|Metric|IXI|BRAVA|
|---|---:|---:|
|Patches / subjects|2,649 /26|1,796 /8|
|Patch-edge occurrences|51,937|19,446|
|Median per-edge maximum error mm|0.009|0.178|
|P95 per-edge maximum error mm|0.149|1.011|
|Worst sampled maximum mm|2.659|5.259|
|Branches definitely exceeding1 mm|0.18%|5.01%|
|Patches containing at least one definite >1-mm branch error|3.32%|35.86%|
|Same-edge Curve F1 at1 mm|0.9986|0.9767|

Long branches are the critical subgroup: among50–100-mm validation branches, definite >1-mm errors occur in58.76% of IXI branches (97 occurrences total) and88.37% of BRAVA branches (576 occurrences total). For BRAVA25–50-mm branches the rate is15.25%; all six >100-mm validation occurrences exceed2 mm. Overlapping crops mean these are not counts of independent anatomical vessels.

On the deliberately difficult BRAVA validation diagnostic subset (72 branches), median maximum error falls2.910→1.483→0.902 mm with8→12→16 controls. P95 falls3.658→2.356→1.397 mm. Extra controls help but16 is not a universal guarantee of submillimetre approximation. These refits keep anchors/connectivity fixed and are target-fitting results, not model results or population-prevalence estimates.

Requested sample_000002_0222 is already a64³ patch. Its longest edge is82.11 mm;8/12/16-control maximum errors are2.930/1.272/0.680 mm, and dense-path coverage within1 mm is60.5%/98.5%/100%.

**Interpretation:** eight controls are generally adequate for the common IXI target branches, so target spline capacity cannot explain most of the previously measured IXI node/relation failures. Long BRAVA branches have a real pre-training approximation bottleneck. A controlled larger-control or adaptive-subdivision representation study is justified; neither learned performance nor query efficiency is established by these GT fits. Subject-level bootstrap intervals and all length/tortuosity strata are in the full report.

## 2. Recovered BRAVA representation analysis and model study

Sources: [Stage1 report](BRAVA_experiment_plan.md), [collected tables](evidence/brava_stage1_20261021/results.md). The one-point reference-path issue was repaired by explicitly retaining the coincident endpoints of the existing SWC link. Both CPU recovery jobs succeeded without retraining or inference.

### Full-subject GT representations — eight validation subjects

|Representation|Mean graph nodes|Common Curve F1|Common Branch F1|
|---|---:|---:|---:|
|Dense SWC|3,143.4|1.0000|1.0000|
|Junction anchors, straight chords|203.0|0.1866|1.0000|
|Curvature-adaptive anchors, straight chords|2,471.4|0.9991|1.0000|
|Junction anchors,8-control fitted splines|203.0|0.8413|1.0000|

All four preserve annotation-relative components/cycles. This analysis uses full-subject graphs; the separate capacity audit above uses the actual model patches. The different Curve F1 values therefore are not contradictory. Internal spline controls are extra geometric parameters, not detector queries; counting graph anchors alone does not establish model-level efficiency.

The earlier model findings remain: IXI zero-shot is strongest among measured starting points (node recall0.2309, positional edge F10.0269, Curve F10.0580). Both fresh BRAVA arms completed23,500 matched updates but recover very few edges; standard has4 true/226 false predicted edges and refinement1 true/68 false over validation. Refiner-minus-standard node-recall95% interval is[-0.030636,-0.025552]. These short single-seed scratch runs do not establish convergence or prove refinement generally harmful. Same-graph IXI spline-minus-chord Curve F1 improves by0.00264 (95% interval[0.001993,0.003249]).

## 3. Earlier limited-data botanical transfer also completed

Source: [full report](pretraining/limited-data-botanical-transfer.md). This is **SyntheticMRI, not IXI or BRAVA**:200 vascular training patches from62 volumes, with or without25,000 leaf pretraining patches. Each arm uses78,150 pretraining and31,300 fine-tuning updates; domain adaptation is disabled. Validation-selected checkpoints were evaluated on the already-authorized held-out SyntheticMRI test population (13,071 patches).

|Test metric|Vascular-only|Leaves + vascular|
|---|---:|---:|
|Native node F1|0.3246|0.3855|
|Native edge F1|0.2588|0.2979|
|Common Curve F1|0.1009|0.1206|
|Common Branch F1|0.0004|0.0005|
|Component MAE|2.4808|2.6998|
|Cycle MAE|3.5468|3.3283|
|Duplicate fraction|0.0068|0.0110|

Leaf diversity helps detection and geometric recovery in this annotation-limited single-seed setting, with mixed structural results: cycles improve, components and duplicates worsen, and absolute Branch F1 remains very low. Both arms have equal optimizer/example-processing budgets but different vascular exposure (approximately3.50M versus2.25M vascular presentations overall). This supports a limited-data transfer benefit under this recipe, without contradicting the earlier full-data vascular-only advantage or establishing statistical superiority across seeds.

## Current priorities

1. Separate BRAVA target approximation from model learning: compare larger spline capacity versus bounded adaptive subdivision using identical dense references.
2. Diagnose the weak scratch-trained BRAVA baseline before a large transfer/architecture matrix; preserve the IXI zero-shot reference.
3. Keep the botanical result scoped to scarce SyntheticMRI annotations; replication and real-data transfer remain separate questions.

No additional training, inference or analysis jobs were submitted during this review.
