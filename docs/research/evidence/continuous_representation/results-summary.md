## Completed results and interpretation — 22 October 2026

Final collector1446434 completed successfully in28m28s. All18 GT tasks completed; replacement GPU smoke1446429, all three training tasks1446432 and prediction tasks1446433 completed. The earlier collector bugs were fixed and verified against saved smoke outputs. No full detector training, test evaluation or further GPU work was launched.

### BRAVA GT: capacity and graph complexity

All1,796 validation patches /8 subjects /19,446 patch-edge occurrences, referenced to original dense paths. Coverage below uses the conservative continuous symmetric maximum bound, not just average distance.

|Representation|Mean symmetric error mm|P95 branch maximum mm|Branches within0.5 mm|Mean raw controls per original branch|
|---|---:|---:|---:|---:|
|Fixed K8|0.0898|1.0110|77.19%|8|
|Fixed K12|0.0437|0.6135|88.36%|12|
|Fixed K16|0.0266|0.4732|94.02%|16|
|Fixed K24|0.0131|0.3011|98.34%|24|
|Oracle adaptive K,0.5 mm|0.0690|0.4271|98.35%|7.79|
|Piecewise K8 inside one edge,0.5 mm|0.0460|0.4131|100%|10.61|
|Same segments with graph splitting|0.0460|0.4131|100%|10.61|

Adaptive K saves controls by spending the available error tolerance: it does not minimize mean distance. Its labels still fail the0.5-mm criterion on1.65% of branches because none of the available K values passes. Piecewise fitting succeeds on every evaluated BRAVA validation branch at0.5 mm, with22.81% of branches requiring subdivision. At0.25 mm, piecewise success is99.84%, with16.31 raw controls/branch; this is not a universal arbitrary-tolerance guarantee.

At0.5 mm, explicit graph splitting increases mean nodes17.14→20.67 (+20.6%), max nodes69→97, and mean all-node candidate pairs206.3→305.7 (+48.2%). All validation patches still fit256 queries. Internal piecewise segments retain17.14 graph nodes and the original pair population. Both store the same106.83 unique coordinate vectors/patch after shared-endpoint accounting, versus82.11 for fixed K8 and255.34 for fixed K24. Extra graph nodes therefore provide no GT geometry benefit over the identical internal segmentation; their learned detector/relation costs remain untested. Internal joins are C0, without enforced tangent continuity.

### IXI and SyntheticMRI extensions

Focused subject-balanced validation subsets of480 patches each, not repeated full-population audits. Fixed K8 already meets0.5 mm for99.14% of IXI branches and99.32% of SyntheticMRI branches. Oracle adaptive selection meets the tolerance for100% using4.38/4.58 controls on average. Piecewise K8 also reaches100%, using8.08/8.05 controls on average. The larger representation bottleneck is concentrated in BRAVA and rare complex branches; these results do not explain general IXI connectivity failures.

### Learned geometry: matched detector and training budget

Common frozen IXI detector;2,048 BRAVA training patches /39 subjects and512 validation patches /8 subjects. Each K8/K16/K24 head completed8,000 batch4 updates, identical order and frozen-state hash, approximately15.6 training-population passes. Actual training took17.6–18.7 minutes (job elapsed18–20 minutes), **shorter than the nominal one-hour target because the8,000-update cap bound the budget**. These are finite-budget geometry pilots, not convergence evidence. Trainable parameters1.578M/1.593M/1.608M.

|Method|Validation common-v2 Curve F1|Mean matched-pair error mm|Endpoint-neutral mean error mm|
|---|---:|---:|---:|
|Original IXI pretrained K8|0.05782|not cached|not cached|
|BRAVA-trained frozen head K8|0.06050|3.10090|1.39998|
|BRAVA-trained frozen head K16|0.06064|3.09981|1.39997|
|BRAVA-trained frozen head K24|0.06087|3.10013|1.40004|
|K24 + geometric compression|0.06058|3.10123|1.40081|
|K24 + separately learned compression selector|0.06062|3.10077|1.40017|

K24−K8 Curve F1 is+0.000363 (paired subject-bootstrap95% interval[+0.000145,+0.000611]). This is a small measurable change, not a substantial graph reconstruction improvement. K8 adaptation versus the original head is+0.002682 [0.001986,0.003408]. These intervals resample8 subjects and recompute pooled common-v2 Curve F1; they do not cover training-seed uncertainty. Endpoint-neutral K24−K8 mean-error interval[-0.001479,+0.001774] mm includes zero. Fine curve capacity therefore has no convincing benefit in that geometry diagnostic under this setup.

Only81/5,445 Hungarian-positive validation pairs (1.49%) have both predicted endpoints within1 mm of their assigned GT endpoints. For that small subset, mean errors K8/K16/K24 are0.8893/0.8875/0.8912 mm. Hungarian assignment is not correct localization. The large gap between near-perfect target fits and ~3.10-mm predicted-curve errors primarily reflects the weak detector/geometry learning setting; it does not invalidate adaptive representations.

All heads share node recall23.47% at2 mm, edge F10.02696, Branch F10.00250, component MAE7.564 and cycle MAE0.410. They predict106 correct and2,313 incorrect edges;97.84% of incorrect edges involve at least one spatially unmatched node. Of5,445 GT edges, only454 have both endpoints recovered at2 mm, and106 are predicted (conditional recall23.35%). These connectivity quantities are unchanged by construction. Node selection/localization and relation learning remain critical bottlenecks.

### Adaptive inference and limits

Geometric simplification compresses predicted K24 curves to4.04 controls on average, with nearly unchanged error and Curve F1. The separately trained classifier uses6.69 controls on validation and offers no accuracy advantage; even oracle GT labels selecting compression give essentially unchanged error. This is evidence for compact storage of these predictions, not genuine variable-capacity neural prediction. Compression tolerances are measured relative to the predicted curve and do not certify accuracy against anatomy. The classifier sees predicted curve features only; GT complexity labels are confined to training.

### Recommended direction

**Most promising representation candidate: piecewise geometry inside each existing graph edge**, because the GT study reaches the chosen accuracy with moderate geometric complexity and no additional detector queries or relations. This is a representation result, not yet a learned-model result. Keep fixed K16/K24 as simple controls; adaptive fixed-K labels are also attractive for output efficiency. Explicit degree-2 splitting should be an alternative only if its added graph supervision yields a measurable benefit.

Before full end-to-end comparisons, establish a useful IXI→BRAVA fine-tuned detector with unchanged losses and fixed-population monitoring. Then test a bounded internal-piecewise geometry head with GT-anchor/oracle-topology geometry learning and a valid inference-time segment decision. A K-specific projection bank and learned internal-piecewise or split-graph generators were **not trained in this batch**. No new joint detector losses or architectural sweeps are justified by these results alone.

Evidence: `docs/research/evidence/continuous_representation/`; standalone viewer `overlays.html`; learning curves `training-curves.png`. Additional paired Curve F1 bootstrap is reproducible with `python3 scripts/summarize_continuous_results.py` using saved caches only.
