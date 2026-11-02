# Experiment register and results

<!-- NATIVE2D CURRENT POINTER 20261028 -->
> Latest native2D result: **Native2D-03 corrected-target pilot completed** (B0/A-xyz6,000 updates each; no filtered graph recovery at0.5). [Final comparison, dataset readiness and stopping decision](../docs/research/native2d_heldout_20261028.md#completed-corrected-target-comparison). Native2D-02 remains preserved with its unstable-target and post-training-timeout caveats.


Latest results: [D08 and Native2D-01 reviewed completion](#d08-and-native2d-01-final-reviewed-results--28-october-2026).

For stable experiment IDs, status distinctions and launch/stop gates, use the [stable-ID vascular experiment registry](VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md). This document retains the detailed historical experiment index; the imported registry has been reconciled with collected N08 results.

Latest completed follow-up (25 October): [two-seed longer local refinement](../docs/research/local_refinement_long_study.md), eight matched 12,000-update runs. Image-conditioned coordinate-only A retains most combined-A gains over B0 and the no-image control; a bounded IXI transfer is recommended but not launched. [Quota audit and archive locations](../docs/research/storage/cleanup-20261025.md).


Consolidated 23 October 2026. This is the compact entry point to the experiment history. It preserves positive, negative and inconclusive findings; detailed protocols, exact tables, uncertainty, launch/recovery records and evidence remain linked below. Historical job submission is not proof of current completion. Compare numbers only within matched populations, target conventions, checkpoint-selection rules and metric versions.

## Completed node-verifier feasibility and BRAVA learnability —23 October2026

[Full protocol/results](../docs/research/node_verifier_feasibility.md), [resolved protocol](../experiments/node_verifier_feasibility/protocol.json), [jobs](../experiments/node_verifier_feasibility/jobs.json), [artifacts](../experiments/node_verifier_feasibility/). This later study is separate from the earlier coordinate-refinement and multi-dataset geometry arms; no test data were evaluated.

V0 is original confidence at 0.5; V1 is original confidence with training-selected threshold/NMS; V2 independent and V3 five-neighbor contextual residual scorers have matched 58,129-parameter capacity. Each dataset uses 256 training /128 subject-disjoint validation patches (IXI 123/26 subjects; SyntheticMRI 64/13 volumes), 2,000 updates, batch 8, AdamW LR 0.001, seed 231023. Stage 1 is frozen, coordinates/candidates unchanged, and genuine original relation scores remain available for all raw pairs at edge cutoff 0.5. Final update 2,000 was predeclared. Ambiguous spatial labels are ignored; duplicates receive reduced negative weight. No new jointly optimized detector loss was added.

**Validation at 2 mm; F1 values on the 0–1 scale:**

| Dataset | Arm | Training-calibrated node F1 | Equal-count node recall | Equal-count node F1 | Equal-count edge F1 |
|---|---|---:|---:|---:|---:|
| IXI | V0 | 0.4677 | 0.4812 | 0.4677 | 0.0513 |
| IXI | V1 | 0.4693 | 0.4812 | 0.4677 | 0.0513 |
| IXI | V2 | 0.3484 | 0.3939 | 0.3829 | 0.0215 |
| IXI | V3 | 0.3655 | 0.4104 | 0.3989 | 0.0195 |
| SyntheticMRI | V0 | 0.7525 | 0.7804 | 0.7525 | 0.4455 |
| SyntheticMRI | V1 | 0.7634 | 0.7804 | 0.7525 | 0.4455 |
| SyntheticMRI | V2 | 0.6767 | 0.6961 | 0.6713 | 0.3093 |
| SyntheticMRI | V3 | 0.6811 | 0.7057 | 0.6805 | 0.3273 |

Equal-count policies retain V0's exact per-patch count, not a GT count. V1 selects radius zero (threshold 0.70 IXI /0.85 SyntheticMRI), hence its equal-count ranking is V0's. V3−V1 equal-count node-F1 95% subject-bootstrap intervals: IXI [−0.0810, −0.0563], SyntheticMRI [−0.0819, −0.0629]; edge-F1 intervals [−0.0418, −0.0222] /[−0.1318, −0.1062]. All use 1,000 subject draws and omit seed uncertainty. Validation-matched-precision diagnostics also fail; their thresholds use validation labels only to describe the PR envelope.

Canonical AP train/validation: IXI V0 0.3080/0.3103, V2 0.4872/0.2187, V3 0.5304/0.2191; SyntheticMRI V0 0.5955/0.5606, V2 0.6919/0.4376, V3 0.7362/0.4614. Tiny clean-label fits succeed and early AP sometimes improves, so this supports a supervision/ranking/generalization problem in this recipe rather than absent gradients or universally uninformative embeddings. **Feasibility gate failed on both datasets; retain V1, revise supervision first.** BRAVA V2/V3 fitting and transfer were not launched.

Full-population cached bounds (distinct from the 128-patch verifier results): 2-mm raw/filtered recall IXI 0.78179/0.49702 (512 patches), SyntheticMRI 0.87411/0.78601 (480), BRAVA D1 0.56318/0.34093 (1,796). BRAVA minimum-distance GT node selection gives unchanged-head edge F1 0.03524 and conditional relation recall 0.06676. These are GT-dependent candidate-selection diagnostics, not deployable results or relation-maximizing oracles. BRAVA's separate 128-patch V0→V1 policy improves node F1 0.2806→0.2935 while edge F1 falls 0.01558→0.01469; matched-count NMS decreases node F1.

**Separate BRAVA learnability check:** eight fixed training patches from eight subjects, 100 nodes/55 edges; full detector unfrozen, 512 updates, batch 4, no augmentation, node CE `[0.25,0.75]`, edge CE, K8 `graph_only`. The prior full-population recovery used edge focal; this is not a matched continuation.

| Initializer | Final raw /filtered recall 2 mm | Eval node F1 | Eval edge F1 | Mean Hungarian distance |
|---|---:|---:|---:|---:|
| Scratch | 0.930 /0.930 | 0.8986 | 0.8393 | 0.886 mm |
| IXI | 0.990 /0.990 | 0.9900 | 0.9818 | 0.330 mm |

Scratch train-mode edge F1 is 0.9273 versus eval 0.8393; IXI 0.9725 versus 0.9818. Localization gradients reach backbone, decoder and coordinate head. This establishes learnability on selected training targets, not full-population generalization or BN/loss causality. The historical BRAVA validation-recovery gate remains unmet; no longer scratch run is justified by this check alone.

Jobs 1483623 (SyntheticMRI verifier, 2m09s), 1483624 (IXI verifier, 2m09s), 1483632 (BRAVA scratch, 1m33s), 1483633 (BRAVA IXI, 1m31s) completed with exit 0. BRAVA jobs used `afterany:1483623:1483624`, limiting GPU use to two. Initial staging attempts 1483320/1483324 failed before inference on workspace quota; replacement staging used node-local temporary storage. Only failed temporary runtime copies were removed; results were preserved. CPU collection reused predictions/weights. For the source report's exact tables and figures, run in the established Python environment:

```bash
python scripts/report_node_verifier.py
```

This study adds physical node/edge and topology diagnostics; it does not supply new spline/Curve-F1 results or replace the historical coordinate-refinement findings.

## Current multi-dataset batch

[Full protocol](../docs/research/multidataset_continuous_graph_experiments.md), [exact plan/manifests](../experiments/multidataset_continuous/plan.json), [submission and replaced jobs](../experiments/multidataset_continuous/submission.json), [recovery](../experiments/multidataset_continuous/recovery.json).

| Dataset/stage | Arms | Data | Training/control |
|---|---|---|---|
| BRAVA detector recovery | IXI initialization; SyntheticMRI initialization; scratch, all K8 | 8,768 train /1,796 validation; 39/8 subjects | 30,000 updates, batch8; whole detector trainable, common losses, backbone LR3.5e-5 /detector LR7e-5; warmup then cosine to30% LR floor |
| IXI frozen geometry | K8/K16/K24; actual adaptive-K bank; four-section C0/C1 heads | 4,096 train /512 validation; 123/26 subjects | 20,000 updates, batch4; frozen detector and BN; identical graph/candidate population |
| SyntheticMRI frozen geometry | Same six arms | 4,096 train /480 validation; 64/13 volumes | Same controlled head budget; native120-query source detector |

The adaptive bank uses K∈{4,8,12,16,24}, K-specific output projections and a separately trained selector on detached training features. Piecewise C0/C1 use29/26 global control vectors, respectively, and no detector queries for joins. Explicit split targets and branch weights are implemented, but split-model training and full end-to-end representation comparisons remain pending prerequisites.

Audit1451641 passed. Smoke1451642 failed before main training on a missing orientation import; replacement1460518 passed. All15 training/prediction tasks1460519 and evaluations1460520 completed. Collector1460521 failed only during Plotly export after numerical aggregation; CPU replay with complete protected plotting packages published all reports/figures successfully. No model work was repeated. See the [23 October review](../docs/research/EXPERIMENT_REVIEW_20261023.md) and [final tables](../docs/research/evidence/multidataset_continuous/report.md).

Large root: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/multidataset_continuous_20261022`. Read current saved progress/results:

```bash
python3 scripts/show_multidataset_graph.py
```

## Completed multi-dataset results —23 October2026

BRAVA final default validation (1,796 patches): IXI-initialized node recall2mm0.34093 /edge F10.02099 /Curve F10.02220; SyntheticMRI-initialized0.27331 /0.01340 /0.01094; scratch0.10557 /0.00040 /0.00100. IXI transfer improves node recovery over zero-shot, but does not recover reliable connectivity. Its training-selected edge threshold0.425 raises edge F1 to0.03696 with21,236 false edges and increased cycles; the practical gate still fails. Scratch is poor on monitored training patches as well.

|Learned head|IXI validation Curve F1|SyntheticMRI validation Curve F1|IXI /SyntheticMRI mean output controls|
|---|---:|---:|---:|
|K8|0.12149|0.58386|8 /8|
|K16|0.12177|0.58324|16 /16|
|K24|0.12205|0.58377|24 /24|
|Actual adaptive K|0.12108|0.58170|4.31 /4.54|
|Four-section C0|0.12170|0.58415|29 /29|
|Four-section C1|0.12216|0.58275|26 /26|

Frozen-graph comparisons: nodes, relations and positional edge metrics are identical across heads. IXI K24/C1 minusK8 Curve F1 differences are +0.000558 [+0.000227, +0.000910] and +0.000666 [+0.000078, +0.001435]. SyntheticMRI adaptive/C1 differences are -0.002158 [-0.003313, -0.000770] and -0.001113 [-0.002604, -0.000028]. All are small; no consistent multi-dataset accuracy advantage is established. Exact figures and limitations are in the linked review/tables; uncertainty covers subjects, not seeds.

Adaptive selection is genuine native-projection prediction, but imbalanced labels hide low complex-class recall. Output compactness is not reduced model size or training cost. Four uniform parameter sections are not the exact earlier error-adaptive GT partition. Recovered-BRAVA geometry, split-model/oracle-node training, end-to-end representation comparisons and isolated node/relation ablations remain pending.

## Historical experiment register

“Completed” refers to the cited scoped study, not every later proposal appearing in the same document.

| Study | Configuration/question | Results and interpretation | Detailed record |
|---|---|---|---|
| Baseline, focal, balancing and Betti matrices | Reconstructed 3D baseline; matched loss/matcher variants and candidate-budget rules | Keep the original experiment definitions; no universal ranking is inferred from a config's existence | [Early matrices](../docs/EXPERIMENTS.md), [config contract](../configs/README.md) |
| Source/grid and data audits | Boundary-corrected SyntheticMRI; source split/restoration and image–graph checks | Validated conventions and documented repair/exclusion history; extracted guides remain uncertain anatomy | [Data audit](../docs/DATA_AUDIT.md), [dataset findings](../docs/SYNTHETIC_MRI_DATASET_REPORT.md), [restoration](../docs/research/spline_learning/synthetic-mri-restoration.md) |
| IXI graph GT and rate–distortion, T01–T06 | Rings, immutable dense references, junction/adaptive/dense products; radius/containment policy; sampled sites/resolutions | Representation fidelity and topology preservation precede model claims; preserve scoped studies and production caveats | [Tracker](../docs/research/IMPLEMENTATION_TRACKER.md), [T05](../docs/research/evidence/T05/README.md), [adaptive tuning](../docs/research/evidence/adaptive_tuning/README.md) |
| Vedo extraction and production | Original Vedo centerlines, explicit mask repair,220-volume checks | Production/QC improved target reproducibility; mask containment and Betti preservation do not certify anatomical connectivity | [Production](../docs/research/VEDO_PRODUCTION_RUN.md), [CT-001 comparison](../docs/research/evidence/vedo_original_ct001/full_volume/README.md) |
| Query-capacity benchmarks |120 vs192 and192 vs256 on their paired cohorts | Earlier limits overflowed some targets; capacity/compute trade-offs must retain their representation/population | [120/192](../docs/research/evidence/ixi_query_benchmark/RESULTS.md), [192/256](../docs/research/evidence/ixi_capacity_192_256/README.md) |
| Metric validation, T07–T08 | Physical branch contraction, missing edges, shortcuts, close bridges, wrong pairing and altered curve interiors | Branch F1 detects failures that geometric coverage and Betti counts can miss | [T07](../docs/research/evidence/T07/README.md), [T08](../docs/research/evidence/T08/README.md) |
| Historical IXI300-epoch run, T09 | Earlier adaptive target lineage and held-out test | Native node/edge F1~0.409/0.307 but physical Branch F1~0.00535; overconnection remained. Different cohort from newer spline-target IXI | [Test record](../docs/research/evidence/T09_long/test_results.md) |
| Oracle curve families | Single Bézier, B-spline and piecewise curves; Catmull–Rom examples | Supplied endpoints/topology establish representability, not neural learnability | [Family overview](../docs/research/spline_learning/README.md), [single cases](../docs/research/spline_learning/catmull-rom-single-cases.md) |
| Early spline V0/V1, weight and staged training | Fixed K8, physical residuals, loss weights and delayed head activation | Feasibility, geometry collapse and loss-route effects motivated controlled isolation; all failed arms remain evidence | [Learning log](../docs/research/spline_learning/spline-learning-report.md), [weights](../docs/research/spline_learning/spline-weight-pilot.md), [staged pilot](../docs/research/spline_learning/spline-staged-learning.md) |
| Coupled scratch straight/B-spline/adaptive degree-2 | Original300-epoch comparison, with best/final checkpoints | Coupled B-spline collapsed; degree-2 graph changes did not establish a general learned advantage. Do not confuse this with later detached training | [Held-out comparison](../docs/research/spline_learning/scratch-comparison-results.md) |
| Frozen token/residual/local geometry and unfreezing | Fixed detector; feature/scale/initialization controls; local CNN; controlled backward routes and longer stability | Residual/local information can improve shape, but longer training and broader unfreezing do not uniformly improve held-out graph quality | [Frozen audit](../docs/research/spline_learning/frozen-spline-investigation.md), [residual probes](../docs/research/spline_learning/residual-token-diagnostics.md), [local geometry](../docs/research/spline_learning/local-geometry-diagnostic.md), [unfreezing](../docs/research/spline_learning/controlled-unfreezing.md), [stability](../docs/research/spline_learning/local-geometry-stability.md) |
| Detached scratch through300 epochs | Matched graph-only versus detached local B-spline;1,952 validation patches | Final native edge F1 0.5185→0.5420, common Curve F1 0.2020→0.3426, Branch F1 0.0079→0.0118. Geometry improvement coexists with poor absolute branch connectivity | [Completed extension](../docs/research/spline_learning/scratch-extension-300.md), [core metrics](../docs/research/metrics/core-metrics-results.md) |
| Plant/leaf/mixed vascular transfer | Matched vascular-only, plants, leaves and mixtures; straight versus detached splines | Full-data matched SyntheticMRI table favors vascular-only. Historical authorized tests are preserved, not reused for tuning | [Full table](../docs/research/pretraining/combined-transfer-results/test-comparison.md) |
| Nonplanar/textured source and rendering ablations | Visual prototypes, source augmentation and controlled transfer | Distinguish prototype plausibility from training evidence. Augmented plants improve some source-control scores without beating vascular-only | [Prototypes](../docs/research/pretraining/nonplanar-source-prototype.md), [texturing](../docs/research/pretraining/textured-source-prototype.md), [trained comparison](../docs/research/pretraining/augmented-transfer-results/test-comparison.md) |
| IXI source strategy and geometry curriculum | IXI-only / plants+IXI / SyntheticMRI+IXI; straight, both-stage spline and delayed spline | Later target-only control is the appropriate strong reference; source/pretraining histories and geometry activation differ across arms | [Protocol/results](../docs/research/pretraining/ixi-transfer.md) |
| IXI four-patch overfit | Straight and spline learnability on four training patches | Memorization succeeds; held-out losses worsen. Trainability on a few images is not domain generalization | [Completed diagnostic](../docs/research/pretraining/ixi-overfit.md) |
| IXI1: supervision audit | GT guides versus image/mask support and preparation | Establishes reference uncertainty and physical conventions before attributing errors to the model | [Report1](../docs/research/metrics/1-ixi-supervision-audit.md) |
| IXI2: feature/node diagnostics | Early versus coarse CNN foreground readouts; node errors | Early features are useful; spatial sampling/resolution confounds a claim that coarse features contain no vessel information | [Report2](../docs/research/metrics/2-ixi-feature-node-diagnostics.md) |
| IXI3: local node refinement | Matched continuation versus local CNN node correction |128-patch pilot: native node F1 0.4857→0.5196, Curve F1 0.1171→0.1676; conditional relation recall barely changes | [Report3](../docs/research/metrics/3-ixi-node-refinement-pilot.md) |
| IXI4: relation isolation | Original/refit token scorer, geometry-only and chord image/geometry probes | Conditional validation AP original0.4702; refit0.3774; geometry0.4043; image+geometry0.4261. Stronger training fit did not establish better validation ranking | [Report4](../docs/research/metrics/4-ixi-relation-isolation.md) |
| IXI5: threshold calibration | Training-selected relation operating points | Recall/Curve gains can produce false connections/cycles; calibration and ranking are separate | [Report5](../docs/research/metrics/5-ixi-threshold-calibration.md) |
| IXI6: all-query attribution | Parent/continuation/refinement; raw256 queries, duplicates, deletion and NMS controls | Normal recall56.27%; GT-assisted raw recovery83.62%;91.1% of false edges touch a spatially unmatched node,35.7% a duplicate candidate. Naive NMS degrades that pilot | [Report6](../docs/research/metrics/6-ixi-query-attribution.md) |
| IXI7: selection/refinement/loss mechanisms | Raw-query oracle, confidence probes, coordinate/classification ablation; focal/CE/mining continuations | Raw selection raises edge F1 0.0737→0.0999 but lowers Curve F1; coordinate residual explains most refinement benefit. Original relation validation AP remains best among tested continuations | [Report7](../docs/research/metrics/7-ixi-mechanism-batch.md) |
| IXI8: Hungarian/generalization/fair calibration | Actual training matcher, spatial populations, same-pair AP and train-selected thresholds | Conditional AP normal0.4702 vs Hungarian0.4658; no substantial matched-population ranking rescue. Original threshold tuning improves recall/F1 while sharply increasing cycles | [Report8](../docs/research/metrics/8-ixi-relation-matching-calibration.md) |
| IXI9: larger image-aware relation residuals |2,048 train/512 validation; token control, endpoints, chord and wider corridor | Conditional validation AP original0.4723; token0.3058; endpoints0.3113; chord0.3216; corridor0.3211. Image evidence helps some residual-control measures but none beats original ranking | [Report9](../docs/research/metrics/9-ixi-image-aware-relation-learning.md) |
| IXI10: controlled node weights/focal | Same edge CE+ratio0.15; node weighted CE, equal CE, equal focal | Validation recall0.5522/0.4225/0.3071 and edge F1 0.1189/0.1165/0.0767. Precision/fragmentation/cycle trade-offs matter | [Report10](../docs/research/metrics/10-ixi-node-focal-ablation.md) |
| IXI11: binary vessel input | Common pre-IXI initializer; paired raw-MRA versus supplied masks generating graph GT | Validation edge F1 0.0415→0.0904, Curve F1 0.0931→0.1586. Mask-to-graph diagnostic, not independently labelled or raw-image deployment performance | [Report11](../docs/research/metrics/11-ixi-segmentation-input-oracle.md) |
| Limited-data botanical transfer |200 SyntheticMRI patches/62 volumes; same target subset with/without25,000 leaf source patches; single seed | Held-out native edge F1 0.2588→0.2979 and Curve F1 0.1009→0.1206; components/duplicates worsen. Equal updates are not equal vascular exposure | [Complete low-label report](../docs/research/pretraining/limited-data-botanical-transfer.md) |
| BRAVA preparation and recovery | Manual SWC variants, image alignment, skull stripping, source graphs,64³ patches and spline caches |56 paired subjects;39/8/9 splits. Explicit scaling/reflection fixes and coincident-SWC-link reporting repair preserved source annotations; brain masks are not vessels | [Preparation](../docs/research/BRAVA_preparation.md), [Stage1/recovery](../docs/research/BRAVA_experiment_plan.md) |
| BRAVA Stage1 models and GT representations | Compatible SyntheticMRI/IXI zero-shot;23,500-update matched scratch/refinement; same-graph chord/spline; full-subject representations | IXI zero-shot is strongest measured starting point; scratch nearly misses all connections. Full-subject GT junction K8 Curve F1~0.8413 vs chords0.1866, with topology fixed. This is distinct from patch/model results | [Stage1 results](../docs/research/evidence/brava_stage1_20261021/results.md) |
| Full K8 capacity audit | All10,000/2,649 IXI and8,768/1,796 BRAVA development patches |23,213 patches,349,357 branch occurrences. Long BRAVA branches are capacity-limited; ordinary IXI targets generally are not | [Capacity report](../docs/research/spline_learning/ixi-brava-control-capacity.md) |
| Fixed/adaptive/piecewise GT study | All BRAVA development; focused IXI/SyntheticMRI; K4/8/12/16/24, oracle K, compression, internal versus graph splitting | Adaptive geometry offers substantial BRAVA GT accuracy/complexity benefits; explicit node splitting has additional graph cost with identical fitted geometry | [Completed comparison](../docs/research/continuous_representation_comparison.md) |
| First learned BRAVA geometry comparison | Common frozen IXI detector;2,048/512 patches; K8/K16/K24;8,000 updates; compression/selectors | Small learned gains; weak endpoints dominate. Earlier selector is learned compression, not the new projection bank | [Results/interpretation](../docs/research/evidence/continuous_representation/results-summary.md) |

## Key controlled result tables

### Full-data SyntheticMRI transfer — historical authorized test

These rows share their original matched protocol and validation-selected checkpoints. Edge F1 is the native detection measure; Branch/Curve are common graph metrics. Do not mix these values with newer physical-edge or different-cohort comparisons.

| Source and geometry | Native edge F1 | Branch F1 | Curve F1 |
|---|---:|---:|---:|
| Vascular-only, straight |0.7219|0.0979|0.4729|
| Vascular-only, spline |0.7258|0.1020|0.5097|
| Plants, spline |0.6879|0.0702|0.4581|
| Leaves, spline |0.6905|0.0664|0.4607|
| Combined sources, spline |0.6809|0.0633|0.4484|
| Augmented plants, spline |0.6935|0.0749|0.4733|

Sources: [matched transfer](../docs/research/pretraining/combined-transfer-results/test-comparison.md), [augmented variants](../docs/research/pretraining/augmented-transfer-results/test-comparison.md). Botanical low-label gains above answer a different question and are retained alongside this result.

### BRAVA validation GT fit — original dense paths

| Representation | Mean symmetric error mm | Within0.5-mm maximum-error bound | Mean controls/branch |
|---|---:|---:|---:|
| Fixed K8 |0.0898|77.19%|8|
| Fixed K16 |0.0266|94.02%|16|
| Fixed K24 |0.0131|98.34%|24|
| Oracle adaptive K |0.0690|98.35%|7.79|
| Internal piecewise K8 |0.0460|100%|10.61|

All1,796 BRAVA validation patches /8 subjects /19,446 overlapping branch occurrences. At0.5 mm, explicit splitting uses the same curves but increases mean graph nodes17.14→20.67 and candidate pairs206→306; internal geometry leaves the graph unchanged. Earlier piecewise fits enforce C0, not C1. Newly implemented C1 heads belong to the current batch and must not inherit these oracle results as learned outcomes.

### Earlier frozen BRAVA geometry —512 validation patches

| Geometry | Curve F1 | Matched-pair mean error mm |
|---|---:|---:|
| Original IXI K8 |0.05782|not cached|
| Trained K8 |0.06050|3.1009|
| Trained K16 |0.06064|3.0998|
| Trained K24 |0.06087|3.1001|

K24−K8 Curve F1 is+0.000363, subject-bootstrap95% interval[0.000145,0.000611]; endpoint-neutral mean-error interval includes zero. Only81/5,445 Hungarian-positive pairs have both endpoints within1 mm. All graph predictions are frozen: node recall2 mm23.47%, positional edge F10.02696, Branch F10.00250. The8,000-update cap yielded~18-minute head training, not a one-hour convergence study.

## Reproduction, evidence and retention

- Exact cohorts, configs, checkpoint/source hashes, job dependencies and recovery reasons live under the linked `experiments/` plans and detailed reports. Use study-specific commands rather than run a generic training entry point with approximate settings.
- Main artifact roots are on `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/`; current multi-dataset root is given above. [Retention register](../docs/research/storage/retained-experiments.md) and [archive/restore record](../docs/research/storage/cleanup-20261016.md) explain historical storage decisions.
- No artifacts or scientific reports were deleted by this consolidation. Old proposals, failures and early stopping/pending statements retain their dates. New aggregate results take priority for current interpretation without erasing earlier evidence.
- The [verbatim compilation](../docs/research/ALL_EXPERIMENTS_AND_RESULTS.md) remains available for completeness; it deliberately contains duplication and historical status statements. This register provides the concise reading order.

## Complete source catalog

The expandable index below preserves access to the detailed histories and supporting evidence beyond the selected findings above. It includes documentation outside `docs/research/`, early loss matrices and evidence reports; numerical data, HTML viewers and checkpoints remain at paths linked from those reports.

<details>
<summary>Research protocols, reviews and execution history (16 documents)</summary>

- [Consolidated experiments and results](../docs/research/ALL_EXPERIMENTS_AND_RESULTS.md) — `docs/research/ALL_EXPERIMENTS_AND_RESULTS.md`
- [BRAVA continuous vascular graph reconstruction — staged experiments](../docs/research/BRAVA_experiment_plan.md) — `docs/research/BRAVA_experiment_plan.md`
- [Brava SWC graphs, skull-stripped brains and IXI-compatible patches](../docs/research/BRAVA_preparation.md) — `docs/research/BRAVA_preparation.md`
- [Experiment review — jobs run on 20 October 2026](../docs/research/EXPERIMENT_REVIEW_20261021.md) — `docs/research/EXPERIMENT_REVIEW_20261021.md`
- [Completed experiment review — 22 October 2026](../docs/research/EXPERIMENT_REVIEW_20261022.md) — `docs/research/EXPERIMENT_REVIEW_20261022.md`
- [Completed experiment review —23 October2026](../docs/research/EXPERIMENT_REVIEW_20261023.md) — `docs/research/EXPERIMENT_REVIEW_20261023.md`
- [Vascular graph roadmap implementation tracker](../docs/research/IMPLEMENTATION_TRACKER.md) — `docs/research/IMPLEMENTATION_TRACKER.md`
- [IXI vascular graph extraction — diagnosis and experimental roadmap](../docs/research/IXI_graph_extraction_diagnostic_roadmap.md) — `docs/research/IXI_graph_extraction_diagnostic_roadmap.md`
- [Project review and IXI diagnostic starting point — 19 October 2026](../docs/research/PROJECT_REVIEW_20261019.md) — `docs/research/PROJECT_REVIEW_20261019.md`
- [Vedo production extraction run](../docs/research/VEDO_PRODUCTION_RUN.md) — `docs/research/VEDO_PRODUCTION_RUN.md`
- [Continuous vascular representations: fixed, adaptive and piecewise splines](../docs/research/continuous_representation_comparison.md) — `docs/research/continuous_representation_comparison.md`
- [Mitigating false loops and off-vessel graph edges](../docs/research/loop_failure_mitigation.md) — `docs/research/loop_failure_mitigation.md`
- [Multi-dataset continuous vascular graph experiments](../docs/research/multidataset_continuous_graph_experiments.md) — `docs/research/multidataset_continuous_graph_experiments.md`
- [Evaluation and Metrics Plan for Adaptive Vascular Graph Extraction](../docs/research/vascular_graph_metrics.md) — `docs/research/vascular_graph_metrics.md`
- [Vascular Graph Representation: Current and Future Directions](../docs/research/vascular_graph_representation.md) — `docs/research/vascular_graph_representation.md`
- [Ordered Research and Implementation Roadmap](../docs/research/vascular_graph_roadmap.md) — `docs/research/vascular_graph_roadmap.md`

</details>

<details>
<summary>Metrics and IXI diagnostic studies (19 documents)</summary>

- [1. IXI supervision and prediction audit — 2026-10-18](../docs/research/metrics/1-ixi-supervision-audit.md) — `docs/research/metrics/1-ixi-supervision-audit.md`
- [Controlled IXI node class-weight and focal-loss continuation](../docs/research/metrics/10-ixi-node-focal-ablation.md) — `docs/research/metrics/10-ixi-node-focal-ablation.md`
- [11 — IXI segmentation-input oracle](../docs/research/metrics/11-ixi-segmentation-input-oracle.md) — `docs/research/metrics/11-ixi-segmentation-input-oracle.md`
- [2. IXI feature and node diagnostics — 2026-10-18](../docs/research/metrics/2-ixi-feature-node-diagnostics.md) — `docs/research/metrics/2-ixi-feature-node-diagnostics.md`
- [3. IXI higher-resolution node refinement pilot — 2026-10-18](../docs/research/metrics/3-ixi-node-refinement-pilot.md) — `docs/research/metrics/3-ixi-node-refinement-pilot.md`
- [4. IXI relation isolation — 2026-10-19](../docs/research/metrics/4-ixi-relation-isolation.md) — `docs/research/metrics/4-ixi-relation-isolation.md`
- [5. IXI training-calibrated relation thresholds](../docs/research/metrics/5-ixi-threshold-calibration.md) — `docs/research/metrics/5-ixi-threshold-calibration.md`
- [6. IXI full-query, duplicate and edge attribution — 2026-10-19](../docs/research/metrics/6-ixi-query-attribution.md) — `docs/research/metrics/6-ixi-query-attribution.md`
- [7. IXI bounded mechanism batch — 2026-10-19](../docs/research/metrics/7-ixi-mechanism-batch.md) — `docs/research/metrics/7-ixi-mechanism-batch.md`
- [8. IXI relation matching, generalization and calibration](../docs/research/metrics/8-ixi-relation-matching-calibration.md) — `docs/research/metrics/8-ixi-relation-matching-calibration.md`
- [IXI image-aware relation residuals: larger controlled experiment](../docs/research/metrics/9-ixi-image-aware-relation-learning.md) — `docs/research/metrics/9-ixi-image-aware-relation-learning.md`
- [Vessel evaluation metrics](../docs/research/metrics/README.md) — `docs/research/metrics/README.md`
- [Branch fragmentation and relation distance diagnostic](../docs/research/metrics/branch-connectivity-diagnostic.md) — `docs/research/metrics/branch-connectivity-diagnostic.md`
- [Vessel core v1: completed validation comparison](../docs/research/metrics/core-metrics-results.md) — `docs/research/metrics/core-metrics-results.md`
- [Generalization diagnostics — 2026-10-14](../docs/research/metrics/generalization-audit.md) — `docs/research/metrics/generalization-audit.md`
- [Metric implementation and evaluation tracker](../docs/research/metrics/implementation-status.md) — `docs/research/metrics/implementation-status.md`
- [1. Geometric metrics](../docs/research/metrics/metrics_ideas.md) — `docs/research/metrics/metrics_ideas.md`
- [Controlled relation and node-head learning pilot](../docs/research/metrics/relation-distance-pilot.md) — `docs/research/metrics/relation-distance-pilot.md`
- [Vessel core v1: common physical evaluation](../docs/research/metrics/vessel-core-v1.md) — `docs/research/metrics/vessel-core-v1.md`

</details>

<details>
<summary>Pretraining, sources and label-budget studies (16 documents)</summary>

- [Augmented biological-source comparison — 2026-10-18](../docs/research/pretraining/augmented-source-comparison.md) — `docs/research/pretraining/augmented-source-comparison.md`
- [Source transfer: test comparison](../docs/research/pretraining/augmented-transfer-results/test-comparison.md) — `docs/research/pretraining/augmented-transfer-results/test-comparison.md`
- [Source transfer: validation comparison](../docs/research/pretraining/augmented-transfer-results/validation-comparison.md) — `docs/research/pretraining/augmented-transfer-results/validation-comparison.md`
- [Plants + leaves + SyntheticMRI: matched source mixture](../docs/research/pretraining/combined-sources.md) — `docs/research/pretraining/combined-sources.md`
- [Source transfer: test comparison](../docs/research/pretraining/combined-transfer-results/test-comparison.md) — `docs/research/pretraining/combined-transfer-results/test-comparison.md`
- [Controlled plant transfer and spline augmentation — 2026-10-14](../docs/research/pretraining/generalization-training.md) — `docs/research/pretraining/generalization-training.md`
- [IXI four-patch fitting diagnostic (2026-10-16)](../docs/research/pretraining/ixi-overfit.md) — `docs/research/pretraining/ixi-overfit.md`
- [IXI source-domain comparison, representation trained in both stages](../docs/research/pretraining/ixi-transfer.md) — `docs/research/pretraining/ixi-transfer.md`
- [HALVS replacement of the plant source domain — 2026-10-15](../docs/research/pretraining/leaves-replacement.md) — `docs/research/pretraining/leaves-replacement.md`
- [Limited-data SyntheticMRI: botanical versus vessel-only pretraining](../docs/research/pretraining/limited-data-botanical-transfer.md) — `docs/research/pretraining/limited-data-botanical-transfer.md`
- [Two-sample nonplanar plant prototype](../docs/research/pretraining/nonplanar-source-prototype.md) — `docs/research/pretraining/nonplanar-source-prototype.md`
- [Mixed plant pretraining recovery — 2026-10-14](../docs/research/pretraining/plants-recovery.md) — `docs/research/pretraining/plants-recovery.md`
- [Plant/leaf augmentation integration](../docs/research/pretraining/source-augmentation-training.md) — `docs/research/pretraining/source-augmentation-training.md`
- [Source transfer: test comparison](../docs/research/pretraining/source-transfer-results/test-comparison.md) — `docs/research/pretraining/source-transfer-results/test-comparison.md`
- [Source transfer: validation comparison](../docs/research/pretraining/source-transfer-results/validation-comparison.md) — `docs/research/pretraining/source-transfer-results/validation-comparison.md`
- [Simple textured-background plant prototype](../docs/research/pretraining/textured-source-prototype.md) — `docs/research/pretraining/textured-source-prototype.md`

</details>

<details>
<summary>Spline learning and representation histories (31 documents)</summary>

- [Continuous vessel graph experiments](../docs/research/spline_learning/README.md) — `docs/research/spline_learning/README.md`
- [Compact adaptive degree-two scratch baseline](../docs/research/spline_learning/adaptive-degree2-scratch-comparison.md) — `docs/research/spline_learning/adaptive-degree2-scratch-comparison.md`
- [Eight-point Catmull–Rom training — 2026-10-09](../docs/research/spline_learning/catmull-rom-learning.md) — `docs/research/spline_learning/catmull-rom-learning.md`
- [Catmull–Rom versus B-spline: existing single cases](../docs/research/spline_learning/catmull-rom-single-cases.md) — `docs/research/spline_learning/catmull-rom-single-cases.md`
- [Controlled unfreezing — 2026-10-09](../docs/research/spline_learning/controlled-unfreezing.md) — `docs/research/spline_learning/controlled-unfreezing.md`
- [Connectivity informed by predicted vessel curves — 2026-10-08](../docs/research/spline_learning/curve-informed-connectivity.md) — `docs/research/spline_learning/curve-informed-connectivity.md`
- [B-spline failure: evidence review and frozen-detector diagnostic](../docs/research/spline_learning/frozen-spline-investigation.md) — `docs/research/spline_learning/frozen-spline-investigation.md`
- [Full validation comparison — 2026-10-09](../docs/research/spline_learning/full-validation-comparison.md) — `docs/research/spline_learning/full-validation-comparison.md`
- [IXI and BRAVA — adequacy of actual 8-control spline targets](../docs/research/spline_learning/ixi-brava-control-capacity.md) — `docs/research/spline_learning/ixi-brava-control-capacity.md`
- [IXI patch: curve-valued branches with uncertain centerlines](../docs/research/spline_learning/ixi_patch.md) — `docs/research/spline_learning/ixi_patch.md`
- [Local image information for geometry — 2026-10-10](../docs/research/spline_learning/local-geometry-diagnostic.md) — `docs/research/spline_learning/local-geometry-diagnostic.md`
- [Local geometry: full validation and controlled unfreezing](../docs/research/spline_learning/local-geometry-followup.md) — `docs/research/spline_learning/local-geometry-followup.md`
- [Longer stability continuation after local-geometry unfreezing](../docs/research/spline_learning/local-geometry-stability.md) — `docs/research/spline_learning/local-geometry-stability.md`
- [Projects cleanup — 2026-10-06](../docs/research/spline_learning/projects-cleanup-2026-10-06.md) — `docs/research/spline_learning/projects-cleanup-2026-10-06.md`
- [The approaches I would seriously consider](../docs/research/spline_learning/representation-implementation.md) — `docs/research/spline_learning/representation-implementation.md`
- [Frozen residual loss and token-input diagnostics — 2026-10-09](../docs/research/spline_learning/residual-token-diagnostics.md) — `docs/research/spline_learning/residual-token-diagnostics.md`
- [SyntheticMRI: paired scratch comparison, 300 epochs](../docs/research/spline_learning/scratch-comparison-300ep.md) — `docs/research/spline_learning/scratch-comparison-300ep.md`
- [Three scratch trainings: completed held-out comparison](../docs/research/spline_learning/scratch-comparison-results.md) — `docs/research/spline_learning/scratch-comparison-results.md`
- [Scratch geometry with detector gradients isolated](../docs/research/spline_learning/scratch-detached-geometry.md) — `docs/research/spline_learning/scratch-detached-geometry.md`
- [Scratch extension to 300 epochs — 2026-10-12](../docs/research/spline_learning/scratch-extension-300.md) — `docs/research/spline_learning/scratch-extension-300.md`
- [Scratch learning with residual local-image geometry](../docs/research/spline_learning/scratch-local-geometry-diagnostic.md) — `docs/research/spline_learning/scratch-local-geometry-diagnostic.md`
- [V0 SyntheticMRI training with up to 10,000 patches](../docs/research/spline_learning/spline-10k-training.md) — `docs/research/spline_learning/spline-10k-training.md`
- [Selection after the initial spline experiments](../docs/research/spline_learning/spline-approach-selection.md) — `docs/research/spline_learning/spline-approach-selection.md`
- [Scratch B-spline collapse diagnosis — 2026-10-08](../docs/research/spline_learning/spline-collapse-diagnosis.md) — `docs/research/spline_learning/spline-collapse-diagnosis.md`
- [Learning continuous vessel branches: implementation and experiment log](../docs/research/spline_learning/spline-learning-report.md) — `docs/research/spline_learning/spline-learning-report.md`
- [Spline learning: generated run results](../docs/research/spline_learning/spline-learning-results.md) — `docs/research/spline_learning/spline-learning-results.md`
- [Staged graph-to-spline learning — 2026-10-08](../docs/research/spline_learning/spline-staged-learning.md) — `docs/research/spline_learning/spline-staged-learning.md`
- [Paired spline-weight pilot — 2026-10-08](../docs/research/spline_learning/spline-weight-pilot.md) — `docs/research/spline_learning/spline-weight-pilot.md`
- [SyntheticMRI restoration — 2026-10-05](../docs/research/spline_learning/synthetic-mri-restoration.md) — `docs/research/spline_learning/synthetic-mri-restoration.md`
- [SyntheticMRI held-out prediction viewers](../docs/research/spline_learning/synthetic-mri-test-predictions.md) — `docs/research/spline_learning/synthetic-mri-test-predictions.md`
- [SyntheticMRI: the same continuous-branch viewer as IXI](../docs/research/spline_learning/synthetic_mri_patch.md) — `docs/research/spline_learning/synthetic_mri_patch.md`

</details>

<details>
<summary>Detailed evidence and historical result reports (38 documents)</summary>

- [Phase 2: representation-quality study (2026-09-21)](../docs/research/evidence/T05/README.md) — `docs/research/evidence/T05/README.md`
- [Early Phase 3 metric check](../docs/research/evidence/T07/README.md) — `docs/research/evidence/T07/README.md`
- [T08: controlled metric-failure checks](../docs/research/evidence/T08/README.md) — `docs/research/evidence/T08/README.md`
- [IXI 256-query integration smoke (2026-09-23)](../docs/research/evidence/T09/README.md) — `docs/research/evidence/T09/README.md`
- [IXI adaptive-GT 300-epoch run and held-out test](../docs/research/evidence/T09_long/README.md) — `docs/research/evidence/T09_long/README.md`
- [IXI held-out test: validation-selected epoch-270 model](../docs/research/evidence/T09_long/test_results.md) — `docs/research/evidence/T09_long/test_results.md`
- [IXI node-type / unmatched-negative pilot (2026-09-24)](../docs/research/evidence/T10_node_type_pilot.md) — `docs/research/evidence/T10_node_type_pilot.md`
- [Adaptive graph tuning on representative real volumes](../docs/research/evidence/adaptive_tuning/README.md) — `docs/research/evidence/adaptive_tuning/README.md`
- [Selected global graph policy comparison](../docs/research/evidence/adaptive_tuning/selected_policy_comparison/README.md) — `docs/research/evidence/adaptive_tuning/selected_policy_comparison/README.md`
- [BRAVA Stage 1 recovery — 22 October 2026](../docs/research/evidence/brava_stage1_20261021/completion-review.md) — `docs/research/evidence/brava_stage1_20261021/completion-review.md`
- [Completed Stage 1 results](../docs/research/evidence/brava_stage1_20261021/results.md) — `docs/research/evidence/brava_stage1_20261021/results.md`
- [New GT results](../docs/research/evidence/continuous_representation/gt-report.md) — `docs/research/evidence/continuous_representation/gt-report.md`
- [New learned geometry results](../docs/research/evidence/continuous_representation/learned-report.md) — `docs/research/evidence/continuous_representation/learned-report.md`
- [Completed results and interpretation — 22 October 2026](../docs/research/evidence/continuous_representation/results-summary.md) — `docs/research/evidence/continuous_representation/results-summary.md`
- [IXI degree-consistency training experiment](../docs/research/evidence/degree_consistency/README.md) — `docs/research/evidence/degree_consistency/README.md`
- [Plants → IXI domain-adaptation run, 2026-09-24](../docs/research/evidence/domain_ixi_pretraining/README.md) — `docs/research/evidence/domain_ixi_pretraining/README.md`
- [IXI test: plain decoder versus soft degree penalty 0.5](../docs/research/evidence/domain_ixi_pretraining/test_decoder_comparison.md) — `docs/research/evidence/domain_ixi_pretraining/test_decoder_comparison.md`
- [IXI node-type model: held-out test evaluation](../docs/research/evidence/domain_ixi_pretraining/test_types300.md) — `docs/research/evidence/domain_ixi_pretraining/test_types300.md`
- [IXI focal-loss ablation: held-out test results](../docs/research/evidence/focal_ixi_300ep/README.md) — `docs/research/evidence/focal_ixi_300ep/README.md`
- [IXI capacity and training-crop audit (2026-09-21)](../docs/research/evidence/ixi_capacity_192_256/README.md) — `docs/research/evidence/ixi_capacity_192_256/README.md`
- [IXI frozen-feature and node diagnostics](../docs/research/evidence/ixi_features_nodes_20261018/results.md) — `docs/research/evidence/ixi_features_nodes_20261018/results.md`
- [Completed results](../docs/research/evidence/ixi_image_relation_learning_20261020/results.md) — `docs/research/evidence/ixi_image_relation_learning_20261020/results.md`
- [Completed IXI mechanism batch evidence](../docs/research/evidence/ixi_mechanisms_20261019/README.md) — `docs/research/evidence/ixi_mechanisms_20261019/README.md`
- [IXI bounded mechanism diagnostics](../docs/research/evidence/ixi_mechanisms_20261019/results.md) — `docs/research/evidence/ixi_mechanisms_20261019/results.md`
- [Completed results](../docs/research/evidence/ixi_node_focal_20261020/results.md) — `docs/research/evidence/ixi_node_focal_20261020/results.md`
- [IXI node refinement: fixed 1000-update comparison](../docs/research/evidence/ixi_node_refinement_20261018/comparison.md) — `docs/research/evidence/ixi_node_refinement_20261018/comparison.md`
- [IXI query-attribution pilot evidence — 2026-10-19](../docs/research/evidence/ixi_query_attribution_20261019/README.md) — `docs/research/evidence/ixi_query_attribution_20261019/README.md`
- [IXI-only 120-versus-192 object-query benchmark](../docs/research/evidence/ixi_query_benchmark/README.md) — `docs/research/evidence/ixi_query_benchmark/README.md`
- [Result: cost of 192 object queries on IXI](../docs/research/evidence/ixi_query_benchmark/RESULTS.md) — `docs/research/evidence/ixi_query_benchmark/RESULTS.md`
- [IXI relation isolation](../docs/research/evidence/ixi_relation_isolation_20261019/results.md) — `docs/research/evidence/ixi_relation_isolation_20261019/results.md`
- [Computed results](../docs/research/evidence/ixi_relation_matching_calibration_20261019/results.md) — `docs/research/evidence/ixi_relation_matching_calibration_20261019/results.md`
- [Completed results](../docs/research/evidence/ixi_segmentation_input_20261020/results.md) — `docs/research/evidence/ixi_segmentation_input_20261020/results.md`
- [IXI train-calibrated relation thresholds](../docs/research/evidence/ixi_threshold_calibration_20261019/results.md) — `docs/research/evidence/ixi_threshold_calibration_20261019/results.md`
- [Job review — 20 October 2026](../docs/research/evidence/job_review_20261020/review.md) — `docs/research/evidence/job_review_20261020/review.md`
- [Completed initial batch](../docs/research/evidence/multidataset_continuous/report.md) — `docs/research/evidence/multidataset_continuous/report.md`
- [IXI and BRAVA — adequacy of actual 8-control spline targets](../docs/research/evidence/spline_target_capacity_20261022/report.md) — `docs/research/evidence/spline_target_capacity_20261022/report.md`
- [TopBrain mixed-pretraining plus vessel fine-tuning time estimate](../docs/research/evidence/topbrain_training_time_estimate/README.md) — `docs/research/evidence/topbrain_training_time_estimate/README.md`
- [Full-volume CT-001: current versus original Vedo](../docs/research/evidence/vedo_original_ct001/full_volume/README.md) — `docs/research/evidence/vedo_original_ct001/full_volume/README.md`

</details>

<details>
<summary>Storage, retention and restoration (2 documents)</summary>

- [Research storage cleanup — 2026-10-16](../docs/research/storage/cleanup-20261016.md) — `docs/research/storage/cleanup-20261016.md`
- [Retained experiment register — 2026-10-16](../docs/research/storage/retained-experiments.md) — `docs/research/storage/retained-experiments.md`

</details>

<details>
<summary>Repository implementation and execution references (23 documents)</summary>

- [Data augmentation contract](../docs/AUGMENTATION.md) — `docs/AUGMENTATION.md`
- [SyntheticMRI candidate-grid audit](../docs/DATA_AUDIT.md) — `docs/DATA_AUDIT.md`
- [SyntheticMRI v2 dataset generation](../docs/DATA_GENERATION.md) — `docs/DATA_GENERATION.md`
- [Data loading contract](../docs/DATA_LOADING.md) — `docs/DATA_LOADING.md`
- [Inference and evaluation](../docs/EVALUATION.md) — `docs/EVALUATION.md`
- [Controlled MRI finetuning experiments](../docs/EXPERIMENTS.md) — `docs/EXPERIMENTS.md`
- [Interactive SyntheticMRI graph visualization](../docs/GRAPH_PREDICTION_3D.md) — `docs/GRAPH_PREDICTION_3D.md`
- [IXI vessel-graph dataset](../docs/IXI_DATASET.md) — `docs/IXI_DATASET.md`
- [Graph loss stack](../docs/LOSSES.md) — `docs/LOSSES.md`
- [3D RelationFormer model contract](../docs/MODEL.md) — `docs/MODEL.md`
- [Running the 3D RelationFormer model](../docs/RUNNING_THE_MODEL.md) — `docs/RUNNING_THE_MODEL.md`
- [Spline representation experiment](../docs/SPLINE_REPRESENTATION.md) — `docs/SPLINE_REPRESENTATION.md`
- [SyntheticMRI dataset split and patch-selection report](../docs/SYNTHETIC_MRI_DATASET_REPORT.md) — `docs/SYNTHETIC_MRI_DATASET_REPORT.md`
- [Training](../docs/TRAINING.md) — `docs/TRAINING.md`
- [repo-graph.md — Graph-Native-Betti-Matching (fork)](../docs/repo-graph.md) — `docs/repo-graph.md`
- [Configuration contract](../configs/README.md) — `configs/README.md`
- [Experiment navigation](../experiments/README.md) — `experiments/README.md`
- [Jean Zay setup and execution](../cluster/jean_zay/README.md) — `cluster/jean_zay/README.md`
- [Boundary-data edge-focal gamma sweep](../configs/experiments/boundary_gamma_sweep_500/README.md) — `configs/experiments/boundary_gamma_sweep_500/README.md`
- [Edge candidate/loss ablation](../configs/experiments/edge_candidate_ablation_600/README.md) — `configs/experiments/edge_candidate_ablation_600/README.md`
- [Seven-recipe focal matrix](../configs/experiments/focal_matrix_600/README.md) — `configs/experiments/focal_matrix_600/README.md`
- [Full-data comparison pipelines](../configs/experiments/full_dataset_comparison/README.md) — `configs/experiments/full_dataset_comparison/README.md`
- [Full-data node-focal pipeline](../configs/experiments/full_dataset_node_focal/README.md) — `configs/experiments/full_dataset_node_focal/README.md`

</details>

Catalog snapshot: **145 source documents**; detailed sources retained at their original paths.

## Image-to-Graph-Vessel historical experiments

Added the [companion-project inventory](image-to-graph-vessel.md), covering source-only plants/HALVS/roads, source mixtures, target fine-tuning, contrastive and BN-calibration variants. [Recovered historical evaluation metrics](evidence/image-to-graph-vessel/historical-evaluation-metrics.csv) retain every extracted metric block and source-log SHA256; the [archive index](evidence/image-to-graph-vessel/archive-index.csv) preserves the full file inventory. These are legacy native metrics with unresolved per-run comparability, not common-v2 results. Config presence and archived log presence are distinguished from verified successful results. No experiments were rerun.

<!-- BEGIN N08 REGISTRY 20261026 -->
## N08 — GT-free representative selection — completed 26 October 2026

**Negative controlled diagnostic; stop further heuristic search.** Follows N06’s oracle-representative analysis. Reused two A-xyz step-12,000 SyntheticMRI caches (128 training/480 validation patches per seed; 32/13 disjoint subjects), no GPU or training. Training-only choices compared uncapped support, guarded reference-rank support, local profile representatives and protected NMS against unchanged inference, confidence NMS and exact per-patch node-count controls.

Primary local-profile Edge F1: **0.637638/0.637196**, versus confidence NMS **0.637675/0.637184**. Differences are inconsistent and both subject-bootstrap intervals include zero. Guarded support adds 106/104 TP but 283/285 FP versus NMS and increases cycles. Confidence NMS loses short branches/nearby junctions; protected NMS largely restores normal false edges. No deployable selection change, training or architecture is recommended. Remaining interpretation: spatial identity/assignment/connectivity coupling, not proven absence of relation-token information. See [N08 full protocol/results](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026). Numeric evidence is in sibling `gnbm-query-selection-20261026/experiments/gt_free_query_selection_20261026`; print with `python3 gnbm-query-selection-20261026/scripts/show_query_selection.py` from `projects`.

<!-- END N08 REGISTRY 20261026 -->


<!-- NATIVE2D IMPLEMENTATION 20261027 -->
## Native2D-01 — implementation and bounded smoke, 27 October 2026

Original2D GrEVe ResNet/deformable transformer port with modern focal gamma2/equal node weights, matched-edge CE ratio0.15, existing Hungarian matching, optional coordinate-only9-sample local refinement and detached endpoint-constrained K8 geometry. Eight synthetic/metric regressions plus real CPU forward/backward passed. Development preparation:2train/2validation64² AV-WIDE crops from different source images, one128² REAVER alignment crop; final biological subject/split provenance unresolved. GPU verification and1000-update two-crop memorization are pending; no complete-model benchmark/full training submitted. Detailed report and limitations: [companion project](image-to-graph-vessel.md#native-2d-journal-extension--27-october-2026). Evidence: sibling `gnbm-native-2d-20261027/experiments/native2d_20261027`; retrieval `python3 scripts/show_native2d.py`.


<!-- NATIVE2D ACCEPTED JOBS 20261027 -->
### Native 2D accepted bounded jobs — 27 October 2026

**SUBMITTED — verify:** GPU smoke **1637307**, one H100,20-minute allocation; original-size architecture backward plus1000-update memorization on two training crops. CPU collector **1637308**, `afterany:1637307`, appends the outcome automatically even if smoke fails. GPU dependency: `afterany:1637025:1637027:1637032:1637034:1637036:1637037:1637039:1637040:1637042:1637044:1637045:1637046`. These are existing BRAVA GPU chains, ending at1637046; the dependency preserves the global two-H100 limit and may delay the small2D check until that longer batch ends. No existing job was cancelled, no full2D training submitted.

Completed before submission: eight unit regressions,13 real-image CPU checks, two training/two validation development crops plus one REAVER alignment overlay. Final main-repository replay passed; existing3D implementation hashes remain unchanged. CUDA and tiny memorization results remain pending, not claimed passed.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-native-2d-20261027/experiments/native2d_20261027`; large runtime/checkpoint output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/native2d_20261027`. Source/config hashes and archive are frozen in `code-identities.json` / `source.zip`. Read saved results with:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation/scripts/show_native2d.py
```



<!-- COMPLETED RESULTS REVIEW 20261028 -->
## D08 and Native2D-01 final reviewed results — 28 October 2026

Both are **DONE — measured**, with different scopes. Earlier submission/pending text remains historical.

| Study | Population and budget | Completed result | Interpretation |
|---|---|---|---|
| D08 scratch versus SyntheticMRI initialization |8,768 train /1,796 val patches;39/8 subjects;300 epochs,82,200 updates/arm, one seed | Final2 mm validation node F1 0.02544→0.35572; edge F1 0.00110→0.06794; Branch F1 0→0.00647; Curve F1 0.00302→0.12174 | Initialization helps under this recipe; scratch remains poor on representative training data and transferred connectivity remains weak. No representation winner selected. |
| Native2D-01 correctness and tiny learning | Two 64² training crops from one AV-WIDE image;1,000 updates with ResNet18/64D; original-width ResNet101 separately tested with zero updates | Node/edge/Branch F1 1.0 on both training crops; Curve F1 0.9789/1.0; CPU/CUDA, spline gradient and state checks passed | Implementation learnability only. No validation/test/generalization or B0/A-xyz/geometry-benefit comparison. |

D08 SyntheticMRI-minus-scratch paired subject95% intervals: node F1 **+0.33027 [0.32262,0.33879]**, edge F1 **+0.06684 [0.06245,0.07032]**. Eight reused validation subjects; no training-seed uncertainty. Default thresholds remain0.5. Training-calibrated thresholds were not evaluated in D08.

Reports: [BRAVA](../docs/research/BRAVA_experiment_plan.md#reviewed-final-d08-results--28-october-2026), [native2D](image-to-graph-vessel.md#native-2d-completion-review--28-october-2026). Compact evidence: `experiments/completed_result_review_20261028/review.json`, milestone CSV, final per-subject CSV and input hashes. Original checkpoints/caches/results are preserved. All production and reporting jobs succeeded; no new jobs were submitted.


<!-- D08 CALIBRATION REGISTER 20261028 -->
## D08 cached calibration — 28 October 2026

D08 CPU calibration completed: trained node cut0.45 gives validation node F1 0.36761; edge-only train-F1 cut0.425374478 gives edge F1 0.06892 on unchanged>0.5-node candidates. Ranking/AP is unchanged. Joint lower-node Edge/Curve F1 remains unavailable because rejected-query relations/new-edge controls were not cached. No GPU, inference or training was run. See the [existing BRAVA report](../docs/research/BRAVA_experiment_plan.md#d08-training-only-operating-point-audit--28-october-2026) and `experiments/d08_operating_points_20261028/`.

<!-- N07 STATUS AUDIT 20261028 -->
## N07/D04 and J00/J02 — status/capacity audit, 28 October 2026

Modern IXI B0/A-xyz/A-cls scientific comparison **NOT RUN**: smoke1569366 FAILED; replacement1583591 TIMEOUT (AF_UNIX path too long), dependents1583592/1583593/1583594 CANCELLED before execution. No scientific checkpoint/results and no fixed final-update budget. No GPU work launched by this audit. L06 remains partial.

Actual contracted K8 train/val capacity: IXI10000/2649 patches, max153/141, five/five >120; N07 subset4096/512 includes one/one overflow. BRAVA8768/1796, max64/69, none >120. Zero degree2 object nodes; index/manifest/subject checks pass. Keep256 for unchanged IXI recovery. [Full existing report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026). Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-ixi-local-components-20261026/experiments/n07_status_audit_20261028`. No efficacy result should be inferred from the infrastructure failure.

<!-- NATIVE2D HELDOUT LIVE 20261028 -->
## Native2D-02: held-out original-width AV-WIDE study — 28 October 2026

Authorized bounded B0 versus A-xyz pilot, following Native2D-01. All measurements below refer to image-disjoint validation, never test data. Patient independence is unverified. Distances are pixels.

Cohort: 17 train / 6 validation images; six reserved test images have header-only inspection. Image 07 excluded for its diagonal self-loop. 1385 train / 506 validation 128² tiles; 96 object queries selected from contracted training GT. All spatial tiles retained, including empty graphs.
**Target-quality limitation:** the original irregular-sampling K8 least-squares fit produces pathological controls (one verified128² tile reaches 56,890 px), causing shared loss spikes. This pilot is preserved but does not support a reliable geometry conclusion. A separately recorded single corrected-target pair uses uniform arc-length samples of unchanged dense paths; no graph labels, image splits, architecture or loss policy change.
Full ResNet101/512D, 3 encoder + 3 decoder layers, 5 relation tokens: gates PASSED; 1200 optimizer updates on two 128² training tiles before the experiment. Shared initialization and graph-only gradient checks passed. Budget fixed to 6000 updates per arm, batch 4, no automatic extension.
Throughput (existing PyTorch deformable attention): B0 0.0575s/update, 69.52 images/s, Axyz 0.0578s/update, 69.25 images/s.
Slurm: gates `1653554`, Axyz `1653991`, collector `1653994`, aborted_B0 `1653987`, B0 `1654241`.
Scheduler outcome: A-xyz completed normally. B0 completed all 6,000 updates and all 506-crop final metrics in about 436s, then its redundant initialization-checkpoint curve evaluation hit the1h45 Slurm limit. Its job state is **TIMEOUT**, despite complete primary final-checkpoint evidence. The initialization artifact is preserved as `B0/excluded-initialization.pt`; the trained-checkpoint index was repaired from the genuine final checkpoint/metrics, without retraining or new inference. No budget extension or threshold tuning was performed.

Primary comparison: **final fixed-budget checkpoints**, all validation tiles. Branch/Curve columns use GT edge-positive crops; all-crop aggregates are also retained in JSON, where empty/empty graphs can receive perfect scores.

| Arm | State | Final update | Node F1 @2 px | Edge F1 @2 px | Branch F1 @1 px | K8 Curve F1 @1 px | Chord Curve F1 @1 px |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Axyz | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Arm | Node mAP | Node mAR | Edge mAP | Edge mAR | Raw node recall @2 px, micro | Filtered node recall @2 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0846 | 0.4316 | 0.0305 | 0.3001 | 0.2353 | 0.0000 |
| Axyz | 0.0833 | 0.4286 | 0.0289 | 0.3187 | 0.2240 | 0.0000 |

**Negative bounded pilot:** both final checkpoints retain zero nodes/edges at the predeclared0.5 confidence thresholds, missing 7,613 GT nodes and 6,308 GT edges. Conditional relation performance is undefined because no filtered endpoint pairs remain; zero duplicates are a consequence of empty predictions. There is no evidence of A-xyz benefit or spline superiority. The8 edge-empty target crops inflate all-crop Branch/Curve macro F1 to0.01581; on 498 edge-positive crops these metrics are exactly zero. Pixel-space raw coverage and unthresholded AP remain visible, and no validation-driven operating-point change is made. Stop at the fixed budget; no automatic extension.

Intermediate validation uses 24 fixed spatial tiles (4/image); final and selected trained checkpoints are evaluated on every validation tile. Curve points at final therefore use a larger population; do not interpret that discontinuity as a training change. Selection uses the same 24 tiles throughout, excludes initialization, and resolves trained ties toward later updates. Best and final outcomes are both saved.

Detection AP/AR uses unthresholded query scores, IoU .50:.05:.95, node limit equal to query capacity and edge top300; reported AP is crop macro. Physical node/edge F1 is micro over crops at 1/2/4 px, including crop-boundary anchors. Branch/Curve F1 is crop macro at 1 px, .25 px sampling. Spline/chord use identical inferred nodes/edges; topology cannot change. Raw/filtered recall, relation AP populations, duplicates and Betti errors are retained per crop. No subject-level uncertainty claim, full-image stitching, independent manual topology or pretraining benefit is established.

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-native-2d-20261027/experiments/native2d_20261028`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D HELDOUT LIVE 20261028 -->

<!-- NATIVE2D TARGETFIT LIVE 20261028 -->
## Native2D-03: corrected-target held-out AV-WIDE study — 28 October 2026

Authorized bounded B0 versus A-xyz pilot, following Native2D-01. All measurements below refer to image-disjoint validation, never test data. Patient independence is unverified. Distances are pixels.

Cohort: 17 train / 6 validation images; six reserved test images have header-only inspection. Image 07 excluded for its diagonal self-loop. 1385 train / 506 validation 128² tiles; 96 object queries selected from contracted training GT. All spatial tiles retained, including empty graphs.
Correction: uniform64 arc-length samples of the same polyline before K8 fitting. All graph/image/split/reference-path tensors are exactly unchanged. Refit 25263 branches; control max absolute coordinate 2007485.8→131.2 px, uniform design condition number4.55. One fresh fixed-budget pair only; original checkpoints are not resumed.
Full ResNet101/512D, 3 encoder + 3 decoder layers, 5 relation tokens: gates PASSED; 1200 optimizer updates on two 128² training tiles before the experiment. Shared initialization and graph-only gradient checks passed. Budget fixed to 6000 updates per arm, batch 4, no automatic extension.
Throughput (existing PyTorch deformable attention): B0 0.0558s/update, 71.72 images/s, Axyz 0.0559s/update, 71.59 images/s.
Slurm: gates `1659581`, B0 `1659729`, Axyz `1659730`, collector `1659731`.

Primary comparison: **final fixed-budget checkpoints**, all validation tiles. Branch/Curve columns use GT edge-positive crops; all-crop aggregates are also retained in JSON, where empty/empty graphs can receive perfect scores.

| Arm | State | Final update | Node F1 @2 px | Edge F1 @2 px | Branch F1 @1 px | K8 Curve F1 @1 px | Chord Curve F1 @1 px |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Axyz | complete | 6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

| Arm | Node mAP | Node mAR | Edge mAP | Edge mAR | Raw node recall @2 px, micro | Filtered node recall @2 px |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 0.0880 | 0.4497 | 0.0375 | 0.3138 | 0.2374 | 0.0000 |
| Axyz | 0.0920 | 0.4120 | 0.0267 | 0.2918 | 0.2142 | 0.0000 |

**Negative bounded pilot:** both final checkpoints retain zero nodes/edges at the predeclared0.5 confidence thresholds, missing 7,613 GT nodes and 6,308 GT edges. Conditional relation performance is undefined because no filtered endpoint pairs remain; zero duplicates are a consequence of empty predictions. There is no evidence of A-xyz benefit or spline superiority. The8 edge-empty target crops inflate all-crop Branch/Curve macro F1 to0.01581; on 498 edge-positive crops these metrics are exactly zero. Pixel-space raw coverage and unthresholded AP remain visible, and no validation-driven operating-point change is made. Stop at the fixed budget; no automatic extension.

Intermediate validation uses 24 fixed spatial tiles (4/image); final and selected trained checkpoints are evaluated on every validation tile. Curve points at final therefore use a larger population; do not interpret that discontinuity as a training change. Selection uses the same 24 tiles throughout, excludes initialization, and resolves trained ties toward later updates. Best and final outcomes are both saved.

Detection AP/AR uses unthresholded query scores, IoU .50:.05:.95, node limit equal to query capacity and edge top300; reported AP is crop macro. Physical node/edge F1 is micro over crops at 1/2/4 px, including crop-boundary anchors. Branch/Curve F1 is crop macro at 1 px, .25 px sampling. Spline/chord use identical inferred nodes/edges; topology cannot change. Raw/filtered recall, relation AP populations, duplicates and Betti errors are retained per crop. No subject-level uncertainty claim, full-image stitching, independent manual topology or pretraining benefit is established.

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-native-2d-20261027/experiments/native2d_20261028_targetfix`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D TARGETFIT LIVE 20261028 -->

<!-- NATIVE2D LONG LIVE 20261028 -->
## Native2D-04 — longer fixed-epoch training and pure plant transfer, 28 October 2026

User authorized longer runs with the original node/edge confidence thresholds0.5, plants50 epochs, then vessel-only finetuning. This supersedes the previous17.3-epoch pilot stopping budget; earlier experiments remain preserved. No threshold calibration or architecture/loss/matcher change.


| Run | Slurm | State | Epoch | Node mAP | Edge mAP | Node F1 @2 units | Edge F1 @2 units |
|---|---|---|---:|---:|---:|---:|---:|
| plants_50 | — | not submitted | — | — | — | — | — |
| avwide_scratch_B0 | — | not submitted | — | — | — | — | — |
| avwide_scratch_Axyz | — | not submitted | — | — | — | — | — |
| avwide_plants_B0 | — | not submitted | — | — | — | — | — |
| avwide_plants_Axyz | — | not submitted | — | — | — | — | — |
| reaver_scratch_B0 | — | not submitted | — | — | — | — | — |
| reaver_scratch_Axyz | — | not submitted | — | — | — | — | — |
| reaver_plants_B0 | — | not submitted | — | — | — | — | — |
| reaver_plants_Axyz | — | not submitted | — | — | — | — | — |
| octasynth_scratch_B0 | — | not submitted | — | — | — | — | — |
| octasynth_scratch_Axyz | — | not submitted | — | — | — | — | — |
| octasynth_plants_B0 | — | not submitted | — | — | — | — | — |
| octasynth_plants_Axyz | — | not submitted | — | — | — | — | — |

Primary comparisons use the final fixed-epoch checkpoints on complete validation populations; selected checkpoints are secondary. Selection uses fixed interim source-image groups, excludes initialization and resolves trained ties toward later epochs. Pixel and micron metrics are reported separately. Raw versus filtered recall, conditional relation populations/AP, duplicates, components/cycles and same-graph K8/chord geometry remain in per-run validation JSON. Plant pretraining reports detection/graph metrics on its sparse annotations; its geometry is not an independently traced dense reference.

No test arrays or predictions are used. AV-WIDE remains17 training /6 validation images with unknown patient independence; image07 excluded. REAVER is the separately verified diaphragm60x4/2-image cohort in microns. OCTA uses official source-image membership and regenerated128² crops with coordinate/crop-boundary QC. Plant parent IDs are disjoint across official train/val/test membership.

Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/GNBM/gnbm-native-2d-20261027/experiments/native2d_long_20261028`. Source/config hashes: `source.zip`, `code-identities.json`; dataset manifests/audits: `<dataset>/`; epoch losses and full state checkpoints: `runs/<name>/`; final metrics: `final-heldout.json`; plots: `curves-<dataset>.png`. Stop at50 plant /100 vessel epochs; exceptions stop jobs without automatic retries, new sweeps or budget extensions.
<!-- END NATIVE2D LONG LIVE 20261028 -->
