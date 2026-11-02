# Vascular Graph Extraction — Experiment Registry & Research Decision Log

<!-- NATIVE2D CURRENT POINTER 20261028 -->
> Latest native2D result: **Native2D-03 corrected-target pilot completed** (B0/A-xyz6,000 updates each; no filtered graph recovery at0.5). [Final comparison, dataset readiness and stopping decision](../docs/research/native2d_heldout_20261028.md#completed-corrected-target-comparison). Native2D-02 remains preserved with its unstable-target and post-training-timeout caveats.


Latest reviewed completions (28 October 2026): [D08 and Native2D-01](#reviewed-completion-and-priorities--28-october-2026). Proposed next steps: [eight-week journal closure plan](#10-proposed-order-of-work--eight-week-journal-closure-plan-28-october-2026). D08 is measured but graph recovery remains weak; native 2D has passed implementation/tiny-fit checks only.

**Snapshot:** 26 October 2026 — supplied evening N06/D06 registry, reconciled with completed N08  
**Project:** `Graph-Native-Betti-Matching-Representation` (and explicitly named isolated worktrees)  
**Purpose:** one compact, durable index of **what was tested, what the tests actually established, what was merely planned/submitted, and what should be tried next**. Read this *before* writing or launching a new Codex experiment prompt.  
**Scope:** RelationFormer graph extraction, node/query selection, relation learning, continuous-edge representations, cross-domain transfer, and evaluation. Historical experiments are indexed rather than exhaustively reproduced. Detailed reports remain authoritative.


**Repository integration:** [experiment history](experiments.md) · [research roadmap](roadmap.md) · [unchanged attachment](evidence/experiment-registry/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.received.md). Stable IDs in this registry index the detailed reports; this import does not authorize new experiments. N08 below is updated from collected repository evidence; unrelated status claims remain the supplied snapshot pending their own audit.

> **Status caveat.** This is a report-based snapshot, **not a live query of Jean-Zay or Codex sessions**. Some jobs have known Slurm outcomes; others are mentioned in earlier prompts but have **not been verified as submitted/completed**. Do not convert “prompt written” to “experiment running.” New runs should update this file with Slurm IDs, code/checkpoint identities and actual collected measurements.

---

## 0. How to maintain and use this file

### Status vocabulary (use exactly these meanings)

| Status | Meaning |
|---|---|
| **DONE — measured** | Finished and collected; numerical conclusions supported by a report or saved evidence. |
| **DONE — negative/mixed** | Finished, but no demonstrated net benefit; preserve failures and limitations. |
| **PARTIAL / RECOVERY** | Some stages completed, but failure/cancellation or missing collection prevents a complete result. |
| **SUBMITTED — verify** | A submission/job ID is recorded; scientific completion has not been verified here. |
| **STATUS UNKNOWN** | Mentioned as underway in earlier conversations; no reliable completion/submission record here. |
| **NOT RUN** | Hypothesis/prompt/proposed experiment, with no confirmed execution. |
| **DEFERRED** | Deliberately blocked by an unmet prerequisite or weak evidence. |

**Two separate tracking objects:** (a) a *Codex prompt* is a request to an agent; (b) a *Slurm experiment* is a submitted, executed and ideally collected scientific run. A prompt may be drafted but never sent; a Slurm job may complete without the scientific collector. Record both statuses separately.

**Update protocol for a new result:**
1. Find the existing experiment ID below; **do not invent a new experiment name for a near-duplicate**.
2. Append a dated outcome with dataset/split, exact checkpoint SHA, training recipe, what was frozen/trainable, seed(s), steps, thresholds, numerical results and Slurm/collector status.
3. Record controls and limitations, including whether the result is GT-assisted/oracle, annotation-relative, trained geometry or GT-only fitting.
4. Update **§3 matrix**, **§5 next actions**, and **§6 decision gates**. Mark a step *DONE* only after collection/verification; mark a failure honestly.
5. Prefer appending evidence to the **existing domain report**, using this file as the index. Do not scatter nearly identical summaries into many new research Markdown files.

### Short current answer

- **Modern IXI D04/N07 completed:** one-seed5,500-update six-arm study. A-cls raises physical2mm edge F1 **0.087926→0.101536** with a positive paired-subject interval and count-matched support; Curve improves, duplicates/cycle error worsen. Coordinate-only improves localization without a clear default-edge advantage; combined/head-transfer benefits are unproven. [Final IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#reviewed-final-modern-ixi-results--28-october-2026).
- **Best controlled modern SyntheticMRI direction:** local image-conditioned **coordinate-only** node refinement (`A-xyz`), with two-seed improvement over continued baseline B0; **not a demonstrated topology or query-selection solution**.
- **Largest measured SyntheticMRI opportunity:** recover/select the right node-query representatives. Modern A-xyz edge F1 is approximately **0.634** at normal inference, while GT-assisted confidence-preferred query selection yields **0.834–0.838**. **N06 is now DONE:** about one-third of recovered GT vertices switch representatives; confidence-preferred alternatives are slightly less localized but have better same-neighbor relation ranking, strongly associated with training-style assignment. The oracle is **not deployable**.
- **N08 GT-free selection: DONE — negative/mixed.** The training-selected local rule achieved Edge F1 **0.637638/0.637196**, versus identical-count confidence NMS **0.637675/0.637184**; both paired graph-F1 intervals include zero. Guarded support added **106/104 TP but 283/285 FP**, with more cycles. No tested rule consistently beat count-matched controls. Stop this heuristic search; the oracle remains nondeployable.
- **Relation-image features:** Older IXI edge-logit adapters remain R05 (early gains, late overfit). Modern **R06 is now DONE — negative/mixed**, testing relation-specific 552D CNN token residuals on frozen A-xyz in both seeds. The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched.
- **BRAVA:** D06 remains a completed negative BN diagnosis. **D07 is now DONE:** 4k off−on train node/edge F1 +0.2908/+0.1551; validation +0.0285/+0.0062. Large fit gain; validation edge CI includes zero, raw 2-mm recall falls and loss/cycles worsen. Memorization is plausible; no recovered detector or follow-up launched. See the existing BRAVA report.
- **Infrastructure and capacity:** recovered modern IXI D04/N07 training/collection is complete; the earlier Q0 no-result finding remains a historical snapshot. L06 is separately partial (SyntheticMRI BN half complete, IXI failed). Current contracted IXI targets reach153 anchors; retain256 and defer Q120. Prior failed jobs and their costs remain preserved.

---

## 1. Controlled reference configurations and non-negotiable distinctions

### S0 — Modern clean SyntheticMRI reference

- **Baseline:** RelationFormer K8, 120 object queries, source checkpoint at epoch 95 after 100-epoch scratch SyntheticMRI training. `experiments/fine_feature_architecture/runs/baseline/best.pt`; SHA256 `d4bc78ad3eb7c1c29c9be8ec3e2927e88f2f45e6f34a3cb8be9ade5fd9d2431c`.
- **Working graph loss:** node **unweighted softmax focal γ=2**, class weights `[1,1]`; matched-pair edge **cross-entropy**, weights `[1,1]`, ratio upsampling `0.15`. Do not silently call the historical edge-focal configuration this baseline.
- **B0:** same architecture continued, all graph/CNN/transformer layers trainable. **A:** early-CNN local feature refinement of coordinates and object logits. **A-xyz:** same local sampling, **coordinates only**, logits unchanged; geometry remains K8, `graph_only` spline loss route. **A-control:** image features zeroed.
- **Controlled long study:** 12,000 full-model updates for each arm, two fine-tuning seeds `364505`, `92741`, identical parent checkpoint. Val 480 patches / 13 subjects. Approximate mean 2-mm edge F1: **B0 0.595**, **A 0.637**, **A-xyz 0.634**; 1-mm node F1 improves from **0.489 to 0.570**. A-xyz chosen for simplicity, not because confidence-refinement ablation demonstrated a major benefit.
- **What remains weak:** cycle-count error ~1.5 per patch, some duplication, and a large raw-query-versus-filtered-recall gap. Local image-feature perturbation demonstrates reliance on the information *after training*; it is not itself the improvement-vs-B0 estimate.
- **Original relation head receives:** paired original decoder node embeddings (552D each) and two global relation-token embeddings (552D each). **A-xyz does not recompute relation scores using corrected positions or local sampled CNN features** within a forward pass.

### Evaluation safeguards

- Report **1 mm and 2 mm positional** node/edge accounting; 1:1 maximum-cardinality/minimum-distance matching. Branch/Curve F1 often use a separate **common-v2 1-mm anchor / 0.25-mm geometry** definition; never merge those protocols.
- **Conditional relation recall/AP** evaluates a selected population of pairs with recovered endpoints; it is **not overall edge recall or edge precision**. A ~95% conditional recall can coexist with ~0.63 edge F1.
- Separate **spatially unmatched at 2 mm** from **Hungarian-unassigned at training**; a Hungarian assignment does not guarantee a physically matched node. A node located on foreground is not necessarily a valid vertex: our unsplit target represents junctions, termini and patch-induced boundary vertices, **not arbitrary intermediate vessel locations**.
- An **oracle** uses GT for selection/conditioning; it is an *upper-bound diagnostic*, not an inference-time improvement. **SMD/Curve F1 can increase with overprediction.** Report signed node/edge counts, components β₀, cycles β₁, duplicates and exact edge TP/FP/FN together.
- **BRAVA lacks a vascular segmentation mask**: `seg/` contains a **brain mask for loader compatibility**, and SWC annotation coverage is incomplete. Its FP/FN and Betti discrepancies are **annotation-relative**, not categorical anatomical truth.

---

## 2. Measured experiment history — what *has* been tested

The IDs below are stable anchors for future prompts.

### NODE / QUERY

| ID | Experiment & protocol | Status | What we learned / what we did **not** learn | Main evidence |
|---|---|---|---|---|
| **N01** | Modern SyntheticMRI B0 vs A vs A-xyz vs A-control, two seeds, fully unfrozen 12k continuation | **DONE — measured** | Local fine CNN features improve node localization and edge F1; confidence-logit residual contributes little. The same relation probabilities may connect better-localized endpoints. No consistent topology fix. | Modern fine-feature architecture report; `6-ixi-query-attribution.md` Oct26 addendum |
| **N02** | Earlier IXI frozen detector + 1k node-head continuation ± local-image refiner | **DONE — measured** | Local image head improves retained 2-mm node recall **0.512→0.563** and positional edge F1 **0.058→0.074** (128 validation patches). Frozen continuation is not equivalent to modern full-model IXI training. | `6-ixi-query-attribution.md` |
| **N03** | KNN/contextual node verifier V0–V3 (confidence, NMS, token MLP, KNN neighborhood) | **DONE — negative/mixed** | V3 harmed node F1 on SyntheticMRI and IXI; a token-context verifier is **not** evidence against a yet-untested image-conditioned `A-cls`. | `node_verifier_feasibility.md` |
| **N04** | Hard spatial NMS 0.5/1 mm on legacy IXI predictions | **DONE — negative** | Removed real GT anchors and true edges; decreased edge F1 from **0.0737** to **0.0626/0.0508**. Not a general uniqueness solution. | `6-ixi-query-attribution.md` |
| **N05** | GT-assisted query deletion, spatial-nearest raw oracle, confidence-preferred raw oracle, modern two-seed SyntheticMRI | **DONE — measured (diagnostic)** | A-xyz normal **0.634**, delete-unmatched **0.750/0.753**, nearest oracle **0.720/0.729**, confidence oracle **0.834/0.838** edge F1. **All raw-query oracles are GT-assisted; no causal token-quality mechanism established.** | `6-ixi-query-attribution.md`, Oct26 |
| **N06** | Paired nearest-versus-confidence oracle, controlled query swaps, relation ranking, small exact-token intervention, two SyntheticMRI seeds | **DONE — measured diagnostic** | ~33% of recovered anchors select different representatives; alternatives are ~0.38 mm farther yet better supported on synthetic vessel masks. Fixed-neighbor conditional AP improves ~0.873→0.959, beyond scalar recalibration. Confidence choices agree more with Hungarian assignments, but causality from training history is **unproven**. A small off-manifold token intervention suggests relation-relevant embedding content beyond the classifier direction. | `6-ixi-query-attribution.md`, Oct26 oracle addendum |
| **N07** | Standalone image-conditioned A-cls on modern IXI, fully trainable six-arm control | **DONE — measured** |5,500 updates,256 queries,512 val/26 subjects,one target seed. A-cls physical2mm edge F1 **0.101536 vs B0 0.087926**, delta+0.013610 [0.008265,0.019401]; count-matched delta+0.010862 [0.005577,0.016087]. Curve F1 .214892→.224389, but duplicates313→535 and cycle MAE1.998→2.580. Combined A/xyz head transfer not established. | [Reviewed final IXI results](../docs/research/metrics/3-ixi-node-refinement-pilot.md#reviewed-final-modern-ixi-results--28-october-2026) |
| **N08** | **GT-free relation-aware representative selection / uniqueness** on frozen modern A-xyz queries and original symmetric pair scores | **DONE — negative/mixed** | Two seeds, train-only settings, 128 train/480 validation patches per seed. Local profile Edge F1 **0.637638/0.637196**, versus identical-count NMS **0.637675/0.637184**; no consistent advantage and both graph-F1 CIs include zero. Guarded support adds **106/104 TP but 283/285 FP** and increases cycles. NMS loses short-edge/near-junction recovery. **No deployable selection change; stop further heuristic search.** | [N08 collected report](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026); [evidence](../experiments/gt_free_query_selection_20261026/worktree.json) |

### RELATION / GRAPH

| ID | Experiment & protocol | Status | What we learned / what we did **not** learn | Main evidence |
|---|---|---|---|---|
| **R01** | Legacy IXI full-query attribution, matched/unmatched/duplicate edge categories | **DONE — measured** | **668/733 (91.1%)** false edges touched spatially unmatched nodes; only 262/733 touched duplicates. Recovered-endpoint conditional relation recall was **12.57%**. Not a universal 91% law. | `6-ixi-query-attribution.md`, original Oct19 |
| **R02** | Modern SyntheticMRI full-query / geometry / exposure / overprediction attribution, B0 vs A-xyz, both seeds | **DONE — measured** | **82–83%** A-xyz FP touch spatially unmatched endpoints, but only ~35% touch Hungarian-unassigned. About **90.7–90.8%** FN GT edges lack a recovered endpoint pair. Net node count **underpredicted** ~0.8–0.9/patch despite false nodes; β₁ bias **+1.45–1.53**. | `6-ixi-query-attribution.md`, Oct26 |
| **R03** | Relation error vs GT arc length/tortuosity, modern SyntheticMRI and historical IXI | **DONE — measured** | A-xyz conditional recall shortest vs longest GT arc quartile **~99% vs ~90%**, BUT full-GT recall is **better on longer branches** because short-edge endpoint pairs are harder to recover. Near nonadjacent matched GT pairs (<5 mm) have ~47% FP rate in the examined SyntheticMRI populations. Length effect ≠ dominant overall loss. | `6-ixi-query-attribution.md`, Oct26 |
| **R04** | Legacy IXI original-vs-Hungarian-vs-spatial-oracle selection and retrained focal/regularized CE/matched/unmatched hard-negative relation heads | **DONE — negative/mixed** | No head improved original validation ranking; naive extra negatives can suppress valid connections. Already attempted mining **on older IXI**; do not claim “unmatched negatives never tested.” | `EXPERIMENT_REVIEW_20261021.md`; historical `8-ixi-relation-matching-calibration.md` |
| **R05** | Older IXI frozen detector: residual relation logit adapters — token/geometry, endpoint CNN, straight chord, 3D corridor | **DONE — mixed/overfit** | At **2k**, conditional AP original **0.4723**, endpoint **0.4907**, chord **0.5016**, corridor **0.5019**; image arms exceeded token control **0.4766**. At **65.6k**, all adapters were worse than original (chord **0.3216**). All used fixed older IXI detector/features, edge focal, same candidates. | `9-ixi-image-aware-relation-learning.md`, Oct26 early-score addendum |
| **R06** | Modern relation-specific 552D local-CNN token residual; frozen A-xyz, two seeds, R0/R1/R2 + continued-head/shuffled controls | **DONE — negative/mixed** | 364505: R0/R1/R2 conditional AP 0.97763/0.97408/0.97357; edge F1 0.63413/0.62870/0.63046; 92741: R0/R1/R2 conditional AP 0.97797/0.97505/0.97482; edge F1 0.63373/0.63012/0.63075. The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched. | [R06 measured report](../docs/research/metrics/9-ixi-image-aware-relation-learning.md#r06--relation-specific-552d-cnn-token-enrichment--27-october-2026) |
| **R07** | Modern SyntheticMRI reliable matched-negative redistribution, fixed CE/positive balance/pair budget, two frozen A-xyz seeds | **DONE — negative/mixed** | All matched negatives already enter S0. Fourfold reliable-negative exposure gives tiny fixed-cutoff F1 gains vs S0 (+0.000755/+0.000684), removing25/19 FP but losing3/1 TP; all-retained AP worsens in both seeds. Both remain below original A-xyz. Gate failed; no unfrozen/R08 launch. | [R07 reviewed results](../docs/research/metrics/8-ixi-relation-matching-calibration.md#r07-reviewed-conclusions-and-recovery-record--27-october-2026) |
| **R08** | Conservative unmatched-query relation supervision with spatial-reliability gates, and standalone S1/S2 controlled comparisons | **NOT RUN in this form** | Risk: alternative spatially plausible queries may be positive and Hungarian-assigned queries can be poorly localized. Do not label all unmatched as negative. | Dependent on R07 + reliability audit |
| **R09** | Learned path-aligned pairwise CNN feature aggregation inspired by Qi et al. | **DEFERRED** | Qi et al. demonstrated **segmentation** benefit, not graph adjacency. Need to distinguish direct GT edge from any connected vascular path through another junction; a pair-conditioned GT-free path for negative pairs is unresolved. R05 has already sampled chord/corridor. | Only if modern R06 or strong length-conditioned diagnostics warrant |
| **R10** | Combined image-enriched tokens + improved supervision / fully unfrozen graph optimization | **DEFERRED** | Combine only after independent wins, not as a first experiment. | R06 + R07 decision gates |

### GRAPH LOSS / BACKBONE / EVALUATION

| ID | Experiment & protocol | Status | What we learned / what remains | Main evidence |
|---|---|---|---|---|
| **L01** | Historical focal-node, focal-edge, combined-focal, unmatched-hard-pair source/synthetic MRI scratch matrix | **DONE — mixed** | Node focal helped some AP/Betti metrics; aggressive edge focal/hard negatives reduced loops but fragmented graphs. **Not** the same controlled source/population as S0. | `Focal_loss.md` |
| **L02** | Modern clean SyntheticMRI **node focal γ2 + matched edge CE** from scratch | **DONE — measured** | Strong S0 starting point, 2-mm edge F1 ~0.588 at clean baseline, then ~0.595 continued B0. Avoid silently replacing edge CE with focal. | Modern fine-feature report; query attribution |
| **L03** | Historical IXI class-weight/node-focal continuations | **DONE — negative/mixed** | More conservative node logits increase precision but remove real recovered anchors; these older initialization/loss-transition failures do **not** disprove S0 in-domain or future IXI-from-S0 transfer. | `EXPERIMENT_REVIEW_20261021.md` |
| **L04** | Two-scale deformable attention (C), ~3k SyntheticMRI pilot + fine-level intervention | **DONE — negative pilot** | Little/no reconstruction improvement; extra attention level used only a small effective portion and rolling fine features had little effect. A+C end-to-end is **NOT RUN**. | Multiscale/modern fine-feature report (verify path) |
| **L05** | SMD, β₀/β₁, node/edge, Branch/Curve diagnostics | **DONE — evaluation framework / ongoing** | SMD measures geometric coverage but can improve with overprediction; β₀/β₁ **absolute errors** hide signed bias. Use multiple complementary metrics, not isolated SMD or Curve F1. | Roadmap, `6-ixi-query-attribution.md`, metric docs |
| **L06** | Training-only BN recalibration on archived IXI and SyntheticMRI checkpoints | **PARTIAL / RECOVERY** | Q0 verified IXI1584593_0 failed empty-mask pair accounting; SyntheticMRI1584593_1 complete, collector1584594 cancelled. Synthetic 2mm node F1 .752528→.756058, edge F1 .445495→.439856; descriptive half only. Reuse it; no GPU recovery launched. | [Q0 partial BN result](../docs/research/spline_learning/spline-collapse-diagnosis.md#q0-verified-partial-training-only-bn-result--28-october-2026) |
| **L07** | Historical spline-model collapse and BN-vs-attention diagnosis | **DONE — diagnostic** | Failed scratch spline encoder becomes nearly input-invariant; switching batch-vs-saved BN does not restore it. Potential early spline coordinate gradients/optimization issue; not established single cause. | `spline-collapse-diagnosis.md` |

### GEOMETRY / REPRESENTATION

| ID | Experiment & protocol | Status | What we learned / what remains | Main evidence |
|---|---|---|---|---|
| **G01** | K8 target-fitting capacity audit on actual IXI and BRAVA centerlines | **DONE — GT-only** | IXI K8 generally adequate; BRAVA long curves have meaningful >1-mm maximum-error bottleneck. Pure oracle fitting does not prove learnability. | `EXPERIMENT_REVIEW_20261022.md`; spline capacity report |
| **G02** | Independently trained K8/K16/K24/native adaptive K/fixed C0/C1 **geometry heads on frozen IXI and SyntheticMRI detectors** | **DONE — measured** | Native adaptive K predicts ~4.3/4.5 controls on average, but little robust Curve F1 gain; K24/C1 not a demonstrated universal reconstruction improvement. These were **real trained geometry heads** but detectors frozen. | `EXPERIMENT_REVIEW_20261023.md` |
| **G03** | BRAVA Stage 1 scratch K8 standard vs coordinate refinement, 23.5k updates; zero-shot source comparisons | **DONE — negative/weak graphs** | IXI-init zero-shot was better than fresh scratch, but reconstructed graphs remained weak. Curve/head conclusions are unreliable when only a handful of correct edges exist. | `BRAVA_experiment_plan.md` |
| **G04** | BRAVA 30k-update IXI/SyntheticMRI/scratch recovery arms, older recipe | **DONE — weak graphs** | IXI-init gets node recall **0.341** but edge F1 **0.021**; not a recovered detector, threshold changes increase loops. Historical initialization/loss/patch population differs from G05. | `EXPERIMENT_REVIEW_20261023.md` |
| **G05** | **Modern** BRAVA original-K8 vs true degree-2 `split_K8` (fully trainable detector), and K24/fixed-four-section C1 **geometry heads** on the shared evolving original detector, 4k steps | **DONE — measured, weak graphs** | Original: node F1 **0.1183**, edge F1 **0.0089**, Curve F1 **0.0241**, Branch F1 **0**. K24/C1 Curve **0.0242/0.0243**, no meaningful graph gain. `split_K8`: Curve **0.0665**, edge F1 **0.0080**, β₁ MAE **0.785** vs **0.176**, larger graph. Do not call these independently trained full detector K24/C1 arms; geometry-to-detector spline gradients are detached. | `BRAVA_experiment_plan.md` Oct26 |
| **G06** | BRAVA GT oracle fitting K8 vs K24 vs fixed C1, 141 hard validation branches | **DONE — GT-only** | Mean sampled maximum error **1.571→0.397→0.360 mm**. Learned inference unable to exploit this reliably at 4k; not evidence that richer heads are ineffective. | `BRAVA_experiment_plan.md` Oct26 |
| **G07** | Trainable GT-error-adaptive variable-section C1, learned path segmentation, coupled geometry-detector gradients, controlled query-budget/degree-2 efficiency | **NOT RUN / DEFERRED** | Do not confuse fixed 4-section C1 with **GT-error-adaptive** subdivision; geometry-only improvements under `graph_only` cannot directly fix detector learning. | Geometry roadmap, pending conditional work |

### DOMAIN TRANSFER / DATA

| ID | Experiment & protocol | Status | What we learned / what remains | Main evidence |
|---|---|---|---|---|
| **D01** | Original biological/road image-to-graph transfer study over vascular domains | **DONE — historical measured** | Biological source pretraining favored generalization in earlier RelationFormer study, with domain/sample-size confounds controlled as reported in its source paper; this is a **different experimental setting** from S0. | MICCAI paper, source-transfer reports |
| **D02** | Scarce SyntheticMRI: 200 vascular patches alone vs 200 + 25k leaf patches (same target budget) | **DONE — measured** | Initially failed Slurm smoke but later recovered and **completed** (do not retain the 21 Oct “never run” status). Test node F1 **0.3246→0.3855**, edge F1 **0.2588→0.2979**; Branch F1 ~0.0004/0.0005, other topology mixed. Single seed/unequal vascular exposures. | `EXPERIMENT_REVIEW_20261022.md`, limited-data botanical report |
| **D03** | Historical IXI binary vascular-mask-input oracle vs image input | **DONE — diagnostic training** | Mask information helps but doesn't solve node selection/relations; it is an **image representation oracle**, not deployable MRI-only input. | `EXPERIMENT_REVIEW_20261021.md`, IXI mechanism report |
| **D04** | Modern SyntheticMRI → IXI fully unfrozen B0/A-xyz/A-cls transfer | **DONE — measured** | All six1653860 tasks and collector1653861 completed0:0; same augmented input/target stream,256 queries,5,500 updates. B0/fresh-xyz/transfer-xyz/A-cls/A/control edge F1 .087926/.094314/.094642/.101536/.091748/.087834. A-cls gain exceeds count control; localization helps separately; graph quality remains weak and topology mixed. No extension. | [Completed study](../docs/research/metrics/3-ixi-node-refinement-pilot.md#reviewed-final-modern-ixi-results--28-october-2026) |
| **D05** | BRAVA dataset transformation, orientation correction, graph target preparation | **DONE — measured QC** | 56 paired images; 39/8/9 subjects; 8,768/1,796/2,043 model-ready patches; five hash-pinned x-reflections; **brain masks ≠ vessel masks**. | `BRAVA_preparation.md` |
| **D06** | Representative BRAVA train/val K8 monitor, saved-vs-batch BN, training-only BN recalibration, confidence calibration, augmentation geometric-alignment tests | **DONE — diagnostic, mechanism partly unresolved** | At 4k, matched saved-BN **train node/edge F1 0.1325/0.0114**, val **0.1183/0.0089**; failure occurs on optimized-on training patches (not just val). Training-only BN recalibration **does not rescue** and batch statistics make performance worse. Train-selected node threshold 0.4 helps modestly; augmentation transformations passed alignment checks. No retraining was done. | `BRAVA_experiment_plan.md`, Oct26 learning-diagnosis addendum |
| **D07** | BRAVA original-K8 fully trainable augmentation off vs matched fresh augmentation on; same clean source/1,248 patches/seed/order/4k updates | **DONE — measured** | 4k off−on train node/edge F1 +0.2908/+0.1551; validation +0.0285/+0.0062. Large fit gain; validation edge CI includes zero, raw 2-mm recall falls and loss/cycles worsen. Memorization is plausible; no recovered detector or follow-up launched. | [D07 report](../docs/research/BRAVA_experiment_plan.md#d07-controlled-original-k8-augmentation-onoff--27-october-2026); [artifacts](../experiments/brava_augmentation_20261026/analysis.json) |
| **D08** | BRAVA full-cohort long-training initialization comparison | **DONE — measured** |300 epochs/82,200 target updates;8,768 train/39 subjects,1,796 val/8. Final2 mm node/edge F1: scratch 0.02544/0.00110, SyntheticMRI 0.35572/0.06794; transferred Branch/Curve F1 0.00647/0.12174. Pretraining helps, absolute graph quality remains weak; no automatic extension. | [Reviewed BRAVA results](../docs/research/BRAVA_experiment_plan.md#reviewed-final-d08-results--28-october-2026) |

---

## 3. One-glance test matrix (the checklist for future prompts)

**Legend:** ✅ measured; ⚠️ mixed/negative/partial; ⬜ not confirmed tested; ⏸ gated/deferred.

| Scientific question | Tested already? | Current decision / blocker | Related IDs |
|---|---|---|---|
| Does local fine CNN refinement improve node localization? | ✅ | Yes on modern SyntheticMRI, two seeds. Use A-xyz as reference. | N01 |
| Does refining class logits alongside xyz help? | ✅ | No meaningful independent gain in A ablation. | N01 |
| Does a historical contextual/KNN verifier help? | ⚠️ | No; don't replay V3 as a new idea. | N03 |
| Does standalone image-conditioned `A-cls` help on modern IXI? | ⚠️ | Yes for physical edge/Curve F1 in one controlled seed, including B0 count matching; duplicates/cycle MAE worsen. Combined A adds no supported benefit. | N07,D04 |
| Is120 capacity-safe for current contracted IXI targets? | ✅ | No: full train/val max153/141 with5/5 overflows; modern4096/512 subsets contain1/1 overflows. Zero degree-2 object nodes. Retain256; Q120 gate fails. | J02 / Q0 |
| Can simple spatial NMS solve duplicates? | ⚠️ | Tested; loses correct anchors/edges. | N04 |
| Why does confidence GT oracle beat nearest GT oracle? | ✅ | ~33% representatives change; confidence choices are usually less accurately localized but relation-richer, Hungarian-aligned alternatives. Mechanism partly established; training-history causality remains unproven. | N05–N06 |
| Can a GT-free method reproduce useful oracle query selection? | ⚠️ | N08 completed: tested local/global relation rules do not consistently beat exact node-count controls in both seeds; global support adds false edges/cycles. Oracle gap remains diagnostic. | N08 |
| Do long/tortuous GT edges have worse conditional recall? | ✅ | Yes, **conditional**, not overall dominant loss. | R03 |
| Did we ever feed fine CNN endpoint/path features to relation scoring? | ✅ | Yes, legacy IXI frozen-detector residual **edge-logit** adapters; early gain, late overfit. | R05 |
| Did we refine the **552D node tokens** for a modern A-xyz relation head? | ✅ | Completed frozen two-seed R0/R1/R2, continued-head and shuffled-feature controls; see measured R06 result. | R06 |
| Did we already try extra matched/unmatched hard negatives? | ✅ | Yes on historical IXI and mixed-training settings; no demonstrated overall gain. | R04, L01 |
| Did we test *modern reliable near-nonadjacent matched-negative sampling* with edge CE? | ✅ | R07 completed as a bounded two-seed fixed-detector multiplicity comparison; every matched negative already enters S0. S1 does not pass the predefined two-seed ranking/F1/TP gate. Do not promote the sampler or launch unfrozen/R08 training automatically. | R07 |
| Did we test reliable unmatched-query labels with ambiguity safeguards on S0? | ⬜ | Not in this controlled form. | R08 |
| Is a larger multiscale attention transformer already a win? | ⚠️ | Short C pilot negative; A+C/long clean test not established. | L04 |
| Was BRAVA K24/C1 **trained**? | ✅ | Yes, learned independent geometry heads on **shared trainable detector**; not independent full detector replicates. | G05 |
| Is K24/C1 geometrically more expressive on BRAVA GT? | ✅ | Yes, oracle fits; learned graph benefit not demonstrated. | G06 |
| Was BRAVA degree-2 **graph-changing** subdivision tested? | ✅ | Yes, increases partial Curve F1 and cycles/output size. | G05 |
| Was genuine trained error-adaptive C1 / coupled detector geometry tested? | ⬜ | Not established; defer. | G07 |
| Is the weak BRAVA detector due to BN or poor larger-cohort optimization? | ⚠️ | D06 found no BN rescue; D07 augmentation-off mostly improves train fit. D08 completed 300 epochs: pretraining improves val node/edge F1 to 0.35572/0.06794, but train remains 0.39449/0.08989 and scratch nearly fails. More exposure helps but is insufficient under this recipe; persistent fitting/optimization and a generalization gap remain. | D06–D08 |
| Does full-cohort BRAVA training rescue scratch or benefit from SyntheticMRI initialization? | ✅ | D08 completed300 epochs: transferred node/edge F1 0.35572/0.06794 versus scratch 0.02544/0.00110. Scratch remains weak on train; transferred endpoints/graphs remain limiting. | D08 |
| Does the native2D port run and fit a tiny real-image set? | ✅ | CPU/CUDA and original-width architecture checks pass; two-crop node/edge/Branch F1 1.0. Full 2D training, subject-disjoint generalization and controlled A-xyz benefit remain untested. | Native2D-01 |
| Was limited-data leaf transfer tested? | ✅ | Yes, after infrastructure recovery; historical 21 Oct pending status superseded. | D02 |
| Did training-only BN recalc complete on both domains? | ⚠️ | Only SyntheticMRI GPU stage completed; IXI failed, collector cancelled. | L06 |

---

## 4. Prompt/run inventory — evening update

| Workstream | Verified status from supplied material | Avoid duplication / next action |
|---|---|---|
| **P2 / N06 — Oracle representatives** | **DONE.** Paired two-seed SyntheticMRI analysis, same-neighbor controls, single-query swaps, exact embedding replay job `1597409`. No new model trained. | **Do not repeat.** N08 has also completed without consistent count-controlled benefit; do not relaunch its tested heuristic set. |
| **Q1 / N08 — GT-free representatives** | **DONE — negative/mixed.** Collected CPU-only two-seed cache analysis; no GPU or training jobs. Primary local rule is neutral versus count-matched NMS; guarded support increases FP/cycles. | **Stop further heuristic search.** Review spatial identity/assignment–relation coupling. R06/R07 remain conditional hypotheses, not automatically authorized next runs. |
| **P3 / D06 — BRAVA learning diagnosis** | **DONE.** Recovered GPU evaluation `1598848` and dependent collection documented as measured. Representative cohorts, BN/augmentation checks and confidence calibrations reported. | **Do not repeat.** Advance to **D07** augmentation-on/off **training** ablation. |
| **D08 — full-cohort BRAVA long training** | **DONE — measured**, both arms completed 300 epochs/82,200 updates; all six collectors completed. Pretrained val edge F1 0.06794, scratch 0.00110; training graphs remain weak. | Do not duplicate or automatically extend. Review cached assignment/localization/confidence and target coverage; no architecture/loss change or test evaluation authorized by this result. |
| **P1 / L06 — historical training-only BN recovery** | **PARTIAL / RECOVERY, verified Q0.** IXI failed empty-mask accounting, SyntheticMRI complete, collector cancelled. | Preserve completed SyntheticMRI half. Future IXI-only diagnostic recovery needs narrow dtype correction; no GPU submitted by Q0. |
| **D04/N07 — modern IXI B0/A-xyz/A-cls transfer** | **DONE — measured28Oct.** All six5,500-update arms and collector completed; input streams match exactly. | A-cls is the strongest physical-edge candidate but increases duplicates/cycles. Preserve B0/xyz controls; no automatic extension or architecture combination. |
| **R06 — relation-specific 552D token fusion** | **DONE — negative/mixed.** Frozen modern comparison completed, with original graph supervision. | The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched. |
| **R07/R08 — modern reliable matched/unmatched negative supervision** | **R07 DONE**, collected modern matched-pair comparison. **R08 NOT RUN**; no unmatched supervision launched. | S1 does not pass the predefined two-seed ranking/F1/TP gate. Do not promote the sampler or launch unfrozen/R08 training automatically. |
| **R09/G07/L04 A+C** | **DEFERRED** | No new path tracker, K24/C1 geometry sweep or multiscale attention before simpler mechanisms improve graphs. |

The status of actual current Codex sessions and queued Slurm tasks cannot be inferred from these attachments; inspect live Jean-Zay before submitting anything.

---

## 5. Updated priority queue — next prompts for Sol

### Q0 — IXI/BN status and actual contracted-target capacity — **DONE — measured, 28 October 2026**

- J00 verified D04 is PARTIAL / RECOVERY and N07 production training never ran: replacement smoke1583591 hit a multiprocessing socket-path error and timed out;1583592/1583593/1583594 cancelled. No duplicate GPU job was submitted.
- L06 remains partial: completed SyntheticMRI half collected descriptively; IXI failed empty-mask accounting and collector cancelled. No BN rescue conclusion.
- J02 audited all23,213 train/val VTP–K8 pairs (CPU1653086): IXI full max153/141,5/5 above120; modern subsets max153/124,1/1 above120. BRAVA max64/69, no overflow. All object graphs already degree-2 contracted.
- **120 safe for current IXI: no. Completed modern IXI result: no.** Keep256. One permitted next GPU run is recovery of the existing correctness/benchmark smoke with short TMPDIR; train only after its gates pass. The next independent queue task is2D-QC. [Existing detailed report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#q0-ixi-status-and-contracted-target-capacity-audit--28-october-2026).

### Q1 — N08: GT-free relation-aware query/representative selection — **DONE — negative/mixed; stopping gate reached**

**Collected outcome (supersedes the former READY status):** the bounded cache-only study completed on both seeds. Training selected a common local-profile weight of 1.0 before validation; all node counts were matched per patch to confidence and spatial controls. At 2 mm, local-minus-NMS Edge F1 was **−0.000037 [−0.000395, +0.000301]** and **+0.000012 [−0.000471, +0.000454]**. The shared-anchor AP change was tiny and inconsistent across seeds. The guarded rule increased both false edges and cycle bias. **Stop; no additional selection sweep, architecture or training was launched.**

**Remaining interpretation:** existing relation information is useful under GT-supplied identities (N06), but these GT-free rules cannot reliably infer that identity. This supports investigating spatial-quality/assignment/connectivity coupling; it does not prove relation tokens lack information or that a particular supervision change will work. See the [full N08 report](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026). Original Q1 scope below is retained for provenance, **not** as an outstanding launch request.


- Reuse frozen two-seed modern SyntheticMRI A-xyz full raw queries, refined coordinates, node confidences, original symmetric relation scores and validation protocol.
- Design a small **GT-free** selection method: e.g. conservative local candidate groups with confidence/uniqueness and *calibrated* relation support from independent candidate neighbors. **Do not** let self-consistent spurious loops or high-degree hubs automatically dominate.
- **Controls:** standard threshold decoding, simple confidence-based suppression, matched-retained-node count, and optional simple GT-free confidence-ranking baseline. The GT-assisted oracle remains a **diagnostic ceiling** only.
- Lock candidate grouping/selection hyperparameters with training data and verify behavior on all fixed 480 validation patches/13 subjects for both seeds. Do not use GT to form groups, choose representatives or construct relation scores.
- Report node/edge F1 at 1/2 mm, all-retained/conditional AP with populations, true/false edge changes, missed-anchor recovery, duplicate counts, β₀/β₁ signed bias and MAE, GT shortcuts and high/low-confidence strata. Detect recall loss or silent edge pruning. No model training in first stage.
- **Stop** if there is no consistent improvement over matched-count confidence control in both seeds, or improvements arise only from more/fewer predicted nodes. Otherwise recommend the smallest trainable selection mechanism to investigate later.

### Q2 — D07: BRAVA K8 augmentation-on/off training ablation — DONE, measured 27 October 2026

- Start from exactly the existing clean SyntheticMRI checkpoint and original BRAVA **K8** target. Use the same 1,248 train/256 val patches, original 39/8 subject split, seed, batch 8, 4,000 updates, optimizer, LR schedule, node focal γ2, matched-pair edge CE, `graph_only` geometry route, and saved-BN evaluation.
- Compare **unchanged augmentation on** versus **geometric/intensity augmentation off** as the only experimental variable; reuse fully documented old augmentation-on control if parity is demonstrable, otherwise run a paired control. No K24/C1/split_K8 or node embedding changes.
- Record source hashes, order of training sample IDs, evaluation batch 1 and 8 parity, original 16-patch tiny reference (monitor only), and **representative 160-train vs identical 256-validation** curves at fixed 1k/2k/4k checkpoints.
- Main question: does disabling augmentation improve *optimized-on representative training fit*, raw 1/2-mm node recovery, confidence retention, edge F1 and conditional relations, without merely overfitting validation? Report train/val differences, assignment churn, signed counts, β₀/β₁ and SWC-annotation caveat.
- **Stop after 4k** and report the causal ablation. If both remain poor, next experiment should isolate training exposure/budget or matching rather than increasing spline capacity.

### Q3 — R07: reliable nearby nonadjacent matched-pair supervision — DONE, collected 27 October 2026; original scope below retained for provenance

- First audit negative pair exposure on modern A-xyz and quantify truly reliable close nonadjacent pairs versus ambiguous alternate representatives, plus recall on longer positives.
- If warranted, compare existing matched CE sampling to a matched-budget negative sampler that emphasizes **unambiguously** nonadjacent nearby anchors; keep network/optimizer and candidate sets fixed. Preserve all positives and avoid blanket unmatched negatives.
- Evaluate ranking AP, long-positive recall, near-shortcut false-positive rates, edge TP/FP/FN, signed edge bias, cycles and subject uncertainty. Stop if it merely reduces all edges.

### Q4 — R06: relation-specific 552D node-token fusion — DONE — negative/mixed, 27 October 2026

- If GT-free selection cannot exploit existing embeddings, ask Astra High to choose a minimal design and matched controls. Sol High then implements R0 original, R1 token-only residual, R2 local-CNN residual fusion at **same 552D dimensionality** with unchanged node coordinates, early/fixed candidate evaluation and strict overfit controls.
- **Do not** repeat historical 65.6k frozen IXI endpoint/chord/corridor logit experiments. Combine R06 with R07 only after independently justified wins.

### D08 — full-cohort BRAVA long-training closeout — **DONE — measured**, 28 October 2026

Both initializations completed 300 epochs under identical inputs and supervision. SyntheticMRI improves train and validation reconstruction, but train/val edge F1 remains 0.08989/0.06794; scratch remains near zero. Additional exposure alone did not recover a functional cohort detector under this schedule. Review cached assignment/localization/confidence and annotation-relative target coverage before recommending another run. Do not launch an extension or new representation automatically. See the [completion review](../docs/research/BRAVA_experiment_plan.md#d08-completion-review--28-october-2026).

### Q5 — geometry and long-range path models — **DEFERRED**

- BRAVA degree-2/K24/C1 capacity, image-conditioned spline/path edge verification, deformable/multiscale attention, and new adaptive GT representations remain tracked, not forgotten.
- Unlock BRAVA geometry work only after representative training graph detection is functional; unlock path models only if a significant residual long-edge problem persists after node selection/relation supervision.

---

## 6. Updated decision gates

| Observation | Consequence |
|---|---|
| **Q1 GT-free relation-aware selection improves both seeds over matched-count controls without worse recall/topology** | Consider a small trainable relation-aware query verifier or structured selection objective; keep simple frozen inference rule as control. |
| **Q1 fails or amplifies loops/shortcuts — observed in N08** | Stop the tested heuristic search. The oracle advantage was not recovered by these GT-free rules. Review spatial identity/assignment–relation coupling; consider R06 or R07 only after a focused causal/reliability audit. N08 alone establishes neither a token-information deficit nor a successful new supervision recipe. |
| **Q2 augmentation-off noticeably improves representative BRAVA training fit** | Repeat with carefully matched exposure and then a different seed; assess whether generalization suffers. |
| **Q2 augmentation-off does not improve training fit** | Isolate optimization budget, per-patch exposure, assignment dynamics and annotation/image difficulty. Do **not** infer from small 16-patch overfit alone that population training simply needs more updates. |
| **R07 improves nonadjacent edge ranking at maintained positive recall** | Confirm modest fully trainable study, then evaluate whether adding reliable unmatched supervision is warranted. |
| **R07 produces tiny F1/precision trade-offs without better AP or original-checkpoint recovery — observed** | Retain original A-xyz. Do not promote this sampler, extend training or launch R08/unfrozen confirmation automatically. Investigate spatial-quality/assignment and training-to-inference label coverage before another supervision policy. |
| **R06 improves ranking beyond token-only residual and valid baseline** | Test short fully unfrozen confirmation; beware R05 overfitting and reused IXI subjects. |
| **R06 fails to beat R0/R1 at the fixed final endpoint — observed in both seeds** | Retain original A-xyz; stop this frozen residual. Review spatial-quality/assignment/connectivity supervision and training-versus-inference coverage. Preserve N08 negatives; do not automatically launch more selection heuristics, an unfrozen model or an R06+R07 combination. |
| **D08: 300 epochs improve pretrained train/val recovery, but both remain weak — observed** | Preserve the measured initialization/exposure gain and scratch failure. Diagnose persistent cohort fitting/optimization as well as generalization; do not claim a pure generalization-only cause, bad raw MRI or a K8 limit. Review cached assignment/confidence/target coverage before another extension; no automatic launch. |
| **D08 initialization benefit with persistently weak graph recovery — observed** | Keep transferred K8 as the measured BRAVA reference; inspect cached training-only operating points and endpoint/assignment errors before more geometry or duration. No new run authorized. |
| **Native2D-01 correctness/tiny-fit passes — observed** | Implementation gate passes; require annotation/crossing QC, biological grouping, scale and a fixed development protocol before full training or a B0/A-xyz claim. |
| **Q0/J02: current contracted IXI targets still overflow120 — observed** | Do not launch Q120, remove hard patches or regenerate labels. Keep the existing256 reference;192 is capacity-sufficient for audited train/val only, without an accuracy claim. Recover D04 smoke before production training. |
| **Modern IXI A-cls gains edge/Curve at matched counts but worsens duplicates/cycles — observed** | Treat as a single-seed accuracy/topology trade-off. Keep B0 and xyz controls; require independent confirmation before broader promotion. Do not claim a topology solution, universal xyz+cls benefit or automatically launch a new backbone/long run. |
| **Modern IXI/BN completion recovered** | Re-evaluate cross-domain prioritization using the actual new results; don't transfer historical weak IXI claims to modern A-xyz. |

---

## 7. Required controls for any future Codex prompt

Copy this short checklist into any new agent instruction:

- [ ] **Read this registry and the specific source report(s)** before proposing or launching experiments.
- [ ] Name the **existing ID** being extended (e.g. `R06`) or explicitly justify a new ID; check whether the arm **already exists**, including failed/earlier frozen variants.
- [ ] Declare dataset version, annotation semantics, subject split, patch list, checkpoint path/hash, seeds, loss/matcher/optimizer, training duration, trainable/frozen components, node/edge selection and all thresholds.
- [ ] Preserve **exactly the same candidate node population** to isolate relation learning, where that is the intended comparison. Preserve equal matched budget/seed when changing graph supervision.
- [ ] Distinguish **GT-only fitting**, **GT-assisted oracle**, **frozen-detector head training**, **independently fully trained models** and **coupled end-to-end training**.
- [ ] Check live `squeue`/`sacct` and previous submission records; plan around **at most two concurrent H100s across ALL agents**, not per-agent. No duplicate existing jobs; prefer CPU and cached scores for diagnostics.
- [ ] Use isolated branches/experiment outputs; no simultaneous edits to the same roadmap/report. Respect WORK inode limits and stage large caches on SCRATCH. Do not overwrite earlier checkpoints or evaluation archives.
- [ ] Match training and validation **subjects** properly, not merely patches; retain test isolation. Treat repeat-use of validation as exploratory; prefer new subjects/locked confirmations when selecting architectures.
- [ ] Use node/edge PR/F1, raw vs filtered/oracle recall, conditional relation AP, FP subtype, signed graph counts, β₀/β₁ and Branch/Curve F1 together. Report prevalence and tolerance; avoid improving one metric by uncontrolled overprediction.
- [ ] **Append actual measured results, failed jobs and remaining uncertainty** to an existing domain report **and update this registry**. Summarize what was **not** tested; never label a scheduled collector “completed” before checking outputs.

---

## 8. Source reports and evidence index

These are **file names / repository-relative paths to locate**; repository location may vary by worktree. They are not copies of the experiments. Prefer the latest dated section when earlier “pending” statements have been superseded.

1. **`6-ixi-query-attribution.md`** — original IXI full-query/duplicate accounting **and Oct26 modern SyntheticMRI oracle/long-range/FP study**. Main source for `N02/N04/N05`, `R01–R03`, `P2`.
2. **`9-ixi-image-aware-relation-learning.md`** — frozen IXI logit residual arms, 1k/2k/65.6k AP, early-checkpoint uncertainty and overfitting. `R05`.
3. **`BRAVA_experiment_plan.md`** — Stage 1 scratch/zero-shot, **Oct26 fully trainable original/split and K8/K24/C1 geometry** results, confidence/oracle checks and diagnostic limitations. `G03–G06`, `D06`.
4. **`BRAVA_preparation.md`** — SWC coordinate units, five reflections, splits, lack of vessel masks; `D05`.
5. **`spline-collapse-diagnosis.md`** — old scratch spline failure and **Oct26 BN submission** `1584593/1584594`; `L06/L07`.
6. **`EXPERIMENT_REVIEW_20261021.md`** — historical IXI mask-input, node-focal and Hungarian/matched/unmatched relation comparisons. **Some statuses superseded by Oct22**.
7. **`EXPERIMENT_REVIEW_20261022.md`** — BRAVA/IXI K8 fitting bounds, recovered BRAVA Stage 1, **completed scarce-data botanical transfer**. `G01/D02`.
8. **`EXPERIMENT_REVIEW_20261023.md`** or **`multidataset_continuous_graph_experiments.md`** — BRAVA 30k recovery arms and real learned geometry heads on frozen IXI/SyntheticMRI detectors. `G02/G04`.
9. **`node_verifier_feasibility.md`** — failed V0–V3 verifier and why it is not A-cls. `N03`.
10. **`Focal_loss.md`** — historical focal/matched/unmatched graph-loss matrix; `L01`.
11. **`IXI_graph_extraction_diagnostic_roadmap.md`**, **`IMPLEMENTATION_TRACKER.md`**, experiment manifests and source checksum files — background and older proposal IDs; verify if stages have since been superseded.
12. **`Qi et al. — A Flexible Structure-Guided Feature Aggregation Paradigm for Efficient Vascular Representation.pdf`** — useful inspiration for **structure-aligned image aggregation**, but only **segmentation** evidence, not proven graph-edge refinement.

### N08 collected evidence and reproduction

- Report: [query attribution, N08](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026).
- Isolated branch/worktree: `research/gt-free-query-selection-20261026`, sibling `gnbm-query-selection-20261026`.
- Outputs: `gnbm-query-selection-20261026/experiments/gt_free_query_selection_20261026`; [repository pointer](../experiments/gt_free_query_selection_20261026/worktree.json). `summary.json`, `training-selection.json`, per-patch/per-subject tables, ranking populations and source/checksum audits are preserved.
- Checkpoints: A-xyz step 12,000, seeds **364505/92741**, SHA256 **`a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9`** / **`9646b7b266dec3463d94d16cfbcef0332044a9bf0d599c2319a2eff2c93d3e53`**. Original symmetric scores and coordinates unchanged; no optimizer or model updates.
- Decisions: training-only common settings; edge cutoff 0.5; physical 1/2-mm evaluation; exact per-patch node-count controls; 2,000 whole-subject graph bootstrap draws and 500 AP draws, seed 261026. Reused validation evidence is exploratory.
- Slurm: **none for N08**; all work reused caches on CPU. N06's job 1597409 was the earlier exact embedding replay, not N08 inference or training.
- Print saved results from the parent `projects` directory: `python3 gnbm-query-selection-20261026/scripts/show_query_selection.py`.

### Latest known Slurm identifiers (historical; not evidence of currently running work)

| Series | Jobs | Known outcome |
|---|---|---|
| Modern BRAVA representation | CPU target `1584757`; tiny `1586043/1586044`; pilots `1586045/1586046`; CPU evaluations `1586047–1586052` | **Completed**, results in BRAVA report |
| Training-only BN | GPU array `1584593_0` IXI / `_1` SyntheticMRI; collector `1584594` | **IXI failed / SyntheticMRI completed / collector cancelled** |
| Older IXI image relation | GPU `1331638–1331645` and reporting recovery `1344394/1344395` | **Completed and analyzed**, including saved early checkpoints |
| Modern full-model IXI transfer/refinement | **No verified ID supplied** | **STATUS UNKNOWN**, check P1 |

---

## 9. Dated change log

- **2026-10-26 — Initial registry.** Consolidated modern SyntheticMRI A-xyz/error attribution, legacy IXI relation adapter evidence, historical matched/unmatched mining, completed BRAVA K24/C1 and genuine degree-2 comparison, scarce-data botanical recovery, user-supplied BN failure accounting, and the unlaunched Astra prompts. Defined the active sequence **P1 status/BN → P2 oracle queries → P3 BRAVA diagnosis → P4 carefully chosen relation intervention**.

- **2026-10-26 (evening) — N06/D06 collected:** Detailed two-seed oracle-representative attribution shows improved relation-relevant node embeddings/assignment consistency for confidence-preferred queries; NMS remains ineffective beyond node-count control. BRAVA representative train and validation performance both remain low; BN recalibration fails to rescue, augmentation transformations pass alignment. Added **Q1 GT-free selection** and **Q2 matched augmentation-off** as immediate Sol work, with R07 and R06 subsequent; prior P2/P3 marked complete. BN and modern IXI status remain unverified in supplied attachments.

- **2026-10-26 — Repository import and N08 reconciliation.** Preserved the supplied registry byte-for-byte under `research/evidence/experiment-registry/`; added this working copy to the research index. N08/Q1 is **DONE — negative/mixed**, with quantitative controls, stopping decision, checkpoint identities and evidence links. Updated §3 matrix, §4 inventory, §5 actions and §6 gates. All unrelated experiment statuses remain as supplied, rather than being guessed from earlier conversations. No jobs were launched during this documentation update.

**Next edit owner:** agents auditing Q0 or explicitly assigned Q2/conditional relation work. Q1 is complete; do not repeat it automatically. Preserve stable IDs and old failed experiments; do not replace the whole roadmap.

- **2026-10-27 — D07 submitted:** smoke 1600406 completed; matched original-K8-only training array 1612527 and dependent collector 1612529 accepted. CPU protocol/source/data/order and collector checks passed. Maximum two concurrent H100s reserved under the common submission lock; no follow-up or architecture sweep submitted.


<!-- N08 REQUEST RECONCILIATION 20261027 -->
### 27 October 2026 — N08 renewed-request audit; no duplicate experiment

N08 remains **DONE — negative/mixed**. The renewed inference-only selection request matches the completed two-seed, training-selected, exact-count-controlled study. Saved outputs/parity records and the GT-free implementation were rechecked; five focused tests passed. No numeric results or historical statuses were overwritten; no jobs or new model inference were launched. **Single next-direction recommendation: no further selection work on the present evidence; preserve unchanged A-xyz inference.** N06's oracle advantage does not justify a trainable selector or relation-token architecture without additional independent evidence. See the [dated N08 recommendation](../docs/research/metrics/6-ixi-query-attribution.md#n08-request-reconciliation-and-single-next-direction-recommendation--27-october-2026). Earlier conditional relation ideas remain hypotheses, not authorized follow-up runs.

<!-- BEGIN D07 COLLECTED 20261027 -->
### D07 collected outcome — 27 October 2026

4k off−on train node/edge F1 +0.2908/+0.1551; validation +0.0285/+0.0062. Disabling augmentation improves representative training reconstruction on node and edge F1. Validation node F1 improves modestly, but the edge-F1 gain is uncertain across subjects (its paired 95% interval includes zero). Validation raw 2-mm localization recall decreases and Hungarian distance worsens. The much larger train gain, declining clean training loss and worse final validation loss are consistent with increased memorization/overfitting; this is not a broadly improved or recovered detector. The matched comparison supports an optimization cost of this augmentation regime under the fixed budget, while one seed and reused development subjects limit the generalization claim. Connectivity remains weak in both training arms and on validation. Recommended follow-up, not launched: separately isolate exposure/training budget with a predefined schedule, or assignment/optimization with fixed exposure; change one factor at a time and retain the generalization gap as an explicit endpoint.

GPU smoke `1600406` passed; matched training array `1612527` reached 4k in both arms, then failed the final numerical batch-parity assertion. Initial collector `1612529` was cancelled. Inference-only recovery `1612855` used the saved 4k weights with zero optimizer updates; CPU collection `1612856` produced fixed 1k/2k/4k metrics. Failed logs/checker source remain preserved. Source `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/fine_feature_architecture/runs/baseline/best.pt`, SHA256 `d4bc78ad3eb7c1c29c9be8ec3e2927e88f2f45e6f34a3cb8be9ade5fd9d2431c`. Exact six checkpoint hashes and paths are in `experiments/brava_augmentation_20261026/analysis.json`. Fresh K8-only controls were used because historical auxiliary-head whole-run parity was not established. Seed 261039, batch eight, 1,248 train/256 validation patches, 39/8 subjects, fixed target/matcher/loss/schedule; saved-BN batch1/8 numerical drift is reported. No new architecture, test evaluation or threshold selection.

Decision gate: distinguish improved optimized-on training fit from validation improvement and memorization. Both arms receive approximately 25.64 draws/patch; earlier 16-patch overfit used different exposure/schedules and cannot prove duration alone is the cause. Incomplete SWC errors are annotation-relative; brain masks are not vessel labels. Any exposure/assignment follow-up remains a recommendation only.

[D07 report](../docs/research/BRAVA_experiment_plan.md#d07-controlled-original-k8-augmentation-onoff--27-october-2026); [artifacts](../experiments/brava_augmentation_20261026/analysis.json)
<!-- END D07 COLLECTED 20261027 -->


<!-- R07 SUBMITTED 20261027 -->
### R07 audit and bounded submission — 27 October 2026

All matched negatives are already included under cap 9,999; current issue is relative contribution/representative reliability, not missing binary exposure. Each training seed: 111 nearby negatives, 18 reliable, 7/6 currently accepted FP. Accepted GPU smoke 1613828 → two-seed array 1613829 (S0/S1 per seed; 1k updates, batch8) → CPU collector 1613830. Reuse 128 train/480 val patches per seed; detector/queries/geometry frozen. Six tests and real-cache CPU parity passed. R07 remains independent of negative N08; no query selector or R08 implemented. Results pending. See existing relation-supervision report and isolated artifacts.

<!-- BEGIN R06 COLLECTED 20261027 -->
### R06 collected — 27 October 2026

The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched.

364505: R0/R1/R2 conditional AP 0.97763/0.97408/0.97357; edge F1 0.63413/0.62870/0.63046; 92741: R0/R1/R2 conditional AP 0.97797/0.97505/0.97482; edge F1 0.63373/0.63012/0.63075. Frozen residual/head controls use exact node candidates and unchanged graph CE/sampling. The image ablation, early/final curves and graph/topology results are in the existing relation report. No R07 combination, path model, query merging or transformer change.
<!-- END R06 COLLECTED 20261027 -->


<!-- R07 FAILED SMOKE AUDIT 20261027 -->
### R07 failed-smoke accounting — 27 October 2026

**PARTIAL / RECOVERY: GPU smoke failed; relation training did not run.** Slurm 1613828 exited 1 after 34 seconds at strict checkpoint loading: `node_refinement.network.2.weight` / bias are `[3,128]` / `[3]` in the coordinate-only A-xyz checkpoint, but the constructed model expects `[5,128]` / `[5]`. This is a model-construction/checkpoint compatibility failure, not a negative scientific result. The preceding CPU smoke exercised the relation head and existing token caches, so it did not establish full detector-construction compatibility.

The `afterok` gates cancelled training array 1613829_[0-1%2] and collector 1613830 before execution. No feature cache, GPU-smoke completion, trained R07 head or final summary exists. The completed CPU supervision audit and original checkpoints remain intact. The earlier submission entry is historical; it must not be read as completed training. Compatibility correction and a new gated submission are needed; this status check did not modify model code or launch jobs. Evidence: isolated `experiments/r07_matched_supervision_20261027/status-check-20261027.json` and `gpu-1613828_4294967294.err`.

<!-- R07 CHECKPOINT LOADER RECOVERY 20261027 -->
### R07 loader fix and recovered submission — 27 October 2026

Import-order root cause reproduced and fixed without checkpoint/config/loss changes. Full CPU strict loading passed for both source checkpoints; six sampler/evaluation tests and real-cache smoke passed. Accepted 1624151 → 1624152_0/1 → 1624153, global maximum two H100s. Old failed attempt archived. R07 scientific results remain pending; no R08 or unfrozen run launched.

<!-- R07 COMPLETED TRAINING FIXED CUTOFF REVIEW 20261027 -->
### R07 completed-training review — 27 October 2026

Both GPU tasks 1624152_0/1 completed. At 1k updates/0.5 cutoff, S1 Edge F1 0.631252/0.631416, below original 0.634134/0.633730; conditional AP 0.974666/0.975581, below original 0.977632/0.977971. S1 versus continued S0 removes 25/19 FP but loses 3/1 TP and does not improve conditional AP. Fixed-cutoff results/subject intervals appended to existing relation report. Full CPU collection restarted as 1624612 after a float32 threshold-tie correction, with zero new training. Keep status partial until final collection; no R08 or unfrozen run.

<!-- R07 COLLECTED 20261027 -->
S1 does not pass the predefined two-seed ranking/F1/TP gate. Do not promote the sampler or launch unfrozen/R08 training automatically. No R08 or unfrozen confirmation launched. See existing relation-supervision report.

<!-- BEGIN R07 REVIEWED 20261027 -->
### R07 reviewed outcome — 27 October 2026

All matched negatives already enter S0. Fourfold reliable-negative exposure gives tiny fixed-cutoff F1 gains vs S0 (+0.000755/+0.000684), removing25/19 FP but losing3/1 TP; all-retained AP worsens in both seeds. Both remain below original A-xyz. Gate failed; no unfrozen/R08 launch.

Recovery smoke1624151 and GPU seeds1624152_0/1 completed; CPU1624153 was superseded solely for a verified float32 calibrated-threshold decoding bug, corrected CPU1624612 completed. Seven focused checks passed; no new GPU training during postprocessing recovery. Assignment/spatial audit and all source provenance remain in the isolated R07 folder.
<!-- END R07 REVIEWED 20261027 -->

<!-- R06 REPEATED REQUEST AUDIT 20261027 -->
### R06 renewed-request review — 27 October 2026

Exact requested training comparison already DONE — negative/mixed; no duplicate training or unfrozen confirmation. CNN features are used but final R2 AP/F1 remain below original R0; no convincing R1 advantage, early gains do not persist. Only missing common-v2 Branch/Curve metrics are being completed by CPU job 1625936, using existing R06 scores and original K8 curves. Checkpoint/sample/GT identity and direct baseline-geometry smoke passed. Existing N08/R07 negatives retained; no combined experiment launched. See the dated existing relation-learning report.

<!-- R06 GEOMETRY REPORTING COMPLETED 20261027 -->
R06 remains DONE — negative/mixed. Missing Branch/Curve metrics were completed using saved R06 scores and unchanged original K8 curves; no new inference/training, query selection, R07 combination or unfrozen confirmation. See the existing relation-learning report.


<!-- NATIVE2D IMPLEMENTATION 20261027 -->
## Native2D-01 — native 2D journal implementation, 27 October 2026

**DONE — measured (implementation and tiny memorization; reviewed 28 October 2026):** original-size GPU backward and1,000-update two-crop learning passed. Node/edge/Branch F1 reaches1.0 on each training crop; Curve F1 0.9789/1.0. This is not a full 2D generalization result. The earlier launch details below are historical; completion supersedes pending status. This is a new dimensional port, not a repetition of N06/N08/R06/R07. No full training or new detector losses. Original2D backbone/attention with modern focal/CE Hungarian supervision, optional A-xyz and endpoint-constrained K8 `graph_only`;3D implementations/checkpoints preserved. Data readiness: paired AV-WIDE graphs/images and REAVER manual mask/skeleton/raw exist; OCTA-Synth images are missing; true subject grouping and crossing policy remain training gates. [Existing report](image-to-graph-vessel.md#native-2d-journal-extension--27-october-2026). Jobs1637307/1637308 completed successfully on 28 October. See the dated completion review below. No full-training authorization is implied.


<!-- NATIVE2D ACCEPTED JOBS 20261027 -->
### Native 2D accepted bounded jobs — 27 October 2026

**SUBMITTED — verify:** GPU smoke **1637307**, one H100,20-minute allocation; original-size architecture backward plus1000-update memorization on two training crops. CPU collector **1637308**, `afterany:1637307`, appends the outcome automatically even if smoke fails. GPU dependency: `afterany:1637025:1637027:1637032:1637034:1637036:1637037:1637039:1637040:1637042:1637044:1637045:1637046`. These are existing BRAVA GPU chains, ending at1637046; the dependency preserves the global two-H100 limit and may delay the small2D check until that longer batch ends. No existing job was cancelled, no full2D training submitted.

Completed before submission: eight unit regressions,13 real-image CPU checks, two training/two validation development crops plus one REAVER alignment overlay. Final main-repository replay passed; existing3D implementation hashes remain unchanged. CUDA and tiny memorization results remain pending, not claimed passed.

Evidence: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261027`; large runtime/checkpoint output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/native2d_20261027`. Source/config hashes and archive are frozen in `code-identities.json` / `source.zip`. Read saved results with:

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/scripts/show_native2d.py
```



<!-- NATIVE2D REGISTRY COLLECTED 20261027 -->
### Native2D-01 smoke outcome — 27 October 2026

Correctness/tiny-fit gate passed; no full training. See the [existing 2D report](image-to-graph-vessel.md#native-2d-smoke-collection--27-october-2026) and `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261027/collection.json`.

### D08 completion review — 28 October 2026

Verified all 24 long-training tasks and six milestone collectors completed; 300 epochs/82,200 updates per arm, all paired input streams identical, all losses finite, and final checkpoint SHA256 values rechecked. Updated D08, the BRAVA matrix, run inventory, next actions and decision gate. The existing BRAVA report contains final train/val results, paired subject intervals, endpoint attribution and measured GPU costs. No new job or test evaluation was launched.


<!-- FINAL D08 AND NATIVE2D REVIEW 20261028 -->
## Reviewed completion and priorities — 28 October 2026

D08 and Native2D-01 are **DONE — measured** within their stated scopes. All 24 BRAVA production GPU tasks/six collectors and the2D smoke/collector completed with exit0. D08 shows controlled initialization gains but does not establish sufficient full-graph recovery: validation edge F1 0.06794, Branch F1 0.00647. Raw-to-filtered node coverage and endpoint availability remain major limitations; no further long run or geometry matrix is authorized. Recommend a cached training-only operating-point audit next, with unchanged validation application and explicit candidate populations. Do not repeat N08/R06/R07 automatically.

Native2D-01 passes implementation and tiny-set learnability only. The tiny ResNet18 result does not compare architectures or establish held-out accuracy; the original-width ResNet101 was checked with zero optimizer updates. Full 2D training remains **DEFERRED** until source recovery, biological grouping, scale and crossings are resolved. No test data were evaluated.

Detailed final interpretation: [BRAVA](../docs/research/BRAVA_experiment_plan.md#reviewed-final-d08-results--28-october-2026), [native2D](image-to-graph-vessel.md#native-2d-completion-review--28-october-2026). Review evidence: `experiments/completed_result_review_20261028/`. This update reruns no model and changes no checkpoint, target, decoder, metric definition or threshold.

---

<!-- BEGIN JOURNAL EIGHT WEEK PLAN 20261028 -->
## 10. Proposed order of work — eight-week journal closure plan (28 October 2026)

**Status: PARTLY EXECUTED — Q0/J00/J02 and modern IXI I01/D04/N07 completed 28 October; other steps retain their recorded status.** This section records user-requested research decisions and experiments; actual execution is tracked by the dated audit and job records. Deadline assumption: ~8 weeks to produce the journal submission; confirm actual journal date before applying the dates rigidly. All preceding D08/Native2D-01 results, N05/N06 oracles and N08/R06/R07 negative results stay unchanged; Q0 resolves the former D04 status caveat below. The objective is **one coherent extension**: biologically informed pretraining + reliable image-conditioned query/graph learning + continuous K8 reconstruction, evaluated on real and synthetic 3D and a controlled 2D cohort. Do not start additional independent novelty tracks.

### A. Non-negotiable diagnosis and already-completed experiments

- **Original Hungarian matcher is already one-to-one:** `models/matcher.py`, `HungarianMatcher.forward`, uses `linear_sum_assignment(5 * L1(pred_node, GT_node) - 2 * P(node))` (normalized coordinates, class probability); every GT is assigned at most once and every query at most once. No separate "GT visited" flag is needed. A close *duplicate* query may instead receive an unsuitable distinct GT assignment or be left background; Hungarian assignment is not physical localization. Do **not** relabel this existing one-to-one property as a new intervention.
- **Confidence is already part of matching**, and the GT-assisted confidence-preferred raw-query oracle N05/N06 was evaluated *at inference/diagnosis*, not as a new training matcher. N08 tested a different, GT-free postprocessing selection with negative/mixed outcomes. The novel *unverified* idea is a **spatially eligible, confidence-preferred, global one-to-one training assignment** (M01), not greedy GT-by-GT sorting and not a second GT match.
- **CORRECTION 28 October — query capacity and degree-2 nodes:** Modern SyntheticMRI uses 120 queries; BRAVA D07 used 120, while BRAVA Stage 1 and D08 used 256. Historical IXI used 256. Earlier IXI capacity logs (`docs/research/evidence/ixi_query_benchmark/RESULTS.md`, `ixi_capacity_192_256/README.md`) reported 59/884 patches above 120 (maximum 177) and 23 training patches above 192 (maximum 239), **but these benchmarks used an older graph population containing intermediate degree-2 samples**. They **cannot justify 256 queries for current continuous K8 training targets**. Verified modern code `data/spline_targets.py::contract_spline_graph` calls `metrics.branch_connectivity.contract_degree_two`; `ixi-transfer.md` says intermediate degree-2 samples remain in *curve guides*, not object targets. The final target still contains real junctions, terminals, crop-boundary anchors, and limited topology-preserving anchors. **Count the exact serialized nodes in the current IXI contracted train/val (and compatible dataset manifest) before deciding if 120 suffices.** If every required patch fits within 120, prioritize a clean 120-versus-256 IXI accuracy ablation; if not, report overflow identities and choose a population-preserving capacity. No silent truncation, filtering or target regeneration for just one arm. The historical counts document older-representation capacity only; they are not the modern maximum.
- **Q0/J02 measured follow-up, 28 October:** the entire current contracted IXI train/val cache has max153/141 anchors and5/5 patches above120; the modern4096/512 subsets have max153/124 and1/1 overflows. All audited object graphs have zero degree-2 nodes. The historical239 remains a different pre-contraction source, but removing it as justification does not make120 safe. Retain256; Q120 prerequisite fails.
- **BRAVA D08 (256 queries, full 300 epochs)**: SyntheticMRI initialization val node/edge F1 at 2mm 0.35572/0.06794 versus scratch 0.02544/0.00110; raw recall 0.65530, retained recall 0.28643; 96.69% missed GT edges lack a retained recovered endpoint pair, 89.40% false edges involve a spatially unmatched endpoint, conditional relation AP 0.60471. Pretraining helps, but insufficient *training-cohort* fitting and generalization remain. No repeating 300 epochs automatically. BRAVA annotation-relative errors due incomplete SWC; `seg/` is brain not vessel mask.
- **Modern SyntheticMRI:** B0/A-xyz controlled 12k, two seeds, 120 queries; A-xyz ~0.634 vs B0 ~0.595 2mm edge F1. Existing R06 image-rich relation embeddings, R07 matched-negative oversampling and N08 GT-free selection did not beat predeclared controls. Do not retest them as new ideas. K8 generally sufficient for IXI; BRAVA geometry is not the dominant current bottleneck.
- **IXI modern D04/N07 now DONE — measured,28October:** Q0 first established the failed infrastructure chain; its short-TMPDIR recovery then completed all six256-query arms and collector. A-cls has a measured edge/Curve gain with worse duplication/cycles; coordinate localization improves but the full graph remains weak. Historical failures stay in the dated audit.
- **Other historical studies to recover (not repeat):** IXI 120/192 and 192/256 *capacity/throughput* studies above; historical FGW matcher attempts and their checkpoint/timepoint outcomes; IXI node focal ablation and mask-input oracle. A capacity benchmark is not a converged graph-accuracy ablation.

### B. Experimental backlog — scientific hypotheses and stop gates

| ID / rank | Specific question | Minimal comparison and prerequisite | Primary evidence / decision gate | Status |
|---|---|---|---|---|
| J00 / P0 | What is already running or measured on modern IXI? | Completed read-only Slurm/artifact/source audit; no new GPU job. | No modern production result:1569366 failed,1583591 timed out on AF_UNIX path; downstream cancelled. D04 partial, N07 not run; L06 Synthetic half complete only. | DONE — measured (Q0) |
| J01 / P0 | Does BRAVA D08 mainly lose useful raw queries at confidence filtering? | CPU re-decode saved 300-epoch D08 source/pretrained caches; **select cutoffs on BRAVA train only**, carry unchanged to 8-subject validation. Same original model/candidates. | Full node/edge PR and physically matched counts, raw vs filtered recall, confidence distributions, duplicates, shortcut/cycle and TP/FP; do not advertise GT-assisted selection as deployable. | NOT RUN |
| J02 / P0 | Is120 sufficient for current contracted IXI and BRAVA GT? | All23,213 retained train/val VTP/NPZ targets verified with actual loader semantics and unchanged manifests. | IXI max153/141,5/5 overflow120; modern subsets1/1. BRAVA max64/69, no overflow. Zero degree-2 object vertices; no capacity exclusions. Retain256 IXI. | DONE — measured (Q0) |
| M01 / P1 | Can spatially gated confidence-first Hungarian improve query assignment stability? | Modern strong SyntheticMRI 120-query clean source, original matcher vs globally optimal **eligible-set** cost. Eligibility in true physical units, with no duplicate GT allocation, explicit fallback for no candidate, match cardinality and deterministic tie handling. Compare matched optimizer updates and initial states. | Assignment churn, eligible GT fractions, selected-query localization and confidence, duplicate/FN/FP, node/edge F1 and conditional AP on both seeds. Stop if only confidence or false-node count shifts with no graph benefit. | NOT RUN; bounded, lower novelty |
| N09 / P1 (final novelty candidate 1) | Can training-only **node and graph-aware query denoising** stabilize the correct representative and incident relations? | Start from **S0 clean SyntheticMRI epoch95 B0 (120 queries)** before experimental target continuation; two controlled seeds used in N01, NOT an already failed IXI/BRAVA head or late checkpoint. A-xyz module zero initialized in all relevant arms. Baseline `B0`/`A-xyz` continuation vs `A-xyz+node-only DN` vs `A-xyz+node+incident-edge DN`. Preserve ordinary Hungarian matching and original matched-edge CE for free queries, K8 `graph_only` detached geometry and same training patches/updates. No M01 matcher change. | First compare unperturbed free-query output parity, DN masking/no-GT-inference, gradients, recoverable 1/2mm nodes, assignment stability, correct/duplicate ranking, native edge F1, conditional AP, Branch/Curve, signed counts/cycles in two seeds. A DN loss change must beat **node-only DN** to justify graph-aware novelty. Stop after predefined short pilot if no free-query gain. | NOT RUN |
| N10 / P1 (final novelty candidate 2) | Can an **image-conditioned spatial node proposal** help recover missed junction/end/boundary anchors? | Audit 3D early CNN and transformer reference-point interface. Start with **training-time spatial heatmap** for annotated node types (using GT **only during training**) and a GT-free proposal readout; use original S0 B0 start. Compare B0; A-xyz; proposal-only; **proposal + A-xyz**, same initialization/updates/queries/K8/loss. If real proposal-fed queries are feasible, explicitly test coordinates and query embeddings while preventing trivial GT leak; do not assume a heatmap loss changes free-query inference. | Raw 1/2mm query recovery and false candidates first; then filtered node/edge F1 and cycles, with proposal precision, calibrated node counts, edge-support. Proposals must discover missing anchors beyond simply refining already-nearby queries; compare parameter/compute matched auxiliary-head control. Stop if raw coverage or graph F1 does not improve. | NOT RUN |
| Q120 / P1 conditional | Does120 outperform256 after degree-2 contraction? | Q0/J02 proves current full and modern-subset IXI overflow120; required condition is false. | No IXI120/256 training under unchanged populations. Do not exclude overflowing patches; retain256. BRAVA capacity-safe but not automatically authorized as replacement study. | DEFERRED — capacity gate failed |
| I01 / P0 | Can IXI match SyntheticMRI with the modern method? | Recovered modern six-arm256-query comparison completed;5,500 updates,one target seed. | A-cls improves IXI edge F1 .087926→.101536 and Curve .214892→.224389 but cycles worsen. This bounded study does not recover SyntheticMRI-level accuracy or isolate image-quality/backbone causes. | DONE — measured |
| T01 / P1 | Does botanical pretraining improve sample efficiency beyond vascular-only? | Matched subject-level low-label 3D budgets, modern baseline, vascular-only vs botanically mixed and ideally original roads control with equal source/target exposures, updates and seeds. D02 single-seed unequal source exposures is not sufficient. | Held-out node/edge/Branch/Curve + confidence intervals, source-domain mixing and compute disclosed. Stronger evidence necessary for GrEVe extension claim. | NOT RUN (controlled confirmation) |
| 2D01 / P0 | Establish native2D valid full datasets and first controlled training | Continue Native2D-01 (tiny-fit only) after audited images/graphs/patient grouping/scale/crossings. First train B0 and A-xyz on one sound dataset with matched seeds/updates/query capacities, compare straight vs K8 **on identical inferred graphs**. | Subject- or specimen-disjoint held-out metrics, native pixel/µm tolerances, graph topology and provenance. Start bounded, scale only after gate. | **DONE bounded Native2D-03 pilot**; negative fixed-cutoff reconstruction, no extension |
| E01 / P2 (only if residual) | Can direct-edge shortcut confusion be reduced? | Don't repeat R06/R07. On an adequate-node model, quantify false direct edges between two correctly matched *nonadjacent* vertices where an intermediate anchor exists; optional lightweight direct-adjacency structural loss **only if** substantial residual errors remain after N09/N10. | Same node candidates, conditional AP/precision/recall, missed positives, β1/cycle; stop if node recall still dominates or shortcut reduction fragments real branches. | DEFERRED |
| B01 / P3 | Does another 3D medical backbone solve the image-feature gap? | One lower-risk encoder control (MONAI SegResNet / nnU-Net-style residual encoder), **conditional on** I01 demonstrating feature bottleneck and sufficient time. Do not run a Swin/MedNeXt sweep. Preserve transformer/matcher/queries/K8 and audit feature pyramid/stride, A-xyz early features and normalization. | Matched training, memory/time and real-data node recall; parameter/compute effects and checkpoint compatibility disclosed. | DEFERRED |

### C. Exact design safeguards for the final two novelty candidates

**N09 graph-aware denoising**

1. Baseline *free object queries* remain the only inference graph nodes; use a separate set of **training-only denoising queries**, initialized from GT nodes with a physically meaningful random position perturbation. Include perturbed true GT node labels and, if required, carefully masked negative/false-anchor denoising cases. GT cannot enter the normal decoder branch or the inference path. Inspect the current transformer input/reference-point design before implementing, as standard DN-DETR code cannot be copied without verifying compatibility.
2. For the *node-only DN* ablation, predict recovered GT anchor coordinates and objectness for DN queries. For *graph-aware DN*, add incident-adjacency supervision **only among appropriately indexed DN anchor pairs**; use the same shared relation classifier where feasible. Do not supervise unrelated DN/free pairs as negatives. Apply explicit attention masks between DN groups and **DN→free and free→DN** to prohibit leaking perfect GT anchors into ordinary outputs. Check no inference dependence on DN or labels. If needed, share parameters but use a separate loss accumulator.
3. The normal free-query Hungarian matcher, class focal γ2, matched CE ratio .15, K8 graph-only geometry, augmentation and sampling stay **identical** to B0/A-xyz controls. Do not combine with M01 or proposals in the first causal study. Use the same S0 epoch95, same object embeddings and two documented N01 seeds. Train baseline and new arms for identical **target updates / target sample exposure**; extra DN tokens increase compute, record it.
4. Unit tests: original model output parity before DN activation, attention mask target-leak checks, no train-only tensor in eval mode, loss gradient flow to intended parameters, equivariance to permutation of DN GT node ordering / edge orientation, finite synthetic/real mini-batch, 8/16-patch memorization, exact node-count handling and edge orientation symmetry. Evaluate *unmodified normal decoder*.
5. Suggested **bounded** pilot: 2,000–4,000 target updates per arm, only 1–2 predeclared DN strengths (not broad hyperparameter search), fixed checkpoints; train-cohort selection then full original validation on promising arms. If gain consistent in both seeds, confirm on 12,000-update reference and then a small real-IXI transfer, never BRAVA first. Promote graph-aware innovation only if it improves over node-only DN and N01 A-xyz and does not inflate FP/cycles.

**N10 image-conditioned proposals with and without A-xyz**

1. Audit existing 3D early CNN feature stride and deformable-transformer reference initialization. Create an auxiliary **class-aware 3D node anchor heatmap** on training GT nodes (junction, endpoint, crop boundary), preserving label semantics. GT-derived segmentation/skeleton masks must **not** become model inputs at inference. Vessel foreground ≠ graph node.
2. First verify spatial proposal precision/recall on held-out patients and how proposals compete with ordinary free-query raw recovery. If injecting proposals into decoder requires replacing learned query/reference positions, test a precise GT-free inference path and matched dimensionality; no oracle GT proposal usage at validation.
3. Controlled factorial with a *shared S0 parent* and fixed 120-query SyntheticMRI reference: (i) B0, (ii) A-xyz, (iii) proposal, (iv) proposal+A-xyz. Zero-initialize any new correction/projection path and check common outputs initially. No graph-aware DN, custom matching, new relation head or backbone switch in this experiment. Keep total update/sample count fixed; record additional head/decoder compute.
4. Gate on **raw** recall at 1/2 mm and unique anchors first, then full graph metrics; a better heatmap alone is not sufficient. Compare gains in node coverage beyond A-xyz and whether combining yields additive or redundant effects. Run a short paired pilot, confirm only if promising, then IXI.

**M01 match-policy isolation**

- Global assignment is still **injective** (one-to-one). Candidate q is eligible for GT j only when a physical-position gate is satisfied. Optimize a lexicographic objective: maximize eligible GT coverage first, then confidence-aware cost within those assignments, with deterministic tie handling. Explicitly define what happens for GT vertices with no eligible query, as silently dropping their node/edge supervision may cause a contradictory objective. If the chosen policy uses a fallback, report fallback frequency and which labels remain valid.
- The first run must retain ordinary node classification supervision for unmatched queries and ordinary relation supervision; if duplicate-foreground alternatives are systematically penalized by the background loss, characterize that separately rather than adding a duplicate-ignore policy in the same arm.
- Compare fixed 120-query SyntheticMRI S0 fully trainable baseline vs M01 on two seeds and early/final checkpoints; fixed data and weights. Record the current exact Hungarian cost and pure-nearest / confidence-favored matching diagnostics. Don't present an N05 GT oracle as a deployable model.

### D. 2D dataset inventory and exact blockers (from `docs/research/image-to-graph-vessel.md`, audited 27–28 October)

| Dataset | Confirmed files/labels | What is missing before journal training | Next permitted action |
|---|---|---|---|
| **AV-WIDE** | **Per dataset README: each `.mat` includes original color image `I2` plus graph struct `G`** (verified for inspected images; `G.V/G.A` connectivity). 30 files locally; two same-image crops were memorized. **No separate image download required for AV-WIDE.** | Verify all 30 `I2`/`G` pairs, coordinates and paths; eye/patient grouping; image/subject-disjoint splits; spacing if available; ambiguous crossings/AV overlaps, clipped boundaries, label provenance and usable cohort size. | **First 2D retinal training candidate** after quick dataset-wide data/target QC, not after unrelated dataset recovery. |
| **REAVER** | 36 raw inputs in `_Original_Images` and manual-analysis output in **`ImageJ_Manual`**; manually derived mask/skeleton reported in the audit. Automated subfolders `AngioQuant_Auto`, `AngioTool_Auto`, `ImageJ_Auto`, `RAVE_Auto`, `REAVER_Auto` are algorithm outputs, **not GT**. Calibration 0.207194025 µm/px verified on one sample only. | Cross-check all 36 input ↔ `ImageJ_Manual` masks/skeletons; confirm exact meaning of TIFF color channels, sample identity/grouping, crossing/short-component policies, graph extraction overlays and per-image physical scale. | **Second 2D target**; do not replace `ImageJ_Manual` with automated outputs or claim the derived graph was independently annotated. |
| **OCTA-Synth** | 480/220/2000 train/val/test **graph directories only**. | **Raw input images and labels/masks absent** in current local copy, generator/version/split provenance; confirm complete correct image ↔ graph pairing before any training. | Locate/recover original image/mask generator outputs; no substitute dummy input and no test-data development. |
| **AV-DRIVE** | 20 train / 20 test `.mat` graph files. Dataset description groups all AV datasets as one MAT per image, **but explicitly confirms embedded `I2` only for AV-WIDE**; repository audit reported paired original images excluded for AV-DRIVE. | Read actual MAT variable keys for a representative train file before declaring images absent with certainty; if `I2`/other raw image absent, recover matching DRIVE images; verify correspondence, licenses and patient/eye grouping. | Not a blocker for beginning AV-WIDE. Never treat `.mat` graph presence alone as confirmed paired raw input. |
| **AV-INSPIRE** | 40 graph `.mat` files. Same dataset description confirms a MAT per image, **not necessarily an embedded color image**; earlier repository audit reports source images excluded. | Inspect MAT variable keys for embedded raw imagery; otherwise recover original paired images, then align/check all graphs and eye/patient IDs. | Later 2D dataset, independent of AV-WIDE/REAVER feasibility. |
| **RETA (`16960855`)** | Image/MAT/model/support RAR archives; README suggests 54 training / 27 test. | Archives not extracted/validated; precise labels, completeness and graph extraction provenance unknown; public test vessel masks reportedly unavailable. | Inspect archives and licenses/readme, audit train-only image/label pairs; lock test. No claim of test GT. |
| **HALVS** | 144 train / six validation source image-mask pairs, botanical source. | **Not a vessel target dataset**; plant identity/source grouping, mask→graph extraction validity, exposure/augmentation controls, independent source images needed for biological pretraining. | Use for botanical source only, not retinal vessel performance. |
| **20cities2D** | Prepared directories empty. | All required road source imagery/graphs missing locally. | Recover only if roads baseline is needed and legally accessible; otherwise label road control unavailable. |

Any additional user-held 2D datasets not in this inventory are **unknown**, not assumed ready. 2D pixels or microns must not be reported as 1/2 mm MRI accuracy; use 1/2px for the existing retinal development convention and predeclare µm tolerances for REAVER. Projected vascular crossings can create nonphysical graph junctions, so inspect both original graph topology and overlay before generating target connectivity. Image-only data cannot undergo IXI-style extraction without a trusted vessel mask/centerline or explicitly declared pseudo-labeling pipeline.

### E. Eight-week schedule and freeze gates (provisional dates)

| Timebox | Expected outputs and decision | Stop / deferral rule |
|---|---|---|
| **Week 1 (Oct 28–Nov 3)** | J00 modern IXI status; J01 BRAVA D08 cached operating point; **J02 verify current post-contraction node counts (old overflow counts invalid here)**; 2D full-file/patient ID and source recovery audit. Freeze core metric definitions and held-out test membership. | Do not spend H100 time redoing jobs or using invalid 120-query IXI targets. |
| **Week 2 (Nov 4–10)** | Resolve modern IXI B0/A-xyz baseline and whether focal-versus-CE is confounded; run M01 short matched Hungarian comparison *only if* capacity and cache audit warrant; prepare first 2D subject-safe B0/A-xyz train manifests. | If M01 fails predefined gate, freeze original matcher immediately. |
| **Week 3 (Nov 11–17)** | N09 node-only DN and graph-DN correctness/masking tests, then bounded two-seed SyntheticMRI pilots. Begin full 2D B0 reference on AV-WIDE or REAVER if readiness gates passed. | If no free-query raw recall/edge F1 benefit, stop DN; no transfer to BRAVA. |
| **Week 4 (Nov 18–24)** | Confirm N09 only if pilot passes; prepare/run N10 proposal-only and proposal+A-xyz paired short pilot. Summarize 2D B0/A-xyz held-out outcomes. | No new matcher, relation head or backbone added to these causal experiments. |
| **Week 5 (Nov 25–Dec 1)** | Select **one** DN or proposal method for journal (or retain A-xyz if both fail), verify 3D SyntheticMRI + IXI transfer. Finish one 2D controlled dataset, preferably second if available and annotation-safe. | Research-design freeze after this week: no new independent architectural novelties. |
| **Week 6 (Dec 2–8)** | Controlled botanical transfer / target annotation efficiency (T01), priority on a few strong comparisons over an extensive matrix; finalize primary 3D and 2D ablations and conventional pipeline comparisons if data ready. | Do not claim botanical superiority from D02 alone; if new comparison fails, report the limitation and preserve GrEVe findings honestly. |
| **Week 7 (Dec 9–15)** | Lock experiments; prepare complete result tables, per-subject uncertainty, visualizations, topology/coverage diagnostics, limitations and reproducibility records. Draft full article, compare explicitly to GrEVe. | No repeated validation-based architecture selection, no test evaluation before protocol freeze. |
| **Week 8 (Dec 16–23)** | Reproduce selected plots/tables, consistency and leakage review, supervisor feedback, manuscript editing, reference and artifact cleanup. Optionally **one** final locked test-set evaluation for the selected model(s) according to journal protocol. | No new novelty chase. Reserve Dec 24–28 as optional buffer if submission date allows. |

**Minimum viable journal package (even if new methods fail):** (1) verified modern 3D Baseline/A-xyz plus matched physical topology metrics; (2) continuous K8 versus straight on identical node/edge sets; (3) controlled biological pretraining/label efficiency; (4) at least one valid 2D dataset experiment if annotated-data gates pass; (5) transparent real-data IXI and BRAVA generalization limitations; (6) suitable segmentation→skeleton→graph conventional baseline where mask availability allows, or explicitly document why unavailable; (7) subject-disjoint confidence/uncertainty and compute. BRAVA must **not** be presented as recovered under D08.

**Resource and provenance contract for every new run:** Read §7 checklist; verify `sacct`, `squeue`, worktrees and registry, no more than **two concurrent H100s globally**, reuse fixed source/checkpoint SHA, deterministic target patch IDs and unchanged splits, source vs target optimization exposure, strict config/model state compatibility when changing query count or backbone, CPU correctness → two-step GPU smoke → pilot → fixed collector. Record failures as failures, use unique experiment IDs and write outcomes into domain report and this registry. Large caches/checkpoints on SCRATCH; never overwrite older runs or inspect test data during development. Separate predeclared primary metrics from secondary/oracle diagnostics, count only independent subjects in confidence intervals and report multiple-seed uncertainty when claiming method gains.

**Stop-adding-ideas rule:** By Week 5, choose **at most one** new query-learning contribution (N09 DN *or* N10 proposals/proposal+A-xyz) to confirm; M01 matcher and Q120 are controls, not additional independent novel contributions. If neither works, retain the established A-xyz method and build the journal extension around controlled pretraining, dimensional transfer, continuous geometry and rigorous real-data evaluation. A journal extension should not depend on every experimental hypothesis succeeding.

<!-- END JOURNAL EIGHT WEEK PLAN 20261028 -->


---

## 11. Executable Codex sequence — first requests to send (revision 28 October 2026)

**Purpose:** This is the **actionable order** superseding the provisional sequencing in §10E wherever they differ. Historical completed results in §§0–9 remain authoritative. Q0/J00+J02 and the recovered modern IXI I01/D04/N07 study are **DONE — measured** (28 October2026); Q120 remains **DEFERRED** because current targets overflow120. Other steps remain proposed/conditional unless an existing completed result is explicitly cited. Each request has a bounded deliverable and exit gate. Start with the first request, **do not send all prompts to independent agents simultaneously**, and check existing live worktrees/jobs before execution. CPU dataset preparation may run in parallel with GPU work once ownership is assigned. Do not submit duplicate IXI jobs or more than two total H100 jobs concurrently.

### Exact priority / dependency order

| Order | ID | Codex task | Compute / prerequisites | Stop or promotion gate | Status |
|---:|---|---|---|---|---|
| **1** | `J00+J02` | **Q0 complete: IXI status and actual contracted GT audit.** | CPU1653086 complete; full23,213 targets and source hashes checked; no GPU submission. | No modern IXI training result;120 not capacity-safe (153/141 full maxima). Keep256 and recover smoke separately. | DONE — measured,28 Oct |
| **2** | `2D-QC` | **Audit all AV-WIDE and REAVER image/annotation pairs; prepare disjoint manifests/overlays**. Verify AV-DRIVE/INSPIRE MAT keys but do not wait for other image recovery. | CPU/data only, may overlap step 1 with isolated ownership. | Start 2D experiments using **AV-WIDE `I2`+`G`** as soon as target/identity checks pass; REAVER `ImageJ_Manual` separately. No undocumented crossing merging. | **DONE** source audit; corrected AV-WIDE fit; REAVER diaphragm targets prepared |
| **3** | `J01` | **BRAVA D08 cached confidence/endpoint audit**. | CPU only; D08 final stored checkpoints and predictions. | Training-derived fixed node/edge operating points transferred once to val; record raw/filtered/duplicate/FP/β errors. No model retrain. | NOT RUN |
| **4** | `I01` | **Modern IXI B0/A-xyz/A-cls comparison completed.** |1653860_0–5 and collector1653861 complete;5,500 updates,256 queries,unchanged sources/subjects/targets. | A-cls improves physical-edge/Curve with a duplicate/cycle cost; one seed,weak absolute graphs. No automatic longer run. | DONE — measured28Oct |
| **5** | `Q120` | **120-vs-256 IXI accuracy study deferred.** | Q0 verified overflow in both full and modern-study contracted targets. | Do not run an unequal/filtered120 arm or alter targets. Keep256; no automatic BRAVA substitute. | DEFERRED — prerequisite failed |
| **6** | `2D-B0` | **First genuine 2D training**: AV-WIDE B0 reference on validated image/graph pairs; compare optional A-xyz under identical protocol. | GPU only after step 2; benchmark full 2D architecture (tiny ResNet18 was merely smoke); use exact subject or image grouping available. | Complete image-disjoint validation with transparently limited patient independence if IDs unavailable. No test-set selection. Report B0/A-xyz and same-graph chords vs K8. | **DONE** corrected full-model B0/A-xyz6k each,506 held-out tiles; F1=0 at0.5 |
| **7** | `M01` | **Short gated confidence-aware Hungarian matcher ablation**, separate from DN/proposals. | One short controlled modern SyntheticMRI training study after J00; no proposal/DN changes. | Global 1:1 matching with physical eligibility, no GT dropped without defined fallback. Require real free-query/edge benefit; **stop early if neutral**. | NOT RUN |
| **8** | `N09` | **Final novelty candidate A: graph-aware query denoising**, with node-only DN control. | Short two-seed modern SyntheticMRI study; start *both* controls and DN arms from clean S0 epoch-95, 120 queries, matching raw training conditions. **Do not inherit M01 matcher.** | Verify no GT leakage, parity before training, and free-query improvement over both A-xyz and node-only DN. Only if gate passes: 12k-length matched confirmation and IXI transfer. | NOT RUN |
| **9** | `N10` | **Final novelty candidate B: image-conditioned proposals** and **proposal+A-xyz**. | Two-seed bounded modern SyntheticMRI; same clean S0 initialization, original Hungarian and fixed K8; separate from N09/M01. | Measure **new** recoverable GT nodes (not only refined existing ones), native edge gains, FP/β and compute; stop if negative. Do **not** combine with N09 unless both independently pass and time remains. | NOT RUN |
| **10** | `2D-REAVER` | **Independent 2D real-image experiment** with `ImageJ_Manual` annotations; optional broader dataset recovery. | After step 2 dataset-specific QC; safe with GPU scheduling; 1/2 pixel only if defined, or documented µm units. | Independent subject/specimen if known, otherwise declare image-disjoint; correct manual graph provenance. | **PREPARED**, separate micron cohort; GPU training not run |
| **11** | `T01` | **Biological pretraining / scarce target labels** under matched source and target budgets, using strongest frozen selected method (no new head sweep). | After 3D node method selected; compare botanical vs vascular-only and original road reference only when compatible. | Matched label availability (at **subject** level), patch exposures, total updates and multiple seeds; preserve negative results. | NOT RUN |
| **12** | `J-FREEZE` | **Freeze journal experimental story and write it.** Include conventional mask→skeleton→graph baseline where annotations permit, locked 2D/3D results, uncertainty, topology and same-graph geometry, quality controlled figures. | Parallel writing can start immediately, formal model freeze by **December 1** if deadline near late December. | No new novelty after freeze, no unreported validation peeking; final locked test exactly once per predeclared model. | NOT RUN |

**Deliberately deferred:** more relation-head/negative-mining architectures (R05–R10 already narrow options), C1/K24/adaptive geometry sweeps (weak detector), multiple medical backbones, repeated 300-epoch BRAVA scratch, 2D source recovery for every dataset before training AV-WIDE, and unconstrained combinations of matcher + DN + proposals. `B01` is allowed **only** if the modern IXI study demonstrates a feature-domain limitation after preprocessing/target audit and the freeze deadline permits. SegResNet and nnU-Net ResEnc are **not established improvements over SE-ResNet for this graph task**; evidence comes from segmentation, which is a different objective.

### Codex request 1 — send first: current IXI status + *post-contraction* capacity (CPU-first)

```text
Audit the live Graph-Native-Betti-Matching-Representation project and the attached experiment registry. Read §10A and §11, `ixi-transfer.md`, `data/spline_targets.py`, and the historical IXI capacity reports before acting. DO NOT train or submit GPU jobs.

A. Resolve the actual Slurm/checkpoint/collector status of modern fully unfrozen IXI B0, A-xyz and A-cls (D04/N07) and the incomplete BN control (L06). Do not duplicate a submitted, running or completed arm.
B. Count nodes in the EXACT serialized model GT for current contracted K8 IXI train/val (and BRAVA reference), not pre-contraction `adaptive`/dense VTP nodes. Verify degree-2 contraction, isolated/path-boundary anchors, graph construction version, patch membership, any capacity-based exclusions, max/p95/p99 and per-subject counts, and how many retained current patches exceed 120/192/256. Recover historical 239-node provenance and explicitly label it PRE-CONTRACTION if confirmed.
C. Verify that all graphs presented to a prospective 120-query arm are representable; do not change labels or silently drop hard patches. For existing 256-vs-120 source transfer, document the learned-query copy and the added-query initialization.
D. Update the existing registry with an evidence-backed one-page decision: 120 safe yes/no for current IXI, completed IXI model results yes/no, and one concrete permitted next run. Include exact paths/checkpoint hashes, source/target split and job IDs.
Return concise findings. No new architecture, optimizer, dataset rewrite or model training.
```

### Codex request 2 — send independently: 2D data audit and first training gate (CPU first)

```text
Read the latest journal registry §10D/§11 and `image-to-graph-vessel.md`, then inspect `/datasets` directly. Do a read-only, dataset-wide pairing and annotation audit, prioritizing AV-WIDE and REAVER. Do not launch full training until the gates pass.

AV-WIDE: the `.mat` contains BOTH `I2` (original color image) and `G` (graph); do not waste effort searching for external AV-WIDE photographs. Validate all 30 `I2/G.V/G.A` pairings, coordinate orientation, paths, crossings, duplicate and crop-boundary nodes; inspect real overlays and any available patient/eye IDs. REAVER: use `_Original_Images` as input and `ImageJ_Manual` as manual-analysis annotation; inspect R/G/B channel semantics and extraction on ALL 36 samples. Do not call any `*_Auto` algorithm outputs independent GT. Verify calibration per image and specimen grouping. Check MAT variable keys in AV-DRIVE/AV-INSPIRE before declaring whether they embed images; missing paired images should NOT block AV-WIDE.

Produce fixed train/validation image-grouped manifests, QC report, source-provenance and crossing decisions. If patient identities are unavailable, label the split image-disjoint, not patient-disjoint. Benchmark full original-width 2D training throughput after QC and propose a small B0 vs A-xyz experiment with same source/optimizer/queries/graph targets, modern focal+CE, K8 `graph_only`, fixed subject-safe checkpoints. Include one production-size GPU smoke BEFORE long runs. Preserve all 3D code and prior 2D smoke files. Report readiness by dataset, no silent pseudo-GT.
```

### Codex request 3 — BRAVA D08 no-training operating-point diagnosis

```text
Read D08 final BRAVA report and journal registry. Reuse archived 300-epoch scratch/SyntheticMRI checkpoints and raw-query/relation caches. No GPU training, no data regeneration, no test evaluation.

Using only BRAVA training subjects, preselect a limited node/edge confidence grid and/or a calibrated ranking policy, with fixed false-positive-count and node-count comparisons. Apply its selected operating point unchanged to the eight validation subjects. Distinguish (i) raw query not near any GT, (ii) recoverable node rejected by threshold, (iii) spatial duplicates, (iv) missing endpoint pairs, and (v) wrong relations conditional on recovered endpoints; stratify by terminal, junction, crop-boundary and subject. Report 1/2mm F1, raw/retained recall, TP/FP/FN, conditional AP/recall, matched-distance, branch/curve, signed node/edge and Betti biases, annotation caveats and uncertainty. Compare ordinary inference to GT-assisted upper bounds ONLY as diagnostics. Do not propose another long training run before delivering this report. Update registry with measured status and decision.
```

### Codex request 4 — contingent 120/256 modern IXI training pilot

```text
ONLY if §11 request 1 confirmed all current IXI contracted target graphs fit <=120 anchors and modern IXI status was reconciled: prepare a SMALL paired full-detector 120-vs-256 query study on identical IXI training/validation subjects, graph caches and patches. Use the strongest compatible source-only SyntheticMRI B0 initialization (copy its 120 learned object embeddings unchanged; for 256 add exactly 136 documented seeded rows, relocating learned relation tokens correctly); identical other shared parameters, K8 graph_only, current matching and graph objectives, augmentation, optimizer, LR, exposure in patches and checkpoints. Do not filter high-node patches or change targets. First compare throughput/peak memory and run a correct GPU smoke. A bounded matched pilot precedes any long study. Measure raw/filtered node recall, node/edge PR/F1, duplicate count, assignment churn, calibration, full graph topology, Branch/Curve and subject uncertainty. Compare against existing 256 checkpoint carefully: nonidentical training histories are descriptive, not a clean causal arm. If 120 loses important coverage, retain 256. Record all runs in the existing registry; no test evaluation.
```

### Codex request 5 — independent spatially gated Hungarian (low-priority control)

```text
First confirm the original Hungarian already enforces global one-to-one GT/query assignment and includes node confidence: 5*L1 - 2*P(node). Review prior N05/N06 oracle and FGW outcomes. Implement only a physically gated, globally one-to-one, confidence-preferred assignment as an ablation; do NOT implement greedy per-GT matching or claim already-tested confidence inclusion is new. Define eligible radius in physical mm, preserve valid match cardinality where feasible, explicitly define fallback for no eligible candidate, avoid dropping supervision, preserve original background/relation labels and matched-pair budget. Compare to the ORIGINAL matcher from exactly the clean SyntheticMRI source epoch95, matched full-model train steps and two seeds, same K8 and focal+CE. No A-xyz/DN/proposals changes in this experiment. First CPU toy adversarial tests (close GT nodes and duplicated queries), GPU smoke, then short fixed-budget pilot. Stop if no real gain on free-query node/edge performance and duplicate/assignment stability beyond matched-confidence controls. Append measured or negative outcomes; no automatic long run.
```

### Codex request 6 — graph-aware denoising (last high-priority novelty)

```text
Review S0/N01/N05/N06/N08 and the latest journal registry. Implement a bounded training-only query-denoising study, starting ALL arms from the clean SyntheticMRI K8 epoch95 model (120 object queries), BEFORE modern A-xyz/B0 continuation; keep two matched seeds and original Hungarian (NOT the M01 experimental matcher). Baselines: B0 and zero-initialized A-xyz. Treatments: (1) A-xyz + node-only DN, (2) A-xyz + node-and-incident-graph DN. The regular/free queries keep unchanged node focal gamma2, matched edge CE ratio .15 and detached K8 geometry; auxiliary perturbed GT queries have separately normalized denoising objectives. Define masking so no auxiliary GT queries/coordinates/edges leak to normal decoder predictions. Their labels must never be used at inference. Test initial free-query output parity, mask/no-GT-inference, distinct-nearby-GT conflict, correct positive/negative incident pairs, gradient routes and loss balance. Use fixed short pilots first, then extend ONLY if regular/free-query raw recovery and native graph metrics improve over BOTH A-xyz and node-only DN on two seeds without severe FP/cycle cost. Report assignment churn, confidence-versus-duplicate ranking, 1/2mm node/edge, conditional AP, Branch/Curve, signed β counts and compute. Do not introduce M01 matcher or image proposals in these arms. Keep all results, including negatives, in the registry.
```

### Codex request 7 — image-conditioned proposals and proposal+A-xyz (independent)

```text
Read N01, N06, N08 and the journal registry. Independently of DN and M01, test training-only image-conditioned node proposal learning from the SAME clean SyntheticMRI 120-query epoch95 source, using unchanged matching, focal+CE, K8 graph_only, matched data/updates/seeds. Compare B0, A-xyz, proposal-only, and proposal+A-xyz. Begin with a small spatial node-likelihood heatmap from early CNN features, explicit supervision of junction/terminal/crop-boundary anchor types, and a GT-free inference mechanism that can place/propose queries (not merely a disconnected auxiliary heatmap). Preserve the known query count and account for additional parameters and compute. Strictly forbid GT-derived coordinates at inference; test proposal-to-decoder gradient and duplicates when several proposals neighbor a junction. Run correctness tests and a bounded pilot; prioritize recovery of GT anchors that B0/A-xyz have NO nearby raw query. Promote only if it improves raw unique-recovered-node recall and native edge F1 without uncontrolled false nodes/cycles on two seeds. Do not combine with denoising unless both independently succeed and deadline permits.
```

### Practical scheduling and stop rule

- **Request 1 completed as Q0/J00/J02 on 28 October. Next: request 2 (2D-QC)** using isolated outputs and distinct report ownership; do not repeat the status/capacity audit. Send request 3 in a CPU slot after checking current BRAVA cache-analysis ownership. Request 4 is **deferred**, because Q0 found current contracted IXI targets above120. Request 5 is **lower priority than launching verified 2D B0 and the IXI reference**; cut it if time slips.
- After requests 1–3, **start first 2D B0/A-xyz held-out study** as soon as AV-WIDE QC passes. It is not dependent on OCTA-Synth, AV-DRIVE or AV-INSPIRE image recovery; REAVER can follow after manual annotation QC. The 2D original-width architecture only passed a one-step smoke, so benchmark it instead of extrapolating tiny-model timings.
- Run request 6 before request 7, both from the SAME clean S0 parent and with the ORIGINAL matcher. If DN has no short-pilot free-query benefit, stop without 12k extension. Run the proposal experiment regardless only if the research-freeze deadline leaves room. Do not use a poorly performing IXI/BRAVA checkpoint to establish either method's mechanism.
- **By approximately 1 December 2026**, retain **at most one** new novelty candidate to confirm on IXI and optionally 2D; or retain A-xyz if neither works. Reserve December principally for robust botanical/source controls, real-data evaluation, figures, and writing. Aim for one coherent graph-native extension rather than numerous unverified modules.
- Full method validation must use independent subject-level units when known, paired seeds, physically appropriate tolerances and one-to-one recovered-node/edge criteria. For BRAVA, unavailable SWC vessels are not proven false anatomy. For the 2D projected retinal vasculature, close 2D line crossings need not imply anatomical junctions. Do not report pixel vs millimeter thresholds interchangeably.

<!-- END CODEX EXECUTABLE PLAN 20261028 -->

<!-- N07 STATUS AUDIT 20261028 -->
## J00/J02 completed — N07 execution and current target counts, 28 October 2026

N07/D04 is **prepared, not scientifically tested**: initial smoke1569366 FAILED, replacement1583591 TIMEOUT after30m04s, all scientific dependents CANCELLED before execution. Root blocker is the overlong multiprocessing TMPDIR socket path; the exact exception was reproduced on CPU. No target-domain training budget was finalized and no modern A-cls efficacy result exists. L06 remains partial (IXI failed empty-pair dtype indexing, SyntheticMRI artifact exists, joint collector cancelled). No GPU submission or runtime/model changes in this audit.

| Population | Patches / subjects | p95 / p99 nodes | Max nodes | >120 / >192 / >256 |
|---|---:|---:|---:|---:|
| IXI train | 10,000 / 123 | 62.00 / 86.00 | 153 | 5 / 0 / 0 |
| IXI val | 2,649 / 26 | 66.00 / 87.52 | 141 | 5 / 0 / 0 |
| BRAVA train | 8,768 / 39 | 38.00 / 47.00 | 64 | 0 / 0 / 0 |
| BRAVA val | 1,796 / 8 | 38.00 / 51.00 | 69 | 0 / 0 / 0 |

These are **actual serialized post-contraction nodes**, not old VTP counts; zero degree2 objects, exact manifest/index agreement and disjoint subjects verified. N07's fixed4096/512 subset still has one153-node training and one124-node validation target. Thus **120 is not safe for unchanged IXI**, independently of the obsolete239-node capacity argument. Keep256 for recovery; BRAVA development targets fit120. No object-capacity exclusions are recorded in these current manifests; existing IXI parallel/closed-branch exclusions remain disclosed.

One next bounded action: repair short SCRATCH multiprocessing TMPDIR and rerun the existing correctness gate, then recover the matched256-query B0/A-xyz/A-cls study if it passes. No new architecture/loss or120-query overflow filtering. Detailed chronology, initializer/query-copy convention and verified hashes are in the [existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#n07d04-execution-and-current-target-capacity-audit--28-october-2026). Artifacts: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/n07_status_audit_20261028`.


<!-- BEGIN Q0 REGISTRY REVIEW 20261028 -->
## Q0 completion decision — 28 October 2026

**J00 and J02 DONE — measured.** Read-only status/source audit plus CPU1653086 (completed exit0,2m48s), no GPU/model training/test evaluation. IXI modern B0/A-xyz/A-cls production training was cancelled after smoke failure/timeout; D04 remains PARTIAL / RECOVERY, N07 is NOT RUN. L06 remains partial, preserving its completed SyntheticMRI half (2mm node F1 .752528→.756058; edge F1 .445495→.439856). No inference or recalibration was repeated.

Current exact contracted IXI full train/val targets:10,000/2,649 patches,123/26 subjects; max153/141, p95 62/66, p99 86/87.52, five/five over120 and zero over192/256. The modern4096/512 subsets include one overflow each (153/124). All object graphs contain zero degree-2 nodes. BRAVA train/val max64/69 and no120 overflow. The old239 count belongs to the archived pre-contraction population; its source VTP is unavailable for recount. Current decisions use freshly verified loader VTP/NPZ contents, not that old count. No capacity filtering or target regeneration was introduced.

**Decision:**120 safe for current IXI **no**; completed modern IXI efficacy results **no**. Keep256, defer Q120. One permitted next GPU job is the existing256-query correctness/benchmark smoke recovered with a short TMPDIR; no production training until its gates pass. Next independent queue task:2D-QC. Source hashes, exact overflow IDs/per-subject counts, seed262017 added-query initialization, job logs and manifests are in the [existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#q0-ixi-status-and-contracted-target-capacity-audit--28-october-2026) and [audit pointer](../experiments/q0_ixi_audit_20261028/worktree.json). Completed/negative N08/R06/R07 and D08/Native2D results are unchanged.
<!-- END Q0 REGISTRY REVIEW 20261028 -->

<!-- N07 SOCKET RECOVERY 20261028 -->
### N07/D04 recovery accepted — 28 October 2026

User explicitly authorized resubmitting the existing batch after the CPU audit. Fixed the short SCRATCH multiprocessing TMPDIR **after env.sh**; package staging remains unchanged. Five focused CPU tests, including reproducing then resolving actual FD-sharing failure, passed. No model/data/loss/threshold/update-budget changes.

Accepted chain **1653858** smoke(afterany:1653554,30min) → **1653859** zero-shot(afterok,75min) → **1653860_0–5** six-arm training(afterok,array%1,3h each) → **1653861** CPU collection(afterok,2h). One H100 for this study; two globally. Same4096/512 IXI patches,123/26 subjects,256-query fine-tuning; final common1500–6000 update count remains benchmark-gated. No scientific result claimed.

Prior job records/error artifacts preserved. See [existing report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#user-authorized-socket-path-recovery--28-october-2026), `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/ixi_local_components_20261026/socket-recovery-20261028/jobs.json` and source/test verification. No additional experiments submitted. Retrieval: `python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/scripts/show_ixi_local_components.py`.

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

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261028`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D HELDOUT LIVE 20261028 -->

<!-- BEGIN IXI RECOVERY VERIFIED SMOKE 20261028 -->
### Existing IXI recovery jobs verified; smoke completed — 28 October 2026

At `2026-10-28T10:05:01.360009+00:00`, recovered smoke **1653858** verified COMPLETED0:0 for all six arms; five startup tests pass. Original benchmark rule selects **5,500 updates**, fixed checkpoints1,500/3,000/5,500. Same256-query fine-tuning, sources, expansion seed262017, subjects, augmentation, matcher, losses, K8 and batch1 evaluation; no scientific changes. Existing zero-shot1653859 is **RUNNING**, training array1653860 **PENDING**, collector1653861 **PENDING**. Training is gated on successful zero-shot and uses oneH100 (%1); separately serialized2D work fits the global two-H100 limit. **No new/duplicate jobs were submitted during this check. Scientific efficacy and final input-stream parity remain pending.** See the [existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#recovered-smoke-passed-existing-jobs-verified--28-october-2026) and saved existing-chain audit.

Zero-shot prerequisite1653859 subsequently completed exit0:0 (7m55s), verified against all three source completion artifacts. Training1653860 is eligible/pending priority; collector1653861 remains dependent. No duplicate submission.
<!-- END IXI RECOVERY VERIFIED SMOKE 20261028 -->

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

Evidence and checkpoints: `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_20261028_targetfix`. [Protocol and dataset readiness](../docs/research/native2d_heldout_20261028.md). Learning curves: `learning-curves.png`; full metrics: `B0/best-heldout.json`, `Axyz/best-heldout.json`; final metrics: `<arm>/validation-<budget>.json`; weights/optimizer: `<arm>/best.pt`, `<arm>/latest.pt`. Failed gates and jobs are reported without automatic resubmission.
<!-- END NATIVE2D TARGETFIT LIVE 20261028 -->

<!-- BEGIN IXI FINAL REVIEW 20261028 -->
## Modern IXI N07/D04 collected outcome — 28 October 2026

**DONE — measured.** Six5,500-update tasks1653860_0–5 and CPU collector1653861 completed exit0. Identical augmented image/target hashes, source initialization,256-query expansion and subject-disjoint4096/512 membership verified; six final checkpoint SHA256 values recorded. One target seed261026;26 validation subjects; no test evaluation. Source epoch/12k continuation identities and all failed smoke evidence preserved.

A-cls is the physical-edge/Curve leader:2mm edge F1 0.101536 vs B0 0.087926, difference+0.013610 [0.008265,0.019401]; B0 node-count-matched difference+0.010862 [0.005577,0.016087]. Curve F1 .214892→.224389. It adds127TP and222FP edges, duplicates313→535 and cycle MAE1.998→2.580; no topology cure. Raw versus retained recall remains .686613/.302168;93.02% of missed edges lack recovered retained endpoints. Fresh/transferred xyz improve1mm localization and raw coverage but default2mm edge differences include zero. Transferred-minus-fresh and combined-A benefits are not established. Zero-image control is near B0. The whole study remains a bounded weak-graph pilot; independent-seed confirmation is needed before stronger promotion.

Current recovery uses3.7942 allocated H100-hours including zero-shot/smoke;4.3247 including the two earlier failed attempts, excluding source pretraining/CPU. No further training, inference or combined experiment launched. L06 remains separately partial. Full definitions, matched controls, trajectories, image interventions, checkpoint identities and limitations: [existing IXI report](../docs/research/metrics/3-ixi-node-refinement-pilot.md#reviewed-final-modern-ixi-results--28-october-2026).
<!-- END IXI FINAL REVIEW 20261028 -->
