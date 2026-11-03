# Vascular Graph Extraction — Experiment Registry & Research Decision Log

**Snapshot:** 26 October 2026 — revised after the evening N06/D06 reports  
**Project:** `Graph-Native-Betti-Matching-Representation` (and explicitly named isolated worktrees)  
**Purpose:** one compact, durable index of **what was tested, what the tests actually established, what was merely planned/submitted, and what should be tried next**. Read this *before* writing or launching a new Codex experiment prompt.  
**Scope:** RelationFormer graph extraction, node/query selection, relation learning, continuous-edge representations, cross-domain transfer, and evaluation. Historical experiments are indexed rather than exhaustively reproduced. Detailed reports remain authoritative.

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

- **Best controlled modern SyntheticMRI direction:** local image-conditioned **coordinate-only** node refinement (`A-xyz`), with two-seed improvement over continued baseline B0; **not a demonstrated topology or query-selection solution**.
- **Largest measured SyntheticMRI opportunity:** recover/select the right node-query representatives. Modern A-xyz edge F1 is approximately **0.634** at normal inference, while GT-assisted confidence-preferred query selection yields **0.834–0.838**. **N06 is now DONE:** about one-third of recovered GT vertices switch representatives; confidence-preferred alternatives are slightly less localized but have better same-neighbor relation ranking, strongly associated with training-style assignment. The oracle is **not deployable**.
- **Relation-image features:** **already tested on an older frozen IXI detector** as endpoint/chord/corridor residual heads. Early AP improvements were real on reused validation subjects; long continuation overfit. A modern relation-specific **552D token-enrichment** design is **not yet tested**.
- **BRAVA:** target geometry has real long-curve capacity issues, but the fully trainable detector currently reconstructs very few accurate connections. The representative K8 monitor **also fails on optimized-on training patches**; saved BN performs best among tested BN modes and augmentation alignment tests pass. **D06 is now DONE (diagnostic, not complete causal explanation)**. Next isolated test is **augmentation-off versus augmentation-on** with matched K8 training, not another K24/C1/path sweep.
- **Pending infrastructure:** the training-only BN experiment is **partial** (IXI GPU task failed; SyntheticMRI GPU task completed; collector cancelled). Modern IXI fully unfrozen B0/A-xyz/A-cls execution status is **not verified** from the supplied reports.

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
| **N06** | Paired nearest-versus-confidence oracle, controlled query swaps, relation ranking, small exact-token intervention, two SyntheticMRI seeds | **DONE — measured diagnostic** | ~33% of recovered anchors select different representatives; alternatives are ~0.38 mm farther yet better supported on synthetic vessel masks. Fixed-neighbor conditional AP improves ~0.873→0.959, beyond scalar recalibration. Confidence choices agree more with Hungarian assignments, but causality from training history is **unproven**. A small off-manifold token intervention suggests relation-relevant embedding content beyond the classifier direction. | `6-ixi-query-attribution(1).md`, Oct26 oracle addendum |
| **N07** | Pure image-aware node classification `A-cls` on modern IXI or SyntheticMRI | **STATUS UNKNOWN / not confirmed completed** | Designed/discussed; a modern IXI agent was reportedly handling it, but a Slurm/collector audit is required. **Do not confuse with A (combined logits+coordinates), N03 or legacy IXI node-head continuation.** | Pending IXI audit **P1** |
| **N08** | **GT-free relation-aware representative selection / uniqueness** on frozen modern A-xyz queries and original pair logits | **NOT RUN — NEXT** | N06 demonstrates distinct information content for eligible representatives. Confidence-only NMS gave only +0.003–0.004 edge F1 and lost ~1.4pp node recall. Test conservative inference-valid relation consistency against original, NMS and matched-node-count controls; beware self-reinforcing false loops/shortcuts. | Sol next study **Q1** |

### RELATION / GRAPH

| ID | Experiment & protocol | Status | What we learned / what we did **not** learn | Main evidence |
|---|---|---|---|---|
| **R01** | Legacy IXI full-query attribution, matched/unmatched/duplicate edge categories | **DONE — measured** | **668/733 (91.1%)** false edges touched spatially unmatched nodes; only 262/733 touched duplicates. Recovered-endpoint conditional relation recall was **12.57%**. Not a universal 91% law. | `6-ixi-query-attribution.md`, original Oct19 |
| **R02** | Modern SyntheticMRI full-query / geometry / exposure / overprediction attribution, B0 vs A-xyz, both seeds | **DONE — measured** | **82–83%** A-xyz FP touch spatially unmatched endpoints, but only ~35% touch Hungarian-unassigned. About **90.7–90.8%** FN GT edges lack a recovered endpoint pair. Net node count **underpredicted** ~0.8–0.9/patch despite false nodes; β₁ bias **+1.45–1.53**. | `6-ixi-query-attribution.md`, Oct26 |
| **R03** | Relation error vs GT arc length/tortuosity, modern SyntheticMRI and historical IXI | **DONE — measured** | A-xyz conditional recall shortest vs longest GT arc quartile **~99% vs ~90%**, BUT full-GT recall is **better on longer branches** because short-edge endpoint pairs are harder to recover. Near nonadjacent matched GT pairs (<5 mm) have ~47% FP rate in the examined SyntheticMRI populations. Length effect ≠ dominant overall loss. | `6-ixi-query-attribution.md`, Oct26 |
| **R04** | Legacy IXI original-vs-Hungarian-vs-spatial-oracle selection and retrained focal/regularized CE/matched/unmatched hard-negative relation heads | **DONE — negative/mixed** | No head improved original validation ranking; naive extra negatives can suppress valid connections. Already attempted mining **on older IXI**; do not claim “unmatched negatives never tested.” | `EXPERIMENT_REVIEW_20261021.md`; historical `8-ixi-relation-matching-calibration.md` |
| **R05** | Older IXI frozen detector: residual relation logit adapters — token/geometry, endpoint CNN, straight chord, 3D corridor | **DONE — mixed/overfit** | At **2k**, conditional AP original **0.4723**, endpoint **0.4907**, chord **0.5016**, corridor **0.5019**; image arms exceeded token control **0.4766**. At **65.6k**, all adapters were worse than original (chord **0.3216**). All used fixed older IXI detector/features, edge focal, same candidates. | `9-ixi-image-aware-relation-learning.md`, Oct26 early-score addendum |
| **R06** | Modern relation-specific **552D node-token residual fusion** with local CNN features, fixed modern A-xyz candidate set | **NOT RUN** | Different from R05's **edge-logit** adapters. No demonstrated benefit yet; design after query diagnostic N06. | Future Astra design → Sol implementation |
| **R07** | Modern SyntheticMRI **matched-pair hard-negative sampling** emphasizing reliable close nonadjacent GT vertices at fixed pair budget | **NOT RUN in this form** | Earlier mining R04 exists, but not this specific modern checkpoint/current CE recipe/controlled matched-pair objective. | Future targeted Sol experiment, *after N06* |
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
| **L06** | Training-only BN recalibration on archived IXI and SyntheticMRI checkpoints | **PARTIAL / RECOVERY** | Slurm **1584593_0** (IXI) FAILED after 49s, **1584593_1** (SyntheticMRI) COMPLETED after 1m37s, collector **1584594** CANCELLED. No validated paired results in supplied report; recover only failed job + collector after integrity checks. | `spline-collapse-diagnosis.md` Oct26; user `sacct` output |
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
| **D04** | Modern SyntheticMRI → IXI fully unfrozen transfer, B0/pretrained-fresh A-xyz, standalone A-cls ± combined A | **STATUS UNKNOWN** | Discussed/promoted to another Sol Codex prompt, but **no verified final report or accepted Slurm IDs** in this provided set. Check repository, worktrees, `sacct`, caches and reports before rerunning. | Execution audit P1 |
| **D05** | BRAVA dataset transformation, orientation correction, graph target preparation | **DONE — measured QC** | 56 paired images; 39/8/9 subjects; 8,768/1,796/2,043 model-ready patches; five hash-pinned x-reflections; **brain masks ≠ vessel masks**. | `BRAVA_preparation.md` |
| **D06** | Representative BRAVA train/val K8 monitor, saved-vs-batch BN, training-only BN recalibration, confidence calibration, augmentation geometric-alignment tests | **DONE — diagnostic, mechanism partly unresolved** | At 4k, matched saved-BN **train node/edge F1 0.1325/0.0114**, val **0.1183/0.0089**; failure occurs on optimized-on training patches (not just val). Training-only BN recalibration **does not rescue** and batch statistics make performance worse. Train-selected node threshold 0.4 helps modestly; augmentation transformations passed alignment checks. No retraining was done. | `BRAVA_experiment_plan(1).md`, Oct26 learning-diagnosis addendum |
| **D07** | BRAVA original-K8 fully trainable **augmentation off vs unchanged augmentation on**, same source/1,248 patches/seed/stream/4k updates | **NOT RUN — NEXT** | D06 flags augmentation as an unresolved *optimization* factor; correct geometric transforms do not prove augmentation aids learning. First controlled short train ablation, then assess underfitting/insufficient exposure if no gain. | Sol next study **Q2** |

---

## 3. One-glance test matrix (the checklist for future prompts)

**Legend:** ✅ measured; ⚠️ mixed/negative/partial; ⬜ not confirmed tested; ⏸ gated/deferred.

| Scientific question | Tested already? | Current decision / blocker | Related IDs |
|---|---|---|---|
| Does local fine CNN refinement improve node localization? | ✅ | Yes on modern SyntheticMRI, two seeds. Use A-xyz as reference. | N01 |
| Does refining class logits alongside xyz help? | ✅ | No meaningful independent gain in A ablation. | N01 |
| Does a historical contextual/KNN verifier help? | ⚠️ | No; don't replay V3 as a new idea. | N03 |
| Does standalone image-conditioned `A-cls` help on modern IXI? | ⬜ | First verify ongoing modern IXI report/job. | N07, D04 |
| Can simple spatial NMS solve duplicates? | ⚠️ | Tested; loses correct anchors/edges. | N04 |
| Why does confidence GT oracle beat nearest GT oracle? | ✅ | ~33% representatives change; confidence choices are usually less accurately localized but relation-richer, Hungarian-aligned alternatives. Mechanism partly established; training-history causality remains unproven. | N05–N06 |
| Can a GT-free method reproduce useful oracle query selection? | ⬜ | Must avoid GT leakage and graph-consistency shortcuts. | N08 |
| Do long/tortuous GT edges have worse conditional recall? | ✅ | Yes, **conditional**, not overall dominant loss. | R03 |
| Did we ever feed fine CNN endpoint/path features to relation scoring? | ✅ | Yes, legacy IXI frozen-detector residual **edge-logit** adapters; early gain, late overfit. | R05 |
| Did we refine the **552D node tokens** for a modern A-xyz relation head? | ⬜ | No confirmed test; future architecture after oracle analysis. | R06 |
| Did we already try extra matched/unmatched hard negatives? | ✅ | Yes on historical IXI and mixed-training settings; no demonstrated overall gain. | R04, L01 |
| Did we test *modern reliable near-nonadjacent matched-negative sampling* with edge CE? | ⬜ | Not in this controlled form. | R07 |
| Did we test reliable unmatched-query labels with ambiguity safeguards on S0? | ⬜ | Not in this controlled form. | R08 |
| Is a larger multiscale attention transformer already a win? | ⚠️ | Short C pilot negative; A+C/long clean test not established. | L04 |
| Was BRAVA K24/C1 **trained**? | ✅ | Yes, learned independent geometry heads on **shared trainable detector**; not independent full detector replicates. | G05 |
| Is K24/C1 geometrically more expressive on BRAVA GT? | ✅ | Yes, oracle fits; learned graph benefit not demonstrated. | G06 |
| Was BRAVA degree-2 **graph-changing** subdivision tested? | ✅ | Yes, increases partial Curve F1 and cycles/output size. | G05 |
| Was genuine trained error-adaptive C1 / coupled detector geometry tested? | ⬜ | Not established; defer. | G07 |
| Is the weak BRAVA detector due to BN or poor larger-cohort optimization? | ⚠️ | Saved BN best among tested variants; no BN rescue. Both representative train and val fit poorly. Augmentation-vs-exposure/optimization causal attribution unresolved. | D06–D07 |
| Was limited-data leaf transfer tested? | ✅ | Yes, after infrastructure recovery; historical 21 Oct pending status superseded. | D02 |
| Did training-only BN recalc complete on both domains? | ⚠️ | Only SyntheticMRI GPU stage completed; IXI failed, collector cancelled. | L06 |

---

## 4. Prompt/run inventory — evening update

| Workstream | Verified status from supplied material | Avoid duplication / next action |
|---|---|---|
| **P2 / N06 — Oracle representatives** | **DONE.** Paired two-seed SyntheticMRI analysis, same-neighbor controls, single-query swaps, exact embedding replay job `1597409`. No new model trained. | **Do not repeat.** Advance to **N08**, a *GT-free* query selection experiment. |
| **P3 / D06 — BRAVA learning diagnosis** | **DONE.** Recovered GPU evaluation `1598848` and dependent collection documented as measured. Representative cohorts, BN/augmentation checks and confidence calibrations reported. | **Do not repeat.** Advance to **D07** augmentation-on/off **training** ablation. |
| **P1 / L06 — historical training-only BN recovery** | **UNVERIFIED in current attachments.** Last verified: `1584593_0` failed, `_1` completed, `1584594` cancelled. | Verify saved artifacts/job history before recovery; do not assume P1 was never sent or that it completed. |
| **D04/N07 — modern IXI B0/A-xyz/A-cls transfer** | **UNVERIFIED in supplied reports.** Old IXI experiment details do not substitute for current fully trainable results. | Find actual checkpoint, submission and collector first. Do not launch duplicate transfer. |
| **R06 — relation-specific 552D token fusion** | **NOT RUN.** N06 shows representation differences, but not the value of training another module. | Keep as alternative after testing GT-free selection; use Astra only for architecture decision if required. |
| **R07/R08 — modern reliable matched/unmatched negative supervision** | **NOT RUN in this modern controlled form.** Historical mining was tested on weaker protocols. | After N08, test R07 first if close-nonadjacent confusion remains limiting. |
| **R09/G07/L04 A+C** | **DEFERRED** | No new path tracker, K24/C1 geometry sweep or multiscale attention before simpler mechanisms improve graphs. |

The status of actual current Codex sessions and queued Slurm tasks cannot be inferred from these attachments; inspect live Jean-Zay before submitting anything.

---

## 5. Updated priority queue — next prompts for Sol

### Q0 — Quick status/BN integrity and modern IXI audit — **Sol 6.1 Medium** — CONDITIONAL

- Verify whether P1/L06 was already executed. If BN IXI stage or collector remains incomplete, recover only failed work, preserving completed SyntheticMRI outputs and locked scientific settings.
- Check `squeue`, `sacct`, worktrees, submission records and result collectors for modern IXI A-cls/A-xyz/B0. **No duplicate submission.**
- Update L06, D04, N07 from hard evidence. This is administration/recovery, not a new scientific train run.

### Q1 — N08: GT-free relation-aware query/representative selection — **Sol 6.1 High** — READY, CPU-FIRST

- Reuse frozen two-seed modern SyntheticMRI A-xyz full raw queries, refined coordinates, node confidences, original symmetric relation scores and validation protocol.
- Design a small **GT-free** selection method: e.g. conservative local candidate groups with confidence/uniqueness and *calibrated* relation support from independent candidate neighbors. **Do not** let self-consistent spurious loops or high-degree hubs automatically dominate.
- **Controls:** standard threshold decoding, simple confidence-based suppression, matched-retained-node count, and optional simple GT-free confidence-ranking baseline. The GT-assisted oracle remains a **diagnostic ceiling** only.
- Lock candidate grouping/selection hyperparameters with training data and verify behavior on all fixed 480 validation patches/13 subjects for both seeds. Do not use GT to form groups, choose representatives or construct relation scores.
- Report node/edge F1 at 1/2 mm, all-retained/conditional AP with populations, true/false edge changes, missed-anchor recovery, duplicate counts, β₀/β₁ signed bias and MAE, GT shortcuts and high/low-confidence strata. Detect recall loss or silent edge pruning. No model training in first stage.
- **Stop** if there is no consistent improvement over matched-count confidence control in both seeds, or improvements arise only from more/fewer predicted nodes. Otherwise recommend the smallest trainable selection mechanism to investigate later.

### Q2 — D07: BRAVA K8 augmentation-on/off training ablation — **Sol 6.1 High** — READY, BOUNDED GPU

- Start from exactly the existing clean SyntheticMRI checkpoint and original BRAVA **K8** target. Use the same 1,248 train/256 val patches, original 39/8 subject split, seed, batch 8, 4,000 updates, optimizer, LR schedule, node focal γ2, matched-pair edge CE, `graph_only` geometry route, and saved-BN evaluation.
- Compare **unchanged augmentation on** versus **geometric/intensity augmentation off** as the only experimental variable; reuse fully documented old augmentation-on control if parity is demonstrable, otherwise run a paired control. No K24/C1/split_K8 or node embedding changes.
- Record source hashes, order of training sample IDs, evaluation batch 1 and 8 parity, original 16-patch tiny reference (monitor only), and **representative 160-train vs identical 256-validation** curves at fixed 1k/2k/4k checkpoints.
- Main question: does disabling augmentation improve *optimized-on representative training fit*, raw 1/2-mm node recovery, confidence retention, edge F1 and conditional relations, without merely overfitting validation? Report train/val differences, assignment churn, signed counts, β₀/β₁ and SWC-annotation caveat.
- **Stop after 4k** and report the causal ablation. If both remain poor, next experiment should isolate training exposure/budget or matching rather than increasing spline capacity.

### Q3 — R07: reliable nearby nonadjacent matched-pair supervision — **Sol 6.1 High** — AFTER Q1/IXI AUDIT

- First audit negative pair exposure on modern A-xyz and quantify truly reliable close nonadjacent pairs versus ambiguous alternate representatives, plus recall on longer positives.
- If warranted, compare existing matched CE sampling to a matched-budget negative sampler that emphasizes **unambiguously** nonadjacent nearby anchors; keep network/optimizer and candidate sets fixed. Preserve all positives and avoid blanket unmatched negatives.
- Evaluate ranking AP, long-positive recall, near-shortcut false-positive rates, edge TP/FP/FN, signed edge bias, cycles and subject uncertainty. Stop if it merely reduces all edges.

### Q4 — R06: relation-specific 552D node-token fusion — **DEFER until Q1 outcome**

- If GT-free selection cannot exploit existing embeddings, ask Astra High to choose a minimal design and matched controls. Sol High then implements R0 original, R1 token-only residual, R2 local-CNN residual fusion at **same 552D dimensionality** with unchanged node coordinates, early/fixed candidate evaluation and strict overfit controls.
- **Do not** repeat historical 65.6k frozen IXI endpoint/chord/corridor logit experiments. Combine R06 with R07 only after independently justified wins.

### Q5 — geometry and long-range path models — **DEFERRED**

- BRAVA degree-2/K24/C1 capacity, image-conditioned spline/path edge verification, deformable/multiscale attention, and new adaptive GT representations remain tracked, not forgotten.
- Unlock BRAVA geometry work only after representative training graph detection is functional; unlock path models only if a significant residual long-edge problem persists after node selection/relation supervision.

---

## 6. Updated decision gates

| Observation | Consequence |
|---|---|
| **Q1 GT-free relation-aware selection improves both seeds over matched-count controls without worse recall/topology** | Consider a small trainable relation-aware query verifier or structured selection objective; keep simple frozen inference rule as control. |
| **Q1 fails or amplifies loops/shortcuts** | Treat oracle gap as largely inaccessible by naive local selection; consider R06 relation-specific token fusion, or R07 if negatives remain primary error. |
| **Q2 augmentation-off noticeably improves representative BRAVA training fit** | Repeat with carefully matched exposure and then a different seed; assess whether generalization suffers. |
| **Q2 augmentation-off does not improve training fit** | Isolate optimization budget, per-patch exposure, assignment dynamics and annotation/image difficulty. Do **not** infer from small 16-patch overfit alone that population training simply needs more updates. |
| **R07 improves nonadjacent edge ranking at maintained positive recall** | Confirm modest fully trainable study, then evaluate whether adding reliable unmatched supervision is warranted. |
| **R06 improves ranking beyond token-only residual and valid baseline** | Test short fully unfrozen confirmation; beware R05 overfitting and reused IXI subjects. |
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

**Next edit owner:** Sol agents working on Q0, Q1 and Q2. Preserve stable IDs and old failed experiments; do not replace the whole roadmap.
