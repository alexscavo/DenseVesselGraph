# IXI vascular graph extraction — diagnosis and experimental roadmap

> **Research consolidation, 23 October 2026:** [Consolidated current priorities](../../research/roadmap.md). This document retains its detailed and dated historical context.

**20 October — paired segmentation-input diagnostic:** [report 11](metrics/11-ixi-segmentation-input-oracle.md) implements raw-MRA versus supplied binary vessel-label input from a common pre-IXI SyntheticMRI initializer. These masks generated the graph references (provenance category 2), so this tests mask-to-graph reconstruction. The 2,048/512 paired population is gated by complete alignment checks and GPU/CPU smoke collection; benchmark-selected matched updates and two-H100 dependencies are recorded in the report. This is separate from the existing image-feature and node-loss studies; no results or changed scientific priorities are inferred before completion.

**Discussion snapshot:** 19 October 2026  
**Revision:** extended local-refinement, hard-negative, and multiscale feasibility protocols after follow-up Q&A.  
**Status:** research plan; the proposed experiments below have **not** been run unless explicitly marked *completed*.  
**Focus:** unusually weak graph reconstruction on real IXI MRA, despite better behavior on SyntheticMRI; duplicate/extra nodes, missed nodes, and unreliable connectivity in the 3D RelationFormer family.

**Implementation update, 19 October:** P0-A/B/C/D tooling and eight-patch smoke
are implemented; frozen-checkpoint pilots 1291324/1291328/1291329 completed
model evaluation, and corrected CPU collection completed after reporting failures.
Local-refinement results: oracle deletion raises positional edge F1 0.0737→0.0892,
while both fixed NMS radii reduce it. This round stops at these pilots.
See the [execution record](metrics/6-ixi-query-attribution.md) and
[project review](PROJECT_REVIEW_20261019.md). This does not mark P1/P2 or
full-validation attribution as completed.

**Next bounded batch:** raw-query oracle selection, confidence/token probes,
inference-time coordinate/classification separation and isolated relation-loss
pilots completed successfully; [report 7](metrics/7-ixi-mechanism-batch.md) records
results. Coordinate correction explains most refinement benefit. Raw-query oracle
selection improves node recovery but only modestly improves edges; tested relation
continuations worsen validation AP despite fixed-cutoff recall gains.

> **Working principle:** do not start another large training sweep or add several new losses at once. Use already-saved predictions to establish what is failing; perform controlled, cheap interventions next; change the model only when a diagnostic supports that change. The previous code and loss formulation are baselines, **not constraints** that must be kept unchanged.

## 1. Scope, current understanding, and points of agreement

### 1.1 Observations reported during our Q&A

- On real IXI, the model often places **multiple predicted nodes near one GT node** (some inside the segmented foreground, some slightly outside). Occasionally it also predicts nodes where no vessel appears to exist. The user considers many predicted locations approximately plausible, but **node uniqueness and graph connectivity** are poor.
- The model learns to reject many obvious nonedges early; historically, reducing the edge candidate budget to a few hundred or approximately 1,000 while doing hard-negative mining performed worse than supervising a much larger candidate pool. Probability-based weighting/focal experiments were not consistently better than the original approach.
- A high node count increases the potential number of candidate connections: for `N` retained nodes there are `N(N-1)/2` undirected pairs, although only some are predicted as edges.
- SyntheticMRI behavior is less problematic, which makes data/appearance shift a reasonable hypothesis but **does not prove** that generalization is the sole issue.

### 1.2 Training versus inference: the central mechanism to test

```text
TRAIN (GT graph available)
  image -> backbone -> transformer -> object queries
                                      |
                            node coordinates + scores
                                      |
                             Hungarian / configured matcher
                                      |
                         M matched queries + U unmatched
                           |                     |
                   node/edge GT losses       no-object loss
                   mostly on M-M pairs       (baseline edge loss
                                             does not supervise U-M/U-U)

INFERENCE (NO GT, NO Hungarian matching)
  image -> backbone -> transformer -> object queries
                                      |
                        node-confidence filtering
                                      |
                         ALL retained query pairs
                                      |
                       relation-score filtering
                                      |
                                predicted graph
```

- Hungarian matching assigns **at most one query per GT node** in the usual set-prediction objective. A duplicate query can therefore be classified as background during training even if it is close to the vessel. Node/background classification applies to all queries; coordinate regression applies to matched queries. Thus, the model is **already** trained to suppress duplicate objects, but may not successfully do so.
- **The Hungarian matcher is not an inference-time duplicate remover.** In the archived `training/evaluation/inference.py`, predicted queries are retained by object-confidence/class filtering, with no default spatial NMS. Relations are evaluated on all unordered pairs of retained queries.
- Standard matched-pair relation supervision addresses M-M pairs; U-M and U-U pairs involving unmatched object queries are absent unless the optional unmatched-negative path is enabled. This **may** make the relation head unreliable on duplicate/false-positive detections at inference.
- This is a **hypothesis**, not a demonstrated causal explanation. The main failure may still arise among correctly located nodes; we need to measure the fractions.
- During node-refinement **training**, Hungarian/configured matching and its losses were retained. During **inference**, there is still no matcher. The available report does not independently verify whether the new matcher cost in the updated runtime uses refined or original proposal coordinates; check the current live code before asserting that detail.

### 1.3 Mechanisms must be treated separately

| Potential bottleneck | Concrete symptom | Relevant first test |
|---|---|---|
| **Appearance / image evidence** | Real MRI vessels not adequately recognized | Binary-mask-input oracle; fine-feature readout |
| **Node localization** | Nodes lie a little off centerline | Existing local CNN refinement; coordinate-only ablation |
| **Duplicate-node suppression** | Several high-confidence queries at one target | Duplicate/confidence audit; oracle pruning; GT-free NMS control |
| **Missing endpoints** | True GT edges lack one or both nodes | Recall by node subtype, confidence-rejected near-GT queries |
| **Relation learning** | Wrong connections even between recovered GT nodes | M-M conditional analysis; hard-negatives/ranking pilots |
| **Train/validation transfer** | Near-perfect fitting on tiny training set, weak on unseen subjects | Stratified train/val subject and appearance audit |
| **Target-label reliability** | Segmentation-derived bridges, missed vessels, uncertain loops | Paired raw-MRI/segmentation/GT graph visual QC |

**Agreed direction:** prioritize read-only/error-attribution experiments, keep the promising local-CNN feature pathway in view, consider multiscale transformer access later, and revisit hard-negative supervision **after identifying whether duplicate nodes or incorrect relations dominate**. No particular new loss or architecture is yet accepted as the final solution.

## 2. What is already known — do not confuse with proposed tests

### 2.1 IXI supervision and current graph quality

- **Completed — `metrics/1-ixi-supervision-audit.md`:** GT nodes were all on the supplied segmentation in a 16-patch audit, but this only demonstrates consistency with the same segmentation used to make GT, **not anatomical ground-truth correctness**. Among predicted nodes, 34.25% were inside the mask and 65.57% within 1 mm of foreground voxel centers. Full-validation 2-mm positional node recall was 50.82%; 69.38% of GT edges missed at least one endpoint; among those with recovered endpoints, direct relation recall was 13.62%.
- **Completed — initial IXI 300-epoch held-out test:** node F1 0.409, edge F1 0.307, physical Branch F1 0.00535, with excessive edges/cycles. Another documented IXI baseline averaged **179.76 predicted versus 50.39 target edges per patch**. These measurements come from specific checkpoints/protocols and must not be combined as if they were from one model.
- **Completed — four-patch overfit (`pretraining/ixi-overfit.md`):** the straight arm fit four real IXI training patches with train node and edge F1 = 1.0 after 1,000 updates, while the four-patch validation edge F1 was ~0.099. This demonstrates small-sample memorization capacity, **not** good generalization or full-dataset fitting.

### 2.2 Node features and local refinement

- **Completed — `metrics/2-ixi-feature-node-diagnostics.md`:** early 32³ CNN features (64 channels) yielded validation foreground-probe AP **0.6551**, while the 2³ projected-CNN and encoder readouts yielded **0.0271** and **0.0246**. A raw-intensity control achieved **0.6802**. The coarse-grid probe cannot prove that information has been destroyed in the learned representation; it strongly motivates access to finer spatial features.
- The same node diagnostic recovered **63.43% training vs 51.12% validation** GT nodes at 2 mm, indicating a generalization gap **and** residual training-set error. Its validation duplicate-candidate count was **209 on 64 patches** under that diagnostic's definition; do not conflate that count with all false-positive nodes or graph edges.
- **Completed — `metrics/3-ixi-node-refinement-pilot.md`:** early CNN `layer0.relu1` feature map at 32³ sampled on a `3×3×3` lattice centered on each proposed node at offsets `−4,0,+4` **input voxels** along each axis; combine 27×64 local features, object token, and proposed coordinates; small MLP predicts bounded coordinate and classification-logit corrections. Separate from the sampling offsets, coordinate corrections are bounded to **±4 voxels per axis**.
- In a **128-patch validation / 1,000-update / single-seed** comparison, baseline vs local-image: node F1 **0.4857 → 0.5196**, edge F1 **0.1553 → 0.1673**, curve F1 **0.1171 → 0.1676**. Zero-image control stayed near baseline. Endpoint-pair recovery improved **30.20% → 37.96%**; conditional relation recall stayed approximately **12.35% → 12.48%**. Good evidence for finer local image information, **not** evidence that the relation head was fixed.
- The local refinement adjusts the final node positions/confidences; in the tested architecture the **relation head receives decoder object/relation embeddings, not directly the corrected coordinates or sampled fine features**. Final graph geometry uses refined endpoints.

### 2.3 Edge-selection and relation experiments already tried

- **Completed — `metrics/5-ixi-threshold-calibration.md`:** changing **edge** threshold only (node outputs fixed) from **0.500 to 0.382** improved validation positional edge F1 **0.0737 → 0.1608** but worsened cycle-count error (β₁ MAE) **1.0391 → 19.5234**. More edges reduced component fragmentation but created many false cycles. A conservative **0.4977** threshold barely improved F1. **Do not repeat this edge-threshold sweep as a proposed solution.** A careful *node*-threshold/NMS study is a different question and remains open.
- **Completed — `metrics/4-ixi-relation-isolation.md`:** existing relation head validation AP was **0.4702** on recovered-endpoint pairs, despite low recall at the 0.5 cutoff; refitting improved training AP but lowered validation AP. Nearby nonadjacent pairs are frequent distractors and long true branches are harder; differences in probe populations/architectures limit interpretation.
- **Completed — `metrics/relation-distance-pilot.md`:** a **5× weight on nonedges with GT endpoints <5 mm**, with negative weight mass renormalized, reduced near-false-edge rate in a controlled SyntheticMRI pilot (relation-only **73.7% → 66.1%**), but did **not** improve overall Curve F1/TOPO/APLS over the starting checkpoint; fragmentation worsened. This is evidence that naive geometric negative weighting alone is insufficient, not proof that all targeted negative learning fails.
- **Completed — `evidence/degree_consistency/README.md`:** a combined experiment added expected-degree consistency, unmatched hard negatives, and changed object classification weight. Interim results reduced cycles and improved some edge scores while worsening fragmentation/Branch F1. These factors were changed together; **do not attribute the outcome uniquely to hard negatives**.
- **Historical hard-negative comparisons:** reducing `max_per_graph` while adding mining confounded candidate coverage with mining; existing focal configurations may also remove positive ratio upsampling or enable unmatched candidates. Treat each as an indication to design **factor-isolated** tests, not as a definitive comparison of hardness weighting itself.

### 2.4 Existing regularized edge sampling: exactly what it does

In the archived `training/losses/criterion.py`, `max_per_graph: 9999` is a **candidate limit**, not a guarantee of 9,999 unique supervision pairs. If there are 40 matched nodes, only `40×39/2 = 780` possible unordered M-M pairs exist. The baseline includes the available positives and negatives within the cap, then `_ratio_upsample` **duplicates the minority label class** to approximate a target positive/negative ratio of 0.15 (it can also duplicate negatives if they are the minority relative to that ratio). Duplication is a form of **loss reweighting**, not independent evidence.

**Decision:** no a priori rejection of Berger et al.'s regularized edge-sampling idea. Retain a reproducible baseline; test alternate explicit class normalization, unmatched negatives, and targeted ranking as separate, controlled changes. Normalizations/gradient scales must be audited so a new loss does not merely shrink the positive signal.

## 3. Prioritized test plan

Legend: **P0** = read-only, cheap, before new training; **P1** = small controlled train/evaluate; **P2** = architecture/data redesign contingent on evidence. All new experiments start **not started** unless marked completed in Section 2.

### P0-A — Freeze a reproducible IXI baseline and evaluation manifest

**Question:** Are we comparing the same model/patches/protocol across diagnostics?

**Procedure**
1. Choose one IXI checkpoint/lineage as primary (prefer the documented delayed-spline checkpoint with local refinement available) and retain one unmodified straight-graph comparator if needed. Record `resolved-config.yaml`, checkpoint hash/step, matcher configuration, training split, augmentation, 256-query count, node and edge thresholds, BN mode, preprocessing, and reference representation.
2. Evaluate/reuse cached outputs for the **same disjoint-subject train and validation IDs**; preserve original full validation and a short fixed-patch diagnostic subset. Clearly distinguish native benchmark node/edge F1 from the newer 2-mm positional graph protocol and Branch/Curve F1.
3. Cache **all 256 query scores, predicted coordinates (original + refined, when applicable), query IDs, decoder tokens if feasible, full pair relation logits, matched GT IDs for diagnostic analysis, image/segmentation/GT graph identifiers**, and geometry. Do not save only edges above the current threshold.
4. Validate normalized-axis-to-voxel/physical conversions, empty graphs, crop-boundary nodes, and that the exact same edge scorer/undirected symmetrization is used for cached pairs and normal inference.

**Deliverables:** immutable manifest, per-patch prediction cache, metric reproduction check, a few fixed linked 3D or slice overlays. **Pass gate:** saved predictions reproduce current metrics before running new ablations.

### P0-B — Duplicate-node, confidence, and GT-proximity audit **[first scientific test]**

**Question:** Are most false detections redundant queries around real GT nodes, or genuinely off-vessel hallucinations?

**Procedure**
1. For each patch, match retained predictions to GT nodes **one-to-one by physical distance** at 1 and 2 mm, without using prediction degrees. Reserve accepted matches before assigning duplicate/displaced categories.
2. For unmatched predictions, label mutually exclusive diagnostic categories: (a) near an **already matched GT node** (likely duplicate), (b) near an **unrecovered GT node** (potentially useful but rejected under one-to-one assignment), (c) vessel-foreground-supported but not near a GT node (possible unannotated vessel or extra degree-2 sample), (d) off-vessel/unexplained. Report boundaries and the chosen distance-to-mask thresholds; do **not** equate all unmatched queries with anatomical false positives.
3. Plot object-confidence distributions by category; duplicate multiplicity per GT, nearest-neighbor separation of predictions, fraction of duplicate clusters containing ≥2 high-confidence queries, categories by node type (junction, degree-2, boundary, interior terminal) and IXI acquisition site.
4. Repeat using **all 256 raw queries before confidence filtering** to separate poor confidence ranking from outright missing proposals.

**Readouts:** node precision/recall, duplicate precision and prevalence, matched vs unmatched score calibration, raw-query recoverability, foreground support, counts per patch. **Interpretation:** many high-confidence duplicates favor uniqueness/matching/NMS investigation; confident remote off-vessel nodes favor image specificity/negative object supervision; many near-GT but confidence-rejected queries favor calibration (without automatically lowering thresholds).

**Caveats:** 1–2 mm proximity is a metric, not proof of duplicated anatomical nodes; true junctions/degree-2 subdivisions can be close in physically dense graphs. Cross-check 3D views, not only maximum-intensity projections.

### P0-C — Decompose predicted edges by endpoint type **[second scientific test]**

**Question:** What fraction of spurious connectivity is caused by the extra/duplicate nodes?

**Procedure**
1. On the same cached predictions, classify each retained edge as **M–M**, **M–U**, or **U–U** using the spatial one-to-one diagnostic matching above; separately subdivide U into duplicate/near-GT/unsupported categories.
2. An M–M edge is a TP only if both matched GT endpoints are truly adjacent. M–U and U–U edges count as FP under the declared node-to-GT scoring rule; flag candidate annotation omissions for visual QC.
3. Report **edge TP/FP/FN counts and precision by category**, not merely the fraction of all edges. Include missed direct relations among GT edges with both endpoints found; stratify by GT arc/chord length, distractor spatial distance, predicted edge-score bin, and node confidence.
4. Compare **fraction of false edges incident to unmatched nodes**, and connected-component/cycle changes when edges incident to those nodes are deleted (diagnostic, not a deployable decoder). Count per-patch β₀, β₁, degree histograms and excessive degree near duplicate clusters.

**Deliverables:** per-patch CSV + population totals + 3–5 fixed 3D examples highlighting M–U/U–U links. **Interpretation:** U-involved FP dominance favors duplicate suppression/unmatched supervision; M–M FP dominance favors direct relation discrimination. Both can coexist.

### P0-D — Oracle pruning upper bound vs realistic GT-free duplicate suppression

**Question:** If extra nodes disappeared, would the already-trained relation scorer become good enough?

**Oracle diagnostic (uses GT; never deploy as inference):**
1. Keep the checkpoint, query IDs, relation scores and edge threshold fixed. From retained nodes, keep **one** one-to-one matched query per GT node, preferentially the spatially assigned query; remove unmatched predictions and their incident edges. **Do not insert missing GT nodes, relabel edges, or recompute relation scores using GT.**
2. Compare original graph to oracle-pruned graph on exactly the same held-out validation patches. Report changes in M–M edge precision/recall, overall positional edge F1, Branch/Curve F1, β₀/β₁ errors, and retained-node fraction. Remember that deleting nodes can artificially improve precision while harming coverage; report raw TP/FP/FN and geometry.
3. Optional optimistic variant: retain the closest raw query for each GT even if its object confidence was below threshold; label separately because this tests an **oracle object-selection ceiling**, not simple duplicate removal.

**Non-oracle decoder control (GT-free; validation development only):**
4. Apply simple spatial suppression to near-coincident predicted queries using node confidence, in **physical mm** with predeclared small distances and an `off` control. Try only a small, justified grid; do not blindly suppress legitimate nearby bifurcations, boundary nodes or degree-2 nodes. Optional radius-aware or type-aware suppression belongs **after** simple suppression and visual review.
5. Re-evaluate at the original node/edge confidence thresholds. A node-confidence-only sweep is a separate control; a combined node+edge search can easily overfit validation and should not be the first intervention.

**Interpretation:** large oracle gain but small NMS gain means duplicate detection/assignment is promising but simple distance suppression is insufficient. Small oracle gain means the connectivity model and/or missing endpoints remain the bottleneck.

### P0-E — Training/validation and GT reliability audit

**Question:** Is the gap caused by appearance/acquisition shift, supervision density, graph complexity, or inconsistent labels?

- Compare subject/site/spacing strata (IXI Guys, HH, IOP as applicable), raw intensity distributions after loader normalization, background texture/noise/contrast, small-vessel foreground fraction, node density, degree-2 proportion, boundary fraction, graph edge-length/radius distributions, and train/validation composition. Always aggregate by **subject**, not only correlated patches.
- Revisit uncertain raw MRA + segmentation + extracted graph examples in consecutive slices (potential missing thin vessels, fused loops/bridges, false GT nodes). Preserve segmentation-derived GT as *provisional*; no global regeneration based on a few examples.
- Compare early/best/final checkpoints using the **same** diagnostic decomposition to distinguish a genuinely new problem from late overfitting.

**Gate:** address any high-confidence preprocessing/misalignment bug before changing objectives. Dataset shift is plausible, not yet quantified as the sole explanation.

### P1-A — Binary vessel-mask input: representation-versus-image oracle **[highest-priority new training experiment]**

**Question:** Could the existing graph architecture recover IXI graphs on unseen subjects if vessel locations were unambiguous in the input?

**Arms:** (1) original raw-MRA input baseline; (2) **GT binary segmentation as input** with identical GT graph, split, 256 queries and graph losses. A raw intensity / contrast-normalized control or mask+MRI combination can follow **only if needed**. Mask input is *not a clinically available inference solution*: it is an upper-bound-like diagnostic, and GT is derived from that mask.

**Protocol:** keep matched initialization or train from scratch consistently, match update budget/optimizer, no augmentation that breaks image-mask correspondence, use subject-disjoint validation, report both training fit and validation. Preserve the base object/relation decoder and GT representation. Check that the input mask truly corresponds to the graph and that train and val preprocessing is identical.

**Decision:** strong unseen-subject improvement with mask input favors image/feature/normalization work; persistent mask-input failure favors graph targets, queries, node uniqueness, matcher and relation objectives. Mixed outcome means both contribute. Even near-perfect mask-input reconstruction would not establish clinical graph correctness because the labels are derived from segmentation.

### P1-B — Node local-image refinement: reproduce, isolate, and extend

**Why prioritize it:** this is already a positive, controlled result, unlike most proposed architectural changes. On a fixed 128-patch IXI validation subset after 1,000 updates, early-feature sampling improved node F1 `0.4857 → 0.5196` and Curve F1 `0.1171 → 0.1676`, while the zero-image control did not improve; nevertheless, conditional relation recall was almost unchanged. It is **not** a full-validation, multi-seed demonstration.

**Existing implementation to preserve for comparison (not necessarily forever):** for each preliminary predicted node position, sample **27 locations** on a `3×3×3` lattice with offsets `−4,0,+4` **input voxels** per axis from early 32³ / 64-channel CNN features. Concatenate sampled vectors, decoder object embedding and preliminary coordinates, then predict coordinate and object-class-logit residuals with a 128-hidden-unit MLP. The **sampling ±4** determines *where features are read*; the separate **correction ±4** bounds *how far coordinates can move* (per axis). The relation head in the tested model still consumes unmodified decoder embeddings, **not** the refined coordinates/features. Training retains node Hungarian matching; no matcher runs at inference. Verify in **live code** whether matching consumes original or refined coordinate outputs.

**Ordered test protocol:**

1. **Reproduce the exact pilot and inspect effects on uniqueness** with identical diagnostic patches/checkpoint/update budget. Compare saved predictions before/after refinement: original-to-refined displacement, GT correspondence, confidence changes, duplicate count around one GT node, and off-foreground distances. Confirm that improvements are not merely gained by accepting more near-duplicate nodes.
2. **Separate coordinate and confidence paths**: same parent checkpoint and node-head-update policy for (a) node-head continuation only, (b) coordinate residual only, (c) class-logit residual only, (d) both residuals, plus a zero-image/vector control. Check whether coordinate-only improves positioning, class-only changes query selection/duplicates, and whether combining both is synergistic. Record active/frozen parameter hashes and gradient norms; keep the edge head frozen in this isolation test.
3. **Check sampling scale and geometry**, keeping the model unchanged otherwise: compare the original 27-sample lattice with a very small set of alternatives (e.g., smaller/larger support, physical-mm offsets adjusted for acquisition spacing, or a second early/mid CNN scale). Check off-image clamping, axis order, exact convolution-grid alignment, anisotropic spacing and patch boundaries. Do **not** turn this into an unrestricted hyperparameter sweep.
4. **Generalization and stability:** verify gains across held-out IXI subjects/acquisition sites, on full validation using a locked evaluation protocol; repeat top candidate with at least one further seed or subject-level bootstrap, and continue training only if short-run curves justify it. Compare gain relative to training and validation, including interior terminals (previously weak), junctions, boundaries, and crowded regions.
5. **Fine evidence for *relations* (new, not previously tested):** try a lightweight relation-head adapter supplied with the refined coordinates, and then/or pooled local image features for both endpoints. Compare to an equal-capacity adapter without image features. Keep the original decoder and node predictor fixed for a first diagnostic and use symmetric undirected pair scoring. The question is whether local evidence can improve long-positive versus nearby-negative discrimination *without* needing an entire multiscale transformer.
6. **Only after isolated benefit**, try partial unfreezing/end-to-end fine-tuning with an appropriate low LR and controlled initialization. Compare against a matched unfreezing control (no local-image access), so an improvement cannot be ascribed merely to training more weights.

**Measurements/gates:** node recall and precision at 1/2 mm, native node F1, raw query recoverability, duplicate predictions **per GT node**, confidence distributions, subtype recall, Curve/Branch F1, endpoint-pair recovery. For relation adapter: conditional M–M relation AP/PR and nearby-nonedge false-positive rate, plus *full decoded graph* edge F1, β₀/β₁ error and false cycle visualizations. **Keep** the approach if the validation gain repeats without growing duplicate/false-edge counts; **pause** architectural elaboration if gains vanish on full validation or mainly reflect selecting additional redundant nodes.

### P1-C — Explicit relation supervision for unmatched object queries

**Hypothesis motivated by observed extra nodes:** training-time Hungarian matching supervises relations primarily among queries assigned to GT nodes (M–M). At inference, all **retained** high-object-confidence queries can participate, including redundant/unmatched ones (M–U/U–U). This may explain extra edges **if** P0-C confirms a large U-involved false-edge burden. Unmatched queries already receive no-object classification loss; this is *additional relation supervision*, not a substitute for node uniqueness.

**Proposed starting objective (not an accepted final formulation):**

`L_edge = L_regularized_M-M + λ_U · L_unmatched_negative`

Retain the currently successful M–M regularized sampling **as a controlled baseline**; we can redesign it later, but should not change it simultaneously with the new U-involved branch. Normalize the auxiliary term on its own so its gradient does not erase positive-edge learning. For undirected graphs compute the same symmetric scores as inference. Top-k mining ranks on **detached predicted P(edge)** while optimizing original logits.

**Controlled small experiments:**

1. **No change:** original M–M edge loss and no U-involving term.
2. **Random U-negative control:** fixed number of U–M/U–U candidates, same training budget and auxiliary loss scale as mined arm.
3. **High-P(edge) U negatives:** probability-based hard-negative selection, exactly matched candidate budget; compare gentle weight ramps after a warmup rather than immediate dominance.
4. **Optional mixture:** random easy/medium negatives + selected hard negatives, or an explicit diversity cap to avoid repeatedly training on one ambiguous cluster; only if (3) collapses recall.
5. **Node-head interaction only if useful:** compare relation-head-only/adapter training with an otherwise matched arm that permits gradients into object-token features. In a *later separate experiment*, combine U supervision with duplicate-aware node classification; otherwise attribution is impossible.

**Important verification:** log the actual counts of M–M, M–U and U–U candidates; their logits, confidence and proximity to a vessel/GT, how many are near duplicates of true nodes, and the extra gradient norm relative to M–M. `Unmatched` is a **Hungarian-assignment result**, not always an anatomically false location; ambiguous GT or duplicate detections may yield unreliable negative labels. Avoid assuming every U-involved pair is a clean negative under imperfect annotations.

**Go/no-go:** fewer U-involved false decoded edges and spurious cycles on subject-held-out validation **without losing true edge recall or substantially increasing fragmentation**. If errors are overwhelmingly M–M or endpoint recovery is poor, this is not the lead intervention. Prior degree-consistency+unmatched+classification pilot changed multiple losses and cannot answer this isolated question.

### P1-D — Redesign matched–matched supervision only if M–M errors remain substantial

**Why revisit:** historical hard-negative experiments tended to remove many supervised pairs when reducing `max_per_graph`, or combined focal/hard mining with changed class balancing. These results do **not** prove that emphasizing informative M–M negatives cannot work. Conversely, they do not justify assuming hard mining is helpful. The existing loss is a starting point, **not sacred**.

**First audit the old loss mathematically and empirically:** `max_per_graph: 9999` is a **cap**, not 9,999 unique pairs; with 40 matched nodes there are at most 780 unordered pairs. Berger-style regularized sampling duplicates the minority class to achieve the desired positive/negative ratio, making some gradients recur. Capture per-patch number of GT positives, unique negatives, duplicates, reduction denominators, sampling randomness and aggregate positive/negative gradient contributions. Test analytic **class-weighted CE** against literal duplication under matched normalization as a numerical equivalence/efficiency control (not expected to be an accuracy breakthrough).

**Incremental training matrix (change only one ingredient per arm):**

1. **Baseline:** original positive/negative regularized edge loss and full available M–M pair pool.
2. **Same-coverage, probability-weighted CE:** retain all unique positives and negatives, but apply a smooth, bounded extra importance to high-`P(edge)` negatives (e.g., gentle focal-style hardness with class-normalized contribution); keep positive mass/total edge-loss scale controlled. Do *not* silently turn off positive upsampling in this comparison.
3. **Same-budget random versus hard:** compare sampling a fixed number of negative *updates* uniformly versus by high `P(edge)`, while retaining a full-coverage anchor loss. This separates 'hardness' from 'fewer examples'.
4. **Structured near-negative contrast:** preferentially contrast a GT-positive edge `(i,j)` against nearby **nonadjacent** `(i,k)` (particularly a shorter false connection competing with a longer true branch). Choose using GT structure + physical distance, optionally model scores. Record how often plausible positive vs negative pairs differ by local vessel appearance. A previously tried *simple 5× near-negative weight* did not improve global reconstruction; this proposed comparison is more specific.
5. **Positive/negative ranking (separate novel arm):** `L_edge = L_calibrated_CE + λ_rank·mean softplus(s_ik - s_ij + margin)` for positive `(i,j)` vs negative `(i,k)` sharing node `i`. Normalize independently, ramp gently, preserve true positive coverage, and compare a random-negative ranking control. Confirm it does not merely re-rank while destroying useful calibrated edge probabilities.

**Assessment:** matched-pair AP/PR (threshold-free), long-positive recall by physical arc length, false-positive rate among close GT nonadjacent pairs, *all-retained-query* edge F1, false cycles, fragmentation, Branch F1, node recall and training/validation gap. A locally lower FPR alone is not success if missing true long branches gets worse. Use the **same** split, seed, initialization, pair budget and effective class/gradient scaling before drawing conclusions.

**Gate:** do not prioritize M–M hard mining unless P0-C shows important errors remain between reliably localized GT nodes. Do not stack ranking, focal, unmatched mining and degree penalties in the first comparison.

### P2-A — Multiscale deformable transformer: explicit feasibility and efficacy gates

**Research hypothesis:** the observed value of early high-resolution CNN features suggests the original single-level (`2³` image-feature grid in the discussed diagnostic) transformer interface may not give decoder queries sufficient *spatially local* evidence. In a multiscale model, decoder cross-attention could sample projected CNN features from coarse and finer levels **during query updates**, potentially benefiting the object embeddings consumed by **both node and relation heads**. This has **not** been implemented or shown effective; high-resolution features are not automatically better for topology.

**Why not start with the full multiscale redesign:** the cheaper local-refinement head already helps nodes, while duplicate retention and uncertain relation labels may be independent of feature resolution. A full 3D multiscale deformable encoder also has implementation and GPU costs, and is a major confound if changed together with new losses.

**Feasibility study — mandatory before a long training run:**

1. **Inspect actual model tensors and code (no training):** print backbone feature stages, shapes, channels, strides, receptive-field context, positional encodings, projection path, deformable-attention CUDA operator assumptions and decoder interface for an actual `64³` IXI input. Treat the illustrative `2³ + 8³ + 16³` selection as a proposal, **not** documented working shapes. Identify one feasible additional level (e.g., 8³ first), keeping the original coarse level.
2. **Image/coordinate correctness checks (CPU/GPU smoke):** verify voxel↔feature-grid↔normalized reference-point transforms at every level, anisotropic voxel spacings, input-axis ordering, border interpolation, out-of-range sampling, valid masks/level offsets and non-empty gradients to projected fine features. Use tiny known-position synthetic patterns (e.g., a bright point or straight vessel) to check sampled locations—not just a shape test.
3. **Smallest functioning architecture:** retain existing CNN and most encoder/decoder design; add **decoder-only second-scale cross-attention** to `2³ + one finer feature level` while retaining deformable sparse sampling. Reuse original pretrained weights where compatible; initialize new projections/adapters explicitly and log which modules are new. Check whether the 3D CUDA kernel supports multiple levels or needs a modified implementation. **Do not naïvely concatenate full 16³/32³ grids into dense attention.**
4. **Resource and stability gate:** on the same H100 and batch/patch size as the control, measure peak memory, iteration latency, numerical stability, forward/backward success and gradients through both coarse and fine branches. If memory is prohibitive, consider fewer scales/lower-resolution features/decoder sampling-point budgets rather than a full architecture rewrite.
5. **Tiny fitting sanity check:** test whether the new model can overfit the **same four real IXI training patches** without collapse, matching controls in optimizer schedule and evaluation. Check duplicate node count and recovered GT anchors—not only training loss. This demonstrates trainability, **not** generalization.
6. **Controlled subject-held-out pilot:** fix matched update budget, training data, GT graph representation, Hungarian matcher, edge/node losses, confidence decoding, augmentation, checkpoint selection and evaluator. Compare (a) original single-level model, (b) existing cheap local refinement, (c) decoder-only 2-scale attention. Include (d) an adapter with the *extra parameter count but no fine features* if feasible to test whether benefit comes from image resolution rather than capacity or optimizer changes. Use an explicit fair checkpoint initialization/fine-tune protocol; log what pretrained tensors transferred.
7. **Decide whether finer decoder features help relations:** examine node location and subtype recall, duplicates around GT anchors, **conditional** M–M relation AP and long-positive/near-negative discrimination, plus **full decoded** edge/Branch/Curve F1, β₀/β₁ and cycle visualizations. Relate any topology gain to endpoint recovery versus genuine relation-head improvement. Compare against **feeding local refined features directly to the relation head** (P1-B step 5), which is significantly cheaper.
8. **Only if decoder-only multiscale wins:** try a 3-scale decoder or multiscale encoder–decoder as a *separate* ablation, then larger/full-validation, multi-seed training. Do not make multiscale encoder, node losses, matcher and relation loss changes together.

**Go/no-go decision rules:**

- **Go forward:** correct feature alignment and gradients, feasible H100 budget, stable fitting, and reproducible *subject-held-out* improvements over both the original single-level control and cheaper local refinement, preferably in genuinely missing nodes and/or relation AP **without more duplicates, extra cycles or large fragmentation**.
- **Pause/stop:** new scales do not reach decoder queries correctly; cost is disproportionate to improvement; only training scores rise; validation gains disappear by subject/site; extra fine tokens sharpen positions while retaining duplicate predictions; or full-graph topology deteriorates despite better node F1.
- **Ambiguous:** local refinement wins for nodes but decoder-only multiscale helps relations. Explore a hybrid or local-feature relation adapter **before** committing to a full multiscale encoder.

**Status:** proposal only. Multiscale transformer has **not** been tested in the supplied experiments. Treat success criteria as pre-declared research decisions, not empirical findings.

### P2-B — Longer-term matching or node-uniqueness redesign if duplicates dominate

Options to investigate **only after** P0-B/D: strengthen no-object discrimination/calibration without harming true-node recall; controlled query-repulsion or duplicate-aware training objective; duplicate classification against well-separated GT anchors; revisiting matcher cost/assignment under duplicate clusters (including any Gromov–Wasserstein alternative) **without** changing data/losses at the same time. Do not introduce an unverified hard minimum node separation in tortuous or densely branched vasculature.

**Gate:** must improve subject-held-out node uniqueness and downstream graph topology beyond GT-free NMS, at acceptable recall. Matcher changes affect which nodes receive location and relation supervision; benchmark against an identical initialization and training schedule.

## 4. Minimal evaluation dashboard for every candidate

Report **per patch, per subject, and pooled** metrics on the same split/protocol, with confidence intervals or subject bootstrap when possible:

| Group | Metrics (with explicit definitions) |
|---|---|
| **Nodes** | GT node recall/precision/F1 at 1 and 2 mm; raw 256-query recoverability; count bias; duplicate clusters per GT; distance-to-mask; subtype (junction, degree-2, interior terminal, boundary) |
| **Edges** | Positional TP/FP/FN and F1; **M–M / M–U / U–U** FP counts; conditional direct-relation recall among recovered endpoint pairs; long-positive recall; near-nonedge FPR; relation AP/PR curve |
| **Graph topology** | β₀ and β₁ count errors **plus** Branch F1 and explicit edge correctness; degree histogram, fragmentation, false-shortcuts/bridges, real 3D visual inspection |
| **Geometry** | Curve F1 and same-endpoint straight-chord baseline; ACD/HD95 where available; distinguish geometry improvement from endpoint correction |
| **Generalization** | Train vs subject-disjoint validation; acquisition site/resolution/node density; early, best and final checkpoint trends; two or more seeds for leading candidates |
| **Cost** | Unique scored candidate pairs, duplicated class counts, trainable parameter count, effective edge-loss/gradient scales, GPU memory and updates/sec |

**Critical rules:**
- A lower β₁ MAE is **not** proof of correct connections; adding/removing edges can make cycle counts look better while actual branches remain wrong.
- Edge/matcher metrics calculated on GT-recovered endpoints are **conditional** and must be accompanied by endpoint coverage; do not compare their values directly with full-graph decoded metrics.
- SMD/Curve coverage can benefit from overprediction; never take apparent curve coverage as proof of useful or correct topology.
- Prefer validation-driven decisions. The historical IXI test set has already been inspected/evaluated in past reports; document that exploration and reserve a new independent final holdout if making new generalization claims.

## 5. Suggested execution order and decision tree

```text
[0] Reproduce frozen checkpoint + same prediction cache
 |
 v
[1] Duplicate/confidence audit + [2] FP edges by M-M / M-U / U-U
 |
 v
[3] Oracle GT-guided node pruning + GT-free NMS control
 |
 +-- Oracle pruning greatly helps and U-involved FP dominate
 |      -> prioritize node uniqueness/confidence;
 |         try isolated unmatched-negative relation supervision;
 |         only then consider matcher/duplicate-aware objective.
 |
 +-- Oracle pruning barely helps; M-M errors dominate
 |      -> prioritize relation scores and hard matched distractors;
 |         test structured negative-vs-positive ranking.
 |
 +-- Missing GT endpoints dominate even after pruning
 |      -> binary-mask input oracle + local-feature isolation;
 |         if fine image evidence helps, try fine-feature-to-relation
 |         then multiscale decoder/transformer.
 |
 +-- Significant train/val gap in every branch
        -> subject/site/image-statistics/GT QC;
           reconsider generalization strategy before bigger losses.

Only promote well-controlled validation winners to longer, multi-seed
training, then to a separately protected final evaluation.
```

**Practical next session:** start with **P0-A → P0-B → P0-C → P0-D**, which can be done largely from frozen predictions. In parallel only if compute/people permit, prepare (but do not over-interpret) **P1-A mask-input** as the cleanest new learning diagnostic. Do not run P1-C/P1-D/P2 indiscriminately in parallel: the result of node/edge error attribution should determine which path to follow.

## 6. Implementation checklist (for Codex / future project sessions)

- [ ] Locate the **live** active `representation` branch and compare to the dated archive before editing; archive code is a snapshot and newer scripts such as `models/node_refinement.py` may not be present there.
- [ ] Read `docs/research/metrics/{1-ixi-supervision-audit,2-ixi-feature-node-diagnostics,3-ixi-node-refinement-pilot,4-ixi-relation-isolation,5-ixi-threshold-calibration}.md` and the relevant canonical experiment notes.
- [ ] Confirm model/checkpoint/config/GT patch manifest; verify current configured matcher (Hungarian vs optional alternatives), node refinement call order, and frozen/current inference thresholds.
- [ ] Reuse `infer_graphs()` logic to cache full query probabilities/pair scores **before filtering**; ensure query IDs are stable and relation scores use same symmetric pair orientation as actual inference.
- [ ] Reuse existing `metrics/` physical node matching and branch tools, not a new inconsistent metric definition; add CPU crafted tests for duplicate nodes, boundary cases, ties, empty graphs, and oracle deletion.
- [ ] Run P0-B/C/D on a predeclared small subject-disjoint validation subset, inspect fixed examples in 3D, then expand to full validation with unchanged definitions.
- [ ] Commit only compact `.md`/CSV/JSON summaries and representative views. Keep full prediction caches and checkpoints on SCRATCH, and avoid large sweeps on Jean Zay until a diagnostic supports them.
- [ ] For each P1 experiment, record exact changes from control, initial-state hashes, dataset order, optimizer settings, number of *unique* supervised pairs, loss normalizations, gradient checks, and training/validation plots.

## 7. Evidence and code references

The following are **repository-relative** paths in the live `Graph-Native-Betti-Matching` project (use the live canonical files if newer):

- `docs/research/metrics/1-ixi-supervision-audit.md` — foreground/GT support and endpoint versus relation accounting.
- `docs/research/metrics/2-ixi-feature-node-diagnostics.md` — CNN vs coarse transformer feature probes, node subtypes and duplicates.
- `docs/research/metrics/3-ixi-node-refinement-pilot.md` — 27-point early-CNN sampling, paired controls and observed gains.
- `docs/research/metrics/4-ixi-relation-isolation.md` — endpoint-conditioned relation AP, nearby negatives, long positives.
- `docs/research/metrics/5-ixi-threshold-calibration.md` — threshold trade-off, fragmentation and false-cycle cost.
- `docs/research/pretraining/ixi-overfit.md` — real IXI four-patch fitting test.
- `docs/research/metrics/relation-distance-pilot.md` — controlled 5-mm negative-weighting ablation and its failure to improve whole-graph metrics.
- `docs/research/evidence/degree_consistency/README.md` and `docs/research/evidence/T10_node_type_pilot.md` — earlier IXI overprediction, node types and confounded unmatched-negative comparisons.
- `docs/LOSSES.md`, `docs/MODEL.md`, `docs/EVALUATION.md`, `docs/IXI_DATASET.md` — implementation/data/evaluation contracts.
- `training/losses/criterion.py`, `training/losses/focal.py`, `training/evaluation/inference.py`, `models/matcher.py`, `models/relationformer.py`, `models/node_refinement.py` (latter reported in newer docs) — code points to verify.
- Original concepts: **Shit et al., RelationFormer (2022)**; **Berger et al., Cross-domain and Cross-dimension Learning for Image-to-Graph Transformers (2024)**; **GrEVE** manuscript supplied for the broader research background.

---

### Decisions to revisit after the first P0 results

1. Is the dominant failure **duplicate nodes that survive object filtering**, **missed true nodes**, or **wrong relations between otherwise correct nodes**?
2. How much graph quality improves after *oracle removal of extras*, with no change to relation scores?
3. Does an input segmentation mask resolve most of the train/validation gap, or does the model still fail to build the graph?
4. Only then: which single intervention has the highest expected information value—unmatched hard negatives, duplicate-aware object training, local-image relation features, structured ranking, or multiscale decoding?

**Do not treat any of these as already answered.**


## Completed matching and calibration follow-up

Cache-only retrospective Hungarian and five-head training-calibrated evaluation completed; see [report 8](metrics/8-ixi-relation-matching-calibration.md) for measured results, uncertainty and revised conditional priorities. No further training or architecture change was launched. Earlier conclusions remain recorded above.
# 2026-10-20 plan update: controlled node weighting/focal transfer

New colleague evidence motivates a three-arm IXI node-loss continuation (CE [.25,.75], CE [1,1], focal gamma2 [1,1]), with common CE edge supervision and ratio0.15. See [report10](metrics/10-ixi-node-focal-ablation.md) for the initialization/edge-transition limitation, fixed subject-disjoint manifests, uniqueness endpoints and submission record. This is a learning-curve pilot; prior findings are retained below. Image-information A–D in [report9](metrics/9-ixi-image-aware-relation-learning.md) remain unchanged. Do not repeat the small frozen-feature CE continuation, synthetic focal matrix or unmatched-mining sweep. Edge focal+ratio is deferred pending this node study rather than added automatically. Completed results and revised evidence-based priorities will be appended by the collectors.



Completed the larger image-aware relation-adapter batch; see [report 9](metrics/9-ixi-image-aware-relation-learning.md) for ranking/calibration results and conditional priorities. Previous findings remain recorded. No transformer/mask-input experiment was included.

Completed the controlled IXI node class-weight/focal continuation; see [report 10](metrics/10-ixi-node-focal-ablation.md). Interpret learning curves and uniqueness before extending loss studies; no new edge sweep was included.
