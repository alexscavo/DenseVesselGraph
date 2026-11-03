# Learning continuous (curved) edges with RelationFormer — consolidated history (to 29 Oct 2026)

**Status.** Covers the spline-learning lineage of 5–13 Oct 2026 (SyntheticMRI, RelationFormer):
spline heads, losses, gradient routes, the scratch collapse and its diagnosis, local-image geometry,
Catmull–Rom, and curve-informed connectivity. Consolidated on 3 Nov 2026 from documents archived
under `docs/archive/legacy-20261103/docs/research/spline_learning/`. **Superseded** as the
project's main model by the [dense traced model](dense_traced_model_20261102.md). Its recipe
conclusions survive inside RelationFormer B0, and the oracle-geometry study
([`network_proposals_20261030.md`](network_proposals_20261030.md) §0, §6.7) later confirmed them.

## Context

The question was whether a graph with only junction, termination and crop-boundary nodes can
represent tortuous vessels if each edge carries a continuous curve, and whether RelationFormer
can *learn* that curve from the image. Almost all experiments use **SyntheticMRI** (64³ patches,
subject-disjoint splits; after the dataset restoration, 64 train / 13 validation / 14 test
subjects). A few single-case fitting studies use **IXI**, and a target-capacity audit uses **IXI
and BRAVA**. The model is always RelationFormer: 120 object queries, two relation tokens, and a
spline MLP added beside the relation head.

The protocol evolved:
- **First**, four training patches.
- **Then**, a 10k-patch fine-tune with a held-out *test* evaluation.
- **Then**, 300-epoch scratch runs, also scored on the held-out *test* set.
- **Finally**, short paired diagnostics (5k–25k updates). These used fixed monitor sets of
  512 training and 128 validation patches, and the full 1,952-patch validation split for
  confirmation. They did not use the test set.

Guides (the generator's dense centrelines) and masks are uncertain references, not anatomical
ground truth. Every result is single-seed.

Ground-truth representation questions are covered in
[`history_representation.md`](history_representation.md) and are not repeated here:
junction-only vs adaptive vs dense targets, and how many controls a target needs (IXI/BRAVA K8
capacity).

## Shorthand

- **K8 / K10.** Clamped cubic B-spline with 8 (or 10) controls. The 2 end controls are the
  predicted nodes; the interior controls (6 for K8, giving 18 scalars per 3D edge) are
  predicted. Interior controls start on the chord at the **Greville abscissae**
  `0, 1/15, 1/5, 2/5, 3/5, 4/5, 14/15, 1`.
- **V0 / V1.** The two first spline losses:
  - V0 is smooth-L1 (β = 1 mm) on the fitted interior *control* coordinates;
  - V1 is smooth-L1 on 64 *sampled curve* points.
  - V2 (per-control shared MLP) and V3 (image-attending control queries) were **PROPOSED,
    never run** ([design](../archive/legacy-20261103/docs/research/spline_learning/representation-implementation.md)).
- **Absolute vs residual target.**
  - Absolute: the loss compares predicted and GT control positions, so it also penalises
    endpoint (node) error.
  - Residual: the endpoint error, interpolated at each Greville position, is subtracted, so
    only the deviation from the chord is supervised. This is the later B0 `target: residual`.
- **Gradient routes** (for the geometry loss):
  - `graph_only` / "detached": geometry sees detached tokens and endpoints, so it never
    updates the detector;
  - `tokens_only`: geometry gradients reach the shared tokens but not the endpoint
    coordinates;
  - `joint`: geometry gradients reach both.
- **Frozen detector.** The backbone, transformer and node/relation heads have fixed weights and
  BN statistics, and only the geometry head trains.
- **Same-endpoint chord.** The same predicted graph drawn with straight edges. It separates a
  curve's shape gain from changes in the graph.
- **Endpoint-neutral error.** The predicted deviation from the chord, transplanted onto the GT
  chord and compared with the guide. It measures shape independently of node localization; it
  is an evaluation device only.
- **Native node/edge F1.** RelationFormer's box-based detection metric: IoU 0.5 with node size
  0.2 in normalized coordinates. It is *not* the later physical "node F1 @2 mm".
- **Common-v2 (`shared-guides-v2`).** The representation-independent scorer:
  - degree-2 chains are contracted;
  - nodes are matched one-to-one, role-aware, within **1 mm**;
  - **Branch F1** counts recovered complete branches;
  - **Curve F1** counts guide samples within 1 mm, at 0.25 mm spacing.
  - The later "common-v2 Branch / Curve F1" in the current docs descends from it.
- **Local-image geometry / local corrector.** A small module that samples the detector's first
  CNN activation (64 channels, 32³, before max-pool, detached) at 32 points along the current
  curve. A 1D conv plus MLP then adds a mm correction to the interior controls. This is B0's
  "local spline corrector".
- **Image sensitivity.** The mean absolute difference of a representation (CNN features,
  transformer memory, tokens) between different input images on a fixed probe. Near zero
  means the model ignores its input ("collapse").
- **B0.** The later baseline RelationFormer recipe: K8 head, residual target, local corrector,
  `graph_only` route ([`network_proposals_20261030.md`](network_proposals_20261030.md) §1B).
- **Oracle-geometry study.** The 30 Oct experiment that gives the true graph and learns only
  the edge shape from image-corridor features
  ([`network_proposals_20261030.md`](network_proposals_20261030.md) §0).

## 1. Can a curved edge be learned at all? [DONE]

**Four-patch overfit, V0 vs V1, fine-tuned vs scratch.** Four training patches (one subject,
79 branches), 500 updates. The fine-tuned arms start from the SyntheticMRI-only 300-epoch
straight-edge checkpoint (855,000 updates). The K8 oracle fit is 0.0242 mm from the guides.
([results](../archive/legacy-20261103/docs/research/spline_learning/spline-learning-results.md),
[selection](../archive/legacy-20261103/docs/research/spline_learning/spline-approach-selection.md))

| Arm | Curve error vs fitted target (mm) | Same-endpoint chord (mm) | Symmetric distance to guide (mm) |
|---|---:|---:|---:|
| V0, fine-tuned | 0.1093 | 0.4812 | **0.1010** |
| V1, fine-tuned | 0.1311 | 0.4821 | 0.1214 |
| V0, scratch | 0.3811 | 0.4803 | 0.3498 |

- All three arms recovered the full topology (98 nodes, 79 edges), so detection F1 could not
  separate them.
- **Decision:** V0 control supervision with joint fine-tuning was chosen as the main approach,
  and V1 was kept as an ablation. This was a 4-patch training-set result, not a held-out
  comparison. V1 was never revisited.
- **Scaling with a pretrained detector [DONE, test split used].** V0 was fine-tuned for 50
  epochs on 9,709 training patches. The validation-selected checkpoint was then evaluated
  once on **13,071 held-out test patches** (the 14 test subjects then available).
  - Native node/edge F1: 0.8841 / 0.8157.
  - Strict 1-mm physical branch F1: only 0.1698, with **17.41%** of target branches
    recovered.
  - Matched-branch guide distance: 0.6396 mm, on that small recovered subset only.
  - The gap between box-based detection and physical branch recovery recurs in every later
    experiment.
  ([10k protocol](../archive/legacy-20261103/docs/research/spline_learning/spline-10k-training.md))

## 2. Scratch: straight vs K8 spline vs adaptive degree-2 [DONE — spline arm failed]

**Setup.** Three arms trained from scratch for 300 epochs (750,000 updates) on the same 10,000
training patches. They share the seed, schedule and graph-loss weights; the spline arm adds
weight 1.474. Arms:
- straight chords;
- K8 V0 (absolute target, joint route);
- **compact adaptive degree-2**: straight segments with degree-2 nodes chosen by an
  error ≤ max(0.75·radius, 0.1 mm) rule. All of its targets fit the 120-query budget.

Scoring: common-v2 on the 13,071 **test** patches. Checkpoints were selected by validation
graph loss, not on test.
([results](../archive/legacy-20261103/docs/research/spline_learning/scratch-comparison-results.md),
[protocol](../archive/legacy-20261103/docs/research/spline_learning/scratch-comparison-300ep.md),
[adaptive arm](../archive/legacy-20261103/docs/research/spline_learning/adaptive-degree2-scratch-comparison.md))

| Final epoch 300 (test) | Straight | K8 spline | Adaptive degree-2 |
|---|---:|---:|---:|
| Anchor F1 | 0.1353 | 0.0001 | 0.1031 |
| Branch F1 | 0.0114 | 0.0000 | 0.0050 |
| Curve F1 | 0.2305 | 0.0014 | 0.2471 |
| Native node / edge F1 | 0.653 / 0.545 | 0.000 / 0.004 | 0.567 / 0.511 |

- **Adaptive vs straight:** adaptive had slightly higher Curve F1, and straight had higher
  anchor and Branch F1. There was no overall winner and no significance test.
- **All arms were weak.** Topology recovery at 1 mm was weak for every arm: straight-edge
  branch coverage was 0.94%.
- **Checkpoint selection was misaligned.** The validation-loss-selected checkpoints (epochs
  35 / 170 / 25) scored *worse* than the final ones, so selecting by graph loss does not track
  reconstruction quality.
- **The spline arm collapsed.** Its predictions were essentially independent of the input:
  the same 38 nodes and 3 edges on every patch.

## 3. Why did the scratch B-spline collapse? [DONE — mechanism partly isolated]

**What the failure was.** The spline run never fit its own training set: node loss stayed
around 0.21–0.24 from early training (epoch 300: train/val 0.20171 / 0.20324, versus
0.00731 / 0.07341 for straight edges). Saved predictions on two patches differed by at most
1.79e-7 in normalized coordinates.
([diagnosis](../archive/legacy-20261103/docs/research/spline_learning/spline-collapse-diagnosis.md))

**Ruled out or localized (diagnostics on the final checkpoints) [DONE]:**
- *Stale inference BN statistics* are ruled out: using batch statistics left the outputs
  unchanged.
- *The failure sits in the transformer encoder*, not the CNN. The CNN feature difference
  between two images is 0.85446; it falls layer by layer to 3.42e-8 at the encoder output.
  The straight model keeps 0.87522. Poor deformable-attention coverage was present in the
  healthy straight model too, and resetting the decoder offsets did not help, so it is not
  the explanation.
- *The geometry gradient was large but aligned.* On one initial batch, the weighted
  spline/graph gradient ratios were 1.35 at the tokens, 7.49 at the node coordinates and 1.24
  at the image features, with coordinate cosine +0.779. That is scale imbalance, not opposing
  gradients.

**Short interventions [DONE — negative or inconclusive]:**
- **10× smaller spline weight** (scratch, 512/128 patches, 5k updates): both arms still had
  node F1 0. Curve F1 was 0.00145 at weight 1.474 vs 0.00081 at 0.1474. Neither arm
  collapsed in 5k updates, so the test could not settle the cause.
  ([weight pilot](../archive/legacy-20261103/docs/research/spline_learning/spline-weight-pilot.md))
- **Staged ramp from a learned graph** (straight scratch checkpoint at epoch 35, spline ramped
  in over 2,500 updates): no collapse, but no shape benefit either. Spline Curve F1 was
  0.15400 vs 0.15448 for its own chords, and both arms regressed slightly from the start.
  ([staged](../archive/legacy-20261103/docs/research/spline_learning/spline-staged-learning.md))

**The decisive paired scratch tests (25k updates, 10k patches, residual target and local
corrector in both) [DONE]:**

| Arm (128 val patches) | Node F1 | Edge F1 | Curve F1 | Memory image sensitivity |
|---|---:|---:|---:|---:|
| Graph-only control (spline weight 0) | 0.2364 | 0.1957 | 0.0564 | 0.894 |
| Geometry through tokens (`tokens_only`) | 0.0000 | 0.0083 | 0.0040 | 3.4e-6 |
| Geometry detached (`graph_only`) | 0.2057 | 0.1928 | 0.0561 | 0.850 |

- With `tokens_only`, sensitivity fell from 0.8491 at update 4k to 0.0016 at 9k, even with
  residual targets, local features and detached endpoints.
- Detaching the geometry avoided collapse. Real-model checks confirmed that detached geometry
  gradients into the detector are exactly zero (569 tensors receive gradient under the token
  route). The small detector differences matched repeated-run numerical noise. Batch-statistics
  BN did not revive the collapsed model.
- **Conclusion:** the collapse is linked to *geometry gradients reaching the shared tokens from
  random initialization*. The internal mechanism was never identified.
  ([scratch local](../archive/legacy-20261103/docs/research/spline_learning/scratch-local-geometry-diagnostic.md),
  [detached](../archive/legacy-20261103/docs/research/spline_learning/scratch-detached-geometry.md))

## 4. Does the spline head help or hurt the detector? [DONE; later confirmed]

**Pretrained, controlled unfreezing.** Starting from a frozen-detector residual head, three
routes were each trained for 5k updates: `graph_only`, `tokens_only` and `joint`. The
transformer, queries and graph heads were unfrozen; the CNN and BN stayed fixed.
- None of the routes collapsed; memory sensitivity stayed about 0.92.
- On full validation (1,952 patches):
  - edge F1: frozen 0.86363, `graph_only` 0.87198, `tokens_only` 0.87096;
  - Curve F1: 0.68441, 0.67955 and 0.68296 respectively.
- So geometry gradients into the detector gave **no demonstrated benefit**. The working rule
  became: train the detector on graph losses, and adapt the geometry head on detached inputs.
- Unfreezing with the local corrector gave the same picture: the three routes were within
  about 0.002 Branch F1 of each other on full validation.
  ([unfreezing](../archive/legacy-20261103/docs/research/spline_learning/controlled-unfreezing.md),
  [full validation](../archive/legacy-20261103/docs/research/spline_learning/full-validation-comparison.md),
  [local follow-up](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-followup.md))

**Confirmed by the current work.** This rule became B0's `graph_only` route; it was first
implemented in `scripts/controlled_spline_unfreeze.py` in this lineage. On 2 Nov, a
straight-edge RelationFormer was trained with B0's recipe and spline weight 0. Its node, edge
and Branch F1 were within ±0.005 of B0 on all three 3D datasets; the spline added only Curve F1
(+0.015 to +0.048). "Training with or without splines gives identical graphs"
([`network_proposals_20261030.md`](network_proposals_20261030.md) §6.7,
[report](../../experiments/b0_straight_dense_20261101/report.md)). Under `graph_only` the
spline head *cannot* hurt the detector. The old harm (§3) occurred only with token-coupled
geometry from scratch.

## 5. What should the geometry loss supervise? Absolute vs residual targets, token inputs, data breadth [DONE]

**Setup.** Frozen pretrained detector (the 855k-update straight checkpoint), head only, 5k
updates.
([frozen](../archive/legacy-20261103/docs/research/spline_learning/frozen-spline-investigation.md),
[factorial](../archive/legacy-20261103/docs/research/spline_learning/residual-token-diagnostics.md))

Caveat: the 1,952 candidate validation patches belonged to the pretrained checkpoint's
validation population, so they were "diagnostic validation", not an unbiased test.

- **Absolute targets learn localization compensation.** Training guide error improved 40.5%
  over chords, but validation got 8.3% worse. Decoded validation Curve F1 fell from 0.65540
  (chords) to 0.58567. The endpoint-neutral error *worsened*, so the head was mostly bending
  curves to fix node errors, which does not transfer.
- **Residual targets learn shape.**
  - With full tokens on 512 patches, training endpoint-neutral error was 50.7% below the
    chords, but validation still did not beat them (Curve F1 0.61590).
  - Zeroing the token inputs gave chord-like behaviour: no fit and no harm.
- **Breadth fixes the overfit.**
  - Training the residual head on 10k patches instead of 512, at the same update budget, gave
    validation Curve F1 0.67865 vs 0.65540 for chords, with endpoint-neutral +6.07%.
  - At 15k updates: 0.68995 and +10.27%.
  - An absolute-target head at the matched 10k / 5k-update budget reached 0.67461, with
    endpoint-neutral −3.64%.
  - Only about half of the branches improved individually.
- **Carried forward:** the residual target became B0's `target: residual`.

## 6. Does local image information help geometry? [DONE; later confirmed and extended]

**Local corrector on a frozen detector.** Three paired arms, 5k updates: the token-only head;
plus the local corrector; plus the same corrector fed zeroed features (a capacity control).
([diagnostic](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-diagnostic.md),
[follow-up](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-followup.md))

| Full validation (1,952 patches) | Curve F1 | Endpoint-neutral error (mm) |
|---|---:|---:|
| Token head only | 0.68411 | 0.46036 |
| + local image corrector | **0.84012** | **0.35986** |
| + corrector with zeroed features | 0.68475 | 0.46009 |

- The local corrector cut endpoint-neutral error versus same-endpoint chords by 30.1% overall
  and by 47.2% on 3,116 curved long branches. The detector was unchanged: native node F1
  0.91041, edge F1 0.86363, common Branch F1 0.27625. The gain is information, not capacity,
  since the zero-feature arm matches the token head.
- **Short branches got worse.** On the 128-patch subset, local 0.1016 mm vs chord 0.0669 mm.
  Inspected cases still looked nearly straight on many branches.
- **The SyntheticMRI curvature caveat** was raised by the user. On 1,509 validation branches
  ≥5 mm, median fitted tortuosity is 1.0159 and only 2.5% exceed 1.10. Small average gains
  on SyntheticMRI therefore cannot diagnose a failure of curvature learning. IXI was
  explicitly deferred because no reliable IXI detector or continuous targets existed then.
- **Longer pretrained continuation (25k more updates, `graph_only`, full validation) [DONE].**
  - Geometry was stable: Curve F1 0.843 → 0.845, endpoint-neutral 0.353 → 0.343 mm.
  - Branch F1 fell from 0.2926 to 0.2814 and β₀ error rose.
  - Training-monitor Branch F1 rose from 0.321 to 0.392: the connectivity was overfitting.
  ([stability](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-stability.md))

**Confirmed and extended by the oracle-geometry study (30 Oct).** With the true graph given, a
K8 head reading image features in a corridor around the chord was 2–3× closer to the centreline
than the chord, on real data too:
- SyntheticMRI 0.519 → 0.151 mm;
- IXI 0.537 → 0.186 mm;
- BRAVA 1.479 → 0.615 mm (K24: 0.49).

This resolves the old "SyntheticMRI is too straight, IXI deferred" caveat. The dense traced
model then made path-sampled image features its geometry input
([`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) §4.5).

## 7. Scratch vs pretrained-and-finetuned [DONE]

**Setup.** The detached scratch pilot (§3) was extended to 300 epochs, paired with a
graph-only scratch control. Both use local corrector, residual target and `graph_only` route.
Total cost was about 37 GPU-h for both runs. Scored on full validation; the test set was not
used.
([extension](../archive/legacy-20261103/docs/research/spline_learning/scratch-extension-300.md),
[CSV](../archive/legacy-20261103/docs/research/spline_learning/scratch-extension-final-results.csv))

| Full validation | Node F1 | Edge F1 | Branch F1 | Curve F1 (own chords) |
|---|---:|---:|---:|---:|
| Scratch graph-only control, epoch 300 | 0.634 | 0.519 | 0.0079 | 0.202 |
| Scratch detached K8, epoch 300 | 0.650 | 0.542 | 0.0118 | 0.343 (0.226) |
| Pretrained detector + frozen local head | 0.910 | 0.864 | 0.276 | 0.840 (—) |
| Pretrained + 25k `graph_only` continuation | 0.909 | 0.875 | 0.281 | 0.845 (—) |

- **Scratch learning works with detached geometry.** There was no collapse in 300 epochs
  (memory sensitivity 0.869), and curves added +0.116 Curve F1 over their own chords.
- **Shape overfits.**
  - Final endpoint-neutral error was 4.32% *worse* than chords on validation, despite 57.64%
    better on the training monitor.
  - The epoch-170 checkpoint kept +2.37%.
  - On the 128-patch subset, curved long branches improved 22%, while short branches got
    117% worse.
- **Topology is not solved.** Branch F1 stayed near 0.01, with about 10.9 predicted vs 16.4
  target branches per patch.
- **The comparison is confounded.** The pretrained detector saw 91,200 patches for 300 epochs,
  and the validation subjects were in its own validation population. The gap measures the
  value of that prior graph training, not of geometry.

**Superseded.** The dense traced model trains from scratch without pretraining and beats every
RelationFormer variant ([`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) §1).
The later literature-scale B0 reruns also showed RelationFormer underfits real data
([`network_proposals_20261030.md`](network_proposals_20261030.md) §0). The "pretraining is
needed" question is now moot for the chosen model.

## 8. Can curves inform connectivity? [DONE — superseded]

**Setup.** Before edge selection, a candidate K8 curve was predicted for each pair. MRI
features from a new small full-resolution encoder were sampled along that curve, and a
zero-initialized correction was added to the relation logits. Image evidence was switched
off/on in otherwise identical joint fine-tunes: same epoch-35 straight checkpoint, 512/128
patches, 5k updates.

**Result.**
- Native edge F1: 0.46077 → 0.50654.
- Curve F1: 0.15971 → 0.21339 (own chords 0.15729 → 0.21172).

Image evidence improved *connectivity*, but the shape contribution was small. Both arms shared
a change to symmetric relation scoring, so the result is not comparable with older pilots.
([curve-informed connectivity](../archive/legacy-20261103/docs/research/spline_learning/curve-informed-connectivity.md);
results are in the table of
[frozen-spline-investigation](../archive/legacy-20261103/docs/research/spline_learning/frozen-spline-investigation.md))

**Node localization was the bottleneck.** In the staged-pilot models only 9.7–11.9% of GT
nodes were within 1 mm, so only 12–20 of 2,059 reference edges had both endpoints recovered.
This anticipates the later finding that about 90% of missed edges lack an endpoint pair
([`network_proposals_20261030.md`](network_proposals_20261030.md) §0).

**Superseded by the dense traced model.** Its edge classifier and geometry both read image
features along a candidate path (a corridor, then the traced centreline). That gives Curve F1
+0.10 over 16-NN candidates on IXI and BRAVA
([`dense_traced_model_20261102.md`](dense_traced_model_20261102.md) §1).

## 9. Catmull–Rom vs B-spline [DONE as fitting; training not evaluated; superseded]

- **Fitting, two single cases [DONE].** Centripetal Catmull–Rom (interpolating, α = 0.5)
  and K8/K10 B-splines (least-squares handles) were compared at equal budget, against the
  guides.
  - IXI (12 branches): the B-spline was better. At 8 points, B-spline 0.03391 mm vs
    Catmull–Rom 0.05460 mm; tortuosity error 0.01058 vs 0.01925.
  - SyntheticMRI (19 mild branches): the two were about equal (0.02295 vs 0.02459 mm at 8
    points; Catmull–Rom slightly better at 10).
  - The targets were built differently: Catmull–Rom used equal arc-length guide samples, not
    least squares.
  ([single cases](../archive/legacy-20261103/docs/research/spline_learning/catmull-rom-single-cases.md))
- **Scratch 300-epoch Catmull–Rom training [RUNNING when last documented; no final result
  in the docs].** Same protocol as the collapsed K8 scratch arm (absolute target, joint route,
  weight 1.474).
  - Its run directory's `training-status.json` records completion at epoch 300, but no
    evaluation was ever documented.
  - Its logged epoch-300 validation node loss (0.203, in `validation-losses.jsonl`) matches the
    collapsed B-spline arm (0.20324). It probably failed the same way, but this was not
    verified.
  ([learning](../archive/legacy-20261103/docs/research/spline_learning/catmull-rom-learning.md))
- **Superseded.** The oracle-geometry study found that the curve family matters little. At
  equal output count a learned polyline matched or slightly beat the K8 B-spline on mean
  distance, and the spline was slightly better on Hausdorff
  ([`network_proposals_20261030.md`](network_proposals_20261030.md) §0). What matters is
  that the geometry is an edge attribute, conditioned on the image along the path.

## 10. Single-case representation viewers [DONE]

Offline 3D viewers fit curve families to the guides with the true topology given. They test
representability, not learning.
([IXI](../archive/legacy-20261103/docs/research/spline_learning/ixi_patch.md),
[SyntheticMRI](../archive/legacy-20261103/docs/research/spline_learning/synthetic_mri_patch.md),
[README](../archive/legacy-20261103/docs/research/spline_learning/README.md))

- **IXI patch** (272 dense nodes, reduced to 14 anchors and 12 branches). On the most tortuous
  branch (tortuosity 1.875), mean residual to the guide was:
  - chord 4.644 mm;
  - single cubic Bézier 1.149 mm;
  - K8 0.231 mm.
  Sine-series and Chebyshev bases with the same budget were similar or worse.
- **SyntheticMRI patch.** K10 reached 0.0153 mm, but the most tortuous branch had tortuosity
  only about 1.03.

The population-level version of this question (K8 adequacy on IXI and BRAVA, length
dependence, 12/16-control refits) is in
[`history_representation.md`](history_representation.md), from
[ixi-brava-control-capacity](../archive/legacy-20261103/docs/research/spline_learning/ixi-brava-control-capacity.md).

## 11. Data and infrastructure notes [DONE]

- **SyntheticMRI restoration (5 Oct).** The working copy had lost most full generator graphs
  (27 complete subjects left). It was restored from a 44 GB archive (130,560 boundary patches
  and 136 segmentation headers verified, 91 complete graph subjects); the old folder is kept
  as `syntheticMRI.before-restore-627540`
  ([restoration](../archive/legacy-20261103/docs/research/spline_learning/synthetic-mri-restoration.md),
  [old inventory](../archive/legacy-20261103/docs/research/spline_learning/archive/synthetic-mri-source-health-2026-10-05.json)).
- **Continuous-edge target rules.** Training VTPs hold chords only, so full paths are rebuilt
  from the generator graphs. Crops clip on voxel cells and snap boundary endpoints to the last
  inside guide sample. Patches with closed or parallel branches are excluded *whole* (one curve
  per node pair): 291 of 10,000 training candidates and 2.75% of available test patches
  ([10k protocol](../archive/legacy-20261103/docs/research/spline_learning/spline-10k-training.md)).
- **Storage cleanup (6 Oct).** `projects` went from about 53 GB to 630 MB; spline targets and
  runs moved to SCRATCH, pilots and the recovery archive to STORE
  ([cleanup](../archive/legacy-20261103/docs/research/spline_learning/projects-cleanup-2026-10-06.md)).
  Offline viewers of real test predictions of the 50-epoch fine-tune also exist
  ([test viewers](../archive/legacy-20261103/docs/research/spline_learning/synthetic-mri-test-predictions.md)).

## What carried forward / what was dropped

- **Carried forward into B0:** K8 B-spline with Greville initialization, zero-initialized last
  layer (curve starts as the chord) and endpoints copied from nodes; the **residual target**
  (§5); the **local image corrector** (§6); the **`graph_only` route** (§4). B0 is exactly this
  end-state. The dense traced model keeps geometry as an edge attribute (K8/K16/K24 per edge),
  reads image features along the candidate path, and fits geometry with a mm chamfer instead
  of control regression.
- **Still true for the current work:** native box-based F1 greatly overstates physical branch
  recovery; validation graph loss is a poor checkpoint selector; geometry overfits short
  branches; node localization, not curve shape, limits edge recovery (also the dense model's
  main remaining limit).
- **Dropped:** V1 (sampled-curve loss); absolute control targets; token-coupled or joint
  geometry gradients; proposals V2/V3; Catmull–Rom as a training target (never evaluated);
  curve-evidence relation scoring inside RelationFormer (replaced by the dense model's
  path-feature edge classifier); the "do we need pretraining?" question (§7).
- **Never established:** the internal mechanism of the token-route collapse; seed robustness
  of any result here; subject-paired CIs (none in this lineage; the current docs have them).

## Sources

All under `docs/archive/legacy-20261103/docs/research/spline_learning/` after the move.

- [README.md](../archive/legacy-20261103/docs/research/spline_learning/README.md) — folder index; curve-family oracle (Bézier, B-spline, piecewise) in toy 2D/3D viewers.
- [representation-implementation.md](../archive/legacy-20261103/docs/research/spline_learning/representation-implementation.md) — design notes: heads A–D, residual controls, V0–V3 tree.
- [spline-learning-report.md](../archive/legacy-20261103/docs/research/spline_learning/spline-learning-report.md) — master log of the whole lineage, 5–13 Oct.
- [spline-learning-results.md](../archive/legacy-20261103/docs/research/spline_learning/spline-learning-results.md) — four-patch V0/V1 overfit results.
- [spline-approach-selection.md](../archive/legacy-20261103/docs/research/spline_learning/spline-approach-selection.md) — guide-distance verification; V0 selected.
- [spline-10k-training.md](../archive/legacy-20261103/docs/research/spline_learning/spline-10k-training.md) — 10k fine-tune protocol, boundary rules, held-out test result.
- [synthetic-mri-test-predictions.md](../archive/legacy-20261103/docs/research/spline_learning/synthetic-mri-test-predictions.md) — two offline viewers of test predictions.
- [synthetic-mri-restoration.md](../archive/legacy-20261103/docs/research/spline_learning/synthetic-mri-restoration.md) — dataset restoration and checks.
- [archive/synthetic-mri-source-health-2026-10-05.json](../archive/legacy-20261103/docs/research/spline_learning/archive/synthetic-mri-source-health-2026-10-05.json) — pre-restoration inventory of missing source files.
- [projects-cleanup-2026-10-06.md](../archive/legacy-20261103/docs/research/spline_learning/projects-cleanup-2026-10-06.md) — storage cleanup and artifact locations.
- [scratch-comparison-300ep.md](../archive/legacy-20261103/docs/research/spline_learning/scratch-comparison-300ep.md) — straight vs K8 scratch protocol, common-v2 definition, straight test results.
- [adaptive-degree2-scratch-comparison.md](../archive/legacy-20261103/docs/research/spline_learning/adaptive-degree2-scratch-comparison.md) — adaptive degree-2 arm targets and run.
- [scratch-comparison-results.md](../archive/legacy-20261103/docs/research/spline_learning/scratch-comparison-results.md) — three-arm held-out test tables.
- [spline-collapse-diagnosis.md](../archive/legacy-20261103/docs/research/spline_learning/spline-collapse-diagnosis.md) — BN, attention and gradient diagnosis; also appended 26–28 Oct BN-provenance notes.
- [spline-weight-pilot.md](../archive/legacy-20261103/docs/research/spline_learning/spline-weight-pilot.md) — 1.474 vs 0.1474 weight pilot.
- [spline-staged-learning.md](../archive/legacy-20261103/docs/research/spline_learning/spline-staged-learning.md) — graph-first ramped spline pilot.
- [curve-informed-connectivity.md](../archive/legacy-20261103/docs/research/spline_learning/curve-informed-connectivity.md) — image evidence along candidate curves for edge scores.
- [frozen-spline-investigation.md](../archive/legacy-20261103/docs/research/spline_learning/frozen-spline-investigation.md) — evidence review, ranked hypotheses, frozen absolute-target head.
- [residual-token-diagnostics.md](../archive/legacy-20261103/docs/research/spline_learning/residual-token-diagnostics.md) — residual vs absolute, token ablation, data breadth, 15k continuation.
- [controlled-unfreezing.md](../archive/legacy-20261103/docs/research/spline_learning/controlled-unfreezing.md) — graph-only / tokens / joint routes from the pretrained model.
- [full-validation-comparison.md](../archive/legacy-20261103/docs/research/spline_learning/full-validation-comparison.md) — the three unfreezing candidates on 1,952 validation patches.
- [local-geometry-diagnostic.md](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-diagnostic.md) — local CNN corrector; SyntheticMRI curvature audit.
- [local-geometry-followup.md](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-followup.md) — local corrector full validation and unfreezing routes.
- [local-geometry-stability.md](../archive/legacy-20261103/docs/research/spline_learning/local-geometry-stability.md) — selection rule and 25k `graph_only` continuation.
- [scratch-local-geometry-diagnostic.md](../archive/legacy-20261103/docs/research/spline_learning/scratch-local-geometry-diagnostic.md) — scratch token-route collapse vs graph-only control.
- [scratch-detached-geometry.md](../archive/legacy-20261103/docs/research/spline_learning/scratch-detached-geometry.md) — scratch detached route and gradient-isolation checks.
- [scratch-extension-300.md](../archive/legacy-20261103/docs/research/spline_learning/scratch-extension-300.md) — detached scratch K8 vs graph-only to 300 epochs.
- [scratch-extension-final-results.csv](../archive/legacy-20261103/docs/research/spline_learning/scratch-extension-final-results.csv) — full-precision final table of that extension.
- [catmull-rom-single-cases.md](../archive/legacy-20261103/docs/research/spline_learning/catmull-rom-single-cases.md) — Catmull–Rom vs B-spline fits on two cases.
- [catmull-rom-learning.md](../archive/legacy-20261103/docs/research/spline_learning/catmull-rom-learning.md) — scratch Catmull–Rom training protocol (no documented result).
- [ixi_patch.md](../archive/legacy-20261103/docs/research/spline_learning/ixi_patch.md) — IXI single-patch curve-family viewer and fits.
- [synthetic_mri_patch.md](../archive/legacy-20261103/docs/research/spline_learning/synthetic_mri_patch.md) — SyntheticMRI single-patch viewer (K10).
- [ixi-brava-control-capacity.md](../archive/legacy-20261103/docs/research/spline_learning/ixi-brava-control-capacity.md) — K8 target adequacy on IXI and BRAVA (summarised in `history_representation.md`).
