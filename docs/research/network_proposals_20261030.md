# Vessel-graph networks: baseline, anchored, and two-stage dense — 30 October 2026

## Read this first: what is done, what is proposed, what was chosen

Every item in this document carries one of three labels:

- **[DONE]** — already run, with measured numbers.
- **[RUNNING]** — implemented and launched on the cluster; no results yet.
- **[PROPOSED]** — an idea or variant that is not built.

### Timeline

**Existed before 30 Oct (earlier work, [DONE]):**
- **RelationFormer B0**, the project's current 3D image-to-graph model (network 1 below).
  It was trained on SyntheticMRI, IXI and BRAVA in earlier studies; the largest run is
  "D08" on BRAVA (see the glossary).
- Its add-ons: the **K8 spline head** (curved edges) and the **A-xyz** node-coordinate
  refinement.

**Diagnostics run on 30 Oct ([DONE]).** These are measurements, not new methods:
1. **Node-heatmap probe.** A plain 3D U-Net was trained only to *find graph nodes*,
   asking whether nodes are detectable from the image at all. Result: much better than
   RelationFormer on IXI and BRAVA (§0).
2. **Oracle-topology geometry study.** With the true graph given, a small network was
   trained only to *draw the shape of each edge*, asking whether curved edges can be
   learned from real images. Result: yes, 2–3× closer to the true centreline than
   straight edges (§0).

**Proposed on 30 Oct, then chosen, built and run ([DONE]):**
- **Network 3, the two-stage dense model — CHOSEN by the user on 30 Oct.** It combines
  the two diagnostics into a full graph predictor. Jobs 1731051 (IXI), 1731052 (BRAVA),
  1731053 (SyntheticMRI), re-scored with the corrected evaluation.
- **Network 2, the heatmap-anchored RelationFormer.** Kept as a comparison arm. Jobs
  1732539 (IXI) and 1732540 (BRAVA).
- **Network 1 at "literature-scale" budget.** RelationFormer B0 retrained on full IXI for
  100 epochs, so that no option is discarded for lack of training. Job 1732163.

All three networks are scored with the same evaluator, cohorts and threshold rule (§5).
**Results are in §6: the two-stage dense model wins on every edge and graph metric.**

## Glossary

**Project shorthand**

- **WS1–WS5.** The five workstreams of the 30 Oct plan:
  - WS1: 2D backbone fix (ImageNet initialization for 2D RelationFormer);
  - WS2: node-detectability probe;
  - WS3: new graph model (network 2 or 3);
  - WS4: proof that curved edges are learnable and better;
  - WS5: GRALE.
- **Probe / node-heatmap probe / WS2 probe.** [DONE] A 3D U-Net trained for about 7 min
  only to predict a heatmap of node locations
  ([`scripts/node_heatmap_probe_20261030.py`](../../scripts/node_heatmap_probe_20261030.py)).
  It is a *diagnostic*, not a graph model: it outputs nodes, not edges.
- **Probe checkpoint.** The saved weights of that probe, one per dataset, in
  `$SCRATCH/experiments/gnbm/node_heatmap_probe_20261030/<dataset>/model.pt`. Network 2
  reuses them, frozen, as its node proposer.
- **Oracle topology / oracle-geometry study.** [DONE] "Oracle" means the true (ground-truth)
  graph is given to the model. Only the edge *shape* is learned, which isolates geometry
  from detection
  ([`scripts/oracle_geometry_20261030.py`](../../scripts/oracle_geometry_20261030.py)).
- **B0.** The baseline RelationFormer recipe: no node refinement, K8 spline head.
- **A-xyz / A-cls.** Earlier variants of B0 with a small *local refinement* head:
  - A-xyz nudges each node's coordinates using high-resolution early CNN features;
  - A-cls also corrects the node confidence.
- **D08.** The completed long BRAVA run (Oct 2026): RelationFormer B0, 300 epochs,
  82,200 updates, initialized from a SyntheticMRI model. It is the "trained long enough"
  evidence for BRAVA.
- **Literature-scale control.** RelationFormer retrained with a budget similar to the
  original papers (§5).
- **Exposures.** Number of training patches seen = updates × batch size.
- **`monitor_train`.** A fixed set of 256 *training* patches. It measures training-set fit
  and is used to choose decision thresholds. Validation patches are never used to choose
  anything.
- **Train-selected thresholds.** The node and edge confidence cut-offs that maximize F1 on
  `monitor_train`, then applied unchanged to validation.

**Graph and geometry terms**

- **Node.** A graph vertex: a junction, a vessel end, or a point where a vessel crosses the
  patch border. Degree-2 points along a vessel are *not* nodes; they are merged into edges.
- **Chord.** The straight segment between an edge's two end nodes.
- **K8 spline.** A clamped cubic B-spline with 8 control points. The 2 end controls are
  the nodes; the 6 interior controls bend the curve.
- **Polyline.** A curve made of straight pieces through M interior points. It is
  equivalent to inserting M degree-2 points *inside* an edge, as an edge attribute, without
  adding them as graph nodes.
- **Dense path / guide.** The original, finely sampled vessel centreline between two nodes,
  used as the geometric ground truth.
- **Corridor.** A 3D tube of image features sampled around a chord (24 samples along it,
  11×11 across, ±12 voxels). The dense model uses it to look *between* two candidate nodes.
- **`graph_only` route.** A gradient rule in RelationFormer: the spline (geometry) loss
  trains only the spline head and never changes the node/edge detector.
- **Greville abscissae.** The standard parameter positions associated with each B-spline
  control point. Interior controls start on the chord at these positions.

**Detection terms**

- **Heatmap (CenterNet-style).** A per-voxel map that is 1 at each node and decays as a
  Gaussian around it. The network predicts it; local maxima ("**peaks**") are the
  detected nodes.
- **NMS.** Non-maximum suppression: keep a voxel only if it is the maximum of its 3×3×3
  neighbourhood, so one blob gives one node.
- **Object queries / relation tokens.** In RelationFormer (DETR-style), a fixed number of
  learned slots, each of which may become a node. Relation tokens are extra global slots
  used when scoring pairs.
- **Hungarian matcher.** A one-to-one assignment (scipy `linear_sum_assignment`) between
  predictions and GT nodes. RelationFormer uses it during training to decide which query
  is "the" node; it does not exist at inference.
- **Matched / unmatched.** A prediction is matched if it is paired with a GT node within
  2 mm by this one-to-one rule.

**Architecture building blocks**

- **SE-ResNet.** RelationFormer's 3D CNN encoder; a ResNet with squeeze-and-excitation
  channel attention.
- **Deformable DETR / deformable attention.** A transformer whose attention looks at a
  few learned sampling points instead of everywhere.
- **Stride 32.** The encoder downsamples by 32: a 64³ patch becomes 2×2×2 feature cells.
- **MONAI BasicUNet.** A standard 3D U-Net from the MONAI library, used as-is. It is an
  encoder–decoder that keeps full 64³ output resolution.
- **Instance norm.** The U-Net's normalization layer. Training and evaluation modes behave
  identically.
- **bf16.** Half-precision arithmetic used to speed up the U-Net.

**Metrics**

- **Node F1 @2 mm.** Predicted nodes are matched one-to-one to GT nodes within 2 mm.
- **Edge F1.** An edge is correct iff both of its nodes are matched and those GT nodes are
  connected.
- **MM / MU / UU false edges.** A false edge between two matched nodes (MM), one matched
  and one unmatched (MU), or two unmatched nodes (UU).
- **common-v2 Branch / Curve F1.** The project's physical metrics:
  - Branch F1 counts correctly reconstructed vessel branches after degree-2 contraction;
  - Curve F1 counts centreline samples within 1 mm of the other graph.
- **β₀ / β₁.** The number of connected components and the number of independent cycles.
- **Mean symmetric distance (mm).** Average distance between a predicted curve and its dense
  path, in both directions.

---

Every statement below about (1) and (3) was checked against the code linked next to it.
External methods are cited by name and year only:
- Relationformer (Shit et al., ECCV 2022)
- Deformable DETR (Zhu et al., ICLR 2021)
- DETR (Carion et al., ECCV 2020)
- CenterNet, "Objects as Points" (Zhou et al., 2019)
- U-Net (Ronneberger et al., 2015)
- SE-Net (Hu et al., 2018)
- the cross-domain image-to-graph transformer of Berger et al. (WACV 2025)

---

## 0. Summary

On real MRA, RelationFormer fails to fit its own training set: IXI train node/edge F1 is
about 0.39/0.09. BRAVA D08 ran 82k updates at batch 32, about 2.6M patch exposures,
comparable to the Relationformer paper's synthetic-vessel budget. It reached train
0.394/0.090 and val 0.356/0.068. SyntheticMRI does fit (0.82/0.64).

The errors are node-driven:
- about 90% of missed edges lack a recovered endpoint pair;
- 83–91% of false edges touch an unmatched node.

The architecture limits node resolution. The SE-ResNet encoder has total stride 32, so a 64³
patch reaches the deformable transformer as a single-level **2×2×2** map. The dense node
probe below shows that the nodes can be found from the image:
- **What it is:** a full-resolution 3D U-Net with a CenterNet heatmap
  ([`scripts/node_heatmap_probe_20261030.py`](../../scripts/node_heatmap_probe_20261030.py)).
- **Budget:** 10k updates at batch 8, about 7 min on one H100.
- **Thresholds:** selected on the training set.
- **Validation node F1 @2 mm:**
  - SyntheticMRI 0.886 (RelationFormer B0 0.793, A-xyz 0.817);
  - IXI 0.639 (RelationFormer 0.365–0.387);
  - BRAVA 0.539 (D08 0.356).
- **Fit:** train ≈ val on all three.

An oracle-topology experiment
([`scripts/oracle_geometry_20261030.py`](../../scripts/oracle_geometry_20261030.py)) tests
edge shape. With the ground-truth endpoints and edges given, a K8 spline learned from
image-corridor features beats the straight chord. Mean symmetric distance to the dense
centerline on validation:
- SyntheticMRI 0.519→0.151 mm;
- IXI 0.537→0.186 mm;
- BRAVA 1.479→0.615 mm.

Subject-paired 95% CIs are far from zero. On BRAVA, 46.8% of GT edges deviate by more than
2 mm from their chord.

The same [DONE] study also trained a **polyline** arm: the same network, decoding its 6
interior points as straight pieces instead of a spline. At equal output count the polyline
matched or slightly beat the spline on mean distance:
- SyntheticMRI 0.146 vs 0.151 mm;
- IXI 0.179 vs 0.186 mm;
- BRAVA 0.583 vs 0.615 mm; with 12 interior points, BRAVA reaches 0.527 mm.

The spline was slightly better on worst-case (Hausdorff) distance. So what is established is:
**curved edge geometry, predicted as an attribute of the edge, beats straight edges.** It
is *not* established that the B-spline beats a polyline. The case against degree-2 points
is about putting them in the graph as *detected nodes*, which inflates detection. As
interior points of an edge they work fine.

Three networks follow from this evidence:
1. **RelationFormer B0** — existing model [DONE]. It is also being rerun at a
   literature-scale budget [RUNNING].
2. **Heatmap-anchored RelationFormer** — proposed on 30 Oct, built as a comparison arm
   [RUNNING]. It keeps the RelationFormer relation machinery but takes its nodes and its
   assignment from the frozen probe's heatmap.
3. **Two-stage dense graph model** — proposed on 30 Oct, **chosen by the user**, built
   [RUNNING]. It has:
   - a full-resolution node heatmap;
   - k-NN candidate pairs;
   - an edge classifier and a geometry regressor that both read image features sampled in
     a corridor around each candidate chord;
   - no Hungarian matching anywhere.

| | (1) RelationFormer B0 [existing] | (2) Heatmap-anchored RelationFormer [built, running] | (3) Two-stage dense model [chosen, built, running] |
|---|---|---|---|
| **Input** | 64³ MRA patch, intensity − foreground mean | same, plus top-K peaks of a frozen WS2 heatmap net | same 64³ patch |
| **Node mechanism** | 120 (SyntheticMRI) or 256 (IXI/BRAVA) learned object queries; sigmoid MLP regresses coordinates from tokens | object queries = top-K heatmap peaks; coordinates = peak + bounded residual (±3 vox) | CenterNet heatmap at 64³; peaks from 3³ max-pool NMS + threshold; sub-voxel offset head |
| **Assignment (train)** | Hungarian on *predicted* coordinates + class probability, recomputed every step | fixed one-to-one matching of *anchor positions* to GT within 2 mm | the same one-to-one physical matching (≤2 mm) of *predicted peaks* to GT that evaluation uses; GT-node candidates with 0.5 mm jitter are added |
| **Edge mechanism** | relation MLP on [token_i, token_j, 2 relation tokens]; training uses matched pairs only; inference scores all pairs | unchanged from (1) | each candidate pair (16-NN in mm) is scored by a 3D CNN over a 24×11×11 feature corridor + endpoint features + endpoint heat |
| **Geometry** | K8 clamped cubic B-spline; MLP residuals on tokens + local CNN corrector; `graph_only` gradient route | unchanged from (1) | K8 clamped cubic B-spline from 6 interior points predicted from the corridor features; chamfer (mm) to the dense path |
| **Train/inference consistency** | low: Hungarian at train, thresholds only at inference; unmatched queries never supervised as edge endpoints | high for nodes (identical assignment); relation pool still matched-only unless extended | high: candidates are built and labelled from predicted peaks with the evaluation matcher (two caveats in §3B.8) |
| **Feature resolution seen by the node/edge decision** | 2×2×2 (stride 32) memory; A-xyz adds a 32³ local map for coordinates only | nodes at 64³ (frozen net); relations still from 2×2×2 tokens | 64³, 32 channels, for nodes, edges and geometry |
| **Compute (measured where available)** | D08 BRAVA: 0.31 s/update at batch 32 on H100, 7.89 GPU-h per 82.2k updates (benchmark) | 0.30 s/update at batch 32 (smoke benchmark); ~15k updates per dataset | probe: ~7 min / 10k updates at batch 8; dense graph: ~0.12–0.13 s/update during GT-only warm-up, ~0.2 s/update afterwards, at batch 8 |

---

## 1. RelationFormer B0 — existing model [DONE]; literature-scale rerun [DONE]

This is the model the project has used so far. Nothing in this section is new. It is
described here so the other two networks can be compared against it.

Code:
- [`models/relationformer.py`](../../models/relationformer.py)
- [`models/seresnet.py`](../../models/seresnet.py)
- [`models/deformable_transformer.py`](../../models/deformable_transformer.py)
- [`models/ops/`](../../models/ops/)
- [`models/position_encoding.py`](../../models/position_encoding.py)
- [`models/matcher.py`](../../models/matcher.py)
- [`training/losses/criterion.py`](../../training/losses/criterion.py)
- [`training/losses/focal.py`](../../training/losses/focal.py)
- [`training/evaluation/inference.py`](../../training/evaluation/inference.py)
- [`models/spline.py`](../../models/spline.py)
- [`models/local_spline_geometry.py`](../../models/local_spline_geometry.py)
- [`models/node_refinement.py`](../../models/node_refinement.py)
- [`configs/base.yaml`](../../configs/base.yaml)

The concrete "B0" recipe values come from
[`experiments/local_refinement_long/plan.json`](../../experiments/local_refinement_long/plan.json),
which is the source of the D08 initialization, and from the D08 description in
[`BRAVA_experiment_plan.md`](../archive/legacy-20261103/docs/research/BRAVA_experiment_plan.md).

### 1A. Simple description

B0 is the 3D Relationformer:
- a CNN encoder (SE-ResNet);
- a Deformable-DETR transformer encoder/decoder;
- a fixed set of learned **object queries** plus **relation tokens**.

Each object query becomes a token. Small heads turn each token into "is this a node?" and
"where is it?".

To decide whether nodes *i* and *j* are connected, an MLP reads the concatenation of the
two node tokens and the two shared relation tokens. A second MLP, the K8 spline head, reads
the same pair representation plus the endpoint geometry. It predicts how the edge bends
away from its straight chord.

During training, a Hungarian matcher pairs each GT node with one query. Only those queries
are told "you are a node", and only pairs of matched queries supervise the relation and
spline heads. At inference there is no matcher: every query above a node threshold
becomes a node, and every pair of such nodes is scored.

The CNN reduces the 64³ patch to a 2×2×2 grid before the transformer sees it. So all
node and edge reasoning works from 8 spatial feature vectors. The A-xyz variant adds a
small local-feature head that nudges coordinates using a higher-resolution early CNN map.

```
 64³ MRA ──► SE-ResNet (stride 32) ──► 2×2×2×2048 ──1×1 conv──► 8 tokens ×552 (+3D sine pos)
                                                                  │
                                         Deformable transformer encoder (6 layers)
                                                                  │ memory 8×552
 learned query_embed (Q×552) ──► Deformable decoder (4 layers, self-attn + deformable cross-attn)
                                                                  │ tokens Q×552
          ┌───────────────────────────┬──────────────────────────┴─────────────┐
   object tokens (120 or 256)                           relation tokens (2)    │
     │ class Linear → {bg,node}                                                │
     │ coord MLP → sigmoid (6)                                                 │
     │ [A-xyz: +local 32³ feature refinement]                                  │
     └── pairs (i,j): [o_i,o_j,r_1,r_2] (2208) ─► relation MLP → {no-edge, edge}
                                              └► SplineHead (+local CNN corrector) → K8 controls
```

### 1B. Detailed description

#### 1B.1 Tensor shapes (B = batch, configured values)

| Stage | Shape | Notes |
|---|---|---|
| Input | `[B,1,64,64,64]` | raw MRA minus foreground mean 0.3334 (`SyntheticMRIDataset`) |
| SE-ResNet `layer0.conv1` 7³, stride 2, pad 3 (+BN, ReLU) | `[B,64,32,32,32]` | `relu1` output is hooked as the "local features" for A-xyz and the spline local corrector |
| `layer0.pool` max-pool 3, stride 2, `ceil_mode` | `[B,64,16,16,16]` | |
| `layer1` (4 SE-ResNet bottlenecks, stride 1) | `[B,256,16,16,16]` | bottleneck expansion 4 |
| `layer2` (4 blocks, stride 2) | `[B,512,8,8,8]` | |
| `layer3` (16 blocks, stride 2) | `[B,1024,4,4,4]` | |
| `layer4` (4 blocks, stride 2) | `[B,2048,2,2,2]` | `num_features = 2048`; total stride 32 |
| `input_proj` 1×1×1 conv | `[B,552,2,2,2]` → flattened `[B,8,552]` | single feature level (`feature_levels: 1`) |
| 3D sine positional encoding + level embedding | `[B,8,552]` | 184 sine/cos channels per axis |
| Transformer encoder (6 layers) | memory `[B,8,552]` | |
| Decoder queries | `Q = object_queries + relation_tokens` = 122 (SyntheticMRI) or 258 (IXI/BRAVA) | `dummy_tokens: 0` |
| Decoder (4 layers) | tokens `[B,Q,552]` | |
| `class_embed` Linear(552→2) | `[B,Q_obj,2]` | `Q_obj` = 120 or 256 |
| `coord_embed` MLP(552→552→552→6), sigmoid | `[B,Q_obj,6]` | `[:3]` = normalized node centre (D,H,W); `[3:]` = box size, used only by the GIoU loss |
| Pair feature | `[P, 552·(2+2)=2208]` | `[o_i, o_j, r_1, r_2]` |
| `relation_embed` MLP(2208→552→552→2) | `[P,2]` | |
| `SplineHead` MLP(2208+10→552→552→18) | `[P,6,3]` residuals (mm) → 8 controls → 64 samples `[P,64,3]` | |

#### 1B.2 Blocks and their origin

- **SE-ResNet encoder.** **Adapted from MONAI.**
  - The `SENet` class in [`models/seresnet.py`](../../models/seresnet.py) is MONAI's SENet
    with configurable per-stage strides.
  - The residual blocks are MONAI's `SEResNetBottleneck`, **third-party as-is**. The
    squeeze-and-excitation design is SE-Net (Hu et al., 2018).
  - Configuration: depths `[4,4,16,4]`, stem 7³ stride-2 conv (`input_3x3=False`,
    `inplanes=64`), SE reduction 16, `groups=1`, BatchNorm.
  - `strides: [1,2,2,2]` sets only the four stages. The stem conv (stride 2) and the
    max-pool (stride 2) add a further ×4, which gives the total stride of 32.
- **3D sine positional encoding.** **Adapted** from the DETR sine embedding to 3D
  ([`models/position_encoding.py`](../../models/position_encoding.py)).
- **Deformable transformer.**
  - The encoder/decoder layers ([`models/deformable_transformer.py`](../../models/deformable_transformer.py))
    are **adapted** from Deformable DETR (Zhu et al., 2021) to 3D, through the Relationformer
    code lineage.
  - The multi-scale deformable attention op ([`models/ops/`](../../models/ops/)) is the
    Deformable DETR CUDA kernel extended to 3D. It is built as `MultiScaleDeformableAttention3D`:
    **third-party design, adapted to 3D**.
  - Configuration: d_model 552, 6 heads (92 dims/head; the module warns that this is not a
    power of 2), 4 sampling points, 1 level, FFN 1280, dropout 0, ReLU.
  - Encoder self-attention is deformable attention over the 8 memory tokens.
  - Each decoder layer runs, in order:
    1. `nn.MultiheadAttention` self-attention over all Q tokens, **third-party as-is** (PyTorch);
    2. deformable cross-attention into the 8-token memory;
    3. the FFN.
  - Decoder input: `tgt` starts at zero. `query_pos` is the learned `query_embed`
    (`nn.Embedding(Q,552)`). Reference points are `sigmoid(Linear(query_embed))`: static,
    learned and input-independent. There is no iterative box refinement.
  - With `relation_attention: true`, the **last** token skips deformable cross-attention
    (`tgt[:, :-1]` is hard-coded). The legacy relation-token cross-attention block
    (`cross_attn2`, `norm12`) is kept for checkpoint compatibility but frozen and unused.
    With 2 relation tokens, the first one (index `Q_obj`) *does* receive deformable
    cross-attention and only the second skips it.
- **Heads.** `class_embed`, `coord_embed`, `relation_embed` and the
  token/relation-token layout come from Relationformer (Shit et al., 2022), **adapted**.
  `MLP` uses ReLU between linear layers.
- **Domain-adversarial discriminators** (Berger et al., WACV 2025 lineage): present in the
  code (`domain_adaptation.enabled`) but **disabled** in B0.
- **K8 spline head** (`SplineHead`, [`models/spline.py`](../../models/spline.py)): **new in
  this project**.
  - `cubic_basis(8, 64)` builds a clamped uniform cubic B-spline. Knots are
    `[0,0,0,0, .2,.4,.6,.8, 1,1,1,1]`; the basis matrix is 64×8 and its last row is forced
    to the last control point, so both endpoints are interpolated.
  - The Greville abscissae of the 8 controls are
    `0, 1/15, 1/5, 2/5, 3/5, 4/5, 14/15, 1`.
  - Features are the 2208-d pair vector plus 10 physical-geometry values: both endpoints
    in mm (6), chord (3) and chord length (1).
  - The MLP outputs 6×3 residuals in mm. Each residual is added to the point at Greville
    abscissa *t* on the physical chord, `(1−t)·a + t·b`.
  - The last layer is zero-initialised, so the initial curve is exactly the chord.
  - Endpoint controls are copied from the predicted nodes, so edges stay connected.
- **Local spline corrector** (`LocalSplineGeometry`, enabled in the B0 recipe config):
  **new in this project**.
  - It samples the **detached** `layer0.relu1` map (`[64,32³]`) plus a ones channel at 32
    points along the **detached** base curve.
  - Layers: `Conv1d(65→16,k3)`+GELU, then `Linear(16·32+10→64)`+GELU, then
    `Linear(64→18)`, zero-initialised.
  - The output is a mm correction added to the interior controls.
- **A-xyz node refinement** (`LocalNodeRefinement` with `compact_output=True`,
  `refine_classification=False`, `detach_features=False`, radius 4 voxels): **new in this
  project**.
  - It samples the `layer0.relu1` map on a 3×3×3 lattice at ±4 voxels around each
    (detached) predicted node, using `grid_sample` with border padding.
  - It concatenates the token, the 27×64 samples and the current xyz (552+1728+3 = 2283).
  - Layers: `Linear(2283→128)`, ReLU, `Linear(128→3)`, zero-initialised.
  - Update: `xyz ← clamp(xyz + 4·tanh(δ)/64, 0, 1)`.
  - The image-feature gradient reaches the stem conv; the node logits are untouched.
  - B0 has `node_refinement.enabled: false`.

#### 1B.3 Assignment (training only)

`HungarianMatcher` ([`models/matcher.py`](../../models/matcher.py)) is **adapted from
DETR**. The solver is scipy `linear_sum_assignment`, **third-party as-is**. For each patch:

`C[q,g] = 5 · ‖x̂_q − x_g‖₁ (normalized DHW coords) − 2 · p_q(node)`

where `p_q` is the softmax node probability. It is solved over all `Q_obj × N_gt`. Every
GT node is matched, because `Q_obj ≥ N_gt` is asserted. The assignment is recomputed every
step from the *current* predictions, so it changes from update to update.

#### 1B.4 Losses (B0 recipe)

All losses are computed on matched queries except the class loss, which covers all queries.
Weights are from the B0 plan config.

| Key | Formula | Weight |
|---|---|---|
| `class` | softmax focal, γ=2, class weights [1,1]: `−α_t(1−p_t)^γ log p_t`, averaged over all `B·Q_obj` queries; label 1 iff the query is Hungarian-matched (`base.yaml` default is weighted CE [0.25, 0.75]) | 4.0 |
| `nodes` | `Σ‖x̂ − x‖₁ / N_matched` over matched queries (normalized coords) | 2.0 |
| `boxes` | `Σ(1 − GIoU_3D)` between the predicted 6-d box and a GT box of fixed size 0.2 centred on the node, / N_matched | 3.0 |
| `cardinality` | L1 between the count of `argmax==node` and the matched count. It is built from an integer argmax count, so it carries **no gradient** and only shifts the logged total | 0.8 |
| `edges` | CE (class weights [1,1]) on all pairs of *matched* queries in a patch. Positives are the GT edges among them; negatives are all other matched pairs (`max_per_graph: 9999`). Each pair gets a random direction flip, then `ratio_upsample` duplicates the minority class to a 0.15 positive:negative ratio. Unmatched queries are never in this pool (`include_unmatched: false`) | 6.0 |
| `spline` | smooth-L1 (β = 1 mm) on the interior-control error, `mode: controls`, `target: residual`. The error is `Δc − [(1−t)Δa + t·Δb]`: the endpoint error interpolated at the Greville *t* is removed, so geometry is learned relative to the predicted chord. Applied to GT edges whose two endpoints are both matched; pairs are canonically oriented (lexicographic in mm); targets are the cached K8 fit controls | 1.474 |

**`graph_only` gradient route.** In `RoutedCriterion` in
[`scripts/controlled_spline_unfreeze.py`](../../scripts/controlled_spline_unfreeze.py), the
spline loss sees `tokens.detach()` and `pred_nodes.detach()`. Together with the detached
local features, this means geometry gradients update only the spline head and its local
corrector. They never reach the detector.

Optimizer (B0 recipe): AdamW, LR 7e-5 (encoder 3.5e-5 in D08), weight decay 1e-4, batch 32.
D08 used a 5-epoch linear warm-up and then cosine decay to 1e-6 over 300 epochs.

#### 1B.5 Inference

`infer_graphs` ([`training/evaluation/inference.py`](../../training/evaluation/inference.py)):

1. **Nodes.** A query is a node if `p_node > node_threshold`, or if `argmax == node` when
   no threshold is set. There is no NMS and no duplicate suppression.
2. **Edge scores.** For every unordered pair of kept queries (`torch.combinations`), the
   relation logits are averaged over both orderings: `½(R([o_i,o_j,r]) + R([o_j,o_i,r]))`.
   Training uses one random ordering per pair.
3. **Edges.** A pair is kept if `p_edge > edge_threshold` (or by argmax). The optional
   degree caps and penalties are off in B0.
4. **Geometry.** Kept pairs are canonically oriented, and the spline head produces 8
   controls and 64 curve samples per edge.

D08 used fixed 0.5/0.5 thresholds. The comparison in §5 uses train-only selected thresholds.

#### 1B.6 Where the failure modes come from

- **Resolution.**
  - Every token is computed from deformable attention into an 8-vector memory. A sampling
    location inside the 2×2×2 grid is a trilinear blend of at most 8 vectors.
  - Sub-voxel node positions must be regressed by `coord_embed` from this blended
    representation. A-xyz helps (SyntheticMRI 0.793→0.817) because it reads a 32³ map,
    but only for coordinates.
  - The node probe shows that full resolution lifts IXI val node F1 from about 0.37 to
    0.639 at a tiny fraction of the budget.
- **Train/inference mismatch.**
  - In training, the node label and all edge and spline supervision depend on a Hungarian
    assignment of *predicted* coordinates that changes as the predictions move.
  - At inference no assignment exists, and the relation head scores pairs that include
    unmatched (background or duplicate) queries. It was never trained on such queries.
  - This matches the measured attribution: 83–91% of false edges touch an unmatched node.
- **Duplicates.** Nothing suppresses two confident queries on the same GT node. Hungarian
  labels one as positive and the rest as background, and which query that is can change
  between steps.
- **Edge recall is bounded by node recall.** A GT edge supervises the relation head only
  when *both* endpoints are matched. At inference it can be correct only if both endpoints
  are recovered within 2 mm. About 90% of missed edges lack a recovered endpoint pair.
- **"More distant node more informative".** The relation MLP sees two summary tokens and
  two global relation tokens. It never sees the image *between* the two nodes. For distant
  pairs, connectivity must be inferred from two local summaries of a 2³ memory.
- **Underfitting on real data.** IXI train edge F1 is about 0.09 and BRAVA D08 train edge
  F1 is 0.090 at about 2.6M exposures, while SyntheticMRI fits (0.82/0.64). The model
  underfits its own real training sets, so this is not only a generalization gap.

---

## 2. Heatmap-anchored RelationFormer — proposed 30 Oct, built and run as a comparison arm [DONE]

Status:
- Proposed on 30 Oct as one of two ways to fix RelationFormer. The user chose network 3
  as the main direction; this one is kept as a comparison arm.
- Implemented on 30 Oct (new files only):
  - [`models/anchored_relationformer.py`](../../models/anchored_relationformer.py)
  - [`models/anchor_matcher.py`](../../models/anchor_matcher.py)
  - [`scripts/anchored_relationformer_20261030.py`](../../scripts/anchored_relationformer_20261030.py)
  - [`tests/test_anchored_relationformer.py`](../../tests/test_anchored_relationformer.py)
- 7/7 unit tests pass, and a GPU smoke run passed: on 8 BRAVA patches after 500 updates,
  node F1 0.968 and edge F1 0.945. That is a memorization check, not a result.
- Production runs: jobs 1732539 (IXI) and 1732540 (BRAVA), about 15k updates at batch 32
  each. Completed; results are in §6.

### 2A. Simple description

The probe checkpoint is the frozen node-heatmap U-Net from the 30 Oct diagnostic; see the
glossary. It finds candidate nodes as heatmap peaks. RelationFormer is then given its
object queries **at those peaks**, instead of learning query positions from scratch:
- each query starts from a peak's position and score;
- its deformable attention looks around that peak;
- its output coordinate may only move the peak by up to ±3 voxels.

Training no longer needs a Hungarian matcher on moving predictions. Each peak is paired
once with the GT node within 2 mm, with exactly the rule the evaluation uses, and that
pairing is the same at training and evaluation. The relation head, the spline head and the
losses stay as in B0.

This variant isolates one question: does fixing node proposals and assignment rescue the
Relationformer relation machinery? Relation features would still come from the 2×2×2
memory.

```
 64³ MRA ──► frozen WS2 U-Net ──► heatmap ──NMS──► top-K peaks (xyz, score)
    │                                                  │
    │                         query_pos = MLP(sine(xyz)); content = learned + Linear(score)
    │                         reference points = xyz
    └──► SE-ResNet → 2³ memory → transformer encoder ──► deformable decoder ──► tokens
                                                              │
                         node = peak + 3·tanh(Δ) voxels; class Linear; relation MLP; K8 spline
                         assignment: peak↔GT one-to-one within 2 mm (fixed, = evaluation)
```

### 2B. Detailed description (as built)

- **Proposal net** (`HeatmapProposer`). It is the probe: MONAI `BasicUNet` with 5 outputs
  (see §3B for the architecture), a **third-party as-is** backbone whose checkpoint was
  trained in this project.
  - It is **frozen**: always in eval mode, `requires_grad=False`, run under `no_grad` with
    bf16. It runs on the same augmented patch the detector sees.
  - Peaks are found with 3³ max-pool NMS, as in the probe and in §3.
  - The top-K peaks are kept, with K = 256 = the number of object queries. That gives
    `[B,256,3]` voxel coordinates plus `[B,256]` scores.
  - There is no score floor. Because low-score background maxima also count, all 256 slots
    were real peaks in every smoke patch; on average 127 per patch scored ≥ 0.05.
  - If fewer than K peaks exist, the remaining slots get score 0 at random in-patch
    positions from a per-sample seeded RNG. This is implemented and tested but rarely used.
  - Anchors are clamped to [−0.5, 63.5] voxels.
- **Queries** (new in this project):
  - `query_pos[k] = MLP(sine_embedding(peak_k))` with shape `[B,K,552]`. The sine
    embedding is the 3D sine code evaluated at a point.
  - `tgt[k] = e_learned + Linear(score_k)`, with **one shared** learned vector. A
    per-slot vector would encode each query's score rank, since queries are sorted by
    score.
  - Decoder reference points = the peaks, converted to the (x=W, y=H, z=D) order and the
    `(i+0.5)/64` voxel-centre convention that the decoder's `grid_sample(align_corners=False)`
    expects. A unit test recovers the anchor voxel to 1e-4, and fails for the wrong order or
    without the +0.5 shift.
  - Relation tokens stay learned, as in B0.
  - Shared weights (encoder, transformer, heads) are initialized from the SyntheticMRI B0
    checkpoint, as in D08.
- **Coordinates.** `x̂_k = peak_k + 3·tanh(Δ_k)` voxels, where `Δ_k = MLP(token_k)`. This
  bounded residual is analogous to `LocalNodeRefinement`, which uses 4 voxels. At
  initialization the predicted nodes equal the anchors exactly.
- **Assignment.** One-to-one `linear_sum_assignment` on mm distances between *anchor
  positions* (peaks) and GT nodes. Pairs beyond 2 mm are disallowed, exactly as in the
  `match`/`match_nodes` evaluation matcher.
  - The assignment is fixed per patch for a given augmentation draw, because the
    proposal net is frozen.
  - The same assignment defines the node labels, the matched-edge pool and the spline
    targets.
  - At evaluation, the same rule is applied to the final nodes.
- **Losses.** Unchanged from D08: node focal, matched-edge CE with ratio upsampling, K8
  spline on the `graph_only` route, box/coordinate terms. The coordinate loss is the L1
  between the bounded-residual node and its assigned GT node.
- **Training recipe.** D08's loop and optimizer (AdamW, LR 7e-5, encoder 3.5e-5, cosine),
  batch 32. IXI: 14,976 updates; BRAVA: 15,070 updates. The 5-epoch warm-up is kept in
  epochs.
- **Evaluation differences from network 3, as built:**
  - candidate edges are all pairs of retained queries, as in RelationFormer, not 16-NN;
  - thresholds use strict `>`, as `infer_graphs` does;
  - otherwise the same matcher, metrics and train-only threshold selection.
- **Expected effect.** Node recall and localization are inherited from the probe (IXI val
  node F1 0.639 versus 0.37), there is no assignment churn, and duplicates are handled by
  NMS.
- **Unchanged weaknesses.**
  - Relation and spline features still come from tokens attending to the 2×2×2 memory.
    Anchoring reference points chooses *where* to interpolate inside an 8-vector grid; it
    does not add resolution.
  - Unmatched anchors are still absent from the relation pool unless it is extended.
  - Recall can never exceed the frozen proposal net's raw peak recall.

---

## 3. Two-stage dense graph model — proposed 30 Oct, CHOSEN by the user, built and run [DONE]

Status:
- Proposed on 30 Oct after the probe and oracle-geometry diagnostics. Chosen by the user
  as the main direction.
- It reuses the probe's U-Net design for nodes (stage 1) and the oracle-geometry corridor
  network for edges (stage 2), and puts them into one model trained end to end.
- Production runs: jobs 1731051/1731052/1731053, 15k updates at batch 8. Completed;
  results are in §6.
- Two evaluation caveats found during review are **now fixed in the code** (§3B.8). The
  first runs' saved models are re-scored with the fixed evaluation (`--evaluate-only` →
  `final-v2.json`).

Code:
- [`scripts/dense_graph_20261030.py`](../../scripts/dense_graph_20261030.py)
- [`scripts/node_heatmap_probe_20261030.py`](../../scripts/node_heatmap_probe_20261030.py),
  which provides `targets`, `focal`, `make_dataset`, `collate` and `PLAN`
- [`scripts/oracle_geometry_20261030.py`](../../scripts/oracle_geometry_20261030.py),
  which provides `frame`, `polyline`, `chamfer` and `to_mm`
- [`models/spline.py`](../../models/spline.py), which provides `cubic_basis`
- Slurm launcher: [`cluster/jean_zay/dense_graph.slurm`](../../cluster/jean_zay/dense_graph.slurm)

### 3A. Simple description

**Stage 1.** A standard 3D U-Net runs at full resolution. It outputs:
- a node heatmap, CenterNet-style, with one Gaussian blob per graph node;
- a 3-vector sub-voxel offset;
- a centerline mask, an auxiliary output that keeps the features vessel-aware;
- 32 general feature channels.

Nodes are the local maxima of the heatmap above a threshold.

**Stage 2.** Each node is paired with its 16 nearest neighbours in millimetres. For each
candidate pair:
1. a local frame is built around the straight chord;
2. the 32-channel U-Net features are resampled in a 24×11×11 tube (a "corridor") that
   covers the chord, slightly past both ends, and ±12 voxels sideways;
3. a small shared 3D CNN encodes the corridor.

Two heads read this encoding together with the endpoint features and heatmap scores:
- an **edge** head, which answers "are these two nodes directly connected?";
- a **geometry** head, which gives 6 interior points of a cubic B-spline that bends the
  edge to follow the vessel.

There is no Hungarian matching over moving predictions. During training, the candidate
pairs are built from the model's *own predicted peaks*. A peak is labelled with the GT node it matches within 2 mm,
using the evaluation's one-to-one matcher. A pair is positive if those two GT nodes share
an edge. Pairs built from jittered GT nodes are also used: alone during a 2000-update
warm-up, and alongside the predicted pairs afterwards. Thresholds are chosen on training
patches only.

```
 64³ MRA ─► MONAI BasicUNet (full res) ─► 37 ch @64³
                                     ├ ch0  heat logit ──► CenterNet focal
                                     ├ ch1-3 offset   ──► L1 at GT peak voxels
                                     ├ ch4  centerline ──► weighted BCE
                                     └ ch5-36 features F (32)
 heat ─► 3³ max-pool NMS + threshold ─► peaks (+offset) [detached]
 peaks ─► 16-NN in mm ─► candidate pairs (a,b)
 for each pair:  frame(t,u,w) ─► grid_sample F on 24×11×11 corridor ─► 3D CNN ─► 6912
                 + |d|,d (4) + F(a),F(b) (64) + heat(a),heat(b) (2) = 6982
                     ├─► edge MLP → logit ──► BCE (pos_weight)
                     └─► geometry MLP → 6×3 local offsets ×12 vox ─► K8 B-spline (64 pts) ─► chamfer (mm)
```

### 3B. Detailed description

#### 3B.1 Tensor shapes (batch B = 8)

| Stage | Shape | Notes |
|---|---|---|
| Input | `[8,1,64,64,64]` | intensity − 0.3334 |
| `BasicUNet` `conv_0` | `[8,32,64³]` | |
| `down_1…down_4` (max-pool 2 + TwoConv) | `[8,32,32³]`, `[8,64,16³]`, `[8,128,8³]`, `[8,256,4³]` | `features=(32,32,64,128,256,32)` |
| `upcat_4…upcat_1` (transposed-conv upsampling + skip concat + TwoConv) | `[8,128,8³]`, `[8,64,16³]`, `[8,32,32³]`, `[8,32,64³]` | |
| `final_conv` 1×1×1 | `[8,37,64,64,64]` | ch0 heat logit (bias initialized to −4.6, so σ ≈ 0.01); ch1–3 offsets (voxels, DHW); ch4 centerline logit; ch5–36 features F |
| Peaks (per patch) | `[N,3]` voxel coordinates + `[N]` scores | sorted by descending score; coordinates clamped to [−0.5, 63.5] |
| Candidate pairs | `[P,2]`, P ≤ 16·N | unique undirected `(i<j)` |
| Training pairs after subsampling | all positives + ≤256 negatives | |
| Corridor sample grid | `[P,24,11,11,3]` | |
| Sampled corridor | `[P,32,24,11,11]` | |
| Corridor CNN | `[P,64,24,11,11]` → `[P,64,12,6,6]` → `[P,64,12,3,3]` → flatten 6912 | |
| Pair vector `x` | `[P,6982]` | 6912 + 1 (\|d\|/64) + 3 (d/64) + 64 (F at a, b) + 2 (heat at a, b) |
| `edge_head` | `[P,1]` | Linear(6982→256), GELU, Linear(256→1) |
| `geometry_head` | `[P,18]` → `[P,6,3]` × 12 voxels | Linear(6982→256), GELU, Linear(256→18), zero-initialized |
| Curves | spline `[P,64,3]`; polyline `[P,64,3]`; chord `[P,2,3]` (voxels) | |

#### 3B.2 Blocks and their origin

- **MONAI `BasicUNet`**: **third-party as-is**. U-Net (Ronneberger et al., 2015).
  - Called as `BasicUNet(spatial_dims=3, in_channels=1, out_channels=37, features=(32,32,64,128,256,32))`.
  - All other arguments are MONAI defaults: LeakyReLU (slope 0.1), instance norm with
    affine parameters, conv bias, no dropout, `"deconv"` upsampling. Each `TwoConv` is
    two 3×3×3 conv blocks.
  - Instance norm keeps no running statistics, so `train()`/`eval()` give identical
    normalization.
  - The project's only change is the initial bias −4.6 on output channel 0 (CenterNet
    prior).
- **Heatmap target and penalty-reduced focal loss**: formulation from CenterNet
  (Zhou et al., 2019), **re-implemented** in `targets`/`focal` (probe script).
  - Each GT node in voxels is rounded to a voxel `c`.
  - Heat is `H(v) = max_n exp(−‖v − c_n‖² / (2σ²))` with σ = 1.5 voxels. `H = 1` exactly
    at the `c_n` voxels.
  - Loss:
    `L_node = −(1/N_pos) Σ_v [ 1{H=1}(1−p)² log p + 1{H<1}(1−H)⁴ p² log(1−p) ]`,
    with `p = σ(logit)` clamped to [1e-4, 1−1e-4]. This is α=2, β=4 with no extra weight.
- **Sub-voxel offsets**: CenterNet-style. L1 between the predicted offset and
  `v_n − round(v_n)` at the peak voxels only.
- **Centerline auxiliary**: **new in this project**.
  - Each GT edge's cached dense guide (64 points) is linearly upsampled to 256 points.
  - The points are rounded to voxels, set to 1, and dilated by a 3³ max-pool.
  - Loss: `BCEWithLogits` with `pos_weight = 5`, averaged over all voxels.
- **Peak extraction** (`peaks`): `heat == maxpool3d(heat, 3, stride 1, pad 1)` and
  `heat ≥ τ`. Coordinates = voxel index + predicted offset. Runs under `torch.no_grad`
  on `out.detach()`. **New in this project**: a standard CenterNet-style decoding, without
  CenterNet's top-K.
- **Candidate rule** (`candidate_pairs`): **new in this project**.
  - `torch.cdist` in mm, using the patch's normalized→mm `metric`/64.
  - `topk(k = min(16, N−1))` per node.
  - Pairs are symmetrized with `(min, max)` and `unique`.
  - A pair is kept if *either* node is among the other's 16 nearest.
  - GT edge coverage by this rule is 98.6–99.9%.
- **Local frame** (`frame`, oracle script): **new in this project**.
  - `d = b − a`, `t = d/‖d‖`.
  - `u = normalize(t × e)`, where `e` is the coordinate axis with the smallest `|t|`
    component.
  - `w = t × u`.
  - All values are in voxel coordinates.
- **Corridor sampling**: PyTorch `F.grid_sample`, **third-party as-is** (trilinear, zeros
  padding, `align_corners=True`).
  - Sample points: `p = a + s·d + o_u·u + o_w·w` with `s ∈ linspace(−0.1, 1.1, 24)` and
    `o_u, o_w ∈ linspace(−12, 12, 11)` voxels, a 2.4-voxel lateral spacing.
  - Points are normalized as `p/63·2−1` and flipped to (x=W, y=H, z=D).
  - Endpoint features and heat are sampled with the same call, at `a` and `b`.
- **Corridor CNN and heads**: **new in this project** (inherited from the oracle-geometry
  script).
  - CNN: `Conv3d(32→64,3,p1)`, GELU, `Conv3d(64→64,3,stride 2,p1)`, GELU,
    `Conv3d(64→64,3,stride (1,2,2),p1)`, GELU.
  - The edge and geometry heads share the CNN output `x`.
- **Geometry decoding**: **new in this project**.
  - The geometry head output is reshaped to 6×3 and multiplied by the radius (12 voxels),
    giving local offsets `(ℓ_t, ℓ_u, ℓ_w)`. These are unbounded: there is no tanh.
  - Interior point *k*: `a + g_k·d + ℓ_t,k·t + ℓ_u,k·u + ℓ_w,k·w`.
  - Mode `spline` (the default):
    - `g_k` are the interior Greville abscissae of `cubic_basis(8, 64)`:
      `1/15, 1/5, 2/5, 3/5, 4/5, 14/15`.
    - Controls `[a, 6 interior, b]` (8 points) are multiplied by the 64×8 basis. This
      gives a clamped cubic B-spline that interpolates `a` and `b`.
    - With zero initialization, the curve is exactly the chord.
  - Mode `polyline`: `g_k = k/7`, and `[a, interior, b]` is resampled to 64 points.
  - Mode `chord`: returns `[a, b]`, and the geometry loss is skipped.
- **Matcher** (`match`): scipy `linear_sum_assignment`, **third-party as-is**, on the mm
  distance matrix. Entries above 2 mm are set to 1e6, and pairs above 2 mm are dropped
  after solving. This is the same one-to-one rule as the evaluation's `match_nodes`, which
  uses a different finite penalty value.
- **Chamfer** (oracle script): both curves are resampled to 128 points along their
  parameter. `(mean_i min_j ‖p_i − r_j‖ + mean_j min_i ‖p_i − r_j‖)/2` in mm, per edge.

#### 3B.3 Losses and weights

`L = L_node + L_offset + L_centerline + L_edge + L_geometry`. All weights are 1.

- `L_node`: penalty-reduced focal, above.
- `L_offset`: L1 at GT peak voxels.
- `L_centerline`: BCE with pos_weight 5.
- `L_edge`: per patch-term, `BCEWithLogits(logits, labels, pos_weight = min((N−P₊)/max(P₊,1), 20))`.
  This is averaged over all patch-terms in the batch: up to 16 per batch after warm-up,
  as explained in 3B.4.
- `L_geometry`: chamfer in mm. It is averaged over *all* positive pairs in the batch,
  from both candidate sources.

The geometry reference is the cached dense path (`guides`, 64 points per edge). It is
oriented so that it starts at the GT node matched to `a`. The predicted curve's endpoints
are the candidate positions, which are jittered GT nodes or predicted peaks, not the GT
nodes. So endpoint error contributes to the chamfer, and the geometry head cannot remove it.

#### 3B.4 Training labels without Hungarian

For each patch and each candidate source:

- **GT source** (every step).
  - Candidates are the GT nodes plus Gaussian jitter. Per axis, σ = 0.5 mm divided by the
    norm of the first row of the voxel→mm matrix.
  - Mapping: identity. A pair is positive iff it is a GT edge.
  - Before update 2000 this is the only source, so the edge and geometry heads learn on
    meaningful candidates before the heatmap is usable.
- **Predicted source** (updates > 2000).
  - Candidates: peaks with σ(heat) ≥ 0.2, taking the highest-scoring ones up to a cap.
  - The cap was 120 in the first runs. It is now `MAX_PEAKS = 200`, because IXI has up to
    153 GT nodes per patch.
  - Mapping: `match(peaks_mm, gt_mm, 2.0)`.
  - A pair is positive iff both peaks are matched and the matched GT pair is a GT edge.
  - Every pair that touches an unmatched peak (false positive or duplicate) is negative.

The GT source is **kept for the whole run**: alone before update 2000, alongside the
predicted source afterwards. So after warm-up each patch contributes two edge terms. This
is intended, and the docstring now says so.

In both sources:
- all positives are kept;
- negatives are randomly subsampled to at most 256 per patch-term;
- the geometry loss covers positive pairs only.

#### 3B.5 Inference and thresholds

Current (corrected) protocol, `final_evaluation`:
1. Run the U-Net (bf16 autocast). Extract the peaks with σ(heat) ≥ `node_t`, the selected
   node threshold.
2. Build the 16-NN candidate pairs **on that retained node set**. This is the same rule as
   in training.
3. Compute the edge probability `σ(edge_logit)` and the curve for every pair.
4. In `evaluate`:
   - keep edges with probability ≥ `edge_t`;
   - polylines are `[node_a, curve[1:-1], node_b]`.

The first runs' built-in evaluation (`final.json`) used an older protocol. It built the
pairs on all peaks ≥ 0.10 and then filtered nodes to `node_t`, so low-score peaks could
take 16-NN slots. The corrected re-scoring is written to `final-v2.json`.
5. Score:
   - `match_nodes` (2 mm, one-to-one) feeds node TP/FP/FN;
   - `edge_accounting` gives edge TP/FP/FN, the missing-endpoint fraction of FN, the
     MM/MU/UU false-edge classes and the β₀/β₁ errors;
   - `evaluate_common_vessels` gives the common-v2 Branch/Curve/anchor F1 and the guide
     distance. It contracts degree-2 nodes and uses 1 mm anchors with 0.25 mm sampling.

Threshold selection (`select_node_threshold`, then `select_edge_threshold`):
- runs on the `monitor_train` patches (256, no augmentation);
- `node_t ∈ {0.10, 0.15, …, 0.70}` maximizes node F1, which does not depend on edges;
- the training patches are then re-predicted with pairs built at `node_t`;
- `edge_t ∈ {0.10, …, 0.90}` maximizes edge F1.

Validation is evaluated at the selected thresholds and also at a fixed 0.5/0.5.
Validation is never used for selection.

#### 3B.6 Hyperparameters (defaults in `main`)

| Parameter | Value |
|---|---|
| updates | 15,000 |
| GT-only warm-up | 2,000 updates |
| batch | 8, sampled with replacement (`RandomSampler`, `updates × batch` draws) |
| augmentation | three-plane quarter turns + Gaussian noise (p 0.35, std ≤ 0.015); no zoom |
| optimizer | AdamW, LR 1e-3, weight decay 1e-4 |
| schedule | `OneCycleLR` (`pct_start` 0.05, PyTorch defaults otherwise) |
| gradient clipping | global norm 5.0 |
| precision | bf16 autocast for the U-Net; heads and losses in fp32 |
| seed | 364505 |
| training peak threshold | 0.2; at most 200 predicted candidates per patch (120 in the first runs) |
| neighbours | k = 16 |
| negatives | ≤ 256 per patch-term |
| corridor | 24 × 11 × 11, s ∈ [−0.1, 1.1], ±12 voxels |
| interior points M | 6 (K = 8 controls) |
| modes | `spline` (default), `polyline`, `chord` |
| heat σ | 1.5 voxels |
| centerline pos_weight | 5 |
| edge pos_weight cap | 20 |

#### 3B.7 Why it addresses each diagnosed failure

- **Resolution.** Nodes, edges and geometry all read 64³ features, against a 2³ memory in
  (1). The node half of this design is exactly the WS2 probe, which reached IXI 0.639 and
  BRAVA 0.539 val node F1 with train ≈ val, against 0.37 and 0.356.
- **Train/inference mismatch.** No Hungarian on moving predictions.
  - Training candidates come from the same peak extractor, the same k-NN rule and the
    same one-to-one 2 mm matcher that evaluation uses.
  - Unmatched peaks *are* supervised: all their pairs are negatives. B0 never shows
    unmatched queries to the relation head.
  - Thresholds are chosen on training data.
  - Instance norm and zero dropout make `train()` and `eval()` forward passes identical.
- **Duplicates.**
  - The focal loss rewards one maximum per node, and the 3³ NMS removes adjacent maxima.
  - A surviving duplicate is unmatched, so its edges are trained as negatives. The edge
    head also sees each endpoint's heat value.
- **"More distant node more informative".** The edge scorer samples the image corridor
  *between* every candidate pair, plus the chord length and direction. Evidence of a
  vessel joining the two nodes is in its input directly, at any distance within the
  16-NN set.
- **Continuous edges.** The geometry head is the oracle-geometry design with the same
  corridor features, which cut validation mean symmetric distance by 3–4× against the
  chord (for example IXI 0.537→0.186 mm). Edges keep exact node endpoints, because the
  clamped B-spline interpolates them.

#### 3B.8 Known limitations and risks

- **Local, pairwise decisions.** Each pair is scored independently. There is no global
  graph reasoning: no degree prior, no cycle or β₀ consistency, no mutual exclusion.
  - Specific risk: **transitive skip edges**. For three nodes a–b–c along one vessel, the
    a–c corridor also contains vessel. The label is negative, but the head can only learn
    to reject it from the features of the intermediate junction visible in the corridor,
    since heat is given only at the endpoints.
  - Edge scores are not symmetrized. The pair order is "lower index first": by descending
    score for peaks, by GT index for the GT source.
- **Candidate-rule ceiling.** The 16-NN rule misses about 1–1.5% of GT edges, which can
  never be predicted.
- **Corridor extent.**
  - The lateral half-width is 12 voxels and the axial extent is 10% beyond each endpoint.
    An edge whose path leaves this tube cannot be seen in full: tortuous loops, or vessels
    that bulge backwards.
  - The interior-point offsets are scaled by the same radius but not bounded.
  - This matters most on BRAVA, where 46.8% of GT edges deviate more than 2 mm from their
    chord.
  - Lateral sampling is 2.4 voxels, coarse for thin vessels, which relies on the U-Net
    features' receptive field.
- **Label incompleteness (BRAVA).** The SWC annotations are incomplete. Real but
  unannotated junctions become false-positive peaks: the heat focal loss suppresses them,
  and every pair touching them is negative. Unannotated vessels count as background for
  the centerline auxiliary. Brain masks are not vessel labels.
- **Train/inference candidate details.** Two caveats were found in review; both are now
  **fixed**:
  1. Inference used to build the k-NN on all peaks ≥ 0.10 and then filter to `node_t`.
     It now builds it on the nodes retained at `node_t`. The first runs are re-scored
     (`final-v2.json`).
  2. The training cap was 120 peaks, below IXI's maximum of 153 GT nodes per patch. It is
     now 200 (affects future training runs only).

  One difference remains: training uses peaks ≥ 0.2, while inference uses the selected
  `node_t`, which can be different. The candidate *rule* is identical; the node set it is
  applied to can differ slightly.
- **Compute and budget.** 15k × 8 = 120k patch exposures, far below RelationFormer's
  budgets: about 2.6M for D08, about 1.0M for the literature-scale control. A loss for (3)
  would not be explained by over-training.

---

## 4. Recommended: two-stage dense model, in detail

### 4.1 Exact order of one training update

1. **Sample.** Draw 8 patches with replacement and apply quarter-turn and noise
   augmentation. `collate` returns images `[8,1,64³]`, per-patch normalized nodes `[N,3]`,
   edges `[E,2]`, and geometry dicts (`metric` 3×3 normalized→mm, `guides` `[E,64,3]`).
2. **Build dense targets** on the GPU: heat `[8,1,64³]`, offset `[8,3,64³]`, peak mask,
   and the dilated centerline mask.
3. **Forward the U-Net once** under bf16 autocast: `out [8,37,64³]`.
4. **Dense losses:** `L_node`, `L_offset`, `L_centerline`.
5. **Prepare the pair inputs.**
   - `feats = out[:,5:].float()` keeps its gradient.
   - `heat = sigmoid(out[:,:1]).detach()` carries no gradient.
6. **Extract peaks** if `step > 2000`: `peaks(out.detach(), 0.2)` under `no_grad`.
   Candidate coordinates are therefore constants.
7. **Per patch, per source** (GT always; predicted after warm-up), in `patch_terms`:
   1. Convert the candidates to mm and build the 16-NN pairs.
   2. Label the pairs: identity mapping for GT, `match` for peaks.
   3. Subsample the negatives to 256.
   4. Compute `pair_features` (corridor `grid_sample` + CNN + endpoint samples).
   5. Compute the edge logits and the BCE.
   6. For positives: compute curves and the oriented dense references, then the chamfer
      in mm.
8. **Reduce:** `L_edge` = mean over patch-terms; `L_geometry` = mean over all positive
   pairs. The total is the unweighted sum of the five terms.
9. **Update:** `zero_grad`, `backward`, clip the global norm at 5, AdamW step, OneCycle
   step. Log every 100 updates.

After training:
1. save `model.pt`;
2. predict `monitor_train` at 0.10 and select `(node_t, edge_t)`;
3. predict `val_ids`;
4. write `final.json` with the train-monitor results, validation at the selected
   thresholds, and validation at 0.5/0.5.

### 4.2 Where gradients flow

| Loss | Reaches |
|---|---|
| `L_node` | U-Net (via ch0) |
| `L_offset` | U-Net (via ch1–3) |
| `L_centerline` | U-Net (via ch4) |
| `L_edge` | edge head → corridor CNN → `grid_sample` input (`feats`, ch5–36) → **all shared U-Net weights**; also endpoint feature samples → U-Net |
| `L_geometry` | geometry head → corridor CNN → `feats` → **all shared U-Net weights** |

There is no gradient to the peak positions or sub-voxel offsets from `L_edge` or
`L_geometry`. Peaks are computed from `out.detach()` under `no_grad`, and the sampling
grids are therefore constants. Endpoint heat values are detached too. So the offset
channel is trained only by `L_offset`, and the heat channel only by `L_node`. The U-Net
trunk and the 32 feature channels are shaped jointly by all five losses at equal weight.
This is the intended coupling: edge and geometry evidence improve the shared features,
while node placement stays a pure detection problem.

### 4.3 Planned arms and what each isolates

- **`--mode spline`** (K8, M = 6): the proposed model.
- **`--mode polyline`** (M = 6 vertices): the same number of outputs with C⁰ decoding,
  which tests the continuous representation.
- **`--mode chord`**: no geometry loss and straight edges. It gives the topology-only
  contribution, and the curve metrics then measure the value of learned geometry within
  the full pipeline.

The oracle-geometry study already ran the spline-versus-chord comparison with GT topology.
The dense model tests whether that gain survives predicted nodes and edges.

### 4.4 Possible next refinements [PROPOSED, not implemented]

1. Train with candidates at the same node threshold used at inference.
2. Symmetrize the edge logit over (a,b)/(b,a).
3. Pass a heat profile sampled along the corridor to the edge head, against skip edges.
4. Add a light global step: degree-aware greedy selection, or one round of message
   passing over the candidate graph.
5. Use an adaptive corridor radius for long or tortuous edges.

Each would be a separate matched ablation.

---

## 5. How the comparison will be run

- **Cohorts.** Use the fixed subject-disjoint splits in
  [`experiments/multidataset_continuous/plan.json`](../../experiments/multidataset_continuous/plan.json),
  with `monitor_train` = 256 training patches for threshold selection:

  | Dataset | `train_ids` | `val_ids` | Full cohort (train/val) |
  |---|---:|---:|---|
  | SyntheticMRI | 4,096 | 480 | 25,000 train |
  | IXI | 4,096 | 512 | 10,000 / 2,649 |
  | BRAVA | 8,768 | 1,796 | 8,768 / 1,796 |

  No test subjects are loaded.
- **Metrics.** One evaluator for every network: `evaluate` in the dense script.
  - Node F1 @2 mm by one-to-one `match_nodes` in physical mm.
  - Edge F1 by `edge_accounting`: a predicted edge is TP iff both endpoints are matched and
    the matched GT pair is a GT edge.
  - The missed-edge-without-endpoint fraction and the MM/MU/UU false-edge classes.
  - β₀/β₁ MAE.
  - common-v2 Branch/Curve/anchor F1 and guide distance via `evaluate_common_vessels`.

  RelationFormer predictions must be converted to the same `rows` format (nodes, node
  scores, all candidate pair scores, curves) before scoring, so that thresholds can be
  swept identically.
- **Thresholds.** Selected on `monitor_train` only (node then edge, same grids). Validation
  is also reported at a fixed 0.5/0.5. Validation never selects a threshold, checkpoint or
  hyperparameter.
- **Literature-scale RelationFormer control.**
  - Recipe: B0 on IXI full 10k training patches, 100 epochs, batch 32, initialized from the
    SyntheticMRI checkpoint.
  - That is about 1.0M patch exposures (about 31k updates).
  - It is evaluated on the plan's `val_ids` with the evaluator above.
  - It answers whether more optimization alone closes the IXI underfit (train node/edge
    F1 about 0.39/0.09). D08 already suggests it does not on BRAVA (train 0.394/0.090
    after about 2.6M exposures).
- **Comparison.** Report train-monitor and validation results side by side, with
  subject-paired bootstrap CIs on the val differences, as in the oracle-geometry study.
  An architecture "wins" only if it improves validation *and* closes the train-fit gap
  that motivated this work.

---

## 6. Results — 30 Oct 2026 [DONE]

All numbers are **validation**, subject-disjoint, with node and edge thresholds selected
on `monitor_train` only. Node and edge F1 use a 2 mm tolerance; Branch and Curve F1 are
common-v2. Each arm uses the cohort listed in §5 (`plan.json`), except the
literature-scale control, which trained on the full IXI train split.

### 6.1 All arms

| Model | Training | IXI node / edge / Branch / Curve | BRAVA node / edge / Branch / Curve |
|---|---|---|---|
| (1) RelationFormer B0, earlier modern run | 5.5k updates (IXI) | 0.365 / 0.088 / 0.010 / 0.215 | — |
| (1) RelationFormer B0, D08 (earlier) | 82.2k × 32, SyntheticMRI init | — | 0.356 / 0.068 / 0.006 / 0.122 |
| (1) RelationFormer B0, literature scale (new) | 31.2k × 32, full 10k IXI, SyntheticMRI init | 0.508 / 0.156 / 0.024 / 0.344 | — |
| (2) Heatmap-anchored RelationFormer | ~15k × 32, SyntheticMRI init, frozen probe | 0.640 / 0.263 / 0.123 / 0.472 | 0.547 / 0.154 / 0.035 / 0.229 |
| **(3) Two-stage dense model** | **15k × 8, from scratch** | **0.648 / 0.315 / 0.163 / 0.677** | **0.543 / 0.217 / 0.054 / 0.365** |

On SyntheticMRI, the dense model reaches node 0.890, edge 0.792, Branch 0.429 and Curve
0.936. The best earlier RelationFormer there (A-xyz) had 0.817 / 0.634 / 0.27 / 0.67.

The dense model's training-monitor scores are within about 0.05 of its validation scores:
- IXI train node/edge 0.662 / 0.362;
- BRAVA 0.553 / 0.229.

So it is not overfitting.

Sources:
- `$SCRATCH/experiments/gnbm/dense_graph_20261030/<dataset>/spline6/final-v2.json`
  (corrected evaluation; the original `final.json` gives slightly lower IXI/BRAVA edge and
  Curve F1);
- [`../../experiments/anchored_rf_20261030/`](../../experiments/anchored_rf_20261030/)
  `final-{IXI,BRAVA}.json`;
- [`../../experiments/ixi_long_20261030/final.json`](../../experiments/ixi_long_20261030/final.json).

### 6.2 Paired comparison: dense (3) vs anchored (2)

Both arms are decoded from saved per-patch rows with their own rule and thresholds. The
pooled counts reproduce each arm's reported numbers exactly. Differences are bootstrapped
over validation subjects (2000 resamples):

| Dense − anchored (95% subject CI) | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|
| IXI (26 subjects, 512 patches) | +0.007 [+0.001, +0.015] | **+0.053** [+0.042, +0.064] | **+0.039** [+0.030, +0.049] | **+0.204** [+0.191, +0.218] |
| BRAVA (8 subjects, 1,796 patches) | −0.004 [−0.007, −0.001] | **+0.063** [+0.057, +0.068] | **+0.019** [+0.017, +0.021] | **+0.136** [+0.114, +0.156] |

Script: [`../../scripts/compare_dense_anchored_20261030.py`](../../scripts/compare_dense_anchored_20261030.py).
Output: `../../experiments/dense_graph_20261030/paired-dense-vs-anchored-{IXI,BRAVA}.json`.

### 6.3 What this shows

1. **More training helps RelationFormer, but does not close the gap.** On IXI, literature-scale
   training lifted B0 from 0.365/0.088 to 0.508/0.156 node/edge F1. It was still slowly
   improving: val node F1 was 0.434 / 0.481 / 0.508 at epochs 25 / 50 / 100. It stays far
   below the dense model, which used about 8× fewer exposures, no pretraining and 2.4× less
   training data.
2. **Anchoring fixes RelationFormer's nodes, not its edges.** Anchored node F1 matches the
   dense model, since both take nodes from a full-resolution heatmap. Edges, branches and
   curves stay clearly worse. This fits the diagnosis that the relation and spline decisions
   still read the 2×2×2 memory.
3. **The two-stage dense model is the best option measured.** It wins edge F1 by +0.05–0.06,
   Branch F1 by +0.02–0.04 and Curve F1 by +0.14–0.20 over the anchored arm, with every CI
   excluding zero. It also beats every RelationFormer run on record. This supports the
   user's choice.
4. **Real-data graphs are still far from solved.** IXI edge F1 is 0.315 and BRAVA 0.217, and
   Branch F1 stays low (0.16 / 0.05). Many errors remain node-driven: 82–86% of missed
   edges lack a recovered endpoint pair.

### 6.4 Caveats

- Budgets and initialization differ between arms:
  - (2) and the literature-scale (1) start from a SyntheticMRI checkpoint at batch 32;
  - (3) is from scratch at batch 8.
  These differences favour the RelationFormer arms, so the dense model's lead is
  conservative.
- Single seed per arm.
- Validation only: no test data was loaded.
- BRAVA has only 8 validation subjects, so its CIs reflect few subjects.
- The literature-scale control was run on IXI only. BRAVA's long-budget evidence is D08.

### 6.5 Average precision (validation) [DONE]

Script: [`../../scripts/dense_ap_20261030.py`](../../scripts/dense_ap_20261030.py). Output:
`../../experiments/dense_graph_20261030/average-precision-validation.json`.

Definitions are the same as in the literature-scale RelationFormer control:
- **Node AP @2 mm:** all proposals are ranked by score and greedily matched to GT within
  2 mm. "Proposals" means every peak ≥ 0.01 for the dense model and all 256 queries for
  RelationFormer.
- **Edge ranking AP:** AP over the candidate pairs among the nodes retained at the selected
  threshold. Two variants: all retained pairs, and pairs whose two endpoints are both matched.
- **Box mAP/mAR:** the GrEVe/RelationFormer COCO-style metric from
  `training/evaluation/metrics.py`. IoU 0.50:0.95, at most 40 detections, GT node boxes of
  side 0.2, edge boxes spanning the two end nodes with half-width 0.1. It is applied to the
  graph decoded at the train-selected thresholds.
  - The dense and anchored arms have no box head, so they use fixed side-0.2 boxes on the
    predicted nodes, which is the GT box convention.
  - RelationFormer B0 uses its regressed boxes.

| Model / dataset | Node AP @2 mm | Edge ranking AP (all / both endpoints matched) | Box node mAP / mAR | Box edge mAP / mAR |
|---|---|---|---|---|
| Dense, SyntheticMRI | 0.860 | 0.938 / 0.991 | 0.638 / 0.701 | 0.622 / 0.681 |
| Dense, IXI | 0.613 | 0.537 / 0.691 | 0.337 / 0.409 | 0.215 / 0.298 |
| Dense, BRAVA | 0.502 | 0.288 / 0.690 | 0.220 / 0.367 | 0.089 / 0.188 |
| Anchored RF, IXI | 0.606 | 0.383 / 0.614 | 0.312 / 0.377 | 0.146 / 0.263 |
| Anchored RF, BRAVA | 0.503 | 0.159 / 0.597 | 0.213 / 0.349 | 0.048 / 0.167 |
| RelationFormer B0, literature scale, IXI | 0.446 | 0.165 / 0.486 | 0.126 (regressed boxes) | 0.066 |

### 6.6 Traced connectivity vs 16-NN (validation) [DONE]

**Question.** Does centreline-traced connectivity recover more of the long edges (arc ≥ 20 mm,
especially BRAVA) without losing short-edge precision?

**What was compared.** Two arms, both warm-started from each dataset's `spline6/model.pt` and
continued for 7.5k updates × 8 at LR 3e-4:
- **Traced + adaptive K** ("traced"; jobs 1743935 / 1743937 / 1743939). Script:
  [`../../scripts/dense_graph_traced_20261030.py`](../../scripts/dense_graph_traced_20261030.py).
  - Candidates: pairs whose geodesic Voronoi regions on the predicted centreline touch, plus
    the 16-NN pairs as a fallback.
  - The corridor follows the traced path.
  - Geometry uses K8, K16 or K24 per edge, chosen by a fixed arc / tortuosity rule.
- **16-NN continued** ("16-NN", the control; jobs 1743936 / 1743938 / 1743940): the dense
  model of §3 unchanged, with the same warm start and budget.

Thresholds were selected on `monitor_train` only, as in §6.1.

| Dataset | Arm | Thresholds (node / edge) | Train node / edge | Val node / edge / Branch / Curve |
|---|---|---|---|---|
| SyntheticMRI | 16-NN | 0.45 / 0.90 | 0.900 / 0.820 | 0.893 / 0.793 / 0.421 / 0.923 |
| SyntheticMRI | traced | 0.45 / 0.90 | 0.900 / 0.809 | 0.894 / 0.797 / 0.441 / 0.945 |
| IXI | 16-NN | 0.30 / 0.80 | 0.670 / 0.378 | 0.653 / 0.319 / 0.168 / 0.673 |
| IXI | traced | 0.35 / 0.75 | 0.668 / 0.344 | 0.649 / 0.313 / 0.179 / **0.770** |
| BRAVA | 16-NN | 0.30 / 0.75 | 0.559 / 0.222 | 0.545 / 0.202 / 0.050 / 0.291 |
| BRAVA | traced | 0.30 / 0.75 | 0.558 / 0.205 | 0.546 / 0.206 / 0.062 / **0.395** |

**Paired difference** (traced − 16-NN, 95% CI over validation subjects, 2000 resamples). The
pooled counts reproduce both arms' reported numbers exactly.

| Dataset | Node F1 | Edge F1 | Branch F1 | Curve F1 |
|---|---|---|---|---|
| SyntheticMRI (13 subj.) | +0.001 [−0.003, +0.004] | +0.004 [−0.002, +0.010] | +0.020 [+0.013, +0.027] | +0.022 [+0.018, +0.027] |
| IXI (26 subj.) | −0.004 [−0.011, +0.001] | −0.006 [−0.013, +0.001] | +0.011 [+0.001, +0.022] | **+0.097** [+0.085, +0.112] |
| BRAVA (8 subj.) | +0.001 [−0.001, +0.003] | +0.003 [+0.001, +0.006] | +0.012 [+0.008, +0.016] | **+0.104** [+0.096, +0.113] |

**GT edges recovered, by arc length** (fraction of validation GT edges; each arm at its own
thresholds; "endp." = fraction with an endpoint missing):

| Dataset | Arc (mm) | GT edges | 16-NN recovered (endp. / rejected) | Traced recovered (endp. / rejected) |
|---|---|---|---|---|
| IXI | 0–5 | 6,801 | 0.221 (0.734 / 0.045) | 0.211 (0.759 / 0.031) |
| IXI | 5–10 | 2,063 | 0.419 (0.473 / 0.108) | 0.412 (0.539 / 0.050) |
| IXI | 10–20 | 1,456 | 0.362 (0.473 / 0.157) | 0.388 (0.533 / 0.076) |
| IXI | 20–40 | 565 | 0.170 (0.481 / 0.304) | **0.299** (0.545 / 0.143) |
| IXI | ≥ 40 | 83 | 0.012 (0.434 / 0.410) | **0.145** (0.554 / 0.241) |
| BRAVA | 0–5 | 5,011 | 0.157 (0.769 / 0.074) | 0.148 (0.777 / 0.074) |
| BRAVA | 5–10 | 4,122 | 0.223 (0.736 / 0.041) | 0.212 (0.753 / 0.036) |
| BRAVA | 10–20 | 5,271 | 0.187 (0.731 / 0.082) | 0.199 (0.742 / 0.058) |
| BRAVA | 20–40 | 3,791 | 0.083 (0.778 / 0.134) | **0.106** (0.790 / 0.102) |
| BRAVA | ≥ 40 | 1,251 | 0.024 (0.814 / 0.137) | **0.052** (0.824 / 0.103) |
| SyntheticMRI | 20–40 / ≥ 40 | 1,411 / 145 | 0.809 / 0.566 | 0.819 / **0.710** |

**Traced candidates.** These are the share of validation candidate pairs that come from
tracing rather than from the 16-NN fallback:
- SyntheticMRI 9.6%, IXI 13.1%, BRAVA 12.0%.
- K16 / K24 were used on 3.4% / 0.4% (SyntheticMRI), 4.9% / 1.0% (IXI) and 6.5% / 2.8%
  (BRAVA) of candidates.

**Reading.**
1. **Long edges: yes, in relative terms.** Tracing cuts the "endpoints found but edge
   rejected" share at ≥ 20 mm: by about half on IXI and by about a quarter on BRAVA. IXI
   ≥ 40 mm rises from 0.012 to 0.145, and BRAVA ≥ 40 mm from 0.024 to 0.052. In absolute
   terms BRAVA long edges are still almost all missed. Their limit is now the nodes: 78–82%
   of BRAVA edges ≥ 20 mm lack a recovered endpoint pair in both arms, and tracing cannot
   fix that.
2. **Short edges and precision: essentially unchanged.**
   - Edge F1 differences are within ±0.006. The IXI CI touches zero; the BRAVA gain is
     +0.003.
   - Overall edge precision is unchanged on SyntheticMRI and BRAVA (0.864 vs 0.863;
     0.285 vs 0.286). It is 0.022 lower on IXI (0.361 vs 0.383), where the train-selected
     thresholds moved (node 0.30 → 0.35, edge 0.80 → 0.75).
   - Recovery of 0–10 mm edges drops by ≤ 0.011, almost entirely through more missing
     endpoints rather than more rejections.
3. **Curve F1 is the clear win.** +0.10 on IXI and BRAVA, with tight CIs, plus a small
   Branch F1 gain (+0.01–0.02). Two caveats apply:
   - Tracing and adaptive K changed together, so this run cannot say how much comes from
     the traced path and how much from K16 / K24.
   - Part of the BRAVA gap is the control's drop (see below).
4. **Topology.** β0 MAE improves on IXI (4.29 → 1.68) and BRAVA (4.77 → 4.08). β1 MAE
   worsens on IXI (2.72 → 3.50) and improves on BRAVA (0.79 → 0.62).

**Recommendation for the next 3D model.** Keep traced connectivity: it costs nothing
measurable in node or edge F1 and gives +0.10 Curve F1 on both real datasets. Do not expect
it to move edge F1. On BRAVA, long edges are now limited by node recall.

**Note on the BRAVA control.** The 16-NN continued run's BRAVA Curve F1 fell from 0.365
(`spline6`, §6.1) to 0.291, while its train-selected node threshold moved from 0.25 to 0.30.
The traced arm (0.395) is above both. This looks like a threshold-selection effect and
should not be over-interpreted.

**Note on the long-edge reference.** The handoff quoted 0.106 (20–40 mm) and 0.025 (≥ 40 mm)
for the BRAVA control. Recomputing it from `knn_continued/final-validation-rows.pkl` at its
own thresholds (0.30 / 0.75) gives 0.083 / 0.024 (table above); 0.106 / 0.052 is the
**traced** arm.

**Caveats.**
- Single seed per arm.
- Validation only: no test data was loaded.
- BRAVA has 8 validation subjects.
- The arms are not fully symmetric. The traced arm's edge head (its input size changed)
  and all geometry heads start fresh from the warm start (see `warm-start.json`). The
  control inherits every tensor. This favours the control.
- Tracing and adaptive K are confounded (see Reading, item 3).

Sources:
- `$SCRATCH/experiments/gnbm/dense_graph_20261030/<dataset>/{traced_adaptive,knn_continued}/final.json`
  (+ `final-validation-rows.pkl`).
- Control arc-length breakdown: `../../experiments/dense_graph_20261030/knn-continued-arc-length-<dataset>.json`
  (job 1748432, `cluster/jean_zay/knn_arc_breakdown_cpu.slurm`).
- Paired bootstrap: [`../../scripts/compare_traced_knn_20261030.py`](../../scripts/compare_traced_knn_20261030.py)
  → `../../experiments/dense_graph_20261030/paired-traced-vs-knn-<dataset>.json`
  (job 1748998).

### 6.7 Dense vs RelationFormer B0 vs straight-edge RelationFormer, same evaluator (1 Nov) [DONE]

**Update, 2 Nov — trained straight lineage scored [DONE].** On the user's request a
straight-edge RelationFormer was trained with B0's exact recipe and spline loss weight 0
(SyntheticMRI 12k, IXI 100 epochs, BRAVA 300 epochs; `scripts/straight_lineage_20261101.py`).
It was scored with the same evaluator: `../../experiments/b0_straight_dense_20261101/<dataset>-with-trained.json`.

Validation, train-selected thresholds, node / edge / Branch / Curve F1:

| Dataset | Trained straight | B0 |
|---|---|---|
| SyntheticMRI | 0.794 / 0.608 / 0.197 / 0.567 | 0.794 / 0.607 / 0.197 / 0.615 |
| IXI | 0.504 / 0.154 / 0.022 / 0.314 | 0.508 / 0.156 / 0.024 / 0.344 |
| BRAVA | 0.364 / 0.069 / 0.005 / 0.146 | 0.368 / 0.073 / 0.005 / 0.161 |

B0 − trained straight, with subject 95% CIs:
- node, edge and Branch F1 within ±0.005, CIs spanning about zero (BRAVA edge +0.005 [0.000, 0.008]);
- Curve F1 +0.048, +0.030 and +0.015.

The trained straight model's Curve F1 matches B0's graphs drawn as chords to within 0.01.
So training with the spline loss does not change the detector; it only improves geometry, as
the `graph_only` route predicts. The text below describes the chord proxy that preceded
this run.

Full report: [`../../experiments/b0_straight_dense_20261101/report.md`](../../experiments/b0_straight_dense_20261101/report.md).

- **No straight run was trained.** With B0's `graph_only` route, and no gradient clipping in
  3D, the spline head cannot affect the detector. A straight-only model therefore decodes
  B0's graph and draws each edge as its chord, so B0's graphs were scored with splines and with
  chords.
- **Coverage:** B0 SyntheticMRI (model-12000) and BRAVA (D08 epoch 300) were newly scored with
  the IXI-long evaluator on the dense model's populations; IXI reuses epoch 100.

Validation results, train-selected thresholds (node / edge / Branch / Curve F1):

| Dataset | Dense | B0 | Straight |
|---|---|---|---|
| SyntheticMRI | **0.890 / 0.792 / 0.429 / 0.936** | 0.794 / 0.607 / 0.197 / 0.615 | 0.794 / 0.607 / 0.197 / 0.564 |
| IXI | **0.648 / 0.315 / 0.163 / 0.677** | 0.508 / 0.156 / 0.024 / 0.344 | 0.508 / 0.156 / 0.024 / 0.321 |
| BRAVA | **0.543 / 0.217 / 0.054 / 0.365** | 0.368 / 0.073 / 0.005 / 0.161 | 0.368 / 0.073 / 0.005 / 0.136 |

- **Dense vs both:** every dense − B0 and dense − straight difference has a 95% subject CI
  that excludes zero.
- **Spline vs chord:** the spline adds only Curve F1, +0.02 to +0.05 over chords.
- **Exception:** at a fixed 0.5 / 0.5 cut-off on BRAVA, the dense model scores 0.130 / 0.012
  and is below B0 (0.356 / 0.068). Its threshold must be calibrated on training patches.
