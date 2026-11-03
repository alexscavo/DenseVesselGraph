# 2. IXI feature and node diagnostics — 2026-10-18

## Research question and status

Follow-up to [the supervision audit](1-ixi-supervision-audit.md). **Completed and inspected.** Smoke 1262339 completed in 1m00s; full diagnostics 1262340 completed in 1m15s (Slurm elapsed, including setup). Both exit 0. The detector remained frozen, including BN buffers.

- Smoke job **1262339**: four train/four validation patches, 20 probe updates; GPU inference, hooks, frozen-state checks, probe fitting, node accounting and rendering.
- Full job **1262340**, `afterok:1262339`: 128 training / 64 validation patches, 1000 fixed probe updates. One H100, six CPUs, two-hour allocation (a limit, not a runtime estimate).
- Same IXI-only delayed-spline best checkpoint as the supervision audit. Straight pretraining followed by spline fine-tuning. No updates to the detector, no BN recalibration, no data augmentation. Strict checkpoint loading.
- Deterministic subject-round-robin patch selection; train and validation filename-derived subject sets asserted disjoint. No test samples. Read exact IDs/checkpoint step from the output summary.

## 1. Frozen-feature foreground probes

Train small linear voxel classifiers on frozen features:

1. early CNN (`encoder.layer0.relu1`),
2. projected CNN features immediately before transformer encoder,
3. transformer encoder memory reshaped back to the same spatial grid,
4. raw input intensity control,
5. encoder features with randomly permuted training labels (negative control).

All probes use the same 512 sampled training voxels per patch, balanced 256 foreground / 256 background. Scarce foreground uses replacement. Feature standardization uses training samples only. Each readout uses Adam, LR 0.01, weight decay 1e-4, batch 2048, fixed 1000 updates. No validation-based selection or architecture search. Save fitting histories: an unconverged readout is not evidence that features lack information.

Evaluate all probes on identical independently sampled 2048 uniform voxels per training/validation patch, at natural foreground prevalence. Primary score: average precision, with prevalence as the random-ranking reference. Also save ROC AUC and descriptive Dice at 0.5. Scores/Dice are uncalibrated because fitting uses balanced samples. Training evaluation can overlap fitting voxels; it is deliberately a within-training-patch diagnostic. Validation subjects remain disjoint.

Early CNN and transformer stages differ in spatial resolution, so that comparison confounds depth and resolution. **Projected CNN versus encoder** has equal channels/spatial grid and is the main controlled comparison. All sampling uses trilinear normalized dense-grid interpolation (`align_corners=False`); this approximates the true CNN receptive-field phase. Poor absolute fine-vessel reconstruction could reflect coarse spatial sampling. Linear readout failure cannot establish that no nonlinear decoder could recover information.

Detector eval mode is retained throughout feature extraction. All parameters have gradients disabled. Assert unchanged BN/other buffers, unchanged parameter version counters, and absent detector gradients before fitting readouts.

## 2. Node localization breakdown

Use the existing node decoding threshold/argmax rule. Do not change object confidence or graph predictions. Categories derive from GT boundary flags and graph degree, with boundary taking priority; interior junction degree ≥3, termination degree 1, other explicitly retained.

At both 1 and 2 mm:

1. Maximum-cardinality positional matching identifies recovered GT nodes.
2. Reserve remaining predictions within the tolerance of an already recovered GT node as **duplicate candidates** (proximity flag, not anatomical proof).
3. Match remaining predictions/GT up to twice the tolerance to identify **displaced** nodes.
4. Remaining GT nodes are **missing**; remaining predictions are **unmatched other**.

Save per-node rows, coordinates, displacement errors and nearest distance from *any* object query (including confidence-rejected queries). A nearby unselected query flags confidence/query-assignment ambiguity, not a recovered object. The displaced radius is an explicit diagnostic convention, not a new benchmark threshold. Train/validation comparisons are stratified by GT node type; different patch difficulty remains a confound.

A CPU crafted example checked recovered/displaced/missing/duplicate accounting and a perfect foreground AP example. It exposed an ambiguity in secondary matching; duplicate reservation now occurs before displacement matching. Syntax and Slurm shell checks passed. GPU checks are gated by the smoke job.

## Reading the results

- Good projected-CNN AP, weak encoder AP: loss of **linear accessibility** across encoder is supported, not proof of complete information collapse.
- Good CNN and encoder probes, poor node recovery: investigate object-token decoding, node objectives/query assignment and confidence filtering.
- Weak probes including early CNN: investigate feature learning, image/reference mismatch and spatial resolution; compare the raw-intensity control before blaming the transformer.
- Good training-node recovery but poor validation: evidence for a generalization gap; inspect subtype and patch difficulty before attributing it entirely to overfitting.
- Boundary-only failures: prioritize cropping/target conventions. Junction-only failures: junction supervision and node context deserve attention.
- High duplicate-candidate count: inspect object matching/confidence. High displaced count: focus on coordinate localization. Missing nodes require examining both confidence-rejected queries and image features.

No new detector loss, architecture change, GT regeneration or full training is authorized by these diagnostic outcomes alone. Results guide the next discussion.

## Outputs and reproduction

Full results: `experiments/ixi_features_nodes/20261018/full-1262340/`:

- `results.md`: generated score/node tables with embedded images.
- `summary.json`: checkpoint, exact IDs, feature shapes, protocol, aggregate scores/counts.
- `probe-per-patch.csv`, `node-errors.csv`: detailed analysis.
- `probe-history.json`: optimization checks.
- `overview.png`, `train-*.png`, `val-*.png`: comparative panels. First two patches per split, selected without looking at performance. Node overlays show points within 1.5 voxels of the displayed slice; absence in a slice is not absence in the whole patch.
- `probes.pt`: tiny fitted readouts and standardization parameters. No feature caches or duplicate detector weights saved.

Logs: `experiments/spline_learning/logs/ixi-features-nodes-{1262339,1262340}.{out,err}`.

Implementation: `scripts/diagnose_ixi_features_nodes.py`, `cluster/jean_zay/ixi_features_nodes.slurm`. Run via `sbatch ... smoke`, then `sbatch --dependency=afterok:SMOKE_JOB ... full`. Each output directory includes its job ID and refuses overwrite. Submission metadata is tracked; generated outputs remain local until compact evidence is reviewed and archived.


## Results and interpretation

Same checkpoint step 28170. 128 training patches, 64 validation patches, natural evaluation prevalence 0.80% / 0.86%. These AP values are **voxel foreground probe AP**, not graph node/edge mAP.

| Probe | Train AP | Validation AP | Feature grid |
|---|---:|---:|---|
| Early CNN |0.6318|0.6551|32³, 64 channels|
| Projected CNN |0.0263|0.0271|2³, 552 channels|
| Transformer encoder |0.0287|0.0246|2³, 552 channels|
| Raw intensity |0.6437|0.6802|64³, 1 channel|
| Encoder, shuffled fitting labels |0.0231|0.0157|2³, 552 channels|

The early CNN has useful foreground information that transfers to these validation subjects. Raw intensity alone is also informative against these masks; neither result establishes learned topology or foreground specificity against every bright non-vessel structure. The inspected 000166 slice shows a strong early-probe response at a bright boundary outside the mask.

**No strong encoder-specific collapse is demonstrated for this trained IXI checkpoint:** projected-CNN and encoder readouts are similarly weak. Their 2×2×2 grid limits a linear spatial readout to very smooth fields. Information may still be encoded in channels and require a different decoder. These results cannot resolve collapse in the earlier failed scratch checkpoint, which was not loaded here. Coarse spatial representation/readout is a more immediate hypothesis to test than attributing everything to the transformer.

Probe fitting histories: early-CNN balanced BCE 0.580→0.174; projected CNN 0.739→0.404; encoder 0.767→0.396. They optimized, with some continued slow improvement; this is not a convergence proof. Shuffled-label BCE remained near 0.70. Its AP exceeds prevalence, so do not claim statistical superiority over a randomization distribution from this single control. Correlated patches/voxels and incidental spatial ranking remain relevant.

### Nodes at 2 mm

| GT type | Train recovered | Validation recovered |
|---|---:|---:|
| All |1726/2721 = 63.43%|847/1657 = 51.12%|
| Boundary |785/1127 = 69.65%|412/665 = 61.95%|
| Interior junction |593/877 = 67.62%|275/573 = 47.99%|
| Interior termination |348/717 = 48.54%|159/417 = 38.13%|

Validation also contains 283 displaced GT nodes (17.08%, matched between 2 and 4 mm) and 527 missing (31.80%). Duplicate candidates: 325 train / 209 validation; unmatched other predictions: 765 / 476. Raw counts must not be compared without accounting for population size.

There is a node generalization gap, especially for junctions, **plus substantial training-set error**. This is not explained by pure validation overfitting alone. Interior terminations are hardest in both splits; boundary nodes are the strongest category, so there is no evidence of a boundary-only implementation failure here.

547 of 810 unrecovered validation GT nodes have some object query within 2 mm. This includes low-confidence queries and already-assigned queries; it is not unique matching and does not prove confidence suppression. It motivates a future confidence/query-assignment check, not automatic threshold lowering (which could exacerbate duplicates).

### Inspected artifacts and next decision

Inspected full-run overview and both fixed validation slice panels directly. They confirm smooth projected/encoder readouts and spatially detailed early-CNN responses. Node slice overlays are examples, not an exhaustive node-label audit.

Recommended next intervention to discuss: give node localization access to higher-resolution image features, with a controlled comparison from the same checkpoint, or first test a coordinate-aware readout to separate channel encoding from spatial-resolution limits. Preserve the graph-only control and identical data/updates. A confidence/assignment diagnostic is a cheaper complementary check. Do not claim that increasing feature resolution is already proven to fix graph extraction. Relation isolation remains a separate, unrun experiment.


### Spatial-resolution control (CPU, no additional model training)

Reused the exact evaluation voxel sampling, verifying per-patch foreground counts and the raw-intensity AP check. Mean-pool raw/mask to a coarse grid and trilinearly reconstruct:

| Reference/control | Train foreground AP | Validation foreground AP |
|---|---:|---:|
| Raw intensity pooled to 2³ |0.0112|0.0110|
| GT mask pooled to 2³ |0.0789|0.0665|
| GT mask pooled to 32³ |0.9451|0.9512|

Even a coarse reconstruction built directly from labels loses most fine-vessel localization. This confirms a substantial spatial-resolution confound in the simple pointwise probe. The pooled mask is **not a rigorous upper bound** for learned channel representations or a model prediction. It supports testing access to finer spatial features, but cannot establish that channel-encoded detail has been destroyed.

Direct raw-intensity AP differs from the float fitted scalar readout by 0.000286 on training and <0.000001 on validation (float rank ties/saturation can affect AP). Per-patch labels/counts reproduce exactly; the difference is recorded, not hidden. `scripts/ixi_probe_resolution_control.py` writes `resolution-controls.json`; the Slurm wrapper now includes it for future full runs. This completed run used the same script as a separate CPU follow-up.

### Preserved evidence

Reviewed results, CSVs, fitting histories, small fitted probes and the five PNG panels are copied to [the evidence folder](../../../../../research/evidence/ixi_features_nodes_20261018/results.md). The original job output remains intact. This is about 2.5 MB; no detector weights or feature caches were duplicated. The research interpretation above is authoritative over the automatically generated descriptive tables.


## Image inspection: method and observed examples

The overview plots measured foreground average precision and GT node outcome fractions. For fixed training/validation examples, panels display raw MRI, the binary mask, node crosses, and spatial foreground scores from fitted linear classifiers. These score maps are obtained by applying each readout to frozen features, trilinearly interpolating logits to the input grid, then applying sigmoid. A single greatest-mask-area slice along axis0 is displayed. Node overlays include points within1.5voxels of the slice. The maps visualize what a small supervised readout can recover from each feature stage.

Inspected `val-sample_000040_0078.png`: the early-CNN map follows the bright curved vessel region, whereas projected-CNN/encoder maps vary smoothly over broad areas. In `val-sample_000166_0036.png`, early-CNN scores also light up a bright tissue boundary outside the vessel mask, showing that fine spatial response does not guarantee vessel specificity. The raw-intensity control highlights bright image structures as expected. These observations agree with the coarse2³ versus early32³ feature grids and motivate resolution-aware interpretation of AP.
