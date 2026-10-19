# IXI relation isolation — 2026-10-19

## Question and submitted design

Use the completed local-image node-refinement model (parent IXI delayed-spline step28170 plus 1000-update `trained-heads.pt`). Freeze the whole detector, including BN buffers, and extract fixed features. This separates relation scoring from changes in nodes, representations and geometry.

256 training and 128 validation patches selected by the same deterministic subject-round-robin method; split subject IDs asserted disjoint. Four/four for smoke. Save exact IDs and per-patch pair counts. The validation set is used for diagnostic comparison, not a final held-out benchmark.

### A. Recovered endpoints: existing versus refitted relation head

Match decoded predicted nodes to GT nodes by maximum-cardinality positional matching within 2 mm. Enumerate every pair of matched nodes and label by GT adjacency. Score with the original relation MLP, symmetrizing endpoint order exactly as inference does (mean forward/reverse logits, then softmax).

Refit a copy of that same pretrained relation MLP on the fixed training-patch matched-pair features. Node/query features and shared relation tokens remain fixed. Balanced positive/negative minibatches, AdamW LR7e-5, weight decay1e-4, 1000 updates. This changes relation supervision/sampling to a deliberately isolated conditional task; it does not reproduce end-to-end graph training.

Interpretation: if refitting improves validation AP, the existing relation head/supervision is a tractable limitation even with fixed token features. If improvement is restricted to training, generalization remains limiting. Failure of a short refit is inconclusive about all possible optimization settings.

### B. GT endpoints: image evidence versus geometry-only classifier

Enumerate every unordered GT-node pair. Positive means adjacent in the cached GT graph; negative means nonadjacent. Binary adjacency merges parallel edges and excludes self-pairs, so this diagnostic concerns connectivity rather than multiplicity.

Use two equal-size MLPs (128 hidden units):

- **image_geometry:** normalized endpoint midpoint, absolute normalized separation, physical chord length, and local early-CNN features sampled along the GT endpoint chord.
- **geometry_only:** identical inputs/architecture, with the image-feature block zeroed after training-only standardization.

Image sampling: 16 uniformly spaced chord positions. At each, sample a seven-point stencil (centre and ±2 input voxels along each axis), mean over the stencil, then average the sequence with its reversal. This gives an endpoint-order-invariant 16×64 feature vector. Sampling follows the early stride2 convolution's voxel-centre mapping. The image features describe the straight chord vicinity; long curved vessels may leave this region. Use GT endpoints to isolate pair classification with known node locations, not to simulate deployable graph prediction. GT path curves are used only for labels/length strata and visual reference, not as classifier inputs.

Balanced 256-positive/256-negative minibatches sampled with replacement, 1000 AdamW updates, LR1e-4, weight decay1e-4. Feature standardization uses all sampled training candidate pairs only. Fitting histories and small readouts are saved; all cached image/token features are discarded after the process finishes.

### Matched comparisons and metrics

Primary: average precision on every eligible pair, at natural candidate prevalence, separately for training and validation. Save pair counts, prevalence, precision/recall/FPR at0.5 and all pair scores. Balanced fitting makes0.5 descriptive rather than a calibrated deployment threshold.

- Compare native/refitted-token/image/geometry models on the **identical recovered-endpoint pair population**.
- Compare image versus geometry on **all GT pairs**, separately. AP across these two populations is not directly comparable.
- Positive-edge recall by GT arc-length quartiles from the sampled training positives only.
- FPR among nearby nonadjacent pairs: GT endpoint chord ≤5mm, declared descriptive radius.

A strong GT-image classifier with weak matched-token scoring points toward relation-feature access. A strong geometry control indicates distance/location priors explain much of the performance. Weak chord-image scores can reflect insufficient sampled context, ambiguous labels or difficult generalization, not necessarily absence of vessel evidence in the volume.

## Visual inspection method

Two fixed validation patches, selected by the existing deterministic sample order. For each, render the first positive pair in lexicographic node-index order and the nearest nonadjacent GT pair, selected independently of classifier performance.

Each panel shows three maximum-intensity projections of the native raw volume. Green polylines show cached GT paths; cyan dashed chords identify adjacent pairs and magenta dashed chords identify nonadjacent pairs. Endpoints are marked, and image/geometry classifier scores are printed in the title. The dashed chord shows the centre of the sampled image region; it is not a predicted vessel path. Projections are useful for seeing alignment and close distractors, but structures at different depths may overlap. Adjacent-slice review is required to resolve anatomical crossings.

Record specific observations only after the generated examples are inspected. `visual-examples.json` records selected pair indices, distances, labels and scores; `*-pairs.png` and `results.md` make the inspection reproducible.

## Correctness and artifacts

Small CPU checks: feature shape, endpoint-order invariance, ideal classifier AP and empty-population handling. GPU smoke covers checkpoint+adapter loading, frozen-state digest, pair extraction, all three fitting paths, metrics and rendering. Full diagnostics depend on successful smoke completion.

Code: `scripts/diagnose_ixi_relations.py`; job wrapper: `cluster/jean_zay/ixi_relation_isolation.slurm`.

Output: `experiments/ixi_relation_isolation/{smoke,full}-JOBID/`: `summary.json`, `comparison.csv`, `patch-counts.csv`, `fit-history.json`, `pair-scores.npz`, `probe-heads.pt`, `visual-examples.json`, fixed pair PNGs and `results.md`. Full model weights are not duplicated.

## Submission

Smoke **1278142** (30-minute limit); full diagnostics **1278148**, `afterok:1278142` (2-hour limit). These limits are scheduler allocations, not expected runtimes. Models share frozen feature extraction within a job. Results and images are written automatically at completion.

Cross-family token/image comparisons also differ in fitting population (matched pairs versus all GT pairs), initialization (pretrained versus fresh) and classifier architecture. They test feasibility under each diagnostic setup; a difference alone cannot be attributed solely to image-versus-token features. The image/geometry-only pair is the directly controlled feature ablation.


## Smoke inspection — 2026-10-19

Smoke1278142 completed successfully in1m05s, including feature extraction, all three readout fits, metrics, frozen-state check and figures. Inspected `sample_000040_0078-pairs.png` directly. The cyan positive-pair chord cuts across the inside of a curved reference path, demonstrating why chord-centred sampling can miss part of a tortuous branch. The magenta nonadjacent pair is closer (6.83mm versus13.23mm for this positive) and lies near bright vessel structures in the projections. This is an example of why distance alone can be misleading, while projection overlap still requires depth-aware review. “Nonadjacent” means no direct GT edge; nodes may still be connected through other graph nodes.

The displayed smoke classifier scores come from10 updates on four training patches and are execution checks, not performance conclusions. The example is preserved as [the inspected smoke panel](../evidence/ixi_relation_isolation_20261019/smoke-sample_000040_0078-pairs.png). Full-run panels use the same deterministic selection rule and will carry the fitted1000-update scores.
