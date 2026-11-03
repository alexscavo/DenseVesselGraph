# 4. IXI relation isolation — 2026-10-19

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

The displayed smoke classifier scores come from10 updates on four training patches and are execution checks, not performance conclusions. The example is preserved as [the inspected smoke panel](../../../../../research/evidence/ixi_relation_isolation_20261019/smoke-sample_000040_0078-pairs.png). Full-run panels use the same deterministic selection rule and will carry the fitted1000-update scores.


## Completed full diagnostic — 2026-10-19

Full job1278148 completed in44seconds, exit0; verified256 training/128 validation IDs, smoke=false,1000 fitting updates per probe, unchanged frozen state. Feature extraction and small readout fitting share one GPU process.

### Results: recovered-endpoint population

All four rows use the same24,054 validation candidate pairs, with1,130 adjacent pairs (4.70% prevalence).

| Scorer | Training AP | Validation AP | Validation precision @0.5 | Validation recall @0.5 |
|---|---:|---:|---:|---:|
| Existing relation head |0.4848|**0.4702**|0.6860|0.1257|
| Refit relation head |**0.5976**|0.3774|0.2571|0.8142|
| Geometry-only probe |0.4020|0.4043|0.1505|0.8841|
| Image+geometry probe |0.4848|0.4261|0.1864|0.8929|

The current relation head contains useful ranking information, but operates at low recall with its existing threshold. Balanced refitting improves training ranking and worsens validation ranking: evidence of a generalization problem in this conditional refit. Its large recall increase is accompanied by lower precision and must not be presented as improved connectivity. New probes trained on balanced pairs are not calibrated at0.5.

### Results: all GT-endpoint pairs

77,550 validation candidates,2,977 positives (3.84% prevalence). Geometry-only AP: train0.3692 / val0.3483. Image+geometry AP: train0.4733 / val0.3505. Added chord-local image information helps fitting but provides little aggregate validation AP gain in this setup. On the matched subset it gives a modest gain over geometry alone (0.4043→0.4261), still below the existing token relation head. Different fitting populations/architectures prevent attributing cross-family gaps solely to features.

Native matched-pair recall is15.62% in the shortest positive arc quartile and3.80% in the longest, consistent with difficulty on long branches. At0.5, nearby-negative FPR on matched pairs is2.44% native,72.10% after token refit,100% geometry-only and89.98% image+geometry. These uncalibrated-probe values expose the precision/recall tradeoff, not a fair final threshold comparison.

### Full-run image observations

Inspected both fixed final panels directly. For `sample_000040_0078`, the adjacent branch bends away from the cyan chord; image scoring raises this positive above the geometry score (0.608 versus0.286), but also assigns the closer nonadjacent pair a high score (0.811 versus0.694). Thus this example visibly illustrates a curved positive missed by simple distance scoring and a nearby distractor that remains difficult. For `sample_000166_0036`, the short positive scores high (image0.895, geometry0.835), while the nonadjacent pair still exceeds0.5 (0.580/0.566). Raw projections show substantial tissue background and can superimpose different vessel depths. Labels come from GT adjacency; projections alone do not verify anatomical connection.

Final panels, exact pair IDs and scores are preserved with the tables under `docs/research/evidence/ixi_relation_isolation_20261019/`. Smoke and full panels have distinct filenames.

### Next research decision

Retain the original relation scorer while testing its operating point with thresholds calibrated on training data and applied unchanged to validation. Its validation AP shows useful ranking that the present cutoff may underuse; quantify the precision/topology cost before changing decoding. Investigate nearby-negative discrimination and image context beyond a straight chord if ranking remains limiting. A repeatable validation benefit from this simple image relation probe has not been established. Keep the successful node-refinement result distinct from these relation results. Further training or threshold experiments have not been launched here.
