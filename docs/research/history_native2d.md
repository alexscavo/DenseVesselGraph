# Native 2D RelationFormer and the Image-to-Graph-Vessel inventory — consolidated history (to 29 Oct 2026)

**Status.** This file covers the native 2D RelationFormer work, Native2D-01 to Native2D-05 (27–29 Oct):
- the AV-WIDE, REAVER and OCTA-Synth datasets;
- B0 vs A-xyz;
- mixed plant/vessel pretraining;
- a straight-edge control.

It also covers the inventory of the companion Image-to-Graph-Vessel repository. It consolidates
archived documents and was written on 3 Nov 2026. **Superseded by the native 2D dense model**,
which trains from scratch ([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §5;
[report](../../experiments/dense_graph_2d_20261030/report.md)). It beats 2D RelationFormer B0 on
all three 2D datasets.

## Context

**Question.** Can the original 2D image-to-graph RelationFormer learn graphs from real 2D vessel
images? The model is a ResNet101 encoder with a deformable-DETR transformer, the GrEVe-era model
inherited from Image-to-Graph-Vessel. It was trained with this project's modern losses, an optional
local coordinate refiner (A-xyz) and a K8 spline head. Two follow-up questions:
- does mixed plant/vessel pretraining help?
- do learned curved edges beat straight chords?

**Datasets:**
- AV-WIDE (retinal fundus);
- a calibrated REAVER microscopy subset (diaphragm, 60× magnification);
- OCTA-Synth (synthetic OCT angiography);
- for pretraining, the native 2D grapevine plant crops.

**Protocol:**
- fixed 128² crop grids, split by source image;
- validation only, with **test arrays never opened**;
- fixed node/edge confidence cutoff 0.5;
- one seed.

Distances are in pixels, or in microns for REAVER, never millimetres.

## Shorthand

- **Native2D-01 … -05**: the five dated native 2D studies (implementation, two AV-WIDE pilots,
  the 15-run pretraining study, the straight control).
- **B0**: the baseline RelationFormer recipe, here the full 2D model:
  - ResNet101, 512-wide, 3+3 transformer layers, 4 feature levels;
  - 5 relation tokens, 96 object queries;
  - node focal loss γ 2, matched-edge cross-entropy with ratio 0.15, Hungarian matching;
  - K8 head.
- **A-xyz**: B0 plus a zero-initialized local refiner that nudges node coordinates using early CNN
  features. In 2D it uses 9 samples at −4/0/+4 px.
- **K8**: endpoint-constrained cubic B-spline with 8 controls per edge, trained on the `graph_only`
  route (the geometry loss never reaches the detector).
- **Chord**: the straight segment between an edge's two nodes. "Same-graph K8 vs chord" means both
  are scored on identical predicted nodes and edges.
- **scratch100**: 100 vessel-only epochs from random initialization.
- **mixed50+100**: 50 epochs of mixed plant + target pretraining, then 100 target-only epochs.
- **Raw / filtered node recall**: GT nodes covered by any query, or only by queries above the 0.5
  cutoff.
- **Branch / Curve F1**: the common physical metrics, here at 1 native unit. Branch F1 counts
  reconstructed branches; Curve F1 counts centreline samples within tolerance.
- **WS1**: workstream 1 of the 30 Oct plan, "2D backbone fix (ImageNet initialization)"
  ([network_proposals_20261030.md](network_proposals_20261030.md) glossary).
- **D4**: the eight rotations and flips of a square crop.

## 1. Image-to-Graph-Vessel companion repository: inventory [DONE, audit of 23 Oct]

The sibling repository holds separate **2D and 3D RelationFormer pipelines** for one thesis:
biological pretraining sources (plants, HALVS leaves) transfer better to vessel graphs than roads
do. The 3D target was SyntheticMRI and the 2D target OCTA-Synth
([image-to-graph-vessel](../archive/legacy-20261103/research/image-to-graph-vessel.md)).

**Facts that still matter:**
- **The adversarial domain loss was dead.** It was computed and logged, but missing from the
  optimized loss list
  ([../experiments/domain_adaptation.md](../experiments/domain_adaptation.md)).
- **Names and flags are not reliable.** Some config names say "SupCL" while setting
  `USE_SUPCL=false`. `AUTO_RESUME` is unwired. Several YAMLs and generators hardcode `/data/scavone`
  paths.
- **Plant inputs are segmentation-derived renderings,** not the original RGB photographs.
- **No continuous spline head exists there.** Curved edges belong to this repository only.
- **No 2D checkpoint was recoverable.** The `checkpoints/mixed_plants_halvs_2d` folder is empty
  ([native2d_heldout](../archive/legacy-20261103/docs/research/native2d_heldout_20261028.md)).

**Historical metrics** ([historical-evaluation-metrics.csv](../archive/legacy-20261103/research/evidence/image-to-graph-vessel/historical-evaluation-metrics.csv),
indexed in [archive-index.csv](../archive/legacy-20261103/research/evidence/image-to-graph-vessel/archive-index.csv)):
- final metric blocks from 32 archived August evaluation logs;
- legacy box-AP definitions;
- values below are rounded to 4 decimals.

| Archived evaluation | Node mAP | Edge mAP | SMD |
|---|---:|---:|---:|
| 2D OCTA-Synth, from biological | 0.6936 | 0.6201 | 0.0010 |
| 2D OCTA-Synth, from roads only | 0.1984 | 0.1192 | 0.0041 |
| 3D SyntheticMRI, from "baseline" | 0.0929 | 0.1572 | 0.0147 |
| 3D, from all available | 0.1232 | 0.1886 | 0.0154 |
| 3D, from plants only | 0.0826 | 0.1483 | 0.0155 |
| 3D, from HALVS only | 0.0368 | 0.0426 | 0.0182 |
| 3D, from roads only | 0.0550 | 0.0985 | 0.0182 |

- **In 2D, the biological source looks far better than roads.**
- **In 3D, the ranking is not stable.** The `wd1` variants reverse it: roads-only-wd1 node mAP
  0.1190 against plants-only-wd1 0.0882.
- **Caveats on these numbers:**
  - the evaluation split, checkpoints and data membership were never reconciled;
  - "baseline" is a run label, not proof of no pretraining;
  - the domain loss was inert;
  - these numbers are not comparable with the common-v2 metrics.

**Decision (23 Oct).** Keep the repository as provenance and do not port it silently. Do not
re-enable its DA, SupCL, hard-negative mining or BN-calibration paths just because they exist.

## 2. Native port, data audit and tiny fit (Native2D-01) [DONE]

**Design decision** ([decisions](../archive/legacy-20261103/research/decisions.md) and
[architecture](../archive/legacy-20261103/research/architecture.md), native 2D sections of 27 Oct):
- reuse the original 2D RelationFormer instead of emulating 2D with a 3D volume;
- port the modern focal / matched-edge / A-xyz / K8 contracts;
- leave out the negative 3D variants N08, R06 and R07 (earlier 3D detector and relation-head
  variants that had come out negative);
- keep pixels where no calibration exists;
- treat mask/skeleton-derived graphs as derived labels;
- never repair artery/vein crossings automatically.

**Correctness** ([image-to-graph-vessel](../archive/legacy-20261103/research/image-to-graph-vessel.md),
Native2D-01 sections):
- eight regressions and 13 real-image CPU/CUDA checks passed;
- the full-width ResNet101 model (72,440,408 parameters) passed one forward/backward pass;
- a tiny ResNet18 model reached node/edge/Branch F1 1.0 on two AV-WIDE training crops by update
  500, with loss 11.3064 → 0.1200.
- This shows the implementation can learn. It says nothing about generalization.

**Dataset audit** ([native2d_heldout](../archive/legacy-20261103/docs/research/native2d_heldout_20261028.md),
[native2d_long](../archive/legacy-20261103/docs/research/native2d_long_20261028.md)):

| Dataset | Outcome |
|---|---|
| AV-WIDE | `.mat` files embed the image `I2` and graph `G`. Images 17 train / 6 val / 6 reserved test; image 07 excluded (self-loop with no path). 1,385 / 506 tiles; training maximum 76 nodes → 96 queries. Split by image only; patient identity unknown. |
| REAVER | 36 images audited: 25 strict raw/manual pairs, 11 channel exceptions. Only the 4-train / 2-val diaphragm60x subset was used: 256 / 128 tiles, 0.207194 µm/px, 24 queries. The `*_Auto` folders are algorithm outputs, not ground truth. |
| OCTA-Synth | Local copy empty at first (graph folders were placeholders). Later supplied packed: 500 full images, official split 88 / 40 / 372; regenerated 10,648 / 4,840 tiles; training maximum 52 nodes. |
| Plants (2D) | Native `plants_3d2cut/patches`: 25,867 / 6,398 crops kept, 79 quarantined (zero-length branches); split by parent image. This is grapevine woody branching. |
| AV-DRIVE, AV-INSPIRE | Graphs only; original images excluded by the dataset README. Not used. |
| HALVS 2D, RETA, 20cities | Not ready: no checkpoints, not extracted, or empty. |

## 3. AV-WIDE bounded pilots, B0 vs A-xyz from scratch (Native2D-02, -03) [DONE — negative]

**Protocol** ([native2d_heldout](../archive/legacy-20261103/docs/research/native2d_heldout_20261028.md)):
- 6,000 updates per arm at batch 4;
- identical initialization and exposure hashes;
- all 506 validation tiles.

**Target-fit defect found in Native2D-02:**
- one branch with irregular polyline sampling produced a K8 control at −56,890 px inside a 128²
  crop, which caused loss spikes;
- Native2D-03 refit every target from 64 uniform arc-length samples, bringing the maximum control
  coordinate from 2,007,485.85 to 131.245 px;
- Native2D-02 therefore supports no geometry conclusion.

**Corrected targets can be represented.** On ground-truth graphs, corrected K8 reaches Curve F1
0.99471 against 0.53083 for chords (256 sampled branches). This is a statement about target
capacity, not about learning.

**Result in both pilots:**
- at the 0.5 cutoff, **B0 and A-xyz predict zero nodes and zero edges**, so node/edge/Branch/Curve
  F1 = 0;
- unthresholded node mAP is 0.0880 for B0 and 0.0920 for A-xyz;
- raw node recall @2 px is 0.2374 and 0.2142;
- filtered recall is 0.

The docs stopped at the budget. No threshold sweep and no extension were run.

## 4. Mixed plant/vessel pretraining vs scratch, three datasets (Native2D-04) [DONE — benefit confounded]

**Protocol** ([native2d_long](../archive/legacy-20261103/docs/research/native2d_long_20261028.md)):
- B0 and A-xyz trained as **scratch100** and as **mixed50+100** on each dataset;
- the user corrected a pure-plant protocol to mixed pretraining before any run started;
- pretraining used 51,734 draws per epoch (50:50 plants/target, with replacement) at batch 32;
- vessel-only stages used batch 4;
- all 15 runs completed.

Final checkpoints on full validation, cutoff 0.5
([native2d_results](../archive/legacy-20261103/docs/research/native2d_results_20261029.md)):

| Dataset | Model | Node F1 @2 | Edge F1 @2 | Branch F1 | K8 Curve F1 |
|---|---|---:|---:|---:|---:|
| AV-WIDE | B0 mixed | 0.2045 | 0.0123 | 0.0039 | 0.0853 |
| AV-WIDE | A-xyz mixed | 0.1963 | 0.0088 | 0.0037 | 0.0656 |
| REAVER (µm) | B0 mixed | 0.7455 | 0.4085 | 0.2361 | 0.5478 |
| REAVER (µm) | A-xyz mixed | 0.2040 | 0.0000 | 0.0000 | 0.0129 |
| OCTA-Synth | B0 mixed | 0.8038 | 0.5950 | 0.2318 | 0.6285 |
| OCTA-Synth | A-xyz mixed | 0.8101 | 0.5923 | 0.4079 | 0.6881 |

Branch and Curve F1 are averaged over crops that contain GT edges.

- **All six scratch100 models predict empty graphs** (F1 = 0). Only mixed-pretrained models
  produced graphs.
- **This benefit is confounded.**
  - Mixed pretraining also adds about 1.29 million target-crop exposures per dataset.
  - It is single seed.
  - Validation sets are small: 6 AV-WIDE images, 2 REAVER images.
  - The backbone problem described in §6 applies to these runs.
- **A-xyz is mixed:**
  - on OCTA it improves 1 px localization (node F1 @1 px 0.5061 → 0.6744) and Branch F1;
  - it collapses on REAVER;
  - it raises OCTA duplicates (2,059 → 3,091) and false cycles (4,518 → 4,836, against 0 in GT).
- **AV-WIDE stays near zero.**

## 5. Curved vs straight edges in 2D (Native2D-04 same-graph; Native2D-05 trained straight) [DONE — no measurable spline benefit]

- **Same-graph comparison:** K8 minus chord Curve F1 ranges from −0.0014 to +0.0165 across the six
  mixed models ([native2d_results](../archive/legacy-20261103/docs/research/native2d_results_20261029.md)).
- **Trained straight control** on OCTA-Synth (Native2D-05): the same mixed50+100 recipe with the
  spline head and loss removed
  ([native2d_straight](../archive/legacy-20261103/docs/research/native2d_straight_20261029.md);
  [report](../../experiments/native2d_straight_20261029/report.md)):

| OCTA-Synth B0, final | Node mAP | Edge mAP | Node F1 @2 px | Edge F1 @2 px | Curve F1 |
|---|---:|---:|---:|---:|---:|
| Curved K8 | 0.8049 | 0.4447 | 0.8038 | 0.5950 | 0.6285 |
| Straight only | 0.7992 | 0.5877 | 0.7932 | 0.5903 | 0.6036 |

- **OCTA edges are almost straight.** In the two displayed validation crops, 89.2% (B0) and
  93.5% (A-xyz) of predicted edges bend less than 0.25 px, and so do 90.9% of GT edges. That
  sample is too small to describe the whole dataset.
- **Conclusion:** OCTA-Synth cannot show a spline benefit. Evidence that curved edges help was
  sought in 3D instead, through the 30 Oct oracle-topology geometry study
  ([network_proposals_20261030.md](network_proposals_20261030.md) §0).

## 6. What the later documents say (30 Oct – 2 Nov)

**[DONE] The backbone fix (WS1) did not rescue scratch training.**
- **The bug:** the native port set `init_weights: null`. That leaves a random ResNet101 whose
  FrozenBatchNorm layers are identities, trained at the 2e-5 backbone learning rate meant for a
  pretrained backbone. The original GrEVe recipe used ImageNet weights with ImageNet mean/std input
  ([script docstring](../../scripts/train_native2d_imagenet_20261030.py)). So every Native2D-02 to
  -05 run used this un-normalized random backbone.
- **The fix arm:** vessel-only B0 for 100 epochs with ImageNet initialization.
  - AV-WIDE: node/edge F1 0.020 / 0.001
    ([dense 2D report](../../experiments/dense_graph_2d_20261030/report.md)).
  - REAVER: 0 / 0
    ([final-heldout.json](../../experiments/native2d_imagenet_20261030/runs/reaver_imagenet_B0/final-heldout.json)).
- **Gaps:** no ImageNet arm was run with mixed pretraining or on OCTA-Synth. The current design
  doc names WS1 only in its glossary and has no WS1 results section.

**[DONE] A dense 2D model from scratch beats the best RelationFormer.** The 2D port of the two-stage
dense model (16-nearest-neighbour edges, no tracing) was trained from scratch for 15k updates ×
batch 8. It beats mixed-pretrained B0 everywhere. Validation node / edge F1 @2 units, with
thresholds chosen on the training monitor
([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §5):

| Dataset | RelationFormer B0 (mixed) | Dense 16-NN, from scratch |
|---|---|---|
| AV-WIDE | 0.204 / 0.012 | 0.454 / 0.133 |
| REAVER | 0.745 / 0.408 | 0.806 / 0.589 |
| OCTA-Synth | 0.804 / 0.595 | 0.903 / 0.782 |

- The dense model also wins at fixed 0.5 / 0.5 thresholds (AV-WIDE 0.393 / 0.045).
- AV-WIDE is still node-limited: node recall @2 is 0.41.
- REAVER overfits and has 2 validation images.
- Single seed, no bootstrap
  ([report](../../experiments/dense_graph_2d_20261030/report.md)).

**[PROPOSED] Next step:** porting traced connectivity to 2D
([dense_traced_model_20261102.md](dense_traced_model_20261102.md) §6).

## What carried forward / what was dropped

**Carried forward:**
- **The validated 2D cohorts:**
  - AV-WIDE 17/6 images;
  - REAVER diaphragm60x 4/2 images in microns;
  - OCTA-Synth official 88/40 split;
  - their crop and target pipeline, including the arc-length K8 fit;
  - the pixel/micron unit rules.
- **"Splines improve curves, not graphs".** In 2D the effect was too small to see, because the
  edges are nearly straight.

**Dropped:**
- the native 2D RelationFormer line, superseded by the dense 2D model;
- A-xyz in 2D, which was unstable across datasets;
- plant pretraining for 2D. Its apparent benefit is confounded with extra target exposure and was
  measured with a broken backbone;
- the legacy Image-to-Graph-Vessel recipes, kept as provenance only.

## Sources

Archived originals (paths after the move):
- [native2d_heldout_20261028.md](../archive/legacy-20261103/docs/research/native2d_heldout_20261028.md): dataset audit, Native2D-02 and -03 pilots, target-fit correction.
- [native2d_long_20261028.md](../archive/legacy-20261103/docs/research/native2d_long_20261028.md): Native2D-04 protocol and the OCTA, REAVER and plant cohorts.
- [native2d_results_20261029.md](../archive/legacy-20261103/docs/research/native2d_results_20261029.md): Native2D-04 final results and overlays.
- [native2d_straight_20261029.md](../archive/legacy-20261103/docs/research/native2d_straight_20261029.md): Native2D-05 straight control and the edge-bending measurement.
- [research/image-to-graph-vessel.md](../archive/legacy-20261103/research/image-to-graph-vessel.md): companion-repository inventory, plus Native2D-01 to -05 status sections.
- [research/experiments.md](../archive/legacy-20261103/research/experiments.md), [roadmap.md](../archive/legacy-20261103/research/roadmap.md), [decisions.md](../archive/legacy-20261103/research/decisions.md), [architecture.md](../archive/legacy-20261103/research/architecture.md), [VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md](../archive/legacy-20261103/research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md): only their Native2D sections are summarized here (these are status mirrors of the above, plus the 2D dataset inventory, §10D of the registry).
- [research/evidence/image-to-graph-vessel/](../archive/legacy-20261103/research/evidence/image-to-graph-vessel/historical-evaluation-metrics.csv): legacy evaluation metrics and archive index CSVs.

Experiment reports (in place):
- [../../experiments/native2d_mixed_20261028/report.md](../../experiments/native2d_mixed_20261028/report.md): complete 15-run report.
- [../../experiments/native2d_straight_20261029/report.md](../../experiments/native2d_straight_20261029/report.md): straight-control report.
- [../../experiments/dense_graph_2d_20261030/report.md](../../experiments/dense_graph_2d_20261030/report.md): dense 2D vs RelationFormer, and the ImageNet reference point.
- Companion: [history_pretraining_transfer.md](history_pretraining_transfer.md), covering 3D source pretraining and transfer.
