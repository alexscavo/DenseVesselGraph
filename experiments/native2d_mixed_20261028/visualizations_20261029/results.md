# Completed native 2D results — 29 October 2026

All15 jobs completed successfully: three50-epoch mixed plant/vessel pretraining stages and twelve100-epoch vessel stages. Results below use final100-epoch checkpoints and the complete held-out validation cohorts, at unchanged node/edge confidence0.5. No test inference, threshold changes, additional training or checkpoint selection was performed for these figures.

## Final mixed-pretraining comparisons

Node/edge F1 are micro counts at2 pixels for AV-WIDE/OCTA-Synth and2 microns for the validated REAVER diaphragm60x subset. AP/AR are the existing unthresholded crop-macro IoU detection metrics; undefined empty-GT cases are omitted from those macro means. F1 includes all validation crops, including empty ones.

| Dataset | Model | Node mAP | Edge mAP | Node mAR | Edge mAR | Node F1 | Edge F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| avwide | B0 | 0.2760 | 0.0774 | 0.5178 | 0.5124 | 0.2045 | 0.0123 |
| avwide | Axyz | 0.2713 | 0.0691 | 0.5151 | 0.5001 | 0.1963 | 0.0088 |
| reaver | B0 | 0.4737 | 0.1190 | 0.6855 | 0.6989 | 0.7455 | 0.4085 |
| reaver | Axyz | 0.1815 | 0.0075 | 0.5145 | 0.0331 | 0.2040 | 0.0000 |
| octasynth | B0 | 0.8049 | 0.4447 | 0.8683 | 0.8631 | 0.8038 | 0.5950 |
| octasynth | Axyz | 0.8513 | 0.5736 | 0.9068 | 0.8999 | 0.8101 | 0.5923 |

All six scratch100 models retain zero nodes/edges at0.5, so their reconstruction F1 is0. Their raw detection AP/AR and raw node recall are nonzero and remain in the full results; an empty filtered graph is not an absence of raw query predictions.

## Localization and geometry

OCTA-Synth A-xyz improves tighter localization: node F1@1px0.5061→0.6744 and edge F1@1px0.2210→0.3975. At2px, edge F1 is nearly unchanged0.5950→0.5923 despite improved detection AP. Detection-box ranking and geometrically correct graph edges measure different properties.

The following Branch/Curve values are macro averages over GT-edge-positive validation crops only (498 AV-WIDE,109 REAVER,4,093 OCTA-Synth), avoiding empty/empty inflation. Common-v2 uses1 native-unit anchor/curve tolerances and0.25-unit samples. Straight chords use exactly the same predicted nodes and edges as each model’s spline geometry.

| Dataset | Model | Branch F1 | K8 Curve F1 | Chord Curve F1 | K8 minus chord |
|---|---|---:|---:|---:|---:|
| avwide | B0 | 0.0039 | 0.0853 | 0.0858 | -0.0005 |
| avwide | Axyz | 0.0037 | 0.0656 | 0.0634 | +0.0022 |
| reaver | B0 | 0.2361 | 0.5478 | 0.5492 | -0.0014 |
| reaver | Axyz | 0.0000 | 0.0129 | 0.0128 | +0.0001 |
| octasynth | B0 | 0.2318 | 0.6285 | 0.6154 | +0.0130 |
| octasynth | Axyz | 0.4079 | 0.6881 | 0.6715 | +0.0165 |

A-xyz improves OCTA Branch/Curve metrics, with modest additional K8-over-chord coverage gains. The B0/A-xyz Branch/Curve difference also contains changes in localization and predicted graphs; it is not an isolated learned-spline effect. AV-WIDE remains poor, and A-xyz regresses substantially on REAVER.

## Remaining graph errors

Counts below are sums over validation crops, including overlapping boundary crops; they are not unique full-image anatomical graph counts. Duplicates use the2-native-unit matching tolerance.

| Dataset | Model | Raw node recall | Filtered node recall | Conditional relation recall | Duplicates | Predicted / GT cycles |
|---|---|---:|---:|---:|---:|---|
| avwide | B0 | 0.3229 | 0.1541 | 0.6506 | 52 | 676 / 196 |
| avwide | Axyz | 0.3078 | 0.1432 | 0.5362 | 40 | 651 / 196 |
| reaver | B0 | 0.9595 | 0.6851 | 0.8689 | 38 | 29 / 0 |
| reaver | Axyz | 0.8582 | 0.1215 | undefined | 13 | 2 / 0 |
| octasynth | B0 | 0.8648 | 0.7803 | 0.9569 | 2059 | 4518 / 0 |
| octasynth | Axyz | 0.8756 | 0.7864 | 0.9461 | 3091 | 4836 / 0 |

OCTA still has false cycles and duplicates; A-xyz increases both counts despite stronger local geometry metrics. The experiments do not establish reliable complete topology recovery.

## Visualizations and artifacts

Figures show final-checkpoint overlays on two GT-positive crops from different validation images in each cohort. Selection uses the first/middle source-image groups and median GT node count within each group, with no prediction-score selection. Green is the supplied/derived GT path; blue is predicted nodes and K8 curves. All24 displayed checkpoint/crop cases (including scratch checks) match saved full-validation node/edge counts and F1 exactly. Layout was re-rendered from saved predictions, without another inference run.

- [octasynth overlay PNG](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-octasynth.png) / [PDF](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-octasynth.pdf).
- [reaver overlay PNG](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-reaver.png) / [PDF](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-reaver.pdf).
- [avwide overlay PNG](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-avwide.png) / [PDF](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/comparison-avwide.pdf).
- [Final F1 summary](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/final-f1-summary.png).
- [All12 final target results, CSV](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/final-results.csv); [JSON](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/final-results.json).
- [Prediction parity checks](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/prediction-parity.json), [selected samples](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/selection.json) and [raw selected predictions](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/visualizations_20261029/selected-predictions.json).
- [Complete15-run report](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/report.md); checkpoints and final/best metrics in `/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/runs`.
- Existing epoch learning curves: [AV-WIDE](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/curves-avwide.png), [REAVER](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/curves-reaver.png), [OCTA-Synth](/lustre/fswork/projects/rech/vnc/upz25mj/projects/DenseVesselGraph/experiments/native2d_mixed_20261028/curves-octasynth.png). Interim points use fixed subsets; final points use full validation.

## Interpretation limits

Mixed pretraining uses extra target-vessel exposures as well as plants (roughly1.29 million additional target crop exposures per cohort), so the improvement over scratch100 cannot isolate a biological-data effect. Each B0/A-xyz target pair has identical shared initialization and exposure hashes, verified again for this report. One training seed was used. AV-WIDE has6 validation images with unknown patient independence; REAVER has2 validation images in one calibrated diaphragm cohort; OCTA has40 held-out synthetic source images with unknown independence beyond supplied IDs. Pixel and micron scores are separate. Tests remain untouched. Fixed training budgets are finished; no extension or new sweep was launched.
