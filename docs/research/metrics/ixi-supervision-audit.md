# IXI supervision and prediction audit — 2026-10-18

## Scope and conclusion

Completed a validation-only, read-only audit of the IXI-only **delayed B-spline** model (`ixi_delayed_spline`, saved validation step 028170). This is the straight pretraining → spline fine-tuning arm, not spline supervision throughout pretraining. No training, decoding thresholds, targets, or benchmark metrics changed.

The selected references are spatially consistent with the segmentation. This **does not certify anatomical topology or completeness**: graphs and masks can share extraction errors. Predictions show substantial foreground preference over random points, but poor localization and connectivity. Both node placement and relation prediction warrant attention.

## Protocol

- 16 validation patches, one per filename-derived subject; deterministic SHA256 ranking independent of predictions/results. Selection is reproducible in `summary.json`.
- Raw volumes and masks loaded in native array axes. Coordinates follow loader convention `index / shape`; physical metric comes from spline cache, not potentially identity patch headers.
- Valid extent is voxel faces `[-0.5, size-0.5]`, including boundary nodes. Within that extent, sample border intensity/mask by nearest extension. Truly outside points cannot count as supported.
- Curves resampled at intervals ≤0.25 mm. Mask occupancy uses nearest voxel; proximity uses exact distance to foreground voxel centres, not a smooth surface distance.
- 4,096 seeded uniform points per patch in the voxel-centre bounding box serve as a simple random spatial control. They are not a learned-model baseline.
- Numbers below are macro averages over patches; empty predicted curves are excluded from curve-support averages and recorded with zero count/null support in CSV.
- Foreground completeness flag: a mask voxel is explained if within the radius at its nearest GT guide point plus one largest voxel spacing. Radius is distance to background. This is a permissive thickness-aware **heuristic**, not a recall metric; it can miss annotation defects and flag legitimate geometry.
- Full-validation endpoint/relation counts reuse `relation-diagnostic.json` alongside the saved predictions. Positional maximum-cardinality matching, no degree matching; 1 and 2 mm sensitivity. This differs from the main benchmark's matching, so counts must not be substituted for benchmark TP/F1.

## Foreground support

| Points | Inside mask | Within 1 mm of foreground voxel centres |
|---|---:|---:|
| GT nodes |100.0%|100.0%|
| GT full-path curves |99.52%|100.0%|
| Predicted nodes |34.25%|65.57%|
| Predicted spline curves |28.73%|62.04%|
| Random spatial points |0.94%|3.06%|

Mean thickness-aware unexplained foreground: **2.61%**. This is encouraging for spatial supervision, not proof all true vessels have labels. GT checks use original full paths, not fitted GT spline controls; target-fit error/topological correctness remain separate questions. Model curves are evaluated where present, so good support would not imply complete reconstruction.

## Node versus relation accounting

Counts are over all 51,937 supported GT edges across 2,649 validation patches. No closed/parallel targets were excluded in this saved diagnostic.

| Matching tolerance | Missing ≥1 endpoint | Both endpoints recovered, no direct relation | Correct direct relation |
|---|---:|---:|---:|
|1 mm|48,096 (92.60%)|3,251 (6.26%)|590 (1.14%)|
|2 mm|36,033 (69.38%)|13,738 (26.45%)|2,166 (4.17%)|

At 2 mm, node recall is **50.82%**. Conditional direct relation recall is **13.62%** among the 15,904 recoverable endpoint pairs. For the longest arc-length quartile it is **6.57%**, versus **14.45%** for the shortest. Long relations are harder here, but node errors already prevent most GT edges being recovered. Conditional rates also depend on which nodes are recovered; they do not isolate a causal relation-head defect. Full length bins and nearby-negative pair selection rates are preserved in `full-validation-relations.json`.

## Visual review and limitations

Generated 16 orthogonal slice panels and a standalone slice viewer with independent mask/GT/prediction toggles. Each PNG selects the greatest mask-area slice per axis; it is not a complete volume review. Inspected panels for `sample_000001_0133`, `sample_000012_0032`, `sample_000163_0050`, and `sample_000166_0057` directly. GT generally follows bright vessel structures in these slices; predictions are sparse and sometimes displaced. The 000163 example was also selected for review because the preliminary completeness heuristic flagged it. Slice fragments must not be interpreted as disconnected 3D branches.

Still needed before declaring GT “good enough” anatomically: review suspected omissions/connections through adjacent slices, check difficult small vessels/boundary terminations, and obtain independent annotation review for ambiguous cases. Agreement between a mask and a graph extracted from that mask is not independent validation. No threshold here automatically approves or rejects a dataset.

The foreground check measures graph output support, not a segmentation head or encoder probe. It establishes that output locations are not spatially random, but cannot identify where inside the network foreground information is lost.

## Next decision

Review the viewer's ambiguous cases first. If no material annotation errors emerge, prioritize foreground-supported node localization and conditional connectivity. A small frozen-feature foreground probe can distinguish whether the image features encode foreground from whether graph decoding uses it; an oracle-endpoint relation experiment can further isolate connectivity. These are **proposals, not launched experiments**. Do not introduce new losses or regenerate all GT based on this small audit alone.

## Files and reproduction

Artifacts: `experiments/ixi_supervision_audit/20261018/`:

- `summary.json`, `per-patch.csv`, `full-validation-relations.json`: compact tracked evidence.
- `viewer.html`: download this file alone and open locally; no server or internet required.
- `sample_*.png`: static raw/overlay panels. HTML/PNGs stay local and are reproducible, avoiding binary growth in git.

```bash
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/audit_ixi_supervision.py \
  --dataset /lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015 \
  --prediction-dir /lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_transfer_20261015/runs/ixi_delayed_spline/best-full-validation/validation-028170 \
  --output experiments/ixi_supervision_audit/20261018 --samples 16
```

Checks: syntax/imports; small physical-coordinate/anisotropic-axis check; voxel-face boundary support, outside exclusion and empty predictions; successful 16-patch run; partition counts add exactly. Initial audit-only bounds and orthogonality checks were corrected for actual voxel-face clipping and affine float precision before recording results. No dataset/model changes. Interactive browser execution was not tested on the cluster; static rendered images were inspected.


## Image inspection: method and observed examples

The raw MRI is displayed in its native voxel axes, with grayscale display limits at the patch's 1st/99.5th intensity percentiles. Static panels show the slice with greatest mask area along each axis; the HTML viewer allows adjacent-slice navigation and independent overlay toggles. Cyan contours mark mask foreground; green sampled paths/crosses show GT curves/nodes; magenta marks predictions. Only points within 0.75 voxel of a displayed slice are overlaid.

Inspected static examples: `sample_000012_0032` shows GT paths running inside bright branching structures and predicted points sometimes displaced from them; `sample_000001_0133` shows GT support on multiple bright vessel sections with predictions concentrated on a smaller portion; `sample_000166_0057` shows strong background texture and sparse predictions despite visible reference vessels. `sample_000163_0050` was additionally reviewed after a completeness flag. These views motivate review of possible omissions but do not resolve all anatomy through a complete slice sequence.

Quantitative checks use nearest-voxel mask occupancy, physical nearest-foreground distance, raw intensity sampled at graph points and thickness-aware mask coverage. They complement the displayed image evidence; they are not pixelwise model segmentation predictions.
