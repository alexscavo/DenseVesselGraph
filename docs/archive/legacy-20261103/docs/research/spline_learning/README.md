# Continuous vessel graph experiments

> **Research consolidation, 23 October 2026:** [Consolidated experiment history and current batch](../../../research/experiments.md). This document retains its detailed and dated historical context.

## Current status (2026-10-16)

The scratch 300-epoch and source-transfer comparisons are complete. Start with
[the final scratch results](scratch-extension-300.md), [source-transfer results](../pretraining/combined-transfer-results/test-comparison.md), and [the current experiment map](../../../../../../experiments/README.md).
The current exploratory work is [nonplanar plant embedding](../pretraining/nonplanar-source-prototype.md) and [textured backgrounds](../pretraining/textured-source-prototype.md); these are visual prototypes, not validated training improvements.

Earlier feasibility, collapse and unfreezing experiments below are retained evidence.
[Storage/retention report](../storage/cleanup-20261016.md).

For the real-image follow-up, see [the IXI patch experiment](ixi_patch.md), with MRI context, uncertain centerline guides, mask support, and a portable 3D viewer.

For the matching SyntheticMRI viewer, starting with the **10-control B-spline**, see [the SyntheticMRI patch experiment](synthetic_mri_patch.md).

For the learning implementation, decisions, target preparation and experiment status, see [the spline learning report](spline-learning-report.md).

For the main-method selection, see [V0/V1 verification and decision](spline-approach-selection.md). For the expanded experiment, boundary handling, W&B projections and launch status, see [the 10k training protocol](spline-10k-training.md).

Open [2d.html](../../../../../../../graph-representation/experiments/2d.html) or [3d.html](../../../../../../../graph-representation/experiments/3d.html) directly in a browser. Both are offline pages with no build or external dependency. In 3D, drag to rotate and scroll to zoom.

## Question

Can a graph with only junction and termination nodes preserve tortuous vessel geometry by storing a continuous curve on each edge? This is a **representation oracle** experiment: topology, endpoints, and dense reference centerlines are supplied. It does not test whether a model can predict them from an image.

The synthetic graph is a small bifurcating tree with two junctions, four terminations, and five branches. Each reference branch interpolates seven sparse waypoints with a smooth Catmull-Rom curve. The waypoints add broad arcs and an S-bend to the right branch; the tortuosity slider scales their transverse and depth offsets. The 2D page shows soft tubular vessel strokes; the 3D page shows a projection of simple spatial centerlines. Neither is a realistic medical image or a voxel simulation.

## Curve families

| Family | Free geometry values per edge in dimension `d` | Main limitation |
| --- | ---: | --- |
| Single cubic Bézier | `2d` | Cannot express several independent bends |
| Clamped cubic B-spline with `n` controls | `(n - 2)d` | Needs more controls as detail increases |
| Piecewise cubic Bézier with `s` segments | `(3s - 1)d` | This baseline is only position continuous at segment joins |

All curves fix their endpoints to the corresponding graph nodes. Pink internal controls and piecewise segment joins are edge parameters, not graph vertices. Drag a pink point to edit the selected edge; in 3D, Shift-drag moves it in depth. **Refit selected edge** restores its least-squares fit. Fits minimize squared distances between reference samples and curve samples at the same parameter value. The piecewise baseline obtains its internal joins from reference samples, while counting them in its parameter cost. This experiment uses known topology and centerlines, so it says nothing yet about learnability from an image.

The displayed mean symmetric curve distance averages each sampled point's closest distance to the other sampled polyline in both directions. It is shown in reference pixels at the initial viewport scale; the 3D calculation uses the original 3D coordinates and does not depend on camera rotation or zoom. Tortuosity is centerline length divided by endpoint chord length. The object-token count remains six for every setting.

## Suggested first checks

1. At low tortuosity, compare all families with their smallest parameter count.
2. Raise tortuosity on the right tortuous branch. Check whether one cubic misses the S-bend and whether additional spline controls recover it.
3. Compare error against geometry values per edge; look for a useful elbow rather than demanding zero error.
4. In 3D, rotate the same fit to inspect depth error that is hidden in one projection.

The source rationale is in [vascular_graph_representation.md](../vascular_graph_representation.md), especially the function-valued edge section. The [research roadmap](../vascular_graph_roadmap.md) places this as exploratory Phase 13 after the adaptive-node work. Those source documents remain in their project's canonical research directory; documentation specific to this experiment lives here.

## Current work (2026-10-06)

- [Current spline training protocol, fixes and job status](spline-10k-training.md).
- [Implementation decisions and earlier measured results](spline-learning-report.md).
- [Verified dataset restoration](synthetic-mri-restoration.md).
- [Project cleanup and artifact locations](projects-cleanup-2026-10-06.md).

The source-health inventory in `archive/` describes the old, disappearing source; use the restored dataset's `restore-verification.json` for current availability. Earlier 2D/3D and IXI examples remain reference experiments for the representation design.

[Actual held-out test predictions: two interactive 3D viewers](synthetic-mri-test-predictions.md).

- [Catmull–Rom versus B-spline: existing single cases](catmull-rom-single-cases.md) — offline viewers, equal-budget interpolation points, construction and metrics.

- [Three scratch trainings: completed comparison tables](scratch-comparison-results.md) — final and validation-selected checkpoints, common metrics, native diagnostics, counts and limitations.

## Latest completed scratch comparison

[300-epoch graph-only versus detached B-spline results](scratch-extension-300.md#final-results--2026-10-13): all trainings/evaluations completed; geometry overlap improves without encoder collapse, but shape overfits and complete-branch recovery remains weak.
