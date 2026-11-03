# Project review and IXI diagnostic starting point — 19 October 2026

This note connects `GrEVe.pdf`, the repository map, the historical experiment
reports, and the new [IXI diagnostic roadmap](IXI_graph_extraction_diagnostic_roadmap.md).
It records the interpretation used to implement the first diagnostic jobs.

## Paper and current research question

The supplied manuscript, *Structural Congruence Matters: Biological Pretraining
for Vascular Graph Extraction*, studies botanical versus road pretraining,
border-aware clipping, degree-2 retention, annotation efficiency, and multi-domain
generalization. The current repository extends the 3D part through adaptive
targets, continuous branch geometry, and physical topology evaluation.

The manuscript's experiments and this checkout's newer experiments have different
targets, dataset sizes, preprocessing histories, and evaluation implementations.
Their numeric results cannot be pooled into a single controlled comparison.
In particular, newer source-transfer controls do not establish a universal
advantage for botanical pretraining: MRI-only is strongest in the recorded
matched SyntheticMRI comparison, and IXI-only is strongest in the recent IXI
comparison. This narrows the claims that can be carried into a journal paper.

## Current implementation and data

- The active checkout is `Graph-Native-Betti-Matching-Representation`, branch
  `representation`, starting commit `8a4b6c4`. The sibling checkout is on
  `graph-extraction`. The September repository map is historical.
- A 3D SE-ResNet feeds a single-level deformable transformer; object queries
  predict node presence and coordinates, and an MLP scores pairs of object
  embeddings with shared relation embeddings. IXI uses 256 object queries.
- Training matching is Hungarian for the diagnostic lineage. It receives
  the final predictions, including local node refinement when enabled.
  Classification supervises all queries; coordinate regression and the standard
  relation loss depend on assigned queries. Baseline inference has no matcher
  or spatial suppression and considers every retained node pair.
- Undirected inference averages the two orientation **logits before softmax**.
  A cache/scorer that averages probabilities instead would change the experiment.
- Current IXI targets are in
  `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI/transfer_splines_k8_20261015`:
  10,000 training, 2,649 validation, and 2,178 test patches. This is a selected
  graph-containing population, not an exhaustive whole-volume deployment test.
- Coordinates are native array D/H/W, normalized by axis size. Physical distance
  uses each target cache's metric. Patch headers alone may not preserve source
  spacing. Exact crop faces extend half a voxel beyond voxel centres.
- Continuous targets preserve shared anatomical/crop anchors and full extracted
  guides; internal B-spline controls are geometry parameters, not object queries.
  Segmentation-derived guides are uncertain anatomical references.
- Brava exists at `/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/Brava`, with
  `brains`, `mask_hull`, and `swc_files`. This inventory does not establish
  image/SWC alignment, vessel-mask availability, or a validated train/val/test
  protocol. `mask_hull` must not be assumed to be a vessel segmentation.

## What the existing experiments support

| Area | Evidence and limitation |
|---|---|
| Graph extraction/targets | Ring handling, exact boundary clipping, adaptive simplification, Vedo repair and production checks improved the target pipeline. Mask containment and preserved Betti numbers do not certify anatomical connectivity. |
| Metrics | Native box-based edge F1 is not endpoint-correct adjacency. Branch F1 detects wrong connections that SMD and Betti counts can miss. Curvature and topology need separate measurements. |
| Coupled scratch spline training | Several runs collapsed or lost image sensitivity. Detaching geometry gradients allowed graph learning, but this does not identify a universal failure mechanism. |
| Residual geometry | Residual targets and local CNN samples improved geometry in controlled frozen-detector experiments. Same-endpoint chords isolate the benefit of curvature. Longer training did not consistently improve branches or held-out shape. |
| Source transfer | Plants/leaves/mixed sources and source rendering variants were tested. Recent matched tables favor the target-only controls; additional source diversity is not automatically beneficial. |
| IXI fitting | Four training patches can be memorized. Poor validation after that diagnostic shows that capacity to memorize is insufficient evidence of transfer. |
| IXI nodes | Substantial endpoint loss and a train/validation gap remain. Early CNN foreground readouts are much stronger than coarse 2³ readouts, but grid resolution confounds claims about information loss. |
| IXI local refinement | On 128 validation patches, node-head continuation versus local-image refinement gives native node F1 0.4857→0.5196 and Curve F1 0.1171→0.1676. Endpoint recovery improves; conditional relation recall barely changes. |
| IXI relations | The frozen head has useful conditional ranking (validation AP 0.4702), but low recall at the current cutoff. Refit and chord-local image probes did not establish improved generalization. |
| Decoder/loss interventions | Lower edge thresholds increase recall and false cycles. Earlier degree/unmatched/classification changes were confounded. Simple near-negative weighting and focal experiments do not justify a universal rejection or adoption of hard mining. |

Primary result trails: [scratch results](spline_learning/scratch-extension-300.md),
[local geometry](spline_learning/local-geometry-followup.md),
[stability](spline_learning/local-geometry-stability.md),
[augmented source tables](pretraining/augmented-transfer-results/test-comparison.md),
[IXI transfer](pretraining/ixi-transfer.md),
[metric definitions](metrics/vessel-core-v1.md), and
[IXI reports 1–5](metrics/README.md).
`ALL_EXPERIMENTS_AND_RESULTS.md` is a verbatim compilation; its old “pending”
statements are not current scheduler status. Canonical result tables take priority.

## First implementation decision

Implement P0-A/B/C/D before selecting a new learning intervention. Freeze the
delayed-spline parent and compare its node-head continuation and local-image
refinement on the same pre-existing train/validation IDs. Cache every query and
pair, distinguish M–M/M–U/U–U edges, remove unmatched queries only in a clearly
labelled oracle, and compare small physical-distance suppression controls.

The [execution report](metrics/6-ixi-query-attribution.md) records code, gates,
jobs, output locations and results. The next training candidate is the paired
raw-image versus binary-mask input oracle. The existing engine already accepts
`training.input: segmentation`; implementing another loader is unnecessary.
Its paired initialization, augmentation, update budget and validation protocol
still need to be fixed before launch. Loss redesign and multiscale attention
remain contingent on the attribution results.
