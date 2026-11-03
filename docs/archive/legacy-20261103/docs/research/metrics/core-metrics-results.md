# Vessel core v1: completed validation comparison

Date: 2026-10-13. [Protocol](vessel-core-v1.md),
[completed/pending tracker](implementation-status.md), [CSV](core-metrics-results.csv).

## Comparison and provenance

We rescored existing decoded predictions from the **final epoch-300 checkpoints**
on all **1,952 identical SyntheticMRI validation patches**. This is validation,
not held-out test evaluation. No training or model inference was performed;
checkpoints, preprocessing, split membership and legacy results were preserved.

| Output name | Model/checkpoint |
|---|---|
| `straight_final` | `graph_control_final`, graph-only scratch control, `graph_control/epoch300/model.pt`; straight chords |
| `bspline_detached_final` | `detached_spline_final`, scratch model with detached geometry branch, `detached_spline/epoch300/model.pt`; cubic B-splines with 8 total controls (6 internal plus shared endpoints) |

These are the latest detached-head scratch comparison, not the original failed
joint B-spline run or the four-patch pretrained overfit. Both were trained on
the same 10k training selection. Geometry gradients in the detached arm do not
update detector tokens, endpoints or shared local image features.

Paired checks confirmed identical ordered sample IDs, dataset root, protocol,
and evaluated GT geometry SHA256:
`4764dae84eea7481e47c561c888955dbe3cc9aa4e6d2c11e3f61d7792cfabf13`.
Both models have all 1,952 cases valid for topology, route scores and global
distance. References use the same uncertain full-path guides and graph topology
as the previous evaluation; they are not asserted to be perfect centerlines.

## Main results

| Metric | Straight | Detached B-spline |
|---|---:|---:|
| Curve F1, 0.5 mm | 0.0642 | 0.1533 |
| Curve F1, 1 mm | 0.2020 | 0.3426 |
| Curve F1, 2 mm | 0.4490 | 0.5539 |
| Average centerline distance (mm) | 4.092 | 3.581 |
| HD95 (mm) | 17.069 | 16.776 |
| TOPO-3D F1 (adapted) | 0.0910 | 0.1860 |
| APLS-3D (adapted) | 0.0390 | 0.1042 |
| Anchor F1 | 0.1288 | 0.1389 |
| Junction F1 | 0.0220 | 0.0294 |
| Endpoint F1 | 0.1791 | 0.1891 |
| Branch connectivity F1 | 0.0079 | 0.0118 |
| Betti-0 absolute error | 2.661 | 2.702 |
| Betti-1 absolute error | 2.268 | 2.133 |
| Native nodes | 21.36 | 21.72 |
| Native edges | 16.24 | 16.38 |
| Internal points | 0.00 | 98.29 |
| Payload bytes (float32/int32) | 386.25 | 1571.07 |

Curve, anchor, junction, endpoint and branch F1 pool supported counts across
patches. APLS/TOPO are means of defined per-patch scores. Average centerline
distance pools the two directed sample distances separately, then averages them;
HD95 is the mean per-patch maximum of the directed 95th percentiles. Betti errors
and representation counts are per-patch means. Higher is better for F1/APLS;
lower is better for distances and Betti errors. Representation size describes
cost, not accuracy.

## What these results establish

- The B-spline model improves spatial overlap and adapted local reachability /
  shortest-path agreement under this common protocol. However, TOPO/APLS and
  exact branch/junction recovery remain low for both models; the network still
  makes substantial graph errors. Large HD95 also exposes distant/missing parts.
- Route metrics depend on both spatial correspondence and connectivity. This
  comparison alone cannot establish that curved geometry improved adjacency:
  the two models have different decoded detections, and geometry gradients were
  detached. The preserved within-model diagnostic holds the spline model's
  endpoints/connectivity fixed: its chord curve F1 at 1 mm is **0.2263**, versus
  **0.3426** with its predicted curves. This establishes an overlap improvement
  attributable to curvature in those predictions, rather than better adjacency.
- Global average distance (4.092 vs 3.581 mm) measures the entire decoded graph.
  Earlier common-v2 matched-branch distances (about 0.660 vs 0.460 mm)
  condition on successfully matched branches and omit unmatched geometry. The
  larger global values reflect a different evaluation population.
- The spline payload is approximately **4.07 times** the straight payload at
  similar predicted node/edge counts. The six internal 3D controls per edge add
  coordinates. An efficiency advantage over dense centerline or adaptive-node
  representations has not been demonstrated by this two-model comparison.
- Curve F1 at 1 mm and branch F1 reproduce legacy common-v2
  scores. The new suite adds route metrics, corrected real-anchor accounting and
  an explicit native payload estimate; legacy definitions/results remain intact.

These are single-seed validation results with uncertain guides. TOPO/APLS here
are documented 3D adaptations, not interchangeable with 2D road benchmark
scores. Physical thresholds, neighborhood radius and deterministic sampling
budgets are fixed diagnostic choices, with calibration still pending. They were
not tuned to improve this comparison. See the tracker for deferred metrics,
held-out evaluation and centerline-tool experiments.

## Verification and execution

- Seven targeted known-graph tests passed, covering subdivisions, breaks,
  curved paths, cycles/isolates/duplicates, explicit centerlines, empty outputs
  and false bridges. Python and Slurm syntax checks passed.
- Four real saved cases per arm passed locally and in each CPU task before
  the full rescore. No GPU was required.
- CPU jobs **1019220_0/1** completed with exit 0: **2:27** straight and **9:59**
  B-spline including setup/smoke. Full evaluator timings were **140.79 s** and
  **589.63 s**. These are evaluation times, not model inference times.
- Initial array **1019081_0/1** failed before evaluation because extracting the
  GPU runtime exhausted node `/tmp`. The replacement uses the module base
  interpreter (Python 3.11.9 / NumPy 1.26.4 / SciPy 1.14.0); data and protocol
  were unchanged and failed logs retained.

## Files and reproduction

Full artifacts are under:

```text
/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012/core_metrics_v1/
  comparison.{csv,json,md}
  straight_final/{summary.json,provenance.json,per-patch.csv,per-patch.json}
  bspline_detached_final/{summary.json,provenance.json,per-patch.csv,per-patch.json}
```

The CSV beside this report is the compact paired table. All new aggregate
metrics, preserved legacy aggregates and provenance without long sample lists
are committed in
`experiments/spline_learning/metadata/metrics/core-v1-results.json`.
Large predictions and per-patch files remain outside Git.

To rescore the same saved predictions using the NumPy/SciPy base interpreter,
choose a fresh output directory (existing arm directories are protected):

```bash
python scripts/evaluate_vessel_core.py --arm straight_final --output-root /path/to/fresh-output
python scripts/evaluate_vessel_core.py --arm bspline_detached_final --output-root /path/to/fresh-output
```

The second completed arm creates the paired table after verifying inputs.
`cluster/jean_zay/evaluate_vessel_core.slurm` records the CPU environment and
four-case smoke gate used for the completed jobs.
