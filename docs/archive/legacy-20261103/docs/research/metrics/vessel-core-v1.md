# Vessel core v1: common physical evaluation

## Scope and references

Compare straight graphs, continuous curves and centerline graphs through one
physical geometry/connectivity interface. Score geometry, topology and native
representation size separately. Use uncertain centerline guides as geometric
references, not exact anatomical truth. Foreground mask scoring is a later step.

APLS follows the clipped relative shortest-path error, bidirectional comparison
and harmonic combination described by [SpaceNet](https://arxiv.org/html/1807.01232v3).
TOPO follows the reachable-neighborhood comparison described by
[Sat2Graph, section 4.4](https://arxiv.org/html/2007.09547).
Both implementations below are **explicit 3D adaptations**, not direct numerical
reproductions of the 2D road benchmarks. Source road tolerances, random seed choices,
minimum path lengths and optional angle constraints are not transferred unchanged.

## Input interface

`metrics/vessel_graph.py` provides `VesselGraph`:

```python
# Dense centerline graph, SWC-derived graph, or straight graph: physical units.
graph = VesselGraph.from_arrays(nodes_mm, edges)

# Explicit branch geometry and endpoint connectivity, including curved polylines.
graph = VesselGraph.from_arrays(nodes_mm, edges, edge_polylines_mm)

# Saved network prediction: cached metric maps normalized DHW to physical space.
graph = VesselGraph.from_prediction(record, metric, curved=True)

# Bare tool-exported polylines: geometry metrics only; topology is unavailable.
graph = VesselGraph.from_centerlines(polylines_mm)

# Shared IDs give explicit connectivity for each centerline vertex.
graph = VesselGraph.from_centerlines(polylines_mm, point_ids=vertex_id_sequences)
```

An array graph uses one polyline per edge, with endpoints agreeing with its node
coordinates. Bare polylines never invent junctions at crossings or nearby points.
Shared centerline point IDs must have consistent coordinates; repeated mesh segments
in root-to-tip paths become one underlying segment. Distinct intermediate IDs retain
parallel paths. Graph array inputs preserve duplicate edges and self-loops explicitly;
their excess topology is penalized rather than silently removed.

Raster skeletonization, VTK/SWC file readers and running external extraction models
are **not implemented yet**. Their outputs can already be evaluated once exported
through the array/polyline interface. A binary mask needs a declared conversion to
connectivity before receiving graph topology scores.

## Common conventions

- Coordinates and lengths are physical millimetres in 2D or 3D. Saved 3D predictions
  use the existing cache matrix, exactly as the previous common evaluator did.
- Degree-two chains contract for evaluation while retaining their geometry.
  Densified evaluation points are not counted as native model objects.
- Junctions are degree >=3, endpoints degree 1, isolates degree 0. Cropped boundary
  endpoints follow the existing GT extraction; they are not joined outside a patch.
- Pure cycles have virtual matching anchors, excluded consistently from **real
  anchor** counts. Loop branches and parallel branches remain topology objects.
- Geometry-only inputs report topology scores as unavailable, never perfect.
- All parameters below are fixed for both models before full evaluation. They
  are initial diagnostic choices, not a medical tolerance calibration or test-set tuning.

| Parameter | Value | Purpose |
|---|---:|---|
| Global geometry sampling spacing | 0.25 mm | Same arc-length convention as previous common evaluation |
| Curve support tolerances | 0.5, 1, 2 mm | Show sensitivity to uncertain guide positions |
| Anchor/junction matching tolerance | 1 mm | Preserve the previous spatial scale for comparison |
| Route graph sampling spacing | 0.5 mm | Bound spatial route/snapping approximation |
| Route/seed snap tolerance | 1 mm | Reject distant correspondences |
| TOPO geodesic neighborhood radius | 10 mm | Local connectivity scale in a cropped vessel patch |
| Maximum TOPO seeds per direction | 32 | Bounded deterministic evaluation cost |
| Maximum APLS controls per direction | 64 | Bounded shortest-path comparison cost |

`Protocol` in `metrics/vessel_suite.py` exposes these settings. Parameter or matching
changes require a separately labelled protocol/configuration and new output root.

## Metrics and aggregation

### Global geometry

Global sampled points are compared to **continuous polyline segments** of all
reference/predicted branches; false and missing geometry therefore participate.
Cubic B-splines are evaluated at 257 parameter locations, then sampled by physical
arc length. This is a finite curve approximation, not an analytic spline distance.

- **Curve precision/recall/F1** at each tolerance: supported predicted/reference
  sample counts, pooled across cases. Counts are explicit in per-patch outputs.
- **Average centerline distance, mm:** half the two mean directed distances.
  Dataset summary pools each direction's distance sums/sample counts. This is the
  chosen unsquared Chamfer/ASSD-style **centerline** measure, not a surface ASSD or
  squared-distance Chamfer. Do not add equivalent names as independent evidence.
- **HD95, mm:** maximum of the two directed 95th percentiles for each patch;
  dataset headline is the mean per-patch HD95, not a global pooled percentile.
- **Length:** predicted/reference total physical length and absolute error.

If only one graph has vessel geometry, curve F1 is zero and its distance is
undefined (`null`). Distance summaries expose valid-case counts and missing-geometry
fractions; do not mistake their conditional averages for complete-case performance.
Both-empty geometry has perfect overlap by convention. Isolated nodes are assessed
by topology/detection counts, not by pretending they are vessel curves.

### Connectivity and node roles

- **Real anchor, junction and endpoint P/R/F1:** role-aware one-to-one spatial
  matching. The cost prioritizes maximum matching cardinality, then distance.
  Isolates participate in real-anchor matching. Cycle centroids do not contribute
  to real-anchor TP/FP/FN.
- **Branch connectivity P/R/F1:** both matched endpoint identities must connect
  in GT. Multiplicity matters. Geometry does not choose the most flattering
  correspondence. Dataset counts pool TP/FP/FN across patches.
- **Betti-0/1 errors:** mean absolute component/cycle count difference per patch,
  with prediction/reference counts retained. These are not sufficient to establish
  correct adjacency when counts happen to agree.
- **Matched tortuosity error:** open matched branches only. Closed-branch
  arc/chord ratios are undefined; count their exclusions instead of emitting NaN.
  Global geometry and branch FP/FN remain visible beside this conditional metric.

This new implementation corrects ignored isolate flags and inconsistent virtual
cycle counts without changing legacy code/results. The legacy evaluator's native
node/edge AP/AR/F1 remains available as a separate detection measure; its edge
boxes and SMD still describe straight endpoint chords in input coordinates.

### APLS-3D adaptation

1. Create a graph of contracted anchors and arc-length samples, preserving explicit
   connectivity. Segment weights partition the underlying polyline **arc length**,
   not endpoint chord length. Parallel routes use their minimum length for shortest
   paths; branch counts and Betti metrics capture extra parallel edges separately.
2. Select at most 64 controls: physical lexicographic anchor priority followed by
   evenly selected remaining route vertices. If anchors exceed the cap, this first
   diagnostic version uses the first anchors in that order; control-count/spatial
   sensitivity should be examined before a publication claim.
3. Project controls to the closest target sampled segment within 1 mm and split
   that segment. Splitting preserves route length and does not merge nearby vessels.
4. For each connected, positive-length source control pair, score
   `min(1, abs(predicted_length - reference_length) / reference_length)`.
   Missing correspondences/routes receive error 1. Zero-length routes and pairs
   disconnected in the source are excluded from that direction; their counts are
   exposed. The reverse direction penalizes extra predicted routes.
5. Each direction scores `1 - mean_error`; combine by harmonic mean. A direction
   with no meaningful source pairs is unavailable. If only one direction has pairs,
   the combined score is zero; if neither has pairs, APLS is unavailable.

Dataset APLS is the mean defined per-case score, with valid-case count. Spatial
snapping and actual route lengths make this a connectivity-and-geometry measure.

### Reachable-neighborhood TOPO-3D adaptation

Select up to 32 seeds in each direction using the same deterministic anchor/route
selection rule. Match a seed to the nearest target sampled node within 1 mm.
Collect all nodes reachable within **10 mm of graph distance**, and maximize a
one-to-one spatial matching between the two neighborhoods at 1 mm. There is no
2D road direction/angle gate. Source prediction neighborhoods determine precision;
source GT neighborhoods determine recall. An unmatched seed contributes its
neighborhood samples as misses. Per-patch counts pool across seeds; dataset headline
is mean per-patch F1. This deliberately differs from random 2D road-seed protocols.

This tests local reachable structure but still depends on spatial agreement. Report
it alongside APLS, branch/junction matching, Betti and geometric scores. Radius,
sampling and seed-budget sensitivity remain a later calibration task.

### Native representation cost

Count original nodes, edges, degree-two nodes, internal controls/polyline points,
coordinate/index scalars and a normalized payload in bytes. Use float32 coordinates
and int32 endpoint indices, shared node endpoints, and offsets for ragged polylines.
A fixed-eight-control B-spline has six additional internal 3D points per edge;
its shared basis/knot convention is not repeated per edge. Bare polylines include
their point arrays and offsets. Evaluation samples, model weights, confidence/box
fields, radii and JSON formatting are excluded. This is a stated encoding estimate,
not actual file size or neural network parameter count.

Fewer objects/bytes are useful only at comparable geometry/topology. Straight and
8-control spline heads predict the same kinds of junction/termination nodes, so
this comparison cannot yet establish an advantage over dense adaptive centerlines.

## Evaluation and artifacts

`scripts/evaluate_vessel_core.py` streams the existing saved predictions and uses
the same ordered 1,952 validation IDs and cached GT paths. It computes evaluated
GT/prediction geometry hashes, records Python/NumPy/SciPy versions, and checks the
GT hash, sample order and protocol agree between models before building comparison
tables. Original checkpoints, splits, legacy predictions and summaries are retained.

Commands, after the existing project environment is available:

```bash
python scripts/evaluate_vessel_core.py --arm straight_final --max-samples 4
python scripts/evaluate_vessel_core.py --arm bspline_detached_final --max-samples 4
python scripts/evaluate_vessel_core.py --arm straight_final
python scripts/evaluate_vessel_core.py --arm bspline_detached_final
```

Use `--output-root` for repetitions: the runner rejects existing arm directories.
The CPU Slurm script runs the small smoke first and uses the module's NumPy/SciPy
base interpreter, without extracting the GPU training runtime into node `/tmp`.

Run root: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012/`

New outputs under `core_metrics_v1/`:

- `straight_final/` and `bspline_detached_final/`: provenance, per-patch JSON/CSV,
  summary JSON with new metrics and explicitly retained legacy summaries.
- `comparison.csv`, `comparison.md`, `comparison.json`: paired core headlines.

Validation remains a development comparison, not a held-out test or seed-robustness
claim. No new training, threshold optimization or model inference is performed.

The completed 2026-10-13 evaluation is recorded in
[the paired result report](core-metrics-results.md) and
[the implementation/pending-work tracker](implementation-status.md).
