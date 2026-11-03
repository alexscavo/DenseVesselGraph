# Metric implementation and evaluation tracker

Updated: 2026-10-14. User accepted a core suite first, straight versus B-spline now,
and compatibility with centerline extraction tools/models for future comparisons.

| Item | Status | Evidence / remaining work |
|---|---|---|
| Literature shortlist | Recorded | Original `metrics_ideas.md`; verified SpaceNet/Sat2Graph definitions in protocol |
| Preserve prior evaluations | Done | New version/files; legacy code and outputs unchanged |
| Physical straight/B-spline adapter | Implemented and checked | Seven known-graph tests; four real saved cases per model passed |
| Centerline graph/polyline interface | Implemented and checked | Explicit IDs preserve connections; bare polylines get geometry only |
| Degree-two contraction, isolates, loops, multiplicity | Implemented and checked | Known subdivisions, isolates, self-loop and duplicate-edge cases |
| Curve F1 at 0.5/1/2 mm | Implemented | Global geometry, with empty-case conventions |
| Symmetric average centerline distance | Implemented | One stated Euclidean measure; no redundant Chamfer/ASSD claims |
| Global HD95 | Implemented | Mean per-case HD95 in dataset summary |
| Junction/endpoint/anchor and branch P/R/F1 | Implemented | Corrected role/count rules in the new evaluator |
| APLS-3D | Implemented and checked | Arc weights, segment snapping, missing-route penalties, both directions |
| TOPO-3D adapted reachable neighborhoods | Implemented and checked | Finite geodesic radius, spatial one-to-one matching; no angle gate |
| Betti and branch/length/object counts | Implemented | Same physical graph abstraction for all supported inputs |
| Representation scalar/byte estimate | Implemented | Native coordinates/connectivity/controls, no evaluator samples |
| Closed-branch tortuosity | Handled | Exclude undefined ratios and expose count; open matched diagnostic retained |
| Full 1,952-case paired rescore | Completed | CPU array **1019220_0/1**, both exit 0; identical ordered IDs/protocol/GT geometry hash |
| Full-validation result review and tables | Completed | [Report](core-metrics-results.md), [CSV](core-metrics-results.csv); all 1,952 distance/topology cases valid per model |
| Branch length / nearby false-link diagnostic | Completed | [Report](branch-connectivity-diagnostic.md); CPU **1024937**, 1,952 paired patches; 3 fixtures and 4-patch smoke passed |
| Controlled relation-only / node-head intervention | Completed and reviewed | [Four-arm results](relation-distance-pilot.md); **1026770_0–3**, exit 0; all 1,952 validation cases; modest near-link effect, no overall improvement, large train/validation gap |
| Curve-aware SMD | Deferred | Existing native SMD is retained as a labelled chord/input-space legacy score |
| Foreground-mask support / clDice | Deferred | Needs a declared curve-to-mask/radius convention |
| Curvature derivatives / surface ASSD | Deferred | Noisy guide references and surface/radius requirements |
| Graph edit distance / CRR | Deferred | Later than the current core suite |
| Inference time / peak memory | Deferred | Saved predictions cannot establish these; needs timed model inference |
| Adaptive degree-two baseline | Deferred by user | Later comparison with matched evaluation membership |
| External centerline tools/models | Interface ready; experiment pending | Need actual tool outputs, format adapters and fair preprocessing |
| VTK/SWC/raster skeleton file import | Pending | Generic arrays/polylines already accepted; no silent junction inference |
| Tolerance/radius/sampling-budget calibration | Pending | Fixed diagnostic defaults now; review stability before paper claims |
| Held-out test, second seed, uncertainty intervals | Pending research phase | Not implied by this validation rescore |

## Verification and job history

- Python/Slurm syntax passed. NumPy/SciPy/pytest available through the module base
  interpreter. Seven meaningful fixture checks passed: identity/subdivision,
  small break, curved length, isolate/cycle/duplicate, explicit centerlines,
  empty predictions, and false bridge. These test metric behavior rather than
  model convergence.
- Four actual saved cases per model passed locally. Their outputs are smoke
  evidence only, not full-validation results. No model/checkpoint was loaded.
- Initial CPU array **1019081_0/1** failed before evaluation while extracting the
  training-runtime archive: node `/tmp` was full. Preserved those logs. Small fix:
  use the protected module base interpreter with only NumPy/SciPy, as in passing
  local checks; no package upgrades or deletion of other temporary files.
- Resubmitted **1019220_0/1** with the same metric protocol/data/output intent;
  four-case CPU smoke gates passed before full evaluation. Both completed:
  straight task 0 in **2 min 27 s**, B-spline task 1 in **9 min 59 s** (Slurm
  wall time including setup/smoke). NumPy 1.26.4, SciPy 1.14.0, Python 3.11.9.
  Logs:
  `experiments/spline_learning/logs/vessel-core-1019220_<task>.{out,err}`.
- Paired collector verified ordered IDs, dataset root, protocol and reference
  geometry SHA256. Full geometry hashes and aggregates are recorded in
  `experiments/spline_learning/metadata/metrics/core-v1-results.json`.
  Curve F1 at 1 mm and branch F1 agree with preserved legacy
  common-v2 results. New route metrics add evidence without changing old scores.

## Decisions to retain

1. Geometry/topology/efficiency remain distinct: a good overlap score does not
   establish connectivity, and Betti counts alone do not identify wrong adjacency.
2. New metrics are versioned and legacy scores remain reproducible. APLS/TOPO
   adaptations are named/documented rather than asserted equivalent to road scores.
3. GT topology and uncertain guides are shared between representations. Evaluation
   may snap/project correspondences; it never alters predictions or trains on GT.
4. Centerline geometry can be scored without invented connectivity. Explicit
   point identities or a graph are required for graph topology.
5. Representation bytes refer to a normalized encoding, not compressed files or
   model weights. Controls/evaluation samples must not be confused with object tokens.
6. Keep failed attempts and unfinished tasks visible; do not report queued jobs
   or four-case smokes as completed 1,952-case results.
