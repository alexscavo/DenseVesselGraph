# Branch fragmentation and relation distance diagnostic

Date: 2026-10-13. Follow-up to the [core evaluation](core-metrics-results.md).

## Question and agreed scope

The user observed that a vessel branch without intervening true junctions can
be split into small predicted edges, while nearby points on different vessel
portions are incorrectly connected. The desired edge remains a full branch
between consecutive junctions/terminations, including preserved crop-boundary
nodes. Passing through true junctions is not the objective.

An auxiliary dense prediction means extra training supervision on image voxels,
not extra decoded graph objects. It has not been selected as an intervention:
the user questioned its relevance to overprediction. First inspect existing
models for node-recovery errors, missed direct relations and nearby false edges.

## Diagnostic definitions

`scripts/diagnose_branch_relations.py` reads the same final straight and detached
B-spline predictions and the same 1,952 validation patches as the core suite.
There is no new model inference, training, score threshold or changed GT.

- Match raw nodes one-to-one by physical position, prioritizing match count,
  without conditioning on predicted degree. Use 1 mm and a 2 mm sensitivity check.
- An eligible GT edge has both endpoints recovered. Measure direct relation
  recall conditional on this event, alongside endpoint-pair recovery.
- Split GT edges into quartiles of physical arc length and endpoint distance.
  Cut points derive solely from common GT, before looking at model predictions.
- Among **all pairs of matched nodes**, measure true and false connection rates
  by GT endpoint-distance quartile. The false-positive denominator includes
  eligible nonadjacent pairs that the model correctly left unconnected. Thus a
  frequency of short predicted edges alone cannot establish a distance bias.
- For eligible edges with no direct connection, search for a predicted path.
  Also search for paths whose internal nodes are all degree two and do not map
  to other GT anchors. Such chains can explain unnecessary fragmentation.
- Check one deterministic shortest-hop route (prefer the degree-two chain if
  available) against the complete GT guide using bidirectional HD95 at 0.5 mm
  sampling, with tolerance matching the node threshold. This is a descriptive
  spatial check, not anatomical truth or an exhaustive search over all routes.
- Exclude closed/parallel GT branches from direct-branch recall because binary
  adjacency cannot distinguish their multiplicity. Expose excluded counts.
  The separate candidate-pair diagnostic still evaluates binary adjacency.

The audit distinguishes failure mechanisms at the saved decoder operating point;
it cannot inspect scores of rejected candidates or prove a causal architectural
defect. Matching is uncertain near close vessels. Conditional relation rates
describe the localized subset, so counts and localization coverage are essential.

## Results

The full paired run completed successfully: **1,952 patches**, **31,934 GT branches**, CPU job **1024937**, 66 seconds wall time (51.15 seconds inside the evaluator). No closed/parallel GT branches were excluded in this population.

### Conditional relation recall by GT branch arc length

Counts are correct direct edges / branches with both endpoints recovered. At **2 mm**:

| GT arc length | Straight | Detached B-spline |
|---|---:|---:|
| <4.97 mm | 92.9% (1058/1139) | 93.8% (1351/1440) |
| 4.97–10.03 mm | 79.9% (737/922) | 79.0% (833/1055) |
| 10.03–17.71 mm | 69.3% (616/889) | 65.4% (697/1065) |
| ≥17.71 mm | 38.8% (551/1421) | 42.8% (784/1830) |

At **1 mm**, shortest/longest recall remains **96.6% / 37.8%** for straight and **95.9% / 50.0%** for B-spline. Thus the pattern persists at the stricter threshold, on fewer eligible branches.

### Selection of false nearby relations

For B-spline at 2 mm, nonadjacent matched pairs closer than **4.90 mm** are connected in **123/171 cases (71.9%)**. For pairs at least **17.19 mm** apart, it is **479/80,277 (0.60%)**. Straight gives **80/116 (69.0%)** versus **414/72,234 (0.57%)**.

The same pattern appears at 1 mm: B-spline **19/28 (67.9%)** versus **53/13,788 (0.38%)**, straight **14/17 (82.4%)** versus **58/12,328 (0.47%)**. Close-negative counts are small at this stricter threshold. These are false-positive rates among nonadjacent candidate pairs, not fractions of selected edges that are false.

### Localization, fragmentation and object counts

| Measure, 2 mm | Straight | Detached B-spline |
|---|---:|---:|
| Raw node recall | 43.0% | 45.3% |
| GT branches with both endpoints found | 13.7% | 16.9% |
| Eligible branches missing direct edge | 1409 | 1725 |
| Of those, connected through another path | 499 | 594 |
| Of those, a chain with degree-two internal nodes exists | 216 | 244 |
| Of those, checked indirect path follows guide | 227 | 349 |

The last three rows each describe subsets of missing-direct-edge cases; they are not additive or all nested. For B-spline, **1,131/1,725** eligible missed direct branches have no connecting predicted path at all. The **244** degree-two-chain cases establish that unnecessary subdivision occurs, but contracting such chains alone cannot resolve most misses. At 1 mm only **1.75% / 2.40%** of all GT branches have both endpoints found (straight / B-spline).

Globally, the B-spline model predicts **42,388 nodes versus 41,646 GT nodes** (+1.8%) and **31,976 edges versus 31,934 GT edges** (+0.13%). Straight predicts **41,695 nodes and 31,702 edges**. Total counts are therefore close; false local objects/links coexist with misses elsewhere. Uniformly reducing object or edge counts would not directly solve the observed problem.

### Interpretation and next decision

The visual hypothesis is supported: both models strongly favor short relations, including nearby false ones, and miss many longer true relations even after conditioning on endpoint recovery. This establishes a decoded-output pattern, not its sole cause. Arc length and chord distance are strongly related in this SyntheticMRI set; the diagnostic does not disentangle length from image quality, branch context or localization difficulty. Most predicted edges still contain an unmatched endpoint at strict physical tolerances.

A minimal next learning experiment would hold detector/geometry weights fixed and compare relation-only continuations with identical initialization and steps: existing supervision versus deliberate emphasis on nearby nonedges, while retaining positive examples across branch lengths. This would test whether the relation scorer learned a distance shortcut without mixing in changing node positions. The exact sampling/weighting must be agreed first; no new training has been launched. If it fails, review existing curve-evidence experiments before changing features. Localization remains a parallel limitation.

For a relation-only experiment, detector batch-normalization statistics must also be frozen. Evaluate candidate precision/recall by distance and decoded graph topology on disjoint validation cases; improving training edge loss alone is insufficient. A blanket edge-length reward, forced spanning tree, or increased/decreased edge threshold is not justified by these data.


## Reproduction and evidence

Three targeted fixtures passed: short false links versus long true links,
continuous degree-two fragmentation versus a real break, and missing endpoints
excluded from conditional relation failures. Four real saved patches per model
passed before CPU job **1024937** evaluated the full paired set.

```bash
sbatch cluster/jean_zay/diagnose_branch_relations.slurm --output /path/to/fresh-output
```

Outputs: `summary.json` and per-branch CSVs for each model and tolerance under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012/branch_relations_v1/`.
The compact summary and CSV tables are kept in
`experiments/spline_learning/metadata/metrics/branch-relations-v1/`.

## Existing implementation relevant to interpretation

The current GT already removes ordinary intermediate degree-two nodes. Inference
considers every retained node pair; no length/radius restriction forbids long
relations. The standard relation MLP receives image-dependent endpoint and shared
relation tokens, without explicit sampled path evidence. Geometry receives local
image features in the detached spline arm, but its gradients do not improve the
graph detector.

An optional curve-image-evidence scorer already exists and was studied in an
earlier pilot: [previous implementation and protocol](../spline_learning/curve-informed-connectivity.md).
Any renewed experiment must build on that evidence rather than present path
sampling as a new, untested idea. Current node types are unsupervised; core
junction scores infer type from predicted degree and therefore combine placement
and connectivity errors.
