# Controlled relation and node-head learning pilot

Date: 2026-10-13. User authorized isolating relation learning and trying node
learning in parallel after the [branch diagnostic](branch-connectivity-diagnostic.md).

## Research question

The final models recover short true branches much more often than long ones,
even when both endpoints are localized. They also frequently connect nearby GT
nonadjacent nodes. This pilot asks whether changing the emphasis of the existing
relation loss improves this pattern, and whether adapting node heads changes
the outcome. It does not assume the relation head is the only failure: most GT
branches still lack two accurately localized endpoints.

## Four controlled arms

| Array task | Arm | Trainable parameters | Nearby-negative weighting |
|---|---|---|---|
| 0 | `relation_control` | Existing relation MLP | Original weights |
| 1 | `relation_near` | Existing relation MLP | 5x relative weight below 5 mm |
| 2 | `node_control` | Relation MLP, node classification and coordinate heads | Original weights |
| 3 | `node_near` | Relation MLP, node classification and coordinate heads | 5x relative weight below 5 mm |

All four load the same **final epoch-300 detached B-spline checkpoint**:
`scratch_extend300_20261012/detached_spline/epoch300/model.pt` under the established
scratch run root. They are continuations of that successful scratch-trained
model; they are not fresh scratch runs or continuations of the old collapsed arm.

“Node learning” in this first bounded pilot means adapting `class_embed` and
`coord_embed` on fixed image/object features. CNN, transformer, projection,
query embeddings, spline head and all buffers remain frozen. No node-type head
or auxiliary dense prediction is introduced. This scope is deliberately narrower
than unfreezing shared representations; failure here would not exclude learning
better node representations with a later intervention.

Every module stays in evaluation mode during updates, keeping BN running
statistics fixed. Gradients still update the selected heads. Relation-only arms
must preserve exact node outputs and object tokens on a fixed real input.
Node arms must preserve tokens while changing node coordinates. Node changes can
change Hungarian assignments; that is an intended consequence recorded by this
factor, not a matcher modification. Frozen spline parameters can produce different
curves when the supplied endpoints or selected edges change.

## Objective and schedule

- **1,000 optimizer updates**, batch size 4: 4,000 patch presentations sampled
  from the same existing 10k training membership. Same seed/order in all arms.
  No augmentation, split changes, new data or held-out test use.
- Fresh AdamW moments, constant LR **7e-5 for the relation head**, **7e-6 for
  node heads**. Weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8.
- Retain existing graph-loss weights: class 4, coordinates 2, boxes 3,
  cardinality 0.8, edges 6. Geometry weight is **0**; geometry parameters are
  frozen and its training computation is skipped. Graph losses that involve
  frozen heads have no update effect in relation-only arms.
- Existing focal edge loss, gamma 2, class weights [1,1]. All matched candidate
  pairs and every positive edge remain supervised; there is no length cap or
  removal of long positives. Matching and inference thresholds are unchanged.
- In weighted arms, nonedges with **GT endpoint distance <5 mm** get raw weight
  5 versus 1 for other nonedges. Renormalize negative weights **within each
  graph** so their total equals the original negative count. Positive weights
  remain 1. Thus this reallocates negative supervision without adding a loss or
  deliberately increasing total negative/positive weight balance.
- GT coordinates define training weights after the existing Hungarian query/GT
  correspondence. They never enter inference. Close predictions are not simply
  forbidden. All candidates remain available, including nearby true edges.
- 5 mm is a rounded operational choice informed by the diagnostic's 4.90 mm
  close-pair boundary; factor 5 is one bounded intervention, not a tuned sweep.
  These are exploratory validation-informed decisions, not held-out test claims.

This preserves the original supervision on Hungarian-matched queries, including
queries whose current coordinates are poor. Unmatched-query negatives and new
image-path features are not added in this experiment. If the intervention fails,
those remain separate hypotheses; existing curve-evidence work must be reviewed
before repeating it.

## Evaluation and interpretation

Reuse the established **512 train / 128 validation monitoring IDs**. Evaluate at
updates 0, 500 and 1,000; the final validation expands to the identical **1,952
validation patches**. Train monitoring is at 0 and 1,000. Consequently the final
full-validation aggregate must be compared to the saved full baseline, not
directly to the initial 128-case aggregate. Per-case predictions permit a paired
128-case comparison too.

Track existing native/common geometry and detection metrics, same-endpoint
chords, node recovery, conditional edge recall by GT arc/chord length, false
connection rates by candidate distance, and indirect paths/degree-two chains.
Final full validation also computes the new core suite, including TOPO-3D,
APLS-3D, branch/junction F1 and Betti errors. Diagnostic cut points stay fixed
from the previous full GT audit, shared by every arm and train/validation report.

W&B runs use offline group `relation-distance-pilot-20261013`, with losses,
native/common metrics, relation diagnostics, final core metrics and two fixed
validation projections. Large outputs and checkpoints remain on scratch.

Interpret comparisons in this order:

1. `relation_near` versus `relation_control`: with unchanged nodes, does nearby
   false-link rate fall while retaining long-edge recall and improving decoded
   topology? Suppressing all connections is not success.
2. `node_near` versus `node_control`: does the same change help with adaptive
   node heads? Compare node precision/recall and endpoint-pair coverage too.
3. Node arms versus their relation-only counterparts: does endpoint recovery
   improve, and do graph metrics benefit? Conditional relation recall alone can
   change because a different subset of nodes/edges becomes eligible.

Lower training loss does not establish better graphs. A negative 1,000-update
result would bound this particular intervention, not prove the spline approach
or long-relation learning infeasible. Decide extensions after inspecting results.

## Implementation, checks and submission

- Config: `configs/relation_distance_pilot.yaml`.
- Criterion: `training/losses/relation_distance.py`; explicit pilot subclass,
  leaving the default criterion unchanged.
- Runner: `scripts/train_relation_distance_pilot.py`, reusing loaders, matcher,
  geometry evaluation, tracking and the new diagnostic/core functions.
- Slurm: `cluster/jean_zay/relation_distance_pilot.slurm`.
- Two targeted CPU tests passed in the module environment: preserving positive
  and negative weight mass; control loss/gradient equivalence and physical
  weighting with permuted query/GT assignments. Python/Slurm syntax passed.
- GPU verification job **1026765** runs those checks in the fixed runtime and
  three real optimizer steps for **each of the four arms**, including initial /
  final metrics, gradient checks and frozen-state/probe checks.
- Pilot array **1026770_0–3** is queued with `afterok:1026765`; invalid dependency
  cancels the pilots. Up to four separate GPUs may run concurrently, subject to
  availability. Each task requests a two-hour maximum, not a duration estimate.

GPU verification **completed successfully in 3 min 5 s, exit 0**. All four real
smokes passed. Initial weights, initial decoded metrics and training batch order
were identical across arms. Frozen state and transformer tokens stayed identical;
relation-only node outputs stayed identical, while node-head coordinates updated.
Compact hashes and per-arm verification timings are recorded with the submission.
The four full pilots subsequently **completed successfully**; see the completed results below. Compact submission provenance is in
`experiments/spline_learning/metadata/metrics/relation-distance-pilot/submission.json`.

Outputs:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/relation_distance_pilot_20261013/`.
Each arm records initial/final summaries, optimizer/model checkpoint, frozen-state
and initial-weight hashes, batch-order hash, pair counts and explicit trainable
parameter names. Compare these before interpreting the four-way result.

```bash
sbatch --time=00:30:00 cluster/jean_zay/relation_distance_pilot.slurm verify
# Replace VERIFY_JOB with the returned ID:
sbatch --array=0-3%4 --dependency=afterok:VERIFY_JOB --kill-on-invalid-dep=yes \
  cluster/jean_zay/relation_distance_pilot.slurm
```

Output directories reject reuse. A rerun requires a separately recorded fresh
output root; completed or failed artifacts should remain available.

## Completed results — 2026-10-14

All four 1,000-update jobs completed with exit 0 in **17 min 33 s to 18 min 37 s**, including full validation. All 1,952 patches were scored in every arm. Paired initial weights, batch order and evaluation membership agree; frozen-state and transformer-token checks passed. No further model testing was needed.

Compare final full-validation scores with the saved full starting checkpoint, rather than the initial 128-case monitor.

| Model | Curve F1, 1 mm | TOPO-3D F1 | APLS-3D | Betti-0 error | Betti-1 error |
|---|---:|---:|---:|---:|---:|
| starting_checkpoint | 0.3426 | 0.1860 | 0.1042 | 2.7018 | 2.1332 |
| relation_control | 0.3336 | 0.1806 | 0.0990 | 3.0236 | 1.9759 |
| relation_near | 0.3351 | 0.1826 | 0.1007 | 3.3207 | 1.7208 |
| node_control | 0.3367 | 0.1834 | 0.1018 | 2.9180 | 2.0353 |
| node_near | 0.3342 | 0.1820 | 0.1009 | 3.3397 | 1.7382 |

At **2 mm**, conditional relation scores use only branches whose endpoints are both recovered:

| Model | Node recall | Both endpoints found | Long-branch recall (≥17.71 mm) | Near nonedge false-positive rate (<4.90 mm) |
|---|---:|---:|---:|---:|
| starting_checkpoint | 45.3% | 16.9% | 42.8% | 71.9% |
| relation_control | 45.3% | 16.9% | 39.3% | 73.7% |
| relation_near | 45.3% | 16.9% | 41.1% | 66.1% |
| node_control | 45.4% | 17.0% | 40.4% | 71.3% |
| node_near | 45.4% | 17.0% | 40.4% | 67.1% |

### Interpretation

- **Weighting has a modest specific effect:** with fixed nodes, near false-positive rate falls from **73.7% to 66.1%**, while long-branch recall moves from **39.3% to 41.1%**. Overall conditional relation recall remains about 66%. These comparisons use identical eligible nodes/pairs; they establish an effect of this weighting at the saved operating point, not a solved connectivity problem.
- **Node-head adaptation is small:** 2 mm node recall rises from **45.27% to 45.37%**, and endpoint-pair recovery from **16.88% to 16.95%**. Because relation features contain fixed tokens, not node-head outputs, both node arms produce the same node updates; relation weighting chiefly changes edge selection. This does not establish what would happen when shared image/token features are trainable.
- **Decoded graph quality does not improve over the starting checkpoint:** all four have lower Curve F1, TOPO and APLS. Weighting reduces cycles but increases disconnected-component error, consistent with removing links while leaving fragmentation. It should not be selected as an overall improvement based only on fewer near false links.
- **A large generalization gap is already present before these updates:** on the fixed 512 training patches, raw node recall is **97.81% at 1 mm / 98.77% at 2 mm**, versus **18.75% / 45.27%** on full validation. Training conditional long-branch recall is **99.70% / 99.66%**, versus validation **50.0% / 42.8%**. The train and validation populations are disjoint; both use the same physical diagnostic. Continued head learning barely changes localization and does not close this gap.
- The weighted relation-only arm reduces 2 mm training near-nonedge FPR from **0.66% to 0.22%**, while validation remains **66.1%**. Thus the current detector/scorer can distinguish connections on its training population but transfers poorly. This supports investigating overfitting or train/validation distribution differences; it does **not** prove which one is responsible or that token capacity alone is insufficient.

### Research decision

Do not extend these exact pilots merely because they completed successfully. The next useful question is why node/connection learning transfers poorly. A cheap candidate is to audit already saved earlier/best-checkpoint predictions with the same decomposition, then inspect sampling/augmentation and train/validation differences if the gap persists. Existing curve-image-evidence experiments remain relevant, but these results alone do not justify immediately adding more architecture. Further learning experiments require a new decision.

Compact full initial/final aggregates and provenance: `experiments/spline_learning/metadata/metrics/relation-distance-pilot/results.json`. Paired headline CSV: `experiments/spline_learning/metadata/metrics/relation-distance-pilot/comparison.csv`. Full per-patch predictions, diagnostics and offline W&B runs stay in the scratch arm folders above.
