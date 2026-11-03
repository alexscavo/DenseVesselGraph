# Controlled unfreezing — 2026-10-09

## Decision and question

The user authorized controlled unfreezing and parallel diagnostics. Frozen
residual learning on 10k patches reached validation curve F1 0.68995 and
10.27% endpoint-neutral shape improvement over chords, with fixed graph
detection. [Previous results](residual-token-diagnostics.md) support proceeding
to the joint-optimization phase. Scratch training remains deferred.

Question: with a successful graph detector and geometry head, does geometry
supervision harm graph learning through shared tokens or predicted coordinates?
Three arms isolate these routes without altering forward geometry or matching:

| Task | Arm | Geometry gradients to tokens | Geometry gradients through predicted endpoints |
|---|---|---|---|
| 823645_0 | `graph_only` | Blocked | Blocked |
| 823645_1 | `tokens_only` | Allowed | Blocked |
| 823645_2 | `joint` | Allowed | Allowed |

In **every** arm the spline MLP trains. The graph-only arm detaches both its
inputs, so the detector receives exactly graph-only gradients while the head
can adapt to changing features. This avoids confounding the comparison with
a stale frozen geometry head. There is no geometry-to-edge classifier module
in this model, so head updates cannot change graph predictions directly.

## Fixed protocol

- Same full model weights from `frozen_followup_20261009/residual_continue/model.pt`
  (15k cumulative frozen-head updates), including identical learned geometry.
- Same 10,000 training patches, original 512 training monitoring cases, and
  ordered 128 validation cases. No test-set access. Single seed 364505.
- **5,000 new updates per arm**, batch four. Fresh AdamW in all arms; no
  source optimizer state reused because trainable parameter groups change.
- Unfreeze transformer encoder/decoder, image projection, query embeddings,
  and node/class/relation heads at constant **7e-6**. Train spline MLP at
  **7e-5**. Existing AdamW betas/epsilon and weight decay **1e-4** unchanged.
- CNN (`model.encoder`) stays frozen. `model.decoder.encoder` is the distinct
  transformer encoder and is trainable. All BN buffers remain fixed. Model
  stays in eval mode with autograd enabled, disabling dropout in every arm.
  Thus stochastic layers cannot explain arm differences.
- Existing graph losses/weights and residual-control objective retained;
  spline weight **1.4739989612791453**. No new loss, matcher, target, image
  feature module or architecture. Graph-loss training progress is held at
  epoch 1/progress 0 for these bounded paired runs (classification curricula
  are disabled in the source configuration). No LR warmup or decay.
- All forward loss values are identical for identical model states across
  arms. Only geometry backward routes differ. Detaching endpoint coordinates
  still allows token gradients to alter future node predictions indirectly.

The smaller detector LR and frozen CNN define a conservative first unfreezing
stage. These runs do not test full-network training or reproduce scratch
initialization. There is no automatic further unfreezing after completion.

## Measurements and verification

At 0/1k/5k updates, evaluate decoded node/edge/common geometry metrics,
same-endpoint chords, and GT-matched-positive spline geometry independently.
Retain raw guide, fitted-target, residual and endpoint-neutral errors, plus
the <=1 mm endpoint subset. Centerline guides remain imperfect references.
GT-matched assignment identities are saved per patch. Report the fraction of
GT nodes whose assigned query changed since initialization and the previous
geometry evaluation; this measures assignment changes on fixed cases, not
intrinsic random instability or necessarily a bad permutation.

Every 500 updates, record validation losses and fixed-probe CNN, transformer
memory, token and node-output image sensitivity. At training updates 1/1k/5k,
record graph versus weighted-spline gradient norms and cosine alignment for
transformer encoder/decoder, node-coordinate head, projection, queries and
relation head. These are sparse single-batch probes, not population estimates.
Zero geometry gradients to the detector/control and coordinate head/token-only
arm are asserted. Track frozen CNN parameters and all buffers by digest.

Initial decoded metrics must match the source checkpoint. Record initial model
hash, dataset/evaluation membership and actual batch-order hash for a final
cross-arm audit. Each task has an independent three-update GPU smoke followed
by a fresh full trial. CPU tests verified:

1. Equal spline forward values across all routes; intended token/endpoint
   gradients blocked or nonzero, while head gradients remain nonzero.
2. Graph-only control detector gradients exactly match graph-only total loss.
3. CNN parameters/BN buffers remain unchanged after an optimizer update.
4. Existing frozen-head and geometric checks still pass (seven tests total).

Offline W&B, validation projections, branch-level metrics, final model and
optimizer state are saved. No prior checkpoint or result is overwritten.

## Interpretation before results

- If joint harms graph F1/image sensitivity while graph-only remains healthy,
  geometry gradient coupling is implicated in this fine-tuning regime.
- If tokens-only preserves performance while joint degrades, the predicted-
  coordinate route is implicated. This includes the endpoint-conditioned MLP
  route; residual supervision already removes the direct chord error term.
- If both geometry-coupled arms degrade similarly, shared-token gradients are
  a stronger candidate. Encoder-memory sensitivity and gradient probes help
  localize the change; a small cosine alone is not proof of causation.
- If every arm degrades, examine graph continuation/LR/matching behavior
  before attributing the problem to splines.
- If joint improves geometry and connectivity without sensitivity loss, it
  supports staged pretrained joint training under this recipe. It still does
  not establish the initiating cause of the original 300-epoch collapse.

Stop for evidence review afterwards. Do not rank methods by total loss alone.

## Files and submission

Code repository: `Graph-Native-Betti-Matching-Representation`.

- `scripts/controlled_spline_unfreeze.py`: diagnostic-only gradient routing,
  unfreezing, optimizer groups, monitoring and assertions; reuses existing
  criterion, loaders, metrics and pilot visualization helpers.
- `scripts/frozen_spline_diagnostic.py`: optional assignment identity output;
  default behavior unchanged for previous experiments.
- `tests/test_controlled_spline_unfreeze.py`: two targeted gradient/state tests.
- `configs/spline_controlled_unfreeze.yaml`: common short protocol.
- `cluster/jean_zay/spline_controlled_unfreeze.slurm`: three independent tasks,
  one H100 each, one-hour ceiling; each is GPU-smoke gated.

Submitted **823645_0/1/2**, initially pending for priority. Actual concurrent
execution depends on scheduler availability. Syntax and seven CPU tests passed;
GPU checks and results are pending at submission.

Run root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/controlled_unfreeze_20261009/`.
Logs: `experiments/spline_learning/logs/controlled-unfreeze-823645_<task>.{out,err}`.
Compact evidence: `experiments/spline_learning/metadata/controlled-unfreeze/`.

## Initial attempt stopped before trial updates

All tasks in 823645 exited 1 after roughly five minutes. All three separate
GPU smokes completed their three updates successfully. Each actual trial
stopped during initial evaluation on the same assertion: training common
curve F1 differed from the source run by **1.138147492e-6**, just above the
1e-6 diagnostic tolerance (0.7014372277 versus 0.7014383659). Validation curve
F1 was identical. This is not a trained unfreezing result or model collapse.

Saved-prediction comparison verified identical ordered sample IDs, query IDs
and edges on all 512/128 cases. Maximum normalized-coordinate/curve drift was
5.96e-7; maximum score drift was 5.61e-6. The three new arms produced identical
initial common metrics. These are tiny numerical differences in inference
that affect a thresholded curve metric; the particular kernel causing the
roundoff has not been isolated and is not necessary to resolve this failure.

Small fix: diagnostic aggregate-metric equality tolerance becomes 1e-5.
Metric distance thresholds, model weights, data, training and objectives are
unchanged. Replaying the saved initial common metrics reproduces three failures
under the old tolerance and passes under the new tolerance. Python/Slurm
syntax checks also passed. Runtime import warnings in the logs were not the
terminating error; the GPU smokes and full initial evaluations completed.

Resubmission uses the new output root `controlled_unfreeze_20261009_attempt2`
under the same runs directory. Original failed trials and successful smoke
artifacts are preserved. Each resubmitted task retains its GPU smoke gate.

Resubmitted as **824964_0/1/2** with the same arm ordering. Logs use `controlled-unfreeze-824964_<task>.{out,err}`. Results pending.

## Completed results: 824964_0/1/2

All completed 5,000 updates, exit 0:0, in **17m59s / 16m06s / 16m14s**.
Cross-arm audit verified identical initialized weights, initial metrics,
trainable parameters, optimizer settings, data/evaluation membership, and
actual batch-order hashes. CNN parameters and all buffers remained fixed.
The intended gradient blocking passed the sampled runtime assertions.

### Validation (128 cases; higher F1 is better)

| Arm | Node F1 | Edge F1 | Common branch F1 | Curve F1 | Own-chord curve F1 |
|---|---:|---:|---:|---:|---:|
| Initial frozen residual model | 0.90753 | 0.85858 | 0.25497 | **0.68995** | 0.65540 |
| Graph-only detector + detached geometry head | **0.90753** | **0.87126** | **0.27387** | 0.68420 | 0.65214 |
| Joint tokens, endpoints detached | 0.90544 | 0.86380 | 0.27148 | 0.68857 | **0.65284** |
| Fully joint | 0.90536 | 0.86197 | 0.26897 | 0.68195 | 0.64801 |

Native node/edge F1 and the representation-independent common branch/curve
scores use different matching rules; do not interpret their numerical scales
interchangeably. All three improve edge/branch F1 over initialization, but none
improves validation curve F1 over the initial frozen residual model.

Training-subset curve F1 progresses from 0.70144 to 0.70945 / 0.71599 / 0.71129.
Thus there is some training improvement without a corresponding validation
curvature gain. There is no single model that wins all measured objectives.

### GT-matched validation geometry and representation diagnostics

| Arm | Raw guide error mm ↓ | Endpoint-neutral guide error mm ↓ | Neutral improvement over chord ↑ | GT-query assignments changed |
|---|---:|---:|---:|---:|
| Initial | 0.89677 | **0.44987** | **10.27%** | — |
| Graph-only detector | 0.88313 | 0.45100 | 10.05% | 6.27% |
| Tokens, endpoints detached | **0.87929** | 0.45065 | 10.12% | 6.19% |
| Fully joint | 0.88406 | 0.45261 | 9.73% | 5.86% |

All 2,059 GT-positive validation branches remain evaluated. Raw geometry
improves slightly while endpoint-neutral shape is approximately unchanged or
slightly worse. Assignment changes are similar across arms and are lowest
for fully joint, so these data do not support a geometry-specific matching
instability explanation. A changed query assignment is not necessarily wrong.

Transformer-memory image sensitivity remains approximately **0.920** (initial
0.92030; finals 0.91775 / 0.91968 / 0.92019). No collapse is observed on the
fixed probes. The legacy `cnn_pair_mae` field measures **projected CNN features**
returned by the model, not raw frozen CNN activations; its small drift is
compatible with the trainable input projection and does not imply CNN updates.

Sparse gradient probes show weighted geometry/graph norm ratios around
0.11–0.20 in the transformer encoder and 0.14–0.21 in the decoder for coupled
arms. Encoder/decoder cosines are mostly near zero. In fully joint training,
the coordinate-head ratio is only about 0.027–0.071; its final cosine is
negative (~−0.48), but a single small-gradient probe cannot establish a cause.
These are three minibatch measurements, not population gradient estimates.

### Interpretation and decision

1. **Controlled pretrained unfreezing is stable here.** Neither geometry route
   reproduces the original catastrophic encoder collapse. This does not rule
   out a problem during scratch learning, with absolute targets, larger LR,
   train-mode stochastic layers or unfrozen CNN. Those conditions differ.
2. **There is no demonstrated benefit from geometry gradients into the detector
   in this short run.** The graph-only detector control has the best connectivity.
   Token-only coupling gives slightly better curve F1 than that control, but
   it does not beat the initial frozen model. Opening the endpoint route gives
   no measured advantage. Differences are single-seed validation observations,
   not established statistically significant rankings.
3. **Keep separate detector and geometry optimization as the conservative
   working route.** Train graph detection with its graph losses and adapt the
   residual head on detached detector features/endpoints. Keep the frozen
   residual checkpoint as the curvature reference; it still has the best
   validation curve F1. There is no basis yet for claiming a combined improvement
   in both connectivity and tortuosity or for restarting scratch training.

Next decision: verify this connectivity/geometry trade-off on a larger fixed
validation subset (and fixed-case visualizations) before committing to a
longer recipe; avoid selecting it from total loss alone. Do not use the test
set to keep choosing interventions. The initiating cause of the original
300-epoch failure remains unresolved. **No additional jobs were launched**
while reviewing these results.

Compact summaries, histories, gradient probes, comparison CSV and completion
audit are in `experiments/spline_learning/metadata/controlled-unfreeze/`.
Full artifacts remain under the `_attempt2` run root above. Original failed
attempts and all prior checkpoints are preserved.
