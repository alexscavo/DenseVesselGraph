# B-spline failure: evidence review and frozen-detector diagnostic

Date: 2026-10-09. Scope: Phase 1 review and preparation/checking of **Phase 2
only**. Meaningful GPU training requires the user's approval. No Phase 3/4
training, new full run, matcher change, or new loss is authorized here.

## 1. What the evidence establishes

| Experiment | Initialization / population / budget | Finding and limit |
|---|---|---|
| Original four-patch V0 | Original straight detector or scratch; four patches from one subject; 500 updates, joint training | Pretrained fitted-curve error 0.1093 mm vs same-endpoint chord 0.4812; scratch 0.3811 vs 0.4803. Both learn some geometry. No disjoint validation. |
| Earlier full fine-tuning | Original straight checkpoint, 9,709 training / 1,952 validation patches, 50 epochs | Native test node/edge F1 0.8841/0.8157, matched decoded guide distance 0.6396 mm on 13,071 test patches. Spline success extends beyond four patches; this is not the later equal-data scratch comparison. |
| Paired 300-epoch scratch | 10,000 training / 1,952 validation / 13,071 test; identical seed, graph population and shared initialization | Final common curve F1: straight 0.23051, spline 0.00135, adaptive 0.24711. Spline training node loss remains poor, and outputs become almost input-independent. |
| BN diagnostic | Final checkpoints, eight validation patches, no optimization | Saved vs current-batch BN does not rescue spline outputs. CNN image sensitivity persists. Rules out the proposed inference-BN repair on these probes, not all possible normalization effects during optimization. |
| Attention diagnostic | Four validation patches; hooks plus disposable decoder-offset intervention | CNN difference 0.85446 becomes transformer memory difference 3.42e-8. Straight memory difference 0.87522. Resetting decoder offsets does not restore outputs. Localizes the trained failure; does not identify its initiating cause. |
| Initial gradients | One actual training batch, no update | Weighted spline/graph norm ratios: tokens 1.35, coordinates 7.49, image features 1.24; coordinate cosine +0.779. Scale imbalance demonstrated on one batch, not systematic gradient opposition. |
| Weight pilot | Scratch, 512 training / 128 validation, 5,000 updates; weights 1.474 vs 0.1474 | Both poor, both input-sensitive. Only two original full-data epochs including warmup; cannot settle the eventual collapse mechanism. |
| Staged pilot | Same-data straight checkpoint epoch 35, fresh AdamW; graph-only vs joint spline ramp; 5,000 updates | Both slightly regress. Spline common curve F1 0.15400 vs own chords 0.15448. No demonstrated geometry benefit. Graph was never frozen. |
| Curve-evidence pilot | Same epoch-35 checkpoint/subset, joint training, identical symmetric scoring; evidence off/on | Final native edge F1 0.46077/0.50654. Curve F1 0.15971/0.21339; own chords 0.15729/0.21172. Improved connectivity but small shape contribution. Not a head-only experiment. |

The evidence pilot's initialization, batch-order, schedule, and membership
hashes agree between arms. Its change to symmetric training relation scoring
was shared by both arms; comparisons against older pilots are not a pure
image-evidence ablation.

Relevant prior reports: [collapse](spline-collapse-diagnosis.md),
[four-patch results](spline-learning-results.md),
[weight pilot](spline-weight-pilot.md), [staged pilot](spline-staged-learning.md),
[curve evidence](curve-informed-connectivity.md),
[scratch comparison](scratch-comparison-results.md). Also reviewed the original
`representation-implementation.md` and the code repository's
`docs/research/vascular_graph_roadmap.md`: preserve Hungarian matching and
separate topology, geometry, and representation capacity.

### What the encoder probe actually measures

The diagnostic hooks receive attention `input_flatten` with layout `[batch,
spatial positions, channels]`. Successive encoder-attention inputs expose the
representation between encoder layers. Decoder cross-attention receives the
final encoder memory. Thus the quoted differences are correctly interpreted
as between-image differences, not attention-head or token-axis differences.
The encoder's near-constant output is real on the recorded probes. Poor
attention support also exists in the healthy straight model, so unsupported
sampling alone is not a spline-specific explanation. No repeat GPU BN or
attention experiment is needed to re-establish these answered questions.

## 2. Implementation audit and ranked hypotheses

The head predicts six residual 3D vectors from the two object tokens, two
shared relation tokens, physical endpoints, chord vector, and chord length.
There is no direct local image sampling in the original head. Image information
must arrive through detector tokens. The output is a fixed clamped cubic
B-spline with eight controls including the two predicted nodes.

For internal control k, in physical coordinates:

`C_pred[k] = (1-g[k])*x_pred_start + g[k]*x_pred_end + R_theta[k]`.

The current V0 loss is Smooth L1 of `C_pred[k] - C_GT[k]`, beta 1 mm, averaged
over six internal controls, coordinates, and matched positive branches. It is
**absolute-control supervision despite the residual output parameterization**.
Consequently it penalizes node-position errors as well as the residual shape.
Gradients reach endpoints directly through the chord and indirectly through
the MLP's endpoint inputs, and reach shared tokens/image features. With the
zero-initialized final MLP layer, residual-path input gradients initially
vanish while the chord-to-endpoint gradient remains active.

Hungarian assignment uses class probability and normalized-coordinate L1
distance only, under `no_grad`. Spline supervision neither changes the matching
cost nor differentiates through assignment. GT edge identities select query
pairs through the node assignment; predicted edge acceptance is not required
for positive-pair geometry training. Canonical physical orientation reverses
both target controls and target samples together. The cache/graph identities
and metric transform are checked when loading targets. No verified matching,
orientation, or units bug was found in this audit.

Ranked, falsifiable hypotheses:

1. **Endpoint/localization coupling during joint optimization.** Frozen graph
   learning should permit geometry improvement without detector drift if the
   features are sufficient. The one-batch gradient result motivates this; it
   does not prove it initiated the scratch collapse.
2. **Noisy or changing correspondence and endpoint error.** With frozen
   detection and evaluation mode, assignment must remain fixed. Compare all
   matched pairs with the subset whose two endpoints are within 1 mm, and
   report endpoint-neutral geometry separately. No matcher change is proposed.
3. **Insufficient transferable shape information in pair tokens/head.** If
   frozen geometry fails on training and validation despite fixed correspondences,
   this remains plausible. Failure could also reflect optimization budget or
   absolute-target compensation, so it is not proof of insufficient capacity.
4. **Generalization/initialization and data breadth.** Training improvement
   without disjoint-validation improvement supports this direction. The original
   successful detector saw 91,200 patches; the newer staged checkpoint saw
   10,000. These initializations are not interchangeable controls.

## 3. First diagnostic: freeze the successful original detector

Use the same source checkpoint as successful four-patch and full fine-tuning:

`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/finetune_vessels_full_300ep/models/best_edge_f1_checkpoint.pt`.

Its recorded epoch is 300, iteration 855,000, with native validation edge F1
0.86321 on its original 19,200 validation patches. Those metrics do not replace
evaluation on the proposed frozen diagnostic population.

- Reuse the fixed **512 training / 128 validation** pilot membership (64/13
  subjects), eight controls, seed 364505, batch four. No test evaluation.
- Verify against pretraining membership: all 10,000 candidate training patches
  belong to original pretraining training; none of the 1,952 candidate validation
  patches or their subjects do. They belong to its validation population and
  were eligible for checkpoint selection. This is diagnostic validation,
  not a new unbiased test set.
- Freeze backbone, transformer, node and relation heads; only
  `spline_head.network.*` is trainable. Keep the detector in `eval()` throughout,
  including BN statistics and dropout. The frozen forward runs under `no_grad`.
- Keep **the existing absolute-control objective**, weight 1.474, beta 1 mm.
  Use fresh head-only AdamW at constant LR 7e-5, weight decay 1e-4, for
  **5,000 updates**. No warmup/ramp: this is an isolated head-learning diagnostic,
  not a matched causal comparison against the prior joint continuation.
  The inherited base configuration's 300-epoch field does not set this runner's
  stopping budget; the pilot config and provenance record 5,000 updates and the
  constant head-only schedule explicitly.
- Record loss/sensitivity every 500 updates; evaluate both splits at updates
  0, 1,000, and 5,000. Retain fixed projections and offline W&B records.
- Assert unchanged detector **parameters and buffers**, and unchanged
  per-sample Hungarian assignment hashes. Verify finite head gradients and
  absent detector gradients. Ordinary node/edge losses do not update anything.

Existing experiments do not already answer this experiment: all earlier
successful and staged spline training updated the detector jointly.

### Separate measurements

**GT-matched geometry:** every assigned GT-positive pair, irrespective of
classification or decoded connectivity. Report branch counts, paired curve vs
same-predicted-endpoint-chord errors against the fitted curve and original
full guide, relative improvement `(chord_error - curve_error)/chord_error`,
fraction of branches improved, and residual-control error. Aggregate errors
per branch, not per batch. Retain per-branch results and the prespecified
both-endpoints-within-1-mm subset. Empty subsets remain explicitly empty.

**Endpoint-neutral diagnostic:** evaluate the predicted deviation from its
chord after adding it to the GT chord. GT positions never enter the head input
or training objective. This removes a fixed localization contribution for
interpretation; it is not a deployable prediction or a benchmark replacement.

**Full decoded graph:** native node/edge F1 and common guide topology/geometry,
plus the exact same decoded nodes/edges represented by straight chords.
Node/edge detection should remain unchanged. This measures the effect of
geometry on a fixed predicted graph and cannot claim recovered missing links.

## 4. Residual formulation: assessment, not implementation of Phase 3

For this clamped B-spline, use the **Greville abscissae**:
`g = [0, 1/15, 1/5, 2/5, 3/5, 4/5, 14/15, 1]` for eight controls.
They satisfy `basis(t) @ g = t`, so chord control positions produce a linearly
parameterized chord. Equal control-index fractions would not.

The proposed `R_GT = C_GT - chord_GT(g)` is mathematically valid when computed
in the same physical coordinates and pair orientation. Compare to the
physical residual produced by the MLP; replacing the absolute objective would
remove its direct chord-localization term. These vectors describe deviation
from a chord, including tangential parameterization effects; they are not an
intrinsic curvature scalar.

Residual supervision alone leaves the MLP-input route from endpoint coordinates
to residuals. Detaching endpoint inputs as well would remove that route while
ordinary node losses could still train localization. Shared-token gradients
would remain and could still affect the transformer. These are distinct
interventions. Do not add both residual and absolute losses, or implement
detachment now; decide after the frozen diagnostic.

## 5. Decision rules and limits

| Result | Interpretation / next discussion |
|---|---|
| Training and validation improve over identical-endpoint chords | Frozen features and head support geometry learning. Joint optimization becomes a stronger target, but the original scratch-collapse cause remains unproven. Discuss one residual-loss or controlled unfreezing comparison. |
| Training improves, validation does not | Investigate transfer/generalization and guide/shape variation before scaling. |
| Endpoint-neutral improves but raw geometry barely changes | Fixed endpoint error limits deployed geometry; keep localization separate in conclusions. |
| Neither training nor validation improves | Check whether fixed endpoints/assignments dominate and whether the chosen budget learned anything; then isolate head/features/targets. Do not conclude B-splines are infeasible. |
| Node/edge predictions, BN buffers, or assignment hashes change | The freeze protocol failed; invalidate the experiment and fix it before interpretation. |

One checkpoint, seed, and finite budget cannot establish a universal mechanism.
The stronger original checkpoint makes this a learnability diagnostic, not an
equal-data demonstration that freezing causes success. If unfreezing is later
justified, use a graph-only continuation from the **same** checkpoint and
conditions. Stop here for approval and interpretation; no automatic Phase 3/4.

## Implementation and reproducible command

In `Graph-Native-Betti-Matching-Representation`:

- `scripts/train_spline_weight_pilot.py`: optional frozen path, training-split
  evaluation, provenance, fixed-state/assignment checks; existing pilots default
  to their original behavior.
- `scripts/frozen_spline_diagnostic.py`: head-only update and separate matched
  geometry measurements using the existing matcher, head, and loss.
- `configs/spline_frozen_head_pilot.yaml`: checkpoint, split, and update budget.
- `cluster/jean_zay/spline_frozen_head.slurm`: three-update smoke, then fresh
  initialization for the bounded trial, one H100 with a 30-minute allocation.
- `tests/test_frozen_spline_diagnostic.py`: CPU freeze/BN/gradient checks, paired
  chord measurements, Greville identity and endpoint-coupling check.

Prepared command, **not submitted**:

```bash
sbatch --output=experiments/spline_learning/logs/frozen-head-%j.out \
       --error=experiments/spline_learning/logs/frozen-head-%j.err \
       cluster/jean_zay/spline_frozen_head.slurm
```

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/frozen_head_pilot_20261009/`.
Smoke and actual run use separate directories and refuse to overwrite existing
results. Compact CPU verification is in `experiments/spline_learning/metadata/frozen-head/`.

## Local verification result

Nine focused CPU tests passed, including existing spline checks. The final three diagnostic tests also passed after the fixture refinement. Loading the real checkpoint on CPU succeeded with only the expected missing new spline tensors: exactly 1,540,098 MLP parameters are trainable, and all 89 BN layers remain in evaluation mode. Syntax and whitespace checks passed. These are correctness checks, not evidence of real-data geometry learning; GPU smoke and the 5,000-update trial have not been submitted.

## Authorized submission

The user subsequently authorized the prepared diagnostic. Submitted Slurm **812623**: one H100, 30-minute limit. It runs the three-update GPU smoke first, then restarts from the original checkpoint for the 5,000-update trial only if the smoke succeeds. Source hashes match the reviewed preparation. No completed GPU result is claimed at submission; stop after this Phase 2 experiment for interpretation. Logs: `experiments/spline_learning/logs/frozen-head-812623.{out,err}`.

## Startup correction and resubmission

Job 812623 failed after 1m02s, before any training updates: the generic evaluation loader intentionally accepts only val/test, but the new diagnostic requested train. The diagnostic now builds an unshuffled loader over a separate view of the existing selected training dataset; membership and augmentation settings are unchanged. At the user’s request, skipped a separate real-sample test and resubmitted directly after syntax/whitespace checks as **813146**. The in-job GPU smoke still gates the 5,000-update trial. New output root has suffix `_attempt2`; failed artifacts are preserved.

## Completed diagnostic — job 813146

Completed successfully in **14m17s**, including the smoke and 5,000-update
trial. The actual trial took 717s including monitoring. The saved export audit
confirmed exactly identical node coordinates, node/edge scores, edges, query
identities, and sample order before/after on all 512 training and 128 validation
patches. Detector parameter/buffer and Hungarian-assignment checks passed.
Input sensitivity also remained exactly unchanged. No new test-set evaluation.

| Metric (final; chord → spline) | Training | Disjoint validation |
|---|---:|---:|
| GT-matched branch guide distance, mm ↓ | 0.96148 → 0.57203 | 0.95368 → 1.03307 |
| GT-matched fitted-curve distance, mm ↓ | 1.25866 → 0.71961 | 1.24956 → 1.40366 |
| Full decoded common curve F1 ↑ | 0.64812 → 0.83403 | 0.65540 → 0.58567 |
| Endpoint-neutral matched guide distance, mm ↓ | 0.51414 → 0.55563 | 0.50138 → 0.70664 |
| Native node F1 (unchanged) | 0.90845 | 0.90753 |
| Native edge F1 (unchanged) | 0.86265 | 0.85858 |

GT-matched distances cover 8,454 training / 2,059 validation branches; decoded
curve F1 evaluates full predictions and is a separate measurement. Relative
matched-guide improvement is **40.5% training**, **−8.3% validation**. For the
localized subset (both endpoints within 1 mm), guide improvement is 41.2% on
2,665 training branches and −14.4% on 619 validation branches. Thus validation
failure is not confined to the worst localized endpoints.

At update 1,000, validation curve F1 improved slightly to 0.66316 before later
regressing. Validation control loss reached 0.30996 there (initial 0.31400),
then rose to 0.37477 at update 5,000 while training loss continued falling.
This is evidence of overfitting at the final budget, not proof that no early
validation benefit exists. No intermediate checkpoint was retained or selected
as a new benchmark model.

### Interpretation and next decision

Freezing works as designed and the head can learn training geometry. However,
it does **not** by itself yield transferable curvature prediction. The original
scratch-collapse cause remains unresolved; joint detector degradation cannot
explain this experiment's validation failure because the detector is unchanged.

Absolute-control learning appears to include compensation for fixed endpoint
errors: residual-control error increases (training 0.56750 → 0.96709 mm;
validation 0.55166 → 1.00938 mm), and the all-pair endpoint-neutral diagnostic
worsens despite large raw training gains. This supports a localization/shape
confounding hypothesis, not a complete causal attribution. Some genuine shape
learning is present in the localized training subset: endpoint-neutral guide
error improves 20.9%; its validation counterpart worsens 33.1%.

The smallest next discriminative experiment is a **frozen-detector residual-
control objective replacing the absolute-control objective**, with the same
checkpoint, seed, samples, head initialization, optimizer, and budget. The
existing completed run supplies its absolute-loss control. All endpoints are
already detached by freezing, so no separate detachment intervention is needed
for that experiment. Compare residual/endpoint-neutral and raw guide geometry,
including the same early monitoring points. Do not unfreeze, add an image
module, or launch a long run on these results. Await discussion/approval before
implementing or launching Phase 3.

Compact verified summary, history, comparison CSV, and completion audit:
`experiments/spline_learning/metadata/frozen-head/`. Full output remains in the
`frozen_head_pilot_20261009_attempt2/absolute_controls/` run directory.

## Authorized parallel follow-up

The user authorized next tests and requested multiple diagnostics together. Submitted **815195_0/1/2**, completing a frozen 2×2 absolute/residual × full-token/endpoint-only comparison using job 813146 as the existing fourth arm. No unfreezing or long training is included. [Protocol and interpretation](residual-token-diagnostics.md).
