# Frozen residual loss and token-input diagnostics — 2026-10-09

## Why these tests

The completed [frozen absolute-control diagnostic](frozen-spline-investigation.md)
learned training geometry but regressed on validation. Detector state and
correspondences remained fixed. Its endpoint-neutral results suggest that
absolute-control supervision partly learns compensation for localization error.
It remains unclear whether detector tokens supply transferable shape information.

The user authorized the next tests and requested several diagnostics together.
Run three new arms in parallel to complete a small 2×2 comparison. Reuse the
completed absolute/full-token run as the fourth arm; do not repeat it.

| Arm | Geometry objective | Spline-head token input | Status |
|---|---|---|---|
| Existing `absolute_controls` | Absolute control coordinates | Original tokens | Completed, job 813146 |
| `residual_full` | Control deviation from chord | Original tokens | New task 0 |
| `absolute_endpoints` | Absolute control coordinates | Zeroed token features | New task 1 |
| `residual_endpoints` | Control deviation from chord | Zeroed token features | New task 2 |

Endpoint-only arms still use **image-predicted endpoints**, their physical
coordinates, chord direction/vector, and length. They are not image-blind.
Zeroing pair-token features preserves the MLP architecture, initial tensors,
and nominal parameter count. It reduces effective input information/capacity;
interpret any change as this specific ablation, not proof that tokens contain
no geometric information.

## Fixed conditions

All four arms use the original successful straight checkpoint, the same
512 training / 128 validation patches (64/13 subjects), seed 364505, batch four,
eight controls, fresh head-only AdamW with constant LR 7e-5 and weight decay
1e-4, 5,000 updates, and spline weight 1.4739989612791453. All detector parameters
and BN buffers remain frozen in evaluation mode. No new data, test-set scoring,
matching changes, image-feature module, or unfreezing is introduced.

Monitoring matches the completed control: loss/sensitivity every 500 updates;
full training/validation matched-positive and decoded-graph evaluations at
0, 1,000, and 5,000 updates. Each curve is compared with its own identical-
endpoint chord. Save raw-guide errors, fitted-target errors, residual errors,
endpoint-neutral diagnostics, the localized endpoint subset, fixed projections,
and W&B offline data. Do not compare total losses across different objectives.

## Residual objective and implementation

Only the spline objective changes in residual arms; it replaces the absolute
error instead of adding a second term. For an internal control at its fixed
Greville coordinate g, in physical millimetres:

```
endpoint_delta = predicted_endpoints - GT_endpoints
residual_error = (predicted_control - GT_control)
                 - ((1-g)*endpoint_delta[start] + g*endpoint_delta[end])
loss = SmoothL1(residual_error, beta=1 mm)
```

This equals predicted residual minus GT residual. Physical coordinate mapping
and canonical pair orientation are unchanged. Predicted endpoints still anchor
the output curve; GT endpoints only define supervision. Detachment is already
provided by the frozen detector's `no_grad` forward, so there is no additional
detachment intervention. The existing absolute geometric error metrics retain
their original meanings regardless of objective.

- `loss.spline.target` selects `absolute` (unchanged default) or `residual`.
  Validation restricts the new residual option to B-spline control loss.
- `model.spline.token_inputs` selects `full` (unchanged default) or `zero`.
  The head applies masking in its shared forward, including inference.
- The pilot runner records these settings and verifies initial tensor hash,
  checkpoint, detector hash, loss weight, learning rate/schedule, membership,
  probe IDs, assignment hashes, and final batch-order hash against the saved
  corresponding smoke/trial control. Comparison failures invalidate the run.

Eleven focused CPU tests passed, including residual error in physical units,
removal of direct translation coupling at zero residual, finite/nonzero head
gradients, the remaining endpoint-feature route once residual weights are
nonzero, token masking in train/eval modes, exact detector/BN freezing, and
existing B-spline behavior. Syntax/Slurm checks passed. Each new GPU task runs
its own three-update smoke before restarting the actual trial from the original
checkpoint and seed. Original results remain untouched.

## Interpretation and stopping point

- Residual/full vs absolute/full tests whether replacing localization-contaminated
  targets improves held-out residual shape and raw geometry. Improvement would
  support this target-confounding explanation; it would not by itself identify
  the initiating cause of the original scratch encoder collapse.
- Full vs endpoint-only, within each objective, tests the incremental predictive
  value of tokens under this recipe. A training-only advantage suggests
  memorization; a validation advantage supports transferable token information.
- If residual helps only endpoint-neutral geometry, fixed node localization may
  still limit deployed curves. If raw error improves while endpoint-neutral
  geometry worsens, avoid labeling the gain as pure curvature recovery.
- Inspect the shared early evaluation (1,000 updates) as well as the fixed
  final budget; the control previously overfit after an early small gain.
- If all variants fail validation, do not automatically unfreeze or launch a
  full training. Revisit feature/target generalization with the measured evidence.

Stop after these frozen diagnostics to discuss results. Phase 4 joint training
would need a justified, matched graph-only continuation from the same checkpoint.

## Locations

Config: `DenseVesselGraph/configs/spline_frozen_factorial_pilot.yaml`.
Submission: `cluster/jean_zay/spline_frozen_factorial.slurm` in that repository.
Each array task requests one H100 for at most 30 minutes; tasks may run in
parallel subject to scheduler availability.

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/frozen_factorial_pilot_20261009/`.
It contains each named arm and separate `smoke-<arm>` directories. Existing
directories are never overwritten.

Compact submission evidence:
`experiments/spline_learning/metadata/frozen-factorial/` in the code repository.

## Submission

Submitted array **815195**: `_0` residual/full tokens; `_1` absolute/endpoints only; `_2` residual/endpoints only. The three tasks are independently smoke-gated and may run concurrently. No completed result is claimed at submission. Logs: `experiments/spline_learning/logs/frozen-factorial-815195_<task>.{out,err}`.

## Completed results

All three jobs completed 5,000 updates with exit 0:0. Elapsed times including
smoke/staging: 815195_0 **14m25s**, _1 **14m27s**, _2 **14m00s**. Initialization,
checkpoint, batch-order, membership and fixed-assignment checks agree with the
completed control. Detector state and image sensitivity remain fixed. Native
validation node/edge F1 is unchanged at 0.90753 / 0.85858 for every arm.

### Full decoded graph: common whole-curve F1 (higher is better)

| Representation / objective / input | Training | Validation |
|---|---:|---:|
| Same-endpoint straight chords | 0.64812 | 0.65540 |
| Absolute / full tokens (previous control) | 0.83403 | 0.58567 |
| Residual / full tokens | 0.79694 | 0.61590 |
| Absolute / endpoints only | 0.65511 | 0.65584 |
| Residual / endpoints only | 0.65447 | 0.64957 |

### GT-matched branch geometry (millimetres; lower is better)

These rows concern 8,454 training / 2,059 validation positive pairs regardless
of decoded edge acceptance, not the subset of recovered decoded branches.

| Objective / input | Train raw guide error | Val raw guide error | Train endpoint-neutral error | Val endpoint-neutral error |
|---|---:|---:|---:|---:|
| Chords | 0.96148 | 0.95368 | 0.51414 | 0.50138 |
| Absolute / full tokens | 0.57203 | 1.03307 | 0.55563 | 0.70664 |
| Residual / full tokens | 0.78370 | 0.97352 | 0.25340 | 0.53300 |
| Absolute / endpoints only | 0.94720 | 0.94739 | 0.53434 | 0.53041 |
| Residual / endpoints only | 0.95061 | 0.95500 | 0.50488 | 0.50778 |

At update 1,000, residual/full validation curve F1 was 0.65541 (essentially
its chord baseline) and endpoint-neutral error 0.49803 (a small improvement
from 0.50138). Later it overfits. Endpoint-only validation F1 at update 1,000
was 0.65964/absolute and 0.65733/residual. Small early gains are not evidence
of robust generalized curvature recovery or a statistical advantage.

### What these tests establish

1. **The residual objective changes what is learned.** With full tokens,
   endpoint-neutral training error falls 50.7% below chords, versus worsening
   8.1% under absolute supervision. Thus the large absolute-loss training gain
   was partly localization compensation, while residual training can learn
   actual deviation from the GT chord on training data.
2. **Residual targets alone do not solve generalization.** Validation raw
   guide error remains 2.1% worse than chords, endpoint-neutral error 6.3%
   worse, and decoded curve F1 below the chord baseline. It substantially
   reduces the damage relative to absolute/full but is not a validated fix.
3. **Token inputs enable training fit, not demonstrated validation transfer
   under this recipe.** Removing them nearly eliminates large training gains
   and the large validation deterioration. The endpoint-only models mostly
   match the chord baseline; they are not a successful curved representation.
   Their lower effective input capacity is a confound when interpreting token
   usefulness. These results do not prove that tokens lack shape information.
4. **This failure is broader than detector drift.** Geometry generalization
   fails with weights, BN statistics and matching fixed. The original scratch
   encoder collapse is still an observed final-state failure whose initiating
   cause remains unidentified; these frozen experiments do not reproduce it.

### Next decision (not launched)

Before unfreezing or adding a new architecture, test whether limited geometry
training data explains the gap: keep the residual/full frozen setup and expand
only training membership to the already prepared 10,000 patches, retaining
these 128 validation patches and a matched update budget. This is a proposed
short diagnostic of data breadth, not a long run or claimed solution. A failure
would still need interpretation against training fit and optimization budget;
it would not prove features are uninformative. Alternatively, a local-feature
head would test a different architectural hypothesis and should be discussed
explicitly before implementing it. No further training was submitted while
recording these results.

Verified summaries, histories, comparison CSV and completion audit are saved
in `experiments/spline_learning/metadata/frozen-factorial/`. Full models,
predictions, projections, per-branch results and offline W&B data remain in the
run directories listed above. Original results are preserved.

## Authorized follow-up: broader geometry training data

The user approved the next diagnostic. Run one new residual/full-token frozen
arm on the existing **10,000 training patches**, reusing completed 512-patch
residual/full job 815195_0 as the control. Keep the original checkpoint, head
initialization, loss weight, optimizer, constant LR, 5,000-update budget, batch
size, and ordered 128 validation patches. The detector and BN remain frozen.
Training geometry evaluation retains exactly the original 512 training cases;
this is a fixed monitoring subset, not an estimate across all 10,000 cases.
Validation and train metrics remain directly comparable on their fixed cases.

Five thousand batch-four updates expose 20,000 examples: approximately two
passes over 10k instead of 39 over 512. Thus the intervention tests broader
membership with less repetition at equal update budget. A negative result can
still reflect insufficient optimization per sample, and cannot establish that
tokens contain no usable geometry. Batch order necessarily changes; its hash
is recorded but is not asserted equal. Initialization, settings, validation
membership, and assignments on the fixed evaluation cases are still checked
against the saved control. No original results are overwritten.

Minimal implementation: generalize the runner's hard-coded 512/128 assertion,
add an explicit expanded-training reference comparison, and evaluate training
geometry on the control's saved membership. No model, loss, matcher, or data
cache changes. Config `configs/spline_frozen_breadth_pilot.yaml`; submission
`cluster/jean_zay/spline_frozen_breadth.slurm`. A separate three-update GPU smoke
must pass before the fresh 5,000-update trial. One H100, 30-minute limit.
Outputs: `runs/frozen_breadth_pilot_20261009/{smoke-residual_full,residual_full}`
under the same spline-learning root. W&B remains offline.

**What this says about the 300-epoch failure:** joint optimization and training
setup remain plausible contributors, but there is no evidence that a single
setup mistake is the sole cause. Original BN diagnostics did not rescue the
collapsed model, and frozen-detector experiments exhibit a separate geometry
generalization problem without detector drift. Successful fitting establishes
representational feasibility on training cases, not reliable transfer. This
follow-up tests data breadth; it does not directly reproduce or identify the
initiating cause of encoder collapse. Stop after results for interpretation.

Submitted as **816293**, initially pending for priority. Syntax/shell checks, five CPU unit tests, and exact dataset-membership checks passed. The first local unit-test attempt encountered stale SymPy files in a reused temporary runtime directory; rerunning in a fresh unique directory passed. Slurm uses its own job-specific staged runtime. Logs: `experiments/spline_learning/logs/frozen-breadth-816293.{out,err}`. Submission evidence: `experiments/spline_learning/metadata/frozen-breadth/submission.json`. No results yet.

## Broader-data result: 816293 completed

Completed successfully in **14m19s**, exit 0:0, with 5,000 trial updates after
the separate smoke. Initialization/settings checks passed against residual/full
815195_0. Detector parameters, BN buffers, sensitivity, matching, and native
node/edge detection remained fixed. Exact ordered evaluation membership was
verified: original 512 training cases and 128 validation cases. The expanded
training set still covers the same 64 subjects; this adds patches, not subjects.

### Fixed-case decoded curve F1 (higher is better)

| Representation / head training population | Training subset | Validation |
|---|---:|---:|
| Same-endpoint straight chords | 0.64812 | 0.65540 |
| Residual/full, 512 patches | 0.79694 | 0.61590 |
| Residual/full, 10,000 patches | 0.67275 | 0.67865 |

The new validation curve F1 improves **0.02325 absolute** over identical-endpoint
chords. At update 1,000 it was 0.66275; the final value improves further rather
than showing the deterioration seen in the 512-patch control.

### GT-matched geometry (millimetres; lower is better)

| Representation | Train raw guide | Val raw guide | Train endpoint-neutral guide | Val endpoint-neutral guide |
|---|---:|---:|---:|---:|
| Chords | 0.96148 | 0.95368 | 0.51414 | 0.50138 |
| Residual/full, 512 patches | 0.78370 | 0.97352 | 0.25340 | 0.53300 |
| Residual/full, 10k patches | 0.92181 | 0.91881 | 0.47632 | 0.47094 |

These are the same 8,454/2,059 GT-positive pairs as before, independent of
decoded edge acceptance. New validation raw guide error improves **3.66%**
against chords, and endpoint-neutral error **6.07%**. The localized-endpoint
subset (619 validation branches) also improves: 5.83% raw / 6.57%
endpoint-neutral. Native validation node F1 / edge F1 remains
**0.90753 / 0.85858**. Connectivity did not improve; this gain is geometric.

### Interpretation and next decision

- Broader patch coverage with fewer repetitions at equal update budget reduces
  the observed overfitting and yields a modest validation curvature gain. This
  supports usable geometric information in the frozen detector features under
  residual supervision; the earlier 512-patch failures do not establish that
  these features are intrinsically insufficient.
- This is one seed and 128 validation patches from 13 subjects. Endpoint-neutral
  mean error improves, but only about 49.9% of branches improve individually;
  benefit is not uniform. No statistical significance or held-out test claim.
  Centerline guides remain imperfect references.
- It still does **not** establish that the original 300-epoch collapse was
  caused only by its training setup, nor isolate a particular setting. This
  run starts from a successful detector, freezes it, and uses residual targets;
  the scratch experiment changed all these conditions. It also does not prove
  residual supervision is necessary: no 10k frozen absolute-target arm was run.
- Recommended next decision: a bounded continuation of this frozen residual
  head, retaining the same monitoring, to assess whether gains grow beyond two
  data passes before risking detector unfreezing. This would test optimization
  budget, not directly identify the scratch-collapse trigger. Any later joint
  continuation should include the matched graph-only control required by the
  original investigation plan. **No new jobs launched in this results review.**

Compact results, history, comparison CSV and completion audit are preserved in
`experiments/spline_learning/metadata/frozen-breadth/`. Full model, branch-level
metrics, projections and offline W&B run `b4exe3tv` remain in the run directory.

## Authorized parallel follow-up: budget and objective

The user approved continuation and requested another diagnostic in parallel.
Training the whole model from scratch remains excluded at this decision point.
Prepare two independent jobs with unchanged detector/BN, eight controls, 10k
training patches, fixed 512 training monitoring cases and 128 validation cases:

| Arm | Initialization | Objective | New updates | Comparison |
|---|---|---|---:|---|
| `residual_continue` | Full model weights from completed 816293 | Residual | 10,000 (15,000 cumulative) | Its own starting metrics and completed 816293 |
| `absolute_full` | Same original straight detector and fresh head initialization as 816293 | Absolute controls | 5,000 | Completed 816293 at the identical 5,000-update budget |

The parallel absolute arm tests whether residual supervision helps once data
breadth is matched. Do not compare it directly against the longer continuation
as if they had equal optimization budgets. Initial tensors, batch-order hash,
matching, detector digest, LR, weight and dataset membership are checked against
816293 for the absolute arm. Its only intended training difference is objective.

The prior checkpoint saved model weights but **not AdamW moments**. Therefore
the residual continuation explicitly restarts AdamW and the seeded loader
shuffle, with unchanged constant LR 7e-5 and weight decay 1e-4. This is a weight
continuation, not an exact interrupted-run resume, and its effect cannot be
attributed exclusively to extra updates. Source checkpoint loading is strict;
model/loss configurations must match, detector digest and evaluation assignment
hashes must remain fixed. Source and new checkpoints remain separate. New final
checkpoints also save optimizer/scheduler states and cumulative update count.

Both use the existing reusable runner. The only additions are optional saved
evaluation membership and optional checkpoint weight continuation with explicit
provenance; no model/loss/matcher modifications. Configs:
`configs/spline_frozen_residual_continue.yaml` and
`configs/spline_frozen_absolute_10k_pilot.yaml`. Submission:
`cluster/jean_zay/spline_frozen_followup.slurm`, array tasks 0/1 respectively.
Each gets one H100 and a 30-minute limit; each runs a separate three-update GPU
smoke before its fresh trial. Monitoring and same-endpoint chord comparisons
remain unchanged; continuation full geometry is evaluated at 0/5k/10k new
updates, absolute at 0/1k/5k. Offline W&B and projections remain enabled.

Output root: `runs/frozen_followup_20261009/`, separate arm and smoke directories.
After results, decide whether longer frozen learning helps and whether the
absolute objective damages shape generalization. No unfreezing, new architecture,
scratch training, or 300-epoch run is part of this submission.

Submitted as **817904_0** (residual continuation) and **817904_1** (absolute control), independently schedulable on one H100 each. Five CPU tests, configuration/checkpoint compatibility, fixed evaluation-membership and syntax checks passed. GPU smoke is queued within each task. Logs: `experiments/spline_learning/logs/frozen-followup-817904_<task>.{out,err}`. Compact submission evidence: `experiments/spline_learning/metadata/frozen-followup/submission.json`. No completed result claimed.

## Parallel follow-up completed: 817904_0/1

Both completed with exit 0:0: residual continuation **17m51s**, absolute
**13m58s**. Comparison audit verified identical training/evaluation membership,
fixed detector digest, sensitivity, matching, and native node/edge scores.
Continuation's initial decoded and matched geometry metrics exactly match the
source checkpoint's final metrics. Absolute and original residual 5k runs have
identical initialized tensors and actual training batch sequence.

### Decoded curve F1 on the fixed evaluation cases

| Objective / cumulative head updates | Train subset | Validation |
|---|---:|---:|
| Same-endpoint chords | 0.64812 | 0.65540 |
| Absolute / 5k | 0.66996 | 0.67461 |
| Residual / 5k (816293) | 0.67275 | 0.67865 |
| Residual / 10k (intermediate evaluation) | 0.69014 | 0.68682 |
| Residual / 15k (continuation final) | 0.70144 | 0.68995 |

### GT-matched validation geometry, all 2,059 pairs

| Objective / updates | Raw guide error mm | Endpoint-neutral guide error mm | Endpoint-neutral improvement vs chord |
|---|---:|---:|---:|
| Chords | 0.95368 | 0.50138 | — |
| Absolute / 5k | 0.92239 | 0.51964 | **−3.64%** |
| Residual / 5k | 0.91881 | 0.47094 | **+6.07%** |
| Residual / 15k | 0.89677 | 0.44987 | **+10.27%** |

Final residual raw guide error improves 5.97% against chords. On the 619
validation pairs with both endpoints within 1 mm, raw guide error improves
9.47% and endpoint-neutral error 12.02%. Native node/edge F1 remains
0.90753 / 0.85858; there is no connectivity change. Training endpoint-neutral
error improves 15.15% for the final residual model, versus a 2.36% worsening
for absolute supervision. The intermediate 10k point is an evaluation, not a
separately saved model checkpoint.

### What is now supported

1. **Keep residual supervision for shape learning.** At the matched 5k budget,
   decoded F1 differs modestly (+0.00404 for residual), but residual improves
   endpoint-neutral shape while absolute worsens it. Absolute fitting improves
   raw geometry without improving transferred residual shape, consistent with
   learning some compensation for localization error. This is not a significance
   claim, nor evidence that absolute losses cannot work in any other setting.
2. **Longer frozen learning improves validation here.** Validation F1 progresses
   0.67865 → 0.68682 → 0.68995, with a smaller later gain. Improvement is not
   restricted to training cases. The continuation restarted AdamW/shuffle, so
   the effect cannot be isolated to extra updates alone. Shape gains are not
   uniform: about 50.7% of validation branches improve endpoint-neutral error.
3. **Frozen pretrained features can support useful curved geometry.** The
   previous small-subset overfitting was not a demonstration of an inherently
   unusable spline head or representation. The causal trigger for original
   scratch encoder collapse is still unproven, and this does not justify a
   new scratch or 300-epoch run.

Next research decision: keep this residual frozen model as the reference. If
the next priority is connectivity and the joint-optimization hypothesis, a
short controlled unfreezing experiment against matched graph-only continuation
is now more informative than another unpaired long run. The exact unfreezing
scope and treatment of endpoint-feature gradients should be specified before
submission; no such job was launched during this results review. All results
remain single-seed diagnostic validation, not held-out test results.

Evidence and comparison audit saved in
`experiments/spline_learning/metadata/frozen-followup/`; full models, metrics,
projections and offline W&B data remain under `runs/frozen_followup_20261009/`.
