# Scratch extension to 300 epochs — 2026-10-12

## Question and decision

User explicitly authorized extending the successful graph-only and detached
B-spline scratch pilots to 300 total epochs. Neither uses an external pretrained
detector. The collapsed token-coupled arm is not extended. The question is whether
curvature learning becomes useful as scratch graph localization improves, while
geometry gradients remain isolated from graph learning.

The preceding [isolation diagnostics](scratch-detached-geometry.md) verified zero
detector gradients from the detached geometry objective. This experiment tests
longer-term learning and stability; it does not identify the exact mechanism of
the earlier coupled collapse, nor demonstrate that curves improve connectivity.

## Fixed conditions and continuation

- Same 10,000 SyntheticMRI training patches, batch 4, no augmentation, seed 364505.
- Same ordered 128 validation and 512 training monitoring cases; memberships
  verified identical across source arms. Full validation remains 1,952 cases.
- Sources: `scratch_local_pilot_20261010_attempt2/graph_control/model.pt` and
  `scratch_detached_pilot_20261011/detached_spline/model.pt` under the existing
  `spline_learning/runs` root. Both completed 25,000 updates / 10 epochs.
- Continue weights **and AdamW moments**, no new warmup. Preserve the original
  750,000-update polynomial schedule, base LR 7e-5, power 0.9, weight decay 1e-4,
  2,500 warmup updates. Existing scheduler formula is unchanged.
- Control spline weight 0, `tokens_only` route. Detached spline weight
  1.4739989612791453, `graph_only` route: detector receives graph losses;
  residual geometry head learns from detached tokens/endpoints/local CNN values.
  Same eight-control B-spline head and local geometry implementation.
- BN updates normally during training; evaluation uses saved running statistics.
- Original checkpoints did not store RNG/sampler state. Both extensions restart
  their random/shuffle streams identically. This is **not** an exact uninterrupted
  continuation of the old random stream. New stage checkpoints preserve Python,
  NumPy, CPU/CUDA torch and loader generator states at exhausted epoch boundaries.
  CUDA arithmetic remains nondeterministic as established by the earlier probe.

## Monitoring and selection

Validation losses, graph metrics, same-endpoint chord comparison, fixed projection
images and image sensitivity every 5 epochs. Fixed 512-case training geometry and
separate GT-matched geometry on train/validation every 20 epochs and at stage ends.
This reduces monitoring frequency versus the short pilots; losses and update
budgets are unchanged. W&B uses the existing offline workflow, group
`spline-scratch-extend300-20261012`.

Save rolling `latest.pt`, final `model.pt` for each stage, and `best.pt`. Best is
selected by **highest validation common branch F1, then native edge F1**; the
source checkpoint is included and the best carries across stages. No selection
on total loss or the test set. Scheduled full validation evaluates both best and
final models, preserving the distinction between GT-matched shape metrics and
full decoded-graph performance. It produces comparison tables and a portable
six-case HTML gallery using existing evaluation code. These are validation
results, not an unbiased held-out test claim.

## Jobs, outputs and reproducibility

The partition allows 20 hours per job. Each model therefore uses two dependent
stages on one H100 each:

| Array | Task 0 | Task 1 | Dependency |
|---|---|---|---|
| 949063 | control, epochs 11–150 | detached, epochs 11–150 | three-update GPU resume smoke in each task |
| 949065 | control, epochs 151–300 | detached, epochs 151–300 | corresponding task in 949063 succeeds |
| 949071 | four full-validation tasks, maximum two concurrent | best and final for both arms | all of 949065 succeeds |

Initial arrays 949026/949030/949034 were cancelled before Python started: Slurm
held the first array with `user_env_retrieval_failed_requeued_held`. Local Slurm
documentation confirms `--export=ALL,NAME=value` implicitly retrieves the login
environment. Replacement submissions use plain `--export=ALL` and positional
arguments; experiment settings were unchanged.

Code/configs in `Graph-Native-Betti-Matching-Representation`:

- `scripts/extend_scratch_geometry.py`: bounded scratch continuation; reuses
  existing training, criterion routing, monitoring and geometry evaluation.
- `configs/spline_scratch_extend300.yaml`: paired source runs and monitoring.
- `cluster/jean_zay/spline_scratch_extend300.slurm`: smoke then continuation.
- `configs/spline_scratch_extend300_validation.yaml`: four checkpoint evaluations.
- `cluster/jean_zay/spline_full_validation.slurm`: accepts optional config argument
  in addition to the existing environment-variable/default interface.

Commands: `sbatch --export=ALL cluster/jean_zay/spline_scratch_extend300.slurm 150`,
then the same with `--dependency=aftercorr:<first-array-id>` and argument `300`.
Evaluation: `sbatch --export=ALL --array=0-3%2 --dependency=afterok:<second-array-id>
cluster/jean_zay/spline_full_validation.slurm configs/spline_scratch_extend300_validation.yaml`.
Use distinct output roots if repeating the experiment; existing directories are
deliberately rejected to preserve artifacts.

Output root:
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/scratch_extend300_20261012/`

Each arm has `epoch150/` and `epoch300/`; evaluation uses `full_validation/`.
Logs are `experiments/spline_learning/logs/scratch-extend{150,300,val}-<array>_<task>.{out,err}`.

## Checks and time estimate

Python/Slurm syntax passed. Two focused CPU tests passed: restored AdamW produces
the identical next parameter update/LR/moments, and epoch-boundary RNG restoration
reproduces the next shuffled epoch and torch random draws. Real GPU smoke gates
check strict model loading, restored scheduler position, finite losses/gradients,
active detector updates, the expected spline gradient, and decoded/matched metrics
on four cases before long training starts.

Both epoch-150 GPU smokes **passed** (25,000→25,003 updates); restored LR was
6.810713331450913e-5 and reached 6.810688054571481e-5 after three updates in
both arms. Both production tasks are running, one H100 each. Initial 128-case
decoded metrics and fixed-probe image sensitivity reproduce the source pilot
values. The first production forward/backward/AdamW update also passed.

Pilot elapsed times were 2,538.6 s (control) and 2,461.4 s (detached) for 25k updates.
Scaling the remaining 725k gives **20.45 h / 19.83 h**. Allow approximately
**20–24 h per model**, **40–48 GPU-hours combined**, or **20–24 h elapsed when
parallel**, plus scheduling delays between stages. Full validation previously
took about 10 minutes per checkpoint: four tasks at concurrency two add roughly
20–30 minutes plus queue time. Later decoded graphs may cost more to evaluate.

Next review: inspect learning/validation trajectories, image sensitivity, curve
gain over identical-endpoint chords, curved-branch strata and overfitting. Retain
best and final evidence even if the last checkpoint is worse. No automatic new
training phase beyond these authorized continuations.

## Added epoch-100 and epoch-200 controls

User requested intermediate comparisons to avoid waiting for all 300 epochs.
The running code already computes decoded validation graphs, same-endpoint chord
metrics, fixed image-sensitivity probes, and train/validation matched geometry
at both milestones (100 and 200 are multiples of the 20-epoch detailed interval).
Reuse these tests on 512 fixed training and 128 fixed validation cases, without
restarting either GPU job or running redundant inference.

`scripts/watch_scratch_milestones.py` polls the existing history every five minutes
and creates paired CSV/JSON/Markdown tables and a portable HTML projection gallery
when both exact-epoch rows are available. The JSON includes original diagnostic
rows and curvature strata. Dataset and evaluation memberships must match.
Results appear under the run root at:

- `milestone_checks/epoch100/comparison.{json,csv,md,html}`
- `milestone_checks/epoch200/comparison.{json,csv,md,html}`

Collapse screening flags transformer memory MAE below 0.001 **and** below 1% of
that arm's epoch-10 value. This is a review flag, not proof of an internal mechanism
or an automatic cancellation rule. Graph detection, token/node sensitivity and
geometry evidence are reported alongside it. Training can continue while reports
are generated; the existing 300-epoch authorization and dependent evaluations
remain active.

Background user service: `gnbm-scratch-milestones-20261012.service`, one standard
library Python process, no reserved GPU, exits after both reports or a 48-hour
deadline. Inspect with `systemctl --user status gnbm-scratch-milestones-20261012.service`.
Reproduce one comparison with `python3 scripts/watch_scratch_milestones.py --once`.

Lightweight verification used actual saved epoch-20 data: paired reports and
embedded images generated successfully in a temporary directory and were deleted
after inspection. Both intact runs passed the collapse screen; the actual saved
collapsed coupled run triggered it. Service verified active.

At the initial progress check, epoch-30 validation edge F1 was 0.36656 / 0.42729
(control/detached), with memory sensitivity 0.95084 / 0.91634. Neither was collapsed
at that checkpoint. These are early validation results, not final performance.

### Progress review: latest common checkpoint epoch 50

Both first-stage jobs still running after 2:56. Control completed monitoring at
epoch 55; detached at epoch 50. At the **same epoch 50**, control/detached node F1
is 0.57319/0.60888, edge F1 0.44457/0.48490, and decoded curve F1 0.14172/0.19589.
Detached same-endpoint chords score 0.16501: curved decoding adds 0.03088 curve F1
at fixed endpoints/connectivity. Image sensitivity is 0.93125/0.90614, so no sign
of the earlier collapse. Latest matched shape assessment at epoch 40 shows
endpoint-neutral validation improvement of 8.97% versus chords (train 17.14%).
Common branch F1 remains very low (0.00514/0.00316 at epoch 50); improving native
edges/curve overlap does not yet establish good complete branch reconstruction.
Continue the authorized training and epoch-100/200 controls. Single-seed results;
detector differences cannot be attributed directly to detached geometry feedback.

### Progress review: latest common checkpoint epoch 75

Both first-stage tasks running after 4:32. Latest monitored epochs: control 85,
detached 75. At equal epoch 75, control/detached node F1 is 0.60290/0.62214,
edge F1 0.47101/0.49738, curve F1 0.17166/0.22368. Detached curves improve over
its identical-endpoint chords (0.17938) by 0.04431 curve F1. Transformer memory
sensitivity 0.91138/0.86518 remains far above the collapse range. Complete branch
F1 remains low (0.00609/0.00377). Latest detached matched evaluation at epoch 60:
endpoint-neutral improvement 25.22% train / 8.44% validation, versus 17.14%/8.97%
at epoch 40. Increasing training gain has not translated into comparable validation
shape improvement; watch overfitting at the planned milestones. Epoch-100/200
paired reports are not ready yet. No further tests or configuration changes.

### Epoch-100 paired control completed

Both jobs running after 6:36; latest monitored epochs control 120 / detached 105.
The automatic epoch-100 paired report is ready. Control/detached node F1
0.59727/0.63882, edge F1 0.49325/0.51777, curve F1 0.17383/0.26381; detached
same-endpoint chord curve F1 0.19868. Both collapse screens negative, memory
sensitivity 0.90688/0.86292. Detached endpoint-neutral validation shape error
improves 7.27% overall and **28.10% on 193 curved long branches** versus chords.
Short-branch neutral shape is worse (0.11989 vs 0.06691 mm); overall training
improvement is 36.75%, so generalization remains limited. Decoded complete-branch
F1 is still low (0.00370/0.00687). Continue toward the authorized epoch-200 review;
no extra diagnostic jobs required. Tables and fixed visualizations are in
`milestone_checks/epoch100/` under the run root.

### Epoch-150 stage completed; continuation awaits GPUs

949063_0/1 completed successfully (8:22:44 control / 9:28:02 detached).
949065_0/1 remain queued with unavailable-node/maintenance reasons; full validation
949071 remains dependent. At epoch 150, control/detached node F1 0.61894/0.64696,
edge F1 0.49602/0.54236, curve F1 0.19195/0.30910. Detached same-endpoint chords
score 0.21559. Memory sensitivity 0.94884/0.85539: neither shows encoder collapse.
Detached validation neutral shape improvement is 2.70% overall and 25.63% for
curved long branches; training improvement is 45.48%. The growing train/validation
gap and declining overall validation neutral gain indicate overfitting despite
better decoded curve overlap. Complete branch F1 stays low (0.00358/0.00845).
The planned epoch-200 comparison is pending the continuation; no new experiments
or configuration changes were made during this status review.

### Continuation time limit reduced at user request

Changed queued continuation array **949065** from 20 hours to **12 hours** with
`scontrol update JobId=949065 TimeLimit=12:00:00`. This changes the scheduler
allocation only; epoch target, training settings, checkpoints and dependent
evaluations are unchanged. The measured remaining runtime is approximately
9–11 hours per arm. Maintenance is reserved for 2026-10-13 08:30–14:30; the
shorter limit permits scheduling before it if suitable GPUs become available.
The Slurm script retains its original 20-hour default for future submissions;
this update applies to the already submitted continuation array.

### 2026-10-13 00:11 — Epoch-200 controls ready; continuation running

The 12-hour limit allowed both continuation tasks to start before maintenance;
949065_0/1 are running after 4:45. Latest monitored epochs: control 225 / detached
230. Both exact epoch-200 comparisons were generated automatically. At epoch 200,
control/detached node F1 0.62427/0.65098, edge F1 0.50940/0.54655, decoded curve
F1 0.19661/0.34230; detached same-endpoint chords 0.22936. Neither collapse screen
fires; memory sensitivity is 0.93479/0.87127. Detached endpoint-neutral validation
shape gain is now **-1.36% overall**, versus +50.70% on training; curved long
validation branches still improve 22.46%. Thus curve overlap improves but shape
generalization/overfitting remains an unresolved limitation. Complete branch F1
is low (0.00532/0.00636). The latest epoch-220 matched evaluation confirms the
same pattern (neutral validation -1.84%, curved long +22.16%). Full validation
awaits both final stages. At measured stage speeds, roughly 4–5 hours remain,
which should fit before the 08:30 maintenance if there is no interruption.

### Scheduled final review and publication

At explicit user request, enabled a one-shot morning review at **2026-10-13 07:00
Europe/Paris**, with **15:00** as a fallback if training or evaluation is still
pending. User timer `gnbm-spline-results-20261013.timer` invokes a bounded Codex
review of this session; it does not keep an agent alive waiting for the results.
The original daily-ping timer remains separate.

The review checks both final trainings and all four full-validation evaluations,
updates this report and the research journal with actual numbers, discoveries,
limitations and next steps, saves a tracked report under
`docs/research/spline_learning/scratch-extension-300.md` in the code repo,
commits only relevant stable changes, and pushes `representation` to `origin`.
User explicitly authorized commit/push. Pending results are recorded honestly;
the fallback handles a later review. No new GPU experiments are part of this task.
The wrapper skips the fallback only after complete result artifacts and a verified
matching local/remote branch head are present.

Reproducible timer/service/wrapper/prompt are in `cluster/jean_zay/` in the code
repo. CLI uses workspace sandboxing, automatic approval review and network access
for the authorized push. Logs and final reply are under
`experiments/spline_learning/logs/morning-results-<timestamp>.{log,md}`;
successful complete publication creates `morning-results-20261013.done` there.

Timer/service syntax and the existing resumable session were verified; Codex is
authenticated. HTTPS push lacked credentials, so configured origin's push URL
to use the existing verified GitHub SSH identity. The initial stable branch/report
publication succeeded (`representation` now tracks `origin/representation`). The
morning completion marker also requires a committed final-results evidence JSON,
so a previously pushed interim report cannot count as completed publication.

## Final results — 2026-10-13

Both runs completed **300 total epochs / 750,000 updates**, with exit code 0.
Control continuation ended **04:36:27**, detached B-spline **04:04:02** Paris time;
all four full-validation jobs completed by **04:54:07**, before 08:30 maintenance.
Final continuation elapsed 9:10:02 / 8:37:37. Together with the initial 25k pilots
and epoch 150 stages, total allocated training was approximately 18:20 / 18:51 per
arm (single GPU each; approximately 37:11 GPU-hours combined, excluding smokes
and validation). Reducing the continuation allocation to 12 hours allowed both
jobs to run before the reservation. No timeout, numerical failure or encoder
image-sensitivity collapse occurred. Final memory MAE is 0.95624 /0.86923.

### Paired full validation: all 1,952 patches

Same ordered samples, input/configuration/evaluation protocol and six fixed
visualization cases; all 31,934 Hungarian-matched GT-positive branches evaluated.
Saved BN statistics, no recalibration; weights/buffers unchanged by evaluation.
Actual epoch 150 and epoch 300 training batch-order hashes match across arms.
These are validation results; the held-out test set was not used.

| model | epoch | node_f 1 | edge_f 1 | branch_f 1 | curve_f 1 | chord_curve_f 1 | endpoint_neutral_gain |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_control_final | 300 | 0.63383 | 0.51851 | 0.00793 | 0.20201 | 0.20201 | -0.00000 |
| graph_control_best | 255.00000 | 0.63109 | 0.51635 | 0.00727 | 0.20058 | 0.20058 | -0.00000 |
| detached_spline_final | 300 | 0.64995 | 0.54195 | 0.01179 | 0.34261 | 0.22633 | -0.04316 |
| detached_spline_best | 170.00000 | 0.64661 | 0.54407 | 0.01057 | 0.32358 | 0.22413 | 0.02370 |

Positive endpoint-neutral gain means shape error improved over chords after
transferring predicted curvature to GT endpoints (evaluation only). This measure
separates shape from localization; decoded curve F1 evaluates complete predicted
geometry with its predicted nodes/connectivity. They answer different questions.

### Discoveries and limits

1. **Scratch learning is possible with detached geometry.** The previous coupled
   token-supervision run collapsed, while this isolated residual/local-image
   setup retained graph learning and image sensitivity for 300 epochs. This supports
   harmful early geometry feedback as a cause under the tested setup, without
   identifying its exact mechanism. BN recalibration is not the explanation
   supported by the existing diagnostic. A single seed is not proof of robustness.
2. **Curves improve decoded vessel overlap:** final B-spline curve F1 0.34261 versus
   its same-endpoint chords 0.22633 (absolute gain 0.11628). Final graph-only curve
   F1 0.20201. Curvature comparisons within one model hold nodes and edges fixed.
   Native node/edge differences between models cannot be attributed causally to
   geometry, because geometry gradients are detached from the detector.
3. **Geometry overfits.** Final endpoint-neutral shape error on full validation
   is 0.53715 mm versus chord 0.51493 mm (**4.32% worse**), despite a 57.64% improvement
   on the 512-case training monitor. The selected spline best checkpoint (epoch 170)
   retains a modest full-validation neutral gain 2.37%, whereas its final checkpoint
   has higher decoded curve F1. More epochs did not uniformly improve shape.
4. On the **128-case monitoring subset only**, the final model improves neutral
   shape on 193 curved long branches by 22.00% (1.30356 vs 1.67118 mm); near-straight
   long branches worsen 11.36%, and 550 short branches worsen 117.36%
   (0.14544 vs 0.06691 mm). Short branches and subtle bends dominate the aggregate
   weakness. Full-validation curvature strata were not enabled in this evaluation
   configuration; do not present the 128-case percentages as full-population results.
5. **Topology is not solved:** final common branch F1 is 0.00793/0.01179. The models
   predict approximately 10.91/10.92 common branches per patch versus 16.36 targets,
   and Betti errors remain substantial. Better native edges and curve overlap
   do not establish accurate complete vessel graphs. Native node boxes and strict
   spatial branch/anchor correspondence use different matching criteria.
6. **Best-checkpoint selection is noisy and objective dependent.** Selected control
   epoch 255 and spline epoch 170 maximize 128-case common branch F1, then native
   edge F1. These choices do not maximize every full-validation metric; selected
   spline best has edge F1 0.54407 vs final 0.54195, but curve F1 0.32358 vs 0.34261.
   Retain both outcomes; do not relabel a different checkpoint as best after
   inspecting the full validation or test set.

### Next research decisions

- Inspect the gap between native detection and common complete-branch recovery
  with the existing fixed cases and exact metric/spacing/matching definitions,
  including boundary, loop and parallel-branch rules, before claiming topology
  success or changing the architecture.
- Address shape generalization and short-branch behavior in a bounded comparison;
  choose geometry checkpoints using an explicit validation geometry objective.
  Any chord fallback/regularization is a proposal requiring a controlled experiment,
  not a modification made during this review. Preserve successful detector training.
- After settling selection and evaluation, use a second seed and the held-out
  test set for a paper-quality comparison. No additional GPU jobs launched here.

### Artifacts and publication

Full tables and portable six-case HTML gallery: `full_validation/comparison.{csv,md,json,html}`
under the run root. Best/final models: `<arm>/epoch 300/{best.pt,model.pt}`.
Final compact evidence is tracked at
`experiments/spline_learning/metadata/local-geometry/scratch-extension-final-results.json`;
all source summary hashes and the paired audit are recorded there.

The 07:00 scheduled reviewer failed immediately because Codex rejects combining
`--approve-for-me` with explicit`--sandbox`. No model or report ran in that attempt.
Removed the redundant sandbox flag (automatic approval review already selects
workspace-write), and completed the authorized documentation/commit/push in this
active review. No approval/sandbox bypass was introduced. The 15:00 fallback will
skip once the published-completion marker is verified.
