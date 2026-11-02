# Fine-feature architecture study execution plan — 24 October 2026

Status: **completed, 25 October 2026**. The baseline finished100 epochs/78,200 updates; predefined selection chose epoch95. Baseline and selected-checkpoint architecture gates passed. All four3000-update pilots, historical evaluation, image interventions and native/common reconstruction evaluations completed. Collector1522540 and final CPU report generation passed. Accounting confirms at most two concurrent H100s (7.903 allocated GPU-hours). Final evidence and assessment are in `docs/research/fine_feature_architecture_study.md`; job identities, checkpoint/config/source hashes and raw accounting are stored alongside this plan. A merits a longer controlled comparison; C has an active fine pathway but no demonstrated gain over continuation. No longer architectural run or A+C model was launched.

## Fixed research protocol

1. Audit the immediate unweighted node focal recipe (`gamma=2`, no curriculum),
   matched-node edge CE and ratio upsampling 0.15 from
   `configs/experiments/full_dataset_node_focal/` and the actual criterion.
   Do not inherit `configs/losses/focal.yaml`: that file also enables edge focal.
2. Reuse all 25,000 prepared SyntheticMRI K8 training patches and the existing
   480 validation patches / 13 subjects. Audit subject separation using original
   patient IDs. No test data. No new target generation or query population.
3. Train one random-initialized model, seed 364505, all active parameters unfrozen,
   batch 32, AdamW 7e-5, one epoch warmup, polynomial exponent 0.9, 100 epochs.
   Keep original matcher, graph losses/weights, 120 queries, K8 local spline head
   and established detached geometry-to-detector route (`graph_only`). Geometry
   head remains trainable. Rotations/noise enabled, spline-incompatible zoom off.
   This is a full 100-epoch scratch experiment, not the previous 4096-patch pilot.
4. Before full training: CPU loss/spatial/migration tests, actual random-model
   forward/backward and eight-patch learnability. Require node F1 >=0.95 and edge
   F1 >=0.90 at 2 mm after at most 1500 tiny updates; otherwise diagnose, do not
   release baseline training. Tiny weights are discarded.
5. Checkpoint selection is predefined: maximum full-validation physical edge F1
   at 2 mm, then node F1 at 1 mm, evaluated every five epochs. This avoids loss
   selection and a weak branch metric dominating node/connectivity selection.
   Keep native metrics and common-v2 branch/curve metrics for the selected model
   and compatible historical source; do not compare incompatible old native
   scores against the new physical scores. Preserve best and resumable latest.
6. Quality gate, on the selected checkpoint: finite metrics, raw recall >=0.82,
   filtered node F1 >=0.70, edge F1 >=0.40, conditional edge recall >=0.70 at 2 mm,
   Hungarian mean distance <=1.8 mm. These are functional-baseline floors, not
   architecture-success criteria. Gate JSON plus nonzero exit on failure blocks
   all dependent training. No automatic extension on failure.
7. While baseline trains, prepare independent A and C and their correctness
   checks. A reuses the 32³ early-CNN 27-sample local refinement with identity
   output initialization and live feature gradients. A-control zeros local image
   inputs with the same residual head and parameter count; its image-input weights
   are inactive, so inference interventions also test image-information use.
   Sampling centers stay detached as in the established head.
8. C uses backbone layer2 (512 channels, 8³) rather than layer3 (1024, 4³), with
   the original coarse 2³ level. Copy all compatible weights and per-head coarse
   rows. New local offset initialization remains within one fine-grid cell.
   Use a fixed 500-update logit warm-in from ratio 1e-4 to 0.1 (9.09% nominal mass). The preflight found that immediately applying 0.1 displaced some proposals by 0.363 normalized units; this correction avoids that abrupt disruption without retaining a permanently suppressive prior.
   Active C is deliberately not constrained to exact baseline parity; quantify
   its initial perturbation and verify coarse-only parity. No new learned gate or queries. The warm-in offset is exactly zero at final inference.
9. After baseline gate and real-checkpoint architecture smoke, run B0/A/A-control/C
   for 3000 matched fully unfrozen updates, batch 32, shared fresh optimizer and
   short warmup/decay schedule with peak LR 7e-6. Same objective and deterministic
   sample/augmentation sequence. Compare final, not validation-picked pilot steps.
10. Evaluate raw/filtered one-to-one recall at 1/2 mm, localization, confidence
    discrimination, duplicate rates, relation/edge and graph reconstruction, and
    training/validation behavior. Paired subject uncertainty. A versus capacity
    control and image ablation; C fine-blocked and spatially shifted inference.
    Feature use requires prediction/metric effects, not attention/gradient alone.

## Scheduling and preservation

At most two H100s concurrently. Baseline may be split into dependent walltime
segments of the same run with full optimizer/scheduler/RNG recovery. Architectural
array concurrency is `%2`, after successful baseline and architecture gates.
All artifacts live under this new study directory; existing results are read-only.
No long architecture training, A+C combination, loss sweep or additional models.

Final report: `docs/research/fine_feature_architecture_study.md`.
