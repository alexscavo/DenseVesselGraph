# Longer stability continuation after local-geometry unfreezing

User authorized full validation, selection and the longer continuation together.
This follows the successful frozen local-image geometry experiment and three
short controlled-unfreezing runs. Remain on SyntheticMRI.

## Selection fixed before full-validation results

Evaluate graph_only, tokens_only and joint on all 1,952 existing validation
patches, using the same protocols and fixed six visualization cases. The frozen
local-image checkpoint's completed full validation is the reference.

A candidate must retain curve F1 and node F1 within 0.01 absolute of the reference,
and endpoint-neutral matched geometry error within 1.05 times the reference.
Among eligible candidates choose highest common branch F1, breaking ties by
curve F1 then native edge F1. These are pragmatic retention tolerances, not
statistical significance bounds. Branch F1 prioritizes recovered connectivity;
the separate guards retain the geometry benefit and node detection. Other
metrics, including Betti errors, remain reported rather than hidden in a score.
If none passes, selection raises before any continuation training.

The selected route and every candidate's values are recorded in selection.json.
This is validation-based model selection, not an unbiased test claim.

## Longer continuation

- Load the selected 5k-unfreezing checkpoint: 25,000 total geometry updates so far.
- Resume both model and AdamW state; no optimizer reset. Existing older checkpoint
  metadata provides cumulative updates as source updates plus completed steps.
- Another 25,000 updates, batch 4 on the same 10k patches: 10 more dataset passes.
- Same constant detector LR 7e-6, spline/local head LR 7e-5, residual objective,
  weight 1.4739989612791453 and selected gradient route.
- Same CNN/BN freeze and eval-mode/dropout policy. No additional unfreezing.
- Shuffle stream restarts with seed 364505; this is an explicitly restarted data
  order, not an exact uninterrupted RNG/data-loader resume.
- Monitor the same 128 validation and 512 training cases. Loss and sensitivity
  every 1k updates; decoded and matched geometry every 5k, including curvature
  strata, node/edge/branch metrics and fixed images.
- Save intermediate model+optimizer snapshots at 5k and 15k; final at 25k.
  Count/digest/finite-gradient checks remain active. Offline W&B records retained.
- First run a 3-update GPU smoke from the selected checkpoint with optimizer
  restoration. The full continuation starts afresh from that source afterward.

After success, evaluate the final checkpoint on all 1,952 validation patches.
Reuse the already completed selected-baseline evaluation through a read-only
symlink in the final comparison directory. No duplicate baseline inference is
needed. Generate before/after tables and a portable six-case projection gallery.
No test samples enter selection or training.

This is a bounded longer-stability experiment, not a 300-epoch training claim.
The full validation comparison, intermediate trajectories and images are needed
before deciding whether further training or additional unfreezing is justified.
No automatic further experiment follows this pipeline.

## Reproducibility and outputs

- `configs/spline_local_unfreeze_full_validation.yaml`: three-candidate evaluation.
- `configs/spline_local_stability.yaml`: fixed selection and extension settings.
- `scripts/select_spline_stability.py`: deterministic selection and resolved configs.
- `scripts/controlled_spline_unfreeze.py`: existing training, with opt-in optimizer
  restoration and intermediate checkpoints; prior runs retain default behavior.
- `cluster/jean_zay/spline_local_stability.slurm`: selection, smoke, continuation.

Outputs under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/runs/`:

- `local_unfreeze_full_validation_20261010/`: candidate evaluation and gallery.
- `local_stability_20261010/selection.json`: chosen route, rule and candidate values.
- `local_stability_20261010/training/<selected>/`: continuation and monitoring.
- `local_stability_20261010/full-validation/comparison.html`: final portable gallery.

The Slurm dependency chain enforces full-validation completion before selection
and successful training before final evaluation. Selection, smoke or training
failure prevents later expensive stages. Existing results remain untouched.

Submitted candidate evaluation **876609_0/1/2**, selection/continuation **876621** (`afterok:876609`), final evaluation **876622_1** (`afterok:876621`). Five targeted CPU tests passed, with Python/Slurm syntax and diff checks. GPU smokes remain queued with each stage. Estimated pipeline wall time excluding queue: roughly 1–1.5 hours; continuation has a two-hour ceiling. Selection is pending full validation.

## Startup failure and correction

Candidate evaluations 876609_0/1/2 completed (9:12–9:39). Continuation 876621 failed after six seconds while staging runtime packages: `mkdir: cannot create directory /lustre/fsn1/jobscratch/upz25mj_876621: Permission denied`. Selection and training never started. Removed blocked final evaluation 876622 and resubmitted continuation as 877101. The two launch scripts now unset JOBSCRATCH before environment setup, using existing node-local /tmp runtime and temporary-file defaults. Syntax/diff checks passed; existing three-update GPU smoke still gates full training. Scientific settings unchanged.

Full validation: graph_only branch F1 0.292610 / curve F1 0.843311; tokens_only 0.290941 / 0.842957; joint 0.291344 / 0.843841. All satisfy the predeclared geometry/node guards. Thus graph_only is selected by branch F1, with small differences and no significance claim. This route continues training the graph detector on graph losses while adapting the local spline head through detached detector inputs.

Replacement final evaluation: **877103_1**, dependency `afterok:877101`.

## Completed stability result

Continuation **877101** completed 25,000 updates in **51:28**; final evaluation **877103_1** completed in **9:04**, both exit 0:0. CNN/buffers remained fixed, and final evaluation did not change model state.

Full validation (1,952 patches):

| Metric | Before extension | After extension |
|---|---:|---:|
| node_f1 | 0.909016 | 0.909413 |
| edge_f1 | 0.872307 | 0.874605 |
| branch_f1 | 0.292610 | 0.281353 |
| curve_f1 | 0.843311 | 0.844922 |
| endpoint_neutral_mm | 0.352656 | 0.343019 |
| matched_guide_mm | 0.730113 | 0.731221 |
| betti0_error | 0.871926 | 0.981557 |
| betti1_error | 1.024078 | 0.793545 |

The local-geometry gain persists without transformer collapse. However, branch F1 decreases by 0.01126 and Betti0 error worsens, while native edge F1, endpoint-neutral geometry and Betti1 error improve. Raw matched guide error is essentially unchanged/slightly worse. Thus longer training is stable in the sense of no collapse, but is not an overall connectivity improvement; do not automatically replace the earlier checkpoint.

The selected graph_only route lets geometry adapt to a detector trained on graph losses, with geometry inputs detached. This does not establish stability of fully coupled geometry gradients or learning from scratch. Full validation remains used for selection, not an unbiased test claim.

Diagnostic-subset trajectory (128 cases; do not compare its absolute values directly to the full-validation table):
```json
[
  {
    "update": 0,
    "edge": 0.8639218422889045,
    "branch": 0.27124505928853754,
    "curve": 0.8415910183767037,
    "shape": 0.34037185523866903
  },
  {
    "update": 5000,
    "edge": 0.8649151360148802,
    "branch": 0.2846589509972913,
    "curve": 0.8480525404848915,
    "shape": 0.3362903709066397
  },
  {
    "update": 10000,
    "edge": 0.8669467787114846,
    "branch": 0.2708692440285644,
    "curve": 0.8562594330041523,
    "shape": 0.3310616005180573
  },
  {
    "update": 15000,
    "edge": 0.8637627432808156,
    "branch": 0.26428046089727875,
    "curve": 0.8549350994839091,
    "shape": 0.33262914224068146
  },
  {
    "update": 20000,
    "edge": 0.8613514145429039,
    "branch": 0.2683229813664596,
    "curve": 0.8434963196998227,
    "shape": 0.3361952753456514
  },
  {
    "update": 25000,
    "edge": 0.8594264221908792,
    "branch": 0.26932668329177056,
    "curve": 0.8510535729395929,
    "shape": 0.33325880977568106
  }
]
```

Portable before/after gallery: `local_stability_20261010/full-validation/comparison.html`. All snapshots and original checkpoints preserved. No new jobs launched in this review. Next decision should address the controlled scratch comparison; preserve the earlier checkpoint as the better full-validation branch-recovery baseline.
