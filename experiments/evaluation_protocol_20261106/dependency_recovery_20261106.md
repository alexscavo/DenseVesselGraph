# Dependency recovery — 6 November 2026

Two pending jobs had `DependencyNeverSatisfied`: IXI traced stage 2 `2052011`
and TubeTK QC `2052268`. Both were replaced; the obsolete pending jobs were cancelled.

| Stage | Replacement job | Dependency |
|---|---|---|
| IXI-loops / dsc3_slim traced stage 2 | 2065823 | None: completed stage-1 artifacts verified |
| TubeTK preparation report collection | 2065819 | None: completed target cache verified |
| TubeTK QC | 2065820 | afterok:2065819 |

At submission verification, IXI and TubeTK collection were pending for Priority;
QC was pending on the live collection job. No replacement had an invalid dependency.

## IXI

`sacct -D -j 2052010` showed an original NODE_FAIL attempt followed by a failed
automatic restart. Before the node failure, the original attempt had saved the
stage-1 checkpoint, all 15,000 training updates, and final validation results.
The restart failed because repeated runtime setup followed an existing directory
symlink and tried to create a nested link inside the protected SymPy package.
`ixi_mechanism_runtime.sh` now uses `ln -sfnT`, replacing the link itself.

Verification: the original `model.pt` loaded on CPU, all model tensors were finite,
checkpoint arguments matched the original recipe, the training log contained
steps 100 through 15,000, and final validation covered all 2,961 patches.
The checkpoint at
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/dense_graph_20261030/IXI-loops/spline6_dsc3_slim_r1106/model.pt`
was therefore reused. The stage-2 command and training settings are unchanged.
A temporary-directory check reproduced nested package links before the fix and
confirmed that sourcing the fixed setup twice leaves package directories untouched.

## TubeTK

`2052267` completed target generation, then failed in `collect()` because the
compute node could not find `git`. The READY marker, manifest and patch index
agree on 12,395 train / 2,514 validation / 2,775 test patches.

The collector now accepts `TUBETK_GIT_HEAD` from the submitting environment. If Git
is unavailable and no revision was supplied, it explicitly records an unknown
revision and retains the preparation script's SHA256 instead of crashing.
Recovery supplies the actual submitting revision. A new `STEP=collect` runs only
report collection, avoiding regeneration of the existing targets.
Syntax checks and the missing-Git provenance checks passed.

Exact replacement commands and cancellation records are in
`dependency_recovery_20261106.json`. No dataset, split, training objective, budget,
or evaluation setting was changed. Full replacement jobs were still pending at
the verification above; their completion is not claimed here.
