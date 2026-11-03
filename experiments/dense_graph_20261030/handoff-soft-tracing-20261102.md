# Handoff: soft-centreline tracing runs (2 Nov 2026)

Owner: Alessandro Scavone. Repo: `projects/DenseVesselGraph`, branch `representation`.
Code: commit `15523ff`, `scripts/dense_graph_traced_20261030.py`, which adds three opt-in flags.
The defaults reproduce `traced_adaptive`.
- `--soft-tracing`: no hard p ≥ 0.5 centreline mask; ink may cross any voxel at cost 1/(p+0.05).
- `--no-knn-fallback`: candidates are geodesic contacts only, with no 16-NN straight pairs.
- `--gap-features`: the edge and geometry heads also read the centreline probability along the path
  (mean, min, fraction < 0.5, longest run < 0.5, mean log cost).

Both smoke tests passed on 2 Nov:
- 1864674, BRAVA, all three flags: 1.12 s/update, traced fraction about 1.0.
- 1864675, IXI, `--gap-features`: 0.90 s/update, traced fraction about 0.13.

## Jobs to watch

All jobs use the same recipe as `traced_adaptive`: warm start from `<ds>/spline6/model.pt`,
`--updates 7500 --lr 3e-4 --warmup-gt 0`, one H100 on t3. The launcher is
`cluster/jean_zay/dense_graph_traced.slurm`.

| Job | Dataset | Run name | Flags | Walltime |
|---|---|---|---|---|
| 1872436 | SyntheticMRI | traced_soft | soft + no-fallback + gap | 3:30 |
| 1872437 | SyntheticMRI | traced_gap | gap | 3:00 |
| 1872438 | IXI | traced_soft | soft + no-fallback + gap | 3:30 |
| 1872439 | IXI | traced_gap | gap | 3:00 |
| 1872440 | BRAVA | traced_soft | soft + no-fallback + gap | 3:30 |
| 1872441 | BRAVA | traced_gap | gap | 3:00 |

Outputs: `$SCRATCH/experiments/gnbm/dense_graph_20261030/<ds>/<run>/`, containing `final.json`,
`final-validation-rows.pkl`, `training.jsonl`, `model.pt` and `warm-start.json`.
Logs: `DenseVesselGraph/experiments/dense_graph_20261030/logs/dt-<ds>-{soft,gap}-<jobid>.{out,err}`.
`sacct` is ground truth.

## What to do when they finish

1. **Watch.** Use a background `sacct` watcher on the six job IDs.
   - On FAILED or TIMEOUT: read the `.err` log and report the cause.
   - Resubmit once, only for an obvious transient or walltime issue: 1 GPU, walltime 1.3× the
     measured need, no `afterany` chains.
   - Do not change the method.
2. **Paired comparison** per dataset, against the existing `traced_adaptive`:
   - `traced_gap` vs `traced_adaptive`: the effect of the gap features alone.
   - `traced_soft` vs `traced_gap`: the effect of the soft domain plus dropping the fallback.
   - Generalise `scripts/compare_traced_knn_20261030.py`: take the two arms from argv, keep
     `ARMS` as the default, and keep the method identical (subject bootstrap, 2000 resamples, seed
     261030; thresholds from each arm's `final.json`, selected on `monitor_train`).
   - Report node / edge / Branch / Curve F1 with paired CIs.
   - Also report `traced_pair_fraction` and the arc-length breakdown (`length_breakdown` in
     `final.json`, where present). The key question: does long-edge recovery (≥ 20 mm and ≥ 40 mm)
     improve on BRAVA and IXI?
3. **Topology.** Extend `scripts/topology_metrics_20261102.py` to include the new arms: β₀/β₁
   errors at the train-selected thresholds.
4. **Document.**
   - In `docs/research/dense_traced_model_20261102.md`, add a short subsection under §4 or §6
     labelled DONE, with a table and plain caveats: single seed, validation only, warm-started
     from the same `spline6`.
   - Do not rewrite earlier results.
   - Commit locally as Alessandro Scavone `<scavoale@gmail.com>` (check `git config` first).
     Do not push.
5. **Report back.** Append a status summary for the user at the end of this file.

Scope: exactly this. No new architectures, sweeps or test-split evaluation.

## Status summary (DONE, 3 Nov 2026)

All six jobs COMPLETED on the first attempt, with no resubmissions: `traced_gap` 1:42–1:49,
`traced_soft` 2:27–2:56. Results are in §4.6 of `docs/research/dense_traced_model_20261102.md`.

**Key question: does long-edge recovery improve on BRAVA and IXI? No.** No ≥ 20 mm or ≥ 40 mm
comparison on either dataset shows a gain: every CI spans zero, except IXI ≥ 20 mm for
`traced_soft` vs `traced_gap`, which is worse.

| Dataset | Arc length | `traced_gap` − `traced_adaptive` | `traced_soft` − `traced_gap` |
|---|---|---|---|
| IXI | ≥ 20 mm | 0.000 [−0.024, +0.023] | −0.060 [−0.089, −0.031] |
| IXI | ≥ 40 mm | −0.036 [−0.086, +0.014] | +0.012 [−0.052, +0.071] |
| BRAVA | ≥ 20 mm | −0.001 [−0.008, +0.007] | −0.001 [−0.007, +0.005] |
| BRAVA | ≥ 40 mm | −0.002 [−0.011, +0.010] | +0.008 [−0.003, +0.020] |

Missing endpoints remain the main limit on BRAVA: 79–80% of edges ≥ 20 mm in all three arms.

Other findings:
- **`traced_gap`:** no effect on SyntheticMRI or BRAVA. Small IXI gain: edge F1 +0.010, Branch F1
  +0.010, Curve F1 +0.012, all with CIs excluding zero. Its IXI node threshold also differs (0.30
  vs 0.35), which confounds that gain.
- **`traced_soft` vs `traced_gap`:**
  - BRAVA gains Curve F1 +0.025 and Branch F1 +0.009, and loses edge F1 −0.003.
  - IXI loses Curve F1 −0.084, edge F1 −0.019 and Branch F1 −0.017.
  - IXI loops fall from 3.25 to 0.86 per patch (GT 0.71), with edges per patch down from 16.9 to
    10.9.

Caveats: single seed, validation only, warm-started from the same `spline6`. The CIs cover subject
sampling only and are uncorrected for multiple comparisons.

Corrections to this handoff:
- The arc-length data in `final.json` is `validation_by_arc_length`, not `length_breakdown`.
- `traced_pair_fraction` is only in `training.jsonl`. `final.json` has
  `validation_candidates.traced_fraction`.
- `traced_adaptive` was not rerun with the new code, so "the defaults reproduce it" is unverified.

Addition beyond the list above: `scripts/compare_arc_length_20261102.py`, which gives the
subject-paired long-edge CIs.

Commits are local on `representation` and not pushed.
