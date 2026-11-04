# Handoff: pass-through decision and U-KAN backbone (3 Nov 2026, evening)

From the VS Code session (projects-cb), which is closing. The user (Alessandro Scavone) asked the
tmux session to keep working on this without interruption. Read `projects/CLAUDE.md` first for
policies: one GPU per job, walltime about 1.3× the measured need, at most ~8 concurrent jobs, no
long serialized `afterany` chains, and smoke-test before production. Commits are made as
Alessandro Scavone <scavoale@gmail.com>. Do not push unless the user asks.

## Repository state

- **Repo:** `projects/DenseVesselGraph`.
- **Checked out:** `KanKan`, at `e442c29`.
- **Branch history:**
  - `KanKan` = `representation` (`20f32ee`) + the U-KAN commit.
  - `main` and `representation` are pushed only up to `2c81b86`.
  - `20f32ee` (pass-through) and `e442c29` (U-KAN) are local only.
- **Working tree:** the jobs below run from the KanKan working tree. `KanKan` contains all the
  pass-through code; the backbone default is `unet`, so pass-through runs are unaffected.
- **Results:** `$SCRATCH/experiments/gnbm/dense_graph_20261030/<dataset>/<run>/`.
  Logs: `experiments/dense_graph_20261030/logs/`. `sacct` is ground truth.

## Context, briefly

- The diagnostics of 3 Nov are in commits `3271d85`, `2c81b86` and `experiments/dense_graph_20261030/*.json`.
- **Edges missed although both endpoints are found** are mainly blocked by *spurious* in-between
  detections C. The predicted centreline is not broken.
- **Two-hop candidates** (`traced_twohop*`) propose those A–B edges but don't get them accepted:
  blocked-edge recovery is unchanged at equal thresholds. They also create triangles
  (IXI 0.75–1.26 per patch).
- **Neighbourhood offsets** (`--offset-radius 1.75`) reduce node error on all datasets, so they are
  now the DEFAULT. Use `--offset-radius 0` to reproduce runs before 3 Nov.
- **`--passthrough`** (new, commit `20f32ee`, implies `--two-hop`) is the node-level decision:
  - A pass-through head scores each traced chain A–C–B, using the joined-path features plus the U-Net
    features at C.
  - At inference, chains are visited by decreasing score. If score ≥ τ and C has no other accepted
    edge, C, A–C and C–B are replaced by A–B.
  - Two-hop rows are never ordinary edges, so no triangles are possible.
  - τ is selected on `monitor_train` after the node and edge thresholds.
  - `final.json` → `passthrough` holds τ, the validation without contraction, and contractions per patch.
  - The unit test passed (`tests/test_traced_two_hop_offsets.py`, job 1908329).
- **U-KAN** (commit `e442c29`, `models/ukan3d.py`):
  - `--backbone ukan` replaces only `down_3`, `down_4`, `upcat_4` and `upcat_3` of the 3D BasicUNet with
    tokenized KAN blocks, following U-KAN (arXiv 2406.02918; github CUHK-AIM-Group/U-KAN `Seg_UKAN/archs.py`,
    `kan.py`). Settings: grid 5, cubic, SiLU base, 3 KAN layers + depthwise conv/BN/ReLU, pre-norm
    residual, concatenated skips.
  - `ukan_mlp` is the same layout with `nn.Linear` (the KAN-vs-MLP control).
  - `ukan_add` uses additive skips.
  - Parameter counts: 5.82M vs 5.75M for the U-Net. The unit test passed earlier (job 1910484).

## Decisions the user made in Q&A (keep to them)

1. **Main experiment:** simply replace the U-Net with U-KAN and change nothing else. Use the U-KAN
   layout (the 2 deepest encoder stages and the first 2 decoder stages).
2. **Arms:** U-KAN and the MLP control only. NO KAN-heads arm; that code was removed.
3. **Pipeline:** the full two stages, replicating the existing pipeline so results compare
   directly with `spline6` and `traced_adaptive`.
4. **Datasets:** all three.
5. **Hyperparameters:** our optimiser, lr, schedule and budget (unchanged), with U-KAN's grid
   (5, cubic). If the KAN arm is unstable, lower lr for the KAN arm only and report it.
6. **Skips:** concatenate. If training fails or the KAN arm clearly misbehaves, try `--backbone ukan_add`.
7. **Priority:** the pass-through jobs first, then the KAN study as slots free (≤ ~8 concurrent).

## Jobs pending at handoff (all smoke/unit tests; none of them produce results to keep)

| Job | What | Check |
|---|---|---|
| 1908330 | `dt-smoke-pass` (IXI `--passthrough`, run `smoke_passthrough`) | `final.json` has `passthrough.threshold`, finite `passthrough` loss in training.jsonl, s/update |
| 1908331 | `dt-smoke-off` (BRAVA, offsets only, run `smoke_offset`) | s/update |
| 1910486 | `dg-smoke-ukan` (IXI stage 1 `--backbone ukan`, run `smoke_ukan`) | finite losses, s/update |
| 1911067 | `dg-smoke-ukanmlp` (IXI `--backbone ukan_mlp`, run `smoke_ukan_mlp`) | same |
| 1911066 | `unit-ukan` (tests/test_ukan3d.py, CPU) | prints ALL TESTS PASSED |

Delete the `smoke_*` run folders on SCRATCH afterwards with `rm -rf "${O:?}"/IXI/smoke_...`.

## Production to submit once the smoke tests pass

Set `O=$SCRATCH/experiments/gnbm/dense_graph_20261030` and run from the repo root. Set walltimes
to 1.3× the measured s/update × updates, plus about 15 min for evaluation. Reference timings:
`traced_twohop` took 2:40, and `spline6` took about 55 min for 15k updates.

**A. Pass-through (6 jobs, first).** Same warm-start recipe as `traced_adaptive`:
```
for d in SyntheticMRI IXI BRAVA; do
  sbatch --job-name=dt-$d-off  --time=03:00:00 cluster/jean_zay/dense_graph_traced.slurm $d --updates 7500 --lr 3e-4 --warmup-gt 0 --init-from $O/$d/spline6/model.pt --run-name traced_offset
  sbatch --job-name=dt-$d-pass --time=05:00:00 cluster/jean_zay/dense_graph_traced.slurm $d --updates 7500 --lr 3e-4 --warmup-gt 0 --init-from $O/$d/spline6/model.pt --run-name traced_passthrough --passthrough
done
```
`traced_offset` is the new baseline: offsets only, no two-hop.

**B. U-KAN study (12 jobs, as slots free).** Stage 2 depends only on its own stage 1 (a single
`afterok`, not a chain):
```
for bb in ukan ukan_mlp; do for d in SyntheticMRI IXI BRAVA; do
  A=$(sbatch --parsable --job-name=dg-$d-$bb --time=<1.3x measured> cluster/jean_zay/dense_graph.slurm $d --mode spline --interior 6 --backbone $bb --run-name spline6_$bb)
  sbatch --dependency=afterok:$A --job-name=dt-$d-$bb --time=<1.3x> cluster/jean_zay/dense_graph_traced.slurm $d --updates 7500 --lr 3e-4 --warmup-gt 0 --offset-radius 0 --backbone $bb --init-from $O/$d/spline6_$bb/model.pt --run-name traced_adaptive_$bb
done; done
```
- **Recipes:** stage 1 uses the `dense_graph_20261030.py` defaults, the same as `spline6`: 15k updates,
  lr 1e-3, warm-up GT 2000, batch 8, seed 364505. Stage 2 uses `--offset-radius 0` to match `traced_adaptive`.
- **Concurrency:** to respect ≤ ~8 concurrent jobs, use a small background submitter that waits
  while ≥ 8 of the user's jobs are running/pending (ignore the ones held on Dependency).
- **The stage-1 reference** for `spline6` is `final-v2.json`, the re-evaluated protocol. Check that
  a new `final.json` from `dense_graph_20261030.py` uses the same evaluation as final-v2. If not,
  re-evaluate with `--evaluate-only` (it writes `final-v2.json`).

## Evaluation when they finish (CPU jobs, `cpu_p1`, `IDR_DEBUG=WARN`)

Templates: `cluster/jean_zay/twohop_analysis_cpu.slurm` and `tracing_blocking_cpu.slurm`.

- **Paired comparisons** (`scripts/compare_traced_knn_20261030.py <ds> <a> <b>` and
  `compare_arc_length_20261102.py <ds> <a> <b>`):
  - `traced_offset` vs `traced_adaptive`: the offset fix alone;
  - `traced_passthrough` vs `traced_offset`;
  - `traced_adaptive_ukan` vs `traced_adaptive`;
  - `traced_adaptive_ukan` vs `traced_adaptive_ukan_mlp`.
- **Stage 1:** `spline6_ukan` / `spline6_ukan_mlp` vs `spline6` on node F1 and node error. Add a
  stage-1 node-error / spurious-peak check by adapting `scripts/offset_refinement_check_20261103.py`
  or `twohop_analysis_20261103.py`, which already gives node-score distributions and node error.
- **Topology:** add the new arms to `arms()` in `scripts/topology_metrics_20261102.py` and run it with those arm names.
- **Pass-through visualizations** (the user explicitly asked to *see how the head works*):
  - Run `scripts/passthrough_figures_20261103.py` (CPU). It is written but not yet run, so fix it if needed.
  - It produces `docs/research/figures/passthrough_20261103/q_distributions.png` and `examples.png`,
    plus `experiments/dense_graph_20261030/passthrough-decisions.json` (contraction precision and recall).
  - Look at the figures before reporting them.
- **Write-up:** add DONE subsections to `docs/research/dense_traced_model_20261102.md` (pass-through;
  offsets) and a U-KAN section, with tables and plain caveats:
  - single seed, validation only;
  - each arm's own train-selected thresholds, with any threshold shift that confounds a delta called out.
- **Commit locally on `KanKan`**, as the user's identity. Report to the user what went where; do not push.

## Report back

Append a status summary for the user at the end of this file:
- what ran, and any resubmissions;
- the key results with CIs;
- links to the figures;
- what is still pending.

---

## Status summary (4 Nov 2026)

All of it finished. Full tables and caveats are in `docs/research/dense_traced_model_20261102.md`, §4.7 (offsets),
§4.8 (pass-through) and §4.9 (U-KAN). Everything is single seed and validation only; CIs are 95% over validation subjects.
Committed locally on `KanKan`, not pushed.

### What ran

- **Pass-through and offsets:** 6 jobs. `traced_offset` 1915387, 1915389, 1915391; `traced_passthrough` 1915388, 1915390, 1915392.
- **U-KAN:** 12 jobs, all first-try. `ukan` 1915466/7, 1915471/2, 1919441/2; `ukan_mlp` 1919521/3, 1919741/2, 1922701/2.
- **Failures:** IXI and BRAVA `traced_passthrough` (1915390, 1915392) crashed in final evaluation after finishing training, with
  `KeyError: 'contracted'` on patches with fewer than two detections. The smokes had no such patch. Fixed in `87c37b4`.
- **Recovery:** `--evaluate-only` from the saved `model.pt` (1922779, 1923079), not retraining. Validation rows are byte-identical
  to those the crashed jobs had saved. Unit test 1922778 passed.
- **Fallbacks:** training was stable, so the lower KAN lr and `ukan_add` were not used.
- **Additions beyond the list, all CPU analysis of saved rows:** two stage-2 smokes on U-KAN checkpoints;
  `scripts/passthrough_effect_20261104.py` and `scripts/compare_shared_threshold_20261104.py`.
- **Changes to the supplied figure script:** it ran, but the plots were misleading, so I changed them. The kept-node example was
  held by the guard and not by the head; a caption was hard-coded; density plots hid the class imbalance. I also added an
  audit of contractions after re-matching.

### Key results

**Pass-through, same model with contraction on − off (Edge F1):**
SyntheticMRI **+0.003 [+0.001, +0.006]**, IXI **+0.005 [+0.002, +0.008]**, BRAVA **+0.010 [+0.006, +0.013]**.
- It removes more false edges than true ones, but only 0.59 / 0.51 / 0.44 of the new A–B edges are correct in the final graph.
- On BRAVA 87% of contractions have C within 2 mm of an endpoint, so there the head mostly removes near-duplicates.
- Long edges do not improve. Against `traced_offset`, ≥ 40 mm: +0.021 [−0.014, +0.063], +0.012 [−0.027, +0.053], −0.004 [−0.019, +0.011].
- Blocked-edge recovery, pass-through − offset: SyntheticMRI +0.132 [−0.007, +0.265], IXI **−0.091 [−0.135, −0.044]**. On BRAVA, against
  `traced_adaptive` at equal thresholds, **−0.112 [−0.165, −0.073]**.
- IXI loops fall (2.82 → 1.44 per patch; GT 0.71), but that comes from training, not from the decision.

**Offsets, `traced_offset` − `traced_adaptive`:**
- SyntheticMRI, equal thresholds: Branch F1 **+0.086 [+0.067, +0.106]**. Median node error 0.65 → 0.57 mm.
- IXI, Curve F1: −0.009 at own thresholds becomes **+0.009 [+0.002, +0.016]** at a shared edge threshold (exact).
- BRAVA, Edge F1: **+0.031 [+0.025, +0.038]**, confounded by node threshold 0.25 vs 0.30; +0.003 [0.000, +0.007] at 0.30 (approximate).

**U-KAN, stage 2, Edge F1:**

| Dataset | U-KAN − U-Net | U-KAN − MLP control |
|---|---|---|
| SyntheticMRI | **−0.010 [−0.016, −0.003]** | **−0.005 [−0.011, 0.000]** |
| IXI, shared edge threshold | **−0.012 [−0.022, −0.002]** | **−0.014 [−0.025, −0.003]** |
| BRAVA, own thresholds (confounded) | **+0.035 [+0.030, +0.041]** | **+0.035 [+0.030, +0.040]** |
| BRAVA, shared thresholds (approximate) | **+0.004 [0.000, +0.009]** | +0.004 [−0.002, +0.011] |

Node F1 and node error are essentially unchanged at both stages. This is at our budget, not a literature-scale one, so it is
not evidence against U-KAN at that scale.

**Across all arms on BRAVA:** node threshold 0.25 gives Edge F1 0.237–0.240, and 0.30 gives 0.206–0.207, with worse β₀ error at
0.25. That gap is larger than any Edge F1 effect above, so the BRAVA own-threshold deltas should not be read as model effects.

### Figures

- [`q_distributions.png`](../../docs/research/figures/passthrough_20261103/q_distributions.png): pass-through score per chain category.
- [`examples.png`](../../docs/research/figures/passthrough_20261103/examples.png): before and after, including a rejection, a near-duplicate and a real error.

### Still open

Nothing is running or pending. These need your decision and were not run:
1. **BRAVA at one fixed node threshold, repeating tracing on GPU.** The BRAVA shared-threshold rows filter saved rows, so they are approximate.
2. **U-KAN at a longer budget, or `ukan_add`.** Neither was triggered by your rules.
3. **Pass-through rule.** 5–17% of contractions add an edge where the edge head had rejected both sides. Requiring accepted sides is untested, and two-hop
   training was not separated from the head.

The U-Net baselines were trained on 30 Oct code and were not rerun on the 3 Nov code. Commits since `825cca0`, and `20f32ee`, `e442c29`, `825cca0` themselves, exist only locally.

