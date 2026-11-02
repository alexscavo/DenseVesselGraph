# repo-graph.md — Graph-Native-Betti-Matching (fork)

> **Native 2D continuation, 28 October 2026:** [Held-out B0/A-xyz protocol and dataset readiness](research/native2d_heldout_20261028.md). Current registry: [vascular experiment registry](../research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md). Native2D-01 was implementation/tiny-fit only; follow the dated continuation instead of the historical “3D only” scope below.

> **Longer native 2D follow-up:** [Fixed 100-epoch vessel comparisons and 50-epoch mixed plant/vessel pretraining](research/native2d_long_20261028.md), including restored OCTA-Synth and unchanged 0.5 thresholds. The user clarified mixed pretraining before any long training started; the superseded pure-plant jobs were cancelled while pending. Its live report distinguishes submitted jobs from completed results.

> **Native2D-04 completed, 29 October:** [Final results and validation visualizations](research/native2d_results_20261029.md). All15 runs finished; mixed-pretrained OCTA reconstructs graphs, while scratch models remain empty at0.5. A-xyz helps tighter OCTA localization/Branch/Curve metrics, regresses on REAVER and does not fix AV-WIDE. False cycles/duplicates and mixed-pretraining exposure confounds remain explicit.

> **Requested straight-edge control, 29 October:** [Original-recipe OCTA B0 straight-only training](research/native2d_straight_20261029.md): same50 mixed +100 vessel-only epochs, spline head/loss disabled. Gate1699959 passed; jobs1699960→1699962 submitted. The earlier displayed OCTA edges were quantified as mostly straight, with the two-crop scope explicit.

> **Research consolidation, 23 October 2026:** [Current architecture and dependencies](../research/architecture.md). This document retains its detailed and dated historical context.

> October 19 orientation: this checkout is now
> `Graph-Native-Betti-Matching-Representation`, branch `representation`.
> Much of the September map below is historical. Start with the
> [current research review](research/PROJECT_REVIEW_20261019.md) and
> [IXI diagnostic execution record](research/metrics/6-ixi-query-attribution.md)
> for current checkpoints, data, hypotheses and job status.

Orientation file for a fresh Claude Code session dropped into this repo.
Read this before re-deriving structure from scratch. It's a map, not a
tutorial — the authoritative docs it points to are the source of truth;
update *this* file if those docs get renamed/moved, not the other way round.

## What this is

Fork of `alessandro-benvenuti/Graph-Native-Betti-Matching`
(origin remote → `github.com/alexscavo/Graph-Native-Betti-Matching`), cloned
into this Jean Zay project on 2026-09-04. It's a clean-room reconstruction of
a 3D graph-extraction pipeline (RelationFormer-style node/edge prediction
over 3D volumes) with graph-native Betti H0/H1 topology losses. Only the 3D
path is in scope — 2D/roads/OCTA/MoCo paths were explicitly not migrated.

Two domains it trains on:
- **plants** (`plants_3d2cut`) — biological source domain, used only as a
  *pretraining* mixing partner.
- **synthetic_mri** — the actual vessel target domain (synthetic vascular
  MRI patches: `raw`/`seg`/`vtp` per split). This is "the vessels" — any
  request to finetune/train "on vessels only" or "MRI only" means
  `data.datasets.synthetic_mri` as the sole/target dataset, no `plants` mix.

This is a sibling/relative of the `Image-to-Graph-Vessel` project already in
this Jean Zay workspace (see top-level `/lustre/.../projects/CLAUDE.md`) —
same broader research question (does biological pretraining transfer better
than roads to vessel graphs?), different, more rigorously-rebuilt codebase.
**Do not assume config keys or gotchas from `Image-to-Graph-Vessel/{3d,2d}`
apply here** — this repo has its own schema (`configs/loader.py`), its own
validated defaults, and deliberately dropped some of that codebase's dead
config surface (domain-adversarial losses, `AUTO_RESUME`, 2D, etc. — see
"Compatibility decisions" in `configs/README.md`).

## Where things actually are (Jean Zay, this project/account)

```
$WORK  = /lustre/fswork/projects/rech/vnc/upz25mj
$SCRATCH = /lustre/fsn1/projects/rech/vnc/upz25mj

repo:            $WORK/projects/Graph-Native-Betti-Matching   (this checkout)
synthetic_mri:   two copies exist, both group-readable (`vnc`):
                   - our own, small: $SCRATCH/datasets/syntheticMRI/patches/syntheticMRI
                     (train 4000 / val 1000 / test 4958) — same layout as the
                     collaborator's, just capped.
                   - upz73jr's, full/current, boundary-corrected — use this
                     one for anything calling itself "full dataset" / "all
                     patches":
                     /lustre/fsn1/projects/rech/vnc/upz73jr/datasets/syntheticMRI/new_patches_boundary
                     (train 91200 / val 19200 / test 20160). Sibling
                     `new_patches` (no `_boundary`) is the same size but the
                     pre-correction version — prefer `_boundary`.
plants:          not in our own $SCRATCH. upz73jr has it, group-readable:
                   /lustre/fsn1/projects/rech/vnc/upz73jr/datasets/plants_3d2cut/patches_3d
                   (standard train/val/test/{raw,seg,vtp} layout) — use this
                   for any mixed-pretraining run (`pretrain_mixed.yaml`).
venv:            built (torch 2.3.1/CUDA 12.4, verified). Lives physically on
                   $SCRATCH/venvs/vascular-graph-extraction-h100-torch231 —
                   $WORK/venvs/vascular-graph-extraction-h100-torch231 is a
                   *symlink* to it (see "$WORK inode quota" below for why).
                   Both env.sh's default $GNBM_VENV and the repo's .venv
                   symlink resolve through that $WORK symlink transparently;
                   no script changes needed to use it normally.
CUDA extension:  built and validated (debug smoke job 1759841, COMPLETED,
                   2m12s) — checkpoint-compatibility, CUDA/model, MRI-loader,
                   and one-epoch train+resume tests all passed.
checkpoints:     none of our own yet, but see "upz73jr" below — reused their
                   debug-job checkpoint to satisfy submit_debug.sh's
                   real-checkpoint compatibility requirement.
running:         finetune_vessels_full_300ep (job 1760082, submitted
                   2026-09-04) — vessels-only, from-scratch, all 91200
                   patches from upz73jr's new_patches_boundary, 300 epochs,
                   4×H100, qos_gpu_h100-t4. Config:
                   configs/finetune_synthetic_mri_vessels_full_300ep.yaml.
                   Output: $SCRATCH/experiments/gnbm/finetune_vessels_full_300ep/.
                   Check `sacct -j 1760082` for current state — don't trust
                   this line, it's a snapshot from submission time.
uv:              user-level ~/.local/bin/uv, symlinked to $WORK/tools/uv/uv
                   (that exact path is what setup_environment.sh expects).
```

## $WORK inode quota — read this before creating any new files here

The `vnc` project's `$WORK` inode quota is **shared across the whole team**
(500,000 files total, project-wide, not per-user) and was already sitting at
~92% before this session touched anything. Two concrete lessons from
2026-09-04:

1. `setup_environment.sh`'s `uv` package-download cache
   (`$WORK/.cache/uv`) dumped ~93,000 tiny files into `$WORK` and pushed the
   project over the edge — a config-file write failed with `EDQUOT`.
   **Always point `UV_CACHE_DIR` at `$SCRATCH` before running that script**
   (`$SCRATCH` has no quota for this project — confirmed via `lfs quota`).
   The stale cache was deleted; this is not a recurring cost once
   `UV_CACHE_DIR` is redirected.
2. A `--system-site-packages` venv is still tens of thousands of files
   (~31k here). It was originally built under `$WORK/venvs/...` (the
   `setup_environment.sh` default) and later relocated to
   `$SCRATCH/venvs/...` with a symlink left at the original `$WORK` path —
   freed ~31k inodes from the shared quota with zero script/env changes
   needed elsewhere. **Do this relocation immediately after any future
   from-scratch `setup_environment.sh` run**, before it becomes load-bearing
   for a running job (do it while any dependent job is still `PENDING`, not
   `RUNNING`).

Similarly, always set `TORCH_EXTENSIONS_DIR` under `$SCRATCH` before the
CUDA-extension build (`submit_debug.sh` / any job that builds
`models/ops`) — it defaults to `$WORK` in `env.sh` otherwise.

Unrelated but worth knowing: `Image-to-Graph-Vessel/.venv` (the sibling
project) is a separate, fully self-contained venv (Python 3.10.20, its own
bundled `torch==2.6.0+cu124`, `include-system-site-packages: false`) sitting
on `$WORK` too, and is **not** interchangeable with this repo's venv — GNBM
hard-requires the H100 module's `torch==2.3.1` and compiles its CUDA
extension against it specifically. Don't try to share/reuse the two.

Check current quota state with `lfs quota -u $USER /lustre/fswork/...` (live)
or `idr_quota_project` (official but **only refreshes once a day** — don't
trust it for same-session before/after comparisons; use `lfs quota` or a
`find -type f | wc -l` delta instead).

Account: `vnc@h100`, partition `gpu_p6`, H100s only, one node max
(1/2/4 GPUs), QoS `qos_gpu_h100-dev` (≤2h, ≤2 GPU, code checks only),
`-t3` (≤20h), `-t4` (≤100h). Compute nodes have no internet — nothing
here needs a download at train time except W&B online sync, which is why
production jobs force `WANDB_MODE=offline`.

## First-time setup (nothing below has been done yet in this fork)

1. `mkdir -p $WORK/tools/uv && ln -s ~/.local/bin/uv $WORK/tools/uv/uv`
   (or install uv fresh there) — `setup_environment.sh` hard-requires this exact path.
2. **Export `UV_CACHE_DIR=$SCRATCH/.cache/uv` before running the next step** —
   otherwise its package-download cache lands on the tight shared `$WORK`
   inode quota (see "$WORK inode quota" above; this bit us in the first
   session here).
3. `bash cluster/jean_zay/setup_environment.sh` (login node, downloads
   wheels from `requirements/jean-zay.txt` + wandb, no GPU/CUDA build). Its
   own trailing `uv pip check` false-flags torch/monai/timm/etc as "not
   installed" — that's just `--system-site-packages` confusing the checker;
   ignore that specific exit-1 if the earlier `import torch` print succeeded.
   Then relocate the built venv off `$WORK` per the quota section above.
4. `wandb login --verify` (once, login node) — not done yet in this fork;
   offline-mode training doesn't need it, only `wandb sync` later does.
5. `bash cluster/jean_zay/submit_debug.sh` — bounded 45-min H100 dev job,
   `qos_gpu_h100-dev`. Builds the deformable-attention CUDA extension for
   sm_90 (**export `TORCH_EXTENSIONS_DIR=$SCRATCH/.cache/torch-extensions/gnbm-h100-torch231-sm90`
   first**, same quota reasoning), runs CUDA forward/backward tests, trains
   one real batch with the full focal+Betti objective, writes one
   checkpoint. Needs `GNBM_MRI_CHECKPOINT` pointing at *some* real
   checkpoint for its compatibility test — none existed in our own
   `$SCRATCH` the first time; reused upz73jr's
   (`/lustre/fsn1/projects/rech/vnc/upz73jr/experiments/gnbm-debug/jean_zay_debug_1134039/models/best_checkpoint.pt`).
   Already done once here (job 1759841, COMPLETED) — re-run only if the
   code under `models/ops` or the PyTorch/CUDA environment changes.
6. Only after 2–5 succeed: production training via `submit_train.sh` (see
   below).

## Config system (details: `configs/README.md`, full key reference:
`docs/RUNNING_THE_MODEL.md` §9)

- Plain-merge YAML, no Hydra: a `defaults:` list is loaded left→right, child
  keys recursively override, `${ENV_VAR}` expands at load time (missing var
  = hard error), result validated by `configs/loader.py` before model/data
  construction, and the fully-resolved YAML is always written to the run dir
  as `resolved-config.yaml` — that file, not the checked-in YAML, is the
  ground truth for what a given run actually did.
- `configs/base.yaml` = everything shared (model/optimizer/loss/aug/eval
  defaults). Everything else is a small overlay on top of it.
- Key per-dataset field: `train_samples` / `validation_samples` — positive
  int caps the loader to a prefix, **`null` = use every discovered patch**.
  Full-dataset recipes (`configs/experiments/full_dataset_node_focal/`,
  `full_dataset_comparison/`, `boundary_gamma_sweep_500/finetune_common.yaml`,
  `focal_matrix_600/finetune_common.yaml`) already set both to `null` — that's
  the established pattern for "use all the vessel patches", not a special flag.
- `configs/experiments/<name>/` = one controlled experiment matrix per
  directory, each with its own `README.md` — check that README before
  reusing/extending one; they encode specific comparisons (loss-matrix
  sweeps, gamma sweeps, edge-candidate ablations), not generic templates.
- No schema drift here the way `Image-to-Graph-Vessel` has — `loader.py`
  actively validates, so a typo'd/dead key raises at load time rather than
  silently no-op'ing. Still: base off an existing full-dataset finetune
  config rather than hand-rolling one, to inherit the already-audited
  early-stopping / F1-checkpoint / evaluation block.

## Launching training (full CLI/env reference: `docs/RUNNING_THE_MODEL.md`)

Generic one-stage launcher, from repo root after `source cluster/jean_zay/env.sh`:

```bash
export SYNTHETIC_MRI_DATASET="$SCRATCH/datasets/syntheticMRI/patches/syntheticMRI"
export GNBM_OUTPUT_DIR="$SCRATCH/experiments/gnbm"
export GNBM_GPUS=4                      # 1, 2, or 4 (one node)
export GNBM_BATCH_SIZE=8                # per-GPU
export GNBM_QOS=qos_gpu_h100-t4         # -dev / -t3 (≤20h) / -t4 (≤100h)
export GNBM_WALLTIME=48:00:00
export WANDB_MODE=offline               # compute nodes have no internet
bash cluster/jean_zay/submit_train.sh CONFIG_PATH UNIQUE_RUN_NAME
```

`submit_train.sh` preflights (dataset dirs exist, config validates, no
GPU/CPU consumed) on the login node before `sbatch`-ing
`cluster/jean_zay/train_h100.slurm`. It refuses to run if the venv/CUDA deps
aren't importable — see "First-time setup" above. Multi-lineage launchers
(`submit_full_dataset_node_focal.sh`, `submit_boundary_gamma_sweep_500.sh`,
`submit_focal_matrix_600.sh`, `submit_edge_candidate_ablation_600.sh`)
auto-chain a pretrain→finetune pair with a checkpoint handoff; each has a
`GNBM_*_DRY_RUN=1` switch to print without submitting.

Init-vs-resume:
- `GNBM_INITIAL_WEIGHTS` = model weights only, fresh optimizer/scheduler —
  use to start any new stage/experiment from a checkpoint.
- `GNBM_RESUME_CHECKPOINT` (+ `GNBM_AUTO_RESUME=1` variant) = full state
  restore, same run continued — use only after a TIMEOUT/interruption of the
  *same* configured schedule.
- Mutually exclusive; unset both to train fully from scratch.

Runs land at `GNBM_OUTPUT_DIR/RUN_NAME/` — `resolved-config.yaml`,
`dataset-manifest.json`, `validation-metrics.jsonl`,
`best_checkpoint.pt` / `best_metric_checkpoint.pt` /
`best_node_f1_checkpoint.pt` / `best_edge_f1_checkpoint.pt` /
`latest_checkpoint.pt`, `checkpoint_epoch=N.pt` (if interval policy),
`training-status.json`, `early-stopping.json`, `wandb-run.json`.

## Monitoring

```bash
squeue -u "$USER" -o "%.18i %.42j %.10T %.12M %.14l %.80R"
tail -f "$WORK/logs/graph-native-betti-matching/JOB_NAME-JOB_ID.out"
sacct -j JOB_ID --format=JobID,JobName%42,State,ExitCode,Elapsed,MaxRSS,AllocTRES%60
bash cluster/jean_zay/sync_wandb_offline.sh "$GNBM_OUTPUT_DIR"   # after offline runs
```

No monitor-script auto-chaining pattern exists in this repo the way it does
in `job-runs/slurm/monitor_*.sh` for `Image-to-Graph-Vessel` — the
multi-lineage `submit_*.sh` scripts here do their own chaining synchronously
inside one submission call instead (see above), so there's nothing to
reconcile after a session restart the way the sibling project needs.

## Evaluation

`python evaluate.py --config CONFIG --checkpoint CKPT --output-dir DIR
--split val|test [...]` → `summary.json`, `per-patch-metrics.csv`,
`predictions.json`, `plots/*.png`. Full flag reference and a Slurm example:
`docs/RUNNING_THE_MODEL.md` §7. 3D interactive prediction viewer:
`scripts/visualize_graph_prediction_3d.py`, documented in
`docs/GRAPH_PREDICTION_3D.md`.

## Docs index (read the specific one before touching that area)

| Doc | Covers |
|---|---|
| `docs/RUNNING_THE_MODEL.md` | **Start here for any launch.** Complete env/CLI/YAML/checkpoint reference. |
| `configs/README.md` | Config composition contract, edge-candidate/balancing semantics, compatibility decisions. |
| `docs/TRAINING.md` | Checkpoint semantics, early-stopping/F1 mechanics, resume vs re-init, smoke tests. |
| `docs/EVALUATION.md` | Metric protocol, thresholds. |
| `docs/LOSSES.md` | Baseline/Betti/focal loss contract. |
| `docs/MODEL.md` | Architecture compatibility values (legacy-checkpoint quirks). |
| `docs/DATA_LOADING.md`, `DATA_AUDIT.md`, `DATA_GENERATION.md` | Dataset contracts, SyntheticMRI audit findings, patch generator. |
| `docs/AUGMENTATION.md` | Per-domain augmentation knobs. |
| `docs/EXPERIMENTS.md` | The controlled ablation matrix design. |
| `docs/experiments/domain_adaptation.md` | Why adversarial domain adaptation is archived/inactive here (unlike `Image-to-Graph-Vessel`). |
| `docs/SYNTHETIC_MRI_DATASET_REPORT.md` | Dataset findings/decisions. |
| `docs/GRAPH_PREDICTION_3D.md` | Interactive 3D prediction viewer. |
| `docs/research/README.md` | Representation/metrics roadmap and its live implementation tracker. |
| `cluster/jean_zay/README.md` | Jean Zay-specific setup/submission quick reference (thinner than `RUNNING_THE_MODEL.md`). |
| `configs/experiments/<name>/README.md` | Per-experiment-matrix design notes — check before reusing a matrix's configs as a template. |

## Collaborator: upz73jr

Same `vnc` project, different Jean Zay login, group-readable `$SCRATCH`.
They have a mature, already-run GNBM setup: `experiments/gnbm/` contains
many completed pretrain/finetune/scratch runs under a `*_seed364505` naming
convention (e.g. `pretrain_full_mixed_baseline_seed364505`,
`finetune_full_mri_node_focal_seed364505`,
`scratch_mri_node_focal_immediate_400_seed364505`), plus the two dataset
copies referenced above. Worth checking their `experiments/gnbm/` for an
existing checkpoint or a near-duplicate run before launching a new one from
scratch — re-deriving something they already have wastes both compute and
the shared `$WORK` quota. Their runs' `resolved-config.yaml` files are the
ground truth for exactly what each one did.

## Maintenance note

This file was created 2026-09-04, the first session in this fork. As of
that session: venv built (relocated to `$SCRATCH`, see quota section),
CUDA extension built and validated (debug job 1759841), one production run
submitted (`finetune_vessels_full_300ep`, job 1760082). Update the
"running:"/"checkpoints:" lines above as runs complete rather than letting
them go stale — check `sacct`/`GNBM_OUTPUT_DIR` as ground truth, this file
is a map, not a live dashboard.
