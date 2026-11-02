#!/usr/bin/env bash
#SBATCH --job-name=ixi-mechanisms-D
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-mechanisms-D-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-mechanisms-D-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH IXI_MECHANISM_FEATURES
source cluster/jean_zay/env.sh
# CPU nodes can have insufficient /tmp space for the fixed runtime archive.
export JOBSCRATCH="$2/runtime/$SLURM_JOB_ID" TMPDIR="$2/runtime/$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; rmdir "$JOBSCRATCH" 2>/dev/null || true; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="$1"; root="$2"
arms=(focal_control regularized hard_unmatched hard_matched)
if [[ "$mode" == smoke ]]; then
  for arm in "${arms[@]}"; do
    python -u scripts/train_ixi_relation_mechanisms.py --arm "$arm" --smoke --output "$root/smoke/D/$arm"
  done
  # Exercise the exact CPU collector path before allowing pilot training.
  export IXI_MECHANISM_FEATURES="$root/smoke/features"
  for arm in "${arms[@]}"; do
    python -u scripts/collect_ixi_mechanisms.py --relation-source "$root/smoke/D/$arm" --output "$root/smoke/D-eval/$arm"
  done
  python -u scripts/collect_ixi_mechanisms.py --root "$root/smoke"
elif [[ "$mode" == pilot ]]; then
  arm="${arms[$SLURM_ARRAY_TASK_ID]}"
  python -u scripts/train_ixi_relation_mechanisms.py --arm "$arm" --output "$root/D/$arm"
else
  exit 2
fi
