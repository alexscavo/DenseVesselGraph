#!/usr/bin/env bash
#SBATCH --job-name=ixi-node-refine
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
#SBATCH --output=experiments/spline_learning/logs/ixi-node-refine-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-node-refine-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
trap 'if [[ "$spline_runtime_dir" == /tmp/gnbm-spline-runtime-* ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(baseline local_image zero_image)
if [[ "${1:-}" == smoke ]]; then
  for arm in "${arms[@]}"; do
    python -u scripts/train_ixi_node_refinement.py --arm "$arm" --smoke
  done
else
  python -u scripts/train_ixi_node_refinement.py --arm "${arms[$SLURM_ARRAY_TASK_ID]}"
fi
