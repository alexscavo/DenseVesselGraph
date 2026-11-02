#!/usr/bin/env bash
#SBATCH --job-name=r07-matched-relations
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=experiments/r07_matched_supervision_20261027/gpu-%A_%a.out
#SBATCH --error=experiments/r07_matched_supervision_20261027/gpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/experiments/gnbm/r07_matched_supervision_20261027/runtime-$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch" XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
if [[ "${1:-run}" == smoke ]]; then
  python -u scripts/r07_matched_supervision.py smoke
else
  seeds=(364505 92741)
  python -u scripts/r07_matched_supervision.py run --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
fi
