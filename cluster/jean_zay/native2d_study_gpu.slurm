#!/usr/bin/env bash
#SBATCH --job-name=native2d-study
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:45:00
#SBATCH --output=experiments/native2d_20261028/%x-%j.out
#SBATCH --error=experiments/native2d_20261028/%x-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SLURM_SUBMIT_DIR/experiments/native2d_20261028/tmp-${SLURM_JOB_ID}"
export TMPDIR="$JOBSCRATCH" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_20261028/mpl"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -u scripts/train_native2d_study.py "${1:?gates or B0 or Axyz}"
