#!/usr/bin/env bash
#SBATCH --job-name=native2d-straight
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=14:00:00
#SBATCH --output=experiments/native2d_straight_20261029/%x-%j.out
#SBATCH --error=experiments/native2d_straight_20261029/%x-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SLURM_SUBMIT_DIR/experiments/native2d_straight_20261029/tmp-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_mixed_20261028/mpl"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
python -u scripts/train_native2d_straight_20261029.py "$@"
