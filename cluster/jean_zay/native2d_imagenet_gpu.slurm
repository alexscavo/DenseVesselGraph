#!/usr/bin/env bash
#SBATCH --job-name=n2d-imagenet
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/native2d_imagenet_20261030/%x-%j.out
#SBATCH --error=experiments/native2d_imagenet_20261030/%x-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/n2d-${SLURM_JOB_ID}"; mkdir -p "$JOBSCRATCH"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_mixed_20261028/mpl"
source cluster/jean_zay/env.sh
source "$GNBM_VENV/bin/activate" 2>/dev/null || true
export TORCH_HOME="$HOME/.cache/torch"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -u scripts/train_native2d_imagenet_20261030.py "$@"
