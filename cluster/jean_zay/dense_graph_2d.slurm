#!/usr/bin/env bash
#SBATCH --job-name=dg2d
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --output=experiments/dense_graph_2d_20261030/logs/%x-%j.out
#SBATCH --error=experiments/dense_graph_2d_20261030/logs/%x-%j.err
# Native 2D two-stage dense graph model (port of GNBM dense_graph_20261030).
# usage (from the worktree root):
#   sbatch --job-name=dg2d-<dataset>-<run> --time=HH:MM:SS cluster/jean_zay/dense_graph_2d.slurm \
#          <avwide|reaver|octasynth> [--run-name NAME] [--updates N] [--warmup-gt N] ...
# Runtime environment of native2d_long_gpu.slurm (env.sh); short TMPDIR under $SCRATCH/tmp
# so multiprocessing AF_UNIX socket paths stay short.
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/tmp/dg2-$SLURM_JOB_ID"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_mixed_20261028/mpl"
trap 'rm -rf -- "$JOBSCRATCH"' EXIT
source cluster/jean_zay/env.sh
export TMPDIR="$JOBSCRATCH"; mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=$SLURM_JOB_ID tmpdir=$TMPDIR (${#TMPDIR} chars) args=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/dense_graph_2d_20261030.py --dataset "$1" \
    --output "$SCRATCH/experiments/gnbm/dense_graph_2d_20261030" \
    --report-dir experiments/dense_graph_2d_20261030/results --workers 6 --metric-workers 14 "${@:2}"
echo "done: $(date -Is)"
