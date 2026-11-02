#!/usr/bin/env bash
#SBATCH --job-name=b0sd-score
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/b0_straight_dense_20261101/logs/%x-%j.out
#SBATCH --error=experiments/b0_straight_dense_20261101/logs/%x-%j.err
# Spline vs chord scoring of B0 graphs plus the dense model, CPU only. Usage: sbatch ... <dataset>
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/b0sd-$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"; trap 'rm -rf -- "$JOBSCRATCH"' EXIT
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 IDR_DEBUG=WARN CUDA_VISIBLE_DEVICES=
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/b0_straight_dense_20261101.py score "$1"
echo "done: score $1"
