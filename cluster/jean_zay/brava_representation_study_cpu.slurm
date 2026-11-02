#!/usr/bin/env bash
#SBATCH --job-name=brava-representation
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/brava_representation_20261026/logs/%A_%a.out
#SBATCH --error=experiments/brava_representation_20261026/logs/%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$SCRATCH/tmp/br-$SLURM_JOB_ID" TMPDIR="$SCRATCH/tmp/br-$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
# Reuse the successfully staged, SHA-verified runtime from CPU job1583536.
# No extraction into node-local /tmp and no per-job archive duplication.
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"
test -d "$brava_runtime/ixi-protected/sympy"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/brava_representation_study.py "$@"
