#!/usr/bin/env bash
#SBATCH --job-name=brava-long-k8
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=experiments/brava_long_20261027/gpu-%A_%a.out
#SBATCH --error=experiments/brava_long_20261027/gpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/blg-$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"
test -d "$brava_runtime/ixi-protected/sympy"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-smoke}" == smoke ]]; then
  "$runtime_python" -m unittest tests.test_brava_long -v
  exec "$runtime_python" -u scripts/brava_long_experiment.py smoke
else
  arms=(scratch synthetic)
  exec "$runtime_python" -u scripts/brava_long_experiment.py train --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --end-epoch "$2"
fi
