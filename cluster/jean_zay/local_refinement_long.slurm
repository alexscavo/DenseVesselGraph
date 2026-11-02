#!/usr/bin/env bash
#SBATCH --job-name=local-refinement-long
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=experiments/local_refinement_long/logs/%A_%a.out
#SBATCH --error=experiments/local_refinement_long/logs/%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="/tmp/local-refinement-long-$SLURM_JOB_ID" TMPDIR="/tmp/local-refinement-long-$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == smoke ]]; then
  "$runtime_python" -m unittest tests.test_focal_losses tests.test_fine_feature_architecture tests.test_local_refinement_long -v
fi
if [[ "$1" == train && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  set -- "$@" --index "$SLURM_ARRAY_TASK_ID"
fi
"$runtime_python" -u scripts/local_refinement_long.py "$@"
