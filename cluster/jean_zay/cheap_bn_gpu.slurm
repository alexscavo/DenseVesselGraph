#!/usr/bin/env bash
#SBATCH --job-name=cheap-training-bn
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cheap_diagnostics_20261026/logs/%A_%a.out
#SBATCH --error=experiments/cheap_diagnostics_20261026/logs/%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$SCRATCH/experiments/gnbm/cheap_diagnostics_20261026/runtime/$SLURM_JOB_ID" TMPDIR="$SCRATCH/experiments/gnbm/cheap_diagnostics_20261026/runtime/$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
trap 'rm -rf -- "$JOBSCRATCH"' EXIT
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -m unittest tests.test_cheap_diagnostics tests.test_evaluator.EvaluatorTests.test_bn_calibration_updates_running_statistics_without_weights -v
datasets=(IXI SyntheticMRI)
"$runtime_python" -u scripts/cheap_research_diagnostics.py bn --dataset "${datasets[$SLURM_ARRAY_TASK_ID]}"
