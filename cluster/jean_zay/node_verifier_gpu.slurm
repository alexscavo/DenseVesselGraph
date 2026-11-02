#!/usr/bin/env bash
#SBATCH --job-name=node-verifier
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/node_verifier_feasibility/logs/%j.out
#SBATCH --error=experiments/node_verifier_feasibility/logs/%j.err
set -eo pipefail
mode="$1"
dataset="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="/tmp/node-verifier-$SLURM_JOB_ID" TMPDIR="/tmp/node-verifier-$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -m unittest tests.test_node_verifier -v
if [[ "$mode" == verifier ]]; then
  "$runtime_python" -u scripts/node_verifier_feasibility.py extract --dataset "$dataset"
  "$runtime_python" -u scripts/node_verifier_feasibility.py fit --dataset "$dataset" --device cuda
elif [[ "$mode" == transfer ]]; then
  "$runtime_python" -u scripts/node_verifier_feasibility.py extract --dataset BRAVA
elif [[ "$mode" == brava ]]; then
  "$runtime_python" -u scripts/brava_learnability.py --initializer "$dataset"
else
  exit 2
fi
