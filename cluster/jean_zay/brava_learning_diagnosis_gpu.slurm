#!/usr/bin/env bash
#SBATCH --job-name=brava-k8-learning-diagnosis
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:25:00
#SBATCH --output=experiments/brava_learning_diagnosis_20261026/gpu-%j.out
#SBATCH --error=experiments/brava_learning_diagnosis_20261026/gpu-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
export JOBSCRATCH="$SCRATCH/experiments/gnbm/brava_learning_diagnosis_20261026/runtime/$SLURM_JOB_ID" TMPDIR="$SCRATCH/experiments/gnbm/brava_learning_diagnosis_20261026/runtime/$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/brava_learning_diagnosis.py
