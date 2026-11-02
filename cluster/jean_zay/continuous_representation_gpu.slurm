#!/usr/bin/env bash
#SBATCH --job-name=continuous-head
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=experiments/spline_learning/logs/continuous-head-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/continuous-head-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$root/runtime/$SLURM_JOB_ID" TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ks=(8 16 24)
if [[ "$mode" == benchmark ]]; then
 "$runtime_python" -m unittest tests.test_continuous_representation tests.test_adaptive_curve_metrics -v
 "$runtime_python" -u scripts/train_continuous_geometry.py benchmark --root "$root"
else
 "$runtime_python" -u scripts/train_continuous_geometry.py "$mode" --root "$root" --k "${ks[$SLURM_ARRAY_TASK_ID]}"
fi
