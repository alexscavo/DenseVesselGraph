#!/usr/bin/env bash
#SBATCH --job-name=ixi-long-k8
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=experiments/ixi_long_20261030/gpu-%j.out
#SBATCH --error=experiments/ixi_long_20261030/gpu-%j.err
# IXI literature-scale RelationFormer B0 control (D08 recipe). Submit only via
#   python scripts/ixi_long_experiment_20261030.py submit-smoke | submit-train
# which pass IXI_LONG_CODE (the frozen code snapshot on SCRATCH).
# Usage: sbatch ... ixi_long_gpu_20261030.slurm smoke | train <end_epoch> | evaluate <epoch>
set -eo pipefail
test -n "${IXI_LONG_CODE:-}" && test -f "$IXI_LONG_CODE/scripts/ixi_long_experiment_20261030.py"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/ilg-$SLURM_JOB_ID"   # short AF_UNIX socket paths
mkdir -p "$JOBSCRATCH"
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"   # same staged runtime as D08
test -d "$brava_runtime/ixi-protected/sympy"
test -f "$brava_runtime/MultiScaleDeformableAttention3D.cpython-311-x86_64-linux-gnu.so"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
script="$IXI_LONG_CODE/scripts/ixi_long_experiment_20261030.py"
case "${1:-smoke}" in
  smoke) exec "$runtime_python" -u "$script" smoke ;;
  train) exec "$runtime_python" -u "$script" train --end-epoch "$2" ;;
  evaluate) exec "$runtime_python" -u "$script" evaluate --epoch "$2" ;;
  *) echo "Unknown mode: $1" >&2; exit 2 ;;
esac
