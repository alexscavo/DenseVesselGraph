#!/usr/bin/env bash
#SBATCH --job-name=multi-vascular
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=05:00:00
#SBATCH --output=experiments/spline_learning/logs/multidataset-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/multidataset-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$root/runtime/$SLURM_JOB_ID" TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$mode" in
 smoke)
  "$runtime_python" -m unittest tests.test_adaptive_geometry tests.test_adaptive_curve_metrics tests.test_multidataset_graph -v
  "$runtime_python" -u scripts/multidataset_graph_training.py smoke --root "$root" ;;
 train)
  "$runtime_python" -u scripts/multidataset_graph_training.py train --root "$root" --arm "$SLURM_ARRAY_TASK_ID"
  "$runtime_python" -u scripts/multidataset_graph_evaluation.py predict --root "$root" --arm "$SLURM_ARRAY_TASK_ID" ;;
 predict) "$runtime_python" -u scripts/multidataset_graph_evaluation.py predict --root "$root" --arm "$SLURM_ARRAY_TASK_ID" ;;
 *) exit 2 ;;
esac
