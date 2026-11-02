#!/usr/bin/env bash
#SBATCH --job-name=ixi-node-collect
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-node-collect-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-node-collect-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset JOBSCRATCH PYTHONPATH IXI_MECHANISM_FEATURES
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(ce_weighted ce_equal focal_equal)
case "$mode" in
  smoke)
    for arm in "${arms[@]}"; do "$runtime_python" -u scripts/evaluate_ixi_node_focal.py --smoke --root "$root" --arm "$arm"; done
    "$runtime_python" -u scripts/collect_ixi_node_focal.py --smoke --root "$root" ;;
  evaluate) "$runtime_python" -u scripts/evaluate_ixi_node_focal.py --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  collect) "$runtime_python" -u scripts/collect_ixi_node_focal.py --root "$root" ;;
  *) exit 2 ;;
esac
