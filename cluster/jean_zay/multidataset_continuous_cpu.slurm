#!/usr/bin/env bash
#SBATCH --job-name=multi-vascular-cpu
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=experiments/spline_learning/logs/multidataset-cpu-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/multidataset-cpu-%A_%a.err
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
 audit) "$runtime_python" -u scripts/audit_multidataset_graph.py --root "$root" ;;
 evaluate) "$runtime_python" -u scripts/multidataset_graph_evaluation.py evaluate --root "$root" --arm "$SLURM_ARRAY_TASK_ID" ;;
 collect)
  # Collector only needs numerical/plotting packages; protected Plotly contains
  # templates/validators omitted from the archived training runtime.
  unset PYTHONPATH
  "$runtime_python" -u scripts/collect_multidataset_graph.py --root "$root" ;;
 *) exit 2 ;;
esac
