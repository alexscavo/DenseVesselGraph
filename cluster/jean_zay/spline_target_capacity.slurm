#!/usr/bin/env bash
#SBATCH --job-name=spline-capacity
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/spline_learning/logs/spline-capacity-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/spline-capacity-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN
unset PYTHONPATH
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
mkdir -p "$root"
if [[ ! -f "$root/plan.json" ]]; then cp experiments/spline_target_capacity/plan.json "$root/plan.json"; fi
cmp "$root/plan.json" experiments/spline_target_capacity/plan.json
if [[ "$mode" == worker ]]; then
  "$runtime_python" -u scripts/analyze_spline_target_capacity.py worker --root "$root" --task "$SLURM_ARRAY_TASK_ID"
elif [[ "$mode" == collect ]]; then
  "$runtime_python" -u scripts/analyze_spline_target_capacity.py collect --root "$root"
else
  exit 2
fi
