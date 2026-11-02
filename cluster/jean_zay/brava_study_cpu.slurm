#!/usr/bin/env bash
#SBATCH --job-name=brava-study-cpu
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spline_learning/logs/brava-study-cpu-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/brava-study-cpu-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export TMPDIR="${SCRATCH:?}/tmp/bg-$SLURM_JOB_ID"
unset PYTHONPATH IXI_MECHANISM_FEATURES
export JOBSCRATCH="${SCRATCH:?}/tmp/bg-$SLURM_JOB_ID"
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(standard coordinate_refinement)
zeros=(synthetic_spline synthetic_straight ixi_spline)
mkdir -p "$root"
if [[ ! -f "$root/plan.json" ]]; then cp experiments/brava_graph_study/plan.json "$root/plan.json"; fi
cmp "$root/plan.json" experiments/brava_graph_study/plan.json
case "$mode" in
  audit) "$runtime_python" -u scripts/audit_brava_study.py --root "$root" ;;
  representations) "$runtime_python" -u scripts/audit_brava_study.py --representations --root "$root" ;;
  smoke)
    for arm in "${zeros[@]}" "${arms[@]}"; do "$runtime_python" -u scripts/evaluate_brava_study.py --smoke --root "$root" --arm "$arm"; done
    "$runtime_python" -u scripts/collect_brava_study.py --smoke --root "$root" ;;
  zero) "$runtime_python" -u scripts/evaluate_brava_study.py --root "$root" --arm "${zeros[$SLURM_ARRAY_TASK_ID]}" ;;
  evaluate) "$runtime_python" -u scripts/evaluate_brava_study.py --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  collect) "$runtime_python" -u scripts/collect_brava_study.py --root "$root" ;;
  *) exit 2 ;;
esac
