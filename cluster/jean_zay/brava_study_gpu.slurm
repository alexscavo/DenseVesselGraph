#!/usr/bin/env bash
#SBATCH --job-name=brava-study
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/spline_learning/logs/brava-study-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/brava-study-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="${SCRATCH:?}/tmp/bg-$SLURM_JOB_ID" TMPDIR="${SCRATCH:?}/tmp/bg-$SLURM_JOB_ID"
unset PYTHONPATH IXI_MECHANISM_FEATURES
source cluster/jean_zay/env.sh
if [[ ! -f "$root/plan.json" ]]; then cp experiments/brava_graph_study/plan.json "$root/plan.json"; fi
cmp "$root/plan.json" experiments/brava_graph_study/plan.json
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(standard coordinate_refinement)
zeros=(synthetic_spline synthetic_straight ixi_spline)
case "$mode" in
  smoke)
    "$runtime_python" -m unittest -v tests.test_brava_graph_study
    "$runtime_python" -u scripts/brava_graph_study.py smoke --root "$root"
    for arm in "${zeros[@]}"; do "$runtime_python" -u scripts/brava_graph_study.py predict --smoke --root "$root" --arm "$arm"; done
    for arm in "${arms[@]}"; do "$runtime_python" -u scripts/brava_graph_study.py predict --smoke --root "$root" --arm "$arm"; done ;;
  benchmark) "$runtime_python" -u scripts/brava_graph_study.py benchmark --root "$root" ;;
  zero) "$runtime_python" -u scripts/brava_graph_study.py predict --root "$root" --arm "${zeros[$SLURM_ARRAY_TASK_ID]}" ;;
  train) "$runtime_python" -u scripts/brava_graph_study.py train --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  predict) "$runtime_python" -u scripts/brava_graph_study.py predict --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  *) exit 2 ;;
esac
