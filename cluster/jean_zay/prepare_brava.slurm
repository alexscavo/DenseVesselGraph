#!/usr/bin/env bash
#SBATCH --job-name=brava-prepare
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/spline_learning/logs/brava-prepare-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/brava-prepare-%A_%a.err
set -eo pipefail
mode="$1"; output="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN
export JOBSCRATCH="$output/runtime/$SLURM_JOB_ID" TMPDIR="$output/runtime/$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
args=()
if [[ "$mode" == subject ]]; then args=(--index "$SLURM_ARRAY_TASK_ID"); fi
"$runtime_python" -u scripts/prepare_brava.py "$mode" --output "$output" --workers "$SLURM_CPUS_PER_TASK" "${args[@]}"
