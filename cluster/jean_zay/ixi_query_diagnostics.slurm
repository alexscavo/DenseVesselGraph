#!/usr/bin/env bash
#SBATCH --job-name=ixi-query-audit
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
#SBATCH --output=experiments/spline_learning/logs/ixi-query-audit-%j.out
#SBATCH --error=experiments/spline_learning/logs/ixi-query-audit-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
trap 'if [[ "$spline_runtime_dir" == /tmp/gnbm-spline-runtime-* ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:-smoke}"
arm="${2:-local_image}"
output="$SCRATCH/experiments/gnbm/ixi_query_diagnostics_20261019/${mode}-${SLURM_JOB_ID}-${arm}"
python -m unittest -v tests.test_ixi_query_diagnostics
args=()
case "$mode" in
  smoke) args+=(--smoke) ;;
  pilot) ;;
  validation) args+=(--full-validation) ;;
  *) echo "Unknown diagnostic mode: $mode" >&2; exit 2 ;;
esac
python -u scripts/diagnose_ixi_queries.py --arm "$arm" --output "$output" "${args[@]}"
python -u scripts/collect_ixi_query_diagnostics.py "$output"
