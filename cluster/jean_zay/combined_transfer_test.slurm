#!/usr/bin/env bash
#SBATCH --job-name=combined-test
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=experiments/spline_learning/logs/combined-test-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/combined-test-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
if [[ "${LIMITED_DATA_VERIFY:-0}" == 1 ]]; then
    export JOBSCRATCH="/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/limited_botanical_200_20261020/runtime/$SLURM_JOB_ID"
    export TMPDIR="$JOBSCRATCH"
fi
source cluster/jean_zay/env.sh
trap 'if [[ -n "${spline_runtime_dir:-}" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
if [[ "${LIMITED_DATA_VERIFY:-0}" == 1 ]]; then
    source cluster/jean_zay/ixi_mechanism_runtime.sh
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:?verify, test or collect}"
plan="${EVALUATION_PLAN:-experiments/combined_pretraining/evaluation-plan.json}"
if [[ "$mode" == collect ]]; then
    python scripts/evaluate_source_transfer.py --config "$plan" --collect test
else
    arms=(combined_straight combined_spline)
    if [[ -n "${EVALUATION_ARMS:-}" ]]; then IFS=: read -r -a arms <<< "$EVALUATION_ARMS"; fi
    arm="${arms[$SLURM_ARRAY_TASK_ID]}"
    if [[ "$mode" == verify ]]; then
        python scripts/evaluate_source_transfer.py --config "$plan" --arm "$arm" --smoke
    elif [[ "$mode" == test ]]; then
        python scripts/evaluate_source_transfer.py --config "$plan" --arm "$arm"
    else
        exit 2
    fi
fi
