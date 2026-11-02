#!/usr/bin/env bash
#SBATCH --job-name=continuous-gt
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=experiments/spline_learning/logs/continuous-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/continuous-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$root/runtime/$SLURM_JOB_ID" TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$mode" == learned ]]; then
 spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
 source cluster/jean_zay/stage_spline_runtime.sh
 source cluster/jean_zay/ixi_mechanism_runtime.sh
fi
case "$mode" in
 gt) "$runtime_python" -u scripts/study_continuous_targets.py worker --root "$root" --task "$SLURM_ARRAY_TASK_ID" ;;
 collect) "$runtime_python" -u scripts/study_continuous_targets.py collect --root "$root" ;;
 learned) "$runtime_python" -u scripts/evaluate_continuous_geometry.py --root "$root" ;;
 *) exit 2 ;;
esac
