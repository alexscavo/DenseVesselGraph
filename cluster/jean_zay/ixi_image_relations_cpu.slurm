#!/usr/bin/env bash
#SBATCH --job-name=ixi-image-eval
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-image-eval-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-image-eval-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset JOBSCRATCH IXI_MECHANISM_FEATURES PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$mode" in
  smoke)
    for arm in original token_geometry endpoints chord corridor; do
      "$runtime_python" -u scripts/evaluate_ixi_image_relations.py --smoke --arm "$arm" --root "$root"
    done
    "$runtime_python" -u scripts/collect_ixi_image_relations.py --smoke --root "$root" ;;
  baseline) "$runtime_python" -u scripts/evaluate_ixi_image_relations.py --arm original --root "$root" ;;
  evaluate)
    arms=(token_geometry endpoints chord corridor)
    "$runtime_python" -u scripts/evaluate_ixi_image_relations.py --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --root "$root" ;;
  collect) "$runtime_python" -u scripts/collect_ixi_image_relations.py --root "$root" ;;
  finish-reporting)
    arms=(original token_geometry endpoints chord corridor)
    "$runtime_python" -u scripts/evaluate_ixi_image_relations.py --finish-reporting --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --root "$root" ;;
  *) exit 2 ;;
esac
