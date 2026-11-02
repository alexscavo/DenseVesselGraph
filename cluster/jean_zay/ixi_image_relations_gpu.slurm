#!/usr/bin/env bash
#SBATCH --job-name=ixi-image-rel
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
#SBATCH --output=experiments/spline_learning/logs/ixi-image-rel-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-image-rel-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset JOBSCRATCH IXI_MECHANISM_FEATURES PYTHONPATH
source cluster/jean_zay/env.sh
mkdir -p "$root"
if [[ ! -f "$root/plan.json" ]]; then cp experiments/ixi_image_relation_learning/plan.json "$root/plan.json"; fi
cmp "$root/plan.json" experiments/ixi_image_relation_learning/plan.json
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
if [[ "$mode" == smoke || "$mode" == prepare ]]; then
  export JOBSCRATCH="$root/runtime/$SLURM_JOB_ID"
  spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
  trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
  source cluster/jean_zay/stage_spline_runtime.sh
  source cluster/jean_zay/ixi_mechanism_runtime.sh
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$mode" in
  smoke)
    "$runtime_python" -m unittest -v tests.test_image_relation_adapter
    "$runtime_python" -u scripts/ixi_image_relation_learning.py smoke --root "$root" ;;
  prepare) "$runtime_python" -u scripts/ixi_image_relation_learning.py prepare --root "$root" ;;
  benchmark) "$runtime_python" -u scripts/ixi_image_relation_learning.py benchmark --root "$root" ;;
  train)
    arms=(token_geometry endpoints chord corridor)
    "$runtime_python" -u scripts/ixi_image_relation_learning.py train --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --root "$root" ;;
  *) exit 2 ;;
esac
