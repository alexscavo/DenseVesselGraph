#!/usr/bin/env bash
#SBATCH --job-name=ixi-mask-input
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
#SBATCH --output=experiments/spline_learning/logs/ixi-mask-input-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-mask-input-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export JOBSCRATCH="$root/runtime/$SLURM_JOB_ID" TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset PYTHONPATH IXI_MECHANISM_FEATURES
source cluster/jean_zay/env.sh
if [[ ! -f "$root/plan.json" ]]; then cp experiments/ixi_segmentation_input/plan.json "$root/plan.json"; fi
cmp "$root/plan.json" experiments/ixi_segmentation_input/plan.json
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arms=(raw_mra vessel_mask)
case "$mode" in
  smoke)
    "$runtime_python" -m unittest -v tests.test_ixi_segmentation_oracle
    "$runtime_python" -u scripts/ixi_segmentation_oracle.py smoke --root "$root"
    for arm in "${arms[@]}"; do "$runtime_python" -u scripts/ixi_segmentation_oracle.py predict --smoke --arm "$arm" --root "$root"; done ;;
  benchmark) "$runtime_python" -u scripts/ixi_segmentation_oracle.py benchmark --root "$root" ;;
  train) "$runtime_python" -u scripts/ixi_segmentation_oracle.py train --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  predict) "$runtime_python" -u scripts/ixi_segmentation_oracle.py predict --root "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" ;;
  *) exit 2 ;;
esac
