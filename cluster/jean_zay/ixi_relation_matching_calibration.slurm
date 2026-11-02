#!/usr/bin/env bash
#SBATCH --job-name=ixi-match-calib
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-match-calib-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-match-calib-%A_%a.err
set -eo pipefail
mode="$1"; root="$2"
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" IDR_DEBUG=WARN WANDB_MODE=offline
export TMPDIR="$root/runtime/$SLURM_JOB_ID"
unset JOBSCRATCH IXI_MECHANISM_FEATURES PYTHONPATH
source cluster/jean_zay/env.sh
# These CPU analyses need only the complete protected module packages. No archive
# extraction or mutable venv package imports; dependencies were checked locally.
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$mode" in
  smoke)
    "$runtime_python" -m unittest -v tests.test_ixi_relation_matching_calibration
    "$runtime_python" -u scripts/ixi_relation_matching_calibration.py prepare --smoke --root "$root/smoke"
    for index in 0 1 2 3 4 5; do
      "$runtime_python" -u scripts/ixi_relation_matching_calibration.py analyse --index "$index" --root "$root/smoke"
    done
    "$runtime_python" -u scripts/collect_ixi_relation_matching_calibration.py --smoke --root "$root/smoke" ;;
  prepare)
    "$runtime_python" -u scripts/ixi_relation_matching_calibration.py prepare --root "$root" ;;
  analyse)
    "$runtime_python" -u scripts/ixi_relation_matching_calibration.py analyse --index "$SLURM_ARRAY_TASK_ID" --root "$root" ;;
  collect)
    "$runtime_python" -u scripts/collect_ixi_relation_matching_calibration.py --root "$root" ;;
  *) exit 2 ;;
esac
