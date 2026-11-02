#!/usr/bin/env bash
#SBATCH --job-name=ixi-mechanisms-cpu
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/spline_learning/logs/ixi-mechanisms-cpu-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/ixi-mechanisms-cpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
source cluster/jean_zay/env.sh
# CPU nodes can have insufficient /tmp space for the fixed runtime archive.
export JOBSCRATCH="$2/runtime/$SLURM_JOB_ID" TMPDIR="$2/runtime/$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
spline_runtime_dir="$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID"
trap 'if [[ "$spline_runtime_dir" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; rmdir "$JOBSCRATCH" 2>/dev/null || true; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="$1"; root="$2"
case "$mode" in
  smoke)
    python -m unittest -v tests.test_ixi_cached_mechanisms
    python -u scripts/prepare_ixi_mechanism_features.py --smoke --output "$root/smoke/features"
    export IXI_MECHANISM_FEATURES="$root/smoke/features"
    for experiment in A B C; do
      python -u scripts/run_ixi_cached_mechanisms.py --experiment "$experiment" --smoke --output "$root/smoke/$experiment"
    done ;;
  prepare)
    python -u scripts/prepare_ixi_mechanism_features.py --output "$root/features" ;;
  abc)
    export IXI_MECHANISM_FEATURES="$root/features"
    experiments=(A B C)
    experiment="${experiments[$SLURM_ARRAY_TASK_ID]}"
    python -u scripts/run_ixi_cached_mechanisms.py --experiment "$experiment" --output "$root/$experiment" ;;
  relation)
    export IXI_MECHANISM_FEATURES="$root/features"
    arms=(focal_control regularized hard_unmatched hard_matched)
    arm="${arms[$SLURM_ARRAY_TASK_ID]}"
    python -u scripts/collect_ixi_mechanisms.py --relation-source "$root/D/$arm" --output "$root/D-eval/$arm" ;;
  collect)
    python -u scripts/collect_ixi_mechanisms.py --root "$root" ;;
  *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac
