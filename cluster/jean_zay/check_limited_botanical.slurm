#!/usr/bin/env bash
#SBATCH --job-name=limited-cpu-check
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=experiments/spline_learning/logs/limited-cpu-check-%j.out
#SBATCH --error=experiments/spline_learning/logs/limited-cpu-check-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv"
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
export JOBSCRATCH="/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/limited_botanical_200_20261020/runtime/$SLURM_JOB_ID"
export TMPDIR="$JOBSCRATCH" IDR_DEBUG=WARN
unset PYTHONPATH
source cluster/jean_zay/env.sh
trap 'if [[ "${spline_runtime_dir:-}" == "$JOBSCRATCH/gnbm-spline-runtime-$SLURM_JOB_ID" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
python -u scripts/check_limited_botanical.py
