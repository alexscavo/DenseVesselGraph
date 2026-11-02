#!/usr/bin/env bash
#SBATCH --job-name=brava-long-cpu
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/brava_long_20261027/cpu-%j.out
#SBATCH --error=experiments/brava_long_20261027/cpu-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/blc-$SLURM_JOB_ID"
mkdir -p "$JOBSCRATCH"
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 IDR_DEBUG=WARN
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"
test -d "$brava_runtime/ixi-protected/sympy"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-preflight}" == preflight ]]; then
  "$runtime_python" -m unittest tests.test_brava_long -v
  "$runtime_python" -u scripts/brava_long_experiment.py preflight
else
  "$runtime_python" -u scripts/collect_brava_long.py --epoch "$2"
fi
