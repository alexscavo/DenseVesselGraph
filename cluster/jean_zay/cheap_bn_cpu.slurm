#!/usr/bin/env bash
#SBATCH --job-name=cheap-bn-collection
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/cheap_diagnostics_20261026/logs/report-%j.out
#SBATCH --error=experiments/cheap_diagnostics_20261026/logs/report-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch" XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib" IDR_DEBUG=WARN
export TMPDIR="$SCRATCH/experiments/gnbm/cheap_diagnostics_20261026/cpu-runtime/$SLURM_JOB_ID"
unset PYTHONPATH JOBSCRATCH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/report_cheap_diagnostics.py bn
