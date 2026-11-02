#!/usr/bin/env bash
#SBATCH --job-name=native2d-long-data
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=experiments/native2d_long_20261028/prepare-%j.out
#SBATCH --error=experiments/native2d_long_20261028/prepare-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SLURM_SUBMIT_DIR/experiments/native2d_long_20261028/tmp-prepare"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_long_20261028/mpl"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -u scripts/prepare_native2d_long.py
