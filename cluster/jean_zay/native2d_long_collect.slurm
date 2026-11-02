#!/usr/bin/env bash
#SBATCH --job-name=native2d-long-report
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=experiments/native2d_mixed_20261028/report-%j.out
#SBATCH --error=experiments/native2d_mixed_20261028/report-%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/experiments/native2d_mixed_20261028/mpl"
python3 scripts/collect_native2d_long.py
