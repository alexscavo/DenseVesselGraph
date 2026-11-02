#!/usr/bin/env bash
#SBATCH --job-name=native2d-fitfix-report
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=experiments/native2d_20261028_targetfix/report-%j.out
#SBATCH --error=experiments/native2d_20261028_targetfix/report-%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export NATIVE2D_STUDY_DIR="$SLURM_SUBMIT_DIR/experiments/native2d_20261028_targetfix"
export MPLCONFIGDIR="$NATIVE2D_STUDY_DIR/mpl"
python3 scripts/collect_native2d_study.py
