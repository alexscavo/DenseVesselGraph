#!/usr/bin/env bash
#SBATCH --job-name=native2d-report
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=experiments/native2d_20261027/collector-%j.out
#SBATCH --error=experiments/native2d_20261027/collector-%j.err
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 scripts/collect_native2d.py
