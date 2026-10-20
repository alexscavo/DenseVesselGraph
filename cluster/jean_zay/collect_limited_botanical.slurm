#!/usr/bin/env bash
#SBATCH --job-name=limited-collect
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=experiments/spline_learning/logs/limited-collect-%j.out
#SBATCH --error=experiments/spline_learning/logs/limited-collect-%j.err
set -euo pipefail
python3 scripts/limited_botanical_transfer.py collect
