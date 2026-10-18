#!/usr/bin/env bash
#SBATCH --job-name=ixi-node-table
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=experiments/spline_learning/logs/ixi-node-table-%j.out
#SBATCH --error=experiments/spline_learning/logs/ixi-node-table-%j.err
set -euo pipefail
python3 scripts/collect_ixi_node_refinement.py
