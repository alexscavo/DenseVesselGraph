#!/usr/bin/env bash
#SBATCH --job-name=loop-ab-compare
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/loop_subdivision_20261105/logs/%x-%j.out
#SBATCH --error=experiments/loop_subdivision_20261105/logs/%x-%j.err
# Loop A/B paired comparison (CPU, saved validation rows).
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj}
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/compare_loop_ab_20261105.py
