#!/usr/bin/env bash
#SBATCH --job-name=ixi-restore-check
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=experiments/loop_subdivision_20261105/logs/%x-%j.out
#SBATCH --error=experiments/loop_subdivision_20261105/logs/%x-%j.err
# Verify restored IXI test images against surviving September patches (read-only).
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/verify_ixi_restore_20261105.py
