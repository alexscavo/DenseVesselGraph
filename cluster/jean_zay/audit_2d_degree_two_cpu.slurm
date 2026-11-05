#!/usr/bin/env bash
#SBATCH --job-name=audit-2d-deg2
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=experiments/loop_subdivision_20261105/logs/%x-%j.out
#SBATCH --error=experiments/loop_subdivision_20261105/logs/%x-%j.err
# Audit: degree-2 nodes of native 2D targets lie on cycles (inserted breaking points).
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/audit_2d_degree_two_20261105.py
