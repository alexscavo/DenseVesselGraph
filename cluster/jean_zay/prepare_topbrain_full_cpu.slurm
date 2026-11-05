#!/usr/bin/env bash
#SBATCH --job-name=topbrain-full
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
# Manifest, shared patient split and provenance for full (unstripped) TopBrain volumes.
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/prepare_topbrain_full_20261104.py
