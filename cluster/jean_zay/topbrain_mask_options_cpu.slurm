#!/usr/bin/env bash
#SBATCH --job-name=topbrain-masks
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:50:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
# Vessel coverage and bone share of candidate TopBrain brain masks (CPU).
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/topbrain_mask_options_20261104.py ${MOD:-}
