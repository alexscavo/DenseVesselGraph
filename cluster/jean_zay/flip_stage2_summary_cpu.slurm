#!/usr/bin/env bash
#SBATCH --job-name=flip-stage2-summary
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/final_try_20261113/logs/%x-%j.out
#SBATCH --error=experiments/final_try_20261113/logs/%x-%j.err
# Summary of the flip stage-2 runs (scripts/flip_stage2_summary_20261113.py; CPU, saved validation rows).
set -eo pipefail
export IDR_DEBUG=WARN; cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/vnc/upz25mj} OMP_NUM_THREADS=1
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/flip_stage2_summary_20261113.py
