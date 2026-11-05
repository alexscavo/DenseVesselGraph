#!/usr/bin/env bash
#SBATCH --job-name=topbrain-qc
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
set -eo pipefail
export IDR_DEBUG=WARN MPLCONFIGDIR=$SCRATCH/tmp/mpl-$SLURM_JOB_ID; cd "$SLURM_SUBMIT_DIR"
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/topbrain_stripping_qc_20261104.py ${QC_DATASET:-}
