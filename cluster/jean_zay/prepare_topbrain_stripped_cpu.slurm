#!/usr/bin/env bash
#SBATCH --job-name=topbrain-strip
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
# Skull-stripped TopBrain + labels clipped to the TopCoW brain mask. ARGS: e.g. "--smoke --output $SCRATCH/tmp/tb-smoke"
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
cd "$SLURM_SUBMIT_DIR"
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u \
    scripts/prepare_topbrain_stripped_20261104.py --workers 6 ${ARGS:-}
