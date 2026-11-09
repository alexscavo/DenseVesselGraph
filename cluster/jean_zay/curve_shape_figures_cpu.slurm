#!/usr/bin/env bash
#SBATCH --job-name=curve-figs
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/arm_compare_20261109/logs/%x-%j.out
#SBATCH --error=experiments/arm_compare_20261109/logs/%x-%j.err
# Figures for dense_traced_model §4.12–4.13 from saved validation rows (scripts/curve_shape_figures_20261109.py).
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
export MPLCONFIGDIR="$SCRATCH/tmp/mpl-$SLURM_JOB_ID"; mkdir -p "$MPLCONFIGDIR"
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python -u scripts/curve_shape_figures_20261109.py
