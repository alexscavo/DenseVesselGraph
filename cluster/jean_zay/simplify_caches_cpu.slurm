#!/usr/bin/env bash
#SBATCH --job-name=simplify-gt
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=experiments/node_miss_diagnosis_20261110/logs/%x-%j.out
#SBATCH --error=experiments/node_miss_diagnosis_20261110/logs/%x-%j.err
# GT graph simplification at a fixed scale (scripts/simplify_graph_caches_20261110.py); arguments are passed through.
# With RUN_SCRIPT=<script> the same environment runs another CPU script (e.g. the saved-row re-scoring).
# Environment: as topbrain_splines_cpu.slurm (module + the fsn1 venv); the staged GPU runtime cannot import pyvista on
# CPU nodes (vtkmodules.vtkImagingSources missing, jobs 25058 / 25123 of 10 Nov 2026).
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN PYTHONDONTWRITEBYTECODE=1
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
"$PY" -u "${RUN_SCRIPT:-scripts/simplify_graph_caches_20261110.py}" "$@"
