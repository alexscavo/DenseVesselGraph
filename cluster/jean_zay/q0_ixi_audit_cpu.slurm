#!/usr/bin/env bash
#SBATCH --job-name=q0-ixi-target-audit
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/q0_ixi_audit_20261028/cpu-%j.out
#SBATCH --error=experiments/q0_ixi_audit_20261028/cpu-%j.err
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYVISTA_OFF_SCREEN=true MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
/lustre/fswork/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python3 -u scripts/audit_q0_ixi_targets.py --workers 4
