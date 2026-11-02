#!/usr/bin/env bash
#SBATCH --job-name=oracle-geometry
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/oracle_geometry_20261030/logs/%x-%j.out
#SBATCH --error=experiments/oracle_geometry_20261030/logs/%x-%j.err
# WS4a oracle-topology geometry; usage: sbatch oracle_geometry.slurm <dataset> --mode <spline|polyline> [--interior M]
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/oracle_geometry_20261030/runtime/$SLURM_JOB_ID"
trap 'rm -rf -- "$JOBSCRATCH" "${probe_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
probe_tmp="$SCRATCH/tmp/ogeo-$SLURM_JOB_ID"; mkdir -p "$probe_tmp"; export TMPDIR="$probe_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/oracle_geometry_20261030.py --dataset "$1" \
    --output "$SCRATCH/experiments/gnbm/oracle_geometry_20261030" --workers 10 "${@:2}"
