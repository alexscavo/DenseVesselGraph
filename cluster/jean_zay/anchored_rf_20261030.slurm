#!/usr/bin/env bash
#SBATCH --job-name=anchored-rf
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=00:15:00
#SBATCH --signal=B:USR1@240
#SBATCH --output=experiments/anchored_rf_20261030/logs/%x-%j.out
#SBATCH --error=experiments/anchored_rf_20261030/logs/%x-%j.err
# Heatmap-anchored RelationFormer arm.
# usage: sbatch [--time=HH:MM:SS] anchored_rf_20261030.slurm smoke
#        sbatch --time=HH:MM:SS -J arf-<DATASET> anchored_rf_20261030.slurm train <IXI|BRAVA>
#        sbatch --time=HH:MM:SS anchored_rf_20261030.slurm evaluate <IXI|BRAVA>
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/anchored_rf_20261030/runtime/$SLURM_JOB_ID"
trap 'rm -rf -- "$JOBSCRATCH" "${arf_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
arf_tmp="$SCRATCH/tmp/arf-$SLURM_JOB_ID"; mkdir -p "$arf_tmp"; export TMPDIR="$arf_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mode="${1:-smoke}"
if [[ "$mode" == smoke ]]; then
  "$runtime_python" -m unittest tests.test_anchored_relationformer -v
  "$runtime_python" -u scripts/anchored_relationformer_20261030.py smoke --workers 10
else
  # Not exec: the python child receives USR1/TERM via the forwarding trap below.
  "$runtime_python" -u scripts/anchored_relationformer_20261030.py "$mode" --dataset "$2" --workers 10 &
  child=$!
  trap 'kill -USR1 "$child" 2>/dev/null' USR1 TERM
  wait "$child" || status=$?
  # wait returns >128 when interrupted by the trapped signal; wait again for the real exit.
  if [[ "${status:-0}" -gt 128 ]]; then wait "$child" || status=$?; fi
  exit "${status:-0}"
fi
