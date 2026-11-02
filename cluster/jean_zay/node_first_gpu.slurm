#!/usr/bin/env bash
#SBATCH --job-name=node-first-gpu
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=experiments/node_first_curriculum_20261028/logs/gpu-%A_%a.out
#SBATCH --error=experiments/node_first_curriculum_20261028/logs/gpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/node_first_curriculum_20261028/runtime/$SLURM_JOB_ID"
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
trap 'rm -rf -- "$JOBSCRATCH" "${nfc_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
# env.sh resets TMPDIR to JOBSCRATCH. Override it only after sourcing env.sh.
nfc_tmp="$SCRATCH/tmp/nfc-$SLURM_JOB_ID"
if (( ${#nfc_tmp} > 64 )); then echo 'Multiprocessing TMPDIR too long' >&2; exit 2; fi
mkdir -p "$nfc_tmp"
export TMPDIR="$nfc_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == smoke ]]; then
  "$runtime_python" -m unittest tests.test_node_first_curriculum -v
else
  set -- "$@" --index "$SLURM_ARRAY_TASK_ID"
fi
"$runtime_python" -u scripts/node_first_curriculum.py "$@"
