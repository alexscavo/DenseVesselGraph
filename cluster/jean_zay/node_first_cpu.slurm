#!/usr/bin/env bash
#SBATCH --job-name=node-first-cpu
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/node_first_curriculum_20261028/logs/cpu-%A_%a.out
#SBATCH --error=experiments/node_first_curriculum_20261028/logs/cpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/node_first_curriculum_20261028/runtime/$SLURM_JOB_ID"
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache"
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
trap 'rm -rf -- "$JOBSCRATCH" "${nfc_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
nfc_tmp="$SCRATCH/tmp/nfc-$SLURM_JOB_ID"
if (( ${#nfc_tmp} > 64 )); then echo 'Multiprocessing TMPDIR too long' >&2; exit 2; fi
mkdir -p "$nfc_tmp"
export TMPDIR="$nfc_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == arm ]]; then set -- "$@" --index "$SLURM_ARRAY_TASK_ID"; fi
if [[ "$1" == smoke ]]; then "$runtime_python" -m unittest tests.test_node_first_collection -v; fi
"$runtime_python" -u scripts/collect_node_first_curriculum.py "$@"
