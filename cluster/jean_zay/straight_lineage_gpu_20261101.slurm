#!/usr/bin/env bash
#SBATCH --job-name=straight-lineage
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --signal=B:USR1@180
#SBATCH --output=experiments/straight_lineage_20261101/logs/%x-%j.out
#SBATCH --error=experiments/straight_lineage_20261101/logs/%x-%j.err
# Straight-edge RelationFormer lineage. Usage (submit from the repo, set --time per stage):
#   sbatch ... straight_lineage_gpu_20261101.slurm <s1|s2|s3|s4> <code snapshot dir> [--smoke] [--end-epoch N]
set -eo pipefail
repo="$SLURM_SUBMIT_DIR"; stage="$1"; code="$2"; shift 2
test -f "$repo/scripts/straight_lineage_20261101.py" && test -d "$code/models"
export GNBM_REPO_DIR="$repo"
export JOBSCRATCH="$SCRATCH/tmp/sl-$SLURM_JOB_ID"   # short AF_UNIX socket paths
mkdir -p "$JOBSCRATCH"; trap 'rm -rf -- "$JOBSCRATCH"' EXIT
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline IDR_DEBUG=WARN
export TORCH_EXTENSIONS_DIR="$repo/.cache/torch" XDG_CACHE_HOME="$repo/.cache" MPLCONFIGDIR="$repo/.cache/matplotlib"
unset PYTHONPATH
source "$repo/cluster/jean_zay/env.sh"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"   # staged runtime used by D08 / ixi_long
test -d "$brava_runtime/ixi-protected/sympy"
test -f "$brava_runtime/MultiScaleDeformableAttention3D.cpython-311-x86_64-linux-gnu.so"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$code"
"$runtime_python" -u "$repo/scripts/straight_lineage_20261101.py" "$stage" --code "$code" "$@"
echo "done: $stage $*"
