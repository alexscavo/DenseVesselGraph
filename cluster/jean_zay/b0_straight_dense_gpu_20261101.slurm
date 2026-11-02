#!/usr/bin/env bash
#SBATCH --job-name=b0sd-infer
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:45:00
#SBATCH --output=experiments/b0_straight_dense_20261101/logs/%x-%j.out
#SBATCH --error=experiments/b0_straight_dense_20261101/logs/%x-%j.err
# B0 (or, with --trained, trained straight-edge) inference with the unchanged ixi_long evaluator. Usage: sbatch ... <SyntheticMRI|BRAVA> [--trained]
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/tmp/b0sd-$SLURM_JOB_ID"   # short AF_UNIX socket paths
mkdir -p "$JOBSCRATCH"; trap 'rm -rf -- "$JOBSCRATCH"' EXIT
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
unset PYTHONPATH
source cluster/jean_zay/env.sh
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
brava_runtime="$SCRATCH/tmp/br-1583536/gnbm-spline-runtime-1583536"   # same staged runtime as D08 / ixi_long
test -f "$brava_runtime/MultiScaleDeformableAttention3D.cpython-311-x86_64-linux-gnu.so"
export PYTHONPATH="$brava_runtime/ixi-protected:$brava_runtime"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$runtime_python" -u scripts/b0_straight_dense_20261101.py infer "$@"
echo "done: infer $*"
