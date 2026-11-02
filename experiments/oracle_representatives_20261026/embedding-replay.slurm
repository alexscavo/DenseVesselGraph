#!/usr/bin/env bash
#SBATCH --job-name=oracle-eight-patch-replay
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:10:00
#SBATCH --output=experiments/oracle_representatives_20261026/replay-%j.out
#SBATCH --error=experiments/oracle_representatives_20261026/replay-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$SLURM_SUBMIT_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache" MPLCONFIGDIR="$SLURM_SUBMIT_DIR/.cache/matplotlib"
export JOBSCRATCH="$SCRATCH/experiments/gnbm/oracle_representatives_20261026/$SLURM_JOB_ID" TMPDIR="$SCRATCH/experiments/gnbm/oracle_representatives_20261026/$SLURM_JOB_ID"
source cluster/jean_zay/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python -u scripts/oracle_embedding_replay.py --device cuda
