#!/usr/bin/env bash
#SBATCH --job-name=r06-relation-tokens
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=experiments/r06_relation_tokens_20261027/gpu-%A_%a.out
#SBATCH --error=experiments/r06_relation_tokens_20261027/gpu-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SLURM_SUBMIT_DIR/experiments/r06_relation_tokens_20261027/runtime"
export TMPDIR="$JOBSCRATCH" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export TORCH_EXTENSIONS_DIR="$SLURM_SUBMIT_DIR/.cache/torch" XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/.cache"
source cluster/jean_zay/env.sh
export PYTHONPATH="/lustre/fswork/projects/rech/vnc/upz25mj/projects/gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026/replay-runtime"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
runtime_python=/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python
if [[ "${1:-run}" == smoke ]]; then
  "$runtime_python" -u scripts/r06_relation_tokens.py smoke
else
  seeds=(364505 92741)
  "$runtime_python" -u scripts/r06_relation_tokens.py run --seed "${seeds[$SLURM_ARRAY_TASK_ID]}"
fi
