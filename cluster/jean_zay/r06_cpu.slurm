#!/usr/bin/env bash
#SBATCH --job-name=r06-relation-report
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/r06_relation_tokens_20261027/cpu-%j.out
#SBATCH --error=experiments/r06_relation_tokens_20261027/cpu-%j.err
set -eo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
/lustre/fshomisc/sup/pub/anaconda-py3/h100/2024.06/envs/pytorch-gpu-2.3.1+py3.11.9/bin/python -u scripts/collect_r06.py
