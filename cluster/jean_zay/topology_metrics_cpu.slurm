#!/usr/bin/env bash
#SBATCH --job-name=topology-metrics
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/dense_graph_20261030/logs/%x-%j.out
#SBATCH --error=experiments/dense_graph_20261030/logs/%x-%j.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export JOBSCRATCH="$SCRATCH/experiments/gnbm/dense_graph_20261030/runtime/$SLURM_JOB_ID"
trap 'rm -rf -- "$JOBSCRATCH" "${cmp_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
cmp_tmp="$SCRATCH/tmp/cmp-$SLURM_JOB_ID"; mkdir -p "$cmp_tmp"; export TMPDIR="$cmp_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
"$runtime_python" -u scripts/topology_metrics_20261102.py
