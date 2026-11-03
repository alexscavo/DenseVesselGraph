#!/usr/bin/env bash
#SBATCH --job-name=twohop-eval
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/dense_graph_20261030/logs/%x-%j.out
#SBATCH --error=experiments/dense_graph_20261030/logs/%x-%j.err
# Two-hop / offset evaluation: paired comparisons, long edges, topology, triangles (CPU)
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/node_heatmap_probe_20261030/runtime/$SLURM_JOB_ID"
trap 'rm -rf -- "$JOBSCRATCH" "${probe_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
probe_tmp="$SCRATCH/tmp/the-$SLURM_JOB_ID"; mkdir -p "$probe_tmp"; export TMPDIR="$probe_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 IDR_DEBUG=WARN
for d in SyntheticMRI IXI BRAVA; do
    "$runtime_python" -u scripts/compare_traced_knn_20261030.py $d traced_twohop traced_adaptive
    "$runtime_python" -u scripts/compare_traced_knn_20261030.py $d traced_twohop_offset traced_twohop
    "$runtime_python" -u scripts/compare_arc_length_20261102.py $d traced_twohop traced_adaptive
    "$runtime_python" -u scripts/compare_arc_length_20261102.py $d traced_twohop_offset traced_twohop
done
"$runtime_python" -u scripts/topology_metrics_20261102.py traced_twohop traced_twohop_offset
"$runtime_python" -u scripts/twohop_analysis_20261103.py
