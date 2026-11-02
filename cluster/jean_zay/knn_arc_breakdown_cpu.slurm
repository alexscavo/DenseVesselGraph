#!/usr/bin/env bash
# Arc-length breakdown of the 16-NN continued controls (knn_continued), at each run's
# train-selected thresholds, for comparison with traced_adaptive (§6.6).
#SBATCH --job-name=knn-arc-breakdown
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
root="$SCRATCH/experiments/gnbm/dense_graph_20261030"
for d in SyntheticMRI IXI BRAVA; do
  run="$root/$d/knn_continued"
  read -r nt et < <("$runtime_python" -c "import json,sys; t=json.load(open(sys.argv[1]))['thresholds']; print(t['node'], t['edge'])" "$run/final.json")
  echo "$d node_t=$nt edge_t=$et"
  "$runtime_python" -u scripts/dense_graph_traced_20261030.py --dataset "$d" --output "$cmp_tmp" \
    --breakdown-rows "$run/final-validation-rows.pkl" --breakdown-thresholds "$nt" "$et" \
    > "experiments/dense_graph_20261030/knn-continued-arc-length-$d.json"
done
echo "done: knn arc breakdown"
