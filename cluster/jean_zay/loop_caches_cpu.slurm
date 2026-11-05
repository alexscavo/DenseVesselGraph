#!/usr/bin/env bash
#SBATCH --job-name=loop-cache
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:30:00
#SBATCH --output=experiments/loop_subdivision_20261105/logs/%x-%j.out
#SBATCH --error=experiments/loop_subdivision_20261105/logs/%x-%j.err
# K8 transfer-spline caches with loop subdivision (breaking-point anchors; no loop patch excluded), 5 Nov 2026.
# Same sources, patches, limits and seed as the existing caches; only --subdivide-loops differs. KIND=ixi|ixi_trainval|synthetic|
# topbrain_mr|topbrain_ct; MODE=smoke writes a *_smoke folder with 2 patches per split.
set -eo pipefail
: "${KIND:?set KIND}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
base=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets; P=optimal_radius_0p75x_c095_vedo_original
case "$KIND" in
  ixi) source="$base/IXI/vascular_graph_sources/$P"; patches="$base/IXI/vascular_patches/$P"
       output="$base/IXI/transfer_splines_k8_loops_20261105"; options=(--train-limit 10000 --test) ;;
  ixi_trainval) source="$base/IXI/vascular_graph_sources/${P}_trainval_20261105"; patches="$base/IXI/vascular_patches/${P}_trainval_20261105"
       output="$base/IXI/transfer_splines_k8_loops_20261105"; options=(--train-limit 10000) ;;  # test images purged: no test yet
  synthetic) source="$base/syntheticMRI"; patches="$source/new_patches_boundary"
       output="$source/transfer_source_k8_25k_loops_20261105"; options=(--train-limit 25000 --val-limit 480) ;;
  topbrain_mr|topbrain_ct) m=${KIND#topbrain_}; T="$base/TopBrain_tight_20261105"
       source="$T/vascular_graph_sources/$P/$m"; patches="$T/vascular_patches/$P/$m"
       output="$T/transfer_splines_k8_loops_20261105/$m"; options=(--train-limit 0 --test) ;;
  *) echo "unknown KIND $KIND" >&2; exit 1 ;;
esac
if [[ "${MODE:-full}" == smoke ]]; then output="${output}_smoke"; options+=(--smoke); fi
"$PY" -u scripts/prepare_transfer_splines.py --source "$source" --patches "$patches" --output "$output" --subdivide-loops "${options[@]}"
if [[ "$KIND" == ixi* ]]; then "$PY" -u scripts/prepare_ixi_monitoring.py --root "$output"; fi
