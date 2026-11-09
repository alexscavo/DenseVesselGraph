#!/usr/bin/env bash
#SBATCH --job-name=tubetk-prep
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/tubetk_preparation_20261105/logs/%x-%j.out
#SBATCH --error=experiments/tubetk_preparation_20261105/logs/%x-%j.err
# TubeTK Skwerer-tree preparation (5 Nov 2026), scripts/prepare_tubetk_skwerer_20261105.py. STEP=
#   smoke    : Normal-008 + Normal-099 volumes/graphs (+ Normal-002: the patch generator needs 3 patients), 3-subject staging, patches, K8 loop cache (--smoke) under $O/smoke
#   subjects : all 97 subjects (completed subjects are reused after their marker check)
#   patches  : subject split (seed 42 + family repair), staging, 64^3 patches, curation
#   targets  : K8 transfer-spline cache, --subdivide-loops, all train/val/test patches, then collect preparation.json
#   collect  : collect preparation.json from the completed target cache without rebuilding it
#   qc       : stripping/overlay figure and the validation-patch 3D HTML
# Export TUBETK_GIT_HEAD=$(git rev-parse HEAD) at submission to retain Git provenance on nodes without Git.
set -eo pipefail
: "${STEP:?set STEP}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
export MPLCONFIGDIR="$SCRATCH/tmp/mpl-$SLURM_JOB_ID"; mkdir -p "$MPLCONFIGDIR"
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
O=${OUTPUT:-/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/TubeTK_skwerer_20261105}; CACHE=transfer_splines_k8_loops_20261105
S=scripts/prepare_tubetk_skwerer_20261105.py
case "$STEP" in
  smoke)
    "$PY" -u $S smoke --output "$O" --workers 4
    "$PY" -u scripts/prepare_transfer_splines.py --source "$O/smoke/sources" --patches "$O/smoke/patches" \
        --output "$O/smoke/${CACHE}_smoke" --subdivide-loops --smoke --test ;;
  subjects) "$PY" -u $S subjects --output "$O" --workers ${WORKERS:-18} ;;
  patches)
    [ -d "$O/sources" ] || "$PY" -u $S stage --output "$O"
    "$PY" -u $S patches --output "$O" --workers ${WORKERS:-16} ;;
  targets)
    "$PY" -u scripts/prepare_transfer_splines.py --source "$O/sources" --patches "$O/patches" --output "$O/$CACHE" \
        --subdivide-loops --train-limit 0 --test
    "$PY" -u $S collect --output "$O" --cache "$CACHE" ;;
  collect) "$PY" -u $S collect --output "$O" --cache "$CACHE" ;;
  qc)
    "$PY" -u scripts/tubetk_stripping_qc_20261105.py "$O"
    "$PY" -u scripts/topbrain_patch_3d_20261105.py --dataset tubetk --root "$O" ;;
  *) echo "unknown STEP $STEP" >&2; exit 1 ;;
esac
