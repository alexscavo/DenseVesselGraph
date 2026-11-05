#!/usr/bin/env bash
#SBATCH --job-name=topbrain-patches
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
# TopBrain (VARIANT=tight|stripped|full, DATASET_DIR e.g. TopBrain_tight_20261105), one modality (MOD=mr|ct): stage graphs -> 64^3 patches (exact-boundary grid) -> audit/curate
# -> K8 transfer-spline cache with all compatible train/val/test patches. CTA uses a fixed HU window.
set -eo pipefail
: "${MOD:?set MOD=mr or MOD=ct}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
export MPLCONFIGDIR="$SCRATCH/tmp/mpl-$SLURM_JOB_ID"; mkdir -p "$MPLCONFIGDIR"
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1  # torch (imported by the patch audit) needs the module libraries
VARIANT="${VARIANT:-full}"
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
DATA=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets
O=$DATA/${DATASET_DIR:-TopBrain_${VARIANT}_20261104}; POLICY=optimal_radius_0p75x_c095_vedo_original
PATCHES=$O/vascular_patches/$POLICY/$MOD; SOURCES=$O/vascular_graph_sources/$POLICY/$MOD
mkdir -p "$PATCHES"; [ -f "$PATCHES/patient_split.csv" ] || cp "$O/patient_split_$MOD.csv" "$PATCHES/patient_split.csv"
window=(); [ "$MOD" = ct ] && window=(--intensity-window -100 400)
"$PY" -u scripts/extract_vessel_patches.py --dataset topbrain_${VARIANT}_$MOD --dataset-root "$DATA" \
    --manifest "$O/manifest.csv" --centerline-backend vedo_original --sources "$SOURCES" --output "$PATCHES" \
    --split-output "$PATCHES/patient_split.csv" --evidence experiments/topbrain_preparation_20261104/evidence \
    "${window[@]}" --workers 8
"$PY" -u scripts/prepare_transfer_splines.py --source "$SOURCES" --patches "$PATCHES" \
    --output "$O/transfer_splines_k8_${SPLINE_TAG:-20261105}/$MOD" --train-limit 0 --test
