#!/usr/bin/env bash
#SBATCH --job-name=ixi-regen
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=experiments/loop_subdivision_20261105/logs/%x-%j.out
#SBATCH --error=experiments/loop_subdivision_20261105/logs/%x-%j.err
# Regenerate the IXI 64^3 patch grid for the 149 train/val subjects whose original image still exists (the 21 test
# images and the never-read patches were purged from SCRATCH). Same grid, pad, stride and normalization as the
# original production run (extract_vessel_patches.py); then audit + curate as there. Originals are only read.
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN MPLCONFIGDIR=$SCRATCH/tmp/mpl-$SLURM_JOB_ID
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
I=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/IXI; P=optimal_radius_0p75x_c095_vedo_original
SRC=$I/vascular_graph_sources/$P; NEW=$I/vascular_graph_sources/${P}_trainval_20261105; OUT=$I/vascular_patches/${P}_trainval_20261105
mkdir -p $NEW/raw $NEW/seg $NEW/graphs/adaptive $OUT
echo "patient_id,split" > $OUT/patient_split.csv
tail -n +2 "$SLURM_SUBMIT_DIR/experiments/loop_subdivision_20261105/ixi_patient_split_recovered.csv" | while IFS=, read pid split; do
  split=${split%$'\r'}
  [ -e "$SRC/raw/$pid.nii.gz" ] || continue   # image purged: skip (all such subjects are test)
  [ "$split" = test ] && continue
  ln -sfn "$(readlink -f $SRC/raw/$pid.nii.gz)" $NEW/raw/$pid.nii.gz
  ln -sfn "$(readlink -f $SRC/seg/$pid.nii.gz)" $NEW/seg/$pid.nii.gz
  ln -sfn "$(readlink -f $SRC/graphs/adaptive/$pid)" $NEW/graphs/adaptive/$pid
  echo "$pid,$split" >> $OUT/patient_split.csv
done
echo "staged $(($(wc -l < $OUT/patient_split.csv)-1)) subjects"
"$PY" -u scripts/generate_synthetic_mri_dataset.py --root $NEW --output-dir $OUT --split-output $OUT/patient_split.csv \
  --patch-size 64 64 64 --pad 0 0 0 --maximum-stride 40 40 40 --workers 16 --resume --allow-existing-split-proportions
"$PY" -u scripts/audit_real_patch_eligibility.py --patch-root $OUT --output experiments/loop_subdivision_20261105/ixi_regen_audit
"$PY" -u scripts/curate_vessel_training_patches.py --patch-root $OUT --all-splits --apply
# identity check against surviving original patches (same grid => same files)
"$PY" - "$I/vascular_patches/$P" "$OUT" <<'PYEOF'
import sys, glob, os, numpy as np, nibabel as nib
old, new = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f'{old}/val/raw/*_data.nii.gz'))[::50] + sorted(glob.glob(f'{old}/train/raw/*_data.nii.gz'))[::500]
bad = 0
for f in files:
    split = f.split('/')[-3]; name = os.path.basename(f)
    for kind, suffix in (('raw', '_data'), ('seg', '_seg')):
        a = f.replace('/raw/', f'/{kind}/').replace('_data', suffix); b = a.replace(old, new)
        if not os.path.exists(b): b = b.replace(f'/{split}/', f'/excluded_{split}/')
        same = os.path.exists(b) and np.array_equal(np.asanyarray(nib.load(a).dataobj), np.asanyarray(nib.load(b).dataobj))
        bad += not same
print(f'IDENTITY regenerated vs original: {2*len(files)-bad}/{2*len(files)} identical', flush=True)
PYEOF
