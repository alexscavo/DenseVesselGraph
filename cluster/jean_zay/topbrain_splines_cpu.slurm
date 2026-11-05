#!/usr/bin/env bash
#SBATCH --job-name=topbrain-splines
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=experiments/topbrain_preparation_20261104/logs/%x-%j.out
#SBATCH --error=experiments/topbrain_preparation_20261104/logs/%x-%j.err
# Stamp mm units on the derived TopBrain labels/masks (TopBrain label headers say 'unknown'), then build the
# K8 transfer-spline cache for one modality (MOD) from the existing patches. Voxel data is not changed.
set -eo pipefail
: "${MOD:?set MOD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 IDR_DEBUG=WARN
cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
O=/lustre/fsn1/projects/rech/vnc/upz25mj/datasets/${DATASET_DIR:-TopBrain_tight_20261105}; POLICY=optimal_radius_0p75x_c095_vedo_original
"$PY" - "$O" "$MOD" <<'PYEOF'
import sys, os, glob, numpy as np, nibabel as nib
O, mod = sys.argv[1], sys.argv[2]
n = 0
for f in sorted(glob.glob(f'{O}/labels_{mod}/*.nii.gz') + glob.glob(f'{O}/masks_{mod}/*.nii.gz')):
    img = nib.load(f)
    if img.header.get_xyzt_units()[0] == 'mm':
        continue
    data = np.asanyarray(img.dataobj); header = img.header.copy(); header.set_xyzt_units('mm')
    tmp = f.replace('.nii.gz', '.tmp.nii.gz'); nib.save(nib.Nifti1Image(data, img.affine, header), tmp)
    assert np.array_equal(np.asanyarray(nib.load(tmp).dataobj), data); os.replace(tmp, f); n += 1
print('stamped mm units on', n, 'files', flush=True)
PYEOF
"$PY" -u scripts/prepare_transfer_splines.py --source "$O/vascular_graph_sources/$POLICY/$MOD" \
    --patches "$O/vascular_patches/$POLICY/$MOD" --output "$O/transfer_splines_k8_20261105/$MOD" --train-limit 0 --test
