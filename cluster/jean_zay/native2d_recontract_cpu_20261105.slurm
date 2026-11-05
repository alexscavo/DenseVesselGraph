#!/usr/bin/env bash
#SBATCH --job-name=n2d-recontract
#SBATCH --account=vnc@cpu
#SBATCH --partition=cpu_p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=experiments/native2d_recontracted_20261105/logs/%x-%j.out
#SBATCH --error=experiments/native2d_recontracted_20261105/logs/%x-%j.err
# Rebuild native 2D crop targets with crop re-contraction into experiments/native2d_recontracted_20261105/,
# then audit degree-2 nodes and load-check the new manifests with the training dataset class.
set -eo pipefail
export IDR_DEBUG=WARN OMP_NUM_THREADS=1 MPLCONFIGDIR=$SCRATCH/tmp/mpl; cd "$SLURM_SUBMIT_DIR"
module purge; module load arch/h100 pytorch-gpu/py3/2.3.1
PY=/lustre/fsn1/projects/rech/vnc/upz25mj/venvs/vascular-graph-extraction-h100-torch231/bin/python
OUT=experiments/native2d_recontracted_20261105
pids=()
for d in ${DATASETS:-avwide reaver octasynth plants}; do
  $PY -u scripts/prepare_native2d_recontracted_20261105.py $d > $OUT/logs/rebuild-$d-$SLURM_JOB_ID.log 2>&1 & pids+=($!)
done
status=0; for p in "${pids[@]}"; do wait $p || status=1; done
$PY -u scripts/audit_2d_degree_two_20261105.py --root $OUT --out $OUT/audit_2d_degree_two.json
$PY -u - <<'PY'
import json, numpy as np
from pathlib import Path
from scripts.train_native2d_long import StudyDataset
root = Path('experiments/native2d_recontracted_20261105')
for d in ['avwide', 'reaver', 'octasynth', 'plants']:
    for split in ['train', 'validation']:
        ds = StudyDataset(root / d / 'manifest.json', split, augment=True, max_queries=96)
        ids = np.linspace(0, len(ds) - 1, 16, dtype=int)
        for i in ids:
            image, target, row = ds[int(i)]
            assert tuple(image.shape) == (3, 128, 128) and len(target['inserted']) == len(target['nodes']) == row['nodes']
        print('LOADCHECK', d, split, len(ds), 'ok', flush=True)
PY
exit $status
