#!/usr/bin/env bash
#SBATCH --job-name=plant-generalization
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=experiments/spline_learning/logs/plant-generalization-%A_%a.out
#SBATCH --error=experiments/spline_learning/logs/plant-generalization-%A_%a.err
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$GNBM_REPO_DIR/.cache/uv" TORCH_EXTENSIONS_DIR="$GNBM_REPO_DIR/.cache/torch-extensions"
export XDG_CACHE_HOME="$GNBM_REPO_DIR/.cache" MPLCONFIGDIR="$GNBM_REPO_DIR/.cache/matplotlib" TMPDIR=/tmp IDR_DEBUG=WARN WANDB_MODE=offline
unset JOBSCRATCH
if [[ "${LIMITED_DATA_VERIFY:-0}" == 1 ]]; then
    source cluster/jean_zay/limited_runtime.sh
fi
source cluster/jean_zay/env.sh
trap 'if [[ -n "${spline_runtime_dir:-}" ]]; then rm -rf -- "$spline_runtime_dir"; fi' EXIT
source cluster/jean_zay/stage_spline_runtime.sh
if [[ "${LIMITED_DATA_VERIFY:-0}" == 1 ]]; then
    source cluster/jean_zay/ixi_mechanism_runtime.sh
    python - <<'PYCHECK'
from multiprocessing.connection import Listener
with Listener() as listener:
    print('Multiprocessing socket startup check passed:', listener.address)
PYCHECK
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config="${GENERALIZATION_CONFIG:-configs/plant_generalization.yaml}"
mode="${1:?verify, pretrain or finetune}"
if [[ "$mode" == verify ]]; then
    if [[ "${LIMITED_DATA_VERIFY:-0}" == 1 ]]; then
        if [[ -n "${LIMITED_RETRY_FROM:-}" ]]; then
            python - "$config" "$LIMITED_RETRY_FROM" <<'PYARCHIVE'
import json,sys
from pathlib import Path
root=Path(json.loads(Path(sys.argv[1]).read_text())['output_root'])
assert not (root/'SMOKE_PASSED').exists(), 'A successful smoke already exists; review before retry'
assert all(p.name.startswith('smoke-') for p in (root/'runs').iterdir()), 'Training outputs exist; do not restart them'
archive=root/'failed-attempts'/sys.argv[2]
archive.mkdir(parents=True,exist_ok=False)
for folder in (root/'runs').glob('smoke-*'):
    folder.rename(archive/folder.name)
print('Preserved failed smoke outputs:',archive)
PYARCHIVE
        fi
        for arm in pretrain_vessels pretrain_mixed vessels_spline mixed_spline; do
            python -u scripts/train_plant_generalization.py --config "$config" --arm "$arm" --until-epoch 1 --smoke
        done
        python - "$config" <<'PY'
import json,sys,yaml
from pathlib import Path
root=Path(yaml.safe_load(Path(sys.argv[1]).read_text())['output_root'])/'runs'
a=json.loads((root/'smoke-pretrain_vessels/provenance.json').read_text())
b=json.loads((root/'smoke-pretrain_mixed/provenance.json').read_text())
assert a['initial_detector_spline_sha256']==b['initial_detector_spline_sha256']
from scripts.limited_data_support import duplicate_diagnostics
cfg=yaml.safe_load((root/'smoke-mixed_spline/resolved-config.yaml').read_text())
duplicates=duplicate_diagnostics(root/'smoke-mixed_spline/validation-000003/predictions.json',
    cfg['data']['datasets']['synthetic_mri']['root'],'val')
assert duplicates['samples']==4 and 0<=duplicates['duplicate_fraction']<=1
for arm in ('pretrain_vessels','pretrain_mixed','vessels_spline','mixed_spline'):
    run=root/('smoke-'+arm)
    summary=json.loads((run/'summary.json').read_text())
    assert summary['steps']==3 and summary['exposure']['samples_processed']==96
    assert summary['exposure']['unique_samples_encountered']['synthetic_mri']<=200
    for name in ('best.pt','latest.pt'):(run/name).unlink(missing_ok=True)
(root.parent/'SMOKE_PASSED').write_text('Four actual batch-32 forward/backward arms; shared initializer and exposure counters verified.\n')
print('Limited-data smoke passed; disposable startup weights removed.')
PY
        exit 0
    fi
    if [[ "${SOURCE_AUGMENTATION_CHECKS:-0}" == 1 ]]; then
        PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/vessel/test_source_augmentation.py tests/vessel/test_plant_spline_projection.py
    fi
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/vessel/test_spline_augmentation.py tests/vessel/test_plant_contraction.py
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_domain_adaptation.py -k "reference_domain_loss_is_in_total_and_reaches_features or domain_config_requires_mixture_and_optimized_loss"
    for arm in pretrain_mixed pretrain_vessels mixed_straight mixed_spline; do
        python -u scripts/train_plant_generalization.py --config "$config" --arm "$arm" --until-epoch 1 --smoke
    done
    if [[ "${SOURCE_AUGMENTATION_CHECKS:-0}" == 1 ]]; then
        # All four smoke arms have finished; their startup weights are no longer needed.
        python - "$config" <<'PY'
import sys,yaml
from pathlib import Path
root=Path(yaml.safe_load(Path(sys.argv[1]).read_text())['output_root'])/'runs'
for arm in ('pretrain_mixed','pretrain_vessels','mixed_straight','mixed_spline'):
    run=root/('smoke-'+arm)
    assert (run/'summary.json').is_file()
    for name in ('best.pt','latest.pt'):
        (run/name).unlink(missing_ok=True)
print('Verified smoke metadata retained; disposable startup weights removed.')
PY
    fi
elif [[ "$mode" == pretrain ]]; then
    arms=(pretrain_mixed pretrain_vessels)
    python -u scripts/train_plant_generalization.py --config "$config" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
elif [[ "$mode" == finetune ]]; then
    arms=(mixed_straight mixed_spline vessels_straight vessels_spline)
    python -u scripts/train_plant_generalization.py --config "$config" --arm "${arms[$SLURM_ARRAY_TASK_ID]}" --until-epoch "${2:?epoch}"
else
    exit 2
fi
