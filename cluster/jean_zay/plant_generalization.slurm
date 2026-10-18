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
source cluster/jean_zay/env.sh
source cluster/jean_zay/stage_spline_runtime.sh
trap 'rm -rf -- "$spline_runtime_dir"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
config="${GENERALIZATION_CONFIG:-configs/plant_generalization.yaml}"
mode="${1:?verify, pretrain or finetune}"
if [[ "$mode" == verify ]]; then
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
