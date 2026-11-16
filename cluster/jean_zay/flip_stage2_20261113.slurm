#!/usr/bin/env bash
#SBATCH --job-name=flip-stage2
#SBATCH --account=vnc@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=experiments/final_try_20261113/logs/%x-%j.out
#SBATCH --error=experiments/final_try_20261113/logs/%x-%j.err
# Stage 2 on the sigma-1.5 stage-1 fine-tunes with / without mirrored-patch augmentation, and flip-averaged node decoding
# (13 Nov 2026). Run from the repo root.
#   smoke:  sbatch flip_stage2_20261113.slurm smoke
#           (decoding_output vs the study's flip_maps on 8 IXI patches; stage-2 --smoke --flip training; evaluate-only
#            --flip-average --nms 5 of it; stage-1 evaluate-only --smoke --flip-average)
#   train:  sbatch flip_stage2_20261113.slurm <entry> control|flip [ref:plain] [ref:fa5] [ref:fa3]
#           ref:* = evaluate-only of traced_unet_wide_long_r1111 first (plain = recorded decoding -> final-v2.json;
#           fa<K> = --flip-average --nms K -> final-v2[-nms<K>]-flipavg.json); then stage 2 with the reference recipe
#           (7.5k updates, warm-up 0, lr 3e-4, curve every 1500) from spline6_unet_wide_ft_s15[_flip]_r1113, the flip arm
#           with --flip; then evaluate-only of the new run with --flip-average --nms 5.
set -eo pipefail
export GNBM_REPO_DIR="$SLURM_SUBMIT_DIR" WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export JOBSCRATCH="$SCRATCH/experiments/gnbm/dense_graph_20261030/runtime/$SLURM_JOB_ID"
trap 'rm -rf -- "$JOBSCRATCH" "${probe_tmp:-$JOBSCRATCH}"' EXIT
source cluster/jean_zay/env.sh
probe_tmp="$SCRATCH/tmp/fs-$SLURM_JOB_ID"; mkdir -p "$probe_tmp"; export TMPDIR="$probe_tmp"
runtime_python="$(python -c 'import sys; print(sys.base_prefix)')/bin/python"
source cluster/jean_zay/stage_spline_runtime.sh
source cluster/jean_zay/ixi_mechanism_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
T() { echo "=== $(date +%T) stage2 $*"; "$runtime_python" -u scripts/dense_graph_traced_20261030.py --output "$O" --workers 10 "$@"; }
REF=traced_unet_wide_long_r1111

if [ "$1" = smoke ]; then
  echo "=== $(date +%T) decoding_output check"
  "$runtime_python" -u - <<'EOF'
import json, os, sys
import torch
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import dense_graph_20261030 as D
import dense_graph_traced_20261030 as T
import node_decode_study_20261113 as S
from node_heatmap_probe_20261030 import PLAN, collate, make_dataset
run = os.environ['SCRATCH'] + '/experiments/gnbm/dense_graph_20261030/IXI-simp2/traced_unet_wide_long_r1111'
final = json.load(open(run + '/final.json'))
T.apply_saved_mode(final)
model = T.TracedDenseGraph(backbone=final['backbone'], feature_channels=final['feature_channels']).cuda()
model.load_state_dict(torch.load(run + '/model.pt', map_location='cuda')['model'])
model.eval()
ds = make_dataset('IXI-simp2', json.loads(PLAN.read_text())['datasets']['IXI-simp2']['monitor_train'][:8], augment=False)
images = collate([ds[i] for i in range(8)])[0].cuda()
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    out = model.forward_image(images)
    D.FLIP_AVERAGE = False
    same = D.decoding_output(model, images, out) is out
    D.FLIP_AVERAGE = True
    dec = D.decoding_output(model, images, out)
    h, f, p = S.flip_maps(model, images, out)
mine = dec[:, :1].sigmoid()
res = dict(off_returns_input=same, max_abs_heat_diff_vs_study=float((mine - h).abs().max()), study_heat_max=float(h.max()),
           other_channels_unflipped=bool(torch.equal(dec[:, 1:], out[:, 1:].float())))
for thr in (0.25, 0.32):
    ours = D.peaks(dec, thr, 5)
    st_unflipped_off = S.decode(h, out[:, 1:4].float(), torch.zeros_like(h[:, 0]), 'raw5', thr)
    st_avg_off = S.decode(h, f, torch.zeros_like(h[:, 0]), 'raw5', thr)
    n = [len(q) for q, _ in ours]
    res[f'thr{thr}'] = dict(nodes=sum(n), nodes_study=sum(len(d['pos']) for d in st_unflipped_off),
                            max_pos_diff_same_offsets=max(float((q - d['pos']).abs().max()) if len(q) else 0.0 for (q, _), d in zip(ours, st_unflipped_off)),
                            mean_pos_shift_study_averaged_offsets=float(torch.cat([(q - d['pos']).norm(dim=-1) for (q, _), d in zip(ours, st_avg_off)]).mean()))
print('CHECK ' + json.dumps(res), flush=True)
EOF
  T --dataset IXI-simp2 --smoke --updates 40 --warmup-gt 0 --lr 3e-4 --eval-every 20 --flip \
    --init-from "$O/IXI-simp2/spline6_unet_wide_ft_s15_flip_r1113/model.pt" --run-name traced_unet_wide_ft_s15_flip_smoke_r1113
  T --dataset IXI-simp2 --smoke --evaluate-only --flip-average --nms 5 --run-name traced_unet_wide_ft_s15_flip_smoke_r1113
  echo "=== $(date +%T) stage1 evaluate-only smoke flip-average"
  "$runtime_python" -u scripts/dense_graph_20261030.py --dataset IXI-simp2 --output "$O" --workers 10 --smoke --evaluate-only \
    --flip-average --run-name spline6_unet_wide_ft_s15_smoke_r1113
  echo "=== $(date +%T) done"
  exit 0
fi

entry="$1"; arm="$2"; shift 2
for r in "$@"; do  # reference evaluations: a failure is reported but does not stop the training
  case "$r" in
    ref:plain) T --dataset "$entry" --evaluate-only --run-name "$REF" || echo "FAILED $r" ;;
    ref:fa*)   T --dataset "$entry" --evaluate-only --flip-average --nms "${r#ref:fa}" --run-name "$REF" || echo "FAILED $r" ;;
  esac
done
case "$arm" in
  control) s1=spline6_unet_wide_ft_s15_r1113; run=traced_unet_wide_ft_s15_r1113; extra=() ;;
  flip)    s1=spline6_unet_wide_ft_s15_flip_r1113; run=traced_unet_wide_ft_s15_flip_r1113; extra=(--flip) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
T --dataset "$entry" --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$entry/$s1/model.pt" --run-name "$run" \
  --eval-every 1500 "${extra[@]}"
T --dataset "$entry" --evaluate-only --flip-average --nms 5 --run-name "$run"
echo "=== $(date +%T) done"
