#!/usr/bin/env bash
# Wider node-heatmap target (13 Nov 2026). Run from the repo root.
# Stage 1 fine-tuned from the long U-Net-wide run spline6_unet_wide_long_r1111 of the entry, with the Gaussian target width
# --sigma S (voxels; 1.5 = control = the width of every earlier run) and the training decoding window --nms 3.
# Recipe: 6000 updates, batch 8, augmentation on, --warmup-gt 0 (warm start, as stage 2), peak lr 3e-4 (OneCycle as every run: 5% warm-up from lr/25, cosine to
# lr/25e4), curve point at step 0 (warm-start check) and every 500 updates; after training, the final evaluation at the
# training window 3 (final.json) and again at windows 5 and 7 (final-nms5.json, final-nms7.json; thresholds re-selected
# on the training monitor for each window).
#
# usage: submit_sigma_finetune_20261113.sh smoke <entry> <sigma>...     (300 updates, curve every 100)
#        submit_sigma_finetune_20261113.sh full <entry> <sigma>...      (6000 updates, curve every 500)
#        env: T (walltime; defaults below); FLIP=1 adds random axis flips to the augmentation (--flip; name suffix _flip)
# Run names: spline6_unet_wide_ft_s<sigma without dot>[_smoke]_r1113, e.g. sigma 2.5 -> spline6_unet_wide_ft_s25_r1113.
set -euo pipefail
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
what="$1"; d="$2"; shift 2
case "$what" in
  smoke) u=300; e=100; t=${T:-00:40:00}; tag=_smoke ;;
  full)  u=6000; e=500; t=${T:-01:05:00}; tag= ;;
  *) echo "unknown: $what" >&2; exit 2 ;;
esac
for s in "$@"; do
  fl=""; [ "${FLIP:-0}" = 1 ] && fl=_flip
  name="spline6_unet_wide_ft_s${s//./}${fl}${tag}_r1113"
  [ -e "$O/$d/$name" ] && { echo "exists, not overwriting: $O/$d/$name" >&2; exit 3; }
  j=$(sbatch --parsable --job-name="dg-$d-ft-s${s//./}$fl$tag" --time=$t cluster/jean_zay/dense_graph.slurm "$d" \
        --init-from "$O/$d/spline6_unet_wide_long_r1111/model.pt" --sigma "$s" --nms 3 --extra-nms 5 7 \
        ${fl:+--flip} --updates "$u" --batch 8 --lr 3e-4 --warmup-gt 0 --run-name "$name" --eval-every "$e" --eval-at-start)
  echo "$d sigma $s: $j -> $O/$d/$name ($u updates, $t)"
done
