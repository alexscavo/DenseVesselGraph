#!/usr/bin/env bash
# Backbone comparison at the long stage-1 schedule on the simplified GT (decisions of 11 Nov 2026). Run from the repo root.
# Stage 1: about 100 epochs capped at 60k updates = IXI 60k, BRAVA 60k, TopBrain CTA 30k. Stage 2: reference recipe
# (7.5k updates, lr 3e-4, warm start; defaults = one K24 head, arc penalty 0.1, output compression 0.5 voxel).
# The U-Net stage 1 at this length already exists: spline6_x4_r1110 (IXI, BRAVA) and spline6_x2_r1110 (TopBrain CTA).
#
# usage: submit_backbones_long_20261111.sh stage1 <backbone> <entry>...      (+ its stage 2, chained afterok)
#        submit_backbones_long_20261111.sh unet_stage2 <entry>...            (stage 2 on the existing long U-Net stage 1)
#        submit_backbones_long_20261111.sh unet_stage2_x2 <entry>...         (same with 15k updates: stage-2 length check)
# AFTER=<job id> delays a submission until that job has ended (keeps about 8 jobs running).
set -euo pipefail
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
updates() { case "$1" in TopBrainCTA-simp2) echo 30000 ;; *) echo 60000 ;; esac; }
unet_stage1() { case "$1" in TopBrainCTA-simp2) echo spline6_x2_r1110 ;; *) echo spline6_x4_r1110 ;; esac; }
dep="${AFTER:+--dependency=afterany:$AFTER}"
what="$1"; shift
case "$what" in
  stage1)
    bb="$1"; shift
    for d in "$@"; do
      u=$(updates "$d")
      case "$bb:$u" in unet_wide:60000) t1=06:30:00 ;; unet_wide:30000) t1=03:00:00 ;; dsc3_slim:60000) t1=11:30:00 ;; dsc3_slim:30000) t1=05:45:00 ;;
                       dsc3:60000) t1=13:00:00 ;; dsc3:30000) t1=06:15:00 ;; esac
      case "$bb" in unet_wide) t2=03:00:00 ;; dsc3_slim) t2=03:30:00 ;; dsc3) t2=03:45:00 ;; esac
      s1=$(sbatch --parsable --job-name="dg-$d-$bb-long" --time=$t1 $dep cluster/jean_zay/dense_graph.slurm "$d" \
             --backbone "$bb" --updates "$u" --run-name "spline6_${bb}_long_r1111" --eval-every $((u / 6)))
      s2=$(sbatch --parsable --job-name="dt-$d-$bb-long" --time=$t2 --dependency="afterok:$s1" cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --backbone "$bb" --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/spline6_${bb}_long_r1111/model.pt" \
             --run-name "traced_${bb}_long_r1111" --eval-every 1500)
      echo "$d $bb: stage 1 $s1 ($u updates) -> stage 2 $s2"
    done ;;
  unet_stage2)
    for d in "$@"; do
      echo "$d unet stage 2: $(sbatch --parsable --job-name="dt-$d-unet-long" --time=02:50:00 $dep cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/$(unet_stage1 "$d")/model.pt" --run-name traced_unet_long_r1111 --eval-every 1500)"
    done ;;
  unet_stage2_x2)
    for d in "$@"; do
      echo "$d unet stage 2 x2: $(sbatch --parsable --job-name="dt-$d-unet-long-x2" --time=05:00:00 $dep cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --updates 15000 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/$(unet_stage1 "$d")/model.pt" --run-name traced_unet_long_x2_r1111 --eval-every 3000)"
    done ;;
  *) echo "unknown: $what" >&2; exit 2 ;;
esac
