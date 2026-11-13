#!/usr/bin/env bash
# Journal open problems B1 (node recall) and B5 (training length), 10 Nov 2026. Run from the repo root.
#
# Fix arm for B1: both stages trained on the GT simplified at 2 mm (plan entries <dataset>-simp2,
# scripts/simplify_graph_caches_20261110.py). Recipe = the reference recipe (stage 1: 15k updates, lr 1e-3; stage 2:
# 7.5k updates, lr 3e-4, warm start, arc penalty 0.1) with one K24 head (--k-rule fixed24, option (a) of the
# number-of-controls decision). The reference trained on the original GT with the same recipe is
# <dataset>:traced_k24_r1109.
# B5: stage 1 with 2x and 4x the updates on the same entries, learning curves from --eval-every.
#
# usage: submit_gt_simplification_20261110.sh fix|x2|x4|stage2x2|k32 <entry> [<entry> ...]
set -euo pipefail
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
what="$1"; shift
for d in "$@"; do
  case "$what" in
    fix)
      s1=$(sbatch --parsable --job-name="dg-$d-r1110" --time=01:45:00 cluster/jean_zay/dense_graph.slurm "$d" \
             --run-name spline6_r1110 --eval-every 2500)
      s2=$(sbatch --parsable --job-name="dt-$d-k24-r1110" --time=02:40:00 --dependency="afterok:$s1" cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/spline6_r1110/model.pt" --run-name traced_k24_r1110 \
             --k-rule fixed24 --eval-every 1500)
      echo "$d fix: stage 1 $s1 -> stage 2 $s2" ;;
    x2)
      echo "$d x2: $(sbatch --parsable --job-name="dg-$d-x2" --time=02:45:00 cluster/jean_zay/dense_graph.slurm "$d" \
             --updates 30000 --run-name spline6_x2_r1110 --eval-every 5000)" ;;
    x4)
      echo "$d x4: $(sbatch --parsable --job-name="dg-$d-x4" --time=04:45:00 ${AFTER:+--dependency=afterany:$AFTER} cluster/jean_zay/dense_graph.slurm "$d" \
             --updates 60000 --run-name spline6_x4_r1110 --eval-every 10000)" ;;
    stage2x2)  # stage 2 with 2x the updates on the 1x stage 1 (run after the fix arm)
      echo "$d stage2 x2: $(sbatch --parsable --job-name="dt-$d-k24-x2" --time=04:45:00 cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --updates 15000 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/spline6_r1110/model.pt" --run-name traced_k24_x2_r1110 \
             --k-rule fixed24 --eval-every 3000)" ;;
    k32)  # stage 2 with one K32 head on the 1x stage 1 (number-of-controls arm; AFTER=<stage-1 job id> to chain)
      echo "$d k32: $(sbatch --parsable --job-name="dt-$d-k32-r1110" --time=02:40:00 ${AFTER:+--dependency=afterok:$AFTER} cluster/jean_zay/dense_graph_traced.slurm "$d" \
             --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/spline6_r1110/model.pt" --run-name traced_k32_r1110 \
             --k-rule fixed32 --eval-every 1500)" ;;
    *) echo "unknown: $what" >&2; exit 2 ;;
  esac
done
