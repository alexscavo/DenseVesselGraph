#!/usr/bin/env bash
# Decoder ceiling by overfitting 10 training patches (12 Nov 2026; scripts/overfit_ceiling_20261112.py). Run from the repo root.
# Entries '<entry>-overfit10' (train = monitor = validation = the same 10 training patches), no augmentation, U-Net wide,
# stage-1 recipe defaults, stage-2 reference recipe (warm start, --warmup-gt 0, lr 3e-4; K24 head, compression 0.5 voxel).
# Stage 2 waits for its stage 1 with a single afterok.
#
# usage: submit_overfit_ceiling_20261112.sh smoke <entry>                (300 + 300 updates, curve every 100)
#        submit_overfit_ceiling_20261112.sh full <entry>...              (U1 stage-1 and U2 stage-2 updates, 20 curve points)
#        env: U1 (default 20000), U2 (default 7500), T1 / T2 walltimes (default 01:45:00 / 01:30:00);
#             ONLY=1 submits stage 1 only, ONLY=2 stage 2 only (after job DEP if set), to stay within ~8 queued jobs
# Schedule (12 Nov 2026): smoke on IXI-loops-overfit10 measured 0.23 s/update (stage 1) and 0.50 s/update (stage 2).
set -euo pipefail
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
what="$1"; shift
case "$what" in
  smoke) u1=300; u2=300; e1=100; e2=100; t1=00:25:00; t2=00:30:00; tag=smoke_r1112 ;;
  full)  u1=${U1:-20000}; u2=${U2:-7500}; e1=$((u1 / 20)); e2=$((u2 / 20)); t1=${T1:-01:45:00}; t2=${T2:-01:30:00}; tag=r1112 ;;
  *) echo "unknown: $what" >&2; exit 2 ;;
esac
only="${ONLY:-}"
for d in "$@"; do
  s1="${DEP:-}"
  [ "$only" = 2 ] || s1=$(sbatch --parsable --job-name="dg-$d-$tag" --time=$t1 cluster/jean_zay/dense_graph.slurm "$d" \
         --backbone unet_wide --no-augment --updates "$u1" --run-name "spline6_unet_wide_overfit_$tag" --eval-every "$e1")
  s2=-
  [ "$only" = 1 ] || s2=$(sbatch --parsable --job-name="dt-$d-$tag" --time=$t2 ${s1:+--dependency="afterok:$s1"} cluster/jean_zay/dense_graph_traced.slurm "$d" \
         --backbone unet_wide --no-augment --updates "$u2" --warmup-gt 0 --lr 3e-4 \
         --init-from "$O/$d/spline6_unet_wide_overfit_$tag/model.pt" --run-name "traced_unet_wide_overfit_$tag" --eval-every "$e2")
  echo "$d: stage 1 $s1 ($u1 updates, $t1) -> stage 2 $s2 ($u2 updates, $t2)"
done
