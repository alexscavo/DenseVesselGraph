#!/usr/bin/env bash
# Long-edge arms of 11 Nov 2026 (stage 2 only, on the long U-Net stage 1 of the simplified entries). Run from the repo root.
#   roles   : node-role head; peaks classified 'intermediate' are removed before tracing (--node-roles)
#   twohop  : pairs separated by one traced node are proposed along the joined traced path (--two-hop; first run 3 Nov
#             on the original GT, where the hair nodes corrupted its labels)
# Reference: traced_unet_long_r1111 (same stage 1, same recipe). usage: submit_long_edge_arms_20261111.sh roles|twohop <entry>...
# AFTER=<job id> delays the submission until that job has ended.
set -euo pipefail
O="$SCRATCH/experiments/gnbm/dense_graph_20261030"
unet_stage1() { case "$1" in TopBrainCTA-simp2) echo spline6_x2_r1110 ;; *) echo spline6_x4_r1110 ;; esac; }
what="$1"; shift
case "$what" in roles) flag=--node-roles; t=03:15:00 ;; twohop) flag=--two-hop; t=04:15:00 ;; *) echo "unknown: $what" >&2; exit 2 ;; esac
for d in "$@"; do
  echo "$d $what: $(sbatch --parsable --job-name="dt-$d-$what-long" --time=$t ${AFTER:+--dependency=afterany:$AFTER} cluster/jean_zay/dense_graph_traced.slurm "$d" \
         --updates 7500 --warmup-gt 0 --lr 3e-4 --init-from "$O/$d/$(unet_stage1 "$d")/model.pt" --run-name "traced_${what}_long_r1111" --eval-every 1500 $flag)"
done
