#!/usr/bin/env bash
#SBATCH --job-name=exp-archive
#SBATCH --account=vnc@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=experiments/dense_graph_20261030/logs/%x-%j.out
#SBATCH --error=experiments/dense_graph_20261030/logs/%x-%j.err
# Pack superseded experiment folders (one zstd tar each) and the current SCRATCH runs into STORE, then verify.
# Nothing is deleted here; removal from WORK happens only after this log is checked.
set -eo pipefail
DEST="$STORE/DenseVesselGraph-experiments-archive-20261103"
mkdir -p "$DEST"
cd "$SLURM_SUBMIT_DIR/experiments"
for d in afternoon_job_review_20261028 brava_augmentation_20261026 brava_graph_study brava_learning_diagnosis_20261026 brava_long_20261027 brava_representation_20261026 cheap_diagnostics_20261026 combined_pretraining completed_result_review_20261028 continuous_representation coordination d08_operating_points_20261028 fine_feature_architecture graph_error_attribution_20261026 gt_free_query_selection_20261026 ixi_features_nodes ixi_image_relation_learning ixi_local_components_20261026 ixi_mechanisms ixi_node_focal_ablation ixi_node_refinement ixi_overfit ixi_query_diagnostics ixi_relation_isolation ixi_relation_matching_calibration ixi_segmentation_input ixi_supervision_audit ixi_threshold_calibration ixi_transfer leaf_pretraining limited_botanical_transfer local_refinement_long multiscale_pretrained n07_status_audit_20261028 native2d_20261027 native2d_imagenet_20261030 native2d_mixed_20261028 native2d_straight_20261029 node_first_curriculum_20261028 node_verifier_feasibility oracle_representatives_20261026 plant_pretraining q0_ixi_audit_20261028 r06_relation_tokens_20261027 r07_matched_supervision_20261027 source_augmentation spline_learning spline_target_capacity storage_cleanup; do
    tar -cf - "$d" | zstd -q -T4 -3 -o "$DEST/$d.tar.zst"
    packed=$(zstd -dc "$DEST/$d.tar.zst" | tar -tf - | wc -l)
    expected=$(find "$d" | wc -l)
    echo -e "ARCHIVE\t$d\t$(du -sb "$d" | cut -f1)\t$(stat -c %s "$DEST/$d.tar.zst")\t$expected\t$packed\t$([ "$packed" = "$expected" ] && echo OK || echo MISMATCH)"
done
cd "$SCRATCH/experiments/gnbm"
tar -cf - --exclude='dense_graph_20261030/runtime' --exclude='dense_graph_20261030/smoke*' dense_graph_20261030 | zstd -q -T4 -3 -o "$DEST/current-dense-graph-runs-scratch-20261103.tar.zst"
packed=$(zstd -dc "$DEST/current-dense-graph-runs-scratch-20261103.tar.zst" | tar -tf - | wc -l)
expected=$(find dense_graph_20261030 -path dense_graph_20261030/runtime -prune -o -path 'dense_graph_20261030/smoke*' -prune -o -print | wc -l)
echo -e "SCRATCH\tdense_graph_20261030\t$(stat -c %s "$DEST/current-dense-graph-runs-scratch-20261103.tar.zst")\t$expected\t$packed\t$([ "$packed" = "$expected" ] && echo OK || echo MISMATCH)"
echo DONE
