## D08 training-only operating-point audit — 28 October 2026

**Completed on CPU from saved epoch-300 predictions. No detector inference, optimizer updates or GPU jobs.**

Calibration uses the **160 representative training patches /39 training subjects** already cached, not all 8,768 training patches. The selected cuts are frozen before evaluating **1,796 validation patches /eight disjoint subjects**. Primary selection maximizes micro positional F1 at 2 mm; the same cuts are also evaluated at 1 mm. Validation does not choose a threshold. Both original D08 checkpoints and every saved coordinate/score remain unchanged.

### Node calibration using all 256 raw queries

A fixed 23-cut confidence grid (0…1, listed in `protocol.json`) is evaluated on training only. Highest cutoff wins exact F1 ties. No GT grouping, query replacement, NMS, classifier fitting or changed matching enters inference. Raw0-cut results diagnose all-query coverage and are not a selected full-graph policy.

| Arm | Policy | Node cut | Train node F1 @2 mm | Val node P/R/F1 @2 mm | Val nodes/patch | Val duplicates/patch | GT edges with endpoints recovered |
|---|---|---:|---:|---|---:|---:|---:|
| scratch | normal_0p5 | 0.500 | 0.02282 | 0.24583/0.01342/0.02544 | 0.935 | 0.007 | 27 |
| scratch | training_node_f1 | 0.350 | 0.16683 | 0.12095/0.21292/0.15427 | 30.175 | 0.281 | 1099 |
| scratch | all_raw | 0.000 | 0.06000 | 0.03233/0.48288/0.06061 | 256.000 | 1.364 | 3634 |
| synthetic | normal_0p5 | 0.500 | 0.39449 | 0.46922/0.28643/0.35572 | 10.464 | 0.346 | 1607 |
| synthetic | training_node_f1 | 0.450 | 0.41641 | 0.40828/0.33431/0.36761 | 14.036 | 0.669 | 2230 |
| synthetic | all_raw | 0.000 | 0.08701 | 0.04388/0.65530/0.08225 | 256.000 | 9.898 | 7826 |

Node-only validation differences (train-selected minus 0.5;1,000 paired whole-subject bootstrap draws, seed 281028):

- scratch: node F1 +0.12883,95% interval [0.12268,0.13487]; cut 0.350. This changes proposal counts as well as recovery.
- synthetic: node F1 +0.01190,95% interval [0.01009,0.01359]; cut 0.450. This changes proposal counts as well as recovery.

### Edge calibration on the unchanged originally retained node population

All cached pairs among original node-confidence >0.5 queries are retained as the ranking population. Select either training-max positional edge F1, or the most recalled tied score group meeting the original training precision. Strict `>` and native-float32 `nextafter` preserve whole score ties; seven targeted tests include the earlier R07 threshold-rounding failure mode. All positives/candidate labels are evaluation labels; no relation retraining or negative mining.

| Arm | Policy | Edge cut | Train edge F1 | Val edge P/R/F1 @2 mm | Val TP /FP /FN | Branch F1 | Cycle MAE |
|---|---|---:|---:|---|---|---:|---:|
| scratch | fixed_0p5 | 0.500000000 | 0.00000 | 0.02245/0.00057/0.00110 | 11/479/19435 | 0.00000 | 0.03396 |
| scratch | train_max_f1 | 0.156702042 | 0.00196 | 0.00780/0.00129/0.00221 | 25/3180/19421 | 0.00000 | 1.09298 |
| scratch | train_preserve_precision | 0.156702042 | 0.00196 | 0.00780/0.00129/0.00221 | 25/3180/19421 | 0.00000 | 1.09298 |
| synthetic | fixed_0p5 | 0.500000000 | 0.08989 | 0.10088/0.05122/0.06794 | 996/8877/18450 | 0.00647 | 1.05902 |
| synthetic | train_max_f1 | 0.425374478 | 0.09130 | 0.08648/0.05729/0.06892 | 1114/11767/18332 | 0.00576 | 1.82461 |
| synthetic | train_preserve_precision | 0.497195870 | 0.09116 | 0.10034/0.05153/0.06809 | 1002/8984/18444 | 0.00630 | 1.08129 |

Training precision targets are scratch=0.00000, synthetic=0.13397. Scratch has zero training TP at the original cutoff; its zero precision constraint is **uninformative**, not evidence of a useful precision-controlled operating point. Training precision is not guaranteed on validation.

Paired validation edge-F1 differences versus 0.5 on these identical node candidates:

- scratch/train_max_f1: +0.00110 [0.00068,0.00159], TP change+14, FP change+2701, cycle-MAE change+1.05902.
- scratch/train_preserve_precision: +0.00110 [0.00068,0.00159], TP change+14, FP change+2701, cycle-MAE change+1.05902.
- synthetic/train_max_f1: +0.00098 [0.00017,0.00192], TP change+118, FP change+2890, cycle-MAE change+0.76559.
- synthetic/train_preserve_precision: +0.00015 [-0.00011,0.00044], TP change+6, FP change+107, cycle-MAE change+0.02227.

### Ranking, candidate identity and cache limits

| Arm | Val conditional AP (pairs/positives) | Val all-retained AP (pairs/positives) |
|---|---|---|
| scratch | 0.37041 (433/27) | 0.02390 (6851/27) |
| synthetic | 0.60471 (32880/1607) | 0.17897 (158454/1607) |

Edge threshold changes **do not change AP or relation ranking** on this fixed candidate population. Differences between scratch and transferred AP still have different candidate populations and cannot isolate relation-head quality. Branch F1 uses the established1 mm role-aware contraction/matching convention; baseline branch TP/FP/FN and original positional counts reproduce exactly. All1/2 mm per-patch/per-subject metrics, near-nonadjacent FP rates, signed components/cycles and subject intervals are saved separately.

**Missing data boundary:** D08 archives save raw256-query positions/confidences, but relation scores only for the original>0.5 node population, and K8 controls only for original predicted edges. They contain no full raw-query decoder embeddings. Therefore the node-calibrated graph Edge/Curve F1 and new-edge learned spline geometry are **not recoverable from these caches alone**. Do not combine node-only gains with cached edge-only gains, substitute GT edges/coordinates, fill missing scores, or report straight chords as the learned Curve F1. Node-calibrated recovered endpoint counts above are diagnostic ceilings, not predicted correct edges.

### Interpretation and next gate

scratch: training-selected node threshold changes validation F1 from 0.02544 to 0.15427, precision from 0.24583 to 0.12095, recall from 0.01342 to 0.21292. Edge-only calibration changes F1 from 0.00110 to 0.00221, while FP changes479→3180 and cycle MAE0.03396→1.09298. Assess these trade-offs together; additional recall is not an improved relation representation.
synthetic: training-selected node threshold changes validation F1 from 0.35572 to 0.36761, precision from 0.46922 to 0.40828, recall from 0.28643 to 0.33431. Edge-only calibration changes F1 from 0.06794 to 0.06892, while FP changes8877→11767 and cycle MAE1.05902→1.82461. Assess these trade-offs together; additional recall is not an improved relation representation.

**Decision:** retain the original 0.5/0.5 graph decoder as the reported reference. The transferred node cut 0.45 is a modest node-only calibration candidate, not an established graph improvement. Its validation precision falls 0.46922→0.40828, duplicate candidates increase 621→1,201, and correct node detections increase 8,818→10,292 while annotation-relative FP increase 9,975→14,916. Endpoint-pair recovery rises 1,607→2,230 of 19,446 GT edges (8.26%→11.47% under the same spatial matching convention), still a small fraction of the graph.

For transferred edge-only max-F1 calibration, the gain is **+0.00098** but adds 118 TP and 2,890 FP. Branch F1 falls 0.00647→0.00576; its paired difference interval excludes zero in the negative direction. Cycle MAE rises 1.05902→1.82461. Near-nonadjacent negative-pair FP rate increases 44.51%→57.54% (304→393 FP among 683 unchanged eligible pairs). The training-precision policy stays near the baseline (edge cut 0.49720), has an edge-F1 interval including zero and slightly worse Branch/cycle outcomes. These are operating-point trades, not a connectivity rescue.

Scratch node calibration gives a much larger F1 change but only 12.10% validation node precision, about 30.17 retained nodes/patch and greater count MAE (16.21→17.66). It does not establish a recovered scratch detector. Its edge-only cutoff0.15670 raises conditional recall on only 27 recovered validation GT edges while producing3,180 FP; do not treat that conditional recall as good whole-graph reconstruction.

This audit tests operating points, not new discrimination or a complete recovered BRAVA graph. Incomplete SWC annotations make unmatched/false edges annotation-relative. Confidence filtering is only one possible bottleneck; improved node recall at a lower cutoff can increase ambiguity, false relations and cycles. Neither an architecture nor a longer training run is justified automatically.

If combined graph evaluation is subsequently justified, the smallest missing computation is one frozen epoch-300 replay of the same 160 training and 1,796 validation patches per chosen arm, with inference batch 8 and strict original-cache node/score parity. Retrieve only relations involving newly retained queries and learned K8 geometry for newly accepted edges; select any additional edge cutoff on training only. This is a bounded inference task, not retraining, and **was not launched here**. No test set is needed.

Calibration grid/decisions were saved before validation in `thresholds.json`, `training-node-curves.json` and `protocol.json`. Source/checkpoint identities, eight baseline parity blocks, seven tests, raw summaries,1,000-draw subject intervals and compressed per-patch/per-subject tables are in `experiments/d08_operating_points_20261028/`. Validation subjects are reused development data; uncertainty does not include training-seed variation. The training calibration monitor is 160 patches rather than the full cohort.

```bash
python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation/scripts/report_d08_operating_points.py --show
```
