## R07 reviewed conclusions and recovery record — 27 October 2026

**Completed: a small precision trade-off versus continued S0, without improved relation ranking or a net gain over original A-xyz.** S1 passes neither the predeclared discrimination/true-edge gate nor the original-checkpoint comparison. Retain the original checkpoint. No fully unfrozen confirmation, R08 or combined R06/N08 mechanism was launched.

1. **Are reliable difficult negatives missing from the current objective? No.** The 9,999-pair cap exceeds all unordered pairs from 120 queries, and the actual frozen-cohort audit confirms every matched negative enters S0. Only 18/28,494 training negatives per seed (0.063%) meet the conservative reliability gate; 92.25% have GT chord >=20 mm. Counts alone do not establish gradient dominance or prove inadequate supervision. S1 tests relative multiplicity, not newly discovered negatives.

2. **Does better sampling improve discrimination and reconstruction? Only a small operating-point trade-off was measured.** At fixed 0.5, S1−S0 Edge F1 is +0.000755 / +0.000684, with subject intervals above zero. It removes 25/19 FP but also loses 3/1 TP. Those losses are much smaller than the FP reductions, so this is not the earlier aggressive-mining collapse; nevertheless it fails the predefined no-TP-loss requirement. Conditional AP decreases slightly, and all-retained AP decreases in both seeds with intervals below zero. Small Branch-F1 gains versus S0 remain below the original checkpoint; Curve-F1 differences versus S0 include zero. Original A-xyz remains better on primary Edge F1, conditional/all-retained AP, Branch F1 and long-edge recall.

3. **Are the remaining errors primarily selection/representation rather than supervision? The endpoint budget remains dominant, but this is not a causal proof.** Earlier attribution put about 90.7–90.8% of missing GT edges in absent recovered-endpoint pairs, which fixed-node R07 cannot repair. The new audit separately identifies Hungarian-versus-spatial identity disagreements. Together these support prioritizing spatial quality, assignment/alternative-query label coherence and training-to-inference candidate coverage over more of this tiny matched-negative subset. They do not establish representation as the sole cause. N08 already rejected the tested selection heuristics; R07 introduces no selector. This matched-only audit does not establish reliable unmatched-query negatives, so it does not justify launching R08.

### Verified exposure and normalization

| Seed | Effective pair slots | Positive / negative slots (both arms) | Reliable hard slots S0 → S1 | Replaced negative slots |
|---:|---:|---|---|---:|
|364505|2054432|267559 / 1786873|1161 → 4644|3483 (0.1949%)|
|92741|2056026|267764 / 1788262|1160 → 4640|3480 (0.1946%)|

Every positive slot and each graph/batch negative count is preserved after the original ratio balancing (0.15, tolerance 0.02). CE mean reduction, coefficient 6, optimizer and patch order match. The 4x exposure change is real, but the actual replaced fraction is about 0.195%, far below the 25% ceiling because the reliable subset is rare.

### Positional validation metrics, fixed 0.5

| Seed | Arm | 1-mm precision / recall / F1 | 2-mm precision / recall / F1 |
|---:|---|---|---|
|364505|original|0.305132 / 0.310457 / 0.307772|0.628696 / 0.639667 / 0.634134|
|364505|S0|0.302890 / 0.307071 / 0.304966|0.626204 / 0.634848 / 0.630497|
|364505|S1|0.304499 / 0.307592 / 0.306038|0.628078 / 0.634458 / 0.631252|
|92741|original|0.309409 / 0.317750 / 0.313524|0.625412 / 0.642271 / 0.633730|
|92741|S0|0.309084 / 0.315927 / 0.312468|0.623901 / 0.637713 / 0.630732|
|92741|S1|0.309746 / 0.315796 / 0.312742|0.625367 / 0.637583 / 0.631416|

The full table above retains TP/FP/FN, nearby FPR, long-edge recall, graph errors and common-v2 Branch/Curve F1 for all fixed and training-calibrated policies. Common-v2 uses its own 1-mm anchor/curve support and 0.25-mm sampling; it is not the 1/2-mm positional edge protocol.

### Fixed 2-mm AP populations and prevalence

| Seed / split | Conditional pairs / positives / prevalence | All-retained pairs / positives / prevalence | Hungarian pairs / positives / prevalence |
|---|---|---|---|
|364505 / train|18943 / 1425 / 0.075226|27307 / 1425 / 0.052184|30626 / 2132 / 0.069614|
|364505 / val|67858 / 5170 / 0.076189|99280 / 5170 / 0.052075|108808 / 7679 / 0.070574|
|92741 / train|19165 / 1436 / 0.074928|27485 / 1436 / 0.052247|30626 / 2132 / 0.069614|
|92741 / val|68128 / 5185 / 0.076107|100100 / 5185 / 0.051798|108808 / 7679 / 0.070574|

Within each population, sample/query IDs, labels and prevalence are identical across heads. Hungarian nominal labels, spatially recovered endpoint labels and all-retained inference labels are not interchangeable. Full 1-mm AP populations are preserved in `summary.json` and `ranking/*.npz`.

### Discrimination, learning curves and operating points

| Seed | S1−S0 all-retained AP [95% subject CI] | S1−original Edge F1 [95% subject CI] |
|---:|---|---|
|364505|-0.000630 [-0.001151, -0.000126]|-0.002882 [-0.004292, -0.001239]|
|92741|-0.000865 [-0.001258, -0.000426]|-0.002313 [-0.003593, -0.000716]|

At 1k, S1 training conditional AP reaches 0.987217/0.989929 while validation is 0.974666/0.975581; both continuations fit training increasingly well while validation ranking declines. Final S1−S0 conditional AP is −0.000100/−0.000110; the second interval is below zero and the first includes zero. Thus the fixed-cutoff F1 increase is not improved discrimination.

Training-max-F1 and original-training-precision policies were calibrated at **2 mm**, then applied unchanged at both positional tolerances. At max-F1, S1 slightly raises TP and FP together relative to S0; nearby-negative FPR is unchanged. Training-precision constraints need not yield exactly matched validation precision. The saved validation-only PR readouts are diagnostic, not deployable thresholds. Neither policy overturns the original-checkpoint/gate conclusion.

![R07 train/validation learning curves](../../../../gnbm-r07-matched-supervision-20261027/experiments/r07_matched_supervision_20261027/learning-curves.png)

### Hungarian versus spatial identity audit

| Seed / split | Assigned queries | Nearest GT differs from Hungarian GT | Assigned query >2 mm from own GT | Comparable nominal-negative / spatial-positive pairs |
|---|---:|---:|---:|---:|
|364505 / train|2764|241|388|33|
|364505 / val|10080|874|1383|163|
|92741 / train|2764|231|373|44|
|92741 / val|10080|866|1357|129|

Raw spatial identities use the established one-to-one 2-mm maximum-cardinality/minimum-distance association. Pair-disagreement counts condition on both Hungarian-assigned endpoints also having spatial identities. Categories overlap; being unselected by the raw nearest-query association does not make a query invalid. These are correspondence disagreements, not proof of wrong annotations, and no targets were relabelled. Positive arc-length quartiles use training-derived boundaries; the training set contributes 533 positive edges per quartile. Exact counts and cutoffs are in `assignment-spatial-audit.json`.

### Execution recovery and verification

Original smoke 1613828 failed before training because an older import constructed a five-output rather than compact three-output coordinate head; dependent 1613829/1613830 were cancelled. Pinning the exact frozen model package restored strict checkpoint loading without changing architecture or weights. Recovery smoke **1624151** and both tasks in **1624152** completed exit 0. All four S0/S1 runs reached 1,000 updates.

CPU collector **1624153** was stopped after a demonstrated float32 tie-boundary bug was found: a float64 `nextafter` threshold could round back onto a float32 score during `>` decoding. The fix steps down in the actual probability dtype before storing the cutoff. All seven targeted checks passed, including this regression. Corrected collector **1624612** completed exit 0; every baseline-precision target is feasible in the corrected training calibration. Only CPU reporting was repeated; fixed-0.5 predictions and all training weights/slots are unchanged. Original failed logs and code identities are retained in the experiment folder.

The result is limited to 128 training-monitor patches/32 subjects, the same 480 development-validation patches/13 subjects and a fixed 1k frozen-head continuation. Two seeds share a pretrained source. Whole-subject intervals quantify neither source-training uncertainty nor all choices of reliability/sampling policy. No test data, BRAVA, R06 token module, new matcher, node refinement, additional loss or unmatched-query supervision entered this comparison.
