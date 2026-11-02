## R06 — Relation-specific 552D CNN token enrichment — 27 October 2026

The predeclared gate for a fully unfrozen confirmation did not pass. Retain original A-xyz inference; no fully unfrozen or combined R06+R07 experiment was launched.

**Measured outcome: the CNN branch is used, but this residual does not improve the original modern A-xyz relation predictor.**

1. **Image information is used, without a net baseline gain.** R2 minus shuffled-image all-retained AP is +0.007732 [0.002736, 0.013344] / +0.011343 [0.006843, 0.015530] across the two seeds. Nevertheless, R2 minus original R0 is −0.009691 / −0.006233, with both subject intervals below zero. Image sensitivity is not evidence that R2 improves the original embeddings.

2. **No convincing advantage over the parameter-matched token control.** Final R2 conditional AP is slightly below R1 in both seeds; R2−R1 all-retained AP is +0.002577 / +0.001388 and edge F1 +0.001755 / +0.000629, with all four subject intervals including zero. R2 often outperforms the continued-head control in all-retained AP, but that control also degrades versus R0. Extra trainable capacity/continued supervision does not rescue the original scorer here.

3. **Early gains do not persist.** R2 validation all-retained AP at 100 updates is 0.828191 / 0.822567 versus R0 0.827375 / 0.821966; at 2k it falls to 0.817684 / 0.815733. These small early fluctuations are descriptive and do not establish a validated early stopping rule. Training conditional AP rises to 0.983150 / 0.986247 while validation falls to 0.973575 / 0.974823. Training Hungarian-pair CE decreases while validation CE increases. Training all-retained AP does not consistently improve either. This raises a supervision-population question alongside the observed generalization gap; the experiment does not isolate its cause or show uniformly better training on every inference candidate.

4. **Graph reconstruction deteriorates.** R2−R0 edge F1 is −0.003678 [−0.005682, −0.000958] / −0.002978 [−0.005733, −0.000496]. It loses 37/38 TP, adds 10/removes 9 FP, and increases component MAE by 0.0750/0.0729 and cycle MAE by 0.04375/0.03333. Original node counts, coordinates and classification logits stay unchanged. Mean retained-pair rank correlations with R0 are 0.9846/0.9869 and 421/419 edge decisions change: the embeddings alter ranking, rather than merely applying a common scalar calibration.

**Recommendation:** keep R0 and the negative result. N08 does not support more of the tested selection heuristics. Prioritize a focused review of spatial quality, assignment/alternative-query label coherence and the matched-training-versus-retained-inference population; keep the independently owned R07 study separate and do not repeat its sampler automatically. The present negative result is limited to 128 training patches, this residual architecture and 2k updates; it does not rule out every use of CNN information.

**Decision gate and scope.** N08 completed negatively against count-matched controls. It supplies no deployable query-selection improvement and does not establish an absence of token information. N06 identifies relation-informative alternatives and residual close-nonadjacent confusion, while normal conditional AP is already about 0.978. This leaves a bounded information-content question. The present experiment is explicitly a modern SyntheticMRI token residual, not the older IXI endpoint/chord/corridor edge-logit adapter.

**Fixed comparison.** Both modern A-xyz seeds 364505/92741 at update 12,000; original 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects. Training the new modules on the monitor is a bounded pilot, not full 25k-patch population learning. No test access. All detector/CNN/transformer/node/refinement/spline weights and BN statistics are frozen. Original coordinates and logits are retained exactly in evaluation; replay/cache parity and frozen-state hashes are recorded. Relation tokens stay unchanged.

R0: original relation head. R1: token-only residual. R2: image-conditioned residual using 27 samples around the existing refined coordinates, at offsets −4/0/+4 input voxels in each axis on layer0/relu1. Both residuals produce 552D corrections, have identical active trainable parameter counts, and zero-initialized output. Each uses token/context LayerNorm→64D projection→ReLU, concatenated into a 552D linear output. R1 fills the 1728D context slot with a fixed repetition of token components: no CNN information, same trainable layers, but a rank-limited redundant input; this is not a claim of equal independent input dimensionality. Only the relation branch consumes the corrected embeddings; node classification/coordinates/spline use originals. R3_head continues only the original relation head and is a supervision/capacity reference, not a parameter-matched arm. R2_shuffled permutes local CNN vectors among queries at inference.

All trainable arms use the unchanged matched-pair CE, ratio-positive upsampling 0.15, cap 9999, no unmatched mining, edge weight 6, AdamW peak LR 7e-6, 20-step warmup, weight decay 1e-4, batch eight, 2,000 updates. Identical patch order, random endpoint orientation, labels and balancing are verified with sampling digests. Save 0/100/300/1000/2000; final 2000 is primary, early points descriptive. Inference batch one; node/edge cutoffs stay 0.5. Geometry-bin quartiles use training GT only.

**Primary table: validation at 2 mm, fixed final 2,000 updates.**

| Seed | Arm, final | Conditional AP | All-retained AP | Edge F1 | TP / FP / FN | Conditional recall | β0 / β1 MAE |
|---:|---|---:|---:|---:|---|---:|---|
|364505|R0|0.977632|0.827375|0.634134|4912 / 2901 / 2767|0.950097|0.8833 / 1.4708|
|364505|R1|0.974082|0.815107|0.628700|4874 / 2952 / 2805|0.942747|0.8979 / 1.5083|
|364505|R2|0.973575|0.817684|0.630456|4875 / 2911 / 2804|0.942940|0.9583 / 1.5146|
|364505|R2_shuffled|0.972648|0.809952|0.629176|4850 / 2888 / 2829|0.938104|0.9313 / 1.4417|
|364505|R3_head|0.972622|0.809140|0.630460|4835 / 2824 / 2844|0.935203|0.9979 / 1.3896|
|92741|R0|0.977971|0.821966|0.633730|4932 / 2954 / 2747|0.951205|0.8854 / 1.5417|
|92741|R1|0.975047|0.814345|0.630123|4897 / 2967 / 2782|0.944455|0.9292 / 1.5604|
|92741|R2|0.974823|0.815733|0.630751|4894 / 2945 / 2785|0.943877|0.9583 / 1.5750|
|92741|R2_shuffled|0.973140|0.804391|0.630711|4859 / 2870 / 2820|0.937126|1.0167 / 1.4375|
|92741|R3_head|0.973110|0.803635|0.630101|4848 / 2861 / 2831|0.935005|1.0271 / 1.4646|

| Seed / arm (final, validation 2 mm) | Nearby nonadjacent FP / candidates | Long-arc Q4 TP / eligible / GT | Tortuosity Q4 TP / eligible / GT | Signed edge / component / cycle bias per patch |
|---|---|---|---|---|
|364505 / R0|51 / 109|1440 / 1601 / 1967|1265 / 1387 / 1952|+0.2792 / +0.2542 / +1.4500|
|364505 / R1|51 / 109|1417 / 1601 / 1967|1242 / 1387 / 1952|+0.3063 / +0.2646 / +1.4875|
|364505 / R2|57 / 109|1411 / 1601 / 1967|1248 / 1387 / 1952|+0.2229 / +0.3542 / +1.4937|
|92741 / R0|51 / 112|1425 / 1588 / 1967|1275 / 1404 / 1952|+0.4313 / +0.2729 / +1.5250|
|92741 / R1|49 / 112|1404 / 1588 / 1967|1261 / 1404 / 1952|+0.3854 / +0.3375 / +1.5437|
|92741 / R2|51 / 112|1399 / 1588 / 1967|1260 / 1404 / 1952|+0.3333 / +0.4042 / +1.5583|

R2 does not recover the long/tortuous-edge deficit: long-arc TP falls 1440→1411 / 1425→1399 at identical eligible endpoints; tortuous TP falls 1265→1248 / 1275→1260. Nearby nonadjacent GT negatives are sparse (109/112 pairs), so their 51→57 / 51→51 FP counts should not be overgeneralized. Positive-bin full-GT and conditional recall, with training-defined cutoffs, are saved separately.


AP uses identical candidates and positive prevalence within each seed/split/tolerance; conditional AP includes only spatially recovered endpoints, while all-retained AP also includes unmatched endpoints. Per-arm candidate/positive counts are in `analysis.json`. Full 1/2-mm counts, near-nonadjacent false-positive rates, training-defined long/tortuous-bin recovery, signed graph counts, ranking correlations and decision changes are saved.

| Seed / split, 2 mm | Conditional pairs / positives / prevalence | All-retained pairs / positives / prevalence |
|---|---|---|
|364505 / train|18943 / 1425 / 0.075226|27307 / 1425 / 0.052184|
|364505 / val|67858 / 5170 / 0.076189|99280 / 5170 / 0.052075|
|92741 / train|19165 / 1436 / 0.074928|27485 / 1436 / 0.052247|
|92741 / val|68128 / 5185 / 0.076107|100100 / 5185 / 0.051798|

| Seed | R2 minus reference, final | Conditional AP difference [95% CI] | All-retained AP difference [95% CI] | Edge F1 difference [95% CI] |
|---:|---|---|---|---|
|364505|R0|-0.004057 [-0.004729, -0.003292]|-0.009691 [-0.014400, -0.004973]|-0.003678 [-0.005682, -0.000958]|
|364505|R1|-0.000508 [-0.001648, +0.000521]|+0.002577 [-0.000519, +0.006328]|+0.001755 [-0.001167, +0.004449]|
|364505|R3_head|+0.000953 [-0.000435, +0.002423]|+0.008544 [+0.004563, +0.013000]|-0.000004 [-0.002201, +0.002706]|
|364505|R2_shuffled|+0.000927 [-0.000655, +0.002543]|+0.007732 [+0.002736, +0.013344]|+0.001280 [-0.001630, +0.004844]|
|92741|R0|-0.003148 [-0.004100, -0.002145]|-0.006233 [-0.011347, -0.001774]|-0.002978 [-0.005733, -0.000496]|
|92741|R1|-0.000224 [-0.001158, +0.000672]|+0.001388 [-0.002415, +0.005347]|+0.000629 [-0.001945, +0.003340]|
|92741|R3_head|+0.001713 [+0.000367, +0.003080]|+0.012098 [+0.008613, +0.015529]|+0.000650 [-0.002013, +0.003012]|
|92741|R2_shuffled|+0.001683 [+0.000204, +0.003106]|+0.011343 [+0.006843, +0.015530]|+0.000040 [-0.003258, +0.002777]|

Intervals resample whole validation subjects (1,000 graph draws; 500 AP draws, seed 271026). These reused development cohorts share a source model; intervals omit source-training and seed uncertainty. No validation threshold or checkpoint selection. Same-population AP changes test ranking beyond a common monotonic logit calibration; graph F1/TP/FP/topology are separate requirements.

Slurm smoke **1613831**, sequential training array **1613949**, CPU collector **1613951**. R06 used at most one H100 at a time under the shared submission lock, with a global limit of two.

![R06 learning curves](../../../../gnbm-r06-relation-token-20261027/experiments/r06_relation_tokens_20261027/learning-curves.png)

Artifacts: sibling worktree `gnbm-r06-relation-token-20261027`, branch `research/r06-relation-token-20261027`, `experiments/r06_relation_tokens_20261027`. Its `large` symlink points to isolated SCRATCH features/scores/checkpoints. `plan.json`, `submission.json`, `smoke.json`, `analysis.json`, `checkpoint-manifest.json`, `ranking/*.npz`, per-patch/per-subject tables and source snapshots preserve provenance. Original A-xyz checkpoint hashes are unchanged: `a1210fe958dffe0e588d38222cb62bf48b58e9b238fc57c027ec1c7a795277c9` / `9646b7b266dec3463d94d16cfbcef0332044a9bf0d599c2319a2eff2c93d3e53`.

The initial conclusion is limited to this frozen-head residual, training cohort and short schedule. If negative, preserve existing query-selection negatives: N08 does not justify more heuristics. The better-supported next question is assignment/spatial-quality/connectivity supervision, keeping R07 independent; neither arbitrary embedding capacity nor blanket unmatched-negative mining is established as a cure.
