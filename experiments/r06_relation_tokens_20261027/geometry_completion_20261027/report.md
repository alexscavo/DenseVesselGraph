### R06 geometry-metric completion from saved predictions — 27 October 2026

The repeated R06 request matches the completed experiment above; no training, inference, architecture change or unfrozen confirmation was repeated. The missing common-v2 Branch/Curve metrics below reuse R06 final score vectors and compatible original frozen K8 candidate curves cached during R07 preparation. **No R07 trained scores or sampling policy enters this comparison.** Checkpoint hashes, split/sample/GT identities, query-pair ordering, original R0 score parity, original geometry parity and reproduced per-arm node/edge counts passed.

| Seed | Scorer, validation | Branch F1 | Curve F1 | Component MAE | Cycle MAE |
|---:|---|---:|---:|---:|---:|
|364505|R0|0.273976|0.670871|0.8833|1.4708|
|364505|R1|0.268968|0.670032|0.8979|1.5083|
|364505|R2|0.268676|0.670009|0.9583|1.5146|
|364505|R2_shuffled|0.269711|0.669427|0.9313|1.4417|
|364505|R3_head|0.268412|0.670014|0.9979|1.3896|
|92741|R0|0.278753|0.669204|0.8854|1.5417|
|92741|R1|0.274380|0.667373|0.9292|1.5604|
|92741|R2|0.271642|0.665883|0.9583|1.5750|
|92741|R2_shuffled|0.272191|0.667206|1.0167|1.4375|
|92741|R3_head|0.272036|0.666099|1.0271|1.4646|

| Seed | R2 comparison | Branch F1 difference [95% subject CI] | Curve F1 difference [95% subject CI] |
|---:|---|---|---|
|364505|R2 minus R0|-0.005301 [-0.008577, -0.002350]|-0.000863 [-0.003800, +0.002081]|
|364505|R2 minus R1|-0.000292 [-0.003727, +0.003479]|-0.000024 [-0.001642, +0.001397]|
|364505|R2 minus R3_head|+0.000263 [-0.003528, +0.005172]|-0.000005 [-0.001948, +0.002164]|
|364505|R2 minus R2_shuffled|-0.001035 [-0.005363, +0.004409]|+0.000582 [-0.002027, +0.003519]|
|92741|R2 minus R0|-0.007111 [-0.010850, -0.002664]|-0.003321 [-0.006533, -0.000472]|
|92741|R2 minus R1|-0.002738 [-0.006465, +0.000865]|-0.001490 [-0.002796, +0.000008]|
|92741|R2 minus R3_head|-0.000395 [-0.003389, +0.002852]|-0.000216 [-0.002085, +0.001835]|
|92741|R2 minus R2_shuffled|-0.000549 [-0.004643, +0.004239]|-0.001323 [-0.004128, +0.001180]|

These geometry metrics supplement the completed negative discrimination/reconstruction result; they do not change the failed predeclared R06 confirmation gate. Image shuffling shows that CNN information is used, but final conditional AP/F1 are below original R0 in both seeds and there is no convincing R1 advantage. A possible Curve F1 gain must be assessed alongside the existing TP/FP/cycle counts; geometric coverage alone does not demonstrate connectivity improvement. Tiny early AP fluctuations are exploratory; no validation-selected stopping rule or new training is introduced. Retain original A-xyz. The better-supported remaining question concerns spatial identity, assignment and relation-label consistency; N08 and the completed R07 sampler negatives remain binding evidence, not invitations to repeat them or launch blanket unmatched mining.

This remains a 128-training/480-validation-patch, two-seed frozen residual pilot. It cannot rule out all CNN relation features or establish a causal training fix. Matched parameter counts do not make R1 redundant token-context input rank equal to independent CNN input rank. No test subjects were inspected.

CPU reporting job: `1625936`. Evidence: sibling `gnbm-r06-relation-token-20261027/experiments/r06_relation_tokens_20261027/geometry_completion_20261027` (`summary.json`, per-patch/per-subject CSVs, identity manifest and smoke). Retrieve with `python3 gnbm-r06-relation-token-20261027/scripts/show_r06.py` from projects.
