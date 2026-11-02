## R07 — modern SyntheticMRI matched-pair negative redistribution — 27 October 2026

Completed frozen-detector relation-only diagnostic. N08 remained negative/mixed and no query-selection mechanism was changed. Same two A-xyz step-12,000 sources; 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects per seed. One 1,000-update relation-only continuation per policy and source; fixed early checkpoints 100/300/1,000. No test data, unmatched mining, image adapter or unfrozen confirmation.

The existing 9,999-pair cap already includes every Hungarian-matched negative on these patches. S1 tests relative negative multiplicity, not missing binary exposure: preserve all positive slots and exact per-graph/batch post-ratio-balancing negative counts; replace at most 25% of negative slots with copies of reliable <5-mm nonadjacent pairs, bounded at 4x their S0 multiplicity. Reliability: each Hungarian query within 2 mm of its own nearest GT anchor, with at least 0.5-mm margin to the next GT anchor. All GT-based gates are training supervision/offline evaluation only. Existing CE, edge weight 6, ratio 0.15/tolerance 0.02 and random endpoint orientation are unchanged.

| Seed | Scorer | Step | Validation policy | Edge F1 2mm | TP / FP / FN | Conditional AP 2mm | Near-negative FPR | Long-edge recall | Branch / Curve F1 | β1 MAE |
|---|---|---:|---|---:|---|---:|---:|---:|---|---:|
|92741|S0|1000|baseline_precision|0.62613|4938/3156/2741|0.97569|0.4821|0.7239|0.2737/0.6662|1.8687|
|92741|S0|1000|fixed|0.63073|4897/2952/2782|0.97569|0.4464|0.7138|0.2745/0.6668|1.5833|
|92741|S0|1000|max_f1|0.63840|4593/2117/3086|0.97569|0.3125|0.6396|0.2680/0.6559|0.6458|
|92741|S1|1000|baseline_precision|0.62659|4935/3138/2744|0.97558|0.4643|0.7245|0.2736/0.6655|1.8521|
|92741|S1|1000|fixed|0.63142|4896/2933/2783|0.97558|0.4286|0.7138|0.2750/0.6670|1.5583|
|92741|S1|1000|max_f1|0.63863|4640/2212/3039|0.97558|0.3125|0.6512|0.2713/0.6598|0.7063|
|92741|original|0|baseline_precision|0.63438|4931/2936/2748|0.97797|0.4554|0.7239|0.2792/0.6692|1.5146|
|92741|original|0|fixed|0.63373|4932/2954/2747|0.97797|0.4554|0.7245|0.2788/0.6692|1.5417|
|92741|original|0|max_f1|0.64274|4754/2360/2925|0.97797|0.3036|0.6792|0.2775/0.6656|0.8417|
|364505|S0|1000|baseline_precision|0.62908|4898/2995/2781|0.97477|0.4954|0.7265|0.2700/0.6697|1.6146|
|364505|S0|1000|fixed|0.63050|4875/2910/2804|0.97477|0.4771|0.7184|0.2699/0.6703|1.5021|
|364505|S0|1000|max_f1|0.63878|4773/2492/2906|0.97477|0.3761|0.6904|0.2704/0.6692|0.9667|
|364505|S1|1000|baseline_precision|0.62919|4903/3003/2776|0.97467|0.4954|0.7295|0.2705/0.6690|1.6313|
|364505|S1|1000|fixed|0.63125|4872/2885/2807|0.97467|0.4495|0.7189|0.2712/0.6702|1.4625|
|364505|S1|1000|max_f1|0.63903|4786/2514/2893|0.97467|0.3761|0.6945|0.2718/0.6701|1.0021|
|364505|original|0|baseline_precision|0.63429|4904/2880/2775|0.97763|0.4587|0.7300|0.2743/0.6704|1.4458|
|364505|original|0|fixed|0.63413|4912/2901/2767|0.97763|0.4679|0.7321|0.2740/0.6709|1.4708|
|364505|original|0|max_f1|0.64325|4780/2403/2899|0.97763|0.3394|0.7006|0.2759/0.6703|0.8625|

Seed 364505, S1−S0 at fixed cutoff: Edge F1 {'delta': 0.0007549822393000394, 'ci95': [0.0002334660254226767, 0.0013581403056465852], 'subjects': 13, 'draws': 1000}; TP {'delta': -3.0, 'ci95': [-10.0, 3.0], 'subjects': 13, 'draws': 1000}; FP {'delta': -25.0, 'ci95': [-34.0, -16.0], 'subjects': 13, 'draws': 1000}; conditional AP {'seed': 364505, 'split': 'val', 'tolerance': 2.0, 'population': 'conditional', 'comparison': 'S1 minus S0 AP', 'delta': -0.00010044800840158796, 'ci95': [-0.00022888565194880973, 1.4747737628906241e-05], 'subjects': 13, 'draws': 500}.

Seed 92741, S1−S0 at fixed cutoff: Edge F1 {'delta': 0.0006844616735348641, 'ci95': [9.153004214462469e-05, 0.001255298863892609], 'subjects': 13, 'draws': 1000}; TP {'delta': -1.0, 'ci95': [-3.0, 0.0], 'subjects': 13, 'draws': 1000}; FP {'delta': -19.0, 'ci95': [-33.0, -4.0], 'subjects': 13, 'draws': 1000}; conditional AP {'seed': 92741, 'split': 'val', 'tolerance': 2.0, 'population': 'conditional', 'comparison': 'S1 minus S0 AP', 'delta': -0.00010950947327237337, 'ci95': [-0.00017497864450132238, -4.850882614635683e-05], 'subjects': 13, 'draws': 500}.

S1 does not pass the predefined two-seed ranking/F1/TP gate. Do not promote the sampler or launch unfrozen/R08 training automatically.

Interpretation: current sampling does not omit these matched negatives. This study tests a bounded change in their relative loss contribution. The prior attribution assigned about 90.7–90.8% of missed GT edges to absent recovered endpoint pairs; that historical budget still bounds relation-only recovery here because node proposals remain fixed. This experiment cannot causally establish that query representations are the sole remaining problem. Potential false-negative flags in the audit reflect conflicting spatial identities, not proof of incorrect annotations.

Only 18 reliable close negatives occur in the 128 training snapshots per seed (17 patches), so this is a limited-data intervention. Frozen clean eval-mode features do not reproduce augmented train-mode query distributions. Two continuation seeds share source history; development validation is reused. Whole-subject intervals do not quantify source-training uncertainty. Reported same-population AP and PR diagnostics must not be compared across differing candidate prevalence. Long edges use the training arc-length upper-quartile boundary.

Operating points: fixed 0.5, training-max positional F1 and training-selected threshold meeting the original training precision; thresholds are applied unchanged to validation. Validation-only PR comparisons at the original precision/recall are explicitly diagnostic. Original node threshold remains 0.5. Common-v2 uses 1-mm anchor/geometry support with 0.25-mm arc sampling, independently of 1/2-mm positional matching. Curves use the unchanged frozen K8/CNN geometry for every retained pair.

Detailed early learning curves, training/validation AP, 1/2-mm positional metrics, pair counts/prevalence, per-subject results, graph biases, loss/multiplicity histories and whole-subject uncertainty are saved in `../gnbm-r07-matched-supervision-20261027/experiments/r07_matched_supervision_20261027`. Checkpoint/cache/source identities and accepted jobs are in `plan.json`, `code-identities.json` and `submission.json`.

Retrieve from projects: `python3 gnbm-r07-matched-supervision-20261027/scripts/show_r07.py`. No subsequent experiment is automatically authorized.
