## Completed results

|Arm|Population|Node precision|Node recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|ce_weighted|train|0.4568|0.6002|0.1341|0.0278|0.3661|4.5664|12.9883|
|ce_weighted|val|0.4365|0.5522|0.1189|0.0216|0.3604|4.3750|13.7246|
|ce_equal|train|0.5686|0.4576|0.1248|0.0230|0.2629|4.5859|3.6836|
|ce_equal|val|0.5474|0.4225|0.1165|0.0181|0.2528|4.5449|3.9160|
|focal_equal|train|0.5386|0.3383|0.0869|0.0077|0.1665|5.2695|2.7422|
|focal_equal|val|0.5293|0.3071|0.0767|0.0063|0.1589|4.8145|2.6875|

Paired subject intervals are in `bootstrap.json`; uniqueness and all intermediate training/validation metrics are in the companion CSVs. A vs B isolates class weighting, B vs C isolates focal modulation. Spatial correct-versus-duplicate AUC is diagnostic, not an inference feature. Pooled raw AUC is reported separately from the bootstrap mean per-patch AUC.

This is a short continuation pilot conditioned on a mature focal-edge checkpoint and a common transition to edge CE/ratio upsampling. Inspect learning-curve trends before drawing convergence conclusions. The node loss can change Hungarian assignments and hence which queries represent supervised endpoints; the matching algorithm and edge candidate policy remain identical. Do not interpret higher node AP/recall as improved uniqueness unless duplicate multiplicity and graph false edges also support it. No result here is a test-set estimate.

Next decision: if B improves over A while C does not improve over B, prioritize weighting rather than focal modulation. If C improves uniqueness over B without losing legitimate-node recall, confirm with a longer matched-budget study. If gains occur only on training subjects or curves have not stabilized, investigate generalization or extend the learning-curve budget before rejecting focal loss. No edge-loss or architecture sweep is automatically warranted.

Output: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_node_focal_20261020-114355`.

