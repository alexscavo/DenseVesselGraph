# IXI bounded mechanism diagnostics

Validation pilot; 1/2 mm positional matching. Common-v2 geometry unchanged.

|Experiment|Policy|Tolerance|Node recall|Node precision|Edge F1|Conditional recall|Branch F1|Curve F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|A|normal|1.0|0.3674|0.3376|0.0296|0.1124|0.0122|0.1686|
|A|previous_oracle|1.0|0.3674|1.0000|0.0372|0.1124|0.0131|0.0335|
|A|raw_oracle|1.0|0.5988|1.0000|0.0636|0.0779|0.0187|0.0535|
|A|normal|2.0|0.5627|0.5171|0.0737|0.1257|0.0122|0.1686|
|A|previous_oracle|2.0|0.5627|1.0000|0.0892|0.1257|0.0127|0.0718|
|A|raw_oracle|2.0|0.8362|1.0000|0.0999|0.0722|0.0194|0.0827|
|C|baseline_continuation|1.0|0.2465|0.2267|0.0172|0.1467|0.0050|0.1179|
|C|both|1.0|0.3674|0.3376|0.0296|0.1124|0.0122|0.1686|
|C|classification_only|1.0|0.2405|0.2211|0.0171|0.1564|0.0056|0.1172|
|C|coordinates_only|1.0|0.3691|0.3319|0.0298|0.1128|0.0116|0.1709|
|C|neither_local_residual|1.0|0.2411|0.2168|0.0170|0.1549|0.0055|0.1182|
|C|baseline_continuation|2.0|0.5118|0.4707|0.0583|0.1243|0.0050|0.1179|
|C|both|2.0|0.5627|0.5171|0.0737|0.1257|0.0122|0.1686|
|C|classification_only|2.0|0.5018|0.4612|0.0582|0.1238|0.0056|0.1172|
|C|coordinates_only|2.0|0.5672|0.5101|0.0730|0.1239|0.0116|0.1709|
|C|neither_local_residual|2.0|0.5041|0.4533|0.0581|0.1236|0.0055|0.1182|
|D_focal_control|focal_control|1.0|0.3674|0.3376|0.0523|0.2860|0.0227|0.3032|
|D_focal_control|focal_control|2.0|0.5627|0.5171|0.1169|0.2867|0.0227|0.3032|
|D_regularized|regularized|1.0|0.3674|0.3376|0.0639|0.4951|0.0360|0.3884|
|D_regularized|regularized|2.0|0.5627|0.5171|0.1410|0.4903|0.0360|0.3884|
|D_hard_unmatched|hard_unmatched|1.0|0.3674|0.3376|0.0659|0.4201|0.0355|0.3355|
|D_hard_unmatched|hard_unmatched|2.0|0.5627|0.5171|0.1381|0.3947|0.0355|0.3355|
|D_hard_matched|hard_matched|1.0|0.3674|0.3376|0.0571|0.3669|0.0241|0.3398|
|D_hard_matched|hard_matched|2.0|0.5627|0.5171|0.1286|0.3708|0.0241|0.3398|

B: confidence.json, discrimination.json, recovery.csv, probes.json and per-subject files under B/.
Paired intervals: bootstrap.json. Long-branch recovery: length-strata.csv. See each experiment for full counts.
D regularized versus hard arms isolates auxiliary objectives. Focal control changes the base objective and is a separate comparison.
Oracle selection uses GT. Reused validation, one seed, cached training subset; no test-set claim.
