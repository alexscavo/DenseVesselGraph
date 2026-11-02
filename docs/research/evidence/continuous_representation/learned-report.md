## New learned geometry results

The detector is frozen and prediction/candidate parity is checked exactly. Hungarian GT-positive diagnostics include misplaced endpoints; the endpoint-neutral diagnostic is evaluation-only. It does not repair inference. B2 here is a separately trained **compression selector**, not a learned bank of K-specific spline heads. Explicit graph splitting and a learned internal-piecewise generator remain pending.

| split   | method                |   branches |   mean_mm |   max_p95_mm |   controls |   endpoint_neutral_mean_mm |   localized_branches |   localized_mean_mm |
|:--------|:----------------------|-----------:|----------:|-------------:|-----------:|---------------------------:|---------------------:|--------------------:|
| train   | K16                   |       2597 |   3.19406 |      13.3296 |   16       |                    1.39565 |                   27 |             1.0649  |
| train   | K24                   |       2597 |   3.1937  |      13.3415 |   24       |                    1.39542 |                   27 |             1.06758 |
| train   | K8                    |       2597 |   3.19711 |      13.4633 |    8       |                    1.40041 |                   27 |             1.06391 |
| train   | compress24_0.5        |       2597 |   3.19507 |      13.4528 |    4.04775 |                    1.3984  |                   27 |             1.0678  |
| train   | learned_K_compression |       2597 |   3.19437 |      13.3415 |    7.10512 |                    1.39656 |                   27 |             1.06718 |
| train   | oracle_K_compression  |       2597 |   3.19405 |      13.3415 |    8.30189 |                    1.39588 |                   27 |             1.06714 |
| val     | K16                   |       5445 |   3.09981 |      13.5747 |   16       |                    1.39997 |                   81 |             0.88752 |
| val     | K24                   |       5445 |   3.10013 |      13.5722 |   24       |                    1.40004 |                   81 |             0.89119 |
| val     | K8                    |       5445 |   3.1009  |      13.597  |    8       |                    1.39998 |                   81 |             0.88934 |
| val     | compress24_0.5        |       5445 |   3.10123 |      13.5927 |    4.036   |                    1.40081 |                   81 |             0.89186 |
| val     | learned_K_compression |       5445 |   3.10077 |      13.5747 |    6.68724 |                    1.40017 |                   81 |             0.89098 |
| val     | oracle_K_compression  |       5445 |   3.10047 |      13.5722 |    7.79063 |                    1.39931 |                   81 |             0.89093 |

### Paired validation-subject differences in endpoint-neutral mean error versus K8

| method                |   paired_subjects |   difference |      low |    high |
|:----------------------|------------------:|-------------:|---------:|--------:|
| K16                   |                 8 |     -1e-05   | -0.00084 | 0.00104 |
| K24                   |                 8 |      0.00012 | -0.00148 | 0.00177 |
| K8                    |                 8 |      0       |  0       | 0       |
| compress24_0.5        |                 8 |      0.00098 | -0.00085 | 0.00299 |
| learned_K_compression |                 8 |      0.00031 | -0.00186 | 0.00277 |
| oracle_K_compression  |                 8 |     -0.00053 | -0.00247 | 0.00191 |

Graph metrics, per-subject tables and complete branch-level files are saved alongside this report. No improvement in connectivity is possible from a frozen geometry head alone.

### Actual training budget

```json
{
  "completed": true,
  "updates": 8000,
  "batch_size": 4,
  "timing": {
    "8": {
      "seconds_per_update": 0.18571161155381966,
      "trainable_parameters": 1577876,
      "smoke_loss": 1.1909348964691162
    },
    "16": {
      "seconds_per_update": 0.1601152940032383,
      "trainable_parameters": 1592708,
      "smoke_loss": 1.0313318967819214
    },
    "24": {
      "seconds_per_update": 0.16169280089282742,
      "trainable_parameters": 1607540,
      "smoke_loss": 1.032149076461792
    }
  },
  "walltime": "02:00:00",
  "detector_sha256": "3055e743e868961125631dbc8032e1b5c096bdc40805bf77d1ef60fef6ce185b",
  "target_seconds": 3600,
  "checkpoints": [
    1000,
    4000,
    8000
  ]
}
```

### Graph reconstruction (frozen nodes and relations)

|Split|Method|Tolerance mm|Node recall|Edge F1|Branch F1|Curve F1|Component MAE|Cycle MAE|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|train|K8|1|0.07473|0.00369|0.00174|0.05669|7.03906|0.45312|
|train|K8|2|0.22395|0.02055|0.00174|0.05669|7.03906|0.45312|
|train|K16|1|0.07473|0.00369|0.00174|0.05700|7.03906|0.45312|
|train|K16|2|0.22395|0.02055|0.00174|0.05700|7.03906|0.45312|
|train|K24|1|0.07473|0.00369|0.00174|0.05703|7.03906|0.45312|
|train|K24|2|0.22395|0.02055|0.00174|0.05703|7.03906|0.45312|
|val|K8|1|0.08005|0.00407|0.00250|0.06050|7.56445|0.41016|
|val|K8|2|0.23470|0.02696|0.00250|0.06050|7.56445|0.41016|
|val|K16|1|0.08005|0.00407|0.00250|0.06064|7.56445|0.41016|
|val|K16|2|0.23470|0.02696|0.00250|0.06064|7.56445|0.41016|
|val|K24|1|0.08005|0.00407|0.00250|0.06087|7.56445|0.41016|
|val|K24|2|0.23470|0.02696|0.00250|0.06087|7.56445|0.41016|
|val|original_pretrained_K8|1|0.08005|0.00407|0.00250|0.05782|7.56445|0.41016|
|val|original_pretrained_K8|2|0.23470|0.02696|0.00250|0.05782|7.56445|0.41016|

![Training curves](evidence/continuous_representation/training-curves.png)