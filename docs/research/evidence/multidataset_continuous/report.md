## Completed initial batch

Matched-final development results; no test evaluation. Frozen geometry arms have identical node/relation results by design. BRAVA recovery arms train the full detector.

| dataset      | arm                   |   step |   node_recall |   raw_query_recall |      f1 |   conditional_relation_ap |   conditional_recall |   common_branch_f1 |   common_curve_f1 |   beta0_mae |   beta1_mae |
|:-------------|:----------------------|-------:|--------------:|-------------------:|--------:|--------------------------:|---------------------:|-------------------:|------------------:|------------:|------------:|
| BRAVA        | D1_IXI                |  30000 |       0.34093 |            0.56318 | 0.02099 |                   0.43697 |              0.12413 |            0.00243 |           0.0222  |    14.338   |     0.12416 |
| BRAVA        | D2_SyntheticMRI       |  30000 |       0.27331 |            0.48499 | 0.0134  |                   0.39735 |              0.11524 |            0.00128 |           0.01094 |    13.3786  |     0.08073 |
| BRAVA        | D3_scratch            |  30000 |       0.10557 |            0.31264 | 0.0004  |                   0.32605 |              0.02424 |            0.0001  |           0.001   |    17.0451  |     0.00334 |
| IXI          | IXI_K8                |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12149 |    17.0469  |     0.95898 |
| IXI          | IXI_K16               |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12177 |    17.0469  |     0.95898 |
| IXI          | IXI_K24               |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12205 |    17.0469  |     0.95898 |
| IXI          | IXI_adaptive          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12108 |    17.0469  |     0.95898 |
| IXI          | IXI_piece_C0          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.1217  |    17.0469  |     0.95898 |
| IXI          | IXI_piece_C1          |  20000 |       0.49702 |            0.78179 | 0.06074 |                   0.45437 |              0.1329  |            0.00528 |           0.12216 |    17.0469  |     0.95898 |
| SyntheticMRI | SyntheticMRI_K8       |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58386 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_K16      |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58324 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_K24      |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58377 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.5817  |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58415 |     2.50833 |     1.51875 |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 |       0.78601 |            0.87411 | 0.47013 |                   0.89837 |              0.77472 |            0.1025  |           0.58275 |     2.50833 |     1.51875 |

### Learned geometry

| dataset      | arm                   |   step | split   |   branches |   mean_mm |   max_p95_mm |   neutral_mm |   controls |
|:-------------|:----------------------|-------:|:--------|-----------:|----------:|-------------:|-------------:|-----------:|
| BRAVA        | D1_IXI                |  20000 | train   |       2677 |   2.8055  |     11.3429  |      1.37327 |    8       |
| BRAVA        | D1_IXI                |  20000 | val     |      19446 |   2.9451  |     11.8893  |      1.41613 |    8       |
| BRAVA        | D1_IXI                |  30000 | train   |       2677 |   2.72847 |     11.1136  |      1.3796  |    8       |
| BRAVA        | D1_IXI                |  30000 | val     |      19446 |   2.85363 |     11.7828  |      1.41333 |    8       |
| BRAVA        | D2_SyntheticMRI       |  30000 | train   |       2677 |   2.96554 |     11.8727  |      1.39862 |    8       |
| BRAVA        | D2_SyntheticMRI       |  30000 | val     |      19446 |   3.06706 |     12.1897  |      1.42832 |    8       |
| BRAVA        | D3_scratch            |  30000 | train   |       2677 |   3.59396 |     11.7253  |      1.40795 |    8       |
| BRAVA        | D3_scratch            |  30000 | val     |      19446 |   3.67252 |     12.2193  |      1.43489 |    8       |
| IXI          | IXI_K16               |  20000 | train   |       4244 |   1.54392 |      7.72345 |      0.49838 |   16       |
| IXI          | IXI_K16               |  20000 | val     |      10968 |   1.88311 |      9.03705 |      0.51565 |   16       |
| IXI          | IXI_K24               |  20000 | train   |       4244 |   1.54347 |      7.72788 |      0.49773 |   24       |
| IXI          | IXI_K24               |  20000 | val     |      10968 |   1.88373 |      9.04837 |      0.51682 |   24       |
| IXI          | IXI_K8                |  20000 | train   |       4244 |   1.54723 |      7.72345 |      0.49971 |    8       |
| IXI          | IXI_K8                |  20000 | val     |      10968 |   1.88304 |      9.05911 |      0.51484 |    8       |
| IXI          | IXI_adaptive          |  20000 | train   |       4244 |   1.55349 |      7.72345 |      0.50104 |    4.46843 |
| IXI          | IXI_adaptive          |  20000 | val     |      10968 |   1.88701 |      9.04175 |      0.51119 |    4.31182 |
| IXI          | IXI_piece_C0          |  20000 | train   |       4244 |   1.54354 |      7.72096 |      0.49838 |   29       |
| IXI          | IXI_piece_C0          |  20000 | val     |      10968 |   1.88336 |      9.04946 |      0.51692 |   29       |
| IXI          | IXI_piece_C1          |  20000 | train   |       4244 |   1.54459 |      7.76445 |      0.49872 |   26       |
| IXI          | IXI_piece_C1          |  20000 | val     |      10968 |   1.88329 |      9.03714 |      0.51646 |   26       |
| SyntheticMRI | SyntheticMRI_K16      |  20000 | train   |       4227 |   0.97893 |      5.98441 |      0.36578 |   16       |
| SyntheticMRI | SyntheticMRI_K16      |  20000 | val     |       7679 |   0.97338 |      5.61535 |      0.40907 |   16       |
| SyntheticMRI | SyntheticMRI_K24      |  20000 | train   |       4227 |   0.97859 |      5.99769 |      0.36489 |   24       |
| SyntheticMRI | SyntheticMRI_K24      |  20000 | val     |       7679 |   0.97167 |      5.59466 |      0.4075  |   24       |
| SyntheticMRI | SyntheticMRI_K8       |  20000 | train   |       4227 |   0.98129 |      6.03457 |      0.36406 |    8       |
| SyntheticMRI | SyntheticMRI_K8       |  20000 | val     |       7679 |   0.9715  |      5.6095  |      0.40446 |    8       |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 | train   |       4227 |   0.98254 |      6.02332 |      0.362   |    4.55642 |
| SyntheticMRI | SyntheticMRI_adaptive |  20000 | val     |       7679 |   0.97383 |      5.6092  |      0.40112 |    4.53965 |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 | train   |       4227 |   0.97838 |      5.98931 |      0.36557 |   29       |
| SyntheticMRI | SyntheticMRI_piece_C0 |  20000 | val     |       7679 |   0.97225 |      5.62123 |      0.4085  |   29       |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 | train   |       4227 |   0.98066 |      6.03668 |      0.36712 |   26       |
| SyntheticMRI | SyntheticMRI_piece_C1 |  20000 | val     |       7679 |   0.97359 |      5.61535 |      0.40901 |   26       |

### Paired subject Curve F1 differences

| dataset      | arm                   | reference       |   curve_f1_difference |      low |     high |
|:-------------|:----------------------|:----------------|----------------------:|---------:|---------:|
| IXI          | IXI_K8                | IXI_K8          |               0       |  0       |  0       |
| IXI          | IXI_K16               | IXI_K8          |               0.00028 | -7e-05   |  0.00065 |
| IXI          | IXI_K24               | IXI_K8          |               0.00056 |  0.00023 |  0.00091 |
| IXI          | IXI_adaptive          | IXI_K8          |              -0.00041 | -0.00108 |  0.00041 |
| IXI          | IXI_piece_C0          | IXI_K8          |               0.00021 | -0.00021 |  0.00066 |
| IXI          | IXI_piece_C1          | IXI_K8          |               0.00067 |  8e-05   |  0.00144 |
| SyntheticMRI | SyntheticMRI_K8       | SyntheticMRI_K8 |               0       |  0       |  0       |
| SyntheticMRI | SyntheticMRI_K16      | SyntheticMRI_K8 |              -0.00062 | -0.00225 |  0.00099 |
| SyntheticMRI | SyntheticMRI_K24      | SyntheticMRI_K8 |              -9e-05   | -0.00149 |  0.00129 |
| SyntheticMRI | SyntheticMRI_adaptive | SyntheticMRI_K8 |              -0.00216 | -0.00331 | -0.00077 |
| SyntheticMRI | SyntheticMRI_piece_C0 | SyntheticMRI_K8 |               0.00029 | -0.00083 |  0.00134 |
| SyntheticMRI | SyntheticMRI_piece_C1 | SyntheticMRI_K8 |              -0.00111 | -0.0026  | -3e-05   |

### BRAVA recovery selection

{
  "dataset": "BRAVA",
  "arm": "D1_IXI",
  "step": 30000,
  "validation_edge_f1": 0.020992285967204823,
  "validation_node_recall": 0.34093419086597804,
  "representation_stage_ready": false,
  "gate": "Predefined feasibility gate: edge F1>0.04 and node recall2mm>0.25; review paired comparisons and annotation limitations before further submission.",
  "automatic_followup_training": false
}

A feasibility gate is not a scientific claim of superiority. Inspect best-versus-final learning curves, independent seed limitations, the selector confusion matrices and long-branch errors. Explicit splitting, recovered-BRAVA geometry learning, full end-to-end representations and node/relation ablations remain separate pending stages.