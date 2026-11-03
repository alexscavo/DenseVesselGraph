# Source transfer: validation comparison

Validation-selected best checkpoints; same SyntheticMRI population and evaluation protocol. Graph-only pretraining followed by straight-edge or detached residual B-spline fine-tuning. Evaluation uses saved BatchNorm statistics, without recalibration. Single seed; no statistical superiority claim. Guides are uncertain extracted references.

These are validation results used for model selection, not held-out test results.

| model | node_mAP | node_mAR | edge_mAP | edge_mAR | node_f1 | edge_f1 | common_branch_f1 | common_curve_f1 | beta0_absolute_error | beta1_absolute_error |
|---|---|---|---|---|---|---|---|---|---|---|
| leaves_straight | 0.4348 | 0.5327 | 0.3131 | 0.4295 | 0.8445 | 0.6814 | 0.0605 | 0.4275 | 2.9923 | 1.9155 |
| leaves_spline | 0.4395 | 0.5370 | 0.3198 | 0.4304 | 0.8482 | 0.6879 | 0.0644 | 0.4570 | 3.1004 | 1.6844 |
| plants_straight | 0.4413 | 0.5401 | 0.3206 | 0.4311 | 0.8493 | 0.6881 | 0.0653 | 0.4274 | 3.1516 | 1.7526 |
| plants_spline | 0.4426 | 0.5418 | 0.3203 | 0.4320 | 0.8485 | 0.6852 | 0.0679 | 0.4558 | 3.1030 | 1.7387 |
| mri_only_straight | 0.4725 | 0.5652 | 0.3716 | 0.4759 | 0.8681 | 0.7195 | 0.0969 | 0.4695 | 2.6829 | 1.6240 |
| mri_only_spline | 0.4749 | 0.5682 | 0.3740 | 0.4766 | 0.8703 | 0.7224 | 0.0956 | 0.5036 | 2.7459 | 1.5533 |
