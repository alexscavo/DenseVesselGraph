# Source transfer: validation comparison

Validation-selected best checkpoints; same SyntheticMRI population and evaluation protocol. Graph-only pretraining followed by straight-edge or detached residual B-spline fine-tuning. Evaluation uses saved BatchNorm statistics, without recalibration. Single seed; no statistical superiority claim. Guides are uncertain extracted references.

These are validation results used for model selection, not held-out test results.

| model | node_mAP | node_mAR | edge_mAP | edge_mAR | node_f1 | edge_f1 | common_branch_f1 | common_curve_f1 | beta0_absolute_error | beta1_absolute_error |
|---|---|---|---|---|---|---|---|---|---|---|
| leaves_straight | 0.4348 | 0.5327 | 0.3131 | 0.4295 | 0.8445 | 0.6814 | 0.0605 | 0.4275 | 2.9923 | 1.9155 |
| leaves_spline | 0.4395 | 0.5370 | 0.3198 | 0.4304 | 0.8482 | 0.6879 | 0.0644 | 0.4570 | 3.1004 | 1.6844 |
| plants_straight | 0.4413 | 0.5401 | 0.3206 | 0.4311 | 0.8493 | 0.6881 | 0.0653 | 0.4274 | 3.1516 | 1.7526 |
| plants_spline | 0.4426 | 0.5418 | 0.3203 | 0.4320 | 0.8485 | 0.6852 | 0.0679 | 0.4558 | 3.1030 | 1.7387 |
| mri_only_straight | 0.4725 | 0.5652 | 0.3716 | 0.4759 | 0.8681 | 0.7195 | **0.0969** | 0.4695 | **2.6829** | 1.6240 |
| mri_only_spline | **0.4749** | **0.5682** | **0.3740** | **0.4766** | **0.8703** | **0.7224** | 0.0956 | **0.5036** | 2.7459 | **1.5533** |
| combined_straight | 0.4238 | 0.5254 | 0.3046 | 0.4211 | 0.8364 | 0.6767 | 0.0599 | 0.4188 | 3.3648 | 1.8263 |
| combined_spline | 0.4267 | 0.5276 | 0.3098 | 0.4251 | 0.8412 | 0.6803 | 0.0635 | 0.4472 | 3.2971 | 1.7956 |
| plants_aug_straight | 0.4472 | 0.5436 | 0.3210 | 0.4379 | 0.8497 | 0.6878 | 0.0680 | 0.4407 | 3.0359 | 1.7664 |
| plants_aug_spline | 0.4525 | 0.5486 | 0.3281 | 0.4394 | 0.8549 | 0.6921 | 0.0711 | 0.4695 | 3.0774 | 1.6035 |
| leaves_aug_straight | 0.4218 | 0.5230 | 0.2987 | 0.4131 | 0.8396 | 0.6728 | 0.0546 | 0.4101 | 3.1875 | 1.8811 |
| leaves_aug_spline | 0.4302 | 0.5298 | 0.3049 | 0.4211 | 0.8420 | 0.6769 | 0.0615 | 0.4421 | 3.1404 | 1.8899 |
| combined_aug_straight | 0.4440 | 0.5457 | 0.3217 | 0.4364 | 0.8546 | 0.6899 | 0.0672 | 0.4349 | 3.1178 | 1.7690 |
| combined_aug_spline | 0.4366 | 0.5403 | 0.3178 | 0.4322 | 0.8531 | 0.6852 | 0.0667 | 0.4574 | 3.1183 | 1.7736 |

**Bold** marks the best value per column (higher AP/AR/F1; lower Betti errors).
