# Source transfer: test comparison

Validation-selected best checkpoints; same SyntheticMRI population and evaluation protocol. Graph-only pretraining followed by straight-edge or detached residual B-spline fine-tuning. Evaluation uses saved BatchNorm statistics, without recalibration. Single seed; no statistical superiority claim. Guides are uncertain extracted references.

| model | node_mAP | node_mAR | edge_mAP | edge_mAR | node_f1 | edge_f1 | common_branch_f1 | common_curve_f1 | beta0_absolute_error | beta1_absolute_error |
|---|---|---|---|---|---|---|---|---|---|---|
| leaves_straight | 0.4387 | 0.5372 | 0.3204 | 0.4364 | 0.8466 | 0.6864 | 0.0650 | 0.4309 | 2.9396 | 1.8872 |
| leaves_spline | 0.4418 | 0.5408 | 0.3238 | 0.4364 | 0.8503 | 0.6905 | 0.0664 | 0.4607 | 3.0601 | 1.7225 |
| plants_straight | 0.4456 | 0.5442 | 0.3245 | 0.4370 | 0.8502 | 0.6867 | 0.0700 | 0.4328 | 3.0908 | 1.7983 |
| plants_spline | 0.4448 | 0.5440 | 0.3238 | 0.4367 | 0.8496 | 0.6879 | 0.0702 | 0.4581 | 3.1230 | 1.7570 |
| mri_only_straight | 0.4738 | 0.5675 | 0.3760 | 0.4813 | 0.8688 | 0.7219 | 0.0979 | 0.4729 | **2.6416** | 1.6382 |
| mri_only_spline | **0.4788** | **0.5722** | **0.3814** | **0.4840** | **0.8708** | **0.7258** | **0.1020** | **0.5097** | 2.7209 | **1.5398** |
| combined_straight | 0.4246 | 0.5268 | 0.3086 | 0.4262 | 0.8377 | 0.6788 | 0.0605 | 0.4219 | 3.2785 | 1.8381 |
| combined_spline | 0.4281 | 0.5295 | 0.3121 | 0.4281 | 0.8408 | 0.6809 | 0.0633 | 0.4484 | 3.2364 | 1.7859 |
| plants_aug_straight | 0.4490 | 0.5473 | 0.3257 | 0.4432 | 0.8521 | 0.6902 | 0.0694 | 0.4444 | 3.0296 | 1.7390 |
| plants_aug_spline | 0.4555 | 0.5522 | 0.3312 | 0.4453 | 0.8568 | 0.6935 | 0.0749 | 0.4733 | 3.0403 | 1.6379 |
| leaves_aug_straight | 0.4230 | 0.5246 | 0.2999 | 0.4155 | 0.8380 | 0.6737 | 0.0565 | 0.4124 | 3.1529 | 1.8335 |
| leaves_aug_spline | 0.4302 | 0.5312 | 0.3077 | 0.4249 | 0.8405 | 0.6785 | 0.0620 | 0.4473 | 3.0567 | 1.8683 |
| combined_aug_straight | 0.4477 | 0.5488 | 0.3267 | 0.4425 | 0.8548 | 0.6902 | 0.0720 | 0.4395 | 3.0594 | 1.7873 |
| combined_aug_spline | 0.4428 | 0.5448 | 0.3239 | 0.4396 | 0.8550 | 0.6892 | 0.0694 | 0.4627 | 3.0418 | 1.7811 |

**Bold** marks the best value per column (higher AP/AR/F1; lower Betti errors).
