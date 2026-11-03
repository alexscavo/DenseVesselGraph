# Source transfer: test comparison

Validation-selected best checkpoints; same SyntheticMRI population and evaluation protocol. Graph-only pretraining followed by straight-edge or detached residual B-spline fine-tuning. Evaluation uses saved BatchNorm statistics, without recalibration. Single seed; no statistical superiority claim. Guides are uncertain extracted references.

| model | node_mAP | node_mAR | edge_mAP | edge_mAR | node_f1 | edge_f1 | common_branch_f1 | common_curve_f1 | beta0_absolute_error | beta1_absolute_error |
|---|---|---|---|---|---|---|---|---|---|---|
| leaves_straight | 0.4387 | 0.5372 | 0.3204 | 0.4364 | 0.8466 | 0.6864 | 0.0650 | 0.4309 | 2.9396 | 1.8872 |
| leaves_spline | 0.4418 | 0.5408 | 0.3238 | 0.4364 | 0.8503 | 0.6905 | 0.0664 | 0.4607 | 3.0601 | 1.7225 |
| plants_straight | 0.4456 | 0.5442 | 0.3245 | 0.4370 | 0.8502 | 0.6867 | 0.0700 | 0.4328 | 3.0908 | 1.7983 |
| plants_spline | 0.4448 | 0.5440 | 0.3238 | 0.4367 | 0.8496 | 0.6879 | 0.0702 | 0.4581 | 3.1230 | 1.7570 |
| mri_only_straight | 0.4738 | 0.5675 | 0.3760 | 0.4813 | 0.8688 | 0.7219 | 0.0979 | 0.4729 | 2.6416 | 1.6382 |
| mri_only_spline | 0.4788 | 0.5722 | 0.3814 | 0.4840 | 0.8708 | 0.7258 | 0.1020 | 0.5097 | 2.7209 | 1.5398 |
