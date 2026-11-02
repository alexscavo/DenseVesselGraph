## Completed results

Paired optimization population: {'train': 2048, 'val': 511}; subjects: {'train': 123, 'val': 26}. Evaluation uses the fixed training monitoring subset and complete paired validation manifest.

|Input|Final validation node mAP|Node mAR|Edge mAP|Edge mAR|SMD|
|---|---:|---:|---:|---:|---:|
|raw_mra|0.1347|0.2633|0.0097|0.0243|0.1089|
|vessel_mask|0.2210|0.3444|0.0276|0.0532|0.0610|

|Input|Node precision|Node recall|Edge F1|Conditional relation recall|Branch F1|Curve F1|Component MAE|Cycle MAE|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|raw_mra|0.4383|0.5488|0.0415|0.0635|0.0061|0.0931|21.9041|0.7065|
|vessel_mask|0.5871|0.6580|0.0904|0.1180|0.0243|0.1586|17.9041|0.8630|

### Interpretation

This is reconstruction from supplied masks that generated the graph references. It measures appearance removal under correlated supervision, not performance from independently annotated anatomical labels or a deployable raw-image pipeline.
Use paired-bootstrap.json to assess localization/recovery, duplicate multiplicity, conditional connectivity and false cycles separately. Node-only gains with persistently weak conditional relations implicate downstream graph reasoning; gains accompanied by duplicates or loops do not establish faithful reconstruction. Little gain in a short run does not establish an architectural impossibility.
Primary comparison is the matched final update. checkpoint-selection.json records a validation-only best checkpoint per arm; it is secondary and need not select the same update. Evaluation folders retain raw fixed-cutoff metrics and final training-calibrated policies. No test labels were used.
Original physical matching and common-v2 definitions are retained. Native box AP/AR and normalized straight-edge SMD have different meanings from positional connectivity and continuous-curve metrics. The pretrained spline head was frozen in this stable node/relation training configuration, so Curve F1 also reflects that geometry decoder constraint.
The common SyntheticMRI initializer already learned raw-image features. There is no prior IXI exposure, but it remains a modality pretraining bias; both arms train their CNN/BN state. Any absence of convergence or severe generalization gap calls for a bounded convergence check before rejecting segmentation input.
Outputs: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_segmentation_input_20261020-143413`. Detailed native/physical per-patch and per-subject tables, localization.csv, uniqueness.csv, thresholds.json and final calibration/ tables are retained.

