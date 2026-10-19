# IXI node refinement: fixed 1000-update comparison

Matched batch-order hashes and identical initial validation scores verified. Frozen state unchanged in each arm.

| Arm/stage | Node recall 2mm | Node precision 2mm | Node F1 | Edge F1 | Curve F1 |
|---|---:|---:|---:|---:|---:|
|baseline/initial|0.5073|0.4686|0.4812|0.1558|0.1169|
|baseline/final|0.5112|0.4701|0.4857|0.1553|0.1171|
|local_image/initial|0.5073|0.4686|0.4812|0.1558|0.1169|
|local_image/final|0.5627|0.5171|0.5196|0.1673|0.1676|
|zero_image/initial|0.5073|0.4686|0.4812|0.1558|0.1169|
|zero_image/final|0.5115|0.4702|0.4829|0.1538|0.1164|

Compare local_image against BOTH baseline and zero_image. Improvement over zero_image is needed to attribute benefit to local image evidence. These validation pilots do not establish test generalization. Relation weights stay frozen; changing selected nodes changes decoded edges.

Full predictions, fixed-example PNGs, W&B offline runs and small trained-head checkpoints: `/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/spline_learning/ixi_node_refinement_20261018`.
