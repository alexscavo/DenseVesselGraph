# Spline representation experiment

Main variant: endpoint-constrained cubic B-splines, eight controls, joint
SyntheticMRI fine-tuning. Predicted nodes anchor the endpoints. V0 supervises
the six internal controls in physical millimetres; V1 supervises sampled curves.

The original full paths and segmentation masks are uncertain references.
Closed and parallel branches unsupported by the current decoder exclude the
whole patch and are reported in the target manifest. These cases must be
checked before generic simple-graph contraction; a two-segment closed guide
can otherwise appear as duplicate undirected edges and crash preparation.

## Current protocol and decisions

The documentation lives separately in
[graph-representation/docs](../../../../../graph-representation/docs/README.md).
See the [training protocol and history](../../../../../graph-representation/docs/spline-10k-training.md)
and [four-patch evidence](../../../../../graph-representation/docs/spline-learning-results.md).

Training is capped at 10,000 candidates, validation at 2,000 fixed candidates,
and the available held-out test population is uncapped. The first main run
uses 50 epochs and validation/projections every five epochs. Unsupported patch
counts and matched-branch coverage accompany evaluation metrics.

Generated caches belong on SCRATCH, with a link from
`experiments/spline_learning/synthetic_mri_k8_10k_val2k`. Keep code/configs in
Git and preserve manifests and compact scientific evidence on WORK.

## Active artifacts

`experiments/spline_learning/README.md` describes the current dataset link, run
outputs, current logs and exact preparation/source metadata. Historical pilots,
four-patch comparisons and recovery snapshots are archived in STORE; consult
the [cleanup record](../../../../../graph-representation/docs/projects-cleanup-2026-10-06.md)
when reproducing them. They are not inputs to the main training run.
