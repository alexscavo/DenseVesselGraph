# Vessel evaluation metrics

- [Ideas and literature shortlist](metrics_ideas.md): the original proposal.
- [Implemented protocol and centerline interface](vessel-core-v1.md): definitions,
  adaptations, data conventions, assumptions and limitations.
- [Implementation and experiment tracker](implementation-status.md): completed
  work and work still to do.
- [Completed paired validation comparison](core-metrics-results.md) and
  [CSV table](core-metrics-results.csv): results, model identities and interpretation.
- [Branch fragmentation and relation-distance diagnostic](branch-connectivity-diagnostic.md):
  endpoint recovery, long relations and nearby false connections in the final models.
- [Controlled relation / node-head pilot](relation-distance-pilot.md): four arms,
  supervision, freeze boundaries, evaluation and submission status.

The first comparison uses the saved **final 300-epoch straight graph control and
detached B-spline model**, on the same 1,952 SyntheticMRI validation patches.
Legacy results retain their original definitions. The new suite has its own
version and output directory. No training or model inference is needed.

- [Generalization audit and plant pretraining recovery](generalization-audit.md): earlier vs final checkpoints, split statistics, and next controlled experiment (2026-10-14).


## IXI diagnostic reports — reading order

1. [GT supervision and foreground support](1-ixi-supervision-audit.md).
2. [Frozen feature probes and node error breakdown](2-ixi-feature-node-diagnostics.md).
3. [Higher-resolution node refinement](3-ixi-node-refinement-pilot.md).
4. [Relation isolation and image inspection](4-ixi-relation-isolation.md).

Each report describes its quantitative methods, image inspection method, observed examples and interpretation.
