# IXI query-attribution pilot evidence — 2026-10-19

See [methods, results and recovery record](../../../archive/legacy-20261103/docs/research/metrics/6-ixi-query-attribution.md).

Each arm contains summaries copied from its completed CPU collector. Original
Slurm jobs failed during reporting after successful frozen-model evaluation;
collection was corrected and rerun from saved caches with exit code 0.
Historical batch-4/32 comparisons are descriptive, not exact reproduction.
Local-image batch-1 positional counts reproduce the prior diagnostic on 384 patches.

- `comparison.csv`: all splits, tolerances and deletion policies.
- `reproduction.json`: historical differences and same-batch comparison count.
- Remaining JSON: node, edge, site, relation-length and subject-bootstrap summaries.
- `overview.png`: validation policy comparison.
- Local-image `sample_000155_0039-3d.png`: one fixed-order example, not selected for quality.

Full caches, CSV rows and interactive 3D views remain under
`/lustre/fsn1/projects/rech/vnc/upz25mj/experiments/gnbm/ixi_query_diagnostics_20261019/`.
The submission record maps each arm to its output path and recovery collector hash.
No full-validation expansion or new model training was launched in this round.
