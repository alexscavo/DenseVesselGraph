# Evidence kept in the tree

The per-study evidence reports were archived on 3 Nov 2026 (see
[`../../archive/README.md`](../../archive/README.md)). Only files that code reads remain here:

- `brava_preparation/`: BRAVA orientation corrections and preparation checks. `scripts/prepare_brava.py`
  reads `orientation-corrections.json`, and `cluster/jean_zay/submit_brava.py` checks its hash.
  The folder is described in [`../BRAVA_preparation.md`](../BRAVA_preparation.md).
- `ixi_mechanisms_20261019/D/focal_control/resolved-config.json`: the fixture of
  `tests/test_ixi_node_focal_ablation.py`.
