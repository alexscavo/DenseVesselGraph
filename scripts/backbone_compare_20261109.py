#!/usr/bin/env python3
"""Size-matched backbone comparison (9 Nov 2026): U-Net, U-Net wide, DSC3 slim, DSC3 on the current recipe.

For one dataset (GPU job): every arm's final traced model is scored on the validation list with
  - node AP @2 mm from all heatmap peaks >= 0.01 (threshold-free; dense_ap/traced_ap definition);
  - full loop-aware node/edge/Branch/Curve F1 and |beta0|, |beta1| errors at the FROZEN GLOBAL decoding settings of the
    evaluation protocol (report.md, global_pooled): V0 node 0.30 / edge 0.75 and V4 low 0.20 / high 0.30 / edge 0.75,
    so no arm benefits from its own threshold choice;
  - each arm's own train-selected result (final.json), for reference.
Arm minus U-Net differences use the protocol's subject-paired bootstrap (protocol_report_20261106.paired).
Writes experiments/backbone_compare_20261109/<dataset>.json. usage: backbone_compare_20261109.py <dataset>
"""
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import dense_graph_traced_20261030 as T  # noqa: E402
from compare_dense_anchored_20261030 import KEYS, dense_patch  # noqa: E402
from decode_study_20261105 import as_row, decode, gt  # noqa: E402
from ixi_query_diagnostics import betti  # noqa: E402
from node_heatmap_probe_20261030 import PLAN, make_dataset  # noqa: E402
from protocol_report_20261106 import paired, pooled  # noqa: E402
from traced_ap_20261105 import node_ap_from_model  # noqa: E402

RUNS = {'IXI-loops': 'traced_adaptive', 'TopBrainCTA-loops': 'traced_adaptive', 'BRAVA': 'traced_offset'}
ARMS = {'unet': None, 'unet_wide': 'traced_unet_wide_r1106', 'dsc3_slim': 'traced_dsc3_slim_r1106', 'dsc3': 'traced_dsc3_r1106'}
SETTINGS = {'V0_global': dict(t_high=0.3, t_edge=0.75), 'V4_global': dict(t_low=0.2, t_high=0.3, t_edge=0.75, weak_tree=True)}
ROWS = []


def per_patch(args):
    i, cfg = args
    r = ROWS[i]
    dr = as_row(r, *decode(r, **cfg))
    c = dense_patch(dr, 0.5, 0.5)
    _, _, _, (g0, g1), _ = gt(r['cache'])
    p0, p1 = betti(len(dr['nodes']), dr['pairs'])
    return [c[k] for k in KEYS] + [abs(p0 - g0), abs(p1 - g1)]


def main(dataset):
    scratch = Path(os.environ['SCRATCH']) / 'experiments/gnbm/dense_graph_20261030' / dataset
    val_ids = json.loads(PLAN.read_text())['datasets'][dataset]['val_ids']
    ds = make_dataset(dataset, val_ids, augment=False)
    out = dict(dataset=dataset, settings=SETTINGS, arms={})
    counts, ids_ref = {}, None
    for arm, run in ARMS.items():
        run = run or RUNS[dataset]
        final = json.loads((scratch / run / 'final.json').read_text())
        assert final.get('backbone', 'unet') == arm, (run, final.get('backbone'))
        T.MODE.update(final.get('mode', {}))
        model = T.TracedDenseGraph(backbone=arm).to('cuda')
        model.load_state_dict(torch.load(scratch / run / 'model.pt', map_location='cuda')['model'])
        entry = dict(run=run, own_thresholds=final['thresholds'], own_validation={k: final['validation'][k] for k in
                     ('node_f1', 'edge_f1', 'common_branch_f1', 'common_curve_f1', 'beta0_mae', 'beta1_mae')},
                     parameters_backbone=int(sum(p.numel() for p in model.unet.parameters())),
                     node_ap_2mm=node_ap_from_model(model, ds, 'cuda'))
        for name, cfg in SETTINGS.items():
            ROWS[:] = T.predict(model, ds, 'cuda', cfg.get('t_low', cfg['t_high']), 8)
            ids = [r['sample_id'] for r in ROWS]
            assert ids_ref is None or ids == ids_ref
            ids_ref = ids
            with mp.get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))) as pool:
                counts[(arm, name)] = np.asarray(pool.map(per_patch, [(i, cfg) for i in range(len(ROWS))], chunksize=16), float)
            entry[name] = pooled(counts[(arm, name)])
        out['arms'][arm] = entry
        print(dataset, arm, json.dumps({k: entry[k] for k in ('parameters_backbone', 'node_ap_2mm', 'V0_global', 'V4_global')}, default=float), flush=True)
        del model
        torch.cuda.empty_cache()
    out['minus_unet'] = {f'{arm} | {name}': paired(counts[(arm, name)], counts[('unet', name)], ids_ref)['difference']
                         for arm in ARMS if arm != 'unet' for name in SETTINGS}
    dst = ROOT / 'experiments/backbone_compare_20261109'
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f'{dataset}.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
