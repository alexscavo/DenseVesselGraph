"""Small synthetic cache/metric parity checks, not scientific model results."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scripts.node_first_curriculum import META, write
from scripts.collect_node_first_curriculum import evaluate_folder, common_rank_pairs, train_threshold, collect
from scripts.node_first_curriculum import plan


def fixture(folder):
    truth = np.array([[.2, .2, .2], [.25, .2, .2], [.25, .25, .2]], np.float32)
    edges = np.array([[0, 1], [1, 2]], np.int64)
    controls = np.array([np.linspace(truth[a], truth[b], 8) for a, b in edges])
    curves = np.array([np.linspace(truth[a], truth[b], 64) for a, b in edges])
    metric = np.eye(3, dtype=np.float32) * 64
    gt = folder / 'target.npz'
    np.savez_compressed(gt, nodes=truth, edges=edges, controls=controls,
                        metric=metric, full_paths=truth[edges].reshape(-1, 3),
                        path_offsets=np.array([0, 2, 4], np.int64))
    raw = np.r_[truth, [[.201, .2, .2], [.8, .8, .8]]].astype(np.float32)
    pairs = np.array([[a, b] for a in range(5) for b in range(a + 1, 5)], np.int64)
    scores = np.array([.9, .85, .8, .1, .2], np.float32)
    relations = np.array([.8 if tuple(p) in {(0, 1), (1, 2)} else .1 for p in pairs], np.float32)
    c = dict(raw_boxes=np.c_[raw, np.full_like(raw, .1)], scores=scores,
             raw_logits=np.c_[np.log1p(-scores), np.log(scores)],
             hungarian_source=np.arange(3), hungarian_target=np.arange(3),
             truth_normalized=truth, truth=truth @ metric.T, gt_edges=edges, metric=metric,
             physical=raw @ metric.T, pairs=pairs, relations=relations,
             graph_nodes=truth, graph_boxes=np.c_[truth, np.full_like(truth, .1)],
             graph_node_scores=scores[:3], graph_query_ids=np.arange(3),
             graph_edges=edges, graph_edge_scores=np.full(2, .8, np.float32),
             graph_spline_controls=controls, graph_spline_curves=curves, graph_spline_metric=metric)
    rec = []
    with zipfile.ZipFile(folder / 'caches.zip', 'w') as z:
        for split in ['train', 'val']:
            key = split + '/fixture.npz'
            b = io.BytesIO()
            np.savez_compressed(b, **c)
            z.writestr(key, b.getvalue())
            rec.append(dict(split=split, sample='fixture', subject=split + '_subject', cache_key=key,
                            target_path=str(gt), loss=dict(nodes=0., edges=.2, total=1.2)))
    write(folder / 'records.json', rec)
    write(folder / 'complete.json', dict(completed=True))


class NodeFirstCollection(unittest.TestCase):
    def test_synthetic_full_metric_and_native_cache_path(self):
        with tempfile.TemporaryDirectory(dir=META) as temporary:
            folder = Path(temporary)
            fixture(folder)
            result = evaluate_folder(folder)
            for split in ['train', 'val']:
                for tol in ['1.0', '2.0']:
                    self.assertEqual(result['physical'][split][tol]['node_f1'], 1)
                    self.assertEqual(result['physical'][split][tol]['edge_f1'], 1)
                self.assertEqual(result['common'][split]['common_branch_f1'], 1)
                self.assertEqual(result['common'][split]['common_curve_f1'], 1)
                self.assertEqual(result['ranking'][split]['conditional']['candidates'], 3)
                self.assertEqual(result['ranking'][split]['conditional']['positives'], 2)

    def test_identical_pair_ap_and_strict_training_threshold(self):
        with tempfile.TemporaryDirectory(dir=META) as temporary:
            folder = Path(temporary)
            fixture(folder)
            rank = common_rank_pairs(folder, folder)
            self.assertEqual(rank['delta'], 0)
            self.assertEqual(rank['candidates'], 3)
            self.assertEqual(rank['positives'], 2)
            cut = train_threshold(folder)
            self.assertLess(cut, float(np.float32(.8)))
            self.assertEqual(cut, float(np.nextafter(np.float32(.8), np.float32(-np.inf))))

    def test_tiny_end_to_end_collection_stops_neutral_pilot(self):
        with tempfile.TemporaryDirectory(dir=META) as temporary:
            root = Path(temporary)
            p = copy.deepcopy(plan())
            p['output_root'] = str(root)
            p['stages']['pilot'] = dict(updates=100, seeds=[1])
            p['subject_bootstrap_draws'] = 5
            with patch('scripts.collect_node_first_curriculum.plan', return_value=p), \
                 patch('scripts.collect_node_first_curriculum.publish'), \
                 patch('scripts.collect_node_first_curriculum.MAIN', root / 'fixture-project'):
                for arm in ['joint', 'node_first']:
                    out = root / 'pilot/1' / arm
                    trajectories = {}
                    for step in [10, 20, 50, 100]:
                        folder = out / f'step-{step}'
                        folder.mkdir(parents=True)
                        fixture(folder)
                        trajectories[str(step)] = evaluate_folder(folder)
                    write(out / 'complete.json', dict(completed=True, initial_state_sha256='random-fixture', batch_chain='same-fixture-data'))
                    write(out / 'evaluation-complete.json', dict(trajectories=trajectories))
                # No publisher or Slurm action is invoked by this fixture.
                collect('pilot', auto_promote=False)
                decision = __import__('json').loads((root / 'pilot/decision.json').read_text())
                self.assertFalse(decision['full_allowed'])
                self.assertEqual(decision['decision'], 'inconclusive_do_not_auto_expand')
                self.assertTrue((root / 'pilot/report.md').exists())


if __name__ == '__main__':
    unittest.main()
