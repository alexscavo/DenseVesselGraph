"""CPU uniqueness and common-v2 graph evaluation of IXI node-loss pilots."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scripts.ixi_cached_mechanisms import measure, summarize_run, save, table
from scripts.ixi_query_diagnostics import categorize
from scripts.diagnose_ixi_queries import graph
from scripts.ixi_node_focal_ablation import ARMS


def uniqueness(c, gt, mask, tolerance):
    result = {}; samples = []
    for population, q in [('raw', np.arange(256)), ('retained', c['retained'])]:
        mapping, categories, _, _ = categorize(c['nodes'], gt['nodes'], gt['metric'], mask, q, tolerance)
        matched = [i for i in q if categories[int(i)] == 'matched']
        duplicate = [i for i in q if categories[int(i)] == 'duplicate_candidate']
        ids = matched+duplicate; y = np.array([1]*len(matched)+[0]*len(duplicate))
        values = c['node_scores'][ids]
        result.update({population+'_'+key: value for key, value in dict(
            matched=len(matched), unmatched=len(q)-len(matched), duplicates=len(duplicate),
            high_confidence_duplicates=int((c['node_scores'][duplicate] >= .9).sum()),
            duplicates_per_recovered_gt=len(duplicate)/len(matched) if matched else None,
            correct_vs_duplicate_auc=float(roc_auc_score(y, values)) if len(matched) and len(duplicate) else None,
            node_precision=len(matched)/len(q) if len(q) else None,
            node_recall=len(matched)/len(gt['nodes']) if len(gt['nodes']) else None).items()})
        if population == 'raw':
            # Query-label AP is descriptive; this is not box/detection mAP.
            labels = np.array([categories[int(i)] == 'matched' for i in q])
            result['raw_query_match_ap'] = float(average_precision_score(labels, c['node_scores'][q])) if labels.any() else None
            samples = [dict(query_id=int(i), confidence=float(c['node_scores'][i]), category=categories[int(i)]) for i in q]
    result.update(node_count_bias=int(len(c['retained'])-len(gt['nodes'])),
                  node_count_absolute_error=int(abs(len(c['retained'])-len(gt['nodes']))))
    return result, samples


def evaluate(root, arm, smoke=False):
    import torch
    torch.set_num_threads(1)
    source = root/('smoke-predictions' if smoke else 'predictions')/arm
    for folder in sorted(source.iterdir()):
        summary = json.loads((folder/'summary.json').read_text()); assert summary['completed']
        records = json.loads((folder/'records.json').read_text()); rows = []; lengths = []; confidences = []
        out = root/('smoke-evaluation' if smoke else 'evaluation')/arm/folder.name
        out.mkdir(parents=True, exist_ok=False)
        for record in records:
            with np.load(record['cache']) as f: c = {k:f[k] for k in f.files}
            with np.load(record['gt']) as f: gt = {k:f[k] for k in f.files}
            mask = np.asarray(nib.load(Path(record['gt']).parent.parent/'seg'/(record['sample_id']+'_seg.nii.gz')).dataobj) > .5
            g = graph(c, c['retained'])
            g['query_ids'] = c['retained']
            for tol in [1, 2]:
                row, ll = measure(c, g, Path(record['gt']), gt, mask, tol)
                extra, qq = uniqueness(c, gt, mask, tol)
                key = dict(split=record['split'], sample_id=record['sample_id'], subject=record['subject'], site=record['site'], tolerance_mm=tol, policy='standard_decoder')
                rows.append(dict(**key, **row, **extra)); lengths.extend(dict(**key, **r) for r in ll)
                confidences.extend(dict(**key, **r) for r in qq)
        summarize_run(out, rows, lengths, dict(arm=arm, step=summary['step'], records=len(records)))
        # Compact confidence labels allow pooled/per-subject AUC and rejected-query review.
        table(out/'query-confidence.csv', confidences)
        extra_rows = []
        for split, tol in sorted({(r['split'],r['tolerance_mm']) for r in rows}):
            for subject in [None]+sorted({r['subject'] for r in rows if r['split'] == split}):
                group = [r for r in rows if r['split'] == split and r['tolerance_mm'] == tol and (subject is None or r['subject'] == subject)]
                extra = {}
                for population in ['raw','retained']:
                    matched = sum(r[population+'_matched'] for r in group)
                    duplicate = sum(r[population+'_duplicates'] for r in group)
                    extra[population+'_duplicates_per_recovered_gt'] = duplicate/matched if matched else None
                    for suffix in ['matched','unmatched','duplicates','high_confidence_duplicates']:
                        extra[population+'_'+suffix] = sum(r[population+'_'+suffix] for r in group)
                qq = [r for r in confidences if r['split'] == split and r['tolerance_mm'] == tol and (subject is None or r['subject'] == subject) and r['category'] in ['matched','duplicate_candidate']]
                y = [r['category'] == 'matched' for r in qq]
                extra['raw_correct_vs_duplicate_auc'] = float(roc_auc_score(y,[r['confidence'] for r in qq])) if len(set(y)) == 2 else None
                extra.update(node_count_bias=float(np.mean([r['node_count_bias'] for r in group])),
                    node_count_mae=float(np.mean([r['node_count_absolute_error'] for r in group])),
                    raw_node_recall=sum(r['raw_matched'] for r in group)/sum(r['gt_nodes'] for r in group))
                extra_rows.append(dict(split=split,tolerance_mm=tol,subject=subject,**extra))
        table(out/'uniqueness.csv', extra_rows)
        print(out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); evaluate(a.root, a.arm, a.smoke)
