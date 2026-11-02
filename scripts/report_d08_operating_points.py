#!/usr/bin/env python3
"""Summarize saved CPU D08 calibration, optionally append existing research docs."""
import argparse
import fcntl
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/d08_operating_points_20261028'


def fmt(value):return 'NA' if value is None else f'{value:.5f}'


def append(path,body,marker):
    with path.with_name(path.name+'.d08-calibration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        text=path.read_text()
        if marker not in text:
            with path.open('a') as stream:stream.write('\n\n'+marker+'\n'+body+'\n')


def report(publish=False):
    data=json.loads((OUT/'summary.json').read_text());s=data['aggregates'];cuts=data['thresholds'];ci=data['paired_subject_intervals']
    lines=['## D08 training-only operating-point audit — 28 October 2026','',
        '**Completed on CPU from saved epoch-300 predictions. No detector inference, optimizer updates or GPU jobs.**',
        '', 'Calibration uses the **160 representative training patches /39 training subjects** already cached, not all8,768 training patches. The selected cuts are frozen before evaluating **1,796 validation patches /eight disjoint subjects**. Primary selection maximizes micro positional F1 at2mm; the same cuts are also evaluated at1mm. Validation does not choose a threshold. Both original D08 checkpoints and every saved coordinate/score remain unchanged.',
        '', '### Node calibration using all256 raw queries', '',
        'A fixed23-cut confidence grid (0…1, listed in `protocol.json`) is evaluated on training only. Highest cutoff wins exact F1 ties. No GT grouping, query replacement, NMS, classifier fitting or changed matching enters inference. Raw0-cut results diagnose all-query coverage and are not a selected full-graph policy.', '',
        '| Arm | Policy | Node cut | Train node F1 @2mm | Val node P/R/F1 @2mm | Val nodes/patch | Val duplicates/patch | GT edges with endpoints recovered |',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for arm in ['scratch','synthetic']:
        for policy in ['normal_0p5','training_node_f1','all_raw']:
            t=s[f'{arm}/train/node/{policy}/2.0'];v=s[f'{arm}/val/node/{policy}/2.0']
            cut=.5 if policy=='normal_0p5' else (0. if policy=='all_raw' else cuts[arm]['node_train_max_f1'])
            lines.append(f"| {arm} | {policy} | {cut:.3f} | {fmt(t['node_f1'])} | {fmt(v['node_precision'])}/{fmt(v['node_recall'])}/{fmt(v['node_f1'])} | {v['predicted_nodes']/v['patches']:.3f} | {v['duplicates']/v['patches']:.3f} | {v['gt_edges_both_endpoints_recovered']} |")
    lines+=['', 'Node-only validation differences (train-selected minus0.5;1,000 paired whole-subject bootstrap draws, seed281028):','']
    for arm in ['scratch','synthetic']:
        b=s[f'{arm}/val/node/normal_0p5/2.0'];v=s[f'{arm}/val/node/training_node_f1/2.0'];interval=ci[f'{arm}/node/2.0']['node_f1']
        lines.append(f"- {arm}: node F1 {v['node_f1']-b['node_f1']:+.5f},95% interval [{fmt(interval[0])},{fmt(interval[1])}]; cut{cuts[arm]['node_train_max_f1']:.3f}. This changes proposal counts as well as recovery.")
    lines+=['', '### Edge calibration on the unchanged originally retained node population', '',
        'All cached pairs among original node-confidence>0.5 queries are retained as the ranking population. Select either training-max positional edge F1, or the most recalled tied score group meeting the original training precision. Strict `>` and native-float32 `nextafter` preserve whole score ties; seven targeted tests include the earlier R07 threshold-rounding failure mode. All positives/candidate labels are evaluation labels; no relation retraining or negative mining.', '',
        '| Arm | Policy | Edge cut | Train edge F1 | Val edge P/R/F1 @2mm | Val TP /FP /FN | Branch F1 | Cycle MAE |',
        '|---|---|---:|---:|---|---|---:|---:|']
    for arm in ['scratch','synthetic']:
        for policy in ['fixed_0p5','train_max_f1','train_preserve_precision']:
            t=s[f'{arm}/train/edge/{policy}/2.0'];v=s[f'{arm}/val/edge/{policy}/2.0'];cut=cuts[arm]['edge'][policy]
            lines.append(f"| {arm} | {policy} | {cut:.9f} | {fmt(t['edge_f1'])} | {fmt(v['precision'])}/{fmt(v['recall'])}/{fmt(v['edge_f1'])} | {v['tp']}/{v['fp']}/{v['fn']} | {fmt(v['branch_f1'])} | {fmt(v['cycle_mae'])} |")
    lines+=['', 'Training precision targets are '+', '.join(f"{a}={cuts[a]['baseline_training_precision']:.5f}" for a in ['scratch','synthetic'])+'. Scratch has zero training TP at the original cutoff; its zero precision constraint is **uninformative**, not evidence of a useful precision-controlled operating point. Training precision is not guaranteed on validation.', '',
        'Paired validation edge-F1 differences versus0.5 on these identical node candidates:','']
    for arm in ['scratch','synthetic']:
        base=s[f'{arm}/val/edge/fixed_0p5/2.0']
        for policy in ['train_max_f1','train_preserve_precision']:
            v=s[f'{arm}/val/edge/{policy}/2.0'];bounds=ci[f'{arm}/edge/{policy}/2.0']['edge_f1']
            lines.append(f"- {arm}/{policy}: {v['edge_f1']-base['edge_f1']:+.5f} [{fmt(bounds[0])},{fmt(bounds[1])}], TP change{v['tp']-base['tp']:+d}, FP change{v['fp']-base['fp']:+d}, cycle-MAE change{v['cycle_mae']-base['cycle_mae']:+.5f}.")
    lines+=['', '### Ranking, candidate identity and cache limits', '',
        '| Arm | Val conditional AP (pairs/positives) | Val all-retained AP (pairs/positives) |',
        '|---|---|---|']
    for arm in ['scratch','synthetic']:
        r=s[f'{arm}/val/ranking/2.0'];a=r['conditional'];b=r['all_retained']
        lines.append(f"| {arm} | {fmt(a['ap'])} ({a['candidates']}/{a['positives']}) | {fmt(b['ap'])} ({b['candidates']}/{b['positives']}) |")
    lines+=['', 'Edge threshold changes **do not change AP or relation ranking** on this fixed candidate population. Differences between scratch and transferred AP still have different candidate populations and cannot isolate relation-head quality. Branch F1 uses the established1mm role-aware contraction/matching convention; baseline branch TP/FP/FN and original positional counts reproduce exactly. All1/2mm per-patch/per-subject metrics, near-nonadjacent FP rates, signed components/cycles and subject intervals are saved separately.', '',
        '**Missing data boundary:** D08 archives save raw256-query positions/confidences, but relation scores only for the original>0.5 node population, and K8 controls only for original predicted edges. They contain no full raw-query decoder embeddings. Therefore the node-calibrated graph Edge/Curve F1 and new-edge learned spline geometry are **not recoverable from these caches alone**. Do not combine node-only gains with cached edge-only gains, substitute GT edges/coordinates, fill missing scores, or report straight chords as the learned Curve F1. Node-calibrated recovered endpoint counts above are diagnostic ceilings, not predicted correct edges.', '',
        '### Interpretation and next gate', '']
    for arm in ['scratch','synthetic']:
        b=s[f'{arm}/val/node/normal_0p5/2.0'];v=s[f'{arm}/val/node/training_node_f1/2.0'];e=s[f'{arm}/val/edge/train_max_f1/2.0'];old=s[f'{arm}/val/edge/fixed_0p5/2.0']
        lines.append(f"{arm}: training-selected node threshold changes validation F1 from{b['node_f1']:.5f} to{v['node_f1']:.5f}, precision from{fmt(b['node_precision'])} to{fmt(v['node_precision'])}, recall from{fmt(b['node_recall'])} to{fmt(v['node_recall'])}. Edge-only calibration changes F1 from{old['edge_f1']:.5f} to{e['edge_f1']:.5f}, while FP changes{old['fp']}→{e['fp']} and cycle MAE{old['cycle_mae']:.5f}→{e['cycle_mae']:.5f}. Assess these trade-offs together; additional recall is not an improved relation representation.")
    lines+=['',
        '**Decision:** retain the original0.5/0.5 graph decoder as the reported reference. The transferred node cut0.45 is a modest node-only calibration candidate, not an established graph improvement. Its validation precision falls0.46922→0.40828, duplicate candidates increase621→1,201, and correct node detections increase8,818→10,292 while annotation-relative FP increase9,975→14,916. Endpoint-pair recovery rises1,607→2,230 of19,446 GT edges (8.26%→11.47% under the same spatial matching convention), still a small fraction of the graph.',
        '',
        'For transferred edge-only max-F1 calibration, the gain is **+0.00098** but adds118 TP and2,890 FP. Branch F1 falls0.00647→0.00576; its paired difference interval excludes zero in the negative direction. Cycle MAE rises1.05902→1.82461. Near-nonadjacent negative-pair FP rate increases44.51%→57.54% (304→393 FP among683 unchanged eligible pairs). The training-precision policy stays near the baseline (edge cut0.49720), has an edge-F1 interval including zero and slightly worse Branch/cycle outcomes. These are operating-point trades, not a connectivity rescue.',
        '',
        'Scratch node calibration gives a much larger F1 change but only12.10% validation node precision, about30.17 retained nodes/patch and greater count MAE (16.21→17.66). It does not establish a recovered scratch detector. Its edge-only cutoff0.15670 raises conditional recall on only27 recovered validation GT edges while producing3,180 FP; do not treat that conditional recall as good whole-graph reconstruction.']
    lines+=['', 'This audit tests operating points, not new discrimination or a complete recovered BRAVA graph. Incomplete SWC annotations make unmatched/false edges annotation-relative. Confidence filtering is only one possible bottleneck; improved node recall at a lower cutoff can increase ambiguity, false relations and cycles. Neither an architecture nor a longer training run is justified automatically.', '',
        'If combined graph evaluation is subsequently justified, the smallest missing computation is one frozen epoch-300 replay of the same160 training and1,796 validation patches per chosen arm, with inference batch8 and strict original-cache node/score parity. Retrieve only relations involving newly retained queries and learned K8 geometry for newly accepted edges; select any additional edge cutoff on training only. This is a bounded inference task, not retraining, and **was not launched here**. No test set is needed.', '',
        'Calibration grid/decisions were saved before validation in `thresholds.json`, `training-node-curves.json` and `protocol.json`. Source/checkpoint identities, eight baseline parity blocks, seven tests, raw summaries,1,000-draw subject intervals and compressed per-patch/per-subject tables are in `experiments/d08_operating_points_20261028/`. Validation subjects are reused development data; uncertainty does not include training-seed variation. The training calibration monitor is160 patches rather than the full cohort.', '',
        '```bash',
        'python3 /lustre/fswork/projects/rech/vnc/upz25mj/projects/Graph-Native-Betti-Matching-Representation/scripts/report_d08_operating_points.py --show',
        '```']
    text='\n'.join(lines)+'\n'
    # Readable units/spacing in the generated report; numerical values unchanged.
    import re
    text=re.sub(r'(?<=\d)mm\b',' mm',text)
    for a,b in [('all256','all 256'),('fixed23','fixed 23'),('node-confidence>','node-confidence >'),('at2 mm','at 2 mm'),('at1 mm','at 1 mm'),('39 training','39 training'),('not all8,768','not all 8,768'),('minus0.5','minus 0.5'),('versus0.5','versus 0.5'),('seed281028','seed 281028'),('cut0.','cut 0.'),('changes118','changes 118'),('adds118','adds 118'),('and2,890','and 2,890'),('rises1,607','rises 1,607'),('of19,446','of 19,446'),('among683','among 683'),('only27','only 27'),('only12.10','only 12.10'),('about30.17','about 30.17'),('increase621','increase 621'),('increase8,818','increase 8,818'),('increase9,975','increase 9,975'),('from0.','from 0.'),('to0.','to 0.'),('from1.','from 1.'),('to1.','to 1.'),('from44.','from 44.'),('0.5/0.5','0.5/0.5'),('original0.','original 0.'),('precisions0.','precisions 0.'),('rises1.','rises 1.'),('falls0.','falls 0.'),('increases44.','increases 44.'),('keeps12.','keeps 12.'),('detector. Its','detector. Its'),('candidate. Its','candidate. Its'),('of0.','of 0.'),('with1,607','with 1,607'),('same160','same 160'),('and1,796','and 1,796'),('batch8','batch 8'),('eight1/2','eight 1/2'),('all1/2','all 1/2'),('1,000-draw','1,000-draw'),('is160','is 160')]:text=text.replace(a,b)
    (OUT/'report.md').write_text(text)
    if publish:
        append(ROOT/'docs/research/BRAVA_experiment_plan.md',text,'<!-- D08 CACHE CALIBRATION 20261028 -->')
        a=s['synthetic/val/node/training_node_f1/2.0'];e=s['synthetic/val/edge/train_max_f1/2.0']
        short=f"D08 CPU calibration completed: trained node cut0.45 gives validation node F1 {a['node_f1']:.5f}; edge-only train-F1 cut{cuts['synthetic']['edge']['train_max_f1']:.9f} gives edge F1 {e['edge_f1']:.5f} on unchanged>0.5-node candidates. Ranking/AP is unchanged. Joint lower-node Edge/Curve F1 remains unavailable because rejected-query relations/new-edge controls were not cached. No GPU, inference or training was run. See the [existing BRAVA report](../docs/research/BRAVA_experiment_plan.md#d08-training-only-operating-point-audit--28-october-2026) and `experiments/d08_operating_points_20261028/`."
        append(ROOT/'research/experiments.md','## D08 cached calibration — 28 October 2026\n\n'+short,'<!-- D08 CALIBRATION REGISTER 20261028 -->')
        append(ROOT/'research/roadmap.md','## D08 operating-point gate measured — 28 October 2026\n\n'+short+' No architecture/training sweep is authorized; review precision/recall and cycles before deciding on the minimal missing inference replay.','<!-- D08 CALIBRATION ROADMAP 20261028 -->')
        append(ROOT/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md','## D08 calibration follow-up — DONE, measured 28 October 2026\n\n'+short,'<!-- D08 CALIBRATION REGISTRY 20261028 -->')
    return text


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--show',action='store_true');parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    if args.show:
        print((OUT/'report.md').read_text() if (OUT/'report.md').exists() else 'D08 cache-only calibration results are not yet collected.');return
    print(report(args.publish))


if __name__=='__main__':main()
