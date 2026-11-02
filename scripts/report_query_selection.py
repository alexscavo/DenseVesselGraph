"""Append completed N08 results to existing reports and experiment register."""
import fcntl,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_query_selection import OUT
from scripts.analyze_graph_errors import read,write

def find(rows,seed,method,split='val',tol=2.,**kw):return next(r for r in rows if r['seed']==seed and r['method']==method and r['split']==split and r['tolerance']==tol and all(r[k]==v for k,v in kw.items()))
def fmt(x):return 'NA' if x is None else f'{x:.4f}'
def ci(r):return f'{r["delta"]:+.6f} [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}]'

def append(name,section,marker):
    audit=read(OUT/'publication-audit.json')[name];path=Path(audit['path']);old=path.read_text();start='<!-- BEGIN '+marker+' -->';end='<!-- END '+marker+' -->';block=start+'\n'+section+'\n'+end
    if hashlib.sha256(old.encode()).hexdigest()!=audit['sha256'] or start in old:
        write(OUT/(name+'-integration-pending.json'),dict(target=str(path),reason='Concurrent edit or prior publication; preserve existing content',section=block));return
    assert path.read_text()==old;path.write_text(old.rstrip()+'\n\n'+block+'\n');write(OUT/(name+'-publication.json'),dict(completed=True,path=str(path),before_sha256=audit['sha256'],after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

def main():
    s=read(OUT/'summary.json');choice=read(OUT/'training-selection.json');changes=read(OUT/'selection-changes.json');primary=s['primary'];passed=s['decision']['consistent_count_controlled_benefit']
    verdict='Passed the predefined count-controlled gate' if passed else 'No consistent benefit over count-matched controls; stop further search'
    lines=['## N08 — GT-free relation-aware representative selection — 26 October 2026','',
    f'**Completed, CPU-only. {verdict}.** This follows N06’s GT-assisted oracle analysis and does not turn its 0.834–0.838 Edge F1 into deployable performance. No detector training, relation head, graph loss, architecture, new GPU inference or saved-prediction changes were introduced.','',
    '### Fixed data, inference interface and controls','',
    'The same two SyntheticMRI A-xyz checkpoints at update 12,000 are used: seeds 364505 and 92741, with the N06 checkpoint/cache checksums preserved in `input-identities.json`. Per seed: 128 training-monitor patches/32 subjects and 480 validation patches/13 disjoint subjects. These monitor patches were used during detector optimization and are not an independent generalization estimate. Test data were not accessed.','',
    'Selectors receive **only** predicted physical coordinates, object confidence, original unordered query pairs and their saved symmetric relation probabilities. The caches contain probabilities after averaging the two endpoint-order logits, rather than the original two-component logits; these probabilities retain the binary ranking/decision information required here. No GT coordinate, edge, node count, Hungarian assignment, vessel mask or oracle membership enters grouping, reference construction or query choice. GT is used afterward for training-only setting selection and offline evaluation.','',
    'Every selected vertex is an existing query, with its original coordinates/confidence. Every selected pair keeps its original relation score and the 0.5 edge cutoff. Alternative policies replace the node-selection decision; the 0.5 object cutoff is **not reapplied afterward**, which would discard rescued queries. Their node budget comes from normal inference or confidence NMS, never from GT.','',
    'The predeclared small candidate set was:','',
    '1. **Normal:** original object-confidence cutoff 0.5.','2. **Confidence NMS:** confidence order, 1-mm radius, original retained candidates. This reproduces the N06 control.','3. **Naive global support:** confidence plus uncapped sum of relation probabilities to fixed reference queries; intentionally retained as a hub/self-support risk control.','4. **Guarded global support:** bounded strongest-reference support, a reference-neighbor rank check, and a nonreciprocal-support penalty. References are up to 12 confidence-NMS queries with confidence ≥0.75. References within 2 mm are excluded from support; reference degree normalizes hub contributions. A single fixed scaffold prevents iterative amplification by newly chosen queries, but cannot certify that references are anatomically correct.','5. **Local profile selection:** one representative per original confidence-NMS seed, chosen inside disjoint cells with maximum radius `min(1 mm, 0.49 × nearest-seed separation)`. The score combines confidence with agreement to fixed predicted external neighbors and a strongest-negative profile penalty. These cells keep conservative local identities; they do not identify GT vertices.','6. **Protected NMS:** a predicted connection above 0.5 between nearby queries, or mean external-profile disagreement above 0.25, vetoes suppression. This tests preserving short predicted branches/distinct junctions, but can preserve false duplicate connections as well.','',
    'Global candidates require confidence ≥0.1. Each family uses a common weight from {0.25, 1}, chosen using training data only. The criterion is mean two-seed training Edge F1 advantage over the stronger confidence/spatial count control, with at most a one-percentage-point training node-recall loss. Ties prefer the smaller weight. No radius, threshold or reference-set search was performed. The selected weights were naive 0.25, guarded 0.25 and local profile 1.0. The **primary local-profile policy was frozen before validation** because it was closest to neutral; none of the families consistently beat both training controls.','',
    'All policies receive both controls at exactly the same predicted node count **per patch**: top original confidence, and confidence NMS truncated/backfilled by original confidence if needed. The pure confidence-NMS count controls for global/local policies are exactly the original NMS sets. No MST, cycle deletion, graph-tree assumption, hard anatomical degree cap or GT edge insertion is used.','',
    '### Validation reconstruction at 2 mm','',
    '| Seed | Policy | Selected nodes | Node P / R / F1 | Edge P / R / F1 | Edge TP / FP | Duplicates |','|---|---|---:|---|---|---:|---:|']
    methods=['normal','confidence_nms','naive_global','guarded_global','local_profile','protected_nms']
    for seed in [364505,92741]:
        for method in methods:
            r=find(s['summary'],seed,method);lines.append(f'| {seed} | {method} | {r["nodes"]} | '+ ' / '.join(fmt(r[k]) for k in ['node_precision','node_recall','node_f1'])+' | '+ ' / '.join(fmt(r[k]) for k in ['edge_precision','edge_recall','edge_f1'])+f' | {r["edge_tp"]} / {r["edge_fp"]} | {r["duplicates"]} |')
    lines+=['',
    'Raw-query 2-mm coverage is unchanged at 0.9232/0.9252 because the detector and all 120 proposals are unchanged. Filtered/selected recovery, 1-mm metrics and confidence-rejected selections are retained separately. Normal graph metrics and NMS counts reproduce N06 exactly.','',
    '### One-millimetre sensitivity','',
    '| Seed | Policy | Node P / R / F1 | Edge P / R / F1 | TP / FP |','|---|---|---|---|---:|']
    for seed in [364505,92741]:
        for method in ['normal','confidence_nms',primary]:
            r=find(s['summary'],seed,method,tol=1.);lines.append(f'| {seed} | {method} | '+ ' / '.join(fmt(r[k]) for k in ['node_precision','node_recall','node_f1'])+' | '+ ' / '.join(fmt(r[k]) for k in ['edge_precision','edge_recall','edge_f1'])+f' | {r["edge_tp"]} / {r["edge_fp"]} |')
    lines+=['','The local-profile difference remains mixed across seeds at 1 mm; it does not reveal a consistent stricter-localization benefit.','',
    '### Matched-count result and ranking','',
    '| Seed | Primary versus matched-count control | Edge F1 difference [subject 95% CI] | TP difference | FP difference |','|---|---|---|---:|---:|']
    for seed in [364505,92741]:
        for control in [primary+'_count_confidence',primary+'_count_spatial']:
            r=find(s['paired'],seed,primary,control=control);lines.append(f'| {seed} | {control} | {ci(r["edge_f1"])} | {r["edge_tp"]["delta"]:+.0f} | {r["edge_fp"]["delta"]:+.0f} |')
    lines+=['',
    'The local policy changes only 22/26 validation query choices versus NMS, in 22/26 patches out of 480 per seed. It selects four/one originally confidence-rejected queries. It adds only **one/two true edges and three/four false edges** versus the identical-count NMS control. Both graph-F1 intervals include zero, and the gain changes sign across seeds. This is not a usable recovery of the GT-oracle representative advantage. Conservatism limits its coverage; the result rules out this small fixed heuristic set, not every possible GT-free selector.','',
    '| Seed | Primary conditional AP | Primary all-retained AP | Shared-GT-anchor AP change versus NMS [subject CI] | Shared-anchor AP change versus normal |','|---|---:|---:|---|---|']
    for seed in [364505,92741]:
        r=find(s['ranking'],seed,primary);lines.append(f'| {seed} | {fmt(r["conditional_ap"])} | {fmt(r["all_retained_ap"])} | {ci(r["shared_nms_difference"])} | {ci(r["shared_normal_difference"])} |')
    lines+=['',
    'AP between independently selected populations is confounded by recovered anchors and positive prevalence. The shared-anchor comparison instead fixes the recovered GT vertices and their pair labels in each patch, comparing their existing representative scores. The local-NMS AP difference is tiny and positive in only one seed; versus normal representatives it is slightly negative in both. Candidate counts, positives and prevalence are saved for conditional, all-retained and both shared-anchor populations. No relation scores were recalibrated or reranked by a learned head.','',
    '### Connectivity costs and close structures','',
    '| Seed | Policy | Signed count error/patch: nodes / edges / β₀ / β₁ | GT shortcuts / near nonadjacent FP | Degree≥6 nodes | Short-edge recall | Close-junction recall |','|---|---|---|---:|---:|---:|']
    for seed in [364505,92741]:
        for method in methods:
            r=find(s['summary'],seed,method);lines.append(f'| {seed} | {method} | '+ ' / '.join(f'{r[k]:+.3f}' for k in ['nodes_signed_error_mean','edges_signed_error_mean','beta0_signed_error_mean','beta1_signed_error_mean'])+f' | {r["fp_gt_shortcut"]} / {r["fp_near_nonadjacent"]} | {r["pred_hubs_degree6"]} | {fmt(r["short_edge_recall"])} | {fmt(r["close_junction_recall"])} |')
    lines+=['',
    'GT shortcut means a wrong direct edge between localized GT anchors separated by at least two annotated GT graph hops. Nearby nonadjacent FP uses <5-mm GT endpoint separation and also includes disconnected GT components. Short positive edges have <5-mm endpoint separation. A close junction has degree≥3 and another GT anchor within 2 mm. These GT strata are strictly offline and never construct inference groups. Signed and absolute node/edge/component/cycle errors, recovered-anchor gains/losses and all denominators are saved. Degree≥6 is a diagnostic count, not a rule declaring such anatomy invalid.','',
    'Guarded support replaces 288/296 query choices versus NMS. It recovers 106/104 extra true edges but adds **283/285 false edges**; duplicates and cycles increase and Edge F1 decreases in both seeds. The bounded/reference-ranked score therefore does not solve self-support or spatial identity ambiguity. The uncapped rule is mixed across seeds and also increases cycles. This is consistent with the anticipated hub/false-support risk; it is not a direct causal proof that every extra false cycle arose from a particular clique.','',
    'Confidence NMS itself removes 139/143 recovered GT anchors relative to normal inference. Short-edge recall falls from 0.4940/0.4950 to 0.4281/0.4291, and close-junction recall also declines. Its small F1 gain is primarily a precision trade-off with lost true edges, not improved recovery of connections. Protected NMS retains most of these nearby nodes but also retains nearly all normal false edges/duplicates, returning essentially to normal graph quality. Consequently neither fewer nodes nor fewer cycles is treated as sufficient success.','',
    '### Conclusion and stopping decision','',
    f'**{verdict}.** All predeclared comparisons are complete. No extra score variants, validation-selected operating points, GPU inference, trainable selector or architecture were launched. The inference module remains an isolated diagnostic; it was not integrated into the model’s standard decoder.','',
    'N06 shows that appropriate existing queries carry useful relation information when GT supplies their spatial identities. N08 shows that **these simple inference-available confidence, distance and relation-support rules do not supply that missing identity reliably**. High internal relation support does not certify correct localization or unique GT-node identity. The strong original conditional relation ranking is preserved, while global support can select internally plausible but spatially wrong representatives.','',
    'The best-supported remaining hypothesis is **deeper coupling between spatial node quality, assignment/uniqueness and connectivity supervision**, rather than a demonstrated absence of information in the relation tokens. A future controlled supervision study could test whether spatially equivalent query alternatives receive coherent relation labels/quality calibration, keeping positives and total negative budgets explicit. That remains a hypothesis, not a result of N08. This negative study does not justify blanket unmatched-negative mining, which previously performed poorly, or a new large relation architecture. No deployable selection change is recommended from these results.','',
    '### Evidence and reproducibility','',
    f'Isolated worktree/branch: `gnbm-query-selection-20261026`, `research/gt-free-query-selection-20261026`; outputs: `{OUT}`. `training-selection.json` records the choice made before validation; `per-patch.csv.gz`, `aggregate.csv`, `per-subject.csv`, `paired-differences.json`, `ranking/`, `selections.jsonl.gz` and `selection-changes.json` retain decisions, all metrics and candidate populations. Node/edge thresholds, physical 1/2-mm matching and graph-count conventions are inherited unchanged. No geometric representation or curve target was modified.','',
    'Graph intervals resample whole subjects 2,000 times, AP intervals 500 times with exact tie handling, seed 261026. Validation is reused development evidence and exploratory; two continuations share one pretrained source, and the intervals omit source-training/seed uncertainty and multiplicity correction. Five focused selector tests and a two-patch-per-seed pipeline smoke passed. Source/cache/checkpoint hashes and baseline-parity checks are retained.','',
    'Print results from the parent `projects` directory:','',
    '```bash','python3 gnbm-query-selection-20261026/scripts/show_query_selection.py','```','',
    'CPU reproduction in the isolated worktree, using the protected research Python: `scripts/run_query_selection.py`, then `scripts/summarize_query_selection.py`. No inference or training is needed to display saved results. The available registry is `research/experiments.md`; no separate registry already containing N06/N08 was present in the audited files, so its existing history is preserved and a dated N08 entry is appended there.','']
    append('report','\n'.join(lines),'N08 GT FREE QUERY SELECTION 20261026')
    registry=f'''## N08 — GT-free representative selection — completed 26 October 2026

**Negative controlled diagnostic; stop further heuristic search.** Follows N06’s oracle-representative analysis. Reused two A-xyz step-12,000 SyntheticMRI caches (128 training/480 validation patches per seed; 32/13 disjoint subjects), no GPU or training. Training-only choices compared uncapped support, guarded reference-rank support, local profile representatives and protected NMS against unchanged inference, confidence NMS and exact per-patch node-count controls.

Primary local-profile Edge F1: **0.637638/0.637196**, versus confidence NMS **0.637675/0.637184**. Differences are inconsistent and both subject-bootstrap intervals include zero. Guarded support adds 106/104 TP but 283/285 FP versus NMS and increases cycles. Confidence NMS loses short branches/nearby junctions; protected NMS largely restores normal false edges. No deployable selection change, training or architecture is recommended. Remaining interpretation: spatial identity/assignment/connectivity coupling, not proven absence of relation-token information. See [N08 full protocol/results](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026). Numeric evidence is in sibling `gnbm-query-selection-20261026/experiments/gt_free_query_selection_20261026`; print with `python3 gnbm-query-selection-20261026/scripts/show_query_selection.py` from `projects`.
'''
    append('registry',registry,'N08 REGISTRY 20261026')
    roadmap='''## N08 stopping decision — 26 October 2026

Completed the bounded, CPU-only GT-free query-selection diagnostic; see [N08](../docs/research/metrics/6-ixi-query-attribution.md#n08--gt-free-relation-aware-representative-selection--26-october-2026) and [experiment register](experiments.md#n08--gt-free-representative-selection--completed-26-october-2026). Local relation-profile selection is neutral versus count-matched confidence NMS across two SyntheticMRI A-xyz seeds; global support increases false edges/cycles. No additional heuristic sweep, learned selector or architecture was launched. Keep original inference as the reference; do not adopt NMS solely for its small Edge F1 gain, because short-branch and nearby-junction recovery decline. The oracle selection gap remains diagnostic rather than deployable. Prioritize understanding spatial node identity, query assignment and relation-label consistency before replacing relation tokens or adding unmatched negatives; N08 alone does not establish a causal training fix.
'''
    # Cooperate with the existing diagnostics lock, and still hash-check before appending.
    path=Path(read(OUT/'publication-audit.json')['roadmap']['path']);lock=path.with_name(path.name+'.diagnostics.lock')
    with lock.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:write(OUT/'roadmap-integration-pending.json',dict(reason='Roadmap actively locked by another agent',section=roadmap))
        else:append('roadmap',roadmap,'N08 ROADMAP 20261026')
    print('N08 report/register/roadmap publication attempted with conflict guards')
if __name__=='__main__':main()
