"""Review collected R07 results; publish no new scores or training outcomes."""
import fcntl,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/r07_matched_supervision_20261027'
MAIN=ROOT.parent/'Graph-Native-Betti-Matching-Representation'


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def append_section(path,marker,section):
    old=path.read_text();begin='<!-- BEGIN '+marker+' -->';end='<!-- END '+marker+' -->'
    block=begin+'\n'+section.rstrip()+'\n'+end
    if begin in old:
        a=old.index(begin);b=old.index(end,a)+len(end);new=old[:a]+block+old[b:]
    else:new=old.rstrip()+'\n\n'+block+'\n'
    assert path.read_text()==old,'Concurrent documentation edit; rerun this publication-only script'
    path.write_text(new)


def main():
    result=read(OUT/'summary.json');assert result['completed']
    publication=read(OUT/'publication.json');assert publication['completed']
    plan=result['plan'];audit=read(OUT/'assignment-spatial-audit.json');jobs=read(OUT/'submission.json')
    def graph(seed,arm,tol=2.,split='val',policy='fixed'):
        step=0 if arm=='original' else 1000
        return next(r for r in result['summary'] if (r['seed'],r['arm'],r['step'],r['split'],r['tolerance'],r['policy'])==(seed,arm,step,split,tol,policy))
    def rank(seed,arm,tol=2.,split='val'):
        return next(r for r in result['ranking'] if (r['seed'],r['arm'],r['step'],r['split'],r['tolerance'])==(seed,arm,0 if arm=='original' else 1000,split,tol))
    def diff(seed,comparison,policy=None,population=None):
        return next(r for r in result['differences'] if r['seed']==seed and r['split']=='val' and r.get('tolerance',2.)==2. and r['comparison']==comparison and r.get('policy')==policy and r.get('population')==population)
    records=[]
    for seed in plan['seeds']:
        completed=[read(OUT/'large'/str(seed)/'runs'/a/'complete.json') for a in ['S0','S1']]
        assert all(r['completed'] and r['updates']==1000 for r in completed)
        assert completed[0]['effective_budget_sha256']==completed[1]['effective_budget_sha256']
        assert completed[0]['order_sha256']==completed[1]['order_sha256']
        assert completed[0]['exposure']['positive_slots']==completed[1]['exposure']['positive_slots']
        assert completed[0]['exposure']['negative_slots']==completed[1]['exposure']['negative_slots']
        records.extend(completed)
        for tol in [1.,2.]:
            rr=[graph(seed,a,tol) for a in ['original','S0','S1']]
            assert len({(r['nodes'],r['node_tp'],r['gt_nodes']) for r in rr})==1
    assert all(r['precision_target_feasible'] for r in result['thresholds'])
    lines=['## R07 reviewed conclusions and recovery record — 27 October 2026','',
      '**Completed: a small precision trade-off versus continued S0, without improved relation ranking or a net gain over original A-xyz.** S1 passes neither the predeclared discrimination/true-edge gate nor the original-checkpoint comparison. Retain the original checkpoint. No fully unfrozen confirmation, R08 or combined R06/N08 mechanism was launched.','',
      '1. **Are reliable difficult negatives missing from the current objective? No.** The 9,999-pair cap exceeds all unordered pairs from 120 queries, and the actual frozen-cohort audit confirms every matched negative enters S0. Only 18/28,494 training negatives per seed (0.063%) meet the conservative reliability gate; 92.25% have GT chord >=20 mm. Counts alone do not establish gradient dominance or prove inadequate supervision. S1 tests relative multiplicity, not newly discovered negatives.','',
      '2. **Does better sampling improve discrimination and reconstruction? Only a small operating-point trade-off was measured.** At fixed 0.5, S1−S0 Edge F1 is +0.000755 / +0.000684, with subject intervals above zero. It removes 25/19 FP but also loses 3/1 TP. Those losses are much smaller than the FP reductions, so this is not the earlier aggressive-mining collapse; nevertheless it fails the predefined no-TP-loss requirement. Conditional AP decreases slightly, and all-retained AP decreases in both seeds with intervals below zero. Small Branch-F1 gains versus S0 remain below the original checkpoint; Curve-F1 differences versus S0 include zero. Original A-xyz remains better on primary Edge F1, conditional/all-retained AP, Branch F1 and long-edge recall.','',
      '3. **Are the remaining errors primarily selection/representation rather than supervision? The endpoint budget remains dominant, but this is not a causal proof.** Earlier attribution put about 90.7–90.8% of missing GT edges in absent recovered-endpoint pairs, which fixed-node R07 cannot repair. The new audit separately identifies Hungarian-versus-spatial identity disagreements. Together these support prioritizing spatial quality, assignment/alternative-query label coherence and training-to-inference candidate coverage over more of this tiny matched-negative subset. They do not establish representation as the sole cause. N08 already rejected the tested selection heuristics; R07 introduces no selector. This matched-only audit does not establish reliable unmatched-query negatives, so it does not justify launching R08.','',
      '### Verified exposure and normalization','',
      '| Seed | Effective pair slots | Positive / negative slots (both arms) | Reliable hard slots S0 → S1 | Replaced negative slots |','|---:|---:|---|---|---:|']
    for seed in plan['seeds']:
        a,b=[r for r in records if r['seed']==seed]
        x,y=a['exposure'],b['exposure']
        lines.append(f'|{seed}|{x["pairs"]}|{x["positive_slots"]} / {x["negative_slots"]}|{x["reliable_hard_after"]} → {y["reliable_hard_after"]}|{y["replaced_slots"]} ({100*y["replaced_slots"]/y["negative_slots"]:.4f}%)|')
    lines+=['','Every positive slot and each graph/batch negative count is preserved after the original ratio balancing (0.15, tolerance 0.02). CE mean reduction, coefficient 6, optimizer and patch order match. The 4x exposure change is real, but the actual replaced fraction is about 0.195%, far below the 25% ceiling because the reliable subset is rare.','',
      '### Positional validation metrics, fixed 0.5','',
      '| Seed | Arm | 1-mm precision / recall / F1 | 2-mm precision / recall / F1 |','|---:|---|---|---|']
    for seed in plan['seeds']:
        for arm in ['original','S0','S1']:
            a,b=graph(seed,arm,1.),graph(seed,arm,2.)
            show=lambda r:' / '.join(f'{r[k]:.6f}' for k in ['edge_precision','edge_recall','edge_f1'])
            lines.append(f'|{seed}|{arm}|{show(a)}|{show(b)}|')
    lines+=['','The full table above retains TP/FP/FN, nearby FPR, long-edge recall, graph errors and common-v2 Branch/Curve F1 for all fixed and training-calibrated policies. Common-v2 uses its own 1-mm anchor/curve support and 0.25-mm sampling; it is not the 1/2-mm positional edge protocol.','',
      '### Fixed 2-mm AP populations and prevalence','',
      '| Seed / split | Conditional pairs / positives / prevalence | All-retained pairs / positives / prevalence | Hungarian pairs / positives / prevalence |','|---|---|---|---|']
    for seed in plan['seeds']:
        for split in ['train','val']:
            r=rank(seed,'original',split=split);parts=[]
            for kind in ['conditional','all_retained','hungarian_pairs']:
                n,k=r[kind+'_candidates'],r[kind+'_positives'];parts.append(f'{n} / {k} / {k/n:.6f}')
            lines.append('|'+str(seed)+' / '+split+'|'+'|'.join(parts)+'|')
    lines+=['','Within each population, sample/query IDs, labels and prevalence are identical across heads. Hungarian nominal labels, spatially recovered endpoint labels and all-retained inference labels are not interchangeable. Full 1-mm AP populations are preserved in `summary.json` and `ranking/*.npz`.','',
      '### Discrimination, learning curves and operating points','',
      '| Seed | S1−S0 all-retained AP [95% subject CI] | S1−original Edge F1 [95% subject CI] |','|---:|---|---|']
    for seed in plan['seeds']:
        a=diff(seed,'S1 minus S0 AP',population='all_retained');b=diff(seed,'S1 minus original',policy='fixed')['edge_f1']
        show=lambda d:f'{d["delta"]:+.6f} [{d["ci95"][0]:+.6f}, {d["ci95"][1]:+.6f}]'
        lines.append(f'|{seed}|{show(a)}|{show(b)}|')
    lines+=['','At 1k, S1 training conditional AP reaches 0.987217/0.989929 while validation is 0.974666/0.975581; both continuations fit training increasingly well while validation ranking declines. Final S1−S0 conditional AP is −0.000100/−0.000110; the second interval is below zero and the first includes zero. Thus the fixed-cutoff F1 increase is not improved discrimination.','',
      'Training-max-F1 and original-training-precision policies were calibrated at **2 mm**, then applied unchanged at both positional tolerances. At max-F1, S1 slightly raises TP and FP together relative to S0; nearby-negative FPR is unchanged. Training-precision constraints need not yield exactly matched validation precision. The saved validation-only PR readouts are diagnostic, not deployable thresholds. Neither policy overturns the original-checkpoint/gate conclusion.','',
      '![R07 train/validation learning curves](../../../../gnbm-r07-matched-supervision-20261027/experiments/r07_matched_supervision_20261027/learning-curves.png)','',
      '### Hungarian versus spatial identity audit','',
      '| Seed / split | Assigned queries | Nearest GT differs from Hungarian GT | Assigned query >2 mm from own GT | Comparable nominal-negative / spatial-positive pairs |','|---|---:|---:|---:|---:|']
    for r in audit['aggregate']:
        lines.append(f'|{r["seed"]} / {r["split"]}|{r["assigned_nodes"]}|{r["assigned_nearest_gt_disagrees"]}|{r["assigned_outside_2mm"]}|{r["nominal_negative_spatial_positive"]}|')
    lines+=['','Raw spatial identities use the established one-to-one 2-mm maximum-cardinality/minimum-distance association. Pair-disagreement counts condition on both Hungarian-assigned endpoints also having spatial identities. Categories overlap; being unselected by the raw nearest-query association does not make a query invalid. These are correspondence disagreements, not proof of wrong annotations, and no targets were relabelled. Positive arc-length quartiles use training-derived boundaries; the training set contributes 533 positive edges per quartile. Exact counts and cutoffs are in `assignment-spatial-audit.json`.','',
      '### Execution recovery and verification','',
      f'Original smoke 1613828 failed before training because an older import constructed a five-output rather than compact three-output coordinate head; dependent 1613829/1613830 were cancelled. Pinning the exact frozen model package restored strict checkpoint loading without changing architecture or weights. Recovery smoke **{jobs["smoke"]}** and both tasks in **{jobs["training_array"]}** completed exit 0. All four S0/S1 runs reached 1,000 updates.','',
      f'CPU collector **{jobs["collector_before_threshold_fix"]}** was stopped after a demonstrated float32 tie-boundary bug was found: a float64 `nextafter` threshold could round back onto a float32 score during `>` decoding. The fix steps down in the actual probability dtype before storing the cutoff. All seven targeted checks passed, including this regression. Corrected collector **{jobs["collector"]}** completed exit 0; every baseline-precision target is feasible in the corrected training calibration. Only CPU reporting was repeated; fixed-0.5 predictions and all training weights/slots are unchanged. Original failed logs and code identities are retained in the experiment folder.','',
      'The result is limited to 128 training-monitor patches/32 subjects, the same 480 development-validation patches/13 subjects and a fixed 1k frozen-head continuation. Two seeds share a pretrained source. Whole-subject intervals quantify neither source-training uncertainty nor all choices of reliability/sampling policy. No test data, BRAVA, R06 token module, new matcher, node refinement, additional loss or unmatched-query supervision entered this comparison.','']
    section='\n'.join(lines)
    (OUT/'reviewed-results.md').write_text(section)
    report=MAIN/'docs/research/metrics/8-ixi-relation-matching-calibration.md'
    registry=MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        append_section(report,'R07 REVIEWED RESULTS 20261027',section)
        old=registry.read_text();rows=old.splitlines()
        brief='All matched negatives already enter S0. Fourfold reliable-negative exposure gives tiny fixed-cutoff F1 gains vs S0 (+0.000755/+0.000684), removing25/19 FP but losing3/1 TP; all-retained AP worsens in both seeds. Both remain below original A-xyz. Gate failed; no unfrozen/R08 launch.'
        for i,row in enumerate(rows):
            if row.startswith('| **R07** |'):
                rows[i]='| **R07** | Modern SyntheticMRI reliable matched-negative redistribution, fixed CE/positive balance/pair budget, two frozen A-xyz seeds | **DONE — negative/mixed** | '+brief+' | [R07 reviewed results](../docs/research/metrics/8-ixi-relation-matching-calibration.md#r07-reviewed-conclusions-and-recovery-record--27-october-2026) |'
        observed='| **R07 produces tiny F1/precision trade-offs without better AP or original-checkpoint recovery — observed** | Retain original A-xyz. Do not promote this sampler, extend training or launch R08/unfrozen confirmation automatically. Investigate spatial-quality/assignment and training-to-inference label coverage before another supervision policy. |'
        if observed not in rows:
            at=next(i for i,row in enumerate(rows) if row.startswith('| **R07 improves nonadjacent'))
            rows.insert(at+1,observed)
        assert registry.read_text()==old,'Concurrent registry edit'
        registry.write_text('\n'.join(rows)+'\n')
        append_section(registry,'R07 REVIEWED 20261027','### R07 reviewed outcome — 27 October 2026\n\n'+brief+'\n\nRecovery smoke1624151 and GPU seeds1624152_0/1 completed; CPU1624153 was superseded solely for a verified float32 calibrated-threshold decoding bug, corrected CPU1624612 completed. Seven focused checks passed; no new GPU training during postprocessing recovery. Assignment/spatial audit and all source provenance remain in the isolated R07 folder.')
    checkpoints=[]
    for f in sorted((OUT/'large').glob('*/runs/*/relation-*.pt')):
        checkpoints.append(dict(path=str(f.resolve()),sha256=sha(f),bytes=f.stat().st_size))
    assert len(checkpoints)==12
    write(OUT/'checkpoint-manifest.json',checkpoints)
    write(OUT/'final-verification.json',dict(passed=True,seeds=plan['seeds'],training_runs=4,updates_per_run=1000,
      head_checkpoints=12,node_predictions_unchanged=True,equal_pair_and_class_budgets=True,
      hard_exposure_multiplier=4,calibrated_precision_targets_feasible=True,focused_tests_passed=7,
      gpu_smoke=jobs['smoke'],gpu_array=jobs['training_array'],cpu_collector=jobs['collector'],
      postprocessing_only_recovery=True,new_gpu_runs_for_postprocessing_fix=0,
      passed_gate=publication['passed_gate'],summary_sha256=sha(OUT/'summary.json'),no_r08_or_unfrozen_run=True))
    write(OUT/'review-publication.json',dict(completed=True,report=str(report),registry=str(registry),report_sha256=sha(report),registry_sha256=sha(registry)))
    files=['scripts/collect_r07.py','tests/test_r07_evaluation.py','scripts/supplement_r07_assignment_audit.py','scripts/finalize_r07_results.py','experiments/r07_matched_supervision_20261027/code-identities.json','experiments/r07_matched_supervision_20261027/threshold-decoding-fix.json']
    with zipfile.ZipFile(OUT/'postprocessing-reviewed-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
    print('R07 verified, reviewed and published; no follow-up training launched.')


if __name__=='__main__':main()
