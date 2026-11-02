"""Review saved metrics and update research summaries; no inference/training."""
import csv,gzip,json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(json.loads((REPO/'experiments/multidataset_continuous/submission.json').read_text())['output_root'])
E=REPO/'docs/research/evidence/multidataset_continuous'


def readcsv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt') as f:return list(csv.DictReader(f))


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])


def bounded(path,begin,end,text):
    value=path.read_text()
    if begin in value:
        start,tail=value.split(begin,1);_,rest=tail.split(end,1);value=start+begin+'\n'+text+'\n'+end+rest
    else:value+='\n'+begin+'\n'+text+'\n'+end+'\n'
    path.write_text(value)


def main():
    assert json.loads((ROOT/'results/collection-complete.json').read_text())['completed']
    results=readcsv(ROOT/'results/all-results.csv');geo=readcsv(ROOT/'results/geometry-summary.csv');paired=readcsv(ROOT/'results/paired-representations.csv');timing=json.loads((ROOT/'smoke.json').read_text())['timings']
    def row(arm,split='val',policy='default'):
        return next(r for r in results if r['arm']==arm and r['split']==split and r['policy']==policy and float(r['tolerance_mm'])==2 and int(r['step'])==(30000 if arm.startswith('D') else 20000))
    def fmt(value,n=5):return f'{float(value):.{n}f}'
    brava=[]
    for arm in ['D1_IXI','D2_SyntheticMRI','D3_scratch']:
        x=row(arm);brava.append([arm,*[fmt(x[k]) for k in ['node_recall','raw_query_recall','f1','common_branch_f1','common_curve_f1','beta0_mae','beta1_mae']]])
    neurons=[]
    for dataset in ['IXI','SyntheticMRI']:
        for method in ['K8','K16','K24','adaptive','piece_C0','piece_C1']:
            arm=dataset+'_'+method;x=row(arm);g=next(z for z in geo if z['arm']==arm and z['split']=='val')
            neurons.append([dataset,method,fmt(x['common_curve_f1']),fmt(g['mean_mm']),fmt(g['neutral_mm']),fmt(g['controls'],2)])
    selectors=[];strata=[];oracle=[]
    for dataset in ['IXI','SyntheticMRI']:
        arm=dataset+'_adaptive';f=ROOT/'evaluation'/arm/'020000/selector-confusion.json';cm=json.loads(f.read_text());matrix=cm['matrix'];total=sum(sum(r) for r in matrix)
        selectors.append(dict(dataset=dataset,accuracy=cm['accuracy'],always_K4_accuracy=sum(matrix[0])/total,recall_K8=matrix[1][1]/sum(matrix[1]),recall_K12=matrix[2][2]/sum(matrix[2]),recall_K16=matrix[3][3]/sum(matrix[3]) if sum(matrix[3]) else None,matrix=matrix))
        for method in ['K8','K24','adaptive','piece_C0','piece_C1']:
            arm=dataset+'_'+method;data=[r for r in readcsv(ROOT/'evaluation'/arm/'020000/geometry.csv.gz') if r['split']=='val']
            for category,chosen in [('endpoints_within_1mm',[r for r in data if float(r['endpoint_max_mm'])<=1]),('K8_difficult',[r for r in data if float(r['K8_fit_sampled_max_mm'])>.5]),('K8_difficult_and_localized',[r for r in data if float(r['K8_fit_sampled_max_mm'])>.5 and float(r['endpoint_max_mm'])<=1]),('long_ge_25mm',[r for r in data if float(r['length_mm'])>=25])]:
                strata.append(dict(dataset=dataset,method=method,category=category,branches=len(chosen),mean_mm=sum(float(r['mean_mm']) for r in chosen)/len(chosen) if chosen else None,neutral_mm=sum(float(r['neutral_mean_mm']) for r in chosen)/len(chosen) if chosen else None))
            if method=='adaptive':oracle.append(dict(dataset=dataset,actual_mean_mm=sum(float(r['mean_mm']) for r in data)/len(data),GT_label_oracle_mean_mm=sum(float(r['oracle_head_mean_mm']) for r in data)/len(data)))
    population=[]
    for arm in ['D1_IXI','D2_SyntheticMRI','D3_scratch']:
        data=[r for r in readcsv(ROOT/'evaluation'/arm/'030000/relation-populations.csv.gz') if r['split']=='val']
        for kind in sorted({r['population'] for r in data}):
            group=[r for r in data if r['population']==kind];totals={k:sum(int(r[k]) for r in group) for k in ['candidates','positive_pairs','predicted_edges','false_edges']}
            population.append(dict(arm=arm,population=kind,false_positive_rate=totals['false_edges']/max(1,totals['candidates']-totals['positive_pairs']),**totals))
    payload=dict(selectors=selectors,geometry_strata=strata,GT_label_oracle=oracle,raw_query_relation_populations=population,source='Saved evaluations only; raw-query populations include confidence-rejected queries; no test data')
    (E/'interpretation-details.json').write_text(json.dumps(payload,indent=2)+'\n')
    paired_table=table(['Dataset','Arm vs K8','Curve F1 difference','95% subject interval'],[[r['dataset'],r['arm'].replace(r['dataset']+'_',''),fmt(r['curve_f1_difference'],6),f"[{fmt(r['low'],6)}, {fmt(r['high'],6)}]"] for r in paired if not r['arm'].endswith('_K8')])
    runtime=[]
    for method in ['K8','K16','K24','adaptive','piece_C0','piece_C1']:
        arm='IXI_'+method;par=timing[arm]['trainable_parameters'];mins=[json.loads((ROOT/'runs'/(dataset+'_'+method)/'complete.json').read_text())['seconds']/60 for dataset in ['IXI','SyntheticMRI']]
        runtime.append([method,f'{par:,}',fmt(mins[0],1),fmt(mins[1],1)])
    text='''# Completed experiment review —23 October2026

## Completion and reporting recovery

All15 training/prediction tasks1460519 and all15 CPU evaluation tasks1460520 completed successfully. BRAVA arms completed30,000 updates on8,768 training patches; each independent IXI/SyntheticMRI head completed20,000 updates on4,096 training patches. No test evaluation or new end-to-end representation training was performed.

Collector1460521 failed **after numerical aggregation**, while loading missing Plotly assets from the archived training runtime. Collection was rerun from saved results using the complete protected plotting environment. All tables, figures, three offline3D viewers and the study report are now published. No inference/retraining was required. The Slurm failure remains in accounting; successful reporting recovery is recorded by `results/collection-complete.json`.

The retrieval-command error was a working-directory issue. From the parent `projects/` directory use `python3 Graph-Native-Betti-Matching-Representation/scripts/show_multidataset_graph.py`; from the repository root use `python3 scripts/show_multidataset_graph.py`.

## 1. BRAVA: node recovery improves, graph connectivity remains inadequate

All1,796 validation patches /8 subjects, default edge threshold0.5 and2-mm positional matching; common-v2 Branch/Curve retain their own1-mm convention. Native graph-query population and GT are the same across these final K8 recovery arms.

'''+table(['Initializer','Node recall','Raw-query recall','Edge F1','Branch F1','Curve F1','Component MAE','Cycle MAE'],brava)+'''

**IXI initialization is best among the recovery arms**, but it is not a recovered high-quality graph detector. Its default edge F1 is0.02099 (95% subject-bootstrap interval[0.01949,0.02260]); the predefined gate required edge F1>0.04 and node recall>0.25. The gate fails.

The same-split historical IXI zero-shot reference has node recall0.23095, edge F10.02687 and Curve F10.05802. Fine-tuning improves node recall to0.34093 but lowers default edge/curve performance to0.02099/0.02220. It predicts249 correct and4,028 incorrect edges versus the reference376/8,160. Fewer false edges therefore coexist with fewer correct edges and greater fragmentation (component MAE7.65→14.34). This descriptive comparison uses the preserved reference, not a new tuned test.

Training-calibrated thresholds do not resolve connectivity: D1's train-selected0.425 gives validation precision0.03482, recall0.03939, edge F10.03696 and Curve F10.09217, with766 correct/21,236 incorrect edges. Branch F1 decreases0.00243→0.00158; cycle MAE rises0.124→2.211. These are recall/calibration trade-offs, not a demonstrated topology improvement.

D2's train-selected0.4 gives edge F10.02495, Curve F10.09674 and cycle MAE3.818. D3's0.375 gives edge F10.00257 and cycle MAE9.854. The threshold grid and selection used training data only.

### What this says about scratch failure

Scratch remains poor **on monitored training patches too**: final training node recall0.1159 and edge F10, versus full-validation node recall0.1056 and edge F10.00040 (4 correct/641 incorrect edges). Poor validation generalization alone cannot explain this. Losses decrease and the new LR floor remains positive; simply appending updates to an old zero-LR run is not the explanation tested here.

The static detach route and GPU gradient check exclude direct spline-loss gradients into the detector in this recipe. They do not identify a single cause of weak localization/relation learning. Intensity/domain differences, optimization/assignment behavior, confidence selection, graph-anchor distributions and incomplete annotations remain competing explanations. A small BRAVA training-patch learnability/BN/assignment diagnostic is justified before another long scratch run.

On the fixed128-patch validation monitor D1's node recall rises0.2271→0.3246 and mean Hungarian distance falls4.137→3.382 mm, while default edge F1 drops0.0292→0.0156. This directly separates better localization from worse default connectivity. D1's final training/validation edge F1 is0.0228/0.0210 on the reported populations; source-population/prevalence differences prevent attributing every train–validation AP difference to generalization.

### Selection and comparison limitations

D1's128-patch monitor selected step20,000 by a very small edge-F1 margin, but full validation scores step30,000 higher (0.02099 versus0.01910). The primary matched-final comparison remains30,000; the full-validation winner is retained by the collector. Any further selection must acknowledge this two-level validation process. D2 expands120 source queries to256; source history/query initialization prevent a pure source-data-only causal interpretation.

## 2. Actual learned geometry on IXI and SyntheticMRI

Every arm uses the same frozen dataset-specific detector, nodes, query population and relation predictions. These are genuine trained geometry heads, not GT-only fits. Adaptive inference selects a native K-specific projection; piecewise heads predict four shared-join sections with C0/C1 constraints. Their uniform parameter sections are not the same as the earlier error-adaptive GT partitions.

Validation:512 IXI patches /26 subjects and480 SyntheticMRI patches /13 volumes. “Mean error” includes incorrectly localized Hungarian endpoints; the endpoint-neutral diagnostic is evaluation-only.

'''+table(['Dataset','Head','Curve F1','Mean error mm','Endpoint-neutral mm','Controls/branch'],neurons)+'''

Complete graph counts and positional node/edge outcomes are identical across frozen geometry variants by construction. IXI's positional edge F1 is0.06074, conditional relation AP0.45437 and Branch F10.00528. SyntheticMRI's corresponding values are0.47013,0.89837 and0.10250. Candidate prevalence/populations differ between datasets, so AP values are not a controlled cross-domain ranking comparison.

### Paired uncertainty

'''+paired_table+'''

Intervals use1,000 paired subject resamples and do not include seed uncertainty. IXI K24 and C1 have small positive Curve F1 differences; SyntheticMRI adaptive K and C1 have small negative differences. There is no substantial, consistent reconstruction advantage across datasets. Endpoint-neutral errors also do not show a broad larger-capacity advantage.

### Difficult branches and correctly localized endpoints

IXI has88 validation branch occurrences with K8 sampled maximum fit error>0.5 mm; only2 also have both predicted endpoints within1 mm. SyntheticMRI has35 difficult occurrences,9 localized. Conclusions about difficult correctly localized branches are therefore weakly supported.

On the difficult IXI subset, mean error K8/K24/C1 is4.571/4.551/4.549 mm. On difficult SyntheticMRI it is2.808/2.812/2.859 mm. This does not reveal a large hidden benefit masked by the short-branch majority. Long-branch (≥25 mm), endpoint-neutral and localization strata are preserved in `geometry-stratified.csv` and `interpretation-details.json`.

Among all correctly localized pairs, mean errors K8/C1 are0.5978/0.5973 mm for IXI (688 pairs) and0.5530/0.5599 mm for SyntheticMRI (1,333 pairs). These conditional cohorts do not imply the same accuracy for the complete decoded graph.

## 3. Adaptive efficiency and selector limitations

Actual native adaptive heads average4.31 controls on IXI and4.54 on SyntheticMRI, versus8 for the fixed baseline. This is real variable-output prediction; it is distinct from the earlier K24 output-compression classifier. Curve F1 is unchanged within the IXI interval and slightly lower on SyntheticMRI.

Selector validation accuracy is91.02% on IXI, but always choosingK4 would score91.58%. On SyntheticMRI it is87.19% versus86.22% for alwaysK4. Complex-class recall is low: K8/K12 recall36.5%/15.0% on IXI and49.8%/10.2% on SyntheticMRI; no K16 case is classified correctly. Overall accuracy alone is misleading under these imbalanced labels.

Using **GT-fit K labels to choose among the trained projections**, as a diagnostic, yields mean errors1.8863 versus actual selection1.8870 mm on IXI and0.9722 versus0.9738 mm on SyntheticMRI. This is not an oracle choosing the best learned curve by its measured error. It shows that perfect reproduction of the current GT-complexity labels alone would not establish a large gain in this trained bank.

'''+table(['Head','Geometry-trained parameters','IXI training min','SyntheticMRI training min'],runtime)+'''

The bank's separately trained selector adds35,717 parameters to its1,666,868 geometry-trained parameters. Reduced output control count is therefore not a smaller model; bank training is also slower than K8. Runtimes are descriptive single-run wall times, not isolated inference benchmarks. Four-section C1 uses26 global control vectors, C0 uses29; C1 continuity is enforced by the basis, without another loss.

## 4. Unmatched-query populations: measured separately, no new mining trained

D1's raw-query analysis includes58.62 million candidate pairs across validation, **including confidence-rejected queries**. Their predicted false-edge counts must not be confused with the4,028 false edges in the normally decoded graph.

False-positive rates among labelled negative raw pairs are0.722% for Hungarian-matched/matched versus1.073% for Hungarian-unmatched-involving pairs; spatially matched/matched gives0.168% versus1.073% for spatially unmatched-involving pairs. Conditioning on population size matters.

There are also3,862 physically GT-positive raw pairs involving at least one Hungarian-unassigned query, versus1,412 among Hungarian-assigned pairs. Blanket labelling of every Hungarian-unmatched-involving pair as negative could therefore suppress spatially valid alternative query choices. This does not invalidate Hungarian training, but it reinforces the need to handle duplicates/assignment ambiguity before another negative-sampling experiment. These are annotation-relative2-mm associations, not independent anatomical truth.

## 5. Decisions and remaining experiments

1. **Keep BRAVA end-to-end representation expansion gated.** IXI initialization is the strongest recovery arm, but neither default nor training-calibrated connectivity meets the intended practical standard. Investigate training-patch learnability and assignment/confidence/BN behavior before another large scratch run or a loss sweep.
2. **Keep K8 as the strong geometry control on IXI/SyntheticMRI.** Their GT K8 fits are already very accurate; extra capacity gives small, mixed learned changes. C1 is a valid, trainable parameterization, not yet the selected journal method.
3. **Treat adaptive K as an efficiency result with selector limitations.** Examine rare-complexity classes and native-projection supervision before claiming an accuracy contribution. Add no joint selector loss by default.
4. **Do not discard piecewise geometry from these pilots.** They train four fixed parameter sections on frozen detectors; they do not test a recovered BRAVA model, the exact GT adaptive partition, or full end-to-end geometry learning.
5. Explicit degree-2 training/oracle-node geometry, recovered-BRAVA head comparisons, full end-to-end representation runs and isolated node/refinement/relation-sampling ablations remain unlaunched. No test data or new training was used for this review.

The overarching question remains open: this batch establishes trainable implementations and efficiency trade-offs, but does not establish a robust cross-dataset reconstruction advantage for adaptive geometry.

## Files and reproduction

- Detailed protocol and generated tables: [multi-dataset report](multidataset_continuous_graph_experiments.md).
- Compact evidence: [tables and figures](evidence/multidataset_continuous/report.md), [paired intervals](evidence/multidataset_continuous/paired-representations.csv), [analysis details](evidence/multidataset_continuous/interpretation-details.json).
- Held-out-development long-branch viewers: [BRAVA](evidence/multidataset_continuous/BRAVA-example.html), [IXI](evidence/multidataset_continuous/IXI-example.html), [SyntheticMRI](evidence/multidataset_continuous/SyntheticMRI-example.html). These show Hungarian-positive diagnostics, which need not be decoded edges.
- Summary command from the repository root: `python3 scripts/show_multidataset_graph.py`.
- Reproduce this interpretation from saved tables: `python3 scripts/report_multidataset_results.py`.
'''
    review=REPO/'docs/research/EXPERIMENT_REVIEW_20261023.md';review.write_text(text)
    # Link targets in the detailed report need one extra research-directory prefix.
    main=REPO/'docs/research/multidataset_continuous_graph_experiments.md';s=main.read_text();marker='\n<!-- INITIAL RESULTS -->';before,after=s.split(marker,1)
    begin='<!-- BEGIN COMPLETED INTERPRETATION -->';end='<!-- END COMPLETED INTERPRETATION -->'
    interpretation=text.replace('# Completed experiment review —23 October2026','## Completed interpretation —23 October2026',1)
    if begin in before:
        prefix,tail=before.split(begin,1);_,suffix=tail.split(end,1);before=prefix+suffix
    before=before.replace('Initial scientific results are pending.','All15 training and evaluation arms completed; numerical results and reporting recovery are documented below.')
    main.write_text(before+'\n'+begin+'\n'+interpretation+'\n'+end+marker+after)
    bounded(REPO/'docs/research/ALL_EXPERIMENTS_AND_RESULTS.md','<!-- BEGIN MULTIDATASET REVIEW 20261023 -->','<!-- END MULTIDATASET REVIEW 20261023 -->',text)
    print(review)


if __name__=='__main__':main()
