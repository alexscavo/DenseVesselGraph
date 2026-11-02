"""Render completed verifier evidence without training or inference."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.node_verifier_feasibility import ROOT, paired, write


def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows])


def fmt(x):return f'{x:.4f}'
def interval(x):return f'[{x[0]:+.4f}, {x[1]:+.4f}]'


def main():
    results={d:json.loads((ROOT/'results'/d/'summary.json').read_text()) for d in ['SyntheticMRI','IXI']}
    rows=[];counts=[];graphs=[];cis=[];precision=[];discs=[];policies=[];ones=[];oracle=[];early=[]
    extra={}
    for dataset,s in results.items():
        patches=json.loads((ROOT/'results'/dataset/'patches.json').read_text())
        extra[dataset]=paired(patches['V1']['val']['calibrated_2.0'],patches['V0']['val']['calibrated_2.0'])
        for name,v in s['results'].items():
            r=v['val']['calibrated_2.0'];c=v['val']['matched_count_2.0'];g=v['val']['precision_envelope']
            rows.append([dataset,name,*[fmt(r[k]) for k in ['node_precision','node_recall','node_f1','duplicate_rate']],f"{int(r['rescued'])}/{int(r['lost'])}"])
            counts.append([dataset,name,int(c['nodes']),*[fmt(c[k]) for k in ['node_recall','node_f1','duplicate_rate']],f"{int(c['rescued'])}/{int(c['lost'])}"])
            graphs.append([dataset,name,*[fmt(r[k]) for k in ['edge_f1','conditional_recall','beta0_mae','beta1_mae']],fmt(c['edge_f1'])])
            selected=g['selected'];precision.append([dataset,name,fmt(g['target']),*[fmt(selected[k]) for k in ['node_precision','node_recall','threshold']]])
            policies.append([dataset,name,fmt(s['policies'][name]['threshold']),s['policies'][name]['radius']])
            a=v['train']['discrimination'];b=v['val']['discrimination']
            discs.append([dataset,name,f"{a['canonical_ap']:.4f}/{b['canonical_ap']:.4f}",f"{a['reliable_auc']:.4f}/{b['reliable_auc']:.4f}",fmt(b['auc_correct_duplicate']),fmt(b['brier']),fmt(b['ece'])])
            q=v['val']['matched_count_1.0'];ones.append([dataset,name,fmt(q['node_recall']),fmt(q['node_f1']),fmt(q['edge_f1'])])
        for control in ['V1','V2']:
            a=s['paired']['V3-'+control]
            for mode in ['calibrated_2.0','matched_count_2.0','precision_envelope_2.0']:
                cis.append([dataset,'V3−'+control,mode,interval(a[mode]['node_recall']),interval(a[mode]['node_f1']),interval(a[mode]['edge_f1'])])
        for tol,r in s['oracle'].items():oracle.append([dataset,tol,fmt(r['node_recall']),fmt(r['node_f1']),fmt(r['edge_f1']),fmt(r['conditional_recall'])])
        for row in s['learning_curve']:
            if row['split']=='val':early.append([dataset,row['arm'],row['step'],fmt(row['node_f1']),fmt(row['canonical_ap']),fmt(row['reliable_auc'])])
    text='''## Completed result and decision

**Decision: revise verifier supervision/training before considering any end-to-end fine-tuning. Keep V1 as the operational control; do not expand the architecture or launch longer training.** The tested contextual verifier fails the main feasibility criterion on both datasets. It improves some metrics over independent rescoring, but is substantially worse than original/calibrated classification at matched counts and matched precision. This is a negative result for the present frozen-decoder-feature recipe, not proof that all contextual verification is impossible.

The first discriminative experiments therefore do not pass the transfer gate: no BRAVA V2/V3 transfer inference or verifier training was launched. BRAVA D1 still receives full cached recovery analysis and a cached V0/V1 comparison; its independent unfrozen detector learnability tests completed below.

### Training-calibrated inference, 2 mm

128 validation patches per dataset. Rescued/lost counts compare unique spatial GT coverage with V0 on the same patch; rescuing some anchors while losing many more is not a net improvement.

'''+table(['Dataset','Arm','Precision','Recall','F1','Duplicates / retained','Rescued / lost'],rows)+'''

Selected node thresholds and NMS radii:

'''+table(['Dataset','Arm','Threshold','NMS mm'],policies)+'''

For original scores, training chooses **no NMS** on both datasets. Threshold calibration slightly improves node F1 while reducing recall; SyntheticMRI edge F1 also improves, IXI edge F1 decreases. The paired V1−V0 node-F1 intervals are '''+interval(extra['SyntheticMRI']['node_f1'])+' (SyntheticMRI) and '+interval(extra['IXI']['node_f1'])+''' (IXI). Thus simple suppression was tested but was not the training-selected best policy for either original detector.

### Equal retained-node counts, 2 mm

Every method retains exactly V0's per-patch predicted count. V1 and V0 have identical rankings because V1 selected radius zero; changing a scalar threshold cannot improve a matched-count ranking.

'''+table(['Dataset','Arm','Retained','Recall','F1','Duplicates / retained','Rescued / lost'],counts)+'''

Context has a modest benefit over V2 on IXI, but both learned scorers lose substantial recall relative to V1. Fewer duplicates alone do not establish useful verification.

### Unchanged-relation graph outcomes, 2 mm

The first four metrics use training-calibrated node selection. The last uses matched node counts. Relation probabilities and their 0.5 cutoff are unchanged throughout.

'''+table(['Dataset','Arm','Edge F1','Conditional recall','Component MAE','Cycle MAE','Matched-count edge F1'],graphs)+'''

Lower component/cycle errors at a more restrictive threshold can accompany fewer recovered nodes and edges. Under matched counts V3 does not repair graph connectivity; it increases component error relative to V1 in both datasets.

### Paired subject uncertainty

95% intervals for differences (0–1 scale). Precision-envelope intervals condition on the descriptively selected validation thresholds and do not include their selection uncertainty. All intervals omit training-seed variation.

'''+table(['Dataset','Comparison','Policy','Recall difference','Node F1 difference','Edge F1 difference'],cis)+'''

### Matched precision: descriptive PR envelope

The common target is V0's validation precision. All learned operating points attain that target within 0.0013 absolute precision on the 0.005 threshold grid. They use validation labels only to measure the PR envelope, never to claim an achievable calibrated policy.

'''+table(['Dataset','Arm','Target precision','Attained precision','Recall','Threshold'],precision)+'''

V3 loses recall versus V1 even when precision is matched. IXI V3 also trails V2 at this operating point, despite its positive matched-count node-F1 difference. There is no consistent contextual ranking advantage across operating conditions.

### Strict localization and oracle bounds on the same pilot patches

Achievable matched-count results at 1 mm:

'''+table(['Dataset','Arm','Recall 1 mm','Node F1 1 mm','Edge F1 1 mm'],ones)+'''

GT-dependent oracle node selection, reported separately:

'''+table(['Dataset','Tolerance mm','Oracle node recall','Oracle node F1','Unchanged-head edge F1','Conditional relation recall'],oracle)+'''

These are **candidate-selection bounds**, not attainable detector/verifier results. The minimum-distance node oracle is not optimized for relation recovery and can choose a different duplicate query than the relation head learned to connect.

### Training versus validation discrimination and calibration

AP uses all canonical spatial labels; reliable AUC and Brier/ECE use only supervised queries. Brier/ECE are for the raw scorer probabilities, before NMS. The V1 threshold sweep does not perform temperature/probability calibration.

'''+table(['Dataset','Arm','AP train / val','Reliable AUC train / val','Val correct/duplicate AUC','Val Brier','Val ECE'],discs)+'''

Training reliable AUC approaches 1, whereas validation canonical AP falls below the original detector. Tiny-set fitting and nonzero gradients rule out basic implementation/optimization failure. Lower IXI Brier despite lower canonical AP/coverage illustrates why calibration alone is not the success criterion under strong negative prevalence and ignored ambiguity.

Early-checkpoint validation trajectories below use raw scorer ranking **without NMS**, at V0's fixed per-patch node count, so their final rows differ from the NMS-assisted primary comparison. No early checkpoint is substituted for the predeclared final result.

'''+table(['Dataset','Arm','Update','Count-matched F1, no NMS','Canonical AP','Reliable AUC'],early)+'''

### What failed: proposals, features, or context?

1. **Proposal insufficiency is real but does not explain the learned degradation.** Full-population raw recall caps any fixed-coordinate verifier at 0.7818 IXI / 0.8741 SyntheticMRI at 2 mm and 0.4216/0.5499 at 1 mm. However, V0/V1 recover substantially more nodes than the learned verifiers from the exact same candidates. V3's loss relative to V1 is a selection failure within that recoverable population.
2. **The decoder features contain some useful signal.** IXI V3 canonical validation AP initially rises from V0's 0.3103 to 0.3614 at update 100; reliable AUC also improves early. A claim that frozen embeddings contain no relevant information is contradicted by these early discrimination results. They do not produce a matched-count coverage improvement.
3. **The present supervision/optimization does not preserve useful inference ranking.** Approximately 22–24% of queries are ignored by the ambiguity rules. Excellent fit on the reliable subset does not imply appropriate scores for ignored alternatives or selection of relation-compatible representatives. Both learned scorers overfit; canonical AP degrades as training discrimination rises. This suggests a label/coverage-objective and generalization issue worth inspecting before more training, rather than a need for a larger context module. The experiment does not causally separate label choice from feature generalization.
4. **Context is insufficient, not universally information-free.** It gives a positive IXI matched-count V3−V2 node-F1 interval and a SyntheticMRI edge-F1 improvement over V2, but fails decisively against V1. Extra architectural capacity or longer unfrozen runs are not supported. This is not a “works on SyntheticMRI, fails on IXI” outcome: the primary criterion fails on both.
5. **Relations remain an independent limitation.** Even GT-dependent node selection leaves low conditional relation recovery on IXI/BRAVA at the unchanged cutoff. Rescoring a different representative can also reduce recovered edges; the verifier must be judged on graph outcomes, not just reliable-query classification.
'''
    report=Path('docs/research/node_verifier_feasibility.md');body=report.read_text();before,rest=body.split('<!-- RESULTS -->');_,after=rest.split('<!-- END RESULTS -->');report.write_text(before+'<!-- RESULTS -->\n'+text+'\n<!-- END RESULTS -->'+after)
    write(ROOT/'results'/'calibration-paired.json',extra)
    # Standalone research figure, derived only from completed saved results.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    names=['V0','V1','V2','V3'];colors=['#555555','#00876c','#6699cc','#cc6633']
    for ax,(dataset,s) in zip(axes[:2],results.items()):
        values=[s['results'][n]['val']['matched_count_2.0']['node_f1'] for n in names]
        ax.bar(names,values,color=colors);ax.set_ylim(0,1);ax.set_title(dataset+' validation');ax.set_ylabel('Node F1 at matched counts (2 mm)')
    for dataset,s in results.items():
        for split,style in [('train','-'),('val','--')]:
            curve=[r for r in s['learning_curve'] if r['arm']=='V3' and r['split']==split]
            axes[2].plot([r['step'] for r in curve],[r['canonical_ap'] for r in curve],style,marker='o',label=dataset+' '+split)
    axes[2].set_xlabel('Verifier updates');axes[2].set_ylabel('Canonical node AP');axes[2].set_ylim(0,1);axes[2].legend(fontsize=8);axes[2].set_title('Contextual verifier discrimination')
    out=Path('docs/research/evidence/node_verifier_feasibility');out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'feasibility.png',dpi=180);fig.savefig(out/'feasibility.pdf');plt.close(fig)
    write(out/'summary.json',dict(datasets=results,calibration_paired=extra))


if __name__=='__main__':main()
