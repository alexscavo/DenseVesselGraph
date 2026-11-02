"""Collect paired subject evidence; append only the new section of an existing report."""
import argparse,copy,hashlib,io,json,pickle,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.brava_representation_study import ROOT,read,ORIGIN,ARMS
from scripts.analyze_spline_target_capacity import write
from scripts.multidataset_graph_evaluation import clean
from scripts.multiscale_pretrained import summarize
from scripts.node_verifier_metrics import discrimination,association,measure,aggregate
from metrics.common_vessel_metrics import aggregate_common_vessels


def common(rows):return aggregate_common_vessels([{k:v for k,v in r.items() if k.startswith('common_')} for r in rows])
def paired(a,b):
 aa={r['sample']:r for r in a};bb={r['sample']:r for r in b};assert aa.keys()==bb.keys()
 subjects=sorted({r['subject'] for r in a});keys=['common_branch_f1','common_curve_f1','common_anchor_f1','original_anchor_recall','beta0_mae','beta1_mae']
 def stats(rr):
  v=common(rr);v.update(original_anchor_recall=sum(r['filtered_original_anchor_tp'] for r in rr)/sum(r['original_anchors'] for r in rr),beta0_mae=sum(r['beta0_error'] for r in rr)/sum(r['patches'] for r in rr),beta1_mae=sum(r['beta1_error'] for r in rr)/sum(r['patches'] for r in rr));return v
 def group(rr):return {s:[r for r in rr if r['subject']==s] for s in subjects}
 ga,gb=group(a),group(b);sa,sb=stats(a),stats(b);observed={k:sb[k]-sa[k] for k in keys};draws={k:[] for k in keys};rng=np.random.default_rng(261039)
 for selection in rng.choice(subjects,(1000,len(subjects)),replace=True):
  x=stats([r for s in selection for r in ga[s]]);y=stats([r for s in selection for r in gb[s]])
  for k in keys:draws[k].append(y[k]-x[k])
 return dict(subjects=len(subjects),draws=1000,metrics={k:dict(delta=observed[k],ci95=np.quantile(draws[k],[.025,.975]).tolist()) for k in keys})


def confidence(group):
 folder=ROOT/'predictions'/group;records=read(folder/'004000-records.json');out={}
 with zipfile.ZipFile(folder/'004000.zip') as z:
  for split in ['train','val']:
   caches=[]
   for rec in records:
    if rec['split']!=split or rec['arm'] not in ['K8','split_K8']:continue
    with np.load(io.BytesIO(z.read(rec['member']))) as f:caches.append({k:f[k] for k in f.files})
   out[split]=discrimination(caches,[c['scores'] for c in caches]);out[split]['oracle_candidate_selection']={}
   for tol in [1.,2.]:
    rr=[measure(c,list(association(c['physical'],c['truth'],tol)),tol) for c in caches]
    out[split]['oracle_candidate_selection'][str(tol)]=aggregate(rr)
 return out


def collect():
 plan=read(ROOT/'plan.json');result=dict(plan=plan,jobs=read(ROOT/'jobs.json'),runs={},comparisons={});rows={};final={};all_summaries={}
 for group in ['original','split']:
  complete=read(ROOT/'runs'/group/'complete.json');assert complete['completed'] and complete['steps']==4000
  result['runs'][group]=dict(complete=complete,tiny=read(ROOT/'tiny'/group/'complete.json'),gradients=read(ROOT/'runs'/group/'gradient-check.json'),confidence=confidence(group),monitors=read(ROOT/'runs'/group/'monitors.json'))
  for step in [1000,2000,4000]:
   path=ROOT/'evaluation'/group/f'{step:06d}';summary=read(path/'summary.json');all_summaries.setdefault(str(step),{}).update(summary)
  rr=read(ROOT/'evaluation'/group/'004000/rows.json')
  for arm in sorted({r['arm'] for r in rr}):rows[arm]=[r for r in rr if r['arm']==arm and r['split']=='val' and r['tolerance']=='2.0']
  final.update(read(ROOT/'evaluation'/group/'004000/summary.json'))
 assert result['runs']['original']['complete']['image_stream_sha256']==result['runs']['split']['complete']['image_stream_sha256']
 for arm in ['K24','C1']:
  assert final[arm]['val']['physical']==final['K8']['val']['physical']
  assert final[arm]['val']['common']['common_branch_f1']==final['K8']['val']['common']['common_branch_f1']
 for arm in ['K24','C1','split_K8']:result['comparisons'][arm+'-minus-K8']=paired(rows['K8'],rows[arm])
 result.update(final=final,trajectories=all_summaries,selections={})
 for arm in ARMS:
  best=max([1000,2000,4000],key=lambda s:(all_summaries[str(s)][arm]['val']['common']['common_branch_f1'],all_summaries[str(s)][arm]['val']['common']['common_curve_f1'],-s));result['selections'][arm]=dict(primary=4000,secondary=best)
 audit=read(ROOT/'target-audit.json');ar=audit['rows'];oracle=[r for x in ar for r in x['oracle']];result['target_summary']=dict(patches=len(ar),nodes_mean=float(np.mean([r['nodes'] for r in ar])),split_nodes_mean=float(np.mean([r['split_nodes'] for r in ar])),nodes_max=max(r['nodes'] for r in ar),split_nodes_max=max(r['split_nodes'] for r in ar),added_nodes=sum(r['split_nodes']-r['nodes'] for r in ar),unresolved_original_branches=sum(len(r['unresolved_branches']) for r in ar),original_branches=len(oracle),outside_sigmoid_nodes=sum(r['outside_sigmoid_nodes'] for r in ar),outside_sigmoid_non_boundary=sum(r['outside_sigmoid_non_boundary'] for r in ar),sigmoid_clamp_max_mm=max(r['sigmoid_clamp_max_mm'] for r in ar),difficult_original_branches=sum(r['K8_sampled_max_mm']>1 for r in oracle),gt_sampled_max={name:dict(median=float(np.median([r[name+'_sampled_max_mm'] for r in oracle])),p95=float(np.quantile([r[name+'_sampled_max_mm'] for r in oracle],.95))) for name in ['K8','K24','C1']})
 write(ROOT/'analysis.json',clean(result));render(result)


def render(result):
 fmt=lambda x:'NA' if x is None else f'{x:.4f}'
 lines=['### Completed bounded measurements — 26 October 2026','', 'Primary: fixed 4,000 updates, one training seed and eight paired validation subjects. Native positional metrics use each arm’s own node/edge targets; subdivision changes those denominators. Use original-anchor recall and common Branch/Curve F1 for graph-representation comparisons. Intervals are paired subject bootstrap intervals, not seed uncertainty.','', '| Arm | Raw R1/R2, native | Filtered R1/R2, native | Node F1, 2 mm | Edge F1, 2 mm | Original-anchor R2 | Branch F1 | Curve F1 |','|---|---|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  v=result['final'][arm]['val'];a,b=v['physical']['1.0'],v['physical']['2.0'];cm=v['common']
  lines.append(f"| {arm} | {a['raw_recall']:.4f}/{b['raw_recall']:.4f} | {a['node_recall']:.4f}/{b['node_recall']:.4f} | {b['node_f1']:.4f} | {b['edge_f1']:.4f} | {v['original_anchor_recall']['2.0']['filtered']:.4f} | {cm['common_branch_f1']:.4f} | {cm['common_curve_f1']:.4f} |")
 lines+=['','| Comparison | Branch F1 delta [95% interval] | Curve F1 delta [95% interval] |','|---|---:|---:|']
 for name,comp in result['comparisons'].items():
  def cell(k):
   x=comp['metrics'][k];return f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
  lines.append(f"| {name} | {cell('common_branch_f1')} | {cell('common_curve_f1')} |")
 lines+=['','| Arm | Train/val node F1, 2 mm | Train/val edge F1, 2 mm | Hungarian mm | Components MAE | Cycles MAE | Duplicates |','|---|---|---|---:|---:|---:|---:|']
 for arm in ARMS:
  t=result['final'][arm]['train']['physical']['2.0'];v=result['final'][arm]['val']['physical']['2.0']
  lines.append(f"| {arm} | {t['node_f1']:.4f}/{v['node_f1']:.4f} | {t['edge_f1']:.4f}/{v['edge_f1']:.4f} | {v['hungarian_mm']:.3f} | {v['beta0_mae']:.3f} | {v['beta1_mae']:.3f} | {v['duplicate_rate']:.4f} |")
 lines+=['','All geometry rows below are GT-edge-conditioned diagnostics: Hungarian associations select the annotated positive edges regardless of predicted confidence or relation scores. They are not geometry metrics on the inferred graph. Subedges are joined per original parent branch. Endpoint-neutral geometry linearly corrects the two original endpoints; oracle-endpoint geometry reevaluates the head with GT endpoints (including every added subdivision anchor) and predicted query features. Difficult means original K8 GT-fit sampled maximum error > 1 mm; localized means both original endpoints within 2 mm. Distances are symmetric mean point-to-polyline distances with 0.1-mm sampling. Zero eligible branches are NA.','', '| Arm | Stratum | Branches | GT-conditioned predicted-endpoint mean mm | Endpoint-neutral mean mm | Oracle-endpoint mean mm |','|---|---|---:|---:|---:|---:|']
 for arm in ARMS:
  for label,x in result['final'][arm]['val']['geometry'].items():lines.append(f"| {arm} | {label} | {x['branches']} | {fmt(x['mean_mm'])} | {fmt(x['neutral_mean_mm'])} | {fmt(x['oracle_endpoint_mean_mm'])} |")
 lines+=['','| Arm | Primary update | Secondary validation-selected update |','|---|---:|---:|']
 for arm,x in result['selections'].items():lines.append(f"| {arm} | {x['primary']} | {x['secondary']} |")
 lines+=['','Full numeric results, assignment/confidence/BN monitors, train/validation trajectories and target audit: `experiments/brava_representation_20261026/analysis.json`. Negative results remain in the same register.','']
 path=Path('docs/research/BRAVA_experiment_plan.md');text=path.read_text();marker='<!-- BRAVA REPRESENTATION 20261026 RESULTS -->';end='<!-- END BRAVA REPRESENTATION 20261026 RESULTS -->';block=marker+'\n'+'\n'.join(lines)+end
 if marker in text:text=text[:text.index(marker)]+block+text[text.index(end)+len(end):]
 else:text+='\n'+block+'\n'
 path.write_text(text)

if __name__=='__main__':collect()
