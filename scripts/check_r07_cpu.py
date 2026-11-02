"""Focused real-cache control parity before submitting the GPU-gated batch."""
import sys,io,zipfile,copy,json,csv,gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/r07_matched_supervision_20261027'
sys.path[:0]=[str(OUT/'frozen-source'),str(OUT/'runtime'),str(ROOT)]
import numpy as np
import torch
from torch.nn import functional as F
from scripts.r07_matched_supervision import balanced_features,score
from scripts.r07_sampling import population,redistribution_indices
from scripts.analyze_graph_errors import read,write,geometry,match_hungarian
from models.relationformer import MLP
from models.matcher import build_matcher
from training.losses.criterion import GraphCriterion
from scripts.collect_r07 import calibration,candidate_context,graph_metrics,aggregate,paired_graph_ci


def main():
    torch.set_num_threads(2)
    p=read(OUT/'plan.json');seed=364505;cfg=p['configs'][str(seed)]
    cp=torch.load(p['inputs'][0]['checkpoint'],map_location='cpu',mmap=True)
    head=MLP(2208,552,2,3)
    head.load_state_dict({k.removeprefix('relation_embed.'):v for k,v in cp['net'].items() if k.startswith('relation_embed.')});del cp
    items=[];prior=ROOT.parent/'gnbm-graph-errors-20261026/experiments/oracle_representatives_20261026'
    with zipfile.ZipFile(p['inputs'][0]['prediction_archive']) as z:
        for rec in [r for r in p['records'][str(seed)] if (prior/f'embeddings-{seed}-{r["sample"]}.npz').exists() and r['split']=='train']:
            with np.load(io.BytesIO(z.read(rec['sample']+'.npz'))) as f:c={k:f[k] for k in f.files}
            with np.load(prior/f'embeddings-{seed}-{rec["sample"]}.npz') as f:c['tokens']=f['tokens'][0]
            gt=geometry(rec['gt']);hs,ht,_=match_hungarian(c,gt,cfg)
            c.update(hungarian_source=hs,hungarian_target=ht,record=rec,population=population(c,gt,hs,ht));items.append(c)
            with torch.no_grad():prob=score(head,torch.as_tensor(c['tokens']),torch.as_tensor(c['pairs'])).numpy()
            np.testing.assert_allclose(prob,c['relations'],atol=1e-5,rtol=0)
    assert len(items)==2
    criterion=GraphCriterion(cfg,build_matcher(cfg),head)
    feature,label,hard,groups=balanced_features(criterion,items,seed)
    manual=F.cross_entropy(head(feature),label)
    torch.manual_seed(seed)
    tokens=torch.as_tensor(np.stack([c['tokens'] for c in items]));edges=[torch.as_tensor(c['gt_edges']) for c in items]
    assignments=[(torch.as_tensor(c['hungarian_source']),torch.as_tensor(c['hungarian_target'])) for c in items]
    original,_=criterion.loss_edges(tokens,None,edges,assignments)
    loss_error=float((manual-original).abs().detach())
    torch.testing.assert_close(manual,original,atol=1e-7,rtol=1e-6)
    a=torch.autograd.grad(manual,tuple(head.parameters()),retain_graph=True);b=torch.autograd.grad(original,tuple(head.parameters()))
    gradient_error=max(float((x-y).abs().max()) for x,y in zip(a,b))
    for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)
    index=redistribution_indices(label.numpy(),hard,groups,np.random.default_rng(seed))
    assert np.array_equal(label.numpy()[index],label.numpy())
    # Real-data evaluator replay must recover the archived graph results.
    rows=[]
    for c in items:
        gt=geometry(c['record']['gt']);context=candidate_context(c,gt,2.)
        r,_=graph_metrics(c,gt,context,c['relations'],.5,10.)
        r.update(sample=c['record']['sample'],subject=c['record']['subject']);rows.append(r)
    archive=ROOT.parent/'gnbm-graph-errors-20261026/experiments/graph_error_attribution_20261026/cases/SyntheticMRI_364505_A-xyz_final/rows.csv.gz'
    with gzip.open(archive,'rt') as f:old={r['sample']:r for r in csv.DictReader(f) if float(r['tolerance'])==2.}
    for r in rows:
        for key in ['nodes','gt_nodes','node_tp','edges','gt_edges','edge_tp','eligible_edges']:
            assert r[key]==int(float(old[r['sample']][key])),(key,r[key],old[r['sample']][key])
    ci=paired_graph_ci(rows,rows,20);assert ci['edge_f1']['delta']==0 and ci['edge_f1']['ci95']==[0.,0.]
    # Deliberately tied scores verify exact > threshold and training policy.
    y=np.array([1,0,1,0]);s=np.array([.9,.8,.8,.1]);cuts=calibration(y,s,2,.6)
    chosen=s>cuts['max_f1'];assert chosen.sum()==3 and y[chosen].sum()==2
    write(OUT/'cpu-smoke.json',dict(passed=True,real_cached_patches=2,original_loss_and_gradients_parity=True,
          loss_absolute_error=loss_error,gradient_max_absolute_error=gradient_error,
          baseline_score_parity_atol=1e-5,positive_negative_counts_equal=True,
          identical_subject_bootstrap_zero=True,threshold_tie_semantics=True,evaluator_summary=aggregate(rows)))
    print('R07 real-cache CPU smoke passed')

if __name__=='__main__':main()
