"""Conditional AP, assignment strata, and accounting checks for oracle study."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scripts.oracle_representatives import OUT,selected_cases,matrices
from scripts.analyze_graph_errors import load_case,write,csvwrite,geometry
from scripts.node_verifier_metrics import association,rank_nms

def main():
    populations={};alignment=[]
    for case in selected_cases():
        for record,c in load_case(case):
            gt=geometry(record['gt']);assert np.array_equal(gt['edges'],c['gt_edges']);assert np.allclose(gt['truth'],c['truth'],atol=2e-5)
            scores,adj=matrices(c,gt);ids=np.flatnonzero(c['scores']>.5);keep,_=rank_nms(c['physical'][ids],c['scores'][ids],1.);control=ids[np.argsort(-c['scores'][ids],kind='stable')[:len(keep)]]
            for tol in [1.,2.]:
                for cond,selected in [('normal',ids),('confidence_nms_1mm',ids[keep]),('matched_count_confidence',control)]:
                    local=association(c['physical'][selected],c['truth'],tol);q=np.array([selected[i] for i in local]);g=np.array(list(local.values()),int);a,b=np.triu_indices(len(q),1)
                    key=(case['seed'],record['split'],tol,cond);populations.setdefault(key,[[],[]]);populations[key][0].extend(adj[g[a],g[b]]);populations[key][1].extend(scores[q[a],q[b]])
    ranking=[]
    for (seed,split,tol,cond),(y,p) in populations.items():ranking.append(dict(seed=seed,split=split,tolerance=tol,condition=cond,candidates=len(y),positives=int(np.sum(y)),conditional_ap=float(average_precision_score(y,p))))
    write(OUT/'conditional-ranking.json',ranking)
    rep=pd.read_csv(OUT/'representatives.csv.gz');pairs=pd.read_csv(OUT/'stable-neighbor-pairs.csv.gz');graphs=pd.read_csv(OUT/'graphs.csv.gz');inter=pd.read_csv(OUT/'interactions.csv.gz')
    rep=rep[rep.changed].copy();rep['assignment_group']=np.where(rep.confidence_hungarian_same_gt,'confidence_same_gt',np.where(rep.nearest_hungarian_same_gt,'nearest_same_gt','neither_same_gt'))
    keys=['seed','split','tolerance','sample','subject','gt_vertex'];joined=pairs.merge(rep,on=keys);strata=[]
    for key,x in joined.groupby(['seed','split','tolerance','assignment_group']):
        r=dict(seed=int(key[0]),split=key[1],tolerance=key[2],group=key[3],representatives=len(x[['sample','gt_vertex']].drop_duplicates()),pairs=len(x),positives=int(x.label.sum()))
        for name in ['nearest','confidence']:
            y=x.label.to_numpy();p=x[name+'_score'].to_numpy();r.update({name+'_ap':float(average_precision_score(y,p)) if y.any() else None,name+'_true_mean':float(p[y==1].mean()) if y.any() else None,name+'_false_mean':float(p[y==0].mean()) if (~y.astype(bool)).any() else None,name+'_tp':int(((p>.5)&(y==1)).sum()),name+'_fp':int(((p>.5)&(y==0)).sum())})
        strata.append(r)
    write(OUT/'assignment-strata.json',strata)
    ids=[]
    for key,x in rep.groupby(['seed','split','tolerance']):
        for name in ['nearest','confidence']:
            for q,n in x[name+'_query'].value_counts().items():ids.append(dict(seed=int(key[0]),split=key[1],tolerance=key[2],condition=name,query=int(q),count=int(n)))
    csvwrite(OUT/'query-identities.csv',ids)
    checks=[]
    for (seed,split,tol),x in graphs.groupby(['seed','split','tolerance']):
        a=x[x.condition=='nearest_shared'];b=x[x.condition=='confidence_shared'];v=pairs[(pairs.seed==seed)&(pairs.split==split)&(pairs.tolerance==tol)];i=inter[(inter.seed==seed)&(inter.split==split)&(inter.tolerance==tol)]
        r=dict(seed=int(seed),split=split,tolerance=tol)
        for label,field in [(1,'edge_tp'),(0,'edge_fp')]:
            delta=int(b[field].sum()-a[field].sum());one=v[v.label==label];both=i[i.label==label]
            d1=int((one.confidence_score>.5).sum()-(one.nearest_score>.5).sum());d2=int((both.both_score>.5).sum()-(both.nearest_score>.5).sum())
            assert delta==d1+d2,(r,field,delta,d1,d2)
            r[field+'_total_delta']=delta;r[field+'_one_changed_endpoint_delta']=d1;r[field+'_both_changed_endpoints_delta']=d2
        checks.append(r)
    write(OUT/'decomposition.json',checks)
    print('conditional AP, assignment strata and exact TP/FP decomposition saved')
if __name__=='__main__':main()
