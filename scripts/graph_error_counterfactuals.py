"""GT-assisted confidence selection and indirect-path checks; cached scores only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
from scripts.analyze_graph_errors import cases,load_case,geometry,OUT,write,csvwrite,evaluate_mapping,aggregate,safe_selection

def confidence_oracle(c,tolerance):
 distance=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
 cost=1-c['scores'][:,None]+1e-5*distance/(1+tolerance)
 cost=np.where(distance<=tolerance,cost,(min(distance.shape)+1)*2.)
 i,j=linear_sum_assignment(cost);mapping={int(a):int(b) for a,b in zip(i,j) if distance[a,b]<=tolerance}
 return np.array(list(mapping),dtype=int),mapping

def reachable(adjacency,a,b,forbidden=frozenset()):
 seen={a};queue=deque([a])
 while queue:
  x=queue.popleft()
  if x==b:return True
  for y in adjacency.get(x,[]):
   if y in seen or (y!=b and y in forbidden):continue
   seen.add(y);queue.append(y)
 return False

def main():
 counter=[];misses=[];summaries=[]
 for case in cases():
  rows=[];rr=[]
  selected=dict(case,records=[r for r in case['records'] if r['split']=='val'])
  for record,c in load_case(selected):
   gt=geometry(record['gt'])
   for tol in [1.,2.]:
    ids,mapping=confidence_oracle(c,tol);r,z=evaluate_mapping(c,gt,tol,ids,mapping)
    r.update(case=case['name'],sample=record['sample'],subject=record['subject'],split='val',tolerance=tol,condition='raw_confidence_oracle',raw_node_tp=len(mapping),selected_original_rejected=int((c['scores'][ids]<=.5).sum()))
    rows.append(r)
    normal,flags=safe_selection(c,gt,tol,'normal');inverse={b:a for a,b in flags['mapping'].items()};edges=c['pairs'][flags['chosen']];links={tuple(sorted(map(int,e))) for e in edges};adj={int(q):[] for q in np.flatnonzero(c['scores']>.5)}
    for a,b in edges:adj[int(a)].append(int(b));adj[int(b)].append(int(a))
    forbidden={q for q in adj if len(adj[q])!=2 or q in flags['mapping']}
    for e,(a,b) in enumerate(gt['edges']):
     if int(a) not in inverse or int(b) not in inverse:continue
     q1,q2=inverse[int(a)],inverse[int(b)]
     if tuple(sorted((q1,q2))) in links:continue
     any_route=reachable(adj,q1,q2);degree2=reachable(adj,q1,q2,forbidden)
     rr.append(dict(case=case['name'],sample=record['sample'],subject=record['subject'],tolerance=tol,gt_edge=e,indirect_connection=any_route,degree2_chain_without_other_gt_anchors=degree2,arc_mm=gt['geometry'][e]['arc_mm']))
  counter.extend(rows);misses.extend(rr)
  for tol in [1.,2.]:
   summary=aggregate([r for r in rows if r['tolerance']==tol]);source=[r for r in rr if r['tolerance']==tol]
   summaries.append(dict(case=case['name'],tolerance=tol,condition='raw_confidence_oracle',**summary,localized_direct_misses=len(source),misses_with_indirect_route=sum(r['indirect_connection'] for r in source),misses_with_degree2_chain=sum(r['degree2_chain_without_other_gt_anchors'] for r in source)))
  print('counterfactual/indirect',case['name'],flush=True)
 write(OUT/'counterfactual-extensions.json',summaries);csvwrite(OUT/'confidence-oracle-per-patch.csv.gz',counter);csvwrite(OUT/'indirect-branch-misses.csv.gz',misses)
if __name__=='__main__':main()
