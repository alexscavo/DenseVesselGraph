"""Derived subject uncertainty and geometry-adjusted associations from saved tables."""
import csv,gzip,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.analyze_graph_errors import OUT,read,write,csvwrite,aggregate

def load_rows(path):
 with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
 for r in rows:
  for k in list(r):
   if k in ['endpoints_recovered','correct','raw_pair_feasible','retained_pair_feasible']:r[k]=r[k]=='True'
   elif k.endswith('_quartile'):r[k]=int(r[k])
   elif k in ['tolerance','seed','step']:r[k]=float(r[k])
 return rows

def adjusted(rows,field,adjustment):
 subjects=sorted({r['subject'] for r in rows});matrix=np.zeros((len(subjects),4,2,2));index={s:i for i,s in enumerate(subjects)}
 for r in rows:
  group=r[field+'_quartile'];stratum=r[adjustment+'_quartile']
  if group not in [0,3] or stratum<0:continue
  group=int(group==3);matrix[index[r['subject']],stratum,group]+=int(r['correct']),int(r['endpoints_recovered'])
 total=matrix.sum(0);valid=(total[:,:,1]>0).all(-1);weights=total[:,:,1].sum(-1)*valid
 def effect(m):
  good=(m[:,:,1]>0).all(-1)&valid;w=weights*good
  if not w.sum():return np.nan
  rate=np.divide(m[:,:,0],m[:,:,1],out=np.zeros((4,2)),where=m[:,:,1]>0)
  return float(((rate[:,1]-rate[:,0])*w).sum()/w.sum())
 point=effect(total);rng=np.random.default_rng(261026);values=[]
 for idx in rng.integers(len(subjects),size=(2000,len(subjects))):values.append(effect(matrix[idx].sum(0)))
 finite=np.asarray(values);finite=finite[np.isfinite(finite)]
 return dict(field=field,adjust_for=adjustment,delta_high_minus_low=point,ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,valid_draws=len(finite),eligible_by_stratum=total[:,:,1].astype(int).tolist(),success_by_stratum=total[:,:,0].astype(int).tolist(),interpretation='descriptive conditional association, not a causal curvature effect')

def fp_interval(rows):
 subjects=sorted({r['subject'] for r in rows});fields=['edge_fp','fp_mu','fp_uu','fp_duplicate','no_two_raw_queries','filtering_loss','retained_assignment_ambiguity','localized_relation_miss','gt_edges']
 matrix=np.array([[sum(float(r.get(k) or 0) for r in rows if r['subject']==s) for k in fields] for s in subjects]);rng=np.random.default_rng(261026);samples=matrix[rng.integers(len(subjects),size=(2000,len(subjects)))].sum(1)
 ratio=np.divide(samples[:,1]+samples[:,2],samples[:,0],out=np.zeros(2000),where=samples[:,0]>0)
 return dict(unmatched_fp_fraction_ci95=np.quantile(ratio,[.025,.975]).tolist(),subjects=len(subjects))

def main():
 summary=read(OUT/'summary.json');result={}
 for name in summary['replay']:
  path=OUT/'cases'/name.replace('/','_');pp=load_rows(path/'positives.csv.gz');rr=load_rows(path/'rows.csv.gz');out={}
  for tol in [1.,2.]:
   positive=[r for r in pp if r['split']=='val' and r['tolerance']==tol];rows=[r for r in rr if r['split']=='val' and r['tolerance']==tol]
   out[str(tol)]=dict(tortuosity_adjusted_for_chord=adjusted(positive,'tortuosity','chord_mm'),curvature_adjusted_for_arc=adjusted(positive,'curvature_per_mm','arc_mm'),false_edge_uncertainty=fp_interval(rows))
  result[name]=out
 write(OUT/'adjusted-geometry.json',result);print('Derived geometry associations and false-edge uncertainty completed.')
if __name__=='__main__':main()
