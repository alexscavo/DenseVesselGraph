"""Add subject-paired early-checkpoint graph intervals without repeating AP analysis."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.cheap_research_diagnostics import META,RELATION,read,write,probability
r=read(META/'relation-analysis.json');records=read(RELATION/'cache/records.json')
with np.load(RELATION/'cache/common-pairs.npz') as f:c={k:f[k] for k in f.files}
sizes=np.diff(c['offsets']);val=np.repeat([x['split']=='val' for x in records],sizes);names=np.repeat([x['subject'] for x in records],sizes)
subjects=sorted(set(names[val]));subject_lookup={x:i for i,x in enumerate(subjects)};ids=np.array([subject_lookup[x] for x in names[val]])
gt=np.array([sum(len(x['gt_arc_lengths']) for x in records if x['split']=='val' and x['subject']==s) for s in subjects])
rng=np.random.default_rng(191032);mult=np.array([np.bincount(d,minlength=len(subjects)) for d in rng.integers(len(subjects),size=(1000,len(subjects)))])
graph_boot={};source={}
for row in r['operating_points']:
 if row['split']!='val' or row['tolerance_mm']!=2:continue
 name=row['scorer']
 if name not in source:
  if name=='original':source[name]=probability(c['original_margin'][val])
  else:
   arm,step=name.split('/')
   with np.load(RELATION/'runs'/arm/f'scores-{int(step):06d}.npz') as f:source[name]=probability(f['margin'][val])
 choose=source[name]>row['threshold'];labels=c['labels_2'][val];tp=np.bincount(ids,weights=(choose&labels),minlength=len(subjects));pred=np.bincount(ids,weights=choose,minlength=len(subjects));aa=mult@tp;bb=mult@pred;gg=mult@gt
 graph_boot[(name,row['policy'])]=2*aa/(bb+gg)
rows=[]
for (name,policy),values in graph_boot.items():
 for reference in ['original']:
  if name==reference:continue
  delta=values-graph_boot[(reference,policy)];lo,hi=np.quantile(delta,[.025,.975])
  point=lambda n:next(x['edge_f1'] for x in r['operating_points'] if x['scorer']==n and x['policy']==policy and x['split']=='val' and x['tolerance_mm']==2)
  rows.append(dict(scorer=name,reference=reference,policy=policy,tolerance_mm=2,delta_edge_f1=point(name)-point(reference),ci95_low=float(lo),ci95_high=float(hi)))
r['paired_graph_intervals']=rows;write(META/'relation-analysis.json',r);write(META/'relation-graph-intervals.json',rows)
print('Added paired edge intervals at existing training-selected operating points:',len(rows))
