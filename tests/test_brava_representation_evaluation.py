"""Exercise original/split evaluation with explicit GT fixtures, never model results."""
import copy,io,json,pickle,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
from scripts import brava_representation_evaluation as ev
from scripts.brava_representation_study import ROOT,read
from scripts.analyze_spline_target_capacity import write

class ChangedGraphEvaluation(unittest.TestCase):
 def test_anatomical_reference_and_parent_reconstruction(self):
  p=read(ROOT/'plan.json');audit=read(ROOT/'target-audit-probe.json')
  with (ROOT/'targets-probe.pkl').open('rb') as f:cache=pickle.load(f)
  sample=max(cache,key=lambda s:len(cache[s]['split_K8']['nodes'])-len(cache[s]['K8']['nodes']))
  self.assertGreater(len(cache[sample]['split_K8']['nodes']),len(cache[sample]['K8']['nodes']))
  original=ev.ROOT
  with tempfile.TemporaryDirectory() as td:
   try:
    ev.ROOT=Path(td);write(ev.ROOT/'plan.json',p);write(ev.ROOT/'target-audit.json',audit)
    for group,arm in [('original','K8'),('split','split_K8')]:
     g=cache[sample][arm];n=len(g['nodes']);pairs=np.array([(a,b) for a in range(n) for b in range(a+1,n)]);links={tuple(sorted(e)) for e in g['edges']}
     c=dict(nodes=g['nodes'],edges=g['edges'],curves=g['curves'],physical=g['nodes']@g['metric'].T,truth=g['nodes']@g['metric'].T,gt_edges=g['edges'],scores=np.full(n,.99),pairs=pairs,relations=np.array([.99 if tuple(e) in links else .01 for e in pairs]),hungarian_distance=np.zeros(n),truth_original=cache[sample]['K8']['nodes']@g['metric'].T,matched_curves=g['curves'],oracle_endpoint_curves=g['curves'],matched_edges=np.arange(len(g['edges'])),parent_branch=g['parent_branch'])
     folder=ev.ROOT/'tiny-predictions'/group;folder.mkdir(parents=True);record=dict(member=arm+'/'+sample+'.npz',arm=arm,split='train',sample=sample,subject=p['subjects'][sample.split('_')[1]],gt=str(Path(p['dataset'])/'train/spline'/(sample+'.npz')))
     buf=io.BytesIO();np.savez_compressed(buf,**c)
     with zipfile.ZipFile(folder/'000512.zip','w') as z:z.writestr(record['member'],buf.getvalue())
     write(folder/'000512-records.json',[record]);ev.evaluate(group,512,tiny=True)
     result=read(ev.ROOT/'tiny-evaluation'/group/'000512/summary.json')[arm]['train']
     self.assertEqual(result['physical']['2.0']['node_f1'],1.)
     self.assertEqual(result['physical']['2.0']['edge_f1'],1.)
     self.assertEqual(result['common']['common_branch_f1'],1.)
     self.assertEqual(result['original_anchor_recall']['2.0']['raw'],1.)
     self.assertEqual(result['geometry']['all']['branches'],len(cache[sample]['K8']['edges']))
   finally:ev.ROOT=original

if __name__=='__main__':unittest.main()
