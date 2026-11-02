"""Replay only the original coordinate MLP on saved exact decoder tokens (CPU)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from scripts.oracle_representatives import OUT,selected_cases
from scripts.analyze_graph_errors import load_case,geometry,csvwrite,write
from scripts.graph_error_counterfactuals import confidence_oracle
from scripts.node_verifier_metrics import association

def main():
    torch.set_num_threads(1);rows=[]
    for case in selected_cases():
        state=torch.load(case['checkpoint'],map_location='cpu',mmap=True)['net']
        chosen={p.name.split(f'embeddings-{case["seed"]}-')[1][:-4]:p for p in OUT.glob(f'embeddings-{case["seed"]}-*.npz')}
        for record,c in load_case(dict(case,records=[r for r in case['records'] if r['sample'] in chosen])):
            gt=geometry(record['gt'])
            with np.load(chosen[record['sample']]) as f:t=torch.from_numpy(f['tokens'][0,:120])
            with torch.inference_mode():
                for layer in range(3):
                    t=torch.nn.functional.linear(t,state[f'coord_embed.layers.{layer}.weight'],state[f'coord_embed.layers.{layer}.bias'])
                    if layer<2:t=t.relu()
                stage1=t.sigmoid().numpy()[:,:3]@gt['metric'].T
            nearest=association(c['physical'],c['truth'],2.);_,conf=confidence_oracle(c,2.)
            n={g:q for q,g in nearest.items()};h={g:q for q,g in conf.items()}
            for g in sorted(set(n)&set(h)):
                if n[g]==h[g]:continue
                row=dict(seed=case['seed'],sample=record['sample'],subject=record['subject'],split=record['split'],gt_vertex=g)
                for name,q in [('nearest',n[g]),('confidence',h[g])]:
                    row[name+'_stage1_error_mm']=float(np.linalg.norm(stage1[q]-c['truth'][g]));row[name+'_refined_error_mm']=float(np.linalg.norm(c['physical'][q]-c['truth'][g]));row[name+'_refinement_displacement_mm']=float(np.linalg.norm(stage1[q]-c['physical'][q]))
                rows.append(row)
    csvwrite(OUT/'embedding-stage1-coordinates.csv.gz',rows);write(OUT/'stage1-coordinate-audit.json',dict(source='exact replay tokens through unchanged frozen coordinate MLP; CPU only; no image inference',rows=len(rows)))
    print('stage1 coordinate replay saved',len(rows))
if __name__=='__main__':main()
