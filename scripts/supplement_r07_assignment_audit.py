"""Read-only R07 audit of nominal Hungarian versus spatial query identities."""
import io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_graph_errors import read,write,match_hungarian
from scripts.node_verifier_metrics import association
OUT=ROOT/'experiments/r07_matched_supervision_20261027'
ORIGINAL=ROOT.parent/'Graph-Native-Betti-Matching-Representation'


def main():
    p=read(OUT/'plan.json');rows=[];bins={}
    original_audit=read(OUT/'supervision-audit.json')['per_patch']
    for seed in p['seeds']:
        cfg=p['configs'][str(seed)];audit={r['sample']:r for r in original_audit if r['seed']==seed}
        lengths=np.concatenate([r['positive_arc_mm'] for r in audit.values() if r['split']=='train'])
        cuts=np.quantile(lengths,[.25,.5,.75]);bins[str(seed)]=cuts.tolist()
        folder=ORIGINAL/f'experiments/local_refinement_long/{seed}/runs/A-xyz'
        with zipfile.ZipFile(folder/'step-12000/caches.zip') as z:
            for record in p['records'][str(seed)]:
                sample=record['sample']
                with np.load(io.BytesIO(z.read(sample+'.npz'))) as f:c={k:f[k] for k in f.files}
                with np.load(record['gt']) as f:gt={k:f[k] for k in ['nodes','edges','metric']}
                source,target,error=match_hungarian(c,gt,cfg);assert error is None or error<2e-4
                distances=np.linalg.norm(c['physical'][:,None]-c['truth'][None],axis=-1)
                own=distances[source,target];nearest=distances[source].argmin(1)
                alternative=distances[source].copy();alternative[np.arange(len(source)),target]=np.inf
                margin=alternative.min(1)-own
                spatial=association(c['physical'],c['truth'],2.)
                sp=np.full(len(c['scores']),-1,int)
                for q,g in spatial.items():sp[q]=g
                hm=np.full(len(c['scores']),-1,int);hm[source]=target
                shared=sp[source]>=0
                retained=c['scores'][source]>.5
                adj=np.zeros((len(c['truth']),)*2,bool);a,b=gt['edges'].T;adj[a,b]=adj[b,a]=True
                left,right=c['pairs'].T
                assigned=(hm[left]>=0)&(hm[right]>=0)
                spatial_both=(sp[left]>=0)&(sp[right]>=0)
                nominal=adj[np.maximum(hm[left],0),np.maximum(hm[right],0)]
                spatial_positive=adj[np.maximum(sp[left],0),np.maximum(sp[right],0)]
                comparable=assigned&spatial_both
                active=(c['scores'][left]>.5)&(c['scores'][right]>.5)
                row=dict(seed=seed,split=record['split'],subject=record['subject'],sample=sample,
                    assigned_nodes=len(source),assigned_outside_2mm=int((own>2).sum()),
                    assigned_nearest_gt_disagrees=int((nearest!=target).sum()),
                    assigned_inside_2mm_but_margin_below_0p5=int(((own<=2)&(margin<.5)).sum()),
                    assigned_also_raw_spatially_selected=int(shared.sum()),
                    assigned_spatial_target_disagrees=int((shared&(sp[source]!=target)).sum()),
                    assigned_not_raw_spatially_selected=int((~shared).sum()),
                    retained_assigned_nodes=int(retained.sum()),
                    retained_assigned_nearest_gt_disagrees=int((retained&(nearest!=target)).sum()),
                    comparable_pair_identities=int(comparable.sum()),
                    nominal_negative_spatial_positive=int((comparable&~nominal&spatial_positive).sum()),
                    nominal_positive_spatial_negative=int((comparable&nominal&~spatial_positive).sum()),
                    retained_nominal_negative_spatial_positive=int((comparable&active&~nominal&spatial_positive).sum()),
                    retained_nominal_positive_spatial_negative=int((comparable&active&nominal&~spatial_positive).sum()))
                arc=np.asarray(audit[sample]['positive_arc_mm'])
                for q in range(4):row['positive_arc_quartile_'+str(q+1)]=int((np.searchsorted(cuts,arc,side='right')==q).sum())
                rows.append(row)
    aggregates=[]
    for seed in p['seeds']:
        for split in ['train','val']:
            rr=[r for r in rows if r['seed']==seed and r['split']==split]
            numeric=[k for k,v in rr[0].items() if isinstance(v,int) and k!='seed']
            aggregates.append(dict(seed=seed,split=split,patches=len(rr),subjects=len({r['subject'] for r in rr}),
                                   **{k:sum(r[k] for r in rr) for k in numeric}))
    write(OUT/'assignment-spatial-audit.json',dict(completed=True,read_only=True,
      definitions='Raw spatial identities use the established one-to-one 2-mm association over all 120 queries; nearest-GT disagreement is separately counted. Pair disagreements condition on both Hungarian-assigned endpoints also having spatial identities. Counts flag supervision/inference identity mismatch, not proof of wrong annotations. Node-category counts overlap.',
      no_new_inference_training_or_sampling=True,positive_arc_quartile_cuts_mm=bins,aggregate=aggregates,per_patch=rows))
    print(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
