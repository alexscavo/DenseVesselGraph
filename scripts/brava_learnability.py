"""Eight-patch unfrozen BRAVA learnability test with unchanged K8 and edge CE."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.node_verifier_feasibility import ROOT, SOURCE, SEED, write
from scripts.node_verifier_metrics import measure, aggregate, association


def main():
    from scripts.multidataset_graph_training import setup
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    from scripts.ixi_node_focal_ablation import atomic_checkpoint
    p=argparse.ArgumentParser();p.add_argument('--initializer',choices=['scratch','ixi_spline'],required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(SEED);np.random.seed(SEED)
    out=ROOT/'brava'/a.initializer;out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(SOURCE.read_text());spec=plan['datasets']['BRAVA'];ids=[];subjects=set()
    for sample in spec['monitor_train']:
        subject=spec['subjects'][sample.split('_')[1]]
        with np.load(Path(spec['root'])/'train'/'spline'/(sample+'.npz')) as g:
            if 5<=len(g['nodes'])<=25 and len(g['edges'])>=3 and subject not in subjects:
                ids.append(sample);subjects.add(subject)
        if len(ids)==8:break
    assert len(ids)==8
    plan['datasets']['BRAVA']['train_ids']=ids
    plan['arms']=[dict(name='brava_learnability_'+a.initializer,initializer=a.initializer,dataset='BRAVA',kind='detector',representation='K8')]
    plan['base_config']['loss']['node']['classification'].update(name='weighted_cross_entropy',class_weights=[.25,.75])
    plan['base_config']['loss']['edge']['classification'].update(name='cross_entropy',class_weights=[1.,1.])
    write(out/'plan.json',plan)
    bundle=setup(out,0);_,_,model,cfg,criterion,opt,dataset,collate,_=bundle
    dataset.augment=False;dataset.intensity_transform=None
    cfg['data']['train_augmentation']=False;cfg['data']['batch_size']=4
    write(out/'config.json',cfg)
    samples=[dataset[i] for i in range(len(dataset))]
    batches=[_move_graph_batch(collate(samples[j:j+4]),'cuda','image',True) for j in [0,4]]
    bn0={n:b.detach().clone() for n,b in model.named_buffers() if n.endswith(('running_mean','running_var'))}
    prior={};observations=[]

    @torch.no_grad()
    def monitor(step):
        saved={n:b.clone() for n,b in model.named_buffers()};record=dict(step=step,modes={})
        for mode in ['eval','train']:
            model.train(mode=='train');rows=[];rawrows=[];orows=[];hungarian=[];assigned=[];unassigned=[];churn=[];positive_relations=[];localized_relations=[]
            for bi,(volumes,targets) in enumerate(batches):
                tokens,pred,_=model(volumes);assignments=criterion.matcher(pred,targets)
                graphs=infer_graphs(tokens,pred,model.relation_embed,object_queries=model.obj_token,relation_tokens=criterion.relation_tokens,node_threshold=-1.,edge_threshold=-1.)
                for b,graph in enumerate(graphs):
                    src,tgt=assignments[b];g=targets['spline'][b];metric=g['metric'].cpu().numpy();xyz=pred['pred_nodes'][b,:,:3].cpu().numpy()@metric.T
                    truth=g['nodes'].cpu().numpy()@metric.T if 'nodes' in g else targets['nodes'][b][:,:3].cpu().numpy()@metric.T
                    probs=pred['pred_logits'][b].softmax(-1)[:,1].cpu().numpy();pairs=graph['query_ids'][graph['edges']].numpy()
                    c=dict(physical=xyz,truth=truth,gt_edges=targets['edges'][b].cpu().numpy(),scores=probs,pairs=pairs,relations=graph['edge_scores'].numpy())
                    rows.append(measure(c,np.flatnonzero(probs>.5)));rawrows.append(measure(c,np.arange(len(probs)),graph=False));orows.append(measure(c,list(association(xyz,truth))))
                    dist=np.linalg.norm(xyz[src.numpy()]-truth[tgt.numpy()],axis=-1);hungarian.extend(dist.tolist());assigned.extend(probs[src.numpy()].tolist());unassigned.extend(np.delete(probs,src.numpy()).tolist())
                    matrix=np.zeros((len(probs),len(probs)));matrix[pairs[:,0],pairs[:,1]]=c['relations'];matrix+=matrix.T
                    target_to_query={int(t):int(s) for s,t in zip(src,tgt)};target_distance={int(t):float(d) for t,d in zip(tgt,dist)}
                    for u,v in c['gt_edges']:
                        value=float(matrix[target_to_query[int(u)],target_to_query[int(v)]])
                        positive_relations.append(value)
                        if target_distance[int(u)]<=2 and target_distance[int(v)]<=2:localized_relations.append(value)
                    key=(mode,bi,b);mapping={int(t):int(s) for s,t in zip(src,tgt)}
                    if key in prior:churn.extend([mapping[t]!=q for t,q in prior[key].items()])
                    prior[key]=mapping
            record['modes'][mode]=dict(filtered=aggregate(rows),raw=aggregate(rawrows),oracle_selection=aggregate(orows),hungarian_mean_mm=float(np.mean(hungarian)),hungarian_median_mm=float(np.median(hungarian)),hungarian_within_1mm=float(np.mean(np.array(hungarian)<=1)),hungarian_within_2mm=float(np.mean(np.array(hungarian)<=2)),assigned_confidence=float(np.mean(assigned)),unassigned_confidence=float(np.mean(unassigned)),assignment_change_fraction=float(np.mean(churn)) if churn else None,hungarian_positive_relation_recall=float(np.mean(np.array(positive_relations)>.5)),localized_hungarian_positive_relation_recall=float(np.mean(np.array(localized_relations)>.5)) if localized_relations else None,localized_hungarian_positive_pairs=len(localized_relations))
        for n,b in model.named_buffers():b.copy_(saved[n])
        record['bn_running_rms_change']=float(np.mean([float((saved[n]-v).square().mean().sqrt()) for n,v in bn0.items()]))
        observations.append(record);write(out/'monitor.json',observations);print(a.initializer,step,record,flush=True)

    monitor(0);history=[];start=time.time()
    for step in range(1,513):
        model.train();volumes,targets=batches[(step-1)%2];opt.zero_grad(set_to_none=True)
        tokens,pred,_=model(volumes);losses=criterion(tokens,pred,targets)
        if step in [1,64,256,512]:
            assignments=criterion.matcher(pred,targets)
            loc=criterion.loss_nodes(pred['pred_nodes'],targets['nodes'],assignments)
            named=[(n,p) for n,p in model.named_parameters() if n.startswith(('coord_embed.','encoder.','decoder.'))]
            gradients=torch.autograd.grad(loc,[p for n,p in named],retain_graph=True,allow_unused=True)
            norms={prefix:float(sum(g.detach().square().sum() for (n,p),g in zip(named,gradients) if g is not None and n.startswith(prefix)).sqrt()) for prefix in ['coord_embed.','encoder.','decoder.']}
            assert norms['coord_embed.']>0 and norms['encoder.']>0
            history.append(dict(step=step,localization_loss=float(loc.detach()),localization_gradient_norms=norms,losses={k:float(v.detach()) for k,v in losses.items() if torch.is_tensor(v) and v.numel()==1}))
        assert torch.isfinite(losses['total']);losses['total'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()
        if step in [16,64,128,256,512]:monitor(step)
    atomic_checkpoint(out/'model-000512.pt',model,cfg,opt,512)
    write(out/'complete.json',dict(completed=True,updates=512,training_patches=ids,subjects=sorted(subjects),seconds=time.time()-start,gradients=history,job_id=os.getenv('SLURM_JOB_ID'),bn='train updates; paired eval/train monitors restore buffers',geometry='unchanged K8 graph_only gradient route',annotation_caution='Scores are relative to incomplete manual traces; unmatched predictions are not confirmed anatomical negatives.'))


if __name__=='__main__':main()
