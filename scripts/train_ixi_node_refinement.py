#!/usr/bin/env python3
"""Paired frozen-detector node-localization pilot with local-image ablation."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch,yaml
from data.loaders import build_data_loaders,build_evaluation_loader
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import _move_graph_batch
from training.tracking import build_tracker
from scripts.train_ixi_transfer import TransferCriterion
from scripts.train_spline_weight_pilot import seed,monitoring_rng,geometry,write_json
from scripts.frozen_spline_diagnostic import training_evaluation_loader
from scripts.controlled_spline_unfreeze import frozen_state_digest
from scripts.train_relation_distance_pilot import prediction_diagnostics
from scripts.diagnose_ixi_features_nodes import choose,node_rows
from scripts.evaluate_vessel_core import records


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/ixi_node_refinement.yaml'));p.add_argument('--arm',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    plan=yaml.safe_load(a.config.read_text());assert a.arm in plan['arms'];seed(plan['seed'])
    cp=load_checkpoint_file(plan['checkpoint']);cfg=copy.deepcopy(cp['config'])
    cfg['data'].update(train_augmentation=False,batch_size=plan['batch_size'],validation_batch_size=4)
    cfg['runtime']['workers']=0 if a.smoke else 4
    cfg['experiment'].update(name='ixi_node_refinement_'+a.arm,seed=plan['seed'])
    cfg['tracking'].update(enabled=not a.smoke,mode='offline',group='ixi-node-refinement-20261018')
    # Geometry is frozen; remove its constant computation. Existing node losses unchanged.
    cfg['loss']['weights']['spline']=0.
    mode=plan['arms'][a.arm]
    if mode is not None:cfg['model']['node_refinement']=dict(enabled=True,image_mode=mode,radius_vox=plan['radius_vox'])
    out=Path('experiments/ixi_node_refinement/smoke')/a.arm if a.smoke else Path(plan['output_root'])/a.arm
    out.mkdir(parents=True,exist_ok=False)
    (out/'resolved-config.yaml').write_text(yaml.safe_dump(cfg));write_json(out/'plan.json',plan)
    root=Path(cfg['data']['datasets']['synthetic_mri']['root'])
    train,_=build_data_loaders(cfg)
    ids={s:choose(root,s,4 if a.smoke else plan['train_eval_samples' if s=='train' else 'validation_samples']) for s in ('train','val')}
    train_eval=training_evaluation_loader(train,cfg,ids['train']);val=build_evaluation_loader(cfg,dataset_name='synthetic_mri',sample_ids=ids['val'])
    train_ids=[r.sample_id for r in train.dataset.datasets[0].records]
    assert set(ids['train'])<=set(train_ids)
    assert not {x.rsplit('_',1)[0] for x in train_ids}&{x.rsplit('_',1)[0] for x in ids['val']}
    write_json(out/'sample-ids.json',dict(train=train_ids,monitoring=ids))
    seed(plan['seed']);model=build_model(cfg)
    loaded=model.load_state_dict(cp['net'],strict=False)
    assert not loaded.unexpected_keys and all(k.startswith('node_refinement.') for k in loaded.missing_keys)
    if mode is None:assert not loaded.missing_keys
    del cp
    model.cuda();model.eval()
    for n,param in model.named_parameters():param.requires_grad_(n.startswith(('class_embed.','coord_embed.','node_refinement.')))
    frozen=frozen_state_digest(model)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    settings=cfg['training']['optimizer']
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=plan['learning_rate'],weight_decay=settings['weight_decay'],eps=settings['epsilon'],betas=tuple(settings['betas']))
    updates=3 if a.smoke else plan['updates'];baseline=json.loads((root/'monitoring.json').read_text())['branch_bins']
    provenance=dict(plan=plan,arm=a.arm,updates=updates,initial_checkpoint=plan['checkpoint'],
        frozen_state_sha256=frozen,trainable=[n for n,p in model.named_parameters() if p.requires_grad],
        bn='fixed saved statistics; all modules eval',augmentation=False,
        objectives='unchanged original losses; spline weight zero since geometry frozen; relation token losses cannot update frozen detector',
        endpoint_sampling='detached input coordinates; early convolution centre i maps to voxel 2*i; 27 samples within +/-4 voxels',
        checkpoint_storage='only trained parameters saved; load parent checkpoint then overlay adapter parameters')
    tracker=build_tracker(cfg,out,resume=False,launch_metadata=provenance)
    history=[];started=time.monotonic()
    probe=next(iter(val));pv,_=_move_graph_batch(probe,'cuda',cfg['training']['input'],True)
    with torch.no_grad():
        t0,p0,_=model(pv)
        if model.node_refinement is not None:
            head=model.node_refinement;model.node_refinement=None
            tb,pb,_=model(pv);model.node_refinement=head
            assert torch.equal(p0['pred_nodes'],pb['pred_nodes']) and torch.equal(p0['pred_logits'],pb['pred_logits']), 'Non-identity initialization'
    def monitor(step):
        with monitoring_rng():
            model.eval();assert frozen_state_digest(model)==frozen
            with torch.no_grad():tt,_,_=model(pv)
            assert torch.equal(tt,t0), 'Frozen token change'
            row=dict(step=step,validation=geometry(model,val,cfg,out,step,tracker,0,True),elapsed_seconds=time.monotonic()-started)
            row['relations']=prediction_diagnostics(out/f'validation-{step:06d}'/'predictions.json',root,'val',baseline,core=False)
            if step in (0,updates):row['train']=geometry(model,train_eval,cfg,out,step,None,0,True,split='train')
            # Save GT subtype counts on the exact decoded nodes used for graph metrics.
            subtype={}
            for split,folder in [('val',f'validation-{step:06d}'),('train',f'train-evaluation-{step:06d}')]:
                if not (out/folder).exists():continue
                rr=[]
                for rec in records(out/folder/'predictions.json'):
                    name=rec['source_sample_id'];pred=np.asarray(rec['nodes_dhw'],float).reshape(-1,3)
                    with np.load(root/split/'spline'/f'{name}.npz') as c:cache={k:c[k] for k in ('nodes','metric','edges','boundary')}
                    # all_queries=decoded here; only outcome/type fields are used below.
                    rr.extend(node_rows(pred,pred,cache,split,name))
                subtype[split]={}
                for tol in (1.,2.):
                    subtype[split][str(tol)]={}
                    for kind in ('boundary','junction','termination','other'):
                        group=[r for r in rr if r['tolerance_mm']==tol and r['population']=='GT' and r['type']==kind]
                        subtype[split][str(tol)][kind]=dict(total=len(group),**{s:sum(r['outcome']==s for r in group) for s in ('recovered','displaced','missing')})
            row['node_subtypes']=subtype
        history.append(row)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        monitor(0);seed(plan['seed']);iterator=iter(train);order=hashlib.sha256()
        for step in range(1,updates+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(train);batch=next(iterator)
            for n in batch[2]:order.update(n.numpy().tobytes())
            volumes,targets=_move_graph_batch(batch,'cuda',cfg['training']['input'],True)
            model.eval();optimizer.zero_grad();tokens,predictions,_=model(volumes);loss=criterion(tokens,predictions,targets)
            assert torch.isfinite(loss['total']);loss['total'].backward()
            for n,param in model.named_parameters():
                if not param.requires_grad:assert param.grad is None,n
                elif param.grad is not None:assert torch.isfinite(param.grad).all(),n
            if step==1:
                for prefix in ['coord_embed.','class_embed.']+(['node_refinement.'] if mode is not None else []):
                    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(prefix)),prefix
            optimizer.step()
            if tracker and step%50==0:tracker.log_training({k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1},iteration=step,epoch=0,learning_rate=plan['learning_rate'])
            if step%plan['monitor_interval']==0 or step==updates:monitor(step)
        provenance.update(batch_order_sha256=order.hexdigest(),final_frozen_state_sha256=frozen_state_digest(model))
        write_json(out/'summary.json',dict(provenance=provenance,initial=history[0],final=history[-1]));write_json(out/'provenance.json',provenance)
        if not a.smoke:
            torch.save(dict(parameters={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},config=cfg,provenance=provenance),out/'trained-heads.pt')
        if tracker:tracker.finish()
    except BaseException:
        if tracker:tracker.finish(exit_code=1)
        raise

if __name__=='__main__':main()
