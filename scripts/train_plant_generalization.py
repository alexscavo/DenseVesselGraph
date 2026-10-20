#!/usr/bin/env python3
"""Controlled plant transfer with shared MRI membership and spline-safe augmentation."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from configs import validate_config
from data.loaders import build_data_loaders, build_evaluation_loader, dataset_sample_manifest
from data.loaders.common import image_graph_collate, seed_data_worker
from models import build_model
from models.checkpoint import load_checkpoint_file
from models.matcher import build_matcher
from training.engine import train_step
from training.optim import build_optimizer, build_scheduler
from training.tracking import build_tracker
from scripts.controlled_spline_unfreeze import RoutedCriterion
from scripts.extend_scratch_geometry import atomic_checkpoint, capture_rng, restore_rng
from scripts.train_spline_weight_pilot import seed, monitoring_rng, geometry, write_json
from scripts.frozen_spline_diagnostic import training_evaluation_loader, matched_geometry
from scripts.train_relation_distance_pilot import prediction_diagnostics


class TransferCriterion(RoutedCriterion):
    def loss_spline(self,tokens,predictions,targets,assignments):
        if float(self.config['loss']['weights']['spline'])==0:
            # Plants deliberately have graph supervision only during pretraining.
            return {'spline':predictions['pred_nodes'].sum()*0.}
        return super().loss_spline(tokens,predictions,targets,assignments)


def configuration(plan, arm):
    cfg=yaml.safe_load(Path(plan['base_config']).read_text())
    pretrain=arm.startswith('pretrain_'); mixed=arm=='pretrain_mixed'
    cfg['experiment'].update(name=arm,seed=plan['seed'])
    cfg['runtime']['workers']=plan['workers']
    cfg['data'].update(train_augmentation=True,batch_size=plan['batch_size'],validation_batch_size=plan['batch_size'])
    mri=cfg['data']['datasets']['synthetic_mri'];mri.update(train_samples=plan.get('train_samples'),validation_samples=None)
    cfg['data']['datasets']={'synthetic_mri':mri}
    if mixed:
        cfg['data']['datasets']['plants']=dict(role='source',root=plan['plant_root'],train_samples=None,validation_samples=None,coordinate_order_on_disk=['y','x'])
    cfg['data']['mixed_sampling'].update(balance_source_target=True,epoch_size='twice_source')
    cfg['augmentation']['plants']['source_augmentation']=copy.deepcopy(plan.get('source_augmentation', {}))
    aug=cfg['augmentation']['synthetic_mri'];aug['rotate_90']['enabled']=True;aug['zoom']['enabled']=False
    aug['gaussian_noise'].update(enabled=True,probability=.35,std_range=[0.,.015]);aug['intensity_clamp']=None
    domain=plan.get('domain_adaptation',{})
    cfg['domain_adaptation'].update(enabled=mixed and domain.get('enabled',False))
    cfg['loss']['enabled']=[x for x in cfg['loss']['enabled'] if x!='domain']
    if cfg['domain_adaptation']['enabled']:
        cfg['domain_adaptation'].update(image_adversarial=True,graph_adversarial=True,consistency_regularization=True,
            class_weights=domain['class_weights'],learning_rate=domain['learning_rate'])
        cfg['loss']['enabled'].append('domain');cfg['loss']['weights']['domain']=domain['weight']
    cfg['training']['epochs']=plan['pretrain_epochs'] if pretrain else plan['finetune_epochs']
    cfg['training']['warmup_epochs']=1
    cfg['training']['optimizer']['base_lr']=plan['learning_rate']
    cfg['loss']['weights']['spline']=plan['spline_weight'] if arm.endswith('_spline') else 0.
    cfg['loss']['spline']['target']='residual'
    cfg['tracking'].update(enabled=True,mode='offline',group=plan['tracking_group'])
    validate_config(cfg)
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/plant_generalization.yaml'))
    p.add_argument('--arm',required=True,choices=['pretrain_mixed','pretrain_vessels','mixed_straight','mixed_spline','vessels_straight','vessels_spline'])
    p.add_argument('--until-epoch',type=int,required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();plan=yaml.safe_load(a.config.read_text())
    limited=plan.get('fixed_update_budget',False)
    if limited and not a.smoke:
        assert (Path(plan['output_root'])/'SMOKE_PASSED').is_file(), 'Full-batch smoke must pass before training'
    if a.smoke and not limited:
        plan['plant_root']=str(Path(plan['plant_root']).parent/'plants-smoke')
    cfg=configuration(plan,a.arm)
    pretrain=a.arm.startswith('pretrain_');weighted=cfg['loss']['weights']['spline']>0
    if not (Path(plan['plant_root'])/'READY').exists():raise ValueError('Plant preparation not complete')
    out=Path(plan['output_root'])/'runs'/(( 'smoke-' if a.smoke else '')+a.arm)
    out.mkdir(parents=True,exist_ok=a.smoke or (out/'latest.pt').exists())
    if (out/'COMPLETE').exists():print('Already complete',out);return
    seed(plan['seed']);train,_=build_data_loaders(cfg)
    mri_cfg=copy.deepcopy(cfg);mri_cfg['data']['datasets'].pop('plants',None)
    mri_cfg['data']['train_augmentation']=False
    mri_cfg['domain_adaptation']['enabled']=False
    mri_cfg['loss']['enabled']=[x for x in mri_cfg['loss']['enabled'] if x!='domain']
    mri_train,_=build_data_loaders(mri_cfg)
    manifest=json.loads(Path(plan['population_manifest']).read_text())
    assert [r.sample_id for r in mri_train.dataset.datasets[0].records]==manifest['train_ids']
    if limited:
        from scripts.limited_data_support import budget_loader, Exposures, initialize
        draws=plan['pretrain_draws_per_epoch'] if pretrain else plan['finetune_draws_per_epoch']
        train=budget_loader(train,3*plan['batch_size'] if a.smoke else draws,plan['seed'])
    elif a.arm=='pretrain_vessels':
        generator=torch.Generator().manual_seed(plan['seed'])
        sampler=RandomSampler(train.dataset,replacement=True,num_samples=3*plan['batch_size'] if a.smoke else plan['pretrain_draws_per_epoch'],generator=generator)
        train=DataLoader(train.dataset,batch_size=plan['batch_size'],sampler=sampler,num_workers=plan['workers'],
            collate_fn=image_graph_collate,worker_init_fn=seed_data_worker,generator=generator,pin_memory=True)
    if a.smoke and a.arm=='pretrain_mixed':train.sampler.num_samples=3*plan['batch_size']
    if pretrain and not a.smoke:assert train.sampler.num_samples==plan['pretrain_draws_per_epoch']
    ids=json.loads((Path(plan['evaluation_membership_run'])/'evaluation-sample-ids.json').read_text())
    val=build_evaluation_loader(mri_cfg,dataset_name='synthetic_mri',sample_ids=ids['val'][:4] if a.smoke else ids['val'])
    train_eval=training_evaluation_loader(mri_train,mri_cfg,ids['train'][:4] if a.smoke else ids['train'])
    write_json(out/'dataset-manifest.json',dict(train=dataset_sample_manifest(train.dataset),validation=dataset_sample_manifest(val.dataset)))
    write_json(out/'evaluation-sample-ids.json',ids)
    (out/'resolved-config.yaml').write_text(yaml.safe_dump(cfg))
    if limited:seed(plan['seed'])  # Dataset construction cannot alter scratch initialization.
    model=build_model(cfg).cuda()
    initializer=None;initial_digest=None
    if limited:initializer,initial_digest=initialize(model,plan,pretrain,a.smoke)
    # Identical architecture/initialization in controls; unused curve parameters stay frozen.
    for parameter in model.spline_head.parameters():parameter.requires_grad_(weighted)
    criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only').cuda()
    optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,len(train))
    start=0;step=0;best=(-1.,-1.);best_epoch=0;initial_weights=None
    if limited and not pretrain:
        initial_weights=Path(plan['output_root'])/'runs'/(('smoke-' if a.smoke else '')+'pretrain_'+a.arm.split('_')[0])/'best.pt'
    exposure=Exposures(train) if limited else None
    if (out/'latest.pt').exists() and not a.smoke:
        checkpoint=load_checkpoint_file(out/'latest.pt');model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        start=checkpoint['epoch'];step=checkpoint['steps'];best=tuple(checkpoint['best_score']);best_epoch=checkpoint['best_epoch']
        restore_rng(checkpoint['rng'],train)
        if limited:
            train.sampler.sampler.generator.set_state(checkpoint['sampling_rng'])
            exposure=Exposures(train,checkpoint['exposure'])
            assert exposure.updates==step
    elif not pretrain:
        initial_weights=Path(plan['output_root'])/'runs'/(('smoke-' if a.smoke else '')+'pretrain_'+a.arm.split('_')[0])/'best.pt'
        checkpoint=load_checkpoint_file(initial_weights)
        # Transfer detector only; initialize both geometry heads from the same fresh seed.
        state={k:v for k,v in checkpoint['net'].items() if not k.startswith('spline_head.') and 'domain_discriminator' not in k}
        report=model.load_state_dict(state,strict=False)
        assert not report.unexpected_keys and all(k.startswith('spline_head.') for k in report.missing_keys)
    write_json(out/'provenance.json',dict(arm=a.arm,pretrain=pretrain,initial_weights=str(initial_weights),
        seed=plan['seed'],draws_per_epoch=train.sampler.num_samples if pretrain or limited else len(train.dataset),source=plan,
        gradient_route='graph_only',plant_geometry_supervised=False,
        best_selection='validation common_branch_f1 then native_edge_f1, unchanged from previous paired study',
        bn='training statistics during updates; saved running statistics during evaluation',
        **(dict(common_initializer=initializer,initial_detector_spline_sha256=initial_digest,
                resumed_at_update=step,
                fixed_stage_updates=len(train)*cfg['training']['epochs'],
                full_batch_size=plan['batch_size']) if limited else {})))
    cfg['tracking']['enabled']=not a.smoke
    tracker=build_tracker(cfg,out,resume=start>0,launch_metadata={'arm':a.arm})
    started=time.monotonic();epoch=start;history=[]
    baseline=json.loads(Path('experiments/spline_learning/metadata/metrics/branch-relations-v1/summary.json').read_text())
    target_root=Path(mri_cfg['data']['datasets']['synthetic_mri']['root'])
    def payload():
        return dict(net=model.state_dict(),config=cfg,epoch=epoch,steps=step,optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),rng=capture_rng(train),best_score=best,best_epoch=best_epoch,
            **(dict(exposure=exposure.state(),sampling_rng=train.sampler.sampler.generator.get_state()) if limited else {}))
    def monitor():
        nonlocal best,best_epoch
        with monitoring_rng():
            metrics=geometry(model,val,mri_cfg,out,step,tracker,epoch,True)
            relation=prediction_diagnostics(out/f'validation-{step:06d}'/'predictions.json',target_root,'val',baseline,core=False)
            train_metrics=None
            if epoch%10==0 or a.smoke:
                train_metrics=geometry(model,train_eval,mri_cfg,out,step,None,epoch,True,split='train')
                prediction_diagnostics(out/f'train-evaluation-{step:06d}'/'predictions.json',target_root,'train',baseline,core=False)
        score=(metrics['common_branch_f1'],metrics['native_edge_f1'])
        if score>best:
            best=score;best_epoch=epoch;atomic_checkpoint(out/'best.pt',payload())
        row=dict(epoch=epoch,step=step,elapsed_seconds=time.monotonic()-started,validation=metrics,train=train_metrics,relation_diagnostic=relation,best_epoch=best_epoch,
                 **(dict(exposure=exposure.summary()) if limited else {}))
        history.append(row)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        if start==0 and a.smoke:monitor()
        limit=min(a.until_epoch,cfg['training']['epochs'])
        stop=False
        for epoch in range(start+1,limit+1):
            epoch_losses={};epoch_batches=0
            for batch_index,batch in enumerate(train):
                step+=1
                if a.smoke:torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();batch_started=time.monotonic()
                loss=train_step(model,criterion,optimizer,scheduler,batch,cfg,'cuda',epoch=epoch,iteration=step,total_iterations=len(train)*cfg['training']['epochs'])
                if limited:
                    assert loss is not None
                    exposure.consume(batch_index,batch)
                    for key,value in loss.items():
                        if torch.is_tensor(value) and value.numel()==1:epoch_losses[key]=epoch_losses.get(key,0.)+float(value.detach())
                    epoch_batches+=1
                if a.smoke:
                    torch.cuda.synchronize()
                    diagnostic=dict(step=step,batch_size=len(batch[2]),seconds=time.monotonic()-batch_started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,domain_loss=float(loss['domain'].detach()) if 'domain' in loss else None,domain_labels=batch[-1].tolist())
                    if cfg['domain_adaptation']['enabled']:
                        assert set(batch[-1].tolist())=={0,1}
                        for module in [model.backbone_domain_discriminator,model.instance_domain_discriminator]:
                            assert any(x.grad is not None and x.grad.abs().sum()>0 for x in module.parameters())
                    with (out/'smoke-batches.jsonl').open('a') as f:f.write(json.dumps(diagnostic)+'\n')
                if batch_index==0:
                    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
                    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.encoder.parameters())
                    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.spline_head.parameters())==weighted
                if tracker and step%100==0:
                    tracker.log_training({k:float(v.detach()) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1},iteration=step,epoch=epoch,learning_rate=optimizer.param_groups[0]['lr'])
                if a.smoke and batch_index==2:break
            if limited:
                assert exposure.updates==step
                with (out/'training-losses.jsonl').open('a') as f:
                    f.write(json.dumps(dict(epoch=epoch,step=step,losses={k:v/epoch_batches for k,v in epoch_losses.items()},
                                           learning_rate=optimizer.param_groups[0]['lr'],exposure=exposure.summary()))+'\n')
                write_json(out/'exposure.json',exposure.state())
            if a.smoke or epoch%plan['monitor_epochs']==0 or epoch==limit:
                monitor()
                stop=plan.get('early_stopping',True) and not pretrain and epoch>=plan['min_finetune_epochs'] and epoch-best_epoch>=plan['patience_epochs']
            atomic_checkpoint(out/'latest.pt',payload())
            if stop or a.smoke:break
        complete=stop or epoch==cfg['training']['epochs']
        if complete or a.smoke:
            with monitoring_rng():
                train_destination=out/f'train-evaluation-{step:06d}'
                if not train_destination.exists():
                    train_metrics=geometry(model,train_eval,mri_cfg,out,step,tracker,epoch,True,split='train')
                else:
                    train_metrics=json.loads((train_destination/'common-v2-summary.json').read_text())
                if not a.smoke:
                    full=build_evaluation_loader(mri_cfg,dataset_name='synthetic_mri')
                    assert len(full.dataset)==1952
                    full_output=out/'full-validation';full_output.mkdir(exist_ok=True)
                    full_metrics=geometry(model,full,mri_cfg,full_output,step,tracker,epoch,True)
                    write_json(out/'final-full-validation.json',full_metrics)
                    prediction_diagnostics(full_output/f'validation-{step:06d}'/'predictions.json',target_root,'val',baseline,core=True)
                    best_checkpoint=load_checkpoint_file(out/'best.pt');model.load_state_dict(best_checkpoint['net'])
                    best_output=out/'best-full-validation';best_output.mkdir(exist_ok=True)
                    best_metrics=geometry(model,full,mri_cfg,best_output,best_checkpoint['steps'],None,best_checkpoint['epoch'],True)
                    write_json(out/'best-full-validation.json',best_metrics)
                    prediction_diagnostics(best_output/f"validation-{best_checkpoint['steps']:06d}"/'predictions.json',target_root,'val',baseline,core=True)
                    if limited:
                        from scripts.limited_data_support import duplicate_diagnostics
                        duplicate_diagnostics(best_output/f"validation-{best_checkpoint['steps']:06d}"/'predictions.json',target_root,'val')
            write_json(out/'summary.json',dict(epoch=epoch,steps=step,early_stopped=stop,train=train_metrics,best_epoch=best_epoch,
                                             **(dict(exposure=exposure.summary()) if limited else {})))
            if not a.smoke:(out/'COMPLETE').write_text('complete\n')
        if tracker:tracker.finish()
    except BaseException:
        if tracker:tracker.finish(exit_code=1)
        raise

if __name__=='__main__':main()
