#!/usr/bin/env python3
"""Small real-data CPU check; full-batch CUDA verification remains a Slurm gate."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from scripts.train_plant_generalization import configuration, TransferCriterion
from scripts.limited_data_support import budget_loader, Exposures, state_digest
from scripts.train_spline_weight_pilot import seed
from data.loaders import build_data_loaders, dataset_sample_manifest
from data.loaders.common import image_graph_collate
from models import build_model
from models.matcher import build_matcher
from training.engine import train_step
from training.optim import build_optimizer, build_scheduler


def main():
    torch.set_num_threads(2)
    plan=yaml.safe_load(Path('configs/leaf_generalization_da_b32.yaml').read_text())
    plan.update(train_samples=200,fixed_update_budget=True,pretrain_draws_per_epoch=50016,
                finetune_draws_per_epoch=10016,workers=0)
    # Check shared graph/geometry computations here; configured domain losses
    # and the actual batch-32 CUDA paths are checked by the dependent GPU smoke.
    plan['domain_adaptation']['enabled']=False
    report={};initial=None
    for arm in ('pretrain_vessels','pretrain_mixed','vessels_spline'):
        cfg=configuration(plan,arm);cfg['runtime']['pin_memory']=False
        cfg['model']['decoder']['use_cuda_extension']=False
        loader,_=build_data_loaders(cfg)
        draws=50016 if arm.startswith('pretrain_') else 10016
        loader=budget_loader(loader,draws,plan['seed'])
        membership=dataset_sample_manifest(loader.dataset)
        assert [m['count'] for m in membership if m['domain_label']==1]==[200]
        assert len(loader)*cfg['training']['epochs']==(78150 if arm.startswith('pretrain_') else 31300)
        indices=list(loader.sampler);exposure=Exposures(loader)
        batch=[None,None,[None]*32,None,None,torch.tensor([exposure.lookup[i][0]=='synthetic_mri' for i in indices[:32]],dtype=torch.long)]
        exposure.consume(0,batch)
        assert exposure.summary()['samples_processed']==32
        assert Exposures(loader,exposure.state()).state()==exposure.state()
        rng=loader.sampler.sampler.generator.get_state();expected=list(loader.sampler)
        loader.sampler.sampler.generator.set_state(rng);assert list(loader.sampler)==expected
        # One MRI + one leaf in the mixed path; two MRI otherwise.
        if arm=='pretrain_mixed':
            sample_indices=[0,len(loader.dataset.datasets[0])]
        else:sample_indices=[0,1]
        real_batch=image_graph_collate([loader.dataset[i] for i in sample_indices])
        assert real_batch[0].shape==(2,1,64,64,64)
        seed(plan['seed']);model=build_model(cfg)
        digest=state_digest(model)
        if initial is None:initial=digest
        assert digest==initial
        weighted=cfg['loss']['weights']['spline']>0
        for parameter in model.spline_head.parameters():parameter.requires_grad_(weighted)
        criterion=TransferCriterion(cfg,build_matcher(cfg),model.relation_embed,spline_head=model.spline_head,route='graph_only')
        optimizer=build_optimizer(cfg,model);scheduler=build_scheduler(cfg,optimizer,len(loader))
        losses=train_step(model,criterion,optimizer,scheduler,real_batch,cfg,'cpu',epoch=1,iteration=1,
                          total_iterations=len(loader)*cfg['training']['epochs'])
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spline_head.parameters())==weighted
        report[arm]=dict(updates=len(loader)*cfg['training']['epochs'],dataset_counts=[m['count'] for m in membership],
            loss=float(losses['total'].detach()),initial_state_sha256=digest,domains=real_batch[-1].tolist(),
            sampler_counter_check=exposure.summary(),cpu_forward_backward_batch_size=2,
            cpu_forward_backward_passed=True)
        del model,criterion,optimizer,scheduler,losses,real_batch,loader
    path=Path('experiments/limited_botanical_transfer/correctness-check.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
