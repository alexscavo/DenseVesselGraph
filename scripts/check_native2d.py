#!/usr/bin/env python3
"""Bounded native2D CPU/GPU checks; optional tiny memorization, never full training."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from native2d import build_model
from native2d.config import load_config
from native2d.losses import GraphCriterion2D
from native2d.data import GraphDataset2D,collate
from native2d.evaluation import decode,evaluate,relation_ranking

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/native2d_20261027'
if (OUT/'worktree.json').exists():OUT=Path(json.loads((OUT/'worktree.json').read_text())['output'])


def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run_checks(device):
    torch.manual_seed(364505);torch.set_num_threads(2)
    cfg=load_config(smoke=True)
    cfg['model']['node_refinement']['enabled']=True
    model=build_model(cfg).to(device);model.eval()
    # Contain optional-head initialization so shared weights remain identical.
    torch.manual_seed(364505)
    base_cfg=copy.deepcopy(cfg);base_cfg['model']['node_refinement']['enabled']=False
    plain=build_model(base_cfg).to(device).eval()
    for name,value in plain.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    del plain
    dataset=GraphDataset2D(OUT/'manifest.json','train',max_queries=16)
    x,targets,gt,rows=collate([dataset[i] for i in range(min(2,len(dataset)))],device)
    criterion=GraphCriterion2D(cfg,model)
    tokens,pred=model(x)
    assert tokens.shape[-1]==64 and pred['pred_nodes'].shape[-1]==4
    torch.testing.assert_close(pred['pred_nodes'],pred['pred_nodes_stage1'],rtol=0,atol=0)
    unchanged=pred['pred_logits'].clone()
    losses=criterion(tokens,pred,targets)
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses['spline'].backward(retain_graph=True)
    assert all(p.grad is None or not p.grad.any() for n,p in model.named_parameters() if not n.startswith('spline_head.'))
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.any() for p in model.spline_head.parameters())
    model.zero_grad(set_to_none=True);losses['total'].backward()
    assert model.node_refinement.network[-1].weight.grad.abs().sum()>0
    assert model.class_embed.weight.grad.abs().sum()>0
    assert model.relation_embed.layers[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    pair=torch.tensor([[0,1],[2,3]],device=device)
    torch.testing.assert_close(model.relation_logits(tokens[0],pair),model.relation_logits(tokens[0],pair.flip(1)),rtol=0,atol=0)
    c,curve=model.curves(tokens[0],pred['pred_nodes'][0],pair,targets['spline'][0]['metric'])
    cr,curver=model.curves(tokens[0],pred['pred_nodes'][0],pair.flip(1),targets['spline'][0]['metric'])
    torch.testing.assert_close(c,cr.flip(1),rtol=0,atol=0)
    torch.testing.assert_close(c[:,[0,-1]],pred['pred_nodes'][0,pair,:2].detach(),rtol=0,atol=0)
    # Optional heads do not consume the old model's RNG/weights at import.
    baseline={k:v.clone() for k,v in model.state_dict().items() if not k.startswith(('node_refinement.','spline_head.'))}
    restored=build_model(cfg).to(device);restored.load_legacy_state(baseline);restored.eval()
    with torch.no_grad():
        t2,p2=restored(x)
    torch.testing.assert_close(tokens,t2,rtol=0,atol=0)
    torch.testing.assert_close(pred['pred_logits'],p2['pred_logits'],rtol=0,atol=0)
    torch.testing.assert_close(pred['pred_nodes'],p2['pred_nodes'],rtol=0,atol=0)
    state_hash=digest(model)
    model.eval()
    with torch.no_grad():
        for k in range(4):
            from native2d.data import transform
            im,tgt=transform(dataset[0][0],dataset[0][1],quarter_turns=k,flip_x=True)
            pp=im.unsqueeze(0).to(device);tt,oo=model(pp)
            assert torch.isfinite(oo['pred_nodes']).all()
    assert digest(model)==state_hash
    result=dict(passed=True,device=str(device),config=cfg,
        samples=[r['sample_id'] for r in rows],token_shape=list(tokens.shape),
        losses={k:float(v.detach()) for k,v in losses.items()},
        checks=['native_BCHW_forward_backward','modern_focal_and_matched_CE','zero_Axyz_identity',
            'optional_refinement_shared_initialization_parity',
            'classification_unchanged','geometry_graph_only_gradient_isolation','finite_gradients',
            'symmetric_relation_logits','spline_order_invariance','exact_spline_endpoints',
            'strict_legacy_core_state_replay','eval_buffer_invariance','real_augmented_forward'])
    return result


def tiny_fit(device,steps):
    torch.manual_seed(364505)
    cfg=load_config(smoke=True);cfg['model']['node_refinement']['enabled']=True
    model=build_model(cfg).to(device);criterion=GraphCriterion2D(cfg,model)
    ds=GraphDataset2D(OUT/'manifest.json','train',max_queries=16)
    # Two real image crops from one training image; no augmentation or validation
    # selection. This is explicitly a memorization check, not a benchmark.
    samples=[ds[i] for i in range(min(2,len(ds)))]
    x,target,gt,rows=collate(samples,device)
    optim=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    history=[];start=time.monotonic()
    schedule={0,1,10,50,100,250,500,steps}
    model.eval() # FrozenBatchNorm and dropout=0; still fully trainable parameters.
    for step in range(steps+1):
        tokens,pred=model(x);loss=criterion(tokens,pred,target)
        if step in schedule:
            metrics=[]
            with torch.no_grad():
                for b,g in enumerate(gt):
                    decoded=decode(model,tokens,pred,b,target['spline'][b]['metric'])
                    item=evaluate(decoded,g)
                    item['ranking']=relation_ranking(model,tokens,pred,b,g)
                    raw=dict(nodes=pred['pred_nodes'][b,:,:2].cpu().numpy(),edges=np.empty((0,2),int))
                    from native2d.evaluation import positional
                    item['raw_node_coverage']=positional(raw['nodes']@g['metric'].T,raw['edges'],g['nodes']@g['metric'].T,g['edges'],2.,inserted=g.get('inserted'))['node']['recall']
                    metrics.append(item)
            row=dict(step=step,elapsed_seconds=time.monotonic()-start,losses={k:float(v.detach()) for k,v in loss.items()},samples=metrics)
            history.append(row);print(json.dumps(dict(step=step,total=row['losses']['total'],node_f1=[m['2.0']['node']['f1'] for m in metrics],edge_f1=[m['2.0']['edge']['f1'] for m in metrics])),flush=True)
            (OUT/'tiny-learning-curve.json').write_text(json.dumps(history,indent=2,allow_nan=True)+'\n')
        if step==steps:break
        optim.zero_grad(set_to_none=True);loss['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optim.step()
    final=history[-1];passed=all(m['2.0']['node']['f1']>=.9 and m['2.0']['edge']['f1']>=.8 for m in final['samples'])
    # Save the negative result too. Failure of the predeclared memorization gate
    # must block full training, rather than silently launch a larger run.
    checkpoint=OUT/'runtime'/'tiny-last.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),config=cfg,steps=steps,seed=364505,
        sample_ids=[r['sample_id'] for r in rows],purpose='tiny_memorization_only'),checkpoint)
    summary=dict(completed=True,memorization_gate_passed=passed,steps=steps,
        runtime_seconds=time.monotonic()-start,updates_per_second=steps/(time.monotonic()-start),
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        samples=[r['sample_id'] for r in rows],checkpoint=str(checkpoint),history=history)
    (OUT/'tiny-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True)+'\n')
    return summary


def full_architecture_check(device):
    """One original-width ResNet101/512D forward/backward, no optimizer update."""
    cfg=load_config(ROOT/'configs/native2d.yaml');cfg['model']['node_refinement']['enabled']=True
    model=build_model(cfg).to(device).eval();criterion=GraphCriterion2D(cfg,model)
    ds=GraphDataset2D(OUT/'manifest.json','train',max_queries=80)
    x,target,_,_=collate([ds[0]],device)
    tokens,pred=model(x);loss=criterion(tokens,pred,target);loss['total'].backward()
    assert torch.isfinite(loss['total'])
    result=dict(passed=True,optimizer_updates=0,backbone='resnet101',hidden_dim=512,
        feature_levels=4,encoder_layers=3,decoder_layers=3,object_queries=80,relation_tokens=5,
        input_shape=list(x.shape),parameters=sum(p.numel() for p in model.parameters()),loss=float(loss['total'].detach()))
    (OUT/'full-architecture-check.json').write_text(json.dumps(result,indent=2)+'\n')
    del model,criterion,tokens,pred,loss
    if device.type=='cuda':torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--steps',type=int,default=0)
    ap.add_argument('--full-architecture-smoke',action='store_true')
    args=ap.parse_args();device=torch.device(args.device)
    result=run_checks(device)
    (OUT/f'{args.device}-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(correctness_passed=True,device=str(device))))
    if args.full_architecture_smoke:full_architecture_check(device)
    if args.steps:tiny_fit(device,args.steps)


if __name__=='__main__':main()
