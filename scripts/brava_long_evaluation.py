"""Fixed BRAVA monitors and milestone caches; original physical graph metrics."""
import io
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scripts.brava_long_experiment import OUT, MILESTONES, SEED, read, write, seed, base_dataset


def patch_metrics(cache, gt, tolerance):
    from scripts.diagnose_branch_relations import match_nodes
    from scripts.ixi_query_diagnostics import edge_accounting, betti
    metric=gt['metric']; nodes=cache['nodes']; pairs=cache['candidate_edges']
    mapping=match_nodes(nodes@metric.T,gt['nodes']@metric.T,tolerance)
    row,_,labels,_,_=edge_accounting(np.arange(len(nodes)),pairs,cache['candidate_scores'],
        cache['candidate_scores']>.5,mapping,gt['edges'],len(gt['nodes']))
    distance=np.linalg.norm((nodes[:,None]-gt['nodes'][None])@metric.T,axis=-1)
    recovered=set(mapping.values())
    duplicates=[i for i in range(len(nodes)) if i not in mapping and any(distance[i,j]<=tolerance for j in recovered)]
    errors=np.array([distance[i,j] for i,j in mapping.items()])
    conditional=np.array([int(a) in mapping and int(b) in mapping for a,b in pairs],dtype=bool)
    g0,g1=betti(len(gt['nodes']),gt['edges'])
    row.update(raw_matched=len(match_nodes(cache['raw_nodes']@metric.T,gt['nodes']@metric.T,tolerance)),
        localization_sum_mm=float(errors.sum()),localization_count=len(errors),
        localization_p95_mm=float(np.quantile(errors,.95)) if len(errors) else None,
        hungarian_sum_mm=float(cache['hungarian_distance'].sum()),hungarian_n=len(cache['hungarian_distance']),
        duplicates=len(duplicates),high_confidence_duplicates=int((cache['node_scores'][duplicates]>=.9).sum()),
        node_count_bias=len(nodes)-len(gt['nodes']),edge_count_bias=row['edges']-len(gt['edges']),
        beta0_bias=row['beta0']-g0,beta1_bias=row['beta1']-g1,
        conditional_candidates=int(conditional.sum()),conditional_positive_pairs=int(labels[conditional].sum()),
        all_retained_candidates=len(pairs),all_retained_positive_pairs=int(labels.sum()))
    return row,labels,conditional


def aggregate(rows):
    from scripts.ixi_query_diagnostics import summarize
    result=summarize(rows)
    def ratio(a,b):return a/b if b else None
    result.update(raw_node_recall=ratio(sum(r['raw_matched'] for r in rows),result['gt_nodes']),
        localization_mean_mm=ratio(sum(r['localization_sum_mm'] for r in rows),sum(r['localization_count'] for r in rows)),
        hungarian_mean_mm=ratio(sum(r['hungarian_sum_mm'] for r in rows),sum(r['hungarian_n'] for r in rows)),
        duplicates_per_patch=float(np.mean([r['duplicates'] for r in rows])),
        high_confidence_duplicates=sum(r['high_confidence_duplicates'] for r in rows),
        node_count_bias_per_patch=float(np.mean([r['node_count_bias'] for r in rows])),
        edge_count_bias_per_patch=float(np.mean([r['edge_count_bias'] for r in rows])),
        component_bias_per_patch=float(np.mean([r['beta0_bias'] for r in rows])),
        cycle_bias_per_patch=float(np.mean([r['beta1_bias'] for r in rows])))
    return result


def ap_summary(parts):
    from sklearn.metrics import average_precision_score
    result={}
    for name in ['conditional','all_retained']:
        scores=np.concatenate([p[name][0] for p in parts])
        labels=np.concatenate([p[name][1] for p in parts])
        result[name+'_ap']=float(average_precision_score(labels,scores)) if labels.any() else None
        result[name+'_candidates']=len(scores)
        result[name+'_positives']=int(labels.sum())
        result[name+'_prevalence']=float(labels.mean()) if len(labels) else None
    return result


def snapshot(plan,cfg,model,criterion,arm,epoch,smoke=False):
    import torch
    from torch.utils.data import DataLoader
    from data.loaders.common import image_graph_collate
    from training.engine import _move_graph_batch
    from training.evaluation.inference import infer_graphs
    folder=OUT/'large'/'predictions'/arm/(f'epoch-{epoch:03d}' if not smoke else 'smoke')
    if (folder/'summary.json').exists():
        result=read(folder/'summary.json')
        assert result['completed'] and result['epoch']==epoch
        return result
    full=epoch in MILESTONES and not smoke
    populations={'train':plan['train_monitor'], 'val':plan['ids']['val'] if full else plan['val_monitor']}
    if smoke:populations={s:ids[:2] for s,ids in populations.items()}
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]; records=[]; losses=[]; ranking={}
    model.eval();started=time.monotonic()
    # Evaluation cannot update BN or alter the subsequent training input stream.
    buffers={n:v.detach().clone() for n,v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    for split,names in populations.items():
        ds=base_dataset(plan,split,names,False)
        loader=DataLoader(ds,batch_size=8,num_workers=2,collate_fn=image_graph_collate,
                          shuffle=False,pin_memory=True)
        archive=folder/(split+'.zip');temp=archive.with_name(archive.name+'.tmp')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_STORED) as zipped,torch.no_grad():
            for offset,batch in enumerate(loader):
                volumes,targets=_move_graph_batch(batch,'cuda','image',True)
                tokens,predictions,_=model(volumes)
                seed(SEED+1000000+offset)
                loss=criterion(tokens,predictions,targets)
                assignment=criterion.matcher(predictions,targets)
                scalar={k:float(v) for k,v in loss.items() if torch.is_tensor(v) and v.numel()==1}
                assert all(np.isfinite(v) for v in scalar.values())
                losses.append(dict(split=split,count=len(volumes),**scalar))
                kwargs=dict(object_queries=256,relation_tokens=2,node_threshold=.5,
                            node_type_degree_caps=False,node_type_degree_penalty=0.)
                candidate=infer_graphs(tokens,predictions,model.relation_embed,edge_threshold=-1.,**kwargs)
                if full or smoke:
                    normal=infer_graphs(tokens,predictions,model.relation_embed,edge_threshold=.5,
                        spline_head=model.spline_head,spline_metrics=[g['metric'] for g in targets['spline']],**kwargs)
                else:normal=None
                for j,c in enumerate(candidate):
                    index=offset*8+j; sample=names[index]
                    cache={k:v.numpy() for k,v in c.items() if torch.is_tensor(v)}
                    pairs=cache.pop('edges'); scores=cache.pop('edge_scores')
                    cache.update(candidate_edges=pairs,candidate_scores=scores,
                        edges=pairs[scores>.5],edge_scores=scores[scores>.5],
                        raw_nodes=predictions['pred_nodes'][j,:,:3].detach().cpu().numpy(),
                        raw_scores=predictions['pred_logits'][j].softmax(-1)[:,1].detach().cpu().numpy())
                    metric=targets['spline'][j]['metric'].cpu().numpy()
                    gt=dict(nodes=targets['nodes'][j].cpu().numpy(),edges=targets['edges'][j].cpu().numpy(),metric=metric)
                    source,target=assignment[j]
                    cache['hungarian_source']=source.cpu().numpy();cache['hungarian_target']=target.cpu().numpy()
                    cache['hungarian_distance']=np.linalg.norm((cache['raw_nodes'][source.cpu().numpy()]-gt['nodes'][target.cpu().numpy()])@metric.T,axis=-1)
                    if normal is not None:
                        n=normal[j]
                        np.testing.assert_array_equal(n['query_ids'].numpy(),cache['query_ids'])
                        assert {tuple(sorted(e)) for e in n['edges'].tolist()}=={tuple(e) for e in cache['edges'].tolist()}
                        cache.update(edges=n['edges'].numpy(),edge_scores=n['edge_scores'].numpy(),spline_controls=n['spline_controls'].numpy())
                        assert cache['spline_controls'].shape==(len(cache['edges']),8,3)
                        np.testing.assert_allclose(cache['spline_controls'][:,[0,-1]],cache['nodes'][cache['edges']],atol=2e-6)
                    record=dict(sample=sample,subject=plan['subjects'][sample],split=split,
                                gt=str(Path(plan['dataset'])/split/'spline'/f'{sample}.npz'),
                                archive=str(archive),entry=sample+'.npz',
                                val_monitor=split=='val' and sample in plan['val_monitor'])
                    records.append(record)
                    for tolerance in [1.,2.]:
                        row,labels,conditional=patch_metrics(cache,gt,tolerance)
                        rows.append(dict(record,tolerance_mm=tolerance,**row))
                        part=dict(all_retained=(cache['candidate_scores'],labels),
                                  conditional=(cache['candidate_scores'][conditional],labels[conditional]))
                        groups=[split]+(['val_monitor'] if record['val_monitor'] else [])
                        for group in groups:ranking.setdefault((group,tolerance),[]).append(part)
                    buffer=io.BytesIO();np.savez_compressed(buffer,**cache)
                    zipped.writestr(sample+'.npz',buffer.getvalue())
                if offset%32==0:print('evaluate',arm,epoch,split,offset*8,'/',len(names),flush=True)
        temp.replace(archive)
    assert all(torch.equal(buffers[n],dict(model.named_buffers())[n]) for n in buffers),'Evaluation altered BN'
    metrics={}
    for group in ['train','val','val_monitor']:
        for tolerance in [1.,2.]:
            rr=[r for r in rows if r['tolerance_mm']==tolerance and
                (r['split']==group if group!='val_monitor' else r['val_monitor'])]
            if rr:
                metrics.setdefault(group,{})[str(tolerance)]=dict(aggregate(rr),**ap_summary(ranking[group,tolerance]))
    loss_summary={split:{k:sum(r[k]*r['count'] for r in losses if r['split']==split)/sum(r['count'] for r in losses if r['split']==split)
                         for k in losses[0] if k not in ['split','count']}
                  for split in populations}
    result=dict(completed=True,arm=arm,epoch=epoch,full_validation=full,smoke=smoke,
        metrics=metrics,losses=loss_summary,records=records,rows=rows,seconds=time.monotonic()-started,
        evaluation_batch_size=8,bn_buffers_unchanged=True,graph_thresholds=[.5,.5],test_data_accessed=False)
    write(folder/'summary.json',result)
    if smoke:
        from metrics.common_vessel_metrics import evaluate_common_vessels
        common=[]
        for record in records:
            with zipfile.ZipFile(record['archive']) as archive,np.load(io.BytesIO(archive.read(record['entry']))) as data:
                common.append(evaluate_common_vessels(dict(nodes=data['nodes'],edges=data['edges'],spline_controls=data['spline_controls']),record['gt']))
        write(OUT/'evaluation-smoke.json',dict(passed=True,samples=len(records),common=common,
              finite_graph_losses=True,decoded_node_edge_parity=True,bn_unchanged=True))
    return result
