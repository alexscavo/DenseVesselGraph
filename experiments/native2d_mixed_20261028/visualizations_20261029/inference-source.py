"""Final-checkpoint validation figures and saved-result tables; no training/test IO."""
import argparse
import csv
import gc
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from native2d import build_model
from native2d.data import collate
from native2d.evaluation import decode,positional
from scripts.train_native2d_long import OUT,DATA,StudyDataset

DEST=OUT/'visualizations_20261029'
DATASETS=['avwide','reaver','octasynth']

def read(p):return json.loads(p.read_text())
def write(p,a):p.write_text(json.dumps(a,indent=2)+'\n')

def summaries():
    records=[]
    for dataset in DATASETS:
        for initialization in ['scratch','mixed']:
            for variant in ['B0','Axyz']:
                name=f'{dataset}_{initialization}_{variant}';run=OUT/'runs'/name
                status=read(run/'status.json');assert status['state']=='complete' and status['epoch']==100
                result=read(run/'final-heldout.json');s=result['summary'];rows=result['samples'];t=s['2.0']
                record=dict(dataset=dataset,initialization=initialization,variant=variant,epoch=100,units=s['units'],tolerance=2.,
                    validation_crops=len(rows),node_mAP=s['node_mAP_crop_macro'],edge_mAP=s['edge_mAP_crop_macro'],
                    node_mAR=s['node_mAR_crop_macro'],edge_mAR=s['edge_mAR_crop_macro'],node_f1=t['node']['f1'],edge_f1=t['edge']['f1'],
                    node_f1_at_1=s['1.0']['node']['f1'],edge_f1_at_1=s['1.0']['edge']['f1'],
                    branch_f1_positive_crop_macro=s['spline_edge_positive_crop_macro']['common_branch_f1'],
                    curve_f1_positive_crop_macro=s['spline_edge_positive_crop_macro']['common_curve_f1'],
                    chord_curve_f1_positive_crop_macro=s['chord_edge_positive_crop_macro']['common_curve_f1'],
                    raw_node_recall=t['raw_node_recall_micro'],filtered_node_recall=t['node']['recall'],
                    conditional_relation_recall=t['conditional_relation_recall'],duplicates=t['duplicates'],
                    predicted_nodes=sum(r['predicted_nodes'] for r in rows),predicted_edges=sum(r['predicted_edges'] for r in rows),
                    gt_nodes=sum(r['target_nodes'] for r in rows),gt_edges=sum(r['target_edges'] for r in rows),
                    predicted_components=sum(r['predicted_beta0'] for r in rows),gt_components=sum(r['target_beta0'] for r in rows),
                    predicted_cycles=sum(r['predicted_beta1'] for r in rows),gt_cycles=sum(r['target_beta1'] for r in rows),
                    checkpoint=str(run/'final.pt'),metrics=str(run/'final-heldout.json'))
                records.append(record)
        for initialization in ['scratch','mixed']:
            a,b=[read(OUT/'runs'/f'{dataset}_{initialization}_{v}'/'protocol.json') for v in ['B0','Axyz']]
            assert a['initialization_sha256']==b['initialization_sha256'] and a['exposure_sha256']==b['exposure_sha256']
    with (DEST/'final-results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(DEST/'final-results.json',records)
    fig,axes=plt.subplots(2,3,figsize=(13,7.6))
    for column,dataset in enumerate(DATASETS):
        chosen=[r for r in records if r['dataset']==dataset];x=np.arange(4)
        labels=['Scratch\nB0','Scratch\nA-xyz','Mixed\nB0','Mixed\nA-xyz']
        for row,key,title in [(0,'node_f1','Node F1'),(1,'edge_f1','Edge F1')]:
            ax=axes[row,column];values=[r[key] for r in chosen]
            bars=ax.bar(x,values,color=['#a5b4c4','#c4cbd4','#1678bc','#df8126'])
            ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.set_title(dataset+' — '+title+' @2 '+('µm' if dataset=='reaver' else 'px'))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    fig.suptitle('Held-out validation: final 100-epoch vessel models; confidence thresholds 0.5\nMixed = 50 epochs plants + target training images, then 100 vessel-only epochs',fontsize=12)
    fig.tight_layout(rect=[0,0,.99,.92]);fig.savefig(DEST/'final-f1-summary.png',dpi=180);fig.savefig(DEST/'final-f1-summary.pdf');plt.close(fig)
    return records

def select(ds):
    groups=sorted({r['subject_id'] for r in ds.rows})
    selected=[]
    for parent in [groups[0],groups[len(groups)//2]]:
        candidates=[i for i,r in enumerate(ds.rows) if r['subject_id']==parent and r['nodes']>=4 and r['edges']>=3]
        if not candidates:candidates=[i for i,r in enumerate(ds.rows) if r['subject_id']==parent and r['edges']>0]
        assert candidates,parent
        median=float(np.median([ds.rows[i]['nodes'] for i in candidates]))
        selected.append(min(candidates,key=lambda i:(abs(ds.rows[i]['nodes']-median),ds.rows[i]['sample_id'])))
    return selected

def guides(target):
    return [target['full_paths'][a:b] for a,b in zip(target['path_offsets'][:-1],target['path_offsets'][1:])]

def draw(ax,image,nodes,paths,color):
    ax.imshow(image,interpolation='nearest')
    for path in paths:ax.plot(path[:,0]*128,path[:,1]*128,color=color,lw=1.1,alpha=.95)
    if len(nodes):ax.scatter(nodes[:,0]*128,nodes[:,1]*128,s=18,c=color,edgecolors='black',linewidths=.35,zorder=4)
    ax.set_xlim(-.5,127.5);ax.set_ylim(127.5,-.5);ax.set_xticks([]);ax.set_yticks([])

@torch.no_grad()
def comparisons():
    torch.set_num_threads(2);device='cuda';selection={};predictions={};parity=[]
    for dataset in DATASETS:
        ds=StudyDataset(DATA/dataset/'manifest.json','validation',max_queries=96)
        ids=select(ds);selection[dataset]=[dict(validation_index=i,**ds.rows[i]) for i in ids]
        images={i:ds.raw(i)[0] for i in ids};truths={i:ds.raw(i)[1] for i in ids}
        for initialization in ['scratch','mixed']:
            for variant in ['B0','Axyz']:
                name=f'{dataset}_{initialization}_{variant}';run=OUT/'runs'/name
                state=torch.load(run/'final.pt',map_location='cpu')
                assert state['epoch']==100 and state['completed'] and state['dataset']==dataset
                model=build_model(state['config']).to(device).eval();model.load_state_dict(state['model'],strict=True);del state
                saved={r['sample_id']:r for r in read(run/'final-heldout.json')['samples']}
                batch_starts=sorted({i//4*4 for i in ids})
                for start in batch_starts:
                    indices=list(range(start,min(start+4,len(ds))))
                    x,t,g,rows=collate([ds[i] for i in indices],device)
                    tokens,pred=model(x)
                    for b,i in enumerate(indices):
                        if i not in ids:continue
                        result=decode(model,tokens,pred,b,t['spline'][b]['metric'],node_threshold=.5,edge_threshold=.5)
                        metric=np.asarray(g[b]['metric'])
                        scores=positional(result['nodes']@metric.T,result['edges'],g[b]['nodes']@metric.T,g[b]['edges'],2.)
                        reference=saved[rows[b]['sample_id']]
                        assert len(result['nodes'])==reference['predicted_nodes'] and len(result['edges'])==reference['predicted_edges']
                        assert scores['node']==reference['2.0']['node'] and scores['edge']==reference['2.0']['edge']
                        parity.append(dict(run=name,sample_id=rows[b]['sample_id'],saved_node_edge_counts_and_f1_exact=True))
                        paths=np.einsum('sk,ekd->esd',model.spline_head.basis.cpu().numpy(),result['spline_controls']) if len(result['edges']) else np.empty((0,64,2))
                        predictions[(name,i)]=dict(nodes=result['nodes'].tolist(),edges=result['edges'].tolist(),spline_controls=result.get('spline_controls',np.empty((0,8,2))).tolist(),node_scores=result['node_scores'].tolist(),edge_scores=result['edge_scores'].tolist(),node_f1=scores['node']['f1'],edge_f1=scores['edge']['f1'],paths=paths.tolist())
                del model,x,t,g,rows,tokens,pred;gc.collect();torch.cuda.empty_cache()
                print('CHECKPOINT '+name,flush=True)
        fig,axes=plt.subplots(2,3,figsize=(11,8.1))
        for row,i in enumerate(ids):
            image=images[i];gt=truths[i]
            draw(axes[row,0],image,gt['nodes'],guides(gt),'#25cf58')
            axes[row,0].set_title(f"Ground truth — {len(gt['nodes'])} nodes / {len(gt['edges'])} edges",fontsize=10)
            for column,variant in enumerate(['B0','Axyz'],1):
                result=predictions[(dataset+'_mixed_'+variant,i)]
                draw(axes[row,column],image,np.asarray(result['nodes']).reshape(-1,2),[np.asarray(p) for p in result['paths']],'#00b8ff')
                axes[row,column].set_title(('B0' if variant=='B0' else 'A-xyz')+f" mixed — N-F1 {result['node_f1']:.3f} / E-F1 {result['edge_f1']:.3f}",fontsize=10)
            axes[row,0].text(0,-.055,ds.rows[i]['sample_id'],transform=axes[row,0].transAxes,fontsize=7)
        unit='µm' if dataset=='reaver' else 'px'
        fig.suptitle(dataset+' — final 100-epoch models after 50 mixed-pretraining epochs\nHeld-out images; node/edge confidence 0.5; displayed F1 @2 '+unit,fontsize=12)
        fig.text(.5,.012,'GT-positive crops selected by ground-truth graph size only, before examining model performance.\nGreen: supplied/derived ground truth. Blue: predicted nodes and K8 curves. No test images.',ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.065,1,.92]);fig.savefig(DEST/('comparison-'+dataset+'.png'),dpi=180);fig.savefig(DEST/('comparison-'+dataset+'.pdf'));plt.close(fig)
    write(DEST/'selection.json',dict(rule='first and middle validation source-image groups; within each group select crop closest to median node count among crops with >=4 nodes and >=3 edges; sample_id tie break; fallback to edge-positive crops; no prediction-based selection',datasets=selection))
    write(DEST/'prediction-parity.json',dict(passed=True,checks=parity))
    write(DEST/'selected-predictions.json',[dict(run=name,validation_index=i,**value) for (name,i),value in predictions.items()])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--summaries-only',action='store_true');args=parser.parse_args();DEST.mkdir(exist_ok=True)
    summaries()
    if not args.summaries_only:comparisons()
    write(DEST/'provenance.json',dict(script=str(Path(__file__).resolve()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),evaluation='final.pt, epoch100, validation only, node/edge cuts0.5, unchanged metrics',test_arrays_opened=0,selection_is_prediction_independent=True))
    print('OUTPUT '+str(DEST),flush=True)

if __name__=='__main__':main()
