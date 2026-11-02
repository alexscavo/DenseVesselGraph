"""Complete missing R06 geometry reporting from immutable scores and frozen curves.

No model loading, inference, fitting, query selection or R07 relation-head scores.
"""
import argparse,csv,gzip,io,json,sys,hashlib,fcntl,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
R07=HERE.parent/'gnbm-r07-matched-supervision-20261027'
sys.path[:0]=[str(R07),str(HERE)]
import numpy as np
from scripts.collect_r07 import curve_metrics,paired_curve_ci
from scripts.analyze_graph_errors import read,write,csvwrite,sha
from metrics.common_vessel_metrics import aggregate_common_vessels

BASE=HERE/'experiments/r06_relation_tokens_20261027'
OUT=BASE/'geometry_completion_20261027'
CURVES=R07/'experiments/r07_matched_supervision_20261027'
MAIN=HERE.parent/'Graph-Native-Betti-Matching-Representation'
FILES={'R0':'R0/scores-0000.zip','R1':'R1/scores-2000.zip','R2':'R2/scores-2000.zip',
       'R2_shuffled':'R2/scores-2000-shuffled.zip','R3_head':'R3_head/scores-2000.zip'}


def numeric(row):
    return {k:v for k,v in row.items() if k.startswith(('common_','native_','representation_'))}


def baseline_rows():
    result={}
    with gzip.open(CURVES/'common-v2-per-patch.csv.gz','rt') as f:
        for r in csv.DictReader(f):
            if r['arm']=='original' and r['policy']=='fixed' and int(r['step'])==0:
                result[int(r['seed']),r['split'],r['sample']]={k:float(v) if v else None for k,v in numeric(r).items()}
    return result


def run(smoke=False):
    OUT.mkdir(exist_ok=True)
    assert read(BASE/'analysis.json')['completed'] and read(CURVES/'publication.json')['completed']
    p=read(BASE/'plan.json');q=read(CURVES/'plan.json');original=baseline_rows()
    rows=[];audits=[];reused=measured=0;identities=[]
    for seed in p['seeds']:
        a=next(r for r in p['inputs'] if r['seed']==seed);b=next(r for r in q['inputs'] if r['seed']==seed)
        assert a['checkpoint_identity']['sha256']==b['checkpoint_identity']['sha256']
        assert [(r['sample'],r['split'],r['gt']) for r in p['records'][str(seed)]]==[(r['sample'],r['split'],r['gt']) for r in q['records'][str(seed)]]
        records=p['records'][str(seed)]
        if smoke:records=[next(r for r in records if r['split']=='val')]
        paths={arm:BASE/'large'/str(seed)/'runs'/name for arm,name in FILES.items()}
        zips={arm:zipfile.ZipFile(path) for arm,path in paths.items()}
        feature_sha=read(CURVES/'large'/str(seed)/'feature-audit.json')['features_sha256']
        assert sha(CURVES/'large'/str(seed)/'features.zip')==feature_sha
        identities.append(dict(seed=seed,checkpoint_sha256=a['checkpoint_identity']['sha256'],
          feature_cache_sha256=feature_sha,
          score_archives={arm:dict(path=str(path),sha256=sha(path)) for arm,path in paths.items()}))
        with zipfile.ZipFile(CURVES/'large'/str(seed)/'features.zip') as feature:
            for i,rec in enumerate(records):
                name=rec['sample']+'.npz'
                with np.load(io.BytesIO(feature.read(name))) as f:
                    c={k:f[k] for k in ['physical','scores','pairs','relations','metric','curve_pair_indices','curve_controls']}
                c['record']=rec;ids=np.flatnonzero(c['scores']>.5);c['context']={2.:dict(ids=ids)}
                active=np.isin(c['pairs'],ids).all(1);base_chosen=active&(c['relations']>.5)
                cache={tuple(np.flatnonzero(base_chosen)):original[seed,rec['split'],rec['sample']]}
                if i==0:
                    direct=curve_metrics(c,base_chosen);saved=next(iter(cache.values()))
                    for key,value in direct.items():
                        if value is not None:np.testing.assert_allclose(value,saved[key],rtol=1e-7,atol=1e-7)
                    audits.append(dict(seed=seed,sample=rec['sample'],direct_vs_saved_baseline=True))
                for arm,z in zips.items():
                    with np.load(io.BytesIO(z.read(name))) as f:probability=f['relations']
                    assert probability.shape==c['relations'].shape
                    if arm=='R0':np.testing.assert_allclose(probability,c['relations'],rtol=0,atol=1e-5)
                    chosen=active&(probability>.5);key=tuple(np.flatnonzero(chosen))
                    if key in cache:reused+=1
                    else:cache[key]=curve_metrics(c,chosen);measured+=1
                    rows.append(dict(seed=seed,arm=arm,split=rec['split'],sample=rec['sample'],subject=rec['subject'],
                                     selected_nodes=len(ids),selected_edges=int(chosen.sum()),**cache[key]))
                if i%128==0:print('R06 curve reporting',seed,i,len(records),flush=True)
        for z in zips.values():z.close()
    if smoke:
        write(OUT/'smoke.json',dict(passed=True,real_patches=2,source_and_manifest_identity=True,
              baseline_geometry_parity=audits,no_inference_or_training=True))
        print('R06 curve completion smoke passed',flush=True);return
    result=[];per_subject=[];differences=[];primary=read(BASE/'analysis.json')['summaries']
    for seed in p['seeds']:
        for split in ['train','val']:
            for arm in FILES:
                rr=[r for r in rows if (r['seed'],r['split'],r['arm'])==(seed,split,arm)]
                ref=next(r for r in primary if (r['seed'],r['split'],r['arm'],r['step'],r['tolerance'])==(seed,split,arm,0 if arm=='R0' else 2000,2.))
                assert sum(r['selected_nodes'] for r in rr)==ref['nodes']
                assert sum(r['selected_edges'] for r in rr)==ref['edges']
                result.append(dict(seed=seed,split=split,arm=arm,**aggregate_common_vessels([numeric(r) for r in rr])))
                for subject in sorted({r['subject'] for r in rr}):
                    per_subject.append(dict(seed=seed,split=split,arm=arm,subject=subject,
                       **aggregate_common_vessels([numeric(r) for r in rr if r['subject']==subject])))
            for control in ['R0','R1','R3_head','R2_shuffled']:
                paired=[[r for r in rows if (r['seed'],r['split'],r['arm'])==(seed,split,arm)] for arm in [control,'R2']]
                differences.append(dict(seed=seed,split=split,comparison='R2 minus '+control,**paired_curve_ci(*paired)))
    data=dict(completed=True,summaries=result,differences=differences,identities=identities,
              direct_baseline_audits=audits,graph_metrics_reused=reused,new_graph_metrics=measured,
              no_new_inference_training_or_relation_scores=True,fixed_final_updates=2000,
              conventions='Unchanged nodes, K8 curves, original node/edge cutoffs 0.5; common-v2 1mm anchors/support, 0.25mm sampling; original dense GT paths.',
              uncertainty='1000 paired whole-subject bootstrap draws, seed271026; reused development subjects, no source/seed uncertainty.')
    write(OUT/'summary.json',data);csvwrite(OUT/'per-patch.csv.gz',rows);csvwrite(OUT/'per-subject.csv',per_subject)
    publish(data)


def publish(data):
    text=['### R06 geometry-metric completion from saved predictions — 27 October 2026','',
      'The repeated R06 request matches the completed experiment above; no training, inference, architecture change or unfrozen confirmation was repeated. The missing common-v2 Branch/Curve metrics below reuse R06 final score vectors and compatible original frozen K8 candidate curves cached during R07 preparation. **No R07 trained scores or sampling policy enters this comparison.** Checkpoint hashes, split/sample/GT identities, query-pair ordering, original R0 score parity, original geometry parity and reproduced per-arm node/edge counts passed.','',
      '| Seed | Scorer, validation | Branch F1 | Curve F1 | Component MAE | Cycle MAE |',
      '|---:|---|---:|---:|---:|---:|']
    for r in data['summaries']:
        if r['split']=='val':text.append(f'|{r["seed"]}|{r["arm"]}|{r["common_branch_f1"]:.6f}|{r["common_curve_f1"]:.6f}|{r["common_betti0_abs_error_mean"]:.4f}|{r["common_betti1_abs_error_mean"]:.4f}|')
    text+=['','| Seed | R2 comparison | Branch F1 difference [95% subject CI] | Curve F1 difference [95% subject CI] |','|---:|---|---|---|']
    for r in data['differences']:
        if r['split']!='val':continue
        a,b=r['branch_f1'],r['curve_f1']
        text.append(f'|{r["seed"]}|{r["comparison"]}|{a["delta"]:+.6f} [{a["ci95"][0]:+.6f}, {a["ci95"][1]:+.6f}]|{b["delta"]:+.6f} [{b["ci95"][0]:+.6f}, {b["ci95"][1]:+.6f}]|')
    text+=['',
      'These geometry metrics supplement the completed negative discrimination/reconstruction result; they do not change the failed predeclared R06 confirmation gate. Image shuffling shows that CNN information is used, but final conditional AP/F1 are below original R0 in both seeds and there is no convincing R1 advantage. A possible Curve F1 gain must be assessed alongside the existing TP/FP/cycle counts; geometric coverage alone does not demonstrate connectivity improvement. Tiny early AP fluctuations are exploratory; no validation-selected stopping rule or new training is introduced. Retain original A-xyz. The better-supported remaining question concerns spatial identity, assignment and relation-label consistency; N08 and the completed R07 sampler negatives remain binding evidence, not invitations to repeat them or launch blanket unmatched mining.','',
      'This remains a 128-training/480-validation-patch, two-seed frozen residual pilot. It cannot rule out all CNN relation features or establish a causal training fix. Matched parameter counts do not make R1 redundant token-context input rank equal to independent CNN input rank. No test subjects were inspected.','',
      f'CPU reporting job: `{read(OUT/"submission.json")["collector"]}`. Evidence: sibling `gnbm-r06-relation-token-20261027/experiments/r06_relation_tokens_20261027/geometry_completion_20261027` (`summary.json`, per-patch/per-subject CSVs, identity manifest and smoke). Retrieve with `python3 gnbm-r06-relation-token-20261027/scripts/show_r06.py` from projects.']
    block='\n'.join(text)+'\n';(OUT/'report.md').write_text(block)
    report=MAIN/'docs/research/metrics/9-ixi-image-aware-relation-learning.md';registry=MAIN/'research/VASCULAR_GRAPH_EXPERIMENT_REGISTRY_2026-10-26_UPDATED.md'
    marker='<!-- R06 GEOMETRY REPORTING COMPLETED 20261027 -->'
    with (OUT/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=report.read_text()
        if marker not in old:
            assert report.read_text()==old;report.write_text(old.rstrip()+'\n\n'+marker+'\n'+block)
        old=registry.read_text()
        if marker not in old:
            assert registry.read_text()==old
            registry.write_text(old.rstrip()+'\n\n'+marker+'\nR06 remains DONE — negative/mixed. Missing Branch/Curve metrics were completed using saved R06 scores and unchanged original K8 curves; no new inference/training, query selection, R07 combination or unfrozen confirmation. See the existing relation-learning report.\n')
    write(OUT/'publication.json',dict(completed=True,report=str(report),registry=str(registry)))
    print('R06 geometry reporting completed; no training repeated',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(args.smoke)
