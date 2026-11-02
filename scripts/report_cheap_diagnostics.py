"""Append isolated diagnostics to existing reports; never overwrite concurrent edits."""
import argparse,fcntl,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cheap_research_diagnostics import META,ORIGINAL,BN_ROOT,read,write
BN_DOC='docs/research/spline_learning/spline-collapse-diagnosis.md'
REL_DOC='docs/research/metrics/9-ixi-image-aware-relation-learning.md'

def append_checked(relative,tag,section,expected=None):
 path=ORIGINAL/relative;old=path.read_text();digest=hashlib.sha256(old.encode()).hexdigest()
 start=f'<!-- BEGIN CHEAP DIAGNOSTICS {tag} 20261026 -->';end=f'<!-- END CHEAP DIAGNOSTICS {tag} 20261026 -->'
 if expected is not None and digest!=expected:
  write(META/(tag+'-integration-pending.json'),dict(target=str(path),reason='Shared file changed since ownership audit; integration deferred',section=section));return False
 block=start+'\n'+section.rstrip()+'\n'+end
 if start in old:
  assert old.count(start)==old.count(end)==1
  a=old.index(start);b=old.index(end)+len(end);new=old[:a]+block+old[b:]
 else:new=old.rstrip()+'\n\n'+block+'\n'
 # Cooperating writers can share this lock; the content check also detects other writers.
 lock=path.with_suffix(path.suffix+'.diagnostics.lock')
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if path.read_text()!=old:
   write(META/(tag+'-integration-pending.json'),dict(target=str(path),reason='Concurrent edit detected immediately before append',section=section));return False
  path.write_text(new)
 write(META/(tag+'-integration.json'),dict(target=str(path),before_sha256=digest,after_sha256=hashlib.sha256(new.encode()).hexdigest(),completed=True));return True

def relation_section():
 r=read(META/'relation-analysis.json');ranking=r['ranking'];cis=r['paired_intervals']
 get=lambda n,s='val',tol=2:next(x for x in ranking if x['scorer']==n and x['population']=='conditional' and x['split']==s and x['tolerance_mm']==tol)['ap']
 s='## Cached early relation learning and overfitting —26 October2026\n\nThis new analysis reuses the completed study’s **saved scores**, without new GPU inference, optimizer updates, architecture or loss changes. The original2048/512 patches and123/26 disjoint subjects remain fixed. Pair offsets/order, sample IDs/splits, labels and frozen original margins were checked exactly at every checkpoint. New evidence: paired subject uncertainty for early versus original/control ranking, plus fixed and **training-selected** edge operating points. Earlier intermediate AP measurements are reused, not presented as newly trained results.\n\n'
 s+='| Scorer | Conditional AP train /val, 1k | Conditional AP train /val, 2k | Conditional AP train /val, 65,600 |\n|---|---:|---:|---:|\n'
 s+=f"| Original | {get('original','train'):.5f} /{get('original'):.5f} | same frozen scorer | same frozen scorer |\n"
 for arm in ['token_geometry','endpoints','chord','corridor']:
  s+='| '+arm+' | '+' | '.join(f"{get(arm+'/'+str(k),'train'):.5f} /{get(arm+'/'+str(k)):.5f}" for k in [1000,2000,65600])+' |\n'
 s+='\n| Early scorer | Reference | AP difference, 2 mm [95% subject interval] |\n|---|---|---:|\n'
 for row in cis:
  if row['tolerance_mm']==2 and any(row['scorer'].endswith('/'+str(k)) for k in [1000,2000]):
   s+=f"| {row['scorer']} | {row['reference']} | {row['delta_ap']:+.5f} [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] |\n"
 s+='\n| Early scorer | Policy | Validation edge F1 | Difference vs original [95% subject interval] |\n|---|---|---:|---:|\n'
 for row in r.get('paired_graph_intervals',[]):
  if row['scorer'] not in ['chord/2000','corridor/2000']:continue
  point=next(x['edge_f1'] for x in r['operating_points'] if x['scorer']==row['scorer'] and x['policy']==row['policy'] and x['split']=='val' and x['tolerance_mm']==2)
  s+=f"| {row['scorer']} | {row['policy']} | {point:.5f} | {row['delta_edge_f1']:+.5f} [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}] |\n"
 s+='\nOriginal validation edge F1 is0.07801 at0.5,0.15330 using its training-max-F1 threshold, and0.08047 under the training baseline-precision policy. Chord/corridor2k reach0.13147/0.13849 under their independently training-selected precision-preserving policies; attained validation precision is0.15400/0.15475 versus original0.15710. Training precision constraints do not force exactly equal validation precision. Edge intervals condition on the selected training thresholds and omit their calibration-sample uncertainty. Early Branch/Curve/topology were not recomputed here, so these ranking/edge gains do not establish complete graph superiority.\n'
 s+='\nThe intervals use1,000 paired whole-subject draws (seed191032), preserving pooled pair prevalence and tied scores. Conditional and all-retained AP remain separate; full1/2-mm tables, pair/positive counts, source-score SHA256 and training-calibrated edge F1/recall/conditional-recall policies are in `'+str((META/'relation-analysis.json').resolve())+'` and adjacentCSV files. Fixed relation threshold0.5 remains unchanged in that policy. No validation thresholds choose the reported training-calibrated policies.\n'
 supported=[x for x in cis if x['tolerance_mm']==2 and x['reference'].startswith('token_geometry/') and x['ci95_low']>0 and x['scorer'].rsplit('/',1)[-1] in ['1000','2000']]
 s+='\nEarly image-versus-matched-token-control intervals exclude zero for: '+(', '.join(x['scorer'] for x in supported) if supported else 'none')+'. This is retrospective development evidence across multiple inspected checkpoints/arms, not an independently validated early-stopping rule. The late training/validation divergence demonstrates overfitting of this frozen-feature residual recipe; it does not justify another long relation run. Retain the original scorer as the operational reference until a predefined short schedule or train-only stopping rule is confirmed on disjoint subjects. Additional data must mean new subjects or an explicit independent confirmation, not merely reusing these validation outcomes.\n'
 s+='\nNo automatic relation training was submitted: saved checkpoints already answer this bounded diagnostic. No new Branch/Curve experiment is implied; the earlier full reconstruction tables remain authoritative for their evaluated checkpoints. Wider corridors are still bounded image context, not multiscale attention or GT paths.\n'
 return s

def bn_section():
 from scripts.report_multiscale_pretrained import paired,selected
 results={};s='## Training-only BN recalibration —26 October2026\n\nThis is a new weight-frozen diagnostic on the compatible IXI delayed-spline and SyntheticMRI vascular checkpoints; it is **not** a reanalysis claiming repair of the old collapsed model above. Calibration uses256 existing training-monitor patches, no augmentation, batch8 ×32 batches; the identical128 validation-monitor patches use batch1 before/after. Node/edge cutoffs stay0.5, targets/splits/matcher unchanged, no test data. Existing saved-BN raw-query caches are reused after fresh three-patch position/confidence/relation parity checks. Only BN buffers change; parameter and non-BN-buffer checksums must agree. Stats are not reset; existing per-layer momentum and exponential updates are retained. Only isolated BN-buffer tensors are saved, never over the original checkpoint.\n\n'
 s+='| Dataset | Tolerance | Node F1 saved → recalibrated | Edge F1 saved → recalibrated | Edge delta [95% subject interval] |\n|---|---|---:|---:|---:|\n'
 for dataset in ['IXI','SyntheticMRI']:
  done=read(BN_ROOT/dataset/'complete.json');assert done['completed'] and done['weights_unchanged'] and done['other_buffers_unchanged'];rr=read(BN_ROOT/dataset/'rows.json');a=[r for r in rr if r['condition']=='saved'];b=[r for r in rr if r['condition']=='train_recalibrated'];results[dataset]=dict(complete=done,paired={t:paired(selected(a,tol=t),selected(b,tol=t)) for t in ['1.0','2.0']})
  for t in ['1.0','2.0']:
   x=done['summary']['saved'][t];y=done['summary']['train_recalibrated'][t];d=results[dataset]['paired'][t]['metrics']['edge_f1'];s+=f"| {dataset} | {t} mm | {x['node_f1']:.5f} → {y['node_f1']:.5f} | {x['edge_f1']:.5f} → {y['edge_f1']:.5f} | {d['delta']:+.5f} [{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] |\n"
 write(META/'bn-analysis.json',results)
 s+='\nPer-patch/subject bootstrap evidence, raw/filtered recovery, Hungarian/localized distances, conditional relation AP/recall, duplicate and component/cycle errors are retained in `'+str((META/'bn-analysis.json').resolve())+'`; runtime outputs and BN-only states: `'+str(BN_ROOT)+'`. Subject intervals omit calibration-population and seed variability. An apparent node gain alone is not improved connectivity; inspect edge uncertainty and topology before adopting recalibration. No threshold retuning or model training occurred.\n'
 for dataset,r in results.items():
  e=r['paired']['2.0']['metrics']['edge_f1'];n=r['paired']['2.0']['metrics']['node_f1']
  verdict='supports an edge-F1 gain in this paired pilot' if e['ci95'][0]>0 else 'shows a significant edge-F1 reduction in this paired pilot' if e['ci95'][1]<0 else 'does not establish an edge-F1 gain'
  s+=f"\n- {dataset}: node-F1 delta{n['delta']:+.5f}; edge-F1 delta{e['delta']:+.5f} {verdict}.\n"
 s+='\nThe generic historical `evaluate.py` validation-loader calibration is not called here. Historical validation-driven BN adaptation remains separately labeled and cannot be silently substituted for this training-only comparison.\n'
 return s

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['relation','bn']);a=parser.parse_args()
 section=relation_section() if a.mode=='relation' else bn_section();target=REL_DOC if a.mode=='relation' else BN_DOC
 # Future BN collection checks the hash recorded AFTER this session's protocol append.
 baseline=META/'publication-hashes.json'
 expected=read(baseline).get(target) if baseline.exists() else read(META/'shared-document-audit.json')[target]['sha256']
 ok=append_checked(target,a.mode,section,expected)
 print('Published' if ok else 'Shared edit conflict; staged integration JSON retained',target,flush=True)
if __name__=='__main__':main()
