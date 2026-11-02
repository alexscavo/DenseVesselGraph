"""One-off, manifest-driven packing of completed secondary artifacts."""
from pathlib import Path
import collections, hashlib, json, os, stat, subprocess, zipfile
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.chdir(ROOT)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],text=True)
files=[Path(s) for s in git('ls-files','--others','--exclude-standard','-z').split('\0') if s]
text=[]
for directory in ['docs','research']:
    for p in Path(directory).rglob('*.md'):
        if not p.is_symlink():text.append(p.read_text(errors='replace'))
reports='\n'.join(text)
# Keep the corrected alignment panel for every audited subject and paper PDFs.
keep_images={'BG0001-alignment.png','BG0002-alignment.png','BH0003-alignment.png','BH0040-alignment.png'}
keep_launchers={'brava_augmentation_cpu.slurm','brava_augmentation_gpu.slurm','limited_runtime.sh','rebase_resume_checkpoint_lr.py','prepare_brava.slurm'}
entries=[]
for p in files:
    name=str(p);group=None;reason=None
    if p.is_symlink() or not p.is_file():continue
    if name.startswith('cluster/jean_zay/') and p.name not in keep_launchers:
        group='historical-launchers';reason='Completed Oct19-25 study launcher/recovery; scientific implementation retained'
    elif name.startswith('experiments/brava_preparation/orientation-preflight-20261020/native_swc/') and p.suffix=='.npz':
        group='audit-exports';reason='Derived preflight SWC exports; original dataset and compact audit remain untouched'
    elif name.startswith('experiments/ixi_image_relation_learning/reporting-smoke/') and p.suffix not in ['.md','.py']:
        group='reporting-smoke';reason='Completed old reporting smoke evidence; final study reports retained'
    elif p.suffix=='.png' and '20261026' not in name and '20261027' not in name and p.name not in reports:
        if name.startswith('docs/research/evidence/brava_preparation/') and (p.name in keep_images or p.name.endswith('-alignment-corrected.png')):continue
        group='secondary-views';reason='Unlinked secondary rendering; report-linked images and corrected alignment panels retained'
    elif name.startswith(('experiments/brava_graph_study/preflight','experiments/multidataset_continuous/local-audit/','experiments/ixi_segmentation_input/preflight/')) and p.name=='plan.json':
        study=Path(*p.parts[:2])/'plan.json'
        if study.exists() and sha(p)==sha(study):group='duplicate-plans';reason='Byte-identical copy of retained '+str(study)
    if group:
        st=p.stat();entries.append(dict(path=name,group=group,reason=reason,bytes=st.st_size,sha256=sha(p),mode=stat.S_IMODE(st.st_mode),mtime_ns=st.st_mtime_ns))
assert not (OUT/'manifest.json').exists(), 'Do not overwrite a previous cleanup'
# All existing reports and dirty tracked files are protected against accidental edits.
protected={}
for s in git('diff','--name-only','-z').split('\0'):
    p=Path(s)
    if s and p.is_file() and not p.is_symlink():protected[s]=sha(p)
for directory in ['docs','research']:
    for p in Path(directory).rglob('*.md'):
        if p.is_file() and not p.is_symlink():protected[str(p)]=sha(p)
(OUT/'protected-before.json').write_text(json.dumps(protected,indent=2)+'\n')
(OUT/'git-status-before.txt').write_text(git('status','--short'))
manifest=dict(repository=str(ROOT),archive='secondary-artifacts.zip',entries=entries,untracked_before=len(files),groups=dict(collections.Counter(e['group'] for e in entries)))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=OUT/manifest['archive']
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for e in entries:z.write(e['path'],e['path'])
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for e in entries:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
# Refuse to remove anything changed by another process since the inventory.
for e in entries:
    p=Path(e['path']);assert not p.is_symlink() and sha(p)==e['sha256'],e['path']
for e in entries:Path(e['path']).unlink()
# Remove only empty directories made empty by this operation.
parents={Path(e['path']).parent for e in entries}
for p in sorted(parents,key=lambda p:len(p.parts),reverse=True):
    while p!=Path('.'):
        try:p.rmdir()
        except OSError:break
        p=p.parent
assert all(Path(p).is_file() and sha(Path(p))==h for p,h in protected.items())
manifest.update(completed=True,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,original_bytes=sum(e['bytes'] for e in entries),files_packed=len(entries),protected_files_verified=len(protected))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='entries'},indent=2))
