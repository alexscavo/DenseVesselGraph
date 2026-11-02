"""List or selectively restore archived files without overwriting changed work."""
import argparse,hashlib,json,os,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('paths',nargs='*');p.add_argument('--group');p.add_argument('--destination',type=Path,default=HERE.parents[2]);a=p.parse_args()
m=json.loads((HERE/'manifest.json').read_text());entries=m['entries']
if not a.paths and not a.group:
 for e in entries:print(e['group'],e['path'])
 raise SystemExit(0)
selected=[e for e in entries if e['path'] in a.paths or e['group']==a.group]
assert set(a.paths)<={e['path'] for e in entries},'Unknown archive path'
with zipfile.ZipFile(HERE/m['archive']) as z:
 for e in selected:
  data=z.read(e['path']);assert hashlib.sha256(data).hexdigest()==e['sha256']
  target=a.destination/e['path']
  if target.exists():
   assert target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==e['sha256'],'Refusing overwrite: '+str(target)
   continue
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(data)
  target.chmod(e['mode']);os.utime(target,ns=(e['mtime_ns'],e['mtime_ns']))
  print(target)
