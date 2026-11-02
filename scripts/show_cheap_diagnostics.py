"""Stdlib-only retrieval; never reruns inference, calibration or optimization."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];meta=root/'experiments/cheap_diagnostics_20261026';p=json.loads((meta/'plan.json').read_text())
r=json.loads((meta/'relation-analysis.json').read_text())
print('Completed cache-only relation analysis:',meta/'relation-analysis.json')
for name in ['original','token_geometry/2000','chord/2000','corridor/2000','chord/65600']:
 x=next(v for v in r['ranking'] if v['scorer']==name and v['split']=='val' and v['population']=='conditional' and v['tolerance_mm']==2)
 print(name,'validation conditional AP',round(x['ap'],6))
if (meta/'jobs.json').exists():print((meta/'jobs.json').read_text())
if (meta/'bn-analysis.json').exists():
 print('BN analysis:',meta/'bn-analysis.json')
 for name,values in json.loads((meta/'bn-analysis.json').read_text()).items():print(name,json.dumps(values['paired']['2.0']['metrics'],indent=2))
else:
 for path in Path(p['bn_output_root']).glob('*/complete.json'):print(path,path.read_text())
 print('BN paired collection not completed yet.')
for path in meta.glob('*integration-pending.json'):print('Deferred shared-document integration:',path)
