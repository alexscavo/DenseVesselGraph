"""CPU-only real-loader regression check for geometric augmentation diagnostics."""
import pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.brava_learning_audit import OUT,PREVIOUS,read,write
from scripts.brava_representation_study import dataset
from scripts.brava_learning_diagnosis import augmentation_audit
p=read(OUT/'plan.json');cfg=read(PREVIOUS/'plan.json')['config']
with (PREVIOUS/'targets.pkl').open('rb') as f:cache=pickle.load(f)
ds=dataset(cfg,cache,p['populations']['representative_train'],'K8','train',False)
result=augmentation_audit(ds);write(OUT/'cpu-augmentation-check.json',result);print(result)
