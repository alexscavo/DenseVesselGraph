"""Construct the actual R07 detector on CPU and strictly load both source states."""
import copy,inspect,json,sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import r07_matched_supervision as run
import torch
from models import build_model


def main():
    torch.set_num_threads(2)
    plan=run.read(run.OUT/'plan.json');rows=[]
    for seed in run.SEEDS:
        identity=next(r for r in plan['inputs'] if r['seed']==seed)
        cp=torch.load(identity['checkpoint'],map_location='cpu',mmap=True)
        config=copy.deepcopy(plan['configs'][str(seed)])
        model=build_model(config).eval().requires_grad_(False)
        actual=tuple(model.node_refinement.network[-1].weight.shape)
        expected=tuple(cp['net']['node_refinement.network.2.weight'].shape)
        assert actual==expected, f'R07 checkpoint mismatch: constructed {actual}, source {expected}'
        model.load_state_dict(cp['net'],strict=True)
        model_source=Path(inspect.getfile(type(model))).resolve()
        assert model_source.is_relative_to((run.OUT/'frozen-source').resolve()),str(model_source)
        assert not model.node_refinement.refine_classification
        rows.append(dict(seed=seed,checkpoint=identity['checkpoint'],shape=list(actual),
                         strict_load=True,model_source=str(model_source),coordinate_only=True))
        del model,cp;gc.collect()
    run.write(run.OUT/'checkpoint-loading-smoke.json',dict(passed=True,device='cpu',sources=rows))
    print('Both coordinate-only source checkpoints loaded strictly using frozen model code.')


if __name__=='__main__':main()
