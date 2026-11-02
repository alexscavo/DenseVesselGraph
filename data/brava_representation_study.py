"""Study-local target overlay; original data and original loader stay unchanged."""
import copy
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from data.augmentations import rotate_coordinates, rotate_volume
from data.spline_augmentation import rotate_spline_geometry
from data.loaders.synthetic_mri import _build_training_intensity_transform


class RepresentationDataset(Dataset):
    def __init__(self, base, caches, arm, augment=False):
        self.base=base;self.caches=caches;self.arm=arm;self.augment=augment
        self.records=base.records;self.images={}
        self.intensity=_build_training_intensity_transform(.35,.015,None) if augment else None
        base.augment=False;base.intensity_transform=None

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index];sample=record.sample_id
        if index not in self.images:self.images[index]=self.base[index]
        old=self.images[index];target=self.caches[sample][self.arm]
        image=old[0][0].clone();mask=old[1][0].clone()
        nodes=torch.from_numpy(target['nodes'].copy());edges=torch.from_numpy(target['edges'].copy()).long()
        geometry={k:torch.from_numpy(target[k].copy()).float() for k in ['controls','curves','guides','metric','branch_weights']}
        if self.augment:
            turns=tuple(random.randint(0,3) for _ in range(3));shape=image.shape[-3:]
            image=rotate_volume(image,turns);mask=rotate_volume(mask,turns)
            nodes=rotate_coordinates(nodes,turns,shape,allow_patch_faces=True)
            weights=geometry['branch_weights'];geometry=rotate_spline_geometry(geometry,turns,shape);geometry['branch_weights']=weights
            self.intensity.set_random_state(seed=random.randrange(2**31));image=self.intensity(image)
        return ([image],[mask],[nodes],[edges],[geometry],[old[5][0]])


def migrate_geometry(old,spec,method):
    """Transfer residual functions by fixed-basis projection, including local output."""
    from scripts.multidataset_graph_training import geometry_head
    from models.adaptive_geometry import uniform_knots
    from scipy.interpolate import BSpline
    head=geometry_head(old,spec,method)
    old_basis=BSpline(uniform_knots(8),np.eye(8),3)(np.linspace(0,1,513))[:,1:-1]
    knots=head.knots.numpy() if hasattr(head,'knots') else uniform_knots(head.controls)
    new_basis=BSpline(knots,np.eye(head.controls),3)(np.linspace(0,1,513))[:,1:-1]
    matrix=np.linalg.lstsq(new_basis,old_basis,rcond=None)[0]
    transform=torch.tensor(np.kron(matrix,np.eye(3)),dtype=old.network[-1].weight.dtype)
    with torch.no_grad():
        for source,dest in [(old.network[-1],head.network[-1]),(old.local_geometry.scorer[-1],head.local_geometry.scorer[-1])]:
            dest.weight.copy_(transform@source.weight);dest.bias.copy_(transform@source.bias)
    return head,dict(controls=head.controls,basis_max_error=float(np.abs(new_basis@matrix-old_basis).max()),method='least-squares residual-function transfer; exact if old knots contained')
