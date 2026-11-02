"""Manifest-based paired 2D image/graph loading; masks never substitute for images."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def transform(image,target,*,quarter_turns=0,flip_x=False):
    """Exact pixel-center transforms including controls, paths and physical metric."""
    target={k:np.array(v,copy=True) for k,v in target.items()}
    for operation in ['rotate']*(quarter_turns%4)+(['flip'] if flip_x else []):
        height,width=image.shape[-2:];old_size=np.array([width,height],float)
        if operation=='rotate':
            matrix=np.array([[0,1],[-1,0]],float);offset=np.array([0,width-1],float)
            image=torch.rot90(image,1,[-2,-1]);new_size=old_size[::-1]
        else:
            matrix=np.diag([-1.,1.]);offset=np.array([width-1,0],float)
            image=image.flip(-1);new_size=old_size
        for key in ('nodes','controls','full_paths'):
            target[key]=((target[key]*old_size)@matrix.T+offset)/new_size
        target['metric']=target['metric']@np.diag(1/old_size)@matrix.T@np.diag(new_size)
        target['shape_hw']=new_size[::-1].astype(int)
    return image.contiguous(),target


class GraphDataset2D(Dataset):
    def __init__(self,manifest,split,*,augment=False,max_queries=None):
        self.root=Path(manifest).resolve().parent
        records=json.loads(Path(manifest).read_text())['samples']
        subjects={}
        for row in records:
            old=subjects.setdefault(row['subject_id'],row['split'])
            if old!=row['split']:raise ValueError(f'Subject leakage: {row["subject_id"]}')
        self.rows=[r for r in records if r['split']==split]
        self.augment=augment;self.max_queries=max_queries

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        image_path=self.root/row['image'];target_path=self.root/row['target']
        image=np.asarray(Image.open(image_path).convert('RGB'),dtype=np.float32)/255.-.5
        image=torch.from_numpy(image.transpose(2,0,1).copy())
        with np.load(target_path) as stored:target={k:stored[k].copy() for k in stored.files}
        if self.max_queries is not None and len(target['nodes'])>self.max_queries:
            raise ValueError('Target exceeds query capacity; do not silently discard nodes')
        if self.augment:
            turns=int(torch.randint(4,()).item());flip=bool(torch.randint(2,()).item())
            image,target=transform(image,target,quarter_turns=turns,flip_x=flip)
        return image,target,row


def collate(samples,device='cpu'):
    images,targets,rows=zip(*samples)
    result=dict(nodes=[torch.as_tensor(t['nodes'],dtype=torch.float32,device=device) for t in targets],
        edges=[torch.as_tensor(t['edges'],dtype=torch.long,device=device) for t in targets],
        spline=[{k:torch.as_tensor(t[k],dtype=torch.float32,device=device) for k in ('metric','controls')} for t in targets])
    return torch.stack(images).to(device),result,list(targets),list(rows)
