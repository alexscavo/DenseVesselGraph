"""Original GrEVe 2D RelationFormer core with optional modern geometry/refinement.

Legacy core parameter names are retained. No 3D module or checkpoint is migrated.
"""
import torch
from torch import nn
from .backbone import build_backbone
from .transformer import build_deforamble_transformer
from .structures import NestedTensor
from .config import legacy_architecture
from .geometry import LocalCoordinateRefinement2D,SplineHead2D


class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers):
        super().__init__()
        h=[hidden_dim]*(num_layers-1)
        self.layers=nn.ModuleList(nn.Linear(a,b) for a,b in zip([input_dim]+h,h+[output_dim]))

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<len(self.layers)-1:x=x.relu()
        return x


class RelationFormer2D(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        e,d=config['model']['encoder'],config['model']['decoder']
        if d['relation_attention']:
            raise ValueError('Use the original 2D all-query deformable attention; legacy RLN_ATTN singles out only one token')
        legacy=legacy_architecture(config)
        self.encoder=build_backbone(legacy)
        self.decoder=build_deforamble_transformer(legacy)
        self.obj_token=d['object_queries'];self.relation_tokens=d['relation_tokens']
        width=d['hidden_dim'];self.num_queries=self.obj_token+self.relation_tokens+d['dummy_tokens']
        self.class_embed=nn.Linear(width,2)
        self.bbox_embed=MLP(width,width,4,3)
        self.relation_embed=MLP(width*(2+self.relation_tokens),width,2,3)
        self.query_embed=nn.Embedding(self.num_queries,2*width)
        channels=self.encoder.num_channels
        self.input_proj=nn.ModuleList(nn.Sequential(nn.Conv2d(c,width,1),nn.GroupNorm(32,width)) for c in channels)
        for i in range(len(channels),d['feature_levels']):
            self.input_proj.append(nn.Sequential(nn.Conv2d(channels[-1] if i==len(channels) else width,
                width,3,stride=2,padding=1),nn.GroupNorm(32,width)))
        if e.get('train_all',True): self.encoder[0].body.requires_grad_(True)
        self.node_refinement=None
        r=config['model']['node_refinement']
        if r['enabled']:
            with torch.random.fork_rng(devices=[]):
                self.node_refinement=LocalCoordinateRefinement2D(width,radius_pixels=r['radius_pixels'],image_mode=r['image_mode'])
        s=config['model']['spline'];self.spline_head=None
        if s['enabled']:self.spline_head=SplineHead2D(width*(2+self.relation_tokens),width,s['controls'],s['samples'])

    def forward(self,images):
        if images.ndim!=4 or images.shape[1] not in (1,3): raise ValueError('Expected BCHW with 1 or 3 image channels')
        if images.shape[1]==1:images=images.expand(-1,3,-1,-1)
        nested=NestedTensor(images,torch.zeros(images.shape[0],*images.shape[-2:],dtype=torch.bool,device=images.device))
        features,pos=self.encoder(nested)
        srcs=[];masks=[]
        for i,feature in enumerate(features):
            src,mask=feature.decompose();srcs.append(self.input_proj[i](src));masks.append(mask)
        for i in range(len(srcs),self.config['model']['decoder']['feature_levels']):
            src=self.input_proj[i](features[-1].tensors if i==len(features) else srcs[-1])
            mask=torch.zeros(src.shape[0],*src.shape[-2:],dtype=torch.bool,device=src.device)
            srcs.append(src);masks.append(mask);pos.append(self.encoder[1](NestedTensor(src,mask)).to(src.dtype))
        tokens,*_=self.decoder(srcs,masks,self.query_embed.weight,pos)
        object_tokens=tokens[:,:self.obj_token]
        nodes=self.bbox_embed(object_tokens).sigmoid()
        early=self.encoder[0].early_features
        original=nodes
        if self.node_refinement is not None:nodes=self.node_refinement(object_tokens,nodes,early,images.shape[-2:])
        return tokens,dict(pred_logits=self.class_embed(object_tokens),pred_nodes=nodes,pred_nodes_stage1=original)

    def pair_features(self,tokens,pairs):
        parts=[tokens[pairs[:,0]],tokens[pairs[:,1]]]
        if self.relation_tokens:
            parts.append(tokens[self.obj_token:self.obj_token+self.relation_tokens].flatten()[None].expand(len(pairs),-1))
        return torch.cat(parts,-1)

    def relation_logits(self,tokens,pairs):
        return (self.relation_embed(self.pair_features(tokens,pairs))+
            self.relation_embed(self.pair_features(tokens,pairs.flip(1))))/2

    def curves(self,tokens,nodes,pairs,metric):
        # Canonical physical ordering, then restore caller endpoint order.
        endpoints=nodes[pairs,:2].detach()
        physical=endpoints@metric.T
        reverse=(physical[:,0,0]>physical[:,1,0])|((physical[:,0,0]==physical[:,1,0])&(physical[:,0,1]>physical[:,1,1]))
        canonical=pairs.clone();canonical[reverse]=canonical[reverse].flip(1)
        controls,curves=self.spline_head(self.pair_features(tokens,canonical).detach(),nodes[canonical,:2].detach(),metric)
        controls=controls.clone();curves=curves.clone()
        controls[reverse]=controls[reverse].flip(1);curves[reverse]=curves[reverse].flip(1)
        return controls,curves

    def load_legacy_state(self,state):
        """Strict legacy-core import; only explicitly new heads may be absent."""
        own=self.state_dict()
        extra=set(state)-set(own)
        missing={k for k in own if k not in state and not k.startswith(('node_refinement.','spline_head.'))}
        if extra or missing:raise ValueError(f'Incompatible legacy state: extra={sorted(extra)}, missing={sorted(missing)}')
        for k,v in state.items():
            if own[k].shape!=v.shape:raise ValueError(f'Incompatible shape for {k}: {v.shape} vs {own[k].shape}')
        self.load_state_dict({**own,**state},strict=True)
