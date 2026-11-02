"""Reuse modern Hungarian/focal/matched-edge supervision; replace 3D geometry only."""
import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou,box_convert
from models.matcher import HungarianMatcher
from training.losses.criterion import GraphCriterion


class GraphCriterion2D(GraphCriterion):
    def __init__(self,config,model,*,validation=False):
        m=config['model']['matcher']
        super().__init__(config,HungarianMatcher(m['class_cost'],m['node_cost'],dimensions=2),
            model.relation_embed,validation=validation,spline_head=model.spline_head)
        object.__setattr__(self,'geometry_model',model)

    def loss_boxes(self,predicted_nodes,target_nodes,assignments):
        predicted,centers=self._matched_nodes(predicted_nodes,target_nodes,assignments)
        if not centers.numel():return predicted_nodes.sum()*0.
        target=torch.cat((centers,.2*torch.ones_like(centers)),-1)
        overlap=generalized_box_iou(box_convert(predicted,'cxcywh','xyxy'),box_convert(target,'cxcywh','xyxy'))
        return (1-overlap.diag()).mean()

    def loss_spline(self,tokens,predictions,targets,assignments):
        errors=[];control_errors=[];endpoint_errors=[]
        for b,(source,target) in enumerate(assignments):
            mapping=dict(zip(target.tolist(),source.tolist()))
            selected=[i for i,(a,c) in enumerate(targets['edges'][b].tolist()) if a in mapping and c in mapping]
            if not selected:continue
            pairs=torch.tensor([[mapping[a],mapping[c]] for a,c in targets['edges'][b][selected].tolist()],device=tokens.device)
            g=targets['spline'][b];metric=g['metric'].to(tokens)
            controls,_=self.geometry_model.curves(tokens[b],predictions['pred_nodes'][b],pairs,metric)
            truth=g['controls'][selected].to(tokens)
            dc=(controls[:,1:-1]-truth[:,1:-1])@metric.T
            endpoints=predictions['pred_nodes'][b,pairs,:2].detach()
            endpoint_delta=(endpoints-truth[:,[0,-1]])@metric.T
            t=self.spline_head.greville[None,1:-1,None]
            delta=dc-((1-t)*endpoint_delta[:,:1]+t*endpoint_delta[:,1:])
            errors.append(F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none',
                beta=float(self.config['loss']['spline']['beta_distance'])).mean((1,2)))
            control_errors.append(dc.norm(dim=-1).mean(1))
            endpoint_errors.append(endpoint_delta.norm(dim=-1).mean(1))
        zero=sum(p.sum()*0. for p in self.spline_head.parameters())
        mean=lambda values:torch.cat(values).mean() if values else zero
        return dict(spline=mean(errors),spline_control_distance=mean(control_errors),
            spline_endpoint_distance=mean(endpoint_errors),spline_positive_edges=tokens.new_tensor(sum(len(x) for x in errors)))
