"""Schedule existing edge CE; preserve every other B0 objective and architecture."""
import math

import torch
from torch import nn


def edge_multiplier(step, total_updates, arm):
    if arm == 'joint':
        return 1.0
    if arm != 'node_first' or not 1 <= step <= total_updates:
        raise ValueError((arm, step, total_updates))
    first, last = total_updates // 10, total_updates // 5
    if first == last:
        raise ValueError('Curriculum needs at least ten updates')
    if step <= first:
        return 0.0
    if step >= last:
        return 1.0
    return .5 * (1 - math.cos(math.pi * (step - first) / (last - first)))


def scheduled_total(losses, criterion, multiplier):
    # Preserve the original float summation exactly for the unchanged control
    # and normal joint phase. Subtracting then adding edge CE would leave a
    # zero-valued gradient and AdamW decay/state updates during node-only steps.
    if multiplier == 1:
        return losses['total']
    terms = []
    for key in criterion.enabled_losses:
        if key == 'edges' and multiplier == 0:
            continue
        term = losses[key] * criterion.weights[key]
        terms.append(term * multiplier if key == 'edges' else term)
    return sum(terms)


def gradient_alignment(node_term, edge_term, parameters):
    parameters = [p for p in parameters if p.requires_grad]
    a = torch.autograd.grad(node_term, parameters, retain_graph=True, allow_unused=True)
    b = torch.autograd.grad(edge_term, parameters, retain_graph=True, allow_unused=True)
    aa = sum(float(x.detach().square().sum()) for x in a if x is not None)
    bb = sum(float(x.detach().square().sum()) for x in b if x is not None)
    dot = sum(float((x.detach() * y.detach()).sum()) for x, y in zip(a, b)
              if x is not None and y is not None)
    return dict(node_norm=aa ** .5, edge_norm=bb ** .5,
                cosine=dot / math.sqrt(aa * bb) if aa * bb else None)


class ScheduledGraphObjective(nn.Module):
    """Wrap model AND criterion for DDP's dynamic unused-parameter discovery.

    Raw edge CE is always computed, preserving candidate RNG and diagnostics.
    Only the scheduled total participates in backward. Other returned tensors
    are detached, so DDP cannot mistake an unused raw edge loss for a backward
    root. Use DDP(find_unused_parameters=True); don't wrap just the detector,
    since its relation head is invoked by the criterion outside model.forward.
    """
    def __init__(self, model, criterion, arm, total_updates):
        super().__init__()
        self.model, self.criterion = model, criterion
        self.arm, self.total_updates = arm, total_updates
        if getattr(criterion, 'degree', {}).get('enabled', False):
            raise ValueError('This B0 comparison does not enable degree losses')
        if any(x.get('enabled', False) for x in getattr(criterion, 'topology', {}).values()):
            raise ValueError('This B0 comparison does not enable topology losses')

    def forward(self, volumes, targets, step, probe=False):
        tokens, predictions, _ = self.model(volumes)
        losses = self.criterion(tokens, predictions, targets)
        multiplier = edge_multiplier(step, self.total_updates, self.arm)
        total = scheduled_total(losses, self.criterion, multiplier)
        metrics = {key: value.detach() for key, value in losses.items()}
        metrics.update(edge_multiplier=total.new_tensor(multiplier).detach(),
                       edge_weighted=losses['edges'].detach() * self.criterion.weights['edges'] * multiplier,
                       joint_total=losses['total'].detach(), total=total.detach())
        if probe:
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                raise ValueError('Gradient probes are single-GPU diagnostics only')
            node_term = sum(losses[k] * self.criterion.weights[k]
                            for k in ['boxes', 'class', 'cardinality', 'nodes'])
            parameters = self.model.decoder.decoder.layers[-1].parameters()
            metrics['gradient_alignment'] = gradient_alignment(
                node_term, losses['edges'] * self.criterion.weights['edges'], parameters)
        return total, metrics
