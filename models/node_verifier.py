"""Frozen-candidate residual object rescoring; never changes nodes or relations."""
import torch
from torch import nn


class NodeVerifier(nn.Module):
    """Shared point/message MLPs with symmetric aggregation of five neighbors.

    The independent control uses the same parameters, feeding its own features
    to the message MLP. Physical coordinates only determine neighbors/offsets;
    decoder embeddings are the image-derived input. No target inputs are used.
    """
    def __init__(self, token_width, hidden=64, context=True, k=5):
        super().__init__()
        self.context, self.k = context, k
        self.point = nn.Sequential(nn.LayerNorm(token_width), nn.Linear(token_width, hidden), nn.ReLU())
        self.message = nn.Sequential(nn.Linear(2 * hidden + 9, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(2 * hidden + 5, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, tokens, logits, coordinates, physical):
        h = self.point(tokens)
        base = torch.cat((logits, coordinates), -1)
        if self.context and h.shape[-2] > 1:
            distance = torch.cdist(physical, physical)
            distance = distance.masked_fill(torch.eye(h.shape[-2], device=h.device, dtype=torch.bool), float('inf'))
            # Include all boundary ties with fractional weight so coincident or
            # exactly equidistant queries do not introduce index dependence.
            cutoff = distance.topk(min(self.k, h.shape[-2]-1), largest=False).values[..., -1:]
            strict, ties = distance < cutoff, distance == cutoff
            weights = strict.to(h.dtype) + ties * ((min(self.k, h.shape[-2]-1)-strict.sum(-1, keepdim=True)) / ties.sum(-1, keepdim=True).clamp_min(1))
            # Evaluate only actual neighbors (normally exactly k per node).
            b, i, j = (weights > 0).nonzero(as_tuple=True)
            offset = (physical[b, j] - physical[b, i]) / 10.
            message = self.message(torch.cat((h[b, i], h[b, j], base[b, j], offset, offset.norm(dim=-1, keepdim=True)), -1))
            pooled = torch.zeros_like(h)
            pooled.index_put_((b, i), message * weights[b, i, j, None], accumulate=True)
            pooled = pooled / weights.sum(-1, keepdim=True).clamp_min(1)
        else:
            pooled = self.message(torch.cat((h, h, base, torch.zeros_like(base[..., :4])), -1))
        correction = self.output(torch.cat((h, pooled, base), -1)).squeeze(-1)
        return logits[..., 1] - logits[..., 0] + correction
