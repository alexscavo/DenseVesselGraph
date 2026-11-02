"""The original 2D transformer's small tensor interface."""
from dataclasses import dataclass
import torch


@dataclass
class NestedTensor:
    tensors: torch.Tensor
    mask: torch.Tensor

    def decompose(self):
        return self.tensors, self.mask


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(0, 1)
    return torch.log(x.clamp_min(eps) / (1-x).clamp_min(eps))
