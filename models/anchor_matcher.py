"""Physical one-to-one assignment of heatmap anchors to ground-truth nodes.

Replaces the Hungarian matcher for the anchored RelationFormer. The assignment
uses only proposal anchors and ground truth (never predicted positions or
scores), so it is fixed for a given input and is the same rule an evaluator
applies (``scripts.diagnose_branch_relations.match_nodes``): Hungarian on
physical millimetre distances with an out-of-tolerance penalty, accepting pairs
at most ``tolerance_mm`` apart. All other queries are no-object.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def physical_assignment(prediction_mm: np.ndarray, truth_mm: np.ndarray, tolerance: float):
    """Same rule as ``match_nodes``: returns sorted (prediction, truth) index arrays."""
    prediction_mm = np.asarray(prediction_mm, dtype=np.float64).reshape(-1, 3)
    truth_mm = np.asarray(truth_mm, dtype=np.float64).reshape(-1, 3)
    if not len(prediction_mm) or not len(truth_mm):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    distances = np.linalg.norm(prediction_mm[:, None] - truth_mm[None], axis=-1)
    penalty = (min(distances.shape) + 1) * (tolerance + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= tolerance, distances, penalty))
    keep = distances[rows, cols] <= tolerance
    return rows[keep].astype(np.int64), cols[keep].astype(np.int64)


class AnchorMatcher(nn.Module):
    """Assign GT nodes to anchors within ``tolerance_mm`` (physical, one-to-one)."""

    requires_structure = False

    def __init__(self, tolerance_mm: float = 2.0):
        super().__init__()
        if tolerance_mm <= 0:
            raise ValueError('tolerance_mm must be positive')
        self.tolerance_mm = float(tolerance_mm)

    @torch.no_grad()
    def forward(self, outputs: Mapping, targets: Mapping, *, predicted_structure=None, candidate_indices=None):
        del predicted_structure, candidate_indices
        anchors = outputs['anchors']
        nodes = targets['nodes']
        if 'spline' not in targets:
            raise ValueError('AnchorMatcher needs per-sample physical metrics in targets["spline"]')
        if len(nodes) != anchors.shape[0]:
            raise ValueError("targets['nodes'] must contain one tensor per sample")
        assignments = []
        for sample in range(anchors.shape[0]):
            metric = targets['spline'][sample]['metric'].detach().double().cpu().numpy()
            source = anchors[sample, :, :3].detach().double().cpu().numpy() @ metric.T
            truth = nodes[sample][:, :3].detach().double().cpu().numpy() @ metric.T
            rows, cols = physical_assignment(source, truth, self.tolerance_mm)
            assignments.append((torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)))
        return assignments


__all__ = ['AnchorMatcher', 'physical_assignment']
