"""Unit/shape checks for the heatmap-anchored RelationFormer (30 Oct 2026).

CPU-only and tiny (batch 1, 64^3, reduced widths) so they run on a login node.
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models.anchor_matcher import AnchorMatcher, physical_assignment
from models.anchored_relationformer import (AnchoredRelationFormer, HeatmapProposer, anchor_sine_embedding,
                                            anchors_to_reference_points, encode_memory, proposal_seed)
from models.deformable_transformer import build_def_detr_transformer
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from models.seresnet import build_seresnet

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'experiments/anchored_rf_20261030/plan.json'


def tiny_config(queries=16):
    cfg = copy.deepcopy(json.loads(PLAN.read_text())['config'])
    cfg['model']['decoder'].update(hidden_dim=24, attention_heads=6, feedforward_dim=32, encoder_layers=1,
                                   decoder_layers=2, object_queries=queries, use_cuda_extension=False)
    cfg['model']['encoder']['depths'] = [1, 1, 1, 1]
    return cfg


class FakeUNet(nn.Module):
    """Returns a fixed 5-channel probe output (heat logit, D/H/W offsets, centerline)."""

    def __init__(self, output):
        super().__init__()
        self.register_buffer('output', output)

    def forward(self, images):
        return self.output.expand(images.shape[0], -1, -1, -1, -1)


def tiny_model(queries=16):
    cfg = tiny_config(queries)
    torch.manual_seed(0)
    proposer = HeatmapProposer(queries, 64, autocast=False)
    model = AnchoredRelationFormer(build_seresnet(cfg), build_def_detr_transformer(cfg), cfg, proposer)
    return cfg, model


def straight_geometry(nodes, edges, metric):
    t = torch.linspace(0, 1, 8)[None, :, None]
    s = torch.linspace(0, 1, 64)[None, :, None]
    a, b = nodes[edges[:, 0]][:, None], nodes[edges[:, 1]][:, None]
    return dict(controls=a + t * (b - a), curves=a + s * (b - a), metric=metric)


class AnchoredRelationFormerTests(unittest.TestCase):
    def test_reference_points_follow_decoder_axis_order_and_voxel_centres(self):
        # Value field = its own (d,h,w) voxel coordinate, sampled exactly as MSDeformAttn does.
        axis = torch.arange(64.)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), 0)  # [3,D,H,W]
        value = grid.flatten(1).T[None, :, None, :]  # [N=1, S, heads=1, 3]
        shapes = torch.tensor([[64, 64, 64]])
        voxels = torch.rand(1, 50, 3, generator=torch.Generator().manual_seed(1)) * 63
        voxels[0, :10] = voxels[0, :10].round()
        reference = anchors_to_reference_points(voxels / 64, 64)
        locations = reference[:, :, None, None, None, :]  # [N,Lq,heads,levels,points,3]
        weights = torch.ones(1, 50, 1, 1, 1)
        sampled = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
        torch.testing.assert_close(sampled[0], voxels[0], atol=1e-4, rtol=0)
        # The conversion is load-bearing: DHW order or no half-voxel shift samples elsewhere.
        wrong_order = ms_deform_attn_core_pytorch(value, shapes, reference.flip(-1)[:, :, None, None, None], weights)
        no_shift = ms_deform_attn_core_pytorch(value, shapes, (voxels / 64).flip(-1)[:, :, None, None, None], weights)
        self.assertGreater(float((wrong_order[0] - voxels[0]).abs().max()), 1)
        self.assertGreater(float((no_shift[0, 10:] - voxels[0, 10:]).abs().max()), .4)

    def test_decoder_receives_anchor_reference_points(self):
        cfg, model = tiny_model()
        anchors = torch.rand(1, 16, 3)
        tgt, pos, reference = model.query_inputs(anchors, torch.rand(1, 16))
        self.assertEqual(tuple(tgt.shape), (1, 18, 24))
        self.assertEqual(tuple(pos.shape), (1, 18, 24))
        torch.testing.assert_close(reference[:, :16], ((anchors * 64 + .5) / 64)[..., [2, 1, 0]], rtol=0, atol=0)
        captured = {}
        layer = model.decoder.decoder.layers[0]
        def capture(module, args):
            captured.setdefault('ref', args[1])

        hook = layer.cross_attn1.register_forward_pre_hook(capture)
        with torch.no_grad():
            model(torch.randn(1, 1, 64, 64, 64), proposals=(anchors, torch.rand(1, 16)))
        hook.remove()
        # relation_attention=True: the first cross-attention sees all but the last token.
        torch.testing.assert_close(captured['ref'][:, :16, 0], reference[:, :16], rtol=0, atol=0)

    def test_encode_memory_reproduces_unchanged_transformer(self):
        cfg = tiny_config()
        torch.manual_seed(3)
        transformer = build_def_detr_transformer(cfg).eval()
        src = torch.randn(2, 24, 2, 2, 2)
        mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
        pos = torch.randn(2, 24, 2, 2, 2)
        query = torch.randn(18, 24)
        with torch.no_grad():
            expected = transformer(src, mask, query, pos)
            memory, shapes, starts, ratios, pos_flat, mask_flat = encode_memory(transformer, src, mask, pos)
            q = query[None].expand(2, -1, -1)
            actual, _ = transformer.decoder(torch.zeros_like(q), transformer.reference_points(q).sigmoid(), memory,
                                            shapes, starts, ratios, pos_flat, q, mask_flat)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_proposer_peaks_offsets_padding_and_freezing(self):
        output = torch.full((1, 5, 64, 64, 64), -8.)
        output[0, 1:4] = 0.
        peaks = [((10, 20, 30), 3., (.25, -.25, .1)), ((40, 5, 60), 1., (0., .3, -.4)), ((63, 63, 0), 2., (.4, .4, -.3))]
        for (d, h, w), logit, offset in peaks:
            output[0, 0, d, h, w] = logit
            output[0, 1:4, d, h, w] = torch.tensor(offset)
        output[0, 0, 11, 20, 30] = 2.9  # adjacent lower value is not a local maximum
        proposer = HeatmapProposer(8, 64, autocast=False)
        proposer.unet = FakeUNet(output)
        anchors, scores, counts = proposer(torch.zeros(2, 1, 64, 64, 64), [proposal_seed('a'), proposal_seed('b')])
        # The flat background (-8 logit plateau) is itself a set of tied local maxima: rank by score, raster ties.
        self.assertEqual(int(counts[0]), 8)
        order = [(10, 20, 30, .25, -.25, .1), (63, 63, 0, .4, .4, -.3), (40, 5, 60, 0., .3, -.4)]
        for k, (d, h, w, *o) in enumerate(order):
            expected = (torch.tensor([d, h, w], dtype=torch.float32) + torch.tensor(o)).clamp(-.5, 63.5) / 64
            torch.testing.assert_close(anchors[0, k], expected, atol=1e-6, rtol=0)
        self.assertTrue(bool((scores[0, :3].diff() <= 0).all()))
        # Padding: only when fewer local maxima than queries; seeded per sample, score 0.
        lone = torch.full((1, 5, 64, 64, 64), 0.)
        lone[0, 0] = torch.linspace(-20, -10, 64 ** 3).view(64, 64, 64)  # strictly increasing: single peak at end
        lone[0, 0, 10, 20, 30] = 5.
        proposer.unet = FakeUNet(lone)
        a1, s1, c1 = proposer(torch.zeros(1, 1, 64, 64, 64), [11])
        a2, s2, c2 = proposer(torch.zeros(1, 1, 64, 64, 64), [11])
        a3, _, _ = proposer(torch.zeros(1, 1, 64, 64, 64), [12])
        self.assertEqual(int(c1[0]), 2)
        torch.testing.assert_close(a1, a2, rtol=0, atol=0)
        self.assertTrue(torch.equal(a1[0, :2], a3[0, :2]) and not torch.equal(a1[0, 2:], a3[0, 2:]))
        self.assertTrue(bool((s1[0, 2:] == 0).all()) and bool((s1[0, :2] > 0).all()))
        self.assertTrue(bool(((a1 >= -.5 / 64) & (a1 <= 63.5 / 64)).all()))
        # Frozen: no trainable parameters and eval mode even inside a training parent.
        parent = nn.Sequential(proposer)
        parent.train()
        self.assertFalse(proposer.training)
        self.assertFalse(any(p.requires_grad for p in HeatmapProposer(4).parameters()))

    def test_anchor_matcher_is_evaluator_rule_deterministic_and_order_invariant(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from diagnose_branch_relations import match_nodes
        generator = torch.Generator().manual_seed(7)
        metric = torch.diag(torch.tensor([30., 28., 33.]))
        gt = torch.rand(25, 3, generator=generator)
        anchors = torch.cat([gt[:18] + torch.randn(18, 3, generator=generator) * .03,
                             torch.rand(46, 3, generator=generator)])[None]
        targets = dict(nodes=[gt], spline=[dict(metric=metric)])
        matcher = AnchorMatcher(2.0)
        first = matcher(dict(anchors=anchors), targets)[0]
        second = matcher(dict(anchors=anchors.clone()), targets)[0]
        self.assertTrue(torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]))
        reference = match_nodes(anchors[0].double().numpy() @ metric.double().numpy().T,
                                gt.double().numpy() @ metric.double().numpy().T, 2.0)
        self.assertEqual(dict(zip(first[0].tolist(), first[1].tolist())), reference)
        self.assertGreater(len(reference), 5)
        permutation = torch.randperm(64, generator=generator)
        permuted = matcher(dict(anchors=anchors[:, permutation]), targets)[0]
        self.assertEqual({(int(permutation[a]), int(b)) for a, b in zip(*permuted)}, set(zip(*map(torch.Tensor.tolist, first))))
        # Uses anchors only: predictions/logits are ignored.
        noisy = matcher(dict(anchors=anchors, pred_nodes=torch.rand(1, 64, 6), pred_logits=torch.randn(1, 64, 2)), targets)[0]
        self.assertTrue(torch.equal(noisy[0], first[0]))
        rows, cols = physical_assignment(np.zeros((0, 3)), np.ones((2, 3)), 2.)
        self.assertEqual((len(rows), len(cols)), (0, 0))

    def test_sine_embedding_shape(self):
        embedding = anchor_sine_embedding(torch.rand(2, 5, 3), 552)
        self.assertEqual(tuple(embedding.shape), (2, 5, 552))
        self.assertTrue(bool(torch.isfinite(embedding).all()))

    def test_gradients_reach_new_query_modules_not_proposer(self):
        from scripts.train_ixi_transfer import TransferCriterion
        cfg, model = tiny_model()
        model.proposer.unet = FakeUNet(torch.randn(1, 5, 64, 64, 64, generator=torch.Generator().manual_seed(2)))
        model.train()
        image = torch.randn(1, 1, 64, 64, 64)
        anchors, scores, _ = model.proposer(image, [proposal_seed('x')])
        nodes = (anchors[0, :6] + .01).clamp(0, 1)
        edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        metric = torch.diag(torch.tensor([32., 32., 32.]))
        targets = dict(nodes=[nodes], edges=[edges], domains=torch.ones(1, dtype=torch.long),
                       spline=[straight_geometry(nodes, edges, metric)])
        criterion = TransferCriterion(cfg, AnchorMatcher(2.0), model.relation_embed, spline_head=model.spline_head,
                                      route='graph_only')
        tokens, predictions, _ = model(image, [proposal_seed('x')])
        torch.testing.assert_close(predictions['anchors'], anchors, rtol=0, atol=0)
        torch.testing.assert_close(predictions['pred_nodes'][..., :3], anchors, rtol=0, atol=0)
        assignment = criterion.matcher(predictions, targets)[0]
        self.assertEqual(len(assignment[0]), 6)
        losses = criterion(tokens, predictions, targets)
        losses['total'].backward()
        for prefix in ['anchor_pos.', 'content_embed', 'score_proj.', 'offset_embed.', 'query_embed.',
                       'decoder.', 'encoder.', 'class_embed.', 'relation_embed.', 'spline_head.']:
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
            self.assertTrue(grads, prefix)
            self.assertGreater(float(sum(g.abs().sum() for g in grads)), 0, prefix)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.proposer.parameters()))
        self.assertFalse(model.proposer.training)
        # Residual is bounded to +-3 voxels around the anchor.
        with torch.no_grad():
            model.offset_embed.layers[-1].bias.fill_(50.)
            _, moved, _ = model(image, [proposal_seed('x')])
        delta = (moved['pred_nodes'][..., :3] - anchors).abs()
        self.assertLessEqual(float(delta.max()), 3 / 64 + 1e-6)
        self.assertGreater(float(delta.max()), 2.9 / 64)


if __name__ == '__main__':
    unittest.main()
