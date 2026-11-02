"""Focused alignment, symmetry, initialization and gradient isolation checks."""
import unittest
import torch
from models.image_relation_adapter import ARMS, ImageRelationAdapter, sample_early, connection_points, residual_logits
from scripts.ixi_image_relation_learning import pair_indices


class ImageRelationAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20); torch.set_num_threads(1)
        self.tokens = torch.randn(258, 8)
        self.nodes = torch.rand(256, 3); self.nodes[4] = self.nodes[3]
        self.confidence = torch.rand(256)
        self.metric = torch.diag(torch.tensor([16., 24., 32.]))
        self.features = torch.randn(64, 8, 8, 8)
        self.pairs = torch.tensor([[0, 1], [2, 5], [3, 4]])

    def call(self, model, pairs=None, features=None):
        return model(self.tokens, self.nodes, self.confidence, self.metric,
                     self.features if features is None else features, self.pairs if pairs is None else pairs, (16, 16, 16))

    def test_stride_two_alignment_and_axis_order(self):
        d, h, w = torch.meshgrid(torch.arange(8.), torch.arange(9.), torch.arange(10.), indexing='ij')
        feature = (d+10*h+100*w)[None]
        point = torch.tensor([[[4/16, 6/18, 8/20]]])
        value = sample_early(feature, point, (16, 18, 20))
        torch.testing.assert_close(value.flatten(), torch.tensor([432.]), atol=.001, rtol=0)

    def test_shared_capacity_and_exact_zero_residual(self):
        states = []
        for arm in ARMS:
            torch.manual_seed(42); model = ImageRelationAdapter(8, arm=arm)
            zero = self.call(model); self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
            original = torch.randn(3, 2); self.assertTrue(torch.equal(residual_logits(original, zero), original))
            states.append(model.state_dict())
        for state in states[1:]:
            self.assertEqual(set(state), set(states[0]))
            self.assertTrue(all(torch.equal(v, states[0][k]) for k, v in state.items()))

    def test_symmetry_after_nonzero_residual_and_frozen_inputs(self):
        for arm in ARMS:
            model = ImageRelationAdapter(8, arm=arm)
            with torch.no_grad(): model.network[-1].weight.normal_(std=.01)
            forward = self.call(model); reverse = self.call(model, self.pairs.flip(1))
            torch.testing.assert_close(forward, reverse, atol=1e-6, rtol=1e-6)
            if arm == 'token_geometry':
                torch.testing.assert_close(forward, self.call(model, features=self.features+100))
            self.features.requires_grad_(True); self.tokens.requires_grad_(True)
            self.call(model).square().sum().backward()
            self.assertIsNone(self.features.grad); self.assertIsNone(self.tokens.grad)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if arm != 'token_geometry': self.assertGreater(float(model.image_projection[0].weight.grad.abs().sum()), 0.)

    def test_corridor_symmetry_physical_width_and_pair_order(self):
        a = connection_points(self.nodes, self.pairs, self.metric)
        b = connection_points(self.nodes, self.pairs.flip(1), self.metric)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
        physical = a @ self.metric.T
        # Opposite side-centres of a square cross section are 8 mm apart.
        torch.testing.assert_close((physical[:, :, 1]-physical[:, :, 7]).norm(dim=-1), torch.full((3, 16), 8.), atol=1e-5, rtol=1e-5)
        pairs = torch.combinations(torch.arange(256))
        torch.testing.assert_close(pair_indices(pairs), torch.arange(32640))
        torch.testing.assert_close(pair_indices(pairs.flip(1)), torch.arange(32640))


if __name__ == '__main__': unittest.main()
