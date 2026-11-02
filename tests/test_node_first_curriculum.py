import copy
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from training.node_first_curriculum import edge_multiplier, scheduled_total, ScheduledGraphObjective


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(3, 4)
        self.node = nn.Linear(4, 1)
        self.relation_embed = nn.Linear(4, 1)

    def forward(self, x):
        z = self.shared(x)
        return z, self.node(z), None


class ToyCriterion(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.relation = model.relation_embed
        self.enabled_losses = ['nodes', 'edges']
        self.weights = {'nodes': 2., 'edges': 6.}

    def forward(self, z, nodes, targets):
        # Mimic candidate RNG consumed even when its loss is inactive.
        torch.randperm(4)
        losses = dict(nodes=(nodes - targets).square().mean(),
                      edges=(self.relation(z) - targets).square().mean())
        losses['total'] = sum(losses[k] * self.weights[k] for k in self.enabled_losses)
        return losses


class NodeFirstTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = ToyModel()
        self.criterion = ToyCriterion(self.model)
        self.x, self.y = torch.randn(5, 3), torch.ones(5, 1)

    def test_schedule_boundaries(self):
        self.assertEqual(edge_multiplier(1, 100, 'node_first'), 0)
        self.assertEqual(edge_multiplier(10, 100, 'node_first'), 0)
        self.assertAlmostEqual(edge_multiplier(15, 100, 'node_first'), .5)
        self.assertEqual(edge_multiplier(20, 100, 'node_first'), 1)
        self.assertEqual(edge_multiplier(100, 100, 'node_first'), 1)
        self.assertEqual(edge_multiplier(1, 100, 'joint'), 1)

    def test_zero_edge_has_no_optimizer_state_or_decay(self):
        objective = ScheduledGraphObjective(self.model, self.criterion, 'node_first', 100)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01, weight_decay=.1)
        before = copy.deepcopy(self.model.state_dict())
        total, logs = objective(self.x, self.y, 1)
        total.backward()
        self.assertTrue(all(p.grad is None for p in self.model.relation_embed.parameters()))
        self.assertGreater(float(self.model.shared.weight.grad.norm()), 0)
        optimizer.step()
        for k, p in self.model.relation_embed.named_parameters():
            torch.testing.assert_close(p, before['relation_embed.' + k], rtol=0, atol=0)
            self.assertNotIn(p, optimizer.state)
        optimizer.zero_grad(set_to_none=True)
        total, logs = objective(self.x, self.y, 11)
        self.assertGreater(float(logs['edge_multiplier']), 0)
        total.backward()
        optimizer.step()
        for p in self.model.relation_embed.parameters():
            self.assertEqual(int(optimizer.state[p]['step']), 1)

    def test_joint_total_and_gradients_have_exact_parity(self):
        tokens, pred, _ = self.model(self.x)
        losses = self.criterion(tokens, pred, self.y)
        self.assertIs(scheduled_total(losses, self.criterion, 1), losses['total'])
        original = torch.autograd.grad(losses['total'], list(self.model.parameters()), retain_graph=True)
        wrapped = torch.autograd.grad(scheduled_total(losses, self.criterion, 1), list(self.model.parameters()))
        for a, b in zip(original, wrapped):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_raw_edge_rng_and_metrics_do_not_change_backward(self):
        states = []
        for arm in ['joint', 'node_first']:
            torch.manual_seed(20)
            _, logs = ScheduledGraphObjective(self.model, self.criterion, arm, 100)(self.x, self.y, 1)
            self.assertTrue(all(not v.requires_grad for v in logs.values()))
            states.append(torch.get_rng_state())
        torch.testing.assert_close(*states, rtol=0, atol=0)

    def test_ddp_unused_relation_head_can_become_active(self):
        if not torch.distributed.is_available():
            self.skipTest('distributed unavailable')
        # One-rank Gloo still exercises reducer unused-parameter discovery and
        # the inactive->active graph transition without requesting a GPU.
        with tempfile.TemporaryDirectory() as directory:
            torch.distributed.init_process_group('gloo', rank=0, world_size=1,
                init_method='file://' + str(Path(directory) / 'rendezvous'))
            try:
                objective = ScheduledGraphObjective(self.model, self.criterion, 'node_first', 100)
                ddp = nn.parallel.DistributedDataParallel(objective, find_unused_parameters=True)
                for step in [1, 11, 20]:
                    ddp.zero_grad(set_to_none=True)
                    total, _ = ddp(self.x, self.y, step)
                    total.backward()
                    if step == 1:
                        self.assertIsNone(self.model.relation_embed.weight.grad)
                    else:
                        self.assertGreater(float(self.model.relation_embed.weight.grad.norm()), 0)
            finally:
                torch.distributed.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
