"""Run: python -m unittest discover -s tests -p test_cross_platform_set.py"""
import unittest

import torch
from torch.nn import functional as F

from loss.cross_platform_set import CrossPlatformSetLoss, validate_set_loss_config
from model.multihead_frame_aggregation import MultiHeadFrameAggregation
from config import cfg


class SetContrastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x = torch.randn(8, 2, 12, requires_grad=True)
        self.y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        self.p = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def test_reference_multiple_positives(self):
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        actual, _ = CrossPlatformSetLoss()(self.x, labels, self.p)
        z = F.normalize(self.x, dim=-1)
        losses = []
        for i in range(8):
            candidates = [j for j in range(8) if self.p[i] != self.p[j]]
            scores = []
            for j in candidates:
                same = (z[i, 0] @ z[j, 0] + z[i, 1] @ z[j, 1]) / 2
                swap = (z[i, 0] @ z[j, 1] + z[i, 1] @ z[j, 0]) / 2
                scores.append(torch.maximum(same, swap) / 0.1)
            scores = torch.stack(scores)
            positive = torch.tensor([labels[i] == labels[j] for j in candidates])
            losses.append(-(scores - scores.logsumexp(0))[positive].mean())
        torch.testing.assert_close(actual, torch.stack(losses).mean())
        actual.backward()
        self.assertTrue(torch.isfinite(self.x.grad).all())

    def test_independent_head_permutation(self):
        criterion = CrossPlatformSetLoss()
        original, _ = criterion(self.x, self.y, self.p)
        permuted = self.x.clone()
        permuted[::2] = permuted[::2].flip(1)
        changed, _ = criterion(permuted, self.y, self.p)
        torch.testing.assert_close(original, changed)

    def test_empty_and_mixed_valid_rows(self):
        for labels, platforms in ((self.y, torch.zeros(8, dtype=torch.long)),
                                  (torch.arange(8), self.p),
                                  (torch.zeros(8, dtype=torch.long), self.p),
                                  (torch.tensor([0, 0, 1, 2, 3, 4, 5, 6]), self.p)):
            x = self.x.detach().clone().requires_grad_()
            loss, _ = CrossPlatformSetLoss()(x, labels, platforms)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_aggregation_gradient_and_unchanged_retrieval(self):
        module = MultiHeadFrameAggregation(branch_dims=(6, 6, 4), hidden_dim=5)
        branches = [torch.randn(8, 3, width, requires_grad=True) for width in (6, 6, 4)]
        output, aux = module(*branches)
        for actual, branch in zip(output, branches):
            torch.testing.assert_close(actual, branch.mean(1))
        self.assertEqual(aux['set_features'].shape, (8, 2, 10))
        loss, _ = CrossPlatformSetLoss()(aux['set_features'], self.y, self.p)
        loss.backward()
        gradient = module.score[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_configs(self):
        for name in ('g2a', 'agvpreid'):
            for suffix in ('set', 'set_fixed', 'supcon_fused'):
                settings = cfg.clone()
                settings.merge_from_file('configs/adapter/innovation/' + name +
                                         '_E4_m3_frame_dual_mean_graph_' + suffix + '.yml')
                validate_set_loss_config(settings)
        self.assertEqual(cfg.MODEL.CROSS_PLATFORM_SET_WEIGHT, 0)

    def test_fused_reference_and_duplicate_head_equivalence(self):
        features = self.x[:, :1]
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        actual, _ = CrossPlatformSetLoss(matching='fused')(features, labels, self.p)
        z = F.normalize(features[:, 0], dim=-1)
        logits = z @ z.t() / 0.1
        candidates = self.p[:, None] != self.p[None, :]
        positives = candidates & (labels[:, None] == labels[None, :])
        logp = logits - logits.masked_fill(~candidates, -torch.inf).logsumexp(1)[:, None]
        expected = -(logp.masked_fill(~positives, 0).sum(1) / positives.sum(1)).mean()
        torch.testing.assert_close(actual, expected)
        for mode in ('fixed', 'optimal'):
            duplicated, _ = CrossPlatformSetLoss(matching=mode)(features.expand(-1, 2, -1), labels, self.p)
            torch.testing.assert_close(actual, duplicated)
        actual.backward()
        self.assertTrue(torch.isfinite(self.x.grad).all())

    def test_fixed_and_optimal_differ_on_swapped_positive(self):
        x = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]],
                          [[-1., 0.], [0., -1.]], [[0., -1.], [-1., 0.]]])
        y = torch.tensor([0, 0, 1, 1])
        p = torch.tensor([0, 1, 0, 1])
        fixed, _ = CrossPlatformSetLoss(matching='fixed')(x, y, p)
        optimal, _ = CrossPlatformSetLoss(matching='optimal')(x, y, p)
        self.assertLess(optimal.item(), fixed.item())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast(self):
        x = self.x.detach().cuda().half().requires_grad_()
        with torch.cuda.amp.autocast():
            loss, _ = CrossPlatformSetLoss()(x, self.y.cuda(), self.p.cuda())
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
