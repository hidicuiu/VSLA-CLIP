"""CPU-only tests of RNG isolation used by paired sampling experiments."""

from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from datasets.dynamic_dfgs import refresh_dynamic_dfgs_graph
from utils.reproducibility import preserve_rng_state, reset_training_seed


def draws():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


class PairedReproducibilityTests(unittest.TestCase):
    def test_reset_repeats_draws_and_disables_benchmark(self):
        reset_training_seed(2025)
        expected = draws()
        draws()
        reset_training_seed(2025)
        self.assertEqual(draws(), expected)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_preserves_all_cpu_streams_on_exception(self):
        reset_training_seed(2026)
        expected = draws()
        reset_training_seed(2026)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            with preserve_rng_state():
                for _ in range(12):
                    draws()
                raise RuntimeError('probe')
        self.assertEqual(draws(), expected)

    def test_legacy_disabled_context_consumes_rng(self):
        reset_training_seed(1234)
        draws()
        expected = draws()
        reset_training_seed(1234)
        with preserve_rng_state(False):
            draws()
        self.assertEqual(draws(), expected)

    def test_cuda_states_are_restored_without_launching_cuda(self):
        saved = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        with patch('torch.cuda.is_initialized', return_value=True), \
                patch('torch.cuda.get_rng_state_all', return_value=saved), \
                patch('torch.cuda.set_rng_state_all') as restore:
            with preserve_rng_state():
                draws()
            restore.assert_called_once_with(saved)

    def test_graph_wrapper_isolates_rng_and_retains_legacy_behavior(self):
        for enabled in (True, False):
            current = cfg.clone()
            current.DATALOADER.DYNAMIC_DFGS_PRESERVE_RNG_STATE = enabled
            reset_training_seed(1234)
            first, second = draws(), draws()
            reset_training_seed(1234)
            with patch('datasets.dynamic_dfgs._refresh_dynamic_dfgs_graph',
                       side_effect=lambda *args: draws()) as probe:
                self.assertEqual(refresh_dynamic_dfgs_graph(current, None, None, 1, None), first)
                probe.assert_called_once()
            self.assertEqual(draws(), first if enabled else second)


if __name__ == '__main__':
    unittest.main()
