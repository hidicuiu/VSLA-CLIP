"""CPU checks for dual-direction evaluation and preserving direction logs."""
import argparse
import importlib.util
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import cfg

spec = importlib.util.spec_from_file_location('agvpreid_test_entry', str(ROOT / 'test.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
TEMP_ROOT = None


class EvaluationTests(unittest.TestCase):
    def config(self):
        current = cfg.clone()
        current.DATASETS.NAMES = 'ag_vpreid'
        current.DATASETS.AGVPREID_DIRECTION = 'g2a'
        current.DATASETS.AGVPREID_SPLIT_DIR = 'case2_groud_to_aerial'
        current.TEST.AGVPREID_DIRECTIONS = ['g2a', 'a2g']
        return current

    def test_opposite_split_is_discovered_and_original_config_is_unchanged(self):
        current = self.config()
        current.freeze()
        configurations = entry.evaluation_configurations(current)
        self.assertEqual([direction for direction, _ in configurations], ['g2a', 'a2g'])
        self.assertEqual(configurations[0][1].DATASETS.AGVPREID_SPLIT_DIR, 'case2_groud_to_aerial')
        self.assertEqual(configurations[1][1].DATASETS.AGVPREID_SPLIT_DIR, '')
        self.assertEqual(configurations[1][1].DATASETS.AGVPREID_DIRECTION, 'a2g')
        self.assertEqual(current.DATASETS.AGVPREID_DIRECTION, 'g2a')
        self.assertTrue(current.is_frozen())

    def test_single_direction_override_and_legacy_default(self):
        current = self.config()
        for directions in (['a2g'], []):
            current.TEST.AGVPREID_DIRECTIONS = directions
            expected = 'a2g' if directions else 'g2a'
            self.assertEqual([d for d, _ in entry.evaluation_configurations(current)], [expected])

    def test_other_datasets_keep_the_existing_single_test(self):
        current = self.config()
        current.DATASETS.NAMES = 'mars'
        self.assertEqual(entry.evaluation_configurations(current), [(None, current)])

    def test_bad_directions_are_rejected(self):
        current = self.config()
        for directions in (['both'], ['g2a', 'g2a'], ['a2g', 'bad']):
            current.TEST.AGVPREID_DIRECTIONS = directions
            with self.assertRaises(ValueError):
                entry.evaluation_configurations(current)

    def test_one_checkpoint_serves_both_directions_and_logs_are_appended(self):
        parent = Path(TEMP_ROOT or tempfile.gettempdir()).resolve()
        output = parent / ('ag_eval_test_' + uuid.uuid4().hex)
        output.mkdir()
        try:
            current = self.config()
            current.OUTPUT_DIR = str(output)
            current.TEST.WEIGHT = 'existing_checkpoint.pth'
            for direction in ('g2a', 'a2g'):
                (output / ('test_log_' + direction + '.txt')).write_text('existing result\n')
            model = Mock()
            loader_calls = []
            def make_loaders(config):
                direction = config.DATASETS.AGVPREID_DIRECTION
                loader_calls.append(direction)
                return (None, None, direction+'_query', direction+'_gallery', None, None, 1604, 1459, 6)
            with patch.object(entry, 'make_dataloader', side_effect=make_loaders), \
                    patch.object(entry, 'make_model', return_value=model) as make_model, \
                    patch.object(entry, 'test', return_value=(np.array([.75, .8]), .67, [1, 2])) as test, \
                    patch.dict(os.environ):
                results = entry.run_evaluation(current)
            self.assertEqual(loader_calls, ['g2a', 'a2g'])
            make_model.assert_called_once()
            model.load_param.assert_called_once_with('existing_checkpoint.pth')
            self.assertEqual(test.call_count, 2)
            self.assertEqual(set(results), {'g2a', 'a2g'})
            for direction in ('g2a', 'a2g'):
                content = (output / ('test_log_'+direction+'.txt')).read_text()
                self.assertTrue(content.startswith('existing result\n'))
                self.assertIn('Evaluation direction: '+direction.upper(), content)
                self.assertEqual(content.count('mAP: 67.00%'), 1)
                self.assertFalse(logging.getLogger('transreid.eval.'+direction).handlers)
        finally:
            if output.resolve().parent != parent or not output.name.startswith('ag_eval_test_'):
                raise RuntimeError('Unexpected fixture directory')
            shutil.rmtree(str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--temp-dir')
    options, remaining = parser.parse_known_args()
    TEMP_ROOT = options.temp_dir
    unittest.main(argv=[sys.argv[0]] + remaining)
