"""Small, data-free checks for the optional multicamera reliability rule."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.dynamic_dfgs import (stratified_multicamera_exclusions,
                                   stratified_two_camera_exclusions)
from config import cfg


class MulticameraExclusionTests(unittest.TestCase):
    def test_two_camera_matches_existing_rule(self):
        scores = [0.1, 0.2, 0.5, 0.6, 0.9] * 2
        pids = list(range(5)) * 2
        cameras = [0] * 5 + [1] * 5
        lengths = [12] * 10
        old, _ = stratified_two_camera_exclusions(
            scores, pids, cameras, lengths, 5, 0.2)
        new, diagnostics = stratified_multicamera_exclusions(
            scores, pids, cameras, lengths, 5, 0.2)
        self.assertEqual(old.tolist(), new.tolist())
        self.assertEqual(diagnostics["num_cameras"], 2)

    def test_three_cameras_and_partial_coverage(self):
        # ID 0 is low in 2/3 cameras; ID 1 in 3/3; ID 5 has only one view.
        scores = ([0.1, 0.2, 0.8, 0.9, 0.95, 0.99] +
                  [0.1, 0.2, 0.8, 0.9, 0.95] +
                  [0.8, 0.1, 0.2, 0.9, 0.95])
        pids = list(range(6)) + list(range(5)) + list(range(5))
        cameras = [0] * 6 + [1] * 5 + [2] * 5
        lengths = [12] * len(scores)
        excluded, diagnostics = stratified_multicamera_exclusions(
            scores, pids, cameras, lengths, 6, 0.3)
        self.assertTrue(excluded[0])
        self.assertTrue(excluded[1])
        self.assertFalse(excluded[2])
        self.assertFalse(excluded[5])
        self.assertEqual(diagnostics["identity_camera_coverage_counts"][1], 1)

    def test_camera_and_length_grouping_axes_are_independent(self):
        # Two cameras crossed with two occupied length bins create four E4
        # threshold groups. Disabling either axis reduces this to two; disabling
        # both uses one global threshold. Identity-level camera consensus stays
        # unchanged in every case.
        scores = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        pids = [0, 1, 2, 3, 0, 1, 2, 3]
        cameras = [0] * 4 + [1] * 4
        lengths = [4, 4, 12, 12] * 2
        cases = [
            (True, True, 4),
            (True, False, 2),
            (False, True, 2),
            (False, False, 1),
        ]
        for by_camera, by_length, expected_groups in cases:
            excluded, diagnostics = stratified_multicamera_exclusions(
                scores, pids, cameras, lengths, 4, 0.25,
                group_by_camera=by_camera,
                group_by_length=by_length)
            self.assertEqual(diagnostics["threshold_group_count"], expected_groups)
            self.assertEqual(diagnostics["group_by_camera"], by_camera)
            self.assertEqual(diagnostics["group_by_length"], by_length)
            self.assertEqual(diagnostics["num_cameras"], 2)
            self.assertEqual(len(excluded), 4)

    def test_g2a_multicamera_config_uses_paired_stage_one(self):
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "adapter", "innovation")
        current = cfg.clone()
        current.merge_from_file(os.path.join(
            config_dir,
            "g2a_E4_pbp_multicamera_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml"))
        self.assertTrue(current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)
        self.assertFalse(current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA)
        self.assertEqual(current.DATALOADER.DFGS_PROBABILITY, 0.25)
        self.assertEqual(current.DATALOADER.DYNAMIC_DFGS_UPDATE_PERIOD, 10)
        self.assertTrue(current.MODEL.STAGE1_WEIGHT)

    def test_g2a_grouping_ablation_configs_are_a_matched_factorial(self):
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "configs", "adapter", "innovation")
        names = {
            "global": "g2a_E4_group_global_pbp_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml",
            "camera": "g2a_E4_group_camera_only_pbp_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml",
            "length": "g2a_E4_group_length_only_pbp_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml",
            "camera_length": "g2a_E4_pbp_multicamera_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml",
        }
        configs = {}
        for name, filename in names.items():
            current = cfg.clone()
            current.merge_from_file(os.path.join(config_dir, filename))
            configs[name] = current
        reference = configs["camera_length"]
        expected = {
            "global": (False, False),
            "camera": (True, False),
            "length": (False, True),
            "camera_length": (True, True),
        }
        for name, current in configs.items():
            self.assertEqual(current.DATASETS, reference.DATASETS)
            self.assertEqual(current.SOLVER, reference.SOLVER)
            self.assertEqual(current.INPUT, reference.INPUT)
            self.assertEqual(current.MODEL, reference.MODEL)
            self.assertTrue(current.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES)
            self.assertTrue(current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)
            self.assertEqual(current.DATALOADER.DYNAMIC_DFGS_GROUP_BY_CAMERA,
                             expected[name][0])
            self.assertEqual(current.DATALOADER.DYNAMIC_DFGS_GROUP_BY_LENGTH,
                             expected[name][1])

    def test_mars_pair_uses_same_model_and_dataset(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        baseline = cfg.clone()
        e4 = cfg.clone()
        baseline.merge_from_file(os.path.join(
            project_root, "configs", "adapter", "reproduce",
            "mars_vifi_pbp_p4k4_seed1234_lr2e4.yml"))
        e4.merge_from_file(os.path.join(
            project_root, "configs", "adapter", "innovation",
            "mars_E4_pbp_multicamera_dynamic_DFGS_V_q025_p4k4_seed1234_lr2e4.yml"))
        self.assertEqual(baseline.SOLVER, e4.SOLVER)
        self.assertEqual(baseline.INPUT, e4.INPUT)
        self.assertEqual(baseline.DATASETS, e4.DATASETS)
        self.assertEqual(baseline.DATASETS.NAMES, "mars")
        self.assertTrue(os.path.isdir(os.path.join(
            project_root, baseline.DATASETS.ROOT_DIR)))
        self.assertTrue(baseline.MODEL.PBP_CAMERA)
        self.assertTrue(e4.MODEL.PBP_CAMERA)
        self.assertEqual(baseline.MODEL.VIFI_WEIGHT, e4.MODEL.VIFI_WEIGHT)
        self.assertEqual(baseline.MODEL.USE_CFAA, e4.MODEL.USE_CFAA)
        self.assertEqual(baseline.MODEL.STAGE1_WEIGHT, "")
        self.assertEqual(e4.MODEL.STAGE1_WEIGHT,
                         os.path.join(baseline.OUTPUT_DIR,
                                      "ViT-B-16_stage1_120.pth").replace("\\", "/"))
        self.assertEqual(baseline.DATALOADER.SAMPLER, "softmax_triplet")
        self.assertEqual(e4.DATALOADER.SAMPLER, "mixed_dfgs_baseline_k4")
        self.assertTrue(e4.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)


if __name__ == "__main__":
    unittest.main()
