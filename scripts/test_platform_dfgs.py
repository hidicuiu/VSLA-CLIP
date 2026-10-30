"""Checks for platform filtering, identity parsing, and paired budgets."""

import argparse
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import cfg
from datasets.data_manager import AGVPReID
from datasets.dynamic_dfgs import (platform_labels_from_cameras,
                                   refresh_dynamic_dfgs_graph,
                                   stratified_multicamera_exclusions,
                                   stratified_platform_exclusions)
from datasets.make_video_dataloader import _make_stage2_sampler

TEST_TEMP_ROOT = None

class PlatformExclusionTests(unittest.TestCase):
    def test_refresh_keeps_real_cameras_and_saves_platform_metadata(self):
        current = cfg.clone()
        current.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES = True
        current.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM = True
        current.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM = [0, 0, 0, 0, 1, 1]
        current.DATALOADER.DYNAMIC_DFGS_NUM_WORKERS = 0
        current.MODEL.PBP_CAMERA = True
        fixture_parent = Path(TEST_TEMP_ROOT or tempfile.gettempdir()).resolve()
        output = fixture_parent / ("ag_vpreid_test_" + uuid.uuid4().hex)
        output.mkdir()
        current.OUTPUT_DIR = str(output)
        data = [((), pid, camera) for pid in range(40) for camera in (0, 4)]
        sampler = _make_stage2_sampler(self._dynamic_config(current), data)
        feature_rows = []
        for index, (_, pid, camera) in enumerate(data):
            feature = torch.tensor([pid + 1.0, camera + 1.0, 1.0, 0.1])
            feature_rows.append((feature, feature, feature, pid, camera, index, 12))

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cameras = []

            def forward(self, clips, cam_label=None, view_label=None):
                self.cameras.extend(cam_label.tolist())
                return clips.float()

        model = RecordingModel()
        try:
            with patch("datasets.dynamic_dfgs.DynamicGraphTrackletDataset", return_value=feature_rows), \
                    patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor), \
                    patch("torch.cuda.amp.autocast", lambda **kwargs: nullcontext()):
                graph = refresh_dynamic_dfgs_graph(
                    current, model, sampler, 1, logging.getLogger("test.platform"))
            self.assertEqual(set(model.cameras), {0, 4})
            self.assertTrue(model.training)
            self.assertEqual(sampler.neighbor_graph, graph)
            self.assertEqual(len(graph), 40)
            self.assertTrue(all(len(neighbors) == 10 and pid not in neighbors
                                for pid, neighbors in graph.items()))
            with open(output / "dynamic_dfgs_graphs" / "epoch_001.json", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertTrue(payload["metadata"]["stratified_platform"])
            self.assertEqual(payload["metadata"]["camera_to_platform"], [0, 0, 0, 0, 1, 1])
            self.assertIn("thresholds_by_platform_and_length", payload["filter_diagnostics"])
        finally:
            if output.resolve().parent != fixture_parent or not output.name.startswith("ag_vpreid_test_"):
                raise RuntimeError("Unexpected test output directory")
            shutil.rmtree(str(output))

    @staticmethod
    def _dynamic_config(current):
        current.DATALOADER.SAMPLER = "mixed_dfgs_baseline_k4"
        current.DATALOADER.DYNAMIC_DFGS_ENABLED = True
        current.DATALOADER.PID_SAMPLES_PER_EPOCH = 4
        return current

    def test_two_low_ground_cameras_do_not_imply_two_low_platforms(self):
        cameras = [0, 1, 4] * 3
        pids = [0] * 3 + [1] * 3 + [2] * 3
        scores = [0.1, 0.1, 0.9, 0.2, 0.2, 0.1, 0.9, 0.9, 0.8]
        camera_excluded, _ = stratified_multicamera_exclusions(
            scores, pids, cameras, [12] * 9, 3, 0.5)
        platform_excluded, diagnostics = stratified_platform_exclusions(
            scores, pids, cameras, [12] * 9, 3, 0.5, [0, 0, 0, 0, 1, 1])
        self.assertTrue(camera_excluded[0])
        self.assertFalse(platform_excluded[0])
        self.assertEqual(cameras, [0, 1, 4] * 3)
        self.assertIn("thresholds_by_platform_and_length", diagnostics)
        self.assertNotIn("thresholds_by_camera_and_length", diagnostics)

    def test_both_platforms_low_and_partial_coverage(self):
        excluded, diagnostics = stratified_platform_exclusions(
            [0.1, 0.1, 0.8, 0.8, 0.9, 0.9, 0.01],
            [0, 0, 1, 1, 2, 2, 3], [0, 4, 1, 5, 2, 4, 3],
            [12] * 7, 4, 0.5, [0, 0, 0, 0, 1, 1])
        self.assertEqual(excluded.tolist(), [True, False, False, False])
        self.assertEqual(diagnostics["single_platform_identity_count"], 1)

    def test_invalid_or_single_platform_mapping_fails(self):
        for cameras, mapping in (([0, 5], []), ([0, 6], [0, 0, 0, 0, 1, 1]),
                                 ([0, 1], [0] * 6), ([0, 1], [0, 2])):
            with self.subTest(cameras=cameras, mapping=mapping):
                with self.assertRaises(ValueError):
                    platform_labels_from_cameras(cameras, mapping)

    def test_nonfinite_reliability_is_not_silently_ranked(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            stratified_platform_exclusions(
                [float("nan"), 0.9], [0, 0], [0, 4], [12, 12], 1, 0.2,
                [0, 0, 0, 0, 1, 1])


class AGVPReIDReaderTests(unittest.TestCase):
    def setUp(self):
        # Use normal inherited permissions: TemporaryDirectory's private mode
        # is inaccessible to the restricted Windows execution token here.
        self.fixture_parent = Path(TEST_TEMP_ROOT or tempfile.gettempdir()).resolve()
        self.root = self.fixture_parent / ("ag_vpreid_test_" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(self.clean_fixture)
        self.add("train", "0", "tracklet_0", 10, 0, [10, 2])
        self.add("train", "0", "tracklet_1", 10, 4)
        self.add("train", "1", "tracklet_2", 20, 1)
        self.add("train", "1", "tracklet_3", 20, 5)
        self.add("case2_groud_to_aerial/query", "100", "tracklet_0", 30, 0)
        self.add("case2_groud_to_aerial/query", "100", "tracklet_1", 40, 1)
        self.add("case2_groud_to_aerial/gallery", "100", "tracklet_0", 30, 4)
        self.add("case2_groud_to_aerial/gallery", "100", "tracklet_1", 40, 5)
        self.add("case2_groud_to_aerial/gallery", "101", "tracklet_2", 50, 5)

    def clean_fixture(self):
        if (self.root.resolve().parent != self.fixture_parent or
                not self.root.name.startswith("ag_vpreid_test_")):
            raise RuntimeError("Refusing to clean a fixture outside the test temporary directory")
        shutil.rmtree(str(self.root))

    def add(self, split, folder, tracklet, pid, camera, frames=(1,)):
        path = self.root / split / folder / tracklet
        path.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            (path / "P{}T2310300A2C{}E1K0F{}.jpg".format(pid, camera, frame)).touch()

    def test_annotated_pids_and_camera_ids_are_preserved_for_evaluation(self):
        dataset = AGVPReID(str(self.root), verbose=False)
        self.assertEqual({pid for _, pid, _ in dataset.train}, {0, 1})
        self.assertEqual({pid for _, pid, _ in dataset.query}, {100})
        self.assertEqual({pid for _, pid, _ in dataset.gallery}, {100, 101})
        self.assertEqual({camera for _, _, camera in dataset.gallery}, {4, 5})
        self.assertTrue(dataset.train[0][0][0].endswith("F2.jpg"))
        self.assertEqual(dataset.audit["training_identities_with_both_platforms"], 2)
        self.assertEqual(dataset.audit["gallery_distractor_identities"], 1)

    def test_direction_is_checked_against_filenames(self):
        with self.assertRaisesRegex(ValueError, "expected 1"):
            AGVPReID(str(self.root), direction="a2g",
                     split_dir="case2_groud_to_aerial", verbose=False)

    def test_training_identity_leakage_fails(self):
        self.add("train", "100", "tracklet_4", 30, 4)
        with self.assertRaisesRegex(ValueError, "overlap"):
            AGVPReID(str(self.root), verbose=False)

    def test_filename_p_code_does_not_override_annotated_folder(self):
        self.add("train", "2", "tracklet_4", 30, 4)
        dataset = AGVPReID(str(self.root), verbose=False)
        self.assertEqual(dataset.num_train_pids, 3)
        self.assertEqual({pid for _, pid, _ in dataset.query}, {100})

    def test_mixed_camera_tracklet_fails(self):
        self.add("train", "0", "tracklet_0", 10, 1)
        with self.assertRaisesRegex(ValueError, "Mixed P/C"):
            AGVPReID(str(self.root), verbose=False)


class PairedConfigurationTests(unittest.TestCase):
    def test_equal_budget_and_model_settings(self):
        root = Path(__file__).resolve().parents[1]
        paths = [
            "reproduce/agvpreid_g2a_vifi_pbp_p4k4_seed1234_lr2e4.yml",
            "innovation/agvpreid_g2a_E4_camera_pbp_p4k4_seed1234_lr2e4.yml",
            "innovation/agvpreid_g2a_E4_platform_pbp_p4k4_seed1234_lr2e4.yml",
        ]
        configs = []
        for path in paths:
            current = cfg.clone()
            current.merge_from_file(str(root / "configs" / "adapter" / path))
            configs.append(current)
        baseline, camera, platform = configs
        for current in configs:
            self.assertEqual(current.DATASETS, baseline.DATASETS)
            self.assertEqual(current.SOLVER, baseline.SOLVER)
            self.assertEqual(current.DATALOADER.PID_SAMPLES_PER_EPOCH, 4)
            self.assertTrue(current.MODEL.PBP_CAMERA)
            self.assertEqual(current.MODEL.VIFI_WEIGHT, baseline.MODEL.VIFI_WEIGHT)
        for current in (camera, platform):
            self.assertEqual(current.MODEL.STAGE1_WEIGHT,
                             baseline.OUTPUT_DIR + "/ViT-B-16_stage1_120.pth")
        self.assertTrue(camera.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)
        self.assertTrue(platform.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM)
        self.assertFalse(platform.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)

        # Uneven numbers of real tracklets must not change the matched budget.
        data = [((), pid, pid % 6) for pid in range(9) for _ in range(pid + 1)]
        samplers = [_make_stage2_sampler(current, data) for current in configs]
        self.assertEqual([len(sampler) for sampler in samplers], [32, 32, 32])
        for sampler in samplers:
            random.seed(1234)
            np.random.seed(1234)
            indices = list(sampler)
            self.assertEqual(len(indices), 32)
            for start in range(0, len(indices), 16):
                pids = [data[index][1] for index in indices[start:start + 16]]
                self.assertEqual(sorted(pids.count(pid) for pid in set(pids)), [4] * 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--temp-dir")
    options, remaining = parser.parse_known_args()
    TEST_TEMP_ROOT = options.temp_dir
    unittest.main(argv=[sys.argv[0]] + remaining)
