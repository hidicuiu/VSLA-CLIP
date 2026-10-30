"""Validate every paired E4 count and camera/length quota before training."""

from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import cfg
from datasets import data_manager
from datasets.dynamic_dfgs import (
    identity_camera_length_profiles,
    load_paired_reference_exclusions,
    sample_stratified_random_exclusion_mask,
)


CONFIG = "configs/adapter/innovation/g2a_E4_random_filter_strictmatched_pbp_p4k4_seed1234.yml"


def main():
    cfg.defrost()
    cfg.merge_from_file(CONFIG)
    cfg.freeze()
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    profiles = identity_camera_length_profiles(
        dataset.train, dataset.num_train_pids)
    reference_dir = cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_REFERENCE_DIR
    rows = []
    for epoch in range(1, cfg.SOLVER.STAGE2.MAX_EPOCHS + 1,
                       cfg.DATALOADER.DYNAMIC_DFGS_UPDATE_PERIOD):
        reference_ids, reference_path = load_paired_reference_exclusions(
            reference_dir, epoch, dataset.num_train_pids)
        seed = (int(cfg.SOLVER.SEED) * 1000003 + int(epoch)) % (2 ** 32)
        mask, profile_counts = sample_stratified_random_exclusion_mask(
            profiles, reference_ids, seed)
        assert int(mask.sum()) == len(reference_ids), (epoch, len(reference_ids), int(mask.sum()))
        expected = Counter(profiles[pid] for pid in reference_ids)
        actual = Counter(profiles[pid] for pid in mask.nonzero().flatten().tolist())
        assert expected == actual, (epoch, expected, actual)
        repeated, _ = sample_stratified_random_exclusion_mask(
            profiles, reference_ids, seed)
        assert (mask == repeated).all(), epoch
        rows.append((epoch, len(reference_ids), len(profile_counts)))
        print("epoch={:03d} excluded={} matched_profiles={} source={}".format(
            epoch, len(reference_ids), len(profile_counts), reference_path))
    print("PASS: {} refreshes have exact counts and camera/length profile quotas".format(len(rows)))


if __name__ == "__main__":
    main()
