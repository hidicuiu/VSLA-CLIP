"""Build platform-label-free reliable hard-positive tracklet rankings."""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import cfg
from datasets import data_manager
from datasets.reliable_positive import (
    SCHEMA_VERSION,
    build_partner_rankings,
    compute_tracklet_statistics,
)
from datasets.video_loader import get_default_video_loader
from dataset_transformer import spatial_transforms as ST
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def deterministic_frame_paths(img_paths, seq_len):
    """Select evenly distributed frames, repeating deterministically if needed."""
    if not img_paths:
        raise ValueError("A tracklet contains no frames")
    positions = np.linspace(0, len(img_paths) - 1, num=seq_len)
    return [img_paths[int(round(position))] for position in positions]


class TrackletFeatureDataset(Dataset):
    def __init__(self, tracklets, seq_len, spatial_transform):
        self.tracklets = tracklets
        self.seq_len = seq_len
        self.spatial_transform = spatial_transform
        self.loader = get_default_video_loader()

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, index):
        img_paths, pid, _ = self.tracklets[index]
        selected = deterministic_frame_paths(img_paths, self.seq_len)
        frames = self.loader(selected)
        self.spatial_transform.randomize_parameters()
        frames = [self.spatial_transform(frame) for frame in frames]
        clip = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
        return clip, int(pid), index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build reliable hard-positive tracklet rankings")
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='DataLoader workers; 0 is the safe default on Windows')
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the RHP graph")

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
    ])
    feature_dataset = TrackletFeatureDataset(
        dataset.train, cfg.DATALOADER.SEQ_LEN, transform)
    loader = DataLoader(
        feature_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = make_model(
        cfg, num_class=dataset.num_train_pids,
        camera_num=dataset.num_camera, view_num=0)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    all_frame_features = []
    all_pids = []
    all_indices = []
    with torch.no_grad():
        for clips, pids, indices in loader:
            frame_features = model(
                clips.cuda(non_blocking=True), get_frame_image=True)
            all_frame_features.append(frame_features.float().cpu())
            all_pids.extend(int(pid) for pid in pids.tolist())
            all_indices.extend(int(index) for index in indices.tolist())

    expected_indices = list(range(len(dataset.train)))
    if all_indices != expected_indices:
        raise RuntimeError("Tracklet feature order does not match the training split")
    frame_features = torch.cat(all_frame_features, dim=0)
    centers, reliabilities = compute_tracklet_statistics(frame_features)
    partners = build_partner_rankings(centers, reliabilities, all_pids)

    payload = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset": str(cfg.DATASETS.NAMES),
            "dataset_root": os.path.abspath(str(cfg.DATASETS.ROOT_DIR)),
            "num_tracklets": len(dataset.train),
            "num_identities": dataset.num_train_pids,
            "sequence_length": cfg.DATALOADER.SEQ_LEN,
            "checkpoint": os.path.abspath(args.checkpoint),
            "feature_dimension": int(centers.shape[1]),
            "score": "(1-cos(center_i,center_j))*sqrt(reliability_i*reliability_j)",
        },
        "reliability": {
            str(index): float(value)
            for index, value in enumerate(reliabilities.tolist())
        },
        "partners": partners,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as output_file:
        json.dump(payload, output_file, indent=2)
    print("Saved RHP graph for {} tracklets and {} identities to {}".format(
        len(dataset.train), dataset.num_train_pids, args.output))


if __name__ == '__main__':
    main()
