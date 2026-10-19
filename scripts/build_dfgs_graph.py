"""Build DFGS text, visual, and text-visual identity graphs offline."""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from dataset_transformer import spatial_transforms as ST
from dataset_transformer import temporal_transforms as TT
from datasets import data_manager
from datasets.video_loader import VideoDataset
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DFGS-T, DFGS-V, and DFGS-TV identity graphs")
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--stage1_checkpoint', required=True)
    parser.add_argument('--stage2_checkpoint', required=True)
    parser.add_argument('--text_output', required=True)
    parser.add_argument('--visual_output', required=True)
    parser.add_argument('--text_visual_output', required=True)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--neighbors', type=int, default=10)
    parser.add_argument('--text_weight', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()


def load_state(model, checkpoint):
    state = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)


def extract_text_features(model, num_classes, chunk_size, device):
    model.eval()
    features = []
    with torch.no_grad():
        for begin in range(0, num_classes, chunk_size):
            labels = torch.arange(
                begin, min(begin + chunk_size, num_classes), device=device)
            features.append(model(label=labels, get_text=True).float().cpu())
    return F.normalize(torch.cat(features, dim=0), dim=1)


def extract_visual_centroids(model, dataset, batch_size, num_workers,
                             sequence_length, device):
    transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
    ])
    temporal = TT.TemporalRestrictedBeginCrop(size=sequence_length)
    loader = DataLoader(
        VideoDataset(
            dataset.train, spatial_transform=transform,
            temporal_transform=temporal),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False)

    sums = None
    counts = torch.zeros(dataset.num_train_pids, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for step, (clips, pids, _) in enumerate(loader, start=1):
            clips = clips.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=True):
                features = model(clips).float().cpu()
            if sums is None:
                sums = torch.zeros(
                    dataset.num_train_pids, features.shape[1],
                    dtype=torch.float32)
            for feature, pid in zip(features, pids):
                pid = int(pid)
                sums[pid] += feature
                counts[pid] += 1
            if step % 50 == 0 or step == len(loader):
                print('Visual extraction: {}/{} batches'.format(
                    step, len(loader)), flush=True)

    if sums is None:
        raise RuntimeError('The training split contains no tracklets')
    missing = torch.where(counts == 0)[0].tolist()
    if missing:
        raise RuntimeError('Missing visual tracklets for IDs: {}'.format(missing))
    centroids = sums / counts.unsqueeze(1)
    return F.normalize(centroids, dim=1), counts


def cosine_similarity(features):
    features = F.normalize(features.float(), dim=1)
    similarity = features @ features.t()
    similarity.fill_diagonal_(-float('inf'))
    return similarity


def graph_from_similarity(similarity, skip, neighbors):
    num_classes = similarity.shape[0]
    if skip < 0 or neighbors <= 0:
        raise ValueError('skip must be >= 0 and neighbors must be > 0')
    if skip + neighbors > num_classes - 1:
        raise ValueError(
            'skip + neighbors must not exceed the number of other identities')
    ranked = similarity.argsort(dim=1, descending=True)
    end = skip + neighbors
    return {
        str(pid): ranked[pid, skip:end].tolist()
        for pid in range(num_classes)
    }


def save_graph(graph, path):
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as output_file:
        json.dump(graph, output_file, indent=2)
    print('Saved {}-identity graph to {}'.format(len(graph), path))


def main():
    args = parse_args()
    if not 0.0 <= args.text_weight <= 1.0:
        raise ValueError('text_weight must be in [0, 1]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to build DFGS graphs')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    dataset = data_manager.init_dataset(
        name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    device = torch.device('cuda')
    model = make_model(
        cfg, num_class=dataset.num_train_pids,
        camera_num=dataset.num_camera, view_num=0)

    print('Loading Stage-1 checkpoint for text features')
    load_state(model, args.stage1_checkpoint)
    model.to(device)
    text_features = extract_text_features(
        model, dataset.num_train_pids, args.chunk_size, device)
    text_similarity = cosine_similarity(text_features)

    print('Loading Stage-2 checkpoint for visual identity centroids')
    load_state(model, args.stage2_checkpoint)
    model.to(device)
    visual_features, counts = extract_visual_centroids(
        model, dataset, args.batch_size, args.num_workers,
        cfg.DATALOADER.SEQ_LEN, device)
    visual_similarity = cosine_similarity(visual_features)

    text_visual_similarity = (
        args.text_weight * text_similarity
        + (1.0 - args.text_weight) * visual_similarity)
    text_visual_similarity.fill_diagonal_(-float('inf'))

    save_graph(
        graph_from_similarity(text_similarity, args.skip, args.neighbors),
        args.text_output)
    save_graph(
        graph_from_similarity(visual_similarity, args.skip, args.neighbors),
        args.visual_output)
    save_graph(
        graph_from_similarity(
            text_visual_similarity, args.skip, args.neighbors),
        args.text_visual_output)
    print('Tracklets per identity: min={}, max={}, mean={:.2f}'.format(
        int(counts.min()), int(counts.max()), float(counts.float().mean())))
    print('Text/visual fusion weights: {:.2f}/{:.2f}'.format(
        args.text_weight, 1.0 - args.text_weight))


if __name__ == '__main__':
    main()
