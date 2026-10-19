"""Diagnose alignment between DFGS-T text neighbors and visual hard identities.

This script is read-only with respect to datasets and checkpoints. It extracts
identity text features from the Stage-1 checkpoint, identity visual centroids
from the Stage-2 best checkpoint, and writes alignment statistics to OUTPUT.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.data import DataLoader

from config import cfg
from dataset_transformer import spatial_transforms as ST
from dataset_transformer import temporal_transforms as TT
from datasets import data_manager
from datasets.video_loader import VideoDataset
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--stage2_checkpoint", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_state(model, path):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)


def cosine_ranking(features):
    features = F.normalize(features.float(), dim=1)
    similarity = features @ features.t()
    similarity.fill_diagonal_(-float("inf"))
    ranking = similarity.argsort(dim=1, descending=True)
    return similarity, ranking


def extract_text_features(model, num_classes, device, chunk_size=128):
    model.eval()
    result = []
    with torch.no_grad():
        for begin in range(0, num_classes, chunk_size):
            labels = torch.arange(begin, min(begin + chunk_size, num_classes), device=device)
            result.append(model(label=labels, get_text=True).float().cpu())
    return torch.cat(result, dim=0)


def extract_visual_centroids(model, dataset, cfg, args, device):
    transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
    ])
    temporal = TT.TemporalRestrictedBeginCrop(size=cfg.DATALOADER.SEQ_LEN)
    loader = DataLoader(
        VideoDataset(dataset.train, spatial_transform=transform, temporal_transform=temporal),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    sums = None
    counts = torch.zeros(dataset.num_train_pids, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for step, (clips, pids, _) in enumerate(loader, start=1):
            clips = clips.to(device, non_blocking=True)
            with amp.autocast(enabled=True):
                features = model(clips).float().cpu()
            if sums is None:
                sums = torch.zeros(dataset.num_train_pids, features.shape[1], dtype=torch.float32)
            for feature, pid in zip(features, pids):
                pid = int(pid)
                sums[pid] += feature
                counts[pid] += 1
            if step % 50 == 0 or step == len(loader):
                print("Visual extraction: {}/{} batches".format(step, len(loader)), flush=True)
    if (counts == 0).any():
        raise RuntimeError("Missing visual tracklets for IDs: {}".format(
            torch.where(counts == 0)[0].tolist()))
    return sums / counts.unsqueeze(1), counts


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    device = torch.device("cuda")

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    os.makedirs(args.output, exist_ok=True)

    dataset = data_manager.init_dataset(name=cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    model = make_model(cfg, num_class=num_classes, camera_num=dataset.num_camera, view_num=0)

    print("Loading Stage-1 checkpoint for text features")
    load_state(model, args.stage1_checkpoint)
    model.to(device)
    text_features = extract_text_features(model, num_classes, device)
    text_similarity, text_ranking = cosine_ranking(text_features)

    print("Loading Stage-2 checkpoint for visual features")
    load_state(model, args.stage2_checkpoint)
    visual_features, tracklet_counts = extract_visual_centroids(model, dataset, cfg, args, device)
    visual_similarity, visual_ranking = cosine_ranking(visual_features)

    with open(args.graph, "r") as handle:
        raw_graph = json.load(handle)
    graph = {int(pid): [int(value) for value in values] for pid, values in raw_graph.items()}
    if set(graph) != set(range(num_classes)):
        raise ValueError("Graph IDs do not match dataset IDs 0..{}".format(num_classes - 1))

    ks = sorted(set([1, 5, args.topk]))
    overlap_by_k = {k: [] for k in ks}
    graph_visual_similarities = []
    graph_visual_ranks = []
    per_identity = []
    indegree = Counter()

    inverse_visual_rank = torch.empty_like(visual_ranking)
    rank_values = torch.arange(visual_ranking.shape[1]).unsqueeze(0).expand_as(visual_ranking)
    inverse_visual_rank.scatter_(1, visual_ranking, rank_values)

    for pid in range(num_classes):
        text_neighbors = text_ranking[pid, :args.topk].tolist()
        visual_neighbors = visual_ranking[pid, :args.topk].tolist()
        graph_neighbors = graph[pid]
        for neighbor in graph_neighbors:
            indegree[neighbor] += 1
            graph_visual_similarities.append(float(visual_similarity[pid, neighbor]))
            graph_visual_ranks.append(int(inverse_visual_rank[pid, neighbor]) + 1)
        row = {
            "pid": pid,
            "tracklets": int(tracklet_counts[pid]),
            "graph_neighbors": " ".join(map(str, graph_neighbors)),
            "text_top10": " ".join(map(str, text_neighbors)),
            "visual_top10": " ".join(map(str, visual_neighbors)),
            "graph_visual_mean_similarity": float(np.mean([
                float(visual_similarity[pid, neighbor]) for neighbor in graph_neighbors
            ])),
            "graph_visual_median_rank": float(np.median([
                int(inverse_visual_rank[pid, neighbor]) + 1 for neighbor in graph_neighbors
            ])),
        }
        for k in ks:
            overlap = len(set(text_ranking[pid, :k].tolist()).intersection(
                visual_ranking[pid, :k].tolist())) / float(k)
            overlap_by_k[k].append(overlap)
            row["top{}_overlap".format(k)] = overlap
        per_identity.append(row)

    edge_count = sum(len(values) for values in graph.values())
    random_visual_similarities = []
    for _ in range(edge_count):
        pid = random.randrange(num_classes)
        neighbor = random.randrange(num_classes - 1)
        if neighbor >= pid:
            neighbor += 1
        random_visual_similarities.append(float(visual_similarity[pid, neighbor]))

    indegrees = np.asarray([indegree[pid] for pid in range(num_classes)], dtype=np.float64)
    graph_sims = np.asarray(graph_visual_similarities, dtype=np.float64)
    random_sims = np.asarray(random_visual_similarities, dtype=np.float64)
    ranks = np.asarray(graph_visual_ranks, dtype=np.float64)
    summary = {
        "dataset": str(cfg.DATASETS.NAMES),
        "num_identities": num_classes,
        "num_tracklets": int(tracklet_counts.sum()),
        "topk": args.topk,
        "mean_overlap": {"top{}".format(k): float(np.mean(overlap_by_k[k])) for k in ks},
        "median_overlap": {"top{}".format(k): float(np.median(overlap_by_k[k])) for k in ks},
        "graph_edge_count": edge_count,
        "graph_visual_similarity": {
            "mean": float(graph_sims.mean()),
            "median": float(np.median(graph_sims)),
            "p25": percentile(graph_sims, 25),
            "p75": percentile(graph_sims, 75),
        },
        "random_visual_similarity": {
            "mean": float(random_sims.mean()),
            "median": float(np.median(random_sims)),
            "p25": percentile(random_sims, 25),
            "p75": percentile(random_sims, 75),
        },
        "graph_neighbor_visual_rank": {
            "mean": float(ranks.mean()),
            "median": float(np.median(ranks)),
            "p25": percentile(ranks, 25),
            "p75": percentile(ranks, 75),
            "fraction_in_visual_top10": float(np.mean(ranks <= 10)),
            "fraction_in_visual_top50": float(np.mean(ranks <= 50)),
        },
        "graph_indegree": {
            "mean": float(indegrees.mean()),
            "std": float(indegrees.std()),
            "min": int(indegrees.min()),
            "max": int(indegrees.max()),
            "zero_indegree_ids": int(np.sum(indegrees == 0)),
        },
        "checkpoints": {
            "stage1": args.stage1_checkpoint,
            "stage2": args.stage2_checkpoint,
            "graph": args.graph,
        },
    }

    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    csv_path = os.path.join(args.output, "per_identity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_identity[0].keys()))
        writer.writeheader()
        writer.writerows(per_identity)
    torch.save({
        "text_features": text_features,
        "visual_features": visual_features,
        "tracklet_counts": tracklet_counts,
    }, os.path.join(args.output, "identity_features.pt"))

    print(json.dumps(summary, indent=2))
    print("Saved summary to {}".format(summary_path))
    print("Saved per-identity details to {}".format(csv_path))


if __name__ == "__main__":
    main()
