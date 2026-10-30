"""Compare paired E4 and strict-random exclusions against checkpoint rankings.

This is a retrospective diagnostic: saved graph JSONs contain exact historical
filtered-candidate counts, but not the historical centroid features or complete
rankings. This script extracts centroids from the supplied checkpoints and
applies every saved exclusion set to those checkpoint rankings.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from dataset_transformer import spatial_transforms as ST
from datasets import data_manager
from datasets.dynamic_dfgs import DynamicGraphTrackletDataset
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def load_cfg(path):
    cfg.defrost()
    cfg.merge_from_file(path)
    cfg.freeze()
    return cfg


def extract_centroids(config_path, checkpoint_path, dataset):
    local_cfg = load_cfg(config_path)
    model = make_model(
        local_cfg,
        num_class=dataset.num_train_pids,
        camera_num=dataset.num_camera,
        view_num=0,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    transform = ST.Compose([
        ST.Scale(local_cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(local_cfg.INPUT.PIXEL_MEAN, local_cfg.INPUT.PIXEL_STD),
    ])
    graph_dataset = DynamicGraphTrackletDataset(
        dataset.train,
        local_cfg.DATALOADER.SEQ_LEN,
        transform,
        return_temporal_views=True,
    )
    loader = DataLoader(
        graph_dataset,
        batch_size=local_cfg.DATALOADER.DYNAMIC_DFGS_BATCH_SIZE,
        shuffle=False,
        num_workers=local_cfg.DATALOADER.DYNAMIC_DFGS_NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    sums = None
    counts = torch.zeros(dataset.num_train_pids, dtype=torch.long)
    indices_seen = []
    with torch.no_grad():
        for step, (clips, _first, _second, pids, camids, indices, _frame_counts) in enumerate(loader, 1):
            clips = clips.cuda(non_blocking=True)
            camids = camids.cuda(non_blocking=True)
            camera_labels = camids if local_cfg.MODEL.PBP_CAMERA else None
            view_labels = camids if local_cfg.MODEL.PBP_VIEW else None
            with torch.cuda.amp.autocast(enabled=True):
                features = model(
                    clips, cam_label=camera_labels, view_label=view_labels)
            features = F.normalize(features.float(), dim=1).cpu()
            if sums is None:
                sums = torch.zeros(
                    dataset.num_train_pids, features.shape[1], dtype=torch.float32)
            for feature, pid in zip(features, pids):
                sums[int(pid)] += feature
                counts[int(pid)] += 1
            indices_seen.extend(int(index) for index in indices.tolist())
            if step % 50 == 0 or step == len(loader):
                print("{} extraction: {}/{} batches".format(
                    os.path.basename(checkpoint_path), step, len(loader)), flush=True)
    if indices_seen != list(range(len(dataset.train))):
        raise RuntimeError("Tracklet order mismatch during centroid extraction")
    if (counts == 0).any():
        raise RuntimeError("Some identities have no tracklets")
    centroids = F.normalize(sums / counts.unsqueeze(1), dim=1)
    del model
    torch.cuda.empty_cache()
    return centroids


def summarize_set(similarity, ranking, excluded_ids, skip, pool_size):
    num_ids = similarity.shape[0]
    excluded = torch.zeros(num_ids, dtype=torch.bool)
    excluded[torch.tensor(excluded_ids, dtype=torch.long)] = True
    candidate_pool = ranking[:, skip:skip + pool_size]
    removed_mask = excluded[candidate_pool]
    removed_counts = removed_mask.sum(dim=1).float()
    removed_ranks = []
    removed_sims = []
    if removed_mask.any():
        rows = torch.arange(num_ids).unsqueeze(1).expand_as(candidate_pool)
        removed_rows = rows[removed_mask]
        removed_ids = candidate_pool[removed_mask]
        inverse = torch.empty_like(ranking)
        inverse.scatter_(1, ranking, torch.arange(num_ids).expand_as(ranking))
        removed_ranks = (inverse[removed_rows, removed_ids] + 1).float().numpy()
        removed_sims = similarity[removed_rows, removed_ids].float().numpy()
    pool_sims = similarity.gather(1, candidate_pool).float().numpy().ravel()
    return {
        "excluded_identity_count": len(excluded_ids),
        "mean_excluded_candidates_per_anchor": float(removed_counts.mean()),
        "median_excluded_candidates_per_anchor": float(removed_counts.median()),
        "anchors_with_any_excluded_candidate_fraction": float((removed_counts > 0).float().mean()),
        "removed_candidate_count": int(removed_counts.sum()),
        "removed_candidate_original_rank_mean": float(np.mean(removed_ranks)) if len(removed_ranks) else None,
        "removed_candidate_original_rank_median": float(np.median(removed_ranks)) if len(removed_ranks) else None,
        "removed_candidate_similarity_mean": float(np.mean(removed_sims)) if len(removed_sims) else None,
        "removed_candidate_similarity_median": float(np.median(removed_sims)) if len(removed_sims) else None,
        "all_candidate_pool_similarity_mean": float(np.mean(pool_sims)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4-config", required=True)
    parser.add_argument("--random-config", required=True)
    parser.add_argument("--e4-checkpoint", required=True)
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--e4-graphs", required=True)
    parser.add_argument("--random-graphs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint feature extraction")
    os.makedirs(args.output, exist_ok=True)
    local_cfg = load_cfg(args.e4_config)
    dataset = data_manager.init_dataset(
        name=local_cfg.DATASETS.NAMES, root=local_cfg.DATASETS.ROOT_DIR)

    checkpoint_specs = [
        ("E4_checkpoint120", args.e4_config, args.e4_checkpoint),
        ("random_checkpoint120", args.random_config, args.random_checkpoint),
    ]
    feature_sets = {}
    for name, config_path, checkpoint_path in checkpoint_specs:
        print("Extracting centroids for {}".format(name), flush=True)
        feature_sets[name] = extract_centroids(config_path, checkpoint_path, dataset)
    torch.save(
        feature_sets,
        os.path.join(args.output, "epoch120_identity_centroids.pt"),
    )

    epochs = sorted(
        int(name[len("epoch_"):-len(".json")])
        for name in os.listdir(args.e4_graphs)
        if name.startswith("epoch_") and name.endswith(".json"))
    result_rows = []
    summaries = []
    for model_name, centroids in feature_sets.items():
        similarity = centroids @ centroids.t()
        similarity.fill_diagonal_(-float("inf"))
        ranking = similarity.argsort(dim=1, descending=True)
        for epoch in epochs:
            graph_paths = {
                "E4": os.path.join(args.e4_graphs, "epoch_{:03d}.json".format(epoch)),
                "strict_random": os.path.join(args.random_graphs, "epoch_{:03d}.json".format(epoch)),
            }
            for method, graph_path in graph_paths.items():
                with open(graph_path, "r", encoding="utf-8") as graph_file:
                    payload = json.load(graph_file)
                metrics = summarize_set(
                    similarity,
                    ranking,
                    payload["excluded_identity_ids"],
                    int(payload["metadata"].get("skip", 0)),
                    int(payload["metadata"]["candidate_pool"]),
                )
                historical = payload.get("filter_diagnostics", {})
                row = {
                    "checkpoint": model_name,
                    "graph_refresh_epoch": epoch,
                    "exclusion_set": method,
                    "historical_saved_mean_filtered_candidates": historical.get("mean_filtered_candidates"),
                    **metrics,
                }
                result_rows.append(row)
                summaries.append(row)

    csv_path = os.path.join(args.output, "paired_filter_rank_similarity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(result_rows[0].keys()))
        writer.writeheader()
        writer.writerows(result_rows)
    json_path = os.path.join(args.output, "paired_filter_rank_similarity.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump({
            "note": "Retrospective rankings from final epoch-120 checkpoints; these are not the historical rankings used at each graph refresh.",
            "dataset": str(local_cfg.DATASETS.NAMES),
            "num_identities": dataset.num_train_pids,
            "num_tracklets": len(dataset.train),
            "skip": int(local_cfg.DATALOADER.DYNAMIC_DFGS_SKIP),
            "candidate_pool": int(local_cfg.DATALOADER.DYNAMIC_DFGS_CANDIDATE_POOL),
            "rows": summaries,
        }, json_file, indent=2)
    print("Saved {} and {}".format(csv_path, json_path), flush=True)


if __name__ == "__main__":
    main()
