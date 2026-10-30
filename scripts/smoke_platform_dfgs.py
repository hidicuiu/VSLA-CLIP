"""Small GPU integration probe using AG images; no training or checkpoint writes.

Uses the configured pretrained model with fresh prompts, not a trained result.
Only a new output directory containing the probe graph/report is created.
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import cfg
from datasets.dynamic_dfgs import refresh_dynamic_dfgs_graph
from datasets.make_video_dataloader import make_dataloader
from datasets.sampler import IdentityBatchSamplerVideo
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(args.output_dir)
    cfg.merge_from_file(args.config)
    if not cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM:
        raise ValueError("Choose the platform configuration for this probe")
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.DYNAMIC_DFGS_NUM_WORKERS = 0
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    os.makedirs(args.output_dir)
    seed = cfg.SOLVER.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loaders = make_dataloader(cfg)
    train_loader, query_loader, gallery_loader = loaders[0], loaders[2], loaders[3]
    train_data = train_loader.dataset.dataset
    selected = []
    observed = set()
    # Forty IDs with one ground and one aerial tracklet each are enough to
    # exercise the 30-candidate pool and retain both platforms in the probe.
    for tracklet in train_data:
        _, pid, camera = tracklet
        key = (pid, cfg.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM[camera])
        if pid < 40 and key not in observed:
            selected.append(tracklet)
            observed.add(key)
    if len(observed) != 80:
        raise ValueError("The first forty identities must cover both platforms")
    sampler = IdentityBatchSamplerVideo(
        selected, batch_size=16, num_instances=4, samples_per_pid=4,
        baseline_replacement_sampling=True)
    model = make_model(cfg, num_class=loaders[6], camera_num=loaders[8], view_num=0).cuda()
    if model.cv_embed.shape[2] != 6:
        raise AssertionError("Platform grouping must retain six camera prompts")
    model.train()
    logging.basicConfig(level=logging.INFO)
    graph = refresh_dynamic_dfgs_graph(cfg, model, sampler, 1, logging.getLogger("platform.probe"))
    if not model.training:
        raise AssertionError("Graph extraction did not restore training mode")
    for pid, neighbors in graph.items():
        if len(neighbors) != 10 or pid in neighbors or len(set(neighbors)) != 10:
            raise AssertionError("Invalid graph neighbors")

    # Read one query/gallery batch through the normal evaluation camera path.
    model.eval()
    batch_shapes = {}
    with torch.no_grad():
        for name, loader in (("query", query_loader), ("gallery", gallery_loader)):
            clips, _, cameras = next(iter(loader))
            features = model(clips.cuda(), cam_label=cameras.cuda())
            if not bool(torch.isfinite(features).all()):
                raise AssertionError("Non-finite {} probe features".format(name))
            batch_shapes[name] = list(features.shape)
    report = {
        "probe_only_not_an_experiment": True,
        "graph_identities": len(graph),
        "graph_tracklets": len(selected),
        "neighbors_per_identity": 10,
        "camera_prompt_count": int(model.cv_embed.shape[2]),
        "full_training_batches_per_epoch": len(train_loader),
        "evaluation_feature_shapes": batch_shapes,
    }
    with open(os.path.join(args.output_dir, "smoke_summary.json"), "x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
