"""Read-only feature audit for the MARS E4 epoch-120 checkpoint.

Run from the repository root. Outputs are written only to --output-dir.
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from config import cfg
from datasets.make_video_dataloader import make_dataloader
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def extract(model, loader, split):
    records = []
    features = []
    offset = 0
    for vids, pids, camids in loader:
        with torch.no_grad():
            batch_features = model(vids.cuda(), cam_label=camids.cuda()).detach().cpu()
        features.append(batch_features)
        for j in range(len(pids)):
            index = offset + j
            paths, pid, camid = loader.dataset.dataset[index]
            feature = batch_features[j]
            finite = bool(torch.isfinite(feature).all())
            records.append({
                "split": split,
                "index": index,
                "pid": int(pid),
                "camid": int(camid),
                "tracklet_first_frame": str(paths[0]),
                "tracklet_frames": len(paths),
                "finite": finite,
                "nan_elements": int(torch.isnan(feature).sum()),
                "inf_elements": int(torch.isinf(feature).sum()),
                "norm": float(torch.linalg.vector_norm(feature.float())) if finite else None,
            })
        offset += len(pids)
    return torch.cat(features), records


def audit(model, query_loader, gallery_loader, checkpoint):
    model.load_param(checkpoint)
    model.cuda().eval()
    qf, qrecords = extract(model, query_loader, "query")
    gf, grecords = extract(model, gallery_loader, "gallery")
    # The normal MARS evaluator prepends query tracks to the gallery.
    gf_full = torch.cat((qf, gf), dim=0)
    qnorm = torch.nn.functional.normalize(qf, dim=1, p=2)
    gnorm = torch.nn.functional.normalize(gf_full, dim=1, p=2)
    dist = -(qnorm @ gnorm.T)
    finite_dist = torch.isfinite(dist)
    bad_q = (~finite_dist).sum(dim=1)
    bad_g = (~finite_dist).sum(dim=0)
    for i, row in enumerate(qrecords):
        row["bad_distance_count"] = int(bad_q[i])
    for i, row in enumerate(qrecords + grecords):
        row["bad_distance_count_as_gallery"] = int(bad_g[i])
    return {
        "checkpoint": checkpoint,
        "query_count": len(qrecords),
        "gallery_count_including_query": len(qrecords) + len(grecords),
        "nonfinite_query_features": sum(not r["finite"] for r in qrecords),
        "nonfinite_gallery_features": sum(not r["finite"] for r in grecords),
        "nonfinite_distances": int((~finite_dist).sum()),
        "affected_query_rows": int((bad_q > 0).sum()),
        "affected_gallery_columns": int((bad_g > 0).sum()),
    }, qrecords + grecords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for name in ("summary.json", "tracklets.csv"):
        if os.path.exists(os.path.join(args.output_dir, name)):
            raise FileExistsError(os.path.join(args.output_dir, name))
    cfg.merge_from_file(args.config)
    # This diagnostic runs in a restricted shell where worker pipes are denied.
    # Evaluation ordering is unchanged because the loaders use shuffle=False.
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    seed = cfg.SOLVER.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loaders = make_dataloader(cfg)
    query_loader, gallery_loader = loaders[2:4]
    model = make_model(cfg, num_class=loaders[6], camera_num=loaders[8], view_num=0)
    summary, records = audit(model, query_loader, gallery_loader, args.checkpoint)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "tracklets.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
