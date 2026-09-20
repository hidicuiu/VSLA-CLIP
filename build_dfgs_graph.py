"""Build the identity neighbor graph from Stage-1 CLIP text features."""

import argparse
import json

import torch
import torch.nn.functional as F

from config import cfg
from model.make_model_clipvideoreid_reidadapter_pbp import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_classes', type=int, default=930)
    parser.add_argument('--camera_num', type=int, default=2)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--neighbors', type=int, default=10)
    parser.add_argument('--chunk_size', type=int, default=128)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = make_model(cfg, num_class=args.num_classes, camera_num=args.camera_num, view_num=0)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.cuda().eval()

    features = []
    with torch.no_grad():
        for begin in range(0, args.num_classes, args.chunk_size):
            labels = torch.arange(begin, min(begin + args.chunk_size, args.num_classes), device='cuda')
            features.append(model(label=labels, get_text=True).float().cpu())
    features = F.normalize(torch.cat(features, dim=0), dim=1)
    similarity = features @ features.t()
    similarity.fill_diagonal_(-float('inf'))
    ranked = similarity.argsort(dim=1, descending=True)
    end = args.skip + args.neighbors
    graph = {str(pid): ranked[pid, args.skip:end].tolist() for pid in range(args.num_classes)}
    with open(args.output, 'w') as output_file:
        json.dump(graph, output_file, indent=2)
    print('Saved {}-ID graph to {} (skip={}, neighbors={})'.format(
        args.num_classes, args.output, args.skip, args.neighbors))


if __name__ == '__main__':
    main()
