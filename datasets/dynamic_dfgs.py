"""Periodic current-model DFGS-V graph construction for video ReID."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset_transformer import spatial_transforms as ST
from datasets.video_loader import get_default_video_loader


def should_refresh_graph(epoch, period):
    if period <= 0:
        raise ValueError("Dynamic DFGS update period must be positive")
    return epoch == 1 or (epoch - 1) % period == 0


def deterministic_frame_paths(img_paths, sequence_length):
    if not img_paths:
        raise ValueError("A training tracklet contains no frames")
    positions = np.linspace(0, len(img_paths) - 1, num=sequence_length)
    return [img_paths[int(round(position))] for position in positions]


class DynamicGraphTrackletDataset(Dataset):
    def __init__(self, tracklets, sequence_length, spatial_transform,
                 return_temporal_views=False):
        self.tracklets = tracklets
        self.sequence_length = sequence_length
        self.spatial_transform = spatial_transform
        self.return_temporal_views = return_temporal_views
        self.loader = get_default_video_loader()

    def __len__(self):
        return len(self.tracklets)

    def __getitem__(self, index):
        img_paths, pid, camid = self.tracklets[index]
        selected = deterministic_frame_paths(img_paths, self.sequence_length)
        if self.return_temporal_views:
            view_paths = deterministic_frame_paths(
                img_paths, 2 * self.sequence_length)
            paths = selected + view_paths[::2] + view_paths[1::2]
        else:
            paths = selected
        frames = self.loader(paths)
        self.spatial_transform.randomize_parameters()
        frames = [self.spatial_transform(frame) for frame in frames]
        clip = torch.stack(frames[:self.sequence_length], dim=0).permute(1, 0, 2, 3)
        if not self.return_temporal_views:
            return clip, int(pid), int(camid), index
        first = torch.stack(
            frames[self.sequence_length:2 * self.sequence_length], dim=0
        ).permute(1, 0, 2, 3)
        second = torch.stack(
            frames[2 * self.sequence_length:], dim=0
        ).permute(1, 0, 2, 3)
        return clip, first, second, int(pid), int(camid), index, len(img_paths)


def temporal_reliability(frame_features):
    """Mean frame-to-tracklet-center cosine consistency in [0, 1]."""
    normalized = F.normalize(frame_features.float(), dim=-1)
    centers = F.normalize(normalized.mean(dim=1), dim=-1)
    consistency = (normalized * centers.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    return consistency.clamp(0.0, 1.0)


def two_view_reliability(first_features, second_features, frame_counts):
    consistency = F.cosine_similarity(
        first_features.float(), second_features.float(), dim=1)
    # Repeated views of a one-frame tracklet provide no reliability evidence.
    single_frame = frame_counts.to(consistency.device) < 2
    return consistency.masked_fill(single_frame, 0.0).clamp(0.0, 1.0)


def build_identity_centroids(tracklet_features, pids, num_identities,
                             reliabilities=None):
    if tracklet_features.ndim != 2:
        raise ValueError("tracklet_features must have shape [N, D]")
    if len(tracklet_features) != len(pids):
        raise ValueError("tracklet_features and pids must have equal length")
    features = F.normalize(tracklet_features.float(), dim=1)
    if reliabilities is None:
        weights = torch.ones(len(features), dtype=features.dtype)
    else:
        if len(reliabilities) != len(features):
            raise ValueError("reliabilities and tracklet_features must have equal length")
        weights = reliabilities.float().clamp_min(1e-6)
    sums = torch.zeros(num_identities, features.shape[1], dtype=features.dtype)
    weight_sums = torch.zeros(num_identities, dtype=features.dtype)
    for feature, raw_pid, weight in zip(features, pids, weights):
        pid = int(raw_pid)
        if pid < 0 or pid >= num_identities:
            raise ValueError("Identity {} is outside [0, {})".format(pid, num_identities))
        sums[pid] += feature * weight
        weight_sums[pid] += weight
    missing = torch.where(weight_sums == 0)[0].tolist()
    if missing:
        raise ValueError("No tracklets found for identities {}".format(missing[:5]))
    return F.normalize(sums / weight_sums.unsqueeze(1), dim=1), weight_sums


def graph_from_centroids(centroids, skip, neighbors):
    num_identities = centroids.shape[0]
    if skip < 0 or neighbors <= 0:
        raise ValueError("skip must be >= 0 and neighbors must be > 0")
    if skip + neighbors > num_identities - 1:
        raise ValueError("skip + neighbors exceeds the available negative identities")
    similarity = F.normalize(centroids.float(), dim=1) @ F.normalize(
        centroids.float(), dim=1).t()
    similarity.fill_diagonal_(-float("inf"))
    ranking = similarity.argsort(dim=1, descending=True)
    return {
        pid: ranking[pid, skip:skip + neighbors].tolist()
        for pid in range(num_identities)
    }


def identity_reliability(tracklet_reliabilities, pids, num_identities):
    """Use each identity's weakest tracklet to avoid hiding an unstable view."""
    grouped = [[] for _ in range(num_identities)]
    for reliability, raw_pid in zip(tracklet_reliabilities, pids):
        grouped[int(raw_pid)].append(float(reliability))
    if any(not values for values in grouped):
        raise ValueError("Every identity needs at least one reliability value")
    minimum = torch.tensor([min(values) for values in grouped])
    mean = torch.tensor([sum(values) / len(values) for values in grouped])
    return minimum, mean


def stratified_two_camera_exclusions(tracklet_reliabilities, pids, camids,
                                     frame_counts, num_identities,
                                     low_quality_fraction,
                                     length_bounds=(8, 16, 32, 64)):
    """Exclude an ID only when both cameras score low within length peers."""
    if not 0.0 <= low_quality_fraction < 1.0:
        raise ValueError("low_quality_fraction must be in [0, 1)")
    n = len(tracklet_reliabilities)
    if not (len(pids) == len(camids) == len(frame_counts) == n):
        raise ValueError("Tracklet scores, IDs, cameras and lengths must align")
    cameras = sorted({int(camera) for camera in camids})
    if len(cameras) != 2:
        raise ValueError("Two-camera reliability filtering requires exactly two camera labels")
    if sorted(length_bounds) != list(length_bounds) or len(set(length_bounds)) != len(length_bounds):
        raise ValueError("length_bounds must be strictly increasing")

    def length_bin(length):
        return sum(int(length) >= bound for bound in length_bounds)

    groups = {}
    for index, (camera, count) in enumerate(zip(camids, frame_counts)):
        groups.setdefault((int(camera), length_bin(int(count))), []).append(index)
    thresholds = {}
    low = [False] * n
    for key, indices in groups.items():
        scores = torch.as_tensor(
            [float(tracklet_reliabilities[index]) for index in indices])
        threshold = float(torch.quantile(scores, low_quality_fraction))
        thresholds[key] = threshold
        for index in indices:
            low[index] = float(tracklet_reliabilities[index]) < threshold

    low_by_identity = {pid: {camera: False for camera in cameras}
                       for pid in range(num_identities)}
    seen_by_identity = {pid: set() for pid in range(num_identities)}
    for pid, camera, is_low in zip(pids, camids, low):
        pid, camera = int(pid), int(camera)
        if pid not in low_by_identity:
            raise ValueError("Identity is outside [0, num_identities)")
        seen_by_identity[pid].add(camera)
        low_by_identity[pid][camera] |= is_low
    if any(seen != set(cameras) for seen in seen_by_identity.values()):
        raise ValueError("Every identity must have a tracklet from both cameras")

    excluded = torch.tensor([
        all(low_by_identity[pid].values()) for pid in range(num_identities)
    ], dtype=torch.bool)
    low_counts = [sum(low_by_identity[pid].values())
                  for pid in range(num_identities)]
    diagnostics = {
        "excluded_identity_count": int(excluded.sum()),
        "single_camera_low_count": sum(count == 1 for count in low_counts),
        "both_camera_low_count": sum(count == 2 for count in low_counts),
        "length_bounds": list(length_bounds),
        "thresholds_by_camera_and_length": {
            str(camera): {
                str(bin_index): {
                    "threshold": thresholds[(camera, bin_index)],
                    "tracklet_count": len(groups[(camera, bin_index)]),
                    "low_tracklet_count": sum(
                        low[index] for index in groups[(camera, bin_index)]),
                }
                for bin_index in range(len(length_bounds) + 1)
                if (camera, bin_index) in groups
            }
            for camera in cameras
        },
    }
    return excluded, diagnostics


def stratified_multicamera_exclusions(tracklet_reliabilities, pids, camids,
                                      frame_counts, num_identities,
                                      low_quality_fraction,
                                      min_observed_cameras=2,
                                      min_low_camera_fraction=0.5,
                                      length_bounds=(8, 16, 32, 64)):
    """Exclude IDs that score low in enough of their observed cameras."""
    if not 0.0 <= low_quality_fraction < 1.0:
        raise ValueError("low_quality_fraction must be in [0, 1)")
    if min_observed_cameras < 2:
        raise ValueError("min_observed_cameras must be at least 2")
    if not 0.0 < min_low_camera_fraction <= 1.0:
        raise ValueError("min_low_camera_fraction must be in (0, 1]")
    n = len(tracklet_reliabilities)
    if not (len(pids) == len(camids) == len(frame_counts) == n):
        raise ValueError("Tracklet scores, IDs, cameras and lengths must align")
    if sorted(length_bounds) != list(length_bounds) or len(set(length_bounds)) != len(length_bounds):
        raise ValueError("length_bounds must be strictly increasing")

    cameras = sorted({int(camera) for camera in camids})
    groups = {}
    for index, (camera, count) in enumerate(zip(camids, frame_counts)):
        length_bin = sum(int(count) >= bound for bound in length_bounds)
        groups.setdefault((int(camera), length_bin), []).append(index)
    thresholds = {}
    low = [False] * n
    for key, indices in groups.items():
        scores = torch.as_tensor(
            [float(tracklet_reliabilities[index]) for index in indices])
        threshold = float(torch.quantile(scores, low_quality_fraction))
        thresholds[key] = threshold
        for index in indices:
            low[index] = float(tracklet_reliabilities[index]) < threshold

    low_by_identity = {pid: {} for pid in range(num_identities)}
    for pid, camera, is_low in zip(pids, camids, low):
        pid, camera = int(pid), int(camera)
        if pid not in low_by_identity:
            raise ValueError("Identity is outside [0, num_identities)")
        low_by_identity[pid][camera] = low_by_identity[pid].get(camera, False) or is_low

    excluded_values = []
    coverage_counts = {}
    low_counts = {}
    for camera_scores in low_by_identity.values():
        observed = len(camera_scores)
        low_count = sum(camera_scores.values())
        coverage_counts[observed] = coverage_counts.get(observed, 0) + 1
        low_counts[low_count] = low_counts.get(low_count, 0) + 1
        excluded_values.append(
            observed >= min_observed_cameras and low_count >= 2 and
            low_count / observed >= min_low_camera_fraction)
    excluded = torch.tensor(excluded_values, dtype=torch.bool)
    diagnostics = {
        "excluded_identity_count": int(excluded.sum()),
        "num_cameras": len(cameras),
        "camera_labels": cameras,
        "min_observed_cameras": min_observed_cameras,
        "min_low_camera_fraction": min_low_camera_fraction,
        "identity_camera_coverage_counts": coverage_counts,
        "identity_low_camera_counts": low_counts,
        "length_bounds": list(length_bounds),
        "thresholds_by_camera_and_length": {
            str(camera): {
                str(bin_index): {
                    "threshold": thresholds[(camera, bin_index)],
                    "tracklet_count": len(groups[(camera, bin_index)]),
                    "low_tracklet_count": sum(low[index] for index in groups[(camera, bin_index)]),
                }
                for bin_index in range(len(length_bounds) + 1)
                if (camera, bin_index) in groups
            }
            for camera in cameras
        },
    }
    return excluded, diagnostics


def reliable_graph_from_centroids(centroids, identity_quality, skip,
                                  neighbors, candidate_pool,
                                  low_quality_fraction, excluded_mask=None):
    """Filter the nearest candidate pool; keep visual ranking and fill gaps."""
    num_identities = centroids.shape[0]
    if not 0.0 <= low_quality_fraction < 1.0:
        raise ValueError("low_quality_fraction must be in [0, 1)")
    if candidate_pool < neighbors:
        raise ValueError("candidate_pool must be at least neighbors")
    if skip < 0 or skip + candidate_pool > num_identities - 1:
        raise ValueError("skip + candidate_pool exceeds negative identities")
    if len(identity_quality) != num_identities:
        raise ValueError("identity_quality must match num_identities")
    normalized = F.normalize(centroids.float(), dim=1)
    similarity = normalized @ normalized.t()
    similarity.fill_diagonal_(-float("inf"))
    ranking = similarity.argsort(dim=1, descending=True)
    threshold = (float(torch.quantile(identity_quality.float(), low_quality_fraction))
                 if excluded_mask is None else None)
    if excluded_mask is not None and len(excluded_mask) != num_identities:
        raise ValueError("excluded_mask must match num_identities")
    excluded = (identity_quality < threshold if excluded_mask is None
                else torch.as_tensor(excluded_mask, dtype=torch.bool))
    graph = {}
    filtered_counts = []
    fallback_counts = []
    for pid in range(num_identities):
        pool = ranking[pid, skip:skip + candidate_pool].tolist()
        reliable = [candidate for candidate in pool if not bool(excluded[candidate])]
        selected = reliable[:neighbors]
        filtered_counts.append(len(pool) - len(reliable))
        fallback_counts.append(max(0, neighbors - len(selected)))
        if len(selected) < neighbors:
            fallback_pool = (pool if excluded_mask is None
                             else ranking[pid, skip:].tolist())
            for candidate in fallback_pool:
                if excluded_mask is not None and bool(excluded[candidate]):
                    continue
                if candidate not in selected:
                    selected.append(candidate)
                    if len(selected) == neighbors:
                        break
        if len(selected) != neighbors:
            raise ValueError("Not enough eligible negative identities to fill graph")
        graph[pid] = selected
    diagnostics = {
        "quality_threshold": threshold,
        "excluded_identity_count": int(excluded.sum()),
        "mean_filtered_candidates": sum(filtered_counts) / num_identities,
        "fallback_neighbor_count": sum(fallback_counts),
        "anchors_using_fallback": sum(count > 0 for count in fallback_counts),
    }
    return graph, diagnostics


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def refresh_dynamic_dfgs_graph(cfg, model, sampler, epoch, logger):
    """Extract current-model tracklets, rebuild the graph, and update sampler."""
    if not hasattr(sampler, "update_neighbor_graph"):
        raise TypeError("Dynamic DFGS requires IdentityBatchSamplerVideo")
    train_data = sampler.data_source
    transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
    ])
    reliable_negatives = bool(cfg.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES)
    quality_weighted = bool(cfg.DATALOADER.DYNAMIC_DFGS_QUALITY_WEIGHTED)
    soft_negatives = bool(cfg.DATALOADER.DYNAMIC_DFGS_SOFT_NEGATIVES)
    stratified_multicamera = bool(cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)
    if stratified_multicamera and cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA:
        raise ValueError("Choose either two-camera or multicamera stratification")
    if stratified_multicamera and not reliable_negatives:
        raise ValueError("Multicamera stratification requires reliable negatives")
    if reliable_negatives and quality_weighted:
        raise ValueError("Reliable negative filtering requires unweighted D1 identity centers")
    if soft_negatives and (not reliable_negatives or
                           not cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA):
        raise ValueError("Soft negatives require D4 camera/length reliability scores")
    feature_dataset = DynamicGraphTrackletDataset(
        train_data, cfg.DATALOADER.SEQ_LEN, transform,
        return_temporal_views=reliable_negatives)
    loader = DataLoader(
        feature_dataset,
        batch_size=cfg.DATALOADER.DYNAMIC_DFGS_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATALOADER.DYNAMIC_DFGS_NUM_WORKERS,
        pin_memory=True,
        drop_last=False)

    base_model = _unwrap_model(model)
    was_training = model.training
    model.eval()
    all_tracklet_features = []
    all_reliabilities = []
    all_temporal_reliabilities = []
    all_pids = []
    all_camids = []
    all_frame_counts = []
    all_indices = []
    logger.info(
        "Refreshing dynamic DFGS-V graph at epoch %d (%d tracklets, quality_weighted=%s, reliable_negatives=%s)",
        epoch, len(feature_dataset), quality_weighted, reliable_negatives)
    try:
        with torch.no_grad():
            for step, batch in enumerate(loader, start=1):
                if reliable_negatives:
                    (clips, first_view, second_view, pids, camids,
                     indices, frame_counts) = batch
                    first_view = first_view.cuda(non_blocking=True)
                    second_view = second_view.cuda(non_blocking=True)
                else:
                    clips, pids, camids, indices = batch
                clips = clips.cuda(non_blocking=True)
                camids = camids.cuda(non_blocking=True)
                camera_labels = camids if cfg.MODEL.PBP_CAMERA else None
                view_labels = camids if cfg.MODEL.PBP_VIEW else None
                with torch.cuda.amp.autocast(enabled=True):
                    tracklet_features = base_model(
                        clips, cam_label=camera_labels, view_label=view_labels)
                    if quality_weighted:
                        frame_features = base_model(clips, get_frame_image=True)
                    if reliable_negatives:
                        first_features = base_model(
                            first_view, cam_label=camera_labels,
                            view_label=view_labels)
                        second_features = base_model(
                            second_view, cam_label=camera_labels,
                            view_label=view_labels)
                all_tracklet_features.append(tracklet_features.float().cpu())
                if quality_weighted:
                    all_reliabilities.append(
                        temporal_reliability(frame_features).float().cpu())
                if reliable_negatives:
                    all_temporal_reliabilities.append(
                        two_view_reliability(
                            first_features, second_features, frame_counts).cpu())
                all_pids.extend(int(pid) for pid in pids.tolist())
                if reliable_negatives:
                    all_camids.extend(int(camera) for camera in camids.cpu().tolist())
                    all_frame_counts.extend(int(count) for count in frame_counts.tolist())
                all_indices.extend(int(index) for index in indices.tolist())
                if step % 50 == 0 or step == len(loader):
                    logger.info("Dynamic DFGS extraction: %d/%d batches", step, len(loader))
    finally:
        if was_training:
            model.train()
    if all_indices != list(range(len(train_data))):
        raise RuntimeError("Dynamic DFGS tracklet order does not match training data")

    tracklet_features = torch.cat(all_tracklet_features, dim=0)
    reliabilities = (torch.cat(all_reliabilities, dim=0)
                     if quality_weighted else None)
    num_identities = len(sampler.pids)
    centroids, identity_weight_sums = build_identity_centroids(
        tracklet_features, all_pids, num_identities, reliabilities)
    identity_minimum = identity_mean = diagnostics = None
    if reliable_negatives:
        temporal_reliabilities = torch.cat(all_temporal_reliabilities, dim=0)
        identity_minimum, identity_mean = identity_reliability(
            temporal_reliabilities, all_pids, num_identities)
        stratified = bool(cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA)
        excluded_mask = stratified_diagnostics = None
        if stratified:
            excluded_mask, stratified_diagnostics = stratified_two_camera_exclusions(
                temporal_reliabilities, all_pids, all_camids,
                all_frame_counts, num_identities,
                cfg.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION)
        elif stratified_multicamera:
            excluded_mask, stratified_diagnostics = stratified_multicamera_exclusions(
                temporal_reliabilities, all_pids, all_camids,
                all_frame_counts, num_identities,
                cfg.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION,
                cfg.DATALOADER.DYNAMIC_DFGS_MIN_OBSERVED_CAMERAS,
                cfg.DATALOADER.DYNAMIC_DFGS_MIN_LOW_CAMERA_FRACTION)
        if soft_negatives:
            # Keep the same ten visual neighbors as D1. Reliability affects
            # their sampling probability, not their graph membership.
            graph = graph_from_centroids(
                centroids, cfg.DATALOADER.DYNAMIC_DFGS_SKIP,
                cfg.DATALOADER.DYNAMIC_DFGS_NEIGHBORS)
            diagnostics = {
                "quality_threshold": None,
                "excluded_identity_count": 0,
                "downweighted_identity_count": int(excluded_mask.sum()),
                "mean_filtered_candidates": 0.0,
                "fallback_neighbor_count": 0,
                "anchors_using_fallback": 0,
            }
        else:
            graph, diagnostics = reliable_graph_from_centroids(
                centroids, identity_minimum,
                cfg.DATALOADER.DYNAMIC_DFGS_SKIP,
                cfg.DATALOADER.DYNAMIC_DFGS_NEIGHBORS,
                cfg.DATALOADER.DYNAMIC_DFGS_CANDIDATE_POOL,
                cfg.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION,
                excluded_mask=excluded_mask)
        if stratified or stratified_multicamera:
            diagnostics.update(stratified_diagnostics)
        if soft_negatives:
            diagnostics["downweighted_identity_count"] = int(excluded_mask.sum())
            diagnostics["excluded_identity_count"] = 0
        if not soft_negatives:
            logger.info(
            "Reliable negatives: mode=%s, threshold=%s, excluded IDs=%d/%d, "
            "mean filtered candidates=%.2f, fallback neighbors=%d across %d anchors",
            ("stratified_two_camera" if stratified else
             "stratified_multicamera" if stratified_multicamera else
             "identity_minimum"),
            diagnostics["quality_threshold"],
            diagnostics["excluded_identity_count"], num_identities,
            diagnostics["mean_filtered_candidates"],
            diagnostics["fallback_neighbor_count"],
            diagnostics["anchors_using_fallback"])
        if stratified:
            logger.info(
                "%s reliability detail at epoch %d: one-camera-low=%d, "
                "both-camera-low=%d, %s=%d/%d, length-bounds=%s",
                "D5 soft" if soft_negatives else "D4 exclusion",
                epoch, diagnostics["single_camera_low_count"],
                diagnostics["both_camera_low_count"],
                "downweighted" if soft_negatives else "excluded",
                (diagnostics["downweighted_identity_count"] if soft_negatives
                 else diagnostics["excluded_identity_count"]), num_identities,
                diagnostics["length_bounds"])
        elif stratified_multicamera:
            logger.info(
                "Multicamera reliability at epoch %d: cameras=%s, coverage=%s, "
                "low-camera-counts=%s, excluded=%d/%d, min-observed=%d, min-low-fraction=%.2f",
                epoch, diagnostics["camera_labels"],
                diagnostics["identity_camera_coverage_counts"],
                diagnostics["identity_low_camera_counts"],
                diagnostics["excluded_identity_count"], num_identities,
                diagnostics["min_observed_cameras"],
                diagnostics["min_low_camera_fraction"])
            for camera, bins in sorted(
                    diagnostics["thresholds_by_camera_and_length"].items()):
                for length_bin, group in sorted(bins.items(), key=lambda item: int(item[0])):
                    logger.info(
                        "%s group at epoch %d: camera=%s length-bin=%s "
                        "tracklets=%d low-tracklets=%d threshold=%.6f",
                        "D5" if soft_negatives else "D4", epoch, camera,
                        length_bin, group["tracklet_count"],
                        group["low_tracklet_count"], group["threshold"])
    else:
        graph = graph_from_centroids(
            centroids,
            cfg.DATALOADER.DYNAMIC_DFGS_SKIP,
            cfg.DATALOADER.DYNAMIC_DFGS_NEIGHBORS)
    low_candidate_ids = (torch.where(excluded_mask)[0].tolist()
                         if soft_negatives else None)
    sampler.update_neighbor_graph(
        graph, low_candidate_ids=low_candidate_ids,
        low_candidate_weight=(cfg.DATALOADER.DYNAMIC_DFGS_LOW_CANDIDATE_WEIGHT
                              if soft_negatives else 1.0))
    if soft_negatives:
        low_slots = sum(pid in sampler.low_candidate_ids
                        for neighbors in graph.values() for pid in neighbors)
        logger.info(
            "Soft negatives at epoch %d: low IDs=%d/%d, low graph slots=%d/%d, weight=%.2f",
            epoch, len(low_candidate_ids), num_identities, low_slots,
            sum(map(len, graph.values())), sampler.low_candidate_weight)

    if cfg.DATALOADER.DYNAMIC_DFGS_SAVE_GRAPHS:
        graph_dir = os.path.join(cfg.OUTPUT_DIR, "dynamic_dfgs_graphs")
        os.makedirs(graph_dir, exist_ok=True)
        graph_path = os.path.join(graph_dir, "epoch_{:03d}.json".format(epoch))
        payload = {
            "metadata": {
                "epoch": epoch,
                "update_period": cfg.DATALOADER.DYNAMIC_DFGS_UPDATE_PERIOD,
                "skip": cfg.DATALOADER.DYNAMIC_DFGS_SKIP,
                "neighbors": cfg.DATALOADER.DYNAMIC_DFGS_NEIGHBORS,
                "quality_weighted": quality_weighted,
                "reliable_negatives": reliable_negatives,
                "num_tracklets": len(train_data),
                "num_identities": num_identities,
                "soft_negatives": soft_negatives,
            },
            "graph": {str(pid): neighbors for pid, neighbors in graph.items()},
        }
        if reliabilities is not None:
            payload["tracklet_reliability"] = [
                float(value) for value in reliabilities.tolist()]
        if reliable_negatives:
            payload["metadata"].update({
                "candidate_pool": cfg.DATALOADER.DYNAMIC_DFGS_CANDIDATE_POOL,
                "low_quality_fraction": cfg.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION,
                "reliability_definition": (
                    "two-view cosine, camera/length-stratified, both cameras low"
                    if stratified else
                    "two-view cosine, camera/length-stratified, observed-camera consensus"
                    if stratified_multicamera else
                    "min tracklet cosine across two deterministic temporal views"),
                "stratified_two_camera": stratified,
                "stratified_multicamera": stratified_multicamera,
            })
            payload["filter_diagnostics"] = diagnostics
            payload["tracklet_temporal_reliability"] = [
                float(value) for value in temporal_reliabilities.tolist()]
            payload["identity_reliability_min"] = [
                float(value) for value in identity_minimum.tolist()]
            payload["identity_reliability_mean"] = [
                float(value) for value in identity_mean.tolist()]
            if stratified or stratified_multicamera:
                if soft_negatives:
                    payload["downweighted_identity_ids"] = low_candidate_ids
                else:
                    payload["excluded_identity_ids"] = torch.where(excluded_mask)[0].tolist()
            if soft_negatives:
                payload["metadata"]["low_candidate_weight"] = sampler.low_candidate_weight
        with open(graph_path, "w", encoding="utf-8") as graph_file:
            json.dump(payload, graph_file, indent=2)
        logger.info("Saved dynamic DFGS graph to %s", graph_path)
    logger.info(
        "Dynamic DFGS identity weight sums: min=%.3f max=%.3f mean=%.3f",
        float(identity_weight_sums.min()), float(identity_weight_sums.max()),
        float(identity_weight_sums.mean()))
    return graph
