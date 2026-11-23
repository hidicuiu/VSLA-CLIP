"""Periodic current-model DFGS-V graph construction for video ReID."""

import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset_transformer import spatial_transforms as ST
from datasets.video_loader import get_default_video_loader
from utils.reproducibility import preserve_rng_state


def should_refresh_graph(epoch, period):
    if period <= 0:
        raise ValueError("Dynamic DFGS update period must be positive")
    return epoch == 1 or (epoch - 1) % period == 0


def load_paired_reference_exclusions(reference_dir, epoch, num_identities):
    """Read and validate the paired E4 exclusion set for one refresh epoch."""
    reference_graph = os.path.join(
        reference_dir, "epoch_{:03d}.json".format(epoch))
    if not os.path.isfile(reference_graph):
        raise FileNotFoundError(
            "Paired E4 reference graph missing for epoch {}: {}".format(
                epoch, reference_graph))
    with open(reference_graph, "r", encoding="utf-8") as reference_file:
        reference_payload = json.load(reference_file)
    metadata = reference_payload.get("metadata", {})
    if int(metadata.get("epoch", -1)) != int(epoch):
        raise ValueError("Paired E4 graph epoch does not match current refresh epoch")
    if int(metadata.get("num_identities", -1)) != int(num_identities):
        raise ValueError("Paired E4 graph identity count does not match this dataset")
    reference_ids = reference_payload.get("excluded_identity_ids")
    if not isinstance(reference_ids, list):
        raise ValueError("Paired E4 graph is missing excluded_identity_ids")
    normalized_ids = [int(pid) for pid in reference_ids]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("Paired E4 graph contains duplicate excluded identities")
    if any(pid < 0 or pid >= num_identities for pid in normalized_ids):
        raise ValueError("Paired E4 graph contains out-of-range identity IDs")
    return normalized_ids, reference_graph


def sample_random_exclusion_mask(num_identities, excluded_count, seed):
    """Sample exactly excluded_count identities without touching global RNG."""
    if excluded_count < 0 or excluded_count > num_identities:
        raise ValueError("Random exclusion count must be within identity count")
    sampled_ids = np.random.RandomState(int(seed)).choice(
        num_identities, size=excluded_count, replace=False)
    mask = torch.zeros(num_identities, dtype=torch.bool)
    mask[torch.as_tensor(sampled_ids, dtype=torch.long)] = True
    if int(mask.sum()) != excluded_count:
        raise RuntimeError("Random exclusion mask does not match requested count")
    return mask


def identity_camera_length_profiles(tracklets, num_identities,
                                    length_bounds=(8, 16, 32, 64)):
    """Return each identity's camera-by-track-length-bin profile."""
    per_identity = [[] for _ in range(num_identities)]
    for img_paths, raw_pid, raw_camera in tracklets:
        pid, camera = int(raw_pid), int(raw_camera)
        if pid < 0 or pid >= num_identities:
            raise ValueError("Identity is outside [0, num_identities)")
        length_bin = sum(len(img_paths) >= bound for bound in length_bounds)
        per_identity[pid].append((camera, int(length_bin)))
    if any(not profile for profile in per_identity):
        raise ValueError("Every identity must have a tracklet length profile")
    return [tuple(sorted(profile)) for profile in per_identity]


def sample_stratified_random_exclusion_mask(identity_profiles,
                                            reference_excluded_ids, seed):
    """Randomly sample within E4's camera/length profile quotas."""
    num_identities = len(identity_profiles)
    reference_excluded_ids = [int(pid) for pid in reference_excluded_ids]
    if len(reference_excluded_ids) != len(set(reference_excluded_ids)):
        raise ValueError("Reference exclusion IDs must be unique")
    if any(pid < 0 or pid >= num_identities for pid in reference_excluded_ids):
        raise ValueError("Reference exclusion ID is outside identity range")
    target_counts = Counter(identity_profiles[pid] for pid in reference_excluded_ids)
    pools = defaultdict(list)
    for pid, profile in enumerate(identity_profiles):
        pools[profile].append(pid)
    rng = np.random.RandomState(int(seed))
    sampled_ids = []
    profile_summary = []
    for profile in sorted(target_counts, key=repr):
        target_count = target_counts[profile]
        pool = pools[profile]
        if target_count > len(pool):
            raise ValueError("Reference exclusion quota exceeds profile population")
        sampled_ids.extend(int(pid) for pid in rng.choice(
            pool, size=target_count, replace=False))
        profile_summary.append({
            "camera_length_bins": [list(value) for value in profile],
            "reference_excluded_count": int(target_count),
            "random_excluded_count": int(target_count),
            "profile_population": len(pool),
        })
    mask = torch.zeros(num_identities, dtype=torch.bool)
    mask[torch.as_tensor(sampled_ids, dtype=torch.long)] = True
    if int(mask.sum()) != len(reference_excluded_ids):
        raise RuntimeError("Random exclusion mask does not match reference count")
    return mask, profile_summary


def sample_candidate_impact_matched_random_exclusion_mask(
        centroids, identity_profiles, reference_excluded_ids, seed,
        skip, candidate_pool, iterations=6000, restarts=4):
    """Sample a random mask matching E4's aggregate candidate-pool impact.

    The mask preserves E4's total exclusion count and camera/length profile
    quotas. A seeded swap search matches the distribution of per-anchor
    filtered counts, removed-candidate ranks, and similarities. The E4 ID
    overlap is capped at the expected random-overlap range, so the control
    does not copy E4's exclusion list.
    """
    centroids = F.normalize(torch.as_tensor(centroids).float(), dim=1)
    num_identities = len(identity_profiles)
    reference_ids = [int(pid) for pid in reference_excluded_ids]
    if centroids.ndim != 2 or centroids.shape[0] != num_identities:
        raise ValueError("Centroids must have one row per identity")
    if not bool(torch.isfinite(centroids).all()):
        raise ValueError("Non-finite identity centroids in candidate-impact matching")
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference exclusion IDs must be unique")
    if any(pid < 0 or pid >= num_identities for pid in reference_ids):
        raise ValueError("Reference exclusion ID is outside identity range")
    if skip < 0 or candidate_pool <= 0 or skip + candidate_pool > num_identities - 1:
        raise ValueError("Invalid skip/candidate_pool for identity count")
    if iterations <= 0 or restarts <= 0:
        raise ValueError("Candidate-impact search iterations/restarts must be positive")

    similarity = centroids @ centroids.t()
    similarity.fill_diagonal_(-float("inf"))
    ranking = similarity.argsort(dim=1, descending=True)
    candidate_ids = ranking[:, skip:skip + candidate_pool].cpu().numpy()
    candidate_sims = similarity.gather(1, ranking[:, skip:skip + candidate_pool]).cpu().numpy()
    reference_mask = np.zeros(num_identities, dtype=bool)
    reference_mask[reference_ids] = True

    # Candidate incidence enables a swap to update all affected anchors.
    incidence = np.zeros((num_identities, num_identities), dtype=np.uint8)
    rank_hist_by_id = np.zeros((num_identities, candidate_pool), dtype=np.int32)
    sim_hist_by_id = np.zeros((num_identities, 8), dtype=np.int32)
    rank_sum_by_id = np.zeros(num_identities, dtype=np.float64)
    similarity_sum_by_id = np.zeros(num_identities, dtype=np.float64)
    actual_ranks = np.arange(skip + 1, skip + candidate_pool + 1)
    rank_bins = np.arange(candidate_pool)
    similarity_edges = np.quantile(
        candidate_sims.reshape(-1), np.arange(0.125, 1.0, 0.125))
    sim_bins = np.digitize(candidate_sims, similarity_edges)
    for anchor in range(num_identities):
        ids = candidate_ids[anchor]
        incidence[anchor, ids] = 1
        np.add.at(rank_hist_by_id, (ids, rank_bins), 1)
        np.add.at(sim_hist_by_id, (ids, sim_bins[anchor]), 1)
        np.add.at(rank_sum_by_id, ids, actual_ranks)
        np.add.at(similarity_sum_by_id, ids, candidate_sims[anchor])

    target_counts = incidence[:, reference_mask].sum(axis=1).astype(np.int32)
    target_count_dist = np.bincount(
        target_counts, minlength=candidate_pool + 1).astype(np.float64)
    target_count_dist /= max(1.0, float(target_count_dist.sum()))
    target_rank_hist = rank_hist_by_id[reference_mask].sum(axis=0)
    target_sim_hist = sim_hist_by_id[reference_mask].sum(axis=0)
    target_total_hits = max(1, int(target_counts.sum()))
    target_mean_hits = float(target_counts.mean())
    target_mean_rank = float(rank_sum_by_id[reference_mask].sum() / target_total_hits)
    target_mean_similarity = float(
        similarity_sum_by_id[reference_mask].sum() / target_total_hits)
    candidate_similarity_scale = max(
        float(np.std(candidate_sims)), 1e-6)
    target_rank_dist = target_rank_hist / max(1, int(target_rank_hist.sum()))
    target_sim_dist = target_sim_hist / max(1, int(target_sim_hist.sum()))

    quotas = Counter(identity_profiles[pid] for pid in reference_ids)
    profile_pools = defaultdict(list)
    for pid, profile in enumerate(identity_profiles):
        profile_pools[profile].append(pid)
    profile_summary = []
    expected_overlap = 0.0
    overlap_variance = 0.0
    for profile, quota in quotas.items():
        available = profile_pools[profile]
        if quota > len(available):
            raise ValueError(
                "Cannot preserve profile quota: quota={} population={} profile={}".format(
                    quota, len(available), profile))
        population = len(available)
        if population > 1:
            rate = float(quota) / population
            expected_overlap += float(quota) * rate
            overlap_variance += (
                float(quota) * rate * (1.0 - rate)
                * float(population - quota) / float(population - 1))
        else:
            expected_overlap += float(quota)
        profile_summary.append({
            "camera_length_bins": [list(value) for value in profile],
            "reference_excluded_count": int(quota),
            "random_excluded_count": int(quota),
            "profile_population": population,
        })
    overlap_cap = min(
        len(reference_ids), int(np.ceil(
            expected_overlap + 2.0 * np.sqrt(overlap_variance))))

    def metrics(mask):
        anchor_counts = incidence[:, mask].sum(axis=1).astype(np.int32)
        anchor_dist = np.bincount(
            anchor_counts, minlength=candidate_pool + 1).astype(np.float64)
        anchor_dist /= max(1.0, float(anchor_dist.sum()))
        rank_hist = rank_hist_by_id[mask].sum(axis=0)
        sim_hist = sim_hist_by_id[mask].sum(axis=0)
        anchor_tv = float(0.5 * np.abs(anchor_dist - target_count_dist).sum())
        rank_dist = rank_hist / max(1, int(rank_hist.sum()))
        sim_dist = sim_hist / max(1, int(sim_hist.sum()))
        anchor_mae = float(np.abs(anchor_counts - target_counts).mean())
        mean_hits = float(anchor_counts.mean())
        total_hits = max(1, int(anchor_counts.sum()))
        mean_rank = float(rank_sum_by_id[mask].sum() / total_hits)
        mean_similarity = float(similarity_sum_by_id[mask].sum() / total_hits)
        mean_hit_error = abs(mean_hits - target_mean_hits) / max(1.0, target_mean_hits)
        mean_rank_error = abs(mean_rank - target_mean_rank) / max(1, candidate_pool - 1)
        mean_similarity_error = abs(mean_similarity - target_mean_similarity) / candidate_similarity_scale
        rank_tv = float(0.5 * np.abs(rank_dist - target_rank_dist).sum())
        sim_tv = float(0.5 * np.abs(sim_dist - target_sim_dist).sum())
        objective = (
            anchor_tv + rank_tv + sim_tv + mean_hit_error
            + 0.25 * mean_rank_error + 0.25 * mean_similarity_error)
        return (objective, anchor_counts, rank_hist, sim_hist, anchor_tv,
                anchor_mae, rank_tv, sim_tv, mean_hit_error,
                mean_rank_error, mean_similarity_error)

    rng = np.random.RandomState(int(seed))
    best_mask = None
    best_metrics = None
    for restart in range(restarts):
        selected_by_profile = {}
        selected_mask = np.zeros(num_identities, dtype=bool)
        for profile, quota in quotas.items():
            pool = np.asarray(profile_pools[profile], dtype=np.int64)
            for _ in range(100):
                chosen = rng.choice(pool, size=int(quota), replace=False)
                proposed = selected_mask.copy()
                proposed[chosen] = True
                if int(np.logical_and(proposed, reference_mask).sum()) <= overlap_cap:
                    break
            else:
                raise RuntimeError("Could not sample a profile-matched mask under overlap cap")
            selected_by_profile[profile] = set(int(pid) for pid in chosen)
            selected_mask[chosen] = True
        current = metrics(selected_mask)
        current_obj, anchor_counts, rank_hist, sim_hist = current[:4]
        restart_best_mask = selected_mask.copy()
        restart_best = current

        profile_choices = [profile for profile, quota in quotas.items() if quota > 0]
        for step in range(iterations):
            profile = profile_choices[int(rng.randint(len(profile_choices)))]
            chosen_ids = selected_by_profile[profile]
            available_ids = profile_pools[profile]
            remove_id = int(rng.choice(tuple(chosen_ids)))
            add_id = int(rng.choice(available_ids))
            if selected_mask[add_id]:
                continue
            proposed_overlap = int(np.logical_and(selected_mask, reference_mask).sum())
            proposed_overlap += int(reference_mask[add_id]) - int(reference_mask[remove_id])
            if proposed_overlap > overlap_cap:
                continue
            candidate_counts = anchor_counts + incidence[:, add_id].astype(np.int32) - incidence[:, remove_id].astype(np.int32)
            candidate_rank_hist = rank_hist + rank_hist_by_id[add_id] - rank_hist_by_id[remove_id]
            candidate_sim_hist = sim_hist + sim_hist_by_id[add_id] - sim_hist_by_id[remove_id]
            candidate_count_dist = np.bincount(
                candidate_counts, minlength=candidate_pool + 1).astype(np.float64)
            candidate_count_dist /= max(1.0, float(candidate_count_dist.sum()))
            candidate_anchor_tv = float(
                0.5 * np.abs(candidate_count_dist - target_count_dist).sum())
            candidate_rank_dist = candidate_rank_hist / max(1, int(candidate_rank_hist.sum()))
            candidate_sim_dist = candidate_sim_hist / max(1, int(candidate_sim_hist.sum()))
            candidate_mae = float(np.abs(candidate_counts - target_counts).mean())
            candidate_mean_hits = float(candidate_counts.mean())
            candidate_total_hits = max(1, int(candidate_counts.sum()))
            candidate_mean_rank = float(
                (rank_sum_by_id[selected_mask].sum()
                 - rank_sum_by_id[remove_id] + rank_sum_by_id[add_id])
                / candidate_total_hits)
            candidate_mean_similarity = float(
                (similarity_sum_by_id[selected_mask].sum()
                 - similarity_sum_by_id[remove_id] + similarity_sum_by_id[add_id])
                / candidate_total_hits)
            candidate_mean_hit_error = (
                abs(candidate_mean_hits - target_mean_hits)
                / max(1.0, target_mean_hits))
            candidate_mean_rank_error = (
                abs(candidate_mean_rank - target_mean_rank)
                / max(1, candidate_pool - 1))
            candidate_mean_similarity_error = (
                abs(candidate_mean_similarity - target_mean_similarity)
                / candidate_similarity_scale)
            candidate_rank_tv = float(0.5 * np.abs(candidate_rank_dist - target_rank_dist).sum())
            candidate_sim_tv = float(0.5 * np.abs(candidate_sim_dist - target_sim_dist).sum())
            candidate_obj = (
                candidate_anchor_tv + candidate_rank_tv + candidate_sim_tv
                + candidate_mean_hit_error + 0.25 * candidate_mean_rank_error
                + 0.25 * candidate_mean_similarity_error)
            temperature = max(0.001, 0.08 * (1.0 - float(step) / iterations))
            if (candidate_obj <= current_obj or
                    rng.rand() < np.exp((current_obj - candidate_obj) / temperature)):
                selected_mask[remove_id] = False
                selected_mask[add_id] = True
                chosen_ids.remove(remove_id)
                chosen_ids.add(add_id)
                anchor_counts = candidate_counts
                rank_hist = candidate_rank_hist
                sim_hist = candidate_sim_hist
                current_obj = candidate_obj
                if candidate_obj < restart_best[0]:
                    restart_best_mask = selected_mask.copy()
                    restart_best = (
                        candidate_obj, anchor_counts.copy(), rank_hist.copy(),
                        sim_hist.copy(), candidate_anchor_tv, candidate_mae,
                        candidate_rank_tv, candidate_sim_tv,
                        candidate_mean_hit_error, candidate_mean_rank_error,
                        candidate_mean_similarity_error)

        if best_metrics is None or restart_best[0] < best_metrics[0]:
            best_mask = restart_best_mask
            best_metrics = restart_best

    (objective, matched_counts, matched_rank_hist, matched_sim_hist,
     anchor_tv, anchor_mae, rank_tv, sim_tv, mean_hit_error,
     mean_rank_error, mean_similarity_error) = best_metrics
    selected_ids = np.flatnonzero(best_mask)
    overlap = int(np.logical_and(best_mask, reference_mask).sum())
    selected_candidate_hits = incidence[:, best_mask].sum(axis=1).astype(np.int32)
    selected_total_hits = max(1, int(selected_candidate_hits.sum()))
    selected_rank_hist = rank_hist_by_id[best_mask].sum(axis=0)
    selected_sim_hist = sim_hist_by_id[best_mask].sum(axis=0)
    per_anchor_abs = np.abs(selected_candidate_hits - target_counts)
    selected_mean_rank = float(rank_sum_by_id[best_mask].sum() / selected_total_hits)
    selected_mean_similarity = float(
        similarity_sum_by_id[best_mask].sum() / selected_total_hits)
    impact_diagnostics = {
        "candidate_impact_matched": True,
        "candidate_impact_search_iterations_per_restart": int(iterations),
        "candidate_impact_search_restarts": int(restarts),
        "candidate_impact_search_objective": float(objective),
        "maximum_random_reference_identity_overlap": int(overlap_cap),
        "expected_random_reference_identity_overlap": float(expected_overlap),
        "reference_excluded_identity_count": len(reference_ids),
        "random_excluded_identity_count": int(len(selected_ids)),
        "overlap_with_reference_exclusions": overlap,
        "reference_mean_filtered_candidates": float(target_counts.mean()),
        "random_mean_filtered_candidates": float(selected_candidate_hits.mean()),
        "mean_filtered_candidate_relative_error": float(mean_hit_error),
        "same_anchor_filtered_count_mae": float(anchor_mae),
        "same_anchor_filtered_count_rmse": float(np.sqrt(np.mean((selected_candidate_hits - target_counts) ** 2))),
        "same_anchor_exact_filtered_count_fraction": float(np.mean(selected_candidate_hits == target_counts)),
        "per_anchor_filtered_count_distribution_total_variation": float(anchor_tv),
        "reference_removed_candidate_mean_rank": float(target_mean_rank),
        "random_removed_candidate_mean_rank": selected_mean_rank,
        "removed_candidate_mean_rank_normalized_error": float(mean_rank_error),
        "reference_removed_candidate_mean_similarity": float(target_mean_similarity),
        "random_removed_candidate_mean_similarity": selected_mean_similarity,
        "removed_candidate_mean_similarity_normalized_error": float(mean_similarity_error),
        "reference_filtered_candidate_count_histogram": np.bincount(target_counts, minlength=candidate_pool + 1).tolist(),
        "random_filtered_candidate_count_histogram": np.bincount(selected_candidate_hits, minlength=candidate_pool + 1).tolist(),
        "candidate_pool_original_ranks_1based": [int(rank) for rank in actual_ranks],
        "reference_removed_candidate_rank_histogram": target_rank_hist.tolist(),
        "random_removed_candidate_rank_histogram": selected_rank_hist.tolist(),
        "removed_candidate_rank_total_variation": float(rank_tv),
        "reference_removed_candidate_similarity_histogram": target_sim_hist.tolist(),
        "random_removed_candidate_similarity_histogram": selected_sim_hist.tolist(),
        "removed_candidate_similarity_bin_edges": [float(x) for x in similarity_edges],
        "removed_candidate_similarity_total_variation": float(sim_tv),
        "profile_quotas": profile_summary,
    }
    matched_mask = torch.zeros(num_identities, dtype=torch.bool)
    matched_mask[torch.from_numpy(selected_ids.copy()).long()] = True
    return matched_mask, impact_diagnostics


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
                                      length_bounds=(8, 16, 32, 64),
                                      group_by_camera=True,
                                      group_by_length=True):
    """Exclude low-reliability IDs using selectable threshold group axes.

    Identity-level camera consensus is unchanged. The flags only control which
    metadata axes define the per-tracklet reliability quantile groups.
    """
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
        camera_group = int(camera) if group_by_camera else None
        length_group = length_bin if group_by_length else None
        groups.setdefault((camera_group, length_group), []).append(index)
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
    grouped_thresholds = {}
    for (camera_group, length_group), indices in groups.items():
        camera_key = str(camera_group) if camera_group is not None else "all"
        length_key = str(length_group) if length_group is not None else "all"
        grouped_thresholds.setdefault(camera_key, {})[length_key] = {
            "threshold": thresholds[(camera_group, length_group)],
            "tracklet_count": len(indices),
            "low_tracklet_count": sum(low[index] for index in indices),
        }

    diagnostics = {
        "excluded_identity_count": int(excluded.sum()),
        "num_cameras": len(cameras),
        "camera_labels": cameras,
        "group_by_camera": bool(group_by_camera),
        "group_by_length": bool(group_by_length),
        "threshold_group_count": len(groups),
        "min_observed_cameras": min_observed_cameras,
        "min_low_camera_fraction": min_low_camera_fraction,
        "identity_camera_coverage_counts": coverage_counts,
        "identity_low_camera_counts": low_counts,
        "length_bounds": list(length_bounds),
        "thresholds_by_camera_and_length": grouped_thresholds,
    }
    return excluded, diagnostics


def platform_labels_from_cameras(camids, camera_to_platform):
    """Derive sampling-only platform labels without replacing camera IDs."""
    if not camera_to_platform:
        raise ValueError("Platform stratification requires DYNAMIC_DFGS_CAMERA_TO_PLATFORM")
    if any(type(platform) is not int or platform not in (0, 1)
           for platform in camera_to_platform):
        raise ValueError("Platform mapping must contain integers 0=ground or 1=aerial")
    labels = []
    for camera in camids:
        camera = int(camera)
        if camera < 0 or camera >= len(camera_to_platform):
            raise ValueError("No platform mapping for camera {}".format(camera))
        labels.append(camera_to_platform[camera])
    if set(labels) != {0, 1}:
        raise ValueError("Platform stratification requires both ground and aerial training tracklets; "
                         "use camera stratification for a single-platform dataset")
    return labels


def stratified_platform_exclusions(tracklet_reliabilities, pids, camids,
                                   frame_counts, num_identities,
                                   low_quality_fraction, camera_to_platform,
                                   length_bounds=(8, 16, 32, 64)):
    """Exclude only when an ID has a low tracklet on BOTH observed platforms.

    Thresholds are estimated within platform/length groups. An ID observed on
    only one platform is kept because it provides no cross-platform consensus.
    """
    platform_ids = platform_labels_from_cameras(camids, camera_to_platform)
    if not bool(torch.isfinite(torch.as_tensor(tracklet_reliabilities)).all()):
        raise ValueError("Non-finite temporal reliability in platform filtering")
    excluded, camera_diagnostics = stratified_multicamera_exclusions(
        tracklet_reliabilities, pids, platform_ids, frame_counts, num_identities,
        low_quality_fraction, min_observed_cameras=2, min_low_camera_fraction=1.0,
        length_bounds=length_bounds)
    diagnostics = {
        "excluded_identity_count": camera_diagnostics["excluded_identity_count"],
        "platform_labels": camera_diagnostics["camera_labels"],
        "platform_names": {"0": "ground", "1": "aerial"},
        "camera_to_platform": {str(camera): camera_to_platform[camera]
                               for camera in sorted(set(map(int, camids)))},
        "identity_platform_coverage_counts": camera_diagnostics["identity_camera_coverage_counts"],
        "identity_low_platform_counts": camera_diagnostics["identity_low_camera_counts"],
        "single_platform_identity_count": camera_diagnostics["identity_camera_coverage_counts"].get(1, 0),
        "length_bounds": camera_diagnostics["length_bounds"],
        "thresholds_by_platform_and_length": camera_diagnostics["thresholds_by_camera_and_length"],
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
    with preserve_rng_state(cfg.DATALOADER.DYNAMIC_DFGS_PRESERVE_RNG_STATE):
        return _refresh_dynamic_dfgs_graph(cfg, model, sampler, epoch, logger)


def _refresh_dynamic_dfgs_graph(cfg, model, sampler, epoch, logger):
    """Extract current-model tracklets, rebuild the graph, and update sampler."""
    if not hasattr(sampler, "update_neighbor_graph"):
        raise TypeError("Dynamic DFGS requires IdentityBatchSamplerVideo")
    graph_readout = cfg.DATALOADER.DYNAMIC_DFGS_READOUT
    if graph_readout not in ('native', 'mean'):
        raise ValueError('DYNAMIC_DFGS_READOUT must be native or mean')
    # One per-call override is shared by the center clip and both reliability
    # views. Never change the persistent training/retrieval aggregation mode.
    graph_kwargs = {'use_mean_readout': True} if graph_readout == 'mean' else {}
    train_data = sampler.data_source
    transform = ST.Compose([
        ST.Scale(cfg.INPUT.SIZE_TEST, interpolation=3),
        ST.ToTensor(),
        ST.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
    ])
    reliable_negatives = bool(cfg.DATALOADER.DYNAMIC_DFGS_RELIABLE_NEGATIVES)
    quality_weighted = bool(cfg.DATALOADER.DYNAMIC_DFGS_QUALITY_WEIGHTED)
    soft_negatives = bool(cfg.DATALOADER.DYNAMIC_DFGS_SOFT_NEGATIVES)
    stratified = bool(cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA)
    stratified_multicamera = bool(cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_MULTICAMERA)
    stratified_platform = bool(cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_PLATFORM)
    group_by_camera = bool(cfg.DATALOADER.DYNAMIC_DFGS_GROUP_BY_CAMERA)
    group_by_length = bool(cfg.DATALOADER.DYNAMIC_DFGS_GROUP_BY_LENGTH)
    random_matched_filter = bool(cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER)
    match_candidate_impact = bool(
        cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_CANDIDATE_IMPACT)
    candidate_match_iterations = int(
        cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_ITERATIONS)
    candidate_match_restarts = int(
        cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_MATCH_RESTARTS)
    random_filter_reference_dir = str(
        cfg.DATALOADER.DYNAMIC_DFGS_RANDOM_FILTER_REFERENCE_DIR).strip()
    if sum((stratified, stratified_multicamera, stratified_platform)) > 1:
        raise ValueError("Choose only one of two-camera, multicamera, or platform stratification")
    if (stratified or stratified_multicamera or stratified_platform) and not reliable_negatives:
        raise ValueError("Stratified filtering requires reliable negatives")
    if stratified_platform:
        # Validate before performing a potentially expensive GPU extraction.
        platform_labels_from_cameras(
            [camera for _, _, camera in train_data],
            cfg.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM)
    if reliable_negatives and quality_weighted:
        raise ValueError("Reliable negative filtering requires unweighted D1 identity centers")
    if soft_negatives and (not reliable_negatives or
                           not cfg.DATALOADER.DYNAMIC_DFGS_STRATIFIED_TWO_CAMERA):
        raise ValueError("Soft negatives require D4 camera/length reliability scores")
    if random_matched_filter and (
            not reliable_negatives or not (stratified or stratified_multicamera or stratified_platform)
            or soft_negatives):
        raise ValueError(
            "Matched random filtering requires hard reliable-negative filtering "
            "with a stratified reliability mode")
    if match_candidate_impact and not random_matched_filter:
        raise ValueError(
            "Candidate-impact matching requires DYNAMIC_DFGS_RANDOM_FILTER")
    paired_reference_count = None
    paired_reference_graph = None
    paired_reference_ids = None
    identity_profiles = None
    if random_matched_filter and random_filter_reference_dir:
        paired_reference_ids, paired_reference_graph = load_paired_reference_exclusions(
            random_filter_reference_dir, epoch, len(sampler.pids))
        paired_reference_count = len(paired_reference_ids)
        identity_profiles = identity_camera_length_profiles(
            train_data, len(sampler.pids))
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
    if graph_readout == 'mean' and not hasattr(base_model, 'frame_aggregation_mode'):
        raise TypeError('Mean graph readout requires explicit video-aggregation support')
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
        "Refreshing dynamic DFGS-V graph at epoch %d (%d tracklets, quality_weighted=%s, reliable_negatives=%s, readout=%s)",
        epoch, len(feature_dataset), quality_weighted, reliable_negatives, graph_readout)
    logger.info('Dynamic DFGS readout=%s applies to identity-center clips and both temporal views; '
                'training/retrieval aggregation is not changed', graph_readout)
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
                        clips, cam_label=camera_labels, view_label=view_labels,
                        **graph_kwargs)
                    if quality_weighted:
                        frame_features = base_model(clips, get_frame_image=True)
                    if reliable_negatives:
                        first_features = base_model(
                            first_view, cam_label=camera_labels,
                            view_label=view_labels, **graph_kwargs)
                        second_features = base_model(
                            second_view, cam_label=camera_labels,
                            view_label=view_labels, **graph_kwargs)
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
    if stratified_platform and not bool(torch.isfinite(tracklet_features).all()):
        bad_indices = torch.where(~torch.isfinite(tracklet_features).all(dim=1))[0].tolist()
        raise ValueError("Non-finite dynamic graph features at tracklet indices {}".format(bad_indices[:10]))
    reliabilities = (torch.cat(all_reliabilities, dim=0)
                     if quality_weighted else None)
    num_identities = len(sampler.pids)
    centroids, identity_weight_sums = build_identity_centroids(
        tracklet_features, all_pids, num_identities, reliabilities)
    identity_minimum = identity_mean = diagnostics = None
    reliability_excluded_ids = None
    random_filter_seed = None
    candidate_impact_diagnostics = None
    if reliable_negatives:
        temporal_reliabilities = torch.cat(all_temporal_reliabilities, dim=0)
        identity_minimum, identity_mean = identity_reliability(
            temporal_reliabilities, all_pids, num_identities)
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
                cfg.DATALOADER.DYNAMIC_DFGS_MIN_LOW_CAMERA_FRACTION,
                group_by_camera=group_by_camera,
                group_by_length=group_by_length)
        elif stratified_platform:
            excluded_mask, stratified_diagnostics = stratified_platform_exclusions(
                temporal_reliabilities, all_pids, all_camids,
                all_frame_counts, num_identities,
                cfg.DATALOADER.DYNAMIC_DFGS_LOW_QUALITY_FRACTION,
                cfg.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM)
        if random_matched_filter:
            reliability_excluded_ids = torch.where(excluded_mask)[0].tolist()
            random_filter_seed = (
                (int(cfg.SOLVER.SEED) * 1000003 + int(epoch)) % (2 ** 32))
            target_excluded_count = (
                paired_reference_count if paired_reference_count is not None
                else len(reliability_excluded_ids))
            if paired_reference_ids is not None:
                if match_candidate_impact:
                    excluded_mask, candidate_impact_diagnostics = (
                        sample_candidate_impact_matched_random_exclusion_mask(
                            centroids,
                            identity_profiles,
                            paired_reference_ids,
                            random_filter_seed,
                            cfg.DATALOADER.DYNAMIC_DFGS_SKIP,
                            cfg.DATALOADER.DYNAMIC_DFGS_CANDIDATE_POOL,
                            candidate_match_iterations,
                            candidate_match_restarts))
                    profile_summary = candidate_impact_diagnostics[
                        "profile_quotas"]
                else:
                    excluded_mask, profile_summary = sample_stratified_random_exclusion_mask(
                        identity_profiles, paired_reference_ids, random_filter_seed)
                if int(excluded_mask.sum()) != target_excluded_count:
                    raise RuntimeError("Paired E4 exclusion count was not matched")
                if candidate_impact_diagnostics is not None:
                    candidate_impact_diagnostics[
                        "paired_reference_excluded_identity_count"] = (
                            paired_reference_count)
                    candidate_impact_diagnostics[
                        "paired_reference_graph"] = paired_reference_graph
                diagnostics_profile_summary = profile_summary
            else:
                if match_candidate_impact:
                    raise ValueError(
                        "Candidate-impact matching requires a paired E4 reference graph")
                excluded_mask = sample_random_exclusion_mask(
                    num_identities, target_excluded_count, random_filter_seed)
                diagnostics_profile_summary = []
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
        if stratified or stratified_multicamera or stratified_platform:
            diagnostics.update(stratified_diagnostics)
        if random_matched_filter:
            # Stratified diagnostics describe the reliability-selected E4 set;
            # the applied mask may be a different paired random set.
            diagnostics["excluded_identity_count"] = int(excluded_mask.sum())
            diagnostics["filter_mode"] = "matched_random"
            diagnostics["reliability_excluded_identity_count"] = len(
                reliability_excluded_ids)
            diagnostics["random_filter_seed"] = random_filter_seed
            if paired_reference_count is not None:
                diagnostics["paired_reference_excluded_identity_count"] = (
                    paired_reference_count)
                diagnostics["paired_reference_graph"] = paired_reference_graph
                diagnostics["paired_length_camera_profile_match"] = True
                diagnostics["paired_length_camera_profile_counts"] = (
                    diagnostics_profile_summary)
                if candidate_impact_diagnostics is not None:
                    diagnostics["candidate_impact_match"] = (
                        candidate_impact_diagnostics)
                logger.info(
                    "Strict paired random filter at epoch %d: E4 exclusions=%d, "
                    "random exclusions=%d, matched camera/length profiles=%d",
                    epoch, paired_reference_count, int(excluded_mask.sum()),
                    len(diagnostics_profile_summary))
                if candidate_impact_diagnostics is not None:
                    logger.info(
                        "Candidate-impact random filter at epoch %d: "
                        "overlap=%d/%d, mean filtered candidates E4=%.3f random=%.3f, "
                        "count-TV=%.3f rank-TV=%.3f similarity-TV=%.3f",
                        epoch,
                        candidate_impact_diagnostics[
                            "overlap_with_reference_exclusions"],
                        candidate_impact_diagnostics[
                            "maximum_random_reference_identity_overlap"],
                        candidate_impact_diagnostics[
                            "reference_mean_filtered_candidates"],
                        candidate_impact_diagnostics[
                            "random_mean_filtered_candidates"],
                        candidate_impact_diagnostics[
                            "per_anchor_filtered_count_distribution_total_variation"],
                        candidate_impact_diagnostics[
                            "removed_candidate_rank_total_variation"],
                        candidate_impact_diagnostics[
                            "removed_candidate_similarity_total_variation"])
        if soft_negatives:
            diagnostics["downweighted_identity_count"] = int(excluded_mask.sum())
            diagnostics["excluded_identity_count"] = 0
        if not soft_negatives:
            logger.info(
            "Reliable negatives: mode=%s, threshold=%s, excluded IDs=%d/%d, "
            "mean filtered candidates=%.2f, fallback neighbors=%d across %d anchors",
            ("matched_random_" if random_matched_filter else "") +
            ("stratified_two_camera" if stratified else
             "stratified_multicamera" if stratified_multicamera else
             "stratified_platform" if stratified_platform else
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
                "low-camera-counts=%s, reliability-excluded=%d/%d, "
                "applied-excluded=%d/%d, min-observed=%d, min-low-fraction=%.2f, "
                "threshold-groups=%d, group-by-camera=%s, group-by-length=%s",
                epoch, diagnostics["camera_labels"],
                diagnostics["identity_camera_coverage_counts"],
                diagnostics["identity_low_camera_counts"],
                diagnostics["reliability_excluded_identity_count"]
                if random_matched_filter else diagnostics["excluded_identity_count"],
                num_identities, diagnostics["excluded_identity_count"], num_identities,
                diagnostics["min_observed_cameras"],
                diagnostics["min_low_camera_fraction"],
                diagnostics["threshold_group_count"],
                diagnostics["group_by_camera"], diagnostics["group_by_length"])
            def group_sort_key(value):
                try:
                    return (0, int(value))
                except ValueError:
                    return (1, value)
            for camera, bins in sorted(
                    diagnostics["thresholds_by_camera_and_length"].items(),
                    key=lambda item: group_sort_key(item[0])):
                for length_bin, group in sorted(bins.items(),
                                                key=lambda item: group_sort_key(item[0])):
                    logger.info(
                        "%s group at epoch %d: camera=%s length-bin=%s "
                        "tracklets=%d low-tracklets=%d threshold=%.6f",
                        "D5" if soft_negatives else "D4", epoch, camera,
                        length_bin, group["tracklet_count"],
                        group["low_tracklet_count"], group["threshold"])
        elif stratified_platform:
            logger.info(
                "Platform reliability at epoch %d: platforms=%s, coverage=%s, "
                "low-platform-counts=%s, excluded=%d/%d, single-platform-kept=%d",
                epoch, diagnostics["platform_names"],
                diagnostics["identity_platform_coverage_counts"],
                diagnostics["identity_low_platform_counts"],
                diagnostics["excluded_identity_count"], num_identities,
                diagnostics["single_platform_identity_count"])
            for platform, bins in sorted(diagnostics["thresholds_by_platform_and_length"].items()):
                for length_bin, group in sorted(bins.items(), key=lambda item: int(item[0])):
                    logger.info(
                        "E4 platform group at epoch %d: platform=%s length-bin=%s "
                        "tracklets=%d low-tracklets=%d threshold=%.6f",
                        epoch, diagnostics["platform_names"][platform], length_bin,
                        group["tracklet_count"], group["low_tracklet_count"], group["threshold"])
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
                "graph_feature_readout": graph_readout,
                "graph_feature_stage": "post_adapter_pre_bn",
                "graph_feature_dimension": int(tracklet_features.shape[1]),
                "temporal_reliability_readout": graph_readout if reliable_negatives else None,
                "soft_negatives": soft_negatives,
                "random_matched_filter": random_matched_filter,
                "random_filter_match_candidate_impact": match_candidate_impact,
                "random_filter_reference_dir": random_filter_reference_dir,
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
                    "two-view cosine, platform/length-stratified, both platforms low"
                    if stratified_platform else
                    "min tracklet cosine across two deterministic temporal views"),
                "stratified_two_camera": stratified,
                "stratified_multicamera": stratified_multicamera,
                "stratified_platform": stratified_platform,
            })
            if stratified_platform:
                payload["metadata"]["camera_to_platform"] = list(
                    cfg.DATALOADER.DYNAMIC_DFGS_CAMERA_TO_PLATFORM)
            payload["filter_diagnostics"] = diagnostics
            payload["tracklet_temporal_reliability"] = [
                float(value) for value in temporal_reliabilities.tolist()]
            payload["identity_reliability_min"] = [
                float(value) for value in identity_minimum.tolist()]
            payload["identity_reliability_mean"] = [
                float(value) for value in identity_mean.tolist()]
            if stratified or stratified_multicamera or stratified_platform:
                if soft_negatives:
                    payload["downweighted_identity_ids"] = low_candidate_ids
                else:
                    payload["excluded_identity_ids"] = torch.where(excluded_mask)[0].tolist()
                    if random_matched_filter:
                        payload["reliability_excluded_identity_ids"] = (
                            reliability_excluded_ids)
                        payload["metadata"]["random_filter_seed"] = random_filter_seed
                        if paired_reference_count is not None:
                            payload["metadata"]["paired_reference_epoch"] = epoch
                            payload["metadata"]["paired_reference_excluded_identity_count"] = (
                                paired_reference_count)
                            payload["metadata"]["paired_reference_graph"] = (
                                paired_reference_graph)
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
