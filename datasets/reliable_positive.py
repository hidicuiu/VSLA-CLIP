"""Utilities for platform-label-free reliable hard-positive sampling."""

import json
import math

import torch
import torch.nn.functional as F


SCHEMA_VERSION = 1


def compute_tracklet_statistics(frame_features):
    """Return normalized centers and temporal consistency reliabilities.

    Args:
        frame_features: Tensor shaped [N, T, D].
    """
    if frame_features.ndim != 3:
        raise ValueError("frame_features must have shape [N, T, D]")
    normalized = F.normalize(frame_features.float(), dim=-1)
    centers = F.normalize(normalized.mean(dim=1), dim=-1)
    reliability = (normalized * centers.unsqueeze(1)).sum(dim=-1).mean(dim=1)
    return centers, reliability.clamp(0.0, 1.0)


def reliable_hard_positive_score(center_a, center_b, reliability_a, reliability_b):
    """Score a same-identity pair by hardness times temporal reliability."""
    hardness = 1.0 - F.cosine_similarity(
        center_a.unsqueeze(0), center_b.unsqueeze(0), dim=-1)[0]
    pair_reliability = math.sqrt(
        max(float(reliability_a), 0.0) * max(float(reliability_b), 0.0))
    return float(hardness) * pair_reliability


def build_partner_rankings(centers, reliabilities, pids):
    """Build descending RHP candidate lists keyed by global tracklet index."""
    if len(centers) != len(reliabilities) or len(centers) != len(pids):
        raise ValueError("centers, reliabilities, and pids must have equal length")
    by_pid = {}
    for index, pid in enumerate(pids):
        by_pid.setdefault(int(pid), []).append(index)

    partners = {}
    for indices in by_pid.values():
        for anchor in indices:
            scored = []
            for candidate in indices:
                if candidate == anchor:
                    continue
                score = reliable_hard_positive_score(
                    centers[anchor], centers[candidate],
                    reliabilities[anchor], reliabilities[candidate])
                scored.append({"index": candidate, "score": score})
            scored.sort(key=lambda item: (-item["score"], item["index"]))
            partners[str(anchor)] = scored
    return partners


def load_rhp_graph(path, data_source):
    """Load and validate an RHP graph against the current training split."""
    with open(path, "r", encoding="utf-8") as graph_file:
        payload = json.load(graph_file)
    metadata = payload.get("metadata", {})
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "Unsupported RHP graph schema version: {}".format(
                metadata.get("schema_version")))
    expected = len(data_source)
    actual = metadata.get("num_tracklets")
    if actual != expected:
        raise ValueError(
            "RHP graph has {} tracklets, but the current training split has {}"
            .format(actual, expected))

    index_to_pid = {index: int(item[1]) for index, item in enumerate(data_source)}
    raw_partners = payload.get("partners")
    if not isinstance(raw_partners, dict):
        raise ValueError("RHP graph is missing the 'partners' mapping")
    graph_anchors = {int(anchor) for anchor in raw_partners}
    missing_anchors = set(index_to_pid).difference(graph_anchors)
    if missing_anchors:
        preview = sorted(missing_anchors)[:5]
        raise ValueError(
            "RHP graph is missing {} anchor tracklets, including {}".format(
                len(missing_anchors), preview))

    partners = {}
    for raw_anchor, candidates in raw_partners.items():
        anchor = int(raw_anchor)
        if anchor not in index_to_pid:
            raise ValueError("RHP graph anchor index {} is out of range".format(anchor))
        valid = []
        for item in candidates:
            candidate = int(item["index"] if isinstance(item, dict) else item)
            if candidate not in index_to_pid:
                raise ValueError(
                    "RHP graph candidate index {} is out of range".format(candidate))
            if candidate == anchor:
                raise ValueError("RHP graph contains a self-pair at index {}".format(anchor))
            if index_to_pid[candidate] != index_to_pid[anchor]:
                raise ValueError(
                    "RHP graph pair ({}, {}) crosses identity labels".format(
                        anchor, candidate))
            valid.append(candidate)
        partners[anchor] = valid
    return partners, metadata
