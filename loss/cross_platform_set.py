"""Multi-positive supervised contrast over two-head video sets (training only)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CrossPlatformSetLoss(nn.Module):
    def __init__(self, temperature=0.1, matching='optimal'):
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Set temperature must be finite and positive')
        if matching not in ('optimal', 'fixed', 'fused'):
            raise ValueError('Set matching must be optimal, fixed or fused')
        self.temperature = temperature
        self.matching = matching

    def forward(self, features, labels, platforms):
        heads = 1 if self.matching == 'fused' else 2
        if features.ndim != 3 or features.shape[1] != heads:
            raise ValueError('Contrast features must have shape [B,{},D]'.format(heads))
        if labels.shape != features.shape[:1] or platforms.shape != labels.shape:
            raise ValueError('Identity/platform labels must have shape [B]')
        with torch.cuda.amp.autocast(enabled=False):
            z = F.normalize(features.float(), dim=-1)
            if self.matching == 'fused':
                same = z[:, 0] @ z[:, 0].t()
                swap = same  # Matching diagnostics are not applicable to a single descriptor.
            else:
                cross = torch.einsum('ihd,jkd->ijhk', z, z)
                same = (cross[:, :, 0, 0] + cross[:, :, 1, 1]) * 0.5
                swap = (cross[:, :, 0, 1] + cross[:, :, 1, 0]) * 0.5
            similarity = torch.maximum(same, swap) if self.matching == 'optimal' else same
            candidates = platforms[:, None] != platforms[None, :]
            positives = candidates & (labels[:, None] == labels[None, :])
            negatives = candidates & ~positives
            valid = positives.any(1) & negatives.any(1)
            # Select valid rows before logsumexp: empty rows must never produce NaN gradients.
            logits = similarity[valid] / self.temperature
            positive_mask = positives[valid]
            denominator = torch.logsumexp(logits.masked_fill(~candidates[valid], -torch.inf), dim=1)
            log_probability = logits - denominator[:, None]
            per_anchor = -(log_probability.masked_fill(~positive_mask, 0).sum(1)
                           / positive_mask.sum(1).clamp_min(1))
            loss = per_anchor.mean() if valid.any() else features.float().sum() * 0.0
            matched_positive = positives & valid[:, None]
            count = matched_positive.sum().clamp_min(1)
            details = {
                'valid_fraction': valid.float().mean().detach(),
                'positive_count': positives[valid].sum(1).float().mean().detach() if valid.any() else loss.detach(),
                'swap_fraction': ((swap > same) & matched_positive).sum().float().detach() / count,
                'matching_gap': ((same - swap).abs() * matched_positive).sum().detach() / count,
                'head_cosine': (z[:, 0] * z[:, 1]).sum(-1).mean().detach() if heads == 2 else loss.detach() * 0,
            }
        return loss, details


def validate_set_loss_config(cfg):
    weight = cfg.MODEL.CROSS_PLATFORM_SET_WEIGHT
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Set loss weight must be finite and nonnegative')
    CrossPlatformSetLoss(cfg.MODEL.CROSS_PLATFORM_SET_TEMPERATURE,
                         cfg.MODEL.CROSS_PLATFORM_SET_MATCHING)
    if weight:
        if (cfg.MODEL.FRAME_AGGREGATION not in ('multihead', 'uniform_fusion')
                or cfg.MODEL.FRAME_AGGREGATION_HEADS != 2
                or cfg.MODEL.FRAME_AGGREGATION_GROUPS != 1):
            raise ValueError('Set supervision requires two whole-frame aggregation heads')
        mapping = cfg.MODEL.CROSS_PLATFORM_SET_CAMERA_TO_PLATFORM
        if (not mapping or set(mapping) != {0, 1}
                or any(type(value) is not int for value in mapping)):
            raise ValueError('Set loss requires an explicit binary camera-to-platform mapping')
