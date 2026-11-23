"""Set-based frame aggregation inspired by M3-ReID's attention diversity.

This is a frame-level adaptation, not a reproduction of its HW/TW/TH module.
Each head pools the three original E4 feature branches separately. Nonlinear
residual fusion preserves their widths and starts at the original mean readout.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


CONDITIONAL_MODES = ('conditional', 'conditional_independent', 'conditional_single')


def validate_frame_aggregation_config(cfg):
    mode = cfg.MODEL.FRAME_AGGREGATION
    weight = cfg.MODEL.FRAME_AGGREGATION_DIVERSITY_WEIGHT
    groups = cfg.MODEL.FRAME_AGGREGATION_GROUPS
    if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
        raise ValueError('Frame aggregation groups must be a positive integer')
    if groups > 1 and (mode not in ('multihead', 'uniform_fusion') or
                       any(width % groups for width in (768, 768, 512))):
        raise ValueError('Feature groups require ordinary multihead/uniform fusion and divisible branch widths')
    if cfg.DATALOADER.DYNAMIC_DFGS_READOUT not in ('native', 'mean'):
        raise ValueError('DYNAMIC_DFGS_READOUT must be native or mean')
    if mode not in ('mean', 'multihead', 'uniform_fusion', 'single_shared') + CONDITIONAL_MODES:
        raise ValueError('Unknown FRAME_AGGREGATION mode: ' + mode)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('Attention diversity weight must be finite and nonnegative')
    if mode != 'multihead' and weight != 0:
        raise ValueError('Attention diversity requires learned multihead aggregation')
    head_weight = cfg.MODEL.FRAME_AGGREGATION_HEAD_LOSS_WEIGHT
    if not math.isfinite(head_weight) or head_weight < 0:
        raise ValueError('Head supervision weight must be finite and nonnegative')
    if mode not in CONDITIONAL_MODES and head_weight:
        raise ValueError('Head supervision is only available for conditional aggregation controls')
    if mode in CONDITIONAL_MODES:
        if cfg.MODEL.FRAME_AGGREGATION_HEADS != 2:
            raise ValueError('Conditional aggregation requires two fusion slots')
        suppression = cfg.MODEL.FRAME_AGGREGATION_SUPPRESSION
        gate_init = cfg.MODEL.FRAME_AGGREGATION_GATE_INIT
        if not math.isfinite(suppression) or not 0 <= suppression < 1:
            raise ValueError('Soft frame suppression must be in [0, 1)')
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError('Initial supplement gate must be in (0, 1)')
        if cfg.MODEL.METRIC_LOSS_TYPE != 'triplet' or cfg.TEST.DISTANCE != 'cosine':
            raise ValueError('Conditional controls currently require the E4 cosine triplet protocol')
    if mode != 'mean':
        if mode == 'single_shared' and cfg.MODEL.FRAME_AGGREGATION_HEADS != 2:
            raise ValueError('Single shared attention requires two fusion input slots')
        if cfg.MODEL.NAME != 'ViT-B-16':
            raise ValueError('Frame aggregation currently supports ViT-B-16 only')
        if cfg.MODEL.FRAME_AGGREGATION_HEADS < 1 or cfg.MODEL.FRAME_AGGREGATION_HIDDEN_DIM < 1:
            raise ValueError('Aggregation head count and hidden width must be positive')
        if weight and cfg.MODEL.FRAME_AGGREGATION_HEADS < 2:
            raise ValueError('Attention diversity requires at least two heads')


def attention_diversity(weights):
    """Mean cosine overlap between different heads, computed in FP32.

Lower overlap encourages different selections, not statistical independence.
All-identical frames cannot be separated by a content-only shared scorer.
"""
    with torch.cuda.amp.autocast(enabled=False):
        weights = weights.float()
        if weights.ndim == 4:
            # Compare heads separately within each feature group, then average.
            weights = weights.permute(0, 2, 1, 3).flatten(0, 1)
        elif weights.ndim != 3:
            raise ValueError('Attention weights must be [B,H,T] or [B,H,G,T]')
        heads = weights.shape[1]
        if heads < 2 or weights.shape[-1] < 2:
            return weights.sum() * 0.0
        normalized = F.normalize(weights, p=2, dim=-1)
        gram = normalized @ normalized.transpose(1, 2)
        pairs = torch.triu_indices(heads, heads, offset=1, device=weights.device)
        return gram[:, pairs[0], pairs[1]].mean()


def load_original_stage1(model, state, strict=False):
    """Allow new aggregation keys only; reuse every original Stage-1 tensor."""
    if getattr(model, 'frame_aggregation_mode', 'mean') == 'mean':
        return model.load_state_dict(state, strict=strict)
    if any(key.startswith('frame_aggregation.') for key in state):
        raise ValueError('Use the original Stage-1 checkpoint, not an earlier aggregation experiment')
    loaded = model.load_state_dict(state, strict=strict)
    if loaded.unexpected_keys or any(
            not key.startswith('frame_aggregation.') for key in loaded.missing_keys):
        raise RuntimeError('Stage-1 must supply every original E4 tensor: {}'.format(loaded))
    return loaded


class MultiHeadFrameAggregation(nn.Module):
    def __init__(self, heads=2, hidden_dim=64, mode='multihead',
                 branch_dims=(768, 768, 512), groups=1):
        super().__init__()
        if mode not in ('multihead', 'uniform_fusion', 'single_shared') or heads < 1 or hidden_dim < 1:
            raise ValueError('Invalid frame aggregation construction')
        if mode == 'single_shared' and heads != 2:
            raise ValueError('Single shared attention requires two fusion input slots')
        self.mode = mode
        self.heads = heads
        self.groups = groups
        self.branch_dims = tuple(branch_dims)
        if len(self.branch_dims) != 3 or min(self.branch_dims) < 1:
            raise ValueError('Expected positive widths for last CLS, final CLS and projection')
        if (isinstance(groups, bool) or not isinstance(groups, int) or groups < 1 or
                any(width % groups for width in self.branch_dims) or
                (groups > 1 and mode not in ('multihead', 'uniform_fusion'))):
            raise ValueError('Invalid feature-group count for frame aggregation')
        input_dim = self.branch_dims[1] + self.branch_dims[2]
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, heads))
        # Small, distinct logits avoid exact head symmetry. Frame softmax has
        # fixed temperature 1 and contains no temporal position embedding.
        nn.init.normal_(self.score[-1].weight, std=0.01)
        nn.init.zeros_(self.score[-1].bias)
        self.fusions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(heads * width),
                nn.Linear(heads * width, hidden_dim, bias=False), nn.GELU(),
                nn.Linear(hidden_dim, width, bias=False))
            for width in self.branch_dims])
        for fusion in self.fusions:
            nn.init.zeros_(fusion[-1].weight)
        if mode == 'single_shared':
            # Preserve every shared/fusion tensor and the RNG stream of the
            # dual control. Keep just its first scorer row as the one learned
            # distribution; no unused second scoring head remains trainable.
            dual_score = self.score[-1]
            with torch.random.fork_rng(devices=[]):
                single_score = nn.Linear(hidden_dim, 1)
            with torch.no_grad():
                single_score.weight.copy_(dual_score.weight[:1])
                single_score.bias.copy_(dual_score.bias[:1])
            self.score[-1] = single_score

        if groups > 1:
            # Construct all original tensors first. Repeating the old score row
            # into each group preserves shared/fusion initialization and the
            # global RNG stream; group-specific gradients can then specialize.
            original_score = self.score[-1]
            with torch.random.fork_rng(devices=[]):
                grouped_score = nn.Linear(hidden_dim, heads * groups)
            with torch.no_grad():
                grouped_score.weight.copy_(original_score.weight.repeat_interleave(groups, dim=0))
                grouped_score.bias.copy_(original_score.bias.repeat_interleave(groups, dim=0))
            self.score[-1] = grouped_score

    def pool_heads(self, feature, weights):
        """Pool original channels; groups partition each branch independently."""
        if feature.ndim != 3:
            raise ValueError('Expected frame features [B,T,D]')
        batch, frames, width = feature.shape
        if weights.ndim == 3:
            if weights.shape != (batch, self.heads, frames):
                raise ValueError('Whole-frame weights must have shape [B,H,T]')
            return torch.einsum('bht,btd->bhd', weights, feature.float())
        if (weights.ndim != 4 or weights.shape != (batch, self.heads, self.groups, frames)
                or width % self.groups):
            raise ValueError('Feature-group weights must have shape [B,H,G,T]')
        grouped = feature.float().reshape(batch, frames, self.groups, width // self.groups)
        return torch.einsum('bhgt,btgd->bhgd', weights, grouped).flatten(2)

    def forward(self, last_cls, final_cls, projection):
        branches = (last_cls, final_cls, projection)
        batch, frames = final_cls.shape[:2]
        if frames < 1 or any(
                feature.ndim != 3 or feature.shape != (batch, frames, width)
                for feature, width in zip(branches, self.branch_dims)):
            raise ValueError('Frame branches must share [batch, time] and configured widths')
        # Preserve the original reduction's dtype/rounding at initialization,
        # including under mixed-precision training.
        means = tuple(feature.mean(dim=1) for feature in branches)
        with torch.cuda.amp.autocast(enabled=False):
            content = torch.cat((final_cls.float(), projection.float()), dim=-1)
            encoded = self.frame_encoder(content)
            context = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
            logits = self.score(torch.cat((encoded, context), dim=-1)).transpose(1, 2)
            if self.groups > 1:
                logits = logits.reshape(batch, self.heads, self.groups, frames)
            if self.mode in ('multihead', 'single_shared'):
                weights = logits.softmax(dim=-1)
                if self.mode == 'single_shared':
                    weights = weights.expand(-1, self.heads, -1)
            else:
                # Same fusion architecture and allocated parameters; scorer is
                # deliberately inactive in this uniform-weight control.
                weights = torch.full_like(logits, 1.0 / frames)
            outputs = []
            for feature, mean, fusion in zip(branches, means, self.fusions):
                head_features = self.pool_heads(feature, weights)
                correction = fusion(head_features.flatten(1))
                outputs.append((mean.float() + correction).to(dtype=mean.dtype))
            auxiliary = dict(diversity_loss=attention_diversity(weights), weights=weights)
        return tuple(outputs), auxiliary
