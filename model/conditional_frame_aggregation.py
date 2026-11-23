"""Primary and conditionally complementary pooling of an unordered frame set.

Complementarity is a hypothesis, not guaranteed by different attention weights.
The independent/single controls retain the same residual maps, gate and live
shared identity classifier. No ground-truth label enters the pooling path.
"""
import math

import torch
from torch import nn

from .multihead_frame_aggregation import CONDITIONAL_MODES, attention_diversity


def soft_coverage_prior(primary_weights, strength):
    """Positive soft coverage prior; detach prevents manipulating head1 to escape it."""
    if not math.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError('Soft frame suppression must be in [0, 1)')
    selected = primary_weights.detach().float()
    relative = selected / selected.max(-1, keepdim=True).values.clamp_min(1e-12)
    return (1 - strength * relative).clamp_min(1e-12).log()


class ConditionalFrameAggregation(nn.Module):
    def __init__(self, num_classes, hidden_dim=64, mode='conditional',
                 branch_dims=(768, 768, 512), suppression=.5, gate_init=.1):
        super().__init__()
        if mode not in CONDITIONAL_MODES or hidden_dim < 1 or num_classes < 2:
            raise ValueError('Invalid conditional aggregation construction')
        if not math.isfinite(suppression) or not 0 <= suppression < 1:
            raise ValueError('Soft frame suppression must be in [0, 1)')
        if not math.isfinite(gate_init) or not 0 < gate_init < 1:
            raise ValueError('Initial supplement gate must be in (0, 1)')
        self.mode, self.heads = mode, 2
        self.branch_dims = tuple(branch_dims)
        if len(self.branch_dims) != 3 or min(self.branch_dims) < 1:
            raise ValueError('Expected three positive feature widths')
        self.suppression = suppression
        input_dim = self.branch_dims[1] + self.branch_dims[2]
        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.primary_scorer = self._scorer(2 * hidden_dim, hidden_dim)
        self.secondary_scorer = self._scorer(4 * hidden_dim, hidden_dim)
        # No affine offsets or residual LayerNorm: zero deviation must stay zero,
        # and arbitrarily small differences must not be normalized to unit size.
        self.residual_maps = nn.ModuleList([
            nn.ModuleList([nn.Sequential(
                nn.Linear(width, hidden_dim, bias=False), nn.GELU(),
                nn.Linear(hidden_dim, width, bias=False)) for _ in range(2)])
            for width in self.branch_dims])
        self.gate = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, 1))
        self.head_norm = nn.LayerNorm(input_dim)
        self.head_classifier = nn.Linear(input_dim, num_classes, bias=False)
        nn.init.normal_(self.head_classifier.weight, std=.001)
        for maps in self.residual_maps:
            for mapping in maps:
                nn.init.zeros_(mapping[-1].weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1 - gate_init)))
        if mode == 'conditional_single':
            # Construct all common tensors before removing the unused scorer,
            # preserving common initialization and the constructor RNG stream.
            self.secondary_scorer = None

    @staticmethod
    def _scorer(input_dim, hidden_dim):
        result = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(),
                               nn.Linear(hidden_dim, 1))
        nn.init.normal_(result[-1].weight, std=.01)
        nn.init.zeros_(result[-1].bias)
        return result

    def _validate(self, branches):
        batch, frames = branches[1].shape[:2]
        if frames < 1 or any(f.ndim != 3 or f.shape != (batch, frames, width)
                             for f, width in zip(branches, self.branch_dims)):
            raise ValueError('Frame branches must share [batch, time] and configured widths')
        return batch, frames

    def _encode(self, branches):
        return self.frame_encoder(torch.cat((branches[1].float(), branches[2].float()), -1))

    def readout(self, branches, weights, encoded=None):
        """Read a fixed frame encoding with supplied weights; used by interventions."""
        batch, frames = self._validate(branches)
        if weights.shape != (batch, 2, frames):
            raise ValueError('Conditional readout requires [batch, 2, time] weights')
        means = tuple(f.mean(1) for f in branches)
        with torch.cuda.amp.autocast(enabled=False):
            weights = weights.float()
            encoded = self._encode(branches) if encoded is None else encoded.float()
            pooled_context = torch.einsum('bht,btd->bhd', weights, encoded)
            gate_input = torch.cat((encoded.mean(1), pooled_context[:, 0],
                                    pooled_context[:, 1] - pooled_context[:, 0]), -1)
            gate = self.gate(gate_input).sigmoid()
            outputs, pooled, primary_updates, secondary_updates = [], [], [], []
            for feature, mean, maps in zip(branches, means, self.residual_maps):
                feature = feature.float()
                pooled.append(torch.einsum('bht,btd->bhd', weights, feature))
                # Algebraically sum(w*f)-mean(f), but uniform weights produce
                # exact zero, even with floating-point reductions.
                deviations = torch.einsum('bht,btd->bhd', weights - 1.0 / frames, feature)
                scale = feature.detach().square().mean((1, 2)).clamp_min(1e-12).sqrt().unsqueeze(-1)
                primary = maps[0](deviations[:, 0] / scale) * scale
                secondary = gate * maps[1](deviations[:, 1] / scale) * scale
                outputs.append((mean.float() + primary + secondary).to(mean.dtype))
                primary_updates.append(primary); secondary_updates.append(secondary)
            base = torch.cat((means[1].float(), means[2].float()), -1)
            denominator = base.norm(dim=-1).clamp_min(1e-8)
            auxiliary = dict(
                head_features=torch.cat((pooled[1], pooled[2]), -1),
                supplement_gate=gate,
                primary_residual_relative_norm=torch.cat(primary_updates[1:], -1).norm(dim=-1) / denominator,
                supplement_residual_relative_norm=torch.cat(secondary_updates[1:], -1).norm(dim=-1) / denominator)
        return tuple(outputs), auxiliary

    def forward(self, last_cls, final_cls, projection):
        branches = (last_cls, final_cls, projection)
        self._validate(branches)
        with torch.cuda.amp.autocast(enabled=False):
            encoded = self._encode(branches)
            context = encoded.mean(1, keepdim=True).expand_as(encoded)
            primary = self.primary_scorer(torch.cat((encoded, context), -1)).squeeze(-1).softmax(-1)
            if self.mode == 'conditional_single':
                secondary = primary
            else:
                if self.mode == 'conditional':
                    selected = torch.einsum('bt,btd->bd', primary, encoded).detach()
                    selected = selected.unsqueeze(1).expand_as(encoded)
                    condition, difference = selected, encoded - selected
                else:
                    condition, difference = torch.zeros_like(encoded), torch.zeros_like(encoded)
                secondary_logits = self.secondary_scorer(
                    torch.cat((encoded, context, condition, difference), -1)).squeeze(-1)
                if self.mode == 'conditional':
                    secondary_logits = secondary_logits + soft_coverage_prior(primary, self.suppression)
                secondary = secondary_logits.softmax(-1)
            weights = torch.stack((primary, secondary), 1)
            outputs, auxiliary = self.readout(branches, weights, encoded)
            auxiliary.update(weights=weights, diversity_loss=attention_diversity(weights), encoded=encoded)
            if self.training:
                auxiliary['head_logits'] = self.head_classifier(self.head_norm(auxiliary['head_features']))
        return outputs, auxiliary
