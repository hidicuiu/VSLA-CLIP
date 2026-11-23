"""Live identity/metric supervision on each pooled head, averaged across heads."""
import torch
from torch import nn
from torch.nn import functional as F

from .losses_video import TripletLoss


class FrameHeadSupervision(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.smoothing = .1 if cfg.MODEL.IF_LABELSMOOTH == 'on' else 0.
        self.id_weight = cfg.MODEL.ID_LOSS_WEIGHT
        self.triplet_weight = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        # Match loss.make_loss_video's actual E4 margin and distance, rather
        # than a different image-ReID triplet implementation/default.
        self.triplet = TripletLoss(margin=.3, distance=cfg.TEST.DISTANCE)

    def forward(self, auxiliary, targets):
        with torch.cuda.amp.autocast(enabled=False):
            logits, features = auxiliary['head_logits'].float(), auxiliary['head_features'].float()
            if logits.ndim != 3 or logits.shape[:2] != features.shape[:2] or logits.shape[1] != 2:
                raise ValueError('Expected two pooled heads and live identity predictions')
            identities, metrics = [], []
            for head in range(2):
                log_probs = F.log_softmax(logits[:, head], dim=-1)
                hard = -log_probs.gather(1, targets.unsqueeze(1)).mean()
                smooth = -log_probs.mean()
                identities.append((1-self.smoothing)*hard + self.smoothing*smooth)
                metrics.append(self.triplet(features[:, head], targets))
            identity, metric = torch.stack(identities).mean(), torch.stack(metrics).mean()
            loss = self.id_weight * identity + self.triplet_weight * metric
        return loss, dict(head_id_loss=identity, head_triplet_loss=metric)
