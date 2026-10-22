"""Training-only, bounded class weighting for a fixed weak-label experiment."""
import numpy as np
import torch
from torch.nn import functional as F

_weights = None


def fit_weights(targets, confidence):
    y=np.asarray(targets,dtype=np.float64);c=np.asarray(confidence,dtype=np.float64)
    if y.ndim!=2 or y.shape[1]!=12 or y.shape!=c.shape or len(y)<1:
        raise ValueError('expected aligned training rows by twelve targets')
    if not np.isfinite(y).all() or not np.isfinite(c).all() or ((y<0)|(y>1)|(c<0)|(c>1)).any():
        raise ValueError('invalid targets or confidence')
    positive=(c*y).sum(0);negative=(c*(1-y)).sum(0)
    if (positive<=0).any() or (negative<=0).any():raise ValueError('both classes require training support')
    weight=np.clip(np.sqrt(negative/positive),1.,3.)
    normalizer=(positive*weight+negative)/(positive+negative)
    return dict(positive_mass=positive.tolist(),negative_mass=negative.tolist(),
                positive_weight=weight.tolist(),normalizer=normalizer.tolist(),
                formula='clip(sqrt(negative_mass/positive_mass),1,3); normalize expected training mass per head')


def configure(targets, confidence):
    global _weights
    receipt=fit_weights(targets,confidence)
    _weights=(receipt['positive_weight'],receipt['normalizer'])
    return receipt


def soft_weighted_bce(logits, targets, confidence, loss_mask=None):
    if _weights is None:raise RuntimeError('configure with frozen TRAIN labels before training')
    if logits.shape!=targets.shape or targets.shape!=confidence.shape:raise ValueError('loss shape mismatch')
    if loss_mask is None:loss_mask=torch.ones_like(confidence)
    if loss_mask.shape!=logits.shape:raise ValueError('mask shape mismatch')
    for value in (targets,confidence,loss_mask):
        if not torch.isfinite(value).all() or ((value<0)|(value>1)).any():raise ValueError('invalid labels or weights')
    positive=logits.new_tensor(_weights[0]);normalizer=logits.new_tensor(_weights[1])
    terms=(positive*targets*F.softplus(-logits)+(1-targets)*F.softplus(logits))/normalizer
    return (terms*confidence*loss_mask).mean()
