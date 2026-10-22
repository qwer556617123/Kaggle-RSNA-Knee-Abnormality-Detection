"""CPU-tested research primitive; no admission or scientific eligibility implied.

The original all-head weak-label BCE is unchanged. Explicit training-only
textual cells add a per-head normalized auxiliary term; unknown cells and
unsupported heads contribute exactly zero gradient to that term.
"""
import torch
from torch.nn import functional as F
from .constants import LABELS

SUPPORTED_HEADS=('ACL','MCL','Medial Meniscus','Lateral Meniscus','Effusion',"Baker's")
AUXILIARY_WEIGHT=0.2

def encode_states(states, heads=SUPPORTED_HEADS):
    if len(set(heads))!=len(heads) or not set(heads)<=set(LABELS):
        raise ValueError('invalid supported heads')
    target=torch.zeros((len(states),len(LABELS)),dtype=torch.float32)
    mask=torch.zeros_like(target,dtype=torch.bool)
    for row, values in enumerate(states):
        if set(values)!=set(LABELS): raise ValueError('exact 12 textual states required')
        if not set(values.values())<={'positive','negative','unknown'}: raise ValueError('invalid state')
        for name in heads:
            column=LABELS.index(name)
            state=values[name]
            if state!='unknown':
                mask[row,column]=True
                target[row,column]=float(state=='positive')
    return target,mask

def auxiliary_bce(logits, targets, mask):
    if logits.ndim!=2 or logits.shape[1]!=len(LABELS) or targets.shape!=logits.shape or mask.shape!=logits.shape:
        raise ValueError('expected aligned batch by 12 tensors')
    if mask.dtype!=torch.bool: raise ValueError('boolean mask required')
    if not torch.isfinite(logits).all(): raise ValueError('nonfinite logits')
    selected=targets[mask]
    if not torch.isfinite(selected).all() or not ((selected==0)|(selected==1)).all():
        raise ValueError('explicit labels must be finite binary values')
    # Sanitize unknown labels BEFORE BCE so NaNs cannot survive a zero mask.
    safe_targets=torch.where(mask,targets,torch.zeros_like(targets)).to(logits.dtype)
    cell=F.binary_cross_entropy_with_logits(logits,safe_targets,reduction='none')
    counts=mask.sum(dim=0)
    active=counts>0
    if not active.any(): return logits.sum()*0
    means=(cell*mask).sum(dim=0)/counts.clamp_min(1)
    return means[active].mean()

def combined_loss(logits, weak_targets, report_targets, report_mask):
    main=F.binary_cross_entropy_with_logits(logits,weak_targets)
    auxiliary=auxiliary_bce(logits,report_targets,report_mask)
    return main+AUXILIARY_WEIGHT*auxiliary, main, auxiliary
