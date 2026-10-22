"""Deterministic clean research trainer for an already accepted cache."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .constants import CONFIDENCE_COLUMNS, LABELS, LOSS_MASK_COLUMNS
from .balanced_soft_loss import soft_weighted_bce
from .research_roi import apply_roi_crops, validate_window_metadata
from .research_validation import EVAL_COLUMNS, MASK_COLUMNS, UID, masked_auc, nested_partition


@dataclass(frozen=True)
class TrainRecipe:
    epochs: int = 3
    batch_size: int = 1
    accumulation_steps: int = 8
    encoder_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = .01
    warmup_fraction: float = .10
    max_grad_norm: float = 1.0
    seed: int = 20260905
    amp: bool = True


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class CachedStudyDataset(Dataset):
    def __init__(self, cache_root, frame, label_table, rows, *, training=False, crop_arm='global', auxiliary_by_uid=None):
        if crop_arm not in {'global', 'center', 'anatomic'}:
            raise ValueError('unknown cache crop arm')
        roots = list(cache_root) if isinstance(cache_root, (list, tuple)) else [cache_root]
        self.roots = [Path(root) for root in roots]
        self.root, self.frame = self.roots[0], frame.iloc[np.asarray(rows)].reset_index(drop=True)
        labels = label_table.set_index(UID)
        if not self.frame[UID].isin(labels.index).all():
            raise ValueError('cache dataset label UID mismatch')
        self.targets = labels.loc[self.frame[UID], LABELS].to_numpy(np.float32)
        self.confidence = labels.loc[self.frame[UID], CONFIDENCE_COLUMNS].to_numpy(np.float32)
        present_masks = [column for column in LOSS_MASK_COLUMNS if column in labels.columns]
        if present_masks and len(present_masks) != len(LOSS_MASK_COLUMNS):
            raise ValueError('loss-mask columns must be present for every target or none')
        self.loss_mask = (labels.loc[self.frame[UID], LOSS_MASK_COLUMNS].to_numpy(np.float32)
                          if present_masks else np.ones_like(self.confidence, dtype=np.float32))
        if (not np.isfinite(self.loss_mask).all() or (self.loss_mask < 0).any()
                or (self.loss_mask > 1).any()):
            raise ValueError('invalid target loss mask')
        index = {}
        for root_id, root in enumerate(self.roots):
            part = json.loads((root/'cache_index.json').read_text())
            if set(index) & set(part):
                raise ValueError('duplicate study across mounted cache parts')
            index.update({uid: dict(location, root_id=root_id) for uid, location in part.items()})
        self.locations = [index[str(uid)] for uid in self.frame[UID]]
        self._images, self._masks = {}, {}
        self.training = training
        self.crop_arm = crop_arm
        self.auxiliary_by_uid = auxiliary_by_uid or {}
        self._metadata = self._load_metadata(index) if crop_arm != 'global' else None

    def _load_metadata(self, index):
        metadata = {}
        for root_id, root in enumerate(self.roots):
            for path in sorted(root.glob('metadata_*.jsonl')):
                for line in path.read_text(encoding='utf-8').splitlines():
                    record = json.loads(line)
                    uid = str(record.get(UID, ''))
                    location = index.get(uid)
                    if location is None or location['root_id'] != root_id:
                        raise ValueError('ROI metadata has a UID outside cache index')
                    if int(record.get('row', -1)) != int(location['row']):
                        raise ValueError('ROI metadata row/index mismatch')
                    if uid in metadata:
                        raise ValueError('duplicate ROI metadata UID')
                    windows = record.get('metadata')
                    validate_window_metadata(windows)
                    metadata[uid] = windows
        expected = set(map(str, self.frame[UID]))
        if not expected <= set(metadata):
            raise ValueError('ROI cache metadata missing study provenance')
        return metadata

    def __len__(self): return len(self.frame)

    def _arrays(self, shard, root_id=0):
        key = (root_id, shard)
        if key not in self._images:
            root = self.roots[root_id]
            self._images[key] = np.load(root/f'images_{int(shard):03d}.npy', mmap_mode='r')
            self._masks[key] = np.load(root/f'mask_{int(shard):03d}.npy', mmap_mode='r')
        return self._images[key], self._masks[key]

    def __getitem__(self, index):
        location = self.locations[index]
        images, masks = self._arrays(location['shard'], location['root_id'])
        selected = np.arange(24)
        if self.training:
            selected = []
            offset = 0
            for pool, count in zip((4,6,6,4,4), (2,3,3,2,2)):
                selected.extend((offset + np.sort(np.random.choice(pool, count, replace=False))).tolist())
                offset += pool
        pixels = np.asarray(images[location['row']][selected], dtype=np.float32) / 255.
        if self.crop_arm != 'global':
            _ = self._metadata[str(self.frame.iloc[index][UID])]
            pixels = apply_roi_crops(pixels, np.asarray(selected), self.crop_arm)
        if self.training:
            pixels = np.clip(pixels * np.random.uniform(.9, 1.1), 0, 1)
        x = torch.from_numpy(pixels).sub(.5).div(.5)
        mask = torch.from_numpy(np.asarray(masks[location['row']][selected] > .5, dtype=bool))
        if not mask.any():
            raise RuntimeError(f'cache study has no valid windows: {self.frame.iloc[index][UID]}')
        slots = np.repeat(np.arange(5, dtype=np.int64), (4, 6, 6, 4, 4))[selected]
        depths = np.concatenate([
            np.linspace(0., 1., count, dtype=np.float32) if count > 1 else np.zeros(1, dtype=np.float32)
            for count in (4, 6, 6, 4, 4)
        ])[selected]
        laterality = {'L': 0, 'R': 1}.get(str(location.get('laterality', 'unknown')).upper(), 2)
        return (x, mask, torch.from_numpy(slots), torch.from_numpy(depths),
                torch.tensor(laterality, dtype=torch.long), torch.from_numpy(self.targets[index]),
                torch.from_numpy(self.confidence[index]), torch.from_numpy(self.loss_mask[index]),
                torch.from_numpy(np.asarray(self.auxiliary_by_uid.get(str(self.frame.iloc[index][UID]), (np.zeros(len(LABELS), np.float32), np.zeros(len(LABELS), np.float32)))[0], dtype=np.float32)),
                torch.from_numpy(np.asarray(self.auxiliary_by_uid.get(str(self.frame.iloc[index][UID]), (np.zeros(len(LABELS), np.float32), np.zeros(len(LABELS), np.float32)))[1], dtype=np.float32)),
                self.frame.iloc[index][UID])


def _collate(batch):
    images, masks, slots, depths, laterality, targets, confidence, loss_mask, auxiliary_values, auxiliary_masks, uids = zip(*batch)
    return (torch.stack(images), torch.stack(masks), torch.stack(slots), torch.stack(depths),
            torch.stack(laterality), torch.stack(targets), torch.stack(confidence),
            torch.stack(loss_mask), torch.stack(auxiliary_values), torch.stack(auxiliary_masks), list(uids))


def _autocast(device, enabled):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if enabled and device.type == 'cuda' else nullcontext()


def _model_input_size(model):
    backbone = getattr(model, 'backbone', None)
    cfg = getattr(backbone, 'default_cfg', {}) if backbone is not None else {}
    size = cfg.get('input_size') if isinstance(cfg, dict) else getattr(cfg, 'input_size', None)
    if isinstance(size, (tuple, list)) and len(size) == 3 and int(size[1]) == int(size[2]):
        return int(size[1])
    return None


def _prepare_images(model, images):
    target = _model_input_size(model)
    if target is not None and images.shape[-1] != target:
        images = torch.nn.functional.interpolate(
            images.flatten(0, 1), size=(target, target), mode='bilinear', align_corners=False
        ).reshape(images.shape[0], images.shape[1], images.shape[2], target, target)
    return images


def _forward_model(model, images, masks, slot_ids=None, depth=None, laterality=None):
    parameters = inspect.signature(model.forward).parameters
    if 'window_mask' in parameters:
        kwargs = {'window_mask': masks}
        if 'slot_ids' in parameters: kwargs['slot_ids'] = slot_ids
        if 'depth' in parameters: kwargs['depth'] = depth
        if 'laterality' in parameters: kwargs['laterality'] = laterality
        return model(images, **kwargs)
    if 'masks' in parameters:
        return model(images, masks)
    return model(images)


def _scheduler(optimizer, total_steps, warmup_fraction):
    warmup = max(1, int(total_steps * warmup_fraction))
    def factor(step):
        if step < warmup: return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _evaluate(model, loader, rows, frame, device, *, score=True):
    model.eval(); predictions, uids = [], []
    with torch.inference_mode():
        for images, masks, slots, depths, laterality, _, _, _, _, _, batch_uids in loader:
            images = _prepare_images(model, images)
            with _autocast(device, True):
                logits = _forward_model(model, images.to(device), masks.to(device),
                                        slots.to(device), depths.to(device), laterality.to(device))
            predictions.append(logits.float().sigmoid().cpu().numpy()); uids.extend(batch_uids)
    prediction = pd.DataFrame(np.concatenate(predictions), columns=LABELS)
    prediction.insert(0, UID, uids)
    if not score:
        return prediction, None
    expected = frame.iloc[np.asarray(rows)][[UID,'fold',*EVAL_COLUMNS,*MASK_COLUMNS]].copy()
    aligned = expected.set_index(UID).loc[prediction[UID]].reset_index()
    truth, mask = aligned[EVAL_COLUMNS].to_numpy(float), aligned[MASK_COLUMNS].to_numpy(bool)
    auc = masked_auc(truth, prediction[LABELS].to_numpy(), mask)
    return prediction, auc


def train_fold(model, frame, label_table, cache_root, fold, output_dir, *, recipe=TrainRecipe(), device=None, crop_arm='global', auxiliary_by_uid=None, auxiliary_weight=0.0, max_seconds=4.25*3600):
    """Train one outer fold; outer predictions are produced only after inner selection."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(recipe.seed + fold)
    started = time.monotonic()
    def check_budget():
        if time.monotonic() - started > max_seconds:
            raise TimeoutError('fold exceeded declared recipe budget; incomplete, no completed checkpoint')
    split = nested_partition(frame, fold)
    train_set = CachedStudyDataset(cache_root, frame, label_table, split.train, training=True, crop_arm=crop_arm, auxiliary_by_uid=auxiliary_by_uid)
    inner_set = CachedStudyDataset(cache_root, frame, label_table, split.inner, crop_arm=crop_arm)
    outer_set = CachedStudyDataset(cache_root, frame, label_table, split.outer, crop_arm=crop_arm)
    train_loader = DataLoader(train_set, batch_size=recipe.batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    inner_loader = DataLoader(inner_set, batch_size=recipe.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    outer_loader = DataLoader(outer_set, batch_size=recipe.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    model.to(device)
    groups = [[], []]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad: groups[0 if (name.startswith('backbone.') or '.backbone.' in name) else 1].append(parameter)
    optimizer = torch.optim.AdamW([{'params': groups[0], 'lr': recipe.encoder_lr}, {'params': groups[1], 'lr': recipe.head_lr}], weight_decay=recipe.weight_decay)
    total_steps = recipe.epochs * math.ceil(len(train_loader)/recipe.accumulation_steps)
    scheduler = _scheduler(optimizer, total_steps, recipe.warmup_fraction)
    scaler = torch.amp.GradScaler('cuda', enabled=recipe.amp and device.type == 'cuda')
    best_auc, best_epoch, best_state, inner_history = -np.inf, 0, None, []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, recipe.epochs+1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        running = []
        for step, (images, masks, slots, depths, laterality, targets, confidence, loss_mask, auxiliary_values, auxiliary_masks, _) in enumerate(train_loader, 1):
            check_budget()
            group_start = ((step-1) // recipe.accumulation_steps) * recipe.accumulation_steps
            group_size = min(recipe.accumulation_steps, len(train_loader) - group_start)
            images = _prepare_images(model, images)
            with _autocast(device, recipe.amp):
                loss = soft_weighted_bce(
                    _forward_model(model, images.to(device), masks.to(device), slots.to(device),
                                   depths.to(device), laterality.to(device)),
                    targets.to(device), confidence.to(device), loss_mask.to(device)
                )
                if auxiliary_weight > 0:
                    if not hasattr(model, 'auxiliary_loss'):
                        raise ValueError('auxiliary_weight requires a model with auxiliary_loss')
                    loss = loss + auxiliary_weight * model.auxiliary_loss(auxiliary_values.to(device), auxiliary_masks.to(device))
                loss = loss / group_size
            if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
            scaler.scale(loss).backward(); running.append(float(loss.detach().cpu()))
            if step % recipe.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.max_grad_norm, error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        _, inner_auc = _evaluate(model, inner_loader, split.inner, frame, device)
        check_budget()
        if not np.isfinite(inner_auc).all():
            raise ValueError('inner macro requires all twelve targets supported')
        macro = float(inner_auc.mean())
        print(json.dumps({'event': 'inner_epoch', 'fold': fold, 'epoch': epoch,
                          'inner_macro_auc': macro, 'elapsed_seconds': time.monotonic()-started}), flush=True)
        inner_history.append({'epoch': epoch, 'loss': float(np.mean(running)), 'macro_auc': macro, 'per_label_auc': dict(zip(LABELS, inner_auc.tolist()))})
        if np.isfinite(macro) and macro > best_auc:
            best_auc, best_epoch, best_state = macro, epoch, copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError('no finite inner checkpoint')
    model.load_state_dict(best_state)
    outer_prediction, _ = _evaluate(model, outer_loader, split.outer, frame, device, score=False)
    check_budget()
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/f'fold_{fold}.pt'; torch.save({'model': model.state_dict(), 'fold': fold, 'selected_epoch': best_epoch, 'inner_history': inner_history}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inner_prediction, _ = _evaluate(model, inner_loader, split.inner, frame, device)
    outer_prediction.insert(1, 'fold', fold)
    inner_prediction.insert(1, 'fold', fold)
    outer_prediction.to_csv(output/f'outer_{fold}.csv', index=False)
    inner_prediction.to_csv(output/f'inner_{fold}.csv', index=False)
    return {'fold': fold, 'status': 'completed', 'selected_epoch': best_epoch, 'inner_macro_auc': best_auc,
            'checkpoint_sha256': digest, 'inner_history': inner_history, 'decode_failures': 0,
            'train_count': len(split.train), 'inner_count': len(split.inner), 'outer_count': len(split.outer)}


def _uid_hash(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def run_three_fold(model_factory, frame, label_table, cache_root, output_dir, run_spec, *, recipe=TrainRecipe(), device=None, crop_arm='global'):
    """Run all clean outer folds and write a validator-compatible manifest."""
    required = ('arm', 'contract_version', 'evaluation_sha256', 'fold_sha256',
                'pretrain_sha256', 'preprocessing_sha256', 'code_version')
    missing = [key for key in required if not run_spec.get(key)]
    if missing:
        raise ValueError(f'missing frozen run spec fields: {missing}')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(run_spec)
    manifest.update({
        'status': 'running', 'checkpoint_selection': 'inner_only',
        'gold_role': 'excluded_with_report_groups',
        'recipe': asdict(recipe), 'recipe_sha256': _json_hash(asdict(recipe)),
        'folds': {}, 'created_by': 'rsna_knee.research_trainer.run_three_fold',
    })
    manifest['run_manifest_sha256'] = _json_hash(
        {k: v for k, v in manifest.items() if k != 'run_manifest_sha256'}
    )
    (output/'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
    )
    outer_frames = []
    for fold in range(3):
        seed_everything(recipe.seed + fold)
        model = model_factory(fold)
        result = train_fold(model, frame, label_table, cache_root, fold, output,
                            recipe=recipe, device=device, crop_arm=crop_arm)
        partition = nested_partition(frame, fold)
        manifest['folds'][str(fold)] = {
            'train_uid_sha256': _uid_hash(frame.iloc[partition.train][UID]),
            'inner_uid_sha256': _uid_hash(frame.iloc[partition.inner][UID]),
            'outer_uid_sha256': _uid_hash(frame.iloc[partition.outer][UID]),
            'excluded_uid_sha256': _uid_hash(frame.iloc[partition.excluded][UID]),
            'checkpoint_sha256': result['checkpoint_sha256'],
            'selected_epoch': result['selected_epoch'],
            'decode_failures': result['decode_failures'],
            'train_count': result['train_count'],
            'inner_count': result['inner_count'],
            'outer_count': result['outer_count'],
        }
        outer_frames.append(pd.read_csv(output/f'outer_{fold}.csv', dtype={UID: str}))
        manifest['folds_completed'] = fold + 1
        manifest['run_manifest_sha256'] = _json_hash(
            {k: v for k, v in manifest.items() if k != 'run_manifest_sha256'}
        )
        (output/'run_manifest.json').write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
        )
    expected_rows = np.concatenate([nested_partition(frame, f).outer for f in range(3)])
    expected = frame.iloc[np.sort(expected_rows)][UID].astype(str)
    oof = pd.concat(outer_frames, ignore_index=True)
    if len(oof) != len(expected) or set(oof[UID]) != set(expected):
        raise ValueError('three-fold OOF is incomplete or contains excluded studies')
    oof = oof.set_index(UID).loc[expected].reset_index()
    oof.to_csv(output/'oof.csv', index=False)
    manifest['oof_sha256'] = hashlib.sha256((output/'oof.csv').read_bytes()).hexdigest()
    manifest['status'] = 'completed'
    manifest['run_manifest_sha256'] = _json_hash(
        {k: v for k, v in manifest.items() if k != 'run_manifest_sha256'}
    )
    (output/'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8'
    )
    return manifest
