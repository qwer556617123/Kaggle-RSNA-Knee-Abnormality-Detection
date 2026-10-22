from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import CONFIDENCE_COLUMNS, LABELS, SLOT_NAMES
from .dicom import build_study_tensor, choose_series
from .modeling import (
    ExponentialMovingAverage,
    ModelConfig,
    auc_vector,
    augment_intensity,
    checkpoint_payload,
    create_study_model,
    pairwise_auc_loss,
    positive_class_weights,
    slot_dropout,
    weighted_soft_bce,
)
from .paths import iter_files_pruned


@dataclass
class CacheConfig:
    image_size: int = 196
    slices: int = 15
    group_size: int = 3
    target_fov_mm: float | tuple[float, ...] = 170.0
    slice_sampling: str = "even"
    slots: tuple[str, ...] = SLOT_NAMES

    @property
    def groups(self) -> int:
        return self.slices // self.group_size


@dataclass
class TrainConfig:
    epochs: int = 3
    warmup_epochs: int = 0
    exact_epochs: int = 0
    batch_size: int = 2
    eval_batch_size: int = 4
    accumulation_steps: int = 4
    learning_rate: float = 2e-5
    head_learning_rate: float = 2e-4
    warmup_head_learning_rate: float | None = None
    exact_learning_rate: float | None = None
    exact_head_learning_rate: float | None = None
    weight_decay: float = 0.02
    layer_decay: float = 1.0
    rank_loss_weight: float = 0.20
    gold_repeat: int = 6
    slot_dropout: float = 0.10
    ema_decay: float = 0.995
    warmup_unfreeze_last_blocks: int | None = 0
    weak_unfreeze_last_blocks: int | None = None
    exact_unfreeze_last_blocks: int | None = 1
    reliability_floor: float = 0.15
    workers: int = 2
    seed: int = 20260807


def find_competition_root(search_root: str | Path = "/kaggle/input") -> Path:
    root = Path(search_root)
    candidates = [
        path.parent for path in iter_files_pruned(root, "train_series.csv")
        if (path.parent / "train.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"competition data not found under {root}")
    return sorted(candidates, key=lambda path: len(str(path)))[0]


def find_artifact(filename: str, search_root: str | Path = "/kaggle/input") -> Path:
    candidates = list(iter_files_pruned(search_root, filename))
    if not candidates:
        raise FileNotFoundError(f"{filename} not found under {search_root}")
    return sorted(candidates, key=lambda path: len(str(path)))[0]


def build_cache(
    studies: Sequence[str],
    series_table: pd.DataFrame,
    dicom_root: str | Path,
    output_dir: str | Path,
    config: CacheConfig,
    *,
    prefix: str,
) -> tuple[np.memmap, np.ndarray, pd.DataFrame]:
    failure_columns = ["StudyInstanceUID", "SeriesInstanceUID", "slot", "error"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / f"{prefix}_images.npy"
    mask_path = output / f"{prefix}_mask.npy"
    ids_path = output / f"{prefix}_ids.json"
    config_path = output / f"{prefix}_cache_config.json"
    ids = [str(study) for study in studies]
    cache_config = asdict(config)
    shape = (
        len(ids), len(config.slots), config.groups, config.group_size,
        config.image_size, config.image_size,
    )
    if cache_path.exists() and mask_path.exists() and ids_path.exists() and config_path.exists():
        saved = json.loads(ids_path.read_text(encoding="utf-8"))
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, mmap_mode="r")
        mask = np.load(mask_path)
        if (
            saved == ids
            and saved_config == json.loads(json.dumps(cache_config))
            and cached.shape == shape
            and mask.shape == (len(ids), len(config.slots))
        ):
            return cached, mask, pd.DataFrame(columns=failure_columns)

    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.uint8, shape=shape)
    mask = np.zeros((len(ids), len(config.slots)), dtype=np.float32)
    choices = choose_series(series_table, dicom_root, slots=config.slots)
    failures: list[dict[str, str]] = []
    started = time.time()
    for index, study_uid in enumerate(ids):
        tensor, study_mask, study_failures = build_study_tensor(
            study_uid,
            choices,
            slots=config.slots,
            slices=config.slices,
            group_size=config.group_size,
            image_size=config.image_size,
            target_fov_mm=config.target_fov_mm,
            slice_sampling=config.slice_sampling,
        )
        cache[index] = tensor
        mask[index] = study_mask
        failures.extend(study_failures)
        if (index + 1) % 50 == 0 or index + 1 == len(ids):
            print(f"cache {prefix}: {index + 1}/{len(ids)} in {time.time() - started:.1f}s", flush=True)
    cache.flush()
    np.save(mask_path, mask)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    config_path.write_text(json.dumps(cache_config, sort_keys=True), encoding="utf-8")
    empty = int((mask.sum(axis=1) == 0).sum())
    if empty > max(2, int(0.01 * len(ids))):
        raise RuntimeError(f"systemic DICOM failure: {empty}/{len(ids)} studies have no decoded slots")
    return (
        np.load(cache_path, mmap_mode="r"),
        mask,
        pd.DataFrame(failures, columns=failure_columns),
    )


def make_supervision(train: pd.DataFrame, weak: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = train[["StudyInstanceUID"]].merge(weak, on="StudyInstanceUID", validate="one_to_one")
    targets = merged[LABELS].to_numpy(dtype=np.float32)
    confidence = merged[CONFIDENCE_COLUMNS].to_numpy(dtype=np.float32)
    gold = train[LABELS].notna().all(axis=1).to_numpy()
    # The label pipeline should already override gold, but enforce the invariant here too.
    targets[gold] = train.loc[gold, LABELS].to_numpy(dtype=np.float32)
    confidence[gold] = 1.0
    return targets, confidence, gold


def fold_calibrated_supervision(
    targets: np.ndarray,
    confidence: np.ndarray,
    gold: np.ndarray,
    raw_weak_targets: np.ndarray | None,
    train_indices: np.ndarray,
    *,
    floor: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate weak-label confidence using only the current fold's training gold.

    ``targets`` already has gold labels overriding weak labels.  The separately
    supplied raw scores let us measure report-label discrimination on those
    gold examples without letting a validation example influence its own fold.
    """
    calibrated_targets = np.asarray(targets, dtype=np.float32).copy()
    calibrated_confidence = np.asarray(confidence, dtype=np.float32).copy()
    reliability = np.ones(targets.shape[1], dtype=np.float32)
    if raw_weak_targets is None:
        return calibrated_targets, calibrated_confidence, reliability
    raw_weak_targets = np.asarray(raw_weak_targets, dtype=np.float32)
    if raw_weak_targets.shape != targets.shape:
        raise ValueError("raw_weak_targets must align with targets")
    train_gold = np.asarray(train_indices, dtype=np.int64)
    train_gold = train_gold[gold[train_gold]]
    if len(train_gold) == 0:
        return calibrated_targets, calibrated_confidence, np.zeros(targets.shape[1], dtype=np.float32)

    reliability = np.zeros(targets.shape[1], dtype=np.float32)
    for label in range(targets.shape[1]):
        truth = targets[train_gold, label].astype(np.int8)
        if len(np.unique(truth)) < 2:
            continue
        score = auc_vector(truth[:, None], raw_weak_targets[train_gold, label, None])[0]
        positives = int(truth.sum())
        negatives = int(len(truth) - positives)
        # AUC above chance is useful only when both classes have enough gold
        # support.  This conservative factor prevents a tiny fold from
        # declaring an otherwise noisy report rule perfectly reliable.
        support = min(1.0, min(positives, negatives) / 10.0)
        reliability[label] = float(np.clip((score - 0.5) / 0.5, 0.0, 1.0) * support)
    calibrated_confidence *= floor + (1.0 - floor) * reliability[None, :]
    calibrated_confidence[gold] = 1.0
    return calibrated_targets, calibrated_confidence, reliability


def _dataset(cache, mask, targets=None, confidence=None, indices=None):
    import torch
    from torch.utils.data import Dataset

    class StudyDataset(Dataset):
        def __init__(self):
            self.indices = np.arange(len(cache)) if indices is None else np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, item):
            index = int(self.indices[item])
            result = {
                "index": index,
                # Memmap-backed cache slices can be read-only.  Copy at the
                # PyTorch boundary so transforms or future model code never
                # attempt a write through a non-writable NumPy view.
                "images": torch.from_numpy(np.array(cache[index], copy=True)),
                "mask": torch.from_numpy(np.array(mask[index], copy=True)),
            }
            if targets is not None:
                result["targets"] = torch.from_numpy(targets[index])
                result["confidence"] = torch.from_numpy(confidence[index])
            return result

    return StudyDataset()


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def _unwrapped(model):
    return model.module if hasattr(model, "module") else model


def set_encoder_trainability(model, unfreeze_last_blocks: int | None) -> None:
    """Freeze/unfreeze a portable tail of a DINO or timm encoder."""
    encoder = _unwrapped(model).encoder
    if unfreeze_last_blocks is None:
        for parameter in encoder.parameters():
            parameter.requires_grad = True
        return
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    count = max(0, int(unfreeze_last_blocks))
    if count == 0:
        return
    backbone = getattr(encoder, "model", encoder)
    blocks = getattr(getattr(backbone, "encoder", None), "layer", None)
    if blocks is None:
        blocks = getattr(backbone, "stages", None)
    if blocks is None:
        raise AttributeError("encoder does not expose DINO layers or timm stages")
    for block in list(blocks)[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for norm_name in ("layernorm", "norm", "head"):
        norm = getattr(backbone, norm_name, None)
        if norm is not None:
            for parameter in norm.parameters():
                parameter.requires_grad = True


def _optimizer(
    model,
    config: TrainConfig,
    *,
    encoder_lr: float,
    head_lr: float,
):
    import torch

    bare = _unwrapped(model)
    encoder, head = [], []
    for name, parameter in bare.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if "encoder" in name else head).append(parameter)
    groups = []
    if encoder:
        # DINO exposes ordered transformer blocks, so make the final blocks
        # adapt faster without over-updating the pretrained lower stack.
        backbone = getattr(bare.encoder, "model", None)
        blocks = getattr(getattr(backbone, "encoder", None), "layer", None)
        if blocks is not None and config.layer_decay < 1.0:
            assigned: set[int] = set()
            active = [block for block in blocks if any(p.requires_grad for p in block.parameters())]
            for distance, block in enumerate(reversed(active)):
                parameters = [p for p in block.parameters() if p.requires_grad]
                if parameters:
                    groups.append({"params": parameters, "lr": encoder_lr * config.layer_decay ** distance})
                    assigned.update(id(p) for p in parameters)
            remaining = [parameter for parameter in encoder if id(parameter) not in assigned]
            if remaining:
                groups.append({"params": remaining, "lr": encoder_lr})
        else:
            groups.append({"params": encoder, "lr": encoder_lr})
    if head:
        groups.append({"params": head, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def predict(model, cache, mask, indices, *, batch_size: int, workers: int, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    loader = _loader(_dataset(cache, mask, indices=indices), batch_size=batch_size, shuffle=False, workers=workers)
    predictions, output_indices = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            slot_mask_value = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                probability = model(images, slot_mask_value).sigmoid()
            predictions.append(probability.float().cpu().numpy())
            output_indices.append(batch["index"].numpy())
    return np.concatenate(output_indices), np.concatenate(predictions)


def train_fold(
    fold: int,
    fold_assignment: np.ndarray,
    cache,
    mask: np.ndarray,
    targets: np.ndarray,
    confidence: np.ndarray,
    gold: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    label_version: str,
    deadline: float | None = None,
    raw_weak_targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    torch.manual_seed(train_config.seed + fold)
    np.random.seed(train_config.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices = np.where(fold_assignment != fold)[0]
    validation_indices = np.where(fold_assignment == fold)[0]
    calibrated_targets, calibrated_confidence, reliability = fold_calibrated_supervision(
        targets,
        confidence,
        gold,
        raw_weak_targets,
        train_indices,
        floor=train_config.reliability_floor,
    )
    selection_targets = (
        np.asarray(raw_weak_targets, dtype=np.float32)
        if raw_weak_targets is not None else calibrated_targets
    )
    gold_train = train_indices[gold[train_indices]]
    repeated_train_indices = train_indices
    if len(gold_train):
        repeated_train_indices = np.concatenate(
            [train_indices, np.tile(gold_train, train_config.gold_repeat - 1)]
        )

    model = create_study_model(model_config).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    ema = ExponentialMovingAverage(model.module if hasattr(model, "module") else model, train_config.ema_decay)
    best_score = -np.inf
    best_state = None
    history = []

    weak_blocks = (
        model_config.unfreeze_last_blocks
        if train_config.weak_unfreeze_last_blocks is None
        else train_config.weak_unfreeze_last_blocks
    )
    stages = [
        (
            "warmup",
            train_config.warmup_epochs,
            train_config.warmup_unfreeze_last_blocks,
            train_config.learning_rate,
            train_config.warmup_head_learning_rate or train_config.head_learning_rate,
            repeated_train_indices,
        ),
        (
            "weak",
            train_config.epochs,
            weak_blocks,
            train_config.learning_rate,
            train_config.head_learning_rate,
            repeated_train_indices,
        ),
        (
            "exact",
            train_config.exact_epochs,
            train_config.exact_unfreeze_last_blocks,
            train_config.exact_learning_rate or train_config.learning_rate,
            train_config.exact_head_learning_rate or train_config.head_learning_rate,
            gold_train,
        ),
    ]
    global_epoch = 0
    for stage_name, stage_epochs, unfreeze_blocks, encoder_lr, head_lr, stage_indices in stages:
        if stage_epochs <= 0 or len(stage_indices) < 1:
            continue
        if deadline is not None and time.time() > deadline - 300:
            raise TimeoutError(f"insufficient notebook budget to start {stage_name} for fold {fold}")
        set_encoder_trainability(model, unfreeze_blocks)
        optimizer = _optimizer(model, train_config, encoder_lr=encoder_lr, head_lr=head_lr)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        pos_weight = torch.from_numpy(
            positive_class_weights(calibrated_targets[stage_indices], calibrated_confidence[stage_indices])
        ).to(device)
        loader = _loader(
            _dataset(cache, mask, calibrated_targets, calibrated_confidence, stage_indices),
            batch_size=train_config.batch_size,
            shuffle=True,
            workers=train_config.workers,
        )
        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            epoch_started = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step, batch in enumerate(loader):
                images = augment_intensity(batch["images"].to(device, non_blocking=True))
                slot_mask_value = slot_dropout(batch["mask"].to(device, non_blocking=True), train_config.slot_dropout)
                y = batch["targets"].to(device, non_blocking=True)
                w = batch["confidence"].to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images, slot_mask_value)
                    bce = weighted_soft_bce(logits, y, w, pos_weight)
                    rank = pairwise_auc_loss(logits, y, w)
                    loss = (bce + train_config.rank_loss_weight * rank) / train_config.accumulation_steps
                scaler.scale(loss).backward()
                if (step + 1) % train_config.accumulation_steps == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
                losses.append(float(loss.item() * train_config.accumulation_steps))

            val_order, val_prediction = predict(
                ema.model, cache, mask, validation_indices,
                batch_size=train_config.eval_batch_size, workers=train_config.workers, device=device,
            )
            # Checkpoint selection is deliberately report-label only.  Gold rows
            # in this validation fold are audit data and cannot tune this fold.
            weak_auc = float(np.nanmean(auc_vector(
                (selection_targets[val_order] > 0.5).astype(np.int8), val_prediction
            )))
            gold_validation = val_order[gold[val_order]]
            gold_auc = float("nan")
            if len(gold_validation):
                local = np.isin(val_order, gold_validation)
                gold_auc = float(np.nanmean(auc_vector(targets[gold_validation].astype(np.int8), val_prediction[local])))
            history.append({
                "stage": stage_name,
                "stage_epoch": stage_epoch + 1,
                "epoch": global_epoch,
                "loss": float(np.mean(losses)),
                "selection_weak_auc": weak_auc,
                "gold_audit_auc": gold_auc,
            })
            print(
                f"fold={fold} stage={stage_name} epoch={stage_epoch + 1}/{stage_epochs} "
                f"loss={np.mean(losses):.5f} selection_weak_auc={weak_auc:.5f} gold_audit_auc={gold_auc:.5f}",
                flush=True,
            )
            if weak_auc > best_score:
                best_score = weak_auc
                best_state = {key: value.detach().cpu().clone() for key, value in ema.model.state_dict().items()}
            epoch_seconds = time.time() - epoch_started
            if deadline is not None and time.time() + 1.20 * epoch_seconds > deadline:
                print(f"fold={fold}: stopping after {stage_name} epoch {stage_epoch + 1} to preserve notebook runtime budget", flush=True)
                break
        del loader, optimizer, scaler
        if deadline is not None and time.time() > deadline - 300:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    ema.model.load_state_dict(best_state)
    val_order, val_prediction = predict(
        ema.model, cache, mask, validation_indices,
        batch_size=train_config.eval_batch_size, workers=train_config.workers, device=device,
    )
    metrics = {
        "fold": fold,
        "best_selection_weak_auc": best_score,
        "reliability": dict(zip(LABELS, reliability)),
        "history": history,
    }
    payload = checkpoint_payload(
        ema.model, model_config, fold=fold,
        label_version=label_version, metrics=metrics,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(payload, Path(output_dir) / f"fold_{fold}.pt")
    return val_order, val_prediction, metrics
