from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import LABELS, SLOT_NAMES
from .paths import iter_files_pruned


@dataclass
class ModelConfig:
    model_name: str = "dinov2-small-hf"
    image_size: int = 196
    slots: tuple[str, ...] = SLOT_NAMES
    groups_per_slot: int = 5
    dropout: float = 0.20
    checkpoint_path: str | None = None
    freeze_encoder: bool = False
    unfreeze_last_blocks: int | None = 6
    allow_random_encoder: bool = False
    architecture_version: str = "shared-v1"
    clinical_prior_strength: float = 0.18
    target_fov_mm: float | tuple[float, ...] = 170.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["slots"] = list(self.slots)
        return result


def find_encoder_source(model_name: str, root: str | Path = "/kaggle/input") -> Path | None:
    root = Path(root)
    if model_name.startswith("dinov2"):
        candidates = [
            config.parent
            for config in iter_files_pruned(root, "config.json")
            if "dinov2" in str(config.parent).casefold()
        ]
        variant = "small" if "small" in model_name else "base"
        preferred = [path for path in candidates if variant in str(path).casefold()]
        candidates = preferred or candidates
        return sorted(candidates, key=lambda path: len(str(path)))[0] if candidates else None
    suffixes = (".pth", ".pt", ".bin", ".safetensors")
    tokens = [token for token in model_name.casefold().replace(".", "_").split("_") if token]
    candidates = [
        path for path in iter_files_pruned(root)
        if path.is_file() and path.suffix.casefold() in suffixes
        # Fold checkpoints are complete study models, not ImageNet encoder
        # weights.  They happen to include the architecture name in their
        # mounted path and must never be passed through the encoder loader.
        and not path.name.startswith("fold_")
        and any(token in str(path).casefold() for token in tokens[:2])
    ]
    return sorted(candidates, key=lambda path: len(str(path)))[0] if candidates else None


def load_timm_encoder_checkpoint(encoder, checkpoint_path: str | Path) -> None:
    """Load an offline timm checkpoint while deliberately discarding its classifier.

    The competition heads are always replaced by the study-level multi-label head.
    A published ImageNet checkpoint may therefore contain ``head.fc.*`` (or an
    equivalent classifier) that must not be treated as a backbone mismatch.
    Every other incompatibility is a hard error: silently accepting a different
    ConvNeXt variant would invalidate the experiment.
    """
    import torch

    path = Path(checkpoint_path)
    if path.suffix.casefold() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required for this offline encoder checkpoint") from exc
        payload = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping):
        raise ValueError(f"unsupported checkpoint payload in {path}")
    state_dict = {
        (str(key)[7:] if str(key).startswith("module.") else str(key)): value
        for key, value in payload.items()
        if not str(key).startswith(("head.", "classifier.", "fc."))
    }
    incompatible = encoder.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(("head.", "classifier.", "fc."))
    ]
    invalid_unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(("head.", "classifier.", "fc."))
    ]
    if invalid_missing or invalid_unexpected:
        raise RuntimeError(
            "offline encoder checkpoint is incompatible; "
            f"missing={invalid_missing}, unexpected={invalid_unexpected}"
        )


def build_encoder(config: ModelConfig):
    checkpoint = config.checkpoint_path
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(f"encoder checkpoint not found: {checkpoint}")
    if config.model_name.startswith("dinov2"):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel

        if checkpoint:
            backbone = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        elif config.allow_random_encoder:
            # Architecture-only recreation; the full trained state is loaded immediately afterward.
            raise ValueError("DINOv2 inference requires its mounted base model directory for architecture recreation")
        else:
            raise ValueError("an offline DINOv2 model directory is required")

        class HFEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.num_features = int(model.config.hidden_size) * 2

            def forward(self, x):
                hidden = self.model(pixel_values=x).last_hidden_state
                return torch.cat((hidden[:, 0], hidden[:, 1:].mean(dim=1)), dim=1)

        encoder = HFEncoder(backbone)
        if config.freeze_encoder:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
        elif config.unfreeze_last_blocks is not None:
            # DINOv2-S/B is expensive to fine-tune end-to-end with 25 images per
            # study.  Keep the public baseline's useful last-block adaptation,
            # while avoiding optimizer/activation pressure from the early stack.
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            blocks = getattr(getattr(backbone, "encoder", None), "layer", None)
            if blocks is None:
                raise AttributeError("unsupported DINOv2 layout: encoder.layer was not found")
            count = max(0, int(config.unfreeze_last_blocks))
            for block in blocks[-count:] if count else ():
                for parameter in block.parameters():
                    parameter.requires_grad = True
            final_norm = getattr(backbone, "layernorm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad = True
        return encoder, encoder.num_features

    import timm

    # CNN families such as ConvNeXt neither need nor consistently accept
    # ``img_size``.  Build their feature extractor first, then validate-load
    # the attached offline checkpoint without its ImageNet classifier head.
    encoder = timm.create_model(config.model_name, pretrained=False, num_classes=0)
    if checkpoint:
        load_timm_encoder_checkpoint(encoder, checkpoint)
    if checkpoint is None and not config.allow_random_encoder:
        raise ValueError("an offline pretrained checkpoint is required; random encoders are prohibited")
    if config.freeze_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    dimension = int(getattr(encoder, "num_features"))
    return encoder, dimension


def create_study_model(config: ModelConfig):
    import torch
    from torch import nn

    encoder, dimension = build_encoder(config)

    def clinical_slot_prior() -> np.ndarray:
        prior = np.zeros((len(LABELS), len(config.slots)), dtype=np.float32)
        for target_index, target in enumerate(LABELS):
            for slot_index, slot_name in enumerate(config.slots):
                slot = slot_name.casefold()
                sagittal = "sagittal" in slot
                coronal = "coronal" in slot
                axial = "axial" in slot
                fluid = "fluid" in slot and "nonfluid" not in slot
                structural = "nonfluid" in slot
                if target == "ACL":
                    score = 1.0 * sagittal + 0.25 * axial
                elif target == "MCL":
                    score = 1.0 * coronal + 0.20 * sagittal
                elif "Meniscus" in target:
                    score = 0.85 * coronal + 0.75 * sagittal
                elif target in ("Medial OA", "Lateral OA"):
                    score = 0.85 * coronal + 0.35 * sagittal + 0.25 * structural
                elif target == "PF OA":
                    score = 1.0 * axial + 0.35 * sagittal
                elif target in ("Effusion", "Synovitis"):
                    score = 0.9 * axial + 0.65 * sagittal + 0.35 * coronal + 0.25 * fluid
                elif target == "Baker's":
                    score = 0.95 * sagittal + 0.45 * axial
                elif target == "Contusion":
                    score = 0.55 + 0.35 * fluid
                else:  # Fracture
                    score = 0.45 + 0.25 * structural + 0.20 * fluid
                prior[target_index, slot_index] = score
        prior -= prior.mean(axis=1, keepdims=True)
        prior /= np.maximum(prior.std(axis=1, keepdims=True), 1e-6)
        return prior

    def masked_softmax(logits, mask, dim: int):
        probabilities = logits.masked_fill(~mask.bool(), -1e4).softmax(dim=dim)
        probabilities = probabilities * mask.to(logits.dtype)
        return probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(1e-6)

    class StudyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.encoder = encoder
            if config.architecture_version == "shared-v1":
                self.group_attention = nn.Sequential(
                    nn.LayerNorm(dimension), nn.Linear(dimension, 1)
                )
            elif config.architecture_version == "target-depth-v2":
                self.depth_queries = nn.Parameter(torch.zeros(len(LABELS), dimension))
                nn.init.normal_(self.depth_queries, std=0.02)
                self.register_buffer(
                    "clinical_slot_prior", torch.from_numpy(clinical_slot_prior())
                )
            else:
                raise ValueError(f"unknown architecture_version={config.architecture_version}")
            self.slot_embedding = nn.Parameter(torch.zeros(len(config.slots), dimension))
            nn.init.normal_(self.slot_embedding, std=0.02)
            self.target_queries = nn.Parameter(torch.zeros(len(LABELS), dimension))
            nn.init.normal_(self.target_queries, std=0.02)
            self.target_norm = nn.LayerNorm(dimension)
            self.dropout = nn.Dropout(config.dropout)
            self.target_weight = nn.Parameter(torch.empty(len(LABELS), dimension))
            self.target_bias = nn.Parameter(torch.zeros(len(LABELS)))
            nn.init.trunc_normal_(self.target_weight, std=0.02)

        def forward(self, images, slot_mask):
            # images: [B, slots, groups, 3, H, W], uint8 or float.
            batch, slots, groups, channels, height, width = images.shape
            x = images.reshape(batch * slots * groups, channels, height, width).float() / 255.0
            # ImageNet normalization is appropriate for the attached DINO/ConvNeXt weights.
            mean = x.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            std = x.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
            features = self.encoder((x - mean) / std)
            if features.ndim > 2:
                features = features.mean(tuple(range(2, features.ndim)))
            features = features.reshape(batch, slots, groups, -1)

            if config.architecture_version == "shared-v1":
                group_scores = self.group_attention(features).squeeze(-1)
                group_weights = group_scores.softmax(dim=2)
                slot_features = (features * group_weights.unsqueeze(-1)).sum(dim=2)
                slot_features = slot_features + self.slot_embedding.unsqueeze(0)
                queries = self.target_queries.unsqueeze(0).expand(batch, -1, -1)
                attention = torch.einsum("btd,bsd->bts", queries, slot_features) / (features.shape[-1] ** 0.5)
                valid = slot_mask[:, None, :].bool()
                weights = masked_softmax(attention, valid, dim=-1)
                target_features = torch.einsum("bts,bsd->btd", weights, slot_features)
            else:
                enriched = features + self.slot_embedding[None, :, None, :]
                depth_logits = torch.einsum(
                    "bsgd,td->btsg", enriched, self.depth_queries
                ) / (features.shape[-1] ** 0.5)
                depth_valid = slot_mask[:, None, :, None].bool().expand_as(depth_logits)
                depth_weights = masked_softmax(depth_logits, depth_valid, dim=-1)
                slot_features = torch.einsum("btsg,bsgd->btsd", depth_weights, enriched)
                attention = torch.einsum(
                    "btsd,td->bts", slot_features, self.target_queries
                ) / (features.shape[-1] ** 0.5)
                attention = attention + config.clinical_prior_strength * self.clinical_slot_prior[None]
                valid = slot_mask[:, None, :].bool()
                weights = masked_softmax(attention, valid, dim=-1)
                target_features = torch.einsum("bts,btsd->btd", weights, slot_features)
            target_features = self.dropout(self.target_norm(target_features))
            return (target_features * self.target_weight.unsqueeze(0)).sum(-1) + self.target_bias

    return StudyModel()


def augment_intensity(
    images,
    *,
    noise_probability: float = 0.35,
    spatial_probability: float = 0.70,
    max_rotation_degrees: float = 5.0,
    scale_range: tuple[float, float] = (0.95, 1.05),
    slice_dropout_probability: float = 0.08,
):
    """Anatomy-safe intensity/spatial augmentation with no reflection.

    One affine is sampled per study and shared by all slots/groups so that
    protocol geometry stays coherent.  A dropped outer 2.5D channel is
    replaced by the centre slice instead of blacking out anatomy.
    """
    import torch
    import torch.nn.functional as functional

    x = images.float() / 255.0
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    scale = 0.90 + 0.20 * torch.rand(shape, device=x.device)
    gamma = 0.88 + 0.24 * torch.rand(shape, device=x.device)
    x = x.clamp(0, 1).pow(gamma) * scale
    if torch.rand((), device=x.device) < noise_probability:
        x = x + 0.008 * torch.randn_like(x)

    if spatial_probability > 0 and torch.rand((), device=x.device) < spatial_probability:
        batch, slots, groups, channels, height, width = x.shape
        angle = (
            (2.0 * torch.rand(batch, device=x.device) - 1.0)
            * max_rotation_degrees
            * torch.pi
            / 180.0
        )
        scale = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(
            batch, device=x.device
        )
        cosine, sine = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.zeros((batch, 2, 3), dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta = theta[:, None, None].expand(-1, slots, groups, -1, -1).reshape(-1, 2, 3)
        flat = x.reshape(-1, channels, height, width)
        grid = functional.affine_grid(theta, flat.shape, align_corners=False)
        x = functional.grid_sample(
            flat, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(batch, slots, groups, channels, height, width)

    if slice_dropout_probability > 0 and x.shape[3] == 3:
        drop = torch.rand((*x.shape[:3], 2), device=x.device) < slice_dropout_probability
        centre = x[:, :, :, 1]
        x[:, :, :, 0] = torch.where(drop[..., 0, None, None], centre, x[:, :, :, 0])
        x[:, :, :, 2] = torch.where(drop[..., 1, None, None], centre, x[:, :, :, 2])
    return (x.clamp(0, 1) * 255.0).to(images.dtype)


def slot_dropout(mask, probability: float = 0.10):
    import torch

    if probability <= 0:
        return mask
    result = mask.clone()
    dropped = (torch.rand_like(result) < probability) & result.bool()
    result[dropped] = 0
    empty = result.sum(dim=1) == 0
    result[empty] = mask[empty]
    return result


def weighted_soft_bce(logits, targets, confidence, positive_weights=None):
    import torch.nn.functional as functional

    loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = confidence
    if positive_weights is not None:
        weights = weights * (1 + (positive_weights[None, :] - 1) * targets)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def pairwise_auc_loss(logits, targets, confidence, margin: float = 0.0):
    import torch
    import torch.nn.functional as functional

    losses = []
    for label_index in range(logits.shape[1]):
        positive = logits[(targets[:, label_index] >= 0.60) & (confidence[:, label_index] >= 0.40), label_index]
        negative = logits[(targets[:, label_index] <= 0.40) & (confidence[:, label_index] >= 0.40), label_index]
        if positive.numel() and negative.numel():
            losses.append(functional.softplus(-(positive[:, None] - negative[None, :] - margin)).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def positive_class_weights(targets: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    positive = (targets * confidence).sum(axis=0)
    negative = ((1 - targets) * confidence).sum(axis=0)
    return np.clip(np.sqrt((negative + 1) / (positive + 1)), 1.0, 3.0).astype(np.float32)


def auc_vector(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    result = np.full(targets.shape[1], np.nan, dtype=np.float64)
    for index in range(targets.shape[1]):
        if len(np.unique(targets[:, index])) == 2:
            result[index] = roc_auc_score(targets[:, index], predictions[:, index])
    return result


class ExponentialMovingAverage:
    def __init__(self, model, decay: float = 0.995):
        from copy import deepcopy

        self.decay = decay
        self.model = deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def update(self, model) -> None:
        source = model.module if hasattr(model, "module") else model
        with __import__("torch").no_grad():
            for destination, value in zip(self.model.state_dict().values(), source.state_dict().values()):
                if destination.is_floating_point():
                    destination.mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
                else:
                    destination.copy_(value)


def checkpoint_payload(
    model,
    config: ModelConfig,
    *,
    fold: int,
    label_version: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    return {
        "state_dict": {key: value.detach().cpu() for key, value in state.items()},
        "model_config": config.to_dict(),
        "fold": int(fold),
        "label_version": label_version,
        "labels": list(LABELS),
        "metrics": dict(metrics),
    }


def model_config_from_mapping(value: Mapping[str, Any]) -> ModelConfig:
    payload = dict(value)
    payload["slots"] = tuple(payload.get("slots", SLOT_NAMES))
    if isinstance(payload.get("target_fov_mm"), list):
        payload["target_fov_mm"] = tuple(payload["target_fov_mm"])
    return ModelConfig(**payload)
