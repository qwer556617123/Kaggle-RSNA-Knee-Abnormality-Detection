from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import rank_columns, validate_submission
from .constants import LABELS
from .ensemble import apply_rank_blend
from .modeling import create_study_model, find_encoder_source, model_config_from_mapping
from .paths import iter_files_pruned
from .training import CacheConfig, build_cache, predict


def discover_model_groups(search_root: str | Path = "/kaggle/input") -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    for checkpoint in iter_files_pruned(search_root):
        if checkpoint.name.startswith("fold_") and checkpoint.suffix == ".pt":
            groups[checkpoint.parent].append(checkpoint)
    return {
        parent: sorted(paths)
        for parent, paths in sorted(groups.items(), key=lambda item: str(item[0]))
        if paths
    }


def _load_checkpoint_model(path: Path, device):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if list(payload.get("labels", [])) != LABELS:
        raise ValueError(f"label order mismatch in {path}")
    config = model_config_from_mapping(payload["model_config"])
    source = find_encoder_source(config.model_name)
    config.checkpoint_path = str(source) if source else None
    config.allow_random_encoder = not config.model_name.startswith("dinov2")
    model = create_study_model(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), config, payload


def predict_model_group(
    checkpoints: Sequence[Path],
    cache,
    mask: np.ndarray,
    *,
    batch_size: int,
    workers: int,
    device,
    checkpoint_limit: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if checkpoint_limit is not None:
        import torch

        if checkpoint_limit < 1:
            raise ValueError("checkpoint_limit must be positive")
        scored = []
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            score = float(payload.get("metrics", {}).get("best_weak_auc", float("-inf")))
            scored.append((score, str(checkpoint), checkpoint))
        checkpoints = [item[2] for item in sorted(scored, reverse=True)[:checkpoint_limit]]
    fold_predictions = []
    metadata = []
    indices = np.arange(len(cache))
    for checkpoint in checkpoints:
        model, config, payload = _load_checkpoint_model(checkpoint, device)
        order, fold_prediction = predict(
            model, cache, mask, indices,
            batch_size=batch_size, workers=workers, device=device,
        )
        if not np.array_equal(order, indices):
            raise RuntimeError(f"prediction order mismatch for {checkpoint}")
        fold_predictions.append(fold_prediction)
        metadata.append({"checkpoint": str(checkpoint), "fold": payload.get("fold")})
        del model
        if device.type == "cuda":
            __import__("torch").cuda.empty_cache()
    prediction = np.mean(np.stack(fold_predictions), axis=0)
    return prediction, {"config": config.to_dict(), "checkpoints": metadata}


def run_inference(
    test: pd.DataFrame,
    test_series: pd.DataFrame,
    dicom_root: str | Path,
    model_root: str | Path,
    working_dir: str | Path,
    *,
    batch_size: int = 4,
    workers: int = 2,
    checkpoint_limit: int | None = None,
    slices: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    import torch

    groups = discover_model_groups(model_root)
    if not groups:
        raise FileNotFoundError(f"no fold checkpoints found under {model_root}")

    first_path = next(iter(groups.values()))[0]
    first_payload = torch.load(first_path, map_location="cpu", weights_only=False)
    first_config = model_config_from_mapping(first_payload["model_config"])
    trained_slices = first_config.groups_per_slot * 3
    selected_slices = trained_slices if slices is None else int(slices)
    if selected_slices < 3 or selected_slices % 3:
        raise ValueError("inference slices must be a positive multiple of the 2.5D group size (3)")
    cache_config = CacheConfig(
        image_size=first_config.image_size,
        slices=selected_slices,
        group_size=3,
        slots=first_config.slots,
        target_fov_mm=first_config.target_fov_mm,
    )
    cache, mask, failures = build_cache(
        test["StudyInstanceUID"].astype(str).tolist(),
        test_series,
        dicom_root,
        working_dir,
        cache_config,
        prefix="test",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_predictions = []
    group_metadata = []
    for parent, checkpoints in groups.items():
        prediction, metadata = predict_model_group(
            checkpoints, cache, mask,
            batch_size=batch_size, workers=workers, device=device,
            checkpoint_limit=checkpoint_limit,
        )
        model_predictions.append(prediction)
        metadata["group"] = str(parent)
        group_metadata.append(metadata)

    weights_candidates = list(iter_files_pruned(model_root, "blend_weights.json"))
    if len(model_predictions) == 1:
        final = model_predictions[0]
        blend_mode = "single-model-fold-mean"
    elif weights_candidates:
        blend_payload = json.loads(weights_candidates[0].read_text(encoding="utf-8"))
        weights = np.asarray(blend_payload["weights"])
        expected = blend_payload.get("model_groups")
        if expected:
            current = [Path(item["group"]).name for item in group_metadata]
            if set(expected) != set(current):
                raise ValueError(f"blend model groups {expected} do not match mounted groups {current}")
            model_predictions = [model_predictions[current.index(name)] for name in expected]
        final = apply_rank_blend(model_predictions, weights)
        blend_mode = f"oof-rank-blend:{weights_candidates[0]}"
    else:
        final = np.mean(np.stack([rank_columns(value) for value in model_predictions]), axis=0)
        blend_mode = "equal-rank-blend"

    empty = mask.sum(axis=1) == 0
    if empty.any():
        valid = ~empty
        if not valid.any():
            raise RuntimeError("all test studies failed DICOM decoding")
        final[empty] = np.median(final[valid], axis=0)
        extra = pd.DataFrame(
            {
                "StudyInstanceUID": test.loc[empty, "StudyInstanceUID"],
                "SeriesInstanceUID": "",
                "slot": "ALL",
                "error": "all selected series failed; assigned per-label test median",
            }
        )
        failures = pd.concat([failures, extra], ignore_index=True)

    final = np.clip(final, 1e-6, 1 - 1e-6)
    submission = pd.DataFrame(final, columns=LABELS)
    submission.insert(0, "StudyInstanceUID", test["StudyInstanceUID"].values)
    validate_submission(submission, test)
    metadata = {
        "model_groups": group_metadata,
        "blend_mode": blend_mode,
        "test_studies": len(test),
        "empty_studies": int(empty.sum()),
        "decode_failures": len(failures),
        "checkpoint_limit": checkpoint_limit,
        "trained_slices": trained_slices,
        "inference_slices": selected_slices,
    }
    return submission, metadata, failures
