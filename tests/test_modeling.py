import pytest
import torch
from unittest.mock import patch

from rsna_knee.modeling import (
    ModelConfig,
    augment_intensity,
    create_study_model,
    find_encoder_source,
    load_timm_encoder_checkpoint,
    model_config_from_mapping,
)


def test_encoder_source_excludes_complete_fold_checkpoint(tmp_path):
    pretrained = tmp_path / "convnext-tiny-fb-in22k-ft-in1k" / "model.safetensors"
    trained_fold = tmp_path / "rsna-knee-train-convnext-tiny" / "fold_0.pt"
    pretrained.parent.mkdir()
    trained_fold.parent.mkdir()
    pretrained.touch()
    trained_fold.touch()
    assert find_encoder_source("convnext_tiny", tmp_path) == pretrained


def test_anatomy_safe_augmentation_preserves_contract():
    torch.manual_seed(7)
    images = torch.randint(0, 256, (2, 5, 5, 3, 24, 24), dtype=torch.uint8)
    result = augment_intensity(
        images,
        noise_probability=1.0,
        spatial_probability=1.0,
        slice_dropout_probability=1.0,
    )
    assert result.shape == images.shape
    assert result.dtype == images.dtype
    assert int(result.min()) >= 0 and int(result.max()) <= 255
    assert torch.equal(result[:, :, :, 0], result[:, :, :, 1])
    assert torch.equal(result[:, :, :, 2], result[:, :, :, 1])


class _DummyEncoder(torch.nn.Module):
    def forward(self, images):
        value = images.mean(dim=(1, 2, 3), keepdim=False)
        return value[:, None].expand(-1, 8)


def test_architecture_versions_preserve_output_contract_and_old_config_default():
    images = torch.randint(0, 256, (2, 5, 3, 3, 14, 14), dtype=torch.uint8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1]], dtype=torch.float32)
    with patch("rsna_knee.modeling.build_encoder", return_value=(_DummyEncoder(), 8)):
        shared = create_study_model(ModelConfig(architecture_version="shared-v1"))
        target = create_study_model(ModelConfig(architecture_version="target-depth-v2"))
    assert shared(images, mask).shape == (2, 12)
    assert target(images, mask).shape == (2, 12)
    assert torch.isfinite(target(images, mask)).all()
    assert model_config_from_mapping({"slots": list(ModelConfig().slots)}).architecture_version == "shared-v1"


class _TinyConvEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Linear(3, 4)


def test_timm_checkpoint_accepts_classifier_head_only_difference(tmp_path):
    source = _TinyConvEncoder()
    checkpoint = tmp_path / "convnext.pth"
    torch.save(
        {
            "state_dict": {
                "features.weight": source.features.weight.detach().clone(),
                "features.bias": source.features.bias.detach().clone(),
                "head.fc.weight": torch.zeros(2, 4),
                "head.fc.bias": torch.zeros(2),
            }
        },
        checkpoint,
    )
    target = _TinyConvEncoder()
    load_timm_encoder_checkpoint(target, checkpoint)
    assert torch.equal(target.features.weight, source.features.weight)


def test_timm_checkpoint_rejects_non_classifier_mismatch(tmp_path):
    checkpoint = tmp_path / "bad-convnext.pth"
    torch.save({"state_dict": {"unexpected.weight": torch.zeros(2, 2)}}, checkpoint)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_timm_encoder_checkpoint(_TinyConvEncoder(), checkpoint)
