import torch

from rsna_knee.feature_model import FeatureHeadConfig, create_feature_attention_head


def test_feature_attention_head_contract_and_missing_slots():
    config = FeatureHeadConfig(input_dimension=32, hidden_dimension=16)
    model = create_feature_attention_head(config)
    model.eval()
    features = torch.randn(2, 5, 9, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 1]], dtype=torch.float32)
    output = model(features, mask)
    assert output.shape == (2, 12)
    assert torch.isfinite(output).all()

    slice_mask = mask[:, :, None].expand(-1, -1, 9).clone()
    slice_mask[:, :, 4] = 0
    masked_output = model(features, mask, slice_mask)
    changed_features = features.clone()
    changed_features[:, :, 4] = 1e5
    changed_output = model(changed_features, mask, slice_mask)
    assert torch.allclose(masked_output, changed_output, atol=1e-5)
