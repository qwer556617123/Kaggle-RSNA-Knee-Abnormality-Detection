import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna_knee.research_coatnet import CleanCoAtNetMIL, SpatialEvidenceMIL, strict_encoder_load, soft_weighted_bce, load_contract
from rsna_knee.research_profile import profiling_windows
from rsna_knee.constants import CONFIDENCE_COLUMNS, LABELS
from rsna_knee.research_validation import table_digest
from tests.synthetic_contract import write_synthetic_contract


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.head = nn.Module()
        self.head.norm = nn.LayerNorm(8)

    def forward(self, x):
        return self.head.norm(self.stem(x).mean((-1,-2)))


class FakeSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.proj = nn.Conv2d(3, 8, 1)

    def forward_features(self, x):
        self.calls += 1
        return self.proj(torch.nn.functional.avg_pool2d(x, 2))


class ResearchCoAtNetTests(unittest.TestCase):
    def test_only_classifier_difference_accepted(self):
        model = FakeEncoder()
        state = dict(model.state_dict())
        state['head.fc.weight'] = torch.zeros(1000,8)
        state['head.fc.bias'] = torch.zeros(1000)
        strict_encoder_load(model,state)
        for key in ('head.norm.weight', 'stem.weight'):
            broken = dict(state)
            del broken[key]
            with self.assertRaises(ValueError):
                strict_encoder_load(model,broken)
        state['unexpected.weight'] = torch.zeros(1)
        with self.assertRaises(ValueError):
            strict_encoder_load(model,state)

    def test_shape_mismatch_rejected(self):
        model = FakeEncoder()
        state = dict(model.state_dict())
        state['stem.weight'] = torch.zeros(4,3,1,1)
        with self.assertRaises(ValueError):
            strict_encoder_load(model,state)

    def test_batch_one_bce_has_gradient(self):
        model = CleanCoAtNetMIL(FakeEncoder(),8)
        y = model(torch.randn(1,3,3,12,12))
        loss = soft_weighted_bce(y,torch.ones_like(y),torch.ones_like(y))
        loss.backward()
        self.assertEqual(y.shape,(1,12))
        self.assertGreater(model.class_weight.grad.abs().sum().item(),0)

    def test_zero_confidence_zero_gradient(self):
        logits = torch.randn(1,12,requires_grad=True)
        soft_weighted_bce(logits,torch.ones_like(logits),torch.zeros_like(logits)).backward()
        self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_spatial_arms_have_same_single_backbone_forward_and_mask_invariance(self):
        outputs = []
        for arm in ('global','center','target_query'):
            encoder = FakeSpatialEncoder()
            model = SpatialEvidenceMIL(encoder, 8, pooling_arm=arm).eval()
            images = torch.randn(1, 4, 3, 16, 16)
            mask = torch.tensor([[1,1,0,0]], dtype=torch.bool)
            slots = torch.tensor([[0,0,1,1]])
            depth = torch.tensor([[0.,1.,0.,1.]])
            laterality = torch.tensor([2])
            altered = images.clone()
            altered[:,2:] = 9999
            with torch.no_grad():
                outputs.append(model(images, mask, slots, depth, laterality))
                altered_output = model(altered, mask, slots, depth, laterality)
            self.assertEqual(encoder.calls, 2)
            torch.testing.assert_close(outputs[-1], altered_output)
            self.assertEqual(outputs[-1].shape, (1,12))

    def test_spatial_all_masked_study_fails(self):
        model = SpatialEvidenceMIL(FakeSpatialEncoder(), 8, pooling_arm='center')
        with self.assertRaises(ValueError):
            model(torch.randn(1,2,3,16,16), torch.zeros(1,2,dtype=torch.bool))

    def test_profile_never_invents_neighbor_slices(self):
        volume = np.zeros((5,8,8),dtype=np.uint8)
        volume[1],volume[3] = 64,192
        result = profiling_windows(volume,np.array([0,1,0,1,0]),4)
        self.assertEqual(tuple(result.shape),(4,3,384,384))
        torch.testing.assert_close(result[:,0],result[:,1])
        self.assertGreater(result[-1].mean().item(),result[0].mean().item())
        with self.assertRaises(ValueError):
            profiling_windows(volume,np.zeros(5),4)

    def test_synthetic_frozen_contract_and_tampered_table(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_synthetic_contract(root)
            frame, control, manifest = load_contract(root)
            _, candidate, _ = load_contract(root, 'evidence')
            self.assertEqual(len(frame), 180)
            self.assertEqual(manifest['nested_splits']['0']['excluded']['count'], 2)
            np.testing.assert_array_equal(control[LABELS], candidate[LABELS])
            self.assertFalse(control[CONFIDENCE_COLUMNS].equals(candidate[CONFIDENCE_COLUMNS]))
            frame.loc[2, 'fold'] = 2
            frame.to_csv(root / 'evaluation.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'contract hash mismatch'):
                load_contract(root)

    def test_target_change_is_rejected_even_with_updated_table_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_synthetic_contract(root)
            candidate = pd.read_csv(root / 'labels_evidence.csv')
            candidate.loc[0, LABELS[0]] = 0.9
            candidate.to_csv(root / 'labels_evidence.csv', index=False, float_format='%.9g')
            manifest_path = root / 'research_manifest.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['label_evidence_sha256'] = table_digest(pd.read_csv(root / 'labels_evidence.csv'))
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed targets'):
                load_contract(root)


if __name__ == '__main__':
    unittest.main()
