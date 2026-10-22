import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna_knee.constants import CONFIDENCE_COLUMNS, LABELS
from rsna_knee.research_trainer import TrainRecipe, run_three_fold, train_fold


class TinyMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3*4*4, 8)
        self.head = nn.Linear(8, 12)

    def forward(self, images, masks):
        x = self.backbone(images.mean(1).flatten(1))
        x = x.masked_fill(~masks.any(1, keepdim=True), 0)
        return self.head(x)


def make_frame(n=180):
    frame = pd.DataFrame({'StudyInstanceUID':[f'u{i}' for i in range(n)], 'report_group':[f'g{i}' for i in range(n)],
                          'fold':np.arange(n)%3, 'is_gold':[1]+[0]*(n-1)})
    rng = np.random.default_rng(5)
    for j,label in enumerate(LABELS):
        values = (rng.random(n) > .65).astype(int)
        frame[f'{label}__eval'] = values
        frame[f'{label}__eval_mask'] = True
    return frame


class ResearchTrainerTests(unittest.TestCase):
    def test_fold_trainer_selects_inner_and_exports_outer(self):
        frame = make_frame()
        labels = pd.DataFrame({'StudyInstanceUID':frame.StudyInstanceUID})
        rng = np.random.default_rng(6)
        for label in LABELS:
            labels[label] = rng.uniform(.1,.9,len(frame)).astype('float32')
            labels[f'{label}__confidence'] = .8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.random.default_rng(7).integers(0,255,size=(len(frame),24,3,4,4),dtype=np.uint8)
            masks = np.ones((len(frame),24),dtype=np.float32)
            np.save(root/'images_000.npy',images); np.save(root/'mask_000.npy',masks)
            (root/'cache_index.json').write_text(json.dumps({uid:{'shard':0,'row':i} for i,uid in enumerate(frame.StudyInstanceUID)}))
            result = train_fold(TinyMIL(), frame, labels, root, 0, root/'run', recipe=TrainRecipe(epochs=1,accumulation_steps=8,amp=False), device=torch.device('cpu'))
            self.assertEqual(result['status'],'completed')
            self.assertGreaterEqual(result['selected_epoch'],1)
            self.assertEqual(result['decode_failures'],0)
            self.assertTrue((root/'run/fold_0.pt').exists())
            self.assertTrue((root/'run/outer_0.csv').exists())

    def test_confidence_columns_are_present(self):
        self.assertEqual(len(CONFIDENCE_COLUMNS),12)

    def test_three_fold_runner_writes_complete_manifest_and_oof(self):
        frame = make_frame()
        labels = pd.DataFrame({'StudyInstanceUID':frame.StudyInstanceUID})
        rng = np.random.default_rng(8)
        for label in LABELS:
            labels[label] = rng.uniform(.1,.9,len(frame)).astype('float32')
            labels[f'{label}__confidence'] = .8
        spec = {
            'arm': 'C', 'contract_version': 'top10-clean-v1',
            'evaluation_sha256': 'eval', 'fold_sha256': 'fold',
            'pretrain_sha256': 'pretrain', 'preprocessing_sha256': 'cache',
            'code_version': 'test',
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = np.random.default_rng(9).integers(0,255,size=(len(frame),24,3,4,4),dtype=np.uint8)
            masks = np.ones((len(frame),24),dtype=np.float32)
            np.save(root/'images_000.npy',images); np.save(root/'mask_000.npy',masks)
            (root/'cache_index.json').write_text(json.dumps({uid:{'shard':0,'row':i} for i,uid in enumerate(frame.StudyInstanceUID)}))
            manifest = run_three_fold(
                lambda fold: TinyMIL(), frame, labels, root, root/'three_fold', spec,
                recipe=TrainRecipe(epochs=1, accumulation_steps=8, amp=False),
                device=torch.device('cpu'),
            )
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(set(manifest['folds']), {'0','1','2'})
            self.assertTrue((root/'three_fold/oof.csv').exists())
            self.assertTrue((root/'three_fold/inner_0.csv').exists())


if __name__ == '__main__':
    unittest.main()
