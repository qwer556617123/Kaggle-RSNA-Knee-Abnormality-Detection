import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABELS
from rsna_knee.research_coatnet import PRETRAIN_SHA, load_contract
from rsna_knee.research_oof import validate_oof, uid_hash
from rsna_knee.research_validation import nested_partition, UID
from tests.synthetic_contract import write_synthetic_contract


class ResearchOOFTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        write_synthetic_contract(root)
        self.frame,_,self.contract = load_contract(root)
        rows = np.sort(np.concatenate([nested_partition(self.frame,f).outer for f in range(3)]))
        self.pred = self.frame.iloc[rows][[UID,'fold']].copy()
        self.pred[LABELS] = np.random.default_rng(7).uniform(size=(len(rows),12))
        self.run = {k:self.contract[k] for k in ('contract_version','evaluation_sha256','fold_sha256')}
        self.run.update(pretrain_sha256=PRETRAIN_SHA,checkpoint_selection='inner_only',gold_role='excluded_with_report_groups',recipe_sha256='recipe',preprocessing_sha256='cache',code_version='test',folds={},arm='C',
                        cache_sha256='cache',sampling='physical_triplets_v1',resolution=384,train_windows=12,eval_windows=24,
                        optimizer='adamw_encoder_2e-5_head_2e-4_wd_0.01_warmup_cosine',augmentation='shared_intensity_gain_0.9_1.1_no_flip',seed=20260905,epochs=3,batch_size=1,accumulation_steps=8,supervision_arm='control',pooling_arm='global',geometry_arm='canonical')
        for fold in range(3):
            part = nested_partition(self.frame,fold)
            self.run['folds'][str(fold)] = {name+'_uid_sha256':uid_hash(self.frame.iloc[getattr(part,name)][UID]) for name in ('train','inner','outer','excluded')}
            self.run['folds'][str(fold)].update(checkpoint_sha256='test',decode_failures=0,selected_epoch=1)

    def test_reorder_is_uid_aligned(self):
        expected,a = validate_oof(self.frame,self.pred,self.run,self.contract)
        _,b = validate_oof(self.frame,self.pred.iloc[::-1],self.run,self.contract)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(expected),178)

    def test_partial_oof_rejected(self):
        with self.assertRaises(ValueError):
            validate_oof(self.frame,self.pred.iloc[1:],self.run,self.contract)

    def test_old_anchor_and_outer_selection_rejected(self):
        for key,value in [('pretrain_sha256','competition-finetuned'),('checkpoint_selection','gold'),('fold_sha256','old-v10')]:
            run = dict(self.run,**{key:value})
            with self.assertRaises(ValueError):
                validate_oof(self.frame,self.pred,run,self.contract)

    def test_cross_fold_predictions_rejected(self):
        self.pred.iloc[0,self.pred.columns.get_loc('fold')] = 8
        with self.assertRaises(ValueError):
            validate_oof(self.frame,self.pred,self.run,self.contract)


if __name__ == '__main__':
    unittest.main()
