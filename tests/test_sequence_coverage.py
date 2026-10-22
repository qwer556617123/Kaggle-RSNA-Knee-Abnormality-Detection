import itertools,unittest
import numpy as np
from rsna_knee.sequence_coverage import select_valid_training_windows

class CoverageTests(unittest.TestCase):
    def test_every_protocol_availability_pattern(self):
        for present in itertools.product((False,True),repeat=5):
            if not any(present): continue
            mask=np.repeat(present,(4,6,6,4,4))
            for seed in range(20):
                out=select_valid_training_windows(mask,np.random.RandomState(seed))
                self.assertEqual(len(out),12);self.assertEqual(len(set(out)),12)
                self.assertEqual(int(mask[out].sum()),min(12,int(mask.sum())))
                for slot in range(5):
                    bounds=np.cumsum((0,4,6,6,4,4))
                    if present[slot]: self.assertTrue(any(bounds[slot]<=i<bounds[slot+1] for i in out))
    def test_complete_studies_exact_parent_rng(self):
        for seed in range(50):
            a=np.random.RandomState(seed);b=np.random.RandomState(seed);expected=[];offset=0
            for pool,count in zip((4,6,6,4,4),(2,3,3,2,2)):
                expected.extend((offset+np.sort(a.choice(pool,count,replace=False))).tolist());offset+=pool
            np.testing.assert_array_equal(select_valid_training_windows(np.ones(24),b),expected)
            self.assertEqual(a.random_sample(),b.random_sample())
    def test_partial_within_slot_masks(self):
        mask=np.zeros(24);mask[[0,2,4,6,8,10,12,14,16,18,20,22]]=1
        for seed in range(20):
            out=select_valid_training_windows(mask,np.random.RandomState(seed))
            self.assertTrue(mask[out].all())
    def test_invalid_coverage_rejected(self):
        for mask in (np.zeros(24),np.ones(23),np.full(24,.5),np.full(24,np.nan)):
            with self.assertRaises(ValueError):select_valid_training_windows(mask)
    def test_trainer_only_selection_diff(self):
        from pathlib import Path
        old=Path('rsna_knee/research_trainer.py').read_text(encoding='utf-8')
        new=Path('rsna_knee/sequence_coverage_trainer.py').read_text(encoding='utf-8')
        new=new.replace('from .sequence_coverage import select_valid_training_windows\n','').replace("            selected = select_valid_training_windows(masks[location['row']])",'''            selected = []
            offset = 0
            for pool, count in zip((4,6,6,4,4), (2,3,3,2,2)):
                selected.extend((offset + np.sort(np.random.choice(pool, count, replace=False))).tolist())
                offset += pool''')
        self.assertEqual(old,new)
if __name__=='__main__':unittest.main()
