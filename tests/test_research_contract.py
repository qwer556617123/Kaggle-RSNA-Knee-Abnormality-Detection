import unittest

from rsna_knee.research_contract import formal_submission_gate, validate_comparison


def spec(arm, *, supervision='control', pooling='global'):
    return {
        'arm': arm, 'contract_version': 'top10-clean-v1', 'evaluation_sha256': 'eval',
        'fold_sha256': 'fold', 'pretrain_sha256': 'pre', 'cache_sha256': 'cache',
        'sampling': 'physical_triplets_v1', 'resolution': 384, 'train_windows': 12,
        'eval_windows': 24, 'optimizer': 'opt', 'augmentation': 'aug', 'seed': 20260905,
        'epochs': 3, 'batch_size': 1, 'accumulation_steps': 8, 'code_version': 'code',
        'supervision_arm': supervision, 'pooling_arm': pooling, 'geometry_arm': 'canonical',
    }


class ResearchContractTests(unittest.TestCase):
    def test_a_allows_only_confidence(self):
        result = validate_comparison(spec('C'), spec('A', supervision='evidence'))
        self.assertEqual(result['allowed_differences'], ['supervision_arm'])

    def test_local_pooling_contracts(self):
        self.assertTrue(validate_comparison(spec('C'), spec('B-center', pooling='center'))['compatible'])
        self.assertTrue(validate_comparison(spec('B-center', pooling='center'), spec('B-local', pooling='target_query'))['compatible'])
        self.assertTrue(validate_comparison(spec('C'), spec('S', pooling='slot_target_query'))['compatible'])

    def test_canonical_roi_comparisons_only_allow_geometry(self):
        center = spec('R-center'); center['geometry_arm'] = 'canonical_center_96mm_v1'
        local = spec('R-anatomic'); local['geometry_arm'] = 'canonical_plane_roi_96mm_v1'
        self.assertTrue(validate_comparison(spec('C'), center)['compatible'])
        self.assertTrue(validate_comparison(center, local)['compatible'])

    def test_masked_supervision_contract_allows_only_loss_mask_arm(self):
        base = spec('M-control')
        candidate = spec('M-mask', supervision='masked')
        self.assertTrue(validate_comparison(base, candidate)['compatible'])
        candidate['geometry_arm'] = 'native'
        with self.assertRaises(ValueError):
            validate_comparison(base, candidate)

    def test_unregistered_change_is_rejected(self):
        changed = spec('A', supervision='evidence'); changed['resolution'] = 336
        with self.assertRaises(ValueError):
            validate_comparison(spec('C'), changed)

    def test_extra_or_missing_difference_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_comparison(spec('C'), spec('A'))
        changed = spec('A', supervision='evidence'); changed['pooling_arm'] = 'target_query'
        with self.assertRaises(ValueError):
            validate_comparison(spec('C'), changed)

    def test_submission_gate_is_fail_closed(self):
        result = formal_submission_gate(data_ok=True, model_ok=True, oof_ok=True, subgroup_ok=True,
                                        runtime_ok=True, smoke_ok=True, diversity_ok=False, quota_ok=True,
                                        candidate_count=1)
        self.assertTrue(result['formal_submission_ready'])
        self.assertFalse(result['batch_diversity_required'])
        self.assertTrue(formal_submission_gate(data_ok=True, model_ok=True, oof_ok=True, subgroup_ok=True,
                                               runtime_ok=True, smoke_ok=True, diversity_ok=True, quota_ok=True,
                                               candidate_count=2)['formal_submission_ready'])

    def test_single_candidate_keeps_all_quality_gates(self):
        good = dict(data_ok=True, model_ok=True, oof_ok=True, subgroup_ok=True,
                    runtime_ok=True, smoke_ok=True, diversity_ok=False, quota_ok=True, candidate_count=1)
        for field in ('data_ok', 'model_ok', 'oof_ok', 'subgroup_ok', 'runtime_ok', 'smoke_ok', 'quota_ok'):
            for value in (False, None):
                with self.subTest(field=field, value=value):
                    self.assertFalse(formal_submission_gate(**dict(good, **{field: value}))['formal_submission_ready'])
        self.assertFalse(formal_submission_gate(**dict(good, candidate_count=2))['formal_submission_ready'])
        for count in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                formal_submission_gate(**dict(good, candidate_count=count))


if __name__ == '__main__':
    unittest.main()
