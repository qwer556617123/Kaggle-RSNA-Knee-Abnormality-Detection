import unittest

from rsna_knee.gates import evaluate_candidate


class CandidateGateTests(unittest.TestCase):
    def test_gold_improvement_passes(self):
        result = evaluate_candidate(
            candidate_gold_auc=0.71,
            candidate_weak_auc=0.79,
            incumbent_gold_auc=0.69,
            incumbent_weak_auc=0.80,
        )
        self.assertTrue(result.accepted)

    def test_weak_improvement_with_gold_guard(self):
        result = evaluate_candidate(
            candidate_gold_auc=0.68,
            candidate_weak_auc=0.82,
            incumbent_gold_auc=0.69,
            incumbent_weak_auc=0.80,
        )
        self.assertTrue(result.accepted)
        rejected = evaluate_candidate(
            candidate_gold_auc=0.66,
            candidate_weak_auc=0.82,
            incumbent_gold_auc=0.69,
            incumbent_weak_auc=0.80,
        )
        self.assertFalse(rejected.accepted)


if __name__ == "__main__":
    unittest.main()
