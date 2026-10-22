import unittest

import pandas as pd

from rsna_knee.constants import CONFIDENCE_COLUMNS, LABELS, LOSS_MASK_COLUMNS
from rsna_knee.research_masked_labels import (
    MaskedSupervisionUnavailable,
    prepare_masked_labels,
    qwen_explicit_verdicts,
)


class MaskedLabelTests(unittest.TestCase):
    def _control(self):
        data = {'StudyInstanceUID': ['u1', 'u2', 'u3']}
        for label in LABELS:
            data[label] = [.2, .8, .5]
            data[f'{label}__confidence'] = [.7, .8, .6]
        return pd.DataFrame(data)

    def test_probability_only_sources_do_not_invent_negative_evidence(self):
        control = self._control()
        probability_only = control[['StudyInstanceUID', *LABELS]].copy()
        probability_only[LABELS] = .01
        with self.assertRaisesRegex(MaskedSupervisionUnavailable, 'explicit YES/NO'):
            prepare_masked_labels(control, {'one': probability_only, 'two': probability_only})

    def test_only_explicit_cross_source_consensus_is_retained(self):
        control = self._control()
        one = control[['StudyInstanceUID']].copy(); two = control[['StudyInstanceUID']].copy()
        for label in LABELS:
            one[f'{label}__verdict'] = ['YES', 'NO', 'UNK']
            two[f'{label}__verdict'] = ['YES', 'YES', 'NO']
        result, audit = prepare_masked_labels(control, {'one': one, 'two': two})
        self.assertTrue(audit.available)
        self.assertEqual(result[LOSS_MASK_COLUMNS[0]].tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(result[['StudyInstanceUID', *LABELS, *CONFIDENCE_COLUMNS]].equals(
            control[['StudyInstanceUID', *LABELS, *CONFIDENCE_COLUMNS]]
        ))

    def test_qwen_unmentioned_and_ambiguous_are_not_negative(self):
        table = pd.DataFrame({'StudyInstanceUID': ['u1', 'u2', 'u3', 'u4'], 'status': ['ok', 'ok', 'ok', 'error']})
        for label in LABELS:
            table[f'{label}__label'] = [1, 0, 0, 1]
            table[f'{label}__mentioned'] = [1, 0, 1, 1]
            table[f'{label}__negated'] = [0, 0, 0, 1]
        verdicts = qwen_explicit_verdicts(table)
        self.assertEqual(verdicts[f'{LABELS[0]}__verdict'].tolist(), ['YES', 'UNK', 'UNK', 'UNK'])


if __name__ == '__main__':
    unittest.main()
