import numpy as np
import pytest

from rsna_knee.research_evidence_states import state_from_counts, state_summary


def test_state_precedence_and_summary():
    states = state_from_counts([1, 0, 0, 2], [0, 3, 0, 1])
    assert states.tolist() == ['explicit_positive', 'explicit_negative', 'silent', 'explicit_positive']
    summary = state_summary(states, ['YES', 'NO', 'UNK', 'NO'])
    assert summary['source_explicit_rows'] == 3
    assert summary['rule_source_conflict_rows'] == 1
    assert summary['state_counts']['silent'] == 1


def test_invalid_evidence_state_inputs_fail_closed():
    with pytest.raises(ValueError):
        state_from_counts([1], [1, 2])
    with pytest.raises(ValueError):
        state_from_counts([-1], [0])
    with pytest.raises(ValueError):
        state_summary(np.array(['invented']), np.array(['YES']))
