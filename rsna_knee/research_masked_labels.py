"""Auditable loss-mask preparation for report-derived weak labels.

This is deliberately stricter than score agreement: an absent verdict column is
not interpreted as negative evidence.  A masked-label experiment is unavailable
until at least two independent source families provide explicit YES/NO states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .constants import LABELS, LOSS_MASK_COLUMNS
from .research_validation import UID


class MaskedSupervisionUnavailable(RuntimeError):
    """Raised when source tables cannot support an explicit-state experiment."""


@dataclass(frozen=True)
class MaskedLabelAudit:
    available: bool
    explicit_source_families: tuple[str, ...]
    per_label: dict[str, dict[str, int]]
    reason: str | None = None


def _validate_source(name: str, table: pd.DataFrame, expected_uids: pd.Series) -> pd.DataFrame:
    if UID not in table or table[UID].duplicated().any():
        raise ValueError(f'invalid source UID table: {name}')
    indexed = table.set_index(UID).reindex(expected_uids)
    if indexed.index.isna().any():
        raise ValueError(f'source UID alignment failed: {name}')
    return indexed


def prepare_masked_labels(control: pd.DataFrame, sources: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, MaskedLabelAudit]:
    """Return a confidence-preserving table with target-only loss masks.

    Each retained target has explicit YES/NO from every explicit-state family
    and all families agree.  Probability-only families are reported but never
    promoted into explicit evidence.
    """
    if UID not in control or control[UID].duplicated().any():
        raise ValueError('invalid control table')
    aligned = {name: _validate_source(name, table, control[UID]) for name, table in sources.items()}
    explicit_families = tuple(sorted(name for name, table in aligned.items()
                                     if all(f'{label}__verdict' in table.columns for label in LABELS)))
    per_label: dict[str, dict[str, int]] = {}
    if len(explicit_families) < 2:
        audit = MaskedLabelAudit(
            available=False, explicit_source_families=explicit_families, per_label=per_label,
            reason='fewer than two independent source families provide explicit YES/NO states',
        )
        raise MaskedSupervisionUnavailable(audit.reason)
    result = control.copy()
    for label, mask_column in zip(LABELS, LOSS_MASK_COLUMNS):
        verdicts = np.stack([
            aligned[name][f'{label}__verdict'].fillna('UNK').astype(str).to_numpy()
            for name in explicit_families
        ], axis=1)
        explicit = np.isin(verdicts, ('YES', 'NO')).all(axis=1)
        same = (verdicts == verdicts[:, :1]).all(axis=1)
        keep = explicit & same
        result[mask_column] = keep.astype(np.float32)
        per_label[label] = {
            'retained_explicit_consensus': int(keep.sum()),
            'masked_unmentioned_or_unk': int((~explicit).sum()),
            'masked_explicit_conflict': int((explicit & ~same).sum()),
        }
    # Supervision only: probabilities and confidence must remain byte-for-byte
    # equivalent to the fixed control table.
    protected = [column for column in control.columns if column not in LOSS_MASK_COLUMNS]
    if not result[protected].equals(control[protected]):
        raise AssertionError('masked-label preparation changed existing control values')
    return result, MaskedLabelAudit(True, explicit_families, per_label)


def qwen_explicit_verdicts(table: pd.DataFrame) -> pd.DataFrame:
    """Convert the existing Qwen evidence columns into a minimal verdict table.

    ``mentioned=0`` remains UNK: it is deliberately not converted into a
    negative label.  Only a successfully parsed, mentioned, non-negated
    positive or mentioned, negated normality statement supplies YES/NO.
    """
    if UID not in table or table[UID].duplicated().any():
        raise ValueError('invalid Qwen UID table')
    output = table[[UID]].copy()
    status = table.get('status', pd.Series('ok', index=table.index)).fillna('error').astype(str).eq('ok')
    for label in LABELS:
        required = (f'{label}__label', f'{label}__mentioned', f'{label}__negated')
        if any(column not in table for column in required):
            raise ValueError(f'Qwen table lacks evidence columns for {label}')
        raw_label = pd.to_numeric(table[required[0]], errors='coerce')
        mentioned = pd.to_numeric(table[required[1]], errors='coerce').eq(1)
        negated = pd.to_numeric(table[required[2]], errors='coerce').eq(1)
        verdict = np.full(len(table), 'UNK', dtype=object)
        verdict[status.to_numpy() & mentioned.to_numpy() & negated.to_numpy()] = 'NO'
        verdict[status.to_numpy() & mentioned.to_numpy() & ~negated.to_numpy() & raw_label.eq(1).to_numpy()] = 'YES'
        # Mentioned-but-label-0/non-negated remains unresolved rather than an
        # invented negative statement.
        output[f'{label}__verdict'] = verdict
    return output


def validate_masked_label_table(control: pd.DataFrame, masked: pd.DataFrame) -> None:
    """Ensure a deployment/trainer sees an immutable supervision-only delta."""
    if UID not in masked or masked[UID].duplicated().any() or not masked[UID].equals(control[UID]):
        raise ValueError('masked label UID order mismatch')
    if any(column not in masked for column in LOSS_MASK_COLUMNS):
        raise ValueError('masked label table lacks complete target masks')
    protected = [column for column in control.columns if column in masked.columns]
    if not masked[protected].equals(control[protected]):
        raise ValueError('masked label table changed frozen targets or confidence')
    values = masked[LOSS_MASK_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('masked label table has invalid masks')
