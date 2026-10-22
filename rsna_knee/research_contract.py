"""Immutable experiment contracts and the final submission gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


COMMON_FIELDS = (
    'contract_version', 'evaluation_sha256', 'fold_sha256', 'pretrain_sha256',
    'cache_sha256', 'sampling', 'resolution', 'train_windows', 'eval_windows',
    'optimizer', 'augmentation', 'seed', 'epochs', 'batch_size',
    'accumulation_steps', 'code_version',
)
VARIANT_FIELDS = ('supervision_arm', 'pooling_arm', 'geometry_arm')
SUBMISSION_POLICY_VERSION = 'single-qualified-candidate-20260912'


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    supervision_arm: str
    pooling_arm: str
    geometry_arm: str
    contract_version: str
    evaluation_sha256: str
    fold_sha256: str
    pretrain_sha256: str
    cache_sha256: str
    sampling: str = 'physical_triplets_v1'
    resolution: int = 384
    train_windows: int = 12
    eval_windows: int = 24
    optimizer: str = 'adamw_encoder_2e-5_head_2e-4_wd_0.01_warmup_cosine'
    augmentation: str = 'shared_intensity_gain_0.9_1.1_no_flip'
    seed: int = 20260905
    epochs: int = 3
    batch_size: int = 1
    accumulation_steps: int = 8
    code_version: str = 'uncommitted'

    def as_dict(self):
        return {key: getattr(self, key) for key in COMMON_FIELDS + VARIANT_FIELDS if hasattr(self, key)} | {
            'experiment_id': self.experiment_id
        }

    def digest(self):
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(payload).hexdigest()


def comparison_policy(base_arm: str, candidate_arm: str) -> frozenset[str]:
    policies = {
        ('C', 'A'): frozenset({'supervision_arm'}),
        ('C', 'B-center'): frozenset({'pooling_arm'}),
        ('B-center', 'B-local'): frozenset({'pooling_arm'}),
        ('C', 'B-local'): frozenset({'pooling_arm'}),
        ('C', 'S'): frozenset({'pooling_arm'}),
        ('C', 'R-center'): frozenset({'geometry_arm'}),
        ('R-center', 'R-anatomic'): frozenset({'geometry_arm'}),
        ('C', 'R-anatomic'): frozenset({'geometry_arm'}),
        ('M-control', 'M-mask'): frozenset({'supervision_arm'}),
    }
    try:
        return policies[(base_arm, candidate_arm)]
    except KeyError as error:
        raise ValueError(f'unsupported comparison {base_arm}->{candidate_arm}') from error


def validate_comparison(base: dict, candidate: dict) -> dict:
    base_arm, candidate_arm = base.get('arm'), candidate.get('arm')
    allowed = comparison_policy(base_arm, candidate_arm)
    differences = {key for key in COMMON_FIELDS + VARIANT_FIELDS if base.get(key) != candidate.get(key)}
    # The whole bundle digest remains an audit field. A differing bundle can
    # be waived only by an actual remote-bundle receipt and deterministic
    # global-path probe; a caller supplied hash is never enough.
    component_receipt = candidate.get('comparison_receipt')
    code_version_waived = False
    if 'code_version' in differences:
        if not isinstance(component_receipt, dict) or component_receipt.get('compatible') is not True:
            raise ValueError('code_version differs without a passing component provenance receipt')
        if component_receipt.get('decision') != 'pass':
            raise ValueError('component provenance receipt did not issue a pass decision')
        differences.remove('code_version')
        code_version_waived = True
    if differences - allowed:
        raise ValueError(f'comparison changed frozen fields: {sorted(differences - allowed)}')
    if differences != allowed:
        raise ValueError(f'comparison did not declare the exact mechanism difference: {sorted(differences)} != {sorted(allowed)}')
    if base.get('contract_version') != 'top10-clean-v1' or candidate.get('contract_version') != 'top10-clean-v1':
        raise ValueError('unsupported contract')
    return {'base_arm': base_arm, 'candidate_arm': candidate_arm,
            'allowed_differences': sorted(allowed), 'observed_differences': sorted(differences),
            'code_version_waived_by_component_receipt': code_version_waived,
            'compatible': True}


def formal_submission_gate(*, data_ok, model_ok, oof_ok, subgroup_ok, runtime_ok,
                           smoke_ok, diversity_ok, quota_ok, candidate_count=1):
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or not 1 <= candidate_count <= 3:
        raise ValueError('candidate_count must be 1..3')
    checks = {
        'data': bool(data_ok), 'model': bool(model_ok), 'oof': bool(oof_ok),
        'subgroups': bool(subgroup_ok), 'runtime': bool(runtime_ok),
        'smoke': bool(smoke_ok),
        'batch_diversity': candidate_count == 1 or bool(diversity_ok),
        'daily_quota': bool(quota_ok),
    }
    return {'formal_submission_ready': all(checks.values()), 'checks': checks,
            'submission_policy': SUBMISSION_POLICY_VERSION,
            'batch_diversity_required': candidate_count > 1,
            'candidate_count': candidate_count,
            'missing': [name for name, passed in checks.items() if not passed]}
