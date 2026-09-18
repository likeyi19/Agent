"""QC-selected neutral rows; unchanged shared fragment-overlap science."""
from dataclasses import asdict, replace
import hashlib
import sys

from . import fragment_feature_matrix_contract as explicit, scatac_matrix_contract as legacy
from .regulatory_matrix_contract import validate_binding

ARTIFACT = explicit.ARTIFACT
CONTRACT = 'scatac-cell-by-features.qc-selected.v1'
RECOVERY_POLICY = explicit.RECOVERY_POLICY
CELL_CONTRACT = 'scatac-cell-selection.v1'
PROFILE = replace(explicit.PROFILE, profile_id='canonical-fragment-record-selected-feature-overlap-counts.v1',
                  rows=legacy.PROFILE.rows)
PROFILE_SHA256 = hashlib.sha256(legacy.canonical(asdict(PROFILE))).hexdigest()
diagnostic = explicit.diagnostic
overlap_diagnostic = explicit.overlap_diagnostic


def manifest_identity(value):
    return hashlib.sha256(CONTRACT.encode() + b'\0' + legacy.canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


def validate_manifest(value):
    return legacy.validate_manifest(value, _contract=sys.modules[__name__])
