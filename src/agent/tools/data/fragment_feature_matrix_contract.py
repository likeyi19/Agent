"""Fragment-derived neutral matrix, distinct from external matrix conservation."""
from dataclasses import asdict, replace
import hashlib
import sys

from . import scatac_matrix_contract as m
from .regulatory_matrix_contract import validate_binding

ARTIFACT = 'agent.scatac-cell-by-features'
CONTRACT = 'scatac-cell-by-features.v1'
RECOVERY_POLICY = 'build-scatac-cell-by-features-v1'
PROFILE = replace(m.PROFILE, profile_id='canonical-fragment-record-feature-overlap-counts.v1',
    overlap='same-contig;max(fragment_start,feature_start)<min(fragment_end,feature_end)',
    contribution='one-per-record-per-distinct-feature;no-fractional-weighting',
    rows='scatac-explicit-cells.v1-caller-order;contiguous;never-resort',
    empty_selection='valid-shape-(0,n_full_features);downstream-readiness-separate')
PROFILE_SHA256 = hashlib.sha256(m.canonical(asdict(PROFILE))).hexdigest()


def diagnostic(value):
    return {k.replace('any_ccre', 'any_feature'): v for k, v in value.items()}


def overlap_diagnostic(total, overlapping):
    return diagnostic(m.overlap_diagnostic(total, overlapping))


def manifest_identity(value):
    return hashlib.sha256(b'agent.cell-by-features-identity.v1\0' + m.canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


def validate_manifest(value):
    return m.validate_manifest(value, _contract=sys.modules[__name__])
