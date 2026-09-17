"""Neutral external matrix profile sharing the accepted conservation contract."""
from dataclasses import asdict, dataclass
import hashlib
import sys

from . import external_matrix_contract as e, scatac_matrix_contract as m
from . import regulatory_feature_reference as r, scatac_reference as reference_helpers

ARTIFACT = 'agent.scatac-cell-by-features'
CONTRACT = 'scatac-cell-by-features.external.v1'
REFERENCE_CONTRACT = r.CONTRACT
RECOVERY_POLICY = 'adopt-scatac-cell-by-features-v1'
PROFILE = 'exact-external-cell-by-features-adoption.v1'
PROFILE_SPEC = dict(e.PROFILE_SPEC, profile_id=PROFILE,
    reference='regulatory-feature-reference.v1;explicit-species-and-assembly;declared-feature-category',
    history='external-declaration;no-fragment-QC-selection-curation-or-model-readiness-authority')
PROFILE_SHA256 = hashlib.sha256(m.canonical(PROFILE_SPEC)).hexdigest()
SEMANTICS = e.SEMANTICS


def validate_binding(species, assembly):
    if type(species) is not dict:
        m.fail('MATRIX_SEMANTICS_INVALID')
    r.validate_species(species)
    reference_helpers._text(assembly)


def identity(value):
    return hashlib.sha256(b'agent.external-cell-by-features.v1\0' + m.canonical(
        {k: v for k, v in value.items() if k != 'identity_sha256'})).hexdigest()


def validate(value):
    return e.validate(value, _profile=sys.modules[__name__])


def load(raw):
    return e.load(raw, _profile=sys.modules[__name__])


@dataclass(frozen=True)
class _MatrixReference:
    """Private legacy-owner field adapter, never a cCRE artifact or qualification."""
    bundle: r.RegulatoryFeatureReference

    @property
    def species(self): return asdict(self.bundle.species)
    @property
    def target_assembly(self): return self.bundle.target_assembly
    @property
    def genome(self): return self.bundle.genome
    @property
    def ccre(self): return self.bundle.features
    @property
    def annotation(self): return None
    @property
    def reference_identity_sha256(self): return self.bundle.reference_identity_sha256


def load_reference(path, *, expected_sha256=None):
    path, bundle, sha = r.load_regulatory_feature_reference(path, expected_sha256=expected_sha256)
    return path, _MatrixReference(bundle), sha


def reinspect_reference(reference):
    r.reinspect_regulatory_feature_reference(reference.bundle)
