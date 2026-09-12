"""Producer-neutral v2 authority requirements; qualification never crosses origins.

The .1 qualification contracts are shared by all .2 scientific DAG adapters.
The legacy version-1 physical validator below retains FASTQ-only support.
"""
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Callable

from agent.schemas.verification_authority import AuthorityError, VerificationScope
from . import scatac_fragments_v2 as v2
from . import fastq_fragment_manifest as fastq, bam_fragment_manifest as bam
from . import external_fragment_manifest as external


@dataclass(frozen=True, init=False)
class StoredFragmentAuthority:
    """Opaque immutable capability supplied by accepted execution persistence.

    Data consumers depend on this contract, never on orchestration or RunStore.
    The private factory is for the repository's persistence adapter only; trusted
    Python code is inside the boundary, arbitrary provider callables are not.
    """
    _validate: Callable

    def __init__(self, *args, **kwargs):
        raise AuthorityError('Authority must be loaded from accepted execution persistence.')

    @classmethod
    def _from_accepted_store(cls, validate):
        handle = object.__new__(cls)
        object.__setattr__(handle, '_validate', validate)
        return handle

    def validate(self, path, sha, *, producer_kind=None):
        return self._validate(path, sha, producer_kind)


@dataclass(frozen=True)
class FragmentsAuthorityContract:
    producer_kind: str
    tool_name: str
    verifier_id: str
    qualification_scope: str
    profile_id: str
    profile_sha256: str
    compatibility_version: str = '1'

    @property
    def qualification(self):
        return dict(kind=self.producer_kind, scope=self.qualification_scope,
                    profile_id=self.profile_id, compatibility_version=self.compatibility_version)

    @property
    def verifier(self):
        return dict(id=self.verifier_id, compatibility_version=self.compatibility_version)

    def require(self, authority):
        value = authority.record
        integrity = VerificationScope.INTEGRITY if value['schema_version'] == 1 else VerificationScope.HISTORICAL_INTEGRITY
        if (value['artifact_type'] != v2.ARTIFACT_TYPE or value['artifact_contract'] != v2.CONTRACT_VERSION
                or value['integrity_scope'] != integrity.value
                or value['scope'] != VerificationScope.SCIENTIFIC.value
                or value['producer_qualification'] != self.qualification
                or value['science_profile'] != self.profile_sha256 or value['verifier'] != self.verifier):
            raise AuthorityError('Incompatible producer-qualified fragments authority.')


def _contract(kind, tool, verifier, scope, profile):
    return FragmentsAuthorityContract(kind, tool, verifier, scope, profile.PROFILE['profile_id'],
                                     hashlib.sha256(profile.PROFILE_BYTES).hexdigest())


CONTRACTS = MappingProxyType({c.producer_kind: c for c in (
    _contract('fastq_fragment_production', 'prepare_scATAC_fragments',
              'agent.fastq-fragments-independent', 'fastq_source_and_producer_record.v1', fastq),
    _contract('bam_fragment_production', 'prepare_scATAC_bam_fragments',
              'agent.bam-fragments-independent', 'bam_read_pair_transformation.v1', bam),
    _contract('external_fragment_adoption', 'import_scATAC_fragments',
              'agent.external-fragments-independent', 'external_source_conservation.v1', external),
)})


def contract_for_tool(tool_name):
    matches = [c for c in CONTRACTS.values() if c.tool_name == tool_name]
    if len(matches) != 1:
        raise AuthorityError('Unsupported fragments producer contract.')
    return matches[0]


def validate_publication_integrity(authority, arguments, result, execution_identity):
    kind = authority.record['producer_qualification'].get('kind')
    if kind not in CONTRACTS:
        raise AuthorityError('Producer qualification is required.')
    CONTRACTS[kind].require(authority)
    if kind == 'fastq_fragment_production':
        from .fastq_verification_authority import check_integrity
        check_integrity(authority, arguments, result, execution_identity)
    else:
        raise AuthorityError('This producer has no enabled persistence/reuse adapter yet.')
