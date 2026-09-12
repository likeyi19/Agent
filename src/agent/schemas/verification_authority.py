"""Versioned verification provenance. Parsed records alone confer no authority."""
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from collections.abc import Mapping

from .orchestration import freeze_json_mapping, _serialize


class VerificationScope(str, Enum):
    SCIENTIFIC = 'scientific_correctness.v1'
    INTEGRITY = 'artifact_integrity_lineage.v1'
    PRESENTATION = 'presentation_validation.v1'


class AuthorityError(ValueError):
    code = 'VERIFICATION_AUTHORITY_INVALID'


def canonical(value):
    return json.dumps(_serialize(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise AuthorityError('Invalid authority digest.')


@dataclass(frozen=True)
class AuthorityValidation:
    """Current validation of prior authority, not fresh scientific verification."""
    authority_sha256: str
    scope: VerificationScope = VerificationScope.INTEGRITY

    def __post_init__(self):
        _sha(self.authority_sha256)
        if self.scope is not VerificationScope.INTEGRITY:
            raise AuthorityError('Authority reuse establishes integrity/lineage only.')


@dataclass(frozen=True)
class VerifiedArtifactAuthority:
    """Closed v1 record; only an accepted RunStore step can anchor its reuse.

    File closure includes physical payloads, sidecars, upstream manifests and
    source/resources. Semantic identities remain explicit in separate fields.
    A receipt is one binding, never the authority issuer.
    """
    record: Mapping

    def __post_init__(self):
        value = self.record
        fields = {'schema_version', 'artifact_type', 'artifact_contract',
                  'publication_path', 'manifest_sha256', 'files', 'execution_identity',
                  'arguments_sha256', 'receipt_sha256', 'upstream', 'resources',
                  'science_profile', 'producer_qualification', 'verifier', 'scope',
                  'completion', 'integrity_scope'}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise AuthorityError('Invalid authority schema.')
        if type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise AuthorityError('Unsupported authority schema.')
        if value['completion'] != 'succeeded' or value['scope'] not in {s.value for s in VerificationScope}:
            raise AuthorityError('Incomplete or unknown verification scope.')
        if value['integrity_scope'] != VerificationScope.INTEGRITY.value:
            raise AuthorityError('Missing canonical integrity scope.')
        qualification = value['producer_qualification']
        if (not isinstance(qualification, Mapping)
                or (qualification and (set(qualification) != {'kind', 'scope', 'profile_id', 'compatibility_version'}
                                      or any(type(v) is not str or not v for v in qualification.values())))):
            raise AuthorityError('Invalid producer qualification scope.')
        for key in ('artifact_type', 'artifact_contract', 'publication_path', 'science_profile'):
            if type(value[key]) is not str or not value[key]:
                raise AuthorityError('Missing authority identity.')
        for key in ('manifest_sha256', 'execution_identity', 'arguments_sha256', 'receipt_sha256'):
            _sha(value[key])
        verifier = value['verifier']
        if (not isinstance(verifier, Mapping) or set(verifier) != {'id', 'compatibility_version'}
                or any(type(v) is not str or not v for v in verifier.values())):
            raise AuthorityError('Missing verifier identity or compatibility version.')
        for key in ('upstream', 'resources'):
            if not isinstance(value[key], Mapping) or not value[key]:
                raise AuthorityError('Missing upstream/resource authority.')
        files = value['files']
        if not isinstance(files, (tuple, list)) or not files:
            raise AuthorityError('Missing physical digest closure.')
        paths = []
        from pathlib import Path
        for entry in files:
            if not isinstance(entry, Mapping) or set(entry) != {'path', 'sha256', 'size_bytes'}:
                raise AuthorityError('Invalid file binding.')
            if type(entry['path']) is not str or not Path(entry['path']).is_absolute():
                raise AuthorityError('Invalid authority path.')
            _sha(entry['sha256'])
            if type(entry['size_bytes']) is not int or entry['size_bytes'] < 0:
                raise AuthorityError('Invalid file size.')
            paths.append(entry['path'])
        if paths != sorted(set(paths)):
            raise AuthorityError('Noncanonical file closure.')
        object.__setattr__(self, 'record', freeze_json_mapping(value, 'artifact_authority'))

    def to_dict(self):
        return _serialize(self.record)

    @property
    def identity_sha256(self):
        return digest(self.record)
