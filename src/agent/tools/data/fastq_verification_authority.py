"""FASTQ authority capture/integrity checks; never alignment or QC science.

This first boundary deliberately hashes all bound files, including raw sources.
It eliminates structural reconstruction, not integrity I/O. No stat-only cache.
"""
from pathlib import Path
import hashlib

from agent.schemas.verification_authority import (
    AuthorityError, VerificationScope, VerifiedArtifactAuthority, digest,
)
from agent.tools._cancellation import cancellation_checkpoint
from . import fastq_fragment_manifest as producer, scatac_fragments as public
from . import scatac_fragments_v2 as v2
from ._fragments_binding import FragmentInputs, preflight, fresh_intake
from ._fragments_common import snapshots, unchanged
from . import _chromap as c
from .fragments_authority_contract import CONTRACTS

CONTRACT = CONTRACTS['fastq_fragment_production']


def file_closure(paths):
    entries = []
    paths = sorted(set(map(str, paths)))
    before = snapshots(paths)
    for name in paths:
        path = Path(name)
        if path != path.resolve() or not path.is_file():
            raise AuthorityError('Noncanonical or unsafe authority file.')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                cancellation_checkpoint()
                sha.update(chunk)
        entries.append(dict(path=name, sha256=sha.hexdigest(), size_bytes=path.stat().st_size))
    unchanged(before)
    return entries


def capture_publication(arguments, result, execution_identity):
    """Capture before deep verification, or compare on reuse; does not grant trust."""
    path = Path(result['manifest_path'])
    destination, token, _ = public._publication(arguments, execution_identity)
    if path != destination / 'fragments' / 'manifest.json' or path != path.resolve():
        raise AuthorityError('Publication does not belong to this execution.')
    receipt = public._read_receipt(destination, arguments, token)
    if receipt['manifest_sha256'] != result['manifest_sha256']:
        raise AuthorityError('Receipt/manifest mismatch.')
    value = producer.load_manifest(path, expected_sha256=result['manifest_sha256'])
    if dict(result) != public._summary(value, path, result['manifest_sha256']):
        raise AuthorityError('Result differs from the publication.')
    record = producer.record_for_manifest(value, path.parent)
    bound = preflight(FragmentInputs(**record['inputs']), c.validate_backend(record['backend']), fresh=False)
    # Bounded inventory reinspection detects added/removed discovered sources.
    fresh_intake(bound['intake'])
    paths = list(bound['snapshots']) + [path, path.parent / 'production.json',
                                      path.parent / 'profile.json', destination / 'receipt.json']
    for library in value['libraries']:
        paths.extend(path.parent / library[k]['path'] for k in ('bgzf', 'tabix'))
    files = file_closure(paths)
    upstream = {
        key: {'manifest_path': record['inputs'][key + '_path'],
              'manifest_sha256': record['inputs'][key + '_sha256'],
              'accepted_scope': 'source_resource_reconstruction_within_fastq_verifier.v1'}
        for key in ('intake', 'context', 'reference', 'index')
    }
    return VerifiedArtifactAuthority(dict(
        schema_version=1, artifact_type=v2.ARTIFACT_TYPE, artifact_contract=v2.CONTRACT_VERSION,
        publication_path=str(path), manifest_sha256=result['manifest_sha256'], files=files,
        execution_identity=execution_identity, arguments_sha256=digest(public._arguments(arguments)),
        receipt_sha256=next(f['sha256'] for f in files if f['path'] == str(destination / 'receipt.json')),
        upstream=upstream, resources={'lineage': record['lineage'], 'backend': record['backend'],
                                     'library_bindings': [dict(namespace=e['namespace'],
                                         whitelist_sha256=e['whitelist_sha256'],
                                         whitelist_set_sha256=e['whitelist_set_sha256'])
                                         for e in record['libraries']]},
        science_profile=record['profile_sha256'], producer_qualification=CONTRACT.qualification,
        verifier=CONTRACT.verifier, scope=VerificationScope.SCIENTIFIC.value, completion='succeeded',
        integrity_scope=VerificationScope.INTEGRITY.value,
    ))


def require_compatible(authority):
    CONTRACT.require(authority)


def check_integrity(authority, arguments, result, execution_identity):
    require_compatible(authority)
    current = capture_publication(arguments, result, execution_identity)
    if current != authority:
        raise AuthorityError('Authority invalidated by publication, lineage or resource changes.')
