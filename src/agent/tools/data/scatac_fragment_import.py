"""Registered external adoption and exact durable publication receipt recovery."""
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict

from . import external_fragment_manifest as m, _external_fragment_io as io, scatac_fragments_v2 as v2
from ._fragments_common import canonical, digest
from .scatac_fragments_v2_verifier import FragmentVerificationRuntime
from .external_fragments_verifier import verify_external_fragments

RECOVERY_POLICY = 'import-scatac-fragments-external-v1'


class ExternalFragmentsResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    source_profile: str
    source_encoding: str
    species: str
    assembly: str
    namespace: str
    n_libraries: int
    n_fragment_records: int
    total_support: int
    strand_mode: str


def verification_runtime():
    return FragmentVerificationRuntime()


def _fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publication(arguments, execution_identity):
    args = m.validate_arguments(arguments)
    token = digest({'arguments': args, 'policy': RECOVERY_POLICY, 'durable_execution_identity': execution_identity})
    return args, Path(args['output_dir']) / ('external-fragments-' + token), token


def _summary(value, path, sha):
    entry = value['libraries'][0]
    record = m.load_adoption_record(entry['provenance']['producer_record']['path'], entry['provenance']['producer_record']['sha256'])
    return ExternalFragmentsResult(status='success', manifest_path=str(path), manifest_sha256=sha,
        artifact_type=v2.ARTIFACT_TYPE, artifact_schema_version=2, contract_version=v2.CONTRACT_VERSION,
        source_profile=m.PROFILE_ID, source_encoding=record['source']['encoding'],
        species=value['reference']['species'], assembly=value['reference']['assembly'], namespace=entry['namespace'],
        n_libraries=1, n_fragment_records=entry['n_fragment_records'], total_support=entry['sum_support'], strand_mode=entry['strand']['mode'])


def _receipt(destination, arguments, expected_token=None):
    try:
        path = destination / 'receipt.json'; before = io.take_snapshots([path])
        with path.open('rb') as source:
            raw = source.read(16385)
        if len(raw) > 16384:
            m.fail('EXTERNAL_FRAGMENTS_RECOVERY_MISMATCH')
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=v2._pairs, parse_constant=lambda _: m.fail())
        v2.shape(value, ('artifact_type', 'schema_version', 'policy', 'execution_identity', 'arguments_sha256', 'manifest_sha256'))
        if (value['artifact_type'] != 'agent.external-fragments-execution-receipt'
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['policy'] != RECOVERY_POLICY or value['arguments_sha256'] != digest(arguments)
                or destination.name != 'external-fragments-' + value['execution_identity']
                or (expected_token is not None and value['execution_identity'] != expected_token)):
            m.fail('EXTERNAL_FRAGMENTS_RECOVERY_MISMATCH')
        v2.sha(value['execution_identity']); v2.sha(value['manifest_sha256']); io.check_snapshots(before)
        return value
    except (OSError, ValueError, KeyError, TypeError):
        m.fail('EXTERNAL_FRAGMENTS_RECOVERY_MISMATCH')


def verify_public_result(arguments, result):
    args = m.validate_arguments(arguments)
    path = Path(result['manifest_path']); destination = path.parent.parent
    if (path != path.resolve() or path.name != 'manifest.json' or path.parent.name != 'fragments'
            or destination.parent != Path(args['output_dir'])):
        m.fail('EXTERNAL_FRAGMENTS_RESULT_MISMATCH')
    before = io.take_snapshots([destination / 'receipt.json', path])
    receipt = _receipt(destination, args)
    if receipt['manifest_sha256'] != result['manifest_sha256']:
        m.fail('EXTERNAL_FRAGMENTS_RECOVERY_MISMATCH')
    verified = verify_external_fragments(path, expected_sha256=result['manifest_sha256'], runtime=verification_runtime())
    value = verified.fragments.manifest; entry = value['libraries'][0]
    record = m.load_adoption_record(entry['provenance']['producer_record']['path'], entry['provenance']['producer_record']['sha256'])
    source = record['source']['resource']; index = record['source_index']; reference = record['reference']
    if (source['path'] != args['source_path'] or source['sha256'] != args['source_sha256']
            or reference['manifest_path'] != args['reference_bundle_path'] or reference['manifest_sha256'] != args['reference_bundle_sha256']
            or record['namespace'] != args['namespace'] or record['source_selection'] != args['source_selection']
            or (None if index is None else index['path']) != args['source_index_path']
            or (None if index is None else index['sha256']) != args['source_index_sha256']
            or dict(result) != _summary(value, path, result['manifest_sha256'])):
        m.fail('EXTERNAL_FRAGMENTS_RESULT_MISMATCH')
    io.check_snapshots(before)
    return verified


def recover_external_fragments(arguments, execution_identity):
    args, destination, token = _publication(arguments, execution_identity)
    if not destination.is_dir() or destination.is_symlink():
        m.fail('EXTERNAL_FRAGMENTS_RECOVERY_UNAVAILABLE')
    receipt = _receipt(destination, args, token); path = destination / 'fragments' / 'manifest.json'
    value = v2.load_fragments_manifest_v2(path, expected_sha256=receipt['manifest_sha256'])
    result = _summary(value, path, receipt['manifest_sha256'])
    verify_public_result(args, result)
    return result


def execute_external_fragments(arguments, execution_identity=None):
    from .external_fragments import prepare_in_stage
    args, destination, token = _publication(arguments, execution_identity)
    output = Path(args['output_dir'])
    if output != output.resolve() or destination.exists() or destination.is_symlink():
        m.fail('EXTERNAL_FRAGMENTS_ARTIFACT_CONFLICT')
    output.mkdir(parents=True, exist_ok=True)
    with (output / ('.' + destination.name + '.lock')).open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            m.fail('EXTERNAL_FRAGMENTS_ARTIFACT_CONFLICT')
        stage = Path(tempfile.mkdtemp(prefix='.external-attempt-', dir=output))
        try:
            fragments = stage / 'fragments'; fragments.mkdir()
            runtime = verification_runtime()
            value, snapshots = prepare_in_stage(args, fragments, runtime)
            manifest = fragments / 'manifest.json'
            verified = verify_external_fragments(manifest, expected_sha256=io.file_sha256(manifest), runtime=runtime, temporary_root=stage)
            # Only two locally produced resource locators change on atomic rename.
            # Their bytes/identities and all source/reference locators stay fixed.
            final = deepcopy(value); p = final['libraries'][0]['provenance']
            p['profile']['resource']['path'] = str(destination / 'fragments' / 'profile.json')
            p['producer_record']['path'] = str(destination / 'fragments' / 'adoption.json')
            payload = v2.canonical_fragments_manifest_v2_bytes(final)
            verified.fragments.check_unchanged(); io.check_snapshots(snapshots)
            manifest.write_bytes(payload); sha = m.sha_bytes(payload)
            (stage / 'receipt.json').write_bytes(canonical(dict(artifact_type='agent.external-fragments-execution-receipt',
                schema_version=1, policy=RECOVERY_POLICY, execution_identity=token,
                arguments_sha256=digest(args), manifest_sha256=sha)))
            for path in stage.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            for path in sorted(stage.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if path.is_dir():
                    _fsync_directory(path)
            _fsync_directory(stage); io.check_snapshots(snapshots)
            if destination.exists() or destination.is_symlink():
                m.fail('EXTERNAL_FRAGMENTS_ARTIFACT_CONFLICT')
            os.rename(stage, destination); _fsync_directory(output)
            path = destination / 'fragments' / 'manifest.json'
            result = _summary(final, path, sha)
            verify_public_result(args, result)
            return result
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def import_scATAC_fragments(source_path, source_sha256, source_profile, reference_bundle_path,
        reference_bundle_sha256, namespace, output_dir, source_selection='unknown',
        source_index_path=None, source_index_sha256=None) -> ExternalFragmentsResult:
    """Adopt one explicitly declared external fragment source without biological transformation."""
    return execute_external_fragments(locals())
