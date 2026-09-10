"""One public fragments capability; all backend configuration is execution-only.

The outer publication envelope closes the verified-publication/checkpoint crash
window without changing the accepted data-layer artifact or conflict semantics.
Only the registry's durable hook can recover a run/plan/step-bound receipt.
"""
from pathlib import Path
from typing import TypedDict

RECOVERY_POLICY = 'prepare-scatac-fragments-fastq-v2'


class ScATACFragmentsResult(TypedDict):
    status: str
    manifest_path: str
    manifest_sha256: str
    artifact_type: str
    artifact_schema_version: int
    contract_version: str
    species: str
    assembly: str
    n_libraries: int
    n_fragment_records: int
    total_support: int


def verification_runtime():
    # Completed artifact verification does not require Chromap or index lookup.
    from ._fragments_common import FragmentsRuntime
    return FragmentsRuntime('/not-used-for-verification')


def resolve_execution(reference_bundle_path, reference_bundle_sha256):
    import os
    from . import _chromap as c, chromap_reference_index as ci
    from .scatac_reference import load_scatac_reference_bundle
    from ._fragments_common import FragmentsRuntime, fail
    executable = os.environ.get('AGENT_CHROMAP_BIN')
    if not executable:
        fail('CHROMAP_UNAVAILABLE')
    backend = c.identify_backend(executable, expected_sha256=c.QUALIFIED_EXECUTABLE_SHA256)
    configured = os.environ.get('AGENT_CHROMAP_INDEX_ROOT')
    if not configured or not Path(configured).is_absolute() or not Path(configured).is_dir():
        fail('CHROMAP_INDEX_UNAVAILABLE')
    try:
        _, reference, _ = load_scatac_reference_bundle(reference_bundle_path, expected_sha256=reference_bundle_sha256)
    except (ValueError, OSError):
        fail('FRAGMENTS_REFERENCE_MISMATCH')
    matches = []
    # One bounded directory level, never recursive discovery or first-match reuse.
    for count, directory in enumerate(Path(configured).iterdir(), 1):
        if count > 4096 or directory.is_symlink():
            fail('CHROMAP_INDEX_INVALID')
        manifest = directory / 'manifest.json'
        if not directory.is_dir() or not manifest.is_file():
            fail('CHROMAP_INDEX_INVALID')
        try:
            value = ci.load_chromap_reference_index(manifest)
        except (ValueError, OSError):
            fail('CHROMAP_INDEX_INVALID')
        if (value.reference_identity_sha256 == reference.reference_identity_sha256
                and value.backend == backend and value.build_policy == c.INDEX_POLICY):
            matches.append(manifest)
    if len(matches) != 1:
        fail('CHROMAP_INDEX_AMBIGUOUS' if matches else 'CHROMAP_INDEX_UNAVAILABLE')
    manifest = matches[0]
    return FragmentsRuntime(executable), str(manifest.resolve()), c.sha256(manifest)


def _arguments(arguments):
    return {key: str(value) for key, value in arguments.items()}


def _publication(arguments, execution_identity):
    from ._fragments_common import digest
    args = _arguments(arguments)
    identity = digest({'arguments': args, 'policy': RECOVERY_POLICY,
                       'durable_execution_identity': execution_identity})
    return Path(args['output_dir']) / ('scatac-fragments-' + identity), identity, digest(args)


def _summary(value, path, sha):
    return ScATACFragmentsResult(status='success', manifest_path=str(path), manifest_sha256=sha,
        artifact_type=value['artifact_type'], artifact_schema_version=value['schema_version'],
        contract_version=value['contract_version'], species=value['reference']['species'],
        assembly=value['reference']['assembly'], n_libraries=len(value['libraries']),
        n_fragment_records=sum(e['n_fragment_records'] for e in value['libraries']),
        total_support=sum(e['sum_support'] for e in value['libraries']))


def _read_receipt(destination, arguments, expected_identity=None):
    import json
    from ._fragments_common import digest, fail, snapshot
    from .scatac_fragments_v2 import _pairs
    receipt = destination / 'receipt.json'
    try:
        snapshot(receipt)
        with receipt.open('rb') as stream:
            payload = stream.read(16385)
        if len(payload) > 16384:
            fail('FRAGMENTS_RECOVERY_MISMATCH')
        value = json.loads(payload, object_pairs_hook=_pairs)
        if (set(value) != {'artifact_type', 'schema_version', 'policy', 'execution_identity',
                          'arguments_sha256', 'manifest_sha256'}
                or value['artifact_type'] != 'agent.fragments-execution-receipt'
                or type(value['schema_version']) is not int or value['schema_version'] != 1
                or value['policy'] != RECOVERY_POLICY
                or value['arguments_sha256'] != digest(_arguments(arguments))
                or destination.name != 'scatac-fragments-' + value['execution_identity']
                or (expected_identity is not None and value['execution_identity'] != expected_identity)):
            fail('FRAGMENTS_RECOVERY_MISMATCH')
        from ._chromap import digest_text
        digest_text(value['execution_identity']); digest_text(value['manifest_sha256'])
        return value
    except (ValueError, OSError, TypeError, KeyError):
        fail('FRAGMENTS_RECOVERY_MISMATCH')


def verify_public_result(arguments, result):
    from ._fragments_common import fail, snapshots, unchanged
    from .fastq_fragments_verifier import verify_fragments
    path = Path(result['manifest_path'])
    destination = path.parent.parent
    output = Path(arguments['output_dir']).resolve()
    if (path != path.resolve() or path.name != 'manifest.json' or path.parent.name != 'fragments'
            or destination.parent != output):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    before = snapshots([destination / 'receipt.json', path])
    receipt = _read_receipt(destination, arguments)
    if receipt['manifest_sha256'] != result['manifest_sha256']:
        fail('FRAGMENTS_RECOVERY_MISMATCH')
    value = verify_fragments(path, runtime=verification_runtime(), expected_sha256=result['manifest_sha256'])
    names = {'intake_manifest_path': 'intake_path', 'intake_manifest_sha256': 'intake_sha256',
        'library_context_path': 'context_path', 'library_context_sha256': 'context_sha256',
        'reference_bundle_path': 'reference_path', 'reference_bundle_sha256': 'reference_sha256'}
    from .fastq_fragment_manifest import record_for_manifest
    record = record_for_manifest(value, path.parent)
    if any(record['inputs'][field] != str(arguments[arg]) for arg, field in names.items()):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    if dict(result) != _summary(value, path, result['manifest_sha256']):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    unchanged(before)
    return value


def recover_fragments(arguments, execution_identity):
    """Exact durable receipt lookup only; no aligner, index-root lookup, or rerun."""
    from ._fragments_common import fail
    from .fastq_fragment_manifest import load_manifest
    destination, token, _ = _publication(arguments, execution_identity)
    if not destination.exists() or destination.is_symlink():
        fail('FRAGMENTS_RECOVERY_UNAVAILABLE')
    receipt = _read_receipt(destination, arguments, token)
    path = destination / 'fragments' / 'manifest.json'
    value = load_manifest(path, expected_sha256=receipt['manifest_sha256'])
    result = _summary(value, path, receipt['manifest_sha256'])
    verify_public_result(arguments, result)
    return result


def execute_fragments(arguments, execution_identity=None):
    import fcntl
    import os
    import shutil
    import tempfile
    from ._fragments_binding import FragmentInputs
    from ._fragments_common import canonical, fail
    from .chromap_reference_index import _fsync_directory
    from .fastq_fragments import prepare_fastq_fragments
    args = _arguments(arguments)
    output = Path(args['output_dir'])
    if not output.is_absolute() or output != output.resolve():
        fail('FRAGMENTS_ARTIFACT_CONFLICT')
    destination, token, args_sha = _publication(args, execution_identity)
    if destination.exists() or destination.is_symlink():
        fail('FRAGMENTS_ARTIFACT_CONFLICT')
    runtime, index_path, index_sha = resolve_execution(args['reference_bundle_path'], args['reference_bundle_sha256'])
    inputs = FragmentInputs(args['intake_manifest_path'], args['intake_manifest_sha256'],
        args['library_context_path'], args['library_context_sha256'], args['reference_bundle_path'],
        args['reference_bundle_sha256'], index_path, index_sha)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ('.' + destination.name + '.lock')).open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            fail('FRAGMENTS_ARTIFACT_CONFLICT')
        stage = Path(tempfile.mkdtemp(prefix='.fragments-envelope-', dir=output))
        try:
            result = prepare_fastq_fragments(inputs=inputs, runtime=runtime, output_dir=stage / 'fragments')
            from .fastq_fragments_verifier import verify_fragments
            from .fastq_fragment_manifest import load_manifest, relocate_manifest
            from ._chromap import sha256
            verify_fragments(result['manifest_path'], runtime=verification_runtime(),
                             expected_sha256=result['manifest_sha256'])
            manifest = Path(result['manifest_path'])
            value = load_manifest(manifest, expected_sha256=result['manifest_sha256'])
            manifest.write_bytes(relocate_manifest(value, destination / 'fragments'))
            result['manifest_sha256'] = sha256(manifest)
            with manifest.open('rb') as stream:
                os.fsync(stream.fileno())
            receipt = dict(artifact_type='agent.fragments-execution-receipt', schema_version=1,
                policy=RECOVERY_POLICY, execution_identity=token, arguments_sha256=args_sha,
                manifest_sha256=result['manifest_sha256'])
            (stage / 'receipt.json').write_bytes(canonical(receipt))
            with (stage / 'receipt.json').open('rb') as stream:
                os.fsync(stream.fileno())
            _fsync_directory(stage / 'fragments')
            _fsync_directory(stage)
            if destination.exists() or destination.is_symlink():
                fail('FRAGMENTS_ARTIFACT_CONFLICT')
            os.rename(stage, destination)
            _fsync_directory(output)
            result['manifest_path'] = str(destination / 'fragments' / 'manifest.json')
            verify_public_result(args, result)
            return result
        finally:
            if stage.exists():
                shutil.rmtree(stage)


def prepare_scATAC_fragments(intake_manifest_path, intake_manifest_sha256,
        library_context_path, library_context_sha256, reference_bundle_path,
        reference_bundle_sha256, output_dir) -> ScATACFragmentsResult:
    """Prepare compatible verified FASTQ inputs into canonical scATAC fragments."""
    return execute_fragments(locals())
