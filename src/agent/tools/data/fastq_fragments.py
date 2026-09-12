"""Qualified FASTQ -> canonical fragments v2 production.

One selected processing library per Chromap invocation. The existing public tool wraps this data-layer API; no
cell calling, QC, or cCRE overlap. Output must be a fresh managed directory.
"""
from dataclasses import asdict
import fcntl
import hashlib
from pathlib import Path
import os
import shutil
import tempfile

from . import _chromap as c
from .chromap_reference_index import _fsync_directory
from ._fragments_binding import FragmentInputs, library_binding, preflight
from ._fragments_canonical import canonicalize
from ._fragments_common import (PACKAGING_POLICY,
    MAX_TOTAL, FragmentsRuntime, fail, run_stage, snapshots, unchanged, verify_packaging)
from ._fragments_fastq import scan_group
from . import fastq_fragment_manifest as m, scatac_fragments_v2 as v2
from .scatac_fragments_v2 import canonical_fragments_manifest_v2_bytes
from .fastq_fragments_verifier import verify_fragments


def prepare_fastq_fragments(*, inputs: FragmentInputs, output_dir, runtime: FragmentsRuntime):
    """Authoritative artifact paths/digests plus explicit managed runtime only.

    Trusted local filesystem; no overwrite/retry/resume of private stages.
    An existing output, including identical output, is an explicit conflict.
    Future orchestration can cancel between stages; active children are reaped.
    """
    if type(inputs) is not FragmentInputs or type(runtime) is not FragmentsRuntime:
        fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
    destination = Path(output_dir)
    if (not destination.is_absolute() or destination != destination.resolve()
            or destination.exists() or destination.is_symlink() or not destination.parent.is_dir()):
        fail('FRAGMENTS_ARTIFACT_CONFLICT')
    try:
        backend = c.identify_backend(runtime.chromap, expected_sha256=c.QUALIFIED_EXECUTABLE_SHA256)
    except c.ChromapError:
        raise
    verify_packaging(runtime)
    bound = preflight(inputs, backend)
    # The current v2 bound counts source entries across all libraries, including
    # each library's intake/context bindings. Reject before alignment, not after
    # producing an artifact that the common boundary cannot represent.
    source_count = sum(len({p for group in groups for _, p in group.files}) + 2
                       for _, groups in bound['libraries'])
    if source_count > v2.MAX_SOURCES:
        fail('FRAGMENTS_V2_CONTRACT_INVALID')
    executable_snapshots = snapshots([runtime.chromap, runtime.bgzip, runtime.tabix, runtime.sort])
    lock = destination.with_name('.' + destination.name + '.lock')
    with lock.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            fail('FRAGMENTS_ARTIFACT_CONFLICT')
        stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-attempt-', dir=destination.parent))
        try:
            entries = []; total = 0
            # Validate all libraries before any aligner execution.
            unchanged(bound['snapshots'], 'FRAGMENTS_SOURCE_CHANGED_BEFORE_EXECUTION')
            scans = {}
            for library, groups in bound['libraries']:
                for group in groups:
                    scans['group:' + group.group_id] = scan_group(group, library.whitelist.barcode_length)
            unchanged(bound['snapshots'])
            (stage / 'libraries').mkdir()
            for library, groups in bound['libraries']:
                directory = stage / 'libraries' / library.namespace; directory.mkdir()
                raw = directory / 'raw.chromap.bed'
                try:
                    argv = c.mapping_argv(runtime.chromap, fasta=bound['bundle'].genome.fasta.path,
                        index=Path(inputs.index_path).parent / bound['index'].index_file,
                        output=raw, groups=groups, whitelist=library.whitelist)
                except c.ChromapError:
                    fail('FRAGMENTS_INPUT_BINDING_UNREPRESENTABLE')
                unchanged(bound['snapshots']); unchanged(executable_snapshots)
                if c.identify_backend(runtime.chromap, expected_sha256=backend.executable_sha256) != backend:
                    fail('CHROMAP_VERSION_UNSUPPORTED')
                run_stage(argv, cwd=directory, code='CHROMAP_EXECUTION_FAILED')
                unchanged(bound['snapshots']); unchanged(executable_snapshots)
                if not raw.is_file() or raw.is_symlink():
                    fail('CHROMAP_OUTPUT_INCOMPLETE')
                if any(p.name not in ('raw.chromap.bed', 'process.log') for p in directory.iterdir()):
                    fail('CHROMAP_OUTPUT_INCOMPLETE')
                whitelist = set(Path(library.whitelist.resource.path).read_text('ascii').splitlines())
                canonical, summary = canonicalize(raw, directory=directory, contigs=bound['contigs'],
                    whitelist=whitelist, barcode_length=library.whitelist.barcode_length, runtime=runtime)
                verify_packaging(runtime)
                bgzf = directory / 'fragments.tsv.gz'
                with bgzf.open('xb') as output:
                    run_stage([runtime.bgzip, *PACKAGING_POLICY['bgzip'], canonical],
                        cwd=directory, code='FRAGMENTS_CANONICALIZATION_FAILED', output=output)
                run_stage([runtime.tabix, *PACKAGING_POLICY['tabix'], bgzf],
                    cwd=directory, code='FRAGMENTS_INDEX_MISMATCH')
                entry = {**library_binding(library, groups), **summary,
                    'scans': {'group:' + g.group_id: scans['group:' + g.group_id] for g in groups},
                    'sources': m.source_resources(groups)}
                for kind, path in (('bgzf', bgzf), ('tabix', Path(str(bgzf) + '.tbi'))):
                    if not path.is_file() or path.is_symlink():
                        fail('FRAGMENTS_INDEX_MISMATCH')
                    entry[kind] = {'path': str(path.relative_to(stage)), 'sha256': c.sha256(path),
                                   'size_bytes': path.stat().st_size}
                entries.append(entry); total += summary['sum_support']
                if total > MAX_TOTAL:
                    fail('FRAGMENTS_SUPPORT_INVALID')
                # Raw aligner output and sorting intermediates never enter publication.
                for path in directory.iterdir():
                    if path.name not in ('fragments.tsv.gz', 'fragments.tsv.gz.tbi'):
                        if not path.is_file() or path.is_symlink():
                            fail('CHROMAP_OUTPUT_INCOMPLETE')
                        path.unlink()
            record = dict(artifact_type=m.ARTIFACT_TYPE, schema_version=1, contract_version=m.CONTRACT_VERSION,
                profile_sha256=hashlib.sha256(m.PROFILE_BYTES).hexdigest(),
                inputs=asdict(inputs), lineage=bound['lineage'], backend=asdict(backend),
                mapping_policy={'flags': c.FIXED_FLAGS, 'implementation': c.IMPLEMENTATION_POLICY,
                                'upstream_tag': 'v0.3.2'},
                packaging_policy=PACKAGING_POLICY, semantics=m.SEMANTICS, libraries=entries)
            (stage / 'profile.json').write_bytes(m.PROFILE_BYTES)
            (stage / 'production.json').write_bytes(m.canonical_record_bytes(record))
            value = m.build_manifest(m.load_record(stage / 'production.json'), stage)
            payload = canonical_fragments_manifest_v2_bytes(value)
            manifest = stage / 'manifest.json'; manifest.write_bytes(payload)
            manifest_sha = hashlib.sha256(payload).hexdigest()
            verify_fragments(manifest, runtime=runtime, expected_sha256=manifest_sha)
            unchanged(bound['snapshots']); unchanged(executable_snapshots)
            c.identify_backend(runtime.chromap, expected_sha256=backend.executable_sha256)
            # V2 binds owned profile/record resources by absolute locator.
            payload = m.relocate_manifest(value, destination)
            manifest.write_bytes(payload); manifest_sha = hashlib.sha256(payload).hexdigest()
            for path in stage.rglob('*'):
                if path.is_file():
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            for directory in sorted(stage.rglob('*'), key=lambda p: len(p.parts), reverse=True):
                if directory.is_dir():
                    _fsync_directory(directory)
            _fsync_directory(stage)
            unchanged(bound['snapshots'])
            if destination.exists() or destination.is_symlink():
                fail('FRAGMENTS_ARTIFACT_CONFLICT')
            os.rename(stage, destination)
            _fsync_directory(destination.parent)
            from .authority_context import publication_moved
            publication_moved(stage, destination)
            return dict(status='success', manifest_path=str(destination / 'manifest.json'),
                manifest_sha256=manifest_sha, artifact_type=v2.ARTIFACT_TYPE, artifact_schema_version=2,
                contract_version=v2.CONTRACT_VERSION, species=bound['bundle'].species,
                assembly=bound['bundle'].target_assembly, n_libraries=len(entries),
                n_fragment_records=sum(e['n_fragment_records'] for e in entries), total_support=total)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
