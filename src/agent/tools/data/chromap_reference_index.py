"""M11.2b derived Chromap index resource; no fragment or Planner integration.

Native index bytes are opaque and content identified, never inferred from the
build key. Binding includes the bundle identity but Chromap reads only FASTA.
The stricter ordered FASTA/FAI gate belongs here, not in ReferenceBundle.v1.
Publication assumes a trusted local filesystem. A persistent flock file guards
all cooperating builders; staging is a sibling directory on the same filesystem.
"""
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from . import _chromap as c
from ._ordered_identity import ordered_identity_sha256
from .scatac_reference import load_scatac_reference_bundle

ARTIFACT_TYPE = 'agent.chromap-reference-index'
CONTRACT_VERSION = 'chromap-reference-index.v1'
MAX_MANIFEST_BYTES = 65536

@dataclass(frozen=True)
class ChromapReferenceIndex:
    reference_identity_sha256: str
    reference_manifest_sha256: str
    fasta_sha256: str
    fai_sha256: str
    ordered_contig_sha256: str
    contig_count: int
    backend: c.BackendIdentity
    index_sha256: str
    index_size_bytes: int
    index_identity_sha256: str = ''
    index_file: str = 'reference.chromap'
    kmer: int = 17
    window: int = 7
    build_policy: str = c.INDEX_POLICY
    artifact_type: str = ARTIFACT_TYPE
    schema_version: int = 1
    contract_version: str = CONTRACT_VERSION


def _identity(value):
    d = asdict(value); d.pop('index_identity_sha256')
    return hashlib.sha256(b'agent.chromap-reference-index.v1\0' + c.json_bytes(d)).hexdigest()


def validate_chromap_reference_index(value):
    if type(value) is ChromapReferenceIndex:
        value = asdict(value)
    if type(value) is not dict or set(value) != set(ChromapReferenceIndex.__dataclass_fields__):
        c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Invalid closed index contract.')
    try:
        d = dict(value); d['backend'] = c.validate_backend(d['backend'])
        result = ChromapReferenceIndex(**d)
        for name in ('reference_identity_sha256', 'reference_manifest_sha256', 'fasta_sha256',
            'fai_sha256', 'ordered_contig_sha256', 'index_sha256', 'index_identity_sha256'):
            c.digest_text(getattr(result, name))
        for name in ('contig_count', 'index_size_bytes'):
            n = getattr(result, name)
            if type(n) is not int or not 0 < n <= 2**63 - 1:
                raise ValueError()
        if (type(result.kmer) is not int or result.kmer != 17 or type(result.window) is not int
            or result.window != 7 or type(result.schema_version) is not int or result.schema_version != 1
            or result.artifact_type != ARTIFACT_TYPE or result.contract_version != CONTRACT_VERSION
            or result.build_policy != c.INDEX_POLICY or result.index_file != 'reference.chromap'
            or result.index_identity_sha256 != _identity(result)):
            raise ValueError()
        return result
    except (ValueError, TypeError):
        c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Invalid index identity or fixed policy.')


def canonical_chromap_reference_index_bytes(value):
    return c.json_bytes(asdict(validate_chromap_reference_index(value)))


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Duplicate JSON key.')
        result[key] = value
    return result


def load_chromap_reference_index(path, *, expected_sha256=None):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(payload) > MAX_MANIFEST_BYTES:
        c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Oversized index manifest.')
    if expected_sha256 is not None:
        c.digest_text(expected_sha256)
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            c.fail('CHROMAP_INDEX_IDENTITY_MISMATCH', 'Index manifest hash differs.')
    try:
        return validate_chromap_reference_index(json.loads(payload, object_pairs_hook=_pairs,
            parse_constant=lambda _: c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Nonfinite JSON.')))
    except (ValueError, UnicodeError, RecursionError):
        c.fail('CHROMAP_INDEX_CONTRACT_INVALID', 'Invalid index JSON.')


def verify_fasta_fai(bundle):
    """Stream exact ordered names/lengths and check FAI offsets/line geometry.

    Bounded line memory (1 MiB), dictionary memory proportional to contigs.
    Sequence case is retained; supported FASTA alphabet is ACGTN/acgtn.
    Header descriptions are allowed; the first token is the exact sequence name.
    """
    fasta = Path(bundle.genome.fasta.path); fai = Path(bundle.genome.fai.path)
    before = (c.snapshot(fasta), c.snapshot(fai))
    records = []; names = set(); digest = hashlib.sha256(); current = None
    with fasta.open('rb') as stream:
        while True:
            offset = stream.tell(); line = stream.readline(1024 * 1024 + 1)
            if not line:
                break
            digest.update(line)
            if len(line) > 1024 * 1024 or b'\r' in line or b'\0' in line:
                c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Invalid FASTA line.')
            text = line.removesuffix(b'\n')
            if text.startswith(b'>'):
                if current is not None:
                    if current[1] == 0:
                        c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Empty FASTA record.')
                    records.append(tuple(current[:5]))
                try:
                    name = text[1:].split()[0].decode('utf-8')
                except (IndexError, UnicodeError):
                    c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Invalid FASTA header.')
                if name in names or ':' in name:
                    c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Duplicate/invalid FASTA name.')
                names.add(name); current = [name, 0, stream.tell(), 0, 0, False]
            else:
                if current is None or re.fullmatch(b'[ACGTNacgtn]+', text) is None or current[5]:
                    c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Invalid FASTA sequence/line geometry.')
                if current[3] == 0:
                    current[3:5] = [len(text), len(line)]
                elif len(text) > current[3] or len(line) > current[4]:
                    c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Inconsistent FASTA line widths.')
                current[5] = len(text) < current[3] or len(line) < current[4]
                current[1] += len(text)
    if current is not None and current[1] > 0:
        records.append(tuple(current[:5]))
    else:
        c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Missing/empty FASTA record.')
    rows = []; fai_digest = hashlib.sha256()
    with fai.open('rb') as stream:
        for line in iter(lambda: stream.readline(65537), b''):
            fai_digest.update(line)
            if len(line) > 65536 or b'\r' in line:
                c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Invalid FAI line.')
            try:
                row = line.removesuffix(b'\n').decode('utf-8').split('\t')
                if len(row) != 5 or any(re.fullmatch('[1-9][0-9]*', s) is None for s in row[1:]):
                    raise ValueError()
                rows.append((row[0], *(int(s) for s in row[1:])))
            except (ValueError, UnicodeError):
                c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'Invalid FAI record.')
    ordered = ordered_identity_sha256(f'{row[0]}\t{row[1]}' for row in records)
    if (records != rows or len(rows) != bundle.genome.contig_count
        or ordered != bundle.genome.ordered_contig_sha256):
        c.fail('CHROMAP_FASTA_FAI_MISMATCH', 'FASTA and FAI dictionaries/geometry disagree.')
    if (digest.hexdigest() != bundle.genome.fasta.sha256
        or fai_digest.hexdigest() != bundle.genome.fai.sha256):
        c.fail('CHROMAP_REFERENCE_CHANGED', 'Reference source hash differs.')
    if before != (c.snapshot(fasta), c.snapshot(fai)):
        c.fail('CHROMAP_SOURCE_CHANGED', 'Reference changed during validation.')
    return tuple((r[0], r[1]) for r in records)


def _binding(bundle, reference_sha, backend):
    return dict(reference_identity_sha256=bundle.reference_identity_sha256,
        reference_manifest_sha256=reference_sha, fasta_sha256=bundle.genome.fasta.sha256,
        fai_sha256=bundle.genome.fai.sha256, ordered_contig_sha256=bundle.genome.ordered_contig_sha256,
        contig_count=bundle.genome.contig_count, backend=backend)


def verify_chromap_reference_index(manifest_path, *, reference_manifest_path,
        expected_reference_sha256, backend, expected_manifest_sha256=None):
    value = load_chromap_reference_index(manifest_path, expected_sha256=expected_manifest_sha256)
    backend = c.validate_backend(backend)
    _, bundle, _ = load_scatac_reference_bundle(reference_manifest_path, expected_sha256=expected_reference_sha256)
    verify_fasta_fai(bundle)
    if any(getattr(value, k) != v for k, v in _binding(bundle, expected_reference_sha256, backend).items()):
        c.fail('CHROMAP_INDEX_BINDING_MISMATCH', 'Index/reference/backend binding differs.')
    p = Path(manifest_path).parent / value.index_file
    before = c.snapshot(p)
    if before[2] != value.index_size_bytes or c.sha256(p) != value.index_sha256:
        c.fail('CHROMAP_INDEX_IDENTITY_MISMATCH', 'Index content differs.')
    if c.snapshot(p) != before:
        c.fail('CHROMAP_SOURCE_CHANGED', 'Index changed during verification.')
    return value


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prepare_chromap_reference_index(*, reference_manifest_path, expected_reference_sha256,
        executable, backend, output_dir):
    """Explicit separate index preparation. Existing matching artifacts are reused.

    No overwrite parameter: a conflicting or incomplete destination fails closed.
    This is not called automatically by the mapping adapter. M11.2b callers must
    use tiny synthetic references; full-genome preparation is not qualified here.
    """
    backend = c.validate_backend(backend)
    if c.identify_backend(executable, expected_sha256=backend.executable_sha256) != backend:
        c.fail('CHROMAP_EXECUTABLE_MISMATCH', 'Executable/platform differs.')
    _, bundle, _ = load_scatac_reference_bundle(reference_manifest_path, expected_sha256=expected_reference_sha256)
    verify_fasta_fai(bundle)
    destination = Path(output_dir)
    if (not destination.is_absolute() or destination.is_symlink()
        or destination != destination.resolve() or not destination.parent.is_dir()):
        c.fail('CHROMAP_INDEX_OUTPUT_CONFLICT', 'Expected absolute sibling destination.')
    lock = destination.with_name('.' + destination.name + '.lock')
    with lock.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        if destination.exists():
            if not destination.is_dir() or not (destination / 'manifest.json').is_file():
                c.fail('CHROMAP_INDEX_OUTPUT_CONFLICT', 'Incomplete/conflicting index destination.')
            return verify_chromap_reference_index(destination / 'manifest.json',
                reference_manifest_path=reference_manifest_path,
                expected_reference_sha256=expected_reference_sha256, backend=backend)
        stage = Path(tempfile.mkdtemp(prefix='.' + destination.name + '-', dir=destination.parent))
        try:
            before = tuple(c.snapshot(p) for p in (reference_manifest_path,
                bundle.genome.fasta.path, bundle.genome.fai.path, executable))
            if c.identify_backend(executable, expected_sha256=backend.executable_sha256) != backend:
                c.fail('CHROMAP_EXECUTABLE_MISMATCH', 'Executable identity changed before build.')
            argv = c.index_argv(executable, bundle.genome.fasta.path, stage / 'reference.chromap')
            with (stage / 'build.log').open('wb') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                c.fail('CHROMAP_INDEX_BUILD_FAILED', 'Chromap index build failed.')
            if before != tuple(c.snapshot(p) for p in (reference_manifest_path,
                bundle.genome.fasta.path, bundle.genome.fai.path, executable)):
                c.fail('CHROMAP_SOURCE_CHANGED', 'Source changed during index build.')
            if c.identify_backend(executable, expected_sha256=backend.executable_sha256) != backend:
                c.fail('CHROMAP_EXECUTABLE_MISMATCH', 'Executable identity changed during build.')
            # Recheck exact source/bundle bytes before publishing.
            _, bundle, _ = load_scatac_reference_bundle(reference_manifest_path, expected_sha256=expected_reference_sha256)
            verify_fasta_fai(bundle)
            p = stage / 'reference.chromap'
            value = ChromapReferenceIndex(**_binding(bundle, expected_reference_sha256, backend),
                index_sha256=c.sha256(p), index_size_bytes=p.stat().st_size)
            value = replace(value, index_identity_sha256=_identity(value))
            (stage / 'manifest.json').write_bytes(canonical_chromap_reference_index_bytes(value))
            for p in stage.iterdir():
                with p.open('rb') as stream:
                    os.fsync(stream.fileno())
            _fsync_directory(stage)
            if destination.exists():
                c.fail('CHROMAP_INDEX_OUTPUT_CONFLICT', 'Destination appeared during build.')
            os.rename(stage, destination)
            _fsync_directory(destination.parent)
            return verify_chromap_reference_index(destination / 'manifest.json',
                reference_manifest_path=reference_manifest_path,
                expected_reference_sha256=expected_reference_sha256, backend=backend)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
