"""Private M11.2b qualification adapter. Never a registered execution tool.

Group arguments must be supplied from reviewed M10 facts by a trusted caller.
This module does not establish full-stream synchronization or production readiness.
Only the fixed paired-end barcode BED policy is qualified. Optional upstream
summaries are disabled: their signed-int counters and non-whitelist accounting
are outside the exact fragment-support contract.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from ._raw_fastq import FastqLayout, LAYOUT_ROLES, ReadMeaning, ReadRole
from .scatac_library_context import BarcodeWhitelistIdentity, inspect_barcode_whitelist

UPSTREAM_COMMIT = '5bd17e1f1c50805e76904efd021603cb6a1b6e23'
UPSTREAM_VERSION = '0.3.2-r518'
SOURCE_TAR_SHA256 = '0234cc870ef10367cde5b1510555ba4c06498461b69a3960b9697507ab4aa0b7'
PATCH_SHA256 = '8d862e77d59213aa19b3fbc5f6014585a4b5f414f3e29143b68513ffd961df33'
QUALIFIED_EXECUTABLE_SHA256 = '7ba402d0d69d08fedd525a4de83a4dd749670e88d4de507c236b07f03a285649'
BUILD_RECORD_SHA256 = '6bb93fb766aee7e752a71cbe36c305f7c8278b24699c6945fb6491d37ffa974a'
BACKEND_POLICY = 'chromap-atac-agent-support-v1'
INDEX_POLICY = 'chromap-index-k17-w7.v1'
SUPPORT_SEMANTICS = ('Number of Chromap-generated paired mappings in the accepted corrected-barcode '
    'and contig/start/length duplicate group, including the representative; '
    'representative-MAPQ filtering, not individual-mapping-MAPQ filtering.')
FIXED_FLAGS = ('--preset', 'atac', '--trim-adapters', '--max-insert-size', '2000',
    '--min-read-length', '30', '--error-threshold', '8', '--min-num-seeds', '2',
    '--max-seed-frequencies', '500,1000', '--max-num-best-mappings', '1',
    '--drop-repetitive-reads', '500000', '--MAPQ-threshold', '30',
    '--remove-pcr-duplicates', '--remove-pcr-duplicates-at-cell-level', '--Tn5-shift',
    '--BED', '--low-mem', '--bc-error-threshold', '1', '--bc-probability-threshold', '0.9',
    '--num-threads', '1', '--cache-size', '4000003', '--cache-update-param', '0.01')
IMPLEMENTATION_POLICY = (
    ('barcode_orientation', 'whole-forward'), ('output_nonwhitelist', False),
    ('barcode_translation', False), ('split_alignment', False), ('multi_allocation', False),
    ('summary', False), ('initial_exact_barcode_sample', 20000000), ('batch_records', 500000),
    ('mapping_random_seed', 11), ('minhash_k', 250), ('support_bits', 64),
    ('support_overflow', 'fail'), ('threads', 1))

class ChromapError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)

def fail(code, message):
    raise ChromapError(code, message)

def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def digest_text(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        fail('CHROMAP_IDENTITY_INVALID', 'Invalid SHA-256 identity.')
    return value

def snapshot(path):
    s = Path(path).stat()
    if not Path(path).is_file():
        fail('CHROMAP_SOURCE_INVALID', 'Expected regular file.')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def policy_sha256():
    return hashlib.sha256(json_bytes((FIXED_FLAGS, IMPLEMENTATION_POLICY))).hexdigest()

@dataclass(frozen=True)
class BackendIdentity:
    executable_sha256: str
    machine: str
    system: str
    byteorder: str
    pointer_bits: int
    policy_sha256: str
    backend_policy: str = BACKEND_POLICY
    upstream_commit: str = UPSTREAM_COMMIT
    upstream_version: str = UPSTREAM_VERSION
    source_tar_sha256: str = SOURCE_TAR_SHA256
    patch_sha256: str = PATCH_SHA256
    build_record_sha256: str = BUILD_RECORD_SHA256


def validate_backend(value):
    if isinstance(value, BackendIdentity):
        value = asdict(value)
    if type(value) is not dict or set(value) != set(BackendIdentity.__dataclass_fields__):
        fail('CHROMAP_IDENTITY_INVALID', 'Invalid closed backend identity.')
    try:
        result = BackendIdentity(**value)
        expected = BackendIdentity(result.executable_sha256, result.machine, result.system,
            result.byteorder, result.pointer_bits, policy_sha256())
        if result != expected or type(result.pointer_bits) is not int or result.pointer_bits != 64:
            raise ValueError()
        if result.system != 'Linux' or result.machine != 'x86_64' or result.byteorder != 'little':
            raise ValueError()
        digest_text(result.executable_sha256)
        if result.executable_sha256 != QUALIFIED_EXECUTABLE_SHA256:
            raise ValueError()
        return result
    except (TypeError, ValueError):
        fail('CHROMAP_IDENTITY_INVALID', 'Unsupported backend identity/policy/platform.')


QUALIFIED_LIBRARIES = (('libc.so.6', 'b2cf6c33b74d2f22543b7a469a75b538911e690f769d0b238843a49465b83793'), ('libgcc_s.so.1', 'fc9d43b2f6c20e53b009238f767c5b949d202389e20de9e202ea684b4ba3729a'), ('libgomp.so.1', 'd46f9225c1883039e8a6853e6d96ca1af11d034ce186a090952e4a7c8a7c2fdc'), ('libm.so.6', '3dd5511ae94785c9f921429b0f2b2f7aabb461b6f0e6de6dfdbef15f24bdfee6'), ('libstdc++.so.6', 'ff0825e113603c3866680d5d52216bc6d8eedf3a59f52a0aef67ff01994db128'), ('libz.so.1', '64c206f0146cc58bbddc4f22054436f4ff278f5a554aa3ce6921ddf7e9133370'))

def verify_linked_runtime(executable):
    # The accepted build is a dynamically linked Linux executable. Reuse must
    # not silently accept a changed zlib/OpenMP/C++/C runtime with the same ELF.
    r = subprocess.run(['/usr/bin/ldd', str(executable)], capture_output=True, timeout=15)
    if r.returncode:
        fail('CHROMAP_RUNTIME_MISMATCH', 'Cannot verify linked runtime.')
    found = []
    for line in r.stdout.decode().splitlines():
        if '=>' not in line:
            continue
        parts = line.split()
        if len(parts) != 4 or parts[1] != '=>' or not Path(parts[2]).is_file():
            fail('CHROMAP_RUNTIME_MISMATCH', 'Unresolved runtime library.')
        before = snapshot(parts[2])
        found.append((parts[0], sha256(parts[2])))
        if snapshot(parts[2]) != before:
            fail('CHROMAP_SOURCE_CHANGED', 'Runtime library changed during inspection.')
    if tuple(sorted(found)) != QUALIFIED_LIBRARIES:
        fail('CHROMAP_RUNTIME_MISMATCH', 'Linked runtime differs from qualification build.')


def identify_backend(executable, *, expected_sha256):
    """Expected hash is trusted qualification configuration, not an LLM input.

    The version string alone cannot attest that the support patch was applied.
    Build attestation and guarded tests establish the expected executable hash.
    """
    digest_text(expected_sha256)
    if expected_sha256 != QUALIFIED_EXECUTABLE_SHA256:
        fail('CHROMAP_EXECUTABLE_UNQUALIFIED', 'Only the reviewed qualification build is accepted.')
    p = Path(executable)
    if not p.is_absolute() or not p.is_file() or not os.access(p, os.X_OK):
        fail('CHROMAP_UNAVAILABLE', 'Explicit executable is unavailable.')
    before = snapshot(p)
    if sha256(p) != expected_sha256:
        fail('CHROMAP_EXECUTABLE_MISMATCH', 'Executable hash differs.')
    verify_linked_runtime(p)
    r = subprocess.run([str(p), '--version'], capture_output=True, timeout=15, check=False)
    if r.returncode or (r.stdout + r.stderr).decode().strip() != UPSTREAM_VERSION:
        fail('CHROMAP_VERSION_UNSUPPORTED', 'Unexpected executable version.')
    if snapshot(p) != before:
        fail('CHROMAP_SOURCE_CHANGED', 'Executable changed during inspection.')
    import struct
    return validate_backend(BackendIdentity(expected_sha256, platform.machine(), platform.system(),
        sys.byteorder, struct.calcsize('P') * 8, policy_sha256()))


def verify_patch_source(source_tar, patch):
    """Exact reviewed git-archive bytes and patch; never fuzzy version acceptance."""
    if sha256(source_tar) != SOURCE_TAR_SHA256 or sha256(patch) != PATCH_SHA256:
        fail('CHROMAP_PATCH_SOURCE_MISMATCH', 'Patch/source identity differs from reviewed pin.')


def input_path(path):
    p = Path(path)
    if (not p.is_absolute() or not p.is_file() or any(c in str(p) for c in ',*?[]\n\r\0')):
        fail('CHROMAP_INPUT_INVALID', 'Input cannot be represented as an exact Chromap path.')
    return str(p)


def index_argv(executable, fasta, output):
    return (str(executable), '--build-index', '--ref', input_path(fasta), '--output', str(output),
            '--kmer', '17', '--window', '7')

@dataclass(frozen=True)
class QualificationGroup:
    group_id: str
    layout: FastqLayout
    files: tuple[tuple[ReadRole, str], ...]


def mapping_argv(executable, *, fasta, index, output, groups, whitelist):
    """One explicitly selected processing library; no lane/library inference.

    No execution, arbitrary flags, scientific defaults, or full-stream pre-scan.
    """
    if not isinstance(whitelist, BarcodeWhitelistIdentity):
        fail('CHROMAP_WHITELIST_REQUIRED', 'An identity-verified whitelist is required.')
    current = inspect_barcode_whitelist(whitelist.resource.path)
    if (current.resource.sha256, current.barcode_set_sha256, current.barcode_length,
        current.n_barcodes) != (whitelist.resource.sha256, whitelist.barcode_set_sha256,
        whitelist.barcode_length, whitelist.n_barcodes):
        fail('CHROMAP_WHITELIST_MISMATCH', 'Whitelist identity differs.')
    if not 1 <= current.barcode_length <= 32:
        fail('CHROMAP_BARCODE_LENGTH_UNSUPPORTED', 'Chromap permits 1..32 barcode bases.')
    if not groups or len(groups) > 4096 or any(type(g) is not QualificationGroup for g in groups):
        fail('CHROMAP_GROUP_INVALID', 'Expected selected qualification groups.')
    if len({g.group_id for g in groups}) != len(groups):
        fail('CHROMAP_GROUP_INVALID', 'Repeated group identity.')
    roles = {meaning: [] for meaning in (ReadMeaning.GENOMIC_1, ReadMeaning.GENOMIC_2, ReadMeaning.BARCODE)}
    used = set()
    for group in sorted(groups, key=lambda g: g.group_id):
        digest_text(group.group_id)
        if type(group.layout) is not FastqLayout or group.layout not in LAYOUT_ROLES:
            fail('CHROMAP_LAYOUT_UNSUPPORTED', 'Unsupported layout.')
        files = dict(group.files)
        expected = LAYOUT_ROLES[group.layout]
        if (len(files) != len(group.files) or any(type(role) is not ReadRole for role in files)
            or not set(expected) - {ReadRole.I1} <= set(files) <= set(expected)):
            fail('CHROMAP_ROLE_INVALID', 'Missing, repeated, or foreign role.')
        for role, meaning in expected.items():
            if meaning is ReadMeaning.SAMPLE_INDEX:
                continue
            path = input_path(files[role])
            if path in used:
                fail('CHROMAP_ROLE_INVALID', 'A role file is reused.')
            used.add(path); roles[meaning].append(path)
    lists = [','.join(roles[m]) for m in roles]
    if any(len(s.encode()) > 120000 for s in lists):
        fail('CHROMAP_INPUT_INVALID', 'Role list exceeds qualified argv size.')
    protected = used | {str(Path(fasta)), str(Path(index)), whitelist.resource.path}
    out = Path(output)
    if out.exists() or out.is_symlink() or str(out) in protected or not out.is_absolute():
        fail('CHROMAP_OUTPUT_CONFLICT', 'Qualification output must be a fresh absolute path.')
    return (str(executable), *FIXED_FLAGS, '--ref', input_path(fasta), '--index', input_path(index),
        '-1', lists[0], '-2', lists[1], '-b', lists[2], '--barcode-whitelist',
        input_path(whitelist.resource.path), '--output', str(out))


def prepare_support_source(source_tar, patch, output_dir):
    """Explicit offline qualification provisioning; no compiler/network invocation.

    Extract only the exact reviewed archive, then apply only the exact patch to
    that fresh tree. No patching of arbitrary existing checkouts is supported.
    """
    import io
    import shutil
    import tarfile
    import tempfile
    destination = Path(output_dir)
    if not destination.is_absolute() or destination.exists() or destination.is_symlink():
        fail('CHROMAP_OUTPUT_CONFLICT', 'Patched source requires a fresh private directory.')
    # Bound small source artifacts and consume the same bytes that were hashed.
    with Path(source_tar).open('rb') as stream:
        source = stream.read(16 * 1024 * 1024 + 1)
    with Path(patch).open('rb') as stream:
        correction = stream.read(64 * 1024 + 1)
    if (hashlib.sha256(source).hexdigest() != SOURCE_TAR_SHA256
        or hashlib.sha256(correction).hexdigest() != PATCH_SHA256):
        fail('CHROMAP_PATCH_SOURCE_MISMATCH', 'Only the reviewed archive and patch may be applied.')
    stage = Path(tempfile.mkdtemp(prefix='.chromap-source-', dir=destination.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(source), mode='r:') as archive:
            archive.extractall(stage, filter='data')
        r = subprocess.run(['/usr/bin/patch', '--batch', '--fuzz=0', '-p1'],
            input=correction, cwd=stage, capture_output=True, check=False)
        if r.returncode:
            fail('CHROMAP_PATCH_FAILED', 'Reviewed patch did not apply exactly.')
        if destination.exists():
            fail('CHROMAP_OUTPUT_CONFLICT', 'Patched source destination appeared.')
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
