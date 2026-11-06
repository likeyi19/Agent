"""Frozen Chromap identity readers for accepted historical publications."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re


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
