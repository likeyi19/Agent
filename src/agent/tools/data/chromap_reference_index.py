"""Frozen Chromap index metadata and shared FASTA/FAI resource qualification.

No index builder or runtime backend gate remains here. Existing accepted FASTQ
closures retain their exact index identities. Shared genome/reference owners
still use the ordered FASTA/FAI qualifier independently of raw native execution.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from . import _chromap as c
from ._ordered_identity import ordered_identity_sha256

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
