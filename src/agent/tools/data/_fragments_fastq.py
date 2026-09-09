"""Complete lockstep FASTQ validation, distinct from M10's bounded inspection.

Retains at most one record per role. Digests cover decoded record bytes and
concatenated 32-byte M10 normalized-ID hashes, never encoded physical files.
"""
from contextlib import ExitStack
import gzip
import hashlib
import re
import zlib

from . import _raw_fastq as f
from ._fragments_common import MAX_TOTAL, fail


def _record(stream):
    lines = []
    for i in range(4):
        line = stream.readline(f.MAX_LINE_BYTES + 1)
        if not line and i == 0:
            return None
        if not line.endswith(b'\n') or len(line) >= f.MAX_LINE_BYTES:
            fail('FRAGMENTS_FASTQ_MALFORMED')
        lines.append(line)
    header, sequence, plus, quality = map(f._content, lines)
    if (not header.startswith(b'@') or not header[1:] or not 33 <= header[1] <= 126
            or header[1:].split(maxsplit=1)[0] in (b'/1', b'/2')
            or not plus.startswith(b'+') or not sequence
            or not f._printable(sequence) or not f._printable(quality)
            or not f._printable(header, header=True) or not f._printable(plus, header=True)
            or len(sequence) != len(quality) or (plus[1:] and plus[1:] != header[1:])):
        fail('FRAGMENTS_FASTQ_MALFORMED')
    # Exact normalized tokens are compared; M10's hash helper defines digesting.
    token = header[1:].split(maxsplit=1)[0]
    if token.endswith((b'/1', b'/2')):
        token = token[:-2]
    return token, f._header_id(header), sequence, b''.join(lines)


def scan_group(group, barcode_length):
    roles = f.LAYOUT_ROLES[group.layout]
    files = dict(group.files)
    ordered = [next(role for role, meaning in roles.items() if meaning == target)
               for target in (f.ReadMeaning.GENOMIC_1, f.ReadMeaning.GENOMIC_2, f.ReadMeaning.BARCODE)]
    contents = [hashlib.sha256() for _ in ordered]
    names = hashlib.sha256(); count = 0
    try:
        with ExitStack() as stack:
            streams = []
            for role in ordered:
                raw = stack.enter_context(open(files[role], 'rb'))
                compressed = raw.peek(2)[:2] == b'\x1f\x8b'
                streams.append(stack.enter_context(gzip.GzipFile(fileobj=raw)) if compressed else raw)
            while True:
                rows = [_record(stream) for stream in streams]
                if all(row is None for row in rows):
                    break
                if any(row is None for row in rows):
                    fail('FRAGMENTS_RECORD_COUNT_MISMATCH')
                if len({row[0] for row in rows}) != 1:
                    fail('FRAGMENTS_READ_NAME_MISMATCH')
                barcode = rows[2][2]
                if len(barcode) != barcode_length or re.fullmatch(b'[ACGTN]+', barcode) is None:
                    fail('FRAGMENTS_BARCODE_LENGTH_UNSUPPORTED' if len(barcode) != barcode_length
                         else 'FRAGMENTS_FASTQ_MALFORMED')
                count += 1
                if count > MAX_TOTAL:
                    fail('FRAGMENTS_SUPPORT_INVALID')
                names.update(rows[0][1])
                for digest, row in zip(contents, rows):
                    digest.update(row[3])
    except (OSError, EOFError, zlib.error):
        fail('FRAGMENTS_FASTQ_MALFORMED')
    if not count:
        fail('FRAGMENTS_FASTQ_MALFORMED')
    return {'record_count': count, 'normalized_id_sha256': names.hexdigest(),
            'decoded_role_sha256': [d.hexdigest() for d in contents],
            'identity_policy': 'decoded-record-bytes;sha256-concatenated-M10-id-hashes.v1'}
