"""Bounded descriptive fragment IO; native ingestion consumes the original file."""
import re
from ._fragments_common import MAX_SUPPORT
import gzip
from pathlib import Path
import struct
import zlib
from contextlib import ExitStack
import io as buffered
from agent.tools._cancellation import cancellation_checkpoint

from . import external_fragment_manifest as m
from .scatac_fragments_v2_verifier import file_sha256, take_snapshots, check_snapshots


def resource(path, expected_sha256=None):
    path = Path(path)
    before = take_snapshots([path])
    value = dict(path=str(path), sha256=file_sha256(path), size_bytes=path.stat().st_size)
    check_snapshots(before)
    if expected_sha256 is not None and value['sha256'] != expected_sha256:
        m.fail('EXTERNAL_FRAGMENTS_SOURCE_MISMATCH')
    if not value['size_bytes']:
        m.fail('EXTERNAL_FRAGMENTS_EMPTY')
    return value


def encoding(path):
    """Inspect gzip extra fields, never the filename or producer metadata."""
    with Path(path).open('rb') as source:
        head = source.read(12)
        if head[:4] == b'\x28\xb5\x2f\xfd':
            return 'zstandard'
        if head[:2] != b'\x1f\x8b':
            return 'plain'
        if len(head) < 12 or head[2] != 8:
            m.fail('EXTERNAL_FRAGMENTS_ENCODING_INVALID')
        if head[3] & 4:
            extra = source.read(struct.unpack('<H', head[10:12])[0])
            offset = 0
            while offset + 4 <= len(extra):
                tag, size = extra[offset:offset+2], struct.unpack('<H', extra[offset+2:offset+4])[0]
                if offset + 4 + size > len(extra):
                    m.fail('EXTERNAL_FRAGMENTS_ENCODING_INVALID')
                if tag == b'BC':
                    return 'bgzf'  # Complete block integrity is checked while decoding.
                offset += 4 + size
            if offset != len(extra):
                m.fail('EXTERNAL_FRAGMENTS_ENCODING_INVALID')
        return 'gzip'


def lines(path, kind):
    try:
        if kind == 'bgzf':
            yield from _bgzf_lines(path)
        else:
            with ExitStack() as stack:
                source = stack.enter_context(gzip.open(path, 'rb') if kind == 'gzip' else open(path, 'rb'))
                if kind == 'zstandard':
                    import zstandard
                    source = stack.enter_context(buffered.BufferedReader(zstandard.ZstdDecompressor().stream_reader(source)))
                for number, line in enumerate(iter(lambda: source.readline(m.MAX_LINE + 1), b'')):
                    if number % 4096 == 0:
                        cancellation_checkpoint()
                    yield line
    except (OSError, EOFError, zlib.error, ValueError) as exc:
        if isinstance(exc, m.ExternalFragmentsError):
            raise
        m.fail('EXTERNAL_FRAGMENTS_ENCODING_INVALID')


def source_index_format(index):
    """Describe supported TBI coordinates; this grants no source pairing proof."""
    try:
        with gzip.open(index, 'rb') as stream:
            header = stream.read(36)
        if len(header) != 36 or header[:4] != b'TBI\1':
            m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
        n, fmt, seq, beg, end, meta, skip, length = struct.unpack('<8i', header[4:])
        if n < 0 or length < 0:
            m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
        if fmt not in (0, 65536) or (seq, beg, end, meta, skip) != (1, 2, 3, 35, 0):
            m.fail('EXTERNAL_FRAGMENTS_INDEX_SEMANTICS_UNSUPPORTED')
        return fmt
    except (OSError, EOFError, struct.error, zlib.error):
        m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')


def _bgzf_lines(path):
    """Use the standard gzip/BGZF decoder for bounded descriptive/history reads."""
    with gzip.open(path, 'rb') as stream:
        for number, line in enumerate(iter(lambda: stream.readline(m.MAX_LINE + 1), b'')):
            if number % 4096 == 0: cancellation_checkpoint()
            if len(line) > m.MAX_LINE: m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
            yield line

def parse_record(line, dictionary=None):
    """Read record structure; a supplied reference additionally owns contig bounds.

    Without a reference this returns the observed contig name, never a qualified
    reference rank. Structural inspection cannot establish coordinate eligibility.
    """
    try:
        fields = line[:-1].decode('utf-8').split('\t')
        if len(fields) not in (5, 6):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        name, left, right, barcode, support = fields[:5]
        strand = fields[5] if len(fields) == 6 else ''
        if ((dictionary is not None and name not in dictionary)
                or dictionary is None and re.fullmatch('[!-~]{1,256}', name) is None
                or any(re.fullmatch('0|[1-9][0-9]{0,18}', n) is None for n in (left, right))
                or re.fullmatch('[!-~]{1,256}', barcode) is None
                or re.fullmatch('[1-9][0-9]{0,19}', support) is None
                or (len(fields) == 6 and strand not in ('+', '-', '.'))):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        left, right, support = int(left), int(right), int(support)
        if (not 0 <= left < right or dictionary is not None and right > dictionary[name][1]
                or not 1 <= support <= MAX_SUPPORT):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
        return (dictionary[name][0] if dictionary is not None else name), left, right, barcode, strand, support, len(fields)
    except (UnicodeError, ValueError) as exc:
        if isinstance(exc, m.ExternalFragmentsError):
            raise
        m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
