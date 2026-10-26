"""Bounded descriptive fragment inspection, with no preprocessing authority."""
import gzip
from pathlib import Path
import struct

from . import _external_fragment_io as io, external_fragment_manifest as m
from .external_fragments import parse_record
from .scatac_fragments_v2 import absolute_path, sha

RECORD_LIMIT = 10_000
POLICY = 'inspect-fragment-structure-v1'
RESULT_FIELDS = ('source_path', 'source_sha256', 'source_index_sha256', 'source_size_bytes',
    'encoding', 'columns', 'record_limit', 'n_records_observed', 'sum_support_observed',
    'n_distinct_barcodes_observed', 'n_contigs_observed', 'reached_eof', 'index_status', 'limitations')
LIMITATIONS = ('Observations cover only the inspected prefix unless reached_eof is true. '
    'Barcode counts are distinct observed strings, not cells. No assembly, contig bounds, '
    'sorting, duplicate-key absence, processing history, barcode correction, cell quality, '
    'TSS enrichment or QC readiness is established. An index header does not establish '
    'correspondence with the source. No canonical fragments or downstream authority is produced.')


def inspect_scATAC_fragments(source_path, source_sha256, source_index_path=None, source_index_sha256=None):
    """Inspect at most 10,000 records using the existing fragment owner's reader."""
    absolute_path(str(source_path)); sha(source_sha256)
    if (source_index_path is None) != (source_index_sha256 is None):
        m.fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
    paths = [source_path]
    if source_index_path is not None:
        absolute_path(str(source_index_path)); sha(source_index_sha256)
        paths.append(source_index_path)
    before = io.take_snapshots(paths)
    source = io.resource(source_path, source_sha256)
    kind = io.encoding(source_path)
    index_status = 'not_supplied'
    if source_index_path is not None:
        io.resource(source_index_path, source_index_sha256)
        if kind != 'bgzf': m.fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
        try:
            with gzip.open(source_index_path, 'rb') as stream:
                header = stream.read(36)
            if len(header) != 36 or header[:4] != b'TBI\1':
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
            n, fmt, seq, beg, end, meta, skip, length = struct.unpack('<8i', header[4:])
            if n < 0 or length < 0 or (fmt, seq, beg, end, meta, skip) != (65536, 1, 2, 3, 35, 0):
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
        except (OSError, EOFError, struct.error):
            m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
        index_status = 'tbi_header_only_pair_unverified'
    count = support = header_count = header_bytes = 0
    columns = None
    barcodes, contigs = set(), set()
    complete = False
    reader = io.lines(source_path, kind)
    try:
        for line in reader:
            if len(line) > m.MAX_LINE or not line.endswith(b'\n') or b'\r' in line or b'\0' in line:
                m.fail('EXTERNAL_FRAGMENTS_RECORD_INVALID')
            if line.startswith(b'#'):
                header_count += 1; header_bytes += len(line)
                if count or header_count > m.MAX_HEADERS or header_bytes > m.MAX_HEADER_BYTES:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                try: line.decode('utf-8')
                except UnicodeError: m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                continue
            name, start, end, barcode, strand, weight, width = parse_record(line)
            if columns is not None and width != columns: m.fail('EXTERNAL_FRAGMENTS_LAYOUT_MIXED')
            columns = width; count += 1; support += weight
            barcodes.add(barcode); contigs.add(name)
            if count == RECORD_LIMIT: break
        else:
            complete = True
    finally:
        reader.close()
    if not count: m.fail('EXTERNAL_FRAGMENTS_EMPTY')
    io.check_snapshots(before)
    return dict(source_path=str(Path(source_path)), source_sha256=source_sha256,
        source_index_sha256=source_index_sha256, source_size_bytes=source['size_bytes'],
        encoding=kind, columns=columns, record_limit=RECORD_LIMIT, n_records_observed=count,
        sum_support_observed=support, n_distinct_barcodes_observed=len(barcodes),
        n_contigs_observed=len(contigs), reached_eof=complete, index_status=index_status,
        limitations=LIMITATIONS)


def verify_result(arguments, result):
    """Recheck a bounded observation and exact source identity before acceptance."""
    if inspect_scATAC_fragments(**arguments) != dict(result):
        m.fail('EXTERNAL_FRAGMENTS_INSPECTION_MISMATCH')
