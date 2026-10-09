"""Original QC/selection table eligibility on an already verified snapshot."""

import re
import zlib

from agent.tools.data import _barcode_qc_contract as qc
from agent.tools.data import _cell_selection_contract as selection
from agent.tools.data.scatac_selection_profile import REASONS, decode_cell_id


class TablePrivacyError(ValueError):
    """The original table is outside the reviewed delivery layout."""

    def __init__(self):
        super().__init__('The accepted scientific table is not eligible for original-file delivery.')


_HEADERS = {
    (qc.CONTRACT, 'table'): qc.HEADER,
    (qc.CONTRACT, 'histogram'): None,
    (selection.CONTRACT, 'decisions'): selection.DECISION_HEADER,
    (selection.CONTRACT, 'selected'): selection.SELECTED_HEADER,
}
_COMPRESSED_CHUNK = 64 * 1024
_DECIMAL = re.compile(r'(?:0|[1-9][0-9]*)\Z')
_PATH_TEXT = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')
# Both accepted owners use GzipFile(filename='', mtime=0, compresslevel=6).
# The closed header excludes original filenames, comments and extra fields.
_GZIP_HEADER = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff'


def _reject(condition):
    if condition:
        raise TablePrivacyError()


def _gzip_lines(stream, max_lines):
    """Read one gzip member with bounded compressed and expanded chunks."""
    stream.seek(0)
    _reject(stream.read(len(_GZIP_HEADER)) != _GZIP_HEADER)
    stream.seek(0)
    decoder = zlib.decompressobj(31)
    remainder = b''
    count = 0
    while not decoder.eof:
        compressed = stream.read(_COMPRESSED_CHUNK)
        _reject(not compressed)
        # max_length also bounds highly compressible input; unconsumed_tail
        # stays within one compressed chunk. No whole table is buffered.
        while True:
            expanded = decoder.decompress(compressed, qc.MAX_ROW + 1)
            compressed = decoder.unconsumed_tail
            remainder += expanded
            while b'\n' in remainder:
                end = remainder.index(b'\n') + 1
                _reject(end > qc.MAX_ROW or count == max_lines)
                line, remainder = remainder[:end], remainder[end:]
                count += 1
                yield line
            _reject(len(remainder) > qc.MAX_ROW)
            if decoder.eof:
                # Reject concatenated members and any trailing metadata or
                # padding. zlib already checks the member CRC and size trailer.
                _reject(bool(decoder.unused_data) or bool(stream.read(1)))
                break
            if not compressed and len(expanded) < qc.MAX_ROW + 1:
                break
    _reject(bool(remainder))


def _identifier(namespace, barcode):
    # These are scientific identities, not anonymized or relabeled content.
    # The owner permits punctuation in opaque barcodes; only unmistakable
    # filesystem/URI identifiers are outside original-byte eligibility.
    qc.identity(namespace, barcode)
    _reject(_PATH_TEXT.search(barcode) is not None)


def _number(value):
    _reject(_DECIMAL.fullmatch(value) is None)


def _fields(line, width):
    fields = line[:-1].decode('ascii').split('\t')
    _reject(len(fields) != width)
    return fields


def _barcode_row(line):
    fields = _fields(line, len(qc.COLUMNS))
    _identifier(*fields[:2])
    for value in fields[2:10]:
        _number(value)
    for offset, missing in ((10, 'ZERO_TSS_BACKGROUND'), (13, 'ZERO_NUCLEOSOME_FREE')):
        for value in fields[offset:offset + 2]:
            if value != 'NA':
                _number(value)
        _reject(fields[offset + 2] not in {'DEFINED', missing})


def _selection_identity(namespace, barcode, rendered):
    _identifier(namespace, barcode)
    _reject(decode_cell_id(rendered) != (namespace, barcode))


def _decision_row(line):
    fields = _fields(line, 7)
    _selection_identity(*fields[:3])
    _reject(fields[3] != 'not_assessed' or fields[4] not in {'true', 'false'}
            or fields[5] not in {'true', 'false'})
    _reject(fields[6] != 'NONE' and any(reason not in REASONS for reason in fields[6].split(',')))


def _selected_row(line):
    fields = _fields(line, 4)
    _number(fields[0])
    _selection_identity(*fields[1:])


def _row_count(contract, role, manifest):
    if role == 'histogram':
        return 1001
    if manifest is None:
        return None
    _reject(manifest['contract_version'] != contract)
    value = manifest['selected_count' if role == 'selected' else 'row_count']
    _reject(type(value) is not int or not 0 <= value <= qc.MAX_BARCODES
            or value == 0 and role != 'selected')
    return value


def validate_scientific_table(stream, contract, role, manifest=None):
    """Check closed table format and gzip metadata without doing science.

    ``stream`` is the seekable, exact-byte snapshot prepared by the shared
    delivery boundary. Accepted authority and payload hashes are established
    before this function. Headers, identifiers and row counts are checked;
    counts, fractions, histogram values and selection decisions are never
    recalculated, and no scientific owner verifier is invoked. The caller
    retains its stream and its original compressed bytes.
    """
    try:
        _reject((contract, role) not in _HEADERS)
        if manifest is not None:
            _reject(manifest['contract_version'] != contract)
        header = _HEADERS[(contract, role)]
        expected = _row_count(contract, role, manifest)
        limit = expected if expected is not None else qc.MAX_BARCODES
        rows = _gzip_lines(stream, limit + int(header is not None))
        if header is not None:
            _reject(next(rows, None) != header)
        count = 0
        for count, line in enumerate(rows, 1):
            if role == 'table':
                _barcode_row(line)
            elif role == 'histogram':
                fields = _fields(line, 2)
                _reject(fields[0] != str(count))
                _number(fields[1])
            elif role == 'decisions':
                _decision_row(line)
            else:
                _selected_row(line)
        _reject(expected is not None and count != expected)
        _reject(count == 0 and role != 'selected')
    except TablePrivacyError:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, zlib.error,
            OverflowError, AttributeError) as exc:
        raise TablePrivacyError() from exc
