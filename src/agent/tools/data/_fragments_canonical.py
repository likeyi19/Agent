"""Production raw BED validation and bounded-memory GNU external sorting."""
import hashlib
from pathlib import Path
import re

from ._fragments_common import MAX_LINE, MAX_SUPPORT, MAX_TOTAL, PACKAGING_POLICY, fail, run_stage


def raw_record(line, contigs, whitelist, barcode_length):
    if len(line) > MAX_LINE or not line.endswith(b'\n') or b'\r' in line:
        fail('FRAGMENTS_OUTPUT_MALFORMED')
    try:
        fields = line[:-1].decode('utf-8').split('\t')
        if len(fields) != 5:
            fail('FRAGMENTS_OUTPUT_MALFORMED')
        chrom, start, end, barcode, support = fields
        if chrom not in contigs:
            fail('FRAGMENTS_COORDINATES_INVALID')
        if any(re.fullmatch('0|[1-9][0-9]*', s) is None for s in (start, end)):
            fail('FRAGMENTS_COORDINATES_INVALID')
        start, end = int(start), int(end)
        if not 0 <= start < end <= contigs[chrom][1]:
            fail('FRAGMENTS_COORDINATES_INVALID')
        if len(barcode) != barcode_length or re.fullmatch('[ACGT]+', barcode) is None or barcode not in whitelist:
            fail('FRAGMENTS_OUTPUT_MALFORMED')
        if re.fullmatch('[1-9][0-9]{0,19}', support) is None or not 1 <= int(support) <= MAX_SUPPORT:
            fail('FRAGMENTS_SUPPORT_INVALID')
        return chrom, start, end, barcode, int(support)
    except (UnicodeError, ValueError) as exc:
        if getattr(exc, 'code', None):
            raise
        fail('FRAGMENTS_OUTPUT_MALFORMED')


def canonicalize(raw_path, *, directory, contigs, whitelist, barcode_length, runtime):
    directory = Path(directory)
    ranks = {name: (i, size) for i, (name, size) in enumerate(contigs)}
    ranked = directory / 'ranked.tmp'; sorted_path = directory / 'sorted.tmp'
    with Path(raw_path).open('rb') as source, ranked.open('wb') as target:
        for line in iter(lambda: source.readline(MAX_LINE + 1), b''):
            chrom, start, end, barcode, support = raw_record(line, ranks, whitelist, barcode_length)
            target.write(f'{ranks[chrom][0]}\t{start}\t{end}\t{barcode}\t{support}\n'.encode())
    run_stage([runtime.sort, *PACKAGING_POLICY['sort'], '-T', directory,
        '-o', sorted_path, '--', ranked], cwd=directory, code='FRAGMENTS_CANONICALIZATION_FAILED')
    path = directory / 'canonical.txt'
    digest = hashlib.sha256(); n = total = maximum = 0; barcodes = set(); previous = None
    with sorted_path.open('rb') as source, path.open('wb') as target:
        for line in iter(lambda: source.readline(MAX_LINE + 1), b''):
            try:
                rank, start, end, barcode, support = line.decode('ascii').rstrip('\n').split('\t')
                rank, start, end, support = map(int, (rank, start, end, support))
                key = (rank, start, end, barcode)
                if previous is not None and key <= previous:
                    fail('FRAGMENTS_OUTPUT_MALFORMED')
                text = f'{contigs[rank][0]}\t{start}\t{end}\t{barcode}\t{support}\n'.encode('utf-8')
                # Revalidate output of the external transformation before accepting it.
                raw_record(text, ranks, whitelist, barcode_length)
            except (ValueError, IndexError, UnicodeError):
                fail('FRAGMENTS_CANONICALIZATION_FAILED')
            previous = key; target.write(text); digest.update(text)
            n += 1; total += support; maximum = max(maximum, support); barcodes.add(barcode)
            if total > MAX_TOTAL or n > MAX_TOTAL:
                fail('FRAGMENTS_SUPPORT_INVALID')
    if not n:
        fail('FRAGMENTS_EMPTY')
    return path, {'canonical_record_stream_sha256': digest.hexdigest(),
        'n_fragment_records': n, 'sum_support': total, 'max_support': maximum,
        'n_distinct_barcodes': len(barcodes)}
