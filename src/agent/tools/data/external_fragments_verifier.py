"""Independent source reparse and exact conservation; no historical reconstruction."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile

from . import external_fragment_manifest as m, _external_fragment_io as io
from . import scatac_fragments_v2 as v2
from ._fragments_common import MAX_SUPPORT, MAX_TOTAL
from .scatac_fragments_v2_verifier import verify_fragments_v2, FragmentVerification


@dataclass(frozen=True)
class ExternalAdoptionVerification:
    fragments: FragmentVerification
    source_content: str = 'verified'
    canonicalization_conservation: str = 'verified'
    source_profile: str = m.PROFILE_ID
    producer_history: str = 'declared_not_reconstructed'


def _reconstruct_source(path, contigs, directory, runtime):
    """Does not import production parse_record, scan_source or canonicalize."""
    kind = io.encoding(path)
    dictionary = {n: (i, length) for i, (n, length) in enumerate(contigs)}
    decoded = hashlib.sha256(); data = hashlib.sha256(); headers = hashlib.sha256()
    header_count = header_size = count = total = maximum = 0; width = None; identifiers = set()
    ranked = directory / 'verification-input.tmp'
    with ranked.open('xb') as target:
        for raw in io.lines(path, kind):
            if len(raw) > m.MAX_LINE or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            decoded.update(raw)
            try:
                text = raw[:-1].decode('utf-8')
            except UnicodeError:
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            if text.startswith('#'):
                if width is not None:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                header_count += 1; header_size += len(raw); headers.update(raw)
                if header_count > m.MAX_HEADERS or header_size > m.MAX_HEADER_BYTES:
                    m.fail('EXTERNAL_FRAGMENTS_HEADER_INVALID')
                continue
            fields = text.split('\t')
            if len(fields) not in (5, 6) or (width is not None and len(fields) != width):
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            width = len(fields); chromosome, left_text, right_text, barcode, support_text = fields[:5]
            if chromosome not in dictionary or not 1 <= len(barcode) <= 256 or any(not 33 <= ord(ch) <= 126 for ch in barcode):
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            # Independently check canonical integer spelling and bounds.
            for text_number, bound in ((left_text, 2**63-1), (right_text, 2**63-1), (support_text, MAX_SUPPORT)):
                if (not text_number or len(text_number) > 20 or any(ch not in '0123456789' for ch in text_number)
                        or (len(text_number) > 1 and text_number[0] == '0') or int(text_number) > bound):
                    m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            left, right, support = int(left_text), int(right_text), int(support_text)
            if not 0 <= left < right <= dictionary[chromosome][1] or support < 1:
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            strand = fields[5] if width == 6 else ''
            if width == 6 and strand not in ('+', '-', '.'):
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_INVALID')
            data.update(raw); count += 1; total += support; maximum = max(maximum, support); identifiers.add(barcode)
            if count > MAX_TOTAL or total > MAX_TOTAL:
                m.fail('EXTERNAL_FRAGMENTS_SUPPORT_OVERFLOW')
            target.write(f'{dictionary[chromosome][0]}\t{left}\t{right}\t{barcode}\t{strand}\t{support}\n'.encode())
    if not count:
        m.fail('EXTERNAL_FRAGMENTS_EMPTY')
    ordered = directory / 'verification-sorted.tmp'
    io.sort_ranked(ranked, ordered, directory, runtime)
    canonical = hashlib.sha256(); previous = None; sorted_count = 0
    with ordered.open('rb') as source:
        for line in iter(lambda: source.readline(m.MAX_LINE + 1), b''):
            rank, left, right, barcode, strand, support = line.decode().rstrip('\n').split('\t')
            key = int(rank), int(left), int(right), barcode, strand
            if previous is not None and key <= previous:
                m.fail('EXTERNAL_FRAGMENTS_DUPLICATE_KEY')
            previous = key; sorted_count += 1
            row = [contigs[int(rank)][0], left, right, barcode, support]
            if width == 6:
                row.append(strand)
            canonical.update(('\t'.join(row) + '\n').encode())
    if sorted_count != count:
        m.fail('EXTERNAL_FRAGMENTS_CONSERVATION_MISMATCH')
    source = dict(resource=io.resource(path), encoding=kind, decoded_sha256=decoded.hexdigest(),
        data_sha256=data.hexdigest(), header_sha256=headers.hexdigest(), header_bytes=header_size,
        header_records=header_count, columns=width, n_records=count, sum_support=total)
    summary = dict(canonical_record_stream_sha256=canonical.hexdigest(), n_fragment_records=count,
                   sum_support=total, max_support=maximum, n_distinct_barcodes=len(identifiers))
    return source, summary


from .authority_context import owned_verification

@owned_verification('external_fragment_adoption')
def verify_external_fragments(manifest_path, *, expected_sha256, runtime, temporary_root=None):
    """Fresh source + conservation + generic v2 verification in managed scratch."""
    try:
        initial = io.take_snapshots([manifest_path])
        manifest = v2.load_fragments_manifest_v2(manifest_path, expected_sha256=expected_sha256)
        if len(manifest['libraries']) != 1:
            m.fail('EXTERNAL_FRAGMENTS_RECORD_MISMATCH')
        entry = manifest['libraries'][0]; p = entry['provenance']
        root = Path(manifest_path).parent
        if (p['kind'] != 'external_fragment_adoption' or p['profile']['id'] != m.PROFILE_ID
                or p['profile']['resource']['path'] != str(root / 'profile.json')
                or p['producer_record'] is None or p['producer_record']['path'] != str(root / 'adoption.json')):
            m.fail('EXTERNAL_FRAGMENTS_PROFILE_UNSUPPORTED')
        snapshots = io.take_snapshots([root / 'profile.json', root / 'adoption.json'])
        if io.resource(root / 'profile.json') != p['profile']['resource'] or p['profile']['resource']['sha256'] != m.sha_bytes(m.PROFILE_BYTES):
            m.fail('EXTERNAL_FRAGMENTS_PROFILE_UNSUPPORTED')
        record = m.load_adoption_record(root / 'adoption.json', p['producer_record']['sha256'])
        if (record['reference'] != manifest['reference'] or record['namespace'] != entry['namespace']
                or p != m.provenance(record, p['profile']['resource'], p['producer_record'])):
            m.fail('EXTERNAL_FRAGMENTS_RECORD_MISMATCH')
        paths = [record['source']['resource']['path']]
        if record['source_index'] is not None:
            paths.append(record['source_index']['path'])
        before = io.take_snapshots(paths)
        generic = verify_fragments_v2(manifest_path, expected_sha256=expected_sha256, runtime=runtime)
        with tempfile.TemporaryDirectory(prefix='agent-external-verify-', dir=temporary_root) as scratch:
            source, summary = _reconstruct_source(paths[0], generic.contigs, Path(scratch), runtime)
            if source != record['source']:
                m.fail('EXTERNAL_FRAGMENTS_SOURCE_MISMATCH')
            if summary != record['canonical'] or any(entry[k] != summary[k] for k in v2.SUMMARY_KEYS):
                m.fail('EXTERNAL_FRAGMENTS_CONSERVATION_MISMATCH')
            expected_strand = dict(mode='present' if source['columns'] == 6 else 'absent',
                definition=m.STRAND_DEFINITION if source['columns'] == 6 else None)
            if entry['strand'] != expected_strand:
                m.fail('EXTERNAL_FRAGMENTS_CONSERVATION_MISMATCH')
            if record['source_index'] is not None and io.resource(paths[1]) != record['source_index']:
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
            io.check_source_index(paths[0], paths[1] if len(paths) == 2 else None,
                                  source['encoding'], generic.contigs, scratch, runtime)
        generic.check_unchanged(); io.check_snapshots(before); io.check_snapshots(snapshots); io.check_snapshots(initial)
        return ExternalAdoptionVerification(generic)
    except m.ExternalFragmentsError:
        raise
    except (ValueError, OSError, KeyError, TypeError, IndexError, UnicodeError):
        m.fail('EXTERNAL_FRAGMENTS_VERIFICATION_FAILED')
