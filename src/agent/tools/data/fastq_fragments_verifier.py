"""FASTQ-specific provenance and whitelist verification for fragments v2.

No Chromap execution, correction/alignment rerun or historical biology claim.
"""
import hashlib
from pathlib import Path
import re
import subprocess
from . import _chromap as c, fastq_fragment_manifest as m
from ._fragment_io import _bgzf_lines, _query
from ._fragments_binding import FragmentInputs, library_binding, preflight
from ._fragments_common import MAX_SUPPORT, MAX_TOTAL, fail, snapshots, unchanged, verify_packaging
from .scatac_fragments_v2_verifier import verify_fragments_v2

def verify_stream(path, entry, contigs, whitelist):
    """Independent parser: no imports/calls to production raw_record/canonicalize."""
    dictionary = {name: (rank, length) for rank, (name, length) in enumerate(contigs)}
    digest = hashlib.sha256(); count = total = maximum = 0; previous = None
    barcodes = set(); per_contig = {}; probes = {}
    for line in _bgzf_lines(path):
        try:
            if b'\r' in line or b'\0' in line:
                raise ValueError()
            columns = line[:-1].decode('utf-8').split('\t')
            if len(columns) != 5:
                raise ValueError()
            name, left, right, barcode, support = columns
            if name not in dictionary or any(re.fullmatch(r'0|[1-9][0-9]*', x) is None for x in (left, right)):
                raise ValueError()
            left, right = int(left), int(right)
            if not 0 <= left < right <= dictionary[name][1]:
                raise ValueError()
            if (re.fullmatch('[ACGT]+', barcode) is None or len(barcode) != entry['barcode_length']
                    or barcode not in whitelist or re.fullmatch('[1-9][0-9]{0,19}', support) is None):
                raise ValueError()
            support = int(support)
            if not 1 <= support <= MAX_SUPPORT:
                raise ValueError()
            key = (dictionary[name][0], left, right, barcode.encode('ascii'))
            if previous is not None and previous >= key:
                raise ValueError()
            previous = key
        except (ValueError, UnicodeError):
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        digest.update(line); count += 1; total += support; maximum = max(maximum, support)
        if total > MAX_TOTAL or count > MAX_TOTAL:
            fail('FRAGMENTS_SUPPORT_INVALID')
        barcodes.add(barcode)
        per_contig.setdefault(name, hashlib.sha256()).update(line)
        # First canonical interval per contig is deterministic content-derived input.
        probes.setdefault(name, (left, right))
    summary = dict(canonical_record_stream_sha256=digest.hexdigest(), n_fragment_records=count,
        sum_support=total, max_support=maximum, n_distinct_barcodes=len(barcodes))
    if not count or any(entry[k] != v for k, v in summary.items()):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    # Independent second stream obtains expected overlap digests for narrow queries.
    overlaps = {name: hashlib.sha256() for name in probes}
    for line in _bgzf_lines(path):
        fields = line[:-1].decode('utf-8').split('\t')
        start, end = probes[fields[0]]
        if int(fields[1]) < end and int(fields[2]) > start:
            overlaps[fields[0]].update(line)
    return per_contig, probes, overlaps


from .authority_context import owned_verification

@owned_verification('fastq_fragment_production')
def verify_fragments(manifest_path, *, runtime, expected_sha256=None):
    """Revalidate full FASTQ provenance, independent rows, then generic v2 IO."""
    from ._fragments_common import FragmentsError
    from ._fragments_fastq import scan_group
    try:
        value = m.load_manifest(manifest_path, expected_sha256=expected_sha256)
        root = Path(manifest_path).parent
        before = snapshots([manifest_path, root / 'profile.json', root / 'production.json'])
        verify_packaging(runtime)
        record = m.record_for_manifest(value, root)
        backend = c.validate_backend(record['backend'])
        bound = preflight(FragmentInputs(**record['inputs']), backend)
        if (bound['lineage'] != record['lineage'] or len(bound['libraries']) != len(record['libraries'])
                or m.build_manifest(record, root) != value):
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
        for (library, groups), entry in zip(bound['libraries'], record['libraries'], strict=True):
            if any(entry[k] != v for k, v in library_binding(library, groups).items()):
                fail('FRAGMENTS_CONTEXT_MISMATCH')
            if entry['sources'] != m.source_resources(groups):
                fail('FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION')
            expected_scans = {'group:' + group.group_id: scan_group(group, library.whitelist.barcode_length)
                              for group in groups}
            if entry['scans'] != expected_scans:
                fail('FRAGMENTS_SOURCE_IDENTITY_MISMATCH')
            white = set(Path(library.whitelist.resource.path).read_text('ascii').splitlines())
            for kind in ('bgzf', 'tabix'):
                artifact = entry[kind]; path = root / artifact['path']
                if path.stat().st_size != artifact['size_bytes'] or c.sha256(path) != artifact['sha256']:
                    fail('FRAGMENTS_INDEX_MISMATCH' if kind == 'tabix' else 'FRAGMENTS_VERIFICATION_MISMATCH')
            path = root / entry['bgzf']['path']
            per_contig, probes, overlaps = verify_stream(path, entry, bound['contigs'], white)
            listing = subprocess.run([runtime.tabix, '-l', str(path)], capture_output=True, check=False)
            names = [n for n, _ in bound['contigs'] if n in per_contig]
            if listing.returncode or listing.stdout.decode().splitlines() != names:
                fail('FRAGMENTS_INDEX_MISMATCH')
            for name in names:
                left, right = probes[name]
                if (_query(runtime, path, name) != per_contig[name].hexdigest()
                        or _query(runtime, path, f'{name}:{left+1}-{right}') != overlaps[name].hexdigest()):
                    fail('FRAGMENTS_INDEX_MISMATCH')
        verified = verify_fragments_v2(manifest_path, expected_sha256=expected_sha256 or c.sha256(manifest_path), runtime=runtime)
        verified.check_unchanged(); unchanged(before); unchanged(bound['snapshots'])
        return value
    except FragmentsError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
