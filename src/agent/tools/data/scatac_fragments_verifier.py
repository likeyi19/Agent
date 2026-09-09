"""Independent full artifact verification; never executes Chromap or a full scan.

Parses BGZF blocks and logical rows independently of production canonicalization.
Checks actual TBI queries for every represented contig against streamed digests.
This does not prove alignment/correction biology or rerun duplicate classification.
"""
from dataclasses import asdict
import hashlib
from pathlib import Path
import re
import struct
import subprocess
import zlib

from . import _chromap as c
from ._fragments_binding import FragmentInputs, library_binding, preflight
from ._fragments_common import (MAX_LINE, MAX_SUPPORT, MAX_TOTAL, fail, snapshot, snapshots,
    unchanged, verify_packaging)
from .scatac_fragments_manifest import load_fragments_manifest

BGZF_EOF = bytes.fromhex('1f8b08040000000000ff0600424302001b0003000000000000000000')


def _bgzf_lines(path):
    """Bounded BGZF block decoding, CRC/ISIZE and mandatory terminal EOF block."""
    pending = bytearray(); saw_eof = False
    try:
        with Path(path).open('rb') as stream:
            while True:
                header = stream.read(12)
                if not header:
                    break
                if saw_eof or len(header) != 12 or header[:4] != b'\x1f\x8b\x08\x04':
                    fail('FRAGMENTS_VERIFICATION_MISMATCH')
                xlen = struct.unpack('<H', header[10:12])[0]
                extra = stream.read(xlen); offset = 0; sizes = []
                while offset + 4 <= len(extra):
                    tag = extra[offset:offset+2]; size = struct.unpack('<H', extra[offset+2:offset+4])[0]
                    payload = extra[offset+4:offset+4+size]
                    if len(payload) != size:
                        fail('FRAGMENTS_VERIFICATION_MISMATCH')
                    if tag == b'BC' and size == 2:
                        sizes.append(struct.unpack('<H', payload)[0] + 1)
                    offset += 4 + size
                if offset != xlen or len(sizes) != 1 or not 12+xlen+8 <= sizes[0] <= 65536:
                    fail('FRAGMENTS_VERIFICATION_MISMATCH')
                remainder = stream.read(sizes[0] - 12 - xlen)
                if len(remainder) != sizes[0] - 12 - xlen:
                    fail('FRAGMENTS_VERIFICATION_MISMATCH')
                block = header + extra + remainder
                decoder = zlib.decompressobj(wbits=31)
                decoded = decoder.decompress(block, 65537)
                if len(decoded) > 65536 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    fail('FRAGMENTS_VERIFICATION_MISMATCH')
                if not decoded:
                    if block != BGZF_EOF:
                        fail('FRAGMENTS_VERIFICATION_MISMATCH')
                    saw_eof = True
                pending.extend(decoded)
                while b'\n' in pending:
                    end = pending.index(10) + 1
                    if end > MAX_LINE:
                        fail('FRAGMENTS_VERIFICATION_MISMATCH')
                    yield bytes(pending[:end]); del pending[:end]
                if len(pending) > MAX_LINE:
                    fail('FRAGMENTS_VERIFICATION_MISMATCH')
        if not saw_eof or pending:
            fail('FRAGMENTS_VERIFICATION_MISMATCH')
    except (OSError, zlib.error, struct.error):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')


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


def _query(runtime, path, region):
    with subprocess.Popen([runtime.tabix, str(path), region], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, env={'LC_ALL': 'C'}) as process:
        h = hashlib.sha256()
        try:
            for chunk in iter(lambda: process.stdout.read(65536), b''):
                h.update(chunk)
            if process.wait():
                fail('FRAGMENTS_INDEX_MISMATCH')
        except BaseException:
            process.kill(); process.wait(); raise
    return h.hexdigest()


def _verify_fragments(manifest_path, *, runtime, expected_sha256=None):
    value = load_fragments_manifest(manifest_path, expected_sha256=expected_sha256)
    verify_packaging(runtime)
    # No executable execution is needed for artifact verification: validated
    # recorded Chromap identity is the authority; no aligner invocation here.
    backend = c.validate_backend(value['backend'])
    bound = preflight(FragmentInputs(**value['inputs']), backend)
    if bound['lineage'] != value['lineage'] or len(bound['libraries']) != len(value['libraries']):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
    root = Path(manifest_path).parent
    before = snapshots([manifest_path] + [root / e[k]['path'] for e in value['libraries'] for k in ('bgzf', 'tabix')])
    for (library, groups), entry in zip(bound['libraries'], value['libraries'], strict=True):
        if any(entry[k] != v for k, v in library_binding(library, groups).items()):
            fail('FRAGMENTS_CONTEXT_MISMATCH')
        for kind in ('bgzf', 'tabix'):
            artifact = entry[kind]; path = root / artifact['path']
            if snapshot(path)[2] != artifact['size_bytes'] or c.sha256(path) != artifact['sha256']:
                fail('FRAGMENTS_INDEX_MISMATCH' if kind == 'tabix' else 'FRAGMENTS_VERIFICATION_MISMATCH')
        # Whitelist memory is bounded by the supplied candidate set, not reads.
        white = set(Path(library.whitelist.resource.path).read_text('ascii').splitlines())
        path = root / entry['bgzf']['path']
        per_contig, probes, overlaps = verify_stream(path, entry, bound['contigs'], white)
        result = subprocess.run([runtime.tabix, '-l', str(path)], capture_output=True, check=False)
        expected_names = [n for n, _ in bound['contigs'] if n in per_contig]
        if result.returncode or result.stdout.decode('utf-8').splitlines() != expected_names:
            fail('FRAGMENTS_INDEX_MISMATCH')
        for name in expected_names:
            if _query(runtime, path, name) != per_contig[name].hexdigest():
                fail('FRAGMENTS_INDEX_MISMATCH')
            left, right = probes[name]
            if _query(runtime, path, f'{name}:{left+1}-{right}') != overlaps[name].hexdigest():
                fail('FRAGMENTS_INDEX_MISMATCH')
    unchanged(before); unchanged(bound['snapshots'])
    return value


def verify_fragments(manifest_path, *, runtime, expected_sha256=None):
    """Fresh lineage, complete independent BGZF rows, and functional TBI checks."""
    from ._fragments_common import FragmentsError
    try:
        return _verify_fragments(manifest_path, runtime=runtime, expected_sha256=expected_sha256)
    except FragmentsError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError):
        fail('FRAGMENTS_VERIFICATION_MISMATCH')
