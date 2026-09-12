"""Independent v2 content/reference/resource verification, never production.

Uses the source-neutral BGZF decoder and tabix query helper.
No FASTQ preflight, whitelist validation, Chromap identity/index, or v1 parser is
used. Profile/source bytes are checked, but no producer profile is qualified and
no historical processing is reconstructed by this module.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess

from . import scatac_fragments_v2 as m, scatac_reference as reference
from ._fragments_common import (FragmentsError, MAX_SUPPORT, MAX_TOTAL, TBI_LIMIT,
                               snapshot, verify_packaging)
from ._fragment_io import _bgzf_lines, _query


@dataclass(frozen=True)
class FragmentVerificationRuntime:
    """Existing qualified packaging tools, with no scientific producer field."""
    bgzip: str = '/usr/bin/bgzip'
    tabix: str = '/usr/bin/tabix'
    sort: str = '/usr/bin/sort'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def take_snapshots(paths):
    result = []
    for item in sorted(set(map(str, paths))):
        path = Path(item)
        if not path.is_absolute() or path != path.resolve():
            m.fail('FRAGMENTS_V2_SOURCE_CHANGED')
        result.append((item, snapshot(path)))
    return tuple(result)


def check_snapshots(before):
    try:
        if take_snapshots(p for p, _ in before) != before:
            m.fail('FRAGMENTS_V2_SOURCE_CHANGED')
    except (OSError, FragmentsError, ValueError):
        m.fail('FRAGMENTS_V2_SOURCE_CHANGED')


@dataclass(frozen=True)
class FragmentVerification:
    """Explicit verification scopes, with an immutable manifest snapshot."""
    manifest_path: str
    manifest_sha256: str
    manifest_bytes: bytes
    contigs: tuple[tuple[str, int], ...]
    snapshots: tuple
    artifact_content: str = 'verified'
    bound_resource_identities: str = 'verified'
    producer_history: str = 'not_verified'
    producer_profile_qualification: str = 'not_established'

    @property
    def manifest(self):
        return json.loads(self.manifest_bytes)

    def check_unchanged(self):
        check_snapshots(self.snapshots)


def _verify_stream(path, entry, contigs):
    """Independent row parser; does not call the downstream reader."""
    dictionary = {name: (rank, length) for rank, (name, length) in enumerate(contigs)}
    digest = hashlib.sha256(); per_contig = {}; probes = {}; barcodes = set()
    count = total = maximum = 0; previous = None
    present = entry['strand']['mode'] == 'present'
    for line in _bgzf_lines(path):
        try:
            fields = line[:-1].decode('utf-8').split('\t')
            if len(fields) != (6 if present else 5) or b'\r' in line or b'\0' in line:
                raise ValueError()
            name, left, right, barcode, support = fields[:5]
            strand = fields[5] if present else ''
            if present and strand not in ('+', '-', '.'):
                raise ValueError()
            if (name not in dictionary or any(re.fullmatch('0|[1-9][0-9]{0,18}', x) is None for x in (left, right))
                    or re.fullmatch('[!-~]{1,256}', barcode) is None
                    or re.fullmatch('[1-9][0-9]{0,19}', support) is None):
                raise ValueError()
            left, right, support = int(left), int(right), int(support)
            if not 0 <= left < right <= dictionary[name][1] or not 1 <= support <= MAX_SUPPORT:
                raise ValueError()
            key = (dictionary[name][0], left, right, barcode.encode('ascii'), strand)
            if previous is not None and key <= previous:
                raise ValueError()
            previous = key
        except (ValueError, UnicodeError):
            m.fail('FRAGMENTS_V2_RECORD_INVALID')
        count += 1; total += support; maximum = max(maximum, support)
        if total > MAX_TOTAL or count > MAX_TOTAL:
            m.fail('FRAGMENTS_V2_SUPPORT_INVALID')
        digest.update(line); barcodes.add(barcode)
        per_contig.setdefault(name, hashlib.sha256()).update(line)
        probes.setdefault(name, (left, right))
    actual = dict(canonical_record_stream_sha256=digest.hexdigest(), n_fragment_records=count,
                  sum_support=total, max_support=maximum, n_distinct_barcodes=len(barcodes))
    if not count or any(entry[key] != value for key, value in actual.items()):
        m.fail('FRAGMENTS_V2_STREAM_MISMATCH')
    # Independent second pass derives expected narrow BED interval queries.
    overlaps = {name: hashlib.sha256() for name in probes}
    for line in _bgzf_lines(path):
        fields = line[:-1].decode('utf-8').split('\t')
        left, right = probes[fields[0]]
        if int(fields[1]) < right and int(fields[2]) > left:
            overlaps[fields[0]].update(line)
    return per_contig, probes, overlaps


def _tabix_contigs(runtime, path):
    listing = subprocess.run([runtime.tabix, '-l', str(path)], capture_output=True, check=False,
                             env={'LC_ALL': 'C'})
    if listing.returncode:
        m.fail('FRAGMENTS_V2_INDEX_MISMATCH')
    return listing.stdout.decode('utf-8').splitlines()


def _verify(manifest_path, expected_sha256, runtime):
    initial = take_snapshots([manifest_path])
    payload, value = m.read_manifest_bytes(manifest_path, expected_sha256)
    value = m.validate_fragments_manifest_v2(value)
    r = value['reference']
    reference_snapshot = take_snapshots([r['manifest_path']])
    _, bundle, _ = reference.load_scatac_reference_bundle(r['manifest_path'], expected_sha256=r['manifest_sha256'])
    if (bundle.reference_identity_sha256 != r['reference_identity_sha256']
            or bundle.genome.ordered_contig_sha256 != r['ordered_contig_sha256']
            or bundle.species != r['species'] or bundle.target_assembly != r['assembly']):
        m.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')
    root = Path(manifest_path).parent
    resources = list(m.bound_provenance_resources(value))
    resources += [e[k] | {'path': str(root / e[k]['path'])}
                  for e in value['libraries'] for k in ('bgzf', 'tabix')]
    paths = [str(manifest_path), r['manifest_path'], bundle.genome.fasta.path,
             bundle.genome.fai.path, bundle.ccre.bed.path] + [a['path'] for a in resources]
    if bundle.annotation:
        paths.append(bundle.annotation.resource.path)
    before = take_snapshots(paths)
    check_snapshots(initial); check_snapshots(reference_snapshot)
    reference.reinspect_scatac_reference_bundle_sources(bundle)
    # The M11.1 FAI parser is the reference dictionary authority. Do not import
    # the Chromap index gate, which imposes a separate FASTA scan/order policy.
    dictionary, fai_sha, ordered = reference._inspect_fai(Path(bundle.genome.fai.path),
                                                        Path(bundle.genome.fasta.path).stat().st_size)
    if fai_sha != bundle.genome.fai.sha256 or ordered != r['ordered_contig_sha256']:
        m.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')
    contigs = tuple(dictionary.items())
    if any(length > TBI_LIMIT for _, length in contigs):
        m.fail('FRAGMENTS_V2_INDEX_POLICY_UNSUPPORTED')
    checked = {}
    for artifact in resources:
        path = artifact['path']
        if path not in checked:
            checked[path] = (Path(path).stat().st_size, file_sha256(path))
        if checked[path] != (artifact['size_bytes'], artifact['sha256']):
            m.fail('FRAGMENTS_V2_RESOURCE_MISMATCH')
    # Reuse the qualified packaging checks; no aligner or producer identity.
    verify_packaging(runtime)
    for entry in value['libraries']:
        path = root / entry['bgzf']['path']
        per_contig, probes, overlaps = _verify_stream(path, entry, contigs)
        names = [name for name, _ in contigs if name in per_contig]
        if _tabix_contigs(runtime, path) != names:
            m.fail('FRAGMENTS_V2_INDEX_MISMATCH')
        for name in names:
            if _query(runtime, path, name) != per_contig[name].hexdigest():
                m.fail('FRAGMENTS_V2_INDEX_MISMATCH')
            left, right = probes[name]
            if _query(runtime, path, f'{name}:{left+1}-{right}') != overlaps[name].hexdigest():
                m.fail('FRAGMENTS_V2_INDEX_MISMATCH')
    check_snapshots(before)
    return FragmentVerification(str(manifest_path), expected_sha256, payload, contigs, before)


from .authority_context import owned_verification

@owned_verification('generic_fragments')
def verify_fragments_v2(manifest_path, *, expected_sha256, runtime):
    """Full contents and exact bound bytes; not producer-history verification."""
    try:
        return _verify(manifest_path, expected_sha256, runtime)
    except m.FragmentsV2Error:
        raise
    except reference.ScATACReferenceError:
        m.fail('FRAGMENTS_V2_REFERENCE_MISMATCH')
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, AttributeError):
        m.fail('FRAGMENTS_V2_VERIFICATION_FAILED')
