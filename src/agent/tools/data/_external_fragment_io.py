"""Shared physical IO only; production and verification own separate row parsers."""
import gzip
import hashlib
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib

from . import external_fragment_manifest as m, scatac_reference as ref
from ._fragments_common import TBI_LIMIT, run_stage, verify_packaging
from .scatac_fragments_verifier import _bgzf_lines, _query
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


def reference(path, expected_sha256):
    _, bundle, _ = ref.load_scatac_reference_bundle(path, expected_sha256=expected_sha256)
    ref.reinspect_scatac_reference_bundle_sources(bundle)
    dictionary, _, _ = ref._inspect_fai(Path(bundle.genome.fai.path), Path(bundle.genome.fasta.path).stat().st_size)
    if any(n > TBI_LIMIT for n in dictionary.values()):
        m.fail('EXTERNAL_FRAGMENTS_INDEX_POLICY_UNSUPPORTED')
    identity = dict(manifest_path=str(path), manifest_sha256=expected_sha256,
        reference_identity_sha256=bundle.reference_identity_sha256, species=bundle.species,
        assembly=bundle.target_assembly, ordered_contig_sha256=bundle.genome.ordered_contig_sha256)
    paths = [path, bundle.genome.fasta.path, bundle.genome.fai.path, bundle.ccre.bed.path]
    if bundle.annotation:
        paths.append(bundle.annotation.resource.path)
    return identity, tuple(dictionary.items()), paths


def encoding(path):
    """Inspect gzip extra fields, never the filename or producer metadata."""
    with Path(path).open('rb') as source:
        head = source.read(12)
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
            opener = gzip.open if kind == 'gzip' else open
            with opener(path, 'rb') as source:
                yield from iter(lambda: source.readline(m.MAX_LINE + 1), b'')
    except (OSError, EOFError, zlib.error, ValueError) as exc:
        if isinstance(exc, m.ExternalFragmentsError):
            raise
        m.fail('EXTERNAL_FRAGMENTS_ENCODING_INVALID')


def check_source_index(source, index, kind, contigs, directory, runtime):
    """Bind explicitly supplied TBI, never implicitly consult a source sidecar.

    The temporary symlinks select exactly these two snapshotted resources. Full
    source-contig and narrow overlap digests are compared to tabix query results.
    """
    if index is None:
        return
    if kind != 'bgzf':
        m.fail('EXTERNAL_FRAGMENTS_INDEX_BINDING_INVALID')
    try:
        with gzip.open(index, 'rb') as stream:
            header = stream.read(36)
            if len(header) != 36 or header[:4] != b'TBI\1':
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
            _, fmt, seq, beg, end, meta, skip, _ = struct.unpack('<8i', header[4:])
            if (fmt, seq, beg, end, meta, skip) != (65536, 1, 2, 3, 35, 0):
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
            while stream.read(65536):
                pass
        full = {}; probes = {}
        for line in lines(source, kind):
            if line.startswith(b'#'):
                continue
            fields = line[:-1].decode().split('\t')
            name = fields[0]
            full.setdefault(name, hashlib.sha256()).update(line)
            probes.setdefault(name, (int(fields[1]), int(fields[2])))
        narrow = {n: hashlib.sha256() for n in full}
        for line in lines(source, kind):
            if line.startswith(b'#'):
                continue
            fields = line[:-1].decode().split('\t'); left, right = probes[fields[0]]
            if int(fields[1]) < right and int(fields[2]) > left:
                narrow[fields[0]].update(line)
        with tempfile.TemporaryDirectory(prefix='source-index-', dir=directory) as temporary:
            target = Path(temporary) / 'source.bgz'
            target.symlink_to(source)
            Path(str(target) + '.tbi').symlink_to(index)
            listed = subprocess.run([runtime.tabix, '-l', str(target)], capture_output=True, check=False)
            if listed.returncode or listed.stdout.decode().splitlines() != list(full):
                m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
            for name in full:
                start, end = probes[name]
                if (_query(runtime, target, name) != full[name].hexdigest()
                        or _query(runtime, target, f'{name}:{start+1}-{end}') != narrow[name].hexdigest()):
                    m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')
    except m.ExternalFragmentsError:
        raise
    except (ValueError, OSError, EOFError, struct.error, UnicodeError):
        m.fail('EXTERNAL_FRAGMENTS_INDEX_MISMATCH')


def sort_ranked(source, target, directory, runtime):
    """Mechanical external sort shared as packaging, not a scientific parser."""
    run_stage([runtime.sort, '--parallel=1', '-S', '64M', '-t', '\t', '-k1,1n',
        '-k2,2n', '-k3,3n', '-k4,4', '-k5,5', '-T', directory, '-o', target, '--', source],
        cwd=directory, code='EXTERNAL_FRAGMENTS_SORT_FAILED')
