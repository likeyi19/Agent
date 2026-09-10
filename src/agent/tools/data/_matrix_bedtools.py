"""M11.5a bounded BEDTools qualification harness, not a matrix constructor.

At most 10,000 records per projection; caller explicitly supplies the executable.
Production-scale staging/aggregation and independent reconstruction are M11.5b.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .scatac_matrix_contract import canonical, fail, integer, interval, overlaps, ScATACMatrixError
from agent.tools._cancellation import cancellation_checkpoint

PROFILE_ID = 'bedtools-ccre-record-incidence.v1'
MAX_CONTIG_LENGTH = 1_000_000_000
MAX_ROWS = 10_000
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
MAX_INPUT_BYTES = 4 * 1024 * 1024
MAX_SECONDS = 30
ENVIRONMENT = {'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            cancellation_checkpoint()
            digest.update(block)
    return digest.hexdigest()


def qualify_runtime(executable):
    """Check pinned binary and actual linked libraries before scientific use."""
    try:
        pin = json.loads(Path(__file__).with_name('_matrix_bedtools_toolchain.json').read_text())
        path = Path(executable)
        if (pin['profile'] != PROFILE_ID or pin['max_contig_length'] != MAX_CONTIG_LENGTH
                or pin['overlap_fraction'] != '1e-12' or not path.is_absolute()
                or not path.is_file() or not os.access(path, os.X_OK)
                or file_sha(path) != pin['sha256']):
            fail('MATRIX_BACKEND_UNQUALIFIED')
        before = path.stat()
        version = subprocess.run([str(path), '--version'], env=ENVIRONMENT,
                                 capture_output=True, timeout=15, check=True)
        if version.stdout.decode().strip() != pin['version']:
            fail('MATRIX_BACKEND_UNQUALIFIED')
        linked = subprocess.run(['/usr/bin/ldd', str(path)], env=ENVIRONMENT,
                                capture_output=True, timeout=15, check=True)
        found = {}
        for line in linked.stdout.decode().splitlines():
            words = line.split()
            if not words:
                continue
            target = words[2] if '=>' in words else words[0]
            if target.startswith('/'):
                resolved = str(Path(target).resolve())
                found[resolved] = file_sha(resolved)
        after = path.stat()
        snapshot = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (found != pin['libraries'] or file_sha(path) != pin['sha256']
                or snapshot(before) != snapshot(after)):
            fail('MATRIX_BACKEND_UNQUALIFIED')
        return dict(profile_id=PROFILE_ID, runtime_sha256=hashlib.sha256(canonical(pin)).hexdigest())
    except (OSError, subprocess.SubprocessError, UnicodeError, KeyError, IndexError, TypeError,
            json.JSONDecodeError) as exc:
        raise ScATACMatrixError('MATRIX_BACKEND_UNQUALIFIED') from exc


def genome_order_bytes(contigs):
    """Two-column BEDTools genome format from an already verified FAI dictionary.

    Never pass the five-column FAI verbatim. No inferred chromosome ordering.
    """
    if not 0 < len(contigs) <= MAX_ROWS:
        fail('MATRIX_BACKEND_LIMIT')
    seen = set()
    output = []; size = 0
    for chrom, length in contigs:
        interval(chrom, 0, length)
        integer(length, 1, MAX_CONTIG_LENGTH)
        if chrom in seen:
            fail('MATRIX_BACKEND_INPUT_INVALID')
        seen.add(chrom)
        line = f'{chrom}\t{length}\n'.encode('utf-8')
        size += len(line)
        if len(line) > 65536 or size > MAX_INPUT_BYTES:
            fail('MATRIX_BACKEND_LIMIT')
        output.append(line)
    return b''.join(output)


def intersection_argv(executable, a, b, genome):
    return (str(executable), 'intersect', '-sorted', '-g', str(genome),
            '-a', str(a), '-b', str(b), '-wa', '-wb', '-f', '1e-12', '-F', '1e-12')


def _projection(rows, contigs):
    if len(rows) > MAX_ROWS:
        fail('MATRIX_BACKEND_LIMIT')
    rank = {c: i for i, (c, _) in enumerate(contigs)}
    bounds = dict(contigs)
    identities = {}
    for row in rows:
        if len(row) != 4:
            fail('MATRIX_BACKEND_INPUT_INVALID')
        chrom, start, end, identity = row
        interval(chrom, start, end); integer(identity)
        if chrom not in bounds or end > bounds[chrom] or identity in identities:
            fail('MATRIX_BACKEND_INPUT_INVALID')
        identities[identity] = (chrom, start, end)
    # Only temporary projections sort. IDs retain authoritative source positions.
    ordered = sorted(rows, key=lambda r: (rank[r[0]], r[1], r[2], r[3]))
    output = []; size = 0
    for row in ordered:
        line = ('\t'.join(map(str, row)) + '\n').encode('utf-8')
        size += len(line)
        if len(line) > 65536 or size > MAX_INPUT_BYTES:
            fail('MATRIX_BACKEND_LIMIT')
        output.append(line)
    return b''.join(output), identities


def validate_incidences(lines, fragments, features):
    """Bounded qualification parser: exact returned records, no duplicate pairs."""
    seen = set()
    for line in lines:
        if len(line) > 65536 or not line.endswith(b'\n'):
            fail('MATRIX_INCIDENCE_INVALID')
        try:
            fields = line[:-1].decode('utf-8').split('\t')
            if len(fields) != 8:
                fail('MATRIX_INCIDENCE_INVALID')
            fid, cid = int(fields[3]), int(fields[7])
            a, b = fragments[fid], features[cid]
            if (fields != [*map(str, a), str(fid), *map(str, b), str(cid)]
                    or not overlaps(a, b)):
                fail('MATRIX_INCIDENCE_INVALID')
        except (UnicodeError, ValueError, KeyError) as exc:
            fail('MATRIX_INCIDENCE_INVALID')
        if (fid, cid) in seen:
            fail('MATRIX_DUPLICATE_INCIDENCE')
        seen.add((fid, cid))
        if len(seen) > MAX_OUTPUT_BYTES // 8:
            fail('MATRIX_BACKEND_LIMIT')
        yield fid, cid


def _run(argv, output, errors):
    cancellation_checkpoint()
    with output.open('xb') as out, errors.open('xb') as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, env=ENVIRONMENT)
        deadline = time.monotonic() + MAX_SECONDS
        try:
            while True:
                cancellation_checkpoint()
                if time.monotonic() > deadline:
                    fail('MATRIX_BACKEND_TIMEOUT')
                if output.stat().st_size > MAX_OUTPUT_BYTES or errors.stat().st_size > MAX_STDERR_BYTES:
                    fail('MATRIX_BACKEND_LIMIT')
                try:
                    status = process.wait(timeout=0.05)
                    break
                except subprocess.TimeoutExpired:
                    pass
            if output.stat().st_size > MAX_OUTPUT_BYTES or errors.stat().st_size > MAX_STDERR_BYTES:
                fail('MATRIX_BACKEND_LIMIT')
            if status:
                fail('MATRIX_INTERSECTION_FAILED')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
    cancellation_checkpoint()


def qualification_intersections(executable, contigs, fragments, features, *, scratch_parent):
    """Tiny bounded mechanics probe. Returned IDs are not a scientific matrix."""
    qualify_runtime(executable)
    genome = genome_order_bytes(contigs)
    a, fragment_ids = _projection(fragments, contigs)
    b, feature_ids = _projection(features, contigs)
    with tempfile.TemporaryDirectory(prefix='.matrix-qualification-', dir=scratch_parent) as directory:
        root = Path(directory)
        for name, data in (('a.bed', a), ('b.bed', b), ('genome.tsv', genome)):
            (root / name).write_bytes(data)
        out, err = root / 'out.tsv', root / 'stderr.txt'
        _run(intersection_argv(executable, root / 'a.bed', root / 'b.bed', root / 'genome.tsv'), out, err)
        with out.open('rb') as f:
            result = tuple(validate_incidences(iter(lambda: f.readline(65537), b''), fragment_ids, feature_ids))
        qualify_runtime(executable)
        return result
