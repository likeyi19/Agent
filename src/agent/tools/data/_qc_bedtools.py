"""Qualified narrow BEDTools argv/runtime contract, never a planner tool."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .scatac_qc_profile import fail

BACKEND_PROFILE = 'bedtools-qc-point-incidence.v1'
ENVIRONMENT = {'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_qc_bedtools(executable):
    """Explicit runtime path; qualify bytes before executing version or ldd."""
    try:
        identity = json.loads(Path(__file__).with_name('_qc_bedtools_toolchain.json').read_text())
        path = Path(executable)
        if (identity['profile'] != BACKEND_PROFILE
                or not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)
                or _sha(path) != identity['sha256']):
            fail('QC_BEDTOOLS_MISMATCH')
        before = path.stat()
        version = subprocess.run([str(path), '--version'], env=ENVIRONMENT,
                                 capture_output=True, timeout=15, check=False)
        if version.returncode or version.stdout.decode().strip() != identity['version']:
            fail('QC_BEDTOOLS_MISMATCH')
        linked = subprocess.run(['/usr/bin/ldd', str(path)], env=ENVIRONMENT,
                                capture_output=True, timeout=15, check=False)
        if linked.returncode:
            fail('QC_BEDTOOLS_MISMATCH')
        found = {}
        for line in linked.stdout.decode().splitlines():
            words = line.split()
            if not words:
                continue
            target = words[2] if '=>' in words else words[0]
            if target.startswith('/'):
                resolved = str(Path(target).resolve())
                found[resolved] = _sha(resolved)
        after = path.stat()
        if (found != identity['libraries'] or _sha(path) != identity['sha256']
                or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            fail('QC_BEDTOOLS_MISMATCH')
        return identity
    except (OSError, ValueError, TypeError, IndexError, subprocess.SubprocessError):
        fail('QC_BEDTOOLS_MISMATCH')


def qc_intersection_argv(executable, *, insertions_path, windows_path, genome_path):
    """Future execution adapter must validate resources and stream stdout.

    Both BED inputs must follow FAI-rank/start order. Endpoint projection must
    be sorted independently: sorted fragments do not imply sorted endpoints.
    No strand matching, merging, support weights or -u overlap collapse.
    """
    verify_qc_bedtools(executable)
    for path in (insertions_path, windows_path, genome_path):
        if not Path(path).is_absolute():
            fail('QC_BEDTOOLS_INPUT_INVALID')
    return (str(executable), 'intersect', '-sorted', '-g', str(genome_path),
            '-a', str(insertions_path), '-b', str(windows_path), '-wa', '-wb')
