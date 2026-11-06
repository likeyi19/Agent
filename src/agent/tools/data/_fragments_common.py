"""Shared integer bounds and frozen packaging/process boundaries."""
import hashlib
import json
from pathlib import Path
import stat


MAX_SUPPORT = 2**64 - 1
MAX_TOTAL = 2**128 - 1
TBI_LIMIT = 2**29
MAX_LINE = 1024 * 1024
TOOLCHAIN = json.loads(Path(__file__).with_name('_fragments_toolchain.json').read_text())
PACKAGING_POLICY = {'bgzip': ['-c', '-l', '6', '-@', '1'], 'tabix': ['-p', 'bed'],
    'index_format': 'TBI', 'sort': ['--parallel=1', '-S', '64M', '-t', '\t',
    '-k1,1n', '-k2,2n', '-k3,3n', '-k4,4'], 'locale': 'C', 'tools': TOOLCHAIN}


class FragmentsError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code):
    raise FragmentsError(code) from None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def snapshot(path):
    p = Path(path)
    s = p.stat(follow_symlinks=False)
    if not stat.S_ISREG(s.st_mode):
        fail('FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION')
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, stat.S_IFMT(s.st_mode))


def snapshots(paths):
    return {str(p): snapshot(p) for p in sorted(set(map(str, paths)))}


def unchanged(before, code='FRAGMENTS_SOURCE_CHANGED_DURING_EXECUTION'):
    try:
        if snapshots(before) != before:
            fail(code)
    except OSError:
        fail(code)
