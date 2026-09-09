"""Frozen M11.2c definitions and process boundaries; no scientific registry."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

from . import _chromap as chromap

ARTIFACT_TYPE = 'agent.scatac-fragments'
CONTRACT_VERSION = 'scatac-fragments.v1'
MAX_SUPPORT = 2**64 - 1
MAX_TOTAL = 2**128 - 1
TBI_LIMIT = 2**29
MAX_LINE = 1024 * 1024
SEMANTICS = {
    'coordinates': '0-based-half-open',
    'order': 'fai-rank,start-numeric,end-numeric,barcode-ascii.v1',
    'support': chromap.SUPPORT_SEMANTICS,
    'tn5': 'Chromap-only:+4-start,-5-end;no-second-shift',
    'distinct_barcodes': 'observed-accepted-fragment-tokens;not-called-cells',
    'individual_support_max': MAX_SUPPORT,
    'aggregate_support_max': MAX_TOTAL,
    'stream_identity': 'sha256(canonical-utf8-five-tab-fields-with-LF-in-order)',
}
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


@dataclass(frozen=True)
class FragmentsRuntime:
    chromap: str
    bgzip: str = '/usr/bin/bgzip'
    tabix: str = '/usr/bin/tabix'
    sort: str = '/usr/bin/sort'


def verify_packaging(runtime):
    for name, identity in TOOLCHAIN.items():
        path = Path(getattr(runtime, name))
        if (not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK)
                or chromap.sha256(path) != identity['sha256']):
            fail('FRAGMENTS_TOOLCHAIN_MISMATCH')
        r = subprocess.run([str(path), '--version'], capture_output=True, check=False)
        if r.returncode or r.stdout.decode().splitlines()[0] != identity['version']:
            fail('FRAGMENTS_TOOLCHAIN_MISMATCH')
        found = {}
        r = subprocess.run(['/usr/bin/ldd', str(path)], capture_output=True, check=False)
        if r.returncode:
            fail('FRAGMENTS_TOOLCHAIN_MISMATCH')
        for line in r.stdout.decode().splitlines():
            fields = line.split()
            target = fields[2] if '=>' in fields else fields[0]
            if target.startswith('/'):
                found[target] = chromap.sha256(target)
        if found != identity['libraries']:
            fail('FRAGMENTS_TOOLCHAIN_MISMATCH')


def run_stage(argv, *, cwd, code, output=None):
    """No shell/retry. Reap the child before propagating interruption."""
    env = dict(os.environ, LC_ALL='C', TMPDIR=str(cwd))
    with (Path(cwd) / 'process.log').open('ab') as log:
        with subprocess.Popen(list(map(str, argv)), cwd=cwd, env=env,
                stdout=output if output is not None else log, stderr=log) as process:
            try:
                status = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
    if status:
        fail(code)
