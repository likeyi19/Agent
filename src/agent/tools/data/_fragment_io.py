"""Source-neutral bounded BGZF decoding and functional tabix query IO."""
import hashlib
from pathlib import Path
import struct
import subprocess
import zlib
from ._fragments_common import MAX_LINE, fail

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
