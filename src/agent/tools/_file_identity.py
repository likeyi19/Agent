"""Exact regular-file identities shared by scientific owners; no scientific IO."""
import hashlib
from pathlib import Path
import stat
from agent.schemas.verification_authority import AuthorityError
from ._cancellation import cancellation_checkpoint


def snapshot(path):
    value = Path(path).stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode): raise AuthorityError('Expected a regular scientific file')
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, stat.S_IFMT(value.st_mode))


def snapshots(paths):
    return {str(path): snapshot(path) for path in sorted(set(map(str, paths)))}


def unchanged(before):
    if snapshots(before) != before: raise AuthorityError('Scientific file changed during execution')


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            cancellation_checkpoint(); sha.update(chunk)
    return sha.hexdigest()


def file_closure(paths):
    paths = sorted(set(map(str, paths))); before = snapshots(paths); entries = []
    for name in paths:
        path = Path(name)
        if path != path.resolve(): raise AuthorityError('Noncanonical scientific file locator')
        entries.append(dict(path=name, sha256=file_sha256(path), size_bytes=before[name][2]))
    unchanged(before)
    return entries


def checked_file(path, expected):
    if len(expected) != 64 or file_sha256(path) != expected:
        raise ValueError(f'Content identity mismatch: {path}')
    return Path(path).resolve()


def identifier(value):
    if not isinstance(value, str) or not value or value.strip() != value or any(ord(c) < 32 for c in value):
        raise ValueError('Nonempty exact single-line identity required')
    return value


def read_json(path, expected=None, *, limit=65536):
    import json
    from agent.tools.data import scatac_matrix_contract as matrix
    path = Path(path)
    matrix.absolute_path(str(path))
    if path != path.resolve():
        raise ValueError('Scientific paths must be canonical absolute paths')
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Scientific metadata exceeds its declared bound')
    if expected is not None:
        matrix.sha(expected)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Scientific content identity mismatch')
    from agent.tools.data.scatac_fragments_v2 import _pairs
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
