"""Local atomic session snapshots, separate from scientific FileRunStore records."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from .session_state import AnalysisSession, SessionError, SessionConflictError, canonical, digest, text

MAX_SESSION_BYTES = 16 * 1024 * 1024


class FileSessionStore:
    """Trusted-local storage; no database, artifact discovery, or scientific verification."""

    def __init__(self, root):
        self.root = Path(root)
        self._check_root()

    def _check_root(self):
        if self.root.is_symlink() or not self.root.is_dir() or self.root != self.root.resolve():
            raise SessionError('Invalid session directory.')

    def _path(self, session_id, suffix):
        self._check_root()
        text(session_id)
        path = self.root / (hashlib.sha256(session_id.encode()).hexdigest() + suffix)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise SessionError('Invalid session file type.')
        return path

    @contextmanager
    def _lock(self, session_id):
        path = self._path(session_id, '.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise SessionError('Invalid session lock.')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def create(self, session_id):
        state = AnalysisSession(session_id)
        with self._lock(session_id):
            if self._path(session_id, '.json').exists():
                raise SessionConflictError('Session already exists.')
            self._write(state)
        return state

    def load(self, session_id):
        with self._lock(session_id):
            return self._load(session_id)

    def _load(self, session_id):
        def pairs(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise SessionError('Duplicate session JSON key.')
                value[key] = item
            return value
        try:
            with self._path(session_id, '.json').open('rb') as stream:
                raw = stream.read(MAX_SESSION_BYTES + 1)
            if len(raw) > MAX_SESSION_BYTES:
                raise SessionError('Session storage limit exceeded.')
            envelope = json.loads(raw, object_pairs_hook=pairs)
            if (type(envelope) is not dict or set(envelope) != {'format', 'record', 'sha256'}
                    or envelope['format'] != 'agent.analysis-session.v1'
                    or envelope['sha256'] != digest(envelope['record'])):
                raise SessionError('Invalid session checksum/envelope.')
            state = AnalysisSession.from_dict(envelope['record'])
            if state.session_id != session_id:
                raise SessionError('Session identity mismatch.')
            return state
        except SessionError:
            raise
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise SessionError('Session unavailable or corrupt.') from exc

    def _update(self, session_id, change):
        """Serialize metadata changes; caller performs generation CAS inside change."""
        with self._lock(session_id):
            before = self._load(session_id)
            after = change(before)
            if not isinstance(after, AnalysisSession) or after.session_id != before.session_id:
                raise SessionError('Invalid session update.')
            if (after.revisions[:len(before.revisions)] != before.revisions
                    or after.navigation[:len(before.navigation)] != before.navigation
                    or len(after.turns) < len(before.turns)):
                raise SessionConflictError('Historical session records are immutable.')
            for old, new in zip(before.turns, after.turns):
                if (old.turn_id, old.base_revision_id, old.base_generation, old.request_id,
                    old.request_sha256, old.selections, old.retained_outputs) != (
                    new.turn_id, new.base_revision_id, new.base_generation, new.request_id,
                    new.request_sha256, new.selections, new.retained_outputs):
                    raise SessionConflictError('Turn identity cannot change.')
                if old.run_id is not None and old.run_id != new.run_id:
                    raise SessionConflictError('Associated run cannot change.')
                if old.status in {'activated', 'stale', 'failed', 'cancelled', 'planned', 'clarification'} and old != new:
                    raise SessionConflictError('Completed turn is immutable.')
                if old.completion_files and old != new and (
                        old.completion_files != new.completion_files
                        or old.run_result_sha256 != new.run_result_sha256):
                    raise SessionConflictError('Application completion cannot change.')
            if after != before:
                self._write(after)
            return after

    def _write(self, state):
        # Round-trip validation before publication prevents unreadable successful writes.
        record = json.loads(canonical(state.to_dict()))
        AnalysisSession.from_dict(record)
        payload = canonical({'format': 'agent.analysis-session.v1', 'record': record,
                             'sha256': digest(record)})
        if len(payload) > MAX_SESSION_BYTES:
            raise SessionError('Session storage limit exceeded.')
        path = self._path(state.session_id, '.json')
        fd, name = tempfile.mkstemp(dir=self.root, prefix='.' + path.stem, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            Path(name).unlink(missing_ok=True)
