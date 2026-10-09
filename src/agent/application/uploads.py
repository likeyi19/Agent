"""Completed browser bytes handed to UA1, without scientific admission or jobs.

The single lab-host deployment uses a trusted local filesystem. Files live under
one explicitly approved managed root; existing registration is the only catalog.
Uploads create no Session, scientific result, or execution instruction.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from threading import BoundedSemaphore

from .local_resources import LocalResourceAdmission, ResourceAdmissionError
from .workspace import ApplicationWorkspaceError


DEFAULT_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_FILES = 128
_UPLOAD_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
_STORAGE_NAME = re.compile(r'[0-9a-f]{64}\.h5ad')
_ATTRIBUTION = 'Browser-supplied H5AD; scientific provenance unverified.'
_MESSAGES = {
    'UPLOAD_REQUEST_INVALID': ('Send one file with a valid upload identity and filename.', 400),
    'UPLOAD_TOO_LARGE': ('The attachment exceeds the configured upload size limit.', 413),
    'UPLOAD_INTERRUPTED': ('The file transfer did not complete. Retry the same attachment.', 400),
    'UPLOAD_CONFLICT': ('This upload identity already belongs to different file bytes or metadata.', 409),
    'UPLOAD_BUSY': ('The bounded upload capacity is busy. Retry the same attachment.', 503),
    'UPLOAD_STORAGE_INVALID': ('The controlled upload location is unavailable or unsafe.', 500),
    'UPLOAD_STORAGE_LIMIT': ('The controlled upload location has reached its file limit.', 413),
}


class UploadError(ValueError):
    """Safe transfer failures, separate from resource and scientific failures."""

    def __init__(self, code):
        self.code = code
        self.message, self.status_code = _MESSAGES[code]
        super().__init__(self.message)


def _request(upload_id, filename):
    if (type(upload_id) is not str or _UPLOAD_ID.fullmatch(upload_id) is None
            or type(filename) is not str or not filename.strip() or len(filename) > 160
            or filename in {'.', '..'} or any(c in filename for c in '/\\:')
            or any(not c.isprintable() for c in filename)):
        raise UploadError('UPLOAD_REQUEST_INVALID')


class H5ADUploadAdmission:
    """Bounded streaming transfer and UA1 handoff, with exact retry identity.

    Source names are generated from the bounded upload identity, never filenames.
    A published source lacking registration after a crash stays undiscoverable;
    replaying identical bytes finishes the same existing registration operation.
    """

    def __init__(self, resources, root, *, max_bytes=DEFAULT_UPLOAD_BYTES, max_concurrent=2):
        if not isinstance(resources, LocalResourceAdmission):
            raise TypeError('The existing LocalResourceAdmission owner is required.')
        if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_UPLOAD_BYTES:
            raise ValueError('max_bytes must be between one byte and one GiB.')
        if type(max_concurrent) is not int or not 1 <= max_concurrent <= 4:
            raise ValueError('max_concurrent must be between one and four.')
        self.resources = resources
        self.max_bytes = max_bytes
        self.max_concurrent = max_concurrent
        self._slots = BoundedSemaphore(max_concurrent)
        self.root = Path(root).absolute()
        try:
            if (self.root != self.root.resolve(strict=True) or not self.root.is_dir()
                    or not any(self.root.is_relative_to(r) for r in resources._approved)):
                raise ValueError('Upload root must already be approved.')
            self._workspace = resources._workspace
            self.completed = self._workspace._ensure_directory(self.root / 'completed')
            self.staging = self._workspace._ensure_directory(self.root / 'staging')
            self._locks = self._workspace._ensure_directory(self.root / 'locks')
            self._check_directories()
        except (ValueError, OSError, RuntimeError) as exc:
            raise UploadError('UPLOAD_STORAGE_INVALID') from exc

    def _check_directories(self):
        for path in (self.root, self.completed, self.staging, self._locks):
            self._workspace._assert_contained(path)
            if path.is_symlink() or not path.is_dir() or path != path.resolve(strict=True):
                raise UploadError('UPLOAD_STORAGE_INVALID')

    @staticmethod
    def _sync_directory(path):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _lease(self, name):
        descriptor = os.open(self._locks / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise UploadError('UPLOAD_STORAGE_INVALID')
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise UploadError('UPLOAD_BUSY') from exc
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _stage(self, token):
        # A short shared admission lock includes in-flight staging files in the
        # bound. It creates no durable work record or resource reservation.
        lease = self._lease('admission.lock')
        try:
            count = 0
            for directory in (self.completed, self.staging):
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not entry.is_file(follow_symlinks=False):
                            raise UploadError('UPLOAD_STORAGE_INVALID')
                        count += 1
                        if count >= MAX_UPLOAD_FILES:
                            raise UploadError('UPLOAD_STORAGE_LIMIT')
            descriptor, name = tempfile.mkstemp(prefix=token + '-', suffix='.part', dir=self.staging)
            return os.fdopen(descriptor, 'wb'), Path(name)
        finally:
            os.close(lease)

    def _owned(self, record):
        path = Path(record.source_path)
        if (record.input_type != 'h5ad' or record.attribution != _ATTRIBUTION
                or path.parent != self.completed or _STORAGE_NAME.fullmatch(path.name) is None):
            raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
        return record

    def choices(self):
        """Only completed browser registrations, without reopening source bytes."""
        try:
            self._check_directories()
            return tuple(self._owned(record).public()
                         for record in self.resources.list_records(source_root=self.completed))
        except (ResourceAdmissionError, UploadError):
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise UploadError('UPLOAD_STORAGE_INVALID') from exc

    def resolve(self, resource_id, *, scientific_inputs=None,
                resource_selection_error=None, verify_source=True):
        self._owned(self.resources.load(resource_id))
        # The facade owns new-consumption versus completed-retry lifetime checks.
        # Metadata-only transport resolution must not reopen a historical source.
        if scientific_inputs is not None or resource_selection_error is not None:
            return self.resources.compose_h5ad(resource_id, scientific_inputs,
                resource_selection_error=resource_selection_error, verify_source=verify_source)
        return self.resources.resolve(resource_id, tool_name='inspect_scATAC', verify_source=verify_source)

    async def receive(self, upload_id, filename, chunks, *, expected_size=None):
        """Consume raw bytes incrementally, finalize durably, then register via UA1."""
        _request(upload_id, filename)
        if expected_size is not None:
            if type(expected_size) is not int or expected_size < 0:
                raise UploadError('UPLOAD_REQUEST_INVALID')
            if expected_size > self.max_bytes:
                raise UploadError('UPLOAD_TOO_LARGE')
        if not self._slots.acquire(blocking=False):
            raise UploadError('UPLOAD_BUSY')
        token = hashlib.sha256(b'agent.web-h5ad-upload.v1\0' + upload_id.encode()).hexdigest()
        destination = self.completed / (token + '.h5ad')
        registration_key = 'web-h5ad-upload:' + upload_id
        lease, stream, temporary = None, None, None
        try:
            self._check_directories()
            # Sixteen fixed stripes bound lock-file growth for arbitrary IDs.
            lease = self._lease('upload-' + token[0] + '.lock')
            try:
                accepted = self._owned(self.resources.load(self.resources.registration_id(registration_key)))
            except ResourceAdmissionError as exc:
                if exc.code != 'LOCAL_RESOURCE_UNAVAILABLE':
                    raise
                accepted = None
            if accepted is not None and (accepted.source_path != str(destination) or accepted.label != filename):
                raise UploadError('UPLOAD_CONFLICT')
            exists = destination.exists() or destination.is_symlink()
            if exists:
                if (destination.is_symlink() or not destination.is_file()
                        or destination != destination.resolve(strict=True)):
                    raise UploadError('UPLOAD_STORAGE_INVALID')
            else:
                stream, temporary = self._stage(token)
            digest, size = hashlib.sha256(), 0
            try:
                async for chunk in chunks:
                    if type(chunk) not in (bytes, bytearray):
                        raise UploadError('UPLOAD_REQUEST_INVALID')
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise UploadError('UPLOAD_TOO_LARGE')
                    view = memoryview(chunk)
                    for start in range(0, len(view), UPLOAD_CHUNK_BYTES):
                        bounded = view[start:start + UPLOAD_CHUNK_BYTES]
                        digest.update(bounded)
                        if stream is not None:
                            stream.write(bounded)
            except (UploadError, OSError):
                raise
            except Exception as exc:
                raise UploadError('UPLOAD_INTERRUPTED') from exc
            if size == 0 or expected_size is not None and size != expected_size:
                raise UploadError('UPLOAD_INTERRUPTED')
            if accepted is not None and (digest.hexdigest(), size) != (accepted.source_sha256, accepted.size_bytes):
                # A deleted registered source must not be repopulated with new
                # bytes merely because exclusive publication finds an empty path.
                raise UploadError('UPLOAD_CONFLICT')
            self._check_directories()
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                stream = None
                if not temporary.is_file() or temporary.is_symlink():
                    raise UploadError('UPLOAD_STORAGE_INVALID')
                # Exclusive hard-link publication is atomic and never replaces
                # existing bytes. Staging and final files share a filesystem.
                os.link(temporary, destination, follow_symlinks=False)
                temporary.unlink()
                temporary = None
                self._sync_directory(self.completed)
                self._sync_directory(self.staging)
            else:
                _, existing_sha, existing_size = self.resources._source(destination, integrity=True)
                if (digest.hexdigest(), size) != (existing_sha, existing_size):
                    raise UploadError('UPLOAD_CONFLICT')
            try:
                record = self.resources.register(registration_key, destination,
                    input_type='h5ad', label=filename, attribution=_ATTRIBUTION,
                    expected_source_sha256=digest.hexdigest(), expected_size_bytes=size)
            except ResourceAdmissionError as exc:
                if exc.code == 'LOCAL_RESOURCE_CONFLICT':
                    raise UploadError('UPLOAD_CONFLICT') from exc
                raise
            if record.source_sha256 != digest.hexdigest() or record.size_bytes != size:
                raise ResourceAdmissionError('LOCAL_RESOURCE_INTEGRITY_INVALID')
            return record
        except (UploadError, ResourceAdmissionError):
            raise
        except (OSError, ValueError, RuntimeError, ApplicationWorkspaceError) as exc:
            raise UploadError('UPLOAD_STORAGE_INVALID') from exc
        except Exception as exc:
            # Stream interruption carries no request payload or private paths.
            raise UploadError('UPLOAD_INTERRUPTED') from exc
        finally:
            try:
                if stream is not None:
                    stream.close()
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            finally:
                if lease is not None:
                    os.close(lease)
                self._slots.release()
