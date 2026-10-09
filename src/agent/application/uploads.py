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

from .local_resources import LocalResourceAdmission, LocalResourceCollectionRecord, ResourceAdmissionError
from .session_state import canonical
from .workspace import ApplicationWorkspaceError


DEFAULT_UPLOAD_BYTES = 256 * 1024 * 1024
MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_UPLOAD_FILES = 128
_UPLOAD_ID = re.compile(r'[A-Za-z0-9_-]{1,64}')
_STORAGE_NAME = re.compile(r'[0-9a-f]{64}\.h5ad')
_FRAGMENTS_NAME = re.compile(r'[0-9a-f]{64}\.fragments')
_BAM_NAME = re.compile(r'[0-9a-f]{64}\.bam')
_ATTRIBUTION = 'Browser-supplied H5AD; scientific provenance unverified.'
_FRAGMENTS_ATTRIBUTION = 'Browser-supplied external fragments; scientific provenance unverified.'
_BAM_ATTRIBUTION = 'Browser-supplied BAM; scientific provenance unverified.'
_FASTQ_ATTRIBUTION = 'Browser-supplied FASTQ member with explicit read attribution; scientific provenance unverified.'
_FASTQ_COLLECTION_ATTRIBUTION = 'Browser-completed explicit FASTQ library; scientific qualification unverified.'
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

    def __init__(self, code, *, max_bytes=None):
        self.code = code
        self.message, self.status_code = _MESSAGES[code]
        if code == 'UPLOAD_TOO_LARGE' and max_bytes is not None:
            self.message = (f'The configured per-file upload limit is {max_bytes} bytes. '
                f'Ask the operator to review upload_max_bytes (supported maximum {MAX_UPLOAD_BYTES} bytes).')
        super().__init__(self.message)


def _request(upload_id, filename):
    if (type(upload_id) is not str or _UPLOAD_ID.fullmatch(upload_id) is None
            or type(filename) is not str or not filename.strip() or len(filename) > 160
            or filename in {'.', '..'} or any(c in filename for c in '/\\:')
            or any(not c.isprintable() for c in filename)):
        raise UploadError('UPLOAD_REQUEST_INVALID')


class H5ADUploadAdmission:
    """Shared scientific-source transfer and existing resource handoff.

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
        if isinstance(record, LocalResourceCollectionRecord):
            if record.attribution != _FASTQ_COLLECTION_ATTRIBUTION:
                raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
            for member in record.members:
                source = self.resources.load(member['resource_id'])
                if isinstance(source, LocalResourceCollectionRecord):
                    raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
                self._owned(source)
                if source.input_type != 'fastq' or source.record_sha256 != member['record_sha256']:
                    raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
            return record
        path = Path(record.source_path)
        if record.input_type == 'h5ad':
            valid = record.attribution == _ATTRIBUTION and _STORAGE_NAME.fullmatch(path.name)
        elif record.input_type == 'external_fragments':
            valid = (record.attribution == _FRAGMENTS_ATTRIBUTION
                     and _FRAGMENTS_NAME.fullmatch(path.name))
        elif record.input_type == 'bam':
            valid = record.attribution == _BAM_ATTRIBUTION and _BAM_NAME.fullmatch(path.name)
        elif record.input_type == 'fastq':
            valid = (record.attribution == _FASTQ_ATTRIBUTION and record.fastq_member is not None
                     and path.name == self._fastq_name(record.fastq_member))
        else:
            valid = False
        if not valid or path.parent != self.completed:
            raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
        if record.source_index is not None and (record.input_type != 'external_fragments'
                or record.source_index['path'] != str(path.with_suffix('.tbi'))):
            raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
        return record

    def choices(self):
        """Only completed browser registrations, without reopening source bytes."""
        try:
            self._check_directories()
            return tuple(self._owned(record).public()
                         for record in self.resources.list_records(source_root=self.completed, listing='choices'))
        except (ResourceAdmissionError, UploadError):
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            raise UploadError('UPLOAD_STORAGE_INVALID') from exc

    def fastq_members(self):
        """Completed bytes with explicit attribution, not selectable libraries."""
        self._check_directories()
        return tuple(self._owned(record).public()
                     for record in self.resources.list_records(source_root=self.completed, listing='fastq_members'))

    @staticmethod
    def _fastq_name(member):
        # A mechanical adapter to the existing owner's syntactic filename
        # contract. No original filename or sequencing observation supplies roles.
        library = hashlib.sha256(b'agent.web-fastq-library.v1\0' +
            canonical(dict(library_id=member['library_id'], fastq_layout=member['fastq_layout']))).hexdigest()
        suffix = '.fastq.gz' if member['compression'] == 'gzip' else '.fastq'
        return f"FQ{library}_S1_L{member['lane']}_{member['role']}_{member['chunk']}{suffix}"

    def complete_fastq_collection(self, collection_id, label, member_ids):
        """Publish one complete selection through the existing source catalog."""
        _request(collection_id, label)
        if type(member_ids) not in (list, tuple) or not member_ids:
            raise ResourceAdmissionError('FASTQ_COLLECTION_INVALID')
        for resource_id in member_ids:
            record = self._owned(self.resources.load(resource_id))
            if isinstance(record, LocalResourceCollectionRecord) or record.input_type != 'fastq':
                raise ResourceAdmissionError('FASTQ_COLLECTION_INVALID')
        return self.resources.register_fastq_collection('web-fastq-collection:' + collection_id,
            member_ids, label=label, attribution=_FASTQ_COLLECTION_ATTRIBUTION)

    def resolve(self, resource_id, *, scientific_inputs=None,
                resource_selection_error=None, verify_source=True):
        record = self._owned(self.resources.load(resource_id))
        # The facade owns new-consumption versus completed-retry lifetime checks.
        # Metadata-only transport resolution must not reopen a historical source.
        if isinstance(record, LocalResourceCollectionRecord):
            return self.resources.compose_fastq_collection(resource_id, scientific_inputs,
                resource_selection_error=resource_selection_error, verify_source=verify_source)
        if record.input_type == 'fastq':
            raise ResourceAdmissionError('FASTQ_COLLECTION_REQUIRED')
        if record.input_type == 'external_fragments':
            if resource_selection_error is not None:
                raise ResourceAdmissionError('LOCAL_RESOURCE_BINDING_INVALID')
            return self.resources.compose_fragments(resource_id, scientific_inputs,
                                                   verify_source=verify_source)
        if record.input_type == 'bam':
            return self.resources.compose_bam(resource_id, scientific_inputs,
                resource_selection_error=resource_selection_error, verify_source=verify_source)
        if scientific_inputs is not None or resource_selection_error is not None:
            return self.resources.compose_h5ad(resource_id, scientific_inputs,
                resource_selection_error=resource_selection_error, verify_source=verify_source)
        return self.resources.resolve(resource_id, tool_name='inspect_scATAC', verify_source=verify_source)

    async def receive(self, upload_id, filename, chunks, *, expected_size=None,
                      input_type='h5ad', index_filename=None, source_size=None, fastq_member=None):
        """Receive one source, optionally followed by an explicitly paired index.

        source_size frames the pair in one raw transfer. Both components finish
        before a single source/index record becomes discoverable. Scientific
        format and index compatibility remain the importer's responsibility.
        """
        _request(upload_id, filename)
        paired = index_filename is not None or source_size is not None
        if input_type not in ('h5ad', 'external_fragments', 'bam', 'fastq'):
            raise UploadError('UPLOAD_REQUEST_INVALID')
        if input_type == 'fastq':
            fastq_member = self.resources.fastq_member_declaration(fastq_member)
        elif fastq_member is not None:
            raise UploadError('UPLOAD_REQUEST_INVALID')
        if paired:
            if (input_type != 'external_fragments' or index_filename is None
                    or type(source_size) is not int or source_size <= 0):
                raise UploadError('UPLOAD_REQUEST_INVALID')
            _request(upload_id, index_filename)
            if source_size > self.max_bytes:
                raise UploadError('UPLOAD_TOO_LARGE', max_bytes=self.max_bytes)
        total_limit = self.max_bytes * (2 if paired else 1)
        if expected_size is not None:
            if type(expected_size) is not int or expected_size < 0:
                raise UploadError('UPLOAD_REQUEST_INVALID')
            if expected_size > total_limit:
                raise UploadError('UPLOAD_TOO_LARGE', max_bytes=self.max_bytes)
        if not self._slots.acquire(blocking=False):
            raise UploadError('UPLOAD_BUSY')
        if input_type == 'fastq':
            slot = {key: value for key, value in fastq_member.items() if key != 'compression'}
            token = hashlib.sha256(b'agent.web-fastq-member.v1\0' + canonical(slot)).hexdigest()
            destination = self.completed / self._fastq_name(fastq_member)
            registration_key = 'web-fastq-upload:' + upload_id
            attribution = _FASTQ_ATTRIBUTION
        else:
            domain, suffix, prefix, attribution = {
                'h5ad': (b'agent.web-h5ad-upload.v1\0', '.h5ad', 'web-h5ad-upload:', _ATTRIBUTION),
                'external_fragments': (b'agent.web-fragments-upload.v1\0', '.fragments',
                                       'web-fragments-upload:', _FRAGMENTS_ATTRIBUTION),
                'bam': (b'agent.web-bam-upload.v1\0', '.bam', 'web-bam-upload:', _BAM_ATTRIBUTION),
            }[input_type]
            token = hashlib.sha256(domain + upload_id.encode()).hexdigest()
            destination = self.completed / (token + suffix)
            registration_key = prefix + upload_id
        components = [{'path': destination, 'stream': None, 'temporary': None,
                       'digest': hashlib.sha256(), 'size': 0}]
        if paired:
            components.append({'path': destination.with_suffix('.tbi'), 'stream': None,
                               'temporary': None, 'digest': hashlib.sha256(), 'size': 0})
        lease = None
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
            if accepted is not None and (accepted.source_path != str(destination)
                    or accepted.label != filename or (accepted.source_index is not None) != paired
                    or accepted.fastq_member != fastq_member):
                raise UploadError('UPLOAD_CONFLICT')
            if fastq_member is not None:
                # Role slots are immutable across transfer IDs too, including a
                # changed compression declaration that would choose another path.
                for record in self.resources.list_records(source_root=self.completed, listing='fastq_members'):
                    self._owned(record)
                    occupied = {key: value for key, value in record.fastq_member.items() if key != 'compression'}
                    if occupied == slot and (accepted is None or record.resource_id != accepted.resource_id):
                        raise UploadError('UPLOAD_CONFLICT')
            for component in components:
                path = component['path']
                if path.exists() or path.is_symlink():
                    if (path.is_symlink() or not path.is_file()
                            or path != path.resolve(strict=True)):
                        raise UploadError('UPLOAD_STORAGE_INVALID')
                else:
                    component['stream'], component['temporary'] = self._stage(token)
            total = 0
            try:
                async for chunk in chunks:
                    if type(chunk) not in (bytes, bytearray):
                        raise UploadError('UPLOAD_REQUEST_INVALID')
                    total += len(chunk)
                    if total > total_limit:
                        raise UploadError('UPLOAD_TOO_LARGE', max_bytes=self.max_bytes)
                    view = memoryview(chunk)
                    offset = 0
                    while offset < len(view):
                        component = components[0]
                        remaining = len(view) - offset
                        if paired:
                            if component['size'] == source_size:
                                component = components[1]
                            else:
                                remaining = min(remaining, source_size - component['size'])
                        count = min(remaining, UPLOAD_CHUNK_BYTES)
                        bounded = view[offset:offset + count]
                        component['size'] += count
                        if component['size'] > self.max_bytes:
                            raise UploadError('UPLOAD_TOO_LARGE', max_bytes=self.max_bytes)
                        component['digest'].update(bounded)
                        if component['stream'] is not None:
                            component['stream'].write(bounded)
                        offset += count
            except (UploadError, OSError):
                raise
            except Exception as exc:
                raise UploadError('UPLOAD_INTERRUPTED') from exc
            if (any(component['size'] == 0 for component in components)
                    or paired and components[0]['size'] != source_size
                    or expected_size is not None and total != expected_size):
                raise UploadError('UPLOAD_INTERRUPTED')
            if accepted is not None:
                identities = [(accepted.source_sha256, accepted.size_bytes)]
                if paired:
                    identities.append((accepted.source_index['sha256'], accepted.source_index['size_bytes']))
                for component, identity in zip(components, identities):
                    if (component['digest'].hexdigest(), component['size']) != identity:
                        raise UploadError('UPLOAD_CONFLICT')
            self._check_directories()
            for component in components:
                stream, temporary = component['stream'], component['temporary']
                if stream is not None:
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    component['stream'] = None
                    if not temporary.is_file() or temporary.is_symlink():
                        raise UploadError('UPLOAD_STORAGE_INVALID')
                else:
                    _, sha, size = self.resources._source(component['path'], integrity=True)
                    if (component['digest'].hexdigest(), component['size']) != (sha, size):
                        raise UploadError('UPLOAD_CONFLICT')
            published = []
            try:
                for component in components:
                    temporary = component['temporary']
                    if temporary is not None:
                        # Never overwrite an existing source or companion.
                        os.link(temporary, component['path'], follow_symlinks=False)
                        published.append(component['path'])
                        temporary.unlink()
                        component['temporary'] = None
            except OSError:
                for path in published:
                    path.unlink(missing_ok=True)
                raise
            if published:
                self._sync_directory(self.completed)
                self._sync_directory(self.staging)
            source = components[0]
            index_arguments = {}
            if paired:
                index = components[1]
                index_arguments = {'source_index_path': index['path'],
                    'expected_source_index_sha256': index['digest'].hexdigest(),
                    'expected_source_index_size_bytes': index['size']}
            try:
                record = self.resources.register(registration_key, destination,
                    input_type=input_type, label=filename,
                    attribution=attribution,
                    expected_source_sha256=source['digest'].hexdigest(),
                    expected_size_bytes=source['size'], fastq_member=fastq_member, **index_arguments)
            except ResourceAdmissionError as exc:
                if exc.code == 'LOCAL_RESOURCE_CONFLICT':
                    raise UploadError('UPLOAD_CONFLICT') from exc
                raise
            if record.source_sha256 != source['digest'].hexdigest() or record.size_bytes != source['size']:
                raise ResourceAdmissionError('LOCAL_RESOURCE_INTEGRITY_INVALID')
            if paired and (record.source_index is None or
                    (record.source_index['sha256'], record.source_index['size_bytes']) !=
                    (index['digest'].hexdigest(), index['size'])):
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
                for component in components:
                    if component['stream'] is not None:
                        component['stream'].close()
                    if component['temporary'] is not None:
                        component['temporary'].unlink(missing_ok=True)
            finally:
                if lease is not None:
                    os.close(lease)
                self._slots.release()
