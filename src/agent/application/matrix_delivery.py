"""Shared accepted-file snapshots and the closed original matrix H5AD policy."""

import json
import re
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from threading import Lock

import h5py
import numpy as np


class MatrixPrivacyError(ValueError):
    """The original matrix is outside the reviewed delivery layout."""

    def __init__(self):
        super().__init__('The accepted matrix is not eligible for original-file delivery.')


_CREATED_CONTRACTS = {
    'scatac-cell-by-ccre.v1',
    'scatac-cell-by-features.v1',
    'scatac-cell-by-features.qc-selected.v1',
}
_ADOPTED_CONTRACTS = {
    'scatac-cell-by-ccre.external.v1',
    'scatac-cell-by-features.external.v1',
}
_TOP_LEVEL = {'X', 'obs', 'var', 'uns', 'layers', 'obsm', 'obsp', 'varm', 'varp'}
_CREATED_UNS = {
    'matrix_semantics', 'matrix_profile_id', 'matrix_profile_sha256',
    'species', 'assembly', 'coordinate_system', 'ordered_selected_sha256',
    'ordered_feature_sha256', 'logical_matrix_sha256',
}
_ADOPTED_UNS = {'species', 'assembly', 'matrix_semantics', 'reference_identity_sha256'}
# Matrix writers consume BED lines bounded to 65,536 bytes. Adopted cell IDs
# are bounded to 4,096 bytes, and species/assembly declarations to 4,096 chars.
# A fixed-width HDF5 read prevents a variable-length string from allocating its
# entire unreviewed payload. The extra byte distinguishes excessive lengths.
_STRING_BYTES = 65_536
_STRING_CHUNK = 16
_PATH_TEXT = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')


def _reject(condition):
    if condition:
        raise MatrixPrivacyError()


def _attrs(item, expected):
    _reject(set(item.attrs) != set(expected))
    for name, value in expected.items():
        actual = item.attrs[name]
        if isinstance(value, (tuple, list)):
            _reject(np.asarray(actual).ndim != 1 or list(actual) != list(value))
        else:
            _reject(not isinstance(actual, str) or actual != value)


def _strings(dataset):
    """Read reviewed identifiers in bounded fixed-width pieces, never X."""
    _reject(h5py.check_string_dtype(dataset.dtype) is None)
    reader = dataset.astype(f'S{_STRING_BYTES + 1}')
    slices = ((),) if dataset.shape == () else (
        slice(left, left + _STRING_CHUNK) for left in range(0, len(dataset), _STRING_CHUNK)
    )
    for selection in slices:
        values = [reader[selection]] if selection == () else reader[selection]
        for value in values:
            raw = bytes(value)
            _reject(len(raw) > _STRING_BYTES)
            text = raw.decode('utf-8', errors='strict')
            _reject(not text or any(ord(c) < 32 or ord(c) == 127 for c in text)
                    or _PATH_TEXT.search(text) is not None)
            yield text


def _safe_objects(file):
    """Visit closed shallow objects without following indirect links."""
    seen = {h5py.h5o.get_info(file.id).addr}
    count = 0

    def visit(group, depth):
        nonlocal count
        _reject(depth > 2)
        for name in group:
            count += 1
            _reject(count > 64 or not isinstance(group.get(name, getlink=True), h5py.HardLink))
            item = group[name]
            address = h5py.h5o.get_info(item.id).addr
            _reject(address in seen)
            seen.add(address)
            if isinstance(item, h5py.Group):
                visit(item, depth + 1)
            else:
                _reject(not isinstance(item, h5py.Dataset) or item.is_virtual or bool(item.external))
                # Both publication writers use ordinary contiguous datasets or
                # gzip-compressed CSR arrays, never third-party filter plugins.
                creation = item.id.get_create_plist()
                _reject(any(creation.get_filter(i)[0] != h5py.h5z.FILTER_DEFLATE
                            for i in range(creation.get_nfilters())))
                _reject(creation.fill_value_defined() != h5py.h5d.FILL_VALUE_DEFAULT)

    visit(file, 0)


def _axis(group, length, columns, string_columns):
    _reject(not isinstance(group, h5py.Group) or set(group) != set(columns))
    _attrs(group, {'encoding-type': 'dataframe', 'encoding-version': '0.2.0',
                   '_index': '_index', 'column-order': columns[1:]})
    for name in columns:
        dataset = group[name]
        _reject(not isinstance(dataset, h5py.Dataset) or dataset.shape != (length,))
        strings = name in string_columns
        _attrs(dataset, {'encoding-type': 'string-array' if strings else 'array',
                         'encoding-version': '0.2.0'})
        if strings:
            for _ in _strings(dataset):
                pass
        else:
            _reject(dataset.dtype.kind != 'i' or dataset.dtype.itemsize != 8)


def _expected_metadata(contract, manifest):
    species = manifest['species']
    if isinstance(species, Mapping):
        species = json.dumps(dict(species), sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False)
    expected = dict(species=species, assembly=manifest['assembly'])
    if contract in _ADOPTED_CONTRACTS:
        expected.update(matrix_semantics=manifest['matrix_semantics'],
                        reference_identity_sha256=manifest['reference']['identity_sha256'])
    else:
        expected.update(matrix_semantics='fragment_counts', matrix_profile_id=manifest['profile']['profile_id'],
                        matrix_profile_sha256=manifest['profile_sha256'],
                        coordinate_system='zero-based-half-open',
                        ordered_selected_sha256=manifest['ordered_selected_sha256'],
                        ordered_feature_sha256=manifest['ordered_feature_sha256'],
                        logical_matrix_sha256=manifest['logical_matrix_sha256'])
    return expected


def validate_matrix_h5ad(stream_or_path, contract, manifest=None):
    """Check unchanged original-byte eligibility on the verified snapshot.

    This checks only the two normalized publication layouts from the five
    accepted matrix contracts. It neither reads sparse counts nor establishes
    scientific authority. ``manifest`` is already accepted owner metadata;
    supplying it also checks the exact scalar metadata and declared shape.
    A caller owns and retains a passed stream; this function closes only HDF5.
    """
    _reject(contract not in _CREATED_CONTRACTS | _ADOPTED_CONTRACTS)
    try:
        with h5py.File(stream_or_path, 'r') as file:
            _reject(file.userblock_size != 0)
            _safe_objects(file)
            _reject(set(file) != _TOP_LEVEL)
            _attrs(file, {'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
            for name in ('layers', 'obsm', 'obsp', 'varm', 'varp', 'uns'):
                _reject(not isinstance(file[name], h5py.Group))
                _attrs(file[name], {'encoding-type': 'dict', 'encoding-version': '0.1.0'})
                _reject(name != 'uns' and bool(len(file[name])))
            matrix = file['X']
            _reject(not isinstance(matrix, h5py.Group) or set(matrix) != {'data', 'indices', 'indptr'}
                    or set(matrix.attrs) != {'encoding-type', 'encoding-version', 'shape'})
            dims = np.asarray(matrix.attrs['shape'])
            _reject(dims.dtype.kind != 'i' or dims.dtype.itemsize != 8 or dims.shape != (2,)
                    or int(dims[0]) < 0 or int(dims[1]) < 1)
            _reject(matrix.attrs['encoding-type'] != 'csr_matrix'
                    or matrix.attrs['encoding-version'] != '0.1.0')
            n, p = map(int, dims)
            if manifest is not None:
                _reject(manifest['contract_version'] != contract or list(dims) != list(manifest['shape']))
            for name in ('data', 'indices', 'indptr'):
                dataset = matrix[name]
                _reject(not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1 or len(dataset.attrs)
                        or dataset.dtype.kind != 'i' or dataset.dtype.itemsize != 8)
            _reject(len(matrix['data']) != len(matrix['indices']) or len(matrix['indptr']) != n + 1)
            created = contract in _CREATED_CONTRACTS
            _axis(file['obs'], n,
                  ('_index', 'namespace', 'barcode_identifier', 'matrix_row_index') if created else ('_index',),
                  {'_index', 'namespace', 'barcode_identifier'})
            _axis(file['var'], p,
                  ('_index', 'chrom', 'start', 'end', 'matrix_column_index') if created else ('_index',),
                  {'_index', 'chrom'})
            fields = _CREATED_UNS if created else _ADOPTED_UNS
            _reject(set(file['uns']) != fields)
            expected = _expected_metadata(contract, manifest) if manifest is not None else None
            for name in fields:
                dataset = file['uns'][name]
                _reject(not isinstance(dataset, h5py.Dataset) or dataset.shape != ())
                _attrs(dataset, {'encoding-type': 'string', 'encoding-version': '0.2.0'})
                value = next(_strings(dataset))
                if expected is not None:
                    _reject(value != expected[name])
                if name == 'species' and contract.startswith('scatac-cell-by-features'):
                    parsed = json.loads(value)
                    _reject(not isinstance(parsed, dict) or set(parsed) != {'scientific_name', 'taxonomy_id'}
                            or not isinstance(parsed['scientific_name'], str)
                            or type(parsed['taxonomy_id']) is not int)
                    _reject(_PATH_TEXT.search(parsed['scientific_name']) is not None)
    except MatrixPrivacyError:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RuntimeError, OverflowError) as exc:
        raise MatrixPrivacyError() from exc


MAX_MATRIX_BYTES = 1024 * 1024 * 1024
MAX_MATRIX_TRANSFERS = 2
MATRIX_CHUNK_BYTES = 1024 * 1024
_DISK_MARGIN_BYTES = 64 * 1024 * 1024


class MatrixDeliveryError(ValueError):
    """Safe transport failure; scientific paths and exceptions stay private."""

    _MESSAGES = {
        'MATRIX_DOWNLOAD_UNAVAILABLE': 'The accepted matrix is unavailable or ineligible for download.',
        'MATRIX_DOWNLOAD_LIMIT': 'The accepted matrix exceeds the 1 GiB download limit.',
        'MATRIX_DOWNLOAD_BUSY': 'Matrix download capacity is unavailable. Try again shortly.',
        'SCIENTIFIC_DOWNLOAD_UNAVAILABLE': 'The accepted scientific file is unavailable or ineligible for download.',
        'SCIENTIFIC_DOWNLOAD_LIMIT': 'The accepted scientific file exceeds the 1 GiB download limit.',
        'SCIENTIFIC_DOWNLOAD_BUSY': 'Scientific download capacity is unavailable. Try again shortly.',
    }

    def __init__(self, code='MATRIX_DOWNLOAD_UNAVAILABLE'):
        self.code = code
        self.message = self._MESSAGES[code]
        super().__init__(self.message)


def _open_matrix_source(path, managed_root):
    """Open every path component without following links, then pin one inode."""
    path, root = Path(path), Path(managed_root)
    if (not path.is_absolute() or not root.is_absolute()
            or any(part in {'.', '..'} for part in path.parts + root.parts)):
        raise ValueError('Invalid managed matrix location.')
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError('Invalid managed matrix location.')
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open('/', directory_flags)
    try:
        for part in root.parts[1:] + relative.parts[:-1]:
            child = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW
                             | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError('Matrix payload is not a regular file.')
            return os.fdopen(descriptor, 'rb')
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(directory)


def _source_state(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class VerifiedMatrixSnapshot:
    """An anonymous private file and its single explicit capacity reservation."""

    filename = 'matrix.h5ad'
    content_type = 'application/octet-stream'

    def __init__(self, stream, *, size_bytes, sha256, release,
                 filename='matrix.h5ad', content_type='application/octet-stream'):
        self._stream = stream
        self._release = release
        self._close_lock = Lock()
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.filename = filename
        self.content_type = content_type

    def iter_chunks(self):
        while True:
            chunk = self._stream.read(MATRIX_CHUNK_BYTES)
            if not chunk:
                return
            yield chunk

    def close(self):
        with self._close_lock:
            if self._release is not None:
                release, self._release = self._release, None
                try:
                    self._stream.close()
                finally:
                    release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Defensive cleanup if response construction/dispatch loses ownership.
        # Normal callers and the ASGI response always close explicitly.
        if hasattr(self, '_close_lock'):
            self.close()


class MatrixDownloadPreparer:
    """One bounded snapshot/transfer capacity shared by every approved file role.

    The historical class/module name is retained for matrix clients. Table
    eligibility stays separate; copy, hashing, reservation and cleanup are reused.
    """

    def __init__(self):
        self._lock = Lock()
        self._active = 0
        self._reserved_bytes = 0

    @property
    def active_count(self):
        with self._lock:
            return self._active

    def _reserve(self, size):
        with self._lock:
            if (self._active >= MAX_MATRIX_TRANSFERS
                    or shutil.disk_usage(tempfile.gettempdir()).free
                    < size + self._reserved_bytes + _DISK_MARGIN_BYTES):
                raise MatrixDeliveryError('MATRIX_DOWNLOAD_BUSY')
            self._active += 1
            self._reserved_bytes += size

    def _release(self, size):
        with self._lock:
            self._active -= 1
            self._reserved_bytes -= size

    def prepare(self, artifact):
        """Copy/hash bounded chunks, validate that same snapshot, then release it."""
        matrix = artifact.contract in _CREATED_CONTRACTS | _ADOPTED_CONTRACTS
        prefix = 'MATRIX' if matrix else 'SCIENTIFIC'
        if matrix:
            if getattr(artifact, 'role', 'matrix') != 'matrix':
                raise MatrixDeliveryError('MATRIX_DOWNLOAD_UNAVAILABLE')
            filename, content_type = 'matrix.h5ad', 'application/octet-stream'
        else:
            table_roles = {
                ('scatac-barcode-qc.v1', 'table'): 'barcodes.tsv.gz',
                ('scatac-barcode-qc.v1', 'histogram'): 'lengths.tsv.gz',
                ('scatac-cell-selection.v1', 'decisions'): 'decisions.tsv.gz',
                ('scatac-cell-selection.v1', 'selected'): 'selected.tsv.gz',
            }
            filename = table_roles.get((artifact.contract, getattr(artifact, 'role', None)))
            if filename is None:
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_UNAVAILABLE')
            content_type = 'application/gzip'
        size = artifact.size_bytes
        if type(size) is not int or size < 1:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_UNAVAILABLE')
        if size > MAX_MATRIX_BYTES:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_LIMIT')
        if type(artifact.sha256) is not str or re.fullmatch('[0-9a-f]{64}', artifact.sha256) is None:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_UNAVAILABLE')
        reserved, snapshot = False, None
        try:
            self._reserve(size)
            reserved = True
            # TemporaryFile is 0600 and anonymous/unlinked on this Linux host.
            # Retaining this descriptor avoids reopening either source or temp.
            snapshot = tempfile.TemporaryFile(mode='w+b')
            checksum, copied = hashlib.sha256(), 0
            with _open_matrix_source(artifact.path, artifact.managed_root) as source:
                before = os.fstat(source.fileno())
                if before.st_size != size:
                    raise ValueError('Changed matrix size.')
                while True:
                    chunk = source.read(MATRIX_CHUNK_BYTES)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > size:
                        raise ValueError('Changed matrix size.')
                    snapshot.write(chunk)
                    checksum.update(chunk)
                if _source_state(os.fstat(source.fileno())) != _source_state(before):
                    raise ValueError('Changed matrix source.')
            if copied != size or checksum.hexdigest() != artifact.sha256:
                raise ValueError('Changed matrix payload.')
            snapshot.flush()
            snapshot.seek(0)
            if matrix:
                validate_matrix_h5ad(snapshot, artifact.contract, artifact.manifest)
            else:
                from .table_delivery import validate_scientific_table
                validate_scientific_table(snapshot, artifact.contract, artifact.role, artifact.manifest)
            snapshot.seek(0)
            resource = VerifiedMatrixSnapshot(snapshot, size_bytes=size,
                sha256=artifact.sha256, release=lambda: self._release(size),
                filename=filename, content_type=content_type)
            snapshot, reserved = None, False
            return resource
        except MatrixDeliveryError as exc:
            if not matrix and exc.code == 'MATRIX_DOWNLOAD_BUSY':
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_BUSY') from exc
            raise
        except (ValueError, TypeError, RuntimeError, OSError, ImportError) as exc:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_UNAVAILABLE') from exc
        finally:
            try:
                if snapshot is not None:
                    snapshot.close()
            finally:
                if reserved:
                    self._release(size)
