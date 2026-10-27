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


def _safe_objects(file, *, maximum_depth=2):
    """Visit closed shallow objects without following indirect links."""
    seen = {h5py.h5o.get_info(file.id).addr}
    count = 0

    def visit(group, depth):
        nonlocal count
        _reject(depth > maximum_depth)
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
        'MODEL_DOWNLOAD_LIMIT': 'The accepted model checkpoint exceeds the 5 GiB download limit.',
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


def _snapshot_source(path, root, size, sha256, destination):
    """Pin one accepted file; shared by ordinary and integrated delivery."""
    checksum, copied = hashlib.sha256(), 0
    with _open_matrix_source(path, root) as source:
        before = os.fstat(source.fileno())
        if before.st_size != size:
            raise ValueError('Changed scientific file size.')
        while chunk := source.read(MATRIX_CHUNK_BYTES):
            copied += len(chunk)
            if copied > size:
                raise ValueError('Changed scientific file size.')
            destination.write(chunk)
            checksum.update(chunk)
        if _source_state(os.fstat(source.fileno())) != _source_state(before):
            raise ValueError('Changed scientific source.')
    if copied != size or checksum.hexdigest() != sha256:
        raise ValueError('Changed scientific payload.')
    destination.flush(); destination.seek(0)


MATRIX_CONTRACTS = {
    'build_scATAC_cell_by_ccre': frozenset({'scatac-cell-by-ccre.v1'}),
    'adopt_scATAC_cell_by_ccre': frozenset({'scatac-cell-by-ccre.external.v1'}),
    'build_scATAC_cell_by_features': frozenset({
        'scatac-cell-by-features.v1', 'scatac-cell-by-features.qc-selected.v1'}),
    'adopt_scATAC_cell_by_features': frozenset({'scatac-cell-by-features.external.v1'}),
}

def selects_matrix_port(registry, tool_name, output_key):
    spec = registry.get(tool_name)
    ports = [p for p in spec.semantic_planning.producer_ports
             if output_key in {member.field_name for member in p.members}]
    return len(ports) == 1 and ports[0].name in {'matrix', 'dataset'}


def accepted_matrix(service, session_id, revision, origin, output, *, step_id=None):
    from dataclasses import asdict
    from agent.schemas.orchestration import _serialize
    from .interactive_schemas import ScientificArtifactHandle
    from .interactive import _ScientificMatrixArtifact
    from .session_state import digest
    from agent.orchestration.prior_outputs import accepted_step_digest
    from agent.orchestration.verification_authority import _accepted
    from agent.tools.data import scatac_matrix_contract as contract
    from .matrix_delivery import _open_matrix_source

    store = service._application.run_store
    run_id = revision.run_id if output is None else output.run_id
    step_id = step_id if output is None else output.step_id
    step, authority, _, anchor = _accepted(store, run_id, step_id)
    result, record = _serialize(step.result), authority.record
    if (output is not None and accepted_step_digest(step) != output.accepted_step_sha256
            or result['contract_version'] not in MATRIX_CONTRACTS[step.tool_name]
            or record['schema_version'] != 2
            or record['scope'] != 'scientific_correctness.v1'
            or record['completion'] != 'succeeded'):
        raise ValueError('Unsupported accepted matrix authority.')
    spec = service._application.registry.get(step.tool_name)
    spec.result_contract.validate(result)
    output_key = 'manifest_path' if output is None else output.output_key
    if not selects_matrix_port(service._application.registry, step.tool_name, output_key):
        raise ValueError('Output does not select a reviewed matrix artifact port.')
    manifest_path = Path(result['manifest_path'])
    workspace = service._application._workspace
    root = workspace.runs / workspace.run_digest(run_id) / 'scientific'
    output_dir = Path(step.resolved_arguments['output_dir'])
    for path in (root, output_dir, manifest_path):
        if not path.is_absolute() or path != path.resolve():
            raise ValueError('Unsafe accepted matrix publication path.')
    output_dir.relative_to(root)
    manifest_path.relative_to(root)
    if (manifest_path.name != 'manifest.json' or manifest_path.parent.name != 'artifact'
            or manifest_path.parent.parent.parent != output_dir):
        raise ValueError('Matrix publication differs from the managed output directory.')
    # Inventory reads only the bounded pinned manifest. No matrix open,
    # payload hash, authority-DAG reconstruction, or scientific IO occurs.
    with _open_matrix_source(manifest_path, root) as stream:
        raw = stream.read(contract.MAX_MANIFEST_BYTES + 1)
    if (len(raw) > contract.MAX_MANIFEST_BYTES
            or hashlib.sha256(raw).hexdigest() != result['manifest_sha256']):
        raise ValueError('Changed matrix manifest.')
    manifest = contract.load_manifest_bytes(raw)
    if (record['publication_path'] != str(manifest_path)
            or record['manifest_sha256'] != result['manifest_sha256']
            or record['artifact_contract'] != manifest['contract_version']
            or record['artifact_type'] != manifest['artifact_type']
            or record['science_profile'] != manifest['profile_sha256']
            or _serialize(record['resources']['manifest_identity']) != manifest):
        raise ValueError('Matrix result, manifest and authority differ.')
    if step.tool_name == 'build_scATAC_cell_by_ccre':
        from agent.tools.data.scatac_matrix import _summary
        expected = _summary(manifest, manifest_path, result['manifest_sha256'])
    elif step.tool_name == 'build_scATAC_cell_by_features':
        from agent.tools.data.fragment_feature_matrix import _summary
        expected = _summary(manifest, manifest_path, result['manifest_sha256'])
    else:
        from agent.tools.data.scatac_matrix_adoption import _summary
        from agent.tools.data.external_matrix_contract import contract_for
        expected = _summary(manifest, manifest_path, result['manifest_sha256'],
                            contract=contract_for(manifest))
    if result != expected:
        raise ValueError('Accepted result differs from the matrix owner summary.')
    payload = manifest_path.parent / manifest['matrix']['path']
    if str(payload) != result['matrix_path'] or result['matrix_sha256'] != manifest['matrix']['sha256']:
        raise ValueError('Matrix payload locator or digest differs.')
    files = {f['path']: f for f in record['files']}
    if (_serialize(files.get(str(manifest_path))) != dict(path=str(manifest_path),
            sha256=result['manifest_sha256'], size_bytes=len(raw))
            or _serialize(files.get(str(payload))) != dict(path=str(payload),
            sha256=manifest['matrix']['sha256'], size_bytes=manifest['matrix']['size_bytes'])):
        raise ValueError('Authority does not pin the matrix payload and manifest.')
    proof = record['resources']['result_metadata']
    for key in ('identity_sha256', 'logical_matrix_sha256', 'nnz', 'total_count', 'zero_row_count'):
        if proof[key] != manifest[key]:
            raise ValueError('Scientific proof metadata differs from the matrix manifest.')
    if 'diagnostic' in manifest and _serialize(proof['diagnostic']) != manifest['diagnostic']:
        raise ValueError('Scientific proof diagnostic differs from the matrix manifest.')
    if _accepted(store, run_id, step_id)[3] != anchor:
        raise ValueError('Accepted matrix anchor changed during inventory.')
    identity = dict(session_id=session_id, revision_id=revision.revision_id,
        turn_id=origin.turn_id, output=None if output is None else asdict(output), artifact_role='scientific_matrix',
        authority_sha256=authority.identity_sha256, manifest_sha256=result['manifest_sha256'],
        payload_sha256=manifest['matrix']['sha256'], size_bytes=manifest['matrix']['size_bytes'])
    if output is None:
        identity['accepted_step'] = dict(run_id=run_id, step_id=step_id,
            accepted_step_sha256=accepted_step_digest(step))
    handle = digest(identity)
    reference = ScientificArtifactHandle(handle, 'scientific_matrix', manifest['matrix']['sha256'],
        revision.revision_id, origin.turn_id, manifest['matrix']['size_bytes'])
    return _ScientificMatrixArtifact(reference, payload, root, reference.sha256,
        reference.size_bytes, manifest['contract_version'], manifest)


def _string_block(item, left, right):
    # Most canonical identifiers are short. Fixed-width probing bounds every
    # allocation; a saturated probe is reread individually at the owner limit.
    # This preserves long identifiers without padding every feature to 64 KiB.
    block = item.astype('S256')[left:right]
    values = []
    for offset, value in enumerate(block):
        raw = bytes(value)
        if len(raw) == 256:
            raw = bytes(item.astype('S65537')[left + offset])
        _reject(len(raw) > 65536)
        values.append(raw.decode('utf-8', errors='strict'))
    return values


def _private_metadata(item):
    """Inspect metadata only; scientific arrays are never interpreted as text."""
    from agent.tools.analysis.peak_h5ad import _PATH, strings
    def private(text):
        return _PATH.search(text) or any(ord(c) < 32 for c in text)
    # A typed provenance key such as dependencies/analysis_path is not itself
    # a private locator. Inspect its values; omit explicit credential fields.
    if private(item.name.rsplit('/', 1)[-1]) or item.name.rsplit('/', 1)[-1] in {'api_key', 'credentials'}:
        return True
    for key, value in item.attrs.items():
        if key in {'api_key', 'credentials'}:
            return True
        if np.asarray(value).dtype.kind in 'OSU' and any(private(str(v)) for v in np.asarray(value).ravel()):
            return True
    if isinstance(item, h5py.Group):
        return any(_private_metadata(child) for child in item.values())
    if h5py.check_string_dtype(item.dtype) is not None:
        selections = [()] if item.ndim == 0 else (
            slice(i, min(i+4096, len(item))) for i in range(0, len(item), 4096))
        return any(private(text) for selection in selections for text in strings(item, selection))
    return False


def _redactions(source, *, preserve_annotation=True):
    omitted = set()
    for name in ('obs', 'var', 'raw/var'):
        if name not in source:
            continue
        frame = source[name]
        for column in frame.attrs.get('column-order', ()):
            if _private_metadata(frame[column]):
                omitted.add(frame[column].name)
    if 'uns' in source:
        for name, item in source['uns'].items():
            if (not preserve_annotation or name != 'cell_type_annotation') and _private_metadata(item):
                omitted.add(item.name)
    return omitted


def _copy(source, destination, omitted=(), _references=None):
    root = _references is None
    if root:
        _references = []
    attributes = dict(source.attrs)
    if 'column-order' in attributes:
        attributes['column-order'] = np.array([v for v in attributes['column-order']
            if source[v].name not in omitted], dtype=h5py.string_dtype())
    for key in tuple(attributes):
        dtype = source.attrs.get_id(key).dtype
        reference = h5py.check_dtype(ref=dtype)
        if reference is not None:
            _reject(reference is not h5py.Reference)
            _references.append((destination, key, attributes.pop(key), dtype))
    destination.attrs.update(attributes)
    for name, item in source.items():
        if item.name in omitted:
            continue
        if item.name in ('/uns/cell_type_annotation/matrix_manifest_path','/uns/cell_type_annotation/source_path'): continue
        if isinstance(item, h5py.Group):
            _copy(item, destination.create_group(name), omitted, _references)
            continue
        layout = dict(chunks=item.chunks, compression=item.compression, compression_opts=item.compression_opts,
            shuffle=item.shuffle, fletcher32=item.fletcher32)
        if item.chunks is not None:
            # Native matrix writers use resizable arrays whose chunk can exceed
            # the current tiny shape. Retain that storage contract when copying.
            layout['maxshape'] = item.maxshape
        target = destination.create_dataset(name, shape=item.shape, dtype=item.dtype, **layout)
        for key, value in item.attrs.items():
            dtype = item.attrs.get_id(key).dtype
            reference = h5py.check_dtype(ref=dtype)
            if reference is not None:
                _reject(reference is not h5py.Reference)
                _references.append((target, key, value, dtype))
            else:
                target.attrs[key] = value
        if item.shape == (): target[()] = item[()]
        elif h5py.check_string_dtype(item.dtype) is not None:
            for left in range(0, len(item), 4096):
                right = min(left + 4096, len(item))
                target[left:right] = _string_block(item, left, right)
        else:
            from itertools import product
            budget = max(1, 1024**2 // item.dtype.itemsize)
            block = []
            for dimension in reversed(item.shape):
                width = max(1, min(dimension, budget))
                block.insert(0, width); budget = max(1, budget // width)
            for start in product(*(range(0, length, width) for length, width in zip(item.shape, block))):
                selection = tuple(slice(left, min(left + width, length))
                    for left, width, length in zip(start, block, item.shape))
                target[selection] = item[selection]
    if root:
        # AnnData's supported legacy dataframe encoding uses object-reference
        # attributes for categories. HDF5 references belong to one file and must
        # be rebound after all named objects have been copied, never byte-copied.
        for target, key, values, dtype in _references:
            mapped = np.empty(np.asarray(values).shape, dtype=dtype)
            for index, value in np.ndenumerate(np.asarray(values)):
                if value:
                    name = source.file[value].name
                    _reject(name is None or name not in destination.file)
                    mapped[index] = destination.file[name].ref
                else:
                    mapped[index] = h5py.Reference()
            target.attrs.create(key, mapped[()] if mapped.shape == () else mapped, dtype=dtype)



class VerifiedMatrixSnapshot:
    """An anonymous private file and its single explicit capacity reservation."""

    filename = 'matrix.h5ad'
    content_type = 'application/octet-stream'

    def __init__(self, stream, *, size_bytes, sha256, release,
                 filename='matrix.h5ad', content_type='application/octet-stream', source_sha256=None):
        self._stream = stream
        self._release = release
        self._close_lock = Lock()
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.source_sha256 = sha256 if source_sha256 is None else source_sha256
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
        """Pin accepted bytes and apply the closed role's reviewed delivery policy."""
        matrix = artifact.contract in _CREATED_CONTRACTS | _ADOPTED_CONTRACTS
        from agent.tools.analysis import neighbors_delivery, neighbors_publication
        from . import annotation_delivery, analysis_delivery, model_delivery
        model_role = getattr(artifact, 'role', None)
        model = artifact.contract == model_delivery.CONTRACT and model_role in model_delivery.FILES
        model_metadata = model and model_role == model_delivery.METADATA
        annotation = (artifact.contract == annotation_delivery.CONTRACT
                      and getattr(artifact, 'role', None) == annotation_delivery.ROLE)
        processed = (artifact.contract in neighbors_publication.PROFILES
                     and getattr(artifact, 'role', None) == neighbors_delivery.ROLE)
        compact = (artifact.contract == analysis_delivery.CONTRACT
                   and getattr(artifact, 'role', None) == analysis_delivery.ROLE)
        integrated = (artifact.contract == analysis_delivery.CONTRACT
                      and getattr(artifact, 'role', None) == analysis_delivery.INTEGRATED_ROLE)
        composed = (artifact.contract == analysis_delivery.COMPOSED_CONTRACT
                    and getattr(artifact, 'role', None) == analysis_delivery.COMPOSED_ROLE)
        export = processed or annotation or compact or integrated or composed
        exporter = analysis_delivery if compact else annotation_delivery if annotation else neighbors_delivery
        prefix = 'MATRIX' if matrix else 'SCIENTIFIC'
        if matrix:
            if getattr(artifact, 'role', 'matrix') != 'matrix':
                raise MatrixDeliveryError('MATRIX_DOWNLOAD_UNAVAILABLE')
            filename, content_type = 'matrix.h5ad', 'application/octet-stream'
        elif model:
            filename = model_delivery.FILES[model_role][0]
            content_type = 'application/json' if model_metadata else 'application/octet-stream'
        elif export:
            filename = (artifact.reference.filename if integrated or composed else
                        analysis_delivery.FILENAMES.get(artifact.manifest.get('stage', 'umap'))
                        if compact else exporter.FILENAME)
            if filename is None:
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_UNAVAILABLE')
            content_type = 'application/octet-stream'
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
        limit = (model_delivery.MAX_METADATA_BYTES if model_metadata else model_delivery.MAX_CHECKPOINT_BYTES) if model else MAX_MATRIX_BYTES
        if size > limit and model and not model_metadata:
            raise MatrixDeliveryError('MODEL_DOWNLOAD_LIMIT')
        if size > limit:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_LIMIT')
        if type(artifact.sha256) is not str or re.fullmatch('[0-9a-f]{64}', artifact.sha256) is None:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_UNAVAILABLE')
        delivered_limit = min(size + neighbors_delivery.DESTINATION_OVERHEAD_BYTES, MAX_MATRIX_BYTES)
        original = artifact.manifest['original_source'] if integrated else None
        original_size = 0 if original is None else original['size_bytes']
        if integrated:
            if type(original_size) is not int or original_size < 1:
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_UNAVAILABLE')
            if original_size > MAX_MATRIX_BYTES:
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_LIMIT')
            delivered_limit = min(original_size + size + neighbors_delivery.DESTINATION_OVERHEAD_BYTES,
                                  MAX_MATRIX_BYTES)
        extra_sources = artifact.manifest['sources'] if composed else ()
        if (composed and (not 1 <= len(extra_sources) <= 32 or any(
                type(s['size_bytes']) is not int or not 0 < s['size_bytes'] <= MAX_MATRIX_BYTES
                or type(s['sha256']) is not str or re.fullmatch('[0-9a-f]{64}', s['sha256']) is None
                for s in extra_sources))):
            raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_LIMIT')
        extra_size = sum(s['size_bytes'] for s in extra_sources)
        if composed:
            components = artifact.manifest['components']
            if (not 1 <= len(components) <= 32 or any(type(c['logical_bytes']) is not int
                    or not 0 <= c['logical_bytes'] <= MAX_MATRIX_BYTES for c in components)):
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_LIMIT')
            delivered_limit = (size + sum(c['logical_bytes'] for c in components)
                + analysis_delivery.COMPOSED_METADATA_BYTES
                + len(components) * analysis_delivery.COMPOSED_FIELD_OVERHEAD_BYTES)
            if delivered_limit > MAX_MATRIX_BYTES:
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_LIMIT')
        # A derivative temporarily owns both source and destination snapshots.
        # Reserve both under the same transfer slot and disk admission policy.
        reservation = size + (model_delivery.MAX_METADATA_BYTES if model_metadata else delivered_limit) if export or model_metadata else size
        reservation += original_size + extra_size
        reserved, snapshot, derivative, original_snapshot = False, None, None, None
        extra_snapshots = []
        try:
            self._reserve(reservation)
            reserved = True
            # TemporaryFile is 0600 and anonymous/unlinked on this Linux host.
            # Retaining this descriptor avoids reopening either source or temp.
            snapshot = tempfile.TemporaryFile(mode='w+b')
            _snapshot_source(artifact.path, artifact.managed_root, size, artifact.sha256, snapshot)
            if matrix:
                validate_matrix_h5ad(snapshot, artifact.contract, artifact.manifest)
            elif model_metadata:
                _, manifest = model_delivery.read_json(snapshot)
                if manifest != artifact.manifest:
                    raise ValueError('Changed model metadata.')
                data = model_delivery.metadata(manifest, artifact.manifest_sha256)
                if len(data) > model_delivery.MAX_METADATA_BYTES:
                    raise ValueError('Model metadata projection exceeds the bound.')
                snapshot.seek(0); snapshot.truncate(); snapshot.write(data); snapshot.flush()
                delivered_size = len(data); checksum = hashlib.sha256(data)
            elif model:
                pass  # Exact accepted state-dict bytes; never deserialize weights for delivery.
            elif export:
                derivative = tempfile.TemporaryFile(mode='w+b')
                if composed:
                    for source in extra_sources:
                        stream = tempfile.TemporaryFile(mode='w+b')
                        extra_snapshots.append(stream)
                        _snapshot_source(source['path'], source['managed_root'], source['size_bytes'], source['sha256'], stream)
                    analysis_delivery.create_composed_export(snapshot, extra_snapshots, derivative,
                        artifact.manifest, max_bytes=delivered_limit)
                    for stream in extra_snapshots:
                        stream.close()
                    extra_snapshots.clear()
                elif integrated:
                    original_snapshot = tempfile.TemporaryFile(mode='w+b')
                    source_path = Path(original['path'])
                    _snapshot_source(source_path, source_path.parent, original_size, original['sha256'], original_snapshot)
                    analysis_delivery.create_integrated_export(snapshot, original_snapshot, derivative,
                        artifact.manifest, source_sha256=artifact.sha256, max_bytes=delivered_limit)
                    original_snapshot.close(); original_snapshot = None
                else:
                    create_export = (analysis_delivery.create_analysis_export if compact else
                                     annotation_delivery.create_annotation_export if annotation
                                     else neighbors_delivery.create_neighbors_export)
                    create_export(snapshot, derivative, artifact.manifest,
                        source_sha256=artifact.sha256, source_size_bytes=size,
                        manifest_sha256=artifact.manifest_sha256, max_bytes=delivered_limit)
                derivative.flush()
                derivative.seek(0)
                checksum, delivered_size = hashlib.sha256(), 0
                while chunk := derivative.read(MATRIX_CHUNK_BYTES):
                    delivered_size += len(chunk)
                    if delivered_size > delivered_limit:
                        raise ValueError('Export exceeds the delivery bound.')
                    checksum.update(chunk)
                if delivered_size < 1:
                    raise ValueError('Empty scientific export.')
                snapshot.close()
                snapshot, derivative = derivative, None
            else:
                from .table_delivery import validate_scientific_table
                validate_scientific_table(snapshot, artifact.contract, artifact.role, artifact.manifest)
            snapshot.seek(0)
            resource = VerifiedMatrixSnapshot(snapshot,
                size_bytes=delivered_size if export or model_metadata else size,
                sha256=checksum.hexdigest() if export or model_metadata else artifact.sha256,
                source_sha256=artifact.sha256, release=lambda: self._release(reservation),
                filename=filename, content_type=content_type)
            snapshot, reserved = None, False
            return resource
        except MatrixDeliveryError as exc:
            if not matrix and exc.code == 'MATRIX_DOWNLOAD_BUSY':
                raise MatrixDeliveryError('SCIENTIFIC_DOWNLOAD_BUSY') from exc
            raise
        except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError, SystemError) as exc:
            raise MatrixDeliveryError(prefix + '_DOWNLOAD_UNAVAILABLE') from exc
        finally:
            try:
                try:
                    if snapshot is not None:
                        snapshot.close()
                finally:
                    if derivative is not None:
                        derivative.close()
                    if original_snapshot is not None:
                        original_snapshot.close()
                    for stream in extra_snapshots:
                        stream.close()
            finally:
                if reserved:
                    self._release(reservation)
