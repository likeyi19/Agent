"""Reviewed download derivative of the qualified compact neighbors format.

The caller admits accepted scientific authority and supplies its exact, already
hashed snapshot. This boundary grants no scientific authority. It accepts only
the owner's closed publication layout, then copies scientific values into a new
file while omitting the two recorded private source locators. No source inputs,
AnnData scientific verifier, or graph construction are invoked here.
"""

from collections.abc import Mapping
import re

import h5py
import numpy as np

from . import neighbors_publication as owner


EXPORT_CONTRACT = 'cell-neighbors-export.v1'
ROLE = 'neighbors_export'
FILENAME = 'neighbors.h5ad'
LABEL = 'Cell neighbors — safe export'
EXPORT_KEY = 'agent_scientific_export'
DESTINATION_OVERHEAD_BYTES = 65_536
MAX_EXPORT_BYTES = 1024 * 1024 * 1024
_COPY_BYTES = 1024 * 1024
_STRING_BYTES = 65_536
_STRING_CHUNK = 16
_TOP_LEVEL = {'obs', 'var', 'uns', 'layers', 'obsm', 'obsp', 'varm', 'varp'}
_VERSIONS = {'scanpy', 'anndata', 'numpy', 'scipy', 'scikit_learn', 'umap_learn', 'igraph'}
_PRIVATE_FIELDS = {'source_embedding_path', 'source_cell_ids_path'}
_PATH_TEXT = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')


class NeighborsPrivacyError(ValueError):
    """A safe failure, containing no scientific source locations."""

    def __init__(self):
        super().__init__('The accepted neighbors publication is ineligible for safe export.')


def _reject(condition):
    if condition:
        raise NeighborsPrivacyError()


def _attrs(item, expected):
    _reject(set(item.attrs) != set(expected))
    for key, value in expected.items():
        actual = item.attrs[key]
        if isinstance(value, (tuple, list)):
            array = np.asarray(actual)
            _reject(array.ndim != 1 or list(array) != list(value))
        else:
            _reject(not isinstance(actual, str) or actual != value)


def _group(item, fields):
    _reject(not isinstance(item, h5py.Group) or set(item) != set(fields))
    _attrs(item, {'encoding-type': 'dict', 'encoding-version': '0.1.0'})


def _strings(item, *, private=False):
    _reject(not isinstance(item, h5py.Dataset) or h5py.check_string_dtype(item.dtype) is None)
    reader = item.astype(f'S{_STRING_BYTES + 1}')
    slices = ((),) if item.shape == () else (
        slice(left, left + _STRING_CHUNK) for left in range(0, len(item), _STRING_CHUNK))
    for selection in slices:
        values = [reader[selection]] if selection == () else reader[selection]
        for value in values:
            raw = bytes(value)
            _reject(len(raw) > _STRING_BYTES)
            text = raw.decode('utf-8', errors='strict')
            _reject(not text or any(ord(c) < 32 or ord(c) == 127 for c in text)
                    or (not private and _PATH_TEXT.search(text) is not None))
            yield text


def _scalar(item, expected, *, private=False):
    _reject(not isinstance(item, h5py.Dataset) or item.shape != ())
    if isinstance(expected, str):
        _attrs(item, {'encoding-type': 'string', 'encoding-version': '0.2.0'})
        _reject(next(_strings(item, private=private)) != expected)
    else:
        _attrs(item, {'encoding-type': 'numeric-scalar', 'encoding-version': '0.2.0'})
        _reject(type(expected) is not int or item.dtype.kind != 'i' or item.dtype.itemsize != 8
                or item[()] != expected)


def _safe_objects(file, max_bytes, *, numeric_kinds=frozenset({'i', 'f'}), max_depth=5, max_objects=96):
    """Reject indirect storage and unreviewed attrs before copying any values."""
    seen = {h5py.h5o.get_info(file.id).addr}
    count, numeric_bytes = 0, 0

    def visit(group, depth):
        nonlocal count, numeric_bytes
        _reject(depth > max_depth)
        for name in group:
            count += 1
            _reject(count > max_objects or not isinstance(group.get(name, getlink=True), h5py.HardLink))
            item = group[name]
            address = h5py.h5o.get_info(item.id).addr
            _reject(address in seen)
            seen.add(address)
            if isinstance(item, h5py.Group):
                visit(item, depth + 1)
            else:
                _reject(not isinstance(item, h5py.Dataset) or item.is_virtual or bool(item.external))
                creation = item.id.get_create_plist()
                # The reviewed neighbors writer uses ordinary uncompressed arrays.
                _reject(creation.get_nfilters() != 0
                        or creation.fill_value_defined() != h5py.h5d.FILL_VALUE_DEFAULT)
                if h5py.check_string_dtype(item.dtype) is None:
                    _reject(item.dtype.kind not in numeric_kinds)
                    numeric_bytes += item.size * item.dtype.itemsize
                    _reject(numeric_bytes > max_bytes)

    visit(file, 0)


def _axis(group, length):
    _reject(not isinstance(group, h5py.Group) or set(group) != {'_index'})
    _attrs(group, {'encoding-type': 'dataframe', 'encoding-version': '0.2.0',
                   '_index': '_index', 'column-order': ()})
    index = group['_index']
    _reject(not isinstance(index, h5py.Dataset) or index.shape != (length,))
    _attrs(index, {'encoding-type': 'string-array', 'encoding-version': '0.2.0'})
    for _ in _strings(index):
        pass


def _identity_metadata(manifest, source_sha256, source_size_bytes, manifest_sha256):
    _reject(not isinstance(manifest, Mapping) or manifest['contract_version'] not in owner.PROFILES
            or type(source_size_bytes) is not int or not 0 < source_size_bytes <= MAX_EXPORT_BYTES
            or manifest['analysis']['sha256'] != source_sha256
            or manifest['analysis']['size_bytes'] != source_size_bytes)
    identities = [source_sha256, manifest_sha256, manifest['cell_order_sha256'],
                  manifest['sources']['embedding']['sha256'], manifest['sources']['cell_ids']['sha256']]
    _reject(any(type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                for value in identities))
    for source in manifest['sources'].values():
        _reject(type(source['size_bytes']) is not int or source['size_bytes'] < 1)
    metadata = dict(contract_version=EXPORT_CONTRACT, source_contract=manifest['contract_version'],
        source_sha256=source_sha256, source_size_bytes=source_size_bytes,
        source_manifest_sha256=manifest_sha256, source_cell_order_sha256=manifest['cell_order_sha256'],
        source_embedding_sha256=manifest['sources']['embedding']['sha256'],
        source_embedding_size_bytes=manifest['sources']['embedding']['size_bytes'],
        source_cell_ids_sha256=manifest['sources']['cell_ids']['sha256'],
        source_cell_ids_size_bytes=manifest['sources']['cell_ids']['size_bytes'],
        payload_scope='ordered-cells-embedding-neighbor-graphs',
        format='compact_analysis_result_h5ad',
        scope='Ordered cells, embedding and neighbor graphs; X=None and no original feature axis.',
        provenance_policy='historical-source-locators-omitted',
        authority_scope='export-does-not-issue-scientific-authority')
    if manifest['contract_version'] == 'cell-neighbors-publication.v2':
        metadata.update(embedding_method=manifest['embedding']['method'], representation=manifest['embedding']['representation'],
                        embedding_dim=manifest['summary']['embedding_dim'])
    return metadata


def _inspect(file, manifest, metadata, max_bytes, *, exported):
    _reject(file.userblock_size != 0)
    generic = manifest['contract_version'] == 'cell-neighbors-publication.v2'
    _safe_objects(file, max_bytes, numeric_kinds=frozenset({'i','f','b'}) if generic else frozenset({'i','f'}),
                  max_depth=6 if generic else 5, max_objects=160 if generic else 96)
    _reject(set(file) != _TOP_LEVEL)
    _attrs(file, {'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
    summary, args = manifest['summary'], manifest['arguments']
    n = summary['n_cells']; dimensions = summary['embedding_dim']
    representation = manifest['embedding']['representation'] if generic else 'X_epizoo'
    _reject(type(n) is not int or n < 3 or type(dimensions) is not int or dimensions <= 0
            or (not generic and dimensions != 512))
    _axis(file['obs'], n)
    _axis(file['var'], 0)
    for name in ('layers', 'varm', 'varp'):
        _group(file[name], ())
    _group(file['obsm'], (representation,))
    embeddings = file['obsm'][representation]
    _reject(not isinstance(embeddings, h5py.Dataset) or embeddings.shape != (n, dimensions)
            or embeddings.dtype != np.dtype('float32'))
    _attrs(embeddings, {'encoding-type': 'array', 'encoding-version': '0.2.0'})
    _group(file['obsp'], ('connectivities', 'distances'))
    for name in ('connectivities', 'distances'):
        graph = file['obsp'][name]
        _reject(not isinstance(graph, h5py.Group) or set(graph) != {'data', 'indices', 'indptr'}
                or set(graph.attrs) != {'encoding-type', 'encoding-version', 'shape'})
        _reject(graph.attrs['encoding-type'] != 'csr_matrix' or graph.attrs['encoding-version'] != '0.1.0')
        dims = np.asarray(graph.attrs['shape'])
        _reject(dims.dtype.kind != 'i' or dims.dtype.itemsize != 8 or dims.shape != (2,)
                or list(dims) != [n, n])
        for field in ('data', 'indices', 'indptr'):
            item = graph[field]
            _reject(not isinstance(item, h5py.Dataset) or item.ndim != 1 or len(item.attrs))
            _reject((field == 'data' and (item.dtype.kind != 'f' or item.dtype.itemsize not in (4, 8)))
                    or (field != 'data' and (item.dtype.kind != 'i' or item.dtype.itemsize not in (4, 8))))
        nnz = summary[name + '_nnz']
        _reject(type(nnz) is not int or nnz < 0 or len(graph['data']) != nnz
                or len(graph['indices']) != nnz or len(graph['indptr']) != n + 1)
    fields = {'agent_milestone6', 'neighbors'} | ({EXPORT_KEY} if exported else set())
    _group(file['uns'], fields)
    provenance = file['uns/agent_milestone6']
    fields = {'schema_version', 'stage', 'cell_order_sha256', 'parameters', 'software_versions'}
    # Older delivered v2 files omitted the embedding record entirely. Keep that
    # reader while new exports retain its safe method/parameter provenance.
    embedding = generic and (not exported or 'embedding' in provenance)
    _group(provenance, fields | ({'embedding'} if embedding else set()) | (set() if exported else _PRIVATE_FIELDS))
    if embedding:
        def check_mapping(group, expected):
            _group(group, expected)
            for key, value in expected.items():
                if isinstance(value, Mapping):
                    check_mapping(group[key], value)
                elif type(value) is bool:
                    _attrs(group[key], {'encoding-type':'numeric-scalar','encoding-version':'0.2.0'})
                    _reject(group[key].shape != () or group[key].dtype.kind != 'b' or bool(group[key][()]) != value)
                else:
                    _scalar(group[key], value, private=key in ('source_path','publication_path'))
        expected = {k:v for k,v in manifest['embedding'].items()
                    if not exported or k not in ('source_path', 'publication_path')}
        check_mapping(provenance['embedding'], expected)
    _scalar(provenance['schema_version'], 1)
    _scalar(provenance['stage'], 'neighbors')
    _scalar(provenance['cell_order_sha256'], manifest['cell_order_sha256'])
    if not exported:
        for field, source in (('source_embedding_path', 'embedding'), ('source_cell_ids_path', 'cell_ids')):
            _scalar(provenance[field], manifest['sources'][source]['path'], private=True)
    _group(provenance['parameters'], ('neighbors',))
    parameters = dict(n_neighbors=args['n_neighbors'], metric=args['metric'], method='umap',
                      transformer='none', random_seed=args['random_seed'], use_rep=representation)
    _group(provenance['parameters/neighbors'], parameters)
    for field, value in parameters.items():
        _scalar(provenance['parameters/neighbors'][field], value)
    versions = summary['software_versions']
    _reject(not isinstance(versions, Mapping) or set(versions) != _VERSIONS)
    _group(provenance['software_versions'], _VERSIONS)
    for field, value in versions.items():
        _reject(type(value) is not str or len(value) > 128
                or re.fullmatch('[A-Za-z0-9._+!-]+', value) is None)
        _scalar(provenance['software_versions'][field], value)
    neighbors = file['uns/neighbors']
    _group(neighbors, ('connectivities_key', 'distances_key', 'params'))
    _scalar(neighbors['connectivities_key'], 'connectivities')
    _scalar(neighbors['distances_key'], 'distances')
    parameters = dict(n_neighbors=args['n_neighbors'], metric=args['metric'], method='umap',
                      random_state=args['random_seed'], use_rep=representation)
    _group(neighbors['params'], parameters)
    for field, value in parameters.items():
        _scalar(neighbors['params'][field], value)
    if exported:
        _group(file['uns'][EXPORT_KEY], metadata)
        for field, value in metadata.items():
            _scalar(file['uns'][EXPORT_KEY][field], value)


class _BoundedDestination:
    """Enforce the transport's reservation during HDF5 file-object writes."""

    def __init__(self, stream, maximum):
        self.stream, self.maximum = stream, maximum

    def write(self, value):
        _reject(self.stream.tell() + len(value) > self.maximum)
        return self.stream.write(value)

    def truncate(self, size=None):
        size = self.stream.tell() if size is None else size
        _reject(type(size) is not int or not 0 <= size <= self.maximum)
        return self.stream.truncate(size)

    def read(self, *args):
        return self.stream.read(*args)

    def readinto(self, value):
        return self.stream.readinto(value)

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()

    def flush(self):
        return self.stream.flush()


def _copy_group(source, destination):
    """Copy only the inspected logical objects, never source-file free space."""
    destination.attrs.update(dict(source.attrs))
    for name, item in source.items():
        if source.name == '/uns/agent_milestone6' and name in _PRIVATE_FIELDS:
            continue
        if source.name == '/uns/agent_milestone6/embedding' and name in {'source_path', 'publication_path'}:
            continue
        if isinstance(item, h5py.Group):
            _copy_group(item, destination.create_group(name))
            continue
        target = destination.create_dataset(name, shape=item.shape, dtype=item.dtype)
        target.attrs.update(dict(item.attrs))
        if item.shape == ():
            target[()] = item[()]
        elif h5py.check_string_dtype(item.dtype) is not None:
            # Fixed-width reads bound variable-length values before allocation.
            reader = item.astype(f'S{_STRING_BYTES + 1}')
            for left in range(0, len(item), _STRING_CHUNK):
                selection = slice(left, left + _STRING_CHUNK)
                target[selection] = [bytes(value).decode('utf-8') for value in reader[selection]]
        else:
            row_bytes = max(item.dtype.itemsize * int(np.prod(item.shape[1:])), 1)
            rows = max(_COPY_BYTES // row_bytes, 1)
            for left in range(0, len(item), rows):
                selection = slice(left, left + rows)
                target[selection] = item[selection]


def _write_metadata(group, value):
    group.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
    for name, item in value.items():
        if isinstance(item, str):
            dataset = group.create_dataset(name, data=item, dtype=h5py.string_dtype())
            encoding = 'string'
        else:
            dataset = group.create_dataset(name, data=item, dtype='int64')
            encoding = 'numeric-scalar'
        dataset.attrs.update({'encoding-type': encoding, 'encoding-version': '0.2.0'})


def validate_neighbors_export(stream, manifest, *, source_sha256, source_size_bytes, manifest_sha256,
                              max_bytes):
    """Check the fresh derivative's closed privacy layout; establish no science."""
    try:
        metadata = _identity_metadata(manifest, source_sha256, source_size_bytes, manifest_sha256)
        _reject(type(max_bytes) is not int or not 0 < max_bytes <= MAX_EXPORT_BYTES)
        with h5py.File(stream, 'r') as file:
            if EXPORT_KEY in file['uns'] and 'format' not in file['uns'][EXPORT_KEY]:
                metadata = {k:v for k,v in metadata.items() if k not in ('format', 'scope')}
            _inspect(file, manifest, metadata, max_bytes, exported=True)
    except NeighborsPrivacyError:
        raise
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, RuntimeError, OverflowError) as exc:
        raise NeighborsPrivacyError() from exc


def create_neighbors_export(source_stream, destination_stream, manifest, *, source_sha256,
                            source_size_bytes, manifest_sha256, max_bytes):
    """Write an explicitly labeled owner derivative of a verified source snapshot.

    Both descriptors and capacity belong to the shared transport. The destination
    must be fresh; source and destination remain caller-owned and open. This
    method never opens original source locators or changes the accepted artifact.
    It neither qualifies the derivative for historical reuse nor adds authority.
    """
    try:
        metadata = _identity_metadata(manifest, source_sha256, source_size_bytes, manifest_sha256)
        _reject(type(max_bytes) is not int or not 0 < max_bytes <= MAX_EXPORT_BYTES
                or max_bytes > source_size_bytes + DESTINATION_OVERHEAD_BYTES)
        destination_stream.seek(0, 2)
        _reject(destination_stream.tell() != 0)
        destination_stream.seek(0)
        with h5py.File(source_stream, 'r') as source:
            _inspect(source, manifest, metadata, max_bytes, exported=False)
            with h5py.File(_BoundedDestination(destination_stream, max_bytes), 'w') as destination:
                _copy_group(source, destination)
                _write_metadata(destination['uns'].create_group(EXPORT_KEY), metadata)
        destination_stream.flush()
        destination_stream.seek(0)
        validate_neighbors_export(destination_stream, manifest, source_sha256=source_sha256,
            source_size_bytes=source_size_bytes, manifest_sha256=manifest_sha256, max_bytes=max_bytes)
        destination_stream.seek(0)
    except NeighborsPrivacyError:
        raise
    except (ValueError, TypeError, KeyError, OSError, UnicodeError, RuntimeError, OverflowError) as exc:
        raise NeighborsPrivacyError() from exc
