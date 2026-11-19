"""HDF5 storage safety; AnnData owns encoding and scientific-format parsing.

The optional privacy walk belongs to delivery, not scientific admission.
"""
import re
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
import math
import h5py
import numpy as np

# SnapATAC2's native backed H5AD uses Zstandard. Only this reviewed filter is
# admitted in addition to HDF5 built-ins; arbitrary plugin filters stay rejected.
try:
    import hdf5plugin
except ImportError:
    hdf5plugin = None

MAX_LOGICAL_BYTES = 2 * 1024**3
_DEFAULT = object()
_LIMITS = ContextVar('h5ad_inspection_limits', default=(MAX_LOGICAL_BYTES, 64 * 1024**2))


@contextmanager
def inspection_limits(logical_bytes, chunk_bytes):
    if any(type(v) is not int or v <= 0 for v in (logical_bytes, chunk_bytes)):
        raise ValueError('H5AD inspection requires finite positive limits')
    token = _LIMITS.set((logical_bytes, chunk_bytes))
    try:
        yield
    finally:
        _LIMITS.reset(token)
_PATH = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')


def strings(item, selection):
    values = item.asstr(encoding='utf-8', errors='strict')[selection]
    return np.asarray(values).reshape(-1).tolist()


def record_strings(item, field):
    """Read complete string fields in batches, without fixed-width conversion."""
    values = item.fields(field)
    for start in range(0, len(item), 64):
        for raw in values[start:start+64]:
            yield raw.decode('utf-8', errors='strict')


def metadata_item(handle, path):
    """Resolve a selected metadata path without following indirect ancestors."""
    item = handle
    for name in path.split('/'):
        if not name or not isinstance(item, h5py.Group):
            raise ValueError('Invalid selected HDF5 metadata path')
        link = item.get(name, getlink=True)
        if link is None:
            return None
        if not isinstance(link, h5py.HardLink):
            raise ValueError('Indirect HDF5 metadata is unsupported')
        item = item[name]
    return item


def inspect(source, *, private_paths=(), privacy=False, observation_fields=None, require_matrix=True,
            max_logical_bytes=_DEFAULT, metadata_paths=None):
    """Reject indirect storage, unsafe filters and excessive allocations."""
    logical_limit, chunk_limit = _LIMITS.get()
    from agent.tools._native_process import _CONTROLLED
    if _CONTROLLED.get():
        chunk_limit = None
    if max_logical_bytes is _DEFAULT:
        max_logical_bytes = logical_limit
        if _CONTROLLED.get():
            # The kernel controls native allocations; encoded scientific size
            # is not a numerical applicability rule. Delivery has its own scope.
            max_logical_bytes = None
    with (nullcontext(source) if isinstance(source,h5py.File) else h5py.File(source,'r')) as f:
        # Native fragment storage may have obs/obsm and zero features, with
        # neither X nor var. Matrix consumers retain the stricter requirement.
        required = {'X','obs','var'} if require_matrix else {'obs'}
        if observation_fields is None and metadata_paths is None and not required <= set(f):
            raise ValueError('H5AD requires X, obs and var. Cell Ranger .h5 is unsupported; convert with standard scientific software preserving cells, features and counts. Renaming the extension is not conversion.')
        if privacy and f.userblock_size:
            raise ValueError('Unreviewed HDF5 user block cannot be delivered')
        seen = set(); objects = 0; size = 0
        def visit(item, depth):
            nonlocal objects, size
            objects += 1
            if objects > 4096 or depth > 8:
                raise ValueError('H5AD metadata exceeds bounded layout')
            address = h5py.h5o.get_info(item.id).addr
            if address in seen:
                raise ValueError('Repeated HDF5 object is unsupported')
            seen.add(address)
            for key in item.attrs:
                members = [np.asarray(item.attrs[key])]
                while members:
                    value = members.pop()
                    if value.dtype.names:
                        # Compound storage may hold pointers to strings. Inspect
                        # each actual member, including nested compound members.
                        members.extend(value[field] for field in value.dtype.names)
                    elif value.dtype.kind in 'OSU':
                        for text in value.ravel():
                            text = text.decode('utf-8', errors='strict') if isinstance(text, bytes) else str(text)
                            size += len(text.encode('utf-8')) + 8
                            if privacy and _PATH.search(text):
                                raise ValueError('Private locator in H5AD metadata')
                    else:
                        size += value.nbytes
                    if max_logical_bytes is not None and size > max_logical_bytes:
                        raise ValueError('H5AD decoded values exceed configured storage limit')
            if isinstance(item, h5py.Group):
                names = observation_fields if observation_fields is not None and item.name == '/obs' else item
                for name in names:
                    if not isinstance(item.get(name, getlink=True), h5py.HardLink):
                        raise ValueError('Indirect HDF5 links are unsupported')
                    if privacy and _PATH.search(name):
                        raise ValueError('Private locator in H5AD field name')
                    visit(item[name], depth+1)
            else:
                if item.is_virtual or item.external:
                    raise ValueError('Unsupported H5AD dataset layout')
                if chunk_limit is not None and item.chunks and math.prod(item.chunks) * item.dtype.itemsize > chunk_limit:
                    raise ValueError('H5AD decoded chunk exceeds configured storage bound')
                creation = item.id.get_create_plist()
                filters = [creation.get_filter(i)[0] for i in range(creation.get_nfilters())]
                if {32001,32015}.intersection(filters) and hdf5plugin is None:
                    raise ValueError('Install hdf5plugin to read native H5AD Blosc/Zstandard storage')
                if any(filter_id not in (1,2,3,32001,32015) for filter_id in filters):
                    raise ValueError('Unsupported HDF5 filter')
                string = h5py.check_string_dtype(item.dtype) is not None
                if string:
                    if item.ndim > 1:
                        raise ValueError('Only scalar/vector strings are supported')
                    selections = [()] if item.ndim == 0 else (slice(i,min(i+4096,len(item))) for i in range(0,len(item),4096))
                    for selection in selections:
                        for text in strings(item, selection):
                            size += len(text.encode('utf-8')) + 8
                            if privacy and item.name not in private_paths and (_PATH.search(text) or any(ord(c)<32 for c in text)):
                                raise ValueError('Private locator or control character in H5AD metadata')
                elif item.dtype.kind in 'biufc':
                    size += (item.size or 0) * item.dtype.itemsize
                elif item.dtype.names and item.ndim == 1:
                    # AnnData rank statistics use flat record arrays with one
                    # numeric/string field per group. No references, nested
                    # objects or variable-length numeric arrays are admitted.
                    for field in item.dtype.names:
                        dtype = item.dtype.fields[field][0]
                        if privacy and _PATH.search(field):
                            raise ValueError('Private locator in H5AD record field')
                        if dtype.kind in 'biufc':
                            size += item.size * dtype.itemsize
                        elif h5py.check_string_dtype(dtype) is not None:
                            for text in record_strings(item, field):
                                size += len(text.encode('utf-8')) + 8
                                if privacy and (_PATH.search(text) or any(ord(c)<32 for c in text)):
                                    raise ValueError('Private locator or control character in H5AD record')
                        else:
                            raise ValueError('Unsupported H5AD record field dtype')
                else:
                    raise ValueError('Unsupported H5AD value dtype')
                if max_logical_bytes is not None and size > max_logical_bytes:
                    raise ValueError('H5AD decoded values exceed configured storage limit')
        if metadata_paths is not None:
            if observation_fields is not None:
                raise ValueError('Select either observation fields or metadata paths.')
            for path in metadata_paths:
                item = metadata_item(f, path)
                if item is None:
                    raise ValueError('Missing selected HDF5 metadata')
                visit(item, 0)
        elif observation_fields is not None:
            if not isinstance(f.get('obs', getlink=True), h5py.HardLink):
                raise ValueError('Indirect HDF5 observation storage is unsupported')
            visit(f['obs'], 0)
        else:
            visit(f,0)
        return size


def _axis_size(handle, axis):
    frame=metadata_item(handle,axis)
    index=frame.attrs.get('_index') if isinstance(frame,h5py.Group) else None
    if isinstance(index,bytes):index=index.decode('utf-8')
    values=metadata_item(handle,axis+'/'+index) if isinstance(index,str) else None
    return len(values) if isinstance(values,h5py.Dataset) and values.ndim==1 else None


def _numeric_shape(item, *, dense=False):
    if isinstance(item,h5py.Dataset):
        return tuple(item.shape) if item.ndim==2 and item.dtype.kind in ('fi' if dense else 'biufc') else None
    if not dense and isinstance(item,h5py.Group) and item.attrs.get('encoding-type') in ('csr_matrix','csc_matrix',b'csr_matrix',b'csc_matrix'):
        from anndata.io import sparse_dataset
        matrix=sparse_dataset(item)
        return tuple(matrix.shape) if len(matrix.shape)==2 and matrix.dtype.kind in 'biufc' else None
    return None


def matrix_measurement(handle, slot):
    from anndata.io import read_elem
    layer=slot.removeprefix('layers.')
    if '/' in slot or not (slot=='X' or slot.startswith('layers.') and layer):
        return None
    path='uns/matrix_semantics' if slot=='X' else 'uns/layer_semantics/'+layer
    item=metadata_item(handle,path)
    if isinstance(item,h5py.Dataset) and item.ndim==0:
        inspect(handle,require_matrix=False,metadata_paths=(path,),max_logical_bytes=4096)
        value=read_elem(item)
        if isinstance(value,str):return value
    return None


def scientific_properties(path, names, *, graph_inputs=()):
    """Read storage capabilities without loading matrices or recomputing science."""
    if not h5py.is_hdf5(path):
        from agent.tools.data.fragment_io import observed_format
        return {name:value for name,value in {'format':observed_format(path) if 'format' in names else None,
                'x':False}.items() if name in names and value is not None}
    inspect(path,require_matrix=False,max_logical_bytes=None)
    properties={'format': 'h5ad'} if 'format' in names else {}
    with h5py.File(path,'r') as handle:
        n_cells,n_features=(_axis_size(handle,axis) for axis in ('obs','var'))
        if 'n_obs' in names and n_cells is not None:properties['n_obs']=n_cells
        def adjacency(key):
            return bool(isinstance(key,str) and key and '/' not in key and
                _numeric_shape(metadata_item(handle,'obsp/'+key))==(n_cells,n_cells))
        for name in names:
            section, separator, key = name.partition('.')
            if section=='regions' and key and '/' not in key:
                table=metadata_item(handle,'uns/'+key)
                properties[name]=False
                if not isinstance(table,h5py.Group) or table.attrs.get('encoding-type') not in ('dataframe',b'dataframe'):
                    continue
                columns=table.attrs.get('column-order',())
                if not len(columns):continue
                first=columns[0].decode('utf-8') if isinstance(columns[0],bytes) else columns[0]
                if not isinstance(first,str) or not first or '/' in first:continue
                column=metadata_item(handle,'uns/'+key+'/'+first)
                if isinstance(column,h5py.Group) and column.attrs.get('encoding-type') in ('categorical',b'categorical'):
                    codes=metadata_item(column,'codes')
                    if not isinstance(codes,h5py.Dataset) or not len(codes):continue
                    column=metadata_item(column,'categories')
                properties[name]=isinstance(column,h5py.Dataset) and column.ndim==1 and bool(len(column)) and (
                    h5py.check_string_dtype(column.dtype) is not None)
                continue
            if section in ('measurement','count_values'):
                from .cell_result import count_measurement
                value=matrix_measurement(handle,key)
                if value is not None and section=='count_values':value=count_measurement(value)
                if value is not None:properties[name]=value
                continue
            if separator and section in ('obs','var','layers','obsm','obsp','uns','raw','varm','varp'):
                item=metadata_item(handle, section+'/'+key) if key and '/' not in key else None
                properties[name]=item is not None
                if section in ('obs','var') and isinstance(item,h5py.Dataset) and item.ndim==1:
                    if len(item)!=(n_cells if section=='obs' else n_features):properties[name]=False
                    elif item.dtype.kind=='b':properties[name]='boolean'
                    elif item.dtype.kind in 'iuf':properties[name]='numeric'
                if section=='uns' and isinstance(item,h5py.Group):
                    if item.attrs.get('encoding-type') in ('dataframe',b'dataframe'):
                        properties[name]='dataframe'
                elif section=='layers':properties[name]=_numeric_shape(item)==(n_cells,n_features)
                elif section=='obsm' and key not in ('fragment_paired','fragment_single','base_values'):
                    shape=_numeric_shape(item)
                    properties[name]=shape is not None and shape[0]==n_cells
                elif section=='obsp':properties[name]=adjacency(key)
            elif separator and section in ('graph','graph_slot'):
                from anndata.io import read_elem
                from .native_clustering import scanpy_graph_slot
                mode, _, neighbors=key.partition('.')
                group=metadata_item(handle,'uns/'+neighbors) if neighbors and '/' not in neighbors else None
                properties[name]=False
                if key=='snapatac2_distances':
                    native=any(row.get('graph')=='distances' and row.get('function')=='snapatac2.pp.knn'
                               for row in graph_inputs)
                    foreign=any(row.get('distance_slot')=='distances' for row in graph_inputs)
                    properties[name]=adjacency('distances') and (native or not (foreign or
                        metadata_item(handle,'uns/neighbors') is not None and adjacency('connectivities')))
                elif mode in ('scanpy_connectivities','scanpy_distances') and isinstance(group,h5py.Group):
                    role=mode.removeprefix('scanpy_')
                    metadata={}
                    if neighbors!='neighbors':
                        fields=('connectivities_key','distances_key')
                        paths=tuple('uns/'+neighbors+'/'+field for field in fields)
                        if any(metadata_item(handle,path) is None for path in paths):continue
                        inspect(handle,require_matrix=False,metadata_paths=paths,max_logical_bytes=1024**2)
                        metadata={field:read_elem(metadata_item(handle,path)) for field,path in zip(fields,paths)}
                        if any(not isinstance(value,str) or not value for value in metadata.values()):continue
                    slot=scanpy_graph_slot(neighbors,role,metadata)
                    if section=='graph_slot':
                        if isinstance(slot,str):properties[name]=slot
                    else:
                        properties[name]=adjacency(slot) and not any(row.get('graph')==slot and
                            row.get('function')=='snapatac2.pp.knn' for row in graph_inputs)
            elif separator and section == 'dense':
                slot=metadata_item(handle,'X' if key=='x' else 'layers/'+key)
                properties[name]=_numeric_shape(slot,dense=True)==(n_cells,n_features)
        paired=metadata_item(handle,'obsm/fragment_paired') is not None
        single=metadata_item(handle,'obsm/fragment_single') is not None
        values=metadata_item(handle,'obsm/base_values') is not None
        properties.update(fragment_storage=paired or single,paired_fragments=paired,
                          native_values=paired or single or values)
        if paired or single or values:
            properties['storage_kind']='paired_fragments' if paired else 'single_end_insertions' if single else 'native_base_values'
        if 'x' in names:properties['x']=_numeric_shape(metadata_item(handle,'X'))==(n_cells,n_features)
        if 'scanpy_x_rep' in names:
            named=metadata_item(handle,'obsm/X')
            shape=_numeric_shape(named)
            properties['scanpy_x_rep']=(shape is not None and shape[0]==n_cells if named is not None else
                                        _numeric_shape(metadata_item(handle,'X'))==(n_cells,n_features))
        if 'scanpy_auto_rep' in names:
            matrix=_numeric_shape(metadata_item(handle,'X'))==(n_cells,n_features)
            pca=_numeric_shape(metadata_item(handle,'obsm/X_pca'))
            properties['scanpy_auto_rep']=matrix or (n_features is not None and n_features>50
                                                    and pca is not None and pca[0]==n_cells)
    return {name:properties[name] for name in names if name in properties}


def read_native_counts(matrix, semantics):
    """Bounded dense/CSR/CSC exchange; never densify a complete sparse matrix."""
    from scipy import sparse
    from agent.tools._cancellation import cancellation_checkpoint
    blocks=[]
    chunk=max(1,min(256,1_000_000//max(1,matrix.shape[1]))) if not hasattr(matrix, 'format') else 256
    for start in range(0,matrix.shape[0],chunk):
        cancellation_checkpoint();block=sparse.csr_matrix(matrix[start:start+chunk])
        if np.any(block.data > 2**53):
            raise ValueError('Counts exceed exact native double precision; no rounding is permitted')
        block=block.astype(np.float64)
        block.sum_duplicates();block.eliminate_zeros();v=block.data
        if not np.isfinite(v).all() or (v<0).any() or (v!=np.floor(v)).any():
            raise ValueError('Native chromVAR requires finite nonnegative integer counts, not transformed measurements')
        if semantics=='binary_accessibility' and (v>1).any():raise ValueError('Declared binary accessibility contains values above one')
        blocks.append(block)
    if not all(matrix.shape):
        raise ValueError('Nonempty cell and peak axes are required')
    return sparse.vstack(blocks, format='csr')


def read_native_matrix(matrix):
    """Sparse R exchange with faithful double conversion, no numerical admission."""
    from scipy import sparse
    from agent.tools._cancellation import cancellation_checkpoint
    blocks = []
    for start in range(0, matrix.shape[0], 256):
        cancellation_checkpoint()
        block = sparse.csr_matrix(matrix[start:start + 256])
        native_double_values(block.data)
        block = block.astype(np.float64)
        blocks.append(block)
    return sparse.vstack(blocks, format='csr') if blocks else sparse.csr_matrix(matrix.shape, dtype=float)


def native_double_values(values):
    """Check actual integer-to-double round trip, not an arbitrary magnitude cap."""
    if values.dtype.kind in 'iu':
        with np.errstate(invalid='ignore'):
            restored = values.astype(np.float64).astype(values.dtype)
        if not np.array_equal(values, restored):
            raise ValueError('Integer measurements cannot be represented exactly as native R doubles')
