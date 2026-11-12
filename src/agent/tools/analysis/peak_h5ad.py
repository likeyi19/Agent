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


def inspect(source, *, private_paths=(), privacy=False, observation_fields=None, require_matrix=True,
            max_logical_bytes=_DEFAULT):
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
        if observation_fields is None and not required <= set(f):
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
        if observation_fields is not None:
            if not isinstance(f.get('obs', getlink=True), h5py.HardLink):
                raise ValueError('Indirect HDF5 observation storage is unsupported')
            visit(f['obs'], 0)
        else:
            visit(f,0)
        return size


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
