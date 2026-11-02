"""HDF5 storage safety; AnnData owns encoding and scientific-format parsing.

The optional privacy walk belongs to delivery, not scientific admission.
"""
import re
from contextlib import nullcontext
import h5py
import numpy as np

MAX_BYTES = 1024**3
MAX_LOGICAL_BYTES = 2 * 1024**3
_PATH = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')


def strings(item, selection):
    values = np.asarray(item.astype('S257')[selection])
    result = []
    for index, value in np.ndenumerate(values):
        raw = bytes(value)
        if len(raw) == 257:
            position = () if item.ndim == 0 else selection.start + index[0]
            raw = bytes(item.astype('S65537')[position])
        if len(raw) > 65536:
            raise ValueError('H5AD string exceeds 65536-byte limit')
        result.append(raw.decode('utf-8', errors='strict'))
    return result


def inspect(source, *, private_paths=(), privacy=False, observation_fields=None, require_matrix=True,
            max_logical_bytes=MAX_LOGICAL_BYTES):
    """Reject indirect storage, unsafe filters and excessive allocations."""
    with (nullcontext(source) if isinstance(source,h5py.File) else h5py.File(source,'r')) as f:
        required = {'X','obs','var'} if require_matrix else {'obs','var'}
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
                attr = item.attrs.get_id(key)
                if np.prod(attr.shape, dtype=np.int64) and attr.get_storage_size() > 65536:
                    raise ValueError('Excessive H5AD attribute')
                value = item.attrs[key]
                if np.asarray(value).size > 4096:
                    raise ValueError('Excessive H5AD attribute')
                if privacy and np.asarray(value).dtype.kind in 'OSU':
                    for text in np.asarray(value).ravel():
                        if _PATH.search(str(text)):
                            raise ValueError('Private locator in H5AD metadata')
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
                creation = item.id.get_create_plist()
                if any(creation.get_filter(i)[0] not in (1,2,3) for i in range(creation.get_nfilters())):
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
                elif item.dtype.kind in 'biuf':
                    size += (item.size or 0) * item.dtype.itemsize
                else:
                    raise ValueError('Unsupported H5AD value dtype')
                if max_logical_bytes is not None and size > max_logical_bytes:
                    raise ValueError('H5AD decoded values exceed 2 GiB limit')
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
    blocks=[];stored=0
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
        stored+=block.data.nbytes+block.indices.nbytes+block.indptr.nbytes
        if stored>2*1024**3:raise ValueError('Selected sparse matrix exceeds the 2 GiB exchange allocation budget; this is a runtime limitation')
        blocks.append(block)
    if not all(matrix.shape):
        raise ValueError('Nonempty cell and peak axes are required')
    return sparse.vstack(blocks, format='csr')


def read_native_matrix(matrix):
    """Sparse R exchange with faithful double conversion, no numerical admission."""
    from scipy import sparse
    from agent.tools._cancellation import cancellation_checkpoint
    blocks = []; stored = 0
    for start in range(0, matrix.shape[0], 256):
        cancellation_checkpoint()
        block = sparse.csr_matrix(matrix[start:start + 256])
        native_double_values(block.data)
        block = block.astype(np.float64)
        stored += block.data.nbytes + block.indices.nbytes + block.indptr.nbytes
        if stored > MAX_LOGICAL_BYTES:
            raise ValueError('Selected sparse matrix exceeds the 2 GiB exchange allocation budget')
        blocks.append(block)
    return sparse.vstack(blocks, format='csr') if blocks else sparse.csr_matrix(matrix.shape, dtype=float)


def native_double_values(values):
    """Check actual integer-to-double round trip, not an arbitrary magnitude cap."""
    if values.dtype.kind in 'iu':
        with np.errstate(invalid='ignore'):
            restored = values.astype(np.float64).astype(values.dtype)
        if not np.array_equal(values, restored):
            raise ValueError('Integer measurements cannot be represented exactly as native R doubles')
