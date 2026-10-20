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


def inspect(source, *, private_paths=(), privacy=False):
    """Reject indirect storage, unsafe filters and excessive allocations."""
    with (nullcontext(source) if isinstance(source,h5py.File) else h5py.File(source,'r')) as f:
        if not {'X','obs','var'} <= set(f):
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
                for name in item:
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
                    size += item.size * item.dtype.itemsize
                else:
                    raise ValueError('Unsupported H5AD value dtype')
                if size > MAX_LOGICAL_BYTES:
                    raise ValueError('H5AD decoded values exceed 2 GiB limit')
        visit(f,0)
        return size
