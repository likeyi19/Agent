"""Shared bounded HDF5 download I/O; no scientific algorithm or authority."""

from collections.abc import Mapping
import re

import h5py
import numpy as np



EXPORT_KEY = 'agent_scientific_export'
DESTINATION_OVERHEAD_BYTES = 65_536
MAX_EXPORT_BYTES = 1024 * 1024 * 1024
_STRING_BYTES = 65_536
_STRING_CHUNK = 16
_PATH_TEXT = re.compile(r'(?:^|[\s\"\'=:(])(?:/|~/|[A-Za-z]:[\\/]|\\\\)|file://')


class H5ADDeliveryError(ValueError):
    """A safe failure, containing no scientific source locations."""

    def __init__(self):
        super().__init__('The accepted HDF5 file is ineligible for safe export.')


def _reject(condition):
    if condition:
        raise H5ADDeliveryError()


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
