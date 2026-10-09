"""Bounded snapshots pin accepted bytes before any transport begins."""
import hashlib
from types import SimpleNamespace

import pytest

from agent.application import matrix_delivery as delivery


def identity(tmp_path, payload=b'accepted original bytes'):
    root = tmp_path / 'scientific'
    root.mkdir(exist_ok=True)
    path = root / 'artifact' / 'matrix.h5ad'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(payload)
    return SimpleNamespace(path=path, managed_root=root,
        sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
        contract='scatac-cell-by-ccre.v1', manifest={})


@pytest.fixture
def no_privacy(monkeypatch):
    monkeypatch.setattr(delivery, 'validate_matrix_h5ad', lambda *args, **kwargs: None)


def test_snapshot_is_unchanged_after_original_replacement(tmp_path, no_privacy):
    item = identity(tmp_path)
    preparer = delivery.MatrixDownloadPreparer()
    snapshot = preparer.prepare(item)
    try:
        item.path.unlink()
        item.path.write_bytes(b'replaced original')
        chunks = list(snapshot.iter_chunks())
        assert b''.join(chunks) == b'accepted original bytes'
        assert snapshot.sha256 == item.sha256
        assert snapshot.size_bytes == item.size_bytes
        assert all(len(c) <= delivery.MATRIX_CHUNK_BYTES for c in chunks)
    finally:
        snapshot.close()
    assert preparer.active_count == 0


def test_close_before_iteration_and_capacity_release(tmp_path, no_privacy):
    item = identity(tmp_path)
    preparer = delivery.MatrixDownloadPreparer()
    one, two = preparer.prepare(item), preparer.prepare(item)
    with pytest.raises(delivery.MatrixDeliveryError, match='capacity'):
        preparer.prepare(item)
    one.close()
    one.close()
    third = preparer.prepare(item)
    assert preparer.active_count == 2
    third.close()
    two.close()
    assert preparer.active_count == 0


@pytest.mark.parametrize('change', ['missing', 'bytes', 'file_symlink', 'parent_symlink', 'escape', 'fifo'])
def test_unsafe_sources_fail_without_snapshots(tmp_path, no_privacy, change):
    item = identity(tmp_path)
    if change == 'missing':
        item.path.unlink()
    elif change == 'bytes':
        item.path.write_bytes(b'changed original bytes!')
    elif change == 'file_symlink':
        other = tmp_path / 'elsewhere'
        item.path.rename(other)
        item.path.symlink_to(other)
    elif change == 'parent_symlink':
        other = tmp_path / 'elsewhere'
        item.path.parent.rename(other)
        item.path.parent.symlink_to(other, target_is_directory=True)
    elif change == 'escape':
        item.managed_root = tmp_path / 'other'
    elif change == 'fifo':
        import os
        item.path.unlink()
        os.mkfifo(item.path)
    preparer = delivery.MatrixDownloadPreparer()
    with pytest.raises(delivery.MatrixDeliveryError):
        preparer.prepare(item)
    assert preparer.active_count == 0


def test_privacy_failure_releases_file_and_capacity(tmp_path, monkeypatch):
    item = identity(tmp_path)
    opened = []
    def reject(stream, *args, **kwargs):
        opened.append(stream)
        raise ValueError('unsafe metadata /private/path')
    monkeypatch.setattr(delivery, 'validate_matrix_h5ad', reject)
    preparer = delivery.MatrixDownloadPreparer()
    with pytest.raises(delivery.MatrixDeliveryError) as error:
        preparer.prepare(item)
    assert '/private' not in str(error.value)
    assert all(stream.closed for stream in opened)
    assert preparer.active_count == 0


def test_payload_and_disk_limits(tmp_path, no_privacy, monkeypatch):
    item = identity(tmp_path)
    preparer = delivery.MatrixDownloadPreparer()
    item.size_bytes = delivery.MAX_MATRIX_BYTES + 1
    with pytest.raises(delivery.MatrixDeliveryError) as error:
        preparer.prepare(item)
    assert error.value.code == 'MATRIX_DOWNLOAD_LIMIT'
    item = identity(tmp_path)
    monkeypatch.setattr(delivery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(delivery.MatrixDeliveryError) as error:
        preparer.prepare(item)
    assert error.value.code == 'MATRIX_DOWNLOAD_BUSY'
    assert preparer.active_count == 0


def test_copy_is_bounded_and_mutation_during_preparation_fails(tmp_path, no_privacy, monkeypatch):
    payload = b'a' * (delivery.MATRIX_CHUNK_BYTES * 3 + 23)
    item = identity(tmp_path, payload)
    original_open = delivery._open_matrix_source
    reads = []
    class MutatingSource:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def read(self, size):
            reads.append(size)
            chunk = self.stream.read(size)
            if len(reads) == 1:
                with item.path.open('r+b') as source:
                    source.seek(delivery.MATRIX_CHUNK_BYTES)
                    source.write(b'b')
            return chunk
    monkeypatch.setattr(delivery, '_open_matrix_source', lambda *args: MutatingSource(original_open(*args)))
    preparer = delivery.MatrixDownloadPreparer()
    with pytest.raises(delivery.MatrixDeliveryError):
        preparer.prepare(item)
    assert reads and all(n <= delivery.MATRIX_CHUNK_BYTES for n in reads)
    assert preparer.active_count == 0


def test_interrupted_iteration_then_close_releases_capacity(tmp_path, no_privacy):
    item = identity(tmp_path, b'a' * (delivery.MATRIX_CHUNK_BYTES + 1))
    preparer = delivery.MatrixDownloadPreparer()
    snapshot = preparer.prepare(item)
    chunks = snapshot.iter_chunks()
    assert len(next(chunks)) == delivery.MATRIX_CHUNK_BYTES
    chunks.close()
    snapshot.close()
    assert preparer.active_count == 0


@pytest.mark.parametrize('contract,role,filename', [
    ('scatac-barcode-qc.v1', 'table', 'barcodes.tsv.gz'),
    ('scatac-barcode-qc.v1', 'histogram', 'lengths.tsv.gz'),
    ('scatac-cell-selection.v1', 'decisions', 'decisions.tsv.gz'),
    ('scatac-cell-selection.v1', 'selected', 'selected.tsv.gz'),
])
def test_table_original_bytes_use_the_same_snapshot_transport(tmp_path, contract, role, filename):
    from test_table_delivery_privacy import compressed, content, manifest
    payload = compressed(content(role))
    item = identity(tmp_path, payload)
    item.contract, item.role = contract, role
    item.manifest = manifest(contract, role)
    preparer = delivery.MatrixDownloadPreparer()
    with preparer.prepare(item) as snapshot:
        assert snapshot.filename == filename
        assert snapshot.content_type == 'application/gzip'
        assert b''.join(snapshot.iter_chunks()) == payload
        assert snapshot.sha256 == item.sha256
    assert preparer.active_count == 0
    assert item.path.read_bytes() == payload


def test_table_and_matrix_share_capacity(tmp_path, no_privacy):
    from test_table_delivery_privacy import compressed, content, manifest
    matrix = identity(tmp_path)
    preparer = delivery.MatrixDownloadPreparer()
    one = preparer.prepare(matrix)
    table = identity(tmp_path, compressed(content('selected')))
    table.contract, table.role = 'scatac-cell-selection.v1', 'selected'
    table.manifest = manifest(table.contract, table.role)
    two = preparer.prepare(table)
    try:
        with pytest.raises(delivery.MatrixDeliveryError) as error:
            preparer.prepare(table)
        assert error.value.code == 'SCIENTIFIC_DOWNLOAD_BUSY'
        assert preparer.active_count == 2
    finally:
        one.close()
        two.close()
    assert preparer.active_count == 0


@pytest.mark.parametrize('change', ['bytes', 'private_metadata', 'role', 'size', 'disk'])
def test_table_preparation_failures_release_shared_capacity(tmp_path, monkeypatch, change):
    from test_table_delivery_privacy import compressed, content, manifest
    payload = compressed(content('selected'))
    item = identity(tmp_path, payload)
    item.contract, item.role = 'scatac-cell-selection.v1', 'selected'
    item.manifest = manifest(item.contract, item.role)
    expected = 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    if change == 'bytes':
        item.path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    elif change == 'private_metadata':
        payload = compressed(content('selected', barcode='/srv/private/source'))
        item.path.write_bytes(payload)
        item.sha256, item.size_bytes = hashlib.sha256(payload).hexdigest(), len(payload)
    elif change == 'role':
        item.role = 'receipt'
    elif change == 'size':
        item.size_bytes = delivery.MAX_MATRIX_BYTES + 1
        expected = 'SCIENTIFIC_DOWNLOAD_LIMIT'
    else:
        monkeypatch.setattr(delivery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
        expected = 'SCIENTIFIC_DOWNLOAD_BUSY'
    preparer = delivery.MatrixDownloadPreparer()
    with pytest.raises(delivery.MatrixDeliveryError) as error:
        preparer.prepare(item)
    assert error.value.code == expected
    assert preparer.active_count == 0
