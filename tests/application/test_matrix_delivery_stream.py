"""Bounded snapshots pin accepted bytes before any transport begins."""
import gzip
import hashlib
import tracemalloc
from types import SimpleNamespace

import pytest

from agent.application import matrix_delivery as delivery
from agent.application.native_delivery import PRIVATE_LABEL
from agent.tools.analysis import chromvar_native, macs3
from agent.tools.analysis.native_matrix import SNAPATAC2


def identity(tmp_path, payload=b'accepted original bytes'):
    root = tmp_path / 'scientific'
    root.mkdir(exist_ok=True)
    path = root / 'artifact' / 'matrix.h5ad'
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(payload)
    return SimpleNamespace(path=path, managed_root=root,
        sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
        contract='scatac-cell-by-ccre.v1', manifest={})


def native_text(path, owner):
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return SimpleNamespace(path=path, managed_root=path.parent, role=path.name,
        sha256=digest, size_bytes=path.stat().st_size, contract=owner.CONTRACT, manifest={'tool_name': owner.TOOL})


@pytest.mark.parametrize('owner, role, raw, expected', [
    (chromvar_native, 'result.tsv',
     b'/private/track\t-2\t+\t"tab\there"\r\nchr2\t@formula\t-\t"two\nlines"',
     f'{PRIVATE_LABEL}\t-2\t+\t"tab\there"\r\nchr2\t{PRIVATE_LABEL}\t-\t"two\nlines"\r\n'.encode()),
    (macs3, 'peaks_peaks.xls', b'# path\r\n\r\nchr1\t10\r# more\rchr2\t\xff\n# no newline',
     b'\r\nchr1\t10\rchr2\t\xff\n'),
])
def test_native_text_projection_preserves_existing_field_and_line_rules(tmp_path, owner, role, raw, expected):
    path = tmp_path / role
    path.write_bytes(raw)
    with delivery.MatrixDownloadPreparer().prepare(native_text(path, owner)) as snapshot:
        assert snapshot.filename == role
        assert b''.join(snapshot.iter_chunks()) == expected
    assert path.read_bytes() == raw


def test_fragments_are_gzip_with_projected_fields_and_stable_order(tmp_path):
    from agent.application.delivery_policy import DeliveryPolicy
    from agent.web.app import _ScientificFileResponse

    path = tmp_path / 'fragments.tsv'
    raw = b'chr2\t2\t20\tbarcode-z\t3\nchr1\t3\t10\tbarcode-a\t2\nchr1\t1\t5\t=private\t1\n' * 100
    path.write_bytes(raw)
    artifact = native_text(path, SNAPATAC2)
    preparer = delivery.MatrixDownloadPreparer(DeliveryPolicy(len(raw), 1024, len(raw), 2 * len(raw), 1))
    results = []
    for _ in range(2):
        with preparer.prepare(artifact) as snapshot:
            response = _ScientificFileResponse(snapshot)
            assert response.headers['content-type'] == 'application/gzip'
            assert response.headers['content-disposition'] == 'attachment; filename="fragments.tsv.gz"'
            assert 'content-encoding' not in response.headers
            with pytest.raises(delivery.MatrixDeliveryError) as error:
                preparer.prepare(artifact)
            assert error.value.code == 'SCIENTIFIC_DOWNLOAD_BUSY'
            compressed = b''.join(snapshot.iter_chunks())
            assert response.headers['content-length'] == str(len(compressed))
            assert response.headers['x-artifact-content-sha256'] == hashlib.sha256(compressed).hexdigest()
            assert response.headers['x-artifact-source-sha256'] == artifact.sha256 != snapshot.sha256
            assert gzip.decompress(compressed) == raw.replace(b'=private', PRIVATE_LABEL.encode()).replace(b'\n', b'\r\n')
            results.append(compressed)
        assert preparer.active_count == 0 and preparer._reserved_bytes == 0
    assert results[0] == results[1] and path.read_bytes() == raw


@pytest.mark.parametrize('owner, role', [(chromvar_native, 'result.tsv'), (macs3, 'peaks_peaks.xls'),
                                        (SNAPATAC2, 'fragments.tsv')])
def test_native_projection_memory_does_not_scale_with_record_count(tmp_path, owner, role):
    path = tmp_path / role
    peaks = []
    preparer = delivery.MatrixDownloadPreparer()
    for blocks in (1, 200):
        with path.open('wb') as output:
            for _ in range(blocks):
                output.write(b'chr2\t2\t20\tbarcode-z\t3\n' * 1000)
        artifact = native_text(path, owner)
        tracemalloc.start()
        try:
            with preparer.prepare(artifact):
                peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
    assert peaks[1] <= peaks[0] + 4 * delivery.MATRIX_CHUNK_BYTES, peaks


def test_invalid_native_text_closes_both_streams_and_releases_reservation(tmp_path, monkeypatch):
    path = tmp_path / 'result.tsv'
    path.write_bytes(b'chr2\t2\t20\n' * 150000 + b'\xff')
    opened = []
    temporary_file = delivery.tempfile.TemporaryFile

    def tracked_file(**kwargs):
        stream = temporary_file(**kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(delivery.tempfile, 'TemporaryFile', tracked_file)
    preparer = delivery.MatrixDownloadPreparer()
    with pytest.raises(delivery.MatrixDeliveryError) as error:
        preparer.prepare(native_text(path, chromvar_native))
    assert error.value.code == 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert len(opened) == 2 and all(stream.closed for stream in opened)
    assert preparer.active_count == 0 and preparer._reserved_bytes == 0


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
