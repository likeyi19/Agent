"""The reviewed processed role uses the existing bounded snapshot lifecycle."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agent.application import matrix_delivery as delivery
from agent.tools.analysis import neighbors_delivery

sys.path.insert(0, str(Path(__file__).parents[1] / 'web'))
from test_unified_scientific_result_delivery import processed_delivery_harness, processed_delivery_work, no_scientific_or_planning_work


@pytest.fixture
def accepted_graph(tmp_path):
    source, service, models = processed_delivery_harness(tmp_path)
    revision = source.session.active_revision_id
    entry = service.scientific_artifact_handles('analysis', revision)[0]
    artifact = service._scientific_artifact('analysis', revision, entry.handle)
    return source, service, models, artifact


def test_processed_snapshots_share_capacity_and_reserve_both_files(accepted_graph, tmp_path, monkeypatch):
    source, service, models, artifact = accepted_graph
    preparer = service._matrix_downloads
    matrix_path = artifact.managed_root / 'transport-matrix-fixture.h5ad'
    matrix_path.write_bytes(b'accepted transport fixture')
    matrix = SimpleNamespace(path=matrix_path, managed_root=artifact.managed_root,
        sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest(), size_bytes=matrix_path.stat().st_size,
        contract='scatac-cell-by-ccre.v1', role='matrix', manifest={})
    # This tiny matrix exercises the shared transport capacity only. The real
    # raw-family acceptance independently validates genuine matrix bytes.
    monkeypatch.setattr(delivery, 'validate_matrix_h5ad', lambda *args: None)
    reservation = artifact.size_bytes + min(artifact.size_bytes + neighbors_delivery.DESTINATION_OVERHEAD_BYTES,
                                            delivery.MAX_MATRIX_BYTES)
    with processed_delivery_work() as reads:
        graph = preparer.prepare(artifact)
        try:
            assert preparer._reserved_bytes == reservation and preparer.active_count == 1
            other = preparer.prepare(matrix)
            try:
                assert preparer.active_count == 2
                assert preparer._reserved_bytes == reservation + matrix.size_bytes
                with pytest.raises(delivery.MatrixDeliveryError) as rejected:
                    preparer.prepare(artifact)
                assert rejected.value.code == 'SCIENTIFIC_DOWNLOAD_BUSY'
                assert graph.source_sha256 == artifact.sha256
                assert graph.sha256 != graph.source_sha256
                assert b''.join(other.iter_chunks()) == matrix_path.read_bytes()
            finally:
                other.close()
        finally:
            graph.close()
            graph.close()
    assert preparer.active_count == preparer._reserved_bytes == 0
    no_scientific_or_planning_work(reads)
    assert not models


@pytest.mark.parametrize('attack', ['oversize', 'invalid_role', 'source_replaced', 'disk_exhausted'])
def test_processed_preparation_failure_releases_capacity(accepted_graph, monkeypatch, attack):
    source, service, models, artifact = accepted_graph
    expected = 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    if attack == 'oversize':
        artifact = replace(artifact, size_bytes=delivery.MAX_MATRIX_BYTES + 1)
        expected = 'SCIENTIFIC_DOWNLOAD_LIMIT'
    elif attack == 'invalid_role':
        artifact = replace(artifact, role='original_neighbors')
    elif attack == 'source_replaced':
        artifact.path.unlink()
        artifact.path.write_bytes(b'unaccepted replacement')
    else:
        monkeypatch.setattr(delivery.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
        expected = 'SCIENTIFIC_DOWNLOAD_BUSY'
    with processed_delivery_work() as reads, pytest.raises(delivery.MatrixDeliveryError) as rejected:
        service._matrix_downloads.prepare(artifact)
    assert rejected.value.code == expected
    assert service._matrix_downloads.active_count == service._matrix_downloads._reserved_bytes == 0
    no_scientific_or_planning_work(reads)
    assert not models


def test_source_mutation_during_copy_rejects_export_and_closes_snapshot(accepted_graph, monkeypatch):
    source, service, models, artifact = accepted_graph
    original_open, opened, reads = delivery._open_matrix_source, [], []

    class MutatingSource:
        def __init__(self, stream):
            self.stream = stream
            opened.append(stream)

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
                with artifact.path.open('r+b') as current:
                    current.seek(7)
                    current.write(b'm')
            return chunk

    monkeypatch.setattr(delivery, '_open_matrix_source', lambda *args: MutatingSource(original_open(*args)))
    with processed_delivery_work() as counts, pytest.raises(delivery.MatrixDeliveryError) as rejected:
        service._matrix_downloads.prepare(artifact)
    assert rejected.value.code == 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert reads and all(size <= delivery.MATRIX_CHUNK_BYTES for size in reads)
    assert all(stream.closed for stream in opened)
    assert service._matrix_downloads.active_count == service._matrix_downloads._reserved_bytes == 0
    no_scientific_or_planning_work(counts)


@pytest.mark.parametrize('failure', [ValueError, RuntimeError, OSError])
def test_owner_export_failure_closes_both_snapshots_and_releases_reservation(accepted_graph, monkeypatch, failure):
    source, service, models, artifact = accepted_graph
    opened = []

    def reject(source_snapshot, destination_snapshot, *args, **kwargs):
        opened.extend((source_snapshot, destination_snapshot))
        raise failure('unsafe private source /server/private/graph.h5ad')

    monkeypatch.setattr(neighbors_delivery, 'create_neighbors_export', reject)
    with processed_delivery_work() as counts, pytest.raises(delivery.MatrixDeliveryError) as rejected:
        service._matrix_downloads.prepare(artifact)
    assert rejected.value.code == 'SCIENTIFIC_DOWNLOAD_UNAVAILABLE'
    assert '/server/private' not in str(rejected.value)
    assert len(opened) == 2 and all(stream.closed for stream in opened)
    assert service._matrix_downloads.active_count == service._matrix_downloads._reserved_bytes == 0
    no_scientific_or_planning_work(counts)


def test_prepared_derivative_stays_pinned_after_source_replacement_and_interruption(accepted_graph):
    source, service, models, artifact = accepted_graph
    preparer = service._matrix_downloads
    with processed_delivery_work() as counts:
        prepared = preparer.prepare(artifact)
        try:
            source_sha256 = prepared.source_sha256
            artifact.path.unlink()
            artifact.path.write_bytes(b'replaced after verified snapshot')
            streamed = b''.join(prepared.iter_chunks())
            assert hashlib.sha256(streamed).hexdigest() == prepared.sha256
            assert source_sha256 == artifact.sha256
            assert b'replaced after verified snapshot' not in streamed
            prepared._stream.seek(0)
            interrupted = prepared.iter_chunks()
            next(interrupted)
            interrupted.close()
        finally:
            prepared.close()
    assert preparer.active_count == preparer._reserved_bytes == 0
    no_scientific_or_planning_work(counts)
    assert not models
