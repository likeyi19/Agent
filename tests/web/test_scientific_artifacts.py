"""The HTTP boundary transports only application-prepared scientific bytes."""
import asyncio
from dataclasses import dataclass, replace
import hashlib

from fastapi.testclient import TestClient
import pytest

from agent.application.matrix_delivery import MatrixDeliveryError
from agent.application.interactive_schemas import PresentedResponse
from agent.application.session_state import Interaction
from agent.web.app import _ScientificFileResponse, create_app
from helpers import harness


@dataclass
class Descriptor:
    handle: str = 'a' * 64
    artifact_type: str = 'scientific_matrix'
    filename: str = 'matrix.h5ad'
    label: str = 'Original accepted matrix'
    revision_id: str = 'historical'
    turn_id: str = 'matrix-turn'
    sha256: str = hashlib.sha256(b'abcd').hexdigest()
    size_bytes: int = 4

    def to_dict(self):
        return self.__dict__.copy()


class Snapshot:
    def __init__(self):
        self.sha256 = hashlib.sha256(b'abcd').hexdigest()
        self.source_sha256 = self.sha256
        self.size_bytes = 4
        self.filename = 'matrix.h5ad'
        self.content_type = 'application/octet-stream'
        self.closed = False
        self.iterated = False

    def iter_chunks(self):
        self.iterated = True
        yield b'ab'
        yield b'cd'

    def close(self):
        self.closed = True


@pytest.mark.parametrize(('artifact_type', 'filename', 'label', 'content_type'), [
    ('scientific_matrix', 'matrix.h5ad', 'Original accepted matrix', 'application/octet-stream'),
    ('scientific_qc_table', 'barcodes.tsv.gz', 'Barcode QC', 'application/gzip'),
    ('scientific_qc_table', 'lengths.tsv.gz', 'Fragment length distribution', 'application/gzip'),
    ('scientific_selection_table', 'decisions.tsv.gz', 'Cell-selection decisions', 'application/gzip'),
    ('scientific_selection_table', 'selected.tsv.gz', 'Selected cells', 'application/gzip'),
])
def test_scientific_inventory_is_separate_from_preparation_and_uses_exact_route(
        monkeypatch, tmp_path, artifact_type, filename, label, content_type):
    fixture = harness(tmp_path)
    calls = []
    descriptor = Descriptor(artifact_type=artifact_type, filename=filename, label=label)
    snapshot = Snapshot()
    snapshot.filename, snapshot.content_type = filename, content_type
    def inventory(session_id, revision_id):
        calls.append(('inventory', session_id, revision_id))
        return (descriptor,)
    def turn_inventory(session_id, turn_id):
        calls.append(('turn', session_id, turn_id))
        return (descriptor,)
    def prepare(session_id, revision_id, handle):
        calls.append(('prepare', session_id, revision_id, handle))
        return snapshot
    monkeypatch.setattr(fixture.service, 'scientific_artifact_handles', inventory, raising=False)
    monkeypatch.setattr(fixture.service, 'scientific_artifacts_for_turn', turn_inventory, raising=False)
    monkeypatch.setattr(fixture.service, 'prepare_scientific_artifact', prepare, raising=False)
    with TestClient(create_app(fixture.service)) as client:
        base = '/api/v1/sessions/analysis/revisions/historical/scientific-artifacts'
        inventory_response = client.get(base)
        assert inventory_response.json() == {'artifacts': [descriptor.to_dict()]}
        assert client.get('/api/v1/sessions/analysis/turns/matrix-turn/scientific-artifacts').json() == inventory_response.json()
        assert calls == [('inventory', 'analysis', 'historical'), ('turn', 'analysis', 'matrix-turn')]
        assert not snapshot.iterated and not snapshot.closed
        downloaded = client.get(base + '/' + descriptor.handle)
        assert downloaded.status_code == 200 and downloaded.content == b'abcd'
        assert calls[-1] == ('prepare', 'analysis', 'historical', descriptor.handle)
        assert downloaded.headers['content-type'] == content_type
        assert downloaded.headers['content-length'] == '4'
        assert downloaded.headers['content-disposition'] == f'attachment; filename="{filename}"'
        assert downloaded.headers['x-artifact-content-sha256'] == snapshot.sha256
        assert downloaded.headers['x-artifact-source-sha256'] == snapshot.sha256
        assert downloaded.headers['cache-control'] == 'no-store'
        assert str(tmp_path) not in str(downloaded.headers)
        assert snapshot.closed
    assert not fixture.models and not fixture.science_calls


@pytest.mark.parametrize(('code', 'status'), [
    ('MATRIX_DOWNLOAD_UNAVAILABLE', 404),
    ('MATRIX_DOWNLOAD_LIMIT', 413),
    ('MATRIX_DOWNLOAD_BUSY', 503),
    ('SCIENTIFIC_DOWNLOAD_UNAVAILABLE', 404),
    ('SCIENTIFIC_DOWNLOAD_LIMIT', 413),
    ('MODEL_DOWNLOAD_LIMIT', 413),
    ('SCIENTIFIC_DOWNLOAD_BUSY', 503),
])
def test_failed_preparation_emits_only_safe_json(monkeypatch, tmp_path, code, status):
    fixture = harness(tmp_path)
    def prepare(*args):
        raise MatrixDeliveryError(code)
    monkeypatch.setattr(fixture.service, 'prepare_scientific_artifact', prepare, raising=False)
    with TestClient(create_app(fixture.service)) as client:
        response = client.get('/api/v1/sessions/analysis/revisions/history/scientific-artifacts/' + 'a' * 64)
        assert response.status_code == status
        assert response.json()['error']['code'] == code
        assert response.headers['content-type'].startswith('application/json')
        assert 'content-disposition' not in response.headers
        assert str(tmp_path) not in response.text
        if status == 503:
            assert response.headers['retry-after'] == '2'
            page = client.get('/api/v1/sessions/analysis/revisions/history/scientific-artifacts/' + 'a' * 64,
                              headers={'Accept': 'text/html'})
            assert page.status_code == 503 and page.headers['retry-after'] == '2'
            assert page.headers['content-type'].startswith('text/html')
            assert int(page.headers['content-length']) == len(page.content)
            assert 'Return to the analysis tab and retry' in page.text
            assert 'Accepted scientific results remain available' in page.text
    assert not fixture.models and not fixture.science_calls


@pytest.mark.parametrize(('fail_at', 'failure'), [
    ('http.response.start', asyncio.CancelledError),
    ('http.response.start', RuntimeError),
    ('http.response.body', asyncio.CancelledError),
    ('http.response.body', OSError),
    ('http.response.body', RuntimeError),
])
def test_stream_lifecycle_releases_snapshot_even_before_first_iteration(fail_at, failure):
    snapshot = Snapshot()
    response = _ScientificFileResponse(snapshot)
    async def receive():
        return {'type': 'http.disconnect'}
    async def send(message):
        if message['type'] == fail_at:
            raise failure('interrupted transmission')
    with pytest.raises(BaseException):
        asyncio.run(response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, receive, send))
    assert snapshot.closed
    assert snapshot.iterated is (fail_at == 'http.response.body')


def test_disconnect_without_stream_iteration_releases_snapshot():
    snapshot = Snapshot()
    response = _ScientificFileResponse(snapshot)
    async def receive():
        return {'type': 'http.disconnect'}
    async def send(message):
        await asyncio.sleep(0)
    asyncio.run(response({'type': 'http', 'asgi': {'spec_version': '2.0'}}, receive, send))
    assert snapshot.closed


def test_response_construction_failure_releases_snapshot():
    class InvalidSnapshot(Snapshot):
        sha256 = None
        def __init__(self):
            super().__init__()
            self.sha256 = None
    snapshot = InvalidSnapshot()
    with pytest.raises(AttributeError):
        _ScientificFileResponse(snapshot)
    assert snapshot.closed
