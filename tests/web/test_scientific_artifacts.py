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
from rich_helpers import rich_harness


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


def test_accepted_original_matrix_http_delivery_and_legacy_message_omission(tmp_path):
    fixture = rich_harness(tmp_path, managed_matrix=True)
    try:
        before_work = fixture.work.snapshot()
        before_session = fixture.application.sessions.load('analysis')
        before_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in fixture.application.workspace_root.rglob('*') if p.is_file()}
        with TestClient(create_app(fixture.service)) as client:
            revision = fixture.initial_revision
            base = f'/api/v1/sessions/analysis/revisions/{revision}'
            inventory = client.get(base + '/scientific-artifacts')
            assert inventory.status_code == 200, inventory.text
            artifacts = inventory.json()['artifacts']
            assert len(artifacts) == 1 and artifacts[0]['artifact_type'] == 'scientific_matrix'
            entry = artifacts[0]
            assert entry['filename'] == 'matrix.h5ad'
            assert str(tmp_path) not in inventory.text
            # The seed has no captured chat presentation, so a real accepted
            # scientific output alone cannot fabricate a conversation binding.
            assert client.get('/api/v1/sessions/analysis/turns/seed/scientific-artifacts').json() == {'artifacts': []}
            delivered = client.get(base + '/scientific-artifacts/' + entry['handle'])
            assert delivered.status_code == 200, delivered.text
            assert hashlib.sha256(delivered.content).hexdigest() == entry['sha256']
            assert len(delivered.content) == entry['size_bytes']
            assert delivered.headers['content-disposition'] == 'attachment; filename="matrix.h5ad"'
            assert client.get(base + '/artifacts').json()['artifacts']
            assert client.get(base + '/evidence', params={'output_name': 'matrix'}).status_code == 200
            forged = client.get(base + '/scientific-artifacts/' + '0' * 64)
            assert forged.status_code == 404
        assert fixture.work.snapshot() == before_work and not fixture.models
        assert fixture.application.sessions.load('analysis') == before_session
        after_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in fixture.application.workspace_root.rglob('*') if p.is_file()}
        assert after_files == before_files
    finally:
        fixture.close()


def test_accepted_multistep_tables_and_matrix_share_captured_result_with_exact_original_bytes(tmp_path):
    # Only the terminal matrix is selected into the Revision. The accepted
    # originating execution independently binds its QC and selection steps.
    fixture = rich_harness(tmp_path, managed_matrix=True, managed_tables=True,
                           selection_names=('matrix',))
    try:
        state = fixture.application.sessions.load('analysis')
        assert [output.name for output in state.revisions[0].outputs] == ['matrix']
        interaction = Interaction('seed', 'Perform QC, select cells, and build a matrix.', None, 0,
            {'relations': {}, 'bases': {}}, status='submitted',
            admitted={'kind': 'execute', 'request_id': 'seed-request'},
            presentation=PresentedResponse('execute', 'activated', 'Analysis completed.').to_dict())
        fixture.application.sessions._store._write(replace(state, interactions=(interaction,)))
        before_work = fixture.work.snapshot()
        before_session = fixture.application.sessions.load('analysis')
        before_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in fixture.application.workspace_root.rglob('*') if p.is_file()}
        filenames = {'barcodes.tsv.gz', 'lengths.tsv.gz', 'decisions.tsv.gz', 'selected.tsv.gz', 'matrix.h5ad'}
        originals = {p.name: p.read_bytes() for p in fixture.application.workspace_root.rglob('*')
                     if p.name in filenames}
        with TestClient(create_app(fixture.service)) as client:
            base = f'/api/v1/sessions/analysis/revisions/{fixture.initial_revision}'
            inventory = client.get('/api/v1/sessions/analysis/turns/seed/scientific-artifacts')
            assert inventory.status_code == 200, inventory.text
            entries = inventory.json()['artifacts']
            assert {entry['filename'] for entry in entries} == filenames
            assert len(entries) == 5
            assert all(entry['turn_id'] == 'seed' and entry['revision_id'] == fixture.initial_revision
                       for entry in entries)
            assert str(tmp_path) not in inventory.text
            assert client.get(base + '/scientific-artifacts').json() == inventory.json()
            for entry in entries:
                delivered = client.get(base + '/scientific-artifacts/' + entry['handle'])
                assert delivered.status_code == 200, delivered.text
                assert delivered.content == originals[entry['filename']]
                assert hashlib.sha256(delivered.content).hexdigest() == entry['sha256']
                assert len(delivered.content) == entry['size_bytes']
                assert delivered.headers['content-disposition'] == f'attachment; filename="{entry["filename"]}"'
                assert delivered.headers['content-type'] == (
                    'application/octet-stream' if entry['filename'] == 'matrix.h5ad' else 'application/gzip')
                assert delivered.headers['cache-control'] == 'no-store'
                assert fixture.service._matrix_downloads.active_count == 0
            # The existing report/evidence surface remains available alongside
            # original scientific files; it does not supply their identities.
            assert client.get(base + '/artifacts').json()['artifacts']
            assert client.get(base + '/evidence', params={'output_name': 'matrix'}).status_code == 200
        assert fixture.work.snapshot() == before_work and not fixture.models
        assert fixture.application.sessions.load('analysis') == before_session
        assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in fixture.application.workspace_root.rglob('*') if p.is_file()} == before_files
    finally:
        fixture.close()


@pytest.mark.parametrize(('filename', 'alteration'), [
    ('barcodes.tsv.gz', 'replace'), ('lengths.tsv.gz', 'missing'),
    ('decisions.tsv.gz', 'replace'), ('selected.tsv.gz', 'missing'),
])
def test_http_rejects_changed_or_missing_accepted_table_before_attachment_bytes(tmp_path, filename, alteration):
    fixture = rich_harness(tmp_path, managed_matrix=True, managed_tables=True)
    try:
        before_work = fixture.work.snapshot()
        before_session = fixture.application.sessions.load('analysis')
        with TestClient(create_app(fixture.service)) as client:
            base = f'/api/v1/sessions/analysis/revisions/{fixture.initial_revision}/scientific-artifacts'
            entry = next(item for item in client.get(base).json()['artifacts'] if item['filename'] == filename)
            payload = next(p for p in fixture.application.workspace_root.rglob(filename))
            original = payload.read_bytes()
            try:
                if alteration == 'replace':
                    payload.write_bytes(b'unaccepted replacement')
                else:
                    payload.unlink()
                delivered = client.get(base + '/' + entry['handle'])
                assert delivered.status_code == 404
                assert delivered.headers['content-type'].startswith('application/json')
                assert 'content-disposition' not in delivered.headers
                assert str(tmp_path) not in delivered.text
                assert fixture.service._matrix_downloads.active_count == 0
            finally:
                payload.write_bytes(original)
        assert fixture.work.snapshot() == before_work and not fixture.models
        assert fixture.application.sessions.load('analysis') == before_session
    finally:
        fixture.close()
