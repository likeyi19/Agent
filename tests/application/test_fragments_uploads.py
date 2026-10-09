"""One streamed source/index unit uses the existing byte registration owner."""
import asyncio
import hashlib
from pathlib import Path

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission, UploadError, UPLOAD_CHUNK_BYTES
from agent.application.workspace import ManagedWorkspace


@pytest.fixture
def uploads(tmp_path):
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace._ensure_directory(workspace.root / 'uploads')
    return H5ADUploadAdmission(LocalResourceAdmission(workspace, approved_source_roots=(root,)), root)


async def chunks(*parts):
    for part in parts:
        yield part


def receive(owner, source=b'source', index=b'index', **kwargs):
    options = dict(input_type='external_fragments')
    if index is not None:
        options.update(index_filename='Source.tbi', source_size=len(source))
    options.update(kwargs)
    return asyncio.run(owner.receive('selected', 'Source.tsv.gz',
        chunks(source + (index or b'')), **options))


def test_pair_becomes_discoverable_only_after_both_complete(uploads, monkeypatch):
    original = uploads.resources.register
    def observed(key, path, **kwargs):
        assert Path(path).read_bytes() == b'source'
        assert Path(kwargs['source_index_path']).read_bytes() == b'index'
        assert uploads.choices() == () and not tuple(uploads.staging.iterdir())
        return original(key, path, **kwargs)
    monkeypatch.setattr(uploads.resources, 'register', observed)
    record = receive(uploads)
    assert record.source_sha256 == hashlib.sha256(b'source').hexdigest()
    assert record.source_index['sha256'] == hashlib.sha256(b'index').hexdigest()
    assert uploads.choices() == (record.public(),)
    assert record.public()['has_source_index'] is True
    fresh = H5ADUploadAdmission(LocalResourceAdmission(uploads.resources._workspace,
        approved_source_roots=(uploads.root,)), uploads.root)
    assert fresh.choices() == uploads.choices() and receive(fresh) == record
    assert not tuple(uploads.resources._workspace.runs.iterdir())


def test_component_framing_handles_arbitrary_chunk_boundaries_with_bounded_writes(uploads, monkeypatch):
    original, writes = uploads._stage, []
    class Observed:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            writes.append(len(data))
            return self.stream.write(data)
        def __getattr__(self, name):
            return getattr(self.stream, name)
    def stage(token):
        stream, path = original(token)
        return Observed(stream), path
    monkeypatch.setattr(uploads, '_stage', stage)
    source, index = b's' * (UPLOAD_CHUNK_BYTES + 5), b'i' * (UPLOAD_CHUNK_BYTES + 9)
    payload = source + index
    record = asyncio.run(uploads.receive('selected', 'Source.bgz',
        chunks(payload[:3], payload[3:]), input_type='external_fragments',
        source_size=len(source), index_filename='Source.tbi', expected_size=len(payload)))
    assert max(writes) <= UPLOAD_CHUNK_BYTES
    assert Path(record.source_path).read_bytes() == source
    assert Path(record.source_index['path']).read_bytes() == index


@pytest.mark.parametrize('parts,options,code', [
    ((b'source',), {'source_size': 6}, 'UPLOAD_INTERRUPTED'),
    ((b'short',), {'source_size': 6}, 'UPLOAD_INTERRUPTED'),
    ((b'sourceindex',), {'expected_size': 12}, 'UPLOAD_INTERRUPTED'),
    ((b'source' + b'i' * 9,), {}, 'UPLOAD_TOO_LARGE'),
    ((b'sourceindex',), {'source_size': 9}, 'UPLOAD_TOO_LARGE'),
    ((b'sourceindex',), {'source_size': 0}, 'UPLOAD_REQUEST_INVALID'),
    ((b'sourceindex',), {'source_size': True}, 'UPLOAD_REQUEST_INVALID'),
    ((b'sourceindex',), {'index_filename': '../Source.tbi'}, 'UPLOAD_REQUEST_INVALID'),
    ((b'sourceindex',), {'input_type': 'h5ad'}, 'UPLOAD_REQUEST_INVALID'),
    ((b'sourceindex',), {'index_filename': None}, 'UPLOAD_REQUEST_INVALID'),
])
def test_incomplete_invalid_or_oversized_pair_publishes_neither_component(uploads, parts, options, code):
    uploads.max_bytes = 8
    values = dict(input_type='external_fragments', source_size=6, index_filename='Source.tbi') | options
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('failed', 'Source.tsv', chunks(*parts), **values))
    assert error.value.code == code
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_interruption_after_source_and_cancellation_after_index_leave_no_registration(uploads):
    async def interrupted():
        yield b'sourcei'
        raise RuntimeError('private data')
    async def cancelled():
        yield b'sourceindex'
        raise asyncio.CancelledError()
    for stream, exception in ((interrupted(), UploadError), (cancelled(), asyncio.CancelledError)):
        with pytest.raises(exception):
            asyncio.run(uploads.receive('failed', 'Source.tsv', stream,
                input_type='external_fragments', source_size=6, index_filename='Source.tbi'))
        assert uploads.choices() == ()
        assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_conflicting_pair_retry_cannot_replace_or_repopulate_any_pinned_component(uploads):
    record = receive(uploads)
    index = Path(record.source_index['path'])
    index.unlink()
    before = uploads.resources._path(record.resource_id).read_bytes()
    for values in ({'index': b'changed'}, {'source': b'changed'}, {'index': None}):
        with pytest.raises(UploadError) as error:
            receive(uploads, **values)
        assert error.value.code == 'UPLOAD_CONFLICT'
        assert not index.exists()
        assert not tuple(uploads.staging.iterdir())
        assert uploads.resources._path(record.resource_id).read_bytes() == before
    assert receive(uploads) == record and index.read_bytes() == b'index'


def test_second_component_publication_failure_rolls_back_only_new_links(uploads, monkeypatch):
    import agent.application.uploads as module
    original = module.os.link
    def fail_second(source, destination, **kwargs):
        if Path(destination).suffix == '.tbi':
            raise OSError('private storage failure')
        return original(source, destination, **kwargs)
    monkeypatch.setattr(module.os, 'link', fail_second)
    with pytest.raises(UploadError) as error:
        receive(uploads)
    assert error.value.code == 'UPLOAD_STORAGE_INVALID'
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_registration_failure_keeps_only_complete_pair_and_retry_reuses_bytes(uploads, monkeypatch):
    original = uploads.resources.register
    def failed(*args, **kwargs):
        raise ResourceAdmissionError('LOCAL_RESOURCE_RECORD_INVALID')
    monkeypatch.setattr(uploads.resources, 'register', failed)
    with pytest.raises(ResourceAdmissionError):
        receive(uploads)
    assert uploads.choices() == () and not tuple(uploads.staging.iterdir())
    inodes = {path: path.stat().st_ino for path in uploads.completed.iterdir()}
    assert len(inodes) == 2
    monkeypatch.setattr(uploads.resources, 'register', original)
    record = receive(uploads)
    assert {path: path.stat().st_ino for path in uploads.completed.iterdir()} == inodes
    assert uploads.choices() == (record.public(),)


def test_index_change_before_registration_fails_existing_owner_identity_check(uploads, monkeypatch):
    original = uploads.resources.register
    def changed(key, path, **kwargs):
        Path(kwargs['source_index_path']).write_bytes(b'changed index')
        return original(key, path, **kwargs)
    monkeypatch.setattr(uploads.resources, 'register', changed)
    with pytest.raises(ResourceAdmissionError) as error:
        receive(uploads)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert uploads.choices() == ()


def test_pair_counts_both_files_against_the_existing_shared_capacity(uploads, monkeypatch):
    import agent.application.uploads as module
    monkeypatch.setattr(module, 'MAX_UPLOAD_FILES', 1)
    with pytest.raises(UploadError) as error:
        receive(uploads)
    assert error.value.code == 'UPLOAD_STORAGE_LIMIT'
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_fragments_cannot_accept_h5ad_resource_selection_failure(uploads):
    record = receive(uploads, index=None)
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resolve(record.resource_id, resource_selection_error='EPIZOO_RESOURCE_REQUIRED')
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'
