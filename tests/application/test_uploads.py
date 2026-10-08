"""Completed byte transport, existing UA1 ownership, and bounded durable discovery."""
import asyncio
import hashlib
import os
from pathlib import Path

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission, UploadError, UPLOAD_CHUNK_BYTES
from agent.application.workspace import ManagedWorkspace


@pytest.fixture
def uploads(tmp_path):
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace._ensure_directory(workspace.root / 'uploads')
    resources = LocalResourceAdmission(workspace, approved_source_roots=(root,))
    return H5ADUploadAdmission(resources, root)


async def chunks(*values):
    for value in values:
        yield value


def receive(owner, data=b'byte source; not scientific qualification', *, key='first', label='matrix.h5ad', **kwargs):
    return asyncio.run(owner.receive(key, label, chunks(data), **kwargs))


def test_completed_bytes_only_enter_existing_registration(uploads, monkeypatch):
    import agent.tools.data.scatac as inspection
    monkeypatch.setattr(inspection, 'inspect_scATAC', lambda *_args, **_kwargs: pytest.fail('Upload executed science'))
    original = uploads.resources.register
    calls = []
    def observed(key, path, **kwargs):
        assert Path(path).parent == uploads.completed
        assert Path(path).read_bytes() == b'first-second'
        assert not tuple(uploads.staging.iterdir())
        assert uploads.choices() == ()
        calls.append((key, kwargs))
        return original(key, path, **kwargs)
    monkeypatch.setattr(uploads.resources, 'register', observed)
    record = asyncio.run(uploads.receive('valid', '<matrix>.h5ad', chunks(b'first-', b'second'), expected_size=12))
    assert len(calls) == 1 and calls[0][0] == 'web-h5ad-upload:valid'
    assert record.source_sha256 == hashlib.sha256(b'first-second').hexdigest()
    assert record.size_bytes == 12 and record.label == '<matrix>.h5ad'
    assert uploads.choices() == (record.public(),)
    assert not (uploads.resources._workspace.root / 'sessions').exists()
    assert not tuple(uploads.resources._workspace.runs.iterdir())
    assert os.stat(record.source_path).st_mode & 0o777 == 0o600


def test_writes_are_bounded_even_for_large_transport_chunks(uploads, monkeypatch):
    original = uploads._stage
    writes = []
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
    payload = b'x' * (UPLOAD_CHUNK_BYTES * 2 + 19)
    record = receive(uploads, payload)
    assert writes == [UPLOAD_CHUNK_BYTES, UPLOAD_CHUNK_BYTES, 19]
    assert Path(record.source_path).read_bytes() == payload


@pytest.mark.parametrize('expected,parts,code', [
    (None, (b'1234', b'56789'), 'UPLOAD_TOO_LARGE'),
    (4, (b'abc',), 'UPLOAD_INTERRUPTED'),
    (2, (b'abc',), 'UPLOAD_INTERRUPTED'),
    (None, (b'',), 'UPLOAD_INTERRUPTED'),
    (9, (b'abc',), 'UPLOAD_TOO_LARGE'),
    (-1, (b'abc',), 'UPLOAD_REQUEST_INVALID'),
    (True, (b'abc',), 'UPLOAD_REQUEST_INVALID'),
])
def test_actual_bytes_and_declared_length_fail_before_publication(uploads, expected, parts, code):
    uploads.max_bytes = 8
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('failed', 'matrix.h5ad', chunks(*parts), expected_size=expected))
    assert error.value.code == code and '/' not in error.value.message
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


@pytest.mark.parametrize('key,label', [
    ('../escape', 'matrix.h5ad'), ('/absolute', 'matrix.h5ad'), ('x' * 65, 'matrix.h5ad'),
    ('safe', '../matrix.h5ad'), ('safe', '/private/matrix.h5ad'), ('safe', 'C:\\private.h5ad'),
    ('safe', '..'), ('safe', 'bad\nname.h5ad'), ('safe', 'x' * 161),
])
def test_client_names_never_choose_storage_paths(uploads, key, label):
    with pytest.raises(UploadError, match='valid upload identity'):
        receive(uploads, key=key, label=label)
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_stream_interruption_and_cancellation_cleanup_staging(uploads):
    async def interrupted():
        yield b'partial'
        raise RuntimeError('private request payload')
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('interrupted', 'matrix.h5ad', interrupted()))
    assert error.value.code == 'UPLOAD_INTERRUPTED'
    async def cancel():
        yield b'partial'
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(uploads.receive('cancel', 'matrix.h5ad', cancel()))
    assert uploads.choices() == () and not tuple(uploads.staging.iterdir())
    assert receive(uploads, key='interrupted').public() in uploads.choices()


def test_exact_retry_conflicting_bytes_and_metadata_never_overwrite(uploads):
    record = receive(uploads)
    source = Path(record.source_path)
    before = source.stat()
    record_bytes = uploads.resources._path(record.resource_id).read_bytes()
    assert receive(uploads) == record
    assert source.stat().st_ino == before.st_ino and source.stat().st_mtime_ns == before.st_mtime_ns
    for options in ({'data': b'different'}, {'label': 'other.h5ad'}):
        with pytest.raises(UploadError) as error:
            receive(uploads, **options)
        assert error.value.code == 'UPLOAD_CONFLICT'
    assert source.read_bytes() == b'byte source; not scientific qualification'
    assert uploads.resources._path(record.resource_id).read_bytes() == record_bytes
    assert len(uploads.choices()) == 1 and not tuple(uploads.staging.iterdir())


def test_postpublication_registration_failure_can_complete_same_identity(uploads, monkeypatch):
    original = uploads.resources.register
    def failed(*_args, **_kwargs):
        raise ResourceAdmissionError('LOCAL_RESOURCE_RECORD_INVALID')
    monkeypatch.setattr(uploads.resources, 'register', failed)
    with pytest.raises(ResourceAdmissionError):
        receive(uploads)
    assert len(tuple(uploads.completed.iterdir())) == 1 and uploads.choices() == ()
    source = next(uploads.completed.iterdir())
    inode = source.stat().st_ino
    monkeypatch.setattr(uploads.resources, 'register', original)
    record = receive(uploads)
    assert Path(record.source_path) == source and source.stat().st_ino == inode
    assert uploads.choices() == (record.public(),)


def test_deleted_registered_source_is_not_repopulated_by_conflicting_retry(uploads):
    record = receive(uploads)
    source = Path(record.source_path)
    source.unlink()
    persisted = uploads.resources._path(record.resource_id).read_bytes()
    for options in ({'data': b'different'}, {'label': 'different.h5ad'}):
        with pytest.raises(UploadError) as error:
            receive(uploads, **options)
        assert error.value.code == 'UPLOAD_CONFLICT'
        assert not source.exists() and not tuple(uploads.staging.iterdir())
        assert uploads.resources._path(record.resource_id).read_bytes() == persisted
    assert receive(uploads) == record
    assert source.read_bytes() == b'byte source; not scientific qualification'


def test_changed_completed_bytes_cannot_publish_a_false_registration(uploads, monkeypatch):
    original = uploads.resources.register
    def changed(key, source, **kwargs):
        Path(source).write_bytes(b'changed after transfer')
        return original(key, source, **kwargs)
    monkeypatch.setattr(uploads.resources, 'register', changed)
    with pytest.raises(ResourceAdmissionError) as error:
        receive(uploads)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert uploads.choices() == ()
    assert not (uploads.resources._workspace.root / 'local_resources').exists()


@pytest.mark.parametrize('expectations,code', [
    ({'expected_source_sha256': '0' * 64, 'expected_size_bytes': 3}, 'LOCAL_RESOURCE_INTEGRITY_INVALID'),
    ({'expected_source_sha256': '0' * 64}, 'LOCAL_RESOURCE_BINDING_INVALID'),
    ({'expected_size_bytes': 3}, 'LOCAL_RESOURCE_BINDING_INVALID'),
    ({'expected_source_sha256': 'bad', 'expected_size_bytes': 3}, 'LOCAL_RESOURCE_BINDING_INVALID'),
])
def test_expected_transfer_identity_is_checked_by_ua1_before_publication(uploads, expectations, code):
    source = uploads.root / 'explicit.h5ad'
    source.write_bytes(b'abc')
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resources.register('explicit', source, label='explicit.h5ad', attribution='test', **expectations)
    assert error.value.code == code
    assert uploads.resources.list_records(source_root=uploads.root) == ()


def test_stream_elements_must_be_bytes_to_preserve_actual_byte_bounds(uploads):
    from array import array
    uploads.max_bytes = 8
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('view', 'view.h5ad', chunks(memoryview(array('I', [1, 2, 3])))))
    assert error.value.code == 'UPLOAD_REQUEST_INVALID'
    assert uploads.choices() == () and not tuple(uploads.staging.iterdir())


def test_concurrency_capacity_and_same_identity_have_no_queue(uploads):
    uploads._slots = __import__('threading').BoundedSemaphore(1)
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        async def held():
            yield b'first'
            entered.set()
            await release.wait()
            yield b'second'
        task = asyncio.create_task(uploads.receive('held', 'held.h5ad', held()))
        await entered.wait()
        assert uploads.choices() == ()
        with pytest.raises(UploadError) as error:
            await uploads.receive('other', 'other.h5ad', chunks(b'other'))
        assert error.value.code == 'UPLOAD_BUSY'
        release.set()
        record = await task
        assert uploads.choices() == (record.public(),)
    asyncio.run(exercise())


def test_separate_owner_cannot_overwrite_inflight_identity(uploads):
    other = H5ADUploadAdmission(uploads.resources, uploads.root)
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        async def held():
            yield b'first'
            entered.set()
            await release.wait()
        task = asyncio.create_task(uploads.receive('held', 'held.h5ad', held()))
        await entered.wait()
        with pytest.raises(UploadError) as error:
            await other.receive('held', 'held.h5ad', chunks(b'other'))
        assert error.value.code == 'UPLOAD_BUSY'
        release.set()
        await task
    asyncio.run(exercise())


def test_durable_discovery_is_safe_scoped_and_preserves_source_lifetime(uploads):
    record = receive(uploads)
    unrelated = uploads.root / 'operator.h5ad'
    unrelated.write_bytes(b'operator source')
    uploads.resources.register('operator', unrelated, label='Operator', attribution='Operator')
    reopened = LocalResourceAdmission(uploads.resources._workspace.root, approved_source_roots=(uploads.root,))
    fresh = H5ADUploadAdmission(reopened, uploads.root)
    assert fresh.choices() == (record.public(),)
    binding = fresh.resolve(record.resource_id)
    Path(record.source_path).unlink()
    assert fresh.choices() == (record.public(),)
    assert fresh.resolve(record.resource_id, verify_source=False) == binding
    with pytest.raises(ResourceAdmissionError) as error:
        fresh.resolve(record.resource_id)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    with pytest.raises(ResourceAdmissionError):
        fresh.resolve('local-' + '0' * 64, verify_source=False)


def test_corrupt_record_fails_closed_without_listing_private_payload(uploads):
    record = receive(uploads)
    uploads.resources._path(record.resource_id).write_bytes(b'private corrupted record')
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.choices()
    assert error.value.code == 'LOCAL_RESOURCE_RECORD_INVALID'
    assert 'private' not in error.value.message


def test_source_replacement_rejects_new_binding_but_not_metadata(uploads):
    record = receive(uploads)
    source = Path(record.source_path)
    source.write_bytes(b'changed source')
    assert uploads.resolve(record.resource_id, verify_source=False).resource_id == record.resource_id
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resolve(record.resource_id)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'


def test_unsafe_completed_destination_and_root_are_rejected(uploads, tmp_path):
    record = receive(uploads)
    source = Path(record.source_path)
    source.unlink()
    outside = tmp_path / 'outside'
    outside.write_bytes(b'outside')
    source.symlink_to(outside)
    with pytest.raises(UploadError) as error:
        receive(uploads)
    assert error.value.code == 'UPLOAD_STORAGE_INVALID' and outside.read_bytes() == b'outside'
    with pytest.raises(UploadError):
        H5ADUploadAdmission(uploads.resources, tmp_path)


def test_completed_and_inflight_file_count_is_bounded(uploads, monkeypatch):
    import agent.application.uploads as module
    monkeypatch.setattr(module, 'MAX_UPLOAD_FILES', 1)
    record = receive(uploads)
    assert receive(uploads) == record  # Exact retries consume no extra staging file.
    with pytest.raises(UploadError) as error:
        receive(uploads, key='second')
    assert error.value.code == 'UPLOAD_STORAGE_LIMIT'
    assert uploads.choices() == (record.public(),)


def test_record_discovery_bounds_never_silently_truncate(uploads, monkeypatch):
    import agent.application.local_resources as module
    receive(uploads, key='first')
    receive(uploads, key='second')
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resources.list_records(source_root=uploads.completed, limit=1)
    assert error.value.code == 'LOCAL_RESOURCE_DISCOVERY_LIMIT'
    monkeypatch.setattr(module, 'MAX_DISCOVERY_RECORDS', 1)
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.choices()
    assert error.value.code == 'LOCAL_RESOURCE_DISCOVERY_LIMIT'
