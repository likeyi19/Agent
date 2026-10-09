"""Explicit FASTQ byte members complete one library in the shared source catalog."""
import asyncio
import hashlib
from pathlib import Path

import pytest

from agent.application.local_resources import LocalResourceAdmission, RegisteredInputCollection, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission, UploadError
from agent.application.workspace import ManagedWorkspace
from agent.schemas.orchestration import _serialize


@pytest.fixture
def uploads(tmp_path):
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace._ensure_directory(workspace.root / 'uploads')
    return H5ADUploadAdmission(LocalResourceAdmission(workspace, approved_source_roots=(root,)), root)


def declaration(role='R1', **changes):
    return dict(library_id='Explicit library', fastq_layout='tenx-atac-r1-r2-r3.v1',
                role=role, lane='001', chunk='001', compression='plain') | changes


async def chunks(*parts):
    for part in parts:
        yield part


def receive(owner, role='R1', *, key=None, data=b'not scientifically qualified', **changes):
    return asyncio.run(owner.receive(key or role, 'misleading_R3.data', chunks(data),
        input_type='fastq', fastq_member=declaration(role, **changes)))


def test_member_attribution_and_catalog_completion_never_execute_science(uploads, monkeypatch):
    from agent.tools.data import raw_scatac, fastq_fragments
    def forbidden(*args, **kwargs):
        pytest.fail('Byte/collection admission executed science')
    monkeypatch.setattr(raw_scatac, 'inspect_raw_scATAC', forbidden)
    monkeypatch.setattr(fastq_fragments, 'prepare_fastq_fragments', forbidden)
    records = [receive(uploads, role) for role in ('R1', 'R2', 'R3')]
    assert uploads.choices() == ()
    assert len(uploads.fastq_members()) == 3
    assert all(record.source_sha256 == hashlib.sha256(b'not scientifically qualified').hexdigest() for record in records)
    for record, role in zip(records, ('R1', 'R2', 'R3')):
        assert Path(record.source_path).name.endswith(f'_S1_L001_{role}_001.fastq')
        assert record.fastq_member['role'] == role
        assert record.public()['fastq_attribution'] == declaration(role)
    completed = uploads.complete_fastq_collection('library', 'Explicit reads', [r.resource_id for r in records])
    assert completed.public()['fastq_collection'] is True
    assert uploads.choices() == (completed.public(),)
    binding = uploads.resolve(completed.resource_id)
    assert isinstance(binding, RegisteredInputCollection)
    assert _serialize(binding.execution_inputs) == dict(raw_input_paths=sorted(r.source_path for r in records),
        raw_assay='TENX_ATAC', fastq_layout='tenx-atac-r1-r2-r3.v1')
    assert not tuple(uploads.resources._workspace.runs.iterdir())
    assert not (uploads.resources._workspace.root / 'sessions').exists()
    reopened = H5ADUploadAdmission(LocalResourceAdmission(uploads.resources._workspace,
        approved_source_roots=(uploads.root,)), uploads.root)
    assert reopened.fastq_members() == uploads.fastq_members()
    assert reopened.choices() == uploads.choices()
    assert reopened.complete_fastq_collection('library', 'Explicit reads', [r.resource_id for r in reversed(records)]) == completed


def test_incomplete_role_group_cannot_publish_a_complete_library(uploads):
    first = receive(uploads)
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.complete_fastq_collection('incomplete', 'Incomplete', [first.resource_id])
    assert error.value.code == 'FASTQ_COLLECTION_INCOMPLETE'
    assert 'R2' in error.value.message and 'R3' in error.value.message and 'barcode' in error.value.message
    assert uploads.choices() == () and len(uploads.fastq_members()) == 1
    assert not uploads.resources._path(uploads.resources.registration_id('web-fastq-collection:incomplete')).exists()
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resolve(first.resource_id)
    assert error.value.code == 'FASTQ_COLLECTION_REQUIRED'


@pytest.mark.parametrize('changes', [
    {'library_id': '../escape'}, {'role': 'I2'}, {'lane': '1'}, {'chunk': '001/escape'},
    {'compression': 'bgzf'}, {'fastq_layout': 'unknown'},
])
def test_invalid_typed_attribution_is_rejected_before_transfer(uploads, changes):
    with pytest.raises(ResourceAdmissionError):
        receive(uploads, **changes)
    assert uploads.fastq_members() == () and uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


@pytest.mark.parametrize('changes', [{'role': 'R2'}, {'library_id': 'Changed library'},
    {'compression': 'gzip'}, {'lane': '002'}, {'chunk': '002'}])
def test_same_upload_identity_cannot_change_member_attribution(uploads, changes):
    record = receive(uploads, key='retry')
    values = declaration() | changes
    async def forbidden_stream():
        pytest.fail('A conflicting retry consumed bytes')
        yield b''
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('retry', 'misleading_R3.data', forbidden_stream(),
            input_type='fastq', fastq_member=values))
    assert error.value.code == 'UPLOAD_CONFLICT'
    assert receive(uploads, key='retry') == record
    assert len(uploads.fastq_members()) == 1 and len(tuple(uploads.completed.iterdir())) == 1


@pytest.mark.parametrize('compression', ['plain', 'gzip'])
def test_other_upload_identity_cannot_reuse_an_occupied_role_slot(uploads, compression):
    record = receive(uploads, key='first')
    with pytest.raises(UploadError) as error:
        receive(uploads, key='other', compression=compression)
    assert error.value.code == 'UPLOAD_CONFLICT'
    assert uploads.fastq_members() == (record.public(),)
    assert len(tuple(uploads.completed.iterdir())) == 1


def test_interrupted_member_and_oversize_leave_existing_members_unchanged(uploads):
    first = receive(uploads)
    async def interrupted():
        yield b'partial'
        raise RuntimeError('private detail')
    with pytest.raises(UploadError) as error:
        asyncio.run(uploads.receive('R2', 'reads.fastq', interrupted(),
            input_type='fastq', fastq_member=declaration('R2')))
    assert error.value.code == 'UPLOAD_INTERRUPTED'
    uploads.max_bytes = 8
    with pytest.raises(UploadError) as error:
        receive(uploads, 'R2')
    assert error.value.code == 'UPLOAD_TOO_LARGE' and '8 bytes' in error.value.message
    assert uploads.fastq_members() == (first.public(),)
    assert uploads.choices() == () and not tuple(uploads.staging.iterdir())


def test_changed_member_prevents_group_publication_and_new_consumption(uploads):
    records = [receive(uploads, role) for role in ('R1', 'R2', 'R3')]
    Path(records[-1].source_path).write_bytes(b'changed')
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.complete_fastq_collection('library', 'Library', [r.resource_id for r in records])
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert uploads.choices() == ()


def test_existing_collection_identity_cannot_be_changed_by_conflicting_completion(uploads):
    records = [receive(uploads, role) for role in ('R1', 'R2', 'R3')]
    completed = uploads.complete_fastq_collection('library', 'Library', [r.resource_id for r in records])
    optional = receive(uploads, 'I1')
    before = uploads.resources._path(completed.resource_id).read_bytes()
    for label, members in (('Changed label', records), ('Library', records + [optional])):
        with pytest.raises(ResourceAdmissionError) as error:
            uploads.complete_fastq_collection('library', label, [r.resource_id for r in members])
        assert error.value.code == 'LOCAL_RESOURCE_CONFLICT'
        assert uploads.resources._path(completed.resource_id).read_bytes() == before
    assert uploads.choices() == (completed.public(),)


def test_collection_registration_failure_does_not_publish_partial_choice(uploads, monkeypatch):
    records = [receive(uploads, role) for role in ('R1', 'R2', 'R3')]
    original = uploads.resources._publish
    def failed(*args, **kwargs):
        raise ResourceAdmissionError('LOCAL_RESOURCE_RECORD_INVALID')
    monkeypatch.setattr(uploads.resources, '_publish', failed)
    with pytest.raises(ResourceAdmissionError):
        uploads.complete_fastq_collection('library', 'Library', [r.resource_id for r in records])
    assert uploads.choices() == () and len(uploads.fastq_members()) == 3
    monkeypatch.setattr(uploads.resources, '_publish', original)
    assert uploads.complete_fastq_collection('library', 'Library', [r.resource_id for r in records]).public() in uploads.choices()


def test_deleted_completed_sources_preserve_metadata_but_prevent_new_consumption(uploads):
    records = [receive(uploads, role) for role in ('R1', 'R2', 'R3')]
    completed = uploads.complete_fastq_collection('library', 'Library', [r.resource_id for r in records])
    binding = uploads.resolve(completed.resource_id)
    Path(records[0].source_path).unlink()
    assert uploads.choices() == (completed.public(),)
    assert uploads.resolve(completed.resource_id, verify_source=False) == binding
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resolve(completed.resource_id)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
