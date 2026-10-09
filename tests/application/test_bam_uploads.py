"""BAM is one declared byte source in the existing shared upload service."""
import asyncio
import hashlib
from pathlib import Path

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.uploads import H5ADUploadAdmission, UploadError
from agent.application.workspace import ManagedWorkspace


@pytest.fixture
def uploads(tmp_path):
    workspace = ManagedWorkspace(tmp_path / 'workspace')
    root = workspace._ensure_directory(workspace.root / 'uploads')
    return H5ADUploadAdmission(LocalResourceAdmission(workspace, approved_source_roots=(root,)), root)


async def chunks(*parts):
    for part in parts:
        yield part


def receive(owner, data=b'unqualified source bytes', **kwargs):
    return asyncio.run(owner.receive('selected', 'Source.bam', chunks(data), input_type='bam', **kwargs))


def test_bam_transport_registers_bytes_without_qualifying_or_inspecting_them(uploads, monkeypatch):
    from agent.tools.data import raw_scatac, bam_fragments
    def forbidden(*args, **kwargs):
        pytest.fail('Byte registration invoked a scientific owner')
    monkeypatch.setattr(raw_scatac, 'inspect_raw_scATAC', forbidden)
    monkeypatch.setattr(bam_fragments, 'prepare_in_stage', forbidden)
    record = receive(uploads)
    assert record.input_type == 'bam' and record.source_index is None
    assert record.source_sha256 == hashlib.sha256(b'unqualified source bytes').hexdigest()
    assert record.size_bytes == 24
    assert Path(record.source_path).suffix == '.bam'
    assert record.public() == dict(resource_id=record.resource_id, input_type='bam',
                                 label='Source.bam', status='registered')
    assert not tuple(uploads.resources._workspace.runs.iterdir())
    assert not (uploads.resources._workspace.root / 'sessions').exists()
    binding = uploads.resolve(record.resource_id)
    assert binding.tool_name == 'inspect_raw_scATAC'
    assert list(binding.execution_inputs['raw_input_paths']) == [record.source_path]
    assert 'source_profile' not in binding.execution_inputs


def test_shared_catalog_keeps_all_existing_input_roles_and_identity_domains(uploads):
    bam = receive(uploads)
    h5ad = asyncio.run(uploads.receive('selected', 'Source.h5ad', chunks(b'h5ad')))
    fragments = asyncio.run(uploads.receive('selected', 'Source.tsv', chunks(b'fragment'),
                                           input_type='external_fragments'))
    records = (bam, h5ad, fragments)
    assert len({r.resource_id for r in records}) == len({r.source_path for r in records}) == 3
    assert {choice['input_type'] for choice in uploads.choices()} == {'bam', 'h5ad', 'external_fragments'}
    reopened = H5ADUploadAdmission(LocalResourceAdmission(uploads.resources._workspace,
        approved_source_roots=(uploads.root,)), uploads.root)
    assert reopened.choices() == uploads.choices()
    before = Path(bam.source_path).stat()
    assert receive(reopened) == bam
    assert Path(bam.source_path).stat().st_ino == before.st_ino
    with pytest.raises(UploadError) as error:
        receive(reopened, b'changed BAM bytes')
    assert error.value.code == 'UPLOAD_CONFLICT'
    assert Path(bam.source_path).read_bytes() == b'unqualified source bytes'


@pytest.mark.parametrize('options,code', [
    ({'expected_size': 25}, 'UPLOAD_INTERRUPTED'),
    ({'expected_size': 0}, 'UPLOAD_INTERRUPTED'),
    ({'index_filename': 'Source.bam.bai', 'source_size': 24}, 'UPLOAD_REQUEST_INVALID'),
    ({'source_size': 24}, 'UPLOAD_REQUEST_INVALID'),
])
def test_failed_or_index_paired_bam_transfer_leaves_no_completed_source(uploads, options, code):
    with pytest.raises(UploadError) as error:
        receive(uploads, **options)
    assert error.value.code == code
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_actual_bam_limit_reports_the_operator_configuration_action(uploads):
    uploads.max_bytes = 8
    with pytest.raises(UploadError) as error:
        receive(uploads)
    assert error.value.code == 'UPLOAD_TOO_LARGE'
    assert '8 bytes' in error.value.message and 'upload_max_bytes' in error.value.message
    assert 'operator' in error.value.message and str(uploads.root) not in error.value.message
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


@pytest.mark.parametrize('cancelled', [False, True])
def test_bam_interruption_or_cancellation_never_publishes_partial_registration(uploads, cancelled):
    async def interrupted():
        yield b'partial BAM'
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError('private transfer detail')
    with pytest.raises(asyncio.CancelledError if cancelled else UploadError):
        asyncio.run(uploads.receive('interrupted', 'Source.bam', interrupted(), input_type='bam'))
    assert uploads.choices() == ()
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())


def test_new_bam_consumption_checks_bytes_but_historical_binding_remains_readable(uploads):
    record = receive(uploads)
    original = uploads.resolve(record.resource_id)
    Path(record.source_path).unlink()
    assert uploads.choices() == (record.public(),)
    assert uploads.resolve(record.resource_id, verify_source=False) == original
    with pytest.raises(ResourceAdmissionError) as error:
        uploads.resolve(record.resource_id)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
