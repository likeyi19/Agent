"""Pinned source/TBI registration and scoped ordinary fragments admission."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.application.local_resources import LocalResourceAdmission, ResourceAdmissionError
from agent.application.session_state import canonical, digest
from agent.schemas.orchestration import AgentPlan, PlanStep, _serialize


@pytest.fixture
def registered(tmp_path):
    source = tmp_path / 'source.tsv.gz'
    source.write_bytes(b'byte registration does not qualify fragments semantics')
    index = tmp_path / 'source.tsv.gz.tbi'
    index.write_bytes(b'explicit paired index bytes')
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    record = owner.register('selected', source, input_type='external_fragments', label='Selected fragments',
        attribution='Explicit supplied source and index', source_index_path=index,
        expected_source_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        expected_source_index_size_bytes=index.stat().st_size)
    return owner, record, index


def submission(binding):
    return dict(registered_input=binding.attribution(), execution_inputs=_serialize(binding.execution_inputs))


def test_pair_is_one_immutable_registration_reopened_and_verified(registered, tmp_path):
    owner, record, index = registered
    assert dict(record.source_index) == dict(path=str(index),
        sha256=hashlib.sha256(index.read_bytes()).hexdigest(), size_bytes=index.stat().st_size)
    assert record.public()['has_source_index'] is True
    assert str(index) not in json.dumps(record.public()) and 'sha256' not in json.dumps(record.public())
    with pytest.raises(TypeError):
        record.source_index['path'] = '/another/source.tbi'
    reopened = LocalResourceAdmission(owner._workspace)
    assert reopened.load(record.resource_id) == record
    binding = reopened.compose_native(record.resource_id,{})
    assert binding.execution_inputs['input_path'] == record.source_path
    assert binding.execution_inputs['input_sha256'] == record.source_sha256
    assert 'source_index_path' not in binding.execution_inputs
    index.write_bytes(b'changed index')
    with pytest.raises(ResourceAdmissionError) as error:
        reopened.validate_binding(binding)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert reopened.validate_binding(binding, verify_source=False) == _serialize(binding.execution_inputs)


def test_unpaired_records_keep_the_exact_existing_envelope_and_digest(tmp_path):
    source = tmp_path / 'source.tsv'
    source.write_bytes(b'plain source')
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    record = owner.register('old-source', source, input_type='external_fragments',
                            label='Unpaired source', attribution='Explicit registration')
    payload = json.loads(owner._path(record.resource_id).read_text())
    assert 'source_index' not in record.to_dict() and 'source_index' not in payload['record']
    assert record.record_sha256 == digest(payload['record'])
    assert owner.load(record.resource_id) == record
    payload['record']['source_index'] = None
    payload['sha256'] = digest(payload['record'])
    owner._path(record.resource_id).write_bytes(canonical(payload))
    with pytest.raises(ResourceAdmissionError) as error:
        owner.load(record.resource_id)
    assert error.value.code == 'LOCAL_RESOURCE_RECORD_INVALID'


@pytest.mark.parametrize('field', ['source_index_path', 'source_index_sha256', 'source_path', 'source_sha256'])
def test_paired_source_identities_are_owned_by_registration(registered, tmp_path, field):
    owner, record, _ = registered
    with pytest.raises(ResourceAdmissionError) as error:
        owner.compose_native(record.resource_id,{field:'overridden'})
    assert error.value.code == 'LOCAL_RESOURCE_BINDING_INVALID'


@pytest.mark.parametrize('values', [
    {'expected_source_index_sha256': 'a' * 64},
    {'expected_source_index_size_bytes': 1},
    {'source_index_path': 'same'},
])
def test_partial_index_identity_or_same_source_path_is_rejected(tmp_path, values):
    source = tmp_path / 'source.tsv'
    source.write_bytes(b'source')
    if values.get('source_index_path') == 'same':
        values = {'source_index_path': source}
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    with pytest.raises(ResourceAdmissionError):
        owner.register('selected', source, input_type='external_fragments', label='Source',
                       attribution='Explicit', **values)
    assert not owner._directory(create=True).joinpath(owner.registration_id('selected') + '.json').exists()


def test_registration_rechecks_source_after_reading_index(tmp_path, monkeypatch):
    source, index = tmp_path / 'source.tsv.gz', tmp_path / 'source.tbi'
    source.write_bytes(b'source')
    index.write_bytes(b'index')
    owner = LocalResourceAdmission(tmp_path / 'workspace', approved_source_roots=(tmp_path,))
    original = owner._source
    def changing(path, **kwargs):
        result = original(path, **kwargs)
        if Path(path) == index:
            source.write_bytes(b'changed source')
        return result
    monkeypatch.setattr(owner, '_source', changing)
    with pytest.raises(ResourceAdmissionError) as error:
        owner.register('selected', source, input_type='external_fragments', label='Source',
                       attribution='Explicit', source_index_path=index)
    assert error.value.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
