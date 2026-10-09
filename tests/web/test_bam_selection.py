"""Explicit BAM context selection does not impose producer rules on inspection."""
from dataclasses import replace
import json

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import _bam_selection, create_app
from agent.web.config import ScientificInputSet, WebConfigurationError, load_web_configuration

from helpers import harness, submission


def context(identifier='selected', *, default=False, **inputs):
    values = dict(library_context_path='/operator/qualified/library.json', library_context_sha256='a' * 64,
        reference_bundle_path='/operator/qualified/reference.json', reference_bundle_sha256='b' * 64,
        source_profile='agent-cb-paired-atac.v1') | inputs
    return ScientificInputSet(identifier, 'Qualified ' + identifier + ' BAM context', values,
                              bam_companion=True, bam_default=default)


@pytest.mark.parametrize('defaults', [(), (context('default', default=True),),
    (context('first', default=True), context('second', default=True))])
def test_explicit_context_has_priority_and_does_not_copy_other_defaults(defaults):
    selected = context('explicit', reference_bundle_sha256='c' * 64)
    assert _bam_selection(selected, defaults + (selected,)) == (selected.inputs(), None)


@pytest.mark.parametrize('contexts,expected', [
    ((), ({}, None)), ((context(),), ({}, None)),
    ((context('first', default=True), context('second', default=True)), ({}, 'BAM_RESOURCE_AMBIGUOUS')),
])
def test_inspection_remains_usable_without_a_unique_context_default(contexts, expected):
    assert _bam_selection(None, contexts) == expected


def test_unique_explicitly_configured_default_and_incomplete_context_are_preserved():
    selected = context('default', default=True)
    assert _bam_selection(None, (context(), selected)) == (selected.inputs(), None)
    partial = ScientificInputSet('partial', 'Inspection declarations', {'species': 'human'}, bam_companion=True)
    assert _bam_selection(partial, (selected,)) == ({'species': 'human'}, None)


@pytest.mark.parametrize('selection', [
    ScientificInputSet('matrix', 'Matrix', {'species': 'human'}, h5ad_companion=True),
    ScientificInputSet('fragments', 'Fragments', {'namespace': 'library'}, fragments_companion=True),
    ScientificInputSet('unmarked', 'Unmarked', {}),
])
def test_other_context_roles_do_not_become_bam_contexts(selection):
    with pytest.raises(InteractiveBoundaryError):
        _bam_selection(selection, (context('default', default=True),))


@pytest.mark.parametrize('changes', [
    {'bam_companion': 1}, {'bam_default': 'yes'}, {'bam_companion': False, 'bam_default': True},
    {'h5ad_companion': True}, {'fragments_companion': True},
])
def test_invalid_or_overlapping_context_designations_fail_config_validation(changes):
    with pytest.raises(WebConfigurationError):
        replace(context(), **changes)


@pytest.mark.parametrize('field,value', [
    ('raw_input_paths', ['/operator/other.bam']), ('source_path', '/operator/other.bam'),
    ('source_sha256', 'c' * 64), ('source_index_path', '/operator/other.bai'),
    ('source_index_sha256', 'c' * 64),
])
def test_context_cannot_supply_registered_source_identities(field, value):
    with pytest.raises(WebConfigurationError):
        context(**{field: value})


def test_operator_loader_and_public_choices_keep_scientific_values_private(tmp_path):
    selected = context(default=True)
    value = dict(workspace_root='workspace', default_profile_id='test', model_profiles=[
        dict(profile_id='test', provider_id='scripted', model_id='scripted/test')], input_sets=[
        dict(input_set_id=selected.input_set_id, label=selected.label, execution_inputs=selected.inputs(),
             bam_companion=True, bam_default=True)])
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(value))
    config = load_web_configuration(path)
    assert config.input_sets == (selected,)
    assert selected.choice() == dict(input_set_id='selected', display_label=selected.label,
                                    bam_companion=True, bam_default=True)
    assert '/operator' not in json.dumps(selected.choice()) and 'sha256' not in json.dumps(selected.choice())


def test_missing_source_fails_without_generation_provider_or_scientific_changes(tmp_path):
    backend = harness(tmp_path)
    selected = context()
    with TestClient(create_app(backend.service, input_sets=(selected,))) as client:
        assert client.get('/api/v1/input-sets').json() == {'choices': [selected.choice()]}
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        before = backend.service._application.sessions.load('session')
        response = client.post('/api/v1/sessions/session/turns', json=submission(input_set_id='selected'))
        assert response.status_code == 400 and response.json()['error']['code'] == 'BAM_SOURCE_REQUIRED'
        assert '/operator' not in response.text
        assert backend.service._application.sessions.load('session') == before
        assert not backend.models and not backend.science_calls


def test_uploaded_bam_uses_existing_registration_and_context_binding(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application._workspace.root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root, max_bytes=1024)
    selected = context()
    matrix = ScientificInputSet('matrix', 'Human matrix', {'species': 'human'}, h5ad_companion=True)
    with TestClient(create_app(backend.service, uploads=uploads, input_sets=(selected, matrix))) as client:
        public = client.put('/api/v1/uploads/bam', params={'filename': 'sample.bam', 'input_type': 'bam'},
            content=b'uploaded byte source', headers={'Content-Type': 'application/octet-stream'}).json()
        record = uploads.resources.load(public['resource_id'])
        binding = uploads.resolve(record.resource_id, scientific_inputs=selected.inputs())
        assert binding.resource_id == record.resource_id and binding.record_sha256 == record.record_sha256
        assert binding.execution_inputs['source_sha256'] == record.source_sha256
        assert binding.execution_inputs['raw_input_paths'] == (record.source_path,)
        assert all(binding.execution_inputs[key] == value for key, value in selected.inputs().items())
        assert not backend.models and not backend.science_calls
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        before = backend.service._application.sessions.load('session')
        for changes in (dict(input_set_id='matrix'), dict(input_set_id='selected', epizoo_resource_id='other')):
            response = client.post('/api/v1/sessions/session/turns',
                json=submission(resource_id=record.resource_id, **changes))
            assert response.status_code == 400 and response.json()['error']['code'] == 'INTERACTIVE_INPUT_INVALID'
        assert backend.service._application.sessions.load('session') == before
        assert not backend.models and not backend.science_calls
