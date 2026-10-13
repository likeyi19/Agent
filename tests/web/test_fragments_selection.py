"""Explicit fragments input choices reuse operator-owned typed InputSets."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import _fragments_selection, create_app
from agent.web.config import ScientificInputSet

from helpers import harness, submission, wait_turn


def context(identifier='reviewed', *, default=False, **inputs):
    values = dict(reference_bundle_path='/operator/reviewed/reference.json',
        reference_bundle_sha256='a' * 64, source_profile='10x-atac-fragments.v1',
        namespace=identifier, source_selection='full_export') | inputs
    return ScientificInputSet(identifier, identifier.title() + ' fragments declarations', values,
                              fragments_companion=True, fragments_default=default)


@pytest.mark.parametrize('defaults', [(), (context('first', default=True),),
    (context('first', default=True), context('second', default=True))])
def test_explicit_context_has_priority_over_any_default_configuration(defaults):
    explicit = context('selected', reference_bundle_sha256='b' * 64)
    assert _fragments_selection(explicit, defaults + (explicit,)) == (explicit.inputs(), None)


def test_unique_configured_default_supplies_only_declared_values():
    chosen = context('default', default=True)
    other = context('other')
    assert _fragments_selection(None, (chosen, other)) == (chosen.inputs(), None)
    assert set(_fragments_selection(None, (chosen,))[0]) == set(chosen.execution_inputs)


@pytest.mark.parametrize('contexts,code', [
    ((), None),
    ((context('nondefault'),), None),
    ((context('first', default=True), context('second', default=True)), 'FRAGMENTS_RESOURCE_AMBIGUOUS'),
])
def test_missing_or_ambiguous_defaults_remain_available_for_interpretation(contexts, code):
    assert _fragments_selection(None, contexts) == ({}, code)


@pytest.mark.parametrize('field', ['reference_bundle_path', 'reference_bundle_sha256', 'source_profile', 'namespace'])
def test_genuinely_missing_declarations_do_not_use_other_context_defaults(field):
    values = context('chosen').inputs()
    del values[field]
    incomplete = ScientificInputSet('chosen', 'Chosen declarations', values, fragments_companion=True)
    assert _fragments_selection(incomplete, (incomplete, context('default', default=True))) == (values, None)


def test_h5ad_context_does_not_become_fragments_context():
    h5ad = ScientificInputSet('mouse', 'Mouse matrix', {'species': 'mouse'}, h5ad_companion=True)
    with pytest.raises(InteractiveBoundaryError):
        _fragments_selection(h5ad, (context('default', default=True),))


def test_safe_choice_and_missing_source_do_not_mutate_session_or_construct_provider(tmp_path):
    backend = harness(tmp_path)
    chosen = context()
    with TestClient(create_app(backend.service, input_sets=(chosen,))) as client:
        listing = client.get('/api/v1/input-sets')
        assert listing.json() == {'choices': [chosen.choice()]}
        assert '/operator' not in listing.text and 'sha256' not in listing.text
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        before = backend.service._application.sessions.load('session')
        response = client.post('/api/v1/sessions/session/turns',
            json=submission(input_set_id=chosen.input_set_id))
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'FRAGMENTS_SOURCE_REQUIRED'
        assert '/operator' not in response.text
        assert backend.service._application.sessions.load('session') == before
        assert not backend.models and not backend.science_calls


@pytest.fixture
def uploaded_fragments_web(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application._workspace.root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root, max_bytes=1024)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        yield client, backend, uploads


def test_http_pair_uses_one_registration_with_exact_private_identities(uploaded_fragments_web):
    client, backend, uploads = uploaded_fragments_web
    source, index = b'uploaded external source bytes', b'uploaded paired TBI bytes'
    params = dict(filename='fragments.tsv.gz', input_type='external_fragments',
                  index_filename='fragments.tsv.gz.tbi', source_size=len(source))
    response = client.put('/api/v1/uploads/paired', params=params,
        content=source + index, headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 201, response.text
    public = response.json()
    assert public['input_type'] == 'external_fragments' and public['has_source_index'] is True
    record = uploads.resources.load(public['resource_id'])
    assert Path(record.source_path).read_bytes() == source
    assert Path(record.source_index['path']).read_bytes() == index
    assert client.get('/api/v1/resources').json()['choices'] == [public]
    assert 'sha256' not in response.text and str(uploads.root) not in response.text
    retried = client.put('/api/v1/uploads/paired', params=params,
        content=source + index, headers={'Content-Type': 'application/octet-stream'})
    assert retried.status_code == 201 and retried.json() == public
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    missing = client.post('/api/v1/sessions/session/turns',
        json=submission(resource_id=public['resource_id'], input_set_id=None, utterance='Clarify what is needed to process this source.'))
    assert missing.status_code == 202, missing.text
    view = wait_turn(client)
    assert view['status'] == 'clarification' and view['run_id'] is None
    state = backend.service._application.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert state.interactions[0].submission['registered_input']['record_sha256'] == record.record_sha256
    assert len(backend.models) == 1 and len(backend.models[0].calls) == 1
    assert not backend.science_calls


@pytest.mark.parametrize('params,status', [
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'index_filename': 'source.tbi'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'source_size': '2'}, 422),
    ({'filename': 'source.h5ad', 'index_filename': 'source.tbi', 'source_size': '2'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'fastq'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'index_filename': 'source.tbi', 'source_size': '0'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'index_filename': 'source.tbi', 'source_size': '-1'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'index_filename': 'source.tbi', 'source_size': '\u0662'}, 422),
    ({'filename': 'source.tsv.gz', 'input_type': 'external_fragments', 'index_filename': 'source.tbi', 'source_size': '1025'}, 413),
    ([('filename', 'source.tsv'), ('input_type', 'external_fragments'), ('input_type', 'h5ad')], 422),
    ({'filename': 'source.tsv', 'input_type': 'external_fragments', 'source_index_path': '/private/source.tbi'}, 422),
])
def test_http_incomplete_unsafe_or_oversized_pair_metadata_publishes_nothing(uploaded_fragments_web, params, status):
    client, backend, uploads = uploaded_fragments_web
    response = client.put('/api/v1/uploads/invalid', params=params, content=b'bytes',
                          headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == status, response.text
    assert client.get('/api/v1/resources').json()['choices'] == []
    assert not tuple(uploads.completed.iterdir()) and not tuple(uploads.staging.iterdir())
    assert not backend.models and not backend.science_calls
    assert '/private' not in response.text
