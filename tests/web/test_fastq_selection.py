"""FASTQ selections stay typed, private and independent of operation planning."""
from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission
from agent.application.uploads import H5ADUploadAdmission
from agent.web.app import _fastq_selection, create_app
from agent.web.config import ScientificInputSet, WebConfigurationError, load_web_configuration
from agent.tools.data import _raw_fastq as owner

from helpers import harness, submission
from test_config import configuration_json, write_configuration


def context(identifier='reviewed', *, default=False):
    return ScientificInputSet(identifier, 'Qualified FASTQ context',
        dict(species='human', source_genome_assembly='hg38',
            library_context_path='/operator/qualified/library.json', library_context_sha256='a' * 64,
            reference_bundle_path='/operator/qualified/reference.json', reference_bundle_sha256='b' * 64),
        fastq_companion=True, fastq_default=default)


@pytest.mark.parametrize('defaults', [(), (context('one', default=True),),
    (context('one', default=True), context('two', default=True))])
def test_explicit_fastq_context_wins_without_hidden_replacement(defaults):
    selected = context('selected')
    assert _fastq_selection(selected, defaults + (selected,)) == (selected.inputs(), None)


@pytest.mark.parametrize('contexts,expected', [((), ({}, None)),
    ((context(),), ({}, None)), ((context(default=True),), (context().inputs(), None)),
    ((context('one', default=True), context('two', default=True)), ({}, 'FASTQ_RESOURCE_AMBIGUOUS'))])
def test_only_unique_explicit_defaults_are_supplied_and_inspection_has_no_producer_requirement(contexts, expected):
    assert _fastq_selection(None, contexts) == expected


def test_invalid_explicit_companion_never_falls_back_to_default():
    wrong = ScientificInputSet('bam', 'BAM context', {}, bam_companion=True)
    with pytest.raises(InteractiveBoundaryError):
        _fastq_selection(wrong, (context(default=True),))


@pytest.mark.parametrize('changes', [dict(fastq_companion='true'), dict(fastq_default=True),
    dict(fastq_companion=True, fastq_default=1), dict(fastq_companion=True, bam_companion=True),
    dict(fastq_companion=True, h5ad_companion=True), dict(fastq_companion=True, fragments_companion=True)])
def test_companion_roles_and_defaults_require_explicit_boolean_contract(changes):
    with pytest.raises(WebConfigurationError):
        ScientificInputSet('fastq', 'FASTQ', {}, **changes)


@pytest.mark.parametrize('field', ['source_path', 'source_sha256', 'raw_input_paths',
    'source_index_path', 'source_index_sha256', 'fastq_layout', 'raw_assay'])
def test_operator_companion_cannot_redefine_registered_member_or_role_identity(field):
    with pytest.raises(WebConfigurationError):
        ScientificInputSet('fastq', 'FASTQ', {field: 'operator-value'}, fastq_companion=True)


def test_safe_context_choice_and_missing_collection_preserve_state(tmp_path):
    backend = harness(tmp_path)
    selected = context()
    with TestClient(create_app(backend.service, input_sets=(selected,))) as client:
        choice = client.get('/api/v1/input-sets')
        assert choice.json() == {'choices': [dict(input_set_id='reviewed',
            display_label='Qualified FASTQ context', fastq_companion=True)]}
        assert '/operator' not in choice.text and 'sha256' not in choice.text
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        before = backend.service._application.sessions.load('session')
        response = client.post('/api/v1/sessions/session/turns',
            json=submission(input_set_id='reviewed'))
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'FASTQ_SOURCE_REQUIRED'
        assert backend.service._application.sessions.load('session') == before
        assert not backend.models and not backend.science_calls


def test_strict_operator_configuration_loads_only_explicit_fastq_context_flags(tmp_path):
    selected = context(default=True)
    values = configuration_json()
    values['input_sets'] = [dict(input_set_id=selected.input_set_id, label=selected.label,
        execution_inputs=selected.inputs(), fastq_companion=True, fastq_default=True)]
    configuration = load_web_configuration(write_configuration(tmp_path, values))
    assert configuration.input_sets == (selected,)
    assert configuration.input_sets[0].choice() == dict(input_set_id='reviewed',
        display_label='Qualified FASTQ context', fastq_companion=True, fastq_default=True)


@pytest.fixture
def supplied(tmp_path):
    backend = harness(tmp_path)
    root = backend.service._application._workspace.root / 'uploads'
    root.mkdir()
    backend.service.resources = LocalResourceAdmission(backend.service._application._workspace,
        approved_source_roots=(root,), registry=backend.service._application.registry)
    uploads = H5ADUploadAdmission(backend.service.resources, root, max_bytes=1024)
    with TestClient(create_app(backend.service, uploads=uploads)) as client:
        yield client, backend, uploads


def upload(client, role='R1', **changes):
    params = dict(filename='Misleading filename.dat', input_type='fastq', library_id='Research library',
        fastq_layout=owner.FastqLayout.A.value, role=role, lane='001', chunk='001', compression='plain') | changes
    return client.put('/api/v1/uploads/member-' + role, params=params,
        content=b'@read\nACGT\n+\nIIII\n', headers={'Content-Type': 'application/octet-stream'})


def test_http_partial_members_are_discoverable_only_as_members_then_complete_atomic_choice(supplied):
    client, backend, uploads = supplied
    member = upload(client).json()
    catalog = client.get('/api/v1/resources').json()
    assert catalog == dict(enabled=True, choices=[], fastq_members=[member])
    assert member['fastq_attribution']['role'] == 'R1'
    assert 'sha256' not in str(catalog) and str(uploads.root) not in str(catalog)
    incomplete = client.post('/api/v1/uploads/fastq-collections',
        json=dict(collection_id='library', label='Research library', member_ids=[member['resource_id']]))
    assert incomplete.status_code == 400
    assert incomplete.json()['error']['code'] == 'FASTQ_COLLECTION_INCOMPLETE'
    assert 'R2' in incomplete.json()['error']['message'] and 'R3' in incomplete.json()['error']['message']
    assert client.get('/api/v1/resources').json()['choices'] == []
    members = [member, upload(client, 'R2').json(), upload(client, 'R3').json()]
    complete = client.post('/api/v1/uploads/fastq-collections', json=dict(collection_id='library',
        label='Research library', member_ids=[value['resource_id'] for value in members]))
    assert complete.status_code == 201, complete.text
    choice = complete.json()
    assert choice['fastq_collection'] is True and choice['input_type'] == 'fastq'
    assert client.get('/api/v1/resources').json()['choices'] == [choice]
    client.post('/api/v1/sessions', json={'session_id': 'session'})
    direct_member = client.post('/api/v1/sessions/session/turns',
        json=submission(resource_id=member['resource_id'], input_set_id=None))
    assert direct_member.status_code == 400 and not backend.models and not backend.science_calls
    assert not backend.service._application.sessions.load('session').revisions
    reopened = H5ADUploadAdmission(LocalResourceAdmission(uploads.resources._workspace,
        approved_source_roots=(uploads.root,)), uploads.root)
    assert reopened.choices() == (choice,)
    assert reopened.fastq_members() == uploads.fastq_members()


@pytest.mark.parametrize('field', ['library_id', 'fastq_layout', 'role', 'lane', 'chunk', 'compression'])
def test_http_requires_all_typed_member_attribution_before_registration(supplied, field):
    client, backend, uploads = supplied
    params = dict(filename='file.fastq', input_type='fastq', library_id='Research library',
        fastq_layout=owner.FastqLayout.A.value, role='R1', lane='001', chunk='001', compression='plain')
    del params[field]
    response = client.put('/api/v1/uploads/invalid', params=params, content=b'bytes',
        headers={'Content-Type': 'application/octet-stream'})
    assert response.status_code == 422
    assert uploads.choices() == uploads.fastq_members() == ()
    assert not backend.models and not backend.science_calls


@pytest.mark.parametrize('changes', [dict(role='I2'), dict(fastq_layout='unsupported'),
    dict(lane='00x'), dict(chunk='-01'), dict(compression='zip'), dict(library_id='../source')])
def test_http_rejects_unsupported_or_conflicting_typed_roles(supplied, changes):
    client, _, uploads = supplied
    response = upload(client, **changes)
    assert response.status_code == 400
    assert response.json()['error']['code'] == 'FASTQ_COLLECTION_INVALID'
    assert uploads.choices() == uploads.fastq_members() == ()


@pytest.mark.parametrize('changes', [dict(member_ids=[]), dict(member_ids=['local-' + '0' * 64]),
    dict(member_ids=['/private/source.fastq']), dict(source_path='/private/file'),
    dict(collection_id='../unsafe'), dict(label='/private/source')])
def test_http_collection_request_cannot_inject_paths_or_complete_missing_records(supplied, changes):
    client, backend, uploads = supplied
    member = upload(client).json()
    body = dict(collection_id='library', label='Research library', member_ids=[member['resource_id']]) | changes
    response = client.post('/api/v1/uploads/fastq-collections', json=body)
    assert response.status_code in (400, 404, 422), response.text
    assert '/private' not in response.text
    assert client.get('/api/v1/resources').json()['choices'] == []
    assert not backend.models and not backend.science_calls
