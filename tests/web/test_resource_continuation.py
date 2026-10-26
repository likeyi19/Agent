"""Ordinary pending InputSet selection preserves the exact uploaded objective."""
import json
import pytest
from fastapi.testclient import TestClient
from agent.schemas.orchestration import _serialize
from test_external_fragments_execution import (fragment_source_factory, web_fragments,
    fragment_companion, upload_fragments, execute, fragment_body, IMPORT_FRAGMENTS, ScriptedModel)

OBJECTIVE = 'Process these fragments using a reviewed reference.'
ANSWER = 'Use the reviewed human reference and its declared source profile and namespace.'


class ResourceModel(ScriptedModel):
    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            if value['utterance'] == OBJECTIVE:
                decision = dict(kind='request_resources', target=IMPORT_FRAGMENTS, arguments=[])
            elif value['utterance'] == 'I am still checking the reference.':
                decision = dict(kind='clarify', reason='ambiguous_subject')
            else:
                decision = dict(kind='answer_resources', pending='@resources', choice='resource.declared', arguments=[])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        return super().complete(prompt=prompt, response_schema=response_schema)


def test_resource_reply_preserves_objective_upload_and_restart(fragment_source_factory, tmp_path):
    arguments = fragment_source_factory()
    backend = web_fragments(tmp_path, arguments, model_type=ResourceModel)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        first = execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))
        assert first['status'] == 'clarification', (first, backend.errors)
        state = backend.service._application.sessions.load('analysis')
        assert state.interactions[0].prerequisite is not None, (first, [m.calls for m in backend.models])
        assert state.interactions[0].prerequisite['field'] == 'resources'
        assert not state.revisions and not state.turns
        again = execute(client, fragment_body(source, turn='still', companion=None,
            utterance='I am still checking the reference.'))
        assert again['status'] == 'clarification', (again, backend.errors)
        result = execute(client, fragment_body(source, turn='answer', companion=None, utterance=ANSWER))
        assert result['status'] == 'succeeded', (result, backend.errors)
        run = backend.service._application.run_store.load(result['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.steps[0].resolved_arguments['namespace'] == arguments['namespace']
        assert run.steps[0].verification.passed
        before = sum(len(m.calls) for m in backend.models)
        assert execute(client, fragment_body(source, turn='answer', companion=None, utterance=ANSWER)) == result
        assert sum(len(m.calls) for m in backend.models) == before
        state = backend.service._application.sessions.load('analysis')
        assert state.generation == 1
        assert state.interactions[-1].admitted['registered_input']['resource_id'] == source['resource_id']
        assert state.interactions[-1].admitted['registered_input']['resource_choice'] == backend.input_sets[0].resource_choice()
        for model in backend.models:
            for prompt in model.calls:
                assert arguments['reference_bundle_path'] not in json.dumps(prompt)
    # Fresh facade loads the exact accepted Session without replay.
    fresh = web_fragments(tmp_path, arguments, model_type=ResourceModel)
    assert fresh.service._application.sessions.load('analysis') == state
    assert not fresh.models


@pytest.mark.parametrize('field', ['catalog', 'source', 'parameter', 'pending_digest'])
def test_persisted_resource_request_rejects_changed_capture(fragment_source_factory, tmp_path, field):
    from agent.application.session_state import AnalysisSession, SessionError
    arguments = fragment_source_factory()
    backend = web_fragments(tmp_path, arguments, model_type=ResourceModel)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        assert execute(client, fragment_body(source, companion=None,
            utterance=OBJECTIVE))['status'] == 'clarification'
    state = _serialize(backend.service._application.sessions.load('analysis').to_dict())
    record = state['interactions'][0]
    if field == 'catalog':
        record['admitted']['resource_catalog'][0]['label'] = 'Changed resource'
    elif field == 'source':
        record['admitted']['inputs']['source_path'] = '/invented/source'
    elif field == 'parameter':
        record['admitted']['inputs']['namespace'] = 'unrequested'
    else:
        record['prerequisite']['resource_context_sha256'] = 'f' * 64
    with pytest.raises(SessionError):
        AnalysisSession.from_dict(state)


@pytest.mark.parametrize('problem', ['wrong_source', 'changed_catalog', 'invalid_choice', 'conflicting_choice', 'stale_generation'])
def test_resource_answer_rejects_changed_scope(fragment_source_factory, tmp_path, problem):
    from dataclasses import replace
    from agent.web.app import create_app
    class NegativeModel(ResourceModel):
        def complete(self, *, prompt, response_schema):
            raw = super().complete(prompt=prompt, response_schema=response_schema)
            parsed = json.loads(raw)
            if problem == 'invalid_choice' and parsed.get('decision', {}).get('kind') == 'answer_resources':
                parsed['decision']['choice'] = 'resource.invented'
            return json.dumps(parsed)
    arguments = fragment_source_factory()
    first_choice = fragment_companion(arguments)
    other_choice = fragment_companion(arguments, input_set_id='other')
    backend = web_fragments(tmp_path, arguments, companions=(first_choice, other_choice), model_type=NegativeModel)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        first = execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))
        assert first['status'] == 'clarification'
        if problem == 'wrong_source':
            other = fragment_source_factory(rows=[b'chr1\t20\t40\tOTHER\t9\n'])
            source = upload_fragments(client, other, upload_id='other')
        body = fragment_body(source, turn='answer', companion='other' if problem == 'conflicting_choice' else None,
                             utterance=ANSWER, generation=1 if problem == 'stale_generation' else 0)
        if problem == 'stale_generation':
            assert client.post('/api/v1/sessions/analysis/turns', json=body).status_code == 409
        elif problem != 'changed_catalog':
            result = execute(client, body)
            assert result['status'] == 'failed', (result, backend.errors)
            assert result['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
    if problem == 'changed_catalog':
        changed = replace(first_choice, label='Changed reviewed definition')
        with TestClient(create_app(backend.service, uploads=backend.uploads, input_sets=(changed, other_choice))) as client:
            result = execute(client, body)
            assert result['status'] == 'failed'
            assert result['error']['code'] == 'INTERACTIVE_INTENT_BINDING_INVALID'
    state = backend.service._application.sessions.load('analysis')
    assert not state.revisions and state.generation == 0
    assert not any(i.admitted and i.admitted.get('kind') == 'execute' for i in state.interactions)


def test_no_catalog_explains_operator_prerequisite_without_losing_objective(fragment_source_factory, tmp_path):
    backend = web_fragments(tmp_path, fragment_source_factory(), companions=(), model_type=ResourceModel)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, fragment_source_factory())
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        view = execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))
        assert view['status'] == 'clarification'
        assert 'operator' in view['response']['text']
        state = backend.service._application.sessions.load('analysis')
        assert state.interactions[0].utterance == OBJECTIVE
        assert state.interactions[0].prerequisite['field'] == 'resources'


def test_resource_reply_runs_existing_four_owner_chain(tmp_path, monkeypatch):
    from test_external_fragments_execution import DownstreamModel, downstream_inputs
    from rich_helpers import ScientificWork
    monkeypatch.setenv('AGENT_QC_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_MATRIX_BEDTOOLS', '/usr/bin/bedtools')
    monkeypatch.setenv('AGENT_QC_ALLOW_SYNTHETIC', '1')
    monkeypatch.delenv('AGENT_QC_RESOURCE_CATALOG', raising=False)
    class Chain(ResourceModel, DownstreamModel):
        def complete(self, *, prompt, response_schema):
            if 'turn_schema_version' in json.loads(prompt):
                return ResourceModel.complete(self, prompt=prompt, response_schema=response_schema)
            return DownstreamModel.complete(self, prompt=prompt, response_schema=response_schema)
    arguments = downstream_inputs(tmp_path)
    backend = web_fragments(tmp_path, arguments, model_type=Chain)
    work = ScientificWork()
    try:
        with TestClient(backend.app) as client:
            source = upload_fragments(client, arguments)
            client.post('/api/v1/sessions', json={'session_id': 'analysis'})
            assert execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))['status'] == 'clarification'
            assert not work.snapshot()
            result = execute(client, fragment_body(source, turn='answer', companion=None, utterance=ANSWER), timeout=90)
            assert result['status'] == 'succeeded', (result, backend.errors)
            run = backend.service._application.run_store.load(result['run_id'])
            assert run.request.prompt == OBJECTIVE
            assert len(run.steps) == 4
            for kind in ('fragments', 'qc', 'selection', 'matrix'):
                assert work.snapshot()['production.' + kind] == 1
                assert work.snapshot()['owner.' + kind] == 1
            assert all(s.verification.artifact_authority['schema_version'] == 2 for s in run.steps)
    finally:
        work.close()


def test_reference_choice_and_spanned_profile_namespace_declarations(fragment_source_factory, tmp_path):
    from agent.web.config import ScientificInputSet
    utterance = 'Use the reviewed reference, 10x ATAC profile, and namespace research-a.'
    class Declarations(ResourceModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt)
            if p.get('utterance') == utterance:
                declarations = []
                for name, literal, value in [('source_profile', '10x ATAC', '10x-atac-fragments.v1'),
                                              ('namespace', 'research-a', None)]:
                    start = utterance.index(literal)
                    item = dict(tool=IMPORT_FRAGMENTS, argument=name, literal=literal,
                                start=start, end=start + len(literal))
                    if value is not None:
                        item['value'] = value
                    declarations.append(item)
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_resources',
                    pending='@resources', choice='resource.declared', arguments=declarations)))
            return super().complete(prompt=prompt, response_schema=response_schema)
    arguments = fragment_source_factory()
    values = fragment_companion(arguments).inputs()
    for name in ('source_profile', 'namespace'):
        values.pop(name)
    choice = ScientificInputSet('declared', 'Reviewed reference only', values, fragments_companion=True)
    backend = web_fragments(tmp_path, arguments, companions=(choice,), model_type=Declarations)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        assert execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))['status'] == 'clarification'
        result = execute(client, fragment_body(source, turn='answer', companion=None, utterance=utterance))
        assert result['status'] == 'succeeded', (result, backend.errors)
        step, = backend.service._application.run_store.load(result['run_id']).steps
        assert step.resolved_arguments['source_profile'] == '10x-atac-fragments.v1'
        assert step.resolved_arguments['namespace'] == 'research-a'


def test_resource_answer_can_continue_to_missing_namespace_without_reupload(fragment_source_factory, tmp_path):
    from agent.web.config import ScientificInputSet
    utterance = 'Use namespace research-b.'
    class Namespace(ResourceModel):
        def complete(self, *, prompt, response_schema):
            p = json.loads(prompt)
            if p.get('utterance') == utterance:
                literal = 'research-b'
                start = utterance.index(literal)
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_parameters',
                    pending='@parameters', arguments=[dict(tool=IMPORT_FRAGMENTS, argument='namespace',
                        literal=literal, start=start, end=start + len(literal))])))
            raw = super().complete(prompt=prompt, response_schema=response_schema)
            parsed = json.loads(raw)
            if parsed.get('schema_version') == 4:
                # This scripted Planner omits unavailable inputs; it does not
                # fabricate a request handle for the missing declaration.
                offered = p['catalog']['request_inputs']
                for step in parsed['decision']['steps']:
                    step['sources'] = [s for s in step['sources']
                        if s['source']['input'] in offered]
                return json.dumps(parsed)
            return raw
    arguments = fragment_source_factory()
    values = fragment_companion(arguments).inputs()
    values.pop('namespace')
    choice = ScientificInputSet('declared', 'Reviewed reference and source profile', values, fragments_companion=True)
    backend = web_fragments(tmp_path, arguments, companions=(choice,), model_type=Namespace)
    with TestClient(backend.app) as client:
        source = upload_fragments(client, arguments)
        client.post('/api/v1/sessions', json={'session_id': 'analysis'})
        assert execute(client, fragment_body(source, companion=None, utterance=OBJECTIVE))['status'] == 'clarification'
        answer = execute(client, dict(turn_id='reference', expected_generation=0, utterance=ANSWER))
        assert answer['status'] == 'clarification', (answer, backend.errors)
        state = backend.service._application.sessions.load('analysis')
        assert state.interactions[-1].prerequisite['field'] == 'parameters'
        assert state.interactions[-1].prerequisite['missing'] == ('namespace',)
        result = execute(client, dict(turn_id='namespace', expected_generation=0, utterance=utterance))
        assert result['status'] == 'succeeded', (result, backend.errors)
        run = backend.service._application.run_store.load(result['run_id'])
        assert run.request.prompt == OBJECTIVE
        assert run.steps[0].resolved_arguments['namespace'] == 'research-b'
        assert run.steps[0].resolved_arguments['source_path'] == backend.service.resources.load(source['resource_id']).source_path
