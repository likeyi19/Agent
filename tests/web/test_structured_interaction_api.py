"""HTTP selection and parameter transport share ordinary immutable submissions."""
from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

import helpers
from helpers import submission, wait_turn
from test_h5ad_composition_api import web_composition
from test_upload_api import attach


def test_catalog_and_source_options_are_read_only(tmp_path):
    backend, uploads, app = web_composition(tmp_path)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        record = attach(client, backend.source.read_bytes()).json()
        view = client.get('/api/v1/parameters?tool=run_snapatac2').json()
        assert any(row['tool'] == 'run_snapatac2' for row in view['tools'])
        function = client.get('/api/v1/parameters?tool=run_scanpy&function=pp.scale')
        assert function.status_code == 200, function.text
        assert 'zero_center' in {row['argument'] for row in function.json()['tools'][0]['kwargs']}
        assert client.get('/api/v1/parameters?tool=run_scanpy&function=unknown').status_code == 400
        offered = client.get('/api/v1/sessions/session/interaction-options', params={'resource_ids': record['resource_id']})
        assert offered.status_code == 200, offered.text
        assert offered.json()['sources'][0]['label'] == record['label']
        assert offered.json()['slots'] == [dict(source='source.0', values=['X'])]
        assert not backend.models and not backend.science_calls
        assert str(tmp_path) not in function.text and str(tmp_path) not in offered.text


def test_exact_accepted_result_identity_reaches_existing_input_results(tmp_path, monkeypatch):
    backend, uploads, app = web_composition(tmp_path)
    with TestClient(app) as client:
        record = attach(client, backend.source.read_bytes()).json()
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        body = submission(resource_id=record['resource_id'], input_set_id='mouse')
        client.post('/api/v1/sessions/session/turns', json=body)
        first = wait_turn(client)
        assert first['status'] == 'succeeded', first
        options = client.get('/api/v1/sessions/session/interaction-options')
        assert options.status_code == 200, options.text
        selected, = options.json()['results']
        assert set(selected['identity']) == {'revision_id', 'output_name', 'accepted_step_sha256'}
        follow = dict(turn_id='reuse', expected_generation=1, utterance='Inspect this accepted result.',
            selected_results=[selected['identity']])
        # The fake planner cannot inspect an inspection scalar. Withhold its
        # execution; admission still must retain the exact selected operand.
        client.post('/api/v1/sessions/session/turns', json=follow)
        result = wait_turn(client, 'reuse')
        state = backend.service._application.sessions.load('session')
        assert state.interactions[-1].admitted['input_results']
        handle = state.interactions[-1].admitted['input_results'][0]
        rows = state.interactions[-1].snapshot['dialogue']['results']
        assert any(row['handle'] == handle for row in rows)
        assert state.revisions[0].revision_id == selected['identity']['revision_id']


def test_parameter_payload_validates_before_any_model_call(tmp_path):
    backend, _, app = web_composition(tmp_path)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        for override in [dict(tool='made_up', argument='value', value=1),
                         dict(tool='run_scanpy', argument='kwargs', value={'foreign_name': 2}, native_function='pp.scale')]:
            body = dict(turn_id='invalid', expected_generation=0, utterance='Run selected method.', parameter_overrides=[override])
            response = client.post('/api/v1/sessions/session/turns', json=body)
            assert response.status_code == 400, response.text
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='no-objective',
            expected_generation=0, utterance='', parameter_overrides=[dict(tool='run_scanpy',
            argument='kwargs', native_function='pp.scale', value={'max_value':2})]))
        assert response.status_code == 400
        assert not backend.models and not backend.science_calls


def test_known_clarification_uses_no_language_generation(tmp_path, monkeypatch):
    original = helpers.ScriptedModel
    class NoClarificationRewrite(original):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            assert 'clarification' not in value, 'Deterministic question called the language model'
            return super().complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(helpers, 'ScriptedModel', NoClarificationRewrite)
    backend, _, app = web_composition(tmp_path)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        client.post('/api/v1/sessions/session/turns', json=dict(turn_id='question', expected_generation=0,
            utterance='clarify parameter'))
        question = wait_turn(client, 'question')
        assert question['status'] == 'clarification'
        assert sum('turn_schema_version' in c for m in backend.models for c in m.calls) == 1
        assert not backend.science_calls


def test_global_parameter_editor_removed_and_candidate_click_is_structured():
    root = Path(__file__).resolve().parents[2] / 'src/agent/web/static'
    html = (root / 'index.html').read_text()
    js = (root / 'app.js').read_text()
    assert 'id="parameter-panel"' not in html
    assert 'id="configure-native"' not in html
    assert 'id="parameter-dialog"' in html
    assert 'parameter_overrides: overrides' in js
    assert 'Apply the selected parameters.' not in js
    assert 'body.candidate_selection = Object.freeze(candidateSelection)' in js
    assert 'Run option ${candidate.option}' not in js
    assert 'origin_turn_id: candidate.origin_turn_id, candidate_id: candidate.candidate_id' in js
    assert 'literal:' not in js and 'start:' not in js


def test_reference_click_preserves_identity_without_interpreter(tmp_path, monkeypatch):
    from test_target_resource_continuation import target_choice
    original = helpers.ScriptedModel
    class ReferenceModel(original):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                assert value['utterance'].startswith('Inspect'), 'Unexpected reference click interpretation'
                self.calls.append(value)
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='request_resources', target='inspect_scATAC', arguments=[])))
            assert 'clarification' not in value, 'Unexpected question rewrite'
            return super().complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(helpers, 'ScriptedModel', ReferenceModel)
    choice = target_choice()
    backend, uploads, app = web_composition(tmp_path, companions=(choice,))
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        source = attach(client, backend.source.read_bytes()).json()
        body = dict(turn_id='task', expected_generation=0, utterance='Inspect with selected reference.', resource_id=source['resource_id'])
        client.post('/api/v1/sessions/session/turns', json=body)
        first = wait_turn(client, 'task')
        selected, = first['clarification_choices']
        client.post('/api/v1/sessions/session/turns', json=dict(turn_id='choice', expected_generation=0,
            utterance=selected['label'], clarification_choice=dict(question_turn_id='task', handle=selected['handle'])))
        result = wait_turn(client, 'choice')
        assert result['status'] == 'succeeded', result
        state = backend.service._application.sessions.load('session')
        assert state.interactions[-1].admitted['inputs']['target_model_manifest_path'] == choice.inputs()['target_model_manifest_path']
        assert sum('turn_schema_version' in c for m in backend.models for c in m.calls) == 1


def test_new_file_answers_captured_role_without_interpreter(tmp_path, monkeypatch):
    original = helpers.ScriptedModel
    class FileModel(original):
        def complete(self, *, prompt, response_schema):
            value = json.loads(prompt)
            if 'turn_schema_version' in value:
                assert value['utterance'] == 'Inspect current file.', 'Unexpected file-role interpretation'
            return super().complete(prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(helpers, 'ScriptedModel', FileModel)
    backend, uploads, app = web_composition(tmp_path)
    with TestClient(app) as client:
        client.post('/api/v1/sessions', json={'session_id': 'session'})
        client.post('/api/v1/sessions/session/turns', json=dict(turn_id='task', expected_generation=0, utterance='Inspect current file.'))
        first = wait_turn(client, 'task')
        assert first['status'] == 'clarification' and not backend.science_calls
        role, = first['clarification_choices']
        assert role['sources'] == []
        from test_chat_attachments import upload
        uploaded = upload(client, 'matrix.h5ad', backend.source.read_bytes(), kind='h5ad').json()
        response = client.post('/api/v1/sessions/session/turns', json=dict(turn_id='file', expected_generation=0,
            utterance=uploaded['label'], attachment_ids=[uploaded['resource_id']],
            clarification_choice=dict(question_turn_id='task', handle=role['handle'] + ':' + uploaded['resource_id'])))
        assert response.status_code == 202, response.text
        result = wait_turn(client, 'file')
        assert result['status'] == 'succeeded', result
        assert backend.science_calls == ['inspect_scATAC']
        assert sum('turn_schema_version' in c for m in backend.models for c in m.calls) == 1
        state = backend.service._application.sessions.load('session')
        assert state.interactions[-1].admitted['registered_input']['members'][0]['resource_id'] == uploaded['resource_id']


def test_catalog_refresh_preserves_checked_result_identity():
    from test_static import _javascript_value, STATIC
    code = (STATIC/'app.js').read_text()
    function = code[code.index('async function refreshRequestOptions'):code.index('element("cancel-parameters").addEventListener')]
    identity = dict(revision_id='rev', output_name='counts', accepted_step_sha256='a'*64)
    script = '''(() => {
      class Node {
        constructor(tag) {this.tag=tag;this.children=[];this.checked=false;}
        append(...nodes) {this.children.push(...nodes);}
        replaceChildren(...nodes) {this.children=nodes;}
        querySelectorAll(selector) {return this.children.flatMap(n=>n.tag==='input' ? n.checked ? [n] : [] : n.querySelectorAll ? n.querySelectorAll(selector) : []);}
      }
      const document={createElement:tag=>new Node(tag),createTextNode:text=>({text})};
      const nodes={'request-result-choices':new Node('div'),'request-results-panel':new Node('details')};
      const element=id=>nodes[id];
      const selectedResourceIds=()=>[];const sessionPath=()=>'/s';
      const URLSearchParams=class {append(){} toString(){return '';}};
      const renderParameterRows=()=>{};const showError=error=>{throw error};
      const state={session:{session_id:'s',generation:1},attachments:[],optionsEpoch:0};
      const identity=IDENTITY;
      const old=document.createElement('input');old.value=JSON.stringify(identity);old.checked=true;
      nodes['request-result-choices'].append(old);
      const api=async()=>({results:[{identity,label:'Counts',tool:'run_snapatac2',current:true}],sources:[]});
      const result={values:[]};
    '''.replace('IDENTITY',json.dumps(identity)) + function + '''
      refreshRequestOptions('s').then(()=>{result.values=nodes['request-result-choices'].querySelectorAll('input:checked').map(input=>JSON.parse(input.value));});
      return result;
    })()'''
    assert _javascript_value(script) == {'values':[identity]}
