"""Ordinary Web factories use selected transports without external requests."""
import json
import sys
from types import ModuleType, SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.web.app import create_app
from agent.web.config import build_interactive_application, load_web_configuration
from helpers import wait_turn


PROFILES = [
    dict(profile_id='openrouter-primary', provider_id='openrouter',
         model_id='openai/gpt-5.6-sol', display_label='Development model',
         request_timeout_seconds=90),
    dict(profile_id='groq-primary', provider_id='groq',
         model_id='openai/gpt-oss-120b', display_label='Groq model'),
    dict(profile_id='openrouter-future', provider_id='openrouter',
         model_id='compatible/future-model', display_label='Third configured model'),
    dict(profile_id='disabled', provider_id='openrouter', model_id='compatible/disabled',
         enabled=False),
    dict(profile_id='incapable', provider_id='openrouter', model_id='compatible/incapable',
         supports_structured_output=False),
]


QWEN_PROFILE = dict(profile_id='alibaba-qwen3-8-max', provider_id='alibaba',
                    model_id='qwen3.8-max', display_label='Qwen3.8-Max · Alibaba Cloud',
                    request_timeout_seconds=120)


def configured_application(tmp_path, monkeypatch, *, fail_chat=False, include_alibaba=False):
    initialized, requests, owner_calls = [], [], []
    decision = json.dumps(dict(turn_schema_version=1,
                              decision=dict(kind='clarify', reason='ambiguous_subject')))

    class Client:
        def __init__(self, **kwargs):
            initialized.append(kwargs)
            self.endpoint = kwargs['base_url']
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_completion))
            self.responses = SimpleNamespace(create=self.response)

        def chat_completion(self, **kwargs):
            requests.append(('chat', kwargs))
            if fail_chat:
                raise RuntimeError('provider-secret-sentinel /private/operator/credential')
            return dict(model=kwargs['model'], choices=[dict(finish_reason='stop',
                        message=dict(role='assistant', content=decision, refusal=None, tool_calls=None,
                                     reasoning_content='PRIVATE_REASONING_SENTINEL'))])

        def response(self, **kwargs):
            requests.append(('responses', kwargs))
            return dict(status='completed', output_text=decision, error=None, output=[])

    sdk = ModuleType('openai')
    sdk.OpenAI = Client
    monkeypatch.setitem(sys.modules, 'openai', sdk)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'injected-openrouter-test-key')
    monkeypatch.setenv('GROQ_API_KEY', 'injected-groq-test-key')
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'injected-alibaba-test-key')
    monkeypatch.setenv('DASHSCOPE_BASE_URL', 'https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1')

    # Scientific dispatch is prohibited for these transport/selection tests.
    from dataclasses import replace
    from agent.orchestration import ToolRegistry, build_default_tool_registry

    def forbidden(**arguments):
        owner_calls.append(arguments)
        raise AssertionError('A model-selection test attempted scientific execution.')

    default = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(default.get(name), function=forbidden)
                                  for name in default.names()))
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(dict(workspace_root=str(tmp_path / 'workspace'),
                                   default_profile_id='groq-primary',
                                   model_profiles=PROFILES + ([QWEN_PROFILE] if include_alibaba else []))))
    configuration = load_web_configuration(path)
    application = build_interactive_application(configuration, registry=registry)
    return application, initialized, requests, owner_calls


def test_catalog_and_history_do_not_construct_or_complete_providers(tmp_path, monkeypatch):
    application, initialized, requests, owners = configured_application(tmp_path, monkeypatch)
    with TestClient(create_app(application)) as client:
        choices = client.get('/api/v1/models').json()['choices']
        assert [choice['profile_id'] for choice in choices] == [
            'openrouter-primary', 'groq-primary', 'openrouter-future']
        assert [choice['profile_id'] for choice in choices if choice['is_default']] == ['groq-primary']
        assert choices[0]['provider_id'] == 'openrouter'
        assert choices[0]['model_id'] == 'openai/gpt-5.6-sol'
        assert client.get('/').status_code == 200
        assert client.get('/api/v1/health').json() == {'status': 'ok'}
        assert client.get('/api/v1/resources').json() == {'enabled': False, 'choices': []}
        assert client.post('/api/v1/sessions', json={'session_id': 'selection'}).status_code == 201
        assert client.get('/api/v1/sessions/selection').json()['turns'] == []
        assert initialized == requests == owners == []


def test_qwen_uses_normal_web_selection_interpretation_and_saved_history(tmp_path, monkeypatch, caplog):
    application, initialized, requests, owners = configured_application(
        tmp_path, monkeypatch, include_alibaba=True)
    with TestClient(create_app(application)) as client:
        choices = client.get('/api/v1/models').json()['choices']
        assert [item['profile_id'] for item in choices] == [
            'openrouter-primary', 'groq-primary', 'openrouter-future', 'alibaba-qwen3-8-max']
        assert [item['profile_id'] for item in choices if item['is_default']] == ['groq-primary']
        assert initialized == requests == owners == []
        client.post('/api/v1/sessions', json={'session_id': 'qwen-selection'})
        for index in range(2):
            accepted = client.post('/api/v1/sessions/qwen-selection/turns', json=dict(
                turn_id=f'qwen-{index}', expected_generation=0,
                utterance='Please clarify the intended analysis.', profile_id=QWEN_PROFILE['profile_id']))
            assert accepted.status_code == 202
            view = wait_turn(client, f'qwen-{index}', session='qwen-selection')
            assert view['status'] == 'clarification' and view['profile_id'] == QWEN_PROFILE['profile_id']
        history = client.get('/api/v1/sessions/qwen-selection').json()
        assert history['generation'] == 0 and history['active_revision_id'] is None
        assert len(history['turns']) == len(initialized) == len(requests) == 2
        assert all(item['model'] == 'qwen3.8-max' and item['extra_body'] == {'enable_thinking': False}
                   and item['response_format']['json_schema']['strict'] for _, item in requests)
        assert all(item['base_url'].startswith('https://workspace.ap-southeast-1.maas.') for item in initialized)
        assert 'PRIVATE_REASONING_SENTINEL' not in json.dumps(history) + caplog.text
        assert 'injected-alibaba-test-key' not in json.dumps(history) + caplog.text
        before = len(requests)
        assert client.get('/api/v1/sessions/qwen-selection').json() == history
        assert len(requests) == before and owners == []


def test_selected_profiles_use_real_default_transport_factories_without_fallback(tmp_path, monkeypatch):
    application, initialized, requests, owners = configured_application(tmp_path, monkeypatch)
    with TestClient(create_app(application)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'selection'})
        bodies = []
        for turn, profile in enumerate(PROFILES[:3]):
            body = dict(turn_id=f'turn-{turn}', expected_generation=0,
                        utterance='Please clarify the intended analysis.', profile_id=profile['profile_id'])
            bodies.append(body)
            accepted = client.post('/api/v1/sessions/selection/turns', json=body)
            assert accepted.status_code == 202, accepted.text
            view = wait_turn(client, body['turn_id'], session='selection')
            assert view['status'] == 'clarification' and view['profile_id'] == profile['profile_id']
            assert view['revision_id'] is None
        assert [kind for kind, _ in requests] == ['chat', 'responses', 'chat']
        assert [request['model'] for _, request in requests] == [profile['model_id'] for profile in PROFILES[:3]]
        assert [request['timeout'] for _, request in requests] == [90, 60, 60]
        assert [item['base_url'] for item in initialized] == [
            'https://openrouter.ai/api/v1', 'https://api.groq.com/openai/v1', 'https://openrouter.ai/api/v1']
        assert all(item['max_retries'] == 0 for item in initialized)
        history = client.get('/api/v1/sessions/selection').json()
        assert history['generation'] == 0 and history['active_revision_id'] is None
        assert [turn['profile_id'] for turn in history['turns']] == [profile['profile_id'] for profile in PROFILES[:3]]
        before = (len(initialized), len(requests))
        assert client.post('/api/v1/sessions/selection/turns', json=bodies[0]).status_code == 202
        assert wait_turn(client, 'turn-0', session='selection')['profile_id'] == 'openrouter-primary'
        assert client.get('/api/v1/sessions/selection').json() == history
        assert before == (len(initialized), len(requests))
        assert owners == []


def test_provider_failure_keeps_selected_model_and_sanitizes_error(tmp_path, monkeypatch):
    application, initialized, requests, owners = configured_application(tmp_path, monkeypatch, fail_chat=True)
    with TestClient(create_app(application)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'selection'})
        response = client.post('/api/v1/sessions/selection/turns', json=dict(
            turn_id='failed', expected_generation=0, utterance='Please clarify.',
            profile_id='openrouter-primary'))
        assert response.status_code == 202
        view = wait_turn(client, 'failed', session='selection', terminal_status='failed')
        # Existing interpretation admission renders provider failure as a safe
        # invalid-decision clarification; adding a transport must preserve it.
        assert view['status'] == 'clarification' and view['profile_id'] == 'openrouter-primary'
        assert view['response']['clarification']['reason'] == 'invalid_decision'
        assert 'provider-secret-sentinel' not in json.dumps(view)
        assert '/private/operator' not in json.dumps(view)
        assert len(initialized) == len(requests) == 1
        assert initialized[0]['base_url'] == 'https://openrouter.ai/api/v1'
        assert owners == []


@pytest.mark.parametrize('profile_id', ['missing', 'disabled', 'incapable'])
def test_unavailable_profiles_fail_before_provider_construction(tmp_path, monkeypatch, profile_id):
    application, initialized, requests, owners = configured_application(tmp_path, monkeypatch)
    with TestClient(create_app(application)) as client:
        client.post('/api/v1/sessions', json={'session_id': 'selection'})
        response = client.post('/api/v1/sessions/selection/turns', json=dict(
            turn_id='invalid', expected_generation=0, utterance='Please clarify.', profile_id=profile_id))
        assert response.status_code == 400
        assert response.json()['error']['code'] == 'INTERACTIVE_MODEL_UNAVAILABLE'
        assert initialized == requests == owners == []
        assert client.get('/api/v1/sessions/selection').json()['turns'] == []
