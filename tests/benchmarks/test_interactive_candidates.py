"""Offline exact-candidate availability and transport contracts."""

from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest

from agent.orchestration.planning_model import PlanningModelError
from benchmarks.interactive.candidates import (
    CANDIDATES, ChatCompletionsPlanningModel, build_candidate, get_candidate,
)


SCHEMA = {'type': 'object', 'properties': {'decision': {'type': 'string'}},
          'required': ['decision'], 'additionalProperties': False}


class FakeEndpoint:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeModels:
    def __init__(self, model_id, *, listed=None, error=None):
        self.model_id, self.listed, self.error, self.calls = model_id, listed, error, []

    def list(self, **kwargs):
        self.calls.append(('list', kwargs))
        if self.error:
            raise self.error
        return {'data': self.listed if self.listed is not None else [{'id': self.model_id}]}

    def get(self, **kwargs):
        self.calls.append(('get', kwargs))
        if self.error:
            raise self.error
        return {'name': 'models/' + self.model_id}


def fake_client(candidate, *, models=None, completion=None, error=None):
    endpoint = FakeEndpoint(completion or {
        'status': 'completed', 'output_text': '{"decision":"example"}'}, error)
    chat = FakeEndpoint(completion or {'choices': [
        {'finish_reason': 'stop', 'message': {'content': '{"decision":"example"}'}}]}, error)
    return SimpleNamespace(models=models or FakeModels(candidate.model_id), responses=endpoint,
                           interactions=endpoint, chat=SimpleNamespace(completions=chat))


def build(candidate, client, **kwargs):
    return build_candidate(candidate, client=client,
                           environment={'DASHSCOPE_BASE_URL': 'https://example.test/compatible/v1'}, **kwargs)


def test_exact_frozen_cohort_has_no_silent_model_substitution():
    assert [(c.candidate_id, c.provider_id, c.model_id) for c in CANDIDATES] == [
        ('groq-gpt-oss-120b', 'groq', 'openai/gpt-oss-120b'),
        ('groq-qwen3.8-27b', 'groq', 'qwen/qwen3.8-27b'),
        ('dashscope-deepseek-v4.1-flash', 'dashscope', 'deepseek-v4.1-flash'),
        ('gemini-3.8-flash', 'gemini', 'gemini-3.8-flash'),
        ('mistral-medium-3-5', 'mistral', 'mistral-medium-3-5'),
    ]
    with pytest.raises(FrozenInstanceError):
        CANDIDATES[0].model_id = 'latest'
    with pytest.raises(ValueError):
        get_candidate('latest')
    assert get_candidate('gemini-3.8-flash') is CANDIDATES[3]


@pytest.mark.parametrize('candidate', CANDIDATES, ids=lambda c: c.candidate_id)
def test_every_candidate_requires_availability_and_keeps_shared_schema(candidate):
    client = fake_client(candidate)
    runtime = build(candidate, client)
    assert runtime.availability_requests == runtime.completion_calls == 0
    with pytest.raises(PlanningModelError) as failure:
        runtime.complete(prompt='same Agent prompt', response_schema=SCHEMA)
    assert failure.value.code == 'CANDIDATE_MODEL_NOT_VALIDATED'
    assert runtime.completion_calls == 0
    assert runtime.validate_availability().status == 'available'
    assert runtime.validate_availability().status == 'available'
    assert len(client.models.calls) == runtime.availability_requests == 1
    assert runtime.model_id == candidate.provider_id + ':' + candidate.model_id
    assert runtime.complete(prompt='same Agent prompt', response_schema=SCHEMA) == '{"decision":"example"}'
    assert runtime.completion_calls == 1
    if candidate.adapter_family == 'gemini-interactions':
        call = client.interactions.calls[0]
        assert call['response_format']['schema'] == SCHEMA
        assert call['input'] == 'same Agent prompt'
        assert client.models.calls == [('get', {'model': candidate.model_id,
            'config': {'http_options': {'timeout': 60000}}})]
    elif candidate.adapter_family == 'openai-chat-completions':
        call = client.chat.completions.calls[0]
        assert call['response_format']['json_schema']['schema'] == SCHEMA
        assert call['response_format']['json_schema']['strict'] is True
        assert call['messages'] == [{'role': 'user', 'content': 'same Agent prompt'}]
    else:
        call = client.responses.calls[0]
        assert call['text']['format']['schema'] == SCHEMA
        assert call['input'] == 'same Agent prompt'
    assert call['model'] == candidate.model_id


def test_dashscope_requires_historical_explicit_endpoint():
    candidate = CANDIDATES[2]
    with pytest.raises(PlanningModelError) as failure:
        build_candidate(candidate, client=fake_client(candidate), environment={})
    assert failure.value.code == 'CANDIDATE_ENDPOINT_MISSING'


@pytest.mark.parametrize('endpoint', [
    'https://key:secret@example.test/v1', 'https://example.test/v1?api_key=secret',
    'https://example.test/v1#secret', 'http://example.test/v1',
])
def test_secret_bearing_or_unsafe_endpoint_is_rejected_without_echo(endpoint):
    candidate = CANDIDATES[2]
    with pytest.raises(PlanningModelError) as failure:
        build_candidate(candidate, client=fake_client(candidate),
                        environment={'DASHSCOPE_BASE_URL': endpoint})
    assert 'secret' not in str(failure.value)
    assert endpoint not in str(failure.value)


def test_missing_exact_model_is_unavailable_and_completion_is_impossible():
    candidate = CANDIDATES[1]
    client = fake_client(candidate, models=FakeModels(candidate.model_id, listed=[{'id': 'different-model'}]))
    runtime = build(candidate, client)
    availability = runtime.validate_availability()
    assert availability.status == 'unavailable'
    assert availability.code == 'CANDIDATE_MODEL_UNAVAILABLE'
    with pytest.raises(PlanningModelError):
        runtime.complete(prompt='x', response_schema=SCHEMA)
    assert runtime.completion_calls == 0
    assert client.responses.calls == []


def test_provider_listed_exact_alias_is_accepted_without_changing_request_model():
    candidate = CANDIDATES[4]
    models = FakeModels(candidate.model_id, listed=[{
        'id': 'mistral-medium-2604', 'aliases': [candidate.model_id],
        'capabilities': {'completion_chat': True},
    }])
    client = fake_client(candidate, models=models)
    runtime = build(candidate, client)
    assert runtime.validate_availability().provider_model_id == 'mistral-medium-2604'
    runtime.complete(prompt='x', response_schema=SCHEMA)
    assert client.chat.completions.calls[0]['model'] == 'mistral-medium-3-5'


def test_unique_exact_model_id_is_authoritative_over_multiple_alias_cards():
    candidate = CANDIDATES[4]
    cards = [{'id': 'alias-target-' + str(i), 'aliases': [candidate.model_id]} for i in range(8)]
    cards.insert(4, {'id': candidate.model_id})
    client = fake_client(candidate, models=FakeModels(candidate.model_id, listed=cards))
    runtime = build(candidate, client)
    availability = runtime.validate_availability()
    assert availability.status == 'available'
    assert availability.provider_model_id == candidate.model_id
    runtime.complete(prompt='same prompt', response_schema=SCHEMA)
    assert client.chat.completions.calls[0]['model'] == candidate.model_id
    assert runtime.availability_requests == runtime.completion_calls == 1


@pytest.mark.parametrize('cards', [
    [{'id': 'mistral-medium-3-5'}, {'id': 'mistral-medium-3-5'}],
    [{'id': 'first', 'aliases': ['mistral-medium-3-5']},
     {'id': 'second', 'aliases': ['mistral-medium-3-5']}],
])
def test_duplicate_exact_ids_or_ambiguous_aliases_never_use_first_match(cards):
    candidate = CANDIDATES[4]
    client = fake_client(candidate, models=FakeModels(candidate.model_id, listed=cards))
    runtime = build(candidate, client)
    assert runtime.validate_availability().code == 'CANDIDATE_AVAILABILITY_UNSUPPORTED'
    with pytest.raises(PlanningModelError):
        runtime.complete(prompt='x', response_schema=SCHEMA)
    assert runtime.completion_calls == 0


@pytest.mark.parametrize('status', [
    {'active': False}, {'archived': True}, {'capabilities': {'completion_chat': False}},
])
def test_provider_declared_unavailable_model_cannot_complete(status):
    candidate = CANDIDATES[4]
    client = fake_client(candidate, models=FakeModels(candidate.model_id, listed=[
        {'id': candidate.model_id, **status}]))
    runtime = build(candidate, client)
    assert runtime.validate_availability().status == 'unavailable'
    with pytest.raises(PlanningModelError):
        runtime.complete(prompt='x', response_schema=SCHEMA)
    assert runtime.completion_calls == 0


def test_model_alias_substring_is_not_an_exact_availability_witness():
    candidate = CANDIDATES[4]
    client = fake_client(candidate, models=FakeModels(candidate.model_id, listed=[
        {'id': 'different-model', 'aliases': 'prefix-' + candidate.model_id}]))
    runtime = build(candidate, client)
    assert runtime.validate_availability().code == 'CANDIDATE_AVAILABILITY_UNSUPPORTED'
    assert runtime.completion_calls == 0


def test_true_provider_manifest_and_availability_do_not_persist_secrets():
    candidate = CANDIDATES[2]
    client = fake_client(candidate)
    runtime = build_candidate(candidate, client=client, environment={
        'DASHSCOPE_API_KEY': 'secret-test-credential',
        'DASHSCOPE_BASE_URL': 'https://example.test/compatible/v1',
    })
    manifest = runtime.manifest()
    assert manifest['provider_id'] == 'dashscope'
    assert manifest['model_id'] == 'deepseek-v4.1-flash'
    assert manifest['adapter_family'] == 'openai-responses'
    assert runtime.candidate.manifest() == manifest
    assert runtime.model_id == 'dashscope:deepseek-v4.1-flash'
    serialized = json.dumps([manifest, runtime.validate_availability().to_dict()])
    assert 'secret-test-credential' not in serialized
    assert 'DASHSCOPE_API_KEY' in serialized


def test_availability_http_failure_is_sanitized_and_not_retried():
    class RateLimited(Exception):
        status_code = 429
    candidate = CANDIDATES[0]
    models = FakeModels(candidate.model_id, error=RateLimited('authorization secret-test-credential'))
    runtime = build(candidate, fake_client(candidate, models=models))
    result = runtime.validate_availability()
    assert result.status == 'operational_failure'
    assert result.code == 'PROVIDER_RATE_LIMITED'
    assert result.http_status == 429
    assert 'secret-test-credential' not in json.dumps(result.to_dict())
    assert runtime.validate_availability() is result
    assert runtime.availability_requests == 1


@pytest.mark.parametrize('http_code,status,expected_code', [
    (404, 'unavailable', 'CANDIDATE_MODEL_UNAVAILABLE'),
    (429, 'operational_failure', 'PROVIDER_RATE_LIMITED'),
    (403, 'operational_failure', 'PROVIDER_AUTHENTICATION_FAILED'),
    (500, 'operational_failure', 'PROVIDER_UNAVAILABLE'),
])
def test_numeric_sdk_code_is_sanitized_as_structured_http_status(http_code, status, expected_code):
    class SDKError(Exception):
        code = http_code
    candidate = CANDIDATES[3]
    client = fake_client(candidate, models=FakeModels(candidate.model_id,
        error=SDKError('secret-test-credential in provider details')))
    runtime = build(candidate, client)
    availability = runtime.validate_availability()
    assert availability.status == status
    assert availability.code == expected_code
    assert availability.http_status == http_code
    assert 'secret-test-credential' not in json.dumps(availability.to_dict())
    assert runtime.availability_requests == 1
    assert runtime.completion_calls == 0


@pytest.mark.parametrize('unsafe_code', [True, 9999, 'secret-test-credential', '429'])
def test_non_http_sdk_code_is_never_parsed_or_persisted(unsafe_code):
    class SDKError(Exception):
        code = unsafe_code
    candidate = CANDIDATES[3]
    client = fake_client(candidate, models=FakeModels(candidate.model_id, error=SDKError('untrusted details')))
    availability = build(candidate, client).validate_availability()
    assert availability.code == 'PLANNING_PROVIDER_ERROR'
    assert availability.http_status is None
    assert 'secret-test-credential' not in json.dumps(availability.to_dict())


@pytest.mark.parametrize('completion,code', [
    ({'choices': []}, 'PROVIDER_COMPLETION_INCOMPLETE'),
    ({'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}]}, 'PROVIDER_COMPLETION_INCOMPLETE'),
    ({'choices': [{'finish_reason': 'stop', 'message': {'content': '', 'refusal': 'x'}}]}, 'PROVIDER_REFUSED'),
    ({'choices': [{'finish_reason': 'stop', 'message': {'content': 'x', 'tool_calls': [{}]}}]}, 'PROVIDER_COMPLETION_INCOMPLETE'),
])
def test_generic_chat_adapter_fails_closed_on_incomplete_response(completion, code):
    client = fake_client(CANDIDATES[4], completion=completion)
    model = ChatCompletionsPlanningModel(client=client, model='mistral-medium-3-5')
    with pytest.raises(PlanningModelError) as failure:
        model.complete(prompt='x', response_schema=SCHEMA)
    assert failure.value.code == code
    assert len(client.chat.completions.calls) == 1


def test_generic_chat_transport_error_is_sanitized_without_retry():
    class RateLimited(Exception):
        status_code = 429
    client = fake_client(CANDIDATES[4], error=RateLimited('secret-test-credential'))
    model = ChatCompletionsPlanningModel(client=client, model='mistral-medium-3-5')
    with pytest.raises(PlanningModelError) as failure:
        model.complete(prompt='x', response_schema=SCHEMA)
    assert failure.value.code == 'PROVIDER_RATE_LIMITED'
    assert 'secret-test-credential' not in str(failure.value)
    assert len(client.chat.completions.calls) == 1


def test_factory_sdk_construction_disables_hidden_retries(monkeypatch):
    import openai
    calls = []
    client = fake_client(CANDIDATES[4])
    def factory(**kwargs):
        calls.append(kwargs)
        return client
    monkeypatch.setattr(openai, 'OpenAI', factory)
    runtime = build_candidate(CANDIDATES[4], environment={'MISTRAL_API_KEY': 'secret-test-credential'})
    assert calls == [{'api_key': 'secret-test-credential', 'base_url': 'https://api.mistral.ai/v1', 'max_retries': 0}]
    assert 'secret-test-credential' not in json.dumps(runtime.manifest())
    assert runtime.availability_requests == runtime.completion_calls == 0
