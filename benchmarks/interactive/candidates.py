"""Qualification-only candidate construction; no semantic provider routing.

Mistral documents Chat Completions and JSON Schema at
https://docs.mistral.ai/api/endpoint/chat, account model discovery at
https://docs.mistral.ai/api/endpoint/models, and the exact requested alias at
https://docs.mistral.ai/models/mistral-medium-3-5-26-04.
DashScope deliberately requires the historical DASHSCOPE_BASE_URL setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
import re
from typing import Mapping
from urllib.parse import urlsplit

from agent.orchestration.planning_model import (
    PlanningModelError, PlanningModelProfile, classify_provider_exception,
    normalized_retry_after_seconds,
)
from agent.providers import GeminiPlanningModel, GroqPlanningModel, OpenAIPlanningModel
from agent.providers.openai_planning import _plain_json


def _endpoint(value):
    if type(value) is not str:
        raise ValueError('An explicit HTTPS endpoint is required.')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or any(not char.isprintable() or char.isspace() for char in value)):
        raise ValueError('Endpoint must be HTTPS without credentials, query, or fragment.')
    return value.rstrip('/')


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    provider_id: str
    model_id: str
    adapter_family: str
    credential_env: str
    profile_id: str
    endpoint: str | None = None
    endpoint_env: str | None = None
    timeout_seconds: float = 60.0
    endpoint_identity: str | None = None

    def __post_init__(self):
        # Reuse the production identity and timeout contract.
        self.profile()
        if self.adapter_family not in {
            'groq-responses', 'openai-responses', 'gemini-interactions',
            'openai-chat-completions',
        }:
            raise ValueError('Unsupported qualification transport family.')
        if (type(self.candidate_id) is not str
                or not re.fullmatch(r'[a-z][a-z0-9._-]{0,127}', self.candidate_id)
                or type(self.credential_env) is not str
                or not re.fullmatch(r'[A-Z][A-Z0-9_]*_API_KEY', self.credential_env)):
            raise ValueError('Invalid candidate identity or credential variable name.')
        if self.endpoint is not None:
            _endpoint(self.endpoint)
        if self.endpoint_identity is not None:
            _endpoint(self.endpoint_identity)
        if self.endpoint is not None and self.endpoint_env is not None:
            raise ValueError('A candidate endpoint must have one explicit source.')
        if (self.endpoint_env is not None and (type(self.endpoint_env) is not str
                or not re.fullmatch(r'[A-Z][A-Z0-9_]*', self.endpoint_env))):
            raise ValueError('Invalid endpoint variable name.')

    def profile(self):
        return PlanningModelProfile(self.profile_id, self.provider_id, self.model_id,
                                    request_timeout_seconds=self.timeout_seconds)

    def manifest(self, *, endpoint_identity=None):
        return dict(candidate_id=self.candidate_id, provider_id=self.provider_id,
                    model_id=self.model_id, adapter_family=self.adapter_family,
                    profile_id=self.profile_id, timeout_seconds=self.timeout_seconds,
                    credential_env=self.credential_env, endpoint_env=self.endpoint_env,
                    endpoint_identity=(endpoint_identity if endpoint_identity is not None
                                       else self.endpoint_identity or self.endpoint))


CANDIDATES = (
    Candidate('groq-gpt-oss-120b', 'groq', 'openai/gpt-oss-120b', 'groq-responses',
              'GROQ_API_KEY', 'q1-groq-gpt-oss-120b', 'https://api.groq.com/openai/v1'),
    Candidate('groq-qwen3.8-27b', 'groq', 'qwen/qwen3.8-27b', 'groq-responses',
              'GROQ_API_KEY', 'q1-groq-qwen3-8-27b', 'https://api.groq.com/openai/v1'),
    Candidate('dashscope-deepseek-v4.1-flash', 'dashscope', 'deepseek-v4.1-flash',
              'openai-responses', 'DASHSCOPE_API_KEY', 'q1-dashscope-deepseek-v4-1-flash',
              endpoint_env='DASHSCOPE_BASE_URL'),
    Candidate('gemini-3.8-flash', 'gemini', 'gemini-3.8-flash', 'gemini-interactions',
              'GEMINI_API_KEY', 'q1-gemini-3-8-flash', 'https://generativelanguage.googleapis.com'),
    Candidate('mistral-medium-3-5', 'mistral', 'mistral-medium-3-5',
              'openai-chat-completions', 'MISTRAL_API_KEY', 'q1-mistral-medium-3-5',
              'https://api.mistral.ai/v1'),
)


def get_candidate(candidate_id):
    matches = [c for c in CANDIDATES if c.candidate_id == candidate_id]
    if len(matches) != 1:
        raise ValueError('Unknown exact qualification candidate identifier.')
    return matches[0]


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _http_status_code(exception):
    """Accept only structured numeric HTTP codes, never provider error prose."""
    for name in ('status_code', 'code'):
        value = getattr(exception, name, None)
        if type(value) is int and 100 <= value <= 599:
            return value
    return None


class _HTTPStatusFailure(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


@dataclass(frozen=True)
class Availability:
    status: str
    code: str
    requested_model_id: str
    operation: str
    provider_model_id: str | None = None
    http_status: int | None = None

    def to_dict(self):
        return asdict(self)


class ChatCompletionsPlanningModel:
    """One generic strict-schema request through an injected compatible client."""

    def __init__(self, *, client, model, timeout=60.0):
        if not callable(getattr(getattr(getattr(client, 'chat', None), 'completions', None), 'create', None)):
            raise TypeError('Chat client must provide chat.completions.create().')
        PlanningModelProfile('chat-adapter', 'compatible', model,
                             request_timeout_seconds=timeout)
        self._client, self._model, self._timeout = client, model, timeout

    @property
    def model_id(self):
        return 'compatible:' + self._model

    def complete(self, *, prompt, response_schema):
        if type(prompt) is not str or not prompt or not isinstance(response_schema, Mapping):
            raise ValueError('A prompt and response schema are required.')
        schema = _plain_json(response_schema)
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=[dict(role='user', content=prompt)],
                response_format=dict(type='json_schema', json_schema=dict(
                    name='agent_plan', strict=True, schema=schema)),
                stream=False, timeout=self._timeout)
        except Exception as exc:
            code, message = classify_provider_exception(exc)
            raise PlanningModelError(message, code=code,
                retry_after_seconds=normalized_retry_after_seconds(exc)) from None
        choices = _field(response, 'choices')
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise PlanningModelError('Provider did not return one completion.',
                                     code='PROVIDER_COMPLETION_INCOMPLETE')
        choice = choices[0]
        message = _field(choice, 'message')
        if _field(message, 'refusal'):
            raise PlanningModelError('Provider refused the request.', code='PROVIDER_REFUSED')
        content = _field(message, 'content')
        if (_field(choice, 'finish_reason') != 'stop' or type(content) is not str
                or not content.strip() or _field(message, 'tool_calls')):
            raise PlanningModelError('Provider completion was incomplete or lacked text.',
                                     code='PROVIDER_COMPLETION_INCOMPLETE')
        return content


class CandidateRuntime:
    """Exact identity plus mandatory read-only availability before completion."""

    def __init__(self, candidate, model, client, endpoint_identity):
        self.candidate = replace(candidate, endpoint_identity=endpoint_identity)
        self._model, self._client = model, client
        self._endpoint_identity = endpoint_identity
        self.availability = None
        self.availability_requests = 0
        self.completion_calls = 0

    @property
    def model_id(self):
        return self.candidate.provider_id + ':' + self.candidate.model_id

    def manifest(self):
        return self.candidate.manifest(endpoint_identity=self._endpoint_identity)

    def validate_availability(self):
        if self.availability is not None:
            return self.availability
        candidate = self.candidate
        operation = 'models.get' if candidate.adapter_family == 'gemini-interactions' else 'models.list'
        try:
            models = getattr(self._client, 'models', None)
            method = getattr(models, 'get' if operation == 'models.get' else 'list', None)
            if not callable(method):
                self.availability = Availability('unavailable', 'CANDIDATE_AVAILABILITY_UNSUPPORTED',
                                                candidate.model_id, operation)
                return self.availability
            self.availability_requests += 1
            if operation == 'models.get':
                card = method(model=candidate.model_id, config={
                    'http_options': {'timeout': int(candidate.timeout_seconds * 1000)}})
                returned = _field(card, 'name', _field(card, 'id'))
                # This is an API namespace prefix, never a model substitution.
                actual = returned.removeprefix('models/') if type(returned) is str else None
                accepted = actual == candidate.model_id
            else:
                response = method(timeout=candidate.timeout_seconds)
                cards = _field(response, 'data')
                if not isinstance(cards, (list, tuple)) or len(cards) > 4096:
                    raise ValueError('Invalid bounded model-list response.')
                # A unique exact ID is authoritative even when other cards
                # advertise the same alias. Only an absent exact ID uses aliases.
                matches = [card for card in cards if _field(card, 'id') == candidate.model_id]
                if not matches:
                    for card in cards:
                        aliases = _field(card, 'aliases', ()) or ()
                        if not isinstance(aliases, (list, tuple)) or any(type(a) is not str for a in aliases):
                            raise ValueError('Invalid model aliases.')
                        if candidate.model_id in aliases:
                            matches.append(card)
                if len(matches) > 1:
                    raise ValueError('Ambiguous model-list identity.')
                accepted = len(matches) == 1
                actual = _field(matches[0], 'id') if accepted else None
                if accepted:
                    PlanningModelProfile('available-model', 'compatible', actual)
                if accepted:
                    card = matches[0]
                    if (_field(_field(card, 'capabilities', {}), 'completion_chat') is False
                            or _field(card, 'active') is False or _field(card, 'archived') is True):
                        accepted = False
                if not accepted and _field(response, 'has_more', False):
                    raise ValueError('Model-list response is incomplete.')
            self.availability = Availability(
                'available' if accepted else 'unavailable',
                'CANDIDATE_MODEL_AVAILABLE' if accepted else 'CANDIDATE_MODEL_UNAVAILABLE',
                candidate.model_id, operation,
                actual if accepted and type(actual) is str else None)
        except Exception as exc:
            code, _ = classify_provider_exception(exc)
            status = _http_status_code(exc)
            if code == 'PLANNING_PROVIDER_ERROR' and status is not None:
                code, _ = classify_provider_exception(_HTTPStatusFailure(status))
            if status == 404 and operation == 'models.get':
                result, code = 'unavailable', 'CANDIDATE_MODEL_UNAVAILABLE'
            elif isinstance(exc, ValueError) or status == 404:
                result, code = 'unavailable', 'CANDIDATE_AVAILABILITY_UNSUPPORTED'
            else:
                result = 'operational_failure'
            self.availability = Availability(result, code, candidate.model_id, operation,
                                            http_status=status if type(status) is int else None)
        return self.availability

    def complete(self, *, prompt, response_schema):
        if self.availability is None or self.availability.status != 'available':
            raise PlanningModelError('Exact candidate model availability was not accepted.',
                                     code='CANDIDATE_MODEL_NOT_VALIDATED')
        self.completion_calls += 1
        return self._model.complete(prompt=prompt, response_schema=response_schema)

    def close(self):
        method = getattr(self._client, 'close', None)
        if callable(method):
            method()


def build_candidate(candidate, *, client=None, environment=None):
    """Construct transport only. No network request or model substitution occurs."""
    if not isinstance(candidate, Candidate):
        raise TypeError('Expected an exact Candidate definition.')
    env = os.environ if environment is None else environment
    endpoint = candidate.endpoint
    try:
        if candidate.endpoint_env is not None:
            endpoint = env.get(candidate.endpoint_env)
            if not endpoint:
                raise PlanningModelError('The configured candidate endpoint is missing.',
                                         code='CANDIDATE_ENDPOINT_MISSING')
        endpoint = _endpoint(endpoint)
        if client is None:
            key = env.get(candidate.credential_env)
            if not key or not key.strip():
                raise PlanningModelError('The candidate credential is not configured.',
                                         code='PLANNING_PROVIDER_CONFIGURATION_FAILED')
            if candidate.adapter_family == 'gemini-interactions':
                from google import genai
                client = genai.Client(api_key=key, http_options={'retry_options': {'attempts': 1}})
            else:
                from openai import OpenAI
                client = OpenAI(api_key=key, base_url=endpoint, max_retries=0)
        if candidate.adapter_family == 'groq-responses':
            model = GroqPlanningModel(model=candidate.model_id, timeout=candidate.timeout_seconds, _client=client)
        elif candidate.adapter_family == 'gemini-interactions':
            model = GeminiPlanningModel(model=candidate.model_id, timeout=candidate.timeout_seconds, _client=client)
        elif candidate.adapter_family == 'openai-responses':
            model = OpenAIPlanningModel(model=candidate.model_id, timeout=candidate.timeout_seconds, _client=client)
        else:
            model = ChatCompletionsPlanningModel(client=client, model=candidate.model_id,
                                                 timeout=candidate.timeout_seconds)
    except PlanningModelError:
        raise
    except ImportError:
        raise PlanningModelError('The optional transport dependency is missing.',
                                 code='PLANNING_PROVIDER_DEPENDENCY_MISSING') from None
    except Exception:
        raise PlanningModelError('Candidate transport configuration is invalid.',
                                 code='PLANNING_PROVIDER_CONFIGURATION_FAILED') from None
    return CandidateRuntime(candidate, model, client, endpoint)
