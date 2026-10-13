"""Network-free request, response and deployment checks for OpenRouter planning."""

from __future__ import annotations

import builtins
import copy
import math
import sys
import traceback
from types import MappingProxyType, SimpleNamespace

import pytest

from agent.orchestration import PlanningModel, PlanningRecoveryPolicy
from agent.providers import (
    OpenRouterPlanningDependencyError,
    OpenRouterPlanningError,
    OpenRouterPlanningModel,
)
import agent.providers.openrouter_planning as module


MODEL = "organization/qualified-model"
OUTPUT = ' \n{"decision":{"kind":"clarify"}}\n '
SCHEMA = {"type": "object", "properties": {"decision": {"type": "object"}},
          "required": ["decision"], "additionalProperties": False}


def response(**changes):
    result = {"model": MODEL, "choices": [{"finish_reason": "stop", "message": {
        "content": OUTPUT, "refusal": None, "tool_calls": None}}]}
    result.update(changes)
    return result


class FakeCompletions:
    def __init__(self, result=None, error=None):
        self.result = response() if result is None else result
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, error=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(result, error))


def adapter(result=None, error=None, **options):
    client = FakeClient(result, error)
    return OpenRouterPlanningModel(model=MODEL, _client=client, **options), client


def test_exact_chat_request_and_provider_qualified_identity_preserve_prompt_schema_and_output():
    model, client = adapter(timeout=17.5)
    schema = MappingProxyType(copy.deepcopy(SCHEMA))
    prompt = "Exact offered context and schema; no additional content."

    assert isinstance(model, PlanningModel)
    assert model.model_id == "openrouter:" + MODEL
    assert model.model == MODEL and model.timeout == 17.5
    assert model.complete(prompt=prompt, response_schema=schema) == OUTPUT
    assert client.chat.completions.calls == [{
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "agent_plan", "strict": True, "schema": SCHEMA}},
        "stream": False,
        "timeout": 17.5,
        "max_completion_tokens": 4096,
        "extra_body": {
            "provider": {"require_parameters": True, "allow_fallbacks": False,
                         "max_price": {"prompt": 2, "completion": 10, "request": 0}},
            "plugins": [{"id": name, "enabled": False} for name in
                        ("response-healing", "context-compression", "web")],
        },
        "extra_headers": {"X-OpenRouter-Metadata": "enabled"},
    }]
    assert schema == SCHEMA
    client.chat.completions.calls[0]["response_format"]["json_schema"]["schema"]["required"].append("new")
    assert schema == SCHEMA


def test_each_explicit_complete_call_dispatches_once_without_lifetime_benchmark_guard():
    model, client = adapter()
    for _ in range(3):
        assert model.complete(prompt="same context", response_schema=SCHEMA) == OUTPUT
    assert len(client.chat.completions.calls) == 3


def test_explicitly_tightened_request_policy_is_copied_and_immutable():
    prices = {"prompt": "1.5", "completion": 5, "request": 0}
    model, client = adapter(max_completion_tokens=1024, provider_max_price=prices)
    prices["prompt"] = 100
    with pytest.raises(TypeError):
        model.provider_max_price["prompt"] = 100
    model.complete(prompt="context", response_schema=SCHEMA)
    request = client.chat.completions.calls[0]
    assert model.max_completion_tokens == request["max_completion_tokens"] == 1024
    assert request["extra_body"]["provider"]["max_price"] == {
        "prompt": 1.5, "completion": 5, "request": 0}


@pytest.mark.parametrize("maximum", [True, 0, -1, 4097, 4096.0, "4096", None, math.inf])
def test_invalid_output_limit_fails_before_client_construction(maximum, monkeypatch):
    monkeypatch.setattr(module, "_default_client", lambda: pytest.fail("constructed a client"))
    with pytest.raises(ValueError):
        OpenRouterPlanningModel(model=MODEL, max_completion_tokens=maximum)


@pytest.mark.parametrize("prices", [
    {}, {"prompt": 2, "completion": 10},
    {"prompt": 2, "completion": 10, "request": 0, "other": 0},
    {"prompt": True, "completion": 10, "request": 0},
    {"prompt": -1, "completion": 10, "request": 0},
    {"prompt": 2.01, "completion": 10, "request": 0},
    {"prompt": 2, "completion": 10.01, "request": 0},
    {"prompt": 2, "completion": 10, "request": 0.01},
    {"prompt": "nan", "completion": 10, "request": 0},
    {"prompt": math.inf, "completion": 10, "request": 0},
    {"prompt": "invalid", "completion": 10, "request": 0},
    {"prompt": "9" * 33, "completion": 10, "request": 0},
])
def test_invalid_or_loosened_price_caps_fail_before_client_construction(prices, monkeypatch):
    monkeypatch.setattr(module, "_default_client", lambda: pytest.fail("constructed a client"))
    with pytest.raises(ValueError):
        OpenRouterPlanningModel(model=MODEL, provider_max_price=prices)


@pytest.mark.parametrize("model", ["", " model ", "model with spaces", "https://a?key=private", None])
def test_invalid_model_identity_is_rejected_before_client_construction(model, monkeypatch):
    monkeypatch.setattr(module, "_default_client", lambda: pytest.fail("constructed a client"))
    with pytest.raises(ValueError):
        OpenRouterPlanningModel(model=model)


@pytest.mark.parametrize("timeout", [True, 0, -1, math.nan, math.inf, 3600.1, "60"])
def test_profile_timeout_bounds_apply_before_client_construction(timeout, monkeypatch):
    monkeypatch.setattr(module, "_default_client", lambda: pytest.fail("constructed a client"))
    with pytest.raises(ValueError):
        OpenRouterPlanningModel(model=MODEL, timeout=timeout)


@pytest.mark.parametrize("prompt,schema", [
    ("", SCHEMA), (None, SCHEMA), ("prompt", []),
    ("prompt", {"invalid": math.nan}), ("prompt", {1: "invalid"}),
])
def test_invalid_prompt_or_schema_fails_before_request(prompt, schema):
    model, client = adapter()
    with pytest.raises((TypeError, ValueError)):
        model.complete(prompt=prompt, response_schema=schema)
    assert client.chat.completions.calls == []


def test_invalid_client_interface_is_rejected():
    with pytest.raises(TypeError):
        OpenRouterPlanningModel(model=MODEL, _client=SimpleNamespace(responses=object()))


@pytest.mark.parametrize("changes", [
    {"choices": None}, {"choices": []}, {"choices": [{}, {}]},
    {"choices": [{"finish_reason": "length", "message": {"content": OUTPUT}}]},
    {"choices": [{"finish_reason": "content_filter", "message": {"content": OUTPUT}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": []}}]},
    {"choices": [{"finish_reason": "stop", "message": {
        "content": OUTPUT, "tool_calls": [{"private": "hidden"}]}}]},
])
def test_incomplete_or_tool_call_response_fails_without_an_adapter_retry(changes):
    model, client = adapter(response(**changes))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.code == "PROVIDER_COMPLETION_INCOMPLETE"
    assert len(client.chat.completions.calls) == 1
    assert "hidden" not in str(raised.value)


def test_refusal_text_is_never_exposed():
    model, client = adapter(response(choices=[{"finish_reason": "stop", "message": {
        "content": OUTPUT, "refusal": "private refusal body"}}]))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.code == "PROVIDER_REFUSED"
    assert "private" not in str(raised.value)
    assert len(client.chat.completions.calls) == 1


def test_response_model_must_have_the_exact_string_type():
    class ModelLike:
        def __eq__(self, other):
            return other == MODEL

    model, client = adapter(response(model=ModelLike()))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.code == "PLANNING_PROVIDER_ERROR"
    assert len(client.chat.completions.calls) == 1


@pytest.mark.parametrize("changes", [
    {"model": None}, {"model": "organization/substituted-model"},
    {"model": MODEL + ":free"}, {"error": {"message": "private provider body"}},
    {"openrouter_metadata": "private metadata"},
    {"openrouter_metadata": {"requested": "other-model"}},
    {"openrouter_metadata": {"pipeline": [{"id": "response-healing"}]}},
    {"openrouter_metadata": {"pipeline": None}},
    {"openrouter_metadata": {"attempt": 2}},
    {"openrouter_metadata": {"attempt": True}},
    {"openrouter_metadata": {"attempt": -1}},
    {"openrouter_metadata": {"attempt": "1"}},
])
def test_model_and_routing_policy_violations_are_terminal_and_sanitized(changes):
    model, client = adapter(response(**changes))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.code == "PLANNING_PROVIDER_ERROR"
    assert raised.value.code not in PlanningRecoveryPolicy().retryable_provider_codes
    assert len(client.chat.completions.calls) == 1
    assert "private" not in str(raised.value)
    assert "substituted" not in str(raised.value)


@pytest.mark.parametrize("metadata", [None, {}, {"requested": MODEL, "pipeline": [], "attempt": 1}])
def test_missing_or_compatible_metadata_returns_the_exact_completion_without_inference(metadata):
    model, _ = adapter(response(openrouter_metadata=metadata))
    assert model.complete(prompt="context", response_schema=SCHEMA) == OUTPUT


class SDKError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__("private-key private provider body https://private.example")
        self.status_code = status_code
        self.response = SimpleNamespace(headers={"Retry-After": retry_after, "secret": "private-key"})


@pytest.mark.parametrize("status,code", [
    (401, "PROVIDER_AUTHENTICATION_FAILED"), (403, "PROVIDER_AUTHENTICATION_FAILED"),
    (413, "PROVIDER_REQUEST_TOO_LARGE"), (429, "PROVIDER_RATE_LIMITED"),
    (408, "PROVIDER_TIMEOUT"), (503, "PROVIDER_UNAVAILABLE"), (400, "PLANNING_PROVIDER_ERROR"),
])
def test_provider_errors_are_classified_and_sanitized_without_sdk_retry(status, code):
    model, client = adapter(error=SDKError(status, "2.5"))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.code == code
    assert raised.value.retry_after_seconds == 2.5
    assert len(client.chat.completions.calls) == 1
    assert "private" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("raw", [None, "invalid", "nan", "inf", -1, True])
def test_unsafe_retry_after_is_discarded(raw):
    model, _ = adapter(error=SDKError(429, raw))
    with pytest.raises(OpenRouterPlanningError) as raised:
        model.complete(prompt="context", response_schema=SCHEMA)
    assert raised.value.retry_after_seconds is None


def test_default_client_reads_only_its_key_and_pins_endpoint_and_zero_sdk_retries(monkeypatch):
    client = FakeClient()
    constructed = []

    def construct(**kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=construct))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-other-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://wrong.example")
    model = OpenRouterPlanningModel(model=MODEL)
    assert constructed == [{"api_key": "synthetic-openrouter-key",
                            "base_url": "https://openrouter.ai/api/v1", "max_retries": 0}]
    assert client.chat.completions.calls == []
    model.complete(prompt="context", response_schema=SCHEMA)
    assert "synthetic-openrouter-key" not in repr(client.chat.completions.calls)
    assert "synthetic-openrouter-key" not in repr(model)


@pytest.mark.parametrize("key", [None, "", "  "])
def test_missing_key_fails_safely_before_sdk_client_construction(key, monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        OpenAI=lambda **kwargs: pytest.fail("SDK client constructed")))
    if key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)
    with pytest.raises(OpenRouterPlanningError) as raised:
        OpenRouterPlanningModel(model=MODEL)
    assert raised.value.code == "PLANNING_PROVIDER_CONFIGURATION_FAILED"


def test_sdk_initialization_failure_does_not_expose_credentials(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("synthetic-openrouter-key private initialization body")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=fail))
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-openrouter-key")
    with pytest.raises(OpenRouterPlanningError) as raised:
        OpenRouterPlanningModel(model=MODEL)
    assert raised.value.code == "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    assert "private" not in "".join(traceback.format_exception(raised.value))
    assert "synthetic-openrouter-key" not in "".join(traceback.format_exception(raised.value))


def test_optional_sdk_absence_has_a_lazy_clear_error(monkeypatch):
    original = builtins.__import__
    def without_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("private import details")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_openai)
    with pytest.raises(OpenRouterPlanningDependencyError) as raised:
        OpenRouterPlanningModel(model=MODEL)
    assert raised.value.code == "PLANNING_PROVIDER_DEPENDENCY_MISSING"
    assert "private" not in "".join(traceback.format_exception(raised.value))
