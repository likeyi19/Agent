"""Network-free tests for the optional Groq planning adapter."""

from __future__ import annotations

import importlib
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    PlanningModel,
    build_default_tool_registry,
)
from agent.orchestration.llm_planner import _response_schema
from agent.providers import (
    GroqPlanningDependencyError,
    GroqPlanningError,
    GroqPlanningModel,
)


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_OUTPUT = (
    '{"schema_version":3,"status":"unsupported","steps":[],'
    '"reason":"safe"}'
)


def _response(
    *,
    status: object = "completed",
    output_text: object = _OUTPUT,
    output: object = (),
    error: object = None,
):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        output=output,
        error=error,
    )


class FakeResponses:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = _response() if result is None else result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.responses = FakeResponses(result, error=error)


def _adapter(
    result=None,
    *,
    error: Exception | None = None,
    model: str = "groq-test",
    timeout: float = 12.5,
) -> tuple[GroqPlanningModel, FakeClient]:
    client = FakeClient(result, error=error)
    return (
        GroqPlanningModel(
            model=model,
            timeout=timeout,
            _client=client,
        ),
        client,
    )


def _schema(input_path: str = "/synthetic/input.h5ad"):
    return _response_schema(
        build_default_tool_registry(),
        AgentRequest(
            "provider-schema",
            "Inspect the supplied dataset.",
            {"input_path": input_path},
        ),
    )


def _install_fake_openai(monkeypatch, factory) -> None:
    openai = ModuleType("openai")
    openai.OpenAI = factory
    monkeypatch.setitem(sys.modules, "openai", openai)


def test_adapter_satisfies_planning_model_protocol() -> None:
    adapter, _ = _adapter()

    assert isinstance(adapter, PlanningModel)


def test_configured_model_identity_and_timeout_are_stable() -> None:
    adapter, _ = _adapter(model=" openai/gpt-oss-20b ", timeout=17)

    assert adapter.model == "openai/gpt-oss-20b"
    assert adapter.model_id == "groq:openai/gpt-oss-20b"
    assert adapter.timeout == 17.0


def test_responses_request_maps_prompt_and_v3_schema_exactly() -> None:
    adapter, client = _adapter()
    prompt = "exact planning prompt"
    response_schema = _schema()

    returned = adapter.complete(prompt=prompt, response_schema=response_schema)

    assert returned == _OUTPUT
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0] == {
        "model": "groq-test",
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "agent_plan",
                "strict": True,
                "schema": json.loads(json.dumps(response_schema)),
            }
        },
        "timeout": 12.5,
    }
    transmitted = client.responses.calls[0]["text"]["format"]["schema"]
    assert transmitted["properties"]["schema_version"]["enum"] == [3]
    assert "anyOf" in transmitted["properties"]["steps"]["items"]


def test_request_enables_no_tools_or_provider_state() -> None:
    adapter, client = _adapter()

    adapter.complete(prompt="prompt", response_schema=_schema())

    request = client.responses.calls[0]
    for forbidden in (
        "tools",
        "tool_choice",
        "previous_response_id",
        "store",
        "background",
        "include",
        "truncation",
    ):
        assert forbidden not in request


def test_adapter_adds_no_dataset_or_file_values() -> None:
    adapter, client = _adapter()
    prompt = "prompt-without-local-data"
    private_path = "/private/secret-dataset.h5ad"

    adapter.complete(prompt=prompt, response_schema=_schema(private_path))

    serialized = json.dumps(client.responses.calls[0], sort_keys=True)
    assert private_path not in serialized
    assert client.responses.calls[0]["input"] == prompt


def test_success_returns_exact_output_text_without_normalization() -> None:
    exact = "  {\n  \"status\": \"plan\"\n}\n"
    adapter, _ = _adapter(_response(output_text=exact))

    returned = adapter.complete(prompt="prompt", response_schema=_schema())

    assert returned == exact


@pytest.mark.parametrize("missing", [None, "", "   ", 3])
def test_missing_output_fails_cleanly(missing) -> None:
    adapter, _ = _adapter(_response(output_text=missing))

    with pytest.raises(GroqPlanningError, match="did not contain output text"):
        adapter.complete(prompt="prompt", response_schema=_schema())


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", None])
def test_noncompleted_response_fails_cleanly(status) -> None:
    adapter, _ = _adapter(_response(status=status))

    with pytest.raises(GroqPlanningError, match="was not completed") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())
    assert raised.value.code == "PROVIDER_COMPLETION_INCOMPLETE"


def test_provider_reported_error_fails_without_exposing_error_object() -> None:
    adapter, _ = _adapter(
        _response(error={"authorization": "Bearer provider-secret"})
    )

    with pytest.raises(GroqPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert "provider-secret" not in str(raised.value)
    assert "authorization" not in str(raised.value).casefold()


def test_refusal_fails_without_exposing_refusal_text() -> None:
    refusal_text = "refusal containing provider-secret"
    refusal = SimpleNamespace(type="refusal", refusal=refusal_text)
    message = SimpleNamespace(type="message", content=[refusal])
    adapter, _ = _adapter(_response(output=[message], output_text=""))

    with pytest.raises(GroqPlanningError, match="was refused") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert refusal_text not in str(raised.value)
    assert raised.value.code == "PROVIDER_REFUSED"


def test_api_exception_is_converted_to_sanitized_adapter_error() -> None:
    secret = "Authorization: Bearer api-secret; raw HTTP body; request-id"
    adapter, client = _adapter(error=RuntimeError(secret))

    with pytest.raises(GroqPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert len(client.responses.calls) == 1
    assert str(raised.value) == "Groq planning request failed."
    assert "api-secret" not in str(raised.value)
    assert "HTTP" not in str(raised.value)
    assert "request-id" not in str(raised.value)


def test_real_client_uses_key_only_at_groq_boundary(monkeypatch) -> None:
    secret = "groq-test-do-not-copy"
    client = FakeClient()
    calls: list[dict[str, object]] = []

    def make_client(**kwargs):
        calls.append(kwargs)
        return client

    _install_fake_openai(monkeypatch, make_client)
    monkeypatch.setenv("GROQ_API_KEY", secret)

    adapter = GroqPlanningModel(model="openai/gpt-oss-20b")
    output = adapter.complete(prompt="safe prompt", response_schema=_schema())

    assert calls == [
        {
            "api_key": secret,
            "base_url": _GROQ_BASE_URL,
            "max_retries": 0,
        }
    ]
    assert secret not in output
    assert secret not in json.dumps(client.responses.calls)


def test_missing_optional_sdk_has_clear_lazy_error(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "configured-but-not-disclosed")
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(
        GroqPlanningDependencyError, match="optional OpenAI SDK"
    ) as raised:
        GroqPlanningModel(model="openai/gpt-oss-20b")
    assert raised.value.code == "PLANNING_PROVIDER_DEPENDENCY_MISSING"


def test_missing_key_has_safe_provider_error(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def make_client(**kwargs):
        calls.append(kwargs)
        return FakeClient()

    _install_fake_openai(monkeypatch, make_client)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(GroqPlanningError, match="GROQ_API_KEY") as raised:
        GroqPlanningModel(model="openai/gpt-oss-20b")

    assert calls == []
    assert raised.value.code == "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    assert "authorization" not in str(raised.value).casefold()


def test_client_initialization_failure_is_sanitized(monkeypatch) -> None:
    secret = "Authorization: Bearer provider-secret"

    def make_client(**kwargs):
        raise RuntimeError(f"{secret}; {kwargs!r}")

    _install_fake_openai(monkeypatch, make_client)
    monkeypatch.setenv("GROQ_API_KEY", "provider-secret")

    with pytest.raises(GroqPlanningError) as raised:
        GroqPlanningModel(model="openai/gpt-oss-20b")

    assert str(raised.value) == "Groq client initialization failed."
    assert raised.value.code == "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    assert "provider-secret" not in str(raised.value)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, None, "60", math.inf, -math.inf, math.nan],
)
def test_invalid_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        GroqPlanningModel(model="groq-test", timeout=timeout, _client=FakeClient())


def test_timeout_seconds_are_forwarded_to_one_generation_call() -> None:
    adapter, client = _adapter(timeout=60.25)

    adapter.complete(prompt="prompt", response_schema=_schema())

    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["timeout"] == 60.25


@pytest.mark.parametrize("model", ["", "   ", None, 3])
def test_invalid_model_is_rejected(model) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GroqPlanningModel(model=model, _client=FakeClient())


def test_invalid_client_boundary_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"responses\.create"):
        GroqPlanningModel(model="groq-test", _client=object())


def test_invalid_prompt_or_schema_is_rejected_before_request() -> None:
    adapter, client = _adapter()

    with pytest.raises(ValueError, match="prompt"):
        adapter.complete(prompt="", response_schema=_schema())
    with pytest.raises(TypeError, match="response_schema"):
        adapter.complete(prompt="prompt", response_schema=[])  # type: ignore[arg-type]

    assert client.responses.calls == []


def test_provider_neutral_orchestration_import_does_not_import_openai(
    monkeypatch,
) -> None:
    monkeypatch.delitem(sys.modules, "openai", raising=False)
    orchestration = importlib.import_module("agent.orchestration")
    importlib.reload(orchestration)

    assert "openai" not in sys.modules
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)


def test_default_runtime_remains_offline() -> None:
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)
