"""Network-free tests for the optional OpenAI planning adapter."""

from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    DeterministicPlanner,
    LLMPlanner,
    PlanningModel,
    PlanningWireMode,
    build_default_tool_registry,
    build_semantic_planning_prompt,
    build_semantic_wire_v4_schema,
)
from agent.orchestration.llm_planner import _response_schema
from agent.providers import (
    OpenAIPlanningDependencyError,
    OpenAIPlanningError,
    OpenAIPlanningModel,
)


_OUTPUT = (
    '{"schema_version":3,"status":"unsupported","steps":[],'
    '"reason":"safe"}'
)
_V4_OUTPUT = json.dumps(
    {
        "schema_version": 4,
        "decision": {
            "kind": "plan",
            "steps": [
                {
                    "step_id": "inspect",
                    "tool": "inspect_scATAC",
                    "sources": [],
                    "control_dependencies": [],
                }
            ],
        },
    }
)


def _response(
    *,
    status: str = "completed",
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
    model: str = "gpt-test",
    timeout: float = 12.5,
) -> tuple[OpenAIPlanningModel, FakeClient]:
    client = FakeClient(result, error=error)
    return (
        OpenAIPlanningModel(
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


def test_adapter_satisfies_planning_model_protocol() -> None:
    adapter, _ = _adapter()

    assert isinstance(adapter, PlanningModel)


def test_configured_model_identity_and_timeout_are_stable() -> None:
    adapter, _ = _adapter(model=" gpt-planner-2026 ", timeout=17)

    assert adapter.model == "gpt-planner-2026"
    assert adapter.model_id == "openai:gpt-planner-2026"
    assert adapter.timeout == 17.0


def test_responses_api_request_maps_prompt_and_schema_exactly() -> None:
    adapter, client = _adapter()
    prompt = "exact planning prompt"
    response_schema = _schema()
    plain_schema = json.loads(json.dumps(response_schema))

    returned = adapter.complete(
        prompt=prompt,
        response_schema=response_schema,
    )

    assert returned == _OUTPUT
    assert len(client.responses.calls) == 1
    request = client.responses.calls[0]
    assert request["model"] == "gpt-test"
    assert request["input"] == prompt
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "agent_plan",
            "strict": True,
            "schema": plain_schema,
        }
    }
    assert plain_schema["properties"]["schema_version"]["enum"] == [3]
    assert "anyOf" in plain_schema["properties"]["steps"]["items"]
    assert request["store"] is False
    assert request["background"] is False
    assert request["timeout"] == 12.5
    assert "tools" not in request
    assert "previous_response_id" not in request
    assert "conversation" not in request


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


def test_semantic_v4_round_trip_uses_generic_responses_transport() -> None:
    secret = "/private/OPENAI-V4-SECRET.h5ad"
    request = AgentRequest(
        "openai-v4", "Inspect the supplied dataset.", {"input_path": secret}
    )
    registry = build_default_tool_registry()
    adapter, client = _adapter(_response(output_text=_V4_OUTPUT))

    plan = LLMPlanner(adapter, wire_mode=PlanningWireMode.V4).plan(
        request, registry
    )

    transmitted = client.responses.calls[0]
    assert transmitted["input"] == build_semantic_planning_prompt(
        request, registry
    )
    assert transmitted["text"]["format"]["schema"] == json.loads(
        json.dumps(build_semantic_wire_v4_schema(registry, request))
    )
    serialized = json.dumps(transmitted, sort_keys=True)
    assert secret not in serialized
    assert "tools" not in transmitted
    schema = transmitted["text"]["format"]["schema"]
    assert [
        step["properties"]["tool"]["enum"][0]
        for step in schema["$defs"]["step"]["anyOf"]
    ] == sorted(registry.names())
    assert schema["$defs"]["input_name"]["enum"] == ["input_path"]
    assert plan.steps[0].arguments == {"path": secret}


@pytest.mark.parametrize("missing", [None, "", "   ", 3])
def test_missing_output_fails_cleanly(missing) -> None:
    adapter, _ = _adapter(_response(output_text=missing))

    with pytest.raises(OpenAIPlanningError, match="did not contain output text"):
        adapter.complete(prompt="prompt", response_schema=_schema())


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", None])
def test_noncompleted_response_fails_cleanly(status) -> None:
    adapter, _ = _adapter(_response(status=status))

    with pytest.raises(OpenAIPlanningError, match="was not completed") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())
    assert raised.value.code == "PROVIDER_COMPLETION_INCOMPLETE"


def test_provider_reported_error_fails_without_exposing_error_object() -> None:
    adapter, _ = _adapter(
        _response(error={"authorization": "Bearer provider-secret"})
    )

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert "provider-secret" not in str(raised.value)
    assert "authorization" not in str(raised.value).casefold()


def test_refusal_fails_without_exposing_refusal_text() -> None:
    refusal_text = "refusal containing provider-secret"
    refusal = SimpleNamespace(type="refusal", refusal=refusal_text)
    message = SimpleNamespace(type="message", content=[refusal])
    adapter, _ = _adapter(_response(output=[message], output_text=""))

    with pytest.raises(OpenAIPlanningError, match="was refused") as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert refusal_text not in str(raised.value)
    assert raised.value.code == "PROVIDER_REFUSED"


def test_sdk_exception_is_converted_to_sanitized_adapter_error() -> None:
    secret = "Authorization: Bearer api-secret; raw HTTP response"
    adapter, client = _adapter(error=RuntimeError(secret))

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert len(client.responses.calls) == 1
    assert str(raised.value) == "OpenAI planning request failed."
    assert "api-secret" not in str(raised.value)
    assert "HTTP" not in str(raised.value)


def test_http_413_is_terminal_request_too_large_without_raw_body() -> None:
    error = RuntimeError("TPM limit and private provider response")
    error.status_code = 413
    adapter, _ = _adapter(error=error)

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert raised.value.code == "PROVIDER_REQUEST_TOO_LARGE"
    assert raised.value.retry_after_seconds is None
    assert str(raised.value) == (
        "The planning request exceeds the provider's accepted size."
    )
    assert "TPM" not in str(raised.value)
    assert "private" not in str(raised.value)


def test_numeric_retry_after_is_normalized_without_persisting_headers() -> None:
    error = RuntimeError("raw provider body")
    error.status_code = 429
    error.response = SimpleNamespace(
        headers={"Retry-After": "2.5", "Authorization": "Bearer secret"}
    )
    adapter, _ = _adapter(error=error)

    with pytest.raises(OpenAIPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert raised.value.code == "PROVIDER_RATE_LIMITED"
    assert raised.value.retry_after_seconds == 2.5
    assert "Authorization" not in str(raised.value)
    assert "secret" not in str(raised.value)


def test_credentials_are_not_copied_into_request_or_result(monkeypatch) -> None:
    secret = "sk-test-do-not-copy"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    adapter, client = _adapter()

    result = adapter.complete(prompt="prompt", response_schema=_schema())

    assert secret not in result
    assert secret not in json.dumps(client.responses.calls)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, None, "60", math.inf, -math.inf, math.nan],
)
def test_invalid_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        OpenAIPlanningModel(model="gpt-test", timeout=timeout, _client=FakeClient())


@pytest.mark.parametrize("model", ["", "   ", None, 3])
def test_invalid_model_is_rejected(model) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OpenAIPlanningModel(model=model, _client=FakeClient())


def test_invalid_client_boundary_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"responses\.create"):
        OpenAIPlanningModel(model="gpt-test", _client=object())


def test_invalid_prompt_or_schema_is_rejected_before_request() -> None:
    adapter, client = _adapter()

    with pytest.raises(ValueError, match="prompt"):
        adapter.complete(prompt="", response_schema=_schema())
    with pytest.raises(TypeError, match="response_schema"):
        adapter.complete(prompt="prompt", response_schema=[])  # type: ignore[arg-type]

    assert client.responses.calls == []


def test_optional_sdk_absence_has_clear_lazy_error(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "openai", None)

    with pytest.raises(
        OpenAIPlanningDependencyError, match="optional OpenAI SDK"
    ) as raised:
        OpenAIPlanningModel(model="gpt-test")
    assert raised.value.code == "PLANNING_PROVIDER_DEPENDENCY_MISSING"


def test_provider_neutral_orchestration_import_does_not_import_openai(
    monkeypatch,
) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    orchestration = importlib.import_module("agent.orchestration")
    importlib.reload(orchestration)

    assert "openai" not in sys.modules
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)


def test_default_client_disables_sdk_retries(monkeypatch) -> None:
    import sys
    from types import ModuleType

    calls: list[dict[str, object]] = []
    fake = ModuleType("openai")
    fake.OpenAI = lambda **kwargs: calls.append(kwargs) or FakeClient()
    monkeypatch.setitem(sys.modules, "openai", fake)

    OpenAIPlanningModel(model="gpt-test")

    assert calls == [{"max_retries": 0}]
