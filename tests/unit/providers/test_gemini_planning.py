"""Network-free tests for the optional Gemini planning adapter."""

from __future__ import annotations

import importlib
import json
import math
from types import SimpleNamespace

import pytest

from agent.orchestration import (
    AgentRuntime,
    DeterministicPlanner,
    PlanningModel,
)
from agent.providers import (
    GeminiPlanningDependencyError,
    GeminiPlanningError,
    GeminiPlanningModel,
)


_OUTPUT = (
    '{"schema_version":2,"status":"unsupported","steps":[],'
    '"reason":"safe"}'
)


def _interaction(
    *,
    status: object = "completed",
    output_text: object = _OUTPUT,
    error: object = None,
):
    return SimpleNamespace(
        status=status,
        output_text=output_text,
        error=error,
    )


class FakeInteractions:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.result = _interaction() if result is None else result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    def __init__(self, result=None, *, error: Exception | None = None) -> None:
        self.interactions = FakeInteractions(result, error=error)


def _adapter(
    result=None,
    *,
    error: Exception | None = None,
    model: str = "gemini-test",
    timeout: float = 12.5,
) -> tuple[GeminiPlanningModel, FakeClient]:
    client = FakeClient(result, error=error)
    return (
        GeminiPlanningModel(
            model=model,
            timeout=timeout,
            _client=client,
        ),
        client,
    )


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "enum": (2,)},
            "status": {"type": "string", "enum": ("plan", "unsupported")},
            "steps": {"type": "array", "items": {"type": "object"}},
            "reason": {"type": ("string", "null")},
        },
        "required": ("schema_version", "status", "steps", "reason"),
        "additionalProperties": False,
    }


def test_adapter_satisfies_planning_model_protocol() -> None:
    adapter, _ = _adapter()

    assert isinstance(adapter, PlanningModel)


def test_configured_model_identity_and_timeout_are_stable() -> None:
    adapter, _ = _adapter(model=" gemini-2.5-flash ", timeout=17)

    assert adapter.model == "gemini-2.5-flash"
    assert adapter.model_id == "gemini:gemini-2.5-flash"
    assert adapter.timeout == 17.0


def test_interactions_request_maps_prompt_and_v2_schema_exactly() -> None:
    adapter, client = _adapter()
    prompt = "exact planning prompt"
    response_schema = _schema()

    returned = adapter.complete(
        prompt=prompt,
        response_schema=response_schema,
    )

    assert returned == _OUTPUT
    assert len(client.interactions.calls) == 1
    request = client.interactions.calls[0]
    assert request == {
        "model": "gemini-test",
        "input": prompt,
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {
                    "schema_version": {"type": "integer", "enum": [2]},
                    "status": {
                        "type": "string",
                        "enum": ["plan", "unsupported"],
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                    "reason": {"type": ["string", "null"]},
                },
                "required": ["schema_version", "status", "steps", "reason"],
                "additionalProperties": False,
            },
        },
        "store": False,
        "background": False,
        "timeout": 12.5,
    }


def test_request_is_stateless_and_enables_no_tools() -> None:
    adapter, client = _adapter()

    adapter.complete(prompt="prompt", response_schema=_schema())

    request = client.interactions.calls[0]
    assert request["store"] is False
    assert request["background"] is False
    for forbidden in (
        "tools",
        "previous_interaction_id",
        "agent",
        "environment",
        "system_instruction",
    ):
        assert forbidden not in request


def test_adapter_adds_no_dataset_or_file_values() -> None:
    adapter, client = _adapter()
    prompt = "prompt-without-local-data"

    adapter.complete(prompt=prompt, response_schema=_schema())

    serialized = json.dumps(client.interactions.calls[0], sort_keys=True)
    assert "/data/" not in serialized
    assert "h5ad" not in serialized
    assert client.interactions.calls[0]["input"] == prompt


def test_success_returns_exact_output_text_without_normalization() -> None:
    exact = "  {\n  \"status\": \"plan\"\n}\n"
    adapter, _ = _adapter(_interaction(output_text=exact))

    returned = adapter.complete(prompt="prompt", response_schema=_schema())

    assert returned == exact


@pytest.mark.parametrize("missing", [None, "", "   ", 3])
def test_missing_output_fails_cleanly(missing) -> None:
    adapter, _ = _adapter(_interaction(output_text=missing))

    with pytest.raises(GeminiPlanningError, match="did not contain output text"):
        adapter.complete(prompt="prompt", response_schema=_schema())


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", None])
def test_noncompleted_interaction_fails_cleanly(status) -> None:
    adapter, _ = _adapter(_interaction(status=status))

    with pytest.raises(GeminiPlanningError, match="was not completed"):
        adapter.complete(prompt="prompt", response_schema=_schema())


def test_provider_reported_error_fails_without_exposing_error_object() -> None:
    adapter, _ = _adapter(
        _interaction(error={"authorization": "Bearer provider-secret"})
    )

    with pytest.raises(GeminiPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert "provider-secret" not in str(raised.value)
    assert "authorization" not in str(raised.value).casefold()


def test_sdk_exception_is_converted_to_sanitized_adapter_error() -> None:
    secret = "Authorization: Bearer api-secret; raw HTTP response; request-id"
    adapter, client = _adapter(error=RuntimeError(secret))

    with pytest.raises(GeminiPlanningError) as raised:
        adapter.complete(prompt="prompt", response_schema=_schema())

    assert len(client.interactions.calls) == 1
    assert str(raised.value) == "Gemini planning request failed."
    assert "api-secret" not in str(raised.value)
    assert "HTTP" not in str(raised.value)
    assert "request-id" not in str(raised.value)


def test_credentials_are_not_copied_into_request_or_result(monkeypatch) -> None:
    secret = "gemini-test-do-not-copy"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    adapter, client = _adapter()

    result = adapter.complete(prompt="prompt", response_schema=_schema())

    assert secret not in result
    assert secret not in json.dumps(client.interactions.calls)


@pytest.mark.parametrize(
    "timeout",
    [0, -1, True, None, "60", math.inf, -math.inf, math.nan],
)
def test_invalid_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        GeminiPlanningModel(
            model="gemini-test", timeout=timeout, _client=FakeClient()
        )


def test_timeout_seconds_are_forwarded_without_unit_conversion() -> None:
    adapter, client = _adapter(timeout=60.25)

    adapter.complete(prompt="prompt", response_schema=_schema())

    assert client.interactions.calls[0]["timeout"] == 60.25


@pytest.mark.parametrize("model", ["", "   ", None, 3])
def test_invalid_model_is_rejected(model) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        GeminiPlanningModel(model=model, _client=FakeClient())


def test_invalid_client_boundary_is_rejected() -> None:
    with pytest.raises(TypeError, match=r"interactions\.create"):
        GeminiPlanningModel(model="gemini-test", _client=object())


def test_invalid_prompt_or_schema_is_rejected_before_request() -> None:
    adapter, client = _adapter()

    with pytest.raises(ValueError, match="prompt"):
        adapter.complete(prompt="", response_schema=_schema())
    with pytest.raises(TypeError, match="response_schema"):
        adapter.complete(prompt="prompt", response_schema=[])  # type: ignore[arg-type]

    assert client.interactions.calls == []


def test_missing_optional_sdk_has_clear_lazy_error(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "google.genai", None)

    with pytest.raises(
        GeminiPlanningDependencyError, match="optional Google Gen AI SDK"
    ):
        GeminiPlanningModel(model="gemini-test")


def test_client_initialization_failure_is_sanitized(monkeypatch) -> None:
    import sys
    from types import ModuleType

    secret = "GEMINI_API_KEY=provider-secret"

    class FailingGenAI:
        @staticmethod
        def Client():
            raise RuntimeError(secret)

    fake_genai = FailingGenAI()
    google = ModuleType("google")
    google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    with pytest.raises(GeminiPlanningError) as raised:
        GeminiPlanningModel(model="gemini-test")

    assert str(raised.value).startswith("Gemini client initialization failed")
    assert "provider-secret" not in str(raised.value)


def test_provider_neutral_orchestration_import_does_not_import_gemini_sdk(
    monkeypatch,
) -> None:
    import sys

    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    orchestration = importlib.import_module("agent.orchestration")
    importlib.reload(orchestration)

    assert "google.genai" not in sys.modules
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)


def test_default_runtime_remains_offline() -> None:
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)
