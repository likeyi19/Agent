"""Offline recovery proof through the real provider transport adapters."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanningModelProfile,
    PlanningWireMode,
    RunMode,
    RunStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.providers import (
    GeminiPlanningModel,
    GroqPlanningModel,
    OpenAIPlanningModel,
    PlanningModelFactoryRegistry,
)


def _payload(*, input_name: str | None = None) -> str:
    sources = []
    if input_name is not None:
        sources.append(
            {"kind": "input", "target": "dataset", "input": input_name}
        )
    return json.dumps(
        {
            "schema_version": 4,
            "decision": {
                "kind": "plan",
                "steps": [
                    {
                        "step_id": "inspect",
                        "tool": "inspect_scATAC",
                        "sources": sources,
                        "control_dependencies": [],
                    }
                ],
            },
        }
    )


def _completed(output: str) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed", output_text=output, output=(), error=None
    )


class _QueueEndpoint:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _ResponsesClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.responses = _QueueEndpoint(outcomes)


class _InteractionsClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.interactions = _QueueEndpoint(outcomes)


def _profile(profile_id: str, provider_id: str, model_id: str):
    return PlanningModelProfile(profile_id, provider_id, model_id)


def _request(request_id: str) -> AgentRequest:
    return AgentRequest(
        request_id,
        "Inspect the supplied dataset.",
        {
            "input_path": "/private/TRANSPORT-RECOVERY-SECRET.h5ad",
            "output_dir": "/private/UNAUTHORIZED-SECRET",
        },
        RunMode.PLAN_ONLY,
    )


def _run(planner: LLMPlanner, request_id: str):
    source = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("PLAN_ONLY executed a tool"))
    registry = ToolRegistry(
        tuple(
            replace(source.get(name), function=guard) for name in source.names()
        )
    )
    result = AgentRuntime(planner=planner, registry=registry).run(
        _request(request_id)
    )
    guard.assert_not_called()
    return result


def test_openai_malformed_semantic_response_repairs_through_transport() -> None:
    client = _ResponsesClient([_completed("not-json"), _completed(_payload())])
    adapter = OpenAIPlanningModel(model="openai-test", _client=client)

    result = _run(
        LLMPlanner(
            adapter,
            wire_mode=PlanningWireMode.V4,
            profile=_profile("primary-openai", "openai", "openai-test"),
            retry_sleeper=lambda _: None,
        ),
        "openai-v4-repair",
    )

    assert result.status is RunStatus.PLANNED
    assert len(client.responses.calls) == 2
    assert "repair" in json.loads(client.responses.calls[1]["input"])
    assert client.responses.calls[0]["text"]["format"]["schema"] == (
        client.responses.calls[1]["text"]["format"]["schema"]
    )


def test_groq_semantic_compiler_failure_repairs_through_transport() -> None:
    client = _ResponsesClient(
        [
            _completed(_payload(input_name="output_dir")),
            _completed(_payload()),
        ]
    )
    adapter = GroqPlanningModel(model="groq-test", _client=client)

    result = _run(
        LLMPlanner(
            adapter,
            wire_mode=PlanningWireMode.V4,
            profile=_profile("primary-groq", "groq", "groq-test"),
            retry_sleeper=lambda _: None,
        ),
        "groq-v4-compiler-repair",
    )

    assert result.status is RunStatus.PLANNED
    assert len(client.responses.calls) == 2
    diagnostic = json.loads(client.responses.calls[1]["input"])["repair"][
        "diagnostic"
    ]
    assert diagnostic["reason_code"] == "unauthorized_request_source"
    assert diagnostic["target_port"] == "dataset"


def test_gemini_transient_error_retries_same_v4_transport_request() -> None:
    transient = RuntimeError("private provider body")
    transient.status_code = 429
    client = _InteractionsClient([transient, _completed(_payload())])
    adapter = GeminiPlanningModel(model="gemini-test", _client=client)

    result = _run(
        LLMPlanner(
            adapter,
            wire_mode=PlanningWireMode.V4,
            profile=_profile("primary-gemini", "gemini", "gemini-test"),
            retry_sleeper=lambda _: None,
        ),
        "gemini-v4-retry",
    )

    assert result.status is RunStatus.PLANNED
    assert len(client.interactions.calls) == 2
    assert client.interactions.calls[0] == client.interactions.calls[1]


def test_openai_primary_fails_over_to_groq_v4_at_global_three_call_ceiling() -> None:
    primary_client = _ResponsesClient(
        [_completed("bad-primary-1"), _completed("bad-primary-2")]
    )
    secondary_client = _ResponsesClient([_completed(_payload())])
    primary = OpenAIPlanningModel(model="openai-primary", _client=primary_client)
    secondary = GroqPlanningModel(model="groq-secondary", _client=secondary_client)
    primary_profile = _profile(
        "primary-openai", "openai", "openai-primary"
    )
    secondary_profile = _profile(
        "secondary-groq", "groq", "groq-secondary"
    )
    factories = PlanningModelFactoryRegistry(
        {"groq": lambda received: secondary}
    )

    result = _run(
        LLMPlanner(
            primary,
            wire_mode=PlanningWireMode.V4,
            profile=primary_profile,
            retry_sleeper=lambda _: None,
            recovery_profiles=(secondary_profile,),
            model_factory_registry=factories,
        ),
        "openai-to-groq-v4-failover",
    )

    assert result.status is RunStatus.PLANNED
    assert len(primary_client.responses.calls) == 2
    assert len(secondary_client.responses.calls) == 1
    assert result.plan is not None
    assert result.plan.planner_name == "llm-profile:secondary-groq:wire-v4"
    secondary_schema = secondary_client.responses.calls[0]["text"]["format"][
        "schema"
    ]
    assert secondary_schema["properties"]["schema_version"]["enum"] == [4]
