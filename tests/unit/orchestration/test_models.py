"""Unit tests for registry-independent orchestration contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from agent.schemas import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    ErrorCategory,
    ExecutionTraceEvent,
    PlanStep,
    RunMode,
    RunStatus,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    TraceEventType,
    VerificationCheck,
    VerificationResult,
)


def test_valid_agent_request_is_json_serializable() -> None:
    request = AgentRequest(
        request_id="request-1",
        prompt="Inspect this scATAC dataset.",
        inputs={"path": "/data/input.h5ad", "options": [1, True, None]},
    )

    serialized = request.to_dict()

    assert serialized == {
        "request_id": "request-1",
        "prompt": "Inspect this scATAC dataset.",
        "inputs": {"options": [1, True, None], "path": "/data/input.h5ad"},
        "mode": "EXECUTE",
    }
    assert json.loads(json.dumps(serialized)) == serialized


@pytest.mark.parametrize(
    "invalid",
    [object(), {"bad": object()}, {"bad": float("nan")}, {1: "bad key"}],
)
def test_agent_request_rejects_non_json_safe_inputs(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentRequest(request_id="request-1", prompt="Inspect", inputs={"x": invalid})


def test_step_output_reference_has_explicit_serialization() -> None:
    reference = StepOutputRef(step_id="inspect", output_key="input_path")
    step = PlanStep(
        step_id="embed",
        tool_name="epizoo_embed_cells",
        arguments={"input_path": reference},
        depends_on=("inspect",),
    )

    assert step.to_dict()["arguments"]["input_path"] == {
        "$ref": {"step_id": "inspect", "output_key": "input_path"}
    }


def _multi_step_plan() -> AgentPlan:
    return AgentPlan(
        plan_id="plan-1",
        request_id="request-1",
        planner_name="test-planner",
        steps=(
            PlanStep(
                step_id="embed",
                tool_name="epizoo_embed_cells",
                arguments={
                    "input_path": StepOutputRef("inspect", "input_path"),
                    "output_dir": "/tmp/output",
                    "species": "mouse",
                },
                depends_on=("inspect",),
            ),
            PlanStep(
                step_id="independent",
                tool_name="not_yet_registered",
                arguments={},
            ),
            PlanStep(
                step_id="inspect",
                tool_name="inspect_scATAC",
                arguments={"path": "/data/input.h5ad"},
            ),
        ),
    )


def test_valid_plan_is_registry_independent_and_has_stable_topological_order() -> None:
    plan = _multi_step_plan()

    assert tuple(step.step_id for step in plan.stable_topological_steps()) == (
        "independent",
        "inspect",
        "embed",
    )
    assert json.loads(json.dumps(plan.to_dict())) == plan.to_dict()


def test_duplicate_step_ids_are_rejected() -> None:
    step = PlanStep("same", "inspect_scATAC", {"path": "a.h5ad"})
    with pytest.raises(ValueError, match="unique"):
        AgentPlan("plan", "request", "planner", (step, step))


def test_missing_dependency_is_rejected() -> None:
    step = PlanStep("embed", "epizoo_embed_cells", {}, ("missing",))
    with pytest.raises(ValueError, match="missing dependencies"):
        AgentPlan("plan", "request", "planner", (step,))


def test_self_dependency_is_rejected() -> None:
    with pytest.raises(ValueError, match="depend on itself"):
        PlanStep("inspect", "inspect_scATAC", {}, ("inspect",))


def test_dependency_cycle_is_rejected() -> None:
    first = PlanStep("first", "inspect_scATAC", {}, ("second",))
    second = PlanStep("second", "inspect_scATAC", {}, ("first",))
    with pytest.raises(ValueError, match="cycle"):
        AgentPlan("plan", "request", "planner", (first, second))


def test_reference_requires_declared_dependency() -> None:
    inspect_step = PlanStep("inspect", "inspect_scATAC", {})
    embed_step = PlanStep(
        "embed",
        "epizoo_embed_cells",
        {"input_path": StepOutputRef("inspect", "input_path")},
    )
    with pytest.raises(ValueError, match="must declare referenced step"):
        AgentPlan("plan", "request", "planner", (inspect_step, embed_step))


def test_agent_error_serialization() -> None:
    error = AgentError(
        category=ErrorCategory.RESOURCE_ERROR,
        code="RESOURCE_NOT_FOUND",
        message="Input file was not found.",
        step_id="inspect",
        tool_name="inspect_scATAC",
        exception_type="FileNotFoundError",
        attempt=1,
        details={"path": "/missing.h5ad"},
    )

    serialized = error.to_dict()
    assert serialized["category"] == "RESOURCE_ERROR"
    assert serialized["details"] == {"path": "/missing.h5ad"}
    json.dumps(serialized)


def test_agent_run_result_serialization() -> None:
    check = VerificationCheck("shape", True, "Shape is valid.")
    verification = VerificationResult(
        passed=True,
        target_type="step",
        target_id="inspect",
        checks=(check,),
    )
    step_result = StepExecutionResult(
        step_id="inspect",
        tool_name="inspect_scATAC",
        status=StepStatus.SUCCEEDED,
        attempt_count=1,
        resolved_arguments={"path": "/data/input.h5ad"},
        result={"n_cells": 2},
        verification=verification,
        duration_seconds=0.2,
    )
    event = ExecutionTraceEvent(
        sequence=0,
        event_type=TraceEventType.RUN_COMPLETION,
        timestamp="2026-01-01T00:00:00Z",
        message="Run completed.",
    )
    result = AgentRunResult(
        run_id="run-1",
        request_id="request-1",
        status=RunStatus.SUCCEEDED,
        planning_only=False,
        plan=_multi_step_plan(),
        steps=(step_result,),
        verification=verification,
        trace=(event,),
    )

    serialized = result.to_dict()
    assert serialized["status"] == "SUCCEEDED"
    assert serialized["steps"][0]["status"] == "SUCCEEDED"
    assert json.loads(json.dumps(serialized)) == serialized


def test_models_and_nested_collections_are_immutable() -> None:
    source = {"nested": ["original"]}
    request = AgentRequest("request", "prompt", source, RunMode.PLAN_ONLY)
    source["nested"].append("changed")

    assert request.to_dict()["inputs"] == {"nested": ["original"]}
    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        request.inputs["new"] = "value"  # type: ignore[index]


def test_serialization_returns_fresh_mutable_data() -> None:
    request = AgentRequest("request", "prompt", {"nested": {"value": 1}})
    first = request.to_dict()
    first["inputs"]["nested"]["value"] = 2

    assert request.to_dict()["inputs"]["nested"]["value"] == 1
