"""Offline tests for AgentRuntime coordination and failure boundaries."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    ArgumentSpec,
    ErrorCategory,
    ErrorClassification,
    PlanExecutor,
    PlannerError,
    PlanStep,
    ResultContract,
    RunMode,
    RunStatus,
    StepOutputRef,
    ToolRegistry,
    ToolSpec,
    TraceEventType,
    VerificationCheck,
    VerificationResult,
)


def _classify(exception: Exception) -> ErrorClassification:
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "FAKE_FAILURE")


def _spec(
    name: str,
    function,
    *,
    arguments: dict[str, ArgumentSpec] | None = None,
    result_fields: dict[str, tuple[type, ...]] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name,
        function,
        arguments or {},
        {},
        ResultContract(f"{name}Result", result_fields or {"value": (str,)}),
        _classify,
    )


def _plan(*steps: PlanStep) -> AgentPlan:
    return AgentPlan("plan-1", "request-1", "fixed", tuple(steps))


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self.plan_value


def _request(*, mode: RunMode = RunMode.EXECUTE) -> AgentRequest:
    return AgentRequest("request-1", "fixed workflow", {}, mode)


def test_valid_execute_one_step_run_succeeds() -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.SUCCEEDED
    assert not result.planning_only
    assert result.plan == plan
    assert len(result.steps) == 1
    assert result.verification is not None and result.verification.passed
    assert not result.errors


def test_valid_execute_multi_step_run_succeeds() -> None:
    producer = Mock(return_value={"value": "upstream"})
    consumer = Mock(return_value={"value": "complete"})
    registry = ToolRegistry(
        (
            _spec("producer", producer),
            _spec(
                "consumer",
                consumer,
                arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )
    plan = _plan(
        PlanStep("producer", "producer", {}),
        PlanStep(
            "consumer",
            "consumer",
            {"source": StepOutputRef("producer", "value")},
            ("producer",),
        ),
    )

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.SUCCEEDED
    assert len(result.steps) == 2
    consumer.assert_called_once_with(source="upstream")


def test_runtime_invokes_planner_once_and_uses_its_plan() -> None:
    registry = ToolRegistry(
        (_spec("tool", Mock(return_value={"value": "done"})),)
    )
    plan = _plan(PlanStep("chosen", "tool", {}))
    planner = FixedPlanner(plan)

    result = AgentRuntime(planner=planner, registry=registry).run(_request())

    assert planner.calls == 1
    assert result.plan is plan
    assert result.steps[0].step_id == "chosen"


def test_plan_only_returns_validated_plan_without_execution() -> None:
    tool = Mock(return_value={"value": "must not run"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))
    executor = PlanExecutor(registry)
    original_preflight = executor.preflight
    executor.preflight = Mock(wraps=original_preflight)  # type: ignore[method-assign]

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, executor=executor
    ).run(_request(mode=RunMode.PLAN_ONLY))

    assert result.status is RunStatus.PLANNED
    assert result.planning_only
    assert result.plan == plan
    assert result.steps == ()
    assert result.verification is not None and result.verification.passed
    executor.preflight.assert_called_once_with(plan)
    tool.assert_not_called()


def test_invalid_plan_only_plan_fails_preflight_without_execution() -> None:
    safe_tool = Mock(return_value={"value": "must not run"})
    registry = ToolRegistry((_spec("safe", safe_tool),))
    plan = _plan(
        PlanStep("safe", "safe", {}),
        PlanStep("unknown", "arbitrary_python", {}),
    )

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(
        _request(mode=RunMode.PLAN_ONLY)
    )

    assert result.status is RunStatus.FAILED
    assert result.planning_only
    assert result.steps == ()
    assert result.errors[0].code == "UNKNOWN_TOOL"
    safe_tool.assert_not_called()










class RaisingPlanner:
    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        raise RuntimeError("unexpected planner bug")


def test_unexpected_planner_exception_becomes_internal_error() -> None:
    result = AgentRuntime(
        planner=RaisingPlanner(), registry=ToolRegistry(())
    ).run(_request())

    assert result.status is RunStatus.FAILED
    assert result.errors[0].category is ErrorCategory.INTERNAL_AGENT_ERROR
    assert result.errors[0].code == "PLANNER_UNEXPECTED_ERROR"
    assert result.errors[0].exception_type == "RuntimeError"


def test_executor_tool_failure_produces_failed_run_and_preserves_root_error() -> None:
    tool = Mock(side_effect=RuntimeError("tool failed"))
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.FAILED
    assert result.steps[0].status.value == "FAILED"
    assert result.errors[0].code == "FAKE_FAILURE"
    assert result.errors[0].category is ErrorCategory.TOOL_EXECUTION_ERROR
    assert result.verification is not None and not result.verification.passed
    assert len(result.errors) >= 2


def test_step_verification_failure_produces_failed_run() -> None:
    tool = Mock(return_value={"wrong": "result"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.FAILED
    assert result.steps[0].verification is not None
    assert not result.steps[0].verification.passed
    assert result.errors[0].category is ErrorCategory.VERIFICATION_ERROR


def test_run_level_verification_failure_changes_final_status(
    monkeypatch,
) -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))
    verification_error = AgentError(
        ErrorCategory.VERIFICATION_ERROR,
        "SYNTHETIC_RUN_FAILURE",
        "Synthetic run verification failure.",
    )
    failed_verification = VerificationResult(
        False,
        "run",
        plan.plan_id,
        (VerificationCheck("synthetic", False, "Failed."),),
        verification_error,
    )
    monkeypatch.setattr(
        "agent.orchestration.runtime.verify_run",
        Mock(return_value=failed_verification),
    )

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.FAILED
    assert result.verification == failed_verification
    assert result.errors[-1].code == "SYNTHETIC_RUN_FAILURE"


def test_successful_result_is_json_serializable_with_ordered_trace() -> None:
    registry = ToolRegistry(
        (_spec("tool", Mock(return_value={"value": "done"})),)
    )
    plan = _plan(PlanStep("step", "tool", {}))

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    json.dumps(result.to_dict())
    sequences = [event.sequence for event in result.trace]
    assert sequences == list(range(len(sequences)))
    assert len(sequences) == len(set(sequences))
    assert result.trace[-1].event_type is TraceEventType.RUN_COMPLETION


def test_runtime_has_no_provider_or_scientific_backend_requirement() -> None:
    fake = Mock(return_value={"value": "offline"})
    registry = ToolRegistry((_spec("offline", fake),))
    plan = _plan(PlanStep("step", "offline", {}))

    result = AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())

    assert result.status is RunStatus.SUCCEEDED
    fake.assert_called_once_with()


def test_runtime_does_not_swallow_keyboard_interrupt() -> None:
    tool = Mock(side_effect=KeyboardInterrupt())
    registry = ToolRegistry((_spec("tool", tool),))
    plan = _plan(PlanStep("step", "tool", {}))

    with pytest.raises(KeyboardInterrupt):
        AgentRuntime(planner=FixedPlanner(plan), registry=registry).run(_request())


def test_planner_error_is_not_allowed_to_escape_runtime() -> None:
    class FailingPlanner:
        def plan(self, request, registry):
            raise PlannerError("INVALID_REQUEST_INPUT", "Bad structured input.")

    result = AgentRuntime(
        planner=FailingPlanner(), registry=ToolRegistry(())
    ).run(_request())

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "INVALID_REQUEST_INPUT"
