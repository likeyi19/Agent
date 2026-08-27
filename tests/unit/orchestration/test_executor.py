"""Offline tests for safe sequential plan execution."""

from __future__ import annotations

from datetime import datetime
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentPlan,
    ArgumentSpec,
    ErrorCategory,
    ErrorClassification,
    PlanExecutor,
    PlanStep,
    RecoveryPolicy,
    ResultContract,
    StepOutputRef,
    StepStatus,
    ToolRegistry,
    ToolSpec,
    TraceEventType,
)


class TransientToolError(RuntimeError):
    pass


def _classify(exception: Exception) -> ErrorClassification:
    if isinstance(exception, TransientToolError):
        return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "TRANSIENT")
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "FAKE_FAILURE")


def _spec(
    name: str,
    function,
    *,
    required_arguments: dict[str, ArgumentSpec] | None = None,
    result_fields: dict[str, tuple[type, ...]] | None = None,
    retryable: frozenset[str] = frozenset(),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        function=function,
        required_arguments=required_arguments or {},
        optional_arguments={},
        result_contract=ResultContract(
            f"{name}Result", result_fields or {"value": (str,)}
        ),
        exception_classifier=_classify,
        retryable_error_codes=retryable,
    )


def _plan(*steps: PlanStep) -> AgentPlan:
    return AgentPlan("plan-1", "request-1", "test-planner", tuple(steps))


def test_valid_one_step_execution() -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("echo", tool),))

    outcome = PlanExecutor(registry).execute(_plan(PlanStep("one", "echo", {})))

    assert tool.call_count == 1
    assert len(outcome.step_results) == 1
    step_result = outcome.step_results[0]
    assert step_result.status is StepStatus.SUCCEEDED
    assert step_result.attempt_count == 1
    assert step_result.verification is not None
    assert step_result.verification.passed
    assert not outcome.errors


def test_valid_multi_step_execution_and_reference_resolution() -> None:
    producer = Mock(return_value={"value": "canonical-input"})
    consumer = Mock(return_value={"value": "embedded"})
    registry = ToolRegistry(
        (
            _spec("producer", producer),
            _spec(
                "consumer",
                consumer,
                required_arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )
    plan = _plan(
        PlanStep("produce", "producer", {}),
        PlanStep(
            "consume",
            "consumer",
            {"source": StepOutputRef("produce", "value")},
            ("produce",),
        ),
    )

    outcome = PlanExecutor(registry).execute(plan)

    assert [result.status for result in outcome.step_results] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]
    consumer.assert_called_once_with(source="canonical-input")
    assert outcome.step_results[1].resolved_arguments["source"] == "canonical-input"


def test_stable_topological_execution_order() -> None:
    calls: list[str] = []

    def tool(name: str):
        def invoke():
            calls.append(name)
            return {"value": name}

        return invoke

    registry = ToolRegistry(
        tuple(_spec(name, tool(name)) for name in ("second", "consumer", "first"))
    )
    plan = _plan(
        PlanStep("second", "second", {}),
        PlanStep("consumer", "consumer", {}, ("first",)),
        PlanStep("first", "first", {}),
    )

    outcome = PlanExecutor(registry).execute(plan)

    assert calls == ["second", "first", "consumer"]
    assert all(result.status is StepStatus.SUCCEEDED for result in outcome.step_results)


def test_resolved_argument_is_revalidated() -> None:
    producer = Mock(return_value={"value": 7})
    consumer = Mock(return_value={"value": "unused"})
    registry = ToolRegistry(
        (
            _spec("producer", producer, result_fields={"value": (int,)}),
            _spec(
                "consumer",
                consumer,
                required_arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )
    plan = _plan(
        PlanStep("produce", "producer", {}),
        PlanStep(
            "consume",
            "consumer",
            {"source": StepOutputRef("produce", "value")},
            ("produce",),
        ),
    )

    outcome = PlanExecutor(registry).execute(plan)

    assert producer.call_count == 1
    consumer.assert_not_called()
    assert outcome.step_results[1].status is StepStatus.FAILED
    assert outcome.step_results[1].error is not None
    assert outcome.step_results[1].error.code == "RESOLVED_ARGUMENTS_INVALID"


def test_unknown_tool_is_rejected_before_any_invocation() -> None:
    valid_tool = Mock(return_value={"value": "called"})
    registry = ToolRegistry((_spec("valid", valid_tool),))
    plan = _plan(
        PlanStep("valid-first", "valid", {}),
        PlanStep("unsafe", "arbitrary_python", {}),
    )

    outcome = PlanExecutor(registry).execute(plan)

    valid_tool.assert_not_called()
    assert outcome.errors[0].code == "UNKNOWN_TOOL"
    assert all(result.attempt_count == 0 for result in outcome.step_results)


def test_invalid_later_step_arguments_reject_whole_plan() -> None:
    first = Mock(return_value={"value": "called"})
    second = Mock(return_value={"value": "called"})
    registry = ToolRegistry(
        (
            _spec("first", first),
            _spec(
                "second",
                second,
                required_arguments={"required": ArgumentSpec((str,))},
            ),
        )
    )
    plan = _plan(
        PlanStep("first", "first", {}),
        PlanStep("second", "second", {}),
    )

    outcome = PlanExecutor(registry).execute(plan)

    first.assert_not_called()
    second.assert_not_called()
    assert outcome.errors[0].code == "INVALID_TOOL_ARGUMENTS"


def test_reference_producer_must_be_declared_dependency() -> None:
    producer = PlanStep("producer", "source", {})
    consumer = PlanStep(
        "consumer",
        "sink",
        {"source": StepOutputRef("producer", "value")},
    )

    with pytest.raises(ValueError, match="must declare referenced step"):
        _plan(producer, consumer)


def test_unknown_referenced_output_key_fails_preflight_without_invocation() -> None:
    producer = Mock(return_value={"value": "called"})
    consumer = Mock(return_value={"value": "called"})
    registry = ToolRegistry(
        (
            _spec("producer", producer),
            _spec(
                "consumer",
                consumer,
                required_arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )
    plan = _plan(
        PlanStep("producer", "producer", {}),
        PlanStep(
            "consumer",
            "consumer",
            {"source": StepOutputRef("producer", "missing")},
            ("producer",),
        ),
    )

    outcome = PlanExecutor(registry).execute(plan)

    producer.assert_not_called()
    consumer.assert_not_called()
    assert outcome.errors[0].code == "INVALID_OUTPUT_REFERENCE"


def test_unverified_dependency_result_never_reaches_consumer() -> None:
    producer = Mock(return_value={"wrong": "not verified"})
    consumer = Mock(return_value={"value": "called"})
    registry = ToolRegistry(
        (
            _spec("producer", producer),
            _spec(
                "consumer",
                consumer,
                required_arguments={"source": ArgumentSpec((str,))},
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

    outcome = PlanExecutor(registry).execute(plan)

    assert outcome.step_results[0].status is StepStatus.FAILED
    assert outcome.step_results[0].verification is not None
    assert not outcome.step_results[0].verification.passed
    assert outcome.step_results[1].status is StepStatus.SKIPPED
    consumer.assert_not_called()


def test_callable_receives_exact_planned_arguments_without_mutation() -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry(
        (
            _spec(
                "tool",
                tool,
                required_arguments={
                    "path": ArgumentSpec((str,)),
                    "overwrite": ArgumentSpec((bool,)),
                },
            ),
        )
    )
    arguments = {"path": "/data/input.h5ad", "overwrite": False}

    outcome = PlanExecutor(registry).execute(
        _plan(PlanStep("step", "tool", arguments))
    )

    tool.assert_called_once_with(**arguments)
    assert dict(outcome.step_results[0].resolved_arguments) == arguments
    assert arguments == {"path": "/data/input.h5ad", "overwrite": False}


def test_tool_success_but_verification_failure_marks_failed() -> None:
    tool = Mock(return_value={"wrong": "shape"})
    registry = ToolRegistry((_spec("tool", tool),))

    outcome = PlanExecutor(registry).execute(_plan(PlanStep("step", "tool", {})))

    result = outcome.step_results[0]
    assert result.status is StepStatus.FAILED
    assert result.verification is not None and not result.verification.passed
    assert result.error is not None
    assert result.error.category is ErrorCategory.VERIFICATION_ERROR


def test_tool_exception_is_classified() -> None:
    tool = Mock(side_effect=ValueError("fake failure"))
    registry = ToolRegistry((_spec("tool", tool),))

    outcome = PlanExecutor(registry).execute(_plan(PlanStep("step", "tool", {})))

    error = outcome.step_results[0].error
    assert error is not None
    assert error.code == "FAKE_FAILURE"
    assert error.exception_type == "ValueError"
    assert error.attempt == 1
    assert not error.recoverable


def test_permanent_failure_skips_downstream_and_remaining_steps() -> None:
    failing = Mock(side_effect=RuntimeError("failed"))
    dependent = Mock(return_value={"value": "dependent"})
    independent = Mock(return_value={"value": "independent"})
    registry = ToolRegistry(
        (
            _spec("failing", failing),
            _spec("dependent", dependent),
            _spec("independent", independent),
        )
    )
    plan = _plan(
        PlanStep("failure", "failing", {}),
        PlanStep("dependent", "dependent", {}, ("failure",)),
        PlanStep("independent", "independent", {}),
    )

    outcome = PlanExecutor(registry).execute(plan)

    assert [result.status for result in outcome.step_results] == [
        StepStatus.FAILED,
        StepStatus.SKIPPED,
        StepStatus.SKIPPED,
    ]
    assert outcome.step_results[1].error.code == "DEPENDENCY_FAILED"
    assert outcome.step_results[2].error.code == "EXECUTION_ABORTED"
    dependent.assert_not_called()
    independent.assert_not_called()
    assert len(outcome.step_results) == len(plan.steps)


def test_retryable_failure_succeeds_on_second_identical_attempt() -> None:
    received: list[dict[str, object]] = []

    def flaky(**kwargs):
        received.append(dict(kwargs))
        if len(received) == 1:
            raise TransientToolError("try again")
        return {"value": "done"}

    registry = ToolRegistry(
        (
            _spec(
                "flaky",
                flaky,
                required_arguments={"value": ArgumentSpec((str,))},
                retryable=frozenset({"TRANSIENT"}),
            ),
        )
    )

    outcome = PlanExecutor(registry).execute(
        _plan(PlanStep("step", "flaky", {"value": "fixed"}))
    )

    assert outcome.step_results[0].status is StepStatus.SUCCEEDED
    assert outcome.step_results[0].attempt_count == 2
    assert received == [{"value": "fixed"}, {"value": "fixed"}]
    assert any(event.event_type is TraceEventType.RECOVERY for event in outcome.trace)


def test_retryable_failure_stops_at_configured_maximum() -> None:
    tool = Mock(side_effect=TransientToolError("still unavailable"))
    registry = ToolRegistry(
        (
            _spec(
                "flaky",
                tool,
                retryable=frozenset({"TRANSIENT"}),
            ),
        )
    )
    executor = PlanExecutor(
        registry, recovery_policy=RecoveryPolicy(max_attempts_per_step=2)
    )

    outcome = executor.execute(_plan(PlanStep("step", "flaky", {})))

    assert tool.call_count == 2
    assert outcome.step_results[0].attempt_count == 2
    assert outcome.step_results[0].status is StepStatus.FAILED


def test_nonretryable_failure_gets_one_attempt_even_with_larger_bound() -> None:
    tool = Mock(side_effect=RuntimeError("permanent"))
    registry = ToolRegistry((_spec("tool", tool),))
    executor = PlanExecutor(
        registry, recovery_policy=RecoveryPolicy(max_attempts_per_step=5)
    )

    outcome = executor.execute(_plan(PlanStep("step", "tool", {})))

    assert tool.call_count == 1
    assert outcome.step_results[0].attempt_count == 1


def test_trace_is_ordered_json_serializable_and_timed() -> None:
    registry = ToolRegistry(
        (_spec("tool", Mock(return_value={"value": "done"})),)
    )

    outcome = PlanExecutor(registry).execute(_plan(PlanStep("step", "tool", {})))

    assert [event.sequence for event in outcome.trace] == list(range(len(outcome.trace)))
    json.dumps([event.to_dict() for event in outcome.trace])
    step_result = outcome.step_results[0]
    assert datetime.fromisoformat(step_result.started_at).tzinfo is not None
    assert datetime.fromisoformat(step_result.finished_at).tzinfo is not None
    assert step_result.duration_seconds is not None
    assert step_result.duration_seconds >= 0


def test_executor_never_resolves_callable_from_plan_text() -> None:
    safe = Mock(return_value={"value": "safe"})
    registry = ToolRegistry((_spec("safe", safe),))
    outcome = PlanExecutor(registry).execute(
        _plan(PlanStep("unsafe", "builtins.eval", {}))
    )

    safe.assert_not_called()
    assert outcome.errors[0].code == "UNKNOWN_TOOL"


def test_keyboard_interrupt_is_not_swallowed() -> None:
    tool = Mock(side_effect=KeyboardInterrupt())
    registry = ToolRegistry((_spec("tool", tool),))

    with pytest.raises(KeyboardInterrupt):
        PlanExecutor(registry).execute(_plan(PlanStep("step", "tool", {})))
