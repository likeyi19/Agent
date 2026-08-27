"""Sequential, registry-bound execution of preflighted Agent plans."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from time import perf_counter
from typing import Mapping, cast

from agent.schemas import (
    AgentError,
    AgentPlan,
    ErrorCategory,
    ExecutionTraceEvent,
    JsonValue,
    PlanStep,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    TraceEventType,
    VerificationCheck,
    VerificationResult,
)

from .registry import ToolArgumentError, ToolRegistry, UnknownToolError
from .verifier import verify_step


@dataclass(frozen=True)
class RecoveryPolicy:
    """Strict upper bound for explicitly permitted same-argument retries."""

    max_attempts_per_step: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts_per_step, bool)
            or not isinstance(self.max_attempts_per_step, int)
            or self.max_attempts_per_step < 1
        ):
            raise ValueError("`max_attempts_per_step` must be a positive integer.")


@dataclass(frozen=True)
class ExecutionOutcome:
    """Executor-owned data used by AgentRuntime to build its final result."""

    step_results: tuple[StepExecutionResult, ...]
    errors: tuple[AgentError, ...]
    trace: tuple[ExecutionTraceEvent, ...]


class _TraceRecorder:
    def __init__(self) -> None:
        self._events: list[ExecutionTraceEvent] = []

    @property
    def events(self) -> tuple[ExecutionTraceEvent, ...]:
        return tuple(self._events)

    def add(
        self,
        event_type: TraceEventType,
        message: str,
        *,
        step_id: str | None = None,
        attempt: int | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self._events.append(
            ExecutionTraceEvent(
                sequence=len(self._events),
                event_type=event_type,
                timestamp=_utc_now(),
                message=message,
                step_id=step_id,
                attempt=attempt,
                details=details or {},
            )
        )

    def extend(self, events: tuple[ExecutionTraceEvent, ...]) -> None:
        for event in events:
            self._events.append(
                ExecutionTraceEvent(
                    sequence=len(self._events),
                    event_type=event.event_type,
                    timestamp=event.timestamp,
                    message=event.message,
                    step_id=event.step_id,
                    attempt=event.attempt,
                    details=event.details,
                )
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy_json_value(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            copied[key] = _copy_json_value(nested, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return tuple(
            _copy_json_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}.")


def _copy_json_mapping(value: object, path: str) -> dict[str, JsonValue]:
    copied = _copy_json_value(value, path)
    if not isinstance(copied, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return dict(copied)


def _preflight_failure(
    plan: AgentPlan,
    failures: list[tuple[str, str, str]],
    checks: list[VerificationCheck],
) -> VerificationResult:
    code, step_id, message = failures[0]
    tool_name = next(
        (step.tool_name for step in plan.steps if step.step_id == step_id), None
    )
    return VerificationResult(
        passed=False,
        target_type="plan",
        target_id=plan.plan_id,
        checks=tuple(checks),
        error=AgentError(
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
            code=code,
            message="; ".join(failure[2] for failure in failures),
            step_id=step_id or None,
            tool_name=tool_name,
            details={
                "failed_step_ids": tuple(
                    failure[1] for failure in failures if failure[1]
                ),
                "failure_count": len(failures),
            },
        ),
    )


class _ReferenceResolutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PlanExecutor:
    """Execute a complete plan sequentially through an immutable ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        recovery_policy: RecoveryPolicy | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")
        if recovery_policy is not None and not isinstance(
            recovery_policy, RecoveryPolicy
        ):
            raise TypeError("`recovery_policy` must be a RecoveryPolicy or None.")
        self._registry = registry
        self._recovery_policy = (
            recovery_policy if recovery_policy is not None else RecoveryPolicy()
        )

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def recovery_policy(self) -> RecoveryPolicy:
        return self._recovery_policy

    def preflight(self, plan: AgentPlan) -> VerificationResult:
        """Validate the entire plan against the registry without side effects."""

        if not isinstance(plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan.")

        checks: list[VerificationCheck] = []
        failures: list[tuple[str, str, str]] = []
        try:
            plan.stable_topological_steps()
        except ValueError as exc:
            checks.append(
                VerificationCheck("plan_structure", False, f"Invalid plan: {exc}")
            )
            failures.append(("INVALID_PLAN_STRUCTURE", "", str(exc)))
        else:
            checks.append(
                VerificationCheck(
                    "plan_structure", True, "Plan dependency structure is valid."
                )
            )

        steps_by_id = {step.step_id: step for step in plan.steps}
        for step in plan.steps:
            try:
                self._registry.validate_arguments(step.tool_name, step.arguments)
            except UnknownToolError as exc:
                checks.append(
                    VerificationCheck(
                        f"{step.step_id}_registry_contract", False, str(exc)
                    )
                )
                failures.append(("UNKNOWN_TOOL", step.step_id, str(exc)))
                continue
            except ToolArgumentError as exc:
                checks.append(
                    VerificationCheck(
                        f"{step.step_id}_registry_contract", False, str(exc)
                    )
                )
                failures.append(("INVALID_TOOL_ARGUMENTS", step.step_id, str(exc)))
                continue
            checks.append(
                VerificationCheck(
                    f"{step.step_id}_registry_contract",
                    True,
                    f"Step {step.step_id!r} arguments satisfy the registry contract.",
                )
            )

            for argument_name, argument in step.arguments.items():
                if not isinstance(argument, StepOutputRef):
                    continue
                producer = steps_by_id.get(argument.step_id)
                reference_valid = (
                    producer is not None
                    and argument.step_id in step.depends_on
                    and argument.step_id != step.step_id
                )
                if reference_valid and producer is not None:
                    try:
                        producer_spec = self._registry.get(producer.tool_name)
                    except UnknownToolError:
                        reference_valid = False
                    else:
                        reference_valid = (
                            argument.output_key
                            in producer_spec.result_contract.required_fields
                        )
                if reference_valid:
                    checks.append(
                        VerificationCheck(
                            f"{step.step_id}_{argument_name}_reference",
                            True,
                            "Output reference names a declared dependency result key.",
                        )
                    )
                else:
                    message = (
                        f"Step {step.step_id!r} argument {argument_name!r} has an "
                        f"invalid output reference to {argument.step_id!r}."
                    )
                    checks.append(
                        VerificationCheck(
                            f"{step.step_id}_{argument_name}_reference",
                            False,
                            message,
                        )
                    )
                    failures.append(
                        ("INVALID_OUTPUT_REFERENCE", step.step_id, message)
                    )

        if failures:
            return _preflight_failure(plan, failures, checks)
        return VerificationResult(
            passed=True,
            target_type="plan",
            target_id=plan.plan_id,
            checks=tuple(checks),
        )

    def _resolve_arguments(
        self,
        step: PlanStep,
        verified_results: Mapping[str, Mapping[str, JsonValue]],
        trace: _TraceRecorder,
    ) -> dict[str, object]:
        resolved: dict[str, object] = {}
        for argument_name, argument in step.arguments.items():
            if not isinstance(argument, StepOutputRef):
                resolved[argument_name] = argument
                continue
            if argument.step_id not in step.depends_on:
                raise _ReferenceResolutionError(
                    "REFERENCE_DEPENDENCY_INVALID",
                    f"Referenced step {argument.step_id!r} is not a declared dependency.",
                )
            producer_result = verified_results.get(argument.step_id)
            if producer_result is None:
                raise _ReferenceResolutionError(
                    "REFERENCE_RESULT_UNAVAILABLE",
                    f"Verified result for step {argument.step_id!r} is unavailable.",
                )
            if argument.output_key not in producer_result:
                raise _ReferenceResolutionError(
                    "REFERENCE_OUTPUT_MISSING",
                    f"Verified step {argument.step_id!r} has no output key "
                    f"{argument.output_key!r}.",
                )
            try:
                resolved_value = _copy_json_value(
                    producer_result[argument.output_key],
                    f"{argument.step_id}.{argument.output_key}",
                )
            except (TypeError, ValueError) as exc:
                raise _ReferenceResolutionError(
                    "REFERENCE_VALUE_INVALID",
                    f"Referenced output is not JSON-safe: {exc}",
                ) from exc
            resolved[argument_name] = resolved_value
            trace.add(
                TraceEventType.STEP_EXECUTION,
                "Resolved a direct upstream result reference.",
                step_id=step.step_id,
                details={
                    "argument_name": argument_name,
                    "producer_step_id": argument.step_id,
                    "output_key": argument.output_key,
                },
            )
        try:
            validated = self._registry.validate_arguments(step.tool_name, resolved)
        except (UnknownToolError, ToolArgumentError) as exc:
            raise _ReferenceResolutionError(
                "RESOLVED_ARGUMENTS_INVALID",
                f"Resolved arguments violate the registry contract: {exc}",
            ) from exc
        return dict(validated)

    def _preflight_failure_outcome(
        self,
        plan: AgentPlan,
        verification: VerificationResult,
        trace: _TraceRecorder,
    ) -> ExecutionOutcome:
        error = verification.error or AgentError(
            ErrorCategory.INTERNAL_AGENT_ERROR,
            "PLAN_PREFLIGHT_FAILED",
            "Plan preflight failed without a structured error.",
        )
        failed_step_id = error.step_id
        now = _utc_now()
        results: list[StepExecutionResult] = []
        for step in plan.steps:
            if step.step_id == failed_step_id:
                step_error = AgentError(
                    category=error.category,
                    code=error.code,
                    message=error.message,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    exception_type=error.exception_type,
                    details=error.details,
                )
                status = StepStatus.FAILED
            else:
                step_error = AgentError(
                    ErrorCategory.INTERNAL_AGENT_ERROR,
                    "SKIPPED_AFTER_PREFLIGHT_FAILURE",
                    "Step was not executed because whole-plan preflight failed.",
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    details={"failed_step_id": failed_step_id},
                )
                status = StepStatus.SKIPPED
                trace.add(
                    TraceEventType.STEP_SKIPPED,
                    "Step skipped because whole-plan preflight failed.",
                    step_id=step.step_id,
                    details={"failed_step_id": failed_step_id},
                )
            results.append(
                StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=status,
                    error=step_error,
                    started_at=now,
                    finished_at=now,
                    duration_seconds=0.0,
                )
            )
        trace.add(
            TraceEventType.RUN_COMPLETION,
            "Plan execution ended during preflight.",
            details={"status": "FAILED"},
        )
        return ExecutionOutcome(tuple(results), (error,), trace.events)

    def execute(self, plan: AgentPlan) -> ExecutionOutcome:
        """Preflight and execute every plan step with fail-safe stop semantics."""

        if not isinstance(plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan.")
        trace = _TraceRecorder()
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "Whole-plan preflight started.",
            details={"plan_id": plan.plan_id},
        )
        preflight = self.preflight(plan)
        if not preflight.passed:
            trace.add(
                TraceEventType.PLAN_VALIDATION,
                "Whole-plan preflight failed; no tools were invoked.",
                details={"error_code": preflight.error.code if preflight.error else None},
            )
            return self._preflight_failure_outcome(plan, preflight, trace)
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "Whole-plan preflight succeeded.",
            details={"plan_id": plan.plan_id},
        )

        ordered_steps = plan.stable_topological_steps()
        verified_results: dict[str, Mapping[str, JsonValue]] = {}
        results_by_id: dict[str, StepExecutionResult] = {}
        errors: list[AgentError] = []
        failed_step_id: str | None = None

        for step in ordered_steps:
            if failed_step_id is not None:
                break
            started_at = _utc_now()
            started_clock = perf_counter()
            trace.add(
                TraceEventType.STEP_EXECUTION,
                "Step is ready for sequential execution.",
                step_id=step.step_id,
                details={"tool_name": step.tool_name},
            )
            try:
                resolved_arguments = self._resolve_arguments(
                    step, verified_results, trace
                )
            except _ReferenceResolutionError as exc:
                finished_at = _utc_now()
                error = AgentError(
                    ErrorCategory.INTERNAL_AGENT_ERROR,
                    exc.code,
                    str(exc),
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    exception_type=type(exc).__name__,
                )
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=StepStatus.FAILED,
                    attempt_count=0,
                    error=error,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=max(0.0, perf_counter() - started_clock),
                )
                errors.append(error)
                failed_step_id = step.step_id
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Step failed during output-reference resolution.",
                    step_id=step.step_id,
                    details={"error_code": error.code},
                )
                break

            spec = self._registry.get(step.tool_name)
            attempt = 0
            returned_result: object | None = None
            tool_error: AgentError | None = None
            while attempt < self._recovery_policy.max_attempts_per_step:
                attempt += 1
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Registered tool attempt started.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={"tool_name": step.tool_name},
                )
                try:
                    returned_result = spec.function(**resolved_arguments)
                except Exception as exc:
                    tool_error = self._registry.classify_exception(
                        step.tool_name,
                        exc,
                        step_id=step.step_id,
                        attempt=attempt,
                    )
                    trace.add(
                        TraceEventType.STEP_EXECUTION,
                        "Registered tool attempt failed.",
                        step_id=step.step_id,
                        attempt=attempt,
                        details={
                            "error_code": tool_error.code,
                            "recoverable": tool_error.recoverable,
                        },
                    )
                    retry = (
                        tool_error.recoverable
                        and attempt < self._recovery_policy.max_attempts_per_step
                    )
                    trace.add(
                        TraceEventType.RECOVERY,
                        "Retry approved with unchanged arguments."
                        if retry
                        else "Retry not permitted; attempt is permanent.",
                        step_id=step.step_id,
                        attempt=attempt,
                        details={
                            "retry": retry,
                            "max_attempts": self._recovery_policy.max_attempts_per_step,
                            "error_code": tool_error.code,
                        },
                    )
                    if retry:
                        continue
                    break
                else:
                    tool_error = None
                    trace.add(
                        TraceEventType.STEP_EXECUTION,
                        "Registered tool returned normally.",
                        step_id=step.step_id,
                        attempt=attempt,
                    )
                    break

            if tool_error is not None:
                finished_at = _utc_now()
                duration = max(0.0, perf_counter() - started_clock)
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=StepStatus.FAILED,
                    attempt_count=attempt,
                    resolved_arguments=cast(
                        Mapping[str, JsonValue], resolved_arguments
                    ),
                    error=tool_error,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration,
                )
                errors.append(tool_error)
                failed_step_id = step.step_id
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Step permanently failed during tool execution.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={"error_code": tool_error.code},
                )
                break

            dependency_results = {
                dependency: verified_results[dependency]
                for dependency in step.depends_on
                if dependency in verified_results
            }
            verification = verify_step(
                step,
                resolved_arguments,
                returned_result,
                self._registry,
                dependency_results=dependency_results,
            )
            trace.add(
                TraceEventType.VERIFICATION,
                "Step result verification succeeded."
                if verification.passed
                else "Step result verification failed.",
                step_id=step.step_id,
                attempt=attempt,
                details={
                    "passed": verification.passed,
                    "error_code": (
                        verification.error.code if verification.error else None
                    ),
                },
            )

            recordable_result: dict[str, JsonValue] | None = None
            try:
                recordable_result = _copy_json_mapping(
                    returned_result, f"step.{step.step_id}.result"
                )
            except (TypeError, ValueError):
                pass

            finished_at = _utc_now()
            duration = max(0.0, perf_counter() - started_clock)

            if not verification.passed:
                error = verification.error or AgentError(
                    ErrorCategory.VERIFICATION_ERROR,
                    "STEP_VERIFICATION_FAILED",
                    "Step result verification failed without a structured error.",
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                )
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=StepStatus.FAILED,
                    attempt_count=attempt,
                    resolved_arguments=cast(
                        Mapping[str, JsonValue], resolved_arguments
                    ),
                    result=recordable_result,
                    verification=verification,
                    error=error,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration,
                )
                errors.append(error)
                failed_step_id = step.step_id
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Step failed because its result was not verified.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={"error_code": error.code},
                )
                break

            if recordable_result is None:
                raise RuntimeError(
                    "Verifier passed a result that cannot be recorded as JSON-safe."
                )
            verified_results[step.step_id] = recordable_result
            results_by_id[step.step_id] = StepExecutionResult(
                step_id=step.step_id,
                tool_name=step.tool_name,
                status=StepStatus.SUCCEEDED,
                attempt_count=attempt,
                resolved_arguments=cast(Mapping[str, JsonValue], resolved_arguments),
                result=recordable_result,
                verification=verification,
                started_at=started_at,
                finished_at=finished_at,
                duration_seconds=duration,
            )
            trace.add(
                TraceEventType.STEP_EXECUTION,
                "Step completed successfully.",
                step_id=step.step_id,
                attempt=attempt,
            )

        if failed_step_id is not None:
            steps_by_id = {step.step_id: step for step in plan.steps}
            for step in plan.steps:
                if step.step_id in results_by_id:
                    continue
                dependency_failure = _depends_on(
                    step.step_id, failed_step_id, steps_by_id
                )
                code = (
                    "DEPENDENCY_FAILED" if dependency_failure else "EXECUTION_ABORTED"
                )
                message = (
                    f"Step was skipped because dependency {failed_step_id!r} failed."
                    if dependency_failure
                    else "Step was skipped after an earlier permanent failure."
                )
                skipped_error = AgentError(
                    ErrorCategory.TOOL_EXECUTION_ERROR,
                    code,
                    message,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    details={"failed_step_id": failed_step_id},
                )
                now = _utc_now()
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=StepStatus.SKIPPED,
                    error=skipped_error,
                    started_at=now,
                    finished_at=now,
                    duration_seconds=0.0,
                )
                trace.add(
                    TraceEventType.STEP_SKIPPED,
                    message,
                    step_id=step.step_id,
                    details={"failed_step_id": failed_step_id},
                )

        trace.add(
            TraceEventType.RUN_COMPLETION,
            "Plan execution completed successfully."
            if not errors
            else "Plan execution completed with failure.",
            details={"status": "SUCCEEDED" if not errors else "FAILED"},
        )
        ordered_results = tuple(results_by_id[step.step_id] for step in plan.steps)
        return ExecutionOutcome(ordered_results, tuple(errors), trace.events)


def _depends_on(
    step_id: str,
    target_dependency: str,
    steps_by_id: Mapping[str, PlanStep],
) -> bool:
    pending = list(steps_by_id[step_id].depends_on)
    visited: set[str] = set()
    while pending:
        dependency = pending.pop()
        if dependency == target_dependency:
            return True
        if dependency in visited:
            continue
        visited.add(dependency)
        producer = steps_by_id.get(dependency)
        if producer is not None:
            pending.extend(producer.depends_on)
    return False


__all__ = ["ExecutionOutcome", "PlanExecutor", "RecoveryPolicy"]
