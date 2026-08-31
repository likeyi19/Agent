"""Sequential, registry-bound execution of preflighted Agent plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from time import perf_counter
from typing import Callable, Mapping, Sequence, cast

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
from .error_policy import (
    build_recovery_policy_snapshot,
    classified_agent_error,
    decide_same_step_recovery,
)
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
    cancelled: bool = False


@dataclass(frozen=True)
class ExecutionProgress:
    """Immutable executor transition emitted for optional durable checkpointing."""

    phase: str
    preflight: VerificationResult
    step_results: tuple[StepExecutionResult, ...]
    errors: tuple[AgentError, ...]
    trace: tuple[ExecutionTraceEvent, ...]
    cancelled: bool = False


ExecutionCheckpoint = Callable[[ExecutionProgress], None]
CancellationCheck = Callable[[], str | None]


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


def _fingerprint_json_mapping(value: Mapping[str, JsonValue]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        error=classified_agent_error(
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
            code=code,
            step_id=step_id or None,
            tool_name=tool_name,
            details={
                "failed_step_ids": tuple(
                    failure[1] for failure in failures if failure[1]
                ),
                "failure_count": len(failures),
            },
            safe_fallback_message="Plan preflight failed.",
        ),
    )


class _ReferenceResolutionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _CompletedStepValidationError(RuntimeError):
    def __init__(self, code: str, message: str, step_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.step_id = step_id


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
        preserved_steps: Mapping[str, StepExecutionResult] | None = None,
    ) -> ExecutionOutcome:
        preserved_steps = preserved_steps or {}
        error = verification.error or AgentError(
            ErrorCategory.INTERNAL_AGENT_ERROR,
            "PLAN_PREFLIGHT_FAILED",
            "Plan preflight failed without a structured error.",
        )
        failed_step_id = error.step_id
        now = _utc_now()
        results: list[StepExecutionResult] = []
        for step in plan.steps:
            if step.step_id in preserved_steps:
                results.append(preserved_steps[step.step_id])
                continue
            if step.step_id == failed_step_id:
                step_error = AgentError(
                    category=error.category,
                    code=error.code,
                    message=error.message,
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    exception_type=error.exception_type,
                    recoverable=error.recoverable,
                    attempt=error.attempt,
                    details=error.details,
                    recovery_disposition=error.recovery_disposition,
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

    def execute(
        self,
        plan: AgentPlan,
        *,
        completed_steps: Sequence[StepExecutionResult] = (),
        checkpoint: ExecutionCheckpoint | None = None,
        should_cancel: CancellationCheck | None = None,
    ) -> ExecutionOutcome:
        """Preflight and execute a plan, optionally restoring verified steps."""

        if not isinstance(plan, AgentPlan):
            raise TypeError("`plan` must be an AgentPlan.")
        if not isinstance(completed_steps, Sequence) or not all(
            isinstance(result, StepExecutionResult) for result in completed_steps
        ):
            raise TypeError(
                "`completed_steps` must be a sequence of StepExecutionResult values."
            )
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("`checkpoint` must be callable or None.")
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError("`should_cancel` must be callable or None.")

        trace = _TraceRecorder()
        completed_by_id = self._validate_completed_step_set(plan, completed_steps)
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
            outcome = self._preflight_failure_outcome(
                plan,
                preflight,
                trace,
                completed_by_id,
            )
            if checkpoint is not None:
                checkpoint(
                    ExecutionProgress(
                        "PREFLIGHT_FAILED",
                        preflight,
                        outcome.step_results,
                        outcome.errors,
                        outcome.trace,
                    )
                )
            return outcome
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "Whole-plan preflight succeeded.",
            details={"plan_id": plan.plan_id},
        )
        recovery_policy_snapshot = build_recovery_policy_snapshot(
            plan,
            self._registry,
            max_attempts_per_step=self._recovery_policy.max_attempts_per_step,
        )

        ordered_steps = plan.stable_topological_steps()
        verified_results: dict[str, Mapping[str, JsonValue]] = {}
        results_by_id: dict[str, StepExecutionResult] = {}
        errors: list[AgentError] = []
        failed_step_id: str | None = None
        cancelled = False

        def observe_cancellation(
            boundary: str,
            *,
            step_id: str | None = None,
            attempt: int | None = None,
        ) -> bool:
            nonlocal cancelled
            if cancelled or should_cancel is None:
                return cancelled
            requested_at = should_cancel()
            if requested_at is None:
                return False
            if not isinstance(requested_at, str) or not requested_at:
                raise TypeError(
                    "`should_cancel` must return a non-empty timestamp string or None."
                )
            cancelled = True
            error = AgentError(
                ErrorCategory.CANCELLATION,
                "RUN_CANCELLED",
                "Run cancellation was observed at a cooperative checkpoint.",
                step_id=step_id,
                attempt=attempt,
                details={
                    "requested_at": requested_at,
                    "observed_boundary": boundary,
                },
            )
            errors.append(error)
            details = {
                "requested_at": requested_at,
                "observed_boundary": boundary,
            }
            trace.add(
                TraceEventType.CANCELLATION_REQUESTED,
                "Durable cancellation request was incorporated into execution trace.",
                step_id=step_id,
                attempt=attempt,
                details=details,
            )
            trace.add(
                TraceEventType.CANCELLATION_OBSERVED,
                "Cancellation took effect at a cooperative execution checkpoint.",
                step_id=step_id,
                attempt=attempt,
                details=details,
            )
            return True

        def snapshot() -> tuple[StepExecutionResult, ...]:
            return tuple(
                results_by_id.get(
                    step.step_id,
                    StepExecutionResult(
                        step.step_id,
                        step.tool_name,
                        StepStatus.PENDING,
                    ),
                )
                for step in plan.steps
            )

        def notify(phase: str) -> None:
            if checkpoint is not None:
                checkpoint(
                    ExecutionProgress(
                        phase,
                        preflight,
                        snapshot(),
                        tuple(errors),
                        trace.events,
                        cancelled,
                    )
                )

        try:
            for step in ordered_steps:
                persisted = completed_by_id.get(step.step_id)
                if persisted is None:
                    continue
                missing_dependencies = tuple(
                    dependency
                    for dependency in step.depends_on
                    if dependency not in verified_results
                )
                if missing_dependencies:
                    raise _CompletedStepValidationError(
                        "PERSISTED_STEP_DEPENDENCY_MISSING",
                        f"Persisted successful step {step.step_id!r} lacks verified "
                        f"dependencies {missing_dependencies!r}.",
                        step.step_id,
                    )
                try:
                    resolved_arguments = self._resolve_arguments(
                        step, verified_results, trace
                    )
                except _ReferenceResolutionError as exc:
                    raise _CompletedStepValidationError(
                        "PERSISTED_STEP_ARGUMENT_RESOLUTION_FAILED",
                        str(exc),
                        step.step_id,
                    ) from exc
                if dict(resolved_arguments) != dict(persisted.resolved_arguments):
                    raise _CompletedStepValidationError(
                        "PERSISTED_RESOLVED_ARGUMENTS_MISMATCH",
                        f"Persisted resolved arguments for step {step.step_id!r} "
                        "do not match arguments reconstructed from the plan.",
                        step.step_id,
                    )
                if persisted.result is None:
                    raise _CompletedStepValidationError(
                        "PERSISTED_STEP_RESULT_MISSING",
                        f"Persisted successful step {step.step_id!r} has no result.",
                        step.step_id,
                    )
                dependency_results = {
                    dependency: verified_results[dependency]
                    for dependency in step.depends_on
                }
                verification = verify_step(
                    step,
                    resolved_arguments,
                    persisted.result,
                    self._registry,
                    dependency_results=dependency_results,
                )
                trace.add(
                    TraceEventType.VERIFICATION,
                    "Persisted successful step revalidation succeeded."
                    if verification.passed
                    else "Persisted successful step revalidation failed.",
                    step_id=step.step_id,
                    details={
                        "passed": verification.passed,
                        "resume": True,
                        "error_code": (
                            verification.error.code if verification.error else None
                        ),
                    },
                )
                if not verification.passed:
                    message = (
                        verification.error.message
                        if verification.error is not None
                        else "Persisted step result failed revalidation."
                    )
                    raise _CompletedStepValidationError(
                        "PERSISTED_STEP_REVALIDATION_FAILED",
                        message,
                        step.step_id,
                    )
                recordable = _copy_json_mapping(
                    persisted.result, f"step.{step.step_id}.result"
                )
                verified_results[step.step_id] = recordable
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=persisted.step_id,
                    tool_name=persisted.tool_name,
                    status=StepStatus.SUCCEEDED,
                    attempt_count=persisted.attempt_count,
                    resolved_arguments=persisted.resolved_arguments,
                    result=recordable,
                    verification=verification,
                    started_at=persisted.started_at,
                    finished_at=persisted.finished_at,
                    duration_seconds=persisted.duration_seconds,
                )
                trace.add(
                    TraceEventType.RECOVERY,
                    "Previously verified step restored without tool execution.",
                    step_id=step.step_id,
                    details={"resume": True},
                )
        except _CompletedStepValidationError as exc:
            error = AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                exc.code,
                str(exc),
                step_id=exc.step_id,
                tool_name=(
                    next(
                        (
                            step.tool_name
                            for step in plan.steps
                            if step.step_id == exc.step_id
                        ),
                        None,
                    )
                ),
                exception_type=type(exc).__name__,
            )
            errors.append(error)
            failed_step_id = exc.step_id
            if exc.step_id is not None:
                persisted = next(
                    (
                        result
                        for result in completed_steps
                        if result.step_id == exc.step_id
                    ),
                    None,
                )
                if persisted is not None:
                    results_by_id[exc.step_id] = persisted
                else:
                    step = next(
                        step for step in plan.steps if step.step_id == exc.step_id
                    )
                    now = _utc_now()
                    results_by_id[exc.step_id] = StepExecutionResult(
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        status=StepStatus.FAILED,
                        error=error,
                        started_at=now,
                        finished_at=now,
                        duration_seconds=0.0,
                    )
            trace.add(
                TraceEventType.RECOVERY,
                "Persisted successful step could not be restored; no tools invoked.",
                step_id=exc.step_id,
                details={"error_code": exc.code, "resume": True},
            )
        else:
            notify("PREFLIGHT_SUCCEEDED")

        observe_cancellation("after_preflight_and_restoration")

        for step in ordered_steps:
            if failed_step_id is not None or cancelled:
                break
            if step.step_id in results_by_id:
                continue
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
                notify("STEP_FAILED")
                break

            spec = self._registry.get(step.tool_name)
            argument_snapshot = _copy_json_mapping(
                resolved_arguments,
                f"step.{step.step_id}.validated_arguments",
            )
            argument_fingerprint = _fingerprint_json_mapping(argument_snapshot)
            attempt = 0
            returned_result: object | None = None
            tool_error: AgentError | None = None
            while attempt < self._recovery_policy.max_attempts_per_step:
                if observe_cancellation(
                    "before_scientific_attempt",
                    step_id=step.step_id,
                    attempt=attempt + 1,
                ):
                    if tool_error is not None:
                        trace.add(
                            TraceEventType.RECOVERY,
                            "Pending same-step retry was suppressed by cancellation.",
                            step_id=step.step_id,
                            attempt=attempt,
                            details={
                                "classification_code": tool_error.code,
                                "recovery_disposition": (
                                    tool_error.recovery_disposition.value
                                ),
                                "decision": "stop_cancelled",
                                "reason": "cancellation_observed",
                                "attempt": attempt,
                                "max_attempts": (
                                    self._recovery_policy.max_attempts_per_step
                                ),
                                "retry_exhausted": False,
                                "argument_fingerprint": argument_fingerprint,
                                "recovery_policy_fingerprint": (
                                    recovery_policy_snapshot.fingerprint
                                ),
                            },
                        )
                        finished_at = _utc_now()
                        results_by_id[step.step_id] = StepExecutionResult(
                            step_id=step.step_id,
                            tool_name=step.tool_name,
                            status=StepStatus.FAILED,
                            attempt_count=attempt,
                            resolved_arguments=cast(
                                Mapping[str, JsonValue], argument_snapshot
                            ),
                            error=tool_error,
                            started_at=started_at,
                            finished_at=finished_at,
                            duration_seconds=max(
                                0.0, perf_counter() - started_clock
                            ),
                        )
                        errors.insert(len(errors) - 1, tool_error)
                    break
                attempt += 1
                attempt_arguments = _copy_json_mapping(
                    argument_snapshot,
                    f"step.{step.step_id}.attempt_{attempt}_arguments",
                )
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Registered tool attempt started.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={
                        "tool_name": step.tool_name,
                        "argument_fingerprint": argument_fingerprint,
                        "recovery_policy_fingerprint": (
                            recovery_policy_snapshot.fingerprint
                        ),
                    },
                )
                results_by_id[step.step_id] = StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    status=StepStatus.RUNNING,
                    attempt_count=attempt,
                    resolved_arguments=cast(
                        Mapping[str, JsonValue], argument_snapshot
                    ),
                    started_at=started_at,
                )
                notify("STEP_RUNNING")
                try:
                    returned_result = spec.function(**attempt_arguments)
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
                    decision = decide_same_step_recovery(
                        tool_error,
                        attempt=attempt,
                        max_attempts=self._recovery_policy.max_attempts_per_step,
                    )
                    if decision.exhausted:
                        tool_error = replace(
                            tool_error,
                            details={
                                **tool_error.details,
                                "retry_exhausted": True,
                                "attempts": attempt,
                                "max_attempts": (
                                    self._recovery_policy.max_attempts_per_step
                                ),
                                "recovery_policy_fingerprint": (
                                    recovery_policy_snapshot.fingerprint
                                ),
                            },
                        )
                    trace.add(
                        TraceEventType.RECOVERY,
                        "Retry approved with unchanged arguments."
                        if decision.retry
                        else "Same-step recovery stopped.",
                        step_id=step.step_id,
                        attempt=attempt,
                        details={
                            "retry": decision.retry,
                            "decision": decision.decision,
                            "reason": decision.reason,
                            "classification_code": tool_error.code,
                            "recovery_disposition": (
                                tool_error.recovery_disposition.value
                            ),
                            "attempt": attempt,
                            "max_attempts": self._recovery_policy.max_attempts_per_step,
                            "error_code": tool_error.code,
                            "retry_exhausted": decision.exhausted,
                            "argument_fingerprint": argument_fingerprint,
                            "recovery_policy_fingerprint": (
                                recovery_policy_snapshot.fingerprint
                            ),
                        },
                    )
                    if decision.retry:
                        notify("STEP_RUNNING")
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

            if cancelled:
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
                        Mapping[str, JsonValue], argument_snapshot
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
                notify("STEP_FAILED")
                break

            dependency_results = {
                dependency: verified_results[dependency]
                for dependency in step.depends_on
                if dependency in verified_results
            }
            verification = verify_step(
                step,
                argument_snapshot,
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
                        Mapping[str, JsonValue], argument_snapshot
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
                    TraceEventType.RECOVERY,
                    "Verification failure is not eligible for automatic rerun.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={
                        "classification_code": error.code,
                        "recovery_disposition": error.recovery_disposition.value,
                        "decision": "stop_verification_failure",
                        "reason": "verification_failure_after_tool_return",
                        "attempt": attempt,
                        "max_attempts": self._recovery_policy.max_attempts_per_step,
                        "retry_exhausted": False,
                        "argument_fingerprint": argument_fingerprint,
                        "recovery_policy_fingerprint": (
                            recovery_policy_snapshot.fingerprint
                        ),
                    },
                )
                trace.add(
                    TraceEventType.STEP_EXECUTION,
                    "Step failed because its result was not verified.",
                    step_id=step.step_id,
                    attempt=attempt,
                    details={"error_code": error.code},
                )
                notify("STEP_FAILED")
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
                resolved_arguments=cast(Mapping[str, JsonValue], argument_snapshot),
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
            notify("STEP_SUCCEEDED")
            if observe_cancellation(
                "after_verified_success_checkpoint",
                step_id=step.step_id,
                attempt=attempt,
            ):
                break

        if cancelled:
            for step in plan.steps:
                if step.step_id in results_by_id:
                    continue
                skipped_error = AgentError(
                    ErrorCategory.CANCELLATION,
                    "EXECUTION_CANCELLED",
                    "Step was not started because run cancellation took effect.",
                    step_id=step.step_id,
                    tool_name=step.tool_name,
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
                    "Step was skipped after cooperative cancellation was observed.",
                    step_id=step.step_id,
                    details={"reason": "cancellation"},
                )
            notify("CANCELLATION_OBSERVED")
        elif failed_step_id is not None:
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
            notify("STEPS_SKIPPED")

        trace.add(
            TraceEventType.RUN_COMPLETION,
            (
                "Plan execution stopped after cooperative cancellation."
                if cancelled
                else (
                    "Plan execution completed successfully."
                    if not errors
                    else "Plan execution completed with failure."
                )
            ),
            details={
                "status": (
                    "CANCELLED"
                    if cancelled
                    else ("SUCCEEDED" if not errors else "FAILED")
                )
            },
        )
        ordered_results = tuple(results_by_id[step.step_id] for step in plan.steps)
        outcome = ExecutionOutcome(
            ordered_results, tuple(errors), trace.events, cancelled=cancelled
        )
        if checkpoint is not None:
            checkpoint(
                ExecutionProgress(
                    "EXECUTION_COMPLETED",
                    preflight,
                    outcome.step_results,
                    outcome.errors,
                    outcome.trace,
                    outcome.cancelled,
                )
            )
        return outcome

    @staticmethod
    def _validate_completed_step_set(
        plan: AgentPlan,
        completed_steps: Sequence[StepExecutionResult],
    ) -> dict[str, StepExecutionResult]:
        completed_by_id: dict[str, StepExecutionResult] = {}
        plan_by_id = {step.step_id: step for step in plan.steps}
        for result in completed_steps:
            if result.step_id in completed_by_id:
                raise _CompletedStepValidationError(
                    "DUPLICATE_PERSISTED_STEP",
                    f"Persisted step {result.step_id!r} appears more than once.",
                    result.step_id,
                )
            step = plan_by_id.get(result.step_id)
            if step is None or step.tool_name != result.tool_name:
                raise _CompletedStepValidationError(
                    "PERSISTED_STEP_IDENTITY_MISMATCH",
                    f"Persisted step {result.step_id!r} does not match the plan.",
                    result.step_id,
                )
            if (
                result.status is not StepStatus.SUCCEEDED
                or result.result is None
                or result.verification is None
                or not result.verification.passed
                or result.error is not None
            ):
                raise _CompletedStepValidationError(
                    "PERSISTED_STEP_NOT_VERIFIED",
                    f"Persisted step {result.step_id!r} is not a verified success.",
                    result.step_id,
                )
            completed_by_id[result.step_id] = result
        return completed_by_id


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


__all__ = [
    "CancellationCheck",
    "ExecutionCheckpoint",
    "ExecutionOutcome",
    "ExecutionProgress",
    "PlanExecutor",
    "RecoveryPolicy",
]
