"""Public runtime coordinating planning, execution, and run verification."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence

from agent.schemas import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    ErrorCategory,
    ExecutionTraceEvent,
    PersistedRunState,
    RUN_STATE_SCHEMA_VERSION,
    RunLifecycleStatus,
    RunMode,
    RunStatus,
    StepExecutionResult,
    StepStatus,
    TraceEventType,
    fingerprint_plan,
)

from .executor import (
    ExecutionCheckpoint,
    ExecutionProgress,
    PlanExecutor,
    _TraceRecorder,
)
from .planner import DeterministicPlanner, Planner, PlannerError
from .registry import ToolRegistry, build_default_tool_registry
from .run_store import RunStore
from .verifier import verify_run


class AgentRuntime:
    """Synchronous Milestone 3 runtime with explicit injected components."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        registry: ToolRegistry | None = None,
        executor: PlanExecutor | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        if planner is not None and not callable(getattr(planner, "plan", None)):
            raise TypeError("`planner` must provide a callable plan() method.")
        if registry is not None and not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry or None.")
        if executor is not None and not isinstance(executor, PlanExecutor):
            raise TypeError("`executor` must be a PlanExecutor or None.")
        if run_store is not None and not isinstance(run_store, RunStore):
            raise TypeError("`run_store` must implement RunStore or be None.")

        if registry is None:
            registry = (
                executor.registry
                if executor is not None
                else build_default_tool_registry()
            )
        if executor is None:
            executor = PlanExecutor(registry)
        elif executor.registry is not registry:
            raise ValueError("`executor` and `registry` must use the same registry instance.")

        self._planner = planner if planner is not None else DeterministicPlanner()
        self._registry = registry
        self._executor = executor
        self._run_store = run_store

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def executor(self) -> PlanExecutor:
        return self._executor

    @property
    def run_store(self) -> RunStore | None:
        return self._run_store

    def run(self, request: AgentRequest) -> AgentRunResult:
        """Plan and optionally execute one request without hidden global state."""

        if not isinstance(request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
        if self._run_store is not None:
            run_id = f"{request.request_id}:run"
            with self._run_store.execution_lease(run_id):
                return self._run_durable(request)
        run_id = f"{request.request_id}:run"
        trace = _TraceRecorder()
        trace.add(
            TraceEventType.PLANNING,
            "Planner invocation started.",
            details={"planner_name": type(self._planner).__name__},
        )
        try:
            plan = self._planner.plan(request, self._registry)
            if not isinstance(plan, AgentPlan):
                raise TypeError("Planner returned a value that is not an AgentPlan.")
        except PlannerError as exc:
            error = AgentError(
                category=exc.category,
                code=exc.code,
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            trace.add(
                TraceEventType.PLANNING,
                "Planning failed with a classified planner error.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with planning failure.",
                details={"status": "FAILED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=request.mode is RunMode.PLAN_ONLY,
                errors=(error,),
                trace=trace.events,
            )
        except Exception as exc:
            error = _internal_error(
                "PLANNER_UNEXPECTED_ERROR",
                "Planner raised an unexpected orchestration error.",
                exc,
            )
            trace.add(
                TraceEventType.PLANNING,
                "Planning failed unexpectedly.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with planning failure.",
                details={"status": "FAILED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=request.mode is RunMode.PLAN_ONLY,
                errors=(error,),
                trace=trace.events,
            )

        trace.add(
            TraceEventType.PLANNING,
            "Planner produced a structured AgentPlan.",
            details={"plan_id": plan.plan_id, "step_count": len(plan.steps)},
        )

        if request.mode is RunMode.PLAN_ONLY:
            return self._run_plan_only(request, plan, run_id, trace)
        return self._run_execute(request, plan, run_id, trace)

    def _run_plan_only(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
    ) -> AgentRunResult:
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "PLAN_ONLY whole-plan preflight started.",
            details={"plan_id": plan.plan_id},
        )
        try:
            preflight = self._executor.preflight(plan)
        except Exception as exc:
            error = _internal_error(
                "PREFLIGHT_UNEXPECTED_ERROR",
                "Plan preflight raised an unexpected orchestration error.",
                exc,
            )
            trace.add(
                TraceEventType.PLAN_VALIDATION,
                "PLAN_ONLY whole-plan preflight failed unexpectedly.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "PLAN_ONLY run completed with failure.",
                details={"status": "FAILED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=True,
                plan=plan,
                errors=(error,),
                trace=trace.events,
            )

        if preflight.passed:
            trace.add(
                TraceEventType.PLAN_VALIDATION,
                "PLAN_ONLY whole-plan preflight succeeded.",
                details={"plan_id": plan.plan_id},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "PLAN_ONLY run completed with a validated plan.",
                details={"status": "PLANNED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.PLANNED,
                planning_only=True,
                plan=plan,
                verification=preflight,
                trace=trace.events,
            )

        error = preflight.error or AgentError(
            ErrorCategory.INTERNAL_AGENT_ERROR,
            "PLAN_PREFLIGHT_FAILED",
            "Plan preflight failed without a structured error.",
        )
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "PLAN_ONLY whole-plan preflight failed; no tools were invoked.",
            details={"error_code": error.code},
        )
        trace.add(
            TraceEventType.RUN_COMPLETION,
            "PLAN_ONLY run completed with failure.",
            details={"status": "FAILED"},
        )
        return AgentRunResult(
            run_id=run_id,
            request_id=request.request_id,
            status=RunStatus.FAILED,
            planning_only=True,
            plan=plan,
            verification=preflight,
            errors=(error,),
            trace=trace.events,
        )

    def _run_durable(self, request: AgentRequest) -> AgentRunResult:
        store = self._required_run_store()
        run_id = f"{request.request_id}:run"
        trace = _TraceRecorder()
        trace.add(
            TraceEventType.PLANNING,
            "Planner invocation started.",
            details={"planner_name": type(self._planner).__name__},
        )
        now = _utc_now()
        state = store.create(
            PersistedRunState(
                schema_version=RUN_STATE_SCHEMA_VERSION,
                revision=0,
                run_id=run_id,
                request=request,
                lifecycle_status=RunLifecycleStatus.PLANNING,
                created_at=now,
                updated_at=now,
                trace=trace.events,
            )
        )

        try:
            plan = self._planner.plan(request, self._registry)
            if not isinstance(plan, AgentPlan):
                raise TypeError("Planner returned a value that is not an AgentPlan.")
            if plan.request_id != request.request_id:
                raise TypeError("Planner returned a plan for a different request ID.")
        except PlannerError as exc:
            error = AgentError(
                category=exc.category,
                code=exc.code,
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            trace.add(
                TraceEventType.PLANNING,
                "Planning failed with a classified planner error.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with planning failure.",
                details={"status": "FAILED"},
            )
            result = AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=request.mode is RunMode.PLAN_ONLY,
                errors=(error,),
                trace=trace.events,
            )
            return self._persist_terminal(state, result)
        except Exception as exc:
            error = _internal_error(
                "PLANNER_UNEXPECTED_ERROR",
                "Planner raised an unexpected orchestration error.",
                exc,
            )
            trace.add(
                TraceEventType.PLANNING,
                "Planning failed unexpectedly.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with planning failure.",
                details={"status": "FAILED"},
            )
            result = AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=request.mode is RunMode.PLAN_ONLY,
                errors=(error,),
                trace=trace.events,
            )
            return self._persist_terminal(state, result)

        trace.add(
            TraceEventType.PLANNING,
            "Planner produced a structured AgentPlan.",
            details={"plan_id": plan.plan_id, "step_count": len(plan.steps)},
        )
        state = self._update_state(
            state,
            plan=plan,
            plan_fingerprint=fingerprint_plan(plan),
            trace=trace.events,
        )

        if request.mode is RunMode.PLAN_ONLY:
            result = self._run_plan_only(request, plan, run_id, trace)
            return self._persist_terminal(state, result)
        return self._run_durable_execute(request, plan, run_id, trace, state)

    def _run_durable_execute(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
        initial_state: PersistedRunState,
        *,
        completed_steps: Sequence[StepExecutionResult] = (),
    ) -> AgentRunResult:
        state = initial_state
        base_trace = trace.events

        def checkpoint(progress: ExecutionProgress) -> None:
            nonlocal state
            if progress.phase == "PREFLIGHT_SUCCEEDED":
                lifecycle = (
                    RunLifecycleStatus.RUNNING
                    if state.lifecycle_status is RunLifecycleStatus.RUNNING
                    else RunLifecycleStatus.VALIDATED
                )
            elif progress.errors or progress.phase in {
                "PREFLIGHT_FAILED",
                "STEP_FAILED",
                "STEPS_SKIPPED",
            }:
                lifecycle = RunLifecycleStatus.FAILED
            else:
                lifecycle = RunLifecycleStatus.RUNNING
            state = self._update_state(
                state,
                lifecycle_status=lifecycle,
                preflight_verification=progress.preflight,
                steps=progress.step_results,
                run_verification=None,
                errors=progress.errors,
                trace=_merge_trace(base_trace, progress.trace),
            )

        result = self._run_execute(
            request,
            plan,
            run_id,
            trace,
            completed_steps=completed_steps,
            checkpoint=checkpoint,
        )
        return self._persist_terminal(state, result)

    def resume(self, run_id: str) -> AgentRunResult:
        """Resume one nonterminal durable run without invoking its planner."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("`run_id` must be a non-empty string.")
        store = self._required_run_store()
        with store.execution_lease(run_id):
            return self._resume_with_lease(run_id)

    def _resume_with_lease(self, run_id: str) -> AgentRunResult:
        state = self._required_run_store().load(run_id)
        if state.lifecycle_status in {
            RunLifecycleStatus.PLANNED,
            RunLifecycleStatus.SUCCEEDED,
            RunLifecycleStatus.FAILED,
            RunLifecycleStatus.INTERRUPTED,
        }:
            return state.to_run_result()
        if state.plan is None:
            return self._fail_interrupted_planning(state)
        if any(result.status is StepStatus.RUNNING for result in state.steps):
            return self._fail_stale_running_step(state)

        trace = _TraceRecorder()
        trace.extend(state.trace)
        trace.add(
            TraceEventType.RECOVERY,
            "Durable run resume started without planner invocation.",
            details={"run_id": state.run_id, "revision": state.revision},
        )
        state = self._update_state(state, trace=trace.events)
        if state.request.mode is RunMode.PLAN_ONLY:
            result = self._run_plan_only(
                state.request,
                state.plan,
                state.run_id,
                trace,
            )
            return self._persist_terminal(state, result)
        completed_steps = tuple(
            result
            for result in state.steps
            if result.status is StepStatus.SUCCEEDED
        )
        return self._run_durable_execute(
            state.request,
            state.plan,
            state.run_id,
            trace,
            state,
            completed_steps=completed_steps,
        )

    def _fail_interrupted_planning(
        self, state: PersistedRunState
    ) -> AgentRunResult:
        error = AgentError(
            ErrorCategory.INTERNAL_AGENT_ERROR,
            "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE",
            "The previous process exited before a resumable plan was persisted.",
        )
        trace = _TraceRecorder()
        trace.extend(state.trace)
        trace.add(
            TraceEventType.RECOVERY,
            "Durable run cannot resume because no plan was persisted.",
            details={"error_code": error.code},
        )
        trace.add(
            TraceEventType.RUN_COMPLETION,
            "Durable run completed with interruption failure.",
            details={"status": "FAILED"},
        )
        state = self._update_state(
            state,
            lifecycle_status=RunLifecycleStatus.FAILED,
            errors=(error,),
            trace=trace.events,
        )
        return state.to_run_result()

    def _fail_stale_running_step(
        self, state: PersistedRunState
    ) -> AgentRunResult:
        running = next(
            result for result in state.steps if result.status is StepStatus.RUNNING
        )
        error = AgentError(
            ErrorCategory.TOOL_EXECUTION_ERROR,
            "STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION",
            "The previous process exited while a scientific tool was running; "
            "its outcome is unknown and it will not be rerun automatically.",
            step_id=running.step_id,
            tool_name=running.tool_name,
            attempt=running.attempt_count,
        )
        now = _utc_now()
        interrupted_steps: list[StepExecutionResult] = []
        for result in state.steps:
            if result.status is StepStatus.RUNNING:
                interrupted_steps.append(
                    StepExecutionResult(
                        result.step_id,
                        result.tool_name,
                        StepStatus.FAILED,
                        result.attempt_count,
                        result.resolved_arguments,
                        error=error,
                        started_at=result.started_at,
                        finished_at=now,
                        duration_seconds=0.0,
                    )
                )
            elif result.status is StepStatus.PENDING:
                skipped_error = AgentError(
                    ErrorCategory.TOOL_EXECUTION_ERROR,
                    "EXECUTION_ABORTED_AFTER_INTERRUPTION",
                    "Step was not executed because an in-flight predecessor has an "
                    "unknown outcome.",
                    step_id=result.step_id,
                    tool_name=result.tool_name,
                    details={"interrupted_step_id": running.step_id},
                )
                interrupted_steps.append(
                    StepExecutionResult(
                        result.step_id,
                        result.tool_name,
                        StepStatus.SKIPPED,
                        error=skipped_error,
                        started_at=now,
                        finished_at=now,
                        duration_seconds=0.0,
                    )
                )
            else:
                interrupted_steps.append(result)

        trace = _TraceRecorder()
        trace.extend(state.trace)
        trace.add(
            TraceEventType.RECOVERY,
            "Stale RUNNING step detected; automatic rerun was refused.",
            step_id=running.step_id,
            attempt=running.attempt_count,
            details={"error_code": error.code},
        )
        trace.add(
            TraceEventType.RUN_COMPLETION,
            "Durable run marked interrupted.",
            details={"status": "INTERRUPTED"},
        )
        state = self._update_state(
            state,
            lifecycle_status=RunLifecycleStatus.INTERRUPTED,
            steps=tuple(interrupted_steps),
            errors=(error,),
            trace=trace.events,
        )
        return state.to_run_result()

    def _persist_terminal(
        self,
        state: PersistedRunState,
        result: AgentRunResult,
    ) -> AgentRunResult:
        lifecycle = {
            RunStatus.PLANNED: RunLifecycleStatus.PLANNED,
            RunStatus.SUCCEEDED: RunLifecycleStatus.SUCCEEDED,
            RunStatus.FAILED: RunLifecycleStatus.FAILED,
        }[result.status]
        preflight = state.preflight_verification
        run_verification = result.verification
        if result.planning_only:
            preflight = result.verification
            run_verification = None
        terminal = self._update_state(
            state,
            lifecycle_status=lifecycle,
            plan=result.plan,
            plan_fingerprint=(
                fingerprint_plan(result.plan) if result.plan is not None else None
            ),
            preflight_verification=preflight,
            steps=_terminal_steps(state, result),
            run_verification=run_verification,
            errors=result.errors,
            trace=_reconcile_trace(state.trace, result.trace),
        )
        return terminal.to_run_result()

    def _update_state(
        self,
        state: PersistedRunState,
        **changes: object,
    ) -> PersistedRunState:
        next_state = replace(
            state,
            revision=state.revision + 1,
            updated_at=_utc_now(),
            **changes,
        )
        return self._required_run_store().update(
            next_state,
            expected_revision=state.revision,
        )

    def _required_run_store(self) -> RunStore:
        if self._run_store is None:
            raise RuntimeError(
                "Durable run persistence/resume requires an injected RunStore."
            )
        return self._run_store

    def _run_execute(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
        *,
        completed_steps: Sequence[StepExecutionResult] = (),
        checkpoint: ExecutionCheckpoint | None = None,
    ) -> AgentRunResult:
        try:
            outcome = self._executor.execute(
                plan,
                completed_steps=completed_steps,
                checkpoint=checkpoint,
            )
        except Exception as exc:
            error = _internal_error(
                "EXECUTOR_UNEXPECTED_ERROR",
                "PlanExecutor raised an unexpected orchestration error.",
                exc,
            )
            trace.add(
                TraceEventType.STEP_EXECUTION,
                "Executor failed unexpectedly.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with executor failure.",
                details={"status": "FAILED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=False,
                plan=plan,
                errors=(error,),
                trace=trace.events,
            )

        trace.extend(outcome.trace)
        errors = list(outcome.errors)
        try:
            run_verification = verify_run(plan, outcome.step_results)
        except Exception as exc:
            error = _internal_error(
                "RUN_VERIFICATION_UNEXPECTED_ERROR",
                "Run-level verification raised an unexpected orchestration error.",
                exc,
            )
            errors.append(error)
            trace.add(
                TraceEventType.VERIFICATION,
                "Run-level verification failed unexpectedly.",
                details={"error_code": error.code},
            )
            trace.add(
                TraceEventType.RUN_COMPLETION,
                "Agent run completed with verification failure.",
                details={"status": "FAILED"},
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.FAILED,
                planning_only=False,
                plan=plan,
                steps=outcome.step_results,
                errors=tuple(errors),
                trace=trace.events,
            )

        trace.add(
            TraceEventType.VERIFICATION,
            "Run-level verification succeeded."
            if run_verification.passed
            else "Run-level verification failed.",
            details={
                "passed": run_verification.passed,
                "error_code": (
                    run_verification.error.code if run_verification.error else None
                ),
            },
        )
        if not run_verification.passed and run_verification.error is not None:
            errors.append(run_verification.error)
        status = (
            RunStatus.SUCCEEDED
            if not errors and run_verification.passed
            else RunStatus.FAILED
        )
        trace.add(
            TraceEventType.RUN_COMPLETION,
            "Agent run completed.",
            details={"status": status.value},
        )
        return AgentRunResult(
            run_id=run_id,
            request_id=request.request_id,
            status=status,
            planning_only=False,
            plan=plan,
            steps=outcome.step_results,
            verification=run_verification,
            errors=tuple(errors),
            trace=trace.events,
        )

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_trace(
    prefix: Sequence[ExecutionTraceEvent],
    suffix: Sequence[ExecutionTraceEvent],
) -> tuple[ExecutionTraceEvent, ...]:
    merged: list[ExecutionTraceEvent] = []
    for event in (*prefix, *suffix):
        merged.append(
            ExecutionTraceEvent(
                sequence=len(merged),
                event_type=event.event_type,
                timestamp=event.timestamp,
                message=event.message,
                step_id=event.step_id,
                attempt=event.attempt,
                details=event.details,
            )
        )
    return tuple(merged)


def _reconcile_trace(
    durable: Sequence[ExecutionTraceEvent],
    returned: Sequence[ExecutionTraceEvent],
) -> tuple[ExecutionTraceEvent, ...]:
    common = 0
    while (
        common < len(durable)
        and common < len(returned)
        and durable[common] == returned[common]
    ):
        common += 1
    return _merge_trace(durable, returned[common:])


def _terminal_steps(
    state: PersistedRunState,
    result: AgentRunResult,
) -> tuple[StepExecutionResult, ...]:
    if result.steps or not state.steps:
        return result.steps
    now = _utc_now()
    root_error = result.errors[0] if result.errors else AgentError(
        ErrorCategory.INTERNAL_AGENT_ERROR,
        "EXECUTION_STATE_UNAVAILABLE",
        "Execution terminated without complete step state.",
    )
    terminal: list[StepExecutionResult] = []
    for step in state.steps:
        if step.status is StepStatus.RUNNING:
            terminal.append(
                StepExecutionResult(
                    step.step_id,
                    step.tool_name,
                    StepStatus.FAILED,
                    step.attempt_count,
                    step.resolved_arguments,
                    error=AgentError(
                        root_error.category,
                        root_error.code,
                        root_error.message,
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                        exception_type=root_error.exception_type,
                        details={"checkpoint_failure": True},
                    ),
                    started_at=step.started_at,
                    finished_at=now,
                    duration_seconds=0.0,
                )
            )
        elif step.status is StepStatus.PENDING:
            terminal.append(
                StepExecutionResult(
                    step.step_id,
                    step.tool_name,
                    StepStatus.SKIPPED,
                    error=AgentError(
                        ErrorCategory.INTERNAL_AGENT_ERROR,
                        "EXECUTION_ABORTED_AFTER_CHECKPOINT_FAILURE",
                        "Step was not executed after durable checkpoint failure.",
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                    ),
                    started_at=now,
                    finished_at=now,
                    duration_seconds=0.0,
                )
            )
        else:
            terminal.append(step)
    return tuple(terminal)


def _internal_error(code: str, message: str, exception: Exception) -> AgentError:
    return AgentError(
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
        code=code,
        message=message,
        exception_type=type(exception).__name__,
    )


__all__ = ["AgentRuntime"]
