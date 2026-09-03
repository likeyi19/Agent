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
    CancellationReceipt,
    CancellationRequest,
    ErrorCategory,
    ExecutionTraceEvent,
    PersistedRunState,
    RecoveryDisposition,
    RUN_STATE_SCHEMA_VERSION,
    RunLifecycleStatus,
    RunMode,
    RunStatus,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    TraceEventType,
    VerificationResult,
    fingerprint_plan,
)

from .executor import (
    CancellationCheck,
    ExecutionCheckpoint,
    ExecutionProgress,
    PlanExecutor,
    _TraceRecorder,
)
from .error_policy import build_recovery_policy_snapshot, classified_agent_error
from .planning_diagnostics import (
    DiagnosedPlanningAttempt,
    PlanningDiagnostic,
    PlanningDiagnosticContext,
    PlanningDiagnosticStage,
    safe_diagnostic_identifier,
)
from .planner import DeterministicPlanner, Planner, PlannerError
from .registry import ToolRegistry, build_default_tool_registry
from .run_store import (
    CancellationRequestedError,
    RecoveryPolicyIncompatibleError,
    RecoveryPolicyUnknownError,
    RunStore,
    RunStoreError,
)
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

    def cancel(self, run_id: str) -> CancellationReceipt:
        """Request cooperative cancellation of one durable nonterminal run."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("`run_id` must be a non-empty string.")
        store = self._required_run_store()
        receipt = store.request_cancellation(run_id)
        if not isinstance(receipt, CancellationReceipt):
            raise TypeError("RunStore returned an invalid cancellation receipt.")
        return receipt

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
            plan, diagnostic_context = self._invoke_planner(request, trace)
        except PlannerError as exc:
            error = classified_agent_error(
                category=exc.category,
                code=exc.code,
                exception_type=type(exc).__name__,
                details=_planner_error_details(exc),
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
        candidate_preflight = self._diagnose_candidate_preflight(
            plan,
            trace,
            diagnostic_context,
        )

        if request.mode is RunMode.PLAN_ONLY:
            return self._run_plan_only(
                request,
                plan,
                run_id,
                trace,
                preflight=candidate_preflight,
            )
        return self._run_execute(request, plan, run_id, trace)

    def _invoke_planner(
        self,
        request: AgentRequest,
        trace: _TraceRecorder,
    ) -> tuple[AgentPlan, PlanningDiagnosticContext | None]:
        # M9.4 must add a cancellation checkpoint between any future attempts;
        # M9.2 deliberately invokes one initial provider call only.
        diagnosed_plan = getattr(self._planner, "plan_with_diagnostics", None)
        if not callable(diagnosed_plan):
            plan = self._planner.plan(request, self._registry)
            if not isinstance(plan, AgentPlan):
                raise TypeError("Planner returned a value that is not an AgentPlan.")
            return plan, None
        try:
            attempt = diagnosed_plan(request, self._registry)
        except PlannerError as exc:
            _record_planning_diagnostics(trace, exc.diagnostics)
            raise
        if not isinstance(attempt, DiagnosedPlanningAttempt):
            raise TypeError(
                "Diagnostic planner returned an invalid planning-attempt value."
            )
        _record_planning_diagnostics(trace, attempt.diagnostics)
        return attempt.plan, attempt.context

    def _diagnose_candidate_preflight(
        self,
        plan: AgentPlan,
        trace: _TraceRecorder,
        context: PlanningDiagnosticContext | None,
    ) -> VerificationResult | None:
        if context is None:
            return None
        try:
            preflight = self._executor.preflight(plan)
        except Exception:
            diagnostic = context.diagnostic(
                PlanningDiagnosticStage.PREFLIGHT,
                "PREFLIGHT_UNEXPECTED_ERROR",
                "failed",
                candidate_constructed=True,
                candidate_preflight_passed=False,
                reason_code="preflight_exception",
            )
            _record_planning_diagnostics(trace, (diagnostic,))
            return None

        if not preflight.passed:
            diagnostic = _preflight_failure_diagnostic(
                context,
                plan,
                preflight,
                self._registry,
            )
            _record_planning_diagnostics(trace, (diagnostic,))
            if preflight.error is not None:
                safe_details = {
                    "failure_count": preflight.error.details.get(
                        "failure_count", 1
                    ),
                    **dict(diagnostic.to_details()),
                }
                preflight = replace(
                    preflight,
                    error=replace(
                        preflight.error,
                        step_id=None,
                        tool_name=diagnostic.tool_name,
                        details=safe_details,
                    ),
                )
            return preflight

        diagnostics = (
            context.diagnostic(
                PlanningDiagnosticStage.PREFLIGHT,
                "CANDIDATE_PREFLIGHT_PASSED",
                "succeeded",
                candidate_constructed=True,
                candidate_preflight_passed=True,
            ),
            context.diagnostic(
                PlanningDiagnosticStage.ACCEPTED,
                "FINAL_PLAN_ACCEPTED",
                "succeeded",
                candidate_constructed=True,
                candidate_preflight_passed=True,
            ),
        )
        _record_planning_diagnostics(trace, diagnostics)
        return preflight

    def _run_plan_only(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
        *,
        should_cancel: CancellationCheck | None = None,
        preflight: VerificationResult | None = None,
    ) -> AgentRunResult:
        trace.add(
            TraceEventType.PLAN_VALIDATION,
            "PLAN_ONLY whole-plan preflight started.",
            details={"plan_id": plan.plan_id},
        )
        try:
            if preflight is None:
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

        requested_at = None if should_cancel is None else should_cancel()
        if requested_at is not None:
            _record_cancellation_observation(
                trace,
                requested_at,
                "after_plan_only_preflight",
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.CANCELLED,
                planning_only=True,
                plan=plan,
                verification=preflight,
                errors=(_cancellation_error(requested_at, "after_plan_only_preflight"),),
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
            "Durable planning state initialized before planner invocation.",
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

        cancellation = self._load_cancellation(run_id)
        if cancellation is not None:
            return self._persist_cancelled(
                state,
                cancellation,
                boundary="before_planner_invocation",
            )
        trace.add(
            TraceEventType.PLANNING,
            "Planner invocation started.",
            details={"planner_name": type(self._planner).__name__},
        )

        try:
            plan, diagnostic_context = self._invoke_planner(request, trace)
            if plan.request_id != request.request_id:
                raise TypeError("Planner returned a plan for a different request ID.")
        except PlannerError as exc:
            error = classified_agent_error(
                category=exc.category,
                code=exc.code,
                exception_type=type(exc).__name__,
                details=_planner_error_details(exc),
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
        candidate_preflight = self._diagnose_candidate_preflight(
            plan,
            trace,
            diagnostic_context,
        )
        recovery_policy_snapshot = (
            None
            if request.mode is RunMode.PLAN_ONLY
            else build_recovery_policy_snapshot(
                plan,
                self._registry,
                max_attempts_per_step=(
                    self._executor.recovery_policy.max_attempts_per_step
                ),
            )
        )
        state = self._update_state(
            state,
            plan=plan,
            plan_fingerprint=fingerprint_plan(plan),
            recovery_policy_snapshot=recovery_policy_snapshot,
            trace=trace.events,
        )

        cancellation = self._load_cancellation(run_id)
        if cancellation is not None:
            return self._persist_cancelled(
                state,
                cancellation,
                boundary="after_plan_persistence",
            )

        if request.mode is RunMode.PLAN_ONLY:
            result = self._run_plan_only(
                request,
                plan,
                run_id,
                trace,
                should_cancel=self._cancellation_check(run_id),
                preflight=candidate_preflight,
            )
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
        arbitrated_progress: ExecutionProgress | None = None

        def checkpoint(progress: ExecutionProgress) -> None:
            nonlocal arbitrated_progress, state
            if progress.cancelled:
                return
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
            try:
                state = self._update_state(
                    state,
                    lifecycle_status=lifecycle,
                    preflight_verification=progress.preflight,
                    steps=progress.step_results,
                    run_verification=None,
                    errors=progress.errors,
                    trace=_merge_trace(base_trace, progress.trace),
                )
            except CancellationRequestedError:
                arbitrated_progress = progress
                raise

        try:
            result = self._run_execute(
                request,
                plan,
                run_id,
                trace,
                completed_steps=completed_steps,
                checkpoint=checkpoint,
                should_cancel=self._cancellation_check(run_id),
            )
        except CancellationRequestedError as exc:
            if arbitrated_progress is None:
                raise
            result = AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.CANCELLED,
                planning_only=False,
                plan=plan,
                steps=arbitrated_progress.step_results,
                errors=arbitrated_progress.errors,
                trace=_merge_trace(base_trace, arbitrated_progress.trace),
            )
            return self._persist_cancelled(
                state,
                exc.request,
                boundary="checkpoint_terminal_arbitration",
                result=result,
                preflight_verification=arbitrated_progress.preflight,
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
            RunLifecycleStatus.CANCELLED,
        }:
            return state.to_run_result()
        if any(result.status is StepStatus.RUNNING for result in state.steps):
            return self._fail_stale_running_step(state)

        cancellation = self._load_cancellation(run_id)
        if cancellation is not None:
            return self._persist_cancelled(
                state,
                cancellation,
                boundary="resume_before_pending_work",
            )
        if state.plan is None:
            return self._fail_interrupted_planning(state)

        if state.request.mode is RunMode.EXECUTE:
            if (
                state.source_schema_version < RUN_STATE_SCHEMA_VERSION
                or state.recovery_policy_snapshot is None
            ):
                raise RecoveryPolicyUnknownError(
                    "Legacy nonterminal EXECUTE run has no authoritative recovery "
                    "policy provenance; use the historical runtime or reconcile it "
                    "manually."
                )
            current_policy = build_recovery_policy_snapshot(
                state.plan,
                self._registry,
                max_attempts_per_step=(
                    self._executor.recovery_policy.max_attempts_per_step
                ),
            )
            if current_policy.fingerprint != state.recovery_policy_snapshot.fingerprint:
                raise RecoveryPolicyIncompatibleError(
                    "Durable run recovery policy is incompatible with this runtime; "
                    "resume with the original compatible policy."
                )

        trace = _TraceRecorder()
        trace.extend(state.trace)
        trace.add(
            TraceEventType.RECOVERY,
            "Durable run resume started without planner invocation.",
            details={
                "run_id": state.run_id,
                "revision": state.revision,
                "decision": "resume_with_compatible_runtime",
                "recovery_policy_fingerprint": (
                    None
                    if state.recovery_policy_snapshot is None
                    else state.recovery_policy_snapshot.fingerprint
                ),
            },
        )
        state = self._update_state(state, trace=trace.events)
        cancellation = self._load_cancellation(run_id)
        if cancellation is not None:
            return self._persist_cancelled(
                state,
                cancellation,
                boundary="resume_after_state_restoration",
            )
        if state.request.mode is RunMode.PLAN_ONLY:
            result = self._run_plan_only(
                state.request,
                state.plan,
                state.run_id,
                trace,
                should_cancel=self._cancellation_check(run_id),
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
            recovery_disposition=RecoveryDisposition.MANUAL_RECONCILIATION,
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
        cancellation = self._load_cancellation(state.run_id)
        if cancellation is not None:
            trace.add(
                TraceEventType.CANCELLATION_REQUESTED,
                "Cancellation intent remained durable while stale work was recovered.",
                details={
                    "requested_at": cancellation.requested_at,
                    "resolution": "INTERRUPTED",
                },
            )
        trace.add(
            TraceEventType.RECOVERY,
            "Stale RUNNING step detected; automatic rerun was refused.",
            step_id=running.step_id,
            attempt=running.attempt_count,
            details={
                "error_code": error.code,
                "classification_code": error.code,
                "recovery_disposition": error.recovery_disposition.value,
                "decision": "stop_unknown_outcome",
                "reason": "manual_reconciliation_required",
                "attempt": running.attempt_count,
                "max_attempts": (
                    None
                    if state.recovery_policy_snapshot is None
                    else state.recovery_policy_snapshot.max_attempts_per_step
                ),
                "retry_exhausted": False,
                "recovery_policy_fingerprint": (
                    None
                    if state.recovery_policy_snapshot is None
                    else state.recovery_policy_snapshot.fingerprint
                ),
            },
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
        if result.status is RunStatus.CANCELLED:
            cancellation = self._load_cancellation(state.run_id)
            if cancellation is None:
                raise RuntimeError(
                    "A cancelled result requires durable cancellation intent."
                )
            return self._persist_cancelled(
                state,
                cancellation,
                boundary="cooperative_checkpoint",
                result=result,
            )
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
        try:
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
        except CancellationRequestedError as exc:
            return self._persist_cancelled(
                state,
                exc.request,
                boundary="normal_terminal_arbitration",
                result=result,
            )
        return terminal.to_run_result()

    def _persist_cancelled(
        self,
        state: PersistedRunState,
        cancellation: CancellationRequest,
        *,
        boundary: str,
        result: AgentRunResult | None = None,
        preflight_verification: VerificationResult | None = None,
    ) -> AgentRunResult:
        trace = _TraceRecorder()
        returned_trace = () if result is None else result.trace
        trace.extend(_reconcile_trace(state.trace, returned_trace))
        if not any(
            event.event_type is TraceEventType.CANCELLATION_OBSERVED
            for event in trace.events
        ):
            _record_cancellation_observation(
                trace,
                cancellation.requested_at,
                boundary,
            )
        trace.add(
            TraceEventType.RUN_CANCELLED,
            "Durable run finalized as cooperatively cancelled.",
            details={
                "requested_at": cancellation.requested_at,
                "observed_boundary": boundary,
                "status": "CANCELLED",
            },
        )

        plan = state.plan if result is None or result.plan is None else result.plan
        preflight = state.preflight_verification
        run_verification = state.run_verification
        candidate_steps = state.steps
        candidate_errors = state.errors
        if result is not None:
            candidate_steps = _terminal_steps(state, result)
            candidate_errors = _merge_errors(state.errors, result.errors)
            if result.planning_only:
                preflight = result.verification
                run_verification = None
            else:
                run_verification = result.verification
        if preflight_verification is not None:
            preflight = preflight_verification
        cancellation_error = _cancellation_error(
            cancellation.requested_at,
            boundary,
        )
        errors = _merge_errors(candidate_errors, (cancellation_error,))
        terminal = self._update_state(
            state,
            lifecycle_status=RunLifecycleStatus.CANCELLED,
            plan=plan,
            plan_fingerprint=(fingerprint_plan(plan) if plan is not None else None),
            preflight_verification=preflight,
            steps=_cancel_remaining_steps(candidate_steps),
            run_verification=run_verification,
            errors=errors,
            trace=trace.events,
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

    def _load_cancellation(self, run_id: str) -> CancellationRequest | None:
        cancellation = self._required_run_store().load_cancellation(run_id)
        if cancellation is not None and not isinstance(
            cancellation, CancellationRequest
        ):
            raise TypeError("RunStore returned invalid cancellation state.")
        return cancellation

    def _cancellation_check(self, run_id: str) -> CancellationCheck:
        def should_cancel() -> str | None:
            cancellation = self._load_cancellation(run_id)
            return None if cancellation is None else cancellation.requested_at

        return should_cancel

    def _run_execute(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
        *,
        completed_steps: Sequence[StepExecutionResult] = (),
        checkpoint: ExecutionCheckpoint | None = None,
        should_cancel: CancellationCheck | None = None,
    ) -> AgentRunResult:
        try:
            outcome = self._executor.execute(
                plan,
                completed_steps=completed_steps,
                checkpoint=checkpoint,
                should_cancel=should_cancel,
            )
        except RunStoreError:
            raise
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
        if outcome.cancelled:
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.CANCELLED,
                planning_only=False,
                plan=plan,
                steps=outcome.step_results,
                errors=tuple(errors),
                trace=trace.events,
            )
        requested_at = None if should_cancel is None else should_cancel()
        if requested_at is not None:
            _record_cancellation_observation(
                trace, requested_at, "before_run_verification"
            )
            errors.append(
                _cancellation_error(requested_at, "before_run_verification")
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.CANCELLED,
                planning_only=False,
                plan=plan,
                steps=outcome.step_results,
                errors=tuple(errors),
                trace=trace.events,
            )
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
        requested_at = None if should_cancel is None else should_cancel()
        if requested_at is not None:
            _record_cancellation_observation(
                trace, requested_at, "after_run_verification"
            )
            errors.append(
                _cancellation_error(requested_at, "after_run_verification")
            )
            return AgentRunResult(
                run_id=run_id,
                request_id=request.request_id,
                status=RunStatus.CANCELLED,
                planning_only=False,
                plan=plan,
                steps=outcome.step_results,
                verification=run_verification,
                errors=tuple(errors),
                trace=trace.events,
            )
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


def _cancellation_error(requested_at: str, boundary: str) -> AgentError:
    return AgentError(
        ErrorCategory.CANCELLATION,
        "RUN_CANCELLED",
        "Run cancellation was observed at a cooperative checkpoint.",
        details={
            "requested_at": requested_at,
            "observed_boundary": boundary,
        },
    )


def _record_cancellation_observation(
    trace: _TraceRecorder,
    requested_at: str,
    boundary: str,
) -> None:
    details = {
        "requested_at": requested_at,
        "observed_boundary": boundary,
    }
    trace.add(
        TraceEventType.CANCELLATION_REQUESTED,
        "Durable cancellation request was incorporated into the run trace.",
        details=details,
    )
    trace.add(
        TraceEventType.CANCELLATION_OBSERVED,
        "Cancellation took effect at a cooperative runtime checkpoint.",
        details=details,
    )


def _merge_errors(
    *groups: Sequence[AgentError],
) -> tuple[AgentError, ...]:
    merged: list[AgentError] = []
    for group in groups:
        for error in group:
            if error not in merged:
                merged.append(error)
    return tuple(merged)


def _cancel_remaining_steps(
    steps: Sequence[StepExecutionResult],
) -> tuple[StepExecutionResult, ...]:
    terminal: list[StepExecutionResult] = []
    for step in steps:
        if step.status is StepStatus.PENDING:
            now = _utc_now()
            terminal.append(
                StepExecutionResult(
                    step.step_id,
                    step.tool_name,
                    StepStatus.SKIPPED,
                    error=AgentError(
                        ErrorCategory.CANCELLATION,
                        "EXECUTION_CANCELLED",
                        "Step was not started because run cancellation took effect.",
                        step_id=step.step_id,
                        tool_name=step.tool_name,
                    ),
                    started_at=now,
                    finished_at=now,
                    duration_seconds=0.0,
                )
            )
        elif step.status is StepStatus.RUNNING:
            now = _utc_now()
            error = AgentError(
                ErrorCategory.INTERNAL_AGENT_ERROR,
                "EXECUTION_STATE_UNAVAILABLE_DURING_CANCELLATION",
                "Execution returned without a terminal result for a started step.",
                step_id=step.step_id,
                tool_name=step.tool_name,
                attempt=step.attempt_count,
            )
            terminal.append(
                StepExecutionResult(
                    step.step_id,
                    step.tool_name,
                    StepStatus.FAILED,
                    step.attempt_count,
                    step.resolved_arguments,
                    error=error,
                    started_at=step.started_at,
                    finished_at=now,
                    duration_seconds=0.0,
                )
            )
        else:
            terminal.append(step)
    return tuple(terminal)


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
                        recoverable=root_error.recoverable,
                        details={**root_error.details, "checkpoint_failure": True},
                        recovery_disposition=root_error.recovery_disposition,
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


def _record_planning_diagnostics(
    trace: _TraceRecorder,
    diagnostics: Sequence[PlanningDiagnostic],
) -> None:
    for diagnostic in diagnostics:
        trace.add(
            TraceEventType.PLANNING,
            "Sanitized planning diagnostic recorded.",
            details=diagnostic.to_details(),
        )


def _planner_error_details(exc: PlannerError) -> dict[str, object]:
    if not exc.diagnostics:
        return {}
    return dict(exc.diagnostics[-1].to_details())


def _preflight_failure_diagnostic(
    context: PlanningDiagnosticContext,
    plan: AgentPlan,
    preflight: VerificationResult,
    registry: ToolRegistry,
) -> PlanningDiagnostic:
    error = preflight.error
    code = "PLAN_PREFLIGHT_FAILED" if error is None else error.code
    stage = {
        "UNKNOWN_TOOL": PlanningDiagnosticStage.TOOL_SELECTION,
        "INVALID_TOOL_ARGUMENTS": PlanningDiagnosticStage.ARGUMENT_BINDING,
        "INVALID_OUTPUT_REFERENCE": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
        "INVALID_PLAN_STRUCTURE": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    }.get(code, PlanningDiagnosticStage.PREFLIGHT)
    step_index = None
    tool_name = None
    argument_name = None
    producer_step_index = None
    output_key = None
    reason_code = "candidate_preflight_failed"
    step = None
    if error is not None and error.step_id is not None:
        step_index = next(
            (
                index
                for index, candidate in enumerate(plan.steps)
                if candidate.step_id == error.step_id
            ),
            None,
        )
        if step_index is not None:
            step = plan.steps[step_index]
            tool_name = (
                safe_diagnostic_identifier(step.tool_name)
                if registry.contains(step.tool_name)
                else None
            )

    if code == "UNKNOWN_TOOL":
        reason_code = "unknown_tool"
    elif code == "INVALID_TOOL_ARGUMENTS" and step is not None:
        reason_code, argument_name = _invalid_argument_diagnostic(step, registry)
    elif code == "INVALID_OUTPUT_REFERENCE" and step is not None:
        reason_code = "invalid_result_field_reference"
        (
            argument_name,
            producer_step_index,
            output_key,
        ) = _invalid_reference_diagnostic(step, plan, registry)
    elif code == "INVALID_PLAN_STRUCTURE":
        reason_code = "dependency_structure_invalid"

    return context.diagnostic(
        stage,
        code,
        "failed",
        candidate_constructed=True,
        candidate_preflight_passed=False,
        step_index=step_index,
        argument_name=argument_name,
        producer_step_index=producer_step_index,
        output_key=output_key,
        tool_name=tool_name,
        reason_code=reason_code,
    )


def _invalid_argument_diagnostic(
    step: object,
    registry: ToolRegistry,
) -> tuple[str, str | None]:
    tool_name = getattr(step, "tool_name", None)
    if not isinstance(tool_name, str) or not registry.contains(tool_name):
        return "unknown_tool", None
    spec = registry.get(tool_name)
    arguments = getattr(step, "arguments", {})
    supplied = set(arguments)
    required = set(spec.required_arguments)
    known = required.union(spec.optional_arguments)
    missing = sorted(required.difference(supplied))
    if missing:
        return "missing_tool_argument", safe_diagnostic_identifier(missing[0])
    unknown = sorted(supplied.difference(known))
    if unknown:
        globally_known = {
            name
            for offered_tool in registry.names()
            for name in (
                *registry.get(offered_tool).required_arguments,
                *registry.get(offered_tool).optional_arguments,
            )
        }
        return (
            "unknown_tool_argument",
            unknown[0] if unknown[0] in globally_known else None,
        )
    for name, value in arguments.items():
        argument_spec = spec.required_arguments.get(
            name, spec.optional_arguments.get(name)
        )
        if argument_spec is None:
            continue
        try:
            argument_spec.validate(name, value)
        except Exception:
            return "invalid_tool_argument", safe_diagnostic_identifier(name)
    return "invalid_tool_arguments", None


def _invalid_reference_diagnostic(
    step: object,
    plan: AgentPlan,
    registry: ToolRegistry,
) -> tuple[str | None, int | None, str | None]:
    steps_by_id = {candidate.step_id: candidate for candidate in plan.steps}
    indices_by_id = {
        candidate.step_id: index for index, candidate in enumerate(plan.steps)
    }
    arguments = getattr(step, "arguments", {})
    dependencies = getattr(step, "depends_on", ())
    step_id = getattr(step, "step_id", None)
    for argument_name, argument in arguments.items():
        if not isinstance(argument, StepOutputRef):
            continue
        producer = steps_by_id.get(argument.step_id)
        valid = (
            producer is not None
            and argument.step_id in dependencies
            and argument.step_id != step_id
        )
        if valid and producer is not None and registry.contains(producer.tool_name):
            valid = (
                argument.output_key
                in registry.get(producer.tool_name).result_contract.required_fields
            )
        else:
            valid = False
        if not valid:
            known_output_keys = {
                field
                for offered_tool in registry.names()
                for field in registry.get(
                    offered_tool
                ).result_contract.required_fields
            }
            return (
                safe_diagnostic_identifier(argument_name),
                indices_by_id.get(argument.step_id),
                argument.output_key
                if argument.output_key in known_output_keys
                else None,
            )
    return None, None, None


def _internal_error(code: str, message: str, exception: Exception) -> AgentError:
    return AgentError(
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
        code=code,
        message=message,
        exception_type=type(exception).__name__,
    )


__all__ = ["AgentRuntime"]
