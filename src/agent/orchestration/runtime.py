"""Public runtime coordinating planning, execution, and run verification."""

from __future__ import annotations

from agent.schemas import (
    AgentError,
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    ErrorCategory,
    RunMode,
    RunStatus,
    TraceEventType,
)

from .executor import PlanExecutor, _TraceRecorder
from .planner import DeterministicPlanner, Planner, PlannerError
from .registry import ToolRegistry, build_default_tool_registry
from .verifier import verify_run


class AgentRuntime:
    """Synchronous Milestone 3 runtime with explicit injected components."""

    def __init__(
        self,
        *,
        planner: Planner | None = None,
        registry: ToolRegistry | None = None,
        executor: PlanExecutor | None = None,
    ) -> None:
        if planner is not None and not callable(getattr(planner, "plan", None)):
            raise TypeError("`planner` must provide a callable plan() method.")
        if registry is not None and not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry or None.")
        if executor is not None and not isinstance(executor, PlanExecutor):
            raise TypeError("`executor` must be a PlanExecutor or None.")

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

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def executor(self) -> PlanExecutor:
        return self._executor

    def run(self, request: AgentRequest) -> AgentRunResult:
        """Plan and optionally execute one request without hidden global state."""

        if not isinstance(request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
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

    def _run_execute(
        self,
        request: AgentRequest,
        plan: AgentPlan,
        run_id: str,
        trace: _TraceRecorder,
    ) -> AgentRunResult:
        try:
            outcome = self._executor.execute(plan)
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


def _internal_error(code: str, message: str, exception: Exception) -> AgentError:
    return AgentError(
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
        code=code,
        message=message,
        exception_type=type(exception).__name__,
    )


__all__ = ["AgentRuntime"]
