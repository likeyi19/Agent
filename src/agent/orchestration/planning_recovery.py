"""Bounded provider-neutral recovery for acquiring one accepted AgentPlan."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Callable

from agent.schemas import ErrorCategory, JsonValue, VerificationResult

from .planner import PlannerError
from .planning_diagnostics import (
    DiagnosedPlanningAttempt,
    PlanningAttemptKind,
    PlanningDiagnostic,
    PlanningDiagnosticContext,
    PlanningDiagnosticStage,
)


PLANNING_RECOVERY_POLICY_VERSION = "planning-recovery-v3"
_TRANSIENT_PROVIDER_CODES = frozenset(
    {
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
        "PROVIDER_COMPLETION_INCOMPLETE",
    }
)
_MAX_POLICY_DELAY_SECONDS = 60.0
_DEFAULT_RETRY_DELAY_SECONDS = 1.0
_CANCELLATION_POLL_SECONDS = 0.1
_REPAIRABLE_STAGES = frozenset(
    {
        PlanningDiagnosticStage.PARSE,
        PlanningDiagnosticStage.SCHEMA,
        PlanningDiagnosticStage.TOOL_SELECTION,
        PlanningDiagnosticStage.ARGUMENT_BINDING,
        PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
        PlanningDiagnosticStage.CANDIDATE,
        PlanningDiagnosticStage.PREFLIGHT,
    }
)


@dataclass(frozen=True)
class PlanningRecoveryPolicy:
    """Immutable planning-only recovery bounds, separate from tool recovery."""

    policy_version: str = PLANNING_RECOVERY_POLICY_VERSION
    retryable_provider_codes: frozenset[str] = _TRANSIENT_PROVIDER_CODES
    max_transport_retries: int = 1
    max_repairs: int = 1
    max_profile_failovers: int = 1
    max_primary_local_recovery_actions: int = 1
    max_total_provider_calls: int = 3
    max_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.policy_version != PLANNING_RECOVERY_POLICY_VERSION:
            raise ValueError("Unsupported planning-recovery policy version.")
        if not isinstance(self.retryable_provider_codes, frozenset) or not all(
            isinstance(code, str) and code in _TRANSIENT_PROVIDER_CODES
            for code in self.retryable_provider_codes
        ):
            raise ValueError(
                "`retryable_provider_codes` must contain only approved transient "
                "provider codes."
            )
        bounded_fields = {
            "max_transport_retries": (self.max_transport_retries, 1),
            "max_repairs": (self.max_repairs, 1),
            "max_profile_failovers": (self.max_profile_failovers, 1),
            "max_primary_local_recovery_actions": (
                self.max_primary_local_recovery_actions,
                1,
            ),
            "max_total_provider_calls": (self.max_total_provider_calls, 3),
        }
        for name, (value, maximum) in bounded_fields.items():
            minimum = 1 if name == "max_total_provider_calls" else 0
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"`{name}` must be an integer between {minimum} and {maximum}."
                )
        if self.max_transport_retries > self.max_primary_local_recovery_actions:
            raise ValueError(
                "Transport retry cannot exceed primary-local recovery actions."
            )
        if self.max_repairs > self.max_primary_local_recovery_actions:
            raise ValueError("Plan repair cannot exceed primary-local recovery actions.")
        delay = self.max_retry_delay_seconds
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(float(delay))
            or not 0 <= float(delay) <= _MAX_POLICY_DELAY_SECONDS
        ):
            raise ValueError(
                "`max_retry_delay_seconds` must be finite and between 0 and 60."
            )
        object.__setattr__(self, "max_retry_delay_seconds", float(delay))

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "retryable_provider_codes": tuple(
                sorted(self.retryable_provider_codes)
            ),
            "max_transport_retries": self.max_transport_retries,
            "max_repairs": self.max_repairs,
            "max_profile_failovers": self.max_profile_failovers,
            "max_primary_local_recovery_actions": (
                self.max_primary_local_recovery_actions
            ),
            "max_total_provider_calls": self.max_total_provider_calls,
            "max_retry_delay_seconds": self.max_retry_delay_seconds,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PlanningRepairContext:
    """Safe structural evidence supplied to one complete regeneration call."""

    previous_failure_stage: PlanningDiagnosticStage
    previous_failure_code: str
    reason_code: str | None = None
    step_index: int | None = None
    argument_name: str | None = None
    producer_step_index: int | None = None
    output_key: str | None = None
    tool_name: str | None = None
    candidate_constructed: bool = False
    candidate_preflight_passed: bool | None = None

    @classmethod
    def from_diagnostic(
        cls, diagnostic: PlanningDiagnostic
    ) -> PlanningRepairContext:
        return cls(
            previous_failure_stage=diagnostic.stage,
            previous_failure_code=diagnostic.code,
            reason_code=diagnostic.reason_code,
            step_index=diagnostic.step_index,
            argument_name=diagnostic.argument_name,
            producer_step_index=diagnostic.producer_step_index,
            output_key=diagnostic.output_key,
            tool_name=diagnostic.tool_name,
            candidate_constructed=diagnostic.candidate_constructed,
            candidate_preflight_passed=diagnostic.candidate_preflight_passed,
        )

    def to_prompt_dict(self) -> dict[str, JsonValue]:
        values: dict[str, JsonValue] = {
            "previous_failure_stage": self.previous_failure_stage.value,
            "previous_failure_code": self.previous_failure_code,
            "candidate_constructed": self.candidate_constructed,
            "candidate_preflight_passed": self.candidate_preflight_passed,
        }
        optional: dict[str, JsonValue] = {
            "reason_code": self.reason_code,
            "step_index": self.step_index,
            "argument_name": self.argument_name,
            "producer_step_index": self.producer_step_index,
            "output_key": self.output_key,
            "tool_name": self.tool_name,
        }
        values.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return values


@dataclass(frozen=True)
class RecoveredPlanningAttempt:
    """A final preflight-passing plan plus bounded-recovery provenance."""

    attempt: DiagnosedPlanningAttempt
    preflight: VerificationResult
    diagnostics: tuple[PlanningDiagnostic, ...]
    provider_call_count: int
    retry_used: bool
    repair_used: bool
    failover_used: bool

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, DiagnosedPlanningAttempt):
            raise TypeError("`attempt` must be a DiagnosedPlanningAttempt.")
        if (
            not isinstance(self.preflight, VerificationResult)
            or not self.preflight.passed
        ):
            raise ValueError("Recovered planning requires passing preflight.")
        if not isinstance(self.diagnostics, tuple) or not all(
            isinstance(item, PlanningDiagnostic) for item in self.diagnostics
        ):
            raise TypeError("`diagnostics` must contain PlanningDiagnostic values.")
        if (
            isinstance(self.provider_call_count, bool)
            or not isinstance(self.provider_call_count, int)
            or self.provider_call_count < 1
        ):
            raise ValueError("`provider_call_count` must be positive.")
        if not isinstance(self.retry_used, bool):
            raise TypeError("`retry_used` must be boolean.")
        if not isinstance(self.repair_used, bool):
            raise TypeError("`repair_used` must be boolean.")
        if not isinstance(self.failover_used, bool):
            raise TypeError("`failover_used` must be boolean.")


class PlanningRecoveryCancelled(RuntimeError):
    """Cooperative cancellation observed between planning attempts."""

    def __init__(
        self,
        requested_at: str,
        diagnostics: tuple[PlanningDiagnostic, ...],
        provider_call_count: int,
    ) -> None:
        super().__init__("Planning recovery was cancelled.")
        self.requested_at = requested_at
        self.diagnostics = diagnostics
        self.provider_call_count = provider_call_count


PlanningAttempt = Callable[
    [PlanningAttemptKind, int, int, PlanningRepairContext | None],
    DiagnosedPlanningAttempt,
]
CandidateValidator = Callable[
    [DiagnosedPlanningAttempt],
    tuple[VerificationResult, tuple[PlanningDiagnostic, ...]],
]
PlanningDiagnosticSink = Callable[[tuple[PlanningDiagnostic, ...]], None]
PlanningCancellationCheck = Callable[[], str | None]
PlanningSleeper = Callable[[float], None]
PlanningFailoverPreparation = Callable[
    [int, int, PlanningRepairContext],
    None,
]


class PlanningRecoveryCoordinator:
    """Apply bounded primary recovery and one configured profile failover."""

    def __init__(
        self,
        policy: PlanningRecoveryPolicy | None = None,
        *,
        sleeper: PlanningSleeper = time.sleep,
    ) -> None:
        if policy is not None and not isinstance(policy, PlanningRecoveryPolicy):
            raise TypeError("`policy` must be PlanningRecoveryPolicy or None.")
        if not callable(sleeper):
            raise TypeError("`sleeper` must be callable.")
        self._policy = policy if policy is not None else PlanningRecoveryPolicy()
        self._sleeper = sleeper

    @property
    def policy(self) -> PlanningRecoveryPolicy:
        return self._policy

    def acquire(
        self,
        *,
        attempt: PlanningAttempt,
        validate_candidate: CandidateValidator,
        diagnostic_sink: PlanningDiagnosticSink,
        should_cancel: PlanningCancellationCheck | None = None,
        prepare_failover: PlanningFailoverPreparation | None = None,
    ) -> RecoveredPlanningAttempt:
        """Acquire one Plan with at most one mutually exclusive local action."""

        if not callable(attempt) or not callable(validate_candidate):
            raise TypeError("Planning attempt and candidate validator must be callable.")
        if not callable(diagnostic_sink):
            raise TypeError("`diagnostic_sink` must be callable.")
        if should_cancel is not None and not callable(should_cancel):
            raise TypeError("`should_cancel` must be callable or None.")
        if prepare_failover is not None and not callable(prepare_failover):
            raise TypeError("`prepare_failover` must be callable or None.")

        diagnostics: list[PlanningDiagnostic] = []
        provider_calls = 0
        transport_retries = 0
        repairs = 0
        failovers = 0
        primary_local_actions = 0
        last_context: PlanningDiagnosticContext | None = None
        repair_context: PlanningRepairContext | None = None

        def emit(values: tuple[PlanningDiagnostic, ...]) -> None:
            diagnostics.extend(values)
            diagnostic_sink(values)

        def check_cancel() -> None:
            requested_at = None if should_cancel is None else should_cancel()
            if requested_at is not None:
                if last_context is not None:
                    emit(
                        (
                            _summary_diagnostic(
                                last_context,
                                provider_calls=provider_calls,
                                retry_used=transport_retries > 0,
                                repair_used=repairs > 0,
                                failover_used=failovers > 0,
                                outcome="cancelled",
                                policy_fingerprint=self._policy.fingerprint,
                            ),
                        )
                    )
                raise PlanningRecoveryCancelled(
                    requested_at,
                    tuple(diagnostics),
                    provider_calls,
                )

        def can_repair(failure: PlanningDiagnostic | None) -> bool:
            return bool(
                kind is PlanningAttemptKind.INITIAL
                and failure is not None
                and failure.stage in _REPAIRABLE_STAGES
                and repairs < self._policy.max_repairs
                and primary_local_actions
                < self._policy.max_primary_local_recovery_actions
                and provider_calls < self._policy.max_total_provider_calls
            )

        def schedule_repair(
            context: PlanningDiagnosticContext,
            failure: PlanningDiagnostic,
        ) -> None:
            nonlocal repairs, primary_local_actions, repair_context, kind
            repair_context = PlanningRepairContext.from_diagnostic(failure)
            repairs += 1
            primary_local_actions += 1
            emit(
                (
                    context.diagnostic(
                        PlanningDiagnosticStage.RECOVERY,
                        "PLAN_REPAIR_SCHEDULED",
                        "succeeded",
                        previous_failure_stage=failure.stage,
                        previous_failure_code=failure.code,
                        recovery_action="repair",
                        retry_used=False,
                        repair_used=True,
                        recovery_policy_fingerprint=self._policy.fingerprint,
                    ),
                )
            )
            check_cancel()
            kind = PlanningAttemptKind.REPAIR

        def can_failover(failure: PlanningDiagnostic | None) -> bool:
            eligible_failure = bool(
                failure is not None
                and (
                    (
                        failure.stage is PlanningDiagnosticStage.PROVIDER
                        and failure.code in self._policy.retryable_provider_codes
                    )
                    or failure.stage in _REPAIRABLE_STAGES
                )
            )
            return bool(
                prepare_failover is not None
                and kind
                in {
                    PlanningAttemptKind.TRANSPORT_RETRY,
                    PlanningAttemptKind.REPAIR,
                }
                and eligible_failure
                and primary_local_actions == 1
                and failovers < self._policy.max_profile_failovers
                and provider_calls < self._policy.max_total_provider_calls
            )

        def schedule_failover(
            context: PlanningDiagnosticContext,
            failure: PlanningDiagnostic,
        ) -> None:
            nonlocal repair_context, kind
            repair_context = PlanningRepairContext.from_diagnostic(failure)
            emit(
                (
                    context.diagnostic(
                        PlanningDiagnosticStage.RECOVERY,
                        "PROFILE_FAILOVER_SCHEDULED",
                        "succeeded",
                        previous_failure_stage=failure.stage,
                        previous_failure_code=failure.code,
                        recovery_action="failover",
                        retry_used=transport_retries > 0,
                        repair_used=repairs > 0,
                        failover_used=False,
                        recovery_policy_fingerprint=self._policy.fingerprint,
                    ),
                )
            )
            check_cancel()
            assert prepare_failover is not None
            try:
                prepare_failover(
                    provider_calls + 1,
                    provider_calls + 1,
                    repair_context,
                )
            except PlannerError as exc:
                emit(exc.diagnostics)
                failed_context = _diagnostic_context(exc.diagnostics) or context
                emit(
                    (
                        failed_context.diagnostic(
                            PlanningDiagnosticStage.RECOVERY,
                            "PROFILE_FAILOVER_SUPPRESSED",
                            "rejected",
                            previous_failure_stage=failure.stage,
                            previous_failure_code=exc.code,
                            recovery_action="none",
                            recovery_suppression_reason=(
                                "failover_model_construction_failed"
                            ),
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=False,
                            recovery_policy_fingerprint=self._policy.fingerprint,
                        ),
                        _summary_diagnostic(
                            failed_context,
                            provider_calls=provider_calls,
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=False,
                            outcome="failover_configuration_failed",
                            policy_fingerprint=self._policy.fingerprint,
                        ),
                    )
                )
                raise PlannerError(
                    exc.code,
                    str(exc),
                    category=exc.category,
                    diagnostics=tuple(diagnostics),
                ) from exc
            check_cancel()
            kind = PlanningAttemptKind.FAILOVER

        def fail(
            exc: PlannerError,
            context: PlanningDiagnosticContext | None,
            failure: PlanningDiagnostic | None,
        ) -> None:
            if context is not None:
                if exc.code == "UNSUPPORTED_REQUEST":
                    code = "PLANNING_RECOVERY_SUPPRESSED"
                    reason = "unsupported_terminal"
                    outcome = "unsupported"
                elif kind is PlanningAttemptKind.FAILOVER:
                    code = "PROFILE_FAILOVER_EXHAUSTED"
                    reason = "profile_failover_exhausted"
                    outcome = "failover_failed"
                elif kind is PlanningAttemptKind.REPAIR:
                    code = "PLAN_REPAIR_EXHAUSTED"
                    reason = "repair_attempt_exhausted"
                    outcome = "repair_failed"
                elif failure is not None and failure.stage in _REPAIRABLE_STAGES:
                    code = "PLAN_REPAIR_SUPPRESSED"
                    reason = _repair_suppression_reason(
                        policy=self._policy,
                        provider_calls=provider_calls,
                        repairs=repairs,
                        primary_local_actions=primary_local_actions,
                    )
                    outcome = "failed"
                else:
                    code = "TRANSPORT_RETRY_SUPPRESSED"
                    reason = _suppression_reason(
                        exc,
                        failure,
                        kind=kind,
                        policy=self._policy,
                        provider_calls=provider_calls,
                        transport_retries=transport_retries,
                    )
                    outcome = (
                        "transport_failed"
                        if kind is PlanningAttemptKind.TRANSPORT_RETRY
                        else "failed"
                    )
                emit(
                    (
                        context.diagnostic(
                            PlanningDiagnosticStage.RECOVERY,
                            code,
                            "rejected",
                            previous_failure_stage=(
                                None if failure is None else failure.stage
                            ),
                            previous_failure_code=exc.code,
                            recovery_action="none",
                            recovery_suppression_reason=reason,
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=failovers > 0,
                            recovery_policy_fingerprint=self._policy.fingerprint,
                        ),
                        _summary_diagnostic(
                            context,
                            provider_calls=provider_calls,
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=failovers > 0,
                            outcome=outcome,
                            policy_fingerprint=self._policy.fingerprint,
                        ),
                    )
                )
            raise PlannerError(
                exc.code,
                str(exc),
                category=exc.category,
                diagnostics=tuple(diagnostics),
            ) from exc

        kind = PlanningAttemptKind.INITIAL
        while True:
            check_cancel()
            if provider_calls >= self._policy.max_total_provider_calls:
                raise RuntimeError(
                    "Planning recovery exceeded its provider-call invariant."
                )
            if kind is PlanningAttemptKind.FAILOVER:
                failovers += 1
            provider_calls += 1
            logical_index = provider_calls
            try:
                candidate = attempt(
                    kind,
                    logical_index,
                    provider_calls,
                    repair_context,
                )
            except PlannerError as exc:
                emit(exc.diagnostics)
                context = _diagnostic_context(exc.diagnostics)
                last_context = context
                check_cancel()
                failure = _last_failure(exc.diagnostics)
                retryable = (
                    kind is PlanningAttemptKind.INITIAL
                    and failure is not None
                    and failure.stage is PlanningDiagnosticStage.PROVIDER
                    and exc.code in self._policy.retryable_provider_codes
                    and transport_retries < self._policy.max_transport_retries
                    and primary_local_actions
                    < self._policy.max_primary_local_recovery_actions
                    and provider_calls < self._policy.max_total_provider_calls
                )
                if retryable:
                    assert context is not None and failure is not None
                    retry_after = _retry_delay(failure, self._policy)
                    transport_retries += 1
                    primary_local_actions += 1
                    emit(
                        (
                            context.diagnostic(
                                PlanningDiagnosticStage.RECOVERY,
                                "TRANSPORT_RETRY_SCHEDULED",
                                "succeeded",
                                previous_failure_stage=failure.stage,
                                previous_failure_code=exc.code,
                                recovery_action="transport_retry",
                                retry_after_seconds=retry_after,
                                retry_used=True,
                                repair_used=False,
                                recovery_policy_fingerprint=(
                                    self._policy.fingerprint
                                ),
                            ),
                        )
                    )
                    check_cancel()
                    self._wait(retry_after, check_cancel)
                    check_cancel()
                    kind = PlanningAttemptKind.TRANSPORT_RETRY
                    repair_context = None
                    continue
                if can_repair(failure):
                    assert context is not None and failure is not None
                    schedule_repair(context, failure)
                    continue
                if can_failover(failure):
                    assert context is not None and failure is not None
                    schedule_failover(context, failure)
                    continue
                fail(exc, context, failure)

            last_context = candidate.context
            emit(candidate.diagnostics)
            check_cancel()
            try:
                preflight, preflight_diagnostics = validate_candidate(candidate)
            except Exception as exc:
                context = candidate.context
                failure = context.diagnostic(
                    PlanningDiagnosticStage.PREFLIGHT,
                    "PREFLIGHT_UNEXPECTED_ERROR",
                    "failed",
                    candidate_constructed=True,
                    candidate_preflight_passed=False,
                    reason_code="preflight_exception",
                    retry_used=transport_retries > 0,
                    repair_used=repairs > 0,
                )
                emit(
                    (
                        failure,
                        context.diagnostic(
                            PlanningDiagnosticStage.RECOVERY,
                            "PLAN_REPAIR_SUPPRESSED",
                            "rejected",
                            previous_failure_stage=failure.stage,
                            previous_failure_code=failure.code,
                            recovery_action="none",
                            recovery_suppression_reason=(
                                "internal_preflight_failure"
                            ),
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=failovers > 0,
                            recovery_policy_fingerprint=self._policy.fingerprint,
                        ),
                        _summary_diagnostic(
                            context,
                            provider_calls=provider_calls,
                            retry_used=transport_retries > 0,
                            repair_used=repairs > 0,
                            failover_used=failovers > 0,
                            outcome="failed",
                            policy_fingerprint=self._policy.fingerprint,
                        ),
                    )
                )
                raise PlannerError(
                    "PREFLIGHT_UNEXPECTED_ERROR",
                    "Plan preflight raised an unexpected orchestration error.",
                    category=ErrorCategory.INTERNAL_AGENT_ERROR,
                    diagnostics=tuple(diagnostics),
                ) from exc

            emit(preflight_diagnostics)
            if not preflight.passed:
                failure = preflight_diagnostics[-1]
                error = preflight.error
                planner_error = PlannerError(
                    "PLAN_PREFLIGHT_FAILED" if error is None else error.code,
                    "Candidate Plan failed authoritative preflight.",
                    category=ErrorCategory.INTERNAL_AGENT_ERROR,
                )
                check_cancel()
                if can_repair(failure):
                    schedule_repair(candidate.context, failure)
                    continue
                if can_failover(failure):
                    schedule_failover(candidate.context, failure)
                    continue
                fail(planner_error, candidate.context, failure)

            check_cancel()
            if failovers:
                outcome = "failover_recovered"
            elif repairs:
                outcome = "repair_recovered"
            elif transport_retries:
                outcome = "transport_recovered"
            else:
                outcome = "initial_success"
            emit(
                (
                    _summary_diagnostic(
                        candidate.context,
                        provider_calls=provider_calls,
                        retry_used=transport_retries > 0,
                        repair_used=repairs > 0,
                        failover_used=failovers > 0,
                        outcome=outcome,
                        policy_fingerprint=self._policy.fingerprint,
                    ),
                )
            )
            return RecoveredPlanningAttempt(
                candidate,
                preflight,
                tuple(diagnostics),
                provider_calls,
                transport_retries > 0,
                repairs > 0,
                failovers > 0,
            )

    def _wait(self, delay: float, check_cancel: Callable[[], None]) -> None:
        remaining = delay
        while remaining > 0:
            check_cancel()
            interval = min(remaining, _CANCELLATION_POLL_SECONDS)
            self._sleeper(interval)
            remaining -= interval
        check_cancel()


def _diagnostic_context(
    diagnostics: tuple[PlanningDiagnostic, ...],
) -> PlanningDiagnosticContext | None:
    return None if not diagnostics else diagnostics[-1].context


def _last_failure(
    diagnostics: tuple[PlanningDiagnostic, ...],
) -> PlanningDiagnostic | None:
    return next(
        (item for item in reversed(diagnostics) if item.outcome in {"failed", "rejected"}),
        None,
    )


def _retry_delay(
    failure: PlanningDiagnostic,
    policy: PlanningRecoveryPolicy,
) -> float:
    proposed = failure.retry_after_seconds
    if proposed is None:
        proposed = _DEFAULT_RETRY_DELAY_SECONDS
    return min(float(proposed), policy.max_retry_delay_seconds)


def _suppression_reason(
    exc: PlannerError,
    failure: PlanningDiagnostic | None,
    *,
    kind: PlanningAttemptKind,
    policy: PlanningRecoveryPolicy,
    provider_calls: int,
    transport_retries: int,
) -> str:
    if exc.code == "UNSUPPORTED_REQUEST":
        return "unsupported_terminal"
    if failure is None or failure.stage is not PlanningDiagnosticStage.PROVIDER:
        return "not_transport_failure"
    if exc.code not in policy.retryable_provider_codes:
        return "provider_failure_not_retryable"
    if kind is not PlanningAttemptKind.INITIAL or transport_retries:
        return "transport_retry_exhausted"
    if policy.max_transport_retries == 0:
        return "transport_retry_disabled"
    if policy.max_primary_local_recovery_actions == 0:
        return "primary_local_recovery_disabled"
    if provider_calls >= policy.max_total_provider_calls:
        return "provider_call_bound_reached"
    return "transport_retry_suppressed"


def _repair_suppression_reason(
    *,
    policy: PlanningRecoveryPolicy,
    provider_calls: int,
    repairs: int,
    primary_local_actions: int,
) -> str:
    if repairs >= policy.max_repairs:
        return "repair_attempt_exhausted"
    if policy.max_repairs == 0:
        return "plan_repair_disabled"
    if primary_local_actions >= policy.max_primary_local_recovery_actions:
        return "primary_local_recovery_consumed"
    if provider_calls >= policy.max_total_provider_calls:
        return "provider_call_bound_reached"
    return "plan_repair_suppressed"


def _summary_diagnostic(
    context: PlanningDiagnosticContext,
    *,
    provider_calls: int,
    retry_used: bool,
    repair_used: bool,
    failover_used: bool,
    outcome: str,
    policy_fingerprint: str | None = None,
) -> PlanningDiagnostic:
    return context.diagnostic(
        PlanningDiagnosticStage.RECOVERY,
        "PLANNING_RECOVERY_SUMMARY",
        (
            "succeeded"
            if outcome
            in {
                "initial_success",
                "transport_recovered",
                "repair_recovered",
                "failover_recovered",
            }
            else "failed"
        ),
        total_provider_call_count=provider_calls,
        final_recovery_outcome=outcome,
        retry_used=retry_used,
        repair_used=repair_used,
        failover_used=failover_used,
        recovery_policy_fingerprint=policy_fingerprint,
    )


__all__ = [
    "PLANNING_RECOVERY_POLICY_VERSION",
    "PlanningRecoveryCancelled",
    "PlanningRecoveryCoordinator",
    "PlanningRecoveryPolicy",
    "PlanningRepairContext",
    "RecoveredPlanningAttempt",
]
