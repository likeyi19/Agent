"""Offline M9.4 tests for bounded transport retry and complete Plan repair."""

from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    FileRunStore,
    LLMPlanner,
    PlanningModelError,
    PlanningModelProfile,
    PlanningRecoveryPolicy,
    PlanExecutor,
    RunStoreError,
    RunMode,
    RunStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.providers import PlanningModelFactoryRegistry


_TRANSIENT = (
    "PROVIDER_RATE_LIMITED",
    "PROVIDER_TIMEOUT",
    "PROVIDER_CONNECTION_FAILED",
    "PROVIDER_UNAVAILABLE",
    "PROVIDER_COMPLETION_INCOMPLETE",
)
_TERMINAL = (
    "PROVIDER_AUTHENTICATION_FAILED",
    "PLANNING_PROVIDER_CONFIGURATION_FAILED",
    "PLANNING_PROVIDER_DEPENDENCY_MISSING",
    "PROVIDER_REFUSED",
    "PLANNING_PROVIDER_ERROR",
)


def _valid_response() -> str:
    optional = {
        name: None
        for name in build_default_tool_registry()
        .get("inspect_scATAC")
        .optional_arguments
    }
    return json.dumps(
        {
            "schema_version": 3,
            "status": "plan",
            "steps": [
                {
                    "step_id": "inspect",
                    "tool_name": "inspect_scATAC",
                    "arguments": {
                        "path": {
                            "binding_type": "input",
                            "input_name": "input_path",
                        },
                        **optional,
                    },
                    "depends_on": [],
                    "description": None,
                }
            ],
            "reason": None,
        }
    )


def _embedding_response() -> str:
    optional = {
        name: None
        for name in build_default_tool_registry()
        .get("epizoo_embed_cells")
        .optional_arguments
    }
    optional["device"] = {
        "binding_type": "input",
        "input_name": "device",
    }
    return json.dumps(
        {
            "schema_version": 3,
            "status": "plan",
            "steps": [
                {
                    "step_id": "embed",
                    "tool_name": "epizoo_embed_cells",
                    "arguments": {
                        "input_path": {
                            "binding_type": "input",
                            "input_name": "input_path",
                        },
                        "output_dir": {
                            "binding_type": "input",
                            "input_name": "output_dir",
                        },
                        "species": {
                            "binding_type": "input",
                            "input_name": "species",
                        },
                        **optional,
                    },
                    "depends_on": [],
                    "description": None,
                }
            ],
            "reason": None,
        }
    )


class ScriptedModel:
    model_id = "scripted-recovery-model"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, object]] = []

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls.append((prompt, response_schema))
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def _registry() -> tuple[ToolRegistry, Mock]:
    source = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("planning executed a scientific tool"))
    return ToolRegistry(
        tuple(replace(source.get(name), function=guard) for name in source.names())
    ), guard


def _request(request_id: str = "recovery-request", *, execute: bool = False):
    return AgentRequest(
        request_id,
        "Inspect the supplied scATAC dataset.",
        {"input_path": "/private/DO-NOT-PERSIST/input.h5ad"},
        RunMode.EXECUTE if execute else RunMode.PLAN_ONLY,
    )


def _profile() -> PlanningModelProfile:
    return PlanningModelProfile(
        "primary-planner", "groq", "openai/gpt-oss-20b", request_timeout_seconds=60
    )


def _run(
    model: ScriptedModel,
    *,
    store=None,
    sleeper=lambda _: None,
    request=None,
    recovery_profiles: tuple[PlanningModelProfile, ...] = (),
    model_factory_registry=None,
):
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            model,
            profile=_profile(),
            retry_sleeper=sleeper,
            recovery_profiles=recovery_profiles,
            model_factory_registry=model_factory_registry,
        ),
        registry=registry,
        run_store=store,
    )
    result = runtime.run(request or _request())
    guard.assert_not_called()
    return result, runtime


def _secondary_profile(
    *,
    provider_id: str = "backup",
    profile_id: str = "secondary-planner",
    model_id: str = "organization/secondary-model",
    enabled: bool = True,
    supports_structured_output: bool = True,
) -> PlanningModelProfile:
    return PlanningModelProfile(
        profile_id,
        provider_id,
        model_id,
        enabled=enabled,
        supports_structured_output=supports_structured_output,
        request_timeout_seconds=60,
    )


def _factory_registry(
    profile: PlanningModelProfile,
    model: ScriptedModel,
    calls: list[PlanningModelProfile] | None = None,
) -> PlanningModelFactoryRegistry:
    def factory(received: PlanningModelProfile):
        if calls is not None:
            calls.append(received)
        return model

    return PlanningModelFactoryRegistry({profile.provider_id: factory})


def _diagnostics(result):
    return [
        dict(event.details)
        for event in result.trace
        if event.details.get("diagnostic_schema_version") == 3
    ]


@pytest.mark.parametrize("code", _TRANSIENT)
def test_transient_failure_retries_once_and_recovers(code: str) -> None:
    delay = 99.0 if code == "PROVIDER_RATE_LIMITED" else None
    model = ScriptedModel(
        [
            PlanningModelError(
                "raw provider secret", code=code, retry_after_seconds=delay
            ),
            _valid_response(),
        ]
    )
    sleeps: list[float] = []

    result, _ = _run(model, sleeper=sleeps.append)
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
    assert model.calls[0] == model.calls[1]
    assert sum(sleeps) == pytest.approx(5.0 if delay is not None else 1.0)
    assert diagnostics[-1]["final_recovery_outcome"] == "transport_recovered"
    assert diagnostics[-1]["total_provider_call_count"] == 2
    assert diagnostics[-1]["retry_used"] is True
    assert diagnostics[-1]["repair_used"] is False
    assert diagnostics[-1]["failover_used"] is False
    assert len(diagnostics[-1]["recovery_policy_fingerprint"]) == 64
    assert {
        item["attempt_kind"] for item in diagnostics if item["stage"] == "provider"
    } == {"initial", "transport_retry"}


@pytest.mark.parametrize("code", _TERMINAL)
def test_terminal_provider_failure_does_not_retry(code: str) -> None:
    model = ScriptedModel([PlanningModelError("secret", code=code)])

    result, _ = _run(model)

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 1
    assert _diagnostics(result)[-1]["total_provider_call_count"] == 1
    assert _diagnostics(result)[-1]["retry_used"] is False


@pytest.mark.parametrize(
    "response,code",
    [
        ("not-json PRIVATE-CANDIDATE", "PLANNER_OUTPUT_INVALID"),
        (
            json.dumps(
                {
                    "schema_version": 3,
                    "status": "unsupported",
                    "steps": [],
                    "reason": "not supported PRIVATE-REASON",
                }
            ),
            "UNSUPPORTED_REQUEST",
        ),
    ],
)
def test_candidate_failure_is_repaired_but_unsupported_remains_terminal(
    response: str, code: str
) -> None:
    model = ScriptedModel([response, _valid_response()])

    result, _ = _run(model)

    if code == "PLANNER_OUTPUT_INVALID":
        assert result.status is RunStatus.PLANNED
        assert len(model.calls) == 2
    else:
        assert result.errors[0].code == code
        assert len(model.calls) == 1
    rendered = json.dumps(result.to_dict())
    assert "PRIVATE-CANDIDATE" not in rendered
    assert "PRIVATE-REASON" not in rendered


def test_retry_exhaustion_is_exactly_two_calls_and_never_uses_future_actions() -> None:
    model = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT"),
            PlanningModelError(code="PROVIDER_TIMEOUT"),
            _valid_response(),
        ]
    )

    result, _ = _run(model)
    summary = _diagnostics(result)[-1]

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 2
    assert summary["total_provider_call_count"] == 2
    assert summary["repair_used"] is False
    assert summary["failover_used"] is False


def test_retry_preserves_request_inputs_and_scientific_parameter_binding() -> None:
    inputs = {
        "input_path": "/synthetic/input.h5ad",
        "output_dir": "/synthetic/output",
        "species": "mouse",
        "device": "cuda:7",
    }
    request = AgentRequest(
        "parameter-preservation",
        "Embed the supplied data on the requested device.",
        inputs,
        RunMode.PLAN_ONLY,
    )
    snapshot = request.to_dict()
    model = ScriptedModel(
        [PlanningModelError(code="PROVIDER_TIMEOUT"), _embedding_response()]
    )

    result, _ = _run(model, request=request)

    assert request.to_dict() == snapshot
    assert model.calls[0] == model.calls[1]
    assert result.plan is not None
    assert result.plan.steps[0].arguments["device"] == "cuda:7"


def test_repair_preserves_request_inputs_and_scientific_parameter_binding() -> None:
    request = AgentRequest(
        "repair-parameter-preservation",
        "Embed the supplied data on the requested device.",
        {
            "input_path": "/synthetic/input.h5ad",
            "output_dir": "/synthetic/output",
            "species": "mouse",
            "device": "cuda:7",
        },
        RunMode.PLAN_ONLY,
    )
    snapshot = request.to_dict()
    model = ScriptedModel([_invalid_candidate("malformed"), _embedding_response()])

    result, _ = _run(model, request=request)

    assert request.to_dict() == snapshot
    assert len(model.calls) == 2
    assert model.calls[0][1] == model.calls[1][1]
    assert result.plan is not None
    assert result.plan.steps[0].arguments["device"] == "cuda:7"


def test_policy_is_frozen_versioned_bounded_and_deterministic() -> None:
    first = PlanningRecoveryPolicy()
    second = PlanningRecoveryPolicy()

    assert first.policy_version == "planning-recovery-v3"
    assert first.max_transport_retries == 1
    assert first.max_repairs == 1
    assert first.max_profile_failovers == 1
    assert first.max_primary_local_recovery_actions == 1
    assert first.max_total_provider_calls == 3
    assert first.max_retry_delay_seconds == 5.0
    assert first.fingerprint == second.fingerprint
    with pytest.raises(ValueError):
        PlanningRecoveryPolicy(max_transport_retries=2)
    with pytest.raises(ValueError):
        PlanningRecoveryPolicy(max_total_provider_calls=4)


@pytest.mark.parametrize(
    ("scenario", "expected_calls", "expected_outcome"),
    [
        ("initial_success", 1, "initial_success"),
        ("initial_unsupported", 1, "unsupported"),
        ("terminal_provider", 1, "failed"),
        ("transport_success", 2, "transport_recovered"),
        ("repair_success", 2, "repair_recovered"),
        ("transport_exhausted", 2, "transport_failed"),
        ("repair_exhausted", 2, "repair_failed"),
        ("transport_failover", 3, "failover_recovered"),
        ("repair_failover", 3, "failover_recovered"),
        ("failover_failed", 3, "failover_failed"),
    ],
)
def test_authoritative_provider_call_accounting_matrix(
    scenario: str,
    expected_calls: int,
    expected_outcome: str,
) -> None:
    unsupported = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "Indispensable information is absent.",
        }
    )
    transient = PlanningModelError(
        code="PROVIDER_TIMEOUT", retry_after_seconds=0
    )
    primary_outcomes: dict[str, list[object]] = {
        "initial_success": [_valid_response()],
        "initial_unsupported": [unsupported],
        "terminal_provider": [
            PlanningModelError(code="PROVIDER_AUTHENTICATION_FAILED")
        ],
        "transport_success": [transient, _valid_response()],
        "repair_success": [_invalid_candidate("malformed"), _valid_response()],
        "transport_exhausted": [transient, transient],
        "repair_exhausted": [
            _invalid_candidate("malformed"),
            _invalid_candidate("malformed"),
        ],
        "transport_failover": [transient, transient],
        "repair_failover": [
            _invalid_candidate("malformed"),
            _invalid_candidate("malformed"),
        ],
        "failover_failed": [transient, transient],
    }
    primary = ScriptedModel(primary_outcomes[scenario])
    uses_failover = scenario in {
        "transport_failover",
        "repair_failover",
        "failover_failed",
    }
    secondary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT")
            if scenario == "failover_failed"
            else _valid_response()
        ]
    )
    secondary_profile = _secondary_profile()
    kwargs = (
        {
            "recovery_profiles": (secondary_profile,),
            "model_factory_registry": _factory_registry(
                secondary_profile, secondary
            ),
        }
        if uses_failover
        else {}
    )

    result, _ = _run(primary, request=_repair_request(scenario), **kwargs)
    summary = _diagnostics(result)[-1]

    assert len(primary.calls) + len(secondary.calls) == expected_calls
    assert summary["total_provider_call_count"] == expected_calls
    assert summary["final_recovery_outcome"] == expected_outcome
    assert expected_calls <= 3


def test_cancellation_during_delay_suppresses_retry_and_execution(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    model = ScriptedModel(
        [PlanningModelError(code="PROVIDER_TIMEOUT"), _valid_response()]
    )
    runtime_box: list[AgentRuntime] = []

    def cancel_during_delay(_: float) -> None:
        runtime_box[0].cancel("cancel-delay:run")

    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            model, profile=_profile(), retry_sleeper=cancel_during_delay
        ),
        registry=registry,
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_request("cancel-delay"))

    assert result.status is RunStatus.CANCELLED
    assert len(model.calls) == 1
    guard.assert_not_called()


def test_cancellation_after_transient_failure_suppresses_retry(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    runtime_box: list[AgentRuntime] = []

    def cancel_and_fail() -> object:
        runtime_box[0].cancel("cancel-after-failure:run")
        return PlanningModelError(code="PROVIDER_CONNECTION_FAILED")

    model = ScriptedModel([cancel_and_fail, _valid_response()])
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_request("cancel-after-failure"))

    assert result.status is RunStatus.CANCELLED
    assert len(model.calls) == 1
    guard.assert_not_called()


def test_durable_retry_checkpoints_failure_and_persists_only_accepted_plan(
    tmp_path,
) -> None:
    store = FileRunStore(tmp_path)
    model = ScriptedModel(
        [PlanningModelError("RAW-SECRET", code="PROVIDER_TIMEOUT"), _valid_response()]
    )

    result, runtime = _run(
        model,
        store=store,
        request=_request("durable-recovery"),
    )
    state = store.load(result.run_id)
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert state.plan is not None
    assert state.preflight_verification is not None
    assert state.preflight_verification.passed
    assert any(item["code"] == "PROVIDER_TIMEOUT" for item in diagnostics)
    assert "RAW-SECRET" not in json.dumps(state.to_dict())

    calls_before_resume = len(model.calls)
    resumed = runtime.resume(result.run_id)
    assert resumed.status is RunStatus.PLANNED
    assert len(model.calls) == calls_before_resume


class _FailFirstUpdateStore:
    def __init__(self, root, *, fail_on: int = 1) -> None:
        self.base = FileRunStore(root)
        self.update_calls = 0
        self.fail_on = fail_on

    def execution_lease(self, run_id):
        return self.base.execution_lease(run_id)

    def create(self, state):
        return self.base.create(state)

    def load(self, run_id):
        return self.base.load(run_id)

    def request_cancellation(self, run_id):
        return self.base.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.base.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        self.update_calls += 1
        if self.update_calls == self.fail_on:
            raise RunStoreError("sanitized checkpoint failure")
        return self.base.update(state, expected_revision=expected_revision)


class _CancelAfterDiagnosticStore:
    def __init__(self, root, code: str) -> None:
        self.base = FileRunStore(root)
        self.code = code
        self.cancelled = False

    def execution_lease(self, run_id):
        return self.base.execution_lease(run_id)

    def create(self, state):
        return self.base.create(state)

    def load(self, run_id):
        return self.base.load(run_id)

    def request_cancellation(self, run_id):
        return self.base.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.base.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        saved = self.base.update(state, expected_revision=expected_revision)
        if not self.cancelled and any(
            event.details.get("code") == self.code for event in saved.trace
        ):
            self.base.request_cancellation(saved.run_id)
            self.cancelled = True
        return saved


class _InterruptAfterDiagnosticStore(_CancelAfterDiagnosticStore):
    class SimulatedProcessExit(BaseException):
        pass

    def update(self, state, *, expected_revision):
        saved = self.base.update(state, expected_revision=expected_revision)
        if not self.cancelled and any(
            event.details.get("code") == self.code for event in saved.trace
        ):
            self.cancelled = True
            raise self.SimulatedProcessExit()
        return saved


def test_diagnostic_checkpoint_failure_suppresses_retry(tmp_path) -> None:
    store = _FailFirstUpdateStore(tmp_path)
    model = ScriptedModel(
        [PlanningModelError(code="PROVIDER_TIMEOUT"), _valid_response()]
    )

    result, _ = _run(model, store=store, request=_request("checkpoint-failure"))

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 1
    assert store.update_calls == 2


def test_crash_before_plan_persistence_is_not_automatically_replanned(tmp_path) -> None:
    class SimulatedProcessExit(BaseException):
        pass

    class CrashingModel:
        model_id = "crashing-planning-model"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, prompt, response_schema):
            self.calls += 1
            raise SimulatedProcessExit()

    store = FileRunStore(tmp_path)
    registry, guard = _registry()
    model = CrashingModel()
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request("planning-crash"))

    interrupted = store.load("planning-crash:run")
    assert interrupted.plan is None
    resumed = runtime.resume("planning-crash:run")
    assert resumed.status is RunStatus.FAILED
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert model.calls == 1
    guard.assert_not_called()


def test_preflight_invalid_candidate_is_not_persisted_as_accepted_plan(tmp_path) -> None:
    payload = json.loads(_valid_response())
    embed_optional = {
        name: None
        for name in build_default_tool_registry()
        .get("epizoo_embed_cells")
        .optional_arguments
    }
    payload["steps"].append(
        {
            "step_id": "embed",
            "tool_name": "epizoo_embed_cells",
            "arguments": {
                "input_path": {
                    "binding_type": "ref",
                    "ref_step_id": "inspect",
                    "ref_output_key": "embedding_path",
                },
                "output_dir": {
                    "binding_type": "input",
                    "input_name": "output_dir",
                },
                "species": {
                    "binding_type": "input",
                    "input_name": "species",
                },
                **embed_optional,
            },
            "depends_on": ["inspect"],
            "description": None,
        }
    )
    store = FileRunStore(tmp_path)
    model = ScriptedModel([json.dumps(payload), json.dumps(payload)])

    request = AgentRequest(
        "invalid-candidate",
        "Embed after inspection.",
        {
            "input_path": "/private/DO-NOT-PERSIST/input.h5ad",
            "output_dir": "/private/DO-NOT-PERSIST/output",
            "species": "mouse",
        },
        RunMode.PLAN_ONLY,
    )
    result, _ = _run(model, store=store, request=request)
    state = store.load(result.run_id)

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 2
    assert state.plan is None
    assert state.plan_fingerprint is None


def test_cancellation_after_accepted_preflight_prevents_execution(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    registry, guard = _registry()
    runtime_box: list[AgentRuntime] = []

    class CancellingPreflightExecutor(PlanExecutor):
        def preflight(self, plan):
            verification = super().preflight(plan)
            runtime_box[0].cancel("cancel-after-preflight:run")
            return verification

    executor = CancellingPreflightExecutor(registry)
    model = ScriptedModel([_valid_response()])
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        executor=executor,
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_request("cancel-after-preflight", execute=True))

    assert result.status is RunStatus.CANCELLED
    assert len(model.calls) == 1
    guard.assert_not_called()


def test_noncanonical_preflight_valid_plan_remains_accepted() -> None:
    model = ScriptedModel([_valid_response()])

    result, _ = _run(model)

    assert result.status is RunStatus.PLANNED
    assert result.plan is not None
    assert tuple(step.tool_name for step in result.plan.steps) == ("inspect_scATAC",)


def _invalid_candidate(kind: str) -> str:
    if kind == "malformed":
        return "not-json RAW-CANDIDATE"
    if kind == "markdown":
        return f"```json\n{_valid_response()}\n```"
    payload = json.loads(_valid_response())
    step = payload["steps"][0]
    if kind == "schema":
        payload.pop("reason")
    elif kind == "unknown_tool":
        step["tool_name"] = "invented_tool_RAW"
        step["arguments"] = {}
    elif kind == "missing_argument":
        step["arguments"] = {}
    elif kind == "invalid_binding":
        step["arguments"]["path"] = {
            "binding_type": "literal",
            "value": "RAW-VALUE",
        }
    elif kind == "invented_input":
        step["arguments"]["path"]["input_name"] = "invented_input_RAW"
    elif kind == "invalid_dependency":
        step["depends_on"] = ["missing-step-RAW"]
    elif kind in {"invalid_step_ref", "invalid_result_field", "preflight"}:
        embed_optional = {
            name: None
            for name in build_default_tool_registry()
            .get("epizoo_embed_cells")
            .optional_arguments
        }
        ref_step_id = "missing-step-RAW" if kind == "invalid_step_ref" else "inspect"
        output_key = "input_path" if kind == "invalid_step_ref" else "missing_output_RAW"
        payload["steps"].append(
            {
                "step_id": "embed",
                "tool_name": "epizoo_embed_cells",
                "arguments": {
                    "input_path": {
                        "binding_type": "ref",
                        "ref_step_id": ref_step_id,
                        "ref_output_key": output_key,
                    },
                    "output_dir": {
                        "binding_type": "input",
                        "input_name": "output_dir",
                    },
                    "species": {
                        "binding_type": "input",
                        "input_name": "species",
                    },
                    **embed_optional,
                },
                "depends_on": [ref_step_id],
                "description": None,
            }
        )
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(kind)
    return json.dumps(payload)


def _repair_request(request_id: str = "repair-case") -> AgentRequest:
    return AgentRequest(
        request_id,
        "Inspect the supplied data and create an embedding when needed.",
        {
            "input_path": "/private/INPUT-PATH-SECRET/input.h5ad",
            "output_dir": "/private/OUTPUT-PATH-SECRET",
            "species": "mouse",
            "device": "cuda:7",
        },
        RunMode.PLAN_ONLY,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "malformed",
        "markdown",
        "schema",
        "unknown_tool",
        "missing_argument",
        "invalid_binding",
        "invented_input",
        "invalid_dependency",
        "invalid_step_ref",
        "invalid_result_field",
        "preflight",
    ],
)
def test_objective_candidate_failures_repair_to_complete_valid_plan(kind: str) -> None:
    model = ScriptedModel([_invalid_candidate(kind), _valid_response()])

    result, _ = _run(model, request=_repair_request(kind))
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
    assert diagnostics[-1]["final_recovery_outcome"] == "repair_recovered"
    assert diagnostics[-1]["total_provider_call_count"] == 2
    assert diagnostics[-1]["repair_used"] is True
    assert diagnostics[-1]["retry_used"] is False
    assert diagnostics[-1]["failover_used"] is False
    assert any(item["code"] == "PLAN_REPAIR_SCHEDULED" for item in diagnostics)
    assert any(
        item["attempt_kind"] == "repair"
        and item["code"] == "PLAN_REPAIR_CALL_STARTED"
        for item in diagnostics
    )


def test_repair_prompt_reuses_original_context_with_only_safe_diagnostic() -> None:
    raw = _invalid_candidate("invalid_binding")
    model = ScriptedModel([raw, _valid_response()])
    request = _repair_request("repair-context")
    request_snapshot = request.to_dict()

    result, _ = _run(model, request=request)
    initial_prompt = json.loads(model.calls[0][0])
    repair_prompt = json.loads(model.calls[1][0])

    assert result.status is RunStatus.PLANNED
    assert request.to_dict() == request_snapshot
    assert model.calls[0][1] == model.calls[1][1]
    assert initial_prompt["request"] == repair_prompt["request"]
    assert initial_prompt["tools"] == repair_prompt["tools"]
    assert initial_prompt["instructions"] == repair_prompt["instructions"]
    assert "repair" not in initial_prompt
    assert set(repair_prompt["repair"]) == {"instruction", "diagnostic"}
    diagnostic = repair_prompt["repair"]["diagnostic"]
    assert diagnostic["previous_failure_stage"] == "argument_binding"
    assert diagnostic["previous_failure_code"] == "PLANNER_BINDING_INVALID"
    assert diagnostic["argument_name"] == "path"
    rendered_repair = json.dumps(repair_prompt["repair"])
    for forbidden in (
        raw,
        "RAW-VALUE",
        "INPUT-PATH-SECRET",
        "OUTPUT-PATH-SECRET",
        "cuda:7",
    ):
        assert forbidden not in rendered_repair
    assert model.calls[0][0] == json.dumps(
        initial_prompt,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_repair_invalid_or_provider_failure_stops_at_two_calls() -> None:
    invalid = _invalid_candidate("malformed")
    invalid_model = ScriptedModel([invalid, invalid, _valid_response()])
    provider_model = ScriptedModel(
        [
            invalid,
            PlanningModelError(code="PROVIDER_TIMEOUT"),
            _valid_response(),
        ]
    )

    invalid_result, _ = _run(invalid_model, request=_repair_request("repair-invalid"))
    provider_result, _ = _run(
        provider_model, request=_repair_request("repair-provider-failure")
    )

    assert invalid_result.status is RunStatus.FAILED
    assert provider_result.status is RunStatus.FAILED
    assert len(invalid_model.calls) == 2
    assert len(provider_model.calls) == 2
    assert _diagnostics(invalid_result)[-1]["final_recovery_outcome"] == "repair_failed"
    assert _diagnostics(provider_result)[-1]["final_recovery_outcome"] == "repair_failed"


def test_repair_explicit_unsupported_is_terminal_without_third_call() -> None:
    unsupported = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "Indispensable information is absent.",
        }
    )
    model = ScriptedModel(
        [_invalid_candidate("malformed"), unsupported, _valid_response()]
    )

    result, _ = _run(model, request=_repair_request("repair-unsupported"))

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "UNSUPPORTED_REQUEST"
    assert len(model.calls) == 2
    assert _diagnostics(result)[-1]["final_recovery_outcome"] == "unsupported"


def test_unexpected_preflight_exception_is_not_repaired() -> None:
    registry, guard = _registry()

    class FailingPreflightExecutor(PlanExecutor):
        def preflight(self, plan):
            raise RuntimeError("RAW INTERNAL PREFLIGHT SECRET")

    model = ScriptedModel([_valid_response(), _valid_response()])
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        executor=FailingPreflightExecutor(registry),
    )

    result = runtime.run(_repair_request("unexpected-preflight"))

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "PREFLIGHT_UNEXPECTED_ERROR"
    assert len(model.calls) == 1
    assert "RAW INTERNAL" not in json.dumps(result.to_dict())
    guard.assert_not_called()


def test_local_catalog_invariant_failure_is_not_repaired() -> None:
    source = build_default_tool_registry().get("inspect_scATAC")
    invalid = replace(
        source,
        required_arguments={
            "path": replace(
                source.required_arguments["path"],
                allow_step_output_ref=False,
            )
        },
    )
    guard = Mock(side_effect=AssertionError("catalog failure executed a tool"))
    registry = ToolRegistry((replace(invalid, function=guard),))
    model = ScriptedModel([_valid_response(), _valid_response()])
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
    )

    result = runtime.run(
        AgentRequest("catalog-invalid", "Inspect data.", {}, RunMode.PLAN_ONLY)
    )

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "PLANNER_CATALOG_INVALID"
    assert len(model.calls) == 0
    guard.assert_not_called()


def test_transport_retry_never_stacks_repair() -> None:
    model = ScriptedModel(
        [
            PlanningModelError(
                code="PROVIDER_TIMEOUT", retry_after_seconds=0
            ),
            _invalid_candidate("malformed"),
            _valid_response(),
        ]
    )

    result, _ = _run(model, request=_repair_request("retry-no-repair"))
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 2
    assert any(item["attempt_kind"] == "transport_retry" for item in diagnostics)
    assert not any(item["attempt_kind"] == "repair" for item in diagnostics)
    assert diagnostics[-1]["retry_used"] is True
    assert diagnostics[-1]["repair_used"] is False


def test_cancellation_after_candidate_failure_suppresses_repair(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    runtime_box: list[AgentRuntime] = []

    def cancel_then_return_invalid() -> str:
        runtime_box[0].cancel("cancel-repair:run")
        return _invalid_candidate("malformed")

    model = ScriptedModel([cancel_then_return_invalid, _valid_response()])
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_repair_request("cancel-repair"))

    assert result.status is RunStatus.CANCELLED
    assert len(model.calls) == 1
    guard.assert_not_called()


def test_cancellation_after_repaired_plan_acceptance_prevents_execution(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    registry, guard = _registry()
    runtime_box: list[AgentRuntime] = []

    class CancellingRepairPreflightExecutor(PlanExecutor):
        def preflight(self, plan):
            verification = super().preflight(plan)
            runtime_box[0].cancel("cancel-repaired-plan:run")
            return verification

    model = ScriptedModel([_invalid_candidate("malformed"), _valid_response()])
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        executor=CancellingRepairPreflightExecutor(registry),
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(
        AgentRequest(
            "cancel-repaired-plan",
            "Inspect the supplied dataset.",
            {"input_path": "/synthetic/input.h5ad"},
            RunMode.EXECUTE,
        )
    )

    assert result.status is RunStatus.CANCELLED
    assert len(model.calls) == 2
    guard.assert_not_called()


def test_durable_failure_and_repair_decision_checkpoint_before_call_two(tmp_path) -> None:
    store = FileRunStore(tmp_path)

    def inspect_checkpoint_then_repair() -> str:
        state = store.load("repair-checkpoint:run")
        codes = tuple(
            event.details.get("code")
            for event in state.trace
            if event.details.get("diagnostic_schema_version") == 3
        )
        assert "PLANNER_OUTPUT_INVALID" in codes
        assert "PLAN_REPAIR_SCHEDULED" in codes
        assert state.plan is None
        return _valid_response()

    model = ScriptedModel(
        [_invalid_candidate("malformed"), inspect_checkpoint_then_repair]
    )

    result, runtime = _run(
        model,
        store=store,
        request=_repair_request("repair-checkpoint"),
    )
    state = store.load(result.run_id)

    assert result.status is RunStatus.PLANNED
    assert state.plan is not None
    assert state.preflight_verification is not None
    assert state.preflight_verification.passed
    assert "RAW-CANDIDATE" not in json.dumps(state.to_dict())
    before_resume = len(model.calls)
    assert runtime.resume(result.run_id).status is RunStatus.PLANNED
    assert len(model.calls) == before_resume


def test_repair_checkpoint_failure_suppresses_call_two(tmp_path) -> None:
    store = _FailFirstUpdateStore(tmp_path, fail_on=2)
    model = ScriptedModel([_invalid_candidate("malformed"), _valid_response()])

    result, _ = _run(
        model,
        store=store,
        request=_repair_request("repair-checkpoint-failure"),
    )

    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 1


def test_interrupted_repair_is_not_replayed_on_resume(tmp_path) -> None:
    class SimulatedProcessExit(BaseException):
        pass

    store = FileRunStore(tmp_path)

    def crash_during_repair() -> str:
        raise SimulatedProcessExit()

    model = ScriptedModel(
        [_invalid_candidate("malformed"), crash_during_repair]
    )
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(model, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(SimulatedProcessExit):
        runtime.run(_repair_request("repair-crash"))

    assert store.load("repair-crash:run").plan is None
    resumed = runtime.resume("repair-crash:run")
    assert resumed.status is RunStatus.FAILED
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert len(model.calls) == 2
    guard.assert_not_called()


@pytest.mark.parametrize(
    "code",
    [
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
        "PROVIDER_COMPLETION_INCOMPLETE",
    ],
)
def test_exhausted_transient_retry_fails_over_to_secondary_success(code: str) -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code=code, retry_after_seconds=0),
            PlanningModelError(code=code, retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert factory_calls == [secondary_profile]
    assert secondary.calls[0] == primary.calls[0]
    assert [
        item["attempt_kind"]
        for item in diagnostics
        if item["code"] == "PROVIDER_CALL_STARTED"
    ] == ["initial", "transport_retry", "failover"]
    assert diagnostics[-1]["final_recovery_outcome"] == "failover_recovered"
    assert diagnostics[-1]["total_provider_call_count"] == 3
    assert diagnostics[-1]["retry_used"] is True
    assert diagnostics[-1]["repair_used"] is False
    assert diagnostics[-1]["failover_used"] is True
    failover_started = next(
        item
        for item in diagnostics
        if item["code"] == "PROFILE_FAILOVER_CALL_STARTED"
    )
    assert failover_started["previous_failure_stage"] == "provider"
    assert failover_started["previous_failure_code"] == code


@pytest.mark.parametrize(
    "kind",
    [
        "malformed",
        "markdown",
        "schema",
        "unknown_tool",
        "missing_argument",
        "invalid_binding",
        "invented_input",
        "invalid_dependency",
        "invalid_step_ref",
        "invalid_result_field",
        "preflight",
    ],
)
def test_failed_candidate_repair_fails_over_to_secondary_success(kind: str) -> None:
    raw = _invalid_candidate(kind)
    primary = ScriptedModel([raw, raw])
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()

    result, _ = _run(
        primary,
        request=_repair_request(f"failover-{kind}"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )
    diagnostics = _diagnostics(result)
    failover_prompt = json.loads(secondary.calls[0][0])

    assert result.status is RunStatus.PLANNED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert "repair" not in failover_prompt
    assert set(failover_prompt["failover"]) == {"instruction", "diagnostic"}
    rendered = json.dumps(failover_prompt["failover"])
    for forbidden in (
        raw,
        "RAW-CANDIDATE",
        "RAW-VALUE",
        "INPUT-PATH-SECRET",
        "OUTPUT-PATH-SECRET",
        "cuda:7",
    ):
        assert forbidden not in rendered
    assert diagnostics[-1]["final_recovery_outcome"] == "failover_recovered"
    assert diagnostics[-1]["repair_used"] is True
    assert diagnostics[-1]["retry_used"] is False
    assert diagnostics[-1]["failover_used"] is True


@pytest.mark.parametrize(
    ("provider_id", "model_id"),
    [
        ("groq", "organization/secondary-model"),
        ("other", "different-family/model"),
    ],
)
def test_failover_is_profile_agnostic_through_factory_registry(
    provider_id: str,
    model_id: str,
) -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile(
        provider_id=provider_id,
        model_id=model_id,
    )
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )

    assert result.status is RunStatus.PLANNED
    assert factory_calls == [secondary_profile]
    profile_usage = [
        item["profile_id"]
        for item in _diagnostics(result)
        if item["code"] == "PROVIDER_CALL_STARTED"
    ]
    assert profile_usage == [
        "primary-planner",
        "primary-planner",
        "secondary-planner",
    ]
    failover_call = next(
        item
        for item in _diagnostics(result)
        if item["code"] == "PROFILE_FAILOVER_CALL_STARTED"
    )
    assert failover_call["provider_id"] == provider_id
    assert len(failover_call["model_identity_digest"]) == 64
    assert model_id not in json.dumps(failover_call)


@pytest.mark.parametrize(
    "profile",
    [
        _secondary_profile(enabled=False),
        _secondary_profile(supports_structured_output=False),
    ],
)
def test_disabled_or_incapable_secondary_configuration_is_rejected(
    profile: PlanningModelProfile,
) -> None:
    factory_calls: list[PlanningModelProfile] = []
    registry = _factory_registry(profile, ScriptedModel([]), factory_calls)

    with pytest.raises(ValueError):
        LLMPlanner(
            ScriptedModel([]),
            profile=_profile(),
            recovery_profiles=(profile,),
            model_factory_registry=registry,
        )

    assert factory_calls == []


def test_invalid_duplicate_or_excess_secondary_configuration_is_rejected() -> None:
    primary = _profile()
    duplicate_id = _secondary_profile(profile_id=primary.profile_id)
    duplicate_model = _secondary_profile(
        provider_id=primary.provider_id,
        model_id=primary.model_id,
    )
    secondary = _secondary_profile()
    registry = _factory_registry(secondary, ScriptedModel([]))

    for profiles in ((duplicate_id,), (duplicate_model,), (secondary, secondary)):
        with pytest.raises(ValueError):
            LLMPlanner(
                ScriptedModel([]),
                profile=primary,
                recovery_profiles=profiles,
                model_factory_registry=registry,
            )
    with pytest.raises(TypeError):
        LLMPlanner(
            ScriptedModel([]),
            profile=primary,
            recovery_profiles=[secondary],  # type: ignore[arg-type]
            model_factory_registry=registry,
        )


def test_missing_secondary_stops_after_primary_call_two() -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            _valid_response(),
        ]
    )

    result, _ = _run(primary)

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 2
    assert _diagnostics(result)[-1]["failover_used"] is False


def test_unknown_secondary_provider_configuration_is_rejected() -> None:
    secondary = _secondary_profile(provider_id="unregistered")
    registry = PlanningModelFactoryRegistry(
        {"backup": lambda _: ScriptedModel([_valid_response()])}
    )

    with pytest.raises(ValueError):
        LLMPlanner(
            ScriptedModel([]),
            profile=_profile(),
            recovery_profiles=(secondary,),
            model_factory_registry=registry,
        )


def test_secondary_factory_configuration_failure_is_terminal_and_sanitized() -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    def failing_factory(profile: PlanningModelProfile):
        factory_calls.append(profile)
        raise PlanningModelError(
            "RAW SECONDARY CONFIG SECRET",
            code="PLANNING_PROVIDER_CONFIGURATION_FAILED",
        )

    registry = PlanningModelFactoryRegistry(
        {secondary_profile.provider_id: failing_factory}
    )

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=registry,
    )
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "PLANNING_PROVIDER_CONFIGURATION_FAILED"
    assert factory_calls == [secondary_profile]
    assert diagnostics[-1]["total_provider_call_count"] == 2
    assert diagnostics[-1]["failover_used"] is False
    assert "RAW SECONDARY" not in json.dumps(result.to_dict())


@pytest.mark.parametrize(
    "first",
    [
        PlanningModelError(code="PROVIDER_AUTHENTICATION_FAILED"),
        PlanningModelError(code="PROVIDER_REFUSED"),
        PlanningModelError(code="PLANNING_PROVIDER_CONFIGURATION_FAILED"),
        PlanningModelError(code="PLANNING_PROVIDER_DEPENDENCY_MISSING"),
    ],
)
def test_terminal_primary_provider_failure_never_constructs_failover(first) -> None:
    primary = ScriptedModel([first])
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 1
    assert factory_calls == []
    assert secondary.calls == []


@pytest.mark.parametrize(
    "terminal_code",
    [
        "PROVIDER_AUTHENTICATION_FAILED",
        "PROVIDER_REFUSED",
        "PLANNING_PROVIDER_CONFIGURATION_FAILED",
        "PLANNING_PROVIDER_DEPENDENCY_MISSING",
    ],
)
@pytest.mark.parametrize("local_path", ["transport_retry", "repair"])
def test_terminal_call_two_failure_never_fails_over(
    terminal_code: str,
    local_path: str,
) -> None:
    first = (
        PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0)
        if local_path == "transport_retry"
        else _invalid_candidate("malformed")
    )
    primary = ScriptedModel(
        [first, PlanningModelError(code=terminal_code), _valid_response()]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 2
    assert factory_calls == []
    assert secondary.calls == []


def test_explicit_unsupported_never_constructs_failover() -> None:
    primary = ScriptedModel(
        [
            json.dumps(
                {
                    "schema_version": 3,
                    "status": "unsupported",
                    "steps": [],
                    "reason": "Indispensable information is absent.",
                }
            )
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )

    assert result.errors[0].code == "UNSUPPORTED_REQUEST"
    assert len(primary.calls) == 1
    assert factory_calls == []


def test_explicit_unsupported_from_failover_is_final() -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    unsupported = json.dumps(
        {
            "schema_version": 3,
            "status": "unsupported",
            "steps": [],
            "reason": "The configured models cannot support this request.",
        }
    )
    secondary = ScriptedModel([unsupported, _valid_response()])
    secondary_profile = _secondary_profile()

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )

    assert result.errors[0].code == "UNSUPPORTED_REQUEST"
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert _diagnostics(result)[-1]["final_recovery_outcome"] == "unsupported"


@pytest.mark.parametrize("path", ["retry_candidate", "repair_transport"])
def test_mixed_primary_failure_paths_use_only_failover_as_call_three(path: str) -> None:
    if path == "retry_candidate":
        primary = ScriptedModel(
            [
                PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
                _invalid_candidate("malformed"),
            ]
        )
    else:
        primary = ScriptedModel(
            [
                _invalid_candidate("malformed"),
                PlanningModelError(code="PROVIDER_TIMEOUT"),
            ]
        )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()

    result, _ = _run(
        primary,
        request=_repair_request(f"mixed-{path}"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert diagnostics[-1]["total_provider_call_count"] == 3
    assert diagnostics[-1]["final_recovery_outcome"] == "failover_recovered"
    if path == "retry_candidate":
        assert diagnostics[-1]["retry_used"] is True
        assert diagnostics[-1]["repair_used"] is False
    else:
        assert diagnostics[-1]["retry_used"] is False
        assert diagnostics[-1]["repair_used"] is True


@pytest.mark.parametrize(
    "secondary_outcome",
    [
        PlanningModelError(code="PROVIDER_TIMEOUT"),
        "not-json FINAL-RAW-CANDIDATE",
    ],
)
def test_failover_is_final_and_cannot_retry_repair_or_call_four(
    secondary_outcome: object,
) -> None:
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([secondary_outcome, _valid_response()])
    secondary_profile = _secondary_profile()

    result, _ = _run(
        primary,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert diagnostics[-1]["total_provider_call_count"] == 3
    assert diagnostics[-1]["final_recovery_outcome"] == "failover_failed"
    assert not any(
        item["attempt_kind"] == "failover"
        and item.get("recovery_action") in {"transport_retry", "repair"}
        for item in diagnostics
    )


def test_candidate_failover_preserves_request_scientific_parameters() -> None:
    primary = ScriptedModel(
        [_invalid_candidate("malformed"), _invalid_candidate("malformed")]
    )
    secondary = ScriptedModel([_embedding_response()])
    secondary_profile = _secondary_profile()
    request = AgentRequest(
        "failover-parameters",
        "Embed the supplied data on the requested device.",
        {
            "input_path": "/synthetic/input.h5ad",
            "output_dir": "/synthetic/output",
            "species": "mouse",
            "device": "cuda:7",
        },
        RunMode.PLAN_ONLY,
    )
    snapshot = request.to_dict()

    result, _ = _run(
        primary,
        request=request,
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )

    assert request.to_dict() == snapshot
    assert result.plan is not None
    assert result.plan.steps[0].arguments["device"] == "cuda:7"


def test_cancellation_after_call_two_suppresses_failover_construction(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    runtime_box: list[AgentRuntime] = []

    def cancel_and_fail() -> object:
        runtime_box[0].cancel("cancel-failover:run")
        return PlanningModelError(code="PROVIDER_TIMEOUT")

    primary = ScriptedModel(
        [PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0), cancel_and_fail]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            primary,
            profile=_profile(),
            retry_sleeper=lambda _: None,
            recovery_profiles=(secondary_profile,),
            model_factory_registry=_factory_registry(
                secondary_profile, secondary, factory_calls
            ),
        ),
        registry=registry,
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_request("cancel-failover"))

    assert result.status is RunStatus.CANCELLED
    assert len(primary.calls) == 2
    assert factory_calls == []
    guard.assert_not_called()


def test_cancellation_after_failover_plan_prevents_execution(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    runtime_box: list[AgentRuntime] = []

    class CancellingFailoverPreflightExecutor(PlanExecutor):
        def preflight(self, plan):
            verification = super().preflight(plan)
            runtime_box[0].cancel("cancel-after-failover:run")
            return verification

    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            primary,
            profile=_profile(),
            retry_sleeper=lambda _: None,
            recovery_profiles=(secondary_profile,),
            model_factory_registry=_factory_registry(secondary_profile, secondary),
        ),
        registry=registry,
        executor=CancellingFailoverPreflightExecutor(registry),
        run_store=store,
    )
    runtime_box.append(runtime)

    result = runtime.run(_request("cancel-after-failover", execute=True))

    assert result.status is RunStatus.CANCELLED
    assert len(secondary.calls) == 1
    guard.assert_not_called()


def test_durable_failover_decision_checkpoint_precedes_secondary_construction(
    tmp_path,
) -> None:
    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()

    def factory(profile: PlanningModelProfile):
        assert profile == secondary_profile
        state = store.load("durable-failover:run")
        codes = tuple(
            event.details.get("code")
            for event in state.trace
            if event.details.get("diagnostic_schema_version") == 3
        )
        assert codes.count("PROVIDER_TIMEOUT") == 2
        assert "PROFILE_FAILOVER_SCHEDULED" in codes
        assert state.plan is None
        return secondary

    factory_registry = PlanningModelFactoryRegistry(
        {secondary_profile.provider_id: factory}
    )

    result, runtime = _run(
        primary,
        store=store,
        request=_request("durable-failover"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=factory_registry,
    )
    state = store.load(result.run_id)

    assert result.status is RunStatus.PLANNED
    assert state.plan is not None
    assert state.preflight_verification is not None
    assert state.preflight_verification.passed
    before_resume = len(secondary.calls)
    assert runtime.resume(result.run_id).status is RunStatus.PLANNED
    assert len(secondary.calls) == before_resume


def test_failover_decision_checkpoint_failure_suppresses_call_three(tmp_path) -> None:
    store = _FailFirstUpdateStore(tmp_path, fail_on=4)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    result, _ = _run(
        primary,
        store=store,
        request=_request("failover-checkpoint-failure"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(
            secondary_profile, secondary, factory_calls
        ),
    )

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 2
    assert factory_calls == []
    assert secondary.calls == []


def test_interrupted_failover_is_not_replayed_on_resume(tmp_path) -> None:
    class SimulatedProcessExit(BaseException):
        pass

    def crash() -> str:
        raise SimulatedProcessExit()

    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([crash])
    secondary_profile = _secondary_profile()
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            primary,
            profile=_profile(),
            retry_sleeper=lambda _: None,
            recovery_profiles=(secondary_profile,),
            model_factory_registry=_factory_registry(secondary_profile, secondary),
        ),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request("failover-crash"))

    assert store.load("failover-crash:run").plan is None
    resumed = runtime.resume("failover-crash:run")
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    guard.assert_not_called()


@pytest.mark.parametrize(
    ("path", "decision_code", "expected_primary_calls"),
    [
        ("retry", "TRANSPORT_RETRY_SCHEDULED", 1),
        ("repair", "PLAN_REPAIR_SCHEDULED", 1),
        ("failover", "PROFILE_FAILOVER_SCHEDULED", 2),
    ],
)
def test_cancellation_at_recovery_decision_checkpoint_suppresses_next_call(
    tmp_path,
    path: str,
    decision_code: str,
    expected_primary_calls: int,
) -> None:
    store = _CancelAfterDiagnosticStore(tmp_path, decision_code)
    if path == "repair":
        primary = ScriptedModel([_invalid_candidate("malformed"), _valid_response()])
    else:
        primary = ScriptedModel(
            [
                PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
                PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            ]
        )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []
    kwargs = (
        {
            "recovery_profiles": (secondary_profile,),
            "model_factory_registry": _factory_registry(
                secondary_profile, secondary, factory_calls
            ),
        }
        if path == "failover"
        else {}
    )

    result, _ = _run(
        primary,
        store=store,
        request=_repair_request(f"cancel-decision-{path}"),
        **kwargs,
    )

    assert result.status is RunStatus.CANCELLED
    assert len(primary.calls) == expected_primary_calls
    assert factory_calls == []
    assert secondary.calls == []


def test_cancellation_during_secondary_construction_suppresses_call_three(
    tmp_path,
) -> None:
    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel([_valid_response()])
    secondary_profile = _secondary_profile()
    factory_calls: list[PlanningModelProfile] = []

    def factory(profile: PlanningModelProfile):
        factory_calls.append(profile)
        store.request_cancellation("cancel-construction:run")
        return secondary

    factories = PlanningModelFactoryRegistry(
        {secondary_profile.provider_id: factory}
    )

    result, _ = _run(
        primary,
        store=store,
        request=_request("cancel-construction"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=factories,
    )

    assert result.status is RunStatus.CANCELLED
    assert len(primary.calls) == 2
    assert factory_calls == [secondary_profile]
    assert secondary.calls == []


def test_cancellation_observed_after_initial_response_prevents_later_work(
    tmp_path,
) -> None:
    store = FileRunStore(tmp_path)

    def cancel_in_flight() -> str:
        store.request_cancellation("cancel-initial-flight:run")
        return _valid_response()

    primary = ScriptedModel([cancel_in_flight, _valid_response()])

    result, _ = _run(
        primary,
        store=store,
        request=_request("cancel-initial-flight", execute=True),
    )

    assert result.status is RunStatus.CANCELLED
    assert len(primary.calls) == 1


def test_failed_failover_candidate_is_never_persisted(tmp_path) -> None:
    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary = ScriptedModel(["not-json RAW-FAILED-FAILOVER"])
    secondary_profile = _secondary_profile()

    result, _ = _run(
        primary,
        store=store,
        request=_request("failed-failover-persistence"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )
    state = store.load(result.run_id)

    assert result.status is RunStatus.FAILED
    assert state.plan is None
    assert state.plan_fingerprint is None
    assert "RAW-FAILED-FAILOVER" not in json.dumps(state.to_dict())


def test_secondary_construction_interruption_is_not_replayed(tmp_path) -> None:
    class SimulatedProcessExit(BaseException):
        pass

    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
            PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        ]
    )
    secondary_profile = _secondary_profile()
    factory_calls = 0

    def crash(_: PlanningModelProfile):
        nonlocal factory_calls
        factory_calls += 1
        raise SimulatedProcessExit()

    factories = PlanningModelFactoryRegistry(
        {secondary_profile.provider_id: crash}
    )
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            primary,
            profile=_profile(),
            retry_sleeper=lambda _: None,
            recovery_profiles=(secondary_profile,),
            model_factory_registry=factories,
        ),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request("construction-crash"))

    assert store.load("construction-crash:run").plan is None
    resumed = runtime.resume("construction-crash:run")
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert factory_calls == 1
    assert len(primary.calls) == 2
    guard.assert_not_called()


@pytest.mark.parametrize("phase", ["retry_delay", "retry_call"])
def test_interrupted_transport_recovery_is_not_replayed(
    tmp_path,
    phase: str,
) -> None:
    class SimulatedProcessExit(BaseException):
        pass

    def crash() -> str:
        raise SimulatedProcessExit()

    def crash_delay(_: float) -> None:
        raise SimulatedProcessExit()

    store = FileRunStore(tmp_path)
    primary = ScriptedModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT"),
            crash if phase == "retry_call" else _valid_response(),
        ]
    )
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(
            primary,
            profile=_profile(),
            retry_sleeper=(crash_delay if phase == "retry_delay" else lambda _: None),
        ),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request(f"{phase}-crash"))

    assert store.load(f"{phase}-crash:run").plan is None
    resumed = runtime.resume(f"{phase}-crash:run")
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert len(primary.calls) == (1 if phase == "retry_delay" else 2)
    guard.assert_not_called()


def test_accepted_candidate_interruption_before_plan_persistence_is_not_replayed(
    tmp_path,
) -> None:
    store = _InterruptAfterDiagnosticStore(
        tmp_path, "PLANNING_RECOVERY_SUMMARY"
    )
    primary = ScriptedModel([_valid_response()])
    registry, guard = _registry()
    runtime = AgentRuntime(
        planner=LLMPlanner(primary, profile=_profile(), retry_sleeper=lambda _: None),
        registry=registry,
        run_store=store,
    )

    with pytest.raises(_InterruptAfterDiagnosticStore.SimulatedProcessExit):
        runtime.run(_request("accepted-before-persist"))

    state = store.load("accepted-before-persist:run")
    assert state.plan is None
    resumed = runtime.resume("accepted-before-persist:run")
    assert resumed.errors[0].code == "PLANNING_INTERRUPTED_BEFORE_PLAN_AVAILABLE"
    assert len(primary.calls) == 1
    guard.assert_not_called()


def test_durable_diagnostics_recursively_exclude_sensitive_recovery_content(
    tmp_path,
) -> None:
    store = FileRunStore(tmp_path)
    raw_candidate = "not-json RAW-CANDIDATE-BODY"
    primary = ScriptedModel([raw_candidate, raw_candidate])
    secondary = ScriptedModel(
        [
            PlanningModelError(
                "RAW PROVIDER EXCEPTION BODY TOKEN-SECRET",
                code="PROVIDER_TIMEOUT",
            )
        ]
    )
    secondary_profile = _secondary_profile(model_id="safe-secondary-model")

    result, _ = _run(
        primary,
        store=store,
        request=AgentRequest(
            "diagnostic-privacy",
            "RAW USER PROMPT SECRET",
            {
                "input_path": "/private/RAW-PATH/input.h5ad",
                "token": "TOKEN-SECRET",
            },
            RunMode.PLAN_ONLY,
        ),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory_registry(secondary_profile, secondary),
    )
    diagnostic_details = [
        dict(event.details)
        for event in store.load(result.run_id).trace
        if event.details.get("diagnostic_schema_version") == 3
    ]
    rendered = json.dumps(diagnostic_details)

    for forbidden in (
        raw_candidate,
        "RAW-CANDIDATE-BODY",
        "RAW PROVIDER EXCEPTION",
        "RAW USER PROMPT SECRET",
        "/private/RAW-PATH",
        "TOKEN-SECRET",
    ):
        assert forbidden not in rendered


def test_preflight_valid_semantic_mismatch_does_not_trigger_recovery() -> None:
    primary = ScriptedModel([_valid_response(), _embedding_response()])
    request = AgentRequest(
        "semantic-boundary",
        "Create the requested EpiZoo embedding; inspection alone is insufficient.",
        {
            "input_path": "/synthetic/input.h5ad",
            "output_dir": "/synthetic/output",
            "species": "mouse",
        },
        RunMode.PLAN_ONLY,
    )

    result, _ = _run(primary, request=request)

    assert result.status is RunStatus.PLANNED
    assert tuple(step.tool_name for step in result.plan.steps) == ("inspect_scATAC",)
    assert len(primary.calls) == 1
    assert _diagnostics(result)[-1]["final_recovery_outcome"] == "initial_success"
