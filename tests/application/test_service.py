from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import threading

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import (
    ApplicationServiceError,
    ApplicationStage,
    ApplicationStatus,
    ResearchAgentApplication,
)
from agent.orchestration import (
    AgentRuntime,
    DeterministicPlanner,
    LLMPlanner,
    PlanningModelError,
    PlanningModelProfile,
    PlanningWireMode,
    RunAlreadyExistsError,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentPlan, AgentRequest, PlanStep, RunMode, RunStatus, StepStatus


def _tiny_h5ad(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0, 1], [0, 2, 0]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["cell-1", "cell-2"], dtype="object")),
        var=pd.DataFrame(index=pd.Index(["peak-1", "peak-2", "peak-3"], dtype="object")),
    ).write_h5ad(path)
    return path


def _counting_registry(calls: list[str]) -> ToolRegistry:
    default = build_default_tool_registry()
    inspection = default.get("inspect_scATAC")

    def counted(**arguments: object) -> object:
        calls.append("inspect_scATAC")
        return inspection.function(**arguments)

    return ToolRegistry(
        tuple(
            replace(spec, function=counted)
            if spec.name == "inspect_scATAC"
            else spec
            for spec in (default.get(name) for name in default.names())
        )
    )


def _request(path: Path, request_id: str = "application-inspect") -> AgentRequest:
    return AgentRequest(
        request_id,
        "Inspect this scATAC-seq dataset and generate a scientific report.",
        {"input_path": str(path)},
    )


def _deterministic_application(
    workspace: Path, **kwargs: object
) -> ResearchAgentApplication:
    return ResearchAgentApplication(
        workspace, planner=DeterministicPlanner(), **kwargs
    )


def _planning_response() -> str:
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
                        }
                    },
                    "depends_on": [],
                    "description": None,
                }
            ],
            "reason": None,
        }
    )


def _semantic_planning_response() -> str:
    return json.dumps(
        {
            "schema_version": 4,
            "decision": {
                "kind": "plan",
                "steps": [
                    {
                        "step_id": "inspect",
                        "tool": "inspect_scATAC",
                        "sources": [],
                        "control_dependencies": [],
                    }
                ],
            },
        }
    )


class _ScriptedPlanningModel:
    def __init__(self, *responses: object, model_id: str = "custom:model") -> None:
        self.model_id = model_id
        self.responses = list(responses)
        self.calls = 0

    def complete(self, *, prompt: str, response_schema: object) -> str:
        del prompt, response_schema
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response


def _profile(
    profile_id: str = "primary-profile",
    provider_id: str = "custom",
    model_id: str = "custom/model",
    **changes: object,
) -> PlanningModelProfile:
    values: dict[str, object] = {
        "profile_id": profile_id,
        "provider_id": provider_id,
        "model_id": model_id,
    }
    values.update(changes)
    return PlanningModelProfile(**values)  # type: ignore[arg-type]


def test_application_configured_new_run_is_llm_first_with_recovery_and_plan_only(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    scientific_calls: list[str] = []
    model = _ScriptedPlanningModel("not-json", _planning_response())
    factories = PlanningModelFactoryRegistry({"custom": lambda _: model})
    application = ResearchAgentApplication(
        tmp_path / "workspace",
        primary_planning_profile=_profile(),
        planning_model_factory_registry=factories,
        registry=_counting_registry(scientific_calls),
    )

    result = application.run(
        replace(_request(source, "llm-first-plan-only"), mode=RunMode.PLAN_ONLY)
    )

    assert isinstance(application.runtime.planner, LLMPlanner)
    assert application.runtime.planner.wire_mode is PlanningWireMode.V3
    assert application.runtime.planner.recovery_policy.policy_version == (
        "planning-recovery-v3"
    )
    assert result.status is ApplicationStatus.PLANNED
    assert result.run_status is RunStatus.PLANNED
    assert model.calls == 2
    assert scientific_calls == []
    assert any(
        event.details.get("final_recovery_outcome") == "repair_recovered"
        for event in result.run_result.trace
    )


def test_application_explicit_wire_v4_reaches_owned_llm_planner(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    model = _ScriptedPlanningModel(_semantic_planning_response())
    created: list[PlanningModelProfile] = []

    def create(profile: PlanningModelProfile) -> _ScriptedPlanningModel:
        created.append(profile)
        return model

    profile = _profile(provider_id="groq", model_id="configured-model")
    application = ResearchAgentApplication(
        tmp_path / "workspace",
        primary_planning_profile=profile,
        planning_model_factory_registry=PlanningModelFactoryRegistry(
            {"groq": create}
        ),
        planning_wire_mode=PlanningWireMode.V4,
    )

    result = application.run(
        replace(_request(source, "application-v4"), mode=RunMode.PLAN_ONLY)
    )

    assert isinstance(application.runtime.planner, LLMPlanner)
    assert application.runtime.planner.wire_mode is PlanningWireMode.V4
    assert application.runtime.planner.profile is profile
    assert created == [profile]
    assert result.status is ApplicationStatus.PLANNED
    assert result.run_status is RunStatus.PLANNED
    assert model.calls == 1


def test_application_optional_secondary_profile_uses_existing_failover(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    primary = _ScriptedPlanningModel(
        PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        PlanningModelError(code="PROVIDER_TIMEOUT", retry_after_seconds=0),
        model_id="primary:model",
    )
    secondary = _ScriptedPlanningModel(
        _planning_response(), model_id="secondary:model"
    )
    created: list[str] = []

    def primary_factory(profile: PlanningModelProfile) -> _ScriptedPlanningModel:
        created.append(profile.profile_id)
        return primary

    def secondary_factory(profile: PlanningModelProfile) -> _ScriptedPlanningModel:
        created.append(profile.profile_id)
        return secondary

    factories = PlanningModelFactoryRegistry(
        {"primary": primary_factory, "secondary": secondary_factory}
    )
    application = ResearchAgentApplication(
        tmp_path / "workspace",
        primary_planning_profile=_profile(
            provider_id="primary", model_id="primary/model"
        ),
        recovery_planning_profile=_profile(
            "secondary-profile", "secondary", "secondary/model"
        ),
        planning_model_factory_registry=factories,
    )

    result = application.run(
        replace(_request(source, "configured-failover"), mode=RunMode.PLAN_ONLY)
    )

    assert result.status is ApplicationStatus.PLANNED
    assert primary.calls == 2
    assert secondary.calls == 1
    assert created == ["primary-profile", "secondary-profile"]
    assert any(
        event.details.get("final_recovery_outcome") == "failover_recovered"
        for event in result.run_result.trace
    )


def test_missing_primary_profile_fails_before_durable_state_creation(
    tmp_path: Path,
) -> None:
    application = ResearchAgentApplication(tmp_path / "workspace")

    with pytest.raises(ApplicationServiceError) as raised:
        application.run(AgentRequest("missing-profile", "Inspect this data.", {}))

    assert raised.value.error.code == "PLANNING_MODEL_PROFILE_REQUIRED"
    assert tuple(application._workspace.run_state.iterdir()) == ()
    assert not isinstance(application.runtime.planner, DeterministicPlanner)


def test_explicit_planner_is_authoritative_and_conflicting_profiles_are_rejected(
    tmp_path: Path,
) -> None:
    planner = DeterministicPlanner()
    application = ResearchAgentApplication(tmp_path / "explicit", planner=planner)

    assert application.runtime.planner is planner
    with pytest.raises(ApplicationServiceError) as raised:
        ResearchAgentApplication(
            tmp_path / "conflict",
            planner=planner,
            primary_planning_profile=_profile(),
        )
    assert raised.value.error.code == "APP_PLANNING_CONFIGURATION_CONFLICT"
    with pytest.raises(ApplicationServiceError) as raised:
        ResearchAgentApplication(
            tmp_path / "wire-conflict",
            planner=planner,
            planning_wire_mode=PlanningWireMode.V4,
        )
    assert raised.value.error.code == "APP_PLANNING_CONFIGURATION_CONFLICT"


@pytest.mark.parametrize(
    ("profile", "factories", "code"),
    [
        (
            _profile(enabled=False),
            PlanningModelFactoryRegistry(
                {"custom": lambda _: _ScriptedPlanningModel(_planning_response())}
            ),
            "PLANNING_MODEL_PROFILE_DISABLED",
        ),
        (
            _profile(supports_structured_output=False),
            PlanningModelFactoryRegistry(
                {"custom": lambda _: _ScriptedPlanningModel(_planning_response())}
            ),
            "PLANNING_MODEL_CAPABILITY_UNSUPPORTED",
        ),
        (
            _profile(provider_id="unknown"),
            PlanningModelFactoryRegistry(
                {"custom": lambda _: _ScriptedPlanningModel(_planning_response())}
            ),
            "PLANNING_MODEL_PROVIDER_UNKNOWN",
        ),
    ],
)
def test_application_preserves_stable_profile_configuration_errors(
    tmp_path: Path,
    profile: PlanningModelProfile,
    factories: PlanningModelFactoryRegistry,
    code: str,
) -> None:
    with pytest.raises(ApplicationServiceError) as raised:
        ResearchAgentApplication(
            tmp_path / code,
            primary_planning_profile=profile,
            planning_model_factory_registry=factories,
        )

    assert raised.value.error.code == code


@pytest.mark.parametrize(
    "code",
    [
        "PLANNING_PROVIDER_DEPENDENCY_MISSING",
        "PLANNING_PROVIDER_CONFIGURATION_FAILED",
    ],
)
def test_application_preserves_sanitized_provider_construction_errors(
    tmp_path: Path, code: str
) -> None:
    secret = "provider-secret"

    def fail(_: PlanningModelProfile) -> _ScriptedPlanningModel:
        raise PlanningModelError(secret, code=code)

    with pytest.raises(ApplicationServiceError) as raised:
        ResearchAgentApplication(
            tmp_path / code,
            primary_planning_profile=_profile(),
            planning_model_factory_registry=PlanningModelFactoryRegistry(
                {"custom": fail}
            ),
        )

    assert raised.value.error.code == code
    assert secret not in str(raised.value)
    assert secret not in raised.value.error.message


def test_low_level_runtime_default_remains_deterministic() -> None:
    assert isinstance(AgentRuntime().planner, DeterministicPlanner)


def test_figureless_application_run_is_verified_json_safe_and_nonmutating(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    request = _request(source)

    result = application.run(request)

    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.run_status is RunStatus.SUCCEEDED
    assert result.evidence is not None
    assert result.visualization is None
    assert result.report is not None
    assert Path(result.report.path).name == "analysis_report.md"
    assert calls == ["inspect_scATAC"]
    assert "output_dir" not in request.inputs
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    persisted = application.run_store.load(result.run_id)
    assert persisted.request.inputs["output_dir"] == str(
        Path(result.workspace_path) / "scientific"
    )


def test_terminal_resume_reuses_verified_reporting_without_scientific_execution(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    first = application.run(_request(source, "resume-success"))
    evidence_bytes = Path(first.evidence.path).read_bytes()  # type: ignore[union-attr]
    report_bytes = Path(first.report.path).read_bytes()  # type: ignore[union-attr]

    lifecycle_application = ResearchAgentApplication(tmp_path / "workspace")
    second = lifecycle_application.resume(first.run_id)

    assert second.status is ApplicationStatus.SUCCEEDED
    assert calls == ["inspect_scATAC"]
    assert Path(second.evidence.path).read_bytes() == evidence_bytes  # type: ignore[union-attr]
    assert Path(second.report.path).read_bytes() == report_bytes  # type: ignore[union-attr]


def test_duplicate_run_preserves_run_store_semantics_and_requires_resume(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    application = _deterministic_application(tmp_path / "workspace")
    request = _request(source, "duplicate-run")
    first = application.run(request)

    with pytest.raises(RunAlreadyExistsError):
        application.run(request)

    assert application.resume(first.run_id).status is ApplicationStatus.SUCCEEDED


def test_missing_report_is_rebuilt_without_rerunning_science(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    first = application.run(_request(source, "rebuild-report"))
    report_bundle = Path(first.report.path).parent  # type: ignore[union-attr]
    shutil.rmtree(report_bundle)

    resumed = application.resume(first.run_id)

    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert resumed.report is not None and Path(resumed.report.path).is_file()
    assert calls == ["inspect_scATAC"]


def test_missing_evidence_is_rebuilt_without_rerunning_science(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    first = application.run(_request(source, "rebuild-evidence"))
    Path(first.evidence.path).unlink()  # type: ignore[union-attr]

    resumed = application.resume(first.run_id)

    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert resumed.evidence is not None and Path(resumed.evidence.path).is_file()
    assert calls == ["inspect_scATAC"]


def test_tampered_evidence_fails_closed_without_rerunning_science(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    first = application.run(_request(source, "tampered-evidence"))
    Path(first.evidence.path).write_text("{}", encoding="utf-8")  # type: ignore[union-attr]

    resumed = application.resume(first.run_id)

    assert resumed.status is ApplicationStatus.FAILED
    assert resumed.error is not None
    assert resumed.error.code == "APP_EVIDENCE_FAILED"
    assert resumed.error.stage is ApplicationStage.EVIDENCE
    assert calls == ["inspect_scATAC"]


def test_tampered_report_fails_closed_without_rerunning_science(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    first = application.run(_request(source, "tampered-report"))
    Path(first.report.path).write_text("tampered", encoding="utf-8")  # type: ignore[union-attr]

    resumed = application.resume(first.run_id)

    assert resumed.status is ApplicationStatus.FAILED
    assert resumed.run_status is RunStatus.SUCCEEDED
    assert resumed.error is not None
    assert resumed.error.code == "APP_REPORT_FAILED"
    assert resumed.error.stage is ApplicationStage.REPORT
    assert calls == ["inspect_scATAC"]


def test_plan_only_has_zero_execution_and_zero_reporting(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    application = _deterministic_application(
        tmp_path / "workspace", registry=_counting_registry(calls)
    )
    request = replace(_request(source, "plan-only"), mode=RunMode.PLAN_ONLY)

    result = application.run(request)

    assert result.status is ApplicationStatus.PLANNED
    assert result.run_status is RunStatus.PLANNED
    assert result.evidence is result.visualization is result.report is None
    assert calls == []


def test_planning_failure_preserves_runtime_error_and_skips_reporting(
    tmp_path: Path,
) -> None:
    application = _deterministic_application(tmp_path / "workspace")
    request = AgentRequest("unsupported", "Write a poem.", {})

    result = application.run(request)

    assert result.status is ApplicationStatus.FAILED
    assert result.run_result.errors
    assert result.error is None
    assert result.evidence is result.visualization is result.report is None


def test_execution_failure_and_terminal_resume_preserve_sanitized_runtime_error(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    secret = "secret-tool-detail"
    default = build_default_tool_registry()

    def fail(**_: object) -> object:
        raise RuntimeError(secret)

    registry = ToolRegistry(
        tuple(
            replace(spec, function=fail)
            if spec.name == "inspect_scATAC"
            else spec
            for spec in (default.get(name) for name in default.names())
        )
    )
    application = _deterministic_application(
        tmp_path / "workspace", registry=registry
    )

    first = application.run(_request(source, "execution-failure"))
    resumed = application.resume(first.run_id)

    assert first.status is resumed.status is ApplicationStatus.FAILED
    assert first.error is resumed.error is None
    assert first.evidence is resumed.evidence is None
    assert secret not in json.dumps(first.to_dict())
    assert resumed.run_result == first.run_result


def test_preflight_failure_executes_no_tool_and_creates_no_reporting(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class InvalidPlanner:
        def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
            return AgentPlan(
                "invalid-plan",
                request.request_id,
                "invalid-test-planner",
                (PlanStep("unknown", "arbitrary_python", {}),),
            )

    registry = _counting_registry(calls)
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=InvalidPlanner(), registry=registry
    )

    result = application.run(AgentRequest("preflight-failure", "Invalid plan.", {}))

    assert result.status is ApplicationStatus.FAILED
    assert result.run_result.errors[0].code == "UNKNOWN_TOOL"
    assert result.evidence is result.visualization is result.report is None
    assert calls == []


@pytest.mark.parametrize(
    "inputs",
    [
        {"input_path": "/data/input.h5ad", "output_dir": "/tmp/untrusted"},
        {"input_path": "/data/input.h5ad", "report_output_dir": "/tmp/untrusted"},
        {"input_path": "/data/input.h5ad", "overwrite": True},
    ],
)
def test_reserved_or_overwriting_request_inputs_are_rejected(
    tmp_path: Path, inputs: dict[str, object]
) -> None:
    application = _deterministic_application(tmp_path / "workspace")

    with pytest.raises(ApplicationServiceError) as caught:
        application.run(AgentRequest("invalid", "Inspect this dataset.", inputs))

    assert caught.value.error.code == "APP_REQUEST_INVALID"


def test_incomplete_evidence_destination_fails_as_output_conflict(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    application = _deterministic_application(tmp_path / "workspace")
    run = application._workspace.run_paths("conflict:run")
    (run.evidence / "partial.tmp").write_text("partial", encoding="utf-8")

    result = application.run(_request(source, "conflict"))

    assert result.status is ApplicationStatus.FAILED
    assert result.error is not None
    assert result.error.code == "APP_OUTPUT_CONFLICT"


def test_active_composition_lock_fails_safely(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    application = _deterministic_application(tmp_path / "workspace")
    first = application.run(_request(source, "active-composer"))
    run = application._workspace.run_paths(first.run_id)

    with application._workspace.composition_lease(run):
        second = application.resume(first.run_id)

    assert second.status is ApplicationStatus.FAILED
    assert second.error is not None
    assert second.error.code == "APP_COMPOSITION_ACTIVE"


def test_cancel_delegates_to_runtime_receipt(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    producer = _deterministic_application(tmp_path / "workspace")
    completed = producer.run(_request(source, "terminal-cancel"))
    application = ResearchAgentApplication(tmp_path / "workspace")

    receipt = application.cancel(completed.run_id)

    assert receipt.run_id == completed.run_id
    assert receipt.disposition.value == "ALREADY_TERMINAL"


def test_cooperative_runtime_cancellation_produces_no_reporting(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    default = build_default_tool_registry()
    inspection = default.get("inspect_scATAC")
    started = threading.Event()
    release = threading.Event()

    def blocking(**arguments: object) -> object:
        started.set()
        assert release.wait(timeout=10)
        return inspection.function(**arguments)

    registry = ToolRegistry(
        tuple(
            replace(spec, function=blocking)
            if spec.name == "inspect_scATAC"
            else spec
            for spec in (default.get(name) for name in default.names())
        )
    )
    application = _deterministic_application(
        tmp_path / "workspace", registry=registry
    )
    request = _request(source, "cancel-running")
    holder: dict[str, object] = {}

    def execute() -> None:
        holder["result"] = application.run(request)

    thread = threading.Thread(target=execute)
    thread.start()
    assert started.wait(timeout=10)
    receipt = application.cancel("cancel-running:run")
    release.set()
    thread.join(timeout=20)

    assert not thread.is_alive()
    assert receipt.disposition.value == "REQUESTED"
    result = holder["result"]
    assert result.status is ApplicationStatus.CANCELLED  # type: ignore[attr-defined]
    assert (
        result.evidence is result.visualization is result.report is None  # type: ignore[attr-defined]
    )


def test_interrupted_durable_run_resumes_without_replanning_completed_step(
    tmp_path: Path,
) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    calls: list[str] = []
    registry = _counting_registry(calls)
    request_id = "interrupted-application"

    class TwoStepPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
            self.calls += 1
            return AgentPlan(
                f"{request.request_id}:two-step",
                request.request_id,
                "two-step-test-planner",
                (
                    PlanStep("inspect-1", "inspect_scATAC", {"path": str(source)}),
                    PlanStep("inspect-2", "inspect_scATAC", {"path": str(source)}),
                ),
            )

    planner = TwoStepPlanner()
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=planner, registry=registry
    )
    original_update = application.run_store.update

    def interrupt(state, *, expected_revision: int):
        persisted = original_update(state, expected_revision=expected_revision)
        if tuple(step.status for step in persisted.steps) == (
            StepStatus.SUCCEEDED,
            StepStatus.PENDING,
        ):
            raise KeyboardInterrupt("simulated application interruption")
        return persisted

    application.run_store.update = interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        application.run(
            AgentRequest(request_id, "Inspect twice for resume testing.", {})
        )
    application.run_store.update = original_update  # type: ignore[method-assign]

    resumed = application.resume(f"{request_id}:run")

    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert calls == ["inspect_scATAC", "inspect_scATAC"]
    assert planner.calls == 1
