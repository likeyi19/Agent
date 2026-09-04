"""Offline semantic wire-v4 recovery and diagnostic integration tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    PlanningModelError,
    PlanningModelProfile,
    PlanningWireMode,
    RunMode,
    RunStatus,
    SemanticProducerPortSpec,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.providers import PlanningModelFactoryRegistry
from benchmarks.planner.benchmark import load_cases


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"


class ScriptedSemanticModel:
    model_id = "scripted-semantic-recovery-model"

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, object]] = []

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls.append((prompt, response_schema))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def _input(target: str, input_name: str) -> dict[str, str]:
    return {"kind": "input", "target": target, "input": input_name}


def _step_source(target: str, step_id: str) -> dict[str, str]:
    return {"kind": "step", "target": target, "step": step_id}


def _step_port_source(
    target: str, step_id: str, source_port: str
) -> dict[str, str]:
    return {
        "kind": "step_port",
        "target": target,
        "step": step_id,
        "source_port": source_port,
    }


def _step(
    step_id: str,
    tool: str,
    *,
    sources: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "step_id": step_id,
        "tool": tool,
        "sources": [] if sources is None else sources,
        "control_dependencies": [],
    }


def _payload(*steps: dict[str, object]) -> str:
    return json.dumps(
        {
            "schema_version": 4,
            "decision": {"kind": "plan", "steps": list(steps)},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _unsupported() -> str:
    return json.dumps(
        {
            "schema_version": 4,
            "decision": {
                "kind": "unsupported",
                "reason": "RAW-UNSUPPORTED-REASON",
            },
        }
    )


def _inspect_steps(
    *, input_name: str = "input_path"
) -> tuple[dict[str, object], ...]:
    return (
        _step(
            "inspect",
            "inspect_scATAC",
            sources=[_input("dataset", input_name)],
        ),
    )


def _downstream_steps(
    *,
    missing_embedding: bool = False,
    explicit_cluster_port: bool = False,
) -> tuple[dict[str, object], ...]:
    neighbor_sources = (
        []
        if missing_embedding
        else [_step_source("embedding", "embed")]
    )
    umap_source = (
        _step_port_source("analysis", "cluster", "clustered_analysis")
        if explicit_cluster_port
        else _step_source("analysis", "cluster")
    )
    return (
        _step("inspect", "inspect_scATAC"),
        _step(
            "embed",
            "epizoo_embed_cells",
            sources=[_step_source("dataset", "inspect")],
        ),
        _step(
            "neighbors",
            "build_cell_neighbors",
            sources=neighbor_sources,
        ),
        _step(
            "cluster",
            "cluster_cells",
            sources=[_step_source("analysis", "neighbors")],
        ),
        _step(
            "umap",
            "compute_cell_umap",
            sources=[umap_source],
        ),
    )


def _transfer_steps(*, swapped: bool = False) -> tuple[dict[str, object], ...]:
    reference_embedding = "query_embed" if swapped else "reference_embed"
    query_embedding = "reference_embed" if swapped else "query_embed"
    return (
        _step(
            "reference_inspect",
            "inspect_scATAC",
            sources=[_input("dataset", "reference_input_path")],
        ),
        _step(
            "reference_embed",
            "epizoo_embed_cells",
            sources=[_step_source("dataset", "reference_inspect")],
        ),
        _step(
            "query_inspect",
            "inspect_scATAC",
            sources=[_input("dataset", "query_input_path")],
        ),
        _step(
            "query_embed",
            "epizoo_embed_cells",
            sources=[_step_source("dataset", "query_inspect")],
        ),
        _step(
            "transfer",
            "transfer_cell_labels",
            sources=[
                _step_source("reference_dataset", "reference_inspect"),
                _step_source("reference_embedding", reference_embedding),
                _step_source("query_dataset", "query_inspect"),
                _step_source("query_embedding", query_embedding),
            ],
        ),
    )


def _profile(
    profile_id: str = "primary-semantic",
    provider_id: str = "primary",
    model_id: str = "organization/primary-semantic-model",
) -> PlanningModelProfile:
    return PlanningModelProfile(profile_id, provider_id, model_id)


def _guarded_registry(source: ToolRegistry | None = None) -> tuple[ToolRegistry, Mock]:
    source = build_default_tool_registry() if source is None else source
    guard = Mock(side_effect=AssertionError("PLAN_ONLY executed a scientific tool"))
    return (
        ToolRegistry(
            tuple(
                replace(source.get(name), function=guard)
                for name in source.names()
            )
        ),
        guard,
    )


def _run(
    model: ScriptedSemanticModel,
    request: AgentRequest,
    *,
    registry: ToolRegistry | None = None,
    recovery_profiles: tuple[PlanningModelProfile, ...] = (),
    model_factory_registry: PlanningModelFactoryRegistry | None = None,
):
    guarded, guard = _guarded_registry(registry)
    planner = LLMPlanner(
        model,
        wire_mode=PlanningWireMode.V4,
        profile=_profile(),
        retry_sleeper=lambda _: None,
        recovery_profiles=recovery_profiles,
        model_factory_registry=model_factory_registry,
    )
    result = AgentRuntime(planner=planner, registry=guarded).run(request)
    guard.assert_not_called()
    return result


def _diagnostics(result) -> list[dict[str, object]]:
    return [
        dict(event.details)
        for event in result.trace
        if event.details.get("diagnostic_schema_version") == 4
    ]


def _failure(result, code: str) -> dict[str, object]:
    return next(item for item in _diagnostics(result) if item["code"] == code)


def _inspect_request(
    request_id: str,
    *,
    extra_inputs: dict[str, object] | None = None,
) -> AgentRequest:
    inputs = {"input_path": "/private/INPUT-PATH-SECRET/input.h5ad"}
    if extra_inputs is not None:
        inputs.update(extra_inputs)
    return AgentRequest(
        request_id,
        "Inspect the supplied scATAC dataset.",
        inputs,
        RunMode.PLAN_ONLY,
    )


def _downstream_request(request_id: str) -> AgentRequest:
    return AgentRequest(
        request_id,
        "Embed and run neighbors, clustering, and UMAP.",
        {
            "input_path": "/private/input.h5ad",
            "output_dir": "/private/output",
            "species": "mouse",
        },
        RunMode.PLAN_ONLY,
    )


def _transfer_request(request_id: str) -> AgentRequest:
    case = next(
        case
        for case in load_cases(CASES_PATH)
        if case.case_id == "label_transfer_canonical"
    )
    return AgentRequest(
        request_id, case.prompt, case.inputs, RunMode.PLAN_ONLY
    )


def _ambiguous_cluster_registry() -> ToolRegistry:
    source = build_default_tool_registry()
    cluster = source.get("cluster_cells")
    semantic = cluster.semantic_planning
    assert semantic is not None
    primary = next(
        port for port in semantic.producer_ports if port.name == "clustered_analysis"
    )
    ambiguous = replace(
        cluster,
        semantic_planning=replace(
            semantic,
            producer_ports=(
                primary,
                SemanticProducerPortSpec(
                    "alternate_analysis",
                    primary.semantic_type,
                    primary.members,
                    primary.lineage_from_port,
                ),
            ),
        ),
    )
    return ToolRegistry(
        tuple(
            ambiguous if name == "cluster_cells" else source.get(name)
            for name in source.names()
        )
    )


def test_parse_failure_repairs_once_with_unchanged_semantic_context() -> None:
    raw = "not-json RAW-FAILED-V4-RESPONSE"
    model = ScriptedSemanticModel([raw, _payload(*_inspect_steps())])
    result = _run(model, _inspect_request("v4-parse-repair"))

    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
    initial = json.loads(model.calls[0][0])
    repaired = json.loads(model.calls[1][0])
    assert model.calls[0][1] == model.calls[1][1]
    for key in ("user_request", "catalog", "wire_v4", "instructions"):
        assert initial[key] == repaired[key]
    assert "repair" not in initial
    diagnostic = repaired["repair"]["diagnostic"]
    assert diagnostic["previous_failure_stage"] == "parse"
    assert diagnostic["reason_code"] == "malformed_json"
    assert diagnostic["required_correction"] == "regenerate_strict_wire_v4_json"
    assert raw not in json.dumps(result.to_dict())
    assert all(
        item["diagnostic_schema_version"] == 4 for item in _diagnostics(result)
    )
    assert _diagnostics(result)[-1]["total_provider_call_count"] == 2


def test_unauthorized_request_source_repairs_with_semantic_identifiers() -> None:
    request = _inspect_request(
        "v4-request-source-repair",
        extra_inputs={"output_dir": "/private/OUTPUT-DIR-SECRET"},
    )
    invalid = _payload(*_inspect_steps(input_name="output_dir"))
    model = ScriptedSemanticModel([invalid, _payload(*_inspect_steps())])

    result = _run(model, request)
    failure = _failure(result, "UNAUTHORIZED_REQUEST_INPUT")
    repair = json.loads(model.calls[1][0])["repair"]["diagnostic"]

    assert result.status is RunStatus.PLANNED
    assert failure["stage"] == "argument_binding"
    assert failure["reason_code"] == "unauthorized_request_source"
    assert failure["step_id"] == "inspect"
    assert failure["target_port"] == "dataset"
    assert failure["input_name"] == "output_dir"
    assert repair["required_correction"] == "select_authorized_request_source"


def test_missing_required_source_repairs_at_semantic_target_port() -> None:
    model = ScriptedSemanticModel(
        [
            _payload(*_downstream_steps(missing_embedding=True)),
            _payload(*_downstream_steps()),
        ]
    )
    result = _run(model, _downstream_request("v4-missing-source-repair"))
    failure = _failure(result, "MISSING_REQUIRED_SOURCE")
    repair = json.loads(model.calls[1][0])["repair"]["diagnostic"]

    assert result.status is RunStatus.PLANNED
    assert failure["stage"] == "argument_binding"
    assert failure["step_id"] == "neighbors"
    assert failure["target_port"] == "embedding"
    assert failure["reason_code"] == "missing_semantic_source"
    assert repair["required_correction"] == "select_required_semantic_source"


def test_ambiguous_source_port_repairs_with_explicit_producer_port() -> None:
    registry = _ambiguous_cluster_registry()
    model = ScriptedSemanticModel(
        [
            _payload(*_downstream_steps()),
            _payload(*_downstream_steps(explicit_cluster_port=True)),
        ]
    )
    result = _run(
        model,
        _downstream_request("v4-source-port-repair"),
        registry=registry,
    )
    failure = _failure(result, "AMBIGUOUS_SOURCE_PORT")
    repair = json.loads(model.calls[1][0])["repair"]["diagnostic"]

    assert result.status is RunStatus.PLANNED
    assert failure["stage"] == "dependency_reference"
    assert failure["step_id"] == "umap"
    assert failure["producer_step_id"] == "cluster"
    assert failure["target_port"] == "analysis"
    assert "source_port" not in failure
    assert repair["required_correction"] == "select_source_port_explicitly"


def test_branch_lineage_mismatch_repairs_without_executor_details() -> None:
    model = ScriptedSemanticModel(
        [
            _payload(*_transfer_steps(swapped=True)),
            _payload(*_transfer_steps()),
        ]
    )
    result = _run(model, _transfer_request("v4-lineage-repair"))
    failure = _failure(result, "BROKEN_BRANCH_LINEAGE")
    repair = json.loads(model.calls[1][0])["repair"]["diagnostic"]

    assert result.status is RunStatus.PLANNED
    assert failure["stage"] == "dependency_reference"
    assert failure["reason_code"] == "branch_lineage_mismatch"
    assert failure["step_id"] == "transfer"
    assert failure["target_port"] == "reference_embedding"
    assert failure["producer_step_id"] == "query_embed"
    assert repair["required_correction"] == "select_producer_with_required_lineage"
    assert "argument_name" not in repair
    assert "output_key" not in repair


def test_unsupported_and_request_too_large_remain_terminal() -> None:
    unsupported = ScriptedSemanticModel([_unsupported(), _payload(*_inspect_steps())])
    unsupported_result = _run(
        unsupported, _inspect_request("v4-unsupported-terminal")
    )
    too_large = ScriptedSemanticModel(
        [
            PlanningModelError(code="PROVIDER_REQUEST_TOO_LARGE"),
            _payload(*_inspect_steps()),
        ]
    )
    too_large_result = _run(too_large, _inspect_request("v4-413-terminal"))

    assert unsupported_result.status is RunStatus.FAILED
    assert unsupported_result.errors[0].code == "UNSUPPORTED_REQUEST"
    assert len(unsupported.calls) == 1
    assert "RAW-UNSUPPORTED-REASON" not in json.dumps(
        unsupported_result.to_dict()
    )
    assert too_large_result.status is RunStatus.FAILED
    assert too_large_result.errors[0].code == "PROVIDER_REQUEST_TOO_LARGE"
    assert len(too_large.calls) == 1


def test_transport_retry_stays_transport_only_and_reuses_exact_context() -> None:
    model = ScriptedSemanticModel(
        [
            PlanningModelError(code="PROVIDER_TIMEOUT"),
            _payload(*_inspect_steps()),
        ]
    )
    result = _run(model, _inspect_request("v4-transport-retry"))
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
    assert model.calls[0] == model.calls[1]
    assert not any(item["attempt_kind"] == "repair" for item in diagnostics)
    assert diagnostics[-1]["retry_used"] is True
    assert diagnostics[-1]["repair_used"] is False


def _factory(
    profile: PlanningModelProfile,
    model: ScriptedSemanticModel,
) -> PlanningModelFactoryRegistry:
    def create(received: PlanningModelProfile):
        assert received == profile
        return model

    return PlanningModelFactoryRegistry({profile.provider_id: create})


def test_semantic_failure_repairs_then_fails_over_in_v4() -> None:
    raw = "not-json RAW-PRIMARY-CANDIDATE"
    primary = ScriptedSemanticModel([raw, raw])
    secondary = ScriptedSemanticModel([_payload(*_inspect_steps())])
    secondary_profile = _profile(
        "secondary-semantic",
        "secondary",
        "organization/secondary-semantic-model",
    )

    result = _run(
        primary,
        _inspect_request("v4-failover"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory(secondary_profile, secondary),
    )
    diagnostics = _diagnostics(result)
    secondary_prompt = json.loads(secondary.calls[0][0])

    assert result.status is RunStatus.PLANNED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert primary.calls[0][1] == secondary.calls[0][1]
    assert secondary.calls[0][1]["properties"]["schema_version"]["enum"] == (4,)
    assert "failover" in secondary_prompt
    assert "repair" not in secondary_prompt
    assert secondary_prompt["failover"]["diagnostic"][
        "required_correction"
    ] == "regenerate_strict_wire_v4_json"
    assert diagnostics[-1]["total_provider_call_count"] == 3
    assert diagnostics[-1]["failover_used"] is True
    assert result.plan is not None
    assert result.plan.planner_name.endswith(":wire-v4")


def test_failover_is_final_and_v4_never_calls_a_fourth_time() -> None:
    primary = ScriptedSemanticModel(["bad-1", "bad-2"])
    secondary = ScriptedSemanticModel(
        ["bad-3", _payload(*_inspect_steps())]
    )
    secondary_profile = _profile(
        "secondary-ceiling",
        "secondary",
        "organization/secondary-ceiling-model",
    )

    result = _run(
        primary,
        _inspect_request("v4-call-ceiling"),
        recovery_profiles=(secondary_profile,),
        model_factory_registry=_factory(secondary_profile, secondary),
    )
    diagnostics = _diagnostics(result)

    assert result.status is RunStatus.FAILED
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 1
    assert diagnostics[-1]["total_provider_call_count"] == 3
    assert diagnostics[-1]["final_recovery_outcome"] == "failover_failed"


def test_incomplete_local_semantic_authority_is_terminal_before_provider() -> None:
    source = build_default_tool_registry()
    incomplete = ToolRegistry(
        (replace(source.get("inspect_scATAC"), semantic_planning=None),)
    )
    model = ScriptedSemanticModel([_payload(*_inspect_steps())])

    result = _run(
        model,
        _inspect_request("v4-local-authority-terminal"),
        registry=incomplete,
    )

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "PLANNER_CATALOG_INVALID"
    assert model.calls == []


def test_semantic_repair_diagnostics_exclude_all_structured_input_values() -> None:
    sentinels: dict[str, object] = {
        "sentinel_label": "LABEL-VALUE-SECRET",
        "sentinel_condition": "CONDITION-VALUE-SECRET",
        "sentinel_checkpoint": "/private/CHECKPOINT-VALUE-SECRET.pt",
        "sentinel_output_dir": "/private/OUTPUT-VALUE-SECRET",
        "sentinel_array": ["ARRAY-VALUE-SECRET", 17],
        "sentinel_nested": {"nested": "NESTED-VALUE-SECRET"},
    }
    request = _inspect_request("v4-private-repair", extra_inputs=sentinels)
    raw = "not-json RAW-PROVIDER-RESPONSE-SECRET"
    model = ScriptedSemanticModel([raw, _payload(*_inspect_steps())])

    result = _run(model, request)
    rendered_diagnostics = json.dumps(_diagnostics(result), sort_keys=True)
    rendered_repair = json.dumps(
        json.loads(model.calls[1][0])["repair"], sort_keys=True
    )

    forbidden = (
        "/private/INPUT-PATH-SECRET/input.h5ad",
        "LABEL-VALUE-SECRET",
        "CONDITION-VALUE-SECRET",
        "CHECKPOINT-VALUE-SECRET",
        "OUTPUT-VALUE-SECRET",
        "ARRAY-VALUE-SECRET",
        "NESTED-VALUE-SECRET",
        "RAW-PROVIDER-RESPONSE-SECRET",
        "<Mock",
        "side_effect",
    )
    for value in forbidden:
        assert value not in rendered_diagnostics
        assert value not in rendered_repair
    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
