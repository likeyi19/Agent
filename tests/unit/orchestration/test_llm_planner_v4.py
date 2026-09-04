"""Opt-in semantic wire-v4 composition through the production LLMPlanner."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    ErrorCategory,
    LLMPlanner,
    PlannerError,
    PlanningWireMode,
    RunMode,
    RunStatus,
    SemanticProducerPortSpec,
    ToolRegistry,
    build_default_tool_registry,
    build_semantic_planning_prompt,
    build_semantic_wire_v4_schema,
)
from agent.orchestration.llm_planner import _build_prompt, _response_schema
from agent.orchestration.planning_diagnostics import PlanningDiagnosticStage
from benchmarks.planner.benchmark import load_cases, oracle_response


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"


class CapturingPlanningModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, object]] = []

    @property
    def model_id(self) -> str:
        return "semantic-integration-model"

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls.append((prompt, response_schema))
        return self.response  # type: ignore[return-value]


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _case(case_id: str):
    return next(case for case in load_cases(CASES_PATH) if case.case_id == case_id)


def _request(
    case_id: str,
    *,
    inputs: dict[str, object] | None = None,
    mode: RunMode = RunMode.EXECUTE,
) -> AgentRequest:
    case = _case(case_id)
    return AgentRequest(
        f"{case_id}-integration",
        case.prompt,
        case.inputs if inputs is None else inputs,
        mode,
    )


def _input(target: str, input_name: str) -> dict[str, str]:
    return {"kind": "input", "target": target, "input": input_name}


def _step_source(target: str, step_id: str) -> dict[str, str]:
    return {"kind": "step", "target": target, "step": step_id}


def _step(
    step_id: str,
    tool: str,
    *,
    sources: list[dict[str, str]] | None = None,
    control_dependencies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "step_id": step_id,
        "tool": tool,
        "sources": [] if sources is None else sources,
        "control_dependencies": (
            [] if control_dependencies is None else control_dependencies
        ),
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


def _inspection_steps() -> tuple[dict[str, object], ...]:
    return (_step("provider-step-101", "inspect_scATAC"),)


def _downstream_steps() -> tuple[dict[str, object], ...]:
    return (
        _step("provider-step-101", "inspect_scATAC"),
        _step(
            "provider-step-102",
            "epizoo_embed_cells",
            sources=[_step_source("dataset", "provider-step-101")],
        ),
        _step(
            "provider-step-103",
            "build_cell_neighbors",
            sources=[_step_source("embedding", "provider-step-102")],
        ),
        _step(
            "provider-step-104",
            "cluster_cells",
            sources=[_step_source("analysis", "provider-step-103")],
        ),
        _step(
            "provider-step-105",
            "compute_cell_umap",
            sources=[_step_source("analysis", "provider-step-104")],
        ),
    )


def _transfer_steps(*, optional: bool = False) -> tuple[dict[str, object], ...]:
    reference_embedding = [_step_source("dataset", "provider-step-101")]
    query_embedding = [_step_source("dataset", "provider-step-103")]
    transfer_sources = [
        _step_source("reference_dataset", "provider-step-101"),
        _step_source("reference_embedding", "provider-step-102"),
        _step_source("query_dataset", "provider-step-103"),
        _step_source("query_embedding", "provider-step-104"),
    ]
    if optional:
        reference_embedding.append(_input("checkpoint", "checkpoint_path"))
        query_embedding.append(_input("checkpoint", "checkpoint_path"))
        transfer_sources.append(_input("overwrite", "overwrite"))
    return (
        _step(
            "provider-step-101",
            "inspect_scATAC",
            sources=[_input("dataset", "reference_input_path")],
        ),
        _step(
            "provider-step-102",
            "epizoo_embed_cells",
            sources=reference_embedding,
        ),
        _step(
            "provider-step-103",
            "inspect_scATAC",
            sources=[_input("dataset", "query_input_path")],
        ),
        _step(
            "provider-step-104",
            "epizoo_embed_cells",
            sources=query_embedding,
        ),
        _step(
            "provider-step-105",
            "transfer_cell_labels",
            sources=transfer_sources,
        ),
    )


def _paired_da_steps() -> tuple[dict[str, object], ...]:
    return (
        _step(
            "provider-step-101",
            "validate_scATAC_feature_space",
            sources=[_input("overwrite", "overwrite")],
        ),
        _step(
            "provider-step-102",
            "build_replicate_pseudobulk",
            sources=[
                _step_source("feature_space", "provider-step-101"),
                _input("overwrite", "overwrite"),
            ],
        ),
        _step(
            "provider-step-103",
            "run_replicate_differential_accessibility",
            sources=[
                _step_source("pseudobulk", "provider-step-102"),
                _input("overwrite", "overwrite"),
            ],
        ),
    )


V4_CASES = {
    "inspect_canonical": _inspection_steps,
    "downstream_canonical": _downstream_steps,
    "label_transfer_canonical": _transfer_steps,
    "label_transfer_optional": lambda: _transfer_steps(optional=True),
    "differential_accessibility_paired_covariates": _paired_da_steps,
}


def _execution_signature(plan) -> tuple[object, ...]:
    return tuple(
        (
            step.step_id,
            step.tool_name,
            dict(step.arguments),
            step.depends_on,
        )
        for step in plan.steps
    )


def _contains_callable(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, dict):
        return any(_contains_callable(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_callable(item) for item in value)
    return False


@pytest.mark.parametrize("case_id", tuple(V4_CASES))
def test_v4_compiles_to_accepted_v3_execution_semantics(
    registry: ToolRegistry,
    case_id: str,
) -> None:
    request = _request(case_id)
    v4_model = CapturingPlanningModel(_payload(*V4_CASES[case_id]()))
    v3_model = CapturingPlanningModel(oracle_response(_case(case_id)))

    v4_plan = LLMPlanner(
        v4_model, wire_mode=PlanningWireMode.V4
    ).plan(request, registry)
    v3_plan = LLMPlanner(v3_model).plan(request, registry)

    assert _execution_signature(v4_plan) == _execution_signature(v3_plan)
    assert v4_plan.planner_name == (
        "llm:semantic-integration-model:wire-v4"
    )
    assert v4_plan.plan_id.startswith(f"{request.request_id}:semantic:")
    assert v3_plan.planner_name == "llm:semantic-integration-model"
    assert v3_plan.plan_id.startswith(f"{request.request_id}:llm:")


def test_v4_sends_only_semantic_prompt_and_schema(
    registry: ToolRegistry,
) -> None:
    secret_path = "/SENTINEL/private/input.h5ad"
    request = _request(
        "inspect_canonical", inputs={"input_path": secret_path}
    )
    model = CapturingPlanningModel(_payload(*_inspection_steps()))
    planner = LLMPlanner(model, wire_mode=PlanningWireMode.V4)

    attempt = planner.plan_with_diagnostics(request, registry)

    assert len(model.calls) == 1
    prompt, schema = model.calls[0]
    assert prompt == build_semantic_planning_prompt(request, registry)
    assert schema == build_semantic_wire_v4_schema(registry, request)
    assert prompt != _build_prompt(request, registry)
    assert schema != _response_schema(registry, request)
    assert secret_path not in prompt
    assert secret_path not in json.dumps(schema)
    assert schema["$defs"]["input_source"]["properties"]["input"][
        "enum"
    ] == ("input_path",)
    assert schema["$defs"]["step"]["properties"]["tool"]["enum"] == (
        tuple(sorted(registry.names()))
    )
    assert not _contains_callable(json.loads(prompt))
    assert not _contains_callable(schema)
    assert attempt.context.planning_wire_schema_version == 4


def test_default_and_explicit_v3_are_byte_compatible(
    registry: ToolRegistry,
) -> None:
    request = _request("inspect_canonical")
    response = oracle_response(_case("inspect_canonical"))
    default_model = CapturingPlanningModel(response)
    explicit_model = CapturingPlanningModel(response)

    default_plan = LLMPlanner(default_model).plan(request, registry)
    explicit_plan = LLMPlanner(
        explicit_model, wire_mode=PlanningWireMode.V3
    ).plan(request, registry)

    assert default_model.calls == explicit_model.calls
    assert default_model.calls[0] == (
        _build_prompt(request, registry),
        _response_schema(registry, request),
    )
    assert default_plan == explicit_plan
    assert default_plan.planner_name == "llm:semantic-integration-model"


@pytest.mark.parametrize("wire_mode", ("v2", "v3", "v4", 3, 4, None))
def test_invalid_or_untyped_wire_mode_fails_at_configuration(
    wire_mode: object,
) -> None:
    with pytest.raises(TypeError, match="PlanningWireMode"):
        LLMPlanner(
            CapturingPlanningModel("unused"),
            wire_mode=wire_mode,  # type: ignore[arg-type]
        )


def test_wire_mode_is_never_auto_detected(registry: ToolRegistry) -> None:
    request = _request("inspect_canonical")

    with pytest.raises(PlannerError):
        LLMPlanner(
            CapturingPlanningModel(oracle_response(_case("inspect_canonical"))),
            wire_mode=PlanningWireMode.V4,
        ).plan(request, registry)
    with pytest.raises(PlannerError):
        LLMPlanner(
            CapturingPlanningModel(_payload(*_inspection_steps())),
            wire_mode=PlanningWireMode.V3,
        ).plan(request, registry)


@pytest.mark.parametrize(
    ("response", "code"),
    (
        ("not-json", "PLANNER_OUTPUT_INVALID"),
        (
            json.dumps(
                {
                    "schema_version": 3,
                    "decision": {"kind": "plan", "steps": []},
                }
            ),
            "PLANNER_OUTPUT_INVALID",
        ),
        (
            _payload(_step("inspect", "not_a_registered_tool")),
            "PLANNER_OUTPUT_INVALID",
        ),
    ),
)
def test_v4_wire_failures_stop_at_planner_boundary(
    registry: ToolRegistry,
    response: str,
    code: str,
) -> None:
    model = CapturingPlanningModel(response)

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(model, wire_mode=PlanningWireMode.V4).plan(
            _request("inspect_canonical"), registry
        )

    assert caught.value.code == code
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    ("steps", "inputs", "code"),
    (
        (
            (
                _step(
                    "inspect",
                    "inspect_scATAC",
                    sources=[_input("dataset", "output_dir")],
                ),
            ),
            {
                "input_path": "/data/input.h5ad",
                "output_dir": "/output",
            },
            "UNAUTHORIZED_REQUEST_INPUT",
        ),
        (
            (
                _step(
                    "inspect",
                    "inspect_scATAC",
                    sources=[_input("unknown_port", "input_path")],
                ),
            ),
            {"input_path": "/data/input.h5ad"},
            "UNKNOWN_TARGET_PORT",
        ),
        (
            (_step("inspect", "inspect_scATAC"),),
            {},
            "MISSING_REQUIRED_SOURCE",
        ),
    ),
)
def test_v4_semantic_compiler_failures_are_stable_planner_errors(
    registry: ToolRegistry,
    steps: tuple[dict[str, object], ...],
    inputs: dict[str, object],
    code: str,
) -> None:
    model = CapturingPlanningModel(_payload(*steps))
    request = AgentRequest("negative-semantic", "Plan safely.", inputs)

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(model, wire_mode=PlanningWireMode.V4).plan(request, registry)

    assert caught.value.code == code
    assert caught.value.diagnostics[-1].stage is PlanningDiagnosticStage.CANDIDATE


def test_v4_ambiguous_producer_port_fails_closed(
    registry: ToolRegistry,
) -> None:
    cluster = registry.get("cluster_cells")
    semantic = cluster.semantic_planning
    assert semantic is not None
    primary = next(
        port for port in semantic.producer_ports if port.name == "clustered_analysis"
    )
    ambiguous_cluster = replace(
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
    custom_registry = ToolRegistry(
        (ambiguous_cluster, registry.get("compute_cell_umap"))
    )
    response = _payload(
        _step("cluster", "cluster_cells"),
        _step(
            "umap",
            "compute_cell_umap",
            sources=[_step_source("analysis", "cluster")],
        ),
    )

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(
            CapturingPlanningModel(response),
            wire_mode=PlanningWireMode.V4,
        ).plan(AgentRequest("ambiguous", "Plan.", {}), custom_registry)

    assert caught.value.code == "AMBIGUOUS_SOURCE_PORT"


def test_v4_reference_query_lineage_swap_fails_closed(
    registry: ToolRegistry,
) -> None:
    steps = list(_transfer_steps())
    transfer = dict(steps[-1])
    transfer["sources"] = [
        _step_source("reference_dataset", "provider-step-101"),
        _step_source("reference_embedding", "provider-step-104"),
        _step_source("query_dataset", "provider-step-103"),
        _step_source("query_embedding", "provider-step-102"),
    ]
    steps[-1] = transfer

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(
            CapturingPlanningModel(_payload(*steps)),
            wire_mode=PlanningWireMode.V4,
        ).plan(_request("label_transfer_canonical"), registry)

    assert caught.value.code == "BROKEN_BRANCH_LINEAGE"


def test_v4_unrepresented_scientific_parameter_fails_closed(
    registry: ToolRegistry,
) -> None:
    case = _case("downstream_canonical")
    inputs = {**case.inputs, "n_neighbors": 17}

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(
            CapturingPlanningModel(_payload(*_downstream_steps())),
            wire_mode=PlanningWireMode.V4,
        ).plan(_request("downstream_canonical", inputs=inputs), registry)

    assert caught.value.code == "UNAUTHORIZED_REQUEST_INPUT"


def test_v4_incomplete_tool_semantic_metadata_fails_before_model_call(
    registry: ToolRegistry,
) -> None:
    incomplete = ToolRegistry(
        (replace(registry.get("inspect_scATAC"), semantic_planning=None),)
    )
    model = CapturingPlanningModel(_payload(*_inspection_steps()))

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(model, wire_mode=PlanningWireMode.V4).plan(
            _request("inspect_canonical"), incomplete
        )

    assert caught.value.code == "PLANNER_CATALOG_INVALID"
    assert model.calls == []


def test_v4_unsupported_decision_preserves_high_level_behavior(
    registry: ToolRegistry,
) -> None:
    response = json.dumps(
        {
            "schema_version": 4,
            "decision": {
                "kind": "unsupported",
                "reason": "No safe semantic workflow is available.",
            },
        }
    )

    with pytest.raises(PlannerError) as caught:
        LLMPlanner(
            CapturingPlanningModel(response),
            wire_mode=PlanningWireMode.V4,
        ).plan(_request("inspect_canonical"), registry)

    assert caught.value.code == "UNSUPPORTED_REQUEST"
    assert caught.value.category is ErrorCategory.USER_INPUT_ERROR
    assert str(caught.value) == (
        "Planning model classified the request as unsupported."
    )
    assert caught.value.diagnostics[-1].stage is (
        PlanningDiagnosticStage.UNSUPPORTED
    )


def test_v4_plan_only_uses_runtime_preflight_and_executes_zero_tools(
    registry: ToolRegistry,
) -> None:
    guard = Mock(side_effect=AssertionError("PLAN_ONLY executed a tool"))
    guarded_registry = ToolRegistry(
        tuple(
            replace(registry.get(name), function=guard)
            for name in registry.names()
        )
    )
    model = CapturingPlanningModel(_payload(*_inspection_steps()))
    planner = LLMPlanner(model, wire_mode=PlanningWireMode.V4)

    result = AgentRuntime(planner=planner, registry=guarded_registry).run(
        _request("inspect_canonical", mode=RunMode.PLAN_ONLY)
    )

    assert result.status is RunStatus.PLANNED
    assert result.plan is not None
    assert result.plan.planner_name.endswith(":wire-v4")
    assert result.verification is not None and result.verification.passed
    guard.assert_not_called()
