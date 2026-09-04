"""Pure provider-facing wire-v4 schema and parser contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentRequest,
    PlannerError,
    SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP,
    SEMANTIC_WIRE_MAX_SOURCES_PER_STEP,
    SemanticPlanCandidate,
    SemanticPlanStep,
    SemanticRequestInputSource,
    SemanticStepOutputSource,
    ToolRegistry,
    build_default_tool_registry,
    build_semantic_wire_v4_schema,
    parse_semantic_wire_v4,
)
from agent.orchestration.llm_planner import _response_schema
from benchmarks.planner.benchmark import load_cases, oracle_response


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _case(case_id: str):
    return next(case for case in load_cases(CASES_PATH) if case.case_id == case_id)


def _request(case_id: str) -> AgentRequest:
    case = _case(case_id)
    return AgentRequest(f"{case_id}-v4", case.prompt, case.inputs)


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


def _wire_step(
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


def _payload(*steps: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 4,
        "decision": {"kind": "plan", "steps": list(steps)},
    }


def _encoded(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _paired_da_steps() -> tuple[dict[str, object], ...]:
    return (
        _wire_step(
            "provider-step-101",
            "validate_scATAC_feature_space",
            sources=[_input("overwrite", "overwrite")],
        ),
        _wire_step(
            "provider-step-102",
            "build_replicate_pseudobulk",
            sources=[
                _step_source("feature_space", "provider-step-101"),
                _input("overwrite", "overwrite"),
            ],
        ),
        _wire_step(
            "provider-step-103",
            "run_replicate_differential_accessibility",
            sources=[
                _step_source("pseudobulk", "provider-step-102"),
                _input("overwrite", "overwrite"),
            ],
        ),
    )


def _transfer_steps(*, optional: bool = False) -> tuple[dict[str, object], ...]:
    reference_embedding_sources = [
        _step_source("dataset", "provider-step-101")
    ]
    query_embedding_sources = [_step_source("dataset", "provider-step-103")]
    transfer_sources = [
        _step_source("reference_dataset", "provider-step-101"),
        _step_source("reference_embedding", "provider-step-102"),
        _step_source("query_dataset", "provider-step-103"),
        _step_source("query_embedding", "provider-step-104"),
    ]
    if optional:
        reference_embedding_sources.append(
            _input("checkpoint", "checkpoint_path")
        )
        query_embedding_sources.append(_input("checkpoint", "checkpoint_path"))
        transfer_sources.append(_input("overwrite", "overwrite"))
    return (
        _wire_step(
            "provider-step-101",
            "inspect_scATAC",
            sources=[_input("dataset", "reference_input_path")],
        ),
        _wire_step(
            "provider-step-102",
            "epizoo_embed_cells",
            sources=reference_embedding_sources,
        ),
        _wire_step(
            "provider-step-103",
            "inspect_scATAC",
            sources=[_input("dataset", "query_input_path")],
        ),
        _wire_step(
            "provider-step-104",
            "epizoo_embed_cells",
            sources=query_embedding_sources,
        ),
        _wire_step(
            "provider-step-105",
            "transfer_cell_labels",
            sources=transfer_sources,
        ),
    )


def _downstream_steps() -> tuple[dict[str, object], ...]:
    return (
        _wire_step("provider-step-101", "inspect_scATAC"),
        _wire_step(
            "provider-step-102",
            "epizoo_embed_cells",
            sources=[_step_source("dataset", "provider-step-101")],
        ),
        _wire_step(
            "provider-step-103",
            "build_cell_neighbors",
            sources=[_step_source("embedding", "provider-step-102")],
        ),
        _wire_step(
            "provider-step-104",
            "cluster_cells",
            sources=[_step_source("analysis", "provider-step-103")],
        ),
        _wire_step(
            "provider-step-105",
            "compute_cell_umap",
            sources=[_step_source("analysis", "provider-step-104")],
        ),
    )


def test_schema_is_closed_generic_and_registry_request_driven(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "schema",
        "Plan safely.",
        {"z_input": "secret-z", "a_input": "/private/secret-a"},
    )
    schema = build_semantic_wire_v4_schema(registry, request)

    assert schema["type"] == "object"
    assert schema["required"] == ("schema_version", "decision")
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["enum"] == (4,)
    assert schema["$defs"]["step"]["properties"]["tool"]["enum"] == tuple(
        sorted(
            name
            for name in registry.names()
            if registry.get(name).planning is not None
        )
    )
    assert schema["$defs"]["input_source"]["properties"]["input"][
        "enum"
    ] == ("a_input", "z_input")
    serialized = _encoded(schema)
    assert "secret-z" not in serialized
    assert "/private/secret-a" not in serialized
    for forbidden in ("arguments", "depends_on", "ref_output_key"):
        assert forbidden not in serialized


def test_schema_is_deterministic_for_equivalent_registry_and_input_names(
    registry: ToolRegistry,
) -> None:
    first_request = AgentRequest(
        "first", "First.", {"input_path": "/private/first.h5ad", "species": "mouse"}
    )
    second_request = AgentRequest(
        "second", "Second.", {"species": "human", "input_path": "/elsewhere"}
    )
    reordered = ToolRegistry(tuple(registry.get(name) for name in reversed(registry.names())))

    first = build_semantic_wire_v4_schema(registry, first_request)
    assert first == build_semantic_wire_v4_schema(registry, first_request)
    assert first == build_semantic_wire_v4_schema(registry, second_request)
    assert first == build_semantic_wire_v4_schema(reordered, first_request)


def test_schema_uses_only_provider_compatible_closed_constructs(
    registry: ToolRegistry,
) -> None:
    schema = build_semantic_wire_v4_schema(
        registry, AgentRequest("schema", "Plan.", {"input_path": "secret"})
    )
    banned_keywords = {
        "allOf",
        "else",
        "if",
        "not",
        "oneOf",
        "patternProperties",
        "then",
    }

    def visit(node: object) -> None:
        if isinstance(node, dict):
            assert banned_keywords.isdisjoint(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            if set(node) == {"$ref"}:
                assert node["$ref"].startswith("#/$defs/")
            for key, value in node.items():
                if key in {"$defs", "properties"}:
                    for child in value.values():
                        visit(child)
                else:
                    visit(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                visit(value)

    visit(schema)
    json.loads(json.dumps(schema, allow_nan=False))


def test_schema_without_request_inputs_omits_input_source_variant(
    registry: ToolRegistry,
) -> None:
    schema = build_semantic_wire_v4_schema(
        registry, AgentRequest("no-inputs", "Use upstream sources.", {})
    )

    assert "input_source" not in schema["$defs"]
    assert schema["$defs"]["source"]["anyOf"] == (
        {"$ref": "#/$defs/step_source"},
        {"$ref": "#/$defs/step_port_source"},
    )


def test_schema_fails_if_planner_visible_tool_lacks_semantic_metadata(
    registry: ToolRegistry,
) -> None:
    incomplete = ToolRegistry(
        (
            replace(
                registry.get("inspect_scATAC"),
                semantic_planning=None,
            ),
        )
    )

    with pytest.raises(PlannerError) as caught:
        build_semantic_wire_v4_schema(
            incomplete, _request("inspect_canonical")
        )

    assert caught.value.code == "PLANNER_CATALOG_INVALID"


def test_simple_inspection_with_unique_implicit_input_parses(
    registry: ToolRegistry,
) -> None:
    request = _request("inspect_canonical")
    candidate = parse_semantic_wire_v4(
        _encoded(_payload(_wire_step("inspect", "inspect_scATAC"))),
        request,
        registry,
    )

    assert candidate == SemanticPlanCandidate(
        (SemanticPlanStep("inspect", "inspect_scATAC"),)
    )


def test_inspection_with_multiple_dataset_inputs_parses_explicit_source(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "inspect-reference",
        "Inspect the reference.",
        {
            "reference_input_path": "/reference.h5ad",
            "query_input_path": "/query.h5ad",
        },
    )
    candidate = parse_semantic_wire_v4(
        _encoded(
            _payload(
                _wire_step(
                    "inspect",
                    "inspect_scATAC",
                    sources=[_input("dataset", "reference_input_path")],
                )
            )
        ),
        request,
        registry,
    )

    assert candidate.steps[0].sources == (
        SemanticRequestInputSource("dataset", "reference_input_path"),
    )


def test_paired_da_round_trip_contains_only_semantic_sources(
    registry: ToolRegistry,
) -> None:
    candidate = parse_semantic_wire_v4(
        _encoded(_payload(*_paired_da_steps())),
        _request("differential_accessibility_paired_covariates"),
        registry,
    )

    assert candidate == SemanticPlanCandidate(
        (
            SemanticPlanStep(
                "provider-step-101",
                "validate_scATAC_feature_space",
                sources=(SemanticRequestInputSource("overwrite", "overwrite"),),
            ),
            SemanticPlanStep(
                "provider-step-102",
                "build_replicate_pseudobulk",
                sources=(
                    SemanticStepOutputSource(
                        "feature_space", "provider-step-101"
                    ),
                    SemanticRequestInputSource("overwrite", "overwrite"),
                ),
            ),
            SemanticPlanStep(
                "provider-step-103",
                "run_replicate_differential_accessibility",
                sources=(
                    SemanticStepOutputSource(
                        "pseudobulk", "provider-step-102"
                    ),
                    SemanticRequestInputSource("overwrite", "overwrite"),
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    ("case_id", "optional"),
    (
        ("label_transfer_canonical", False),
        ("label_transfer_optional", True),
    ),
)
def test_dual_branch_transfer_round_trip_preserves_semantic_choices(
    registry: ToolRegistry,
    case_id: str,
    optional: bool,
) -> None:
    candidate = parse_semantic_wire_v4(
        _encoded(_payload(*_transfer_steps(optional=optional))),
        _request(case_id),
        registry,
    )

    assert tuple(step.tool_name for step in candidate.steps) == (
        "inspect_scATAC",
        "epizoo_embed_cells",
        "inspect_scATAC",
        "epizoo_embed_cells",
        "transfer_cell_labels",
    )
    transfer_sources = candidate.steps[-1].sources
    assert transfer_sources[:4] == (
        SemanticStepOutputSource("reference_dataset", "provider-step-101"),
        SemanticStepOutputSource("reference_embedding", "provider-step-102"),
        SemanticStepOutputSource("query_dataset", "provider-step-103"),
        SemanticStepOutputSource("query_embedding", "provider-step-104"),
    )
    if optional:
        assert candidate.steps[1].sources[-1] == SemanticRequestInputSource(
            "checkpoint", "checkpoint_path"
        )
        assert candidate.steps[3].sources[-1] == SemanticRequestInputSource(
            "checkpoint", "checkpoint_path"
        )
        assert transfer_sources[-1] == SemanticRequestInputSource(
            "overwrite", "overwrite"
        )


def test_downstream_grouped_channels_require_no_mechanical_members(
    registry: ToolRegistry,
) -> None:
    payload = _payload(*_downstream_steps())
    candidate = parse_semantic_wire_v4(
        _encoded(payload), _request("downstream_canonical"), registry
    )

    assert len(candidate.steps[2].sources) == 1
    assert candidate.steps[2].sources == (
        SemanticStepOutputSource("embedding", "provider-step-102"),
    )
    assert "embedding_path" not in _encoded(payload)
    assert "cell_ids_path" not in _encoded(payload)


def test_step_port_variant_maps_to_existing_source_port_field(
    registry: ToolRegistry,
) -> None:
    request = _request("downstream_canonical")
    candidate = parse_semantic_wire_v4(
        _encoded(
            _payload(
                _wire_step("producer", "cluster_cells"),
                _wire_step(
                    "consumer",
                    "compute_cell_umap",
                    sources=[
                        _step_port_source(
                            "analysis", "producer", "clustered_analysis"
                        )
                    ],
                ),
            )
        ),
        request,
        registry,
    )

    assert candidate.steps[1].sources == (
        SemanticStepOutputSource(
            "analysis", "producer", "clustered_analysis"
        ),
    )


def test_control_only_dependencies_round_trip_separately_from_sources(
    registry: ToolRegistry,
) -> None:
    payload = _payload(
        _wire_step("first", "inspect_scATAC"),
        _wire_step(
            "second",
            "inspect_scATAC",
            control_dependencies=["first"],
        ),
    )

    candidate = parse_semantic_wire_v4(
        _encoded(payload), _request("inspect_canonical"), registry
    )

    assert candidate.steps[1].sources == ()
    assert candidate.steps[1].control_dependencies == ("first",)


@pytest.mark.parametrize("version", [0, 3, 5, "4", True])
def test_wrong_schema_version_is_rejected(
    registry: ToolRegistry, version: object
) -> None:
    payload = _payload(_wire_step("inspect", "inspect_scATAC"))
    payload["schema_version"] = version

    with pytest.raises(PlannerError, match="unsupported schema version"):
        parse_semantic_wire_v4(_encoded(payload), _request("inspect_canonical"), registry)


@pytest.mark.parametrize(
    "payload",
    (
        {"schema_version": 4},
        {"schema_version": 4, "decision": {}, "extra": True},
        {"schema_version": 4, "decision": "plan"},
        {"schema_version": 4, "decision": {"kind": "other"}},
        {"schema_version": 4, "decision": {"kind": "plan"}},
        {"schema_version": 4, "decision": {"kind": "unsupported"}},
        {
            "schema_version": 4,
            "decision": {"kind": "unsupported", "reason": "safe", "steps": []},
        },
    ),
)
def test_malformed_root_or_decision_variant_is_rejected(
    registry: ToolRegistry, payload: object
) -> None:
    with pytest.raises(PlannerError):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_plan_without_steps_is_rejected(registry: ToolRegistry) -> None:
    with pytest.raises(PlannerError, match="at least one step"):
        parse_semantic_wire_v4(
            _encoded(_payload()), _request("inspect_canonical"), registry
        )


@pytest.mark.parametrize(
    "step",
    (
        "inspect",
        {"step_id": "inspect", "tool": "inspect_scATAC"},
        {
            **_wire_step("inspect", "inspect_scATAC"),
            "arguments": {},
        },
        _wire_step("inspect", "arbitrary_python"),
    ),
)
def test_malformed_step_is_rejected(
    registry: ToolRegistry, step: object
) -> None:
    with pytest.raises(PlannerError):
        parse_semantic_wire_v4(
            _encoded(_payload(step)),  # type: ignore[arg-type]
            _request("inspect_canonical"),
            registry,
        )


def test_duplicate_step_id_is_rejected(registry: ToolRegistry) -> None:
    payload = _payload(
        _wire_step("duplicate", "inspect_scATAC"),
        _wire_step("duplicate", "inspect_scATAC"),
    )

    with pytest.raises(PlannerError, match="semantic candidate structure"):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


@pytest.mark.parametrize(
    "source",
    (
        {"kind": "unknown", "target": "dataset"},
        {"kind": "input", "target": "dataset"},
        {"kind": "step", "target": "dataset"},
        {"kind": "step_port", "target": "analysis", "step": "producer"},
        {**_input("dataset", "input_path"), "extra": True},
        {**_step_source("dataset", "producer"), "source_port": "dataset"},
    ),
)
def test_malformed_source_variant_is_rejected(
    registry: ToolRegistry, source: dict[str, object]
) -> None:
    payload = _payload(
        _wire_step("inspect", "inspect_scATAC", sources=[source])  # type: ignore[list-item]
    )

    with pytest.raises(PlannerError):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_unavailable_request_input_selector_is_rejected(
    registry: ToolRegistry,
) -> None:
    payload = _payload(
        _wire_step(
            "inspect",
            "inspect_scATAC",
            sources=[_input("dataset", "not_available")],
        )
    )

    with pytest.raises(PlannerError, match="not available"):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_duplicate_structural_json_key_is_rejected(registry: ToolRegistry) -> None:
    response = (
        '{"schema_version":4,"decision":{"kind":"plan","kind":"unsupported",'
        '"steps":[]}}'
    )

    with pytest.raises(PlannerError, match="strict valid JSON"):
        parse_semantic_wire_v4(response, _request("inspect_canonical"), registry)


@pytest.mark.parametrize(
    "response",
    (
        "not-json",
        '[{"schema_version":4}]',
        '{"schema_version":NaN,"decision":{"kind":"plan","steps":[]}}',
        '{"schema_version":Infinity,"decision":{"kind":"plan","steps":[]}}',
    ),
)
def test_non_strict_json_is_rejected(
    registry: ToolRegistry, response: str
) -> None:
    with pytest.raises(PlannerError):
        parse_semantic_wire_v4(response, _request("inspect_canonical"), registry)


def test_oversized_response_is_rejected(registry: ToolRegistry) -> None:
    payload = {
        "schema_version": 4,
        "decision": {"kind": "unsupported", "reason": "x" * 70_000},
    }

    with pytest.raises(PlannerError, match="byte limit"):
        parse_semantic_wire_v4(
            json.dumps(payload), _request("inspect_canonical"), registry
        )


def test_excessive_tree_depth_is_rejected(registry: ToolRegistry) -> None:
    nested: object = "value"
    for _ in range(12):
        nested = [nested]
    payload = {"schema_version": 4, "decision": nested}

    with pytest.raises(PlannerError, match="nested too deeply"):
        parse_semantic_wire_v4(
            json.dumps(payload), _request("inspect_canonical"), registry
        )


def test_excessive_node_count_is_rejected(registry: ToolRegistry) -> None:
    payload = {
        "schema_version": 4,
        "decision": {"kind": "unsupported", "reason": [0] * 4_100},
    }

    with pytest.raises(PlannerError, match="too many values"):
        parse_semantic_wire_v4(
            json.dumps(payload), _request("inspect_canonical"), registry
        )


def test_excessive_step_count_is_rejected(registry: ToolRegistry) -> None:
    payload = _payload(
        *(
            _wire_step(f"inspect-{index}", "inspect_scATAC")
            for index in range(17)
        )
    )

    with pytest.raises(PlannerError, match="step limit"):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_excessive_source_count_is_rejected(registry: ToolRegistry) -> None:
    sources = [
        _input("dataset", "input_path")
        for _ in range(SEMANTIC_WIRE_MAX_SOURCES_PER_STEP + 1)
    ]
    payload = _payload(
        _wire_step("inspect", "inspect_scATAC", sources=sources)
    )

    with pytest.raises(PlannerError, match="source limit"):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_excessive_control_edge_count_is_rejected(
    registry: ToolRegistry,
) -> None:
    controls = [
        f"step-{index}"
        for index in range(
            SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP + 1
        )
    ]
    payload = _payload(
        _wire_step(
            "inspect",
            "inspect_scATAC",
            control_dependencies=controls,
        )
    )

    with pytest.raises(PlannerError, match="edge limit"):
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )


def test_oversized_identifier_and_reason_are_rejected(
    registry: ToolRegistry,
) -> None:
    request = _request("inspect_canonical")
    oversized_step = _payload(_wire_step("x" * 129, "inspect_scATAC"))
    oversized_reason = {
        "schema_version": 4,
        "decision": {"kind": "unsupported", "reason": "x" * 2_049},
    }

    with pytest.raises(PlannerError, match="128-character"):
        parse_semantic_wire_v4(_encoded(oversized_step), request, registry)
    with pytest.raises(PlannerError, match="2048-character"):
        parse_semantic_wire_v4(_encoded(oversized_reason), request, registry)


def test_graph_cycle_and_semantic_target_legality_are_deferred(
    registry: ToolRegistry,
) -> None:
    payload = _payload(
        _wire_step(
            "first",
            "inspect_scATAC",
            sources=[_step_source("not_a_real_port", "second")],
        ),
        _wire_step(
            "second",
            "inspect_scATAC",
            sources=[_step_source("also_not_real", "first")],
        ),
    )

    candidate = parse_semantic_wire_v4(
        _encoded(payload), _request("inspect_canonical"), registry
    )

    assert tuple(step.step_id for step in candidate.steps) == ("first", "second")


def test_unsupported_decision_raises_bounded_user_error(
    registry: ToolRegistry,
) -> None:
    payload = {
        "schema_version": 4,
        "decision": {"kind": "unsupported", "reason": "No compatible tool."},
    }

    with pytest.raises(PlannerError, match="No compatible tool") as caught:
        parse_semantic_wire_v4(
            _encoded(payload), _request("inspect_canonical"), registry
        )

    assert caught.value.code == "UNSUPPORTED_REQUEST"


def test_v4_schema_and_representative_responses_are_materially_smaller(
    registry: ToolRegistry,
) -> None:
    v4_payloads = {
        "inspect_canonical": _payload(
            _wire_step("provider-step-101", "inspect_scATAC")
        ),
        "downstream_canonical": _payload(*_downstream_steps()),
        "label_transfer_canonical": _payload(*_transfer_steps()),
        "label_transfer_optional": _payload(
            *_transfer_steps(optional=True)
        ),
        "differential_accessibility_paired_covariates": _payload(
            *_paired_da_steps()
        ),
    }

    for case_id, v4_payload in v4_payloads.items():
        request = _request(case_id)
        v3_schema_bytes = len(_encoded(_response_schema(registry, request)).encode())
        v4_schema_bytes = len(
            _encoded(build_semantic_wire_v4_schema(registry, request)).encode()
        )
        v3_payload = json.loads(oracle_response(_case(case_id)))
        v3_response_bytes = len(_encoded(v3_payload).encode())
        v4_response_bytes = len(_encoded(v4_payload).encode())
        v3_argument_slots = sum(
            len(step["arguments"]) for step in v3_payload["steps"]
        )
        v4_sources = sum(
            len(step["sources"])
            for step in v4_payload["decision"]["steps"]
        )

        assert v4_schema_bytes <= v3_schema_bytes * 0.4
        assert v4_response_bytes < v3_response_bytes
        assert v4_sources < v3_argument_slots
