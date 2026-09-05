"""Registry-driven semantic planning catalog and prompt contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from agent.orchestration import (
    AgentRequest,
    PlannerError,
    ToolRegistry,
    build_default_tool_registry,
    build_semantic_planning_catalog,
    build_semantic_planning_prompt,
)
from agent.orchestration.llm_planner import _build_prompt
from benchmarks.planner.benchmark import load_cases


ROOT = Path(__file__).resolve().parents[3]
CASES_PATH = ROOT / "benchmarks" / "planner" / "cases.json"

TOOL_PURPOSE = 0
TOOL_INPUTS = 1
TOOL_OUTPUTS = 2
TOOL_CONSTRAINTS = 3

PORT_REQUIRED = 0
PORT_REQUEST_MODE = 1
PORT_REQUEST_SOURCES = 2
PORT_UPSTREAM_TYPES = 3
PORT_REQUIRED_LINEAGE = 4
PORT_GUIDANCE = 5

GUIDANCE_MEANING = 0
GUIDANCE_SCIENTIFIC = 1
GUIDANCE_CHOICES = 2
GUIDANCE_DEFAULT = 3
GUIDANCE_CONSTRAINT = 4

OUTPUT_TYPE = 0
OUTPUT_LINEAGE_FROM = 1


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _case(case_id: str):
    return next(case for case in load_cases(CASES_PATH) if case.case_id == case_id)


def _request(case_id: str) -> AgentRequest:
    case = _case(case_id)
    return AgentRequest(f"{case_id}-semantic", case.prompt, case.inputs)


def _tool(catalog, name: str):
    return catalog["tools"][name]


def test_catalog_is_versioned_complete_and_registry_driven(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("inspect_canonical"), registry
    )

    assert catalog["semantic_catalog_version"] == 1
    assert tuple(catalog["tools"]) == tuple(sorted(registry.names()))
    assert catalog["catalog_format"]["tool"] == (
        "purpose",
        "input_ports",
        "output_ports",
        "constraints",
    )
    assert set(catalog) == {
        "semantic_catalog_version",
        "request_inputs",
        "catalog_format",
        "tools",
    }


def test_request_inputs_expose_only_sorted_names_and_basic_types(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "types",
        "Use the named inputs.",
        {
            "z_null": None,
            "a_string": "secret",
            "f_object": {"secret": "value"},
            "e_array": ["secret"],
            "d_number": 1.25,
            "c_integer": 3,
            "b_boolean": True,
        },
    )
    catalog = build_semantic_planning_catalog(request, registry)

    assert catalog["request_inputs"] == {
        "a_string": "string",
        "b_boolean": "boolean",
        "c_integer": "integer",
        "d_number": "number",
        "e_array": "array",
        "f_object": "object",
        "z_null": "null",
    }


def test_catalog_is_deterministic_for_equivalent_inputs_and_registry(
    registry: ToolRegistry,
) -> None:
    first = AgentRequest(
        "first",
        "First wording.",
        {"species": "mouse", "input_path": "/private/first.h5ad"},
    )
    second = AgentRequest(
        "second",
        "Different wording.",
        {"input_path": "/elsewhere/second.h5ad", "species": "human"},
    )
    reordered = ToolRegistry(
        tuple(registry.get(name) for name in reversed(registry.names()))
    )

    expected = build_semantic_planning_catalog(first, registry)
    assert expected == build_semantic_planning_catalog(first, registry)
    assert expected == build_semantic_planning_catalog(second, registry)
    assert expected == build_semantic_planning_catalog(first, reordered)
    assert build_semantic_planning_prompt(first, registry) == (
        build_semantic_planning_prompt(first, registry)
    )


def test_future_semantic_tool_appears_without_renderer_changes(
    registry: ToolRegistry,
) -> None:
    future = replace(
        registry.get("inspect_scATAC"),
        name="future_semantic_tool",
    )
    extended = ToolRegistry(
        (*tuple(registry.get(name) for name in registry.names()), future)
    )
    catalog = build_semantic_planning_catalog(
        _request("inspect_canonical"), extended
    )

    assert "future_semantic_tool" in catalog["tools"]
    assert _tool(catalog, "future_semantic_tool")[TOOL_OUTPUTS]["dataset"][
        OUTPUT_TYPE
    ] == "raw_scatac_dataset.v1"


def test_planner_visible_tool_without_semantic_metadata_fails_closed(
    registry: ToolRegistry,
) -> None:
    incomplete = ToolRegistry(
        (replace(registry.get("inspect_scATAC"), semantic_planning=None),)
    )

    with pytest.raises(PlannerError) as caught:
        build_semantic_planning_catalog(
            _request("inspect_canonical"), incomplete
        )

    assert caught.value.code == "PLANNER_CATALOG_INVALID"


def test_inspection_catalog_marks_unique_and_ambiguous_request_sources(
    registry: ToolRegistry,
) -> None:
    unique = build_semantic_planning_catalog(
        _request("inspect_canonical"), registry
    )
    unique_port = _tool(unique, "inspect_scATAC")[TOOL_INPUTS]["dataset"]
    ambiguous_request = AgentRequest(
        "inspect-ambiguous",
        "Inspect one dataset.",
        {
            "reference_input_path": "/reference.h5ad",
            "query_input_path": "/query.h5ad",
        },
    )
    ambiguous = build_semantic_planning_catalog(ambiguous_request, registry)
    ambiguous_port = _tool(ambiguous, "inspect_scATAC")[TOOL_INPUTS][
        "dataset"
    ]

    assert unique_port[PORT_REQUIRED] is True
    assert unique_port[PORT_REQUEST_MODE] == "unique_available"
    assert unique_port[PORT_REQUEST_SOURCES] == (("input_path", None),)
    assert ambiguous_port[PORT_REQUEST_MODE] == "explicit_choice_required"
    assert ambiguous_port[PORT_REQUEST_SOURCES] == (
        ("query_input_path", "query"),
        ("reference_input_path", "reference"),
    )
    assert _tool(unique, "inspect_scATAC")[TOOL_OUTPUTS]["dataset"] == (
        "raw_scatac_dataset.v1",
        "dataset",
    )


def test_grouped_request_source_is_offered_only_when_complete(
    registry: ToolRegistry,
) -> None:
    partial = AgentRequest(
        "partial-embedding",
        "Build neighbors.",
        {"embedding_path": "/private/embedding.npy"},
    )
    complete = AgentRequest(
        "complete-embedding",
        "Build neighbors.",
        {
            "embedding_path": "/private/embedding.npy",
            "cell_ids_path": "/private/cells.json",
        },
    )
    partial_port = _tool(
        build_semantic_planning_catalog(partial, registry),
        "build_cell_neighbors",
    )[TOOL_INPUTS]["embedding"]
    complete_port = _tool(
        build_semantic_planning_catalog(complete, registry),
        "build_cell_neighbors",
    )[TOOL_INPUTS]["embedding"]

    assert partial_port[PORT_REQUEST_MODE] == "none_available"
    assert partial_port[PORT_REQUEST_SOURCES] == ()
    assert complete_port[PORT_REQUEST_MODE] == "unique_available"
    assert complete_port[PORT_REQUEST_SOURCES] == (("embedding_path", None),)


def test_embedding_and_downstream_catalog_hide_grouped_execution_members(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("inspect_canonical"), registry
    )
    embedding_output = _tool(catalog, "epizoo_embed_cells")[TOOL_OUTPUTS][
        "embedding"
    ]
    neighbors_input = _tool(catalog, "build_cell_neighbors")[TOOL_INPUTS][
        "embedding"
    ]
    serialized = json.dumps(catalog, sort_keys=True)

    assert embedding_output == ("epizoo_embedding_bundle.v1", "dataset")
    assert neighbors_input[PORT_UPSTREAM_TYPES] == (
        "epizoo_embedding_bundle.v1",
    )
    for forbidden in (
        "cell_ids_path",
        "analysis_path",
        "report_path",
        "ref_output_key",
        "StepOutputRef",
        "field_name",
        '"members"',
        '"arguments"',
        '"result_fields"',
    ):
        assert forbidden not in serialized


def test_label_transfer_catalog_preserves_branch_roles_without_bundle_members(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("label_transfer_canonical"), registry
    )
    inputs = _tool(catalog, "transfer_cell_labels")[TOOL_INPUTS]

    assert inputs["reference_dataset"][PORT_REQUEST_SOURCES] == (
        ("reference_input_path", "reference"),
    )
    assert inputs["query_dataset"][PORT_REQUEST_SOURCES] == (
        ("query_input_path", "query"),
    )
    assert inputs["reference_dataset"][PORT_REQUIRED_LINEAGE] == "reference"
    assert inputs["query_dataset"][PORT_REQUIRED_LINEAGE] == "query"
    assert inputs["reference_embedding"][PORT_UPSTREAM_TYPES] == (
        "epizoo_embedding_bundle.v1",
    )
    assert inputs["query_embedding"][PORT_UPSTREAM_TYPES] == (
        "epizoo_embedding_bundle.v1",
    )
    assert inputs["reference_embedding"][PORT_GUIDANCE] is None
    assert "reference_cell_ids_path" not in json.dumps(catalog)
    assert "query_cell_ids_path" not in json.dumps(catalog)


def test_scientific_parameter_guidance_uses_semantic_port_names(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("label_transfer_optional"), registry
    )
    transfer_inputs = _tool(catalog, "transfer_cell_labels")[TOOL_INPUTS]
    neighbor_guidance = transfer_inputs["n_neighbors"][PORT_GUIDANCE]
    metric_guidance = transfer_inputs["metric"][PORT_GUIDANCE]

    assert neighbor_guidance[GUIDANCE_SCIENTIFIC] is True
    assert neighbor_guidance[GUIDANCE_DEFAULT] == 20
    assert metric_guidance[GUIDANCE_MEANING].startswith("Explicit reference")
    assert metric_guidance[GUIDANCE_CHOICES] == ("euclidean", "cosine")
    assert metric_guidance[GUIDANCE_DEFAULT] == "euclidean"
    assert transfer_inputs["overwrite"][PORT_GUIDANCE][
        GUIDANCE_SCIENTIFIC
    ] is False


def test_scoped_optional_request_inputs_are_exposed_only_to_intended_ports(
    registry: ToolRegistry,
) -> None:
    request = _request("label_transfer_optional")
    catalog = build_semantic_planning_catalog(request, registry)
    embedding_overwrite = _tool(catalog, "epizoo_embed_cells")[TOOL_INPUTS][
        "overwrite"
    ]
    transfer_overwrite = _tool(catalog, "transfer_cell_labels")[TOOL_INPUTS][
        "overwrite"
    ]
    instructions = json.loads(
        build_semantic_planning_prompt(request, registry)
    )["instructions"]
    deterministic_rule = next(
        rule
        for rule in instructions
        if "deterministic_scoped" in rule
    )

    assert embedding_overwrite[PORT_REQUIRED] is False
    assert transfer_overwrite[PORT_REQUIRED] is False
    assert "transfer_overwrite" in request.inputs
    assert "overwrite" not in request.inputs
    assert embedding_overwrite[PORT_REQUEST_MODE] == "none_available"
    assert transfer_overwrite[PORT_REQUEST_MODE] == "deterministic_scoped"
    assert embedding_overwrite[PORT_REQUEST_SOURCES] == ()
    assert transfer_overwrite[PORT_REQUEST_SOURCES] == (
        ("transfer_overwrite", None),
    )
    assert embedding_overwrite[PORT_GUIDANCE][GUIDANCE_DEFAULT] is False
    assert transfer_overwrite[PORT_GUIDANCE][GUIDANCE_DEFAULT] is False
    assert embedding_overwrite[PORT_GUIDANCE][GUIDANCE_MEANING] != (
        transfer_overwrite[PORT_GUIDANCE][GUIDANCE_MEANING]
    )
    assert "compiler-bound" in deterministic_rule
    assert "do not emit" in deterministic_rule
    assert "overwrite" not in deterministic_rule
    assert "epizoo_embed_cells" not in deterministic_rule
    assert "transfer_cell_labels" not in deterministic_rule

    embedding_request_inputs = dict(request.inputs)
    embedding_request_inputs.pop("transfer_overwrite")
    embedding_request_inputs["embedding_overwrite"] = False
    embedding_request = AgentRequest(
        "embedding-scoped-overwrite",
        request.prompt,
        embedding_request_inputs,
    )
    embedding_catalog = build_semantic_planning_catalog(
        embedding_request, registry
    )
    embedding_port = _tool(
        embedding_catalog, "epizoo_embed_cells"
    )[TOOL_INPUTS]["overwrite"]
    transfer_port = _tool(
        embedding_catalog, "transfer_cell_labels"
    )[TOOL_INPUTS]["overwrite"]

    assert embedding_port[PORT_REQUEST_SOURCES] == (
        ("embedding_overwrite", None),
    )
    assert embedding_port[PORT_REQUEST_MODE] == "deterministic_scoped"
    assert transfer_port[PORT_REQUEST_SOURCES] == ()


def test_downstream_scoped_overwrites_are_deterministic_not_fanned_out(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("downstream_explicit_steps"), registry
    )
    scoped = {
        ("build_cell_neighbors", "overwrite"): "neighbors_overwrite",
        ("cluster_cells", "overwrite"): "cluster_overwrite",
        ("compute_cell_umap", "overwrite"): "umap_overwrite",
        ("build_cell_neighbors", "random_seed"): "neighbors_random_seed",
        ("cluster_cells", "random_seed"): "cluster_random_seed",
        ("compute_cell_umap", "random_seed"): "umap_random_seed",
    }

    for (tool_name, port_name), selector in scoped.items():
        port = _tool(catalog, tool_name)[TOOL_INPUTS][port_name]
        assert port[PORT_REQUEST_MODE] == "deterministic_scoped"
        assert port[PORT_REQUEST_SOURCES] == ((selector, None),)

    all_sources = {
        source[0]
        for tool_name, _port_name in scoped
        for port in _tool(catalog, tool_name)[TOOL_INPUTS].values()
        for source in port[PORT_REQUEST_SOURCES]
    }
    assert "overwrite" not in all_sources
    assert "random_seed" not in all_sources

    paired = build_semantic_planning_catalog(
        _request("differential_accessibility_paired_covariates"), registry
    )
    for tool_name in (
        "validate_scATAC_feature_space",
        "build_replicate_pseudobulk",
        "run_replicate_differential_accessibility",
    ):
        assert _tool(paired, tool_name)[TOOL_INPUTS]["overwrite"][
            PORT_REQUEST_MODE
        ] == "optional_explicit"


def test_paired_da_catalog_exposes_artifact_flow_parameters_and_rules(
    registry: ToolRegistry,
) -> None:
    catalog = build_semantic_planning_catalog(
        _request("differential_accessibility_paired_covariates"), registry
    )
    validate = _tool(catalog, "validate_scATAC_feature_space")
    pseudobulk = _tool(catalog, "build_replicate_pseudobulk")
    differential = _tool(
        catalog, "run_replicate_differential_accessibility"
    )

    assert validate[TOOL_OUTPUTS]["feature_space"][OUTPUT_TYPE] == (
        "validated_feature_space.v1"
    )
    assert pseudobulk[TOOL_INPUTS]["feature_space"][PORT_UPSTREAM_TYPES] == (
        "validated_feature_space.v1",
    )
    assert pseudobulk[TOOL_OUTPUTS]["pseudobulk"][OUTPUT_TYPE] == (
        "replicate_pseudobulk.v1"
    )
    assert differential[TOOL_INPUTS]["pseudobulk"][PORT_UPSTREAM_TYPES] == (
        "replicate_pseudobulk.v1",
    )
    design = differential[TOOL_INPUTS]["design_type"][PORT_GUIDANCE]
    assert design[GUIDANCE_CHOICES] == ("independent", "paired")
    assert any("paired design" in note for note in differential[TOOL_CONSTRAINTS])
    assert "pseudobulk_path" not in json.dumps(catalog)


def test_evaluation_catalog_keeps_ground_truth_evaluation_only(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "annotation-evaluation",
        "Evaluate the fixed annotation.",
        {
            "annotation_path": "/private/annotation.h5ad",
            "ground_truth_h5ad_path": "/private/truth.h5ad",
            "ground_truth_label_key": "secret-label-column",
            "output_dir": "/private/output",
        },
    )
    catalog = build_semantic_planning_catalog(request, registry)
    evaluation = _tool(catalog, "evaluate_cell_annotation")
    inputs = evaluation[TOOL_INPUTS]

    assert inputs["annotation"][PORT_REQUIRED_LINEAGE] == "query"
    assert inputs["ground_truth_dataset"][PORT_REQUIRED_LINEAGE] == (
        "ground_truth"
    )
    assert inputs["ground_truth_dataset"][PORT_REQUEST_SOURCES] == (
        ("ground_truth_h5ad_path", "ground_truth"),
    )
    assert any(
        "evaluation-only" in constraint
        for constraint in evaluation[TOOL_CONSTRAINTS]
    )


def test_catalog_does_not_promote_unmapped_execution_parameter(
    registry: ToolRegistry,
) -> None:
    request = AgentRequest(
        "conditional",
        "Validate coordinates.",
        {"input_path": "/private/data.h5ad"},
    )
    catalog = build_semantic_planning_catalog(request, registry)
    validate_inputs = _tool(catalog, "validate_scATAC_feature_space")[
        TOOL_INPUTS
    ]

    assert "layer_key" not in validate_inputs
    assert validate_inputs["matrix_source"][PORT_GUIDANCE][
        GUIDANCE_CHOICES
    ] == ("X", "layer")
    assert validate_inputs["matrix_source"][PORT_GUIDANCE][
        GUIDANCE_CONSTRAINT
    ] is None


def test_prompt_is_compact_semantic_context_not_schema_or_serialization_manual(
    registry: ToolRegistry,
) -> None:
    request = _request("downstream_canonical")
    prompt = build_semantic_planning_prompt(request, registry)
    payload = json.loads(prompt)

    assert payload["semantic_prompt_version"] == 1
    assert payload["user_request"] == request.prompt
    assert payload["catalog"] == json.loads(
        json.dumps(build_semantic_planning_catalog(request, registry))
    )
    assert payload["wire_v4"] == {
        "decisions": ["plan", "unsupported"],
        "schema_version": 4,
        "source_fields": ["target", "source"],
        "source_kinds": {
            "input": ["kind", "input"],
            "step": ["kind", "step"],
            "step_port": ["kind", "step", "source_port"],
        },
        "step_fields": [
            "step_id",
            "tool",
            "sources",
            "control_dependencies",
        ],
    }
    assert "$defs" not in prompt
    assert "additionalProperties" not in prompt
    assert "StepOutputRef" not in prompt
    assert "ref_output_key" not in prompt
    assert any("valid DAG" in rule for rule in payload["instructions"])
    unsupported_rule = next(
        rule for rule in payload["instructions"] if "unsupported" in rule.lower()
    )
    assert "genuine capability/input insufficiency only" in unsupported_rule
    assert "plan if the catalog can satisfy the request" in unsupported_rule
    assert "downstream" not in unsupported_rule


def test_structured_input_values_never_enter_catalog_or_prompt(
    registry: ToolRegistry,
) -> None:
    sentinels = (
        "/SENTINEL/private/input.h5ad",
        "SENTINEL-BIOLOGICAL-LABEL",
        "SENTINEL-CONDITION-VALUE",
        "/SENTINEL/checkpoint.pt",
        "/SENTINEL/output-directory",
        "SENTINEL-ARRAY-ELEMENT",
        "SENTINEL-OBJECT-VALUE",
    )
    request = AgentRequest(
        "privacy",
        "Plan from the available named inputs.",
        {
            "input_path": sentinels[0],
            "label_key": sentinels[1],
            "condition_key": sentinels[2],
            "checkpoint_path": sentinels[3],
            "output_dir": sentinels[4],
            "covariates": [sentinels[5]],
            "metadata": {"value": sentinels[6]},
        },
    )
    catalog_text = json.dumps(
        build_semantic_planning_catalog(request, registry),
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt = build_semantic_planning_prompt(request, registry)

    for sentinel in sentinels:
        assert sentinel not in catalog_text
        assert sentinel not in prompt
    assert set(json.loads(prompt)["catalog"]["request_inputs"]) == set(
        request.inputs
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "inspect_canonical",
        "downstream_canonical",
        "label_transfer_canonical",
        "differential_accessibility_paired_covariates",
    ),
)
def test_semantic_context_is_not_larger_than_v3_prompt(
    registry: ToolRegistry,
    case_id: str,
) -> None:
    request = _request(case_id)
    catalog = json.dumps(
        build_semantic_planning_catalog(request, registry),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    semantic_prompt = build_semantic_planning_prompt(request, registry)
    v3_prompt = _build_prompt(request, registry)

    assert len(catalog.encode("utf-8")) < len(v3_prompt.encode("utf-8"))
    assert len(semantic_prompt.encode("utf-8")) <= len(v3_prompt.encode("utf-8"))
