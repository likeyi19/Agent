"""Offline tests for composable, registry-owned planning semantics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import re

import pytest

from agent.orchestration import (
    AgentRequest,
    ArgumentPlanningSemantics,
    ArtifactSemanticKind,
    PlanningProvenanceRole,
    PlanningSourceEligibility,
    PlanningToolRole,
    ToolPlanningSemantics,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.orchestration.llm_planner import _build_prompt, _sanitized_catalog


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return build_default_tool_registry()


def _catalog_by_tool(registry: ToolRegistry) -> dict[str, dict[str, object]]:
    return {
        str(tool["name"]): dict(tool)
        for tool in _sanitized_catalog(registry)
    }


def _argument(tool: dict[str, object], name: str) -> dict[str, object]:
    for section in ("required_arguments", "optional_arguments"):
        arguments = tool[section]
        assert isinstance(arguments, dict)
        if name in arguments:
            value = arguments[name]
            assert isinstance(value, dict)
            return value
    raise AssertionError(f"Unknown argument {name!r}.")


def _result(tool: dict[str, object], name: str) -> dict[str, object]:
    results = tool["result_fields"]
    assert isinstance(results, dict)
    result = results[name]
    assert isinstance(result, dict)
    return result


def test_all_tools_have_complete_immutable_registry_owned_semantics(registry) -> None:
    assert len(registry.names()) == 11
    for tool_name in registry.names():
        spec = registry.get(tool_name)
        assert spec.planning is not None
        assert isinstance(spec.planning.role, PlanningToolRole)
        assert 0 < len(spec.planning.description) <= 512
        arguments = {
            **spec.required_arguments,
            **spec.optional_arguments,
        }
        assert arguments
        assert all(argument.planning is not None for argument in arguments.values())
        with pytest.raises(FrozenInstanceError):
            spec.planning.role = PlanningToolRole.OPERATION  # type: ignore[misc]


def test_planning_catalog_is_derived_from_registry_metadata(registry) -> None:
    replacement_description = "Registry-derived inspection semantics."
    specs = tuple(
        replace(
            registry.get(name),
            planning=ToolPlanningSemantics(
                PlanningToolRole.INSPECTION,
                replacement_description,
            ),
        )
        if name == "inspect_scATAC"
        else registry.get(name)
        for name in registry.names()
    )

    changed = _catalog_by_tool(ToolRegistry(specs))

    assert changed["inspect_scATAC"]["description"] == replacement_description
    assert changed["inspect_scATAC"]["planning_role"] == "inspection"


def test_planning_vocabularies_are_bounded_and_valid() -> None:
    slug = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    assert all(slug.fullmatch(role.value) for role in PlanningToolRole)
    assert all(slug.fullmatch(source.value) for source in PlanningSourceEligibility)
    assert all(slug.fullmatch(kind.value) for kind in ArtifactSemanticKind)
    assert all(slug.fullmatch(role.value) for role in PlanningProvenanceRole)
    with pytest.raises(ValueError, match="exceeds"):
        ToolPlanningSemantics(PlanningToolRole.OPERATION, "x" * 513)
    with pytest.raises(TypeError, match="source_eligibility"):
        ArgumentPlanningSemantics("description", "input")  # type: ignore[arg-type]


def test_producer_and_consumer_artifact_kinds_compose(registry) -> None:
    tools = _catalog_by_tool(registry)
    relationships = (
        ("inspect_scATAC", "input_path", "epizoo_embed_cells", "input_path"),
        (
            "epizoo_embed_cells",
            "embedding_path",
            "build_cell_neighbors",
            "embedding_path",
        ),
        (
            "epizoo_embed_cells",
            "cell_ids_path",
            "build_cell_neighbors",
            "cell_ids_path",
        ),
        ("build_cell_neighbors", "analysis_path", "cluster_cells", "analysis_path"),
        ("cluster_cells", "analysis_path", "compute_cell_umap", "analysis_path"),
        (
            "cluster_cells",
            "analysis_path",
            "evaluate_cell_clustering",
            "analysis_path",
        ),
        (
            "compute_cell_umap",
            "analysis_path",
            "evaluate_cell_clustering",
            "analysis_path",
        ),
        (
            "epizoo_embed_cells",
            "embedding_path",
            "transfer_cell_labels",
            "reference_embedding_path",
        ),
        (
            "epizoo_embed_cells",
            "embedding_path",
            "transfer_cell_labels",
            "query_embedding_path",
        ),
        (
            "epizoo_embed_cells",
            "cell_ids_path",
            "transfer_cell_labels",
            "reference_cell_ids_path",
        ),
        (
            "transfer_cell_labels",
            "annotation_path",
            "evaluate_cell_annotation",
            "annotation_path",
        ),
        (
            "transfer_cell_labels",
            "annotation_path",
            "build_replicate_pseudobulk",
            "group_annotation_path",
        ),
        (
            "validate_scATAC_feature_space",
            "feature_space_path",
            "build_replicate_pseudobulk",
            "feature_space_path",
        ),
        (
            "build_replicate_pseudobulk",
            "pseudobulk_path",
            "run_replicate_differential_accessibility",
            "pseudobulk_path",
        ),
    )
    for producer, result_name, consumer, argument_name in relationships:
        kind = _result(tools[producer], result_name)["artifact_kind"]
        assert kind in _argument(tools[consumer], argument_name)[
            "accepted_artifact_kinds"
        ]


def test_bindable_artifacts_and_non_bindable_diagnostics_are_distinct(registry) -> None:
    tools = _catalog_by_tool(registry)
    embedding_path = _result(tools["epizoo_embed_cells"], "embedding_path")
    n_cells = _result(tools["epizoo_embed_cells"], "n_cells")

    assert embedding_path["downstream_bindable"] is True
    assert embedding_path["artifact_kind"] == "epizoo_embedding"
    assert n_cells == {"json_types": ("integer",), "downstream_bindable": False}


def test_source_guidance_distinguishes_request_and_composable_arguments(registry) -> None:
    tools = _catalog_by_tool(registry)

    assert _argument(tools["epizoo_embed_cells"], "output_dir")[
        "source_eligibility"
    ] == "request_input_only"
    assert _argument(tools["build_cell_neighbors"], "embedding_path")[
        "source_eligibility"
    ] == "request_input_or_upstream_result"
    assert PlanningSourceEligibility.UPSTREAM_RESULT_ONLY.value == (
        "upstream_result_only"
    )


def test_reference_query_provenance_is_explicit_without_branch_order(registry) -> None:
    transfer = _catalog_by_tool(registry)["transfer_cell_labels"]
    reference_names = (
        "reference_embedding_path",
        "reference_cell_ids_path",
        "reference_h5ad_path",
        "reference_species",
        "reference_checkpoint_path",
    )
    query_names = (
        "query_embedding_path",
        "query_cell_ids_path",
        "query_h5ad_path",
        "query_species",
        "query_checkpoint_path",
    )

    assert all(
        _argument(transfer, name)["provenance_role"] == "reference"
        for name in reference_names
    )
    assert all(
        _argument(transfer, name)["provenance_role"] == "query"
        for name in query_names
    )
    assert not any(
        key in transfer for key in ("workflow", "step_order", "required_predecessors")
    )


def test_scientific_parameters_and_registered_defaults_are_exposed_safely(
    registry,
) -> None:
    tools = _catalog_by_tool(registry)
    assert _argument(tools["build_cell_neighbors"], "n_neighbors") == {
        "json_types": ("integer",),
        "allows_step_output_ref": True,
        "description": "Explicit nearest-neighbor count.",
        "source_eligibility": "request_input_only",
        "accepted_artifact_kinds": (),
        "scientific_parameter": True,
        "default_when_omitted": 15,
    }
    checkpoint = _argument(tools["epizoo_embed_cells"], "checkpoint_path")
    assert checkpoint["scientific_parameter"] is True
    assert checkpoint["default_when_omitted"] == "registered_tool_default"


def test_conditional_m8_semantics_are_declarative_notes(registry) -> None:
    tools = _catalog_by_tool(registry)
    layer = _argument(tools["validate_scATAC_feature_space"], "layer_key")
    annotation = _argument(
        tools["build_replicate_pseudobulk"], "group_annotation_path"
    )

    assert "matrix_source is layer" in layer["conditional_note"]
    assert "group_source is verified_annotation" in annotation["conditional_note"]
    assert tools["run_replicate_differential_accessibility"]["conditional_notes"]


def test_planning_semantics_do_not_become_runtime_argument_rejections(registry) -> None:
    validated = registry.validate_arguments(
        "cluster_cells",
        {
            "analysis_path": "/synthetic/existing-neighbors.h5ad",
            "output_dir": "/synthetic/output",
        },
    )

    assert validated["analysis_path"] == "/synthetic/existing-neighbors.h5ad"


def test_generic_instructions_cover_composition_and_rejection_without_recipes(
    registry,
) -> None:
    private_path = "/private/do-not-disclose/input.h5ad"
    payload = json.loads(
        _build_prompt(
            AgentRequest(
                "metadata-prompt",
                "Analyze the supplied data.",
                {"input_path": private_path, "n_neighbors": 23},
            ),
            registry,
        )
    )
    instructions = " ".join(payload["instructions"]).casefold()
    catalog = json.dumps(payload["tools"], sort_keys=True)

    assert payload["planning_catalog_semantic_version"] == 1
    assert len(payload["tools"]) == 11
    assert "preserve every supplied scientific parameter" in instructions
    assert "never match by json type alone" in instructions
    assert "never invent paths" in instructions
    assert "upstream references" in instructions and "dependencies" in instructions
    assert "artifact, data, or provenance flow" in instructions
    assert "reference, query, and ground-truth" in instructions
    assert "required information or capability is absent" in instructions
    assert "unsupported, ambiguous, conflicting" in instructions
    assert "workflow" not in catalog.casefold()
    assert private_path not in catalog
    for forbidden in (
        "function",
        "module",
        "exception_classifier",
        "retryable_error_codes",
    ):
        assert forbidden not in catalog


def test_compact_prompt_retains_every_registered_scientific_semantic(
    registry,
) -> None:
    payload = json.loads(
        _build_prompt(
            AgentRequest(
                "compact-semantic-catalog",
                "Plan against the complete catalog.",
                {"input_path": "/synthetic/input.h5ad"},
            ),
            registry,
        )
    )
    prompt_tools = payload["tools"]
    source_codes = {
        code: meaning
        for code, meaning in payload["catalog_format"]["source"].items()
    }

    assert set(prompt_tools) == set(registry.names())
    for tool_name in registry.names():
        spec = registry.get(tool_name)
        role, meaning, arguments, results, notes = prompt_tools[tool_name]
        assert role == spec.planning.role.value
        assert meaning == spec.planning.description
        argument_specs = {**spec.required_arguments, **spec.optional_arguments}
        assert set(arguments) == set(argument_specs)
        for argument_name, argument_spec in argument_specs.items():
            planning = argument_spec.planning
            argument_meaning, source_code, options = arguments[argument_name]
            assert argument_meaning == planning.description
            assert source_codes[source_code] == planning.source_eligibility.value
            if planning.accepted_artifact_kinds:
                assert options["a"] == [
                    kind.value for kind in planning.accepted_artifact_kinds
                ]
            if planning.provenance_role is not None:
                assert options["p"] == planning.provenance_role.value
            if planning.scientific_parameter:
                assert options["s"] is True
            if planning.conditional_note is not None:
                assert options["n"] == planning.conditional_note

        assert set(results) == set(spec.result_contract.planning_fields)
        for result_name, planning in spec.result_contract.planning_fields.items():
            result_meaning, artifact, provenance, bindable = results[result_name]
            assert result_meaning == planning.description
            assert artifact == (
                None
                if planning.artifact_kind is None
                else planning.artifact_kind.value
            )
            assert provenance == (
                None
                if planning.provenance_role is None
                else planning.provenance_role.value
            )
            assert bindable is planning.downstream_bindable
        assert notes == list(spec.planning.conditional_notes)
