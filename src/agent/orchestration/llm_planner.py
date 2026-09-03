"""Strict provider-neutral LLM planner for the registered scientific tools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from agent.schemas import (
    AgentPlan,
    AgentRequest,
    ErrorCategory,
    JsonValue,
    PlanStep,
    StepOutputRef,
)

from .planner import PlannerError
from .planning_diagnostics import (
    DiagnosedPlanningAttempt,
    PlanningDiagnostic,
    PlanningDiagnosticContext,
    PlanningDiagnosticStage,
)
from .planning_model import PlanningModel, PlanningModelError, PlanningModelProfile
from .registry import ArgumentSpec, ToolRegistry


_SCHEMA_VERSION = 2
_MAX_RESPONSE_BYTES = 64 * 1024
_MAX_RESPONSE_DEPTH = 10
_MAX_RESPONSE_NODES = 4096
_MAX_STEPS = 16
_MAX_IDENTIFIER_LENGTH = 128
_MAX_DESCRIPTION_LENGTH = 2048
_MAX_REASON_LENGTH = 2048
_MAX_MODEL_ID_LENGTH = 256
_PROVIDER_ERROR_CODES = frozenset(
    {
        "PLANNING_PROVIDER_ERROR",
        "PROVIDER_AUTHENTICATION_FAILED",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
    }
)

_TOOL_DESCRIPTIONS = {
    "inspect_scATAC": (
        "Inspect a scATAC-seq h5ad file and return lightweight matrix and "
        "metadata information without running model inference."
    ),
    "epizoo_embed_cells": (
        "Compute EpiZoo cell embeddings from a scATAC-seq h5ad file and "
        "persist the embeddings and ordered cell identifiers as artifacts."
    ),
    "build_cell_neighbors": (
        "Build a sparse Scanpy nearest-neighbor graph from persisted EpiZoo "
        "embeddings and their ordered cell-ID sidecar."
    ),
    "cluster_cells": (
        "Run fixed-setting Leiden clustering on a valid Milestone 6 neighbors "
        "analysis artifact and persist a new compact h5ad artifact."
    ),
    "compute_cell_umap": (
        "Compute a fixed two-dimensional UMAP from a valid clustered Milestone 6 "
        "analysis artifact and persist a new compact h5ad artifact."
    ),
    "evaluate_cell_clustering": (
        "Evaluate fixed cluster labels in a valid Milestone 6 clustering or UMAP "
        "artifact against an ordered reference annotation and persist four metrics."
    ),
    "transfer_cell_labels": (
        "Transfer biological labels from an annotated reference to ordered query "
        "cells by exact deterministic k-nearest neighbors in EpiZoo embedding space."
    ),
    "evaluate_cell_annotation": (
        "Evaluate a fixed valid Milestone 6.3 annotation against exact ordered "
        "ground-truth labels and persist fixed accuracy, macro-F1, confusion, "
        "per-class, and descriptive confidence diagnostics."
    ),
    "validate_scATAC_feature_space": (
        "Validate an immutable raw sparse scATAC regulatory feature space, its "
        "declared count semantics, species/assembly, ordered identities, and "
        "optional explicit coordinates, then persist a provenance manifest."
    ),
    "build_replicate_pseudobulk": (
        "Build exact sparse SUM pseudobulks for fixed group, biological replicate, "
        "and condition metadata from a verified regulatory feature-space manifest."
    ),
    "run_replicate_differential_accessibility": (
        "Run the fixed replicate-aware edgeR quasi-likelihood differential-"
        "accessibility workflow on a verified Milestone 8.1 pseudobulk artifact."
    ),
}

_ARGUMENT_DESCRIPTIONS = {
    ("inspect_scATAC", "path"): "Input scATAC-seq h5ad path.",
    ("epizoo_embed_cells", "input_path"): "Input scATAC-seq h5ad path.",
    ("epizoo_embed_cells", "output_dir"): "Directory for embedding artifacts.",
    ("epizoo_embed_cells", "species"): "Species supported by EpiZoo.",
    ("epizoo_embed_cells", "checkpoint_path"): "EpiZoo checkpoint path.",
    ("epizoo_embed_cells", "device"): "Execution device.",
    ("epizoo_embed_cells", "overwrite"): "Whether existing artifacts may be replaced.",
    ("build_cell_neighbors", "embedding_path"): "Persisted EpiZoo npy embedding path.",
    ("build_cell_neighbors", "cell_ids_path"): "Ordered EpiZoo cell-ID sidecar path.",
    ("build_cell_neighbors", "output_dir"): "Directory for the neighbors artifact.",
    ("build_cell_neighbors", "n_neighbors"): "Explicit nearest-neighbor count.",
    ("build_cell_neighbors", "metric"): "Euclidean or cosine neighbor metric.",
    ("build_cell_neighbors", "random_seed"): "Explicit nonnegative random seed.",
    ("build_cell_neighbors", "overwrite"): "Whether an existing output may be replaced.",
    ("cluster_cells", "analysis_path"): "Valid Milestone 6 neighbors artifact path.",
    ("cluster_cells", "output_dir"): "Directory for the clustered artifact.",
    ("cluster_cells", "resolution"): "Explicit positive Leiden resolution.",
    ("cluster_cells", "random_seed"): "Explicit nonnegative random seed.",
    ("cluster_cells", "overwrite"): "Whether an existing output may be replaced.",
    ("compute_cell_umap", "analysis_path"): "Valid clustered Milestone 6 artifact path.",
    ("compute_cell_umap", "output_dir"): "Directory for the UMAP artifact.",
    ("compute_cell_umap", "min_dist"): "Explicit nonnegative UMAP minimum distance.",
    ("compute_cell_umap", "spread"): "Explicit positive UMAP spread.",
    ("compute_cell_umap", "random_seed"): "Explicit nonnegative random seed.",
    ("compute_cell_umap", "overwrite"): "Whether an existing output may be replaced.",
    ("evaluate_cell_clustering", "analysis_path"): (
        "Valid Milestone 6 clustering or UMAP artifact path."
    ),
    ("evaluate_cell_clustering", "reference_h5ad_path"): (
        "Reference annotated h5ad path with identical ordered cell IDs."
    ),
    ("evaluate_cell_clustering", "label_key"): "Reference annotation obs column.",
    ("evaluate_cell_clustering", "output_dir"): "Directory for the metrics report.",
    ("evaluate_cell_clustering", "cluster_key"): "Predicted cluster obs column.",
    ("evaluate_cell_clustering", "overwrite"): (
        "Whether an existing evaluation report may be replaced."
    ),
    ("transfer_cell_labels", "reference_embedding_path"): (
        "Persisted reference EpiZoo npy embedding path."
    ),
    ("transfer_cell_labels", "reference_cell_ids_path"): (
        "Ordered reference EpiZoo cell-ID sidecar path."
    ),
    ("transfer_cell_labels", "reference_h5ad_path"): (
        "Annotated reference scATAC h5ad path."
    ),
    ("transfer_cell_labels", "reference_label_key"): (
        "Reference biological string-label obs column."
    ),
    ("transfer_cell_labels", "reference_species"): (
        "Species reported by reference EpiZoo embedding."
    ),
    ("transfer_cell_labels", "reference_checkpoint_path"): (
        "Checkpoint path reported by reference EpiZoo embedding."
    ),
    ("transfer_cell_labels", "query_embedding_path"): (
        "Persisted query EpiZoo npy embedding path."
    ),
    ("transfer_cell_labels", "query_cell_ids_path"): (
        "Ordered query EpiZoo cell-ID sidecar path."
    ),
    ("transfer_cell_labels", "query_h5ad_path"): "Query scATAC h5ad path.",
    ("transfer_cell_labels", "query_species"): (
        "Species reported by query EpiZoo embedding."
    ),
    ("transfer_cell_labels", "query_checkpoint_path"): (
        "Checkpoint path reported by query EpiZoo embedding."
    ),
    ("transfer_cell_labels", "output_dir"): (
        "Directory for the compact label-transfer artifact."
    ),
    ("transfer_cell_labels", "n_neighbors"): "Explicit exact-neighbor count.",
    ("transfer_cell_labels", "metric"): "Euclidean or cosine distance metric.",
    ("transfer_cell_labels", "min_confidence"): (
        "Explicit minimum uniform-vote confidence within zero and one."
    ),
    ("transfer_cell_labels", "overwrite"): (
        "Whether an existing annotation artifact may be replaced."
    ),
    ("evaluate_cell_annotation", "annotation_path"): (
        "Fixed valid Milestone 6.3 label-transfer h5ad path."
    ),
    ("evaluate_cell_annotation", "ground_truth_h5ad_path"): (
        "Ground-truth h5ad path with identical ordered query cell IDs."
    ),
    ("evaluate_cell_annotation", "ground_truth_label_key"): (
        "Ground-truth biological string-label obs column."
    ),
    ("evaluate_cell_annotation", "output_dir"): (
        "Directory for the annotation-evaluation JSON report."
    ),
    ("evaluate_cell_annotation", "overwrite"): (
        "Whether an existing annotation-evaluation report may be replaced."
    ),
    ("validate_scATAC_feature_space", "input_path"): "Immutable raw scATAC h5ad path.",
    ("validate_scATAC_feature_space", "output_dir"): "Directory for the feature-space manifest.",
    ("validate_scATAC_feature_space", "matrix_source"): "Exact matrix source: X or layer.",
    ("validate_scATAC_feature_space", "matrix_semantics"): "Declared regulatory matrix semantics.",
    ("validate_scATAC_feature_space", "species"): "Human or mouse source species.",
    ("validate_scATAC_feature_space", "genome_assembly"): "Compatible hg38 or mm10 genome assembly.",
    ("validate_scATAC_feature_space", "coordinate_source"): "Explicit var columns or no coordinates.",
    ("validate_scATAC_feature_space", "layer_key"): "Exact layer key when matrix_source is layer.",
    ("validate_scATAC_feature_space", "feature_chrom_key"): "Explicit chromosome var column.",
    ("validate_scATAC_feature_space", "feature_start_key"): "Explicit start-coordinate var column.",
    ("validate_scATAC_feature_space", "feature_end_key"): "Explicit end-coordinate var column.",
    ("validate_scATAC_feature_space", "coordinate_system"): "Explicit coordinate convention.",
    ("validate_scATAC_feature_space", "semantics_metadata_key"): "Optional corroborating raw uns key.",
    ("validate_scATAC_feature_space", "overwrite"): "Whether an existing manifest may be replaced.",
    ("build_replicate_pseudobulk", "feature_space_path"): "Verified regulatory feature-space manifest.",
    ("build_replicate_pseudobulk", "replicate_key"): "Raw obs biological replicate/subject column.",
    ("build_replicate_pseudobulk", "group_key"): "Raw group column or fixed predicted_label key.",
    ("build_replicate_pseudobulk", "condition_key"): "Raw obs comparison-condition column.",
    ("build_replicate_pseudobulk", "output_dir"): "Directory for the pseudobulk H5AD.",
    ("build_replicate_pseudobulk", "group_source"): "raw_obs or verified_annotation.",
    ("build_replicate_pseudobulk", "group_annotation_path"): "Valid Milestone 6.3 annotation path.",
    ("build_replicate_pseudobulk", "covariate_keys"): "Ordered raw obs covariate columns.",
    ("build_replicate_pseudobulk", "overwrite"): "Whether an existing pseudobulk may be replaced.",
    ("run_replicate_differential_accessibility", "pseudobulk_path"): (
        "Valid verified Milestone 8.1 replicate-pseudobulk H5AD path."
    ),
    ("run_replicate_differential_accessibility", "group_value"): (
        "Exact biological group value selected for comparison."
    ),
    ("run_replicate_differential_accessibility", "condition_key"): (
        "Condition metadata key recorded by the pseudobulk artifact."
    ),
    ("run_replicate_differential_accessibility", "numerator_condition"): (
        "Exact numerator condition for the directional contrast."
    ),
    ("run_replicate_differential_accessibility", "denominator_condition"): (
        "Exact denominator condition for the directional contrast."
    ),
    ("run_replicate_differential_accessibility", "design_type"): (
        "Explicit independent or paired replicate design."
    ),
    ("run_replicate_differential_accessibility", "output_dir"): (
        "Directory for the compact differential-accessibility H5AD."
    ),
    ("run_replicate_differential_accessibility", "covariates"): (
        "Ordered additive covariate specifications with exact key and kind."
    ),
    ("run_replicate_differential_accessibility", "overwrite"): (
        "Whether the identical DA output may be replaced."
    ),
}


class _DuplicateJsonKey(ValueError):
    pass


def _duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _response_schema(registry: ToolRegistry) -> Mapping[str, JsonValue]:
    tool_names = registry.names()
    argument_names = tuple(
        sorted(
            {
                argument_name
                for tool_name in tool_names
                for argument_name in (
                    *registry.get(tool_name).required_arguments,
                    *registry.get(tool_name).optional_arguments,
                )
            }
        )
    )
    tool_name_schema: dict[str, JsonValue] = {"type": "string"}
    if tool_names:
        tool_name_schema["enum"] = tool_names
    argument_name_schema: dict[str, JsonValue] = {"type": "string"}
    if argument_names:
        argument_name_schema["enum"] = argument_names

    binding_properties: Mapping[str, JsonValue] = {
        "name": argument_name_schema,
        "binding_type": {"type": "string", "enum": ("input", "ref")},
        "input_name": {"type": ("string", "null")},
        "ref_step_id": {"type": ("string", "null")},
        "ref_output_key": {"type": ("string", "null")},
    }
    binding_schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": binding_properties,
        "required": tuple(binding_properties),
        "additionalProperties": False,
    }
    step_properties: Mapping[str, JsonValue] = {
        "step_id": {"type": "string"},
        "tool_name": tool_name_schema,
        "arguments": {"type": "array", "items": binding_schema},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "description": {"type": ("string", "null")},
    }
    step_schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": step_properties,
        "required": tuple(step_properties),
        "additionalProperties": False,
    }
    root_properties: Mapping[str, JsonValue] = {
        "schema_version": {"type": "integer", "enum": (_SCHEMA_VERSION,)},
        "status": {"type": "string", "enum": ("plan", "unsupported")},
        "steps": {"type": "array", "items": step_schema},
        "reason": {"type": ("string", "null")},
    }
    return {
        "type": "object",
        "properties": root_properties,
        "required": tuple(root_properties),
        "additionalProperties": False,
    }


def _python_type_to_json_type(value_type: type) -> str:
    if value_type is str or issubclass(value_type, Path):
        return "string"
    if value_type is bool:
        return "boolean"
    if value_type is int:
        return "integer"
    if value_type is float:
        return "number"
    if value_type in {list, tuple}:
        return "array"
    if value_type is dict:
        return "object"
    if value_type is type(None):
        return "null"
    return "unknown"


def _catalog_choice(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Path):
        return str(value)
    raise PlannerError(
        "PLANNER_CATALOG_INVALID",
        "A registered tool choice cannot be represented safely for planning.",
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
    )


def _argument_catalog(
    tool_name: str, argument_name: str, spec: ArgumentSpec
) -> Mapping[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "json_types": tuple(
            dict.fromkeys(
                _python_type_to_json_type(value_type)
                for value_type in spec.accepted_types
            )
        ),
        "allows_step_output_ref": spec.allow_step_output_ref,
    }
    description = _ARGUMENT_DESCRIPTIONS.get((tool_name, argument_name))
    if description is not None:
        metadata["description"] = description
    if spec.choices:
        metadata["choices"] = tuple(_catalog_choice(value) for value in spec.choices)
    return metadata


def _sanitized_catalog(registry: ToolRegistry) -> tuple[Mapping[str, JsonValue], ...]:
    tools: list[Mapping[str, JsonValue]] = []
    for tool_name in registry.names():
        spec = registry.get(tool_name)
        tools.append(
            {
                "name": spec.name,
                "description": _TOOL_DESCRIPTIONS.get(
                    spec.name, "Registered scientific tool."
                ),
                "required_arguments": {
                    name: _argument_catalog(spec.name, name, argument_spec)
                    for name, argument_spec in sorted(spec.required_arguments.items())
                },
                "optional_arguments": {
                    name: _argument_catalog(spec.name, name, argument_spec)
                    for name, argument_spec in sorted(spec.optional_arguments.items())
                },
                "result_fields": tuple(sorted(spec.result_contract.required_fields)),
            }
        )
    return tuple(tools)


def _catalog_fingerprint(registry: ToolRegistry) -> str:
    encoded = json.dumps(
        _sanitized_catalog(registry),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value_type(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, tuple):
        return "array"
    return "object"


def _build_prompt(request: AgentRequest, registry: ToolRegistry) -> str:
    available_inputs = tuple(
        {
            "name": name,
            "json_type": _json_value_type(request.inputs[name]),
        }
        for name in sorted(request.inputs)
    )
    prompt_payload = {
        "instructions": (
            "Return exactly one JSON object matching the supplied response schema. "
            "Plan only with the listed tools. Represent every executable argument "
            "as one fixed binding object in the step arguments array. An input "
            "binding uses binding_type='input', an available input_name, and null "
            "ref fields. A reference binding uses binding_type='ref', null "
            "input_name, and non-null ref_step_id/ref_output_key naming a declared "
            "upstream dependency. Never emit executable literal values. For a plan, "
            "return non-empty steps and null reason. For an unsupported request, "
            "return empty steps and a non-empty reason."
        ),
        "request": {
            "prompt": request.prompt,
            "available_inputs": available_inputs,
        },
        "tools": _sanitized_catalog(registry),
    }
    return json.dumps(
        prompt_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _invalid_output(
    message: str,
    *,
    code: str = "PLANNER_OUTPUT_INVALID",
    stage: PlanningDiagnosticStage | None = None,
    reason_code: str | None = None,
    diagnostic_fields: Mapping[str, object] | None = None,
) -> PlannerError:
    if stage is None:
        stage = (
            PlanningDiagnosticStage.ARGUMENT_BINDING
            if code == "PLANNER_BINDING_INVALID"
            else PlanningDiagnosticStage.DEPENDENCY_REFERENCE
            if code == "PLANNER_STRUCTURE_INVALID"
            else PlanningDiagnosticStage.SCHEMA
        )
    return PlannerError(
        code,
        message,
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
        diagnostic_stage=stage,
        diagnostic_reason_code=reason_code or (
            "binding_invalid"
            if stage is PlanningDiagnosticStage.ARGUMENT_BINDING
            else "dependency_reference_invalid"
            if stage is PlanningDiagnosticStage.DEPENDENCY_REFERENCE
            else "wire_schema_invalid"
        ),
        diagnostic_fields=diagnostic_fields,
    )


def _parse_json_response(response: object) -> dict[str, object]:
    if not isinstance(response, str):
        raise _invalid_output(
            "Planning model response must be a JSON text string.",
            stage=PlanningDiagnosticStage.PARSE,
            reason_code="response_not_text",
        )
    try:
        response_size = len(response.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _invalid_output(
            "Planning model response contains invalid Unicode text.",
            stage=PlanningDiagnosticStage.PARSE,
            reason_code="response_unicode_invalid",
        ) from exc
    if response_size > _MAX_RESPONSE_BYTES:
        raise _invalid_output(
            f"Planning model response exceeds the {_MAX_RESPONSE_BYTES}-byte limit.",
            stage=PlanningDiagnosticStage.PARSE,
            reason_code="response_too_large",
        )
    try:
        parsed = json.loads(
            response,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, _DuplicateJsonKey, RecursionError, ValueError) as exc:
        raise _invalid_output(
            "Planning model response is not strict valid JSON.",
            stage=PlanningDiagnosticStage.PARSE,
            reason_code="malformed_json",
        ) from exc
    if not isinstance(parsed, dict):
        raise _invalid_output(
            "Planning model response must be one JSON object.",
            stage=PlanningDiagnosticStage.SCHEMA,
            reason_code="root_not_object",
        )
    _validate_tree_limits(parsed)
    return parsed


def _validate_tree_limits(value: object) -> None:
    node_count = 0

    def visit(nested: object, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_RESPONSE_NODES:
            raise _invalid_output("Planning model response contains too many values.")
        if depth > _MAX_RESPONSE_DEPTH:
            raise _invalid_output("Planning model response is nested too deeply.")
        if isinstance(nested, dict):
            for key, child in nested.items():
                if len(key) > _MAX_DESCRIPTION_LENGTH:
                    raise _invalid_output(
                        "Planning model response contains an oversized key."
                    )
                visit(child, depth + 1)
        elif isinstance(nested, list):
            for child in nested:
                visit(child, depth + 1)
        elif isinstance(nested, str) and len(nested) > _MAX_RESPONSE_BYTES:
            raise _invalid_output(
                "Planning model response contains an oversized string."
            )

    visit(value, 0)


def _require_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    supplied = set(value)
    missing = sorted(required - supplied)
    unknown = sorted(supplied - required - optional)
    if missing:
        raise _invalid_output(f"{context} is missing required fields: {missing}.")
    if unknown:
        raise _invalid_output(f"{context} contains unknown fields: {unknown}.")


def _bounded_string(
    value: object,
    *,
    field_name: str,
    maximum: int = _MAX_IDENTIFIER_LENGTH,
    code: str = "PLANNER_OUTPUT_INVALID",
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid_output(
            f"{field_name} must be a non-empty string.", code=code
        )
    if len(value) > maximum:
        raise _invalid_output(
            f"{field_name} exceeds the {maximum}-character limit.", code=code
        )
    return value


def _parse_argument_binding(
    binding: object,
    request: AgentRequest,
    *,
    context: str,
    step_index: int,
    binding_index: int,
) -> tuple[str, object]:
    diagnostic_fields = {
        "step_index": step_index,
        "binding_index": binding_index,
    }
    if not isinstance(binding, dict):
        raise _invalid_output(
            f"{context} must be a fixed argument-binding object.",
            code="PLANNER_BINDING_INVALID",
            reason_code="binding_not_object",
            diagnostic_fields=diagnostic_fields,
        )
    expected_fields = {
        "name",
        "binding_type",
        "input_name",
        "ref_step_id",
        "ref_output_key",
    }
    if set(binding) != expected_fields:
        raise _invalid_output(
            f"{context} must contain every v2 binding field and no others.",
            code="PLANNER_BINDING_INVALID",
            reason_code="binding_fields_invalid",
            diagnostic_fields=diagnostic_fields,
        )
    argument_name = _bounded_string(
        binding["name"],
        field_name=f"{context}.name",
        code="PLANNER_BINDING_INVALID",
    )
    binding_type = binding["binding_type"]
    if binding_type == "input":
        if binding["ref_step_id"] is not None or binding["ref_output_key"] is not None:
            raise _invalid_output(
                f"{context} input binding must have null reference fields.",
                code="PLANNER_BINDING_INVALID",
                reason_code="input_binding_reference_fields_nonnull",
                diagnostic_fields={
                    **diagnostic_fields,
                    "argument_name": argument_name,
                },
            )
        input_name = _bounded_string(
            binding["input_name"],
            field_name=f"{context}.input_name",
            code="PLANNER_BINDING_INVALID",
        )
        if input_name not in request.inputs:
            raise PlannerError(
                "MISSING_REQUIRED_INPUT",
                "Planning response requested unavailable structured input "
                f"{input_name!r}.",
                category=ErrorCategory.USER_INPUT_ERROR,
                diagnostic_stage=PlanningDiagnosticStage.ARGUMENT_BINDING,
                diagnostic_reason_code="request_input_missing",
                diagnostic_fields={
                    **diagnostic_fields,
                    "argument_name": argument_name,
                    "input_name": input_name,
                },
            )
        return argument_name, request.inputs[input_name]
    if binding_type == "ref":
        if binding["input_name"] is not None:
            raise _invalid_output(
                f"{context} reference binding must have null input_name.",
                code="PLANNER_BINDING_INVALID",
                reason_code="reference_binding_input_nonnull",
                diagnostic_fields={
                    **diagnostic_fields,
                    "argument_name": argument_name,
                },
            )
        return argument_name, StepOutputRef(
            step_id=_bounded_string(
                binding["ref_step_id"],
                field_name=f"{context}.ref_step_id",
                code="PLANNER_BINDING_INVALID",
            ),
            output_key=_bounded_string(
                binding["ref_output_key"],
                field_name=f"{context}.ref_output_key",
                code="PLANNER_BINDING_INVALID",
            ),
        )
    raise _invalid_output(
        f"{context}.binding_type must be 'input' or 'ref'.",
        code="PLANNER_BINDING_INVALID",
        reason_code="binding_type_invalid",
        diagnostic_fields={
            **diagnostic_fields,
            "argument_name": argument_name,
        },
    )


def _parse_plan_steps(
    raw_steps: object, request: AgentRequest
) -> tuple[PlanStep, ...]:
    if not isinstance(raw_steps, list):
        raise _invalid_output("Plan response `steps` must be an array.")
    if not raw_steps:
        raise _invalid_output("Plan response must contain at least one step.")
    if len(raw_steps) > _MAX_STEPS:
        raise _invalid_output(
            f"Plan response exceeds the {_MAX_STEPS}-step limit."
        )

    steps: list[PlanStep] = []
    for index, raw_step in enumerate(raw_steps):
        context = f"Plan step {index}"
        if not isinstance(raw_step, dict):
            raise _invalid_output(f"{context} must be an object.")
        _require_fields(
            raw_step,
            required=frozenset(
                {
                    "step_id",
                    "tool_name",
                    "arguments",
                    "depends_on",
                    "description",
                }
            ),
            context=context,
        )
        step_id = _bounded_string(raw_step["step_id"], field_name=f"{context}.step_id")
        tool_name = _bounded_string(
            raw_step["tool_name"], field_name=f"{context}.tool_name"
        )
        raw_arguments = raw_step["arguments"]
        if not isinstance(raw_arguments, list):
            raise _invalid_output(f"{context}.arguments must be an array.")
        arguments: dict[str, object] = {}
        for binding_index, binding in enumerate(raw_arguments):
            argument_name, argument_value = _parse_argument_binding(
                binding,
                request,
                context=f"{context}.arguments[{binding_index}]",
                step_index=index,
                binding_index=binding_index,
            )
            if argument_name in arguments:
                raise _invalid_output(
                    f"{context} contains duplicate argument {argument_name!r}.",
                    code="PLANNER_BINDING_INVALID",
                    reason_code="duplicate_argument",
                    diagnostic_fields={
                        "step_index": index,
                        "binding_index": binding_index,
                        "argument_name": argument_name,
                    },
                )
            arguments[argument_name] = argument_value

        raw_dependencies = raw_step["depends_on"]
        if not isinstance(raw_dependencies, list):
            raise _invalid_output(f"{context}.depends_on must be an array.")
        if len(raw_dependencies) > _MAX_STEPS:
            raise _invalid_output(f"{context}.depends_on contains too many entries.")
        dependencies = tuple(
            _bounded_string(
                dependency, field_name=f"{context}.depends_on[{dependency_index}]"
            )
            for dependency_index, dependency in enumerate(raw_dependencies)
        )

        description = raw_step["description"]
        if description is not None:
            description = _bounded_string(
                description,
                field_name=f"{context}.description",
                maximum=_MAX_DESCRIPTION_LENGTH,
            )
        try:
            steps.append(
                PlanStep(
                    step_id=step_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    depends_on=dependencies,
                    description=description,
                )
            )
        except (TypeError, ValueError) as exc:
            raise _invalid_output(
                f"{context} violates the AgentPlan structure.",
                code="PLANNER_STRUCTURE_INVALID",
                reason_code="step_structure_invalid",
                diagnostic_fields={"step_index": index},
            ) from exc
    return tuple(steps)


def _parse_response(response: object, request: AgentRequest) -> tuple[PlanStep, ...]:
    payload = _parse_json_response(response)
    _require_fields(
        payload,
        required=frozenset({"schema_version", "status", "steps", "reason"}),
        context="Planning response",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != _SCHEMA_VERSION:
        raise _invalid_output(
            f"Planning response uses unsupported schema version {version!r}."
        )
    status = payload["status"]
    if status == "unsupported":
        if payload["steps"] != []:
            raise _invalid_output("Unsupported response must contain empty steps.")
        reason = _bounded_string(
            payload["reason"],
            field_name="Unsupported response reason",
            maximum=_MAX_REASON_LENGTH,
        )
        raise PlannerError(
            "UNSUPPORTED_REQUEST",
            reason,
            category=ErrorCategory.USER_INPUT_ERROR,
            diagnostic_stage=PlanningDiagnosticStage.UNSUPPORTED,
            diagnostic_reason_code="explicit_unsupported_response",
        )
    if status != "plan":
        raise _invalid_output("Planning response has an unsupported status.")
    if payload["reason"] is not None:
        raise _invalid_output("Plan response reason must be null.")
    return _parse_plan_steps(payload["steps"], request)


def _sanitize_model_id(model_id: object) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise TypeError("PlanningModel `model_id` must be a non-empty string.")
    if len(model_id) > _MAX_MODEL_ID_LENGTH:
        raise ValueError(
            f"PlanningModel `model_id` exceeds {_MAX_MODEL_ID_LENGTH} characters."
        )
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id.strip())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-._")
    if not sanitized:
        raise ValueError("PlanningModel `model_id` has no safe identity characters.")
    return sanitized


def _plan_id(request: AgentRequest, steps: tuple[PlanStep, ...]) -> str:
    content = {
        "request_id": request.request_id,
        "steps": tuple(step.to_dict() for step in steps),
    }
    canonical = json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{request.request_id}:llm:{digest}"


class LLMPlanner:
    """Convert one strict planning-model response into existing plan contracts."""

    def __init__(
        self,
        model: PlanningModel,
        *,
        profile: PlanningModelProfile | None = None,
    ) -> None:
        if not callable(getattr(model, "complete", None)):
            raise TypeError("`model` must provide a callable complete() method.")
        if profile is not None and not isinstance(profile, PlanningModelProfile):
            raise TypeError("`profile` must be a PlanningModelProfile or None.")
        if profile is not None and not profile.enabled:
            raise ValueError("`profile` must be enabled.")
        if profile is not None and not profile.supports_structured_output:
            raise ValueError("`profile` must support structured output.")
        self._model = model
        self._model_id = _sanitize_model_id(getattr(model, "model_id", None))
        self._profile = profile
        self._name = (
            f"llm:{self._model_id}"
            if profile is None
            else f"llm-profile:{profile.profile_id}"
        )

    @property
    def model(self) -> PlanningModel:
        return self._model

    @property
    def name(self) -> str:
        return self._name

    @property
    def profile(self) -> PlanningModelProfile | None:
        return self._profile

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        """Return the accepted candidate while preserving the Planner protocol."""

        return self.plan_with_diagnostics(request, registry).plan

    def plan_with_diagnostics(
        self,
        request: AgentRequest,
        registry: ToolRegistry,
    ) -> DiagnosedPlanningAttempt:
        """Construct one plan and sanitized diagnostics with exactly one model call."""

        if not isinstance(request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")

        profile_id = "unprofiled" if self._profile is None else self._profile.profile_id
        provider_id = "custom" if self._profile is None else self._profile.provider_id
        model_identity = (
            self._model_id if self._profile is None else self._profile.model_id
        )
        context = PlanningDiagnosticContext(
            profile_id=profile_id,
            provider_id=provider_id,
            model_identity_digest=hashlib.sha256(
                model_identity.encode("utf-8")
            ).hexdigest(),
            catalog_fingerprint=_catalog_fingerprint(registry),
            offered_tool_names=registry.names(),
            planning_wire_schema_version=_SCHEMA_VERSION,
        )
        diagnostics: list[PlanningDiagnostic] = [
            context.diagnostic(
                PlanningDiagnosticStage.PROVIDER,
                "PROVIDER_CALL_STARTED",
                "started",
            )
        ]
        prompt = _build_prompt(request, registry)
        try:
            response = self._model.complete(
                prompt=prompt,
                response_schema=_response_schema(registry),
            )
        except PlanningModelError as exc:
            provider_code = (
                exc.code
                if exc.code in _PROVIDER_ERROR_CODES
                else "PLANNING_PROVIDER_ERROR"
            )
            diagnostics.append(
                context.diagnostic(
                    PlanningDiagnosticStage.PROVIDER,
                    provider_code,
                    "failed",
                    reason_code="provider_call_failed",
                )
            )
            raise PlannerError(
                provider_code,
                "Planning provider request failed.",
                category=ErrorCategory.ENVIRONMENT_ERROR,
                diagnostics=tuple(diagnostics),
            ) from exc
        except Exception as exc:
            diagnostics.append(
                context.diagnostic(
                    PlanningDiagnosticStage.PROVIDER,
                    "PLANNING_PROVIDER_ERROR",
                    "failed",
                    reason_code="provider_call_failed",
                )
            )
            raise PlannerError(
                "PLANNING_PROVIDER_ERROR",
                "Planning model failed to produce a response.",
                category=ErrorCategory.ENVIRONMENT_ERROR,
                diagnostics=tuple(diagnostics),
            ) from exc

        response_byte_count = _response_byte_count(response)
        diagnostics.append(
            context.diagnostic(
                PlanningDiagnosticStage.PROVIDER,
                "PROVIDER_RESPONSE_RECEIVED",
                "succeeded",
                response_byte_count=response_byte_count,
            )
        )
        try:
            steps = _parse_response(response, request)
        except PlannerError as exc:
            _append_preceding_success_diagnostics(
                diagnostics,
                context,
                exc.diagnostic_stage,
                response_byte_count=response_byte_count,
            )
            stage = exc.diagnostic_stage or PlanningDiagnosticStage.SCHEMA
            fields = exc.diagnostic_fields
            diagnostics.append(
                context.diagnostic(
                    stage,
                    exc.code,
                    "rejected" if stage is PlanningDiagnosticStage.UNSUPPORTED else "failed",
                    response_byte_count=response_byte_count,
                    step_index=_optional_nonnegative_int(fields.get("step_index")),
                    argument_name=_known_argument_name(
                        registry, fields.get("argument_name")
                    ),
                    input_name=_known_input_name(
                        request, fields.get("input_name")
                    ),
                    producer_step_index=_optional_nonnegative_int(
                        fields.get("producer_step_index")
                    ),
                    output_key=_known_output_key(
                        registry, fields.get("output_key")
                    ),
                    tool_name=_known_tool_name(registry, fields.get("tool_name")),
                    reason_code=exc.diagnostic_reason_code,
                )
            )
            raise PlannerError(
                exc.code,
                (
                    "Planning model classified the request as unsupported."
                    if stage is PlanningDiagnosticStage.UNSUPPORTED
                    else str(exc)
                ),
                category=exc.category,
                diagnostics=tuple(diagnostics),
            ) from exc

        _append_successful_response_diagnostics(
            diagnostics,
            context,
            response_byte_count=response_byte_count,
        )
        try:
            plan = AgentPlan(
                plan_id=_plan_id(request, steps),
                request_id=request.request_id,
                planner_name=self._name,
                steps=steps,
            )
        except (TypeError, ValueError) as exc:
            failure = _invalid_output(
                "Planning response violates the AgentPlan structure.",
                code="PLANNER_STRUCTURE_INVALID",
                stage=PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
                reason_code="plan_structure_invalid",
            )
            diagnostics.append(
                context.diagnostic(
                    PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
                    failure.code,
                    "failed",
                    response_byte_count=response_byte_count,
                    reason_code=failure.diagnostic_reason_code,
                )
            )
            raise PlannerError(
                failure.code,
                str(failure),
                category=failure.category,
                diagnostics=tuple(diagnostics),
            ) from exc
        diagnostics.append(
            context.diagnostic(
                PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
                "DEPENDENCY_REFERENCE_STRUCTURE_VALID",
                "succeeded",
                response_byte_count=response_byte_count,
            )
        )
        diagnostics.append(
            context.diagnostic(
                PlanningDiagnosticStage.CANDIDATE,
                "CANDIDATE_PLAN_CONSTRUCTED",
                "succeeded",
                response_byte_count=response_byte_count,
                candidate_constructed=True,
            )
        )
        return DiagnosedPlanningAttempt(plan, context, tuple(diagnostics))


def _response_byte_count(response: object) -> int | None:
    if not isinstance(response, str):
        return None
    try:
        return len(response.encode("utf-8"))
    except UnicodeEncodeError:
        return None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _known_argument_name(registry: ToolRegistry, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    known = {
        name
        for tool_name in registry.names()
        for name in (
            *registry.get(tool_name).required_arguments,
            *registry.get(tool_name).optional_arguments,
        )
    }
    return value if value in known else None


def _known_input_name(request: AgentRequest, value: object) -> str | None:
    return value if isinstance(value, str) and value in request.inputs else None


def _known_output_key(registry: ToolRegistry, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    known = {
        field
        for tool_name in registry.names()
        for field in registry.get(tool_name).result_contract.required_fields
    }
    return value if value in known else None


def _known_tool_name(registry: ToolRegistry, value: object) -> str | None:
    return value if isinstance(value, str) and registry.contains(value) else None


def _append_preceding_success_diagnostics(
    diagnostics: list[PlanningDiagnostic],
    context: PlanningDiagnosticContext,
    failed_stage: PlanningDiagnosticStage | None,
    *,
    response_byte_count: int | None,
) -> None:
    if failed_stage is PlanningDiagnosticStage.PARSE:
        return
    diagnostics.append(
        context.diagnostic(
            PlanningDiagnosticStage.PARSE,
            "JSON_PARSE_SUCCEEDED",
            "succeeded",
            response_byte_count=response_byte_count,
        )
    )
    if failed_stage is PlanningDiagnosticStage.SCHEMA:
        return
    diagnostics.append(
        context.diagnostic(
            PlanningDiagnosticStage.SCHEMA,
            "WIRE_SCHEMA_VALID",
            "succeeded",
            response_byte_count=response_byte_count,
        )
    )
    if failed_stage in {
        PlanningDiagnosticStage.ARGUMENT_BINDING,
        PlanningDiagnosticStage.UNSUPPORTED,
    }:
        return
    diagnostics.append(
        context.diagnostic(
            PlanningDiagnosticStage.ARGUMENT_BINDING,
            "ARGUMENT_BINDINGS_CONSTRUCTED",
            "succeeded",
            response_byte_count=response_byte_count,
        )
    )


def _append_successful_response_diagnostics(
    diagnostics: list[PlanningDiagnostic],
    context: PlanningDiagnosticContext,
    *,
    response_byte_count: int | None,
) -> None:
    _append_preceding_success_diagnostics(
        diagnostics,
        context,
        PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
        response_byte_count=response_byte_count,
    )


__all__ = ["LLMPlanner"]
