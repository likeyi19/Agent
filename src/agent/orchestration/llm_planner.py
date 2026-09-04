"""Strict provider-neutral LLM planner for the registered scientific tools."""

from __future__ import annotations

from enum import Enum
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Mapping, Protocol

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
    PlanningAttemptKind,
    PlanningDiagnostic,
    PlanningDiagnosticContext,
    PlanningDiagnosticStage,
)
from .planning_model import PlanningModel, PlanningModelError, PlanningModelProfile
from .planning_recovery import (
    CandidateValidator,
    PlanningCancellationCheck,
    PlanningDiagnosticSink,
    PlanningRepairContext,
    PlanningRecoveryCoordinator,
    PlanningRecoveryPolicy,
    PlanningSleeper,
    RecoveredPlanningAttempt,
)
from .registry import ArgumentSpec, ResultFieldPlanningSemantics, ToolRegistry


_SCHEMA_VERSION = 3
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
        "PROVIDER_REQUEST_TOO_LARGE",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "PROVIDER_CONNECTION_FAILED",
        "PROVIDER_UNAVAILABLE",
        "PROVIDER_COMPLETION_INCOMPLETE",
        "PROVIDER_REFUSED",
        "PLANNING_PROVIDER_DEPENDENCY_MISSING",
        "PLANNING_PROVIDER_CONFIGURATION_FAILED",
    }
)
_FACTORY_CONFIGURATION_CODES = frozenset(
    {
        "PLANNING_MODEL_PROFILE_DISABLED",
        "PLANNING_MODEL_CAPABILITY_UNSUPPORTED",
        "PLANNING_MODEL_PROVIDER_UNKNOWN",
        "PLANNING_PROVIDER_DEPENDENCY_MISSING",
        "PLANNING_PROVIDER_CONFIGURATION_FAILED",
    }
)


class PlanningWireMode(str, Enum):
    """Explicit planning wire contract selected by an LLMPlanner caller."""

    V3 = "v3"
    V4 = "v4"

    @property
    def schema_version(self) -> int:
        return 3 if self is PlanningWireMode.V3 else 4


_SEMANTIC_COMPILER_STAGES = {
    "UNKNOWN_TOOL": PlanningDiagnosticStage.TOOL_SELECTION,
    "MISSING_REQUEST_SOURCE_MEMBER": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "OVERLAPPING_SOURCE_MEMBERS": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "INVALID_REQUEST_BINDING": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "UNKNOWN_TARGET_PORT": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "DUPLICATE_TARGET_SOURCE": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "UNKNOWN_REQUEST_INPUT": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "UNAUTHORIZED_REQUEST_INPUT": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "AMBIGUOUS_REQUEST_INPUT": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "AMBIGUOUS_SOURCE_SELECTION": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "AMBIGUOUS_OPTIONAL_INPUT_SCOPE": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "MISSING_REQUIRED_SOURCE": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "MISSING_REQUIRED_BINDING": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "INVALID_COMPILED_ARGUMENTS": PlanningDiagnosticStage.ARGUMENT_BINDING,
    "WRONG_SOURCE_PORT": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "AMBIGUOUS_SOURCE_PORT": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "ZERO_VALID_CHANNELS": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "MULTIPLE_VALID_CHANNELS": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "UNKNOWN_DEPENDENCY": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "BROKEN_BRANCH_LINEAGE": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "CONFLICTING_BRANCH_LINEAGE": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
    "INVALID_SEMANTIC_GRAPH": PlanningDiagnosticStage.DEPENDENCY_REFERENCE,
}
_SEMANTIC_COMPILER_REASONS = {
    "MISSING_REQUEST_SOURCE_MEMBER": "incomplete_request_bundle",
    "OVERLAPPING_SOURCE_MEMBERS": "ambiguous_target_source",
    "INVALID_REQUEST_BINDING": "invalid_request_source",
    "UNKNOWN_TARGET_PORT": "unknown_target_port",
    "DUPLICATE_TARGET_SOURCE": "duplicate_target_source",
    "UNKNOWN_REQUEST_INPUT": "unknown_request_source",
    "UNAUTHORIZED_REQUEST_INPUT": "unauthorized_request_source",
    "AMBIGUOUS_REQUEST_INPUT": "ambiguous_request_source",
    "AMBIGUOUS_SOURCE_SELECTION": "ambiguous_target_source",
    "AMBIGUOUS_OPTIONAL_INPUT_SCOPE": "ambiguous_optional_input_scope",
    "MISSING_REQUIRED_SOURCE": "missing_semantic_source",
    "MISSING_REQUIRED_BINDING": "missing_semantic_binding",
    "INVALID_COMPILED_ARGUMENTS": "invalid_compiled_semantics",
    "WRONG_SOURCE_PORT": "producer_channel_incompatible",
    "AMBIGUOUS_SOURCE_PORT": "ambiguous_source_port",
    "ZERO_VALID_CHANNELS": "producer_channel_incompatible",
    "MULTIPLE_VALID_CHANNELS": "ambiguous_source_port",
    "UNKNOWN_DEPENDENCY": "unknown_producer_step",
    "BROKEN_BRANCH_LINEAGE": "branch_lineage_mismatch",
    "CONFLICTING_BRANCH_LINEAGE": "branch_lineage_conflict",
    "INVALID_SEMANTIC_GRAPH": "invalid_semantic_graph",
    "UNKNOWN_TOOL": "unknown_tool",
}


class PlanningModelFactoryResolver(Protocol):
    """Structural view of the accepted non-routing model factory registry."""

    @property
    def provider_ids(self) -> tuple[str, ...]: ...

    def create(self, profile: PlanningModelProfile) -> PlanningModel: ...


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


def _closed_object_schema(
    properties: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": tuple(properties),
        "additionalProperties": False,
    }


def _binding_schema(
    request: AgentRequest,
    argument_spec: ArgumentSpec,
    *,
    optional: bool = False,
) -> Mapping[str, JsonValue]:
    has_inputs = bool(request.inputs)
    if not has_inputs and not argument_spec.allow_step_output_ref:
        raise PlannerError(
            "PLANNER_CATALOG_INVALID",
            "A registered tool argument has no available binding source.",
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
        )
    if optional:
        definition = (
            "input_or_ref_or_null"
            if has_inputs and argument_spec.allow_step_output_ref
            else "input_or_null"
            if has_inputs
            else "ref_or_null"
        )
    else:
        definition = (
            "input_or_ref"
            if has_inputs and argument_spec.allow_step_output_ref
            else "input"
            if has_inputs
            else "ref"
        )
    return {"$ref": f"#/$defs/{definition}"}


def _binding_definitions(request: AgentRequest) -> Mapping[str, JsonValue]:
    input_names = tuple(sorted(request.inputs))
    definitions: dict[str, JsonValue] = {
        "ref": _closed_object_schema(
            {
                "binding_type": {"type": "string", "enum": ("ref",)},
                "ref_step_id": {"type": "string"},
                "ref_output_key": {"type": "string"},
            }
        ),
        "ref_or_null": {
            "anyOf": (
                {"$ref": "#/$defs/ref"},
                {"type": "null"},
            )
        },
    }
    if input_names:
        definitions.update(
            {
                "input": _closed_object_schema(
                    {
                        "binding_type": {
                            "type": "string",
                            "enum": ("input",),
                        },
                        "input_name": {
                            "type": "string",
                            "enum": input_names,
                        },
                    }
                ),
                "input_or_ref": {
                    "anyOf": (
                        {"$ref": "#/$defs/input"},
                        {"$ref": "#/$defs/ref"},
                    )
                },
                "input_or_null": {
                    "anyOf": (
                        {"$ref": "#/$defs/input"},
                        {"type": "null"},
                    )
                },
                "input_or_ref_or_null": {
                    "anyOf": (
                        {"$ref": "#/$defs/input"},
                        {"$ref": "#/$defs/ref"},
                        {"type": "null"},
                    )
                },
            }
        )
    return definitions


def _tool_step_schema(
    tool_name: str,
    request: AgentRequest,
    registry: ToolRegistry,
) -> Mapping[str, JsonValue]:
    spec = registry.get(tool_name)
    argument_properties: dict[str, JsonValue] = {}
    for argument_name, argument_spec in sorted(spec.required_arguments.items()):
        argument_properties[argument_name] = _binding_schema(request, argument_spec)
    for argument_name, argument_spec in sorted(spec.optional_arguments.items()):
        argument_properties[argument_name] = _binding_schema(
            request,
            argument_spec,
            optional=True,
        )
    return _closed_object_schema(
        {
            "step_id": {"type": "string"},
            "tool_name": {"type": "string", "enum": (tool_name,)},
            "arguments": _closed_object_schema(argument_properties),
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "description": {"type": ("string", "null")},
        }
    )


def _response_schema(
    registry: ToolRegistry,
    request: AgentRequest,
) -> Mapping[str, JsonValue]:
    tool_names = registry.names()
    if not tool_names:
        raise PlannerError(
            "PLANNER_CATALOG_INVALID",
            "Planning requires at least one registered tool.",
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
        )
    step_schema: Mapping[str, JsonValue] = {
        "anyOf": tuple(
            _tool_step_schema(tool_name, request, registry)
            for tool_name in tool_names
        )
    }
    root_properties: Mapping[str, JsonValue] = {
        "schema_version": {"type": "integer", "enum": (_SCHEMA_VERSION,)},
        "status": {"type": "string", "enum": ("plan", "unsupported")},
        "steps": {"type": "array", "items": step_schema},
        "reason": {"type": ("string", "null")},
    }
    schema = dict(_closed_object_schema(root_properties))
    schema["$defs"] = _binding_definitions(request)
    return schema


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
    if isinstance(value, (list, tuple)):
        return tuple(_catalog_choice(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PlannerError(
                "PLANNER_CATALOG_INVALID",
                "A registered tool mapping cannot be represented safely for planning.",
                category=ErrorCategory.INTERNAL_AGENT_ERROR,
            )
        return {
            key: _catalog_choice(value[key])
            for key in sorted(value)
        }
    raise PlannerError(
        "PLANNER_CATALOG_INVALID",
        "A registered tool choice cannot be represented safely for planning.",
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
    )


def _argument_catalog(
    spec: ArgumentSpec,
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
    planning = spec.planning
    if planning is not None:
        metadata.update(
            {
                "description": planning.description,
                "source_eligibility": planning.source_eligibility.value,
                "accepted_artifact_kinds": tuple(
                    kind.value for kind in planning.accepted_artifact_kinds
                ),
                "scientific_parameter": planning.scientific_parameter,
            }
        )
        if planning.provenance_role is not None:
            metadata["provenance_role"] = planning.provenance_role.value
        if planning.conditional_note is not None:
            metadata["conditional_note"] = planning.conditional_note
    if spec.choices:
        metadata["choices"] = tuple(_catalog_choice(value) for value in spec.choices)
    if (
        planning is not None
        and planning.default_when_omitted is not inspect.Parameter.empty
    ):
        default_when_omitted = planning.default_when_omitted
        metadata["default_when_omitted"] = (
            "registered_tool_default"
            if default_when_omitted is not None
            and any(issubclass(value_type, Path) for value_type in spec.accepted_types)
            else _catalog_choice(default_when_omitted)
        )
    return metadata


def _result_field_catalog(
    accepted_types: tuple[type, ...],
    planning: ResultFieldPlanningSemantics | None,
) -> Mapping[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "json_types": tuple(
            dict.fromkeys(
                _python_type_to_json_type(value_type)
                for value_type in accepted_types
            )
        ),
        "downstream_bindable": False,
    }
    if planning is not None:
        metadata["description"] = planning.description
        metadata["downstream_bindable"] = planning.downstream_bindable
        if planning.artifact_kind is not None:
            metadata["artifact_kind"] = planning.artifact_kind.value
        if planning.provenance_role is not None:
            metadata["provenance_role"] = planning.provenance_role.value
    return metadata


def _sanitized_catalog(registry: ToolRegistry) -> tuple[Mapping[str, JsonValue], ...]:
    tools: list[Mapping[str, JsonValue]] = []
    for tool_name in registry.names():
        spec = registry.get(tool_name)
        planning = spec.planning
        tools.append(
            {
                "name": spec.name,
                "planning_role": (
                    planning.role.value if planning is not None else "operation"
                ),
                "description": (
                    planning.description
                    if planning is not None
                    else "Registered scientific tool."
                ),
                "required_arguments": {
                    name: _argument_catalog(argument_spec)
                    for name, argument_spec in sorted(spec.required_arguments.items())
                },
                "optional_arguments": {
                    name: _argument_catalog(argument_spec)
                    for name, argument_spec in sorted(spec.optional_arguments.items())
                },
                "result_fields": {
                    name: _result_field_catalog(
                        accepted_types,
                        spec.result_contract.planning_fields.get(name),
                    )
                    for name, accepted_types in sorted(
                        spec.result_contract.required_fields.items()
                    )
                },
                "conditional_notes": (
                    planning.conditional_notes if planning is not None else ()
                ),
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


def _prompt_catalog(registry: ToolRegistry) -> Mapping[str, JsonValue]:
    """Keep scientific semantics while omitting syntax duplicated by the schema."""

    source_codes = {
        "request_input_only": "i",
        "request_input_or_upstream_result": "ir",
        "upstream_result_only": "r",
    }
    tools: dict[str, JsonValue] = {}
    for tool in _sanitized_catalog(registry):
        arguments: dict[str, JsonValue] = {}
        for section_name in ("required_arguments", "optional_arguments"):
            section = tool[section_name]
            assert isinstance(section, Mapping)
            for name, raw_metadata in section.items():
                assert isinstance(name, str) and isinstance(raw_metadata, Mapping)
                source = raw_metadata.get(
                    "source_eligibility",
                    "request_input_or_upstream_result"
                    if raw_metadata.get("allows_step_output_ref") is True
                    else "request_input_only",
                )
                assert isinstance(source, str) and source in source_codes
                options: dict[str, JsonValue] = {}
                optional_fields = {
                    "accepted_artifact_kinds": "a",
                    "provenance_role": "p",
                    "conditional_note": "n",
                    "choices": "c",
                    "default_when_omitted": "d",
                }
                for source_name, compact_name in optional_fields.items():
                    value = raw_metadata.get(source_name)
                    if value not in (None, (), False):
                        options[compact_name] = value
                if raw_metadata.get("scientific_parameter") is True:
                    options["s"] = True
                arguments[name] = (
                    raw_metadata.get(
                        "description", "Registered tool argument."
                    ),
                    source_codes[source],
                    options,
                )

        raw_results = tool["result_fields"]
        assert isinstance(raw_results, Mapping)
        result_semantics: dict[str, JsonValue] = {}
        for name, raw_metadata in raw_results.items():
            assert isinstance(name, str) and isinstance(raw_metadata, Mapping)
            if "description" in raw_metadata:
                result_semantics[name] = (
                    raw_metadata["description"],
                    raw_metadata.get("artifact_kind"),
                    raw_metadata.get("provenance_role"),
                    raw_metadata.get("downstream_bindable") is True,
                )

        name = tool["name"]
        assert isinstance(name, str)
        tools[name] = (
            tool["planning_role"],
            tool["description"],
            arguments,
            result_semantics,
            tool["conditional_notes"],
        )
    return tools


def _build_prompt(
    request: AgentRequest,
    registry: ToolRegistry,
    *,
    repair_context: PlanningRepairContext | None = None,
    failover_context: PlanningRepairContext | None = None,
) -> str:
    prompt_payload = {
        "planning_catalog_semantic_version": 1,
        "instructions": (
            "Return one complete JSON decision matching the schema; select only "
            "needed registered tools.",
            "Bind executable values only from semantically matching named request "
            "inputs or compatible upstream results; never match by JSON type alone.",
            "Never invent paths, species, metadata keys, parameters, artifacts, or "
            "other executable values.",
            "Preserve every supplied scientific parameter used by a selected tool; "
            "null means an unused optional argument and applies the tool default.",
            "Use upstream references and dependencies only for actual artifact, data, "
            "or provenance flow, naming the producer step and result field.",
            "Preserve reference, query, and ground-truth provenance; independent "
            "branches need no fixed order.",
            "Do not substitute another input or scientific operation when required "
            "information or capability is absent.",
            "Return unsupported with no steps for unsupported, ambiguous, conflicting, "
            "or indispensably incomplete requests.",
        ),
        "request": {
            "prompt": request.prompt,
            "available_input_names": tuple(sorted(request.inputs)),
        },
        "catalog_format": {
            "tool": (
                "role",
                "meaning",
                "arguments",
                "bindable_results",
                "notes",
            ),
            "argument": ("meaning", "source", "options"),
            "result": ("meaning", "artifact", "provenance", "bindable"),
            "source": {
                "i": "request_input_only",
                "ir": "request_input_or_upstream_result",
                "r": "upstream_result_only",
            },
            "options": {
                "a": "accepted_artifact_kinds",
                "p": "provenance_role",
                "n": "conditional_note",
                "c": "choices",
                "d": "default_when_omitted",
                "s": "scientific_parameter",
            },
        },
        "tools": _prompt_catalog(registry),
    }
    if repair_context is not None:
        prompt_payload["repair"] = {
            "instruction": (
                "The previous candidate planning decision was rejected at the "
                "specified structural stage. Generate a complete new planning "
                "decision from the original request and contracts. Correct the "
                "diagnosed planning error. Do not patch or quote the previous "
                "output. Preserve all original request-supplied scientific "
                "parameters and produce only a complete schema-v3 decision."
            ),
            "diagnostic": repair_context.to_prompt_dict(),
        }
    if failover_context is not None:
        prompt_payload["failover"] = {
            "instruction": (
                "A configured secondary planning model is making the final "
                "planning attempt after an objectively invalid candidate. "
                "Generate a complete new planning decision from the original "
                "request and contracts. Correct the diagnosed structural error. "
                "Do not patch or quote any previous output. Preserve all original "
                "request-supplied scientific parameters and produce only a "
                "complete schema-v3 decision."
            ),
            "diagnostic": failover_context.to_prompt_dict(),
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


def _parse_json_response(
    response: object,
    *,
    tree_limit_stage: PlanningDiagnosticStage | None = None,
) -> dict[str, object]:
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
    _validate_tree_limits(parsed, stage=tree_limit_stage)
    return parsed


def _validate_tree_limits(
    value: object,
    *,
    stage: PlanningDiagnosticStage | None = None,
) -> None:
    node_count = 0

    def visit(nested: object, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_RESPONSE_NODES:
            raise _invalid_output(
                "Planning model response contains too many values.",
                stage=stage,
                reason_code=(
                    "response_node_limit_exceeded" if stage is not None else None
                ),
            )
        if depth > _MAX_RESPONSE_DEPTH:
            raise _invalid_output(
                "Planning model response is nested too deeply.",
                stage=stage,
                reason_code=(
                    "response_depth_limit_exceeded" if stage is not None else None
                ),
            )
        if isinstance(nested, dict):
            for key, child in nested.items():
                if len(key) > _MAX_DESCRIPTION_LENGTH:
                    raise _invalid_output(
                        "Planning model response contains an oversized key.",
                        stage=stage,
                        reason_code=(
                            "response_key_too_large" if stage is not None else None
                        ),
                    )
                visit(child, depth + 1)
        elif isinstance(nested, list):
            for child in nested:
                visit(child, depth + 1)
        elif isinstance(nested, str) and len(nested) > _MAX_RESPONSE_BYTES:
            raise _invalid_output(
                "Planning model response contains an oversized string.",
                stage=stage,
                reason_code=(
                    "response_string_too_large" if stage is not None else None
                ),
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
    argument_name: str,
    argument_spec: ArgumentSpec,
) -> tuple[str, object]:
    diagnostic_fields = {
        "step_index": step_index,
        "argument_name": argument_name,
    }
    if not isinstance(binding, dict):
        raise _invalid_output(
            f"{context} must be a fixed argument-binding object.",
            code="PLANNER_BINDING_INVALID",
            reason_code="binding_not_object",
            diagnostic_fields=diagnostic_fields,
        )
    binding_type = binding.get("binding_type")
    if binding_type == "input":
        if set(binding) != {"binding_type", "input_name"}:
            raise _invalid_output(
                f"{context} input binding fields are invalid.",
                code="PLANNER_BINDING_INVALID",
                reason_code="input_binding_fields_invalid",
                diagnostic_fields=diagnostic_fields,
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
                    "input_name": input_name,
                },
            )
        return argument_name, request.inputs[input_name]
    if binding_type == "ref":
        if set(binding) != {"binding_type", "ref_step_id", "ref_output_key"}:
            raise _invalid_output(
                f"{context} reference binding fields are invalid.",
                code="PLANNER_BINDING_INVALID",
                reason_code="reference_binding_fields_invalid",
                diagnostic_fields=diagnostic_fields,
            )
        if not argument_spec.allow_step_output_ref:
            raise _invalid_output(
                f"{context} does not permit a step-output reference.",
                code="PLANNER_BINDING_INVALID",
                reason_code="reference_binding_not_allowed",
                diagnostic_fields=diagnostic_fields,
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
        diagnostic_fields=diagnostic_fields,
    )


def _parse_plan_steps(
    raw_steps: object,
    request: AgentRequest,
    registry: ToolRegistry,
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
        if not registry.contains(tool_name):
            raise PlannerError(
                "UNKNOWN_TOOL",
                "Planning response selected a tool outside the executable allowlist.",
                category=ErrorCategory.INTERNAL_AGENT_ERROR,
                diagnostic_stage=PlanningDiagnosticStage.TOOL_SELECTION,
                diagnostic_reason_code="unknown_tool",
                diagnostic_fields={"step_index": index, "tool_name": tool_name},
            )
        tool_spec = registry.get(tool_name)
        raw_arguments = raw_step["arguments"]
        if not isinstance(raw_arguments, dict):
            raise _invalid_output(
                f"{context}.arguments must be an object.",
                code="PLANNER_BINDING_INVALID",
                reason_code="arguments_not_object",
                diagnostic_fields={"step_index": index},
            )
        required_arguments = set(tool_spec.required_arguments)
        optional_arguments = set(tool_spec.optional_arguments)
        expected_arguments = required_arguments.union(optional_arguments)
        supplied_arguments = set(raw_arguments)
        if supplied_arguments != expected_arguments:
            missing_required = sorted(required_arguments.difference(supplied_arguments))
            missing_optional = sorted(optional_arguments.difference(supplied_arguments))
            unknown = sorted(supplied_arguments.difference(expected_arguments))
            if missing_required:
                code = "INVALID_TOOL_ARGUMENTS"
                reason_code = "missing_tool_argument"
                argument_name = missing_required[0]
            elif unknown:
                code = "INVALID_TOOL_ARGUMENTS"
                reason_code = "unknown_tool_argument"
                argument_name = unknown[0]
            else:
                code = "PLANNER_BINDING_INVALID"
                reason_code = "missing_nullable_optional_argument"
                argument_name = missing_optional[0]
            raise _invalid_output(
                f"{context}.arguments must contain exactly the selected tool's "
                "registered argument keys; unused optional arguments must be null.",
                code=code,
                stage=PlanningDiagnosticStage.ARGUMENT_BINDING,
                reason_code=reason_code,
                diagnostic_fields={
                    "step_index": index,
                    "argument_name": argument_name,
                },
            )
        arguments: dict[str, object] = {}
        ordered_argument_specs = (
            *tool_spec.required_arguments.items(),
            *tool_spec.optional_arguments.items(),
        )
        for argument_name, argument_spec in ordered_argument_specs:
            binding = raw_arguments[argument_name]
            if argument_name in optional_arguments and binding is None:
                continue
            if binding is None:
                raise _invalid_output(
                    f"{context}.arguments.{argument_name} cannot be null.",
                    code="PLANNER_BINDING_INVALID",
                    reason_code="required_argument_null",
                    diagnostic_fields={
                        "step_index": index,
                        "argument_name": argument_name,
                    },
                )
            argument_name, argument_value = _parse_argument_binding(
                binding,
                request,
                context=f"{context}.arguments.{argument_name}",
                step_index=index,
                argument_name=argument_name,
                argument_spec=argument_spec,
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


def _parse_response(
    response: object,
    request: AgentRequest,
    registry: ToolRegistry,
) -> tuple[PlanStep, ...]:
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
    return _parse_plan_steps(payload["steps"], request, registry)


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


def _semantic_v4_plan(
    response: object,
    request: AgentRequest,
    registry: ToolRegistry,
    compiler_contract: object,
    *,
    planner_name: str,
) -> AgentPlan:
    from .semantic_compiler import (
        SemanticPlanCompileError,
        compile_semantic_plan,
    )
    from .semantic_wire_v4 import parse_semantic_wire_v4

    try:
        candidate = parse_semantic_wire_v4(response, request, registry)
    except PlannerError as exc:
        if exc.code != "UNSUPPORTED_REQUEST" or exc.diagnostic_stage is not None:
            raise
        raise PlannerError(
            exc.code,
            str(exc),
            category=exc.category,
            diagnostic_stage=PlanningDiagnosticStage.UNSUPPORTED,
            diagnostic_reason_code="explicit_unsupported_response",
        ) from exc
    try:
        return compile_semantic_plan(
            request,
            candidate,
            registry,
            compiler_contract,
            planner_name=planner_name,
        )
    except SemanticPlanCompileError as exc:
        raise PlannerError(
            exc.code,
            str(exc),
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
            diagnostic_stage=_SEMANTIC_COMPILER_STAGES.get(
                exc.code, PlanningDiagnosticStage.CANDIDATE
            ),
            diagnostic_reason_code=_SEMANTIC_COMPILER_REASONS.get(
                exc.code, "semantic_compilation_failed"
            ),
            diagnostic_fields=exc.diagnostic_fields,
        ) from exc


class LLMPlanner:
    """Convert one strict planning-model response into existing plan contracts."""

    def __init__(
        self,
        model: PlanningModel,
        *,
        wire_mode: PlanningWireMode = PlanningWireMode.V3,
        profile: PlanningModelProfile | None = None,
        recovery_policy: PlanningRecoveryPolicy | None = None,
        retry_sleeper: PlanningSleeper | None = None,
        recovery_profiles: tuple[PlanningModelProfile, ...] = (),
        model_factory_registry: PlanningModelFactoryResolver | None = None,
    ) -> None:
        if not callable(getattr(model, "complete", None)):
            raise TypeError("`model` must provide a callable complete() method.")
        if not isinstance(wire_mode, PlanningWireMode):
            raise TypeError("`wire_mode` must be a PlanningWireMode.")
        if profile is not None and not isinstance(profile, PlanningModelProfile):
            raise TypeError("`profile` must be a PlanningModelProfile or None.")
        if profile is not None and not profile.enabled:
            raise ValueError("`profile` must be enabled.")
        if profile is not None and not profile.supports_structured_output:
            raise ValueError("`profile` must support structured output.")
        if not isinstance(recovery_profiles, tuple) or not all(
            isinstance(candidate, PlanningModelProfile)
            for candidate in recovery_profiles
        ):
            raise TypeError(
                "`recovery_profiles` must be a tuple of PlanningModelProfile values."
            )
        if len(recovery_profiles) > 1:
            raise ValueError("At most one secondary recovery profile is supported.")
        if recovery_profiles and profile is None:
            raise ValueError("A secondary profile requires an explicit primary profile.")
        if model_factory_registry is not None and not recovery_profiles:
            raise ValueError(
                "A model factory registry requires one configured recovery profile."
            )
        if recovery_profiles:
            secondary = recovery_profiles[0]
            if not secondary.enabled:
                raise ValueError("The secondary recovery profile must be enabled.")
            if not secondary.supports_structured_output:
                raise ValueError(
                    "The secondary recovery profile must support structured output."
                )
            assert profile is not None
            duplicate_identity = (
                secondary.provider_id,
                secondary.model_id,
                secondary.request_timeout_seconds,
            ) == (
                profile.provider_id,
                profile.model_id,
                profile.request_timeout_seconds,
            )
            if secondary.profile_id == profile.profile_id or duplicate_identity:
                raise ValueError(
                    "The secondary recovery profile must differ from the primary."
                )
            if model_factory_registry is None:
                raise ValueError(
                    "A secondary recovery profile requires a model factory registry."
                )
            provider_ids = getattr(model_factory_registry, "provider_ids", None)
            if (
                not isinstance(provider_ids, tuple)
                or not all(isinstance(value, str) for value in provider_ids)
                or secondary.provider_id not in provider_ids
                or not callable(getattr(model_factory_registry, "create", None))
            ):
                raise ValueError(
                    "The secondary recovery profile provider is not registered."
                )
        self._model = model
        self._model_id = _sanitize_model_id(getattr(model, "model_id", None))
        self._wire_mode = wire_mode
        self._profile = profile
        self._recovery_profiles = recovery_profiles
        self._model_factory_registry = model_factory_registry
        base_name = (
            f"llm:{self._model_id}"
            if profile is None
            else f"llm-profile:{profile.profile_id}"
        )
        self._name = (
            base_name
            if wire_mode is PlanningWireMode.V3
            else f"{base_name}:wire-v4"
        )
        coordinator_kwargs = {}
        if retry_sleeper is not None:
            coordinator_kwargs["sleeper"] = retry_sleeper
        self._recovery = PlanningRecoveryCoordinator(
            recovery_policy, **coordinator_kwargs
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

    @property
    def wire_mode(self) -> PlanningWireMode:
        return self._wire_mode

    @property
    def recovery_policy(self) -> PlanningRecoveryPolicy:
        return self._recovery.policy

    @property
    def recovery_profiles(self) -> tuple[PlanningModelProfile, ...]:
        return self._recovery_profiles

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        """Return the accepted candidate while preserving the Planner protocol."""

        return self.plan_with_diagnostics(request, registry).plan

    def plan_with_diagnostics(
        self,
        request: AgentRequest,
        registry: ToolRegistry,
        *,
        attempt_kind: PlanningAttemptKind = PlanningAttemptKind.INITIAL,
        logical_attempt_index: int = 1,
        provider_call_index: int = 1,
        repair_context: PlanningRepairContext | None = None,
    ) -> DiagnosedPlanningAttempt:
        """Construct one plan and sanitized diagnostics with exactly one model call."""

        if not isinstance(request, AgentRequest):
            raise TypeError("`request` must be an AgentRequest.")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")
        if attempt_kind is PlanningAttemptKind.REPAIR and repair_context is None:
            raise ValueError(
                "Repair attempts require one sanitized candidate-failure context."
            )
        if attempt_kind not in {
            PlanningAttemptKind.REPAIR,
            PlanningAttemptKind.FAILOVER,
        } and repair_context is not None:
            raise ValueError(
                "Candidate-failure context is allowed only for regeneration."
            )

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
            planning_wire_schema_version=self._wire_mode.schema_version,
            attempt_kind=attempt_kind,
            logical_attempt_index=logical_attempt_index,
            provider_call_index=provider_call_index,
        )
        diagnostics: list[PlanningDiagnostic] = []
        if attempt_kind is PlanningAttemptKind.REPAIR:
            assert repair_context is not None
            diagnostics.append(
                context.diagnostic(
                    PlanningDiagnosticStage.RECOVERY,
                    "PLAN_REPAIR_CALL_STARTED",
                    "started",
                    previous_failure_stage=(
                        repair_context.previous_failure_stage
                    ),
                    previous_failure_code=repair_context.previous_failure_code,
                    recovery_action="repair",
                    repair_used=True,
                    recovery_policy_fingerprint=self._recovery.policy.fingerprint,
                )
            )
        elif attempt_kind is PlanningAttemptKind.FAILOVER:
            diagnostics.append(
                context.diagnostic(
                    PlanningDiagnosticStage.RECOVERY,
                    "PROFILE_FAILOVER_CALL_STARTED",
                    "started",
                    previous_failure_stage=(
                        None
                        if repair_context is None
                        else repair_context.previous_failure_stage
                    ),
                    previous_failure_code=(
                        None
                        if repair_context is None
                        else repair_context.previous_failure_code
                    ),
                    recovery_action="failover",
                    failover_used=True,
                    recovery_policy_fingerprint=self._recovery.policy.fingerprint,
                )
            )
        diagnostics.append(
            context.diagnostic(
                PlanningDiagnosticStage.PROVIDER,
                "PROVIDER_CALL_STARTED",
                "started",
            )
        )
        semantic_contract: object | None = None
        if self._wire_mode is PlanningWireMode.V3:
            prompt = _build_prompt(
                request,
                registry,
                repair_context=(
                    repair_context
                    if attempt_kind is PlanningAttemptKind.REPAIR
                    else None
                ),
                failover_context=(
                    repair_context
                    if attempt_kind is PlanningAttemptKind.FAILOVER
                    and repair_context is not None
                    and repair_context.previous_failure_stage
                    is not PlanningDiagnosticStage.PROVIDER
                    else None
                ),
            )
            response_schema = _response_schema(registry, request)
        else:
            from .semantic_compiler import build_semantic_compiler_contract
            from .semantic_prompt import build_semantic_planning_prompt
            from .semantic_wire_v4 import build_semantic_wire_v4_schema

            try:
                semantic_contract = build_semantic_compiler_contract(registry)
            except ValueError as exc:
                raise PlannerError(
                    "PLANNER_CATALOG_INVALID",
                    "Semantic planner authority is incomplete or invalid.",
                    category=ErrorCategory.INTERNAL_AGENT_ERROR,
                ) from exc
            prompt = build_semantic_planning_prompt(
                request,
                registry,
                repair_context=(
                    repair_context
                    if attempt_kind is PlanningAttemptKind.REPAIR
                    else None
                ),
                failover_context=(
                    repair_context
                    if attempt_kind is PlanningAttemptKind.FAILOVER
                    and repair_context is not None
                    and repair_context.previous_failure_stage
                    is not PlanningDiagnosticStage.PROVIDER
                    else None
                ),
            )
            response_schema = build_semantic_wire_v4_schema(registry, request)
        try:
            response = self._model.complete(
                prompt=prompt,
                response_schema=response_schema,
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
                    retry_after_seconds=exc.retry_after_seconds,
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
        plan: AgentPlan | None = None
        steps: tuple[PlanStep, ...] | None = None
        try:
            if self._wire_mode is PlanningWireMode.V3:
                steps = _parse_response(response, request, registry)
            else:
                assert semantic_contract is not None
                plan = _semantic_v4_plan(
                    response,
                    request,
                    registry,
                    semantic_contract,
                    planner_name=self._name,
                )
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
                    step_id=fields.get("step_id"),
                    argument_name=_known_argument_name(
                        registry, fields.get("argument_name")
                    ),
                    input_name=_known_input_name(
                        request, fields.get("input_name")
                    ),
                    producer_step_index=_optional_nonnegative_int(
                        fields.get("producer_step_index")
                    ),
                    producer_step_id=fields.get("producer_step_id"),
                    output_key=_known_output_key(
                        registry, fields.get("output_key")
                    ),
                    tool_name=_known_tool_name(registry, fields.get("tool_name")),
                    target_port=fields.get("target_port"),
                    source_port=fields.get("source_port"),
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
            if plan is None:
                assert steps is not None
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
                (
                    "DEPENDENCY_REFERENCE_STRUCTURE_VALID"
                    if self._wire_mode is PlanningWireMode.V3
                    else "SEMANTIC_DEPENDENCIES_COMPILED"
                ),
                "succeeded",
                response_byte_count=response_byte_count,
            )
        )
        diagnostics.append(
            context.diagnostic(
                PlanningDiagnosticStage.CANDIDATE,
                (
                    "CANDIDATE_PLAN_CONSTRUCTED"
                    if self._wire_mode is PlanningWireMode.V3
                    else "SEMANTIC_CANDIDATE_COMPILED"
                ),
                "succeeded",
                response_byte_count=response_byte_count,
                candidate_constructed=True,
            )
        )
        return DiagnosedPlanningAttempt(plan, context, tuple(diagnostics))

    def plan_with_recovery(
        self,
        request: AgentRequest,
        registry: ToolRegistry,
        *,
        validate_candidate: CandidateValidator,
        diagnostic_sink: PlanningDiagnosticSink,
        should_cancel: PlanningCancellationCheck | None = None,
    ) -> RecoveredPlanningAttempt:
        """Acquire one Plan with bounded primary recovery and configured failover."""

        secondary_planner: LLMPlanner | None = None

        def configured_secondary(
            logical_index: int,
            provider_call_index: int,
            failure_context: PlanningRepairContext | None,
        ) -> None:
            nonlocal secondary_planner
            if secondary_planner is not None:
                return
            if (
                not self._recovery_profiles
                or self._model_factory_registry is None
            ):  # pragma: no cover - coordinator/configuration invariant
                raise RuntimeError("Failover was attempted without configuration.")
            secondary_profile = self._recovery_profiles[0]
            try:
                secondary_model = self._model_factory_registry.create(
                    secondary_profile
                )
                secondary_planner = LLMPlanner(
                    secondary_model,
                    wire_mode=self._wire_mode,
                    profile=secondary_profile,
                    recovery_policy=self._recovery.policy,
                )
            except Exception as exc:
                raw_code = getattr(exc, "code", None)
                code = (
                    raw_code
                    if raw_code in _FACTORY_CONFIGURATION_CODES
                    else "PLANNING_PROVIDER_CONFIGURATION_FAILED"
                )
                context = PlanningDiagnosticContext(
                    profile_id=secondary_profile.profile_id,
                    provider_id=secondary_profile.provider_id,
                    model_identity_digest=hashlib.sha256(
                        secondary_profile.model_id.encode("utf-8")
                    ).hexdigest(),
                    catalog_fingerprint=_catalog_fingerprint(registry),
                    offered_tool_names=registry.names(),
                    planning_wire_schema_version=self._wire_mode.schema_version,
                    attempt_kind=PlanningAttemptKind.FAILOVER,
                    logical_attempt_index=logical_index,
                    provider_call_index=provider_call_index,
                )
                diagnostic = context.diagnostic(
                    PlanningDiagnosticStage.PROVIDER,
                    code,
                    "failed",
                    previous_failure_stage=(
                        None
                        if failure_context is None
                        else failure_context.previous_failure_stage
                    ),
                    previous_failure_code=(
                        None
                        if failure_context is None
                        else failure_context.previous_failure_code
                    ),
                    reason_code="failover_model_construction_failed",
                    recovery_action="failover",
                    failover_used=True,
                    recovery_policy_fingerprint=self._recovery.policy.fingerprint,
                )
                raise PlannerError(
                    code,
                    "Configured secondary planning model could not be constructed.",
                    category=ErrorCategory.ENVIRONMENT_ERROR,
                    diagnostics=(diagnostic,),
                ) from exc
            return

        def attempt(
            kind: PlanningAttemptKind,
            logical_index: int,
            provider_call_index: int,
            repair_context: PlanningRepairContext | None,
        ) -> DiagnosedPlanningAttempt:
            planner = self
            if kind is PlanningAttemptKind.FAILOVER:
                if secondary_planner is None:  # pragma: no cover - invariant
                    raise RuntimeError("Failover model was not prepared.")
                planner = secondary_planner
            return planner.plan_with_diagnostics(
                request,
                registry,
                attempt_kind=kind,
                logical_attempt_index=logical_index,
                provider_call_index=provider_call_index,
                repair_context=repair_context,
            )

        return self._recovery.acquire(
            attempt=attempt,
            validate_candidate=validate_candidate,
            diagnostic_sink=diagnostic_sink,
            should_cancel=should_cancel,
            prepare_failover=(
                configured_secondary if self._recovery_profiles else None
            ),
        )


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
    if (
        context.planning_wire_schema_version == 4
        and failed_stage is PlanningDiagnosticStage.TOOL_SELECTION
    ):
        return
    if failed_stage in {
        PlanningDiagnosticStage.ARGUMENT_BINDING,
        PlanningDiagnosticStage.UNSUPPORTED,
    }:
        return
    diagnostics.append(
        context.diagnostic(
            PlanningDiagnosticStage.ARGUMENT_BINDING,
            (
                "ARGUMENT_BINDINGS_CONSTRUCTED"
                if context.planning_wire_schema_version == 3
                else "SEMANTIC_SOURCES_RESOLVED"
            ),
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


__all__ = ["LLMPlanner", "PlanningWireMode"]
