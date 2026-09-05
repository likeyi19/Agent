"""Experimental registry-driven semantic planning catalog and prompt."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from agent.schemas import AgentRequest, ErrorCategory, JsonValue

from .planner import PlannerError
from .registry import (
    ArgumentSpec,
    SemanticConsumerPortSpec,
    SemanticProducerPortSpec,
    ToolRegistry,
    ToolSpec,
)
from .semantic_wire_v4 import (
    SEMANTIC_WIRE_SCHEMA_VERSION,
    _planner_visible_tool_names,
)

if TYPE_CHECKING:
    from .planning_recovery import PlanningRepairContext


SEMANTIC_PLANNING_CATALOG_VERSION = 1
SEMANTIC_PLANNING_PROMPT_VERSION = 1
_SEMANTIC_REPAIR_CORRECTIONS = {
    "malformed_json": "regenerate_strict_wire_v4_json",
    "response_not_text": "regenerate_strict_wire_v4_json",
    "wire_schema_invalid": "regenerate_valid_wire_v4_structure",
    "unknown_tool": "select_offered_tool",
    "unknown_target_port": "select_authorized_target_port",
    "duplicate_target_source": "select_one_source_for_target_port",
    "unknown_request_source": "select_available_request_source",
    "unauthorized_request_source": "select_authorized_request_source",
    "ambiguous_request_source": "select_request_source_explicitly",
    "ambiguous_target_source": "select_one_semantic_source_explicitly",
    "ambiguous_optional_input_scope": "select_optional_source_scope_explicitly",
    "incomplete_request_bundle": "select_complete_request_source_bundle",
    "missing_semantic_source": "select_required_semantic_source",
    "producer_channel_incompatible": "select_compatible_producer_channel",
    "ambiguous_source_port": "select_source_port_explicitly",
    "unknown_producer_step": "select_existing_producer_step",
    "branch_lineage_mismatch": "select_producer_with_required_lineage",
    "branch_lineage_conflict": "select_consistent_branch_lineage",
    "invalid_semantic_graph": "produce_acyclic_semantic_dag",
}


def _catalog_error(message: str) -> PlannerError:
    return PlannerError(
        "PLANNER_CATALOG_INVALID",
        message,
        category=ErrorCategory.INTERNAL_AGENT_ERROR,
    )


def _json_type(value: JsonValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    raise _catalog_error("A request input has no safe JSON type description.")


def _safe_registered_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Path):
        return "registered_tool_default"
    if isinstance(value, (list, tuple)):
        return tuple(_safe_registered_value(item) for item in value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise _catalog_error(
                "Registered planning metadata contains a non-string mapping key."
            )
        return {
            key: _safe_registered_value(value[key])
            for key in sorted(value)
        }
    raise _catalog_error(
        "Registered planning metadata cannot be represented safely."
    )


def _argument_for_port(
    tool: ToolSpec,
    port: SemanticConsumerPortSpec,
) -> ArgumentSpec | None:
    if len(port.members) != 1:
        return None
    field_name = port.members[0].field_name
    argument = tool.required_arguments.get(
        field_name, tool.optional_arguments.get(field_name)
    )
    if argument is None:
        raise _catalog_error(
            "Semantic planning metadata names an unknown tool argument."
        )
    return argument


def _port_guidance(
    tool: ToolSpec,
    port: SemanticConsumerPortSpec,
) -> tuple[JsonValue, ...] | None:
    argument = _argument_for_port(tool, port)
    if argument is None:
        return None
    planning = argument.planning
    return (
        planning.description if planning is not None else None,
        planning.scientific_parameter if planning is not None else False,
        tuple(
            _safe_registered_value(choice) for choice in argument.choices
        ),
        (
            _safe_registered_value(planning.default_when_omitted)
            if planning is not None
            and planning.default_when_omitted is not inspect.Parameter.empty
            else None
        ),
        planning.conditional_note if planning is not None else None,
    )


def _consumer_port_catalog(
    tool: ToolSpec,
    port: SemanticConsumerPortSpec,
    request: AgentRequest,
    deterministic_scoped_selectors: frozenset[str],
) -> tuple[JsonValue, ...]:
    available_sources: list[JsonValue] = []
    for source in sorted(port.request_sources, key=lambda item: item.selector):
        if not all(member.input_name in request.inputs for member in source.members):
            continue
        available_sources.append(
            (
                source.selector,
                source.lineage.value if source.lineage is not None else None,
            )
        )

    if not port.request_sources:
        request_source_mode = "not_allowed"
    elif not available_sources:
        request_source_mode = "none_available"
    elif (
        not port.required
        and len(available_sources) == 1
        and available_sources[0][0] in deterministic_scoped_selectors
        and not port.accepted_upstream_types
    ):
        request_source_mode = "deterministic_scoped"
    elif not port.required:
        request_source_mode = "optional_explicit"
    elif len(available_sources) == 1:
        request_source_mode = "unique_available"
    else:
        request_source_mode = "explicit_choice_required"
    return (
        port.required,
        request_source_mode,
        tuple(available_sources),
        tuple(sorted(port.accepted_upstream_types)),
        (
            port.required_lineage.value
            if port.required_lineage is not None
            else None
        ),
        _port_guidance(tool, port),
    )


def _producer_port_catalog(
    port: SemanticProducerPortSpec,
) -> tuple[JsonValue, ...]:
    return (port.semantic_type, port.lineage_from_port)


def _tool_catalog(
    tool: ToolSpec,
    request: AgentRequest,
    deterministic_scoped_selectors: frozenset[str],
) -> tuple[JsonValue, ...]:
    semantic = tool.semantic_planning
    planning = tool.planning
    if semantic is None or planning is None:
        raise _catalog_error(
            "A planner-visible tool lacks semantic planning metadata."
        )
    return (
        planning.description,
        {
            port.name: _consumer_port_catalog(
                tool, port, request, deterministic_scoped_selectors
            )
            for port in sorted(semantic.consumer_ports, key=lambda item: item.name)
        },
        {
            port.name: _producer_port_catalog(port)
            for port in sorted(semantic.producer_ports, key=lambda item: item.name)
        },
        tuple(planning.conditional_notes),
    )


def build_semantic_planning_catalog(
    request: AgentRequest,
    registry: ToolRegistry,
) -> Mapping[str, JsonValue]:
    """Project registry semantic authority into request-specific LLM context."""

    if not isinstance(request, AgentRequest):
        raise TypeError("`request` must be an AgentRequest.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    tool_names = _planner_visible_tool_names(registry)
    scoped_selector_destinations: dict[str, set[tuple[str, str]]] = {}
    for name in tool_names:
        semantic = registry.get(name).semantic_planning
        if semantic is None:
            continue
        for port in semantic.consumer_ports:
            execution_fields = {
                member.name: member.field_name for member in port.members
            }
            for source in port.request_sources:
                if any(
                    member.input_name != execution_fields[member.name]
                    for member in source.members
                ):
                    scoped_selector_destinations.setdefault(
                        source.selector, set()
                    ).add((name, port.name))
    deterministic_scoped_selectors = frozenset(
        selector
        for selector, destinations in scoped_selector_destinations.items()
        if len(destinations) == 1
    )
    return {
        "semantic_catalog_version": SEMANTIC_PLANNING_CATALOG_VERSION,
        "request_inputs": {
            name: _json_type(request.inputs[name])
            for name in sorted(request.inputs)
        },
        "catalog_format": {
            "tool": ("purpose", "input_ports", "output_ports", "constraints"),
            "input_port": (
                "required",
                "request_source_mode",
                "available_request_sources",
                "accepted_upstream_types",
                "required_lineage",
                "guidance",
            ),
            "request_source": ("input", "lineage"),
            "guidance": (
                "meaning",
                "scientific_parameter",
                "choices",
                "default_when_omitted",
                "constraint",
            ),
            "output_port": ("semantic_type", "lineage_from"),
        },
        "tools": {
            name: _tool_catalog(
                registry.get(name), request, deterministic_scoped_selectors
            )
            for name in tool_names
        },
    }


def build_semantic_planning_prompt(
    request: AgentRequest,
    registry: ToolRegistry,
    *,
    repair_context: PlanningRepairContext | None = None,
    failover_context: PlanningRepairContext | None = None,
) -> str:
    """Build deterministic semantic planning context without executable values."""

    if repair_context is not None and failover_context is not None:
        raise ValueError("Semantic prompt cannot contain repair and failover together.")

    catalog = build_semantic_planning_catalog(request, registry)
    payload = {
        "semantic_prompt_version": SEMANTIC_PLANNING_PROMPT_VERSION,
        "instructions": (
            "Choose offered tools; compose a valid DAG.",
            "Use offered inputs/compatible outputs; invent no values or capabilities.",
            "Preserve lineage; resolve source choices explicitly.",
            "Ports marked deterministic_scoped are compiler-bound from explicit "
            "request scope; do not emit their request source.",
            "Ports marked optional_explicit are per-step choices: select only request "
            "sources whose scope matches that step; availability never implies "
            "fanout; omission preserves defaults.",
            "Required ports use a unique input unless a producer fits. Sources add "
            "dependencies; controls only order.",
            "Unsupported means genuine capability/input insufficiency only; otherwise "
            "plan if the catalog can satisfy the request.",
        ),
        "user_request": request.prompt,
        "catalog": catalog,
        "wire_v4": {
            "schema_version": SEMANTIC_WIRE_SCHEMA_VERSION,
            "decisions": ("plan", "unsupported"),
            "step_fields": (
                "step_id",
                "tool",
                "sources",
                "control_dependencies",
            ),
            "source_fields": ("target", "source"),
            "source_kinds": {
                "input": ("kind", "input"),
                "step": ("kind", "step"),
                "step_port": ("kind", "step", "source_port"),
            },
        },
    }
    regeneration_context = (
        repair_context if repair_context is not None else failover_context
    )
    if regeneration_context is not None:
        key = "repair" if repair_context is not None else "failover"
        action = (
            "repairing a rejected semantic planning decision"
            if repair_context is not None
            else "making the final planning attempt after a rejected decision"
        )
        diagnostic = regeneration_context.to_semantic_prompt_dict()
        correction = _SEMANTIC_REPAIR_CORRECTIONS.get(
            regeneration_context.reason_code or ""
        )
        if (
            correction is None
            and regeneration_context.previous_failure_stage.value == "parse"
        ):
            correction = "regenerate_strict_wire_v4_json"
        if (
            correction is None
            and regeneration_context.previous_failure_stage.value == "schema"
        ):
            correction = "regenerate_valid_wire_v4_structure"
        if correction is not None:
            diagnostic["required_correction"] = correction
        payload[key] = {
            "instruction": (
                f"The planning model is {action}.",
                "Generate one complete new wire-v4 decision from the original "
                "request and unchanged semantic catalog.",
                "Correct the diagnosed semantic constraint; do not patch, quote, "
                "or reproduce the previous response.",
                "Use only semantic ports and source selectors exposed by the "
                "catalog; never emit executor bindings or literal values.",
            ),
            "diagnostic": diagnostic,
        }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "SEMANTIC_PLANNING_CATALOG_VERSION",
    "SEMANTIC_PLANNING_PROMPT_VERSION",
    "build_semantic_planning_catalog",
    "build_semantic_planning_prompt",
]
