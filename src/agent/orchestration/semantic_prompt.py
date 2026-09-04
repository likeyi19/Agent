"""Experimental registry-driven semantic planning catalog and prompt."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Mapping

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


SEMANTIC_PLANNING_CATALOG_VERSION = 1
SEMANTIC_PLANNING_PROMPT_VERSION = 1


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
            port.name: _consumer_port_catalog(tool, port, request)
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
            "source_kinds": (
                "request_input when request_source_mode is not not_allowed",
                "prior_step when accepted_upstream_types is nonempty",
            ),
            "request_source_modes": (
                "not_allowed",
                "none_available",
                "unique_available",
                "explicit_choice_required",
            ),
        },
        "tools": {
            name: _tool_catalog(registry.get(name), request)
            for name in tool_names
        },
    }


def build_semantic_planning_prompt(
    request: AgentRequest,
    registry: ToolRegistry,
) -> str:
    """Build deterministic semantic planning context without executable values."""

    catalog = build_semantic_planning_catalog(request, registry)
    payload = {
        "semantic_prompt_version": SEMANTIC_PLANNING_PROMPT_VERSION,
        "instructions": (
            "Choose only offered tools needed for the user's scientific intent and "
            "compose a valid DAG.",
            "Use only offered structured input names and compatible prior-step "
            "semantic outputs; never invent inputs, capabilities, or literal values.",
            "Preserve reference, query, and ground-truth roles and every required "
            "lineage constraint.",
            "Explicitly select a semantic source whenever alternatives exist, "
            "including request-versus-prior-step choices.",
            "A unique request source may be omitted only when no selected producer "
            "can also satisfy that port.",
            "Select optional request sources on each intended step when their "
            "application scope could otherwise be ambiguous.",
            "Use control_dependencies only for ordering without value flow; source "
            "references already create value dependencies.",
            "Return unsupported when the offered tools and inputs cannot safely "
            "satisfy the request.",
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
            "source_kinds": {
                "input": ("target", "input"),
                "step": ("target", "step"),
                "step_port": ("target", "step", "source_port"),
            },
        },
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
