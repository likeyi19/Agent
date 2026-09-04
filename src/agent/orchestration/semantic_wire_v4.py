"""Experimental strict wire-v4 schema and parser for semantic plan candidates.

This module is deliberately disconnected from production ``LLMPlanner`` and
``AgentRuntime``. It validates only the provider-facing wire structure; the
registry-driven semantic compiler remains authoritative for source legality.
"""

from __future__ import annotations

from typing import Mapping

from agent.schemas import AgentRequest, ErrorCategory, JsonValue

from .llm_planner import (
    _MAX_REASON_LENGTH,
    _MAX_STEPS,
    _bounded_string,
    _invalid_output,
    _parse_json_response,
    _require_fields,
)
from .planner import PlannerError
from .planning_diagnostics import PlanningDiagnosticStage
from .registry import ToolRegistry
from .semantic_compiler import (
    SemanticPlanCandidate,
    SemanticPlanStep,
    SemanticRequestInputSource,
    SemanticStepOutputSource,
)


SEMANTIC_WIRE_SCHEMA_VERSION = 4
SEMANTIC_WIRE_MAX_SOURCES_PER_STEP = 32
SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP = _MAX_STEPS


def _closed_object_schema(
    properties: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": tuple(properties),
        "additionalProperties": False,
    }


def _planner_visible_tool_names(registry: ToolRegistry) -> tuple[str, ...]:
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    tool_names = tuple(
        sorted(
            name
            for name in registry.names()
            if registry.get(name).planning is not None
        )
    )
    if not tool_names:
        raise PlannerError(
            "PLANNER_CATALOG_INVALID",
            "Semantic wire planning requires a planner-visible registered tool.",
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
        )
    missing_metadata = tuple(
        name
        for name in tool_names
        if registry.get(name).semantic_planning is None
    )
    if missing_metadata:
        raise PlannerError(
            "PLANNER_CATALOG_INVALID",
            "Planner-visible tools lack authoritative semantic metadata.",
            category=ErrorCategory.INTERNAL_AGENT_ERROR,
        )
    return tool_names


def build_semantic_wire_v4_schema(
    registry: ToolRegistry,
    request: AgentRequest,
) -> Mapping[str, JsonValue]:
    """Build one deterministic provider-compatible schema without input values."""

    if not isinstance(request, AgentRequest):
        raise TypeError("`request` must be an AgentRequest.")
    tool_names = _planner_visible_tool_names(registry)
    input_names = tuple(sorted(request.inputs))

    definitions: dict[str, JsonValue] = {
        "step_source": _closed_object_schema(
            {
                "kind": {"type": "string", "enum": ("step",)},
                "target": {"type": "string"},
                "step": {"type": "string"},
            }
        ),
        "step_port_source": _closed_object_schema(
            {
                "kind": {"type": "string", "enum": ("step_port",)},
                "target": {"type": "string"},
                "step": {"type": "string"},
                "source_port": {"type": "string"},
            }
        ),
    }
    source_variants: list[JsonValue] = []
    if input_names:
        definitions["input_source"] = _closed_object_schema(
            {
                "kind": {"type": "string", "enum": ("input",)},
                "target": {"type": "string"},
                "input": {"type": "string", "enum": input_names},
            }
        )
        source_variants.append({"$ref": "#/$defs/input_source"})
    source_variants.extend(
        (
            {"$ref": "#/$defs/step_source"},
            {"$ref": "#/$defs/step_port_source"},
        )
    )
    definitions["source"] = {"anyOf": tuple(source_variants)}
    definitions["step"] = _closed_object_schema(
        {
            "step_id": {"type": "string"},
            "tool": {"type": "string", "enum": tool_names},
            "sources": {
                "type": "array",
                "items": {"$ref": "#/$defs/source"},
            },
            "control_dependencies": {
                "type": "array",
                "items": {"type": "string"},
            },
        }
    )
    definitions["plan_decision"] = _closed_object_schema(
        {
            "kind": {"type": "string", "enum": ("plan",)},
            "steps": {
                "type": "array",
                "items": {"$ref": "#/$defs/step"},
            },
        }
    )
    definitions["unsupported_decision"] = _closed_object_schema(
        {
            "kind": {"type": "string", "enum": ("unsupported",)},
            "reason": {"type": "string"},
        }
    )
    root = dict(
        _closed_object_schema(
            {
                "schema_version": {
                    "type": "integer",
                    "enum": (SEMANTIC_WIRE_SCHEMA_VERSION,),
                },
                "decision": {
                    "anyOf": (
                        {"$ref": "#/$defs/plan_decision"},
                        {"$ref": "#/$defs/unsupported_decision"},
                    )
                },
            }
        )
    )
    root["$defs"] = definitions
    return root


def _parse_source(
    raw_source: object,
    *,
    request: AgentRequest,
    context: str,
    step_index: int,
    step_id: str,
) -> SemanticRequestInputSource | SemanticStepOutputSource:
    if not isinstance(raw_source, dict):
        raise _invalid_output(f"{context} must be an object.")
    kind = raw_source.get("kind")
    if kind == "input":
        _require_fields(
            raw_source,
            required=frozenset({"kind", "target", "input"}),
            context=context,
        )
        input_name = _bounded_string(
            raw_source["input"], field_name=f"{context}.input"
        )
        if input_name not in request.inputs:
            raise _invalid_output(
                f"{context}.input is not available in the current request.",
                code="UNKNOWN_REQUEST_INPUT",
                stage=PlanningDiagnosticStage.ARGUMENT_BINDING,
                reason_code="unknown_request_source",
                diagnostic_fields={
                    "step_index": step_index,
                    "step_id": step_id,
                    "target_port": raw_source.get("target"),
                    "input_name": input_name,
                },
            )
        return SemanticRequestInputSource(
            _bounded_string(
                raw_source["target"], field_name=f"{context}.target"
            ),
            input_name,
        )
    if kind == "step":
        _require_fields(
            raw_source,
            required=frozenset({"kind", "target", "step"}),
            context=context,
        )
        return SemanticStepOutputSource(
            _bounded_string(
                raw_source["target"], field_name=f"{context}.target"
            ),
            _bounded_string(
                raw_source["step"], field_name=f"{context}.step"
            ),
        )
    if kind == "step_port":
        _require_fields(
            raw_source,
            required=frozenset({"kind", "target", "step", "source_port"}),
            context=context,
        )
        return SemanticStepOutputSource(
            _bounded_string(
                raw_source["target"], field_name=f"{context}.target"
            ),
            _bounded_string(
                raw_source["step"], field_name=f"{context}.step"
            ),
            _bounded_string(
                raw_source["source_port"],
                field_name=f"{context}.source_port",
            ),
        )
    raise _invalid_output(
        f"{context}.kind must be 'input', 'step', or 'step_port'."
    )


def _parse_steps(
    raw_steps: object,
    *,
    request: AgentRequest,
    registry: ToolRegistry,
) -> SemanticPlanCandidate:
    if not isinstance(raw_steps, list):
        raise _invalid_output("Wire-v4 plan `steps` must be an array.")
    if not raw_steps:
        raise _invalid_output("Wire-v4 plan must contain at least one step.")
    if len(raw_steps) > _MAX_STEPS:
        raise _invalid_output(
            f"Wire-v4 plan exceeds the {_MAX_STEPS}-step limit."
        )

    offered_tools = frozenset(_planner_visible_tool_names(registry))
    steps: list[SemanticPlanStep] = []
    for step_index, raw_step in enumerate(raw_steps):
        context = f"Wire-v4 step {step_index}"
        if not isinstance(raw_step, dict):
            raise _invalid_output(f"{context} must be an object.")
        _require_fields(
            raw_step,
            required=frozenset(
                {"step_id", "tool", "sources", "control_dependencies"}
            ),
            context=context,
        )
        step_id = _bounded_string(
            raw_step["step_id"], field_name=f"{context}.step_id"
        )
        tool_name = _bounded_string(
            raw_step["tool"], field_name=f"{context}.tool"
        )
        if tool_name not in offered_tools:
            raise PlannerError(
                "UNKNOWN_TOOL",
                "Wire-v4 step selected a tool outside the planner-visible registry.",
                category=ErrorCategory.INTERNAL_AGENT_ERROR,
                diagnostic_stage=PlanningDiagnosticStage.TOOL_SELECTION,
                diagnostic_reason_code="unknown_tool",
                diagnostic_fields={
                    "step_index": step_index,
                    "step_id": step_id,
                    "tool_name": tool_name,
                },
            )

        raw_sources = raw_step["sources"]
        if not isinstance(raw_sources, list):
            raise _invalid_output(f"{context}.sources must be an array.")
        if len(raw_sources) > SEMANTIC_WIRE_MAX_SOURCES_PER_STEP:
            raise _invalid_output(
                f"{context}.sources exceeds the "
                f"{SEMANTIC_WIRE_MAX_SOURCES_PER_STEP}-source limit."
            )
        sources = tuple(
            _parse_source(
                source,
                request=request,
                context=f"{context}.sources[{source_index}]",
                step_index=step_index,
                step_id=step_id,
            )
            for source_index, source in enumerate(raw_sources)
        )

        raw_control = raw_step["control_dependencies"]
        if not isinstance(raw_control, list):
            raise _invalid_output(
                f"{context}.control_dependencies must be an array."
            )
        if (
            len(raw_control)
            > SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP
        ):
            raise _invalid_output(
                f"{context}.control_dependencies exceeds the "
                f"{SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP}-edge limit."
            )
        control_dependencies = tuple(
            _bounded_string(
                dependency,
                field_name=(
                    f"{context}.control_dependencies[{dependency_index}]"
                ),
            )
            for dependency_index, dependency in enumerate(raw_control)
        )
        value_dependencies = {
            source.step_id
            for source in sources
            if isinstance(source, SemanticStepOutputSource)
        }
        # A value source already creates the same DAG edge during compilation.
        # Retain only genuinely control-only edges in the semantic candidate.
        control_dependencies = tuple(
            dependency
            for dependency in control_dependencies
            if dependency not in value_dependencies
        )
        try:
            steps.append(
                SemanticPlanStep(
                    step_id=step_id,
                    tool_name=tool_name,
                    sources=sources,
                    control_dependencies=control_dependencies,
                )
            )
        except (TypeError, ValueError) as exc:
            raise _invalid_output(
                f"{context} violates the semantic candidate structure.",
                stage=PlanningDiagnosticStage.CANDIDATE,
                reason_code="semantic_candidate_invalid",
                diagnostic_fields={
                    "step_index": step_index,
                    "step_id": step_id,
                    "tool_name": tool_name,
                },
            ) from exc

    try:
        return SemanticPlanCandidate(tuple(steps))
    except (TypeError, ValueError) as exc:
        raise _invalid_output(
            "Wire-v4 steps violate the semantic candidate structure.",
            stage=PlanningDiagnosticStage.CANDIDATE,
            reason_code="semantic_candidate_invalid",
        ) from exc


def parse_semantic_wire_v4(
    response: object,
    request: AgentRequest,
    registry: ToolRegistry,
) -> SemanticPlanCandidate:
    """Parse strict wire-v4 JSON into the accepted semantic candidate model."""

    if not isinstance(request, AgentRequest):
        raise TypeError("`request` must be an AgentRequest.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    payload = _parse_json_response(
        response, tree_limit_stage=PlanningDiagnosticStage.PARSE
    )
    _require_fields(
        payload,
        required=frozenset({"schema_version", "decision"}),
        context="Wire-v4 response",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != SEMANTIC_WIRE_SCHEMA_VERSION:
        raise _invalid_output(
            f"Wire-v4 response uses unsupported schema version {version!r}."
        )
    decision = payload["decision"]
    if not isinstance(decision, dict):
        raise _invalid_output("Wire-v4 response `decision` must be an object.")
    kind = decision.get("kind")
    if kind == "unsupported":
        _require_fields(
            decision,
            required=frozenset({"kind", "reason"}),
            context="Wire-v4 unsupported decision",
        )
        reason = _bounded_string(
            decision["reason"],
            field_name="Wire-v4 unsupported reason",
            maximum=_MAX_REASON_LENGTH,
        )
        raise PlannerError(
            "UNSUPPORTED_REQUEST",
            reason,
            category=ErrorCategory.USER_INPUT_ERROR,
        )
    if kind != "plan":
        raise _invalid_output(
            "Wire-v4 decision kind must be 'plan' or 'unsupported'."
        )
    _require_fields(
        decision,
        required=frozenset({"kind", "steps"}),
        context="Wire-v4 plan decision",
    )
    return _parse_steps(
        decision["steps"], request=request, registry=registry
    )


__all__ = [
    "SEMANTIC_WIRE_MAX_CONTROL_DEPENDENCIES_PER_STEP",
    "SEMANTIC_WIRE_MAX_SOURCES_PER_STEP",
    "SEMANTIC_WIRE_SCHEMA_VERSION",
    "build_semantic_wire_v4_schema",
    "parse_semantic_wire_v4",
]
