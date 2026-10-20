"""Historical two-stage selection experiment; never imported by Agent production."""
from typing import Mapping
import json
from agent.schemas import AgentRequest, ErrorCategory, JsonValue
from agent.orchestration.registry import ToolRegistry
from agent.orchestration.planner import PlannerError
from agent.orchestration.planning_scope import PlanningScope, capability_index, CAPABILITY_DESCRIPTIONS

SELECTION_SCHEMA_VERSION = 1

def selection_request(
    request: AgentRequest, registry: ToolRegistry,
) -> tuple[str, Mapping[str, JsonValue]]:
    from agent.orchestration.semantic_prompt import _json_type
    from agent.orchestration.semantic_wire_v4 import _closed_object_schema

    index = capability_index(registry)
    payload = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "instructions": (
            "Select all capability families potentially relevant to the complete request; favor high recall.",
            "Multiple families are allowed. Select context, not exact tools, ordering, workflows or scientific parameters.",
            "Return only the selection decision. Use unsupported only when no offered family is relevant.",
        ),
        "user_request": request.prompt,
        "request_inputs": {
            name: _json_type(request.inputs[name]) for name in sorted(request.inputs)
        },
        "capabilities": {c: CAPABILITY_DESCRIPTIONS[c] for c in index},
    }
    from agent.orchestration.active_context import current_context
    active = current_context()
    if active is not None and active.items:
        payload['active_artifact_types'] = tuple(sorted({i.binding.semantic_type for i in active.items}))
    schema = _closed_object_schema({
        "selection_schema_version": {
            "type": "integer", "enum": (SELECTION_SCHEMA_VERSION,),
        },
        "decision": {"anyOf": (
            _closed_object_schema({
                "kind": {"type": "string", "enum": ("select",)},
                "capability_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": tuple(index)},
                },
            }),
            _closed_object_schema({
                "kind": {"type": "string", "enum": ("unsupported",)},
            }),
        )},
    })
    prompt = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return prompt, schema


def parse_selection(response: object, registry: ToolRegistry) -> PlanningScope:
    from agent.orchestration.llm_planner import _parse_json_response, _require_fields

    try:
        payload = _parse_json_response(response)
        _require_fields(
            payload, required=frozenset({"selection_schema_version", "decision"}),
            context="Scope selection",
        )
        if (
            type(payload["selection_schema_version"]) is not int
            or payload["selection_schema_version"] != SELECTION_SCHEMA_VERSION
        ):
            raise ValueError()
        decision = payload["decision"]
        if not isinstance(decision, dict):
            raise ValueError()
        kind = decision.get("kind")
        if kind == "unsupported":
            _require_fields(
                decision, required=frozenset({"kind"}), context="Unsupported scope",
            )
            raise PlannerError(
                "UNSUPPORTED_REQUEST", "No relevant capability selected.",
                category=ErrorCategory.USER_INPUT_ERROR,
            )
        _require_fields(
            decision, required=frozenset({"kind", "capability_ids"}),
            context="Selected scope",
        )
        if kind != "select" or not isinstance(decision["capability_ids"], list):
            raise ValueError()
        return PlanningScope(registry, tuple(decision["capability_ids"]))
    except (PlannerError, ValueError, TypeError, KeyError) as exc:
        if isinstance(exc, PlannerError) and exc.code == "UNSUPPORTED_REQUEST":
            raise
        raise PlannerError(
            "INVALID_PLANNING_SCOPE", "Invalid capability selection response.",
        ) from exc
