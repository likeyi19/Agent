"""Non-executable capability visibility, derived from registered tool metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from agent.schemas import AgentRequest, ErrorCategory, JsonValue

from .planner import PlannerError
from .registry import ToolRegistry


SCOPE_VERSION = "planning-scope-v1"
SELECTION_SCHEMA_VERSION = 1
# Descriptions only. Membership and scientific contracts belong to ToolRegistry.
CAPABILITY_DESCRIPTIONS = MappingProxyType({
    "species_adaptation": "Neutral target-species matrix construction from verified external fragments, explicit barcodes and regulatory features; exact external matrix adoption; EpiZoo species post-training with explicit resources. No cell calling or target-model inference.",
    "processed_inspection": "Inspect supplied processed scATAC H5AD datasets.",
    "embedding_analysis": (
        "EpiZoo cell embeddings, neighbor graphs, clustering, UMAP and fixed clustering evaluation."
    ),
    "reference_annotation": (
        "EpiZoo embeddings, reference-to-query cell label transfer and fixed annotation evaluation."
    ),
    "differential_accessibility": (
        "Declared regulatory feature/count validation, replicate pseudobulk and replicate-aware differential accessibility."
    ),
    "raw_preprocessing": (
        "Raw FASTQ/BAM intake, FASTQ/BAM/external fragments, barcode QC, explicit candidate selection and canonical cell-by-cCRE construction."
    ),
    "exact_matrix_adoption": (
        "Adopt an external exact canonical cell-by-cCRE matrix; ordinary cell-by-peak projection is unsupported."
    ),
    "marker_annotation": (
        "Primary marker annotation of independently supplied groups on an already accepted canonical matrix, with explicit biological context and pinned resources. Does not cluster."
    ),
})


def fingerprint(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capability_index(registry: ToolRegistry) -> Mapping[str, tuple[str, ...]]:
    from .semantic_wire_v4 import _planner_visible_tool_names

    groups: dict[str, list[str]] = {}
    for name in _planner_visible_tool_names(registry):
        planning = registry.get(name).planning
        assert planning is not None
        if not planning.capability_ids or any(
            capability not in CAPABILITY_DESCRIPTIONS
            for capability in planning.capability_ids
        ):
            raise PlannerError(
                "PLANNER_CATALOG_INVALID",
                "Planner-visible tools require known capability membership.",
                category=ErrorCategory.INTERNAL_AGENT_ERROR,
            )
        for capability in planning.capability_ids:
            groups.setdefault(capability, []).append(name)
    return MappingProxyType({
        key: tuple(sorted(groups[key])) for key in sorted(groups)
    })


@dataclass(frozen=True, init=False)
class PlanningScope:
    """Validated visibility only; never execution or compatibility authority."""

    capability_ids: tuple[str, ...]
    visible_tool_names: tuple[str, ...]
    scope_fingerprint: str

    def __init__(self, registry: ToolRegistry, capability_ids: tuple[str, ...]) -> None:
        index = capability_index(registry)
        if (
            not isinstance(capability_ids, tuple)
            or not capability_ids
            or any(not isinstance(c, str) or c not in index for c in capability_ids)
            or len(set(capability_ids)) != len(capability_ids)
        ):
            raise PlannerError(
                "INVALID_PLANNING_SCOPE",
                "Scope must select unique offered capability IDs.",
            )
        ids = tuple(sorted(capability_ids))
        names = tuple(sorted({name for c in ids for name in index[c]}))
        # Bind selected membership and descriptive/semantic metadata without values.
        from .semantic_prompt import build_semantic_planning_catalog

        catalog = build_semantic_planning_catalog(
            AgentRequest("scope", "scope", {}), registry, visible_tool_names=names,
        )
        digest = fingerprint((
            SCOPE_VERSION, ids,
            {c: (CAPABILITY_DESCRIPTIONS[c], index[c]) for c in ids}, catalog,
            {name: asdict(registry.get(name).semantic_planning) for name in names},
        ))
        object.__setattr__(self, "capability_ids", ids)
        object.__setattr__(self, "visible_tool_names", names)
        object.__setattr__(self, "scope_fingerprint", digest)

    def validate(self, registry: ToolRegistry) -> None:
        if self != PlanningScope(registry, self.capability_ids):
            raise PlannerError(
                "INVALID_PLANNING_SCOPE", "Scope does not match registered metadata.",
            )


def selection_request(
    request: AgentRequest, registry: ToolRegistry,
) -> tuple[str, Mapping[str, JsonValue]]:
    from .semantic_prompt import _json_type
    from .semantic_wire_v4 import _closed_object_schema

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
    from .llm_planner import _parse_json_response, _require_fields

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
