"""Non-executable capability visibility, derived from registered tool metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from agent.schemas import AgentRequest, ErrorCategory

from .planner import PlannerError
from .registry import ToolRegistry


SCOPE_VERSION = "planning-scope-v1"
# Descriptions only. Membership and scientific contracts belong to ToolRegistry.
CAPABILITY_DESCRIPTIONS = MappingProxyType({
    "species_adaptation": "Neutral target-species matrix construction from producer-qualified fragments, explicit barcodes and regulatory features; exact external matrix adoption; EpiZoo species post-training with explicit resources. Qualified target-model embedding inference with exact vocabulary and measurement correspondence. No cell calling.",
    "processed_inspection": "Inspect supplied processed scATAC H5AD datasets.",
    "embedding_analysis": (
        "EpiZoo and centered PCA cell embeddings, shared neighbor graphs, clustering, UMAP and fixed clustering evaluation. For generic clustering prefer qualified vocabulary-compatible EpiZoo; otherwise suitable X measurements can use PCA. Explicit EpiZoo requests never switch to PCA after failure."
    ),
    "reference_annotation": (
        "EpiZoo embeddings, reference-to-query cell label transfer and fixed annotation evaluation."
    ),
    "differential_accessibility": (
        "Declared regulatory feature/count validation, replicate pseudobulk and replicate-aware differential accessibility."
    ),
    "raw_preprocessing": (
        "Raw FASTQ/BAM intake, FASTQ/BAM/external fragments, barcode QC, explicit candidate selection, canonical cell-by-cCRE and qualified custom-region matrix construction."
    ),
    "exact_matrix_adoption": (
        "Adopt an external exact canonical cell-by-cCRE matrix; ordinary cell-by-peak projection is unsupported."
    ),
    "marker_annotation": (
        "MAESTRO cell-type annotation of independently supplied groups from human hg38 raw Cell Ranger ATAC cut-site peak-count H5AD, using compatible gene and marker-signature references and explicit tissue context. Accepts an obs grouping column or exact ordered groups TSV. Historical accepted canonical matrices remain supported. Does not cluster or require a labelled reference-cell dataset."
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

    def __init__(self, registry: ToolRegistry, capability_ids: tuple[str, ...] | None = None) -> None:
        if capability_ids is None:
            # Primary planning exposes every registered planner-visible tool.
            # Family labels are not an execution allowlist or required extension point.
            from .semantic_wire_v4 import _planner_visible_tool_names
            ids = ()
            names = _planner_visible_tool_names(registry)
        else:
            index = capability_index(registry)
            if (not isinstance(capability_ids, tuple) or not capability_ids
                    or any(not isinstance(c, str) or c not in index for c in capability_ids)
                    or len(set(capability_ids)) != len(capability_ids)):
                raise PlannerError('INVALID_PLANNING_SCOPE', 'Scope must select unique offered capability IDs.')
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
        if self != PlanningScope(registry, self.capability_ids or None):
            raise PlannerError(
                "INVALID_PLANNING_SCOPE", "Scope does not match registered metadata.",
            )
