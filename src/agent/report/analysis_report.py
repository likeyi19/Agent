"""Deterministic scientific reports downstream of verified evidence and figures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Literal, Mapping, Sequence, TypedDict

from agent.orchestration.registry import ToolRegistry
from agent.schemas import (
    AgentError,
    AgentRunResult,
    ErrorCategory,
    JsonValue,
    VerificationCheck,
    VerificationResult,
)

from .evidence import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    AnalysisEvidenceResult,
    verify_analysis_evidence,
)
from .visualization import (
    ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
    ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
    AnalysisVisualizationResult,
    verify_analysis_visualizations,
)


ANALYSIS_REPORT_SCHEMA_VERSION = 1
ANALYSIS_REPORT_ARTIFACT_TYPE = "agent.analysis-report"
ANALYSIS_REPORT_BUNDLE_DIRNAME = "analysis_report"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"
ANALYSIS_REPORT_MANIFEST_FILENAME = "report_manifest.json"
REPORT_SPEC_VERSION = 1

_GENERATOR_IDENTITY = "agent.deterministic-analysis-report"
_FACT_DIGEST_DOMAIN = "agent.analysis-report.facts.v1"
_LF = "\n"


class ReportFactRecord(TypedDict):
    """One stable report fact attributed to verified evidence."""

    fact_id: str
    source_step_id: str
    tool_name: str
    field: str
    value: JsonValue


class ReportFigureResult(TypedDict):
    """Compact identity for one copied verified figure."""

    figure_id: str
    figure_kind: str
    figure_path: str
    png_sha256: str


class AnalysisReportResult(TypedDict):
    """Lightweight result for a completed deterministic report bundle."""

    status: Literal["success"]
    manifest_path: str
    manifest_sha256: str
    bundle_path: str
    report_path: str
    report_sha256: str
    schema_version: int
    artifact_type: str
    report_spec_version: int
    evidence_path: str
    evidence_sha256: str
    visualization_manifest_path: str | None
    visualization_manifest_sha256: str | None
    run_id: str
    request_id: str
    plan_id: str
    n_sections: int
    section_ids: list[str]
    n_figures: int
    figures: list[ReportFigureResult]


class AnalysisReportError(ValueError):
    """Fail-closed report error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class _EvidenceSnapshot:
    path: Path
    sha256: str
    payload: Mapping[str, object]
    run_id: str
    request_id: str
    plan_id: str


@dataclass(frozen=True)
class _SourceFigure:
    figure_id: str
    figure_kind: str
    source_step_ids: tuple[str, ...]
    relative_path: str
    source_path: Path
    png_sha256: str
    width_px: int
    height_px: int
    dpi: int
    caption: str


@dataclass(frozen=True)
class _VisualizationSnapshot:
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, object]
    figures: tuple[_SourceFigure, ...]


@dataclass(frozen=True)
class _Fact:
    fact_id: str
    source_step_id: str
    tool_name: str
    field: str
    value: JsonValue

    def to_dict(self) -> ReportFactRecord:
        return {
            "fact_id": self.fact_id,
            "source_step_id": self.source_step_id,
            "tool_name": self.tool_name,
            "field": self.field,
            "value": self.value,
        }


@dataclass(frozen=True)
class _StepFacts:
    step_id: str
    tool_name: str
    facts: tuple[_Fact, ...]


@dataclass(frozen=True)
class _Section:
    section_id: str
    title: str
    source_step_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    figure_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ArtifactIntegrity:
    producing_step_id: str
    tool_name: str
    artifact_kind: str
    authoritative_digest: Mapping[str, JsonValue] | None
    verification_basis: tuple[str, ...]


@dataclass(frozen=True)
class _ReportProjection:
    steps: tuple[_StepFacts, ...]
    facts: tuple[_Fact, ...]
    sections: tuple[_Section, ...]
    artifact_integrity: tuple[_ArtifactIntegrity, ...]


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _plain_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisReportError(
                "REPORT_VALUE_INVALID",
                "Report facts cannot contain non-finite values.",
            )
        return value
    if isinstance(value, Mapping):
        copied: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise AnalysisReportError(
                    "REPORT_VALUE_INVALID",
                    "Report metadata requires string mapping keys.",
                )
            copied[key] = _plain_json(nested)
        return copied
    if isinstance(value, (list, tuple)):
        return tuple(_plain_json(nested) for nested in value)
    raise AnalysisReportError(
        "REPORT_VALUE_INVALID",
        "Report metadata contains an unsupported value type.",
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise AnalysisReportError(
            "REPORT_NOT_JSON_SAFE",
            "Report metadata is not strict JSON-safe data.",
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisReportError(
            "REPORT_ARTIFACT_UNAVAILABLE",
            "A report source or artifact could not be read.",
        ) from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _strict_json(path: Path, *, code: str, description: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise AnalysisReportError(code, description) from exc
    if not isinstance(parsed, dict):
        raise AnalysisReportError(code, description)
    return parsed, payload


def _evidence_path(value: str | Path | AnalysisEvidenceResult) -> Path:
    if isinstance(value, Mapping):
        path = value.get("evidence_path")
        if not isinstance(path, str) or not path:
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "AnalysisEvidenceResult lacks a valid evidence path.",
            )
        return Path(path).expanduser().resolve()
    if not isinstance(value, (str, Path)):
        raise TypeError("`evidence` must be a path or AnalysisEvidenceResult mapping.")
    return Path(value).expanduser().resolve()


def _verified_evidence_snapshot(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    registry: ToolRegistry,
) -> _EvidenceSnapshot:
    try:
        verification = verify_analysis_evidence(run_result, evidence, registry=registry)
    except Exception as exc:
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Report generation requires freshly verified AnalysisEvidence.",
        ) from exc
    if not isinstance(verification, VerificationResult) or not verification.passed:
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Report generation requires freshly verified AnalysisEvidence.",
        )
    path = _evidence_path(evidence)
    parsed, payload = _strict_json(
        path,
        code="REPORT_SOURCE_EVIDENCE_INVALID",
        description="Verified analysis evidence could not be strictly loaded.",
    )
    source_run = parsed.get("run")
    if (
        parsed.get("schema_version") != ANALYSIS_EVIDENCE_SCHEMA_VERSION
        or parsed.get("artifact_type") != ANALYSIS_EVIDENCE_ARTIFACT_TYPE
        or parsed.get("status") != "success"
        or not isinstance(source_run, Mapping)
        or source_run.get("run_id") != run_result.run_id
        or source_run.get("request_id") != run_result.request_id
        or run_result.plan is None
        or source_run.get("plan_id") != run_result.plan.plan_id
    ):
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Verified analysis evidence has inconsistent source identity.",
        )
    digest = _sha256_bytes(payload)
    if isinstance(evidence, Mapping) and evidence.get("evidence_sha256") != digest:
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Analysis evidence differs from its authoritative digest.",
        )
    return _EvidenceSnapshot(
        path=path,
        sha256=digest,
        payload=parsed,
        run_id=run_result.run_id,
        request_id=run_result.request_id,
        plan_id=run_result.plan.plan_id,
    )


def _visualization_manifest_path(
    value: str | Path | AnalysisVisualizationResult,
) -> Path:
    if isinstance(value, Mapping):
        path = value.get("manifest_path")
        if not isinstance(path, str) or not path:
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "AnalysisVisualizationResult lacks a valid manifest path.",
            )
        return Path(path).expanduser().resolve()
    if not isinstance(value, (str, Path)):
        raise TypeError(
            "`visualization` must be a manifest path or AnalysisVisualizationResult."
        )
    return Path(value).expanduser().resolve()


_FIGURE_CAPTIONS: Mapping[str, str] = {
    "umap_leiden": "Verified UMAP coordinates colored by Leiden cluster labels.",
    "clustering_metrics": (
        "Verified NMI, ARI, AMI, and Homogeneity values in fixed metric order."
    ),
    "annotation_confusion": (
        "Verified annotation-evaluation confusion matrix using persisted raw "
        "counts and class order."
    ),
}


def _safe_figure_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Visualization figure path metadata is invalid.",
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or len(relative.parts) != 2
        or relative.parts[0] != "figures"
        or relative.parts[1] in {"", ".", ".."}
        or not relative.parts[1].endswith(".png")
        or re.fullmatch(r"[A-Za-z0-9._-]+\.png", relative.parts[1]) is None
    ):
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Visualization figure path metadata is unsafe.",
        )
    return relative


def _verified_visualization_snapshot(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    evidence_snapshot: _EvidenceSnapshot,
    visualization: str | Path | AnalysisVisualizationResult,
    registry: ToolRegistry,
) -> _VisualizationSnapshot:
    try:
        verification = verify_analysis_visualizations(
            run_result,
            evidence,
            visualization,
            registry=registry,
        )
    except Exception as exc:
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Report generation requires freshly verified AnalysisVisualizations.",
        ) from exc
    if not isinstance(verification, VerificationResult) or not verification.passed:
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Report generation requires freshly verified AnalysisVisualizations.",
        )
    manifest_path = _visualization_manifest_path(visualization)
    if manifest_path.name != ANALYSIS_VISUALIZATION_MANIFEST_FILENAME:
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Visualization manifest has an unexpected filename.",
        )
    parsed, payload = _strict_json(
        manifest_path,
        code="REPORT_SOURCE_VISUALIZATION_INVALID",
        description="Verified visualization manifest could not be strictly loaded.",
    )
    source = parsed.get("source")
    if (
        parsed.get("schema_version") != ANALYSIS_VISUALIZATION_SCHEMA_VERSION
        or parsed.get("artifact_type") != ANALYSIS_VISUALIZATION_ARTIFACT_TYPE
        or parsed.get("status") != "success"
        or not isinstance(source, Mapping)
        or source.get("run_id") != evidence_snapshot.run_id
        or source.get("request_id") != evidence_snapshot.request_id
        or source.get("plan_id") != evidence_snapshot.plan_id
        or source.get("evidence_path") != str(evidence_snapshot.path)
        or source.get("evidence_sha256") != evidence_snapshot.sha256
    ):
        raise AnalysisReportError(
            "REPORT_SOURCE_BINDING_MISMATCH",
            "Visualization does not bind to the supplied run and evidence.",
        )
    manifest_digest = _sha256_bytes(payload)
    if (
        isinstance(visualization, Mapping)
        and visualization.get("manifest_sha256") != manifest_digest
    ):
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Visualization manifest differs from its authoritative digest.",
        )
    raw_figures = parsed.get("figures")
    if not isinstance(raw_figures, list) or not raw_figures:
        raise AnalysisReportError(
            "REPORT_SOURCE_VISUALIZATION_INVALID",
            "Visualization manifest contains no verified figures.",
        )
    figures: list[_SourceFigure] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for raw in raw_figures:
        if not isinstance(raw, Mapping):
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "Visualization manifest has an invalid figure record.",
            )
        figure_id = raw.get("figure_id")
        figure_kind = raw.get("figure_kind")
        source_steps = raw.get("source_step_ids")
        relative = _safe_figure_relative_path(raw.get("relative_path"))
        digest = raw.get("png_sha256")
        width = raw.get("width_px")
        height = raw.get("height_px")
        dpi = raw.get("dpi")
        if (
            not isinstance(figure_id, str)
            or not figure_id
            or figure_id in seen_ids
            or figure_kind not in _FIGURE_CAPTIONS
            or not isinstance(source_steps, list)
            or not source_steps
            or not all(isinstance(value, str) and value for value in source_steps)
            or not _is_sha256(digest)
            or isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
            or isinstance(dpi, bool)
            or not isinstance(dpi, int)
            or dpi <= 0
            or relative.name in seen_names
        ):
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "Visualization manifest has invalid figure identity metadata.",
            )
        source_path = manifest_path.parent / Path(*relative.parts)
        if not source_path.is_file() or source_path.is_symlink():
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "A verified source figure is missing or unsafe.",
            )
        if _sha256_file(source_path) != digest:
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "A verified source figure differs from its manifest digest.",
            )
        seen_ids.add(figure_id)
        seen_names.add(relative.name)
        figures.append(
            _SourceFigure(
                figure_id=figure_id,
                figure_kind=str(figure_kind),
                source_step_ids=tuple(source_steps),
                relative_path=str(relative),
                source_path=source_path,
                png_sha256=str(digest),
                width_px=width,
                height_px=height,
                dpi=dpi,
                caption=_FIGURE_CAPTIONS[str(figure_kind)],
            )
        )
    return _VisualizationSnapshot(
        manifest_path=manifest_path,
        manifest_sha256=manifest_digest,
        payload=parsed,
        figures=tuple(figures),
    )


_REPORT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "inspect_scATAC": (
        "input_path",
        "n_cells",
        "n_features",
        "x_storage_type",
        "x_is_sparse",
        "x_dtype",
        "nnz",
        "density",
    ),
    "epizoo_embed_cells": (
        "n_cells",
        "embedding_dim",
        "embedding_dtype",
        "species",
        "backend",
        "checkpoint_path",
        "device",
        "finite",
        "cell_order_preserved",
    ),
    "build_cell_neighbors": (
        "n_cells",
        "embedding_dim",
        "n_neighbors",
        "metric",
        "neighbors_method",
        "transformer",
        "random_seed",
        "connectivities_nnz",
        "distances_nnz",
        "backend",
    ),
    "cluster_cells": (
        "n_cells",
        "n_clusters",
        "cluster_key",
        "algorithm",
        "resolution",
        "random_seed",
        "backend",
    ),
    "compute_cell_umap": (
        "n_cells",
        "n_components",
        "umap_key",
        "coordinate_dtype",
        "min_dist",
        "spread",
        "random_seed",
        "backend",
    ),
    "evaluate_cell_clustering": (
        "n_cells",
        "n_reference_classes",
        "n_predicted_clusters",
        "nmi",
        "ari",
        "ami",
        "homogeneity",
        "average_method",
        "metric_backend",
    ),
    "transfer_cell_labels": (
        "checkpoint_path",
        "reference_label_key",
        "n_reference_cells",
        "n_query_cells",
        "n_reference_classes",
        "assigned_count",
        "unassigned_count",
        "assignment_rate",
        "embedding_dim",
        "n_neighbors",
        "metric",
        "voting_method",
        "min_confidence",
        "species",
        "species_compatible",
        "checkpoint_compatible",
        "backend",
    ),
    "evaluate_cell_annotation": (
        "annotation_sha256",
        "ground_truth_label_key",
        "n_cells",
        "n_ground_truth_classes",
        "n_assigned_predicted_classes",
        "assigned_count",
        "unassigned_count",
        "assignment_rate",
        "correct_assigned_count",
        "incorrect_assigned_count",
        "overall_accuracy",
        "assigned_accuracy",
        "macro_f1",
        "median_confidence",
        "median_assigned_confidence",
        "median_correct_assigned_confidence",
        "median_incorrect_assigned_confidence",
        "metric_backend",
        "macro_average",
        "zero_division",
    ),
    "validate_scATAC_feature_space": (
        "input_path",
        "n_cells",
        "n_features",
        "matrix_source",
        "layer_key",
        "matrix_semantics",
        "semantics_assertion_source",
        "pseudobulk_eligible",
        "species",
        "genome_assembly",
        "coordinate_source",
        "coordinate_system",
        "nnz",
        "source_dtype",
        "source_sparse_format",
        "feature_space_identity_sha256",
    ),
    "build_replicate_pseudobulk": (
        "n_cells",
        "n_features",
        "n_pseudobulks",
        "n_groups",
        "n_replicates",
        "n_conditions",
        "minimum_cells_per_pseudobulk",
        "maximum_cells_per_pseudobulk",
        "matrix_nnz",
        "total_sum",
        "matrix_semantics",
        "output_value_semantics",
        "aggregation_method",
        "output_dtype",
        "group_source",
        "group_key",
        "replicate_key",
        "condition_key",
        "covariate_keys",
        "all_cells_accounted_for",
        "feature_order_preserved",
        "pseudobulk_sha256",
    ),
}

_TOOL_TITLES: Mapping[str, str] = {
    "inspect_scATAC": "Dataset inspection",
    "epizoo_embed_cells": "EpiZoo representation",
    "build_cell_neighbors": "Cell-neighbor graph",
    "cluster_cells": "Leiden clustering",
    "compute_cell_umap": "UMAP",
    "evaluate_cell_clustering": "Clustering evaluation",
    "transfer_cell_labels": "Cell-label transfer",
    "evaluate_cell_annotation": "Annotation evaluation",
    "validate_scATAC_feature_space": "Regulatory feature space",
    "build_replicate_pseudobulk": "Replicate-aware pseudobulk",
}

_FIELD_LABELS: Mapping[str, str] = {
    "input_path": "Input path",
    "n_cells": "Cells",
    "n_features": "Features",
    "x_storage_type": "Matrix storage type",
    "x_is_sparse": "Sparse matrix",
    "x_dtype": "Matrix dtype",
    "nnz": "Nonzero entries",
    "density": "Matrix density",
    "embedding_dim": "Embedding dimensions",
    "embedding_dtype": "Embedding dtype",
    "species": "Species",
    "backend": "Backend",
    "checkpoint_path": "Checkpoint path",
    "device": "Device",
    "finite": "Finite values verified",
    "cell_order_preserved": "Cell order preserved",
    "n_neighbors": "Neighbors",
    "metric": "Distance metric",
    "neighbors_method": "Neighbor-connectivity method",
    "transformer": "Neighbor transformer",
    "random_seed": "Random seed",
    "connectivities_nnz": "Connectivity nonzero entries",
    "distances_nnz": "Distance nonzero entries",
    "n_clusters": "Clusters",
    "cluster_key": "Cluster key",
    "algorithm": "Algorithm",
    "resolution": "Resolution",
    "n_components": "UMAP dimensions",
    "umap_key": "UMAP key",
    "coordinate_dtype": "Coordinate dtype",
    "min_dist": "UMAP min_dist",
    "spread": "UMAP spread",
    "n_reference_classes": "Reference classes",
    "n_predicted_clusters": "Predicted clusters",
    "nmi": "NMI",
    "ari": "ARI",
    "ami": "AMI",
    "homogeneity": "Homogeneity",
    "average_method": "Metric averaging",
    "metric_backend": "Metric backend",
    "reference_label_key": "Reference label key",
    "n_reference_cells": "Reference cells",
    "n_query_cells": "Query cells",
    "assigned_count": "Assigned cells",
    "unassigned_count": "Unassigned cells",
    "assignment_rate": "Assignment rate",
    "voting_method": "Voting method",
    "min_confidence": "Minimum confidence",
    "species_compatible": "Species compatibility verified",
    "checkpoint_compatible": "Checkpoint compatibility verified",
    "annotation_sha256": "Annotation SHA-256",
    "ground_truth_label_key": "Ground-truth label key",
    "n_ground_truth_classes": "Ground-truth classes",
    "n_assigned_predicted_classes": "Assigned predicted classes",
    "correct_assigned_count": "Correct assigned cells",
    "incorrect_assigned_count": "Incorrect assigned cells",
    "overall_accuracy": "Overall accuracy",
    "assigned_accuracy": "Assigned-only accuracy",
    "macro_f1": "Macro-F1",
    "median_confidence": "Median confidence",
    "median_assigned_confidence": "Median assigned confidence",
    "median_correct_assigned_confidence": "Median correct-assigned confidence",
    "median_incorrect_assigned_confidence": "Median incorrect-assigned confidence",
    "macro_average": "Macro averaging",
    "zero_division": "Zero-division policy",
    "matrix_source": "Regulatory matrix source",
    "layer_key": "Regulatory matrix layer",
    "matrix_semantics": "Regulatory matrix semantics",
    "semantics_assertion_source": "Matrix-semantics assertion source",
    "pseudobulk_eligible": "Pseudobulk eligible",
    "genome_assembly": "Genome assembly",
    "coordinate_source": "Coordinate source",
    "coordinate_system": "Coordinate system",
    "source_dtype": "Source matrix dtype",
    "source_sparse_format": "Source sparse format",
    "feature_space_identity_sha256": "Feature-space identity SHA-256",
    "n_pseudobulks": "Pseudobulk units",
    "n_groups": "Biological groups",
    "n_replicates": "Biological replicates",
    "n_conditions": "Conditions",
    "minimum_cells_per_pseudobulk": "Minimum cells per pseudobulk",
    "maximum_cells_per_pseudobulk": "Maximum cells per pseudobulk",
    "matrix_nnz": "Pseudobulk nonzero entries",
    "total_sum": "Total pseudobulk sum",
    "output_value_semantics": "Pseudobulk value semantics",
    "aggregation_method": "Aggregation method",
    "output_dtype": "Output matrix dtype",
    "group_source": "Group source",
    "group_key": "Group metadata key",
    "replicate_key": "Replicate metadata key",
    "condition_key": "Condition metadata key",
    "covariate_keys": "Covariate metadata keys",
    "all_cells_accounted_for": "All cells accounted for",
    "feature_order_preserved": "Feature order preserved",
    "pseudobulk_sha256": "Pseudobulk SHA-256",
}

_SECTION_SPECS: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("dataset", "Dataset", frozenset({"inspect_scATAC"})),
    (
        "epizoo_representation",
        "EpiZoo Representation",
        frozenset({"epizoo_embed_cells"}),
    ),
    (
        "clustering_umap",
        "Clustering and UMAP",
        frozenset({"build_cell_neighbors", "cluster_cells", "compute_cell_umap"}),
    ),
    (
        "clustering_evaluation",
        "Clustering Evaluation",
        frozenset({"evaluate_cell_clustering"}),
    ),
    (
        "cell_annotation",
        "Cell Annotation",
        frozenset({"transfer_cell_labels"}),
    ),
    (
        "annotation_evaluation",
        "Annotation Evaluation",
        frozenset({"evaluate_cell_annotation"}),
    ),
    (
        "regulatory_feature_space",
        "Regulatory Feature Space",
        frozenset({"validate_scATAC_feature_space"}),
    ),
    (
        "replicate_pseudobulk",
        "Replicate-aware Pseudobulk",
        frozenset({"build_replicate_pseudobulk"}),
    ),
)

_METHOD_FIELDS: Mapping[str, tuple[str, ...]] = {
    "inspect_scATAC": ("x_storage_type", "x_is_sparse", "x_dtype"),
    "epizoo_embed_cells": ("species", "backend", "checkpoint_path", "device"),
    "build_cell_neighbors": (
        "n_neighbors",
        "metric",
        "neighbors_method",
        "transformer",
        "random_seed",
    ),
    "cluster_cells": ("algorithm", "resolution", "random_seed"),
    "compute_cell_umap": ("n_components", "min_dist", "spread", "random_seed"),
    "evaluate_cell_clustering": ("average_method", "metric_backend"),
    "transfer_cell_labels": (
        "n_neighbors",
        "metric",
        "voting_method",
        "min_confidence",
    ),
    "evaluate_cell_annotation": ("metric_backend", "macro_average", "zero_division"),
    "validate_scATAC_feature_space": (
        "matrix_source",
        "layer_key",
        "matrix_semantics",
        "semantics_assertion_source",
        "species",
        "genome_assembly",
        "coordinate_source",
        "coordinate_system",
        "source_sparse_format",
    ),
    "build_replicate_pseudobulk": (
        "aggregation_method",
        "output_dtype",
        "output_value_semantics",
        "group_source",
        "group_key",
        "replicate_key",
        "condition_key",
        "covariate_keys",
    ),
}


def _project_artifact_integrity(payload: Mapping[str, object]) -> tuple[_ArtifactIntegrity, ...]:
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Verified evidence lacks artifact provenance.",
        )
    projected: list[_ArtifactIntegrity] = []
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid artifact provenance.",
            )
        integrity = raw.get("integrity")
        if not isinstance(integrity, Mapping):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid artifact integrity metadata.",
            )
        authoritative = integrity.get("authoritative_digest")
        basis = integrity.get("verification_basis")
        if (
            not isinstance(raw.get("producing_step_id"), str)
            or not isinstance(raw.get("tool_name"), str)
            or not isinstance(raw.get("artifact_kind"), str)
            or (authoritative is not None and not isinstance(authoritative, Mapping))
            or not isinstance(basis, list)
            or not all(isinstance(value, str) and value for value in basis)
        ):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid artifact integrity metadata.",
            )
        plain_authoritative = (
            None if authoritative is None else _plain_json(authoritative)
        )
        if plain_authoritative is not None and not isinstance(
            plain_authoritative, Mapping
        ):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid authoritative digest metadata.",
            )
        projected.append(
            _ArtifactIntegrity(
                producing_step_id=str(raw["producing_step_id"]),
                tool_name=str(raw["tool_name"]),
                artifact_kind=str(raw["artifact_kind"]),
                authoritative_digest=plain_authoritative,
                verification_basis=tuple(basis),
            )
        )
    return tuple(projected)


def _project_report(
    evidence: _EvidenceSnapshot,
    visualization: _VisualizationSnapshot | None,
) -> _ReportProjection:
    workflow = evidence.payload.get("workflow")
    raw_steps = evidence.payload.get("steps")
    if (
        not isinstance(workflow, Mapping)
        or not isinstance(workflow.get("ordered_steps"), list)
        or not isinstance(raw_steps, list)
    ):
        raise AnalysisReportError(
            "REPORT_SOURCE_EVIDENCE_INVALID",
            "Verified evidence lacks its ordered workflow projection.",
        )
    by_id: dict[str, Mapping[str, object]] = {}
    for raw in raw_steps:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("step_id"), str):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid step facts.",
            )
        step_id = str(raw["step_id"])
        if step_id in by_id:
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has duplicate step facts.",
            )
        by_id[step_id] = raw

    facts: list[_Fact] = []
    steps: list[_StepFacts] = []
    for ordered in workflow["ordered_steps"]:
        if not isinstance(ordered, Mapping):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid workflow metadata.",
            )
        step_id = ordered.get("step_id")
        tool_name = ordered.get("tool_name")
        if not isinstance(step_id, str) or not isinstance(tool_name, str):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence has invalid workflow identity.",
            )
        fields = _REPORT_FIELDS.get(tool_name)
        if fields is None:
            raise AnalysisReportError(
                "REPORT_NO_REPORTABLE_CONTENT",
                "Report schema v1 does not support a source scientific tool.",
            )
        raw_step = by_id.get(step_id)
        if raw_step is None or raw_step.get("tool_name") != tool_name:
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified workflow and step facts disagree.",
            )
        raw_facts = raw_step.get("facts")
        if not isinstance(raw_facts, Mapping):
            raise AnalysisReportError(
                "REPORT_SOURCE_EVIDENCE_INVALID",
                "Verified evidence lacks reportable step facts.",
            )
        step_facts: list[_Fact] = []
        for field in fields:
            if field not in raw_facts:
                raise AnalysisReportError(
                    "REPORT_SOURCE_EVIDENCE_INVALID",
                    "Verified evidence lacks a required report fact.",
                )
            fact = _Fact(
                fact_id=f"F{len(facts) + 1:04d}",
                source_step_id=step_id,
                tool_name=tool_name,
                field=field,
                value=_plain_json(raw_facts[field]),
            )
            facts.append(fact)
            step_facts.append(fact)
        steps.append(_StepFacts(step_id, tool_name, tuple(step_facts)))
    if not steps:
        raise AnalysisReportError(
            "REPORT_NO_REPORTABLE_CONTENT",
            "Verified evidence contains no reportable workflow steps.",
        )

    sections: list[_Section] = [
        _Section(
            "analysis_summary",
            "Analysis Summary",
            tuple(step.step_id for step in steps),
            tuple(fact.fact_id for fact in facts),
            tuple(
                figure.figure_id for figure in visualization.figures
            ) if visualization is not None else (),
        )
    ]
    for section_id, title, tools in _SECTION_SPECS:
        included = tuple(step for step in steps if step.tool_name in tools)
        if included:
            sections.append(
                _Section(
                    section_id,
                    title,
                    tuple(step.step_id for step in included),
                    tuple(fact.fact_id for step in included for fact in step.facts),
                )
            )
    if visualization is not None:
        sections.append(
            _Section(
                "figures",
                "Figures",
                tuple(
                    step_id
                    for figure in visualization.figures
                    for step_id in figure.source_step_ids
                ),
                tuple(
                    fact.fact_id
                    for fact in facts
                    if any(
                        fact.source_step_id in figure.source_step_ids
                        for figure in visualization.figures
                    )
                ),
                tuple(figure.figure_id for figure in visualization.figures),
            )
        )
    method_ids = tuple(
        fact.fact_id
        for step in steps
        for fact in step.facts
        if fact.field in _METHOD_FIELDS[step.tool_name]
    )
    sections.append(
        _Section(
            "methods",
            "Methods / Analysis Parameters",
            tuple(step.step_id for step in steps),
            method_ids,
        )
    )
    sections.append(
        _Section(
            "provenance",
            "Provenance and Reproducibility",
            tuple(step.step_id for step in steps),
            tuple(fact.fact_id for fact in facts),
            tuple(
                figure.figure_id for figure in visualization.figures
            ) if visualization is not None else (),
        )
    )
    return _ReportProjection(
        steps=tuple(steps),
        facts=tuple(facts),
        sections=tuple(sections),
        artifact_integrity=_project_artifact_integrity(evidence.payload),
    )


def _inline_code(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    maximum = 0
    current = 0
    for character in serialized:
        if character == "`":
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    fence = "`" * (maximum + 1)
    return f"{fence} {serialized} {fence}"


def _fact_ids_line(facts: Sequence[_Fact]) -> str:
    return "Evidence facts: " + ", ".join(_inline_code(fact.fact_id) for fact in facts) + "."


def _render_step_block(step: _StepFacts, occurrence: int) -> list[str]:
    lines = [f"### {_TOOL_TITLES[step.tool_name]} {occurrence}", ""]
    lines.append(f"- Source step: {_inline_code(step.step_id)}")
    for fact in step.facts:
        label = _FIELD_LABELS[fact.field]
        rendered = _inline_code(fact.value)
        if fact.value is None:
            rendered += " (undefined)"
        lines.append(f"- {label}: {rendered}")
    lines.extend(("", _fact_ids_line(step.facts), ""))
    return lines


def _render_summary(
    projection: _ReportProjection,
    visualization: _VisualizationSnapshot | None,
) -> list[str]:
    present_tools = {step.tool_name for step in projection.steps}
    labels = {
        "inspect_scATAC": "Dataset inspection completed",
        "epizoo_embed_cells": "EpiZoo representation produced",
        "build_cell_neighbors": "Cell-neighbor graph produced",
        "cluster_cells": "Leiden clustering produced",
        "compute_cell_umap": "UMAP coordinates produced",
        "evaluate_cell_clustering": "Clustering evaluation available",
        "transfer_cell_labels": "Cell-label transfer performed",
        "evaluate_cell_annotation": "Annotation evaluation available",
        "validate_scATAC_feature_space": "Regulatory feature space validated",
        "build_replicate_pseudobulk": "Replicate-aware pseudobulk produced",
    }
    lines = ["## Analysis Summary", ""]
    for tool_name in _REPORT_FIELDS:
        if tool_name in present_tools:
            lines.append(f"- {labels[tool_name]}.")
    if visualization is not None:
        lines.append(f"- {len(visualization.figures)} verified figure(s) included.")
    lines.extend(("", _fact_ids_line(projection.facts), ""))
    return lines


def _steps_for_section(
    projection: _ReportProjection, section: _Section
) -> tuple[_StepFacts, ...]:
    allowed = set(section.source_step_ids)
    return tuple(step for step in projection.steps if step.step_id in allowed)


def _render_main_section(
    projection: _ReportProjection, section: _Section
) -> list[str]:
    lines = [f"## {section.title}", ""]
    occurrences: dict[str, int] = {}
    for step in _steps_for_section(projection, section):
        occurrences[step.tool_name] = occurrences.get(step.tool_name, 0) + 1
        lines.extend(_render_step_block(step, occurrences[step.tool_name]))
    return lines


def _render_figures(visualization: _VisualizationSnapshot) -> list[str]:
    lines = ["## Figures", ""]
    for index, figure in enumerate(visualization.figures, start=1):
        report_path = f"figures/{PurePosixPath(figure.relative_path).name}"
        lines.extend(
            (
                f"### Figure {index}: {figure.figure_kind}",
                "",
                figure.caption,
                "",
                f"![{figure.caption}]({report_path})",
                "",
                "Source figure: " + _inline_code(figure.figure_id) + ".",
                "",
            )
        )
    return lines


def _render_methods(projection: _ReportProjection) -> list[str]:
    lines = ["## Methods / Analysis Parameters", ""]
    occurrences: dict[str, int] = {}
    for step in projection.steps:
        selected = tuple(
            fact for fact in step.facts if fact.field in _METHOD_FIELDS[step.tool_name]
        )
        occurrences[step.tool_name] = occurrences.get(step.tool_name, 0) + 1
        lines.extend(
            (
                f"### {_TOOL_TITLES[step.tool_name]} {occurrences[step.tool_name]}",
                "",
                f"- Source step: {_inline_code(step.step_id)}",
            )
        )
        for fact in selected:
            rendered = _inline_code(fact.value)
            if fact.value is None:
                rendered += " (undefined)"
            lines.append(f"- {_FIELD_LABELS[fact.field]}: {rendered}")
        lines.extend(("", _fact_ids_line(selected), ""))
    return lines


def _render_provenance(
    evidence: _EvidenceSnapshot,
    visualization: _VisualizationSnapshot | None,
    projection: _ReportProjection,
) -> list[str]:
    lines = [
        "## Provenance and Reproducibility",
        "",
        f"- Run ID: {_inline_code(evidence.run_id)}",
        f"- Request ID: {_inline_code(evidence.request_id)}",
        f"- Plan ID: {_inline_code(evidence.plan_id)}",
        f"- Evidence SHA-256: {_inline_code(evidence.sha256)}",
        f"- Report specification version: {_inline_code(REPORT_SPEC_VERSION)}",
        "- Ordered scientific tools: "
        + ", ".join(_inline_code(step.tool_name) for step in projection.steps),
    ]
    if visualization is not None:
        lines.append(
            "- Visualization manifest SHA-256: "
            + _inline_code(visualization.manifest_sha256)
        )
    else:
        lines.append("- Visualization provenance: none supplied.")
    lines.extend(
        (
            "- Source validation: the run, evidence, and supplied visualizations "
            "were freshly verified without invoking scientific tools.",
            "",
            "### Source artifact integrity",
            "",
        )
    )
    if not projection.artifact_integrity:
        lines.append("- No persisted scientific artifact was projected by evidence.")
    for artifact in projection.artifact_integrity:
        prefix = (
            f"- {_inline_code(artifact.artifact_kind)} from step "
            f"{_inline_code(artifact.producing_step_id)}: "
        )
        if artifact.authoritative_digest is not None:
            lines.append(prefix + "authoritative digest recorded by evidence.")
        else:
            basis = ", ".join(_inline_code(value) for value in artifact.verification_basis)
            lines.append(prefix + "verifier-based integrity (" + basis + ").")
    lines.extend(
        (
            "",
            "Verifier-based integrity is not presented as universal whole-file "
            "cryptographic hashing.",
            "",
        )
    )
    return lines


def _render_markdown(
    evidence: _EvidenceSnapshot,
    visualization: _VisualizationSnapshot | None,
    projection: _ReportProjection,
) -> bytes:
    lines = ["# Single-cell Epigenomic Analysis Report", ""]
    lines.extend(_render_summary(projection, visualization))
    for section in projection.sections:
        if section.section_id in {"analysis_summary", "figures", "methods", "provenance"}:
            continue
        lines.extend(_render_main_section(projection, section))
    if visualization is not None:
        lines.extend(_render_figures(visualization))
    lines.extend(_render_methods(projection))
    lines.extend(_render_provenance(evidence, visualization, projection))
    return (_LF.join(lines).rstrip() + _LF).encode("utf-8")


def _artifact_integrity_payload(
    values: Sequence[_ArtifactIntegrity],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "producing_step_id": value.producing_step_id,
            "tool_name": value.tool_name,
            "artifact_kind": value.artifact_kind,
            "authoritative_digest": value.authoritative_digest,
            "verification_basis": value.verification_basis,
        }
        for value in values
    )


def _figure_manifest_records(
    visualization: _VisualizationSnapshot | None,
    projection: _ReportProjection,
) -> tuple[Mapping[str, object], ...]:
    if visualization is None:
        return ()
    return tuple(
        {
            "figure_id": figure.figure_id,
            "figure_kind": figure.figure_kind,
            "source_step_ids": figure.source_step_ids,
            "associated_fact_ids": tuple(
                fact.fact_id
                for fact in projection.facts
                if fact.source_step_id in figure.source_step_ids
            ),
            "source_visualization_relative_path": figure.relative_path,
            "source_png_sha256": figure.png_sha256,
            "report_relative_path": (
                f"figures/{PurePosixPath(figure.relative_path).name}"
            ),
            "copied_png_sha256": figure.png_sha256,
            "source_and_copy_bytes_equal": True,
            "width_px": figure.width_px,
            "height_px": figure.height_px,
            "dpi": figure.dpi,
            "caption": figure.caption,
        }
        for figure in visualization.figures
    )


def _manifest_payload(
    evidence: _EvidenceSnapshot,
    visualization: _VisualizationSnapshot | None,
    projection: _ReportProjection,
    report_sha256: str,
) -> Mapping[str, object]:
    facts_payload = tuple(fact.to_dict() for fact in projection.facts)
    source_visualization: Mapping[str, object] | None = None
    if visualization is not None:
        source_visualization = {
            "manifest_path": str(visualization.manifest_path),
            "manifest_sha256": visualization.manifest_sha256,
            "schema_version": ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
            "artifact_type": ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
            "evidence_path": str(evidence.path),
            "evidence_sha256": evidence.sha256,
        }
    return {
        "schema_version": ANALYSIS_REPORT_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_REPORT_ARTIFACT_TYPE,
        "status": "success",
        "report_spec_version": REPORT_SPEC_VERSION,
        "source": {
            "run_id": evidence.run_id,
            "request_id": evidence.request_id,
            "plan_id": evidence.plan_id,
            "evidence_path": str(evidence.path),
            "evidence_sha256": evidence.sha256,
            "evidence_schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
            "evidence_artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
            "visualization": source_visualization,
        },
        "report": {
            "relative_path": ANALYSIS_REPORT_FILENAME,
            "sha256": report_sha256,
            "format": "markdown",
            "encoding": "utf-8",
            "line_endings": "LF",
        },
        "content": {
            "ordered_sections": tuple(section.section_id for section in projection.sections),
            "facts": facts_payload,
            "facts_sha256": _sha256_bytes(
                _canonical_json_bytes(
                    {"domain": _FACT_DIGEST_DOMAIN, "facts": facts_payload}
                )
            ),
            "section_bindings": tuple(
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "source_step_ids": section.source_step_ids,
                    "fact_ids": section.fact_ids,
                    "figure_ids": section.figure_ids,
                }
                for section in projection.sections
            ),
            "source_artifact_integrity": _artifact_integrity_payload(
                projection.artifact_integrity
            ),
        },
        "figures": _figure_manifest_records(visualization, projection),
        "generator": {
            "identity": _GENERATOR_IDENTITY,
            "report_spec_version": REPORT_SPEC_VERSION,
            "deterministic_templates": True,
            "llm_narrative_used": False,
        },
        "publication": {
            "bundle_directory": ANALYSIS_REPORT_BUNDLE_DIRNAME,
            "completion_marker": ANALYSIS_REPORT_MANIFEST_FILENAME,
            "strategy": "staged_directory_with_backup_rollback",
            "nonempty_directory_exchange_universally_atomic": False,
        },
        "validation": {
            "fresh_evidence_verified": True,
            "fresh_visualization_verified_when_supplied": visualization is not None,
            "scientific_tools_invoked": False,
            "arbitrary_scientific_files_opened": False,
            "scientific_metrics_recomputed": False,
            "images_interpreted": False,
            "markdown_exactly_regenerable": True,
            "copied_figures_byte_identical": visualization is not None,
            "large_scientific_payloads_in_manifest": False,
        },
    }


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Report output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, payload: bytes) -> None:
    try:
        with path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AnalysisReportError(
            "REPORT_WRITE_FAILED",
            "A report artifact could not be written safely.",
        ) from exc


def _copy_figure(source: Path, destination: Path, expected_sha256: str) -> None:
    try:
        with source.open("rb") as input_handle, destination.open("wb") as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except OSError as exc:
        raise AnalysisReportError(
            "REPORT_FIGURE_MISMATCH",
            "A verified figure could not be copied safely.",
        ) from exc
    if _sha256_file(destination) != expected_sha256:
        raise AnalysisReportError(
            "REPORT_FIGURE_MISMATCH",
            "A copied report figure differs from its verified source.",
        )


def _files_equal(first: Path, second: Path) -> bool:
    try:
        if first.stat().st_size != second.stat().st_size:
            return False
        with first.open("rb") as first_handle, second.open("rb") as second_handle:
            while True:
                left = first_handle.read(1024 * 1024)
                right = second_handle.read(1024 * 1024)
                if left != right:
                    return False
                if not left:
                    return True
    except OSError as exc:
        raise AnalysisReportError(
            "REPORT_FIGURE_MISMATCH",
            "A report figure could not be compared with its verified source.",
        ) from exc


def _load_report_manifest(path: Path) -> tuple[dict[str, object], bytes]:
    parsed, payload = _strict_json(
        path,
        code="REPORT_MANIFEST_MALFORMED",
        description="Report manifest is not strict valid JSON.",
    )
    if (
        parsed.get("schema_version") != ANALYSIS_REPORT_SCHEMA_VERSION
        or parsed.get("artifact_type") != ANALYSIS_REPORT_ARTIFACT_TYPE
        or parsed.get("status") != "success"
        or parsed.get("report_spec_version") != REPORT_SPEC_VERSION
    ):
        raise AnalysisReportError(
            "REPORT_MANIFEST_MALFORMED",
            "Report manifest identity is invalid.",
        )
    return parsed, payload


def _verify_bundle_contents(
    bundle: Path,
    manifest: Mapping[str, object],
    manifest_bytes: bytes,
    expected_manifest: Mapping[str, object],
    expected_markdown: bytes,
    visualization: _VisualizationSnapshot | None,
) -> None:
    expected_manifest_bytes = _canonical_json_bytes(expected_manifest)
    if (
        manifest_bytes != expected_manifest_bytes
        or manifest != json.loads(expected_manifest_bytes)
    ):
        raise AnalysisReportError(
            "REPORT_MANIFEST_CONTENT_MISMATCH",
            "Report manifest differs from freshly regenerated content.",
        )
    try:
        expected_root = {ANALYSIS_REPORT_FILENAME, ANALYSIS_REPORT_MANIFEST_FILENAME}
        if visualization is not None:
            expected_root.add("figures")
        if {value.name for value in bundle.iterdir()} != expected_root:
            raise ValueError("Unexpected report bundle entries.")
        report_path = bundle / ANALYSIS_REPORT_FILENAME
        if not report_path.is_file() or report_path.is_symlink():
            raise ValueError("Missing or unsafe Markdown report.")
        actual_markdown = report_path.read_bytes()
        if actual_markdown != expected_markdown:
            raise AnalysisReportError(
                "REPORT_MARKDOWN_MISMATCH",
                "Persisted Markdown differs from deterministic regeneration.",
            )
        if _sha256_bytes(actual_markdown) != expected_manifest["report"]["sha256"]:  # type: ignore[index]
            raise AnalysisReportError(
                "REPORT_MARKDOWN_MISMATCH",
                "Persisted Markdown differs from its manifest digest.",
            )
        if visualization is None:
            return
        figures_dir = bundle / "figures"
        if not figures_dir.is_dir() or figures_dir.is_symlink():
            raise ValueError("Missing or unsafe report figures directory.")
        expected_names = {
            PurePosixPath(figure.relative_path).name for figure in visualization.figures
        }
        if {value.name for value in figures_dir.iterdir()} != expected_names:
            raise ValueError("Unexpected report figure entries.")
        for figure in visualization.figures:
            copied = figures_dir / PurePosixPath(figure.relative_path).name
            if not copied.is_file() or copied.is_symlink():
                raise ValueError("Missing or unsafe copied figure.")
            if _sha256_file(copied) != figure.png_sha256 or not _files_equal(
                copied, figure.source_path
            ):
                raise AnalysisReportError(
                    "REPORT_FIGURE_MISMATCH",
                    "A copied report figure differs from its verified source.",
                )
    except AnalysisReportError:
        raise
    except Exception as exc:
        raise AnalysisReportError(
            "REPORT_BUNDLE_INVALID",
            "Report bundle contains missing, unsafe, or unexpected artifacts.",
        ) from exc


def _publish_bundle(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis report already exists: {destination}. "
            "Use overwrite=True to replace it."
        )
    backup: Path | None = None
    staged_installed = False
    parent = destination.parent
    try:
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    dir=parent,
                    prefix=".analysis_report.",
                    suffix=".backup",
                )
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
            staged_installed = True
            _fsync_directory(parent)
        except Exception as publication_error:
            try:
                if backup is not None:
                    if destination.exists():
                        shutil.rmtree(destination)
                    os.replace(backup, destination)
                    backup = None
                    _fsync_directory(parent)
                elif staged_installed and destination.exists():
                    shutil.rmtree(destination)
                    _fsync_directory(parent)
            except Exception as rollback_error:
                raise AnalysisReportError(
                    "REPORT_ROLLBACK_FAILED",
                    "Report publication failed and rollback requires manual recovery.",
                ) from rollback_error
            raise publication_error
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
            _fsync_directory(parent)
    except FileExistsError:
        raise
    except AnalysisReportError:
        raise
    except Exception as exc:
        raise AnalysisReportError(
            "REPORT_PUBLISH_FAILED",
            "Report bundle could not be published safely.",
        ) from exc


def _report_reference(
    report: str | Path | AnalysisReportResult,
) -> tuple[Path, str | None]:
    if isinstance(report, Mapping):
        required = {
            "status",
            "manifest_path",
            "manifest_sha256",
            "bundle_path",
            "report_path",
            "report_sha256",
            "schema_version",
            "artifact_type",
            "report_spec_version",
            "evidence_path",
            "evidence_sha256",
            "visualization_manifest_path",
            "visualization_manifest_sha256",
            "run_id",
            "request_id",
            "plan_id",
            "n_sections",
            "section_ids",
            "n_figures",
            "figures",
        }
        if set(report) != required:
            raise AnalysisReportError(
                "REPORT_RESULT_INVALID",
                "AnalysisReportResult has an invalid schema.",
            )
        manifest_path = report.get("manifest_path")
        manifest_sha256 = report.get("manifest_sha256")
        if (
            report.get("status") != "success"
            or report.get("schema_version") != ANALYSIS_REPORT_SCHEMA_VERSION
            or report.get("artifact_type") != ANALYSIS_REPORT_ARTIFACT_TYPE
            or report.get("report_spec_version") != REPORT_SPEC_VERSION
            or not isinstance(manifest_path, str)
            or not manifest_path
            or not _is_sha256(manifest_sha256)
        ):
            raise AnalysisReportError(
                "REPORT_RESULT_INVALID",
                "AnalysisReportResult contains invalid identity metadata.",
            )
        return Path(manifest_path).expanduser().resolve(), str(manifest_sha256)
    if not isinstance(report, (str, Path)):
        raise TypeError("`report` must be a manifest path or AnalysisReportResult.")
    return Path(report).expanduser().resolve(), None


def _source_snapshots(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    visualization: str | Path | AnalysisVisualizationResult | None,
    registry: ToolRegistry,
) -> tuple[_EvidenceSnapshot, _VisualizationSnapshot | None]:
    evidence_snapshot = _verified_evidence_snapshot(run_result, evidence, registry)
    visualization_snapshot = (
        None
        if visualization is None
        else _verified_visualization_snapshot(
            run_result,
            evidence,
            evidence_snapshot,
            visualization,
            registry,
        )
    )
    return evidence_snapshot, visualization_snapshot


def build_analysis_report(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    output_dir: str | Path,
    *,
    registry: ToolRegistry,
    visualization: str | Path | AnalysisVisualizationResult | None = None,
    overwrite: bool = False,
) -> AnalysisReportResult:
    """Build a deterministic Markdown report from freshly verified sources."""

    if not isinstance(run_result, AgentRunResult):
        raise TypeError("`run_result` must be an AgentRunResult.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    output = _resolve_output_dir(output_dir)
    destination = output / ANALYSIS_REPORT_BUNDLE_DIRNAME
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis report already exists: {destination}. "
            "Use overwrite=True to replace it."
        )

    first_evidence, first_visualization = _source_snapshots(
        run_result, evidence, visualization, registry
    )
    first_projection = _project_report(first_evidence, first_visualization)
    markdown = _render_markdown(
        first_evidence, first_visualization, first_projection
    )
    report_sha256 = _sha256_bytes(markdown)

    staging = Path(
        tempfile.mkdtemp(dir=output, prefix=".analysis_report.", suffix=".tmp")
    )
    published = False
    try:
        _write_bytes(staging / ANALYSIS_REPORT_FILENAME, markdown)
        if first_visualization is not None:
            figures_dir = staging / "figures"
            figures_dir.mkdir()
            for figure in first_visualization.figures:
                _copy_figure(
                    figure.source_path,
                    figures_dir / PurePosixPath(figure.relative_path).name,
                    figure.png_sha256,
                )
            _fsync_directory(figures_dir)

        second_evidence, second_visualization = _source_snapshots(
            run_result, evidence, visualization, registry
        )
        second_projection = _project_report(second_evidence, second_visualization)
        second_markdown = _render_markdown(
            second_evidence, second_visualization, second_projection
        )
        if (
            first_evidence.path != second_evidence.path
            or first_evidence.sha256 != second_evidence.sha256
            or first_evidence.payload != second_evidence.payload
            or (first_visualization is None) != (second_visualization is None)
            or _canonical_json_bytes(
                _manifest_payload(
                    first_evidence,
                    first_visualization,
                    first_projection,
                    report_sha256,
                )
            )
            != _canonical_json_bytes(
                _manifest_payload(
                    second_evidence,
                    second_visualization,
                    second_projection,
                    _sha256_bytes(second_markdown),
                )
            )
            or markdown != second_markdown
        ):
            raise AnalysisReportError(
                "REPORT_SOURCE_CHANGED",
                "A verified report source changed during report preparation.",
            )
        if first_visualization is not None:
            assert second_visualization is not None
            for first_figure, second_figure in zip(
                first_visualization.figures,
                second_visualization.figures,
                strict=True,
            ):
                copied = staging / "figures" / PurePosixPath(first_figure.relative_path).name
                if (
                    first_figure != second_figure
                    or not _files_equal(copied, second_figure.source_path)
                ):
                    raise AnalysisReportError(
                        "REPORT_SOURCE_CHANGED",
                        "A verified source figure changed during report preparation.",
                    )

        manifest = _manifest_payload(
            first_evidence,
            first_visualization,
            first_projection,
            report_sha256,
        )
        manifest_path = staging / ANALYSIS_REPORT_MANIFEST_FILENAME
        _write_bytes(manifest_path, _canonical_json_bytes(manifest))
        loaded, loaded_bytes = _load_report_manifest(manifest_path)
        _verify_bundle_contents(
            staging,
            loaded,
            loaded_bytes,
            manifest,
            markdown,
            first_visualization,
        )
        _fsync_directory(staging)
        _publish_bundle(staging, destination, overwrite=overwrite)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    final_manifest = destination / ANALYSIS_REPORT_MANIFEST_FILENAME
    final_report = destination / ANALYSIS_REPORT_FILENAME
    figure_results: list[ReportFigureResult] = []
    if first_visualization is not None:
        for figure in first_visualization.figures:
            path = destination / "figures" / PurePosixPath(figure.relative_path).name
            figure_results.append(
                {
                    "figure_id": figure.figure_id,
                    "figure_kind": figure.figure_kind,
                    "figure_path": str(path),
                    "png_sha256": _sha256_file(path),
                }
            )
    return {
        "status": "success",
        "manifest_path": str(final_manifest),
        "manifest_sha256": _sha256_file(final_manifest),
        "bundle_path": str(destination),
        "report_path": str(final_report),
        "report_sha256": _sha256_file(final_report),
        "schema_version": ANALYSIS_REPORT_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_REPORT_ARTIFACT_TYPE,
        "report_spec_version": REPORT_SPEC_VERSION,
        "evidence_path": str(first_evidence.path),
        "evidence_sha256": first_evidence.sha256,
        "visualization_manifest_path": (
            None
            if first_visualization is None
            else str(first_visualization.manifest_path)
        ),
        "visualization_manifest_sha256": (
            None
            if first_visualization is None
            else first_visualization.manifest_sha256
        ),
        "run_id": first_evidence.run_id,
        "request_id": first_evidence.request_id,
        "plan_id": first_evidence.plan_id,
        "n_sections": len(first_projection.sections),
        "section_ids": [section.section_id for section in first_projection.sections],
        "n_figures": len(figure_results),
        "figures": figure_results,
    }


def verify_analysis_report(
    run_result: AgentRunResult,
    evidence: str | Path | AnalysisEvidenceResult,
    report: str | Path | AnalysisReportResult,
    *,
    registry: ToolRegistry,
    visualization: str | Path | AnalysisVisualizationResult | None = None,
) -> VerificationResult:
    """Freshly verify report sources and exact deterministic report artifacts."""

    target_id = (
        run_result.run_id if isinstance(run_result, AgentRunResult) else "analysis-report"
    )
    checks: list[VerificationCheck] = []
    try:
        if not isinstance(run_result, AgentRunResult):
            raise TypeError("`run_result` must be an AgentRunResult.")
        if not isinstance(registry, ToolRegistry):
            raise TypeError("`registry` must be a ToolRegistry.")
        evidence_snapshot, visualization_snapshot = _source_snapshots(
            run_result, evidence, visualization, registry
        )
        projection = _project_report(evidence_snapshot, visualization_snapshot)
        markdown = _render_markdown(
            evidence_snapshot, visualization_snapshot, projection
        )
    except (AnalysisReportError, TypeError) as exc:
        code = getattr(exc, "code", "REPORT_SOURCE_EVIDENCE_INVALID")
        checks.append(
            VerificationCheck(
                "report_sources_freshly_verified",
                False,
                "Report sources failed fresh verification.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_report",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Analysis report source verification failed.",
            ),
        )
    checks.append(
        VerificationCheck(
            "report_sources_freshly_verified",
            True,
            "Evidence and supplied visualizations passed fresh verification.",
        )
    )

    try:
        manifest_path, expected_result_digest = _report_reference(report)
        if manifest_path.name != ANALYSIS_REPORT_MANIFEST_FILENAME:
            raise AnalysisReportError(
                "REPORT_RESULT_INVALID",
                "Report manifest has an unexpected filename.",
            )
        bundle = manifest_path.parent
        manifest, manifest_bytes = _load_report_manifest(manifest_path)
        manifest_digest = _sha256_bytes(manifest_bytes)
        if expected_result_digest is not None and manifest_digest != expected_result_digest:
            raise AnalysisReportError(
                "REPORT_RESULT_INVALID",
                "Report manifest differs from its authoritative result digest.",
            )
        source = manifest.get("source")
        if not isinstance(source, Mapping):
            raise AnalysisReportError(
                "REPORT_MANIFEST_MALFORMED",
                "Report manifest lacks source metadata.",
            )
        declares_visualization = source.get("visualization") is not None
        if declares_visualization != (visualization_snapshot is not None):
            raise AnalysisReportError(
                "REPORT_SOURCE_VISUALIZATION_INVALID",
                "Report visualization provenance and caller input disagree.",
            )
        expected_manifest = _manifest_payload(
            evidence_snapshot,
            visualization_snapshot,
            projection,
            _sha256_bytes(markdown),
        )
        _verify_bundle_contents(
            bundle,
            manifest,
            manifest_bytes,
            expected_manifest,
            markdown,
            visualization_snapshot,
        )
        if isinstance(report, Mapping):
            figure_results: list[ReportFigureResult] = []
            if visualization_snapshot is not None:
                for figure in visualization_snapshot.figures:
                    path = bundle / "figures" / PurePosixPath(figure.relative_path).name
                    figure_results.append(
                        {
                            "figure_id": figure.figure_id,
                            "figure_kind": figure.figure_kind,
                            "figure_path": str(path),
                            "png_sha256": figure.png_sha256,
                        }
                    )
            expected_result_fields = {
                "bundle_path": str(bundle),
                "report_path": str(bundle / ANALYSIS_REPORT_FILENAME),
                "report_sha256": _sha256_bytes(markdown),
                "evidence_path": str(evidence_snapshot.path),
                "evidence_sha256": evidence_snapshot.sha256,
                "visualization_manifest_path": (
                    None
                    if visualization_snapshot is None
                    else str(visualization_snapshot.manifest_path)
                ),
                "visualization_manifest_sha256": (
                    None
                    if visualization_snapshot is None
                    else visualization_snapshot.manifest_sha256
                ),
                "run_id": evidence_snapshot.run_id,
                "request_id": evidence_snapshot.request_id,
                "plan_id": evidence_snapshot.plan_id,
                "n_sections": len(projection.sections),
                "section_ids": [section.section_id for section in projection.sections],
                "n_figures": len(figure_results),
                "figures": figure_results,
            }
            if any(report.get(key) != value for key, value in expected_result_fields.items()):
                raise AnalysisReportError(
                    "REPORT_RESULT_INVALID",
                    "AnalysisReportResult does not match verified report artifacts.",
                )
    except (AnalysisReportError, OSError, TypeError) as exc:
        code = getattr(exc, "code", "REPORT_BUNDLE_INVALID")
        checks.append(
            VerificationCheck(
                "report_bundle_valid",
                False,
                "Report bundle failed deterministic verification.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_report",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Analysis report bundle failed verification.",
            ),
        )
    checks.append(
        VerificationCheck(
            "report_bundle_valid",
            True,
            "Manifest, Markdown, attribution, and copied figures passed verification.",
        )
    )
    if expected_result_digest is not None:
        checks.append(
            VerificationCheck(
                "report_manifest_sha256_matches",
                True,
                "Report manifest matches its authoritative result digest.",
            )
        )
    return VerificationResult(
        passed=True,
        target_type="analysis_report",
        target_id=target_id,
        checks=tuple(checks),
    )


__all__ = [
    "ANALYSIS_REPORT_ARTIFACT_TYPE",
    "ANALYSIS_REPORT_BUNDLE_DIRNAME",
    "ANALYSIS_REPORT_FILENAME",
    "ANALYSIS_REPORT_MANIFEST_FILENAME",
    "ANALYSIS_REPORT_SCHEMA_VERSION",
    "REPORT_SPEC_VERSION",
    "AnalysisReportError",
    "AnalysisReportResult",
    "ReportFactRecord",
    "ReportFigureResult",
    "build_analysis_report",
    "verify_analysis_report",
]
