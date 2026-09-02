"""Deterministic, verified evidence projection for completed Agent runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, Mapping, TypedDict

import anndata as ad
import numpy as np

from agent.orchestration.differential_accessibility_verifier import (
    VERIFICATION_R_SCRIPT,
)
from agent.orchestration.registry import ToolRegistry
from agent.orchestration.verifier import verify_run, verify_step
from agent.schemas import (
    AgentError,
    AgentRunResult,
    ErrorCategory,
    JsonValue,
    RunStatus,
    StepOutputRef,
    StepStatus,
    VerificationCheck,
    VerificationResult,
)


ANALYSIS_EVIDENCE_SCHEMA_VERSION = 1
ANALYSIS_EVIDENCE_ARTIFACT_TYPE = "agent.analysis-evidence"
ANALYSIS_EVIDENCE_FILENAME = "analysis_evidence.json"


class AnalysisEvidenceResult(TypedDict):
    """Lightweight result for a persisted verified evidence artifact."""

    status: Literal["success"]
    evidence_path: str
    evidence_sha256: str
    schema_version: int
    artifact_type: str
    run_id: str
    request_id: str
    plan_id: str
    n_steps: int
    tool_names: list[str]
    all_steps_verified: bool


class AnalysisEvidenceError(ValueError):
    """Fail-closed evidence construction error with a stable public code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class _ArtifactProjection:
    result_field: str
    kind: str
    verification_basis: tuple[str, ...]
    digest_field: str | None = None


@dataclass(frozen=True)
class _ToolProjection:
    contract_fields: frozenset[str]
    fact_fields: tuple[str, ...]
    recovery_identity: str
    artifacts: tuple[_ArtifactProjection, ...] = ()


_INSPECTION_FIELDS = frozenset(
    {
        "input_path",
        "n_cells",
        "n_features",
        "x_storage_type",
        "x_is_sparse",
        "x_dtype",
        "nnz",
        "density",
        "obs_columns",
        "var_columns",
        "obs_names_sample",
        "var_names_sample",
    }
)
_EMBEDDING_FIELDS = frozenset(
    {
        "status",
        "input_path",
        "embedding_path",
        "cell_ids_path",
        "n_cells",
        "embedding_dim",
        "embedding_dtype",
        "finite",
        "cell_order_preserved",
        "backend",
        "species",
        "checkpoint_path",
        "device",
    }
)
_NEIGHBORS_FIELDS = frozenset(
    {
        "status",
        "embedding_path",
        "cell_ids_path",
        "analysis_path",
        "n_cells",
        "embedding_dim",
        "n_neighbors",
        "metric",
        "neighbors_method",
        "transformer",
        "random_seed",
        "connectivities_nnz",
        "distances_nnz",
        "finite",
        "cell_order_preserved",
        "backend",
        "software_versions",
    }
)
_CLUSTERING_FIELDS = frozenset(
    {
        "status",
        "input_analysis_path",
        "analysis_path",
        "n_cells",
        "n_clusters",
        "cluster_key",
        "algorithm",
        "resolution",
        "random_seed",
        "cell_order_preserved",
        "backend",
        "software_versions",
    }
)
_UMAP_FIELDS = frozenset(
    {
        "status",
        "input_analysis_path",
        "analysis_path",
        "n_cells",
        "n_components",
        "umap_key",
        "coordinate_dtype",
        "finite",
        "min_dist",
        "spread",
        "random_seed",
        "cell_order_preserved",
        "backend",
        "software_versions",
    }
)
_CLUSTERING_EVALUATION_FIELDS = frozenset(
    {
        "status",
        "analysis_path",
        "reference_h5ad_path",
        "report_path",
        "label_key",
        "cluster_key",
        "n_cells",
        "n_reference_classes",
        "n_predicted_clusters",
        "nmi",
        "ari",
        "ami",
        "homogeneity",
        "finite",
        "cell_order_preserved",
        "metric_backend",
        "average_method",
        "report_schema_version",
        "software_versions",
    }
)
_LABEL_TRANSFER_FIELDS = frozenset(
    {
        "status",
        "annotation_path",
        "annotation_sha256",
        "reference_embedding_path",
        "reference_cell_ids_path",
        "reference_h5ad_path",
        "query_embedding_path",
        "query_cell_ids_path",
        "query_h5ad_path",
        "checkpoint_path",
        "reference_label_key",
        "n_reference_cells",
        "n_query_cells",
        "n_reference_classes",
        "assigned_count",
        "unassigned_count",
        "assignment_rate",
        "embedding_dim",
        "embedding_dtype",
        "n_neighbors",
        "metric",
        "voting_method",
        "min_confidence",
        "backend",
        "species",
        "species_compatible",
        "checkpoint_compatible",
        "cell_order_preserved",
        "finite",
        "reference_embedding_sha256",
        "query_embedding_sha256",
        "reference_cell_ids_sha256",
        "query_cell_ids_sha256",
        "reference_labels_sha256",
        "model_config_sha256",
        "artifact_schema_version",
        "software_versions",
    }
)
_ANNOTATION_EVALUATION_FIELDS = frozenset(
    {
        "status",
        "annotation_path",
        "annotation_sha256",
        "ground_truth_h5ad_path",
        "report_path",
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
        "finite",
        "cell_order_preserved",
        "metric_backend",
        "macro_average",
        "zero_division",
        "report_schema_version",
        "software_versions",
    }
)
_FEATURE_SPACE_FIELDS = frozenset(
    {
        "status",
        "feature_space_path",
        "feature_space_sha256",
        "feature_space_identity_sha256",
        "input_path",
        "source_h5ad_sha256",
        "matrix_source",
        "layer_key",
        "matrix_semantics",
        "semantics_assertion_source",
        "pseudobulk_eligible",
        "species",
        "genome_assembly",
        "coordinate_source",
        "coordinate_system",
        "n_cells",
        "n_features",
        "nnz",
        "source_dtype",
        "source_sparse_format",
        "cell_ids_sha256",
        "feature_ids_sha256",
        "matrix_sha256",
        "coordinates_sha256",
        "artifact_schema_version",
        "software_versions",
    }
)
_PSEUDOBULK_FIELDS = frozenset(
    {
        "status",
        "pseudobulk_path",
        "pseudobulk_sha256",
        "feature_space_path",
        "feature_space_sha256",
        "feature_space_identity_sha256",
        "source_h5ad_path",
        "source_h5ad_sha256",
        "matrix_semantics",
        "output_value_semantics",
        "aggregation_method",
        "output_dtype",
        "group_source",
        "group_key",
        "replicate_key",
        "condition_key",
        "covariate_keys",
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
        "all_cells_accounted_for",
        "feature_order_preserved",
        "artifact_schema_version",
        "software_versions",
    }
)
_DIFFERENTIAL_ACCESSIBILITY_FIELDS = frozenset(
    {
        "status",
        "da_path",
        "da_sha256",
        "artifact_type",
        "artifact_schema_version",
        "pseudobulk_path",
        "pseudobulk_sha256",
        "preparation_sha256",
        "analysis_sha256",
        "group_value",
        "condition_key",
        "numerator_condition",
        "denominator_condition",
        "design_type",
        "n_samples",
        "n_numerator_replicates",
        "n_denominator_replicates",
        "design_rank",
        "residual_degrees_of_freedom",
        "warning_codes",
        "n_warnings",
        "n_input_features",
        "n_tested_features",
        "n_filtered_features",
        "filtering_method",
        "normalization_method",
        "backend_pipeline",
        "production_r_script_sha256",
        "r_version",
        "bioconductor_version",
        "edger_version",
        "package_versions",
    }
)


_TOOL_PROJECTIONS: Mapping[str, _ToolProjection] = {
    "inspect_scATAC": _ToolProjection(
        _INSPECTION_FIELDS,
        tuple(sorted(_INSPECTION_FIELDS)),
        "inspect-scatac-v2",
    ),
    "epizoo_embed_cells": _ToolProjection(
        _EMBEDDING_FIELDS,
        (
            "input_path",
            "n_cells",
            "embedding_dim",
            "embedding_dtype",
            "finite",
            "cell_order_preserved",
            "backend",
            "species",
            "checkpoint_path",
            "device",
        ),
        "epizoo-embed-cells-v2",
        (
            _ArtifactProjection(
                "embedding_path",
                "epizoo_embedding_npy",
                ("fresh_existing_verifier", "existence_and_nonempty"),
            ),
            _ArtifactProjection(
                "cell_ids_path",
                "ordered_cell_ids_text",
                ("fresh_existing_verifier", "existence_and_nonempty"),
            ),
        ),
    ),
    "build_cell_neighbors": _ToolProjection(
        _NEIGHBORS_FIELDS,
        (
            "n_cells",
            "embedding_dim",
            "n_neighbors",
            "metric",
            "neighbors_method",
            "transformer",
            "random_seed",
            "connectivities_nnz",
            "distances_nnz",
            "finite",
            "cell_order_preserved",
            "backend",
            "software_versions",
        ),
        "build-cell-neighbors-v1",
        (
            _ArtifactProjection(
                "analysis_path",
                "cell_neighbors_h5ad",
                (
                    "fresh_existing_verifier",
                    "compact_h5ad_structure",
                    "cell_order_validation",
                    "stage_and_source_provenance",
                ),
            ),
        ),
    ),
    "cluster_cells": _ToolProjection(
        _CLUSTERING_FIELDS,
        (
            "n_cells",
            "n_clusters",
            "cluster_key",
            "algorithm",
            "resolution",
            "random_seed",
            "cell_order_preserved",
            "backend",
            "software_versions",
        ),
        "cluster-cells-v1",
        (
            _ArtifactProjection(
                "analysis_path",
                "clustered_cells_h5ad",
                (
                    "fresh_existing_verifier",
                    "compact_h5ad_structure",
                    "cell_order_validation",
                    "stage_and_source_provenance",
                ),
            ),
        ),
    ),
    "compute_cell_umap": _ToolProjection(
        _UMAP_FIELDS,
        (
            "n_cells",
            "n_components",
            "umap_key",
            "coordinate_dtype",
            "finite",
            "min_dist",
            "spread",
            "random_seed",
            "cell_order_preserved",
            "backend",
            "software_versions",
        ),
        "compute-cell-umap-v1",
        (
            _ArtifactProjection(
                "analysis_path",
                "cell_umap_h5ad",
                (
                    "fresh_existing_verifier",
                    "compact_h5ad_structure",
                    "cell_order_validation",
                    "stage_and_source_provenance",
                ),
            ),
        ),
    ),
    "evaluate_cell_clustering": _ToolProjection(
        _CLUSTERING_EVALUATION_FIELDS,
        (
            "label_key",
            "cluster_key",
            "n_cells",
            "n_reference_classes",
            "n_predicted_clusters",
            "nmi",
            "ari",
            "ami",
            "homogeneity",
            "finite",
            "cell_order_preserved",
            "metric_backend",
            "average_method",
            "report_schema_version",
            "software_versions",
        ),
        "evaluate-cell-clustering-v1",
        (
            _ArtifactProjection(
                "report_path",
                "clustering_evaluation_json",
                (
                    "fresh_existing_verifier",
                    "strict_json_schema",
                    "independent_metric_recomputation",
                    "source_and_label_digests",
                ),
            ),
        ),
    ),
    "transfer_cell_labels": _ToolProjection(
        _LABEL_TRANSFER_FIELDS,
        (
            "checkpoint_path",
            "reference_label_key",
            "n_reference_cells",
            "n_query_cells",
            "n_reference_classes",
            "assigned_count",
            "unassigned_count",
            "assignment_rate",
            "embedding_dim",
            "embedding_dtype",
            "n_neighbors",
            "metric",
            "voting_method",
            "min_confidence",
            "backend",
            "species",
            "species_compatible",
            "checkpoint_compatible",
            "cell_order_preserved",
            "finite",
            "reference_embedding_sha256",
            "query_embedding_sha256",
            "reference_cell_ids_sha256",
            "query_cell_ids_sha256",
            "reference_labels_sha256",
            "model_config_sha256",
            "artifact_schema_version",
            "software_versions",
        ),
        "transfer-cell-labels-v1",
        (
            _ArtifactProjection(
                "annotation_path",
                "cell_label_transfer_h5ad",
                (
                    "authoritative_whole_file_sha256",
                    "fresh_existing_verifier",
                    "compact_h5ad_structure",
                    "cell_order_and_source_digest_validation",
                ),
                digest_field="annotation_sha256",
            ),
        ),
    ),
    "evaluate_cell_annotation": _ToolProjection(
        _ANNOTATION_EVALUATION_FIELDS,
        (
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
            "finite",
            "cell_order_preserved",
            "metric_backend",
            "macro_average",
            "zero_division",
            "report_schema_version",
            "software_versions",
        ),
        "evaluate-cell-annotation-v1",
        (
            _ArtifactProjection(
                "report_path",
                "annotation_evaluation_json",
                (
                    "fresh_existing_verifier",
                    "strict_json_schema",
                    "independent_metric_recomputation",
                    "source_prediction_and_label_digests",
                ),
            ),
        ),
    ),
    "validate_scATAC_feature_space": _ToolProjection(
        _FEATURE_SPACE_FIELDS,
        (
            "input_path",
            "source_h5ad_sha256",
            "feature_space_identity_sha256",
            "matrix_source",
            "layer_key",
            "matrix_semantics",
            "semantics_assertion_source",
            "pseudobulk_eligible",
            "species",
            "genome_assembly",
            "coordinate_source",
            "coordinate_system",
            "n_cells",
            "n_features",
            "nnz",
            "source_dtype",
            "source_sparse_format",
            "cell_ids_sha256",
            "feature_ids_sha256",
            "matrix_sha256",
            "coordinates_sha256",
            "artifact_schema_version",
            "software_versions",
        ),
        "validate-scatac-feature-space-v1",
        (
            _ArtifactProjection(
                "feature_space_path",
                "regulatory_feature_space_json",
                (
                    "authoritative_whole_file_sha256",
                    "fresh_existing_verifier",
                    "source_matrix_and_identity_recomputation",
                ),
                digest_field="feature_space_sha256",
            ),
        ),
    ),
    "build_replicate_pseudobulk": _ToolProjection(
        _PSEUDOBULK_FIELDS,
        (
            "pseudobulk_sha256",
            "feature_space_sha256",
            "feature_space_identity_sha256",
            "source_h5ad_sha256",
            "matrix_semantics",
            "output_value_semantics",
            "aggregation_method",
            "output_dtype",
            "group_source",
            "group_key",
            "replicate_key",
            "condition_key",
            "covariate_keys",
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
            "all_cells_accounted_for",
            "feature_order_preserved",
            "artifact_schema_version",
            "software_versions",
        ),
        "build-replicate-pseudobulk-v1",
        (
            _ArtifactProjection(
                "pseudobulk_path",
                "replicate_pseudobulk_h5ad",
                (
                    "authoritative_whole_file_sha256",
                    "fresh_existing_verifier",
                    "exact_independent_sparse_sum_recomputation",
                    "source_feature_metadata_and_order_validation",
                ),
                digest_field="pseudobulk_sha256",
            ),
        ),
    ),
    "run_replicate_differential_accessibility": _ToolProjection(
        _DIFFERENTIAL_ACCESSIBILITY_FIELDS,
        (
            "da_sha256",
            "pseudobulk_sha256",
            "preparation_sha256",
            "analysis_sha256",
            "group_value",
            "condition_key",
            "numerator_condition",
            "denominator_condition",
            "design_type",
            "n_samples",
            "n_numerator_replicates",
            "n_denominator_replicates",
            "design_rank",
            "residual_degrees_of_freedom",
            "warning_codes",
            "n_warnings",
            "n_input_features",
            "n_tested_features",
            "n_filtered_features",
            "filtering_method",
            "normalization_method",
            "backend_pipeline",
            "production_r_script_sha256",
            "r_version",
            "bioconductor_version",
            "edger_version",
            "package_versions",
            "artifact_schema_version",
        ),
        "run-replicate-differential-accessibility-edger-ql-v1",
        (
            _ArtifactProjection(
                "da_path",
                "replicate_differential_accessibility_h5ad",
                (
                    "authoritative_whole_file_sha256",
                    "fresh_independent_python_reconstruction",
                    "fresh_independent_edger_recomputation",
                    "exact_result_and_provenance_digest_validation",
                ),
                digest_field="da_sha256",
            ),
        ),
    ),
}


def _plain_json(value: object, path: str = "value") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise AnalysisEvidenceError(
                "EVIDENCE_VALUE_INVALID",
                "Analysis evidence cannot contain non-finite values.",
            )
        return value
    if isinstance(value, np.generic):
        return _plain_json(value.item(), path)
    if isinstance(value, np.ndarray):
        return tuple(
            _plain_json(nested, f"{path}[{index}]")
            for index, nested in enumerate(value.tolist())
        )
    if isinstance(value, Mapping):
        copied: dict[str, JsonValue] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise AnalysisEvidenceError(
                    "EVIDENCE_VALUE_INVALID",
                    "Analysis evidence mappings require string keys.",
                )
            copied[key] = _plain_json(nested, f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return tuple(
            _plain_json(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        )
    raise AnalysisEvidenceError(
        "EVIDENCE_VALUE_INVALID",
        "Analysis evidence contains an unsupported value type.",
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
        raise AnalysisEvidenceError(
            "EVIDENCE_NOT_JSON_SAFE",
            "Analysis evidence is not strict JSON-safe data.",
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _resolved_arguments_match(
    step: object,
    resolved_arguments: Mapping[str, object],
    verified_results: Mapping[str, Mapping[str, object]],
) -> bool:
    arguments = getattr(step, "arguments", None)
    if not isinstance(arguments, Mapping) or set(arguments) != set(resolved_arguments):
        return False
    for name, planned in arguments.items():
        if isinstance(planned, StepOutputRef):
            producer = verified_results.get(planned.step_id)
            if producer is None or planned.output_key not in producer:
                return False
            expected = producer[planned.output_key]
        else:
            expected = planned
        if _plain_json(resolved_arguments[name]) != _plain_json(expected):
            return False
    return True


def _project_artifacts(
    step_id: str,
    tool_name: str,
    result: Mapping[str, object],
    projection: _ToolProjection,
) -> list[dict[str, JsonValue]]:
    artifacts: list[dict[str, JsonValue]] = []
    for artifact in projection.artifacts:
        path = result.get(artifact.result_field)
        if not isinstance(path, str) or not path:
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_RESULT_INVALID",
                "A verified source result lacks required artifact metadata.",
            )
        authoritative_digest: dict[str, JsonValue] | None = None
        if artifact.digest_field is not None:
            digest = result.get(artifact.digest_field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise AnalysisEvidenceError(
                    "EVIDENCE_SOURCE_RESULT_INVALID",
                    "A verified source result lacks its authoritative artifact digest.",
                )
            try:
                int(digest, 16)
            except ValueError as exc:
                raise AnalysisEvidenceError(
                    "EVIDENCE_SOURCE_RESULT_INVALID",
                    "A verified source result has an invalid artifact digest.",
                ) from exc
            authoritative_digest = {
                "algorithm": "sha256",
                "value": digest,
                "source_result_field": artifact.digest_field,
            }
        artifacts.append(
            {
                "producing_step_id": step_id,
                "tool_name": tool_name,
                "result_field": artifact.result_field,
                "artifact_kind": artifact.kind,
                "artifact_path": path,
                "integrity": {
                    "authoritative_digest": authoritative_digest,
                    "verification_basis": artifact.verification_basis,
                },
            }
        )
    return artifacts


def _differential_accessibility_derived_facts(
    resolved_arguments: Mapping[str, object], result: Mapping[str, object]
) -> dict[str, JsonValue]:
    path_value = result.get("da_path")
    if not isinstance(path_value, str) or not path_value:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RESULT_INVALID",
            "A verified DA result lacks its artifact path.",
        )
    artifact = None
    try:
        artifact = ad.read_h5ad(Path(path_value).expanduser().resolve(), backed="r")
        provenance = artifact.uns["agent_milestone8_differential_accessibility"]
        comparison = provenance["comparison"]
        filtering = provenance["filter"]
        normalization = provenance["normalization"]
        statistical_test = provenance["statistical_test"]
        result_sha256 = statistical_test["result_sha256"]
        if not isinstance(result_sha256, str) or len(result_sha256) != 64:
            raise ValueError
        int(result_sha256, 16)
        verifier_sha256 = hashlib.sha256(VERIFICATION_R_SCRIPT.read_bytes()).hexdigest()
        return {
            "positive_logfc_meaning": _plain_json(
                comparison["positive_logfc_meaning"]
            ),
            "covariates": _plain_json(resolved_arguments.get("covariates", ())),
            "filter_configuration": _plain_json(filtering),
            "normalization_configuration": _plain_json(normalization),
            "ql_configuration": _plain_json(statistical_test),
            "result_sha256": result_sha256,
            "verifier_r_script_sha256": verifier_sha256,
        }
    except Exception as exc:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RESULT_INVALID",
            "Freshly verified DA provenance could not be projected.",
        ) from exc
    finally:
        if artifact is not None and artifact.file is not None:
            artifact.file.close()


def _prepare_evidence(
    run_result: AgentRunResult,
    registry: ToolRegistry,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if not isinstance(run_result, AgentRunResult):
        raise TypeError("`run_result` must be an AgentRunResult.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if run_result.status is not RunStatus.SUCCEEDED or run_result.planning_only:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RUN_NOT_SUCCEEDED",
            "Analysis evidence requires a successful executed Agent run.",
        )
    plan = run_result.plan
    if plan is None or not plan.steps:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_PLAN_MISSING",
            "Analysis evidence requires a non-empty validated Agent plan.",
        )
    if (
        plan.request_id != run_result.request_id
        or run_result.run_id != f"{run_result.request_id}:run"
        or run_result.errors
    ):
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RUN_IDENTITY_INVALID",
            "Analysis evidence requires a consistent successful source-run identity.",
        )
    if (
        run_result.verification is None
        or not run_result.verification.passed
        or run_result.verification.target_type != "run"
        or run_result.verification.target_id != plan.plan_id
    ):
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RUN_NOT_VERIFIED",
            "Analysis evidence requires matching successful stored run verification.",
        )

    by_id: dict[str, list[object]] = {}
    for result in run_result.steps:
        by_id.setdefault(result.step_id, []).append(result)
    if set(by_id) != {step.step_id for step in plan.steps} or any(
        len(values) != 1 for values in by_id.values()
    ):
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_STEP_SET_INVALID",
            "Analysis evidence requires exactly one result for every planned step.",
        )

    for step in plan.steps:
        stored = by_id[step.step_id][0]
        if (
            stored.tool_name != step.tool_name
            or stored.status is not StepStatus.SUCCEEDED
            or stored.result is None
            or stored.error is not None
            or stored.verification is None
            or not stored.verification.passed
            or stored.verification.target_type != "step"
            or stored.verification.target_id != step.step_id
        ):
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_STEP_NOT_VERIFIED",
                "Every source step must have matching successful stored verification.",
            )

    try:
        fresh_run_verification = verify_run(plan, run_result.steps)
    except Exception as exc:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RUN_REVALIDATION_FAILED",
            "Fresh source-run verification could not be completed.",
        ) from exc
    if not fresh_run_verification.passed:
        raise AnalysisEvidenceError(
            "EVIDENCE_SOURCE_RUN_REVALIDATION_FAILED",
            "Fresh source-run verification failed.",
        )

    verified_results: dict[str, Mapping[str, object]] = {}
    evidence_steps: list[dict[str, object]] = []
    evidence_artifacts: list[dict[str, JsonValue]] = []
    registry_identities: list[dict[str, JsonValue]] = []
    ordered = plan.stable_topological_steps()
    for step in ordered:
        step_result = by_id[step.step_id][0]
        if (
            step_result.tool_name != step.tool_name
            or step_result.status is not StepStatus.SUCCEEDED
            or step_result.result is None
            or step_result.error is not None
            or step_result.verification is None
            or not step_result.verification.passed
            or step_result.verification.target_type != "step"
            or step_result.verification.target_id != step.step_id
        ):
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_STEP_NOT_VERIFIED",
                "Every source step must have matching successful stored verification.",
            )
        projection = _TOOL_PROJECTIONS.get(step.tool_name)
        if projection is None:
            raise AnalysisEvidenceError(
                "EVIDENCE_TOOL_UNSUPPORTED",
                "AnalysisEvidence schema v1 does not support a planned tool.",
            )
        try:
            spec = registry.get(step.tool_name)
        except Exception as exc:
            raise AnalysisEvidenceError(
                "EVIDENCE_TOOL_UNSUPPORTED",
                "Analysis evidence requires every source tool in the supplied registry.",
            ) from exc
        if (
            frozenset(spec.result_contract.required_fields)
            != projection.contract_fields
            or spec.recovery_policy_version != projection.recovery_identity
        ):
            raise AnalysisEvidenceError(
                "EVIDENCE_TOOL_SCHEMA_INCOMPATIBLE",
                "A source tool contract is incompatible with AnalysisEvidence schema v1.",
            )
        if not _resolved_arguments_match(
            step, step_result.resolved_arguments, verified_results
        ):
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_ARGUMENTS_INVALID",
                "Source resolved arguments do not match verified plan bindings.",
            )
        try:
            registry.validate_arguments(step.tool_name, step_result.resolved_arguments)
        except Exception as exc:
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_ARGUMENTS_INVALID",
                "Source resolved arguments violate the supplied registry contract.",
            ) from exc
        dependencies = {
            dependency: verified_results[dependency]
            for dependency in step.depends_on
            if dependency in verified_results
        }
        if len(dependencies) != len(step.depends_on):
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_DEPENDENCY_INVALID",
                "A source step dependency was not freshly verified.",
            )
        try:
            fresh_step_verification = verify_step(
                step,
                step_result.resolved_arguments,
                step_result.result,
                registry,
                dependency_results=dependencies,
            )
        except Exception as exc:
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED",
                "Fresh source-step verification could not be completed.",
            ) from exc
        if not fresh_step_verification.passed:
            raise AnalysisEvidenceError(
                "EVIDENCE_SOURCE_STEP_REVALIDATION_FAILED",
                "Fresh source-step verification failed.",
            )

        facts = {
            field: _plain_json(step_result.result[field], f"{step.step_id}.{field}")
            for field in projection.fact_fields
        }
        if step.tool_name == "run_replicate_differential_accessibility":
            facts.update(
                _differential_accessibility_derived_facts(
                    step_result.resolved_arguments, step_result.result
                )
            )
        evidence_steps.append(
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "attempt_count": step_result.attempt_count,
                "facts": facts,
                "verification": {
                    "freshly_verified": True,
                    "passed": True,
                    "check_names": tuple(
                        check.name for check in fresh_step_verification.checks
                    ),
                },
                "recovery_identity": projection.recovery_identity,
            }
        )
        evidence_artifacts.extend(
            _project_artifacts(
                step.step_id, step.tool_name, step_result.result, projection
            )
        )
        registry_identities.append(
            {
                "tool_name": step.tool_name,
                "result_contract": spec.result_contract.name,
                "recovery_identity": projection.recovery_identity,
            }
        )
        verified_results[step.step_id] = step_result.result

    plan_payload = plan.to_dict()
    run_payload = run_result.to_dict()
    trace_payload = [event.to_dict() for event in run_result.trace]
    content: dict[str, object] = {
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "status": "success",
        "run": {
            "run_id": run_result.run_id,
            "request_id": run_result.request_id,
            "plan_id": plan.plan_id,
            "planner_name": plan.planner_name,
            "source_run_status": run_result.status.value,
            "plan_sha256": _sha256_json(plan_payload),
            "source_run_result_sha256": _sha256_json(run_payload),
        },
        "workflow": {
            "ordered_steps": tuple(
                {
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "depends_on": step.depends_on,
                }
                for step in ordered
            )
        },
        "steps": tuple(evidence_steps),
        "artifacts": tuple(evidence_artifacts),
        "provenance": {
            "trace_sha256": _sha256_json(trace_payload),
            "trace_event_count": len(run_result.trace),
            "registry_tool_identities": tuple(registry_identities),
            "evidence_projection_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        },
        "validation": {
            "fresh_run_verification_passed": True,
            "all_steps_freshly_verified": True,
            "all_dependencies_resolved_from_verified_results": True,
            "prohibited_large_scientific_payloads_included": False,
            "scientific_tools_invoked_during_evidence_processing": False,
        },
    }
    # Force the complete projection through the strict JSON boundary now.
    _canonical_json_bytes(content)
    return content, tuple(step.tool_name for step in ordered)


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


def _load_and_validate_evidence(
    path: Path,
    expected: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except Exception as exc:
        raise AnalysisEvidenceError(
            "EVIDENCE_ARTIFACT_MALFORMED",
            "Analysis evidence is not strict valid JSON.",
        ) from exc
    if not isinstance(parsed, dict):
        raise AnalysisEvidenceError(
            "EVIDENCE_ARTIFACT_MALFORMED",
            "Analysis evidence must contain one JSON object.",
        )
    if parsed.get("schema_version") != ANALYSIS_EVIDENCE_SCHEMA_VERSION:
        raise AnalysisEvidenceError(
            "EVIDENCE_SCHEMA_UNSUPPORTED",
            "Analysis evidence uses an unsupported schema version.",
        )
    if parsed.get("artifact_type") != ANALYSIS_EVIDENCE_ARTIFACT_TYPE:
        raise AnalysisEvidenceError(
            "EVIDENCE_ARTIFACT_TYPE_INVALID",
            "Analysis evidence has an invalid artifact type.",
        )
    if parsed.get("status") != "success":
        raise AnalysisEvidenceError(
            "EVIDENCE_ARTIFACT_STATUS_INVALID",
            "Analysis evidence status is invalid.",
        )
    expected_payload = _canonical_json_bytes(expected)
    if payload != expected_payload or parsed != json.loads(expected_payload):
        raise AnalysisEvidenceError(
            "EVIDENCE_CONTENT_MISMATCH",
            "Analysis evidence differs from freshly verified source results.",
        )
    return parsed, _sha256_bytes(payload)


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Analysis evidence output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _evidence_reference(
    evidence: str | Path | AnalysisEvidenceResult,
) -> tuple[Path, str | None]:
    if isinstance(evidence, Mapping):
        required = {
            "status",
            "evidence_path",
            "evidence_sha256",
            "schema_version",
            "artifact_type",
            "run_id",
            "request_id",
            "plan_id",
            "n_steps",
            "tool_names",
            "all_steps_verified",
        }
        if set(evidence) != required:
            raise AnalysisEvidenceError(
                "EVIDENCE_RESULT_INVALID",
                "AnalysisEvidenceResult has an invalid schema.",
            )
        path_value = evidence.get("evidence_path")
        digest = evidence.get("evidence_sha256")
        if (
            evidence.get("status") != "success"
            or evidence.get("schema_version") != ANALYSIS_EVIDENCE_SCHEMA_VERSION
            or evidence.get("artifact_type") != ANALYSIS_EVIDENCE_ARTIFACT_TYPE
            or evidence.get("all_steps_verified") is not True
            or not isinstance(path_value, str)
            or not path_value
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise AnalysisEvidenceError(
                "EVIDENCE_RESULT_INVALID",
                "AnalysisEvidenceResult contains invalid identity or digest fields.",
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise AnalysisEvidenceError(
                "EVIDENCE_RESULT_INVALID",
                "AnalysisEvidenceResult contains an invalid SHA-256 digest.",
            ) from exc
        return Path(path_value).expanduser().resolve(), digest
    if not isinstance(evidence, (str, Path)):
        raise TypeError(
            "`evidence_path` must be a path or AnalysisEvidenceResult mapping."
        )
    return Path(evidence).expanduser().resolve(), None


def build_analysis_evidence(
    run_result: AgentRunResult,
    output_dir: str | Path,
    *,
    registry: ToolRegistry,
    overwrite: bool = False,
) -> AnalysisEvidenceResult:
    """Freshly verify a successful run and persist compact analysis evidence."""

    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    content, tool_names = _prepare_evidence(run_result, registry)
    assert run_result.plan is not None  # established by _prepare_evidence
    directory = _resolve_output_dir(output_dir)
    output_path = directory / ANALYSIS_EVIDENCE_FILENAME
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Analysis evidence already exists: {output_path}. "
            "Use overwrite=True to replace it."
        )
    payload = _canonical_json_bytes(content)
    digest = _sha256_bytes(payload)
    temporary: Path | None = None
    published = False
    try:
        descriptor, name = tempfile.mkstemp(
            dir=directory,
            prefix=".analysis_evidence.",
            suffix=".tmp",
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _load_and_validate_evidence(temporary, content)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Analysis evidence already exists: {output_path}. "
                "Use overwrite=True to replace it."
            )
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.replace(temporary, output_path)
            temporary = None
            published = True
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (FileExistsError, AnalysisEvidenceError):
        raise
    except OSError as exc:
        if published:
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise AnalysisEvidenceError(
            "EVIDENCE_WRITE_FAILED",
            "Analysis evidence could not be written safely.",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return {
        "status": "success",
        "evidence_path": str(output_path),
        "evidence_sha256": digest,
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "run_id": run_result.run_id,
        "request_id": run_result.request_id,
        "plan_id": run_result.plan.plan_id,
        "n_steps": len(run_result.plan.steps),
        "tool_names": list(tool_names),
        "all_steps_verified": True,
    }


def verify_analysis_evidence(
    run_result: AgentRunResult,
    evidence_path: str | Path | AnalysisEvidenceResult,
    *,
    registry: ToolRegistry,
) -> VerificationResult:
    """Freshly revalidate source steps and compare strict persisted evidence."""

    target_id = (
        run_result.run_id
        if isinstance(run_result, AgentRunResult)
        else "analysis-evidence"
    )
    checks: list[VerificationCheck] = []
    try:
        expected, tool_names = _prepare_evidence(run_result, registry)
    except (AnalysisEvidenceError, TypeError) as exc:
        code = getattr(exc, "code", "EVIDENCE_SOURCE_INVALID")
        checks.append(
            VerificationCheck(
                "source_run_freshly_verified",
                False,
                "Source run failed fresh evidence-boundary verification.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_evidence",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Analysis evidence source verification failed.",
            ),
        )
    checks.append(
        VerificationCheck(
            "source_run_freshly_verified",
            True,
            "Source run and every source step passed fresh verification.",
        )
    )

    try:
        if isinstance(evidence_path, Mapping):
            run_section = expected["run"]
            if not isinstance(run_section, Mapping) or (
                evidence_path.get("run_id") != run_result.run_id
                or evidence_path.get("request_id") != run_result.request_id
                or evidence_path.get("plan_id") != run_section.get("plan_id")
                or evidence_path.get("n_steps") != len(tool_names)
                or evidence_path.get("tool_names") != list(tool_names)
            ):
                raise AnalysisEvidenceError(
                    "EVIDENCE_RESULT_INVALID",
                    "AnalysisEvidenceResult does not match the freshly verified run.",
                )
        path, expected_digest = _evidence_reference(evidence_path)
        _, actual_digest = _load_and_validate_evidence(path, expected)
        if expected_digest is not None and actual_digest != expected_digest:
            raise AnalysisEvidenceError(
                "EVIDENCE_SHA256_MISMATCH",
                "Analysis evidence SHA-256 does not match its result metadata.",
            )
    except (AnalysisEvidenceError, OSError, TypeError) as exc:
        code = getattr(exc, "code", "EVIDENCE_ARTIFACT_UNAVAILABLE")
        checks.append(
            VerificationCheck(
                "evidence_artifact_valid",
                False,
                "Persisted analysis evidence failed strict validation.",
            )
        )
        return VerificationResult(
            passed=False,
            target_type="analysis_evidence",
            target_id=target_id,
            checks=tuple(checks),
            error=AgentError(
                ErrorCategory.VERIFICATION_ERROR,
                code,
                "Persisted analysis evidence failed verification.",
            ),
        )
    checks.append(
        VerificationCheck(
            "evidence_artifact_valid",
            True,
            "Persisted analysis evidence exactly matches fresh deterministic projection.",
        )
    )
    if expected_digest is not None:
        checks.append(
            VerificationCheck(
                "evidence_sha256_matches",
                True,
                "Persisted analysis evidence matches its authoritative result digest.",
            )
        )
    return VerificationResult(
        passed=True,
        target_type="analysis_evidence",
        target_id=target_id,
        checks=tuple(checks),
    )


__all__ = [
    "ANALYSIS_EVIDENCE_ARTIFACT_TYPE",
    "ANALYSIS_EVIDENCE_FILENAME",
    "ANALYSIS_EVIDENCE_SCHEMA_VERSION",
    "AnalysisEvidenceError",
    "AnalysisEvidenceResult",
    "build_analysis_evidence",
    "verify_analysis_evidence",
]
