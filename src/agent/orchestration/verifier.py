"""Lightweight orchestration verification for tool steps and completed runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from agent.tools.analysis.clustering_evaluation import (
    AVERAGE_METHOD,
    EVALUATION_REPORT_SCHEMA_VERSION,
    METRIC_BACKEND,
    _evaluate_sources,
    _load_evaluation_report,
    _report_from_snapshot,
)
from agent.tools.analysis.embedding_analysis import (
    EPIZOO_EMBEDDING_DIM,
    PROVENANCE_KEY,
    UMAP_KEY,
    _load_cell_ids,
    _provenance,
    _read_analysis,
    _validate_cluster_labels,
    _validate_neighbors_artifact,
)
from agent.tools.analysis.label_transfer import (
    LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION,
    LABEL_TRANSFER_BACKEND,
    LABEL_TRANSFER_VOTING_METHOD,
    _file_sha256 as _label_transfer_file_sha256,
    _prepare_sources as _prepare_label_transfer_sources,
    _read_annotation as _read_label_transfer_annotation,
    _validate_annotation_artifact,
)

from agent.schemas import (
    AgentError,
    AgentPlan,
    ErrorCategory,
    PlanStep,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
    VerificationCheck,
    VerificationResult,
)

from .registry import ToolRegistry, UnknownToolError


class _VerificationChecks:
    def __init__(self) -> None:
        self.checks: list[VerificationCheck] = []
        self.failures: list[tuple[str, str, str]] = []

    def add(
        self,
        name: str,
        passed: bool,
        success_message: str,
        failure_message: str,
        error_code: str,
    ) -> None:
        message = success_message if passed else failure_message
        self.checks.append(VerificationCheck(name=name, passed=passed, message=message))
        if not passed:
            self.failures.append((error_code, failure_message, name))

    def result(
        self,
        *,
        target_type: str,
        target_id: str,
        step_id: str | None = None,
        tool_name: str | None = None,
    ) -> VerificationResult:
        if not self.failures:
            return VerificationResult(
                passed=True,
                target_type=target_type,
                target_id=target_id,
                checks=tuple(self.checks),
            )
        code, _, _ = self.failures[0]
        failed_names = tuple(name for _, _, name in self.failures)
        messages = "; ".join(message for _, message, _ in self.failures)
        return VerificationResult(
            passed=False,
            target_type=target_type,
            target_id=target_id,
            checks=tuple(self.checks),
            error=AgentError(
                category=ErrorCategory.VERIFICATION_ERROR,
                code=code,
                message=messages,
                step_id=step_id,
                tool_name=tool_name,
                details={"failed_checks": failed_names},
            ),
        )


def _plain_json_value(value: object, path: str = "result") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string mapping key.")
            converted[key] = _plain_json_value(nested, f"{path}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [
            _plain_json_value(nested, f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}.")


def _path_corresponds(left: object, right: object) -> bool:
    if not isinstance(left, (str, Path)) or not isinstance(right, (str, Path)):
        return False
    if not str(left).strip() or not str(right).strip():
        return False
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(
            right
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _nonnegative_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _verify_common_step(
    step: PlanStep,
    result: object,
    registry: ToolRegistry,
    checks: _VerificationChecks,
) -> Mapping[str, object] | None:
    try:
        spec = registry.get(step.tool_name)
    except UnknownToolError:
        checks.add(
            "tool_registered",
            False,
            "Tool is registered.",
            f"Tool {step.tool_name!r} is not registered for execution.",
            "UNKNOWN_TOOL",
        )
        try:
            _plain_json_value(result)
        except (TypeError, ValueError):
            checks.add(
                "result_lightweight",
                False,
                "Result is lightweight and JSON-safe.",
                "Result is not lightweight and JSON-safe.",
                "RESULT_NOT_LIGHTWEIGHT",
            )
        else:
            checks.add(
                "result_lightweight",
                True,
                "Result is lightweight and JSON-safe.",
                "Result is not lightweight and JSON-safe.",
                "RESULT_NOT_LIGHTWEIGHT",
            )
        return None

    checks.add(
        "tool_registered",
        True,
        f"Tool {step.tool_name!r} is registered.",
        "Tool is not registered.",
        "UNKNOWN_TOOL",
    )
    checks.add(
        "tool_identity",
        spec.name == step.tool_name,
        "Plan step tool identity matches the registered specification.",
        "Plan step tool identity does not match the registered specification.",
        "TOOL_IDENTITY_MISMATCH",
    )

    plain_result: Mapping[str, object] | None = None
    try:
        converted = _plain_json_value(result)
        if not isinstance(converted, Mapping):
            raise TypeError("result must be a mapping")
        plain_result = converted
    except (TypeError, ValueError):
        checks.add(
            "result_lightweight",
            False,
            "Result is lightweight and JSON-safe.",
            "Result is not a lightweight JSON-safe mapping.",
            "RESULT_NOT_LIGHTWEIGHT",
        )
    else:
        checks.add(
            "result_lightweight",
            True,
            "Result is a lightweight JSON-safe mapping.",
            "Result is not a lightweight JSON-safe mapping.",
            "RESULT_NOT_LIGHTWEIGHT",
        )

    try:
        contract_result = plain_result if plain_result is not None else result
        registry.validate_result(step.tool_name, contract_result)
    except (TypeError, ValueError):
        checks.add(
            "result_contract",
            False,
            "Result satisfies the authoritative registry contract.",
            "Result violates the authoritative registry contract.",
            "RESULT_CONTRACT_INVALID",
        )
    else:
        checks.add(
            "result_contract",
            True,
            "Result satisfies the authoritative registry contract.",
            "Result violates the authoritative registry contract.",
            "RESULT_CONTRACT_INVALID",
        )
    return plain_result


def _verify_inspection(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "input_path_matches",
        _path_corresponds(result.get("input_path"), resolved_arguments.get("path")),
        "Inspection result path matches the resolved input path.",
        "Inspection result input_path does not match the resolved path argument.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "positive_cell_count",
        _positive_integer(result.get("n_cells")),
        "Inspection reports a positive cell count.",
        "Inspection n_cells must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "positive_feature_count",
        _positive_integer(result.get("n_features")),
        "Inspection reports a positive feature count.",
        "Inspection n_features must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )

    nnz = result.get("nnz")
    nnz_valid = nnz is None or _nonnegative_integer(nnz)
    checks.add(
        "nonnegative_nnz",
        nnz_valid,
        "Inspection nnz is absent or nonnegative.",
        "Inspection nnz must be None or a nonnegative integer.",
        "RESULT_VALUE_INVALID",
    )

    density = result.get("density")
    density_valid = density is None or (
        not isinstance(density, bool)
        and isinstance(density, (int, float))
        and math.isfinite(float(density))
        and 0.0 <= float(density) <= 1.0
    )
    checks.add(
        "density_range",
        density_valid,
        "Inspection density is absent or finite within [0, 1].",
        "Inspection density must be None or finite within [0, 1].",
        "RESULT_VALUE_INVALID",
    )

    is_sparse = result.get("x_is_sparse")
    if is_sparse is True:
        sparse_metadata_valid = (
            _nonnegative_integer(nnz) and density_valid and density is not None
        )
        n_cells = result.get("n_cells")
        n_features = result.get("n_features")
        if (
            sparse_metadata_valid
            and _positive_integer(n_cells)
            and _positive_integer(n_features)
        ):
            expected_density = int(nnz) / (int(n_cells) * int(n_features))
            sparse_metadata_valid = math.isclose(
                float(density), expected_density, rel_tol=1e-12, abs_tol=1e-15
            )
    elif is_sparse is False:
        sparse_metadata_valid = nnz is None and density is None
    else:
        sparse_metadata_valid = False
    checks.add(
        "sparse_metadata_coherent",
        sparse_metadata_valid,
        "Sparse storage metadata is internally coherent.",
        "Sparse storage metadata is internally inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _artifact_checks(
    result: Mapping[str, object], checks: _VerificationChecks
) -> None:
    for field_name in ("embedding_path", "cell_ids_path"):
        value = result.get(field_name)
        path = Path(value) if isinstance(value, str) and value.strip() else None
        try:
            exists_as_file = path is not None and path.is_file()
        except OSError:
            exists_as_file = False
        checks.add(
            f"{field_name}_exists",
            exists_as_file,
            f"Artifact {field_name} exists as a regular file.",
            f"Artifact {field_name} is missing or is not a regular file.",
            "ARTIFACT_MISSING",
        )
        nonempty = False
        if exists_as_file and path is not None:
            try:
                nonempty = path.stat().st_size > 0
            except OSError:
                nonempty = False
        checks.add(
            f"{field_name}_nonempty",
            nonempty,
            f"Artifact {field_name} is nonempty.",
            f"Artifact {field_name} is empty or unavailable.",
            "ARTIFACT_EMPTY",
        )


def _analysis_artifact_exists(
    result: Mapping[str, object], checks: _VerificationChecks
) -> Path | None:
    value = result.get("analysis_path")
    path = Path(value) if isinstance(value, str) and value.strip() else None
    try:
        exists = path is not None and path.is_file()
        nonempty = exists and path is not None and path.stat().st_size > 0
    except OSError:
        exists = False
        nonempty = False
    checks.add(
        "analysis_artifact_exists",
        exists,
        "Analysis artifact exists as a regular file.",
        "Analysis artifact is missing or is not a regular file.",
        "ARTIFACT_MISSING",
    )
    checks.add(
        "analysis_artifact_nonempty",
        nonempty,
        "Analysis artifact is nonempty.",
        "Analysis artifact is empty or unavailable.",
        "ARTIFACT_EMPTY",
    )
    return path if exists and nonempty else None


def _verify_neighbors(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "success_status",
        result.get("status") == "success",
        "Neighbor tool reports success.",
        "Neighbor result status must be 'success'.",
        "RESULT_STATUS_INVALID",
    )
    checks.add(
        "backend_identity",
        result.get("backend") == "Scanpy",
        "Neighbor result identifies Scanpy.",
        "Neighbor result backend identity is not Scanpy.",
        "TOOL_IDENTITY_MISMATCH",
    )
    for field_name in ("embedding_path", "cell_ids_path"):
        checks.add(
            f"{field_name}_matches",
            _path_corresponds(
                result.get(field_name), resolved_arguments.get(field_name)
            ),
            f"Neighbor result {field_name} matches its resolved input.",
            f"Neighbor result {field_name} does not match its resolved input.",
            "RESULT_PATH_MISMATCH",
        )
    checks.add(
        "reported_invariants",
        result.get("finite") is True
        and result.get("cell_order_preserved") is True
        and result.get("embedding_dim") == EPIZOO_EMBEDDING_DIM
        and _positive_integer(result.get("n_cells")),
        "Neighbor result reports valid finite ordered dimensions.",
        "Neighbor result reports invalid scientific invariants.",
        "RESULT_VALUE_INVALID",
    )
    path = _analysis_artifact_exists(result, checks)
    if path is None:
        return
    try:
        artifact = _read_analysis(path)
        artifact_ids, distances, connectivities = _validate_neighbors_artifact(
            artifact, expected_stages=frozenset({"neighbors"})
        )
        sidecar_value = resolved_arguments.get("cell_ids_path")
        if not isinstance(sidecar_value, (str, Path)):
            raise ValueError("Invalid sidecar path.")
        sidecar_ids = _load_cell_ids(Path(sidecar_value).expanduser().resolve())
        provenance = _provenance(artifact)
        parameters = provenance.get("parameters")
        neighbor_parameters = (
            parameters.get("neighbors") if isinstance(parameters, Mapping) else None
        )
        structure_valid = (
            artifact.n_obs == result.get("n_cells")
            and int(distances.nnz) == result.get("distances_nnz")
            and int(connectivities.nnz) == result.get("connectivities_nnz")
            and result.get("n_neighbors")
            == resolved_arguments.get("n_neighbors", 15)
            and result.get("metric")
            == resolved_arguments.get("metric", "euclidean")
            and result.get("random_seed")
            == resolved_arguments.get("random_seed", 0)
        )
        order_valid = artifact_ids == sidecar_ids
        provenance_valid = (
            provenance.get("stage") == "neighbors"
            and _path_corresponds(
                provenance.get("source_embedding_path"),
                resolved_arguments.get("embedding_path"),
            )
            and _path_corresponds(
                provenance.get("source_cell_ids_path"),
                resolved_arguments.get("cell_ids_path"),
            )
            and isinstance(neighbor_parameters, Mapping)
            and neighbor_parameters.get("n_neighbors") == result.get("n_neighbors")
            and neighbor_parameters.get("metric") == result.get("metric")
            and neighbor_parameters.get("method") == "umap"
            and neighbor_parameters.get("transformer") == "none"
            and neighbor_parameters.get("random_seed") == result.get("random_seed")
            and neighbor_parameters.get("use_rep") == "X_epizoo"
        )
    except Exception:
        structure_valid = False
        order_valid = False
        provenance_valid = False
    checks.add(
        "neighbors_artifact_structure",
        structure_valid,
        "Neighbor artifact has valid sparse graphs and dimensions.",
        "Neighbor artifact is unreadable or scientifically inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "neighbors_cell_order",
        order_valid,
        "Neighbor artifact cell order exactly matches the ordered ID sidecar.",
        "Neighbor artifact cell order does not match the ordered ID sidecar.",
        "RESULT_IDENTITY_MISMATCH",
    )
    checks.add(
        "neighbors_provenance",
        provenance_valid,
        "Neighbor artifact has valid source and stage provenance.",
        "Neighbor artifact has invalid source or stage provenance.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _verify_clustering(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "success_status",
        result.get("status") == "success",
        "Clustering tool reports success.",
        "Clustering result status must be 'success'.",
        "RESULT_STATUS_INVALID",
    )
    checks.add(
        "input_path_matches",
        _path_corresponds(
            result.get("input_analysis_path"),
            resolved_arguments.get("analysis_path"),
        ),
        "Clustering result input path matches its resolved input.",
        "Clustering result input path does not match its resolved input.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "reported_identity",
        result.get("backend") == "Scanpy"
        and result.get("algorithm") == "leiden"
        and result.get("cluster_key") == "leiden"
        and result.get("cell_order_preserved") is True,
        "Clustering result reports the fixed Leiden identity and preserved order.",
        "Clustering result reports invalid identity or order metadata.",
        "TOOL_IDENTITY_MISMATCH",
    )
    path = _analysis_artifact_exists(result, checks)
    if path is None:
        return
    try:
        input_value = resolved_arguments.get("analysis_path")
        if not isinstance(input_value, (str, Path)):
            raise ValueError("Invalid input analysis path.")
        input_artifact = _read_analysis(Path(input_value).expanduser().resolve())
        input_ids, _, _ = _validate_neighbors_artifact(
            input_artifact, expected_stages=frozenset({"neighbors"})
        )
        artifact = _read_analysis(path)
        output_ids, _, _ = _validate_neighbors_artifact(
            artifact, expected_stages=frozenset({"clustering"})
        )
        n_clusters = _validate_cluster_labels(artifact)
        provenance = _provenance(artifact)
        parameters = provenance.get("parameters")
        clustering_parameters = (
            parameters.get("clustering")
            if isinstance(parameters, Mapping)
            else None
        )
        structure_valid = (
            artifact.n_obs == result.get("n_cells")
            and n_clusters == result.get("n_clusters")
            and result.get("resolution")
            == float(resolved_arguments.get("resolution", 1.0))
            and result.get("random_seed")
            == resolved_arguments.get("random_seed", 0)
        )
        order_valid = output_ids == input_ids
        provenance_valid = (
            provenance.get("stage") == "clustering"
            and _path_corresponds(
                provenance.get("source_analysis_path"), input_value
            )
            and isinstance(clustering_parameters, Mapping)
            and clustering_parameters.get("algorithm") == "leiden"
            and clustering_parameters.get("resolution")
            == result.get("resolution")
            and clustering_parameters.get("flavor") == "igraph"
            and clustering_parameters.get("n_iterations") == 2
            and clustering_parameters.get("directed") is False
            and clustering_parameters.get("use_weights") is True
            and clustering_parameters.get("random_seed")
            == result.get("random_seed")
            and clustering_parameters.get("key_added") == "leiden"
        )
    except Exception:
        structure_valid = False
        order_valid = False
        provenance_valid = False
    checks.add(
        "clustering_artifact_structure",
        structure_valid,
        "Clustered artifact has valid sparse graphs and Leiden labels.",
        "Clustered artifact is unreadable or scientifically inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "clustering_cell_order",
        order_valid,
        "Clustered artifact exactly preserves upstream cell order.",
        "Clustered artifact does not preserve upstream cell order.",
        "RESULT_IDENTITY_MISMATCH",
    )
    checks.add(
        "clustering_provenance",
        provenance_valid,
        "Clustered artifact has valid stage provenance.",
        "Clustered artifact has invalid stage provenance.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _verify_umap(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "success_status",
        result.get("status") == "success",
        "UMAP tool reports success.",
        "UMAP result status must be 'success'.",
        "RESULT_STATUS_INVALID",
    )
    checks.add(
        "input_path_matches",
        _path_corresponds(
            result.get("input_analysis_path"),
            resolved_arguments.get("analysis_path"),
        ),
        "UMAP result input path matches its resolved input.",
        "UMAP result input path does not match its resolved input.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "reported_invariants",
        result.get("backend") == "Scanpy"
        and result.get("n_components") == 2
        and result.get("umap_key") == UMAP_KEY
        and result.get("finite") is True
        and result.get("cell_order_preserved") is True,
        "UMAP result reports valid fixed scientific invariants.",
        "UMAP result reports invalid scientific invariants.",
        "RESULT_VALUE_INVALID",
    )
    path = _analysis_artifact_exists(result, checks)
    if path is None:
        return
    try:
        input_value = resolved_arguments.get("analysis_path")
        if not isinstance(input_value, (str, Path)):
            raise ValueError("Invalid input analysis path.")
        input_artifact = _read_analysis(Path(input_value).expanduser().resolve())
        input_ids, _, _ = _validate_neighbors_artifact(
            input_artifact, expected_stages=frozenset({"clustering"})
        )
        _validate_cluster_labels(input_artifact)
        artifact = _read_analysis(path)
        output_ids, _, _ = _validate_neighbors_artifact(
            artifact, expected_stages=frozenset({"umap"})
        )
        _validate_cluster_labels(artifact)
        coordinates = np.asarray(artifact.obsm[UMAP_KEY])
        provenance = _provenance(artifact)
        parameters = provenance.get("parameters")
        umap_parameters = (
            parameters.get("umap") if isinstance(parameters, Mapping) else None
        )
        structure_valid = bool(
            artifact.n_obs == result.get("n_cells")
            and coordinates.shape == (artifact.n_obs, 2)
            and np.isfinite(coordinates).all()
            and str(coordinates.dtype) == result.get("coordinate_dtype")
            and result.get("min_dist")
            == float(resolved_arguments.get("min_dist", 0.5))
            and result.get("spread")
            == float(resolved_arguments.get("spread", 1.0))
            and result.get("random_seed")
            == resolved_arguments.get("random_seed", 0)
        )
        order_valid = output_ids == input_ids
        provenance_valid = (
            provenance.get("stage") == "umap"
            and _path_corresponds(
                provenance.get("source_analysis_path"), input_value
            )
            and isinstance(umap_parameters, Mapping)
            and umap_parameters.get("min_dist") == result.get("min_dist")
            and umap_parameters.get("spread") == result.get("spread")
            and umap_parameters.get("n_components") == 2
            and umap_parameters.get("init_pos") == "spectral"
            and umap_parameters.get("random_seed") == result.get("random_seed")
            and umap_parameters.get("key_added") == UMAP_KEY
        )
    except Exception:
        structure_valid = False
        order_valid = False
        provenance_valid = False
    checks.add(
        "umap_artifact_structure",
        structure_valid,
        "UMAP artifact has valid graph, labels, and finite coordinates.",
        "UMAP artifact is unreadable or scientifically inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "umap_cell_order",
        order_valid,
        "UMAP artifact exactly preserves upstream cell order.",
        "UMAP artifact does not preserve upstream cell order.",
        "RESULT_IDENTITY_MISMATCH",
    )
    checks.add(
        "umap_provenance",
        provenance_valid,
        "UMAP artifact has valid stage provenance.",
        "UMAP artifact has invalid stage provenance.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _verify_clustering_evaluation(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    analysis_value = resolved_arguments.get("analysis_path")
    reference_value = resolved_arguments.get("reference_h5ad_path")
    label_key = resolved_arguments.get("label_key")
    cluster_key = resolved_arguments.get("cluster_key", "leiden")
    output_value = resolved_arguments.get("output_dir")
    identity_valid = (
        result.get("status") == "success"
        and _path_corresponds(result.get("analysis_path"), analysis_value)
        and _path_corresponds(result.get("reference_h5ad_path"), reference_value)
        and result.get("label_key") == label_key
        and result.get("cluster_key") == cluster_key
        and result.get("metric_backend") == METRIC_BACKEND
        and result.get("average_method") == AVERAGE_METHOD
        and result.get("report_schema_version")
        == EVALUATION_REPORT_SCHEMA_VERSION
        and result.get("finite") is True
        and result.get("cell_order_preserved") is True
    )
    checks.add(
        "evaluation_result_identity",
        identity_valid,
        "Evaluation result matches resolved paths, keys, and metric identity.",
        "Evaluation result does not match resolved paths, keys, or metric identity.",
        "RESULT_IDENTITY_MISMATCH",
    )

    report_path: Path | None = None
    expected_report_path: Path | None = None
    try:
        if not isinstance(analysis_value, (str, Path)) or not isinstance(
            output_value, (str, Path)
        ):
            raise ValueError("Invalid evaluation paths.")
        expected_report_path = (
            Path(output_value).expanduser().resolve(strict=False)
            / f"{Path(analysis_value).expanduser().resolve(strict=False).stem}.clustering_metrics.json"
        )
        report_value = result.get("report_path")
        if not _path_corresponds(report_value, expected_report_path):
            raise ValueError("Report path mismatch.")
        report_path = Path(str(report_value)).expanduser().resolve()
        report_exists = report_path.is_file() and report_path.stat().st_size > 0
    except (OSError, RuntimeError, TypeError, ValueError):
        report_exists = False
    checks.add(
        "evaluation_report_exists",
        report_exists,
        "Evaluation report exists at the deterministic output path.",
        "Evaluation report is missing or has an inconsistent output path.",
        "ARTIFACT_MISSING",
    )
    if not report_exists or report_path is None:
        return

    report_valid = False
    source_valid = False
    metrics_valid = False
    provenance_valid = False
    try:
        if (
            not isinstance(analysis_value, (str, Path))
            or not isinstance(reference_value, (str, Path))
            or not isinstance(label_key, str)
            or not isinstance(cluster_key, str)
        ):
            raise ValueError("Invalid resolved evaluation arguments.")
        report = _load_evaluation_report(report_path)
        snapshot = _evaluate_sources(
            analysis_value, reference_value, label_key, cluster_key
        )
        expected_report = _report_from_snapshot(snapshot)
        report_valid = report == expected_report
        source_valid = (
            result.get("n_cells") == snapshot.n_cells
            and result.get("n_reference_classes") == snapshot.n_reference_classes
            and result.get("n_predicted_clusters") == snapshot.n_predicted_clusters
        )
        report_metrics = report["metrics"]
        assert isinstance(report_metrics, Mapping)
        metrics_valid = all(
            isinstance(result.get(name), float)
            and result.get(name) == report_metrics.get(name)
            and math.isclose(
                float(result[name]),
                float(getattr(snapshot, name)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for name in ("nmi", "ari", "ami", "homogeneity")
        )
        report_provenance = report["provenance"]
        assert isinstance(report_provenance, Mapping)
        provenance_valid = report_provenance == expected_report["provenance"]
    except Exception:
        report_valid = False
        source_valid = False
        metrics_valid = False
        provenance_valid = False
    checks.add(
        "evaluation_report_schema",
        report_valid,
        "Evaluation report is strict, canonical, and matches recomputed sources.",
        "Evaluation report is corrupt, stale, or inconsistent with recomputed sources.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "evaluation_source_contract",
        source_valid,
        "Evaluation sources preserve exact cells and valid class counts.",
        "Evaluation sources no longer satisfy the exact cell and class contract.",
        "RESULT_IDENTITY_MISMATCH",
    )
    checks.add(
        "evaluation_metrics_recomputed",
        metrics_valid,
        "All four evaluation metrics match independent recomputation.",
        "Reported evaluation metrics do not match independent recomputation.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "evaluation_provenance_recomputed",
        provenance_valid,
        "Evaluation fingerprints match independently reread sources.",
        "Evaluation fingerprints do not match independently reread sources.",
        "RESULT_METADATA_INCONSISTENT",
    )


def _verify_embedding(
    step: PlanStep,
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    dependency_results: Mapping[str, Mapping[str, object]],
    checks: _VerificationChecks,
) -> None:
    checks.add(
        "backend_identity",
        result.get("backend") == "EpiZoo",
        "Embedding result identifies the registered EpiZoo backend.",
        "Embedding result backend identity is not EpiZoo.",
        "TOOL_IDENTITY_MISMATCH",
    )
    if "species" in resolved_arguments:
        resolved_species = resolved_arguments.get("species")
        species_matches = (
            isinstance(resolved_species, str)
            and isinstance(result.get("species"), str)
            and result["species"].casefold() == resolved_species.strip().casefold()
        )
        checks.add(
            "species_matches",
            species_matches,
            "Embedding result species matches the resolved argument.",
            "Embedding result species does not match the resolved argument.",
            "RESULT_IDENTITY_MISMATCH",
        )
    checks.add(
        "success_status",
        result.get("status") == "success",
        "Embedding tool reports success.",
        "Embedding result status must be 'success'.",
        "RESULT_STATUS_INVALID",
    )
    checks.add(
        "input_path_matches",
        _path_corresponds(
            result.get("input_path"), resolved_arguments.get("input_path")
        ),
        "Embedding result path matches the resolved input path.",
        "Embedding result input_path does not match the resolved input_path argument.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "positive_cell_count",
        _positive_integer(result.get("n_cells")),
        "Embedding result reports a positive cell count.",
        "Embedding n_cells must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "positive_embedding_dimension",
        _positive_integer(result.get("embedding_dim")),
        "Embedding dimension is positive.",
        "Embedding dimension must be a positive integer.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "finite_embeddings",
        result.get("finite") is True,
        "Embedding result reports finite values.",
        "Embedding result does not report finite values.",
        "RESULT_VALUE_INVALID",
    )
    checks.add(
        "cell_order_preserved",
        result.get("cell_order_preserved") is True,
        "Embedding result reports preserved cell order.",
        "Embedding result does not report preserved cell order.",
        "RESULT_VALUE_INVALID",
    )
    _artifact_checks(result, checks)

    input_reference = step.arguments.get("input_path")
    if (
        isinstance(input_reference, StepOutputRef)
        and input_reference.output_key == "input_path"
        and input_reference.step_id in step.depends_on
    ):
        inspection_step_id = input_reference.step_id
    elif "inspect" in step.depends_on:
        inspection_step_id = "inspect"
    else:
        return
    inspection = dependency_results.get(inspection_step_id)
    checks.add(
        "inspection_dependency_available",
        isinstance(inspection, Mapping),
        "Verified inspection dependency result is available.",
        "Verified inspection dependency result is unavailable.",
        "DEPENDENCY_RESULT_MISSING",
    )
    if not isinstance(inspection, Mapping):
        return
    checks.add(
        "dependency_input_path_matches",
        _path_corresponds(result.get("input_path"), inspection.get("input_path")),
        "Embedding and inspection input paths match.",
        "Embedding input_path does not match inspection input_path.",
        "RESULT_PATH_MISMATCH",
    )
    checks.add(
        "dependency_cell_count_matches",
        result.get("n_cells") == inspection.get("n_cells"),
        "Embedding and inspection cell counts match.",
        "Embedding n_cells does not match inspection n_cells.",
        "CELL_COUNT_MISMATCH",
    )


def _verify_label_transfer(
    resolved_arguments: Mapping[str, object],
    result: Mapping[str, object],
    checks: _VerificationChecks,
) -> None:
    source_fields = (
        "reference_embedding_path",
        "reference_cell_ids_path",
        "reference_h5ad_path",
        "query_embedding_path",
        "query_cell_ids_path",
        "query_h5ad_path",
    )
    identity_valid = (
        result.get("status") == "success"
        and all(
            _path_corresponds(result.get(name), resolved_arguments.get(name))
            for name in source_fields
        )
        and result.get("reference_label_key")
        == resolved_arguments.get("reference_label_key")
        and result.get("species") == resolved_arguments.get("reference_species")
        and result.get("species") == resolved_arguments.get("query_species")
        and _path_corresponds(
            result.get("checkpoint_path"),
            resolved_arguments.get("reference_checkpoint_path"),
        )
        and _path_corresponds(
            result.get("checkpoint_path"),
            resolved_arguments.get("query_checkpoint_path"),
        )
        and result.get("embedding_dim") == EPIZOO_EMBEDDING_DIM
        and result.get("embedding_dtype") == "float32"
        and result.get("backend") == LABEL_TRANSFER_BACKEND
        and result.get("voting_method") == LABEL_TRANSFER_VOTING_METHOD
        and result.get("artifact_schema_version")
        == LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION
        and result.get("species_compatible") is True
        and result.get("checkpoint_compatible") is True
        and result.get("cell_order_preserved") is True
        and result.get("finite") is True
    )
    checks.add(
        "label_transfer_result_identity",
        identity_valid,
        "Label-transfer result matches resolved sources and scientific identity.",
        "Label-transfer result does not match resolved sources or scientific identity.",
        "RESULT_IDENTITY_MISMATCH",
    )

    annotation_path: Path | None = None
    annotation_exists = False
    try:
        query_value = resolved_arguments.get("query_h5ad_path")
        output_value = resolved_arguments.get("output_dir")
        if not isinstance(query_value, (str, Path)) or not isinstance(
            output_value, (str, Path)
        ):
            raise ValueError("Invalid label-transfer output arguments.")
        expected_output = (
            Path(output_value).expanduser().resolve(strict=False)
            / f"{Path(query_value).expanduser().resolve(strict=False).stem}.label_transfer.h5ad"
        )
        annotation_value = result.get("annotation_path")
        if not _path_corresponds(annotation_value, expected_output):
            raise ValueError("Annotation path mismatch.")
        annotation_path = Path(str(annotation_value)).expanduser().resolve()
        annotation_exists = (
            annotation_path.is_file() and annotation_path.stat().st_size > 0
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        annotation_exists = False
    checks.add(
        "label_transfer_artifact_exists",
        annotation_exists,
        "Label-transfer annotation exists at its deterministic output path.",
        "Label-transfer annotation is missing or has an inconsistent output path.",
        "ARTIFACT_MISSING",
    )
    if not annotation_exists or annotation_path is None:
        return

    artifact_hash_valid = False
    source_valid = False
    artifact_valid = False
    count_valid = False
    try:
        artifact_hash_valid = (
            _label_transfer_file_sha256(annotation_path)
            == result.get("annotation_sha256")
        )
        sources = _prepare_label_transfer_sources(
            resolved_arguments["reference_embedding_path"],
            resolved_arguments["reference_cell_ids_path"],
            resolved_arguments["reference_h5ad_path"],
            resolved_arguments["reference_label_key"],
            resolved_arguments["query_embedding_path"],
            resolved_arguments["query_cell_ids_path"],
            resolved_arguments["query_h5ad_path"],
            reference_species=resolved_arguments["reference_species"],
            query_species=resolved_arguments["query_species"],
            reference_checkpoint_path=resolved_arguments[
                "reference_checkpoint_path"
            ],
            query_checkpoint_path=resolved_arguments["query_checkpoint_path"],
            n_neighbors=resolved_arguments.get("n_neighbors", 20),
            metric=resolved_arguments.get("metric", "euclidean"),
            min_confidence=resolved_arguments.get("min_confidence", 0.0),
            overwrite=resolved_arguments.get("overwrite", False),
        )
        source_valid = (
            result.get("reference_embedding_sha256")
            == sources.reference_embedding_sha256
            and result.get("query_embedding_sha256")
            == sources.query_embedding_sha256
            and result.get("reference_cell_ids_sha256")
            == sources.reference_cell_ids_sha256
            and result.get("query_cell_ids_sha256")
            == sources.query_cell_ids_sha256
            and result.get("reference_labels_sha256")
            == sources.reference_labels_sha256
            and result.get("model_config_sha256") == sources.model_config_sha256
            and result.get("n_neighbors") == sources.n_neighbors
            and result.get("metric") == sources.metric
            and result.get("min_confidence") == sources.min_confidence
            and result.get("n_reference_cells") == len(sources.reference_ids)
            and result.get("n_query_cells") == len(sources.query_ids)
            and result.get("n_reference_classes")
            == len(sources.reference_label_order)
        )
        artifact = _read_label_transfer_annotation(annotation_path)
        try:
            assigned_count, unassigned_count = _validate_annotation_artifact(
                artifact, sources
            )
        finally:
            file_manager = getattr(artifact, "file", None)
            if file_manager is not None:
                file_manager.close()
        artifact_valid = True
        n_query = len(sources.query_ids)
        count_valid = (
            result.get("assigned_count") == assigned_count
            and result.get("unassigned_count") == unassigned_count
            and assigned_count + unassigned_count == n_query
            and isinstance(result.get("assignment_rate"), float)
            and math.isclose(
                float(result["assignment_rate"]),
                assigned_count / n_query,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
    except Exception:
        source_valid = False
        artifact_valid = False
        count_valid = False

    checks.add(
        "label_transfer_artifact_sha256",
        artifact_hash_valid,
        "Label-transfer annotation matches its authoritative result digest.",
        "Label-transfer annotation digest does not match its result.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "label_transfer_sources",
        source_valid,
        "Label-transfer scientific sources and provenance digests are unchanged.",
        "Label-transfer scientific sources or provenance digests are inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "label_transfer_artifact_structure",
        artifact_valid,
        "Label-transfer annotation has valid compact structure and provenance.",
        "Label-transfer annotation is corrupt or structurally inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )
    checks.add(
        "label_transfer_counts",
        count_valid,
        "Label-transfer assignment counts and rate are consistent.",
        "Label-transfer assignment counts or rate are inconsistent.",
        "RESULT_METADATA_INCONSISTENT",
    )


def verify_step(
    step: PlanStep,
    resolved_arguments: Mapping[str, object],
    result: object,
    registry: ToolRegistry,
    *,
    dependency_results: Mapping[str, Mapping[str, object]] | None = None,
) -> VerificationResult:
    """Verify one lightweight result without invoking scientific tools."""

    if not isinstance(step, PlanStep):
        raise TypeError("`step` must be a PlanStep.")
    if not isinstance(resolved_arguments, Mapping):
        raise TypeError("`resolved_arguments` must be a mapping.")
    if not isinstance(registry, ToolRegistry):
        raise TypeError("`registry` must be a ToolRegistry.")
    if dependency_results is None:
        dependency_results = {}
    elif not isinstance(dependency_results, Mapping):
        raise TypeError("`dependency_results` must be a mapping.")

    checks = _VerificationChecks()
    plain_result = _verify_common_step(step, result, registry, checks)
    if plain_result is not None:
        if step.tool_name == "inspect_scATAC":
            _verify_inspection(resolved_arguments, plain_result, checks)
        elif step.tool_name == "epizoo_embed_cells":
            _verify_embedding(
                step,
                resolved_arguments,
                plain_result,
                dependency_results,
                checks,
            )
        elif step.tool_name == "build_cell_neighbors":
            _verify_neighbors(resolved_arguments, plain_result, checks)
        elif step.tool_name == "cluster_cells":
            _verify_clustering(resolved_arguments, plain_result, checks)
        elif step.tool_name == "compute_cell_umap":
            _verify_umap(resolved_arguments, plain_result, checks)
        elif step.tool_name == "evaluate_cell_clustering":
            _verify_clustering_evaluation(
                resolved_arguments, plain_result, checks
            )
        elif step.tool_name == "transfer_cell_labels":
            _verify_label_transfer(resolved_arguments, plain_result, checks)
    return checks.result(
        target_type="step",
        target_id=step.step_id,
        step_id=step.step_id,
        tool_name=step.tool_name,
    )


def _status_is_consistent(result: StepExecutionResult) -> bool:
    if result.status is StepStatus.SUCCEEDED:
        return result.result is not None and result.error is None
    if result.status is StepStatus.FAILED:
        if result.error is None:
            return False
        if result.verification is not None:
            return not result.verification.passed
        return result.result is None
    if result.status is StepStatus.SKIPPED:
        return result.result is None and (
            result.verification is None or not result.verification.passed
        )
    return False


def verify_run(
    plan: AgentPlan, step_results: Sequence[StepExecutionResult]
) -> VerificationResult:
    """Verify completed-run consistency without inspecting scientific files."""

    if not isinstance(plan, AgentPlan):
        raise TypeError("`plan` must be an AgentPlan.")
    if not isinstance(step_results, Sequence):
        raise TypeError("`step_results` must be a sequence.")

    checks = _VerificationChecks()
    if not all(isinstance(result, StepExecutionResult) for result in step_results):
        checks.add(
            "step_result_types",
            False,
            "All step results have the expected type.",
            "Every run result must be a StepExecutionResult.",
            "RUN_RESULT_INVALID",
        )
        return checks.result(target_type="run", target_id=plan.plan_id)
    checks.add(
        "step_result_types",
        True,
        "All step results have the expected type.",
        "Every run result must be a StepExecutionResult.",
        "RUN_RESULT_INVALID",
    )

    plan_by_id = {step.step_id: step for step in plan.steps}
    grouped: dict[str, list[StepExecutionResult]] = {}
    for result in step_results:
        grouped.setdefault(result.step_id, []).append(result)

    unexpected = tuple(step_id for step_id in grouped if step_id not in plan_by_id)
    checks.add(
        "no_unexpected_step_results",
        not unexpected,
        "No unexpected step results are present.",
        f"Unexpected step results are present: {unexpected}.",
        "UNEXPECTED_STEP_RESULT",
    )
    duplicates = tuple(step_id for step_id, values in grouped.items() if len(values) > 1)
    checks.add(
        "one_result_per_step",
        not duplicates,
        "No duplicate step results are present.",
        f"Duplicate step results are present: {duplicates}.",
        "DUPLICATE_STEP_RESULT",
    )
    missing = tuple(step_id for step_id in plan_by_id if step_id not in grouped)
    checks.add(
        "all_plan_steps_reported",
        not missing,
        "Every plan step has a result.",
        f"Plan steps are missing results: {missing}.",
        "MISSING_STEP_RESULT",
    )

    unique_results = {
        step_id: values[0]
        for step_id, values in grouped.items()
        if step_id in plan_by_id and len(values) == 1
    }
    for step in plan.steps:
        result = unique_results.get(step.step_id)
        if result is None:
            continue
        checks.add(
            f"{step.step_id}_tool_identity",
            result.tool_name == step.tool_name,
            f"Step {step.step_id!r} tool identity matches its plan.",
            f"Step {step.step_id!r} tool name does not match its plan.",
            "STEP_IDENTITY_MISMATCH",
        )
        checks.add(
            f"{step.step_id}_status_consistent",
            _status_is_consistent(result),
            f"Step {step.step_id!r} status and payload are internally consistent.",
            f"Step {step.step_id!r} status and payload are inconsistent.",
            "STEP_STATUS_INCONSISTENT",
        )
        checks.add(
            f"{step.step_id}_succeeded",
            result.status is StepStatus.SUCCEEDED,
            f"Step {step.step_id!r} succeeded.",
            f"Step {step.step_id!r} did not succeed.",
            "STEP_NOT_SUCCEEDED",
        )
        verification_passed = (
            result.verification is not None
            and result.verification.passed
            and result.verification.target_type == "step"
            and result.verification.target_id == step.step_id
        )
        checks.add(
            f"{step.step_id}_verification_passed",
            verification_passed,
            f"Step {step.step_id!r} has passed verification.",
            f"Step {step.step_id!r} lacks matching passed verification.",
            "STEP_VERIFICATION_FAILED",
        )

        dependency_results = [unique_results.get(name) for name in step.depends_on]
        if result.status is StepStatus.SUCCEEDED:
            dependency_consistent = all(
                dependency is not None and dependency.status is StepStatus.SUCCEEDED
                for dependency in dependency_results
            )
        elif result.status is StepStatus.SKIPPED and step.depends_on:
            dependency_consistent = any(
                dependency is not None and dependency.status is not StepStatus.SUCCEEDED
                for dependency in dependency_results
            )
        else:
            dependency_consistent = True
        checks.add(
            f"{step.step_id}_dependencies_consistent",
            dependency_consistent,
            f"Step {step.step_id!r} dependency completion is consistent.",
            f"Step {step.step_id!r} dependency completion is inconsistent.",
            "DEPENDENCY_INCONSISTENT",
        )

    return checks.result(target_type="run", target_id=plan.plan_id)


__all__ = ["verify_run", "verify_step"]
