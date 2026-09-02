"""Explicit allowlist and contracts for executable scientific tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import inspect
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from agent.schemas import AgentError, ErrorCategory, StepOutputRef
from agent.tools import (
    build_replicate_pseudobulk,
    build_cell_neighbors,
    cluster_cells,
    compute_cell_umap,
    epizoo_embed_cells,
    evaluate_cell_annotation,
    evaluate_cell_clustering,
    inspect_scATAC,
    run_replicate_differential_accessibility,
    transfer_cell_labels,
    validate_scATAC_feature_space,
)
from agent.tools.analysis.replicate_pseudobulk import M81ScientificError
from agent.tools.analysis.differential_accessibility import M82ScientificError
from agent.tools.analysis.differential_accessibility_backend import (
    DA_ARTIFACT_SCHEMA_VERSION,
    DA_ARTIFACT_TYPE,
    EDGER_PIPELINE_ID,
)

from .error_policy import classified_agent_error


ToolCallable = Callable[..., object]
ResultValidator = Callable[[Mapping[str, object]], None]


class UnknownToolError(LookupError):
    """Raised when a plan names a tool outside the explicit allowlist."""


class ToolArgumentError(ValueError):
    """Raised when tool arguments do not satisfy a registered contract."""


class ToolResultContractError(ValueError):
    """Raised when a tool result does not satisfy its registered contract."""


@dataclass(frozen=True)
class ErrorClassification:
    """Registry-owned classification before execution context is attached."""

    category: ErrorCategory
    code: str


ExceptionClassifier = Callable[[Exception], ErrorClassification]


@dataclass(frozen=True)
class ArgumentSpec:
    """Runtime contract for one public tool argument."""

    accepted_types: tuple[type, ...]
    choices: tuple[object, ...] = ()
    allow_step_output_ref: bool = True

    def __post_init__(self) -> None:
        if not self.accepted_types or not all(
            isinstance(value, type) for value in self.accepted_types
        ):
            raise TypeError("`accepted_types` must be a non-empty tuple of types.")
        if not isinstance(self.choices, tuple):
            raise TypeError("`choices` must be a tuple.")
        if not isinstance(self.allow_step_output_ref, bool):
            raise TypeError("`allow_step_output_ref` must be a boolean.")

    def validate(self, name: str, value: object) -> None:
        if isinstance(value, StepOutputRef):
            if self.allow_step_output_ref:
                return
            raise ToolArgumentError(
                f"Argument {name!r} does not accept an unresolved StepOutputRef."
            )
        if isinstance(value, bool) and bool not in self.accepted_types:
            valid_type = False
        else:
            valid_type = isinstance(value, self.accepted_types)
        if not valid_type:
            expected = ", ".join(value_type.__name__ for value_type in self.accepted_types)
            raise ToolArgumentError(
                f"Argument {name!r} must have type {expected}; "
                f"received {type(value).__name__}."
            )
        if self.choices and value not in self.choices:
            raise ToolArgumentError(
                f"Argument {name!r} must be one of {self.choices!r}; "
                f"received {value!r}."
            )


@dataclass(frozen=True)
class ResultContract:
    """Authoritative lightweight result shape for a registered tool."""

    name: str
    required_fields: Mapping[str, tuple[type, ...]]
    validator: ResultValidator | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Result contract `name` must be a non-empty string.")
        copied: dict[str, tuple[type, ...]] = {}
        for key, expected_types in self.required_fields.items():
            if not isinstance(key, str) or not key:
                raise TypeError("Result field names must be non-empty strings.")
            if not isinstance(expected_types, tuple) or not expected_types or not all(
                isinstance(value, type) for value in expected_types
            ):
                raise TypeError(
                    f"Result field {key!r} must define a non-empty tuple of types."
                )
            copied[key] = expected_types
        object.__setattr__(self, "required_fields", MappingProxyType(copied))

    def validate(self, result: object) -> None:
        if not isinstance(result, Mapping):
            raise ToolResultContractError(
                f"{self.name} must be a lightweight mapping; "
                f"received {type(result).__name__}."
            )
        missing = [key for key in self.required_fields if key not in result]
        if missing:
            raise ToolResultContractError(
                f"{self.name} is missing required fields: {missing}."
            )
        for key, expected_types in self.required_fields.items():
            value = result[key]
            if isinstance(value, bool) and bool not in expected_types:
                valid_type = False
            else:
                valid_type = isinstance(value, expected_types)
            if not valid_type:
                expected = ", ".join(
                    value_type.__name__ for value_type in expected_types
                )
                raise ToolResultContractError(
                    f"{self.name} field {key!r} must have type {expected}; "
                    f"received {type(value).__name__}."
                )
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ToolResultContractError(
                f"{self.name} must be JSON serializable: {exc}"
            ) from exc
        if self.validator is not None:
            self.validator(result)


@dataclass(frozen=True)
class ToolSpec:
    """Immutable executable-tool specification owned by the registry."""

    name: str
    function: ToolCallable
    required_arguments: Mapping[str, ArgumentSpec]
    optional_arguments: Mapping[str, ArgumentSpec]
    result_contract: ResultContract
    exception_classifier: ExceptionClassifier
    retryable_error_codes: frozenset[str] = field(default_factory=frozenset)
    recovery_policy_version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ToolSpec `name` must be a non-empty string.")
        if not callable(self.function):
            raise TypeError("ToolSpec `function` must be callable.")
        required = dict(self.required_arguments)
        optional = dict(self.optional_arguments)
        overlap = set(required).intersection(optional)
        if overlap:
            raise ValueError(f"Tool arguments cannot be both required and optional: {overlap}.")
        for argument_name, argument_spec in (*required.items(), *optional.items()):
            if not isinstance(argument_name, str) or not argument_name:
                raise TypeError("Tool argument names must be non-empty strings.")
            if not isinstance(argument_spec, ArgumentSpec):
                raise TypeError("Tool argument contracts must be ArgumentSpec values.")
        if not isinstance(self.result_contract, ResultContract):
            raise TypeError("`result_contract` must be a ResultContract.")
        if not callable(self.exception_classifier):
            raise TypeError("`exception_classifier` must be callable.")
        if (
            not isinstance(self.recovery_policy_version, str)
            or not self.recovery_policy_version.strip()
        ):
            raise ValueError(
                "`recovery_policy_version` must be a non-empty string."
            )
        object.__setattr__(self, "required_arguments", MappingProxyType(required))
        object.__setattr__(self, "optional_arguments", MappingProxyType(optional))
        object.__setattr__(
            self, "retryable_error_codes", frozenset(self.retryable_error_codes)
        )


class ToolRegistry:
    """Process-local immutable allowlist of executable scientific tools."""

    __slots__ = ("_specs", "_names")

    def __init__(self, specs: tuple[ToolSpec, ...]) -> None:
        if not isinstance(specs, tuple):
            raise TypeError("`specs` must be a tuple of ToolSpec values.")
        if not all(isinstance(spec, ToolSpec) for spec in specs):
            raise TypeError("Every registry entry must be a ToolSpec.")
        names = tuple(spec.name for spec in specs)
        if len(set(names)) != len(names):
            raise ValueError("Tool registry names must be unique.")
        object.__setattr__(
            self, "_specs", MappingProxyType({spec.name: spec for spec in specs})
        )
        object.__setattr__(self, "_names", names)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ToolRegistry is immutable after construction.")

    def contains(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> tuple[str, ...]:
        return self._names

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise UnknownToolError(
                f"Tool {name!r} is not in the executable allowlist."
            ) from exc

    def validate_arguments(
        self, name: str, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        spec = self.get(name)
        if not isinstance(arguments, Mapping):
            raise ToolArgumentError("Tool arguments must be a mapping.")
        supplied = set(arguments)
        required = set(spec.required_arguments)
        known = required.union(spec.optional_arguments)
        missing = sorted(required.difference(supplied))
        unknown = sorted(supplied.difference(known))
        if missing:
            raise ToolArgumentError(
                f"Tool {name!r} is missing required arguments: {missing}."
            )
        if unknown:
            raise ToolArgumentError(
                f"Tool {name!r} received unknown arguments: {unknown}."
            )
        for argument_name, value in arguments.items():
            argument_spec = spec.required_arguments.get(
                argument_name, spec.optional_arguments.get(argument_name)
            )
            if argument_spec is None:  # pragma: no cover - checked above
                raise ToolArgumentError(f"Unknown argument {argument_name!r}.")
            argument_spec.validate(argument_name, value)
        return MappingProxyType(dict(arguments))

    def validate_result(self, name: str, result: object) -> None:
        self.get(name).result_contract.validate(result)

    def classify_exception(
        self,
        name: str,
        exception: Exception,
        *,
        step_id: str | None = None,
        attempt: int | None = None,
    ) -> AgentError:
        spec = self.get(name)
        classification = spec.exception_classifier(exception)
        retryable = classification.code in spec.retryable_error_codes
        return classified_agent_error(
            category=classification.category,
            code=classification.code,
            step_id=step_id,
            tool_name=name,
            exception_type=type(exception).__name__,
            same_step_retry_eligible=retryable,
            attempt=attempt,
        )


def _classify_tool_exception(exception: Exception) -> ErrorClassification:
    if isinstance(exception, FileNotFoundError):
        return ErrorClassification(ErrorCategory.RESOURCE_ERROR, "RESOURCE_NOT_FOUND")
    if isinstance(exception, FileExistsError):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, "OUTPUT_CONFLICT")
    if isinstance(exception, (ModuleNotFoundError, ImportError)):
        return ErrorClassification(
            ErrorCategory.ENVIRONMENT_ERROR, "DEPENDENCY_UNAVAILABLE"
        )
    if isinstance(exception, MemoryError):
        return ErrorClassification(
            ErrorCategory.RESOURCE_ERROR, "HOST_MEMORY_EXHAUSTED"
        )
    if _is_cuda_out_of_memory(exception):
        return ErrorClassification(ErrorCategory.RESOURCE_ERROR, "CUDA_OUT_OF_MEMORY")
    if isinstance(exception, OSError) and exception.errno == errno.ENOSPC:
        return ErrorClassification(ErrorCategory.RESOURCE_ERROR, "DISK_FULL")
    if isinstance(exception, RuntimeError):
        message = str(exception)
        if message.startswith("CUDA device ") and message.endswith(
            " was requested but CUDA is not available in this runtime."
        ):
            return ErrorClassification(
                ErrorCategory.ENVIRONMENT_ERROR, "CUDA_UNAVAILABLE"
            )
        return ErrorClassification(
            ErrorCategory.TOOL_EXECUTION_ERROR, "TOOL_RUNTIME_ERROR"
        )
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "TOOL_EXCEPTION")


def _classify_embedding_exception(exception: Exception) -> ErrorClassification:
    classification = _classify_tool_exception(exception)
    if (
        classification.code == "TOOL_EXCEPTION"
        and isinstance(exception, OSError)
    ):
        return ErrorClassification(
            ErrorCategory.RESOURCE_ERROR, "ARTIFACT_WRITE_FAILED"
        )
    return classification


def _classify_analysis_exception(exception: Exception) -> ErrorClassification:
    if isinstance(exception, (TypeError, ValueError)):
        return ErrorClassification(ErrorCategory.USER_INPUT_ERROR, "INVALID_ARGUMENT")
    return _classify_embedding_exception(exception)


def _classify_m81_exception(exception: Exception) -> ErrorClassification:
    if isinstance(exception, M81ScientificError):
        category = (
            ErrorCategory.RESOURCE_ERROR
            if exception.code == "INTEGER_SUM_OVERFLOW"
            else ErrorCategory.VERIFICATION_ERROR
            if exception.code == "SOURCE_CHANGED_DURING_READ"
            else ErrorCategory.USER_INPUT_ERROR
        )
        return ErrorClassification(category, exception.code)
    return _classify_analysis_exception(exception)


def _classify_m82_exception(exception: Exception) -> ErrorClassification:
    if isinstance(exception, M82ScientificError):
        if exception.code in {"HOST_MEMORY_EXHAUSTED", "DISK_FULL"}:
            category = ErrorCategory.RESOURCE_ERROR
        elif exception.code in {
            "RSCRIPT_UNAVAILABLE",
            "EDGER_PACKAGE_UNAVAILABLE",
            "EDGER_VERSION_UNSUPPORTED",
            "R_PACKAGE_VERSION_INCOMPATIBLE",
        }:
            category = ErrorCategory.ENVIRONMENT_ERROR
        elif exception.code == "SOURCE_CHANGED_DURING_READ":
            category = ErrorCategory.VERIFICATION_ERROR
        elif exception.code in {
            "DA_NO_FEATURES_AFTER_FILTER",
            "DA_FILTERED_LIBRARY_ZERO",
            "R_BACKEND_EXECUTION_FAILED",
            "R_BACKEND_PROTOCOL_INVALID",
            "DA_NUMERICAL_RESULT_INVALID",
            "ARTIFACT_WRITE_FAILED",
        }:
            category = ErrorCategory.TOOL_EXECUTION_ERROR
        else:
            category = ErrorCategory.USER_INPUT_ERROR
        return ErrorClassification(category, exception.code)
    return _classify_analysis_exception(exception)


def _is_cuda_out_of_memory(exception: Exception) -> bool:
    exception_type = type(exception)
    return (
        exception_type.__name__ == "OutOfMemoryError"
        and exception_type.__module__.startswith("torch")
    )


def _validate_inspection_result(result: Mapping[str, object]) -> None:
    if result["nnz"] is not None and not isinstance(result["nnz"], int):
        raise ToolResultContractError("inspect_scATAC field 'nnz' must be int or None.")
    if result["density"] is not None and not isinstance(result["density"], float):
        raise ToolResultContractError(
            "inspect_scATAC field 'density' must be float or None."
        )


def _validate_embedding_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "epizoo_embed_cells field 'status' must equal 'success'."
        )
    if "embeddings" in result:
        raise ToolResultContractError(
            "epizoo_embed_cells must not return an embedding array in its result."
        )


def _validate_neighbors_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "build_cell_neighbors field 'status' must equal 'success'."
        )
    if result["finite"] is not True or result["cell_order_preserved"] is not True:
        raise ToolResultContractError(
            "build_cell_neighbors must report finite output and preserved cell order."
        )
    forbidden = {"embeddings", "cell_ids", "distances", "connectivities"}
    if forbidden.intersection(result):
        raise ToolResultContractError(
            "build_cell_neighbors must not return large scientific arrays."
        )


def _validate_clustering_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "cluster_cells field 'status' must equal 'success'."
        )
    if result["cell_order_preserved"] is not True:
        raise ToolResultContractError(
            "cluster_cells must report preserved cell order."
        )
    if "labels" in result:
        raise ToolResultContractError(
            "cluster_cells must not return the full Leiden label vector."
        )


def _validate_umap_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "compute_cell_umap field 'status' must equal 'success'."
        )
    if result["finite"] is not True or result["cell_order_preserved"] is not True:
        raise ToolResultContractError(
            "compute_cell_umap must report finite output and preserved cell order."
        )
    if "coordinates" in result:
        raise ToolResultContractError(
            "compute_cell_umap must not return the full coordinate array."
        )


def _validate_evaluation_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "evaluate_cell_clustering field 'status' must equal 'success'."
        )
    if result["finite"] is not True or result["cell_order_preserved"] is not True:
        raise ToolResultContractError(
            "evaluate_cell_clustering must report finite metrics and preserved order."
        )
    if (
        result["metric_backend"] != "scikit-learn"
        or result["average_method"] != "arithmetic"
        or result["report_schema_version"] != 1
    ):
        raise ToolResultContractError(
            "evaluate_cell_clustering has invalid metric or schema identity."
        )
    if (
        result["n_cells"] <= 0
        or result["n_reference_classes"] < 2
        or result["n_predicted_clusters"] < 1
    ):
        raise ToolResultContractError(
            "evaluate_cell_clustering has invalid scientific counts."
        )
    for name, lower in (
        ("nmi", 0.0),
        ("ari", -1.0),
        ("ami", -1.0),
        ("homogeneity", 0.0),
    ):
        value = result[name]
        if not math.isfinite(value) or not lower - 1e-12 <= value <= 1.0 + 1e-12:
            raise ToolResultContractError(
                f"evaluate_cell_clustering field {name!r} is outside its valid range."
            )
    forbidden = {"reference_labels", "predicted_labels", "cell_ids", "embeddings"}
    if forbidden.intersection(result):
        raise ToolResultContractError(
            "evaluate_cell_clustering must not return large scientific arrays."
        )


def _validate_label_transfer_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "transfer_cell_labels field 'status' must equal 'success'."
        )
    if (
        result["finite"] is not True
        or result["cell_order_preserved"] is not True
        or result["species_compatible"] is not True
        or result["checkpoint_compatible"] is not True
    ):
        raise ToolResultContractError(
            "transfer_cell_labels must report valid compatibility and output invariants."
        )
    if (
        result["embedding_dim"] != 512
        or result["embedding_dtype"] != "float32"
        or result["voting_method"] != "uniform_plurality"
        or result["artifact_schema_version"] != 1
    ):
        raise ToolResultContractError(
            "transfer_cell_labels has invalid scientific or schema identity."
        )
    n_query = result["n_query_cells"]
    assigned = result["assigned_count"]
    unassigned = result["unassigned_count"]
    if (
        result["n_reference_cells"] <= 0
        or n_query <= 0
        or result["n_reference_classes"] < 2
        or assigned < 0
        or unassigned < 0
        or assigned + unassigned != n_query
        or not math.isclose(
            result["assignment_rate"], assigned / n_query, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise ToolResultContractError(
            "transfer_cell_labels has inconsistent scientific counts."
        )
    for name in (
        "annotation_sha256",
        "reference_embedding_sha256",
        "query_embedding_sha256",
        "reference_cell_ids_sha256",
        "query_cell_ids_sha256",
        "reference_labels_sha256",
        "model_config_sha256",
    ):
        digest = result[name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ToolResultContractError(
                f"transfer_cell_labels field {name!r} is not a SHA-256 digest."
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ToolResultContractError(
                f"transfer_cell_labels field {name!r} is not a SHA-256 digest."
            ) from exc
    forbidden = {
        "predictions",
        "confidences",
        "cell_ids",
        "reference_labels",
        "neighbor_indices",
        "distances",
        "embeddings",
    }
    if forbidden.intersection(result):
        raise ToolResultContractError(
            "transfer_cell_labels must not return per-cell or large scientific arrays."
        )


def _validate_annotation_evaluation_result(result: Mapping[str, object]) -> None:
    if result["status"] != "success":
        raise ToolResultContractError(
            "evaluate_cell_annotation field 'status' must equal 'success'."
        )
    if result["finite"] is not True or result["cell_order_preserved"] is not True:
        raise ToolResultContractError(
            "evaluate_cell_annotation must report finite output and preserved order."
        )
    if (
        result["metric_backend"] != "scikit-learn"
        or result["macro_average"] != "macro"
        or result["zero_division"] != 0
        or result["report_schema_version"] != 1
    ):
        raise ToolResultContractError(
            "evaluate_cell_annotation has invalid metric or schema identity."
        )
    n_cells = result["n_cells"]
    assigned = result["assigned_count"]
    unassigned = result["unassigned_count"]
    correct = result["correct_assigned_count"]
    incorrect = result["incorrect_assigned_count"]
    if (
        n_cells <= 0
        or result["n_ground_truth_classes"] < 1
        or result["n_assigned_predicted_classes"] < 0
        or min(assigned, unassigned, correct, incorrect) < 0
        or assigned + unassigned != n_cells
        or correct + incorrect != assigned
    ):
        raise ToolResultContractError(
            "evaluate_cell_annotation has inconsistent scientific counts."
        )
    required_scores = (
        "assignment_rate",
        "overall_accuracy",
        "macro_f1",
        "median_confidence",
    )
    nullable_scores = (
        "assigned_accuracy",
        "median_assigned_confidence",
        "median_correct_assigned_confidence",
        "median_incorrect_assigned_confidence",
    )
    for name in required_scores:
        value = result[name]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ToolResultContractError(
                f"evaluate_cell_annotation field {name!r} is outside [0, 1]."
            )
    for name in nullable_scores:
        value = result[name]
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ToolResultContractError(
                f"evaluate_cell_annotation field {name!r} is outside [0, 1]."
            )
    expected_nullable = {
        "assigned_accuracy": assigned == 0,
        "median_assigned_confidence": assigned == 0,
        "median_correct_assigned_confidence": correct == 0,
        "median_incorrect_assigned_confidence": incorrect == 0,
    }
    if any((result[name] is None) != expected for name, expected in expected_nullable.items()):
        raise ToolResultContractError(
            "evaluate_cell_annotation has invalid nullable-field semantics."
        )
    if not math.isclose(
        result["assignment_rate"], assigned / n_cells, rel_tol=0.0, abs_tol=1e-15
    ) or not math.isclose(
        result["overall_accuracy"], correct / n_cells, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ToolResultContractError(
            "evaluate_cell_annotation rates are inconsistent with its counts."
        )
    expected_assigned_accuracy = correct / assigned if assigned else None
    if expected_assigned_accuracy is not None and not math.isclose(
        result["assigned_accuracy"],
        expected_assigned_accuracy,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ToolResultContractError(
            "evaluate_cell_annotation assigned accuracy is inconsistent."
        )
    digest = result["annotation_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ToolResultContractError(
            "evaluate_cell_annotation annotation digest is invalid."
        )
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ToolResultContractError(
            "evaluate_cell_annotation annotation digest is invalid."
        ) from exc
    forbidden = {
        "ground_truth_labels",
        "predicted_labels",
        "prediction_status",
        "prediction_confidence",
        "cell_ids",
        "confusion",
        "per_class",
    }
    if forbidden.intersection(result):
        raise ToolResultContractError(
            "evaluate_cell_annotation must not return per-cell diagnostics."
        )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_feature_space_result(result: Mapping[str, object]) -> None:
    if (
        result["status"] != "success"
        or result["artifact_schema_version"] != 1
        or result["pseudobulk_eligible"] is not True
        or result["matrix_semantics"]
        not in {"fragment_counts", "insertion_counts", "binary_accessibility"}
        or result["matrix_source"] not in {"X", "layer"}
        or result["coordinate_source"] not in {"none", "var_columns"}
        or result["semantics_assertion_source"]
        not in {"structured_request", "structured_request_and_raw_uns"}
    ):
        raise ToolResultContractError("Feature-space result has invalid scientific identity.")
    if min(result["n_cells"], result["n_features"]) <= 0 or result["nnz"] < 0:
        raise ToolResultContractError("Feature-space result has invalid dimensions.")
    for name in (
        "feature_space_sha256",
        "feature_space_identity_sha256",
        "source_h5ad_sha256",
        "cell_ids_sha256",
        "feature_ids_sha256",
        "matrix_sha256",
    ):
        if not _valid_sha256(result[name]):
            raise ToolResultContractError(f"Feature-space field {name!r} is not SHA-256.")
    coordinate_digest = result["coordinates_sha256"]
    if (result["coordinate_source"] == "none") != (coordinate_digest is None):
        raise ToolResultContractError("Feature-space coordinate digest is inconsistent.")
    if coordinate_digest is not None and not _valid_sha256(coordinate_digest):
        raise ToolResultContractError("Feature-space coordinate digest is invalid.")
    forbidden = {"matrix", "cell_ids", "feature_ids", "coordinates"}
    if forbidden.intersection(result):
        raise ToolResultContractError("Feature-space result contains a large payload.")


def _validate_pseudobulk_result(result: Mapping[str, object]) -> None:
    if (
        result["status"] != "success"
        or result["artifact_schema_version"] != 1
        or result["aggregation_method"] != "sum"
        or result["output_dtype"] != "int64"
        or result["group_source"] not in {"raw_obs", "verified_annotation"}
        or result["all_cells_accounted_for"] is not True
        or result["feature_order_preserved"] is not True
    ):
        raise ToolResultContractError("Pseudobulk result has invalid scientific identity.")
    positive = (
        "n_cells", "n_features", "n_pseudobulks", "n_groups", "n_replicates",
        "n_conditions", "minimum_cells_per_pseudobulk", "maximum_cells_per_pseudobulk",
    )
    if any(result[name] <= 0 for name in positive) or result["matrix_nnz"] < 0 or result["total_sum"] < 0:
        raise ToolResultContractError("Pseudobulk result has invalid counts.")
    if result["minimum_cells_per_pseudobulk"] > result["maximum_cells_per_pseudobulk"]:
        raise ToolResultContractError("Pseudobulk cell-count range is invalid.")
    if not isinstance(result["covariate_keys"], list) or not all(
        isinstance(value, str) and value for value in result["covariate_keys"]
    ):
        raise ToolResultContractError("Pseudobulk covariate keys are invalid.")
    for name in (
        "pseudobulk_sha256", "feature_space_sha256",
        "feature_space_identity_sha256", "source_h5ad_sha256",
    ):
        if not _valid_sha256(result[name]):
            raise ToolResultContractError(f"Pseudobulk field {name!r} is not SHA-256.")
    forbidden = {"matrix", "cell_ids", "feature_ids", "unit_assignments", "metadata_vectors"}
    if forbidden.intersection(result):
        raise ToolResultContractError("Pseudobulk result contains a large payload.")


def _validate_differential_accessibility_result(
    result: Mapping[str, object],
) -> None:
    if (
        result["status"] != "success"
        or result["artifact_type"] != DA_ARTIFACT_TYPE
        or result["artifact_schema_version"] != DA_ARTIFACT_SCHEMA_VERSION
        or result["backend_pipeline"] != EDGER_PIPELINE_ID
        or result["filtering_method"] != "edgeR::filterByExpr"
        or result["normalization_method"] != "TMM"
        or result["design_type"] not in {"independent", "paired"}
    ):
        raise ToolResultContractError(
            "Differential-accessibility result has invalid scientific identity."
        )
    positive = (
        "n_samples",
        "n_numerator_replicates",
        "n_denominator_replicates",
        "design_rank",
        "residual_degrees_of_freedom",
        "n_input_features",
        "n_tested_features",
    )
    if any(result[name] <= 0 for name in positive):
        raise ToolResultContractError(
            "Differential-accessibility result has invalid positive counts."
        )
    if result["n_filtered_features"] < 0 or (
        result["n_tested_features"] + result["n_filtered_features"]
        != result["n_input_features"]
    ):
        raise ToolResultContractError(
            "Differential-accessibility feature counts are inconsistent."
        )
    if (
        not isinstance(result["warning_codes"], list)
        or not all(
            isinstance(value, str) and value for value in result["warning_codes"]
        )
        or result["n_warnings"] != len(result["warning_codes"])
    ):
        raise ToolResultContractError(
            "Differential-accessibility warning metadata is invalid."
        )
    for name in (
        "da_sha256",
        "pseudobulk_sha256",
        "preparation_sha256",
        "analysis_sha256",
        "production_r_script_sha256",
    ):
        if not _valid_sha256(result[name]):
            raise ToolResultContractError(
                f"Differential-accessibility field {name!r} is not SHA-256."
            )
    if not isinstance(result["package_versions"], dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(value, str)
        and value
        for key, value in result["package_versions"].items()
    ):
        raise ToolResultContractError(
            "Differential-accessibility package versions are invalid."
        )
    forbidden = {
        "matrix",
        "feature_ids",
        "row_eligibility",
        "design_matrix",
        "contrast",
        "statistics",
    }
    if forbidden.intersection(result):
        raise ToolResultContractError(
            "Differential-accessibility result contains a large payload."
        )


def _assert_signature_matches(spec: ToolSpec) -> None:
    signature = inspect.signature(spec.function)
    parameters = signature.parameters
    registered = set(spec.required_arguments).union(spec.optional_arguments)
    if set(parameters) != registered:
        raise RuntimeError(
            f"Registry contract for {spec.name!r} does not match its public signature."
        )
    signature_required = {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    if signature_required != set(spec.required_arguments):
        raise RuntimeError(
            f"Registry required arguments for {spec.name!r} do not match its "
            "public signature."
        )


def build_default_tool_registry() -> ToolRegistry:
    """Build the fixed Milestone 3 allowlist from public scientific APIs."""

    path_argument = ArgumentSpec((str, Path))
    inspect_spec = ToolSpec(
        name="inspect_scATAC",
        function=inspect_scATAC,
        required_arguments={"path": path_argument},
        optional_arguments={},
        result_contract=ResultContract(
            name="ScATACInspection",
            required_fields={
                "input_path": (str,),
                "n_cells": (int,),
                "n_features": (int,),
                "x_storage_type": (str,),
                "x_is_sparse": (bool,),
                "x_dtype": (str,),
                "nnz": (int, type(None)),
                "density": (float, type(None)),
                "obs_columns": (list,),
                "var_columns": (list,),
                "obs_names_sample": (list,),
                "var_names_sample": (list,),
            },
            validator=_validate_inspection_result,
        ),
        exception_classifier=_classify_tool_exception,
        recovery_policy_version="inspect-scatac-v2",
    )
    embedding_spec = ToolSpec(
        name="epizoo_embed_cells",
        function=epizoo_embed_cells,
        required_arguments={
            "input_path": path_argument,
            "output_dir": path_argument,
            "species": ArgumentSpec((str,), choices=("human", "mouse")),
        },
        optional_arguments={
            "checkpoint_path": path_argument,
            "device": ArgumentSpec((str,)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="EpiZooEmbeddingToolResult",
            required_fields={
                "status": (str,),
                "input_path": (str,),
                "embedding_path": (str,),
                "cell_ids_path": (str,),
                "n_cells": (int,),
                "embedding_dim": (int,),
                "embedding_dtype": (str,),
                "finite": (bool,),
                "cell_order_preserved": (bool,),
                "backend": (str,),
                "species": (str,),
                "checkpoint_path": (str,),
                "device": (str,),
            },
            validator=_validate_embedding_result,
        ),
        exception_classifier=_classify_embedding_exception,
        recovery_policy_version="epizoo-embed-cells-v2",
    )
    neighbors_spec = ToolSpec(
        name="build_cell_neighbors",
        function=build_cell_neighbors,
        required_arguments={
            "embedding_path": path_argument,
            "cell_ids_path": path_argument,
            "output_dir": path_argument,
        },
        optional_arguments={
            "n_neighbors": ArgumentSpec((int,)),
            "metric": ArgumentSpec((str,), choices=("euclidean", "cosine")),
            "random_seed": ArgumentSpec((int,)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="CellNeighborsToolResult",
            required_fields={
                "status": (str,),
                "embedding_path": (str,),
                "cell_ids_path": (str,),
                "analysis_path": (str,),
                "n_cells": (int,),
                "embedding_dim": (int,),
                "n_neighbors": (int,),
                "metric": (str,),
                "neighbors_method": (str,),
                "transformer": (str,),
                "random_seed": (int,),
                "connectivities_nnz": (int,),
                "distances_nnz": (int,),
                "finite": (bool,),
                "cell_order_preserved": (bool,),
                "backend": (str,),
                "software_versions": (dict,),
            },
            validator=_validate_neighbors_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="build-cell-neighbors-v1",
    )
    clustering_spec = ToolSpec(
        name="cluster_cells",
        function=cluster_cells,
        required_arguments={
            "analysis_path": path_argument,
            "output_dir": path_argument,
        },
        optional_arguments={
            "resolution": ArgumentSpec((int, float)),
            "random_seed": ArgumentSpec((int,)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="CellClusteringToolResult",
            required_fields={
                "status": (str,),
                "input_analysis_path": (str,),
                "analysis_path": (str,),
                "n_cells": (int,),
                "n_clusters": (int,),
                "cluster_key": (str,),
                "algorithm": (str,),
                "resolution": (float,),
                "random_seed": (int,),
                "cell_order_preserved": (bool,),
                "backend": (str,),
                "software_versions": (dict,),
            },
            validator=_validate_clustering_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="cluster-cells-v1",
    )
    umap_spec = ToolSpec(
        name="compute_cell_umap",
        function=compute_cell_umap,
        required_arguments={
            "analysis_path": path_argument,
            "output_dir": path_argument,
        },
        optional_arguments={
            "min_dist": ArgumentSpec((int, float)),
            "spread": ArgumentSpec((int, float)),
            "random_seed": ArgumentSpec((int,)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="CellUMAPToolResult",
            required_fields={
                "status": (str,),
                "input_analysis_path": (str,),
                "analysis_path": (str,),
                "n_cells": (int,),
                "n_components": (int,),
                "umap_key": (str,),
                "coordinate_dtype": (str,),
                "finite": (bool,),
                "min_dist": (float,),
                "spread": (float,),
                "random_seed": (int,),
                "cell_order_preserved": (bool,),
                "backend": (str,),
                "software_versions": (dict,),
            },
            validator=_validate_umap_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="compute-cell-umap-v1",
    )
    evaluation_spec = ToolSpec(
        name="evaluate_cell_clustering",
        function=evaluate_cell_clustering,
        required_arguments={
            "analysis_path": path_argument,
            "reference_h5ad_path": path_argument,
            "label_key": ArgumentSpec((str,)),
            "output_dir": path_argument,
        },
        optional_arguments={
            "cluster_key": ArgumentSpec((str,)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="CellClusteringEvaluationToolResult",
            required_fields={
                "status": (str,),
                "analysis_path": (str,),
                "reference_h5ad_path": (str,),
                "report_path": (str,),
                "label_key": (str,),
                "cluster_key": (str,),
                "n_cells": (int,),
                "n_reference_classes": (int,),
                "n_predicted_clusters": (int,),
                "nmi": (float,),
                "ari": (float,),
                "ami": (float,),
                "homogeneity": (float,),
                "finite": (bool,),
                "cell_order_preserved": (bool,),
                "metric_backend": (str,),
                "average_method": (str,),
                "report_schema_version": (int,),
                "software_versions": (dict,),
            },
            validator=_validate_evaluation_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="evaluate-cell-clustering-v1",
    )
    label_transfer_spec = ToolSpec(
        name="transfer_cell_labels",
        function=transfer_cell_labels,
        required_arguments={
            "reference_embedding_path": path_argument,
            "reference_cell_ids_path": path_argument,
            "reference_h5ad_path": path_argument,
            "reference_label_key": ArgumentSpec((str,)),
            "query_embedding_path": path_argument,
            "query_cell_ids_path": path_argument,
            "query_h5ad_path": path_argument,
            "output_dir": path_argument,
            "reference_species": ArgumentSpec(
                (str,), choices=("human", "mouse")
            ),
            "query_species": ArgumentSpec((str,), choices=("human", "mouse")),
            "reference_checkpoint_path": path_argument,
            "query_checkpoint_path": path_argument,
        },
        optional_arguments={
            "n_neighbors": ArgumentSpec((int,)),
            "metric": ArgumentSpec((str,), choices=("euclidean", "cosine")),
            "min_confidence": ArgumentSpec((int, float)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="CellLabelTransferToolResult",
            required_fields={
                "status": (str,),
                "annotation_path": (str,),
                "annotation_sha256": (str,),
                "reference_embedding_path": (str,),
                "reference_cell_ids_path": (str,),
                "reference_h5ad_path": (str,),
                "query_embedding_path": (str,),
                "query_cell_ids_path": (str,),
                "query_h5ad_path": (str,),
                "checkpoint_path": (str,),
                "reference_label_key": (str,),
                "n_reference_cells": (int,),
                "n_query_cells": (int,),
                "n_reference_classes": (int,),
                "assigned_count": (int,),
                "unassigned_count": (int,),
                "assignment_rate": (float,),
                "embedding_dim": (int,),
                "embedding_dtype": (str,),
                "n_neighbors": (int,),
                "metric": (str,),
                "voting_method": (str,),
                "min_confidence": (float,),
                "backend": (str,),
                "species": (str,),
                "species_compatible": (bool,),
                "checkpoint_compatible": (bool,),
                "cell_order_preserved": (bool,),
                "finite": (bool,),
                "reference_embedding_sha256": (str,),
                "query_embedding_sha256": (str,),
                "reference_cell_ids_sha256": (str,),
                "query_cell_ids_sha256": (str,),
                "reference_labels_sha256": (str,),
                "model_config_sha256": (str,),
                "artifact_schema_version": (int,),
                "software_versions": (dict,),
            },
            validator=_validate_label_transfer_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="transfer-cell-labels-v1",
    )
    annotation_evaluation_spec = ToolSpec(
        name="evaluate_cell_annotation",
        function=evaluate_cell_annotation,
        required_arguments={
            "annotation_path": path_argument,
            "ground_truth_h5ad_path": path_argument,
            "ground_truth_label_key": ArgumentSpec((str,)),
            "output_dir": path_argument,
        },
        optional_arguments={"overwrite": ArgumentSpec((bool,))},
        result_contract=ResultContract(
            name="CellAnnotationEvaluationToolResult",
            required_fields={
                "status": (str,),
                "annotation_path": (str,),
                "annotation_sha256": (str,),
                "ground_truth_h5ad_path": (str,),
                "report_path": (str,),
                "ground_truth_label_key": (str,),
                "n_cells": (int,),
                "n_ground_truth_classes": (int,),
                "n_assigned_predicted_classes": (int,),
                "assigned_count": (int,),
                "unassigned_count": (int,),
                "assignment_rate": (float,),
                "correct_assigned_count": (int,),
                "incorrect_assigned_count": (int,),
                "overall_accuracy": (float,),
                "assigned_accuracy": (float, type(None)),
                "macro_f1": (float,),
                "median_confidence": (float,),
                "median_assigned_confidence": (float, type(None)),
                "median_correct_assigned_confidence": (float, type(None)),
                "median_incorrect_assigned_confidence": (float, type(None)),
                "finite": (bool,),
                "cell_order_preserved": (bool,),
                "metric_backend": (str,),
                "macro_average": (str,),
                "zero_division": (int,),
                "report_schema_version": (int,),
                "software_versions": (dict,),
            },
            validator=_validate_annotation_evaluation_result,
        ),
        exception_classifier=_classify_analysis_exception,
        recovery_policy_version="evaluate-cell-annotation-v1",
    )
    feature_space_spec = ToolSpec(
        name="validate_scATAC_feature_space",
        function=validate_scATAC_feature_space,
        required_arguments={
            "input_path": path_argument,
            "output_dir": path_argument,
            "matrix_source": ArgumentSpec((str,), choices=("X", "layer")),
            "matrix_semantics": ArgumentSpec(
                (str,),
                choices=(
                    "fragment_counts",
                    "insertion_counts",
                    "binary_accessibility",
                    "normalized_continuous",
                ),
            ),
            "species": ArgumentSpec((str,), choices=("human", "mouse")),
            "genome_assembly": ArgumentSpec((str,), choices=("hg38", "mm10")),
            "coordinate_source": ArgumentSpec((str,), choices=("none", "var_columns")),
        },
        optional_arguments={
            "layer_key": ArgumentSpec((str, type(None))),
            "feature_chrom_key": ArgumentSpec((str, type(None))),
            "feature_start_key": ArgumentSpec((str, type(None))),
            "feature_end_key": ArgumentSpec((str, type(None))),
            "coordinate_system": ArgumentSpec(
                (str, type(None)),
                choices=("zero_based_half_open", "one_based_closed", None),
            ),
            "semantics_metadata_key": ArgumentSpec((str, type(None))),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="ScATACFeatureSpaceToolResult",
            required_fields={
                "status": (str,), "feature_space_path": (str,),
                "feature_space_sha256": (str,), "feature_space_identity_sha256": (str,),
                "input_path": (str,), "source_h5ad_sha256": (str,),
                "matrix_source": (str,), "layer_key": (str, type(None)),
                "matrix_semantics": (str,), "semantics_assertion_source": (str,),
                "pseudobulk_eligible": (bool,), "species": (str,),
                "genome_assembly": (str,), "coordinate_source": (str,),
                "coordinate_system": (str, type(None)), "n_cells": (int,),
                "n_features": (int,), "nnz": (int,), "source_dtype": (str,),
                "source_sparse_format": (str,), "cell_ids_sha256": (str,),
                "feature_ids_sha256": (str,), "matrix_sha256": (str,),
                "coordinates_sha256": (str, type(None)),
                "artifact_schema_version": (int,), "software_versions": (dict,),
            },
            validator=_validate_feature_space_result,
        ),
        exception_classifier=_classify_m81_exception,
        recovery_policy_version="validate-scatac-feature-space-v1",
    )
    pseudobulk_spec = ToolSpec(
        name="build_replicate_pseudobulk",
        function=build_replicate_pseudobulk,
        required_arguments={
            "feature_space_path": path_argument,
            "replicate_key": ArgumentSpec((str,)),
            "group_key": ArgumentSpec((str,)),
            "condition_key": ArgumentSpec((str,)),
            "output_dir": path_argument,
            "group_source": ArgumentSpec(
                (str,), choices=("raw_obs", "verified_annotation")
            ),
        },
        optional_arguments={
            "group_annotation_path": ArgumentSpec((str, Path, type(None))),
            "covariate_keys": ArgumentSpec((list, tuple)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="ReplicatePseudobulkToolResult",
            required_fields={
                "status": (str,), "pseudobulk_path": (str,),
                "pseudobulk_sha256": (str,), "feature_space_path": (str,),
                "feature_space_sha256": (str,), "feature_space_identity_sha256": (str,),
                "source_h5ad_path": (str,), "source_h5ad_sha256": (str,),
                "matrix_semantics": (str,), "output_value_semantics": (str,),
                "aggregation_method": (str,), "output_dtype": (str,),
                "group_source": (str,), "group_key": (str,),
                "replicate_key": (str,), "condition_key": (str,),
                "covariate_keys": (list,), "n_cells": (int,),
                "n_features": (int,), "n_pseudobulks": (int,),
                "n_groups": (int,), "n_replicates": (int,),
                "n_conditions": (int,), "minimum_cells_per_pseudobulk": (int,),
                "maximum_cells_per_pseudobulk": (int,), "matrix_nnz": (int,),
                "total_sum": (int,), "all_cells_accounted_for": (bool,),
                "feature_order_preserved": (bool,), "artifact_schema_version": (int,),
                "software_versions": (dict,),
            },
            validator=_validate_pseudobulk_result,
        ),
        exception_classifier=_classify_m81_exception,
        recovery_policy_version="build-replicate-pseudobulk-v1",
    )
    differential_accessibility_spec = ToolSpec(
        name="run_replicate_differential_accessibility",
        function=run_replicate_differential_accessibility,
        required_arguments={
            "pseudobulk_path": path_argument,
            "group_value": ArgumentSpec((str,)),
            "condition_key": ArgumentSpec((str,)),
            "numerator_condition": ArgumentSpec((str,)),
            "denominator_condition": ArgumentSpec((str,)),
            "design_type": ArgumentSpec(
                (str,), choices=("independent", "paired")
            ),
            "output_dir": path_argument,
        },
        optional_arguments={
            "covariates": ArgumentSpec((list, tuple)),
            "overwrite": ArgumentSpec((bool,)),
        },
        result_contract=ResultContract(
            name="ReplicateDifferentialAccessibilityToolResult",
            required_fields={
                "status": (str,),
                "da_path": (str,),
                "da_sha256": (str,),
                "artifact_type": (str,),
                "artifact_schema_version": (int,),
                "pseudobulk_path": (str,),
                "pseudobulk_sha256": (str,),
                "preparation_sha256": (str,),
                "analysis_sha256": (str,),
                "group_value": (str,),
                "condition_key": (str,),
                "numerator_condition": (str,),
                "denominator_condition": (str,),
                "design_type": (str,),
                "n_samples": (int,),
                "n_numerator_replicates": (int,),
                "n_denominator_replicates": (int,),
                "design_rank": (int,),
                "residual_degrees_of_freedom": (int,),
                "warning_codes": (list,),
                "n_warnings": (int,),
                "n_input_features": (int,),
                "n_tested_features": (int,),
                "n_filtered_features": (int,),
                "filtering_method": (str,),
                "normalization_method": (str,),
                "backend_pipeline": (str,),
                "production_r_script_sha256": (str,),
                "r_version": (str,),
                "bioconductor_version": (str,),
                "edger_version": (str,),
                "package_versions": (dict,),
            },
            validator=_validate_differential_accessibility_result,
        ),
        exception_classifier=_classify_m82_exception,
        retryable_error_codes=frozenset(),
        recovery_policy_version=(
            "run-replicate-differential-accessibility-edger-ql-v1"
        ),
    )
    specs = (
        inspect_spec,
        embedding_spec,
        neighbors_spec,
        clustering_spec,
        umap_spec,
        evaluation_spec,
        label_transfer_spec,
        annotation_evaluation_spec,
        feature_space_spec,
        pseudobulk_spec,
        differential_accessibility_spec,
    )
    for spec in specs:
        _assert_signature_matches(spec)
    return ToolRegistry(specs)


__all__ = [
    "ArgumentSpec",
    "ErrorClassification",
    "ResultContract",
    "ToolArgumentError",
    "ToolRegistry",
    "ToolResultContractError",
    "ToolSpec",
    "UnknownToolError",
    "build_default_tool_registry",
]
