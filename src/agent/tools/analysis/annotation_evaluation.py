"""Evaluation and confidence diagnostics for fixed cell-label annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Literal, TypedDict

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .embedding_analysis import EPIZOO_EMBEDDING_DIM
from .label_transfer import (
    LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION,
    LABEL_TRANSFER_ARTIFACT_TYPE,
    LABEL_TRANSFER_BACKEND,
    LABEL_TRANSFER_PROVENANCE_KEY,
    LABEL_TRANSFER_STAGE,
    LABEL_TRANSFER_VOTING_METHOD,
)


ANNOTATION_EVALUATION_REPORT_SCHEMA_VERSION = 1
ANNOTATION_EVALUATION_ARTIFACT_TYPE = "agent.cell-annotation-evaluation"
ANNOTATION_METRIC_BACKEND = "scikit-learn"
ANNOTATION_MACRO_AVERAGE = "macro"
ANNOTATION_ZERO_DIVISION = 0
_SCORE_TOLERANCE = 1e-12


class CellAnnotationEvaluationToolResult(TypedDict):
    """Lightweight result for persisted annotation-evaluation diagnostics."""

    status: Literal["success"]
    annotation_path: str
    annotation_sha256: str
    ground_truth_h5ad_path: str
    report_path: str
    ground_truth_label_key: str
    n_cells: int
    n_ground_truth_classes: int
    n_assigned_predicted_classes: int
    assigned_count: int
    unassigned_count: int
    assignment_rate: float
    correct_assigned_count: int
    incorrect_assigned_count: int
    overall_accuracy: float
    assigned_accuracy: float | None
    macro_f1: float
    median_confidence: float
    median_assigned_confidence: float | None
    median_correct_assigned_confidence: float | None
    median_incorrect_assigned_confidence: float | None
    finite: bool
    cell_order_preserved: bool
    metric_backend: Literal["scikit-learn"]
    macro_average: Literal["macro"]
    zero_division: Literal[0]
    report_schema_version: int
    software_versions: dict[str, str]


@dataclass(frozen=True)
class _AnnotationPredictions:
    path: Path
    annotation_sha256: str
    cell_ids: tuple[str, ...]
    predicted_labels: tuple[str | None, ...]
    statuses: tuple[str, ...]
    confidences: np.ndarray
    provenance: dict[str, object]
    provenance_sha256: str


@dataclass(frozen=True)
class _AnnotationEvaluationSnapshot:
    annotation_path: Path
    ground_truth_h5ad_path: Path
    ground_truth_label_key: str
    annotation_sha256: str
    n_cells: int
    n_ground_truth_classes: int
    n_assigned_predicted_classes: int
    assigned_count: int
    unassigned_count: int
    assignment_rate: float
    correct_assigned_count: int
    incorrect_assigned_count: int
    overall_accuracy: float
    assigned_accuracy: float | None
    macro_f1: float
    median_confidence: float
    median_assigned_confidence: float | None
    median_correct_assigned_confidence: float | None
    median_incorrect_assigned_confidence: float | None
    per_class: tuple[dict[str, object], ...]
    confusion: dict[str, object]
    annotation_provenance_sha256: str
    query_cell_ids_sha256: str
    ground_truth_labels_sha256: str
    predicted_labels_sha256: str
    prediction_status_sha256: str
    prediction_confidence_sha256: str
    software_versions: dict[str, str]


def _software_versions() -> dict[str, str]:
    distributions = {
        "scikit_learn": "scikit-learn",
        "anndata": "anndata",
        "numpy": "numpy",
        "pandas": "pandas",
    }
    versions: dict[str, str] = {}
    for key, distribution in distributions.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _resolve_h5ad(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"`{name}` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input h5ad file does not exist: {path}")
    if path.suffix.casefold() != ".h5ad":
        raise ValueError(f"`{name}` must identify a .h5ad file: {path}")
    return path


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Annotation-evaluation output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_report_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Annotation-evaluation report already exists: {path}. "
            "Use overwrite=True to replace it."
        )


def _validate_nonempty_key(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a nonempty string.")
    if value != value.strip():
        raise ValueError(f"`{name}` must not contain surrounding whitespace.")
    return value


def _close_adata(adata: ad.AnnData) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to hash annotation artifact: {path}") from exc
    return digest.hexdigest()


def _json_compatible(value: object, *, path: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite value.")
        return value
    if isinstance(value, np.generic):
        return _json_compatible(value.item(), path=path)
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key.")
            converted[key] = _json_compatible(nested, path=f"{path}.{key}")
        return converted
    if isinstance(value, np.ndarray):
        return [
            _json_compatible(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value.tolist())
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _json_compatible(nested, path=f"{path}[{index}]")
            for index, nested in enumerate(value)
        ]
    raise ValueError(f"{path} contains unsupported value type {type(value).__name__}.")


def _canonical_digest(value: object, *, domain: str) -> str:
    payload = json.dumps(
        {"domain": domain, "value": _json_compatible(value)},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ordered_strings_digest(values: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _predicted_labels_digest(values: Sequence[str | None]) -> str:
    digest = hashlib.sha256()
    digest.update(b"agent.annotation-predicted-labels.v1\0")
    for value in values:
        if value is None:
            digest.update(b"\x00")
        else:
            encoded = value.encode("utf-8")
            digest.update(b"\x01")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _confidence_digest(values: np.ndarray, *, rows_per_chunk: int = 8192) -> str:
    canonical = np.asarray(values, dtype="<f8")
    if canonical.ndim != 1 or not np.isfinite(canonical).all():
        raise ValueError("Prediction confidence digest requires finite values.")
    header = json.dumps(
        {
            "schema": "agent.annotation-prediction-confidence.v1",
            "count": int(canonical.shape[0]),
            "dtype": "float64-le",
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    for start in range(0, canonical.shape[0], rows_per_chunk):
        chunk = np.ascontiguousarray(canonical[start : start + rows_per_chunk])
        digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _strict_biological_labels(
    series: pd.Series, *, source: str
) -> tuple[str, ...]:
    if len(series) <= 0:
        raise ValueError(f"{source} must contain at least one biological label.")
    try:
        missing = series.isna()
    except Exception as exc:
        raise ValueError(f"Unable to validate missing values in {source}.") from exc
    if bool(missing.any()):
        raise ValueError(f"{source} contains missing biological labels.")
    labels: list[str] = []
    for value in series.tolist():
        if not isinstance(value, str):
            raise ValueError(f"{source} must contain only string biological labels.")
        if not value.strip():
            raise ValueError(f"{source} contains blank biological labels.")
        if value != value.strip():
            raise ValueError(
                f"{source} contains biological labels with surrounding whitespace."
            )
        labels.append(value)
    return tuple(labels)


def _ordered_ids(index: pd.Index, *, source: str) -> tuple[str, ...]:
    values = tuple(str(value) for value in index)
    if not values:
        raise ValueError(f"{source} must contain at least one cell.")
    if any(not value for value in values):
        raise ValueError(f"{source} cell identifiers must be nonempty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{source} cell identifiers must be unique.")
    return values


def _strict_provenance(
    value: object,
    *,
    n_cells: int,
    n_reference_classes: int,
    assigned_count: int,
    unassigned_count: int,
) -> dict[str, object]:
    provenance = _json_compatible(value, path="label_transfer_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Annotation artifact has invalid label-transfer provenance.")
    expected_root = {
        "schema_version",
        "artifact_type",
        "stage",
        "inputs",
        "compatibility",
        "digests",
        "parameters",
        "backend",
        "counts",
        "software_versions",
    }
    if set(provenance) != expected_root:
        raise ValueError("Annotation artifact has invalid provenance fields.")
    if (
        provenance["schema_version"] != LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION
        or provenance["artifact_type"] != LABEL_TRANSFER_ARTIFACT_TYPE
        or provenance["stage"] != LABEL_TRANSFER_STAGE
    ):
        raise ValueError("Annotation artifact has invalid schema, type, or stage.")

    inputs = provenance["inputs"]
    compatibility = provenance["compatibility"]
    digests = provenance["digests"]
    parameters = provenance["parameters"]
    backend = provenance["backend"]
    counts = provenance["counts"]
    software = provenance["software_versions"]
    expected_nested = {
        "inputs": {
            "reference_embedding_path",
            "reference_cell_ids_path",
            "reference_h5ad_path",
            "query_embedding_path",
            "query_cell_ids_path",
            "query_h5ad_path",
            "checkpoint_path",
            "reference_label_key",
        },
        "compatibility": {
            "species",
            "species_compatible",
            "checkpoint_compatible",
            "model_config_sha256",
        },
        "digests": {
            "reference_embedding_sha256",
            "query_embedding_sha256",
            "reference_cell_ids_sha256",
            "query_cell_ids_sha256",
            "reference_labels_sha256",
        },
        "parameters": {
            "embedding_dim",
            "embedding_dtype",
            "n_neighbors",
            "metric",
            "voting_method",
            "min_confidence",
            "working_memory_mib",
        },
        "backend": {"name", "version", "n_jobs", "exact"},
        "counts": {
            "n_reference_cells",
            "n_query_cells",
            "n_reference_classes",
            "assigned_count",
            "unassigned_count",
        },
    }
    nested_values = {
        "inputs": inputs,
        "compatibility": compatibility,
        "digests": digests,
        "parameters": parameters,
        "backend": backend,
        "counts": counts,
    }
    for name, expected in expected_nested.items():
        nested = nested_values[name]
        if not isinstance(nested, dict) or set(nested) != expected:
            raise ValueError(f"Annotation provenance section {name!r} is invalid.")
    assert isinstance(inputs, dict)
    assert isinstance(compatibility, dict)
    assert isinstance(digests, dict)
    assert isinstance(parameters, dict)
    assert isinstance(backend, dict)
    assert isinstance(counts, dict)
    if not all(isinstance(item, str) and item for item in inputs.values()):
        raise ValueError("Annotation provenance input paths or keys are invalid.")
    if (
        compatibility.get("species") not in {"human", "mouse"}
        or compatibility.get("species_compatible") is not True
        or compatibility.get("checkpoint_compatible") is not True
        or not _is_sha256(compatibility.get("model_config_sha256"))
    ):
        raise ValueError("Annotation compatibility provenance is invalid.")
    if not all(_is_sha256(digests.get(name)) for name in expected_nested["digests"]):
        raise ValueError("Annotation source digests are invalid.")
    n_neighbors = parameters.get("n_neighbors")
    min_confidence = parameters.get("min_confidence")
    working_memory = parameters.get("working_memory_mib")
    if (
        parameters.get("embedding_dim") != EPIZOO_EMBEDDING_DIM
        or parameters.get("embedding_dtype") != "float32"
        or isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, int)
        or n_neighbors < 1
        or parameters.get("metric") not in {"euclidean", "cosine"}
        or parameters.get("voting_method") != LABEL_TRANSFER_VOTING_METHOD
        or isinstance(min_confidence, bool)
        or not isinstance(min_confidence, (int, float))
        or not math.isfinite(float(min_confidence))
        or not 0.0 <= float(min_confidence) <= 1.0
        or isinstance(working_memory, bool)
        or not isinstance(working_memory, int)
        or working_memory <= 0
    ):
        raise ValueError("Annotation scientific parameter provenance is invalid.")
    if (
        backend.get("name") != LABEL_TRANSFER_BACKEND
        or not isinstance(backend.get("version"), str)
        or not backend["version"]
        or backend.get("n_jobs") != 1
        or backend.get("exact") is not True
    ):
        raise ValueError("Annotation backend provenance is invalid.")
    if not all(
        not isinstance(counts.get(name), bool) and isinstance(counts.get(name), int)
        for name in expected_nested["counts"]
    ):
        raise ValueError("Annotation provenance counts are invalid.")
    if (
        counts["n_reference_cells"] <= 0
        or counts["n_query_cells"] != n_cells
        or counts["n_reference_classes"] != n_reference_classes
        or counts["n_reference_classes"] < 2
        or counts["assigned_count"] != assigned_count
        or counts["unassigned_count"] != unassigned_count
        or counts["assigned_count"] + counts["unassigned_count"] != n_cells
        or n_neighbors > counts["n_reference_cells"]
    ):
        raise ValueError("Annotation provenance counts are inconsistent.")
    if not isinstance(software, dict) or not software or not all(
        isinstance(key, str)
        and key
        and isinstance(item, str)
        and item
        for key, item in software.items()
    ):
        raise ValueError("Annotation software provenance is invalid.")
    return provenance


def _read_annotation_predictions(path: Path) -> _AnnotationPredictions:
    try:
        # Milestone 6.3 annotations are deliberately compact (zero variables and
        # only three obs columns). Reading them in memory lets AnnData represent
        # the required absent X matrix as ``None``; backed AnnData raises a
        # KeyError when an H5AD intentionally has no X dataset.
        annotation = ad.read_h5ad(path)
    except Exception as exc:
        raise ValueError(f"Unable to read label-transfer annotation: {path}") from exc
    try:
        if annotation.n_obs <= 0 or annotation.n_vars != 0 or annotation.X is not None:
            raise ValueError("Annotation artifact must have cells, zero variables, and X=None.")
        if tuple(annotation.obs.columns) != (
            "predicted_label",
            "prediction_confidence",
            "prediction_status",
        ):
            raise ValueError("Annotation artifact has invalid observation columns.")
        if set(annotation.uns) != {LABEL_TRANSFER_PROVENANCE_KEY}:
            raise ValueError("Annotation artifact has unexpected unstructured data.")
        if any(
            len(container) != 0
            for container in (
                annotation.obsm,
                annotation.obsp,
                annotation.layers,
                annotation.varm,
                annotation.varp,
            )
        ) or len(annotation.var.columns) != 0:
            raise ValueError("Annotation artifact contains unexpected scientific arrays.")
        cell_ids = _ordered_ids(annotation.obs_names, source="Annotation artifact")
        labels = annotation.obs["predicted_label"]
        statuses_series = annotation.obs["prediction_status"]
        if not isinstance(labels.dtype, pd.CategoricalDtype):
            raise ValueError("Annotation predicted labels must be categorical.")
        if not isinstance(statuses_series.dtype, pd.CategoricalDtype):
            raise ValueError("Annotation prediction status must be categorical.")
        vocabulary = tuple(labels.cat.categories.tolist())
        if len(vocabulary) < 2 or any(
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            for value in vocabulary
        ):
            raise ValueError("Annotation reference-label vocabulary is invalid.")
        if tuple(statuses_series.cat.categories.tolist()) != (
            "assigned",
            "unassigned",
        ):
            raise ValueError("Annotation prediction-status categories are invalid.")
        statuses = tuple(str(value) for value in statuses_series.tolist())
        if any(value not in {"assigned", "unassigned"} for value in statuses):
            raise ValueError("Annotation prediction status is invalid.")
        confidences = np.asarray(
            annotation.obs["prediction_confidence"], dtype=np.float64
        )
        if confidences.shape != (annotation.n_obs,) or not np.isfinite(confidences).all():
            raise ValueError("Annotation prediction confidences must be finite.")
        if np.any((confidences < 0.0) | (confidences > 1.0)):
            raise ValueError("Annotation prediction confidences must lie in [0, 1].")
        predicted: list[str | None] = []
        assigned_count = 0
        for index, status in enumerate(statuses):
            value = labels.iloc[index]
            if status == "assigned":
                if pd.isna(value) or not isinstance(value, str) or value not in vocabulary:
                    raise ValueError("Assigned annotation cells require a biological label.")
                predicted.append(value)
                assigned_count += 1
            else:
                if not pd.isna(value):
                    raise ValueError("Unassigned annotation cells require a missing label.")
                predicted.append(None)
        provenance = _strict_provenance(
            annotation.uns[LABEL_TRANSFER_PROVENANCE_KEY],
            n_cells=annotation.n_obs,
            n_reference_classes=len(vocabulary),
            assigned_count=assigned_count,
            unassigned_count=annotation.n_obs - assigned_count,
        )
    finally:
        _close_adata(annotation)
    return _AnnotationPredictions(
        path=path,
        annotation_sha256=_file_sha256(path),
        cell_ids=cell_ids,
        predicted_labels=tuple(predicted),
        statuses=statuses,
        confidences=confidences,
        provenance=provenance,
        provenance_sha256=_canonical_digest(
            provenance, domain="agent.label-transfer-provenance.v1"
        ),
    )


def _median_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    median = float(np.median(values))
    if not math.isfinite(median) or not 0.0 <= median <= 1.0:
        raise RuntimeError("Prediction confidence median is invalid.")
    return median


def _validate_score(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeError(f"{name} is not numeric.")
    score = float(value)
    if not math.isfinite(score) or not -_SCORE_TOLERANCE <= score <= 1.0 + _SCORE_TOLERANCE:
        raise RuntimeError(f"{name} is outside [0, 1].")
    return score


def _evaluate_annotation_sources(
    annotation_path: str | Path,
    ground_truth_h5ad_path: str | Path,
    ground_truth_label_key: str,
) -> _AnnotationEvaluationSnapshot:
    resolved_annotation = _resolve_h5ad(annotation_path, "annotation_path")
    resolved_ground_truth = _resolve_h5ad(
        ground_truth_h5ad_path, "ground_truth_h5ad_path"
    )
    if resolved_annotation == resolved_ground_truth:
        raise ValueError("Annotation and ground-truth paths must identify different files.")
    label_key = _validate_nonempty_key(
        ground_truth_label_key, "ground_truth_label_key"
    )
    annotation = _read_annotation_predictions(resolved_annotation)

    try:
        ground_truth = ad.read_h5ad(resolved_ground_truth, backed="r")
    except Exception as exc:
        raise ValueError(
            f"Unable to read ground-truth AnnData: {resolved_ground_truth}"
        ) from exc
    try:
        ground_truth_ids = _ordered_ids(
            ground_truth.obs_names, source="Ground-truth AnnData"
        )
        if label_key not in ground_truth.obs:
            raise ValueError(f"Ground-truth AnnData lacks label key {label_key!r}.")
        ground_truth_labels = _strict_biological_labels(
            ground_truth.obs[label_key].copy(),
            source=f"Ground-truth label column {label_key!r}",
        )
    finally:
        _close_adata(ground_truth)
    if len(annotation.cell_ids) != len(ground_truth_ids):
        raise ValueError("Annotation and ground truth must contain identical cell counts.")
    if annotation.cell_ids != ground_truth_ids:
        if set(annotation.cell_ids) == set(ground_truth_ids):
            raise ValueError("Annotation and ground-truth cells have different exact order.")
        raise ValueError("Annotation and ground-truth cell identities do not match.")
    if len(ground_truth_labels) != len(annotation.cell_ids):
        raise ValueError("Ground-truth label count does not match annotation cells.")

    ground_truth_order = tuple(dict.fromkeys(ground_truth_labels))
    ground_truth_codes = {
        label: index for index, label in enumerate(ground_truth_order)
    }
    observed_predictions = tuple(
        dict.fromkeys(
            value
            for value, status in zip(
                annotation.predicted_labels, annotation.statuses, strict=True
            )
            if status == "assigned" and value is not None
        )
    )
    shared_prediction_order = tuple(
        label for label in ground_truth_order if label in observed_predictions
    )
    external_prediction_order = tuple(
        label for label in observed_predictions if label not in ground_truth_codes
    )
    biological_columns = shared_prediction_order + external_prediction_order
    external_codes = {
        label: len(ground_truth_order) + index
        for index, label in enumerate(external_prediction_order)
    }
    y_true = np.asarray(
        [ground_truth_codes[label] for label in ground_truth_labels], dtype=np.int64
    )
    y_pred = np.full(len(annotation.cell_ids), -1, dtype=np.int64)
    assigned_mask = np.asarray(
        [status == "assigned" for status in annotation.statuses], dtype=bool
    )
    for index in np.flatnonzero(assigned_mask):
        label = annotation.predicted_labels[index]
        assert label is not None
        y_pred[index] = ground_truth_codes.get(label, external_codes.get(label, -1))
    correct_mask = assigned_mask & (y_true == y_pred)
    incorrect_mask = assigned_mask & ~correct_mask
    assigned_count = int(assigned_mask.sum())
    unassigned_count = len(annotation.cell_ids) - assigned_count
    correct_count = int(correct_mask.sum())
    incorrect_count = int(incorrect_mask.sum())
    assignment_rate = assigned_count / len(annotation.cell_ids)
    overall_accuracy = correct_count / len(annotation.cell_ids)
    assigned_accuracy = (
        correct_count / assigned_count if assigned_count > 0 else None
    )
    labels_for_metrics = list(range(len(ground_truth_order)))
    macro_f1 = _validate_score(
        "Macro-F1",
        f1_score(
            y_true,
            y_pred,
            labels=labels_for_metrics,
            average=ANNOTATION_MACRO_AVERAGE,
            zero_division=ANNOTATION_ZERO_DIVISION,
        ),
    )
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels_for_metrics,
        zero_division=ANNOTATION_ZERO_DIVISION,
    )
    per_class: list[dict[str, object]] = []
    for class_index, label in enumerate(ground_truth_order):
        true_positive = int(
            np.sum((y_true == class_index) & (y_pred == class_index))
        )
        per_class.append(
            {
                "label": label,
                "support": int(support[class_index]),
                "true_positive": true_positive,
                "precision": _validate_score("Per-class precision", precision[class_index]),
                "recall": _validate_score("Per-class recall", recall[class_index]),
                "f1": _validate_score("Per-class F1", f1_values[class_index]),
            }
        )

    column_indices = {label: index for index, label in enumerate(biological_columns)}
    unassigned_column = len(biological_columns)
    confusion_counts = np.zeros(
        (len(ground_truth_order), len(biological_columns) + 1), dtype=np.int64
    )
    for index, true_code in enumerate(y_true):
        if annotation.statuses[index] == "unassigned":
            column = unassigned_column
        else:
            predicted = annotation.predicted_labels[index]
            assert predicted is not None
            column = column_indices[predicted]
        confusion_counts[true_code, column] += 1
    confusion = {
        "rows": {
            "kind": "ground_truth_biological_label",
            "labels": list(ground_truth_order),
        },
        "columns": [
            {"kind": "biological_label", "label": label}
            for label in biological_columns
        ]
        + [{"kind": "structural_unassigned", "label": None}],
        "counts": confusion_counts.tolist(),
    }
    confidences = annotation.confidences
    median_confidence = _median_or_none(confidences)
    assert median_confidence is not None
    return _AnnotationEvaluationSnapshot(
        annotation_path=resolved_annotation,
        ground_truth_h5ad_path=resolved_ground_truth,
        ground_truth_label_key=label_key,
        annotation_sha256=annotation.annotation_sha256,
        n_cells=len(annotation.cell_ids),
        n_ground_truth_classes=len(ground_truth_order),
        n_assigned_predicted_classes=len(observed_predictions),
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
        assignment_rate=float(assignment_rate),
        correct_assigned_count=correct_count,
        incorrect_assigned_count=incorrect_count,
        overall_accuracy=float(overall_accuracy),
        assigned_accuracy=(
            float(assigned_accuracy) if assigned_accuracy is not None else None
        ),
        macro_f1=macro_f1,
        median_confidence=median_confidence,
        median_assigned_confidence=_median_or_none(confidences[assigned_mask]),
        median_correct_assigned_confidence=_median_or_none(confidences[correct_mask]),
        median_incorrect_assigned_confidence=_median_or_none(confidences[incorrect_mask]),
        per_class=tuple(per_class),
        confusion=confusion,
        annotation_provenance_sha256=annotation.provenance_sha256,
        query_cell_ids_sha256=_ordered_strings_digest(
            annotation.cell_ids, domain="agent.annotation-query-cell-ids.v1"
        ),
        ground_truth_labels_sha256=_ordered_strings_digest(
            ground_truth_labels, domain="agent.annotation-ground-truth-labels.v1"
        ),
        predicted_labels_sha256=_predicted_labels_digest(
            annotation.predicted_labels
        ),
        prediction_status_sha256=_ordered_strings_digest(
            annotation.statuses, domain="agent.annotation-prediction-status.v1"
        ),
        prediction_confidence_sha256=_confidence_digest(confidences),
        software_versions=_software_versions(),
    )


def _report_from_snapshot(
    snapshot: _AnnotationEvaluationSnapshot,
) -> dict[str, object]:
    return {
        "schema_version": ANNOTATION_EVALUATION_REPORT_SCHEMA_VERSION,
        "artifact_type": ANNOTATION_EVALUATION_ARTIFACT_TYPE,
        "status": "success",
        "inputs": {
            "annotation_path": str(snapshot.annotation_path),
            "ground_truth_h5ad_path": str(snapshot.ground_truth_h5ad_path),
            "ground_truth_label_key": snapshot.ground_truth_label_key,
        },
        "counts": {
            "n_cells": snapshot.n_cells,
            "n_ground_truth_classes": snapshot.n_ground_truth_classes,
            "n_assigned_predicted_classes": snapshot.n_assigned_predicted_classes,
            "assigned_count": snapshot.assigned_count,
            "unassigned_count": snapshot.unassigned_count,
            "correct_assigned_count": snapshot.correct_assigned_count,
            "incorrect_assigned_count": snapshot.incorrect_assigned_count,
        },
        "metrics": {
            "assignment_rate": snapshot.assignment_rate,
            "overall_accuracy": snapshot.overall_accuracy,
            "assigned_accuracy": snapshot.assigned_accuracy,
            "macro_f1": snapshot.macro_f1,
        },
        "confidence_diagnostics": {
            "median_confidence": snapshot.median_confidence,
            "median_assigned_confidence": snapshot.median_assigned_confidence,
            "median_correct_assigned_confidence": snapshot.median_correct_assigned_confidence,
            "median_incorrect_assigned_confidence": snapshot.median_incorrect_assigned_confidence,
        },
        "per_class": [dict(value) for value in snapshot.per_class],
        "confusion": dict(snapshot.confusion),
        "metric_backend": {
            "name": ANNOTATION_METRIC_BACKEND,
            "version": snapshot.software_versions["scikit_learn"],
            "average": ANNOTATION_MACRO_AVERAGE,
            "zero_division": ANNOTATION_ZERO_DIVISION,
            "unassigned_policy": "structural_excluded_from_macro",
        },
        "validation": {"finite": True, "cell_order_preserved": True},
        "provenance": {
            "annotation_sha256": snapshot.annotation_sha256,
            "annotation_provenance_sha256": snapshot.annotation_provenance_sha256,
            "query_cell_ids_sha256": snapshot.query_cell_ids_sha256,
            "ground_truth_labels_sha256": snapshot.ground_truth_labels_sha256,
            "predicted_labels_sha256": snapshot.predicted_labels_sha256,
            "prediction_status_sha256": snapshot.prediction_status_sha256,
            "prediction_confidence_sha256": snapshot.prediction_confidence_sha256,
            "source_validation_boundary": "annotation_artifact",
        },
        "software_versions": dict(snapshot.software_versions),
    }


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not permitted.")


def _nullable_score(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _validate_score(name, value)


def _validate_report_structure(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        raise ValueError("Annotation-evaluation report must contain one JSON object.")
    expected_root = {
        "schema_version",
        "artifact_type",
        "status",
        "inputs",
        "counts",
        "metrics",
        "confidence_diagnostics",
        "per_class",
        "confusion",
        "metric_backend",
        "validation",
        "provenance",
        "software_versions",
    }
    if set(report) != expected_root:
        raise ValueError("Annotation-evaluation report has an invalid root schema.")
    if (
        report["schema_version"] != ANNOTATION_EVALUATION_REPORT_SCHEMA_VERSION
        or report["artifact_type"] != ANNOTATION_EVALUATION_ARTIFACT_TYPE
        or report["status"] != "success"
    ):
        raise ValueError("Annotation-evaluation report identity is invalid.")
    nested_keys = {
        "inputs": {
            "annotation_path",
            "ground_truth_h5ad_path",
            "ground_truth_label_key",
        },
        "counts": {
            "n_cells",
            "n_ground_truth_classes",
            "n_assigned_predicted_classes",
            "assigned_count",
            "unassigned_count",
            "correct_assigned_count",
            "incorrect_assigned_count",
        },
        "metrics": {
            "assignment_rate",
            "overall_accuracy",
            "assigned_accuracy",
            "macro_f1",
        },
        "confidence_diagnostics": {
            "median_confidence",
            "median_assigned_confidence",
            "median_correct_assigned_confidence",
            "median_incorrect_assigned_confidence",
        },
        "metric_backend": {
            "name",
            "version",
            "average",
            "zero_division",
            "unassigned_policy",
        },
        "validation": {"finite", "cell_order_preserved"},
        "provenance": {
            "annotation_sha256",
            "annotation_provenance_sha256",
            "query_cell_ids_sha256",
            "ground_truth_labels_sha256",
            "predicted_labels_sha256",
            "prediction_status_sha256",
            "prediction_confidence_sha256",
            "source_validation_boundary",
        },
    }
    sections: dict[str, dict[str, object]] = {}
    for name, keys in nested_keys.items():
        value = report[name]
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"Annotation report section {name!r} is invalid.")
        sections[name] = value
    inputs = sections["inputs"]
    counts = sections["counts"]
    metrics = sections["metrics"]
    diagnostics = sections["confidence_diagnostics"]
    backend = sections["metric_backend"]
    validation = sections["validation"]
    provenance = sections["provenance"]
    if not all(isinstance(value, str) and value for value in inputs.values()):
        raise ValueError("Annotation report inputs are invalid.")
    if not all(
        not isinstance(counts[name], bool) and isinstance(counts[name], int)
        for name in counts
    ):
        raise ValueError("Annotation report counts are invalid.")
    n_cells = counts["n_cells"]
    assigned = counts["assigned_count"]
    unassigned = counts["unassigned_count"]
    correct = counts["correct_assigned_count"]
    incorrect = counts["incorrect_assigned_count"]
    if (
        n_cells <= 0
        or counts["n_ground_truth_classes"] < 1
        or counts["n_assigned_predicted_classes"] < 0
        or min(assigned, unassigned, correct, incorrect) < 0
        or assigned + unassigned != n_cells
        or correct + incorrect != assigned
    ):
        raise ValueError("Annotation report counts are inconsistent.")
    assignment_rate = _validate_score("Assignment rate", metrics["assignment_rate"])
    overall_accuracy = _validate_score("Overall accuracy", metrics["overall_accuracy"])
    assigned_accuracy = _nullable_score("Assigned accuracy", metrics["assigned_accuracy"])
    _validate_score("Macro-F1", metrics["macro_f1"])
    if not math.isclose(assignment_rate, assigned / n_cells, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Annotation report assignment rate is inconsistent.")
    if not math.isclose(overall_accuracy, correct / n_cells, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Annotation report overall accuracy is inconsistent.")
    expected_assigned_accuracy = correct / assigned if assigned else None
    if assigned_accuracy is None:
        if expected_assigned_accuracy is not None:
            raise ValueError("Annotation report assigned accuracy is unexpectedly null.")
    elif expected_assigned_accuracy is None or not math.isclose(
        assigned_accuracy, expected_assigned_accuracy, rel_tol=0.0, abs_tol=1e-15
    ):
        raise ValueError("Annotation report assigned accuracy is inconsistent.")
    diagnostic_values = {
        name: _nullable_score(name, value) for name, value in diagnostics.items()
    }
    if diagnostic_values["median_confidence"] is None:
        raise ValueError("Overall median confidence must be defined.")
    expected_null = {
        "median_assigned_confidence": assigned == 0,
        "median_correct_assigned_confidence": correct == 0,
        "median_incorrect_assigned_confidence": incorrect == 0,
    }
    for name, should_be_null in expected_null.items():
        if (diagnostic_values[name] is None) != should_be_null:
            raise ValueError(f"Annotation report diagnostic {name!r} has invalid null semantics.")
    if (
        backend
        != {
            "name": ANNOTATION_METRIC_BACKEND,
            "version": backend.get("version"),
            "average": ANNOTATION_MACRO_AVERAGE,
            "zero_division": ANNOTATION_ZERO_DIVISION,
            "unassigned_policy": "structural_excluded_from_macro",
        }
        or not isinstance(backend.get("version"), str)
        or not backend["version"]
    ):
        raise ValueError("Annotation report metric backend is invalid.")
    if validation != {"finite": True, "cell_order_preserved": True}:
        raise ValueError("Annotation report validation flags are invalid.")
    for name in (
        "annotation_sha256",
        "annotation_provenance_sha256",
        "query_cell_ids_sha256",
        "ground_truth_labels_sha256",
        "predicted_labels_sha256",
        "prediction_status_sha256",
        "prediction_confidence_sha256",
    ):
        if not _is_sha256(provenance.get(name)):
            raise ValueError("Annotation report provenance digest is invalid.")
    if provenance.get("source_validation_boundary") != "annotation_artifact":
        raise ValueError("Annotation report source-validation boundary is invalid.")

    per_class = report["per_class"]
    confusion = report["confusion"]
    if not isinstance(per_class, list) or len(per_class) != counts["n_ground_truth_classes"]:
        raise ValueError("Annotation report per-class diagnostics are invalid.")
    if not isinstance(confusion, dict) or set(confusion) != {"rows", "columns", "counts"}:
        raise ValueError("Annotation report confusion schema is invalid.")
    rows = confusion["rows"]
    columns = confusion["columns"]
    matrix = confusion["counts"]
    if (
        not isinstance(rows, dict)
        or set(rows) != {"kind", "labels"}
        or rows.get("kind") != "ground_truth_biological_label"
        or not isinstance(rows.get("labels"), list)
    ):
        raise ValueError("Annotation report confusion rows are invalid.")
    row_labels = rows["labels"]
    if len(row_labels) != len(per_class) or len(set(row_labels)) != len(row_labels):
        raise ValueError("Annotation report confusion row labels are invalid.")
    if not isinstance(columns, list) or not columns:
        raise ValueError("Annotation report confusion columns are invalid.")
    biological_columns: list[str] = []
    for index, descriptor in enumerate(columns):
        if not isinstance(descriptor, dict) or set(descriptor) != {"kind", "label"}:
            raise ValueError("Annotation report confusion descriptor is invalid.")
        if index == len(columns) - 1:
            if descriptor != {"kind": "structural_unassigned", "label": None}:
                raise ValueError("Structural unassigned confusion column is invalid.")
        elif (
            descriptor.get("kind") != "biological_label"
            or not isinstance(descriptor.get("label"), str)
            or not descriptor["label"].strip()
            or descriptor["label"] != descriptor["label"].strip()
        ):
            raise ValueError("Biological confusion column is invalid.")
        else:
            biological_columns.append(descriptor["label"])
    if len(set(biological_columns)) != len(biological_columns):
        raise ValueError("Biological confusion columns must be unique.")
    if len(biological_columns) != counts["n_assigned_predicted_classes"]:
        raise ValueError("Assigned predicted class count is inconsistent.")
    if not isinstance(matrix, list) or len(matrix) != len(row_labels):
        raise ValueError("Annotation report confusion matrix is invalid.")
    biological_column_indices = {
        label: index for index, label in enumerate(biological_columns)
    }
    total = 0
    for entry, label, row in zip(per_class, row_labels, matrix, strict=True):
        if not isinstance(entry, dict) or set(entry) != {
            "label",
            "support",
            "true_positive",
            "precision",
            "recall",
            "f1",
        }:
            raise ValueError("Annotation report per-class entry is invalid.")
        if entry["label"] != label or not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError("Annotation report per-class label is invalid.")
        if (
            isinstance(entry["support"], bool)
            or not isinstance(entry["support"], int)
            or entry["support"] <= 0
            or isinstance(entry["true_positive"], bool)
            or not isinstance(entry["true_positive"], int)
            or not 0 <= entry["true_positive"] <= entry["support"]
        ):
            raise ValueError("Annotation report per-class counts are invalid.")
        for metric in ("precision", "recall", "f1"):
            _validate_score(f"Per-class {metric}", entry[metric])
        if (
            not isinstance(row, list)
            or len(row) != len(columns)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in row)
            or sum(row) != entry["support"]
        ):
            raise ValueError("Annotation report confusion row is inconsistent.")
        total += sum(row)
        expected_true_positive = (
            row[biological_column_indices[label]]
            if label in biological_column_indices
            else 0
        )
        if entry["true_positive"] != expected_true_positive:
            raise ValueError("Annotation report true-positive count is inconsistent.")
    if total != n_cells:
        raise ValueError("Annotation report confusion total is inconsistent.")
    if not isinstance(report["software_versions"], dict) or not report["software_versions"] or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in report["software_versions"].items()
    ):
        raise ValueError("Annotation report software versions are invalid.")
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Annotation report is not strict JSON-safe data.") from exc
    return report


def _load_annotation_evaluation_report(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except Exception as exc:
        raise ValueError(f"Unable to read annotation-evaluation report: {path}") from exc
    return _validate_report_structure(report)


def _atomic_write_report(
    report: Mapping[str, object], output_path: Path, *, overwrite: bool
) -> None:
    _ensure_report_available(output_path, overwrite=overwrite)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            serialized = json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = _load_annotation_evaluation_report(temporary_path)
        if loaded != dict(report):
            raise ValueError("Written annotation-evaluation report failed validation.")
        _ensure_report_available(output_path, overwrite=overwrite)
        os.replace(temporary_path, output_path)
        temporary_path = None
        directory_fd = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def evaluate_cell_annotation(
    annotation_path: str | Path,
    ground_truth_h5ad_path: str | Path,
    ground_truth_label_key: str,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> CellAnnotationEvaluationToolResult:
    """Evaluate a fixed Milestone 6.3 annotation against ordered ground truth."""

    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    resolved_annotation = _resolve_h5ad(annotation_path, "annotation_path")
    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = (
        resolved_output_dir
        / f"{resolved_annotation.stem}.annotation_evaluation.json"
    )
    _ensure_report_available(output_path, overwrite=overwrite)
    snapshot = _evaluate_annotation_sources(
        resolved_annotation,
        ground_truth_h5ad_path,
        ground_truth_label_key,
    )
    report = _report_from_snapshot(snapshot)
    _validate_report_structure(report)
    _atomic_write_report(report, output_path, overwrite=overwrite)
    return {
        "status": "success",
        "annotation_path": str(snapshot.annotation_path),
        "annotation_sha256": snapshot.annotation_sha256,
        "ground_truth_h5ad_path": str(snapshot.ground_truth_h5ad_path),
        "report_path": str(output_path),
        "ground_truth_label_key": snapshot.ground_truth_label_key,
        "n_cells": snapshot.n_cells,
        "n_ground_truth_classes": snapshot.n_ground_truth_classes,
        "n_assigned_predicted_classes": snapshot.n_assigned_predicted_classes,
        "assigned_count": snapshot.assigned_count,
        "unassigned_count": snapshot.unassigned_count,
        "assignment_rate": snapshot.assignment_rate,
        "correct_assigned_count": snapshot.correct_assigned_count,
        "incorrect_assigned_count": snapshot.incorrect_assigned_count,
        "overall_accuracy": snapshot.overall_accuracy,
        "assigned_accuracy": snapshot.assigned_accuracy,
        "macro_f1": snapshot.macro_f1,
        "median_confidence": snapshot.median_confidence,
        "median_assigned_confidence": snapshot.median_assigned_confidence,
        "median_correct_assigned_confidence": snapshot.median_correct_assigned_confidence,
        "median_incorrect_assigned_confidence": snapshot.median_incorrect_assigned_confidence,
        "finite": True,
        "cell_order_preserved": True,
        "metric_backend": ANNOTATION_METRIC_BACKEND,
        "macro_average": ANNOTATION_MACRO_AVERAGE,
        "zero_division": ANNOTATION_ZERO_DIVISION,
        "report_schema_version": ANNOTATION_EVALUATION_REPORT_SCHEMA_VERSION,
        "software_versions": dict(snapshot.software_versions),
    }


__all__ = ["CellAnnotationEvaluationToolResult", "evaluate_cell_annotation"]
