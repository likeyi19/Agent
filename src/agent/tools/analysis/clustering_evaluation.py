"""Supervised evaluation of fixed Milestone 6.1 cell clustering artifacts."""

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
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from .embedding_analysis import (
    PROVENANCE_KEY,
    UMAP_KEY,
    _cell_order_digest,
    _provenance,
    _read_analysis,
    _validate_cluster_labels,
    _validate_neighbors_artifact,
)


EVALUATION_REPORT_SCHEMA_VERSION = 1
EVALUATION_ARTIFACT_TYPE = "agent.cell-clustering-evaluation"
METRIC_BACKEND = "scikit-learn"
AVERAGE_METHOD = "arithmetic"
_SCORE_TOLERANCE = 1e-12


class CellClusteringEvaluationToolResult(TypedDict):
    """Lightweight result for persisted clustering-evaluation metrics."""

    status: Literal["success"]
    analysis_path: str
    reference_h5ad_path: str
    report_path: str
    label_key: str
    cluster_key: str
    n_cells: int
    n_reference_classes: int
    n_predicted_clusters: int
    nmi: float
    ari: float
    ami: float
    homogeneity: float
    finite: bool
    cell_order_preserved: bool
    metric_backend: str
    average_method: str
    report_schema_version: int
    software_versions: dict[str, str]


@dataclass(frozen=True)
class _EvaluationSnapshot:
    analysis_path: Path
    reference_h5ad_path: Path
    label_key: str
    cluster_key: str
    analysis_stage: str
    n_cells: int
    n_reference_classes: int
    n_predicted_clusters: int
    nmi: float
    ari: float
    ami: float
    homogeneity: float
    cell_order_sha256: str
    reference_labels_sha256: str
    predicted_labels_sha256: str
    analysis_provenance_sha256: str
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


def _validate_nonempty_key(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a nonempty string.")
    return value


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
        raise ValueError(f"Evaluation output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_report_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Clustering-evaluation report already exists: {path}. "
            "Use overwrite=True to replace it."
        )


def _ordered_ids(index: pd.Index, *, source: str) -> tuple[str, ...]:
    values = tuple(str(value) for value in index)
    if not values:
        raise ValueError(f"{source} must contain at least one cell.")
    if any(not value for value in values):
        raise ValueError(f"{source} cell identifiers must be nonempty.")
    if len(set(values)) != len(values):
        raise ValueError(f"{source} cell identifiers must be unique.")
    return values


def _label_token(value: object, *, source: str) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{source} contains unsupported boolean label values.")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{source} contains blank-string label values.")
        payload: object = value
        value_type = "string"
    elif isinstance(value, int):
        payload = str(value)
        value_type = "integer"
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{source} contains non-finite label values.")
        payload = value.hex()
        value_type = "float"
    else:
        raise ValueError(
            f"{source} contains unsupported or unhashable label value type "
            f"{type(value).__name__}."
        )
    return json.dumps(
        {"type": value_type, "value": payload},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalized_labels(series: pd.Series, *, source: str) -> tuple[str, ...]:
    if len(series) <= 0:
        raise ValueError(f"{source} must contain at least one label.")
    try:
        missing = series.isna()
    except Exception as exc:
        raise ValueError(f"Unable to validate missing values in {source}.") from exc
    if bool(missing.any()):
        raise ValueError(f"{source} contains missing label values.")
    return tuple(_label_token(value, source=source) for value in series.tolist())


def _ordered_values_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _json_compatible(value: object, *, path: str = "provenance") -> object:
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


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_score(name: str, value: object, *, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeError(f"{name} metric is not numeric.")
    score = float(value)
    if not math.isfinite(score):
        raise RuntimeError(f"{name} metric is non-finite.")
    if score < lower - _SCORE_TOLERANCE or score > upper + _SCORE_TOLERANCE:
        raise RuntimeError(f"{name} metric is outside its valid range.")
    return score


def _close_adata(adata: ad.AnnData) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def _validate_evaluation_analysis(analysis: ad.AnnData) -> tuple[tuple[str, ...], str]:
    """Validate the Milestone 6.1 structures relevant to evaluation."""

    cell_ids, _, _ = _validate_neighbors_artifact(
        analysis, expected_stages=frozenset({"clustering", "umap"})
    )
    _validate_cluster_labels(analysis)
    provenance = _provenance(analysis)
    stage = provenance.get("stage")
    parameters = provenance.get("parameters")
    neighbors = parameters.get("neighbors") if isinstance(parameters, Mapping) else None
    clustering = parameters.get("clustering") if isinstance(parameters, Mapping) else None
    if not isinstance(neighbors, Mapping):
        raise ValueError("Analysis artifact lacks upstream neighbors provenance.")
    n_neighbors = neighbors.get("n_neighbors")
    neighbor_seed = neighbors.get("random_seed")
    if (
        isinstance(n_neighbors, bool)
        or not isinstance(n_neighbors, int)
        or not 2 <= n_neighbors < analysis.n_obs
        or neighbors.get("metric") not in {"euclidean", "cosine"}
        or neighbors.get("method") != "umap"
        or neighbors.get("transformer") != "none"
        or isinstance(neighbor_seed, bool)
        or not isinstance(neighbor_seed, int)
        or neighbor_seed < 0
        or neighbors.get("use_rep") != "X_epizoo"
    ):
        raise ValueError("Analysis artifact has invalid upstream neighbors provenance.")
    if not isinstance(clustering, Mapping):
        raise ValueError("Analysis artifact lacks upstream clustering provenance.")
    resolution = clustering.get("resolution")
    clustering_seed = clustering.get("random_seed")
    if (
        clustering.get("algorithm") != "leiden"
        or isinstance(resolution, bool)
        or not isinstance(resolution, (int, float))
        or not math.isfinite(float(resolution))
        or float(resolution) <= 0
        or clustering.get("flavor") != "igraph"
        or clustering.get("n_iterations") != 2
        or clustering.get("directed") is not False
        or clustering.get("use_weights") is not True
        or isinstance(clustering_seed, bool)
        or not isinstance(clustering_seed, int)
        or clustering_seed < 0
        or clustering.get("key_added") != "leiden"
    ):
        raise ValueError("Analysis artifact has invalid upstream clustering provenance.")
    source_path = provenance.get("source_analysis_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("Analysis artifact lacks its upstream analysis path provenance.")
    if stage == "umap":
        umap = parameters.get("umap") if isinstance(parameters, Mapping) else None
        min_dist = umap.get("min_dist") if isinstance(umap, Mapping) else None
        spread = umap.get("spread") if isinstance(umap, Mapping) else None
        umap_seed = umap.get("random_seed") if isinstance(umap, Mapping) else None
        if (
            not isinstance(umap, Mapping)
            or isinstance(min_dist, bool)
            or not isinstance(min_dist, (int, float))
            or not math.isfinite(float(min_dist))
            or float(min_dist) < 0
            or isinstance(spread, bool)
            or not isinstance(spread, (int, float))
            or not math.isfinite(float(spread))
            or float(spread) <= 0
            or float(min_dist) > float(spread)
            or umap.get("n_components") != 2
            or umap.get("init_pos") != "spectral"
            or isinstance(umap_seed, bool)
            or not isinstance(umap_seed, int)
            or umap_seed < 0
            or umap.get("key_added") != UMAP_KEY
        ):
            raise ValueError("UMAP artifact has invalid UMAP provenance.")
        if UMAP_KEY not in analysis.obsm:
            raise ValueError("UMAP artifact lacks UMAP coordinates.")
        coordinates = np.asarray(analysis.obsm[UMAP_KEY])
        if coordinates.shape != (analysis.n_obs, 2) or not np.isfinite(coordinates).all():
            raise ValueError("UMAP artifact has invalid UMAP coordinates.")
    return cell_ids, str(stage)


def _evaluate_sources(
    analysis_path: str | Path,
    reference_h5ad_path: str | Path,
    label_key: str,
    cluster_key: str,
) -> _EvaluationSnapshot:
    resolved_analysis = _resolve_h5ad(analysis_path, "analysis_path")
    resolved_reference = _resolve_h5ad(reference_h5ad_path, "reference_h5ad_path")
    if resolved_analysis == resolved_reference:
        raise ValueError("Analysis and reference h5ad paths must identify different files.")
    normalized_label_key = _validate_nonempty_key(label_key, "label_key")
    normalized_cluster_key = _validate_nonempty_key(cluster_key, "cluster_key")

    analysis = _read_analysis(resolved_analysis)
    try:
        analysis_ids, stage = _validate_evaluation_analysis(analysis)
        provenance = _provenance(analysis)
        if normalized_cluster_key not in analysis.obs:
            raise ValueError(
                f"Analysis artifact lacks cluster key {normalized_cluster_key!r}."
            )
        predicted = analysis.obs[normalized_cluster_key].copy()
        analysis_provenance_sha256 = _canonical_digest(
            analysis.uns[PROVENANCE_KEY]
        )
    finally:
        _close_adata(analysis)

    try:
        reference = ad.read_h5ad(resolved_reference, backed="r")
    except Exception as exc:
        raise ValueError(
            f"Unable to read reference AnnData file: {resolved_reference}"
        ) from exc
    try:
        reference_ids = _ordered_ids(reference.obs_names, source="Reference AnnData")
        if normalized_label_key not in reference.obs:
            raise ValueError(
                f"Reference AnnData lacks label key {normalized_label_key!r}."
            )
        reference_labels = reference.obs[normalized_label_key].copy()
    finally:
        _close_adata(reference)

    if len(analysis_ids) != len(reference_ids):
        raise ValueError(
            "Analysis and reference artifacts must contain identical cell counts."
        )
    if analysis_ids != reference_ids:
        if set(analysis_ids) == set(reference_ids):
            raise ValueError(
                "Analysis and reference cell identifiers have different exact order."
            )
        raise ValueError("Analysis and reference cell identities do not exactly match.")

    true_tokens = _normalized_labels(
        reference_labels, source=f"Reference label column {normalized_label_key!r}"
    )
    predicted_tokens = _normalized_labels(
        predicted, source=f"Predicted cluster column {normalized_cluster_key!r}"
    )
    if len(true_tokens) != len(analysis_ids) or len(predicted_tokens) != len(analysis_ids):
        raise ValueError("Evaluation label counts do not match the ordered cells.")
    n_reference_classes = len(set(true_tokens))
    n_predicted_clusters = len(set(predicted_tokens))
    if n_reference_classes < 2:
        raise ValueError("Reference labels must contain at least two distinct classes.")
    if n_predicted_clusters < 1:
        raise ValueError("Predicted labels must contain at least one cluster.")

    nmi = _validate_score(
        "NMI",
        normalized_mutual_info_score(
            true_tokens,
            predicted_tokens,
            average_method=AVERAGE_METHOD,
        ),
        lower=0.0,
        upper=1.0,
    )
    ari = _validate_score(
        "ARI",
        adjusted_rand_score(true_tokens, predicted_tokens),
        lower=-1.0,
        upper=1.0,
    )
    ami = _validate_score(
        "AMI",
        adjusted_mutual_info_score(
            true_tokens,
            predicted_tokens,
            average_method=AVERAGE_METHOD,
        ),
        lower=-1.0,
        upper=1.0,
    )
    homogeneity = _validate_score(
        "Homogeneity",
        homogeneity_score(true_tokens, predicted_tokens),
        lower=0.0,
        upper=1.0,
    )
    return _EvaluationSnapshot(
        analysis_path=resolved_analysis,
        reference_h5ad_path=resolved_reference,
        label_key=normalized_label_key,
        cluster_key=normalized_cluster_key,
        analysis_stage=str(stage),
        n_cells=len(analysis_ids),
        n_reference_classes=n_reference_classes,
        n_predicted_clusters=n_predicted_clusters,
        nmi=nmi,
        ari=ari,
        ami=ami,
        homogeneity=homogeneity,
        cell_order_sha256=_cell_order_digest(analysis_ids),
        reference_labels_sha256=_ordered_values_digest(true_tokens),
        predicted_labels_sha256=_ordered_values_digest(predicted_tokens),
        analysis_provenance_sha256=analysis_provenance_sha256,
        software_versions=_software_versions(),
    )


def _report_from_snapshot(snapshot: _EvaluationSnapshot) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "artifact_type": EVALUATION_ARTIFACT_TYPE,
        "status": "success",
        "inputs": {
            "analysis_path": str(snapshot.analysis_path),
            "reference_h5ad_path": str(snapshot.reference_h5ad_path),
            "label_key": snapshot.label_key,
            "cluster_key": snapshot.cluster_key,
        },
        "counts": {
            "n_cells": snapshot.n_cells,
            "n_reference_classes": snapshot.n_reference_classes,
            "n_predicted_clusters": snapshot.n_predicted_clusters,
        },
        "metrics": {
            "nmi": snapshot.nmi,
            "ari": snapshot.ari,
            "ami": snapshot.ami,
            "homogeneity": snapshot.homogeneity,
        },
        "metric_backend": {
            "name": METRIC_BACKEND,
            "version": snapshot.software_versions["scikit_learn"],
            "average_method": AVERAGE_METHOD,
        },
        "validation": {
            "finite": True,
            "cell_order_preserved": True,
        },
        "provenance": {
            "analysis_stage": snapshot.analysis_stage,
            "cell_order_sha256": snapshot.cell_order_sha256,
            "reference_labels_sha256": snapshot.reference_labels_sha256,
            "predicted_labels_sha256": snapshot.predicted_labels_sha256,
            "analysis_provenance_sha256": snapshot.analysis_provenance_sha256,
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


def _validate_report_structure(report: object) -> dict[str, object]:
    if not isinstance(report, dict):
        raise ValueError("Evaluation report must contain one JSON object.")
    expected_root = {
        "schema_version",
        "artifact_type",
        "status",
        "inputs",
        "counts",
        "metrics",
        "metric_backend",
        "validation",
        "provenance",
        "software_versions",
    }
    if set(report) != expected_root:
        raise ValueError("Evaluation report has an invalid root schema.")
    if report["schema_version"] != EVALUATION_REPORT_SCHEMA_VERSION:
        raise ValueError("Evaluation report uses an unsupported schema version.")
    if report["artifact_type"] != EVALUATION_ARTIFACT_TYPE:
        raise ValueError("Evaluation report has an invalid artifact type.")
    if report["status"] != "success":
        raise ValueError("Evaluation report status must be 'success'.")
    nested_keys = {
        "inputs": {
            "analysis_path",
            "reference_h5ad_path",
            "label_key",
            "cluster_key",
        },
        "counts": {"n_cells", "n_reference_classes", "n_predicted_clusters"},
        "metrics": {"nmi", "ari", "ami", "homogeneity"},
        "metric_backend": {"name", "version", "average_method"},
        "validation": {"finite", "cell_order_preserved"},
        "provenance": {
            "analysis_stage",
            "cell_order_sha256",
            "reference_labels_sha256",
            "predicted_labels_sha256",
            "analysis_provenance_sha256",
        },
    }
    for name, keys in nested_keys.items():
        value = report[name]
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError(f"Evaluation report section {name!r} is invalid.")
    if not isinstance(report["software_versions"], dict):
        raise ValueError("Evaluation report software versions are invalid.")
    inputs = report["inputs"]
    counts = report["counts"]
    metrics = report["metrics"]
    backend = report["metric_backend"]
    validation = report["validation"]
    provenance = report["provenance"]
    assert isinstance(inputs, dict)
    assert isinstance(counts, dict)
    assert isinstance(metrics, dict)
    assert isinstance(backend, dict)
    assert isinstance(validation, dict)
    assert isinstance(provenance, dict)
    if not all(isinstance(inputs[key], str) and inputs[key] for key in inputs):
        raise ValueError("Evaluation report inputs are invalid.")
    if not all(
        not isinstance(counts[key], bool) and isinstance(counts[key], int)
        for key in counts
    ):
        raise ValueError("Evaluation report counts are invalid.")
    if (
        counts["n_cells"] <= 0
        or counts["n_reference_classes"] < 2
        or counts["n_predicted_clusters"] < 1
    ):
        raise ValueError("Evaluation report counts violate scientific constraints.")
    for metric_name, lower in (("nmi", 0.0), ("ari", -1.0), ("ami", -1.0), ("homogeneity", 0.0)):
        _validate_score(metric_name, metrics[metric_name], lower=lower, upper=1.0)
    if backend.get("name") != METRIC_BACKEND or backend.get("average_method") != AVERAGE_METHOD:
        raise ValueError("Evaluation report metric backend is invalid.")
    if not isinstance(backend.get("version"), str) or not backend["version"]:
        raise ValueError("Evaluation report metric backend version is invalid.")
    if validation != {"finite": True, "cell_order_preserved": True}:
        raise ValueError("Evaluation report validation flags are invalid.")
    if provenance.get("analysis_stage") not in {"clustering", "umap"}:
        raise ValueError("Evaluation report analysis stage is invalid.")
    for digest_name in (
        "cell_order_sha256",
        "reference_labels_sha256",
        "predicted_labels_sha256",
        "analysis_provenance_sha256",
    ):
        digest = provenance.get(digest_name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Evaluation report provenance digest is invalid.")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError("Evaluation report provenance digest is invalid.") from exc
    if not report["software_versions"] or not all(
        isinstance(key, str)
        and isinstance(value, str)
        and key
        and value
        for key, value in report["software_versions"].items()
    ):
        raise ValueError("Evaluation report software versions are invalid.")
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation report is not strict JSON-safe data.") from exc
    return report


def _load_evaluation_report(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except Exception as exc:
        raise ValueError(f"Unable to read clustering-evaluation report: {path}") from exc
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
            json.dump(
                report,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        loaded = _load_evaluation_report(temporary_path)
        if loaded != dict(report):
            raise ValueError("Written evaluation report failed exact validation.")
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


def evaluate_cell_clustering(
    analysis_path: str | Path,
    reference_h5ad_path: str | Path,
    label_key: str,
    output_dir: str | Path,
    *,
    cluster_key: str = "leiden",
    overwrite: bool = False,
) -> CellClusteringEvaluationToolResult:
    """Evaluate fixed clustering labels against ordered reference annotations."""

    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    resolved_analysis = _resolve_h5ad(analysis_path, "analysis_path")
    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = (
        resolved_output_dir / f"{resolved_analysis.stem}.clustering_metrics.json"
    )
    _ensure_report_available(output_path, overwrite=overwrite)
    snapshot = _evaluate_sources(
        resolved_analysis,
        reference_h5ad_path,
        label_key,
        cluster_key,
    )
    report = _report_from_snapshot(snapshot)
    _atomic_write_report(report, output_path, overwrite=overwrite)
    return {
        "status": "success",
        "analysis_path": str(snapshot.analysis_path),
        "reference_h5ad_path": str(snapshot.reference_h5ad_path),
        "report_path": str(output_path),
        "label_key": snapshot.label_key,
        "cluster_key": snapshot.cluster_key,
        "n_cells": snapshot.n_cells,
        "n_reference_classes": snapshot.n_reference_classes,
        "n_predicted_clusters": snapshot.n_predicted_clusters,
        "nmi": snapshot.nmi,
        "ari": snapshot.ari,
        "ami": snapshot.ami,
        "homogeneity": snapshot.homogeneity,
        "finite": True,
        "cell_order_preserved": True,
        "metric_backend": METRIC_BACKEND,
        "average_method": AVERAGE_METHOD,
        "report_schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "software_versions": dict(snapshot.software_versions),
    }


__all__ = [
    "CellClusteringEvaluationToolResult",
    "evaluate_cell_clustering",
]
