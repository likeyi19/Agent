"""Deterministic reference-to-query label transfer in EpiZoo embedding space."""

from __future__ import annotations

from collections import Counter
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
from sklearn.metrics import pairwise_distances_chunked

from agent.tools.models.epizoo import MODEL_CONFIG

from .embedding_analysis import (
    EPIZOO_EMBEDDING_DIM,
    _load_cell_ids,
    _load_embeddings,
)


LABEL_TRANSFER_PROVENANCE_KEY = "agent_milestone6_label_transfer"
LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION = 1
LABEL_TRANSFER_ARTIFACT_TYPE = "agent.epizoo-cell-label-transfer"
LABEL_TRANSFER_STAGE = "label_transfer"
LABEL_TRANSFER_BACKEND = "scikit-learn exact pairwise distances"
LABEL_TRANSFER_VOTING_METHOD = "uniform_plurality"
EMBEDDING_DIGEST_SCHEMA = "agent.epizoo-embedding-content.v1"
MODEL_CONFIG_DIGEST_SCHEMA = "agent.epizoo-model-config.v1"
_PAIRWISE_WORKING_MEMORY_MIB = 64


class CellLabelTransferToolResult(TypedDict):
    """Lightweight result for a persisted cell-label-transfer artifact."""

    status: Literal["success"]
    annotation_path: str
    annotation_sha256: str
    reference_embedding_path: str
    reference_cell_ids_path: str
    reference_h5ad_path: str
    query_embedding_path: str
    query_cell_ids_path: str
    query_h5ad_path: str
    checkpoint_path: str
    reference_label_key: str
    n_reference_cells: int
    n_query_cells: int
    n_reference_classes: int
    assigned_count: int
    unassigned_count: int
    assignment_rate: float
    embedding_dim: int
    embedding_dtype: str
    n_neighbors: int
    metric: str
    voting_method: str
    min_confidence: float
    backend: str
    species: str
    species_compatible: bool
    checkpoint_compatible: bool
    cell_order_preserved: bool
    finite: bool
    reference_embedding_sha256: str
    query_embedding_sha256: str
    reference_cell_ids_sha256: str
    query_cell_ids_sha256: str
    reference_labels_sha256: str
    model_config_sha256: str
    artifact_schema_version: int
    software_versions: dict[str, str]


@dataclass(frozen=True)
class _TransferSources:
    reference_embedding_path: Path
    reference_cell_ids_path: Path
    reference_h5ad_path: Path
    query_embedding_path: Path
    query_cell_ids_path: Path
    query_h5ad_path: Path
    checkpoint_path: Path
    reference_label_key: str
    reference_embeddings: np.ndarray
    query_embeddings: np.ndarray
    reference_ids: tuple[str, ...]
    query_ids: tuple[str, ...]
    reference_labels: tuple[str, ...]
    reference_label_order: tuple[str, ...]
    species: str
    n_neighbors: int
    metric: str
    min_confidence: float
    reference_embedding_sha256: str
    query_embedding_sha256: str
    reference_cell_ids_sha256: str
    query_cell_ids_sha256: str
    reference_labels_sha256: str
    model_config_sha256: str
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


def _resolve_file(value: str | Path, name: str, *, suffix: str | None = None) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"`{name}` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    if suffix is not None and path.suffix.casefold() != suffix:
        raise ValueError(f"`{name}` must identify a {suffix} file: {path}")
    return path


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError("`output_dir` must be a string or pathlib.Path.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Label-transfer output path is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_output_available(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Label-transfer annotation artifact already exists: {path}. "
            "Use overwrite=True to replace it."
        )


def _close_adata(adata: ad.AnnData) -> None:
    file_manager = getattr(adata, "file", None)
    if file_manager is not None:
        file_manager.close()


def _read_backed_h5ad(path: Path, source: str) -> ad.AnnData:
    try:
        return ad.read_h5ad(path, backed="r")
    except Exception as exc:
        raise ValueError(f"Unable to read {source} AnnData file: {path}") from exc


def _ordered_values_digest(values: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _embedding_content_digest(array: np.ndarray, *, rows_per_chunk: int = 8192) -> str:
    if array.ndim != 2 or array.dtype != np.dtype(np.float32):
        raise ValueError("Embedding digest requires a two-dimensional float32 array.")
    header = json.dumps(
        {
            "schema": EMBEDDING_DIGEST_SCHEMA,
            "shape": [int(array.shape[0]), int(array.shape[1])],
            "dtype": "float32-le",
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    for start in range(0, array.shape[0], rows_per_chunk):
        chunk = np.asarray(array[start : start + rows_per_chunk], dtype="<f4")
        digest.update(np.ascontiguousarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def _model_config_digest() -> str:
    payload = json.dumps(
        {"schema": MODEL_CONFIG_DIGEST_SCHEMA, "config": dict(MODEL_CONFIG)},
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"Unable to hash scientific artifact: {path}") from exc
    return digest.hexdigest()


def _strict_reference_labels(series: pd.Series, key: str) -> tuple[str, ...]:
    if len(series) <= 0:
        raise ValueError("Reference labels must contain at least one value.")
    if bool(series.isna().any()):
        raise ValueError(f"Reference label column {key!r} contains missing values.")
    labels: list[str] = []
    for value in series.tolist():
        if not isinstance(value, str):
            raise ValueError(
                f"Reference label column {key!r} must contain only string labels."
            )
        if not value.strip():
            raise ValueError(f"Reference label column {key!r} contains blank labels.")
        if value != value.strip():
            raise ValueError(
                f"Reference label column {key!r} contains leading or trailing whitespace."
            )
        labels.append(value)
    if len(set(labels)) < 2:
        raise ValueError("Reference labels must contain at least two distinct classes.")
    return tuple(labels)


def _validate_species(value: object, name: str) -> str:
    if value not in {"human", "mouse"}:
        raise ValueError(f"`{name}` must be 'human' or 'mouse'.")
    return str(value)


def _validate_parameters(
    n_neighbors: object, metric: object, min_confidence: object, overwrite: object
) -> tuple[int, str, float]:
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
        raise TypeError("`n_neighbors` must be an integer and not a boolean.")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("`metric` must be 'euclidean' or 'cosine'.")
    if isinstance(min_confidence, bool) or not isinstance(
        min_confidence, (int, float)
    ):
        raise TypeError("`min_confidence` must be a finite number in [0, 1].")
    confidence = float(min_confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("`min_confidence` must be a finite number in [0, 1].")
    if not isinstance(overwrite, bool):
        raise TypeError("`overwrite` must be a boolean.")
    return n_neighbors, str(metric), confidence


def _prepare_sources(
    reference_embedding_path: str | Path,
    reference_cell_ids_path: str | Path,
    reference_h5ad_path: str | Path,
    reference_label_key: str,
    query_embedding_path: str | Path,
    query_cell_ids_path: str | Path,
    query_h5ad_path: str | Path,
    *,
    reference_species: object,
    query_species: object,
    reference_checkpoint_path: str | Path,
    query_checkpoint_path: str | Path,
    n_neighbors: object,
    metric: object,
    min_confidence: object,
    overwrite: object = False,
) -> _TransferSources:
    normalized_k, normalized_metric, normalized_confidence = _validate_parameters(
        n_neighbors, metric, min_confidence, overwrite
    )
    if not isinstance(reference_label_key, str) or not reference_label_key.strip():
        raise ValueError("`reference_label_key` must be a nonempty string.")
    if reference_label_key != reference_label_key.strip():
        raise ValueError("`reference_label_key` must not contain surrounding whitespace.")
    normalized_reference_species = _validate_species(
        reference_species, "reference_species"
    )
    normalized_query_species = _validate_species(query_species, "query_species")
    if normalized_reference_species != normalized_query_species:
        raise ValueError("Reference and query species must match exactly.")

    reference_embedding = _resolve_file(
        reference_embedding_path, "reference_embedding_path", suffix=".npy"
    )
    query_embedding = _resolve_file(
        query_embedding_path, "query_embedding_path", suffix=".npy"
    )
    reference_ids_path = _resolve_file(
        reference_cell_ids_path, "reference_cell_ids_path", suffix=".txt"
    )
    query_ids_path = _resolve_file(
        query_cell_ids_path, "query_cell_ids_path", suffix=".txt"
    )
    reference_h5ad = _resolve_file(
        reference_h5ad_path, "reference_h5ad_path", suffix=".h5ad"
    )
    query_h5ad = _resolve_file(query_h5ad_path, "query_h5ad_path", suffix=".h5ad")
    reference_checkpoint = _resolve_file(
        reference_checkpoint_path, "reference_checkpoint_path"
    )
    query_checkpoint = _resolve_file(query_checkpoint_path, "query_checkpoint_path")

    if reference_h5ad == query_h5ad:
        raise ValueError("Reference and query raw h5ad paths must differ.")
    if reference_embedding == query_embedding:
        raise ValueError("Reference and query embedding paths must differ.")
    if reference_ids_path == query_ids_path:
        raise ValueError("Reference and query cell-ID sidecar paths must differ.")
    if reference_checkpoint != query_checkpoint:
        raise ValueError("Reference and query checkpoint paths must match exactly.")

    reference_embeddings = _load_embeddings(reference_embedding)
    query_embeddings = _load_embeddings(query_embedding)
    reference_ids = _load_cell_ids(reference_ids_path)
    query_ids = _load_cell_ids(query_ids_path)
    if len(reference_ids) != reference_embeddings.shape[0]:
        raise ValueError("Reference cell-ID count must equal reference embedding rows.")
    if len(query_ids) != query_embeddings.shape[0]:
        raise ValueError("Query cell-ID count must equal query embedding rows.")
    if not 1 <= normalized_k <= reference_embeddings.shape[0]:
        raise ValueError(
            "`n_neighbors` must satisfy 1 <= n_neighbors <= n_reference_cells."
        )
    if normalized_metric == "cosine":
        if np.any(np.linalg.norm(reference_embeddings, axis=1) == 0):
            raise ValueError("Cosine distance does not accept zero-norm reference embeddings.")
        if np.any(np.linalg.norm(query_embeddings, axis=1) == 0):
            raise ValueError("Cosine distance does not accept zero-norm query embeddings.")

    # Deliberately do not infer species from or validate ``n_vars`` here. These
    # raw files provide ordered IDs and reference labels only; the accepted
    # Agent embedding producers have already validated their feature spaces.
    reference_adata = _read_backed_h5ad(reference_h5ad, "reference")
    try:
        reference_adata_ids = tuple(str(value) for value in reference_adata.obs_names)
        if reference_adata_ids != reference_ids:
            raise ValueError(
                "Reference cell-ID sidecar must exactly match reference AnnData obs_names."
            )
        if reference_label_key not in reference_adata.obs:
            raise ValueError(
                f"Reference AnnData lacks label key {reference_label_key!r}."
            )
        reference_labels = _strict_reference_labels(
            reference_adata.obs[reference_label_key].copy(), reference_label_key
        )
    finally:
        _close_adata(reference_adata)

    query_adata = _read_backed_h5ad(query_h5ad, "query")
    try:
        query_adata_ids = tuple(str(value) for value in query_adata.obs_names)
        if query_adata_ids != query_ids:
            raise ValueError(
                "Query cell-ID sidecar must exactly match query AnnData obs_names."
            )
    finally:
        _close_adata(query_adata)

    reference_embedding_sha256 = _embedding_content_digest(reference_embeddings)
    query_embedding_sha256 = _embedding_content_digest(query_embeddings)
    reference_cell_ids_sha256 = _ordered_values_digest(
        reference_ids, domain="agent.cell-ids.v1"
    )
    query_cell_ids_sha256 = _ordered_values_digest(
        query_ids, domain="agent.cell-ids.v1"
    )
    if (
        reference_embedding_sha256 == query_embedding_sha256
        and reference_cell_ids_sha256 == query_cell_ids_sha256
    ):
        raise ValueError(
            "Reference and query embedding plus ordered-ID digests indicate self-transfer."
        )
    label_order = tuple(dict.fromkeys(reference_labels))
    return _TransferSources(
        reference_embedding,
        reference_ids_path,
        reference_h5ad,
        query_embedding,
        query_ids_path,
        query_h5ad,
        reference_checkpoint,
        reference_label_key,
        reference_embeddings,
        query_embeddings,
        reference_ids,
        query_ids,
        reference_labels,
        label_order,
        normalized_reference_species,
        normalized_k,
        normalized_metric,
        normalized_confidence,
        reference_embedding_sha256,
        query_embedding_sha256,
        reference_cell_ids_sha256,
        query_cell_ids_sha256,
        _ordered_values_digest(
            reference_labels, domain="agent.reference-labels.v1"
        ),
        _model_config_digest(),
        _software_versions(),
    )


def _deterministic_neighbor_indices(distances: np.ndarray, k: int) -> np.ndarray:
    if distances.ndim != 1 or not np.isfinite(distances).all():
        raise RuntimeError("Nearest-neighbor backend produced invalid distances.")
    if k == distances.shape[0]:
        selected = np.arange(distances.shape[0], dtype=np.int64)
    else:
        boundary = np.partition(distances, k - 1)[k - 1]
        closer = np.flatnonzero(distances < boundary)
        tied = np.flatnonzero(distances == boundary)
        remaining = k - closer.shape[0]
        if remaining < 0 or tied.shape[0] < remaining:
            raise RuntimeError("Unable to resolve deterministic nearest-neighbor boundary.")
        selected = np.concatenate((closer, tied[:remaining])).astype(
            np.int64, copy=False
        )
    order = np.lexsort((selected, distances[selected]))
    return selected[order]


def _transfer_predictions(
    sources: _TransferSources,
) -> tuple[tuple[str | None, ...], np.ndarray, tuple[str, ...]]:
    predictions: list[str | None] = []
    confidences: list[float] = []
    statuses: list[str] = []
    for distance_chunk in pairwise_distances_chunked(
        sources.query_embeddings,
        sources.reference_embeddings,
        metric=sources.metric,
        n_jobs=1,
        working_memory=_PAIRWISE_WORKING_MEMORY_MIB,
    ):
        for row in np.asarray(distance_chunk):
            neighbors = _deterministic_neighbor_indices(row, sources.n_neighbors)
            votes = Counter(sources.reference_labels[index] for index in neighbors)
            top_count = max(votes.values())
            winners = tuple(label for label, count in votes.items() if count == top_count)
            confidence = top_count / sources.n_neighbors
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise RuntimeError("Label-transfer confidence is invalid.")
            if len(winners) == 1 and confidence >= sources.min_confidence:
                predictions.append(winners[0])
                statuses.append("assigned")
            else:
                predictions.append(None)
                statuses.append("unassigned")
            confidences.append(confidence)
    if len(predictions) != len(sources.query_ids):
        raise RuntimeError("Nearest-neighbor backend returned an invalid query count.")
    return tuple(predictions), np.asarray(confidences, dtype=np.float64), tuple(statuses)


def _provenance(
    sources: _TransferSources, *, assigned_count: int, unassigned_count: int
) -> dict[str, object]:
    return {
        "schema_version": LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": LABEL_TRANSFER_ARTIFACT_TYPE,
        "stage": LABEL_TRANSFER_STAGE,
        "inputs": {
            "reference_embedding_path": str(sources.reference_embedding_path),
            "reference_cell_ids_path": str(sources.reference_cell_ids_path),
            "reference_h5ad_path": str(sources.reference_h5ad_path),
            "query_embedding_path": str(sources.query_embedding_path),
            "query_cell_ids_path": str(sources.query_cell_ids_path),
            "query_h5ad_path": str(sources.query_h5ad_path),
            "checkpoint_path": str(sources.checkpoint_path),
            "reference_label_key": sources.reference_label_key,
        },
        "compatibility": {
            "species": sources.species,
            "species_compatible": True,
            "checkpoint_compatible": True,
            "model_config_sha256": sources.model_config_sha256,
        },
        "digests": {
            "reference_embedding_sha256": sources.reference_embedding_sha256,
            "query_embedding_sha256": sources.query_embedding_sha256,
            "reference_cell_ids_sha256": sources.reference_cell_ids_sha256,
            "query_cell_ids_sha256": sources.query_cell_ids_sha256,
            "reference_labels_sha256": sources.reference_labels_sha256,
        },
        "parameters": {
            "embedding_dim": EPIZOO_EMBEDDING_DIM,
            "embedding_dtype": "float32",
            "n_neighbors": sources.n_neighbors,
            "metric": sources.metric,
            "voting_method": LABEL_TRANSFER_VOTING_METHOD,
            "min_confidence": sources.min_confidence,
            "working_memory_mib": _PAIRWISE_WORKING_MEMORY_MIB,
        },
        "backend": {
            "name": LABEL_TRANSFER_BACKEND,
            "version": sources.software_versions["scikit_learn"],
            "n_jobs": 1,
            "exact": True,
        },
        "counts": {
            "n_reference_cells": len(sources.reference_ids),
            "n_query_cells": len(sources.query_ids),
            "n_reference_classes": len(sources.reference_label_order),
            "assigned_count": assigned_count,
            "unassigned_count": unassigned_count,
        },
        "software_versions": dict(sources.software_versions),
    }


def _validate_annotation_artifact(
    artifact: ad.AnnData, sources: _TransferSources
) -> tuple[int, int]:
    if artifact.n_obs != len(sources.query_ids) or artifact.n_vars != 0:
        raise ValueError("Label-transfer artifact has invalid dimensions.")
    if artifact.X is not None:
        raise ValueError("Label-transfer artifact must not contain an X matrix.")
    artifact_ids = tuple(str(value) for value in artifact.obs_names)
    if artifact_ids != sources.query_ids:
        raise ValueError("Label-transfer artifact changed ordered query identifiers.")
    required_columns = {
        "predicted_label",
        "prediction_confidence",
        "prediction_status",
    }
    if set(artifact.obs.columns) != required_columns:
        raise ValueError("Label-transfer artifact has invalid observation columns.")
    labels = artifact.obs["predicted_label"]
    confidences = np.asarray(artifact.obs["prediction_confidence"], dtype=float)
    statuses = tuple(str(value) for value in artifact.obs["prediction_status"])
    if confidences.shape != (artifact.n_obs,) or not np.isfinite(confidences).all():
        raise ValueError("Prediction confidences must be finite.")
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise ValueError("Prediction confidences must lie in [0, 1].")
    vocabulary = set(sources.reference_label_order)
    assigned_count = 0
    for index, status in enumerate(statuses):
        value = labels.iloc[index]
        if status == "assigned":
            if pd.isna(value) or not isinstance(value, str) or value not in vocabulary:
                raise ValueError("Assigned predictions must contain a reference label.")
            assigned_count += 1
        elif status == "unassigned":
            if not pd.isna(value):
                raise ValueError("Unassigned predictions must have a missing label.")
        else:
            raise ValueError("Prediction status must be assigned or unassigned.")
    unassigned_count = artifact.n_obs - assigned_count
    expected = _provenance(
        sources,
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
    )
    actual = artifact.uns.get(LABEL_TRANSFER_PROVENANCE_KEY)
    if not isinstance(actual, Mapping) or dict(actual) != expected:
        raise ValueError("Label-transfer artifact provenance is inconsistent.")
    return assigned_count, unassigned_count


def _read_annotation(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path)
    except Exception as exc:
        raise ValueError(f"Unable to read label-transfer annotation artifact: {path}") from exc


def _atomic_write_annotation(
    artifact: ad.AnnData,
    output_path: Path,
    sources: _TransferSources,
    *,
    overwrite: bool,
) -> None:
    _ensure_output_available(output_path, overwrite=overwrite)
    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.h5ad",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        artifact.write_h5ad(temporary_path)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        written = _read_annotation(temporary_path)
        try:
            _validate_annotation_artifact(written, sources)
        finally:
            _close_adata(written)
        _ensure_output_available(output_path, overwrite=overwrite)
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


def transfer_cell_labels(
    reference_embedding_path: str | Path,
    reference_cell_ids_path: str | Path,
    reference_h5ad_path: str | Path,
    reference_label_key: str,
    query_embedding_path: str | Path,
    query_cell_ids_path: str | Path,
    query_h5ad_path: str | Path,
    output_dir: str | Path,
    *,
    reference_species: Literal["human", "mouse"],
    query_species: Literal["human", "mouse"],
    reference_checkpoint_path: str | Path,
    query_checkpoint_path: str | Path,
    n_neighbors: int = 20,
    metric: Literal["euclidean", "cosine"] = "euclidean",
    min_confidence: float = 0.0,
    overwrite: bool = False,
) -> CellLabelTransferToolResult:
    """Transfer reference labels to query cells using exact EpiZoo-space kNN."""

    _validate_parameters(n_neighbors, metric, min_confidence, overwrite)
    resolved_query_h5ad = _resolve_file(
        query_h5ad_path, "query_h5ad_path", suffix=".h5ad"
    )
    resolved_output_dir = _resolve_output_dir(output_dir)
    output_path = resolved_output_dir / f"{resolved_query_h5ad.stem}.label_transfer.h5ad"
    _ensure_output_available(output_path, overwrite=overwrite)

    sources = _prepare_sources(
        reference_embedding_path,
        reference_cell_ids_path,
        reference_h5ad_path,
        reference_label_key,
        query_embedding_path,
        query_cell_ids_path,
        resolved_query_h5ad,
        reference_species=reference_species,
        query_species=query_species,
        reference_checkpoint_path=reference_checkpoint_path,
        query_checkpoint_path=query_checkpoint_path,
        n_neighbors=n_neighbors,
        metric=metric,
        min_confidence=min_confidence,
        overwrite=overwrite,
    )
    predictions, confidences, statuses = _transfer_predictions(sources)
    assigned_count = sum(status == "assigned" for status in statuses)
    unassigned_count = len(statuses) - assigned_count
    obs = pd.DataFrame(index=pd.Index(sources.query_ids, dtype="object"))
    obs["predicted_label"] = pd.Categorical(
        predictions, categories=sources.reference_label_order
    )
    obs["prediction_confidence"] = confidences
    obs["prediction_status"] = pd.Categorical(
        statuses, categories=("assigned", "unassigned")
    )
    artifact = ad.AnnData(obs=obs)
    artifact.uns[LABEL_TRANSFER_PROVENANCE_KEY] = _provenance(
        sources,
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
    )
    _validate_annotation_artifact(artifact, sources)
    _atomic_write_annotation(
        artifact, output_path, sources, overwrite=overwrite
    )
    annotation_sha256 = _file_sha256(output_path)
    assignment_rate = assigned_count / len(sources.query_ids)
    return {
        "status": "success",
        "annotation_path": str(output_path),
        "annotation_sha256": annotation_sha256,
        "reference_embedding_path": str(sources.reference_embedding_path),
        "reference_cell_ids_path": str(sources.reference_cell_ids_path),
        "reference_h5ad_path": str(sources.reference_h5ad_path),
        "query_embedding_path": str(sources.query_embedding_path),
        "query_cell_ids_path": str(sources.query_cell_ids_path),
        "query_h5ad_path": str(sources.query_h5ad_path),
        "checkpoint_path": str(sources.checkpoint_path),
        "reference_label_key": sources.reference_label_key,
        "n_reference_cells": len(sources.reference_ids),
        "n_query_cells": len(sources.query_ids),
        "n_reference_classes": len(sources.reference_label_order),
        "assigned_count": assigned_count,
        "unassigned_count": unassigned_count,
        "assignment_rate": float(assignment_rate),
        "embedding_dim": EPIZOO_EMBEDDING_DIM,
        "embedding_dtype": "float32",
        "n_neighbors": sources.n_neighbors,
        "metric": sources.metric,
        "voting_method": LABEL_TRANSFER_VOTING_METHOD,
        "min_confidence": sources.min_confidence,
        "backend": LABEL_TRANSFER_BACKEND,
        "species": sources.species,
        "species_compatible": True,
        "checkpoint_compatible": True,
        "cell_order_preserved": True,
        "finite": True,
        "reference_embedding_sha256": sources.reference_embedding_sha256,
        "query_embedding_sha256": sources.query_embedding_sha256,
        "reference_cell_ids_sha256": sources.reference_cell_ids_sha256,
        "query_cell_ids_sha256": sources.query_cell_ids_sha256,
        "reference_labels_sha256": sources.reference_labels_sha256,
        "model_config_sha256": sources.model_config_sha256,
        "artifact_schema_version": LABEL_TRANSFER_ARTIFACT_SCHEMA_VERSION,
        "software_versions": dict(sources.software_versions),
    }


__all__ = ["CellLabelTransferToolResult", "transfer_cell_labels"]
