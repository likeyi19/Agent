"""Replicate-aware regulatory feature-space validation and sparse pseudobulk."""

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
from scipy import sparse

from .annotation_evaluation import _read_annotation_predictions


FEATURE_SPACE_ARTIFACT_TYPE = "agent.regulatory-feature-space"
FEATURE_SPACE_SCHEMA_VERSION = 1
PSEUDOBULK_ARTIFACT_TYPE = "agent.replicate-pseudobulk"
PSEUDOBULK_SCHEMA_VERSION = 1
PSEUDOBULK_PROVENANCE_KEY = "agent_milestone8_pseudobulk"
FEATURE_SPACE_IDENTITY_DOMAIN = "agent.regulatory-feature-space-identity.v1"
PSEUDOBULK_UNIT_DOMAIN = "agent.replicate-pseudobulk-unit.v1"
MATRIX_DIGEST_DOMAIN = "agent.regulatory-sparse-matrix.v1"
_ROW_CHUNK_SIZE = 4096
_SUPPORTED_SPECIES_ASSEMBLIES = {"human": "hg38", "mouse": "mm10"}
_MATRIX_SEMANTICS = {
    "fragment_counts",
    "insertion_counts",
    "binary_accessibility",
    "normalized_continuous",
}
_COUNT_SEMANTICS = {
    "fragment_counts",
    "insertion_counts",
    "binary_accessibility",
}
_COORDINATE_SYSTEMS = {"zero_based_half_open", "one_based_closed"}


class M81ScientificError(ValueError):
    """Scientific contract failure carrying one stable sanitized code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class ScATACFeatureSpaceToolResult(TypedDict):
    status: Literal["success"]
    feature_space_path: str
    feature_space_sha256: str
    feature_space_identity_sha256: str
    input_path: str
    source_h5ad_sha256: str
    matrix_source: str
    layer_key: str | None
    matrix_semantics: str
    semantics_assertion_source: str
    pseudobulk_eligible: bool
    species: str
    genome_assembly: str
    coordinate_source: str
    coordinate_system: str | None
    n_cells: int
    n_features: int
    nnz: int
    source_dtype: str
    source_sparse_format: str
    cell_ids_sha256: str
    feature_ids_sha256: str
    matrix_sha256: str
    coordinates_sha256: str | None
    artifact_schema_version: int
    software_versions: dict[str, str]


class ReplicatePseudobulkToolResult(TypedDict):
    status: Literal["success"]
    pseudobulk_path: str
    pseudobulk_sha256: str
    feature_space_path: str
    feature_space_sha256: str
    feature_space_identity_sha256: str
    source_h5ad_path: str
    source_h5ad_sha256: str
    matrix_semantics: str
    output_value_semantics: str
    aggregation_method: str
    output_dtype: str
    group_source: str
    group_key: str
    replicate_key: str
    condition_key: str
    covariate_keys: list[str]
    n_cells: int
    n_features: int
    n_pseudobulks: int
    n_groups: int
    n_replicates: int
    n_conditions: int
    minimum_cells_per_pseudobulk: int
    maximum_cells_per_pseudobulk: int
    matrix_nnz: int
    total_sum: int
    all_cells_accounted_for: bool
    feature_order_preserved: bool
    artifact_schema_version: int
    software_versions: dict[str, str]


@dataclass(frozen=True)
class _FeatureSpaceSnapshot:
    input_path: Path
    source_h5ad_sha256: str
    matrix_source: str
    layer_key: str | None
    matrix_semantics: str
    semantics_metadata_key: str | None
    semantics_assertion_source: str
    species: str
    genome_assembly: str
    coordinate_source: str
    feature_chrom_key: str | None
    feature_start_key: str | None
    feature_end_key: str | None
    coordinate_system: str | None
    n_cells: int
    n_features: int
    nnz: int
    source_dtype: str
    source_sparse_format: str
    cell_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    chromosomes: tuple[str, ...] | None
    starts: tuple[int, ...] | None
    ends: tuple[int, ...] | None
    cell_ids_sha256: str
    feature_ids_sha256: str
    matrix_sha256: str
    coordinates_sha256: str | None
    feature_space_identity_sha256: str
    software_versions: dict[str, str]


@dataclass(frozen=True)
class _MetadataSnapshot:
    group_values: tuple[str, ...]
    replicate_values: tuple[str, ...]
    condition_values: tuple[str, ...]
    covariate_values: tuple[tuple[object, ...], ...]
    unit_keys: tuple[tuple[str, str, str], ...]
    cell_to_unit: tuple[int, ...]
    unit_ids: tuple[str, ...]
    cell_counts: tuple[int, ...]
    first_cell_indices: tuple[int, ...]
    unit_covariates: tuple[tuple[object, ...], ...]
    group_annotation_path: str | None
    group_annotation_sha256: str | None
    group_values_sha256: str
    metadata_sha256: str
    unit_assignments_sha256: str


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for key, distribution in {
        "anndata": "anndata",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
    }.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = "unavailable"
    return versions


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_digest(domain: str, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_INVALID", "Unable to hash a scientific input."
        ) from exc
    return digest.hexdigest()


def _ordered_strings_digest(values: Sequence[str], *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _strict_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise M81ScientificError(
            "INVALID_ARGUMENT", f"`{name}` must be nonblank without surrounding whitespace."
        )
    return value


def _resolve_h5ad(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise M81ScientificError("INVALID_ARGUMENT", "`input_path` must be path-like.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required input H5AD does not exist: {path}")
    if path.suffix.casefold() != ".h5ad":
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_INVALID", "The feature-space source must be an H5AD file."
        )
    return path


def _resolve_existing_json(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise M81ScientificError("INVALID_ARGUMENT", "Feature-space path must be path-like.")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required feature-space manifest does not exist: {path}")
    if path.suffix.casefold() != ".json":
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest must be JSON."
        )
    return path


def _resolve_output_dir(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise M81ScientificError("INVALID_ARGUMENT", "`output_dir` must be path-like.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise M81ScientificError("INVALID_ARGUMENT", "Output path is not a directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _close_adata(adata: ad.AnnData) -> None:
    manager = getattr(adata, "file", None)
    if manager is not None:
        manager.close()


def _read_backed(path: Path) -> ad.AnnData:
    try:
        return ad.read_h5ad(path, backed="r")
    except Exception as exc:
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_INVALID", "Unable to read the source H5AD."
        ) from exc


def _is_sparse_source(matrix: object) -> bool:
    return sparse.issparse(matrix) or isinstance(
        matrix, (ad.abc.CSRDataset, ad.abc.CSCDataset)
    )


def _sparse_format(matrix: object) -> str:
    if sparse.isspmatrix_csr(matrix) or isinstance(matrix, ad.abc.CSRDataset):
        return "csr"
    if sparse.isspmatrix_csc(matrix) or isinstance(matrix, ad.abc.CSCDataset):
        return "csc"
    raise M81ScientificError(
        "MATRIX_STORAGE_UNSUPPORTED", "M8.1 requires a CSR or CSC sparse matrix."
    )


def _source_matrix(
    adata: ad.AnnData, matrix_source: str, layer_key: str | None
) -> object:
    if matrix_source == "X":
        if layer_key is not None:
            raise M81ScientificError(
                "MATRIX_SOURCE_INVALID", "`layer_key` is prohibited for matrix_source='X'."
            )
        matrix = adata.X
    elif matrix_source == "layer":
        if layer_key is None:
            raise M81ScientificError(
                "MATRIX_SOURCE_INVALID", "`layer_key` is required for matrix_source='layer'."
            )
        if layer_key not in adata.layers:
            raise M81ScientificError(
                "MATRIX_SOURCE_INVALID", "The configured source layer is absent."
            )
        matrix = adata.layers[layer_key]
    else:
        raise M81ScientificError(
            "MATRIX_SOURCE_INVALID", "`matrix_source` must be 'X' or 'layer'."
        )
    if matrix is None:
        raise M81ScientificError("MATRIX_SOURCE_INVALID", "Source matrix is absent.")
    if not _is_sparse_source(matrix):
        raise M81ScientificError(
            "MATRIX_STORAGE_UNSUPPORTED", "M8.1 v1 accepts sparse source matrices only."
        )
    return matrix


def _canonical_integer_chunk(matrix: object, start: int, stop: int, semantics: str) -> sparse.csr_matrix:
    try:
        chunk = matrix[start:stop]
    except Exception as exc:
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_INVALID", "Unable to read a source matrix chunk."
        ) from exc
    if not sparse.issparse(chunk):
        raise M81ScientificError(
            "MATRIX_STORAGE_UNSUPPORTED", "Sparse source slicing produced a dense value."
        )
    canonical = chunk.tocsr(copy=True)
    canonical.sum_duplicates()
    canonical.sort_indices()
    canonical.eliminate_zeros()
    data = canonical.data
    if data.dtype.kind not in {"b", "i", "u", "f"}:
        raise M81ScientificError(
            "MATRIX_VALUES_INVALID", "Source matrix values must be real integer-valued numbers."
        )
    if data.size:
        if not np.isfinite(data).all() or np.any(data < 0):
            raise M81ScientificError(
                "MATRIX_VALUES_INVALID", "Source matrix values must be finite and nonnegative."
            )
        if data.dtype.kind == "f" and np.any(data != np.floor(data)):
            raise M81ScientificError(
                "MATRIX_VALUES_INVALID", "Source count values must be exactly integer-valued."
            )
        if np.any(data > np.iinfo(np.int64).max):
            raise M81ScientificError(
                "INTEGER_SUM_OVERFLOW", "Source values exceed the supported int64 range."
            )
    canonical.data = data.astype(np.int64, copy=False)
    if semantics == "binary_accessibility" and canonical.data.size and np.any(
        canonical.data != 1
    ):
        raise M81ScientificError(
            "MATRIX_VALUES_INVALID", "Binary accessibility must contain only logical 0 or 1 values."
        )
    return canonical


def _matrix_digest_and_nnz(matrix: object, n_rows: int, n_cols: int, semantics: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(MATRIX_DIGEST_DOMAIN.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        _canonical_json_bytes(
            {"shape": [n_rows, n_cols], "dtype": "int64", "semantics": semantics}
        )
    )
    digest.update(b"\n")
    nnz = 0
    for start in range(0, n_rows, _ROW_CHUNK_SIZE):
        chunk = _canonical_integer_chunk(
            matrix, start, min(start + _ROW_CHUNK_SIZE, n_rows), semantics
        )
        nnz += int(chunk.nnz)
        for local_row in range(chunk.shape[0]):
            left, right = chunk.indptr[local_row : local_row + 2]
            columns = np.asarray(chunk.indices[left:right], dtype=">u8")
            values = np.asarray(chunk.data[left:right], dtype=">i8")
            digest.update(int(right - left).to_bytes(8, "big"))
            digest.update(columns.tobytes())
            digest.update(values.tobytes())
    return digest.hexdigest(), nnz


def _strict_ids(values: Sequence[object], source: str) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)
    if not identifiers or any(not value for value in identifiers):
        raise M81ScientificError(
            f"{source.upper()}_IDENTIFIERS_INVALID", f"{source} identifiers must be nonempty."
        )
    if len(set(identifiers)) != len(identifiers):
        raise M81ScientificError(
            f"{source.upper()}_IDENTIFIERS_INVALID", f"{source} identifiers must be unique."
        )
    return identifiers


def _strict_coordinate_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise M81ScientificError(
            "FEATURE_COORDINATES_INVALID", f"Coordinate {name} values must be integers."
        )
    return int(value)


def _coordinates(
    adata: ad.AnnData,
    *,
    coordinate_source: str,
    feature_chrom_key: str | None,
    feature_start_key: str | None,
    feature_end_key: str | None,
    coordinate_system: str | None,
) -> tuple[tuple[str, ...] | None, tuple[int, ...] | None, tuple[int, ...] | None, str | None]:
    keys = (feature_chrom_key, feature_start_key, feature_end_key)
    if coordinate_source == "none":
        if any(value is not None for value in (*keys, coordinate_system)):
            raise M81ScientificError(
                "FEATURE_COORDINATES_INVALID", "Coordinate arguments are prohibited when coordinates are absent."
            )
        return None, None, None, None
    if coordinate_source != "var_columns":
        raise M81ScientificError(
            "FEATURE_COORDINATES_INVALID", "`coordinate_source` must be 'none' or 'var_columns'."
        )
    if any(value is None for value in keys) or coordinate_system not in _COORDINATE_SYSTEMS:
        raise M81ScientificError(
            "FEATURE_COORDINATES_INVALID", "Explicit coordinate columns and coordinate system are required."
        )
    assert all(isinstance(value, str) for value in keys)
    normalized_keys = tuple(_strict_name(value, "coordinate key") for value in keys)
    if len(set(normalized_keys)) != 3 or any(key not in adata.var for key in normalized_keys):
        raise M81ScientificError(
            "FEATURE_COORDINATES_INVALID", "Coordinate columns must be distinct existing var columns."
        )
    chromosomes: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    seen: set[tuple[str, int, int]] = set()
    for chrom_value, start_value, end_value in zip(
        adata.var[normalized_keys[0]].tolist(),
        adata.var[normalized_keys[1]].tolist(),
        adata.var[normalized_keys[2]].tolist(),
        strict=True,
    ):
        chrom = _strict_name(chrom_value, "chromosome")
        start = _strict_coordinate_integer(start_value, "start")
        end = _strict_coordinate_integer(end_value, "end")
        minimum_start = 0 if coordinate_system == "zero_based_half_open" else 1
        if start < minimum_start or end <= start:
            raise M81ScientificError(
                "FEATURE_COORDINATES_INVALID", "Feature coordinates have invalid interval bounds."
            )
        coordinate = (chrom, start, end)
        if coordinate in seen:
            raise M81ScientificError(
                "FEATURE_COORDINATES_INVALID", "Feature coordinate triples must be unique."
            )
        seen.add(coordinate)
        chromosomes.append(chrom)
        starts.append(start)
        ends.append(end)
    digest = _domain_digest(
        "agent.regulatory-feature-coordinates.v1",
        {
            "coordinate_system": coordinate_system,
            "values": list(zip(chromosomes, starts, ends, strict=True)),
        },
    )
    return tuple(chromosomes), tuple(starts), tuple(ends), digest


def _validate_species_assembly(species: object, genome_assembly: object) -> tuple[str, str]:
    normalized_species = _strict_name(species, "species")
    normalized_assembly = _strict_name(genome_assembly, "genome_assembly")
    if _SUPPORTED_SPECIES_ASSEMBLIES.get(normalized_species) != normalized_assembly:
        raise M81ScientificError(
            "SPECIES_ASSEMBLY_INVALID",
            "M8.1 v1 supports only human/hg38 and mouse/mm10 compatibility pairs.",
        )
    return normalized_species, normalized_assembly


def _snapshot_feature_space(
    input_path: str | Path,
    *,
    matrix_source: object,
    matrix_semantics: object,
    species: object,
    genome_assembly: object,
    coordinate_source: object,
    layer_key: object = None,
    feature_chrom_key: object = None,
    feature_start_key: object = None,
    feature_end_key: object = None,
    coordinate_system: object = None,
    semantics_metadata_key: object = None,
) -> _FeatureSpaceSnapshot:
    path = _resolve_h5ad(input_path)
    source_before = _file_sha256(path)
    source_name = _strict_name(matrix_source, "matrix_source")
    semantics = _strict_name(matrix_semantics, "matrix_semantics")
    if semantics not in _MATRIX_SEMANTICS:
        raise M81ScientificError(
            "MATRIX_SEMANTICS_UNSUPPORTED", "Unknown regulatory matrix semantics."
        )
    if semantics not in _COUNT_SEMANTICS:
        raise M81ScientificError(
            "MATRIX_SEMANTICS_UNSUPPORTED", "Normalized/continuous matrices are not pseudobulk count inputs."
        )
    normalized_species, normalized_assembly = _validate_species_assembly(
        species, genome_assembly
    )
    coordinate_name = _strict_name(coordinate_source, "coordinate_source")
    normalized_layer = None if layer_key is None else _strict_name(layer_key, "layer_key")
    normalized_semantics_key = (
        None
        if semantics_metadata_key is None
        else _strict_name(semantics_metadata_key, "semantics_metadata_key")
    )
    normalized_chrom_key = (
        None if feature_chrom_key is None else _strict_name(feature_chrom_key, "feature_chrom_key")
    )
    normalized_start_key = (
        None if feature_start_key is None else _strict_name(feature_start_key, "feature_start_key")
    )
    normalized_end_key = (
        None if feature_end_key is None else _strict_name(feature_end_key, "feature_end_key")
    )
    normalized_coordinate_system = (
        None if coordinate_system is None else _strict_name(coordinate_system, "coordinate_system")
    )
    adata = _read_backed(path)
    try:
        if adata.n_obs <= 0 or adata.n_vars <= 0:
            raise M81ScientificError(
                "FEATURE_SPACE_SOURCE_INVALID", "Feature-space source must contain cells and features."
            )
        matrix = _source_matrix(adata, source_name, normalized_layer)
        sparse_format = _sparse_format(matrix)
        if tuple(matrix.shape) != (adata.n_obs, adata.n_vars):
            raise M81ScientificError(
                "FEATURE_SPACE_SOURCE_INVALID", "Source matrix shape does not match AnnData dimensions."
            )
        if semantics == "fragment_counts" and np.dtype(matrix.dtype).kind == "b":
            raise M81ScientificError(
                "MATRIX_VALUES_INVALID", "Boolean matrices cannot assert fragment-count semantics."
            )
        if semantics == "insertion_counts" and np.dtype(matrix.dtype).kind == "b":
            raise M81ScientificError(
                "MATRIX_VALUES_INVALID", "Boolean matrices cannot assert insertion-count semantics."
            )
        assertion_source = "structured_request"
        if normalized_semantics_key is not None:
            if normalized_semantics_key not in adata.uns:
                raise M81ScientificError(
                    "MATRIX_SEMANTICS_UNSUPPORTED", "Configured semantics metadata is absent."
                )
            if adata.uns[normalized_semantics_key] != semantics:
                raise M81ScientificError(
                    "MATRIX_SEMANTICS_UNSUPPORTED", "Raw semantics metadata disagrees with the declaration."
                )
            assertion_source = "structured_request_and_raw_uns"
        cell_ids = _strict_ids(adata.obs_names, "cell")
        feature_ids = _strict_ids(adata.var_names, "feature")
        chromosomes, starts, ends, coordinates_sha256 = _coordinates(
            adata,
            coordinate_source=coordinate_name,
            feature_chrom_key=normalized_chrom_key,
            feature_start_key=normalized_start_key,
            feature_end_key=normalized_end_key,
            coordinate_system=normalized_coordinate_system,
        )
        matrix_sha256, nnz = _matrix_digest_and_nnz(
            matrix, adata.n_obs, adata.n_vars, semantics
        )
        source_dtype = str(matrix.dtype)
        n_cells = int(adata.n_obs)
        n_features = int(adata.n_vars)
    finally:
        _close_adata(adata)
    source_after = _file_sha256(path)
    if source_after != source_before:
        raise M81ScientificError(
            "SOURCE_CHANGED_DURING_READ", "Source H5AD changed during feature-space validation."
        )
    cell_digest = _ordered_strings_digest(
        cell_ids, domain="agent.regulatory-cell-identities.v1"
    )
    feature_digest = _ordered_strings_digest(
        feature_ids, domain="agent.regulatory-feature-identities.v1"
    )
    identity_payload = {
        "source_h5ad_sha256": source_before,
        "matrix_source": source_name,
        "layer_key": normalized_layer,
        "matrix_semantics": semantics,
        "species": normalized_species,
        "genome_assembly": normalized_assembly,
        "cell_ids_sha256": cell_digest,
        "feature_ids_sha256": feature_digest,
        "matrix_sha256": matrix_sha256,
        "coordinates_sha256": coordinates_sha256,
    }
    return _FeatureSpaceSnapshot(
        path,
        source_before,
        source_name,
        normalized_layer,
        semantics,
        normalized_semantics_key,
        assertion_source,
        normalized_species,
        normalized_assembly,
        coordinate_name,
        normalized_chrom_key,
        normalized_start_key,
        normalized_end_key,
        normalized_coordinate_system,
        n_cells,
        n_features,
        nnz,
        source_dtype,
        sparse_format,
        cell_ids,
        feature_ids,
        chromosomes,
        starts,
        ends,
        cell_digest,
        feature_digest,
        matrix_sha256,
        coordinates_sha256,
        _domain_digest(FEATURE_SPACE_IDENTITY_DOMAIN, identity_payload),
        _software_versions(),
    )


def _feature_manifest(snapshot: _FeatureSpaceSnapshot) -> dict[str, object]:
    coordinates: dict[str, object] = {
        "source": snapshot.coordinate_source,
        "available": snapshot.coordinate_source == "var_columns",
    }
    if snapshot.coordinate_source == "var_columns":
        coordinates.update(
            {
                "chrom_key": snapshot.feature_chrom_key,
                "start_key": snapshot.feature_start_key,
                "end_key": snapshot.feature_end_key,
                "coordinate_system": snapshot.coordinate_system,
                "coordinates_sha256": snapshot.coordinates_sha256,
            }
        )
    return {
        "schema_version": FEATURE_SPACE_SCHEMA_VERSION,
        "artifact_type": FEATURE_SPACE_ARTIFACT_TYPE,
        "status": "success",
        "source": {
            "input_path": str(snapshot.input_path),
            "source_h5ad_sha256": snapshot.source_h5ad_sha256,
        },
        "matrix": {
            "source": snapshot.matrix_source,
            "layer_key": snapshot.layer_key,
            "semantics": snapshot.matrix_semantics,
            "semantics_metadata_key": snapshot.semantics_metadata_key,
            "semantics_assertion_source": snapshot.semantics_assertion_source,
            "pseudobulk_eligible": True,
            "source_dtype": snapshot.source_dtype,
            "source_sparse_format": snapshot.source_sparse_format,
            "logical_dtype": "int64",
            "nnz": snapshot.nnz,
            "matrix_sha256": snapshot.matrix_sha256,
        },
        "biology": {
            "species": snapshot.species,
            "genome_assembly": snapshot.genome_assembly,
            "compatibility_allowlist_version": 1,
        },
        "dimensions": {"n_cells": snapshot.n_cells, "n_features": snapshot.n_features},
        "identities": {
            "cell_ids_sha256": snapshot.cell_ids_sha256,
            "feature_ids_sha256": snapshot.feature_ids_sha256,
            "feature_space_identity_sha256": snapshot.feature_space_identity_sha256,
        },
        "coordinates": coordinates,
        "validation": {
            "source_sparse": True,
            "finite_nonnegative_integer_values": True,
            "binary_content_validated": snapshot.matrix_semantics == "binary_accessibility",
            "feature_order_preserved": True,
            "cell_order_preserved": True,
            "feature_alignment_performed": False,
            "coordinate_inference_performed": False,
        },
        "software_versions": dict(snapshot.software_versions),
    }


def _strict_json_load(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except Exception as exc:
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest is not strict JSON."
        ) from exc
    if not isinstance(value, dict):
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest must be one object."
        )
    return value


def _load_feature_manifest(path: str | Path) -> tuple[Path, dict[str, object], str]:
    resolved = _resolve_existing_json(path)
    manifest = _strict_json_load(resolved)
    if manifest.get("schema_version") != FEATURE_SPACE_SCHEMA_VERSION or manifest.get(
        "artifact_type"
    ) != FEATURE_SPACE_ARTIFACT_TYPE:
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest identity is invalid."
        )
    expected_keys = {
        "schema_version", "artifact_type", "status", "source", "matrix", "biology",
        "dimensions", "identities", "coordinates", "validation", "software_versions",
    }
    if set(manifest) != expected_keys or manifest.get("status") != "success":
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest schema is invalid."
        )
    return resolved, manifest, _file_sha256(resolved)


def _snapshot_from_manifest(manifest: Mapping[str, object]) -> _FeatureSpaceSnapshot:
    try:
        source = manifest["source"]
        matrix = manifest["matrix"]
        biology = manifest["biology"]
        coordinates = manifest["coordinates"]
        if not all(isinstance(value, Mapping) for value in (source, matrix, biology, coordinates)):
            raise TypeError
        assert isinstance(source, Mapping) and isinstance(matrix, Mapping)
        assert isinstance(biology, Mapping) and isinstance(coordinates, Mapping)
        return _snapshot_feature_space(
            str(source["input_path"]),
            matrix_source=matrix["source"],
            layer_key=matrix.get("layer_key"),
            matrix_semantics=matrix["semantics"],
            semantics_metadata_key=matrix.get("semantics_metadata_key"),
            species=biology["species"],
            genome_assembly=biology["genome_assembly"],
            coordinate_source=coordinates["source"],
            feature_chrom_key=coordinates.get("chrom_key"),
            feature_start_key=coordinates.get("start_key"),
            feature_end_key=coordinates.get("end_key"),
            coordinate_system=coordinates.get("coordinate_system"),
        )
    except M81ScientificError:
        raise
    except Exception as exc:
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Feature-space manifest fields are invalid."
        ) from exc


def _atomic_write_json(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Scientific output already exists: {path}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _strict_json_load(temporary)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Scientific output already exists: {path}")
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.replace(temporary, path)
            temporary = None
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_scATAC_feature_space(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    matrix_source: str,
    matrix_semantics: str,
    species: str,
    genome_assembly: str,
    coordinate_source: str,
    layer_key: str | None = None,
    feature_chrom_key: str | None = None,
    feature_start_key: str | None = None,
    feature_end_key: str | None = None,
    coordinate_system: str | None = None,
    semantics_metadata_key: str | None = None,
    overwrite: bool = False,
) -> ScATACFeatureSpaceToolResult:
    """Validate and persist exact regulatory feature-space provenance."""

    if not isinstance(overwrite, bool):
        raise M81ScientificError("INVALID_ARGUMENT", "`overwrite` must be boolean.")
    snapshot = _snapshot_feature_space(
        input_path,
        matrix_source=matrix_source,
        matrix_semantics=matrix_semantics,
        species=species,
        genome_assembly=genome_assembly,
        coordinate_source=coordinate_source,
        layer_key=layer_key,
        feature_chrom_key=feature_chrom_key,
        feature_start_key=feature_start_key,
        feature_end_key=feature_end_key,
        coordinate_system=coordinate_system,
        semantics_metadata_key=semantics_metadata_key,
    )
    directory = _resolve_output_dir(output_dir)
    output_path = directory / f"{snapshot.input_path.stem}.regulatory_feature_space.json"
    payload = _canonical_json_bytes(_feature_manifest(snapshot))
    _atomic_write_json(output_path, payload, overwrite=overwrite)
    manifest_path, manifest, manifest_digest = _load_feature_manifest(output_path)
    if manifest != _feature_manifest(snapshot):
        raise M81ScientificError(
            "FEATURE_SPACE_ARTIFACT_INVALID", "Published feature-space manifest changed content."
        )
    return {
        "status": "success",
        "feature_space_path": str(manifest_path),
        "feature_space_sha256": manifest_digest,
        "feature_space_identity_sha256": snapshot.feature_space_identity_sha256,
        "input_path": str(snapshot.input_path),
        "source_h5ad_sha256": snapshot.source_h5ad_sha256,
        "matrix_source": snapshot.matrix_source,
        "layer_key": snapshot.layer_key,
        "matrix_semantics": snapshot.matrix_semantics,
        "semantics_assertion_source": snapshot.semantics_assertion_source,
        "pseudobulk_eligible": True,
        "species": snapshot.species,
        "genome_assembly": snapshot.genome_assembly,
        "coordinate_source": snapshot.coordinate_source,
        "coordinate_system": snapshot.coordinate_system,
        "n_cells": snapshot.n_cells,
        "n_features": snapshot.n_features,
        "nnz": snapshot.nnz,
        "source_dtype": snapshot.source_dtype,
        "source_sparse_format": snapshot.source_sparse_format,
        "cell_ids_sha256": snapshot.cell_ids_sha256,
        "feature_ids_sha256": snapshot.feature_ids_sha256,
        "matrix_sha256": snapshot.matrix_sha256,
        "coordinates_sha256": snapshot.coordinates_sha256,
        "artifact_schema_version": FEATURE_SPACE_SCHEMA_VERSION,
        "software_versions": dict(snapshot.software_versions),
    }


def _strict_labels(series: pd.Series, key: str) -> tuple[str, ...]:
    if bool(series.isna().any()):
        raise M81ScientificError("METADATA_VALUES_INVALID", f"Metadata {key!r} contains missing values.")
    labels: list[str] = []
    for value in series.tolist():
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise M81ScientificError(
                "METADATA_VALUES_INVALID", f"Metadata {key!r} must contain strict text labels."
            )
        labels.append(value)
    return tuple(labels)


def _canonical_covariate(value: object, key: str) -> object:
    if pd.isna(value):
        raise M81ScientificError("METADATA_VALUES_INVALID", f"Covariate {key!r} contains missing values.")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise M81ScientificError("METADATA_VALUES_INVALID", f"Covariate {key!r} is nonfinite.")
        return normalized
    if isinstance(value, str) and value.strip() and value == value.strip():
        return value
    raise M81ScientificError(
        "METADATA_VALUES_INVALID", f"Covariate {key!r} has an unsupported value."
    )


def _metadata_snapshot(
    source: _FeatureSpaceSnapshot,
    *,
    replicate_key: str,
    group_key: str,
    condition_key: str,
    group_source: str,
    group_annotation_path: str | Path | None,
    covariate_keys: tuple[str, ...],
) -> _MetadataSnapshot:
    adata = _read_backed(source.input_path)
    try:
        required_raw = {replicate_key, condition_key, *covariate_keys}
        if group_source == "raw_obs":
            required_raw.add(group_key)
        missing = sorted(required_raw.difference(adata.obs.columns))
        if missing:
            raise M81ScientificError(
                "METADATA_COLUMN_MISSING", "Required raw observation metadata is absent."
            )
        replicate_values = _strict_labels(adata.obs[replicate_key].copy(), replicate_key)
        condition_values = _strict_labels(adata.obs[condition_key].copy(), condition_key)
        annotation_path: str | None = None
        annotation_sha256: str | None = None
        if group_source == "raw_obs":
            if group_annotation_path is not None:
                raise M81ScientificError(
                    "GROUP_ANNOTATION_INVALID", "Annotation path is prohibited for raw_obs groups."
                )
            group_values = _strict_labels(adata.obs[group_key].copy(), group_key)
        elif group_source == "verified_annotation":
            if group_annotation_path is None or group_key != "predicted_label":
                raise M81ScientificError(
                    "GROUP_ANNOTATION_INVALID", "Verified groups require a Milestone 6.3 predicted_label artifact."
                )
            try:
                annotation = _read_annotation_predictions(
                    Path(group_annotation_path).expanduser().resolve()
                )
            except Exception as exc:
                raise M81ScientificError(
                    "GROUP_ANNOTATION_INVALID", "Group annotation failed Milestone 6.3 validation."
                ) from exc
            if annotation.cell_ids != source.cell_ids:
                raise M81ScientificError(
                    "CELL_IDENTITY_MISMATCH", "Annotation cell identity/order differs from raw H5AD."
                )
            if any(value is None for value in annotation.predicted_labels):
                raise M81ScientificError(
                    "GROUP_ANNOTATION_INVALID", "Every annotation cell must be assigned for pseudobulk."
                )
            group_values = tuple(str(value) for value in annotation.predicted_labels)
            annotation_path = str(annotation.path)
            annotation_sha256 = annotation.annotation_sha256
        else:
            raise M81ScientificError(
                "GROUP_ANNOTATION_INVALID", "Unsupported pseudobulk group source."
            )
        covariates = tuple(
            tuple(_canonical_covariate(value, key) for value in adata.obs[key].tolist())
            for key in covariate_keys
        )
    finally:
        _close_adata(adata)
    if not (
        len(group_values)
        == len(replicate_values)
        == len(condition_values)
        == source.n_cells
    ):
        raise M81ScientificError("CELL_IDENTITY_MISMATCH", "Metadata length differs from cell count.")
    unit_index: dict[tuple[str, str, str], int] = {}
    unit_keys: list[tuple[str, str, str]] = []
    cell_to_unit: list[int] = []
    counts: list[int] = []
    first: list[int] = []
    unit_covariates: list[tuple[object, ...]] = []
    replicate_condition_covariates: dict[tuple[str, str], tuple[object, ...]] = {}
    for index, (group, replicate, condition) in enumerate(
        zip(group_values, replicate_values, condition_values, strict=True)
    ):
        covariate_tuple = tuple(values[index] for values in covariates)
        pair = (replicate, condition)
        previous_pair = replicate_condition_covariates.setdefault(pair, covariate_tuple)
        if previous_pair != covariate_tuple:
            raise M81ScientificError(
                "COVARIATE_NOT_CONSTANT", "Covariates must be constant within replicate-condition pairs."
            )
        key = (group, replicate, condition)
        unit = unit_index.get(key)
        if unit is None:
            unit = len(unit_keys)
            unit_index[key] = unit
            unit_keys.append(key)
            counts.append(0)
            first.append(index)
            unit_covariates.append(covariate_tuple)
        counts[unit] += 1
        cell_to_unit.append(unit)
    unit_ids = tuple(
        "pb-"
        + _domain_digest(
            PSEUDOBULK_UNIT_DOMAIN,
            {
                "feature_space_identity_sha256": source.feature_space_identity_sha256,
                "group": group,
                "replicate": replicate,
                "condition": condition,
            },
        )
        for group, replicate, condition in unit_keys
    )
    if len(set(unit_ids)) != len(unit_ids):
        raise M81ScientificError("PSEUDOBULK_METADATA_MISMATCH", "Pseudobulk unit IDs collided.")
    group_digest = _ordered_strings_digest(
        group_values, domain="agent.pseudobulk-group-values.v1"
    )
    metadata_payload = {
        "group": group_values,
        "replicate": replicate_values,
        "condition": condition_values,
        "covariate_keys": covariate_keys,
        "covariates": covariates,
    }
    return _MetadataSnapshot(
        group_values,
        replicate_values,
        condition_values,
        covariates,
        tuple(unit_keys),
        tuple(cell_to_unit),
        unit_ids,
        tuple(counts),
        tuple(first),
        tuple(unit_covariates),
        annotation_path,
        annotation_sha256,
        group_digest,
        _domain_digest("agent.pseudobulk-source-metadata.v1", metadata_payload),
        _domain_digest(
            "agent.pseudobulk-unit-assignments.v1",
            {"unit_ids": unit_ids, "cell_to_unit": cell_to_unit},
        ),
    )


def _production_aggregate(source: _FeatureSpaceSnapshot, metadata: _MetadataSnapshot) -> sparse.csr_matrix:
    source_before = _file_sha256(source.input_path)
    if source_before != source.source_h5ad_sha256:
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_MISMATCH", "Source H5AD changed after feature-space validation."
        )
    adata = _read_backed(source.input_path)
    try:
        matrix = _source_matrix(adata, source.matrix_source, source.layer_key)
        aggregate = sparse.csr_matrix(
            (len(metadata.unit_keys), source.n_features), dtype=np.int64
        )
        for start in range(0, source.n_cells, _ROW_CHUNK_SIZE):
            stop = min(start + _ROW_CHUNK_SIZE, source.n_cells)
            chunk = _canonical_integer_chunk(matrix, start, stop, source.matrix_semantics)
            row_indices = np.asarray(metadata.cell_to_unit[start:stop], dtype=np.int64)
            membership = sparse.csr_matrix(
                (
                    np.ones(stop - start, dtype=np.int64),
                    (row_indices, np.arange(stop - start, dtype=np.int64)),
                ),
                shape=(len(metadata.unit_keys), stop - start),
            )
            maximum_value = int(chunk.data.max()) if chunk.data.size else 0
            maximum_members = max(metadata.cell_counts)
            if maximum_value == 0 or maximum_value <= np.iinfo(np.int64).max // maximum_members:
                partial = membership @ chunk
            else:
                row_maps: list[dict[int, int]] = [dict() for _ in metadata.unit_keys]
                for local_row in range(chunk.shape[0]):
                    target = row_maps[metadata.cell_to_unit[start + local_row]]
                    left, right = chunk.indptr[local_row : local_row + 2]
                    for column, value in zip(
                        chunk.indices[left:right], chunk.data[left:right], strict=True
                    ):
                        total = target.get(int(column), 0) + int(value)
                        if total > np.iinfo(np.int64).max:
                            raise M81ScientificError(
                                "INTEGER_SUM_OVERFLOW", "Pseudobulk integer sum overflowed."
                            )
                        target[int(column)] = total
                rows: list[int] = []
                columns: list[int] = []
                values: list[int] = []
                for row, row_map in enumerate(row_maps):
                    for column in sorted(row_map):
                        rows.append(row)
                        columns.append(column)
                        values.append(row_map[column])
                partial = sparse.csr_matrix(
                    (
                        np.asarray(values, dtype=np.int64),
                        (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
                    ),
                    shape=aggregate.shape,
                )
            combined = aggregate.astype(np.uint64) + partial.astype(np.uint64)
            if combined.data.size and np.any(combined.data > np.iinfo(np.int64).max):
                raise M81ScientificError(
                    "INTEGER_SUM_OVERFLOW", "Pseudobulk integer sum overflowed."
                )
            aggregate = combined.astype(np.int64)
        aggregate.sum_duplicates()
        aggregate.sort_indices()
        aggregate.eliminate_zeros()
        result = aggregate
    finally:
        _close_adata(adata)
    if _file_sha256(source.input_path) != source_before:
        raise M81ScientificError(
            "SOURCE_CHANGED_DURING_READ", "Source H5AD changed during pseudobulk aggregation."
        )
    return result


def _matrix_digest_from_csr(matrix: sparse.csr_matrix, semantics: str) -> str:
    digest, _ = _matrix_digest_and_nnz(matrix, matrix.shape[0], matrix.shape[1], semantics)
    return digest


def _output_value_semantics(semantics: str) -> str:
    if semantics == "binary_accessibility":
        return "accessible_cell_count"
    return semantics


def _checked_library_sizes(matrix: sparse.csr_matrix) -> tuple[int, ...]:
    maximum = np.iinfo(np.int64).max
    values: list[int] = []
    for row in range(matrix.shape[0]):
        left, right = matrix.indptr[row : row + 2]
        total = sum(int(value) for value in matrix.data[left:right])
        if total > maximum:
            raise M81ScientificError(
                "INTEGER_SUM_OVERFLOW", "Pseudobulk library size overflowed."
            )
        values.append(total)
    return tuple(values)


def _artifact_provenance(
    source: _FeatureSpaceSnapshot,
    feature_path: Path,
    feature_sha256: str,
    metadata_snapshot: _MetadataSnapshot,
    matrix: sparse.csr_matrix,
    *,
    group_source: str,
    group_key: str,
    replicate_key: str,
    condition_key: str,
    covariate_keys: tuple[str, ...],
    library_sizes: tuple[int, ...],
) -> dict[str, object]:
    coordinate: dict[str, object] = {
        "available": source.coordinate_source == "var_columns",
        "source": source.coordinate_source,
    }
    if source.coordinate_source == "var_columns":
        coordinate.update(
            {
                "coordinate_system": source.coordinate_system,
                "coordinates_sha256": source.coordinates_sha256,
            }
        )
    return {
        "schema_version": PSEUDOBULK_SCHEMA_VERSION,
        "artifact_type": PSEUDOBULK_ARTIFACT_TYPE,
        "stage": "replicate_pseudobulk",
        "source": {
            "feature_space_path": str(feature_path),
            "feature_space_sha256": feature_sha256,
            "feature_space_identity_sha256": source.feature_space_identity_sha256,
            "source_h5ad_path": str(source.input_path),
            "source_h5ad_sha256": source.source_h5ad_sha256,
            "matrix_source": source.matrix_source,
            "layer_key": source.layer_key,
        },
        "biology": {
            "species": source.species,
            "genome_assembly": source.genome_assembly,
            "matrix_semantics": source.matrix_semantics,
            "semantics_assertion_source": source.semantics_assertion_source,
            "output_value_semantics": _output_value_semantics(source.matrix_semantics),
        },
        "metadata": {
            "group_source": group_source,
            "group_key": group_key,
            "replicate_key": replicate_key,
            "condition_key": condition_key,
            "covariate_keys": list(covariate_keys),
            "covariate_columns": [f"covariate_{index:03d}" for index in range(len(covariate_keys))],
            "group_annotation_path": metadata_snapshot.group_annotation_path,
            "group_annotation_sha256": metadata_snapshot.group_annotation_sha256,
            "replicate_may_span_conditions": True,
        },
        "aggregation": {
            "unit": ["group", "replicate", "condition"],
            "method": "sum",
            "ordering": "first_occurrence_in_source_cell_order",
            "output_sparse_format": "csr",
            "output_dtype": "int64",
            "n_cells": source.n_cells,
            "n_features": source.n_features,
            "n_pseudobulks": len(metadata_snapshot.unit_keys),
            "matrix_nnz": int(matrix.nnz),
            "total_sum": int(sum(library_sizes)),
        },
        "digests": {
            "cell_ids_sha256": source.cell_ids_sha256,
            "feature_ids_sha256": source.feature_ids_sha256,
            "coordinates_sha256": source.coordinates_sha256,
            "group_values_sha256": metadata_snapshot.group_values_sha256,
            "metadata_sha256": metadata_snapshot.metadata_sha256,
            "unit_assignments_sha256": metadata_snapshot.unit_assignments_sha256,
            "pseudobulk_matrix_sha256": _matrix_digest_from_csr(matrix, _output_value_semantics(source.matrix_semantics)),
        },
        "coordinates": coordinate,
        "validation": {
            "all_cells_accounted_for": sum(metadata_snapshot.cell_counts) == source.n_cells,
            "feature_order_preserved": True,
            "normalization_performed": False,
            "cell_filtering_performed": False,
            "feature_filtering_performed": False,
            "feature_intersection_performed": False,
            "feature_reordering_performed": False,
            "feature_remapping_performed": False,
            "coordinate_inference_performed": False,
        },
        "software_versions": dict(source.software_versions),
    }


def _categorical(values: Sequence[str]) -> pd.Categorical:
    return pd.Categorical(values, categories=list(dict.fromkeys(values)), ordered=True)


def _artifact(
    source: _FeatureSpaceSnapshot,
    metadata_snapshot: _MetadataSnapshot,
    matrix: sparse.csr_matrix,
    provenance: Mapping[str, object],
    library_sizes: tuple[int, ...],
) -> ad.AnnData:
    obs_data: dict[str, object] = {
        "group": _categorical([value[0] for value in metadata_snapshot.unit_keys]),
        "replicate": _categorical([value[1] for value in metadata_snapshot.unit_keys]),
        "condition": _categorical([value[2] for value in metadata_snapshot.unit_keys]),
        "n_cells": np.asarray(metadata_snapshot.cell_counts, dtype=np.int64),
        "first_cell_index": np.asarray(metadata_snapshot.first_cell_indices, dtype=np.int64),
        "library_size": np.asarray(library_sizes, dtype=np.int64),
    }
    for index in range(len(metadata_snapshot.covariate_values)):
        values = [row[index] for row in metadata_snapshot.unit_covariates]
        column = f"covariate_{index:03d}"
        obs_data[column] = _categorical(values) if values and isinstance(values[0], str) else values
    obs = pd.DataFrame(obs_data, index=pd.Index(metadata_snapshot.unit_ids, dtype=str))
    var_data: dict[str, object] = {}
    if source.chromosomes is not None:
        var_data = {
            "chrom": _categorical(source.chromosomes),
            "start": np.asarray(source.starts, dtype=np.int64),
            "end": np.asarray(source.ends, dtype=np.int64),
        }
    var = pd.DataFrame(var_data, index=pd.Index(source.feature_ids, dtype=str))
    return ad.AnnData(X=matrix, obs=obs, var=var, uns={PSEUDOBULK_PROVENANCE_KEY: dict(provenance)})


def _atomic_write_h5ad(artifact: ad.AnnData, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Scientific output already exists: {output_path}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output_path.parent, prefix=f".{output_path.stem}.", suffix=".tmp.h5ad"
        )
        os.close(descriptor)
        temporary = Path(name)
        artifact.write_h5ad(temporary)
        try:
            validated = ad.read_h5ad(temporary)
        except Exception as exc:
            raise M81ScientificError(
                "ARTIFACT_WRITE_FAILED", "Temporary pseudobulk artifact could not be reopened."
            ) from exc
        try:
            if (
                tuple(validated.obs_names) != tuple(artifact.obs_names)
                or tuple(validated.var_names) != tuple(artifact.var_names)
                or tuple(validated.obs.columns) != tuple(artifact.obs.columns)
                or tuple(validated.var.columns) != tuple(artifact.var.columns)
                or not sparse.isspmatrix_csr(validated.X)
                or np.dtype(validated.X.dtype) != np.dtype(np.int64)
                or (validated.X != artifact.X).nnz != 0
                or set(validated.uns) != {PSEUDOBULK_PROVENANCE_KEY}
            ):
                raise M81ScientificError(
                    "ARTIFACT_WRITE_FAILED", "Temporary pseudobulk artifact failed validation."
                )
        finally:
            _close_adata(validated)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Scientific output already exists: {output_path}")
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.replace(temporary, output_path)
            temporary = None
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (FileExistsError, M81ScientificError):
        raise
    except OSError as exc:
        raise M81ScientificError("ARTIFACT_WRITE_FAILED", "Pseudobulk artifact write failed.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_replicate_pseudobulk(
    feature_space_path: str | Path,
    replicate_key: str,
    group_key: str,
    condition_key: str,
    output_dir: str | Path,
    *,
    group_source: str,
    group_annotation_path: str | Path | None = None,
    covariate_keys: list[str] | tuple[str, ...] = (),
    overwrite: bool = False,
) -> ReplicatePseudobulkToolResult:
    """Aggregate exact sparse regulatory counts by group, replicate and condition."""

    if not isinstance(overwrite, bool):
        raise M81ScientificError("INVALID_ARGUMENT", "`overwrite` must be boolean.")
    normalized_replicate = _strict_name(replicate_key, "replicate_key")
    normalized_group = _strict_name(group_key, "group_key")
    normalized_condition = _strict_name(condition_key, "condition_key")
    normalized_group_source = _strict_name(group_source, "group_source")
    if len({normalized_replicate, normalized_group, normalized_condition}) != 3:
        raise M81ScientificError("INVALID_ARGUMENT", "Group, replicate and condition keys must be distinct.")
    if not isinstance(covariate_keys, (list, tuple)):
        raise M81ScientificError("INVALID_ARGUMENT", "`covariate_keys` must be a list or tuple.")
    normalized_covariates = tuple(_strict_name(value, "covariate key") for value in covariate_keys)
    if len(set(normalized_covariates)) != len(normalized_covariates) or set(normalized_covariates).intersection(
        {normalized_replicate, normalized_group, normalized_condition}
    ):
        raise M81ScientificError("INVALID_ARGUMENT", "Metadata keys must be unique and conceptually distinct.")
    manifest_path, manifest, manifest_sha256 = _load_feature_manifest(feature_space_path)
    source = _snapshot_from_manifest(manifest)
    if manifest != _feature_manifest(source):
        raise M81ScientificError(
            "FEATURE_SPACE_SOURCE_MISMATCH", "Feature-space manifest no longer matches its source."
        )
    metadata_snapshot = _metadata_snapshot(
        source,
        replicate_key=normalized_replicate,
        group_key=normalized_group,
        condition_key=normalized_condition,
        group_source=normalized_group_source,
        group_annotation_path=group_annotation_path,
        covariate_keys=normalized_covariates,
    )
    matrix = _production_aggregate(source, metadata_snapshot)
    library_sizes = _checked_library_sizes(matrix)
    provenance = _artifact_provenance(
        source,
        manifest_path,
        manifest_sha256,
        metadata_snapshot,
        matrix,
        group_source=normalized_group_source,
        group_key=normalized_group,
        replicate_key=normalized_replicate,
        condition_key=normalized_condition,
        covariate_keys=normalized_covariates,
        library_sizes=library_sizes,
    )
    artifact = _artifact(source, metadata_snapshot, matrix, provenance, library_sizes)
    directory = _resolve_output_dir(output_dir)
    output_path = directory / f"{source.input_path.stem}.replicate_pseudobulk.h5ad"
    _atomic_write_h5ad(artifact, output_path, overwrite=overwrite)
    artifact_digest = _file_sha256(output_path)
    return {
        "status": "success",
        "pseudobulk_path": str(output_path),
        "pseudobulk_sha256": artifact_digest,
        "feature_space_path": str(manifest_path),
        "feature_space_sha256": manifest_sha256,
        "feature_space_identity_sha256": source.feature_space_identity_sha256,
        "source_h5ad_path": str(source.input_path),
        "source_h5ad_sha256": source.source_h5ad_sha256,
        "matrix_semantics": source.matrix_semantics,
        "output_value_semantics": _output_value_semantics(source.matrix_semantics),
        "aggregation_method": "sum",
        "output_dtype": "int64",
        "group_source": normalized_group_source,
        "group_key": normalized_group,
        "replicate_key": normalized_replicate,
        "condition_key": normalized_condition,
        "covariate_keys": list(normalized_covariates),
        "n_cells": source.n_cells,
        "n_features": source.n_features,
        "n_pseudobulks": len(metadata_snapshot.unit_keys),
        "n_groups": len(set(metadata_snapshot.group_values)),
        "n_replicates": len(set(metadata_snapshot.replicate_values)),
        "n_conditions": len(set(metadata_snapshot.condition_values)),
        "minimum_cells_per_pseudobulk": min(metadata_snapshot.cell_counts),
        "maximum_cells_per_pseudobulk": max(metadata_snapshot.cell_counts),
        "matrix_nnz": int(matrix.nnz),
        "total_sum": int(sum(library_sizes)),
        "all_cells_accounted_for": sum(metadata_snapshot.cell_counts) == source.n_cells,
        "feature_order_preserved": True,
        "artifact_schema_version": PSEUDOBULK_SCHEMA_VERSION,
        "software_versions": dict(source.software_versions),
    }


__all__ = [
    "FEATURE_SPACE_ARTIFACT_TYPE",
    "FEATURE_SPACE_SCHEMA_VERSION",
    "M81ScientificError",
    "PSEUDOBULK_ARTIFACT_TYPE",
    "PSEUDOBULK_PROVENANCE_KEY",
    "PSEUDOBULK_SCHEMA_VERSION",
    "ReplicatePseudobulkToolResult",
    "ScATACFeatureSpaceToolResult",
    "build_replicate_pseudobulk",
    "validate_scATAC_feature_space",
]
