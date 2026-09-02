"""Independent M8.2-C reconstruction and edgeR verification boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.tools.analysis.replicate_pseudobulk import (
    PSEUDOBULK_ARTIFACT_TYPE,
    PSEUDOBULK_PROVENANCE_KEY,
    PSEUDOBULK_SCHEMA_VERSION,
    M81ScientificError,
    _artifact_provenance as _m81_artifact_provenance,
    _canonical_covariate as _m81_canonical_covariate,
    _canonical_integer_chunk as _m81_canonical_integer_chunk,
    _feature_manifest as _m81_feature_manifest,
    _file_sha256 as _file_sha256,
    _load_feature_manifest as _m81_load_feature_manifest,
    _metadata_snapshot as _m81_metadata_snapshot,
    _output_value_semantics as _m81_output_value_semantics,
    _read_backed as _m81_read_backed,
    _snapshot_from_manifest as _m81_snapshot_from_manifest,
    _source_matrix as _m81_source_matrix,
    _software_versions as _software_versions,
)


DA_ARTIFACT_TYPE = "agent.replicate-differential-accessibility"
DA_ARTIFACT_SCHEMA_VERSION = 1
DA_PROVENANCE_KEY = "agent_milestone8_differential_accessibility"
DA_PREPARATION_TYPE = "agent.replicate-da-preparation"
DA_PREPARATION_SCHEMA_VERSION = 1
EDGER_PROTOCOL_VERSION = 1
EDGER_PIPELINE_ID = "agent.edger-ql-native-v4.v1"
EDGER_RSCRIPT_ENVIRONMENT_VARIABLE = "AGENT_EDGER_RSCRIPT"
VERIFICATION_R_SCRIPT = (
    Path(__file__).resolve().parent / "r" / "edger_ql_verify_v1.R"
)
EXPECTED_VERIFICATION_R_SCRIPT_SHA256 = (
    "b07d52873d4ef2fb3d8cd521d689019cd3d73053b6f436c7de41ea89e8db8697"
)
PRODUCTION_R_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "analysis"
    / "r"
    / "edger_ql_v1.R"
)

EXPECTED_BACKEND_VERSIONS = {
    "r": "4.6.1",
    "bioconductor": "3.23",
    "biocmanager": "1.30.27",
    "edger": "4.10.4",
    "limma": "3.68.5",
    "locfit": "1.5.9.12",
    "statmod": "1.5.2",
    "lattice": "0.23.1",
}
FILTER_CONFIGURATION = {
    "method": "edgeR::filterByExpr",
    "grouping": "condition_denominator_0_numerator_1",
    "min_count": 10,
    "min_total_count": 15,
    "large_n": 10,
    "min_prop": 0.7,
}
NORMALIZATION_CONFIGURATION = {
    "method": "edgeR::normLibSizes",
    "method_argument": "TMM",
    "reference_column": "automatic",
    "logratio_trim": 0.30,
    "sum_trim": 0.05,
    "do_weighting": True,
    "a_cutoff": -1e10,
}
QL_CONFIGURATION = {
    "method": "edgeR::glmQLFit/glmQLFTest",
    "dispersion": None,
    "abundance_trend": True,
    "robust": True,
    "winsor_tail_p": [0.05, 0.10],
    "legacy": False,
    "top_proportion": None,
    "keep_unit_matrix": False,
    "prior_count": 0.125,
    "poisson_bound": True,
    "estimate_disp_called": False,
}

_MAX_CAPTURE_BYTES = 64 * 1024
_MAX_STATUS_BYTES = 64 * 1024
_MAX_EXACT_FLOAT64_INTEGER = 2**53
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
_DENSE_WORKING_COPIES = 12
_FEATURE_RESULT_COLUMNS = 6
_DESIGN_WORKING_COPIES = 4
_FIXED_MEMORY_OVERHEAD_BYTES = 64 * 1024 * 1024
_MEMORY_SAFETY_FACTOR = 1.25
_MINIMUM_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
_MAXIMUM_AVAILABLE_FRACTION = 0.70


class DAVerificationError(ValueError):
    """Fail-closed independent DA verification error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DAVerificationMetadata:
    verifier_r_script_sha256: str
    result_sha256: str
    preparation_sha256: str
    analysis_sha256: str


@dataclass(frozen=True)
class _VerifiedPseudobulk:
    path: Path
    sha256: str
    feature_space_identity_sha256: str
    feature_ids_sha256: str
    pseudobulk_matrix_sha256: str
    matrix_semantics: str
    output_value_semantics: str
    condition_key: str
    covariate_keys: tuple[str, ...]
    covariate_columns: tuple[str, ...]
    row_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    groups: tuple[str, ...]
    replicates: tuple[str, ...]
    conditions: tuple[str, ...]
    n_cells: tuple[int, ...]
    library_sizes: tuple[int, ...]
    covariate_values: tuple[tuple[object, ...], ...]
    obs: pd.DataFrame
    var: pd.DataFrame
    counts: sparse.csr_matrix


@dataclass(frozen=True)
class _CovariateEncoding:
    key: str
    kind: str
    source_column: str
    design_columns: tuple[str, ...]
    values: tuple[bool | int | float | str, ...]
    categorical_levels: tuple[tuple[str, bool | int | float | str, str | None], ...]


@dataclass(frozen=True)
class _Preparation:
    pseudobulk: _VerifiedPseudobulk
    group_value: str
    condition_key: str
    numerator_condition: str
    denominator_condition: str
    design_type: str
    covariates: tuple[tuple[str, str], ...]
    row_states: tuple[tuple[bool, str], ...]
    included_positions: tuple[int, ...]
    included_ids: tuple[str, ...]
    replicate_order: tuple[str, ...]
    numerator_replicates: tuple[str, ...]
    denominator_replicates: tuple[str, ...]
    warnings: tuple[tuple[str, Mapping[str, object]], ...]
    encodings: tuple[_CovariateEncoding, ...]
    design_columns: tuple[str, ...]
    design: np.ndarray
    contrast: np.ndarray
    rank: int
    residual_df: int
    rank_tolerance: float
    estimability_tolerance: float
    inclusion_sha256: str
    design_sha256: str
    contrast_sha256: str
    preparation_sha256: str


@dataclass(frozen=True)
class _RResult:
    filter_mask: np.ndarray
    tested_indices: np.ndarray
    post_filter_library_sizes: np.ndarray
    normalization_factors: np.ndarray
    effective_library_sizes: np.ndarray
    statistics: np.ndarray
    versions: Mapping[str, str]


def _error(code: str, message: str) -> DAVerificationError:
    return DAVerificationError(code, message)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(nested) for nested in value]
    raise TypeError("unsupported value")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
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


def _float_array_digest(
    domain: str,
    values: np.ndarray,
    *,
    row_ids: Sequence[str],
    column_ids: Sequence[str],
) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f8")
    header = {
        "schema_version": 1,
        "shape": list(canonical.shape),
        "dtype": "float64-little-endian",
        "row_ids": list(row_ids),
        "column_ids": list(column_ids),
    }
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(_canonical_json_bytes(header))
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _binary_digest(
    domain: str,
    values: np.ndarray,
    *,
    dtype: str,
    metadata: Mapping[str, object],
) -> str:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "shape": list(canonical.shape),
                "dtype": np.dtype(dtype).str,
                **dict(metadata),
            }
        )
    )
    digest.update(b"\n")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _independent_pseudobulk_sum(source: object, metadata: object) -> sparse.csr_matrix:
    rows: list[dict[int, int]] = [dict() for _ in metadata.unit_keys]
    raw = _m81_read_backed(source.input_path)
    try:
        matrix = _m81_source_matrix(raw, source.matrix_source, source.layer_key)
        for start in range(0, source.n_cells, 4096):
            stop = min(start + 4096, source.n_cells)
            chunk = _m81_canonical_integer_chunk(
                matrix, start, stop, source.matrix_semantics
            )
            for local_row in range(chunk.shape[0]):
                target = rows[metadata.cell_to_unit[start + local_row]]
                left, right = chunk.indptr[local_row : local_row + 2]
                for column, value in zip(
                    chunk.indices[left:right], chunk.data[left:right], strict=True
                ):
                    total = target.get(int(column), 0) + int(value)
                    if total > np.iinfo(np.int64).max:
                        raise M81ScientificError(
                            "INTEGER_SUM_OVERFLOW", "Independent sum overflowed."
                        )
                    target[int(column)] = total
    finally:
        raw.file.close()
    indptr = [0]
    indices: list[int] = []
    data: list[int] = []
    for row in rows:
        for column in sorted(row):
            if row[column]:
                indices.append(column)
                data.append(row[column])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(rows), source.n_features),
    )


def _verify_pseudobulk(path_value: object) -> _VerifiedPseudobulk:
    if not isinstance(path_value, (str, Path)):
        raise _error("PSEUDOBULK_ARTIFACT_INVALID", "Invalid pseudobulk path.")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise _error("PSEUDOBULK_ARTIFACT_INVALID", "Pseudobulk is unavailable.")
    digest_before = _file_sha256(path)
    artifact: ad.AnnData | None = None
    try:
        artifact = ad.read_h5ad(path, backed="r")
        if set(artifact.uns) != {PSEUDOBULK_PROVENANCE_KEY}:
            raise _error(
                "PSEUDOBULK_ARTIFACT_INVALID", "Pseudobulk structure is invalid."
            )
        provenance = _json_value(artifact.uns[PSEUDOBULK_PROVENANCE_KEY])
        if not isinstance(provenance, dict):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Pseudobulk provenance is invalid."
            )
        source_record = provenance.get("source")
        metadata_record = provenance.get("metadata")
        if (
            provenance.get("schema_version") != PSEUDOBULK_SCHEMA_VERSION
            or provenance.get("artifact_type") != PSEUDOBULK_ARTIFACT_TYPE
            or provenance.get("stage") != "replicate_pseudobulk"
            or not isinstance(source_record, Mapping)
            or not isinstance(metadata_record, Mapping)
        ):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Pseudobulk provenance is invalid."
            )
        feature_path, manifest, feature_sha256 = _m81_load_feature_manifest(
            source_record["feature_space_path"]
        )
        source = _m81_snapshot_from_manifest(manifest)
        if manifest != _m81_feature_manifest(source):
            raise _error(
                "FEATURE_SPACE_SOURCE_MISMATCH", "Feature-space source changed."
            )
        raw_keys = metadata_record.get("covariate_keys")
        raw_columns = metadata_record.get("covariate_columns")
        if not isinstance(raw_keys, list) or not isinstance(raw_columns, list):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Covariate provenance is invalid."
            )
        covariate_keys = tuple(str(value) for value in raw_keys)
        covariate_columns = tuple(str(value) for value in raw_columns)
        if covariate_columns != tuple(
            f"covariate_{index:03d}" for index in range(len(covariate_keys))
        ):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Covariate provenance is invalid."
            )
        metadata = _m81_metadata_snapshot(
            source,
            replicate_key=str(metadata_record["replicate_key"]),
            group_key=str(metadata_record["group_key"]),
            condition_key=str(metadata_record["condition_key"]),
            group_source=str(metadata_record["group_source"]),
            group_annotation_path=metadata_record.get("group_annotation_path"),
            covariate_keys=covariate_keys,
        )
        expected = _independent_pseudobulk_sum(source, metadata)
        library_sizes = tuple(
            sum(int(value) for value in expected.data[expected.indptr[row] : expected.indptr[row + 1]])
            for row in range(expected.shape[0])
        )
        expected_provenance = _m81_artifact_provenance(
            source,
            feature_path,
            feature_sha256,
            metadata,
            expected,
            group_source=str(metadata_record["group_source"]),
            group_key=str(metadata_record["group_key"]),
            replicate_key=str(metadata_record["replicate_key"]),
            condition_key=str(metadata_record["condition_key"]),
            covariate_keys=covariate_keys,
            library_sizes=library_sizes,
        )
        if provenance != _json_value(expected_provenance):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Pseudobulk provenance changed."
            )
        expected_obs_columns = (
            "group",
            "replicate",
            "condition",
            "n_cells",
            "first_cell_index",
            "library_size",
            *covariate_columns,
        )
        expected_var_columns = (
            ("chrom", "start", "end") if source.chromosomes is not None else ()
        )
        if (
            artifact.raw is not None
            or any(
                len(container)
                for container in (
                    artifact.layers,
                    artifact.obsm,
                    artifact.obsp,
                    artifact.varm,
                    artifact.varp,
                )
            )
            or tuple(artifact.obs.columns) != expected_obs_columns
            or tuple(artifact.var.columns) != expected_var_columns
            or tuple(str(value) for value in artifact.obs_names) != metadata.unit_ids
            or tuple(str(value) for value in artifact.var_names) != source.feature_ids
            or not isinstance(artifact.X, ad.abc.CSRDataset)
            or np.dtype(artifact.X.dtype) != np.dtype(np.int64)
        ):
            raise _error(
                "PSEUDOBULK_ARTIFACT_INVALID", "Pseudobulk structure changed."
            )
        observed_units = tuple(
            zip(
                (str(value) for value in artifact.obs["group"]),
                (str(value) for value in artifact.obs["replicate"]),
                (str(value) for value in artifact.obs["condition"]),
                strict=True,
            )
        )
        if (
            observed_units != metadata.unit_keys
            or tuple(int(value) for value in artifact.obs["n_cells"])
            != metadata.cell_counts
            or tuple(int(value) for value in artifact.obs["first_cell_index"])
            != metadata.first_cell_indices
            or tuple(int(value) for value in artifact.obs["library_size"])
            != library_sizes
        ):
            raise _error(
                "PSEUDOBULK_METADATA_MISMATCH", "Pseudobulk rows changed."
            )
        for index, key in enumerate(covariate_keys):
            observed = tuple(
                _m81_canonical_covariate(value, key)
                for value in artifact.obs[covariate_columns[index]].tolist()
            )
            if observed != tuple(row[index] for row in metadata.unit_covariates):
                raise _error(
                    "PSEUDOBULK_METADATA_MISMATCH", "Pseudobulk covariates changed."
                )
        if source.chromosomes is not None and (
            tuple(str(value) for value in artifact.var["chrom"])
            != source.chromosomes
            or tuple(int(value) for value in artifact.var["start"]) != source.starts
            or tuple(int(value) for value in artifact.var["end"]) != source.ends
        ):
            raise _error(
                "PSEUDOBULK_FEATURE_MISMATCH", "Feature coordinates changed."
            )
        for row in range(expected.shape[0]):
            observed = _m81_canonical_integer_chunk(
                artifact.X,
                row,
                row + 1,
                _m81_output_value_semantics(source.matrix_semantics),
            )
            reference = expected[row : row + 1]
            if not (
                np.array_equal(observed.indptr, reference.indptr)
                and np.array_equal(observed.indices, reference.indices)
                and np.array_equal(observed.data, reference.data)
            ):
                raise _error(
                    "PSEUDOBULK_AGGREGATION_MISMATCH", "Exact pseudobulk SUM changed."
                )
        digests = provenance.get("digests")
        if not isinstance(digests, Mapping):
            raise _error(
                "PSEUDOBULK_PROVENANCE_MISMATCH", "Pseudobulk digests are invalid."
            )
        result = _VerifiedPseudobulk(
            path=path,
            sha256=digest_before,
            feature_space_identity_sha256=source.feature_space_identity_sha256,
            feature_ids_sha256=source.feature_ids_sha256,
            pseudobulk_matrix_sha256=str(digests["pseudobulk_matrix_sha256"]),
            matrix_semantics=source.matrix_semantics,
            output_value_semantics=_m81_output_value_semantics(
                source.matrix_semantics
            ),
            condition_key=str(metadata_record["condition_key"]),
            covariate_keys=covariate_keys,
            covariate_columns=covariate_columns,
            row_ids=metadata.unit_ids,
            feature_ids=source.feature_ids,
            groups=tuple(value[0] for value in metadata.unit_keys),
            replicates=tuple(value[1] for value in metadata.unit_keys),
            conditions=tuple(value[2] for value in metadata.unit_keys),
            n_cells=metadata.cell_counts,
            library_sizes=library_sizes,
            covariate_values=metadata.unit_covariates,
            obs=artifact.obs.copy(),
            var=artifact.var.copy(),
            counts=expected,
        )
    except DAVerificationError:
        raise
    except Exception as exc:
        raise _error(
            getattr(exc, "code", "PSEUDOBULK_ARTIFACT_INVALID"),
            "Independent M8.1 verification failed.",
        ) from exc
    finally:
        if artifact is not None:
            artifact.file.close()
    if _file_sha256(path) != digest_before:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed during verification."
        )
    return result


def _strict_text(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _error(code, "A required DA identity is invalid.")
    return value


def _categorical_identity(value: object) -> tuple[str, bool | int | float | str]:
    canonical = _m81_canonical_covariate(value, "DA categorical covariate")
    if isinstance(canonical, bool):
        return "boolean", canonical
    if isinstance(canonical, int):
        return "integer", canonical
    if isinstance(canonical, float):
        return "float", canonical
    if isinstance(canonical, str):
        return "text", canonical
    raise _error("DA_COVARIATE_INVALID", "Categorical covariate is invalid.")


def _encode_covariates(
    source: _VerifiedPseudobulk,
    included: Sequence[int],
    specifications: Sequence[tuple[str, str]],
) -> tuple[tuple[_CovariateEncoding, ...], tuple[str, ...], list[np.ndarray]]:
    encodings: list[_CovariateEncoding] = []
    design_columns: list[str] = []
    design_values: list[np.ndarray] = []
    key_to_index = {key: index for index, key in enumerate(source.covariate_keys)}
    for covariate_index, (key, kind) in enumerate(specifications):
        source_index = key_to_index[key]
        source_column = source.covariate_columns[source_index]
        raw_values = tuple(
            source.covariate_values[position][source_index]
            for position in included
        )
        if kind == "numeric":
            numeric: list[float] = []
            for raw in raw_values:
                value = _m81_canonical_covariate(raw, key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise _error(
                        "DA_COVARIATE_INVALID", "Numeric covariate is invalid."
                    )
                converted = float(value)
                if not math.isfinite(converted):
                    raise _error(
                        "DA_COVARIATE_INVALID", "Numeric covariate is invalid."
                    )
                numeric.append(converted)
            if len(set(numeric)) < 2:
                raise _error(
                    "DA_COVARIATE_INVARIANT", "Numeric covariate is invariant."
                )
            column = f"covariate_{covariate_index:03d}_numeric"
            design_columns.append(column)
            design_values.append(np.asarray(numeric, dtype=np.float64))
            encodings.append(
                _CovariateEncoding(
                    key, kind, source_column, (column,), tuple(numeric), ()
                )
            )
            continue

        identities: list[tuple[str, bool | int | float | str]] = []
        levels: list[tuple[str, bool | int | float | str]] = []
        for raw in raw_values:
            identity = _categorical_identity(raw)
            identities.append(identity)
            if identity not in levels:
                levels.append(identity)
        if len(levels) < 2:
            raise _error(
                "DA_COVARIATE_INVARIANT", "Categorical covariate is invariant."
            )
        records: list[tuple[str, bool | int | float | str, str | None]] = []
        generated: list[str] = []
        for level_index, (value_type, value) in enumerate(levels):
            column = (
                None
                if level_index == 0
                else f"covariate_{covariate_index:03d}_level_{level_index:03d}"
            )
            records.append((value_type, value, column))
            if column is not None:
                generated.append(column)
                design_columns.append(column)
                design_values.append(
                    np.asarray(
                        [
                            1.0 if identity == (value_type, value) else 0.0
                            for identity in identities
                        ],
                        dtype=np.float64,
                    )
                )
        encodings.append(
            _CovariateEncoding(
                key,
                kind,
                source_column,
                tuple(generated),
                tuple(value for _, value in identities),
                tuple(records),
            )
        )
    return tuple(encodings), tuple(design_columns), design_values


def _reconstruct_preparation(
    source: _VerifiedPseudobulk, arguments: Mapping[str, object]
) -> _Preparation:
    if source.matrix_semantics not in {"fragment_counts", "insertion_counts"}:
        raise _error(
            "DA_MATRIX_SEMANTICS_INELIGIBLE", "Matrix semantics are ineligible."
        )
    group = _strict_text(arguments.get("group_value"), "DA_GROUP_NOT_FOUND")
    condition_key = _strict_text(
        arguments.get("condition_key"), "DA_CONDITION_NOT_FOUND"
    )
    numerator = _strict_text(
        arguments.get("numerator_condition"), "DA_CONDITION_NOT_FOUND"
    )
    denominator = _strict_text(
        arguments.get("denominator_condition"), "DA_CONDITION_NOT_FOUND"
    )
    design_type = _strict_text(arguments.get("design_type"), "DA_DESIGN_INVALID")
    if design_type not in {"independent", "paired"}:
        raise _error("DA_DESIGN_INVALID", "Design type is invalid.")
    if condition_key != source.condition_key:
        raise _error(
            "DA_CONDITION_KEY_MISMATCH", "Condition key differs from M8.1."
        )
    if numerator == denominator:
        raise _error("DA_CONDITION_NOT_FOUND", "Conditions must differ.")
    if group not in source.groups:
        raise _error("DA_GROUP_NOT_FOUND", "Group is absent.")
    available_conditions = {
        condition
        for row_group, condition in zip(
            source.groups, source.conditions, strict=True
        )
        if row_group == group
    }
    if numerator not in available_conditions or denominator not in available_conditions:
        raise _error("DA_CONDITION_NOT_FOUND", "A condition is absent.")

    row_states: list[tuple[bool, str]] = []
    included: list[int] = []
    for position, (row_group, condition) in enumerate(
        zip(source.groups, source.conditions, strict=True)
    ):
        if row_group != group:
            state = (False, "group_not_selected")
        elif condition not in {numerator, denominator}:
            state = (False, "condition_not_selected")
        else:
            state = (True, "included")
            included.append(position)
        row_states.append(state)
    if any(source.library_sizes[position] == 0 for position in included):
        raise _error("DA_ZERO_LIBRARY", "An included library is zero.")
    included_replicates = tuple(source.replicates[position] for position in included)
    included_conditions = tuple(source.conditions[position] for position in included)
    numerator_replicates = tuple(
        dict.fromkeys(
            replicate
            for replicate, condition in zip(
                included_replicates, included_conditions, strict=True
            )
            if condition == numerator
        )
    )
    denominator_replicates = tuple(
        dict.fromkeys(
            replicate
            for replicate, condition in zip(
                included_replicates, included_conditions, strict=True
            )
            if condition == denominator
        )
    )
    replicate_order = tuple(dict.fromkeys(included_replicates))
    warnings: list[tuple[str, Mapping[str, object]]] = []
    if design_type == "independent":
        if len(numerator_replicates) < 2 or len(denominator_replicates) < 2:
            raise _error(
                "DA_REPLICATION_INSUFFICIENT", "Replication is insufficient."
            )
        if set(numerator_replicates).intersection(denominator_replicates):
            raise _error("DA_PAIRING_INVALID", "Independent replicates overlap.")
        if len(numerator_replicates) == 2 or len(denominator_replicates) == 2:
            warnings.append(
                (
                    "DA_LOW_REPLICATION",
                    {
                        "numerator_replicates": len(numerator_replicates),
                        "denominator_replicates": len(denominator_replicates),
                        "recommended_minimum_per_condition": 3,
                    },
                )
            )
    else:
        pair_counts: dict[tuple[str, str], int] = {}
        for replicate, condition in zip(
            included_replicates, included_conditions, strict=True
        ):
            key = (replicate, condition)
            pair_counts[key] = pair_counts.get(key, 0) + 1
        if (
            any(count != 1 for count in pair_counts.values())
            or set(numerator_replicates) != set(denominator_replicates)
        ):
            raise _error("DA_PAIRING_INVALID", "Paired observations are invalid.")
        if len(numerator_replicates) < 3:
            raise _error(
                "DA_REPLICATION_INSUFFICIENT", "Replication is insufficient."
            )
    one_cell = tuple(position for position in included if source.n_cells[position] == 1)
    if one_cell:
        warnings.append(
            (
                "DA_ONE_CELL_PSEUDOBULK",
                {
                    "pseudobulk_count": len(one_cell),
                    "pseudobulk_ids": [source.row_ids[position] for position in one_cell],
                    "cell_counts": [source.n_cells[position] for position in one_cell],
                },
            )
        )

    raw_covariates = arguments.get("covariates", ())
    if not isinstance(raw_covariates, (list, tuple)):
        raise _error("DA_COVARIATE_INVALID", "Covariates are invalid.")
    specifications: list[tuple[str, str]] = []
    for raw in raw_covariates:
        if not isinstance(raw, Mapping) or set(raw) != {"key", "kind"}:
            raise _error("DA_COVARIATE_INVALID", "Covariates are invalid.")
        key = _strict_text(raw.get("key"), "DA_COVARIATE_INVALID")
        kind = _strict_text(raw.get("kind"), "DA_COVARIATE_INVALID")
        if kind not in {"categorical", "numeric"}:
            raise _error("DA_COVARIATE_INVALID", "Covariate kind is invalid.")
        specifications.append((key, kind))
    covariate_keys = tuple(key for key, _ in specifications)
    if len(set(covariate_keys)) != len(covariate_keys) or any(
        key not in source.covariate_keys for key in covariate_keys
    ):
        raise _error("DA_COVARIATE_INVALID", "Covariate key is invalid.")

    condition_values = np.asarray(
        [1.0 if value == numerator else 0.0 for value in included_conditions],
        dtype=np.float64,
    )
    design_columns: list[str] = ["intercept", "condition_numerator"]
    columns: list[np.ndarray] = [
        np.ones(len(included), dtype=np.float64),
        condition_values,
    ]
    if design_type == "paired":
        for replicate_index, replicate in enumerate(replicate_order[1:], start=1):
            design_columns.append(f"replicate_{replicate_index:03d}")
            columns.append(
                np.asarray(
                    [1.0 if value == replicate else 0.0 for value in included_replicates],
                    dtype=np.float64,
                )
            )
    encodings, covariate_columns, covariate_values = _encode_covariates(
        source, included, specifications
    )
    design_columns.extend(covariate_columns)
    columns.extend(covariate_values)
    design = np.column_stack(columns).astype(np.float64, copy=False)
    contrast = np.zeros(design.shape[1], dtype=np.float64)
    contrast[1] = 1.0
    _, singular_values, right_vectors = np.linalg.svd(design, full_matrices=True)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = max(design.shape) * _FLOAT64_EPSILON * largest
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    estimability_tolerance = (
        max(design.shape)
        * _FLOAT64_EPSILON
        * max(1.0, float(np.linalg.norm(contrast, ord=2)))
    )
    if rank < design.shape[1]:
        null_space = right_vectors[rank:, :]
        if null_space.size and float(np.linalg.norm(null_space @ contrast, ord=2)) > estimability_tolerance:
            raise _error("DA_CONTRAST_NOT_ESTIMABLE", "Contrast is not estimable.")
        raise _error("DA_DESIGN_RANK_DEFICIENT", "Design is rank deficient.")
    residual_df = design.shape[0] - design.shape[1]
    if design.shape[0] <= design.shape[1] or residual_df < 2:
        raise _error("DA_RESIDUAL_DF_INSUFFICIENT", "Residual DF is insufficient.")

    included_ids = tuple(source.row_ids[position] for position in included)
    warning_payload = [
        {"code": code, "metadata": dict(metadata)} for code, metadata in warnings
    ]
    inclusion_payload = {
        "schema_version": DA_PREPARATION_SCHEMA_VERSION,
        "pseudobulk_sha256": source.sha256,
        "feature_space_identity_sha256": source.feature_space_identity_sha256,
        "group": group,
        "condition_key": condition_key,
        "numerator": numerator,
        "denominator": denominator,
        "design_type": design_type,
        "rows": [
            {
                "source_position": position,
                "pseudobulk_id": source.row_ids[position],
                "group": source.groups[position],
                "replicate": source.replicates[position],
                "condition": source.conditions[position],
                "n_cells": source.n_cells[position],
                "library_size": source.library_sizes[position],
                "included": state[0],
                "reason": state[1],
            }
            for position, state in enumerate(row_states)
        ],
        "replicate_order": list(replicate_order),
        "covariates": [
            {"key": key, "kind": kind} for key, kind in specifications
        ],
        "warnings": warning_payload,
    }
    inclusion_sha256 = _domain_digest(
        "agent.replicate-da-inclusion.v1", inclusion_payload
    )
    design_sha256 = _float_array_digest(
        "agent.replicate-da-design-matrix.v1",
        design,
        row_ids=included_ids,
        column_ids=design_columns,
    )
    contrast_sha256 = _float_array_digest(
        "agent.replicate-da-contrast.v1",
        contrast.reshape(1, -1),
        row_ids=("numerator_minus_denominator",),
        column_ids=design_columns,
    )
    encoding_payload = [
        {
            "key": encoding.key,
            "kind": encoding.kind,
            "source_column": encoding.source_column,
            "design_columns": list(encoding.design_columns),
            "values": list(encoding.values),
            "categorical_levels": [
                {
                    "value_type": value_type,
                    "value": value,
                    "design_column": column,
                }
                for value_type, value, column in encoding.categorical_levels
            ],
        }
        for encoding in encodings
    ]
    preparation_sha256 = _domain_digest(
        "agent.replicate-da-preparation.v1",
        {
            "pseudobulk_sha256": source.sha256,
            "feature_space_identity_sha256": source.feature_space_identity_sha256,
            "feature_ids_sha256": source.feature_ids_sha256,
            "pseudobulk_matrix_sha256": source.pseudobulk_matrix_sha256,
            "inclusion_sha256": inclusion_sha256,
            "design_sha256": design_sha256,
            "contrast_sha256": contrast_sha256,
            "design_columns": design_columns,
            "design_shape": list(design.shape),
            "design_rank": rank,
            "residual_degrees_of_freedom": residual_df,
            "rank_tolerance": rank_tolerance,
            "estimability_tolerance": estimability_tolerance,
            "covariate_encodings": encoding_payload,
            "warnings": warning_payload,
        },
    )
    return _Preparation(
        source,
        group,
        condition_key,
        numerator,
        denominator,
        design_type,
        tuple(specifications),
        tuple(row_states),
        tuple(included),
        included_ids,
        replicate_order,
        numerator_replicates,
        denominator_replicates,
        tuple(warnings),
        encodings,
        tuple(design_columns),
        design,
        contrast,
        rank,
        residual_df,
        rank_tolerance,
        estimability_tolerance,
        inclusion_sha256,
        design_sha256,
        contrast_sha256,
        preparation_sha256,
    )


def _resolve_rscript() -> Path:
    raw = os.environ.get(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE)
    if raw is None or not raw.strip():
        raise _error("RSCRIPT_UNAVAILABLE", "Compatible Rscript is not configured.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise _error("RSCRIPT_UNAVAILABLE", "Compatible Rscript must be absolute.")
    path = candidate.resolve()
    if not path.is_file() or not os.access(path, os.X_OK) or path.name != "Rscript":
        raise _error("RSCRIPT_UNAVAILABLE", "Compatible Rscript is unavailable.")
    return path


def _host_available_memory_bytes() -> int:
    candidates: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            candidates.append(pages * page_size)
    except (OSError, ValueError, TypeError):
        pass
    try:
        maximum_text = Path("/sys/fs/cgroup/memory.max").read_text(
            encoding="ascii"
        ).strip()
        current = int(
            Path("/sys/fs/cgroup/memory.current").read_text(encoding="ascii")
        )
        if maximum_text != "max":
            maximum = int(maximum_text)
            if maximum > current:
                candidates.append(maximum - current)
    except (OSError, ValueError):
        pass
    if not candidates:
        raise _error(
            "HOST_MEMORY_EXHAUSTED",
            "Available host memory could not be determined for verification.",
        )
    return min(candidates)


def _preflight_verifier_memory(preparation: _Preparation) -> None:
    n_features = len(preparation.pseudobulk.feature_ids)
    n_samples = len(preparation.included_positions)
    n_columns = len(preparation.design_columns)
    dense_bytes = n_features * n_samples * 8
    base = (
        dense_bytes * _DENSE_WORKING_COPIES
        + n_features * _FEATURE_RESULT_COLUMNS * 8
        + n_samples * n_columns * 8 * _DESIGN_WORKING_COPIES
        + 2 * _FIXED_MEMORY_OVERHEAD_BYTES
    )
    estimated = math.ceil(base * _MEMORY_SAFETY_FACTOR)
    available = _host_available_memory_bytes()
    reserve = max(
        _MINIMUM_MEMORY_RESERVE_BYTES,
        math.ceil(available * (1.0 - _MAXIMUM_AVAILABLE_FRACTION)),
    )
    if estimated > max(0, available - reserve):
        raise _error(
            "HOST_MEMORY_EXHAUSTED",
            "Host memory preflight rejected independent DA verification.",
        )


def _subprocess_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "R_ENVIRON",
        "R_ENVIRON_USER",
        "R_PROFILE",
        "R_PROFILE_USER",
        "R_LIBS",
        "R_LIBS_SITE",
        "R_LIBS_USER",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    return environment


def _write_binary(path: Path, values: np.ndarray, dtype: str) -> None:
    path.write_bytes(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _stage_verification(directory: Path, preparation: _Preparation) -> str:
    counts_digest = hashlib.sha256()
    counts_digest.update(b"agent.replicate-da-selected-counts.v1\0")
    counts_digest.update(
        _canonical_json_bytes(
            {
                "shape": [
                    len(preparation.included_positions),
                    len(preparation.pseudobulk.feature_ids),
                ],
                "dtype": "float64-little-endian",
                "row_ids": list(preparation.included_ids),
                "feature_ids_sha256": preparation.pseudobulk.feature_ids_sha256,
            }
        )
    )
    counts_digest.update(b"\n")
    with (directory / "verification_counts.bin").open("wb") as handle:
        for position in preparation.included_positions:
            dense = np.asarray(
                preparation.pseudobulk.counts[position : position + 1].toarray(),
                dtype="<f8",
            ).reshape(-1)
            if dense.size and np.any(dense > _MAX_EXACT_FLOAT64_INTEGER):
                raise _error(
                    "DA_NUMERICAL_RESULT_INVALID", "Counts exceed exact float64."
                )
            payload = dense.tobytes(order="C")
            handle.write(payload)
            counts_digest.update(payload)
    selected_counts_sha256 = counts_digest.hexdigest()
    _write_binary(
        directory / "verification_design.bin", preparation.design, "<f8"
    )
    _write_binary(
        directory / "verification_contrast.bin", preparation.contrast, "<f8"
    )
    condition = np.asarray(
        [
            1
            if preparation.pseudobulk.conditions[position]
            == preparation.numerator_condition
            else 0
            for position in preparation.included_positions
        ],
        dtype="<i4",
    )
    _write_binary(directory / "verification_condition.bin", condition, "<i4")
    manifest = (
        ("protocol_version", str(EDGER_PROTOCOL_VERSION)),
        ("preparation_sha256", preparation.preparation_sha256),
        ("n_features", str(len(preparation.pseudobulk.feature_ids))),
        ("n_samples", str(len(preparation.included_positions))),
        ("n_design_columns", str(len(preparation.design_columns))),
        ("design_sha256", preparation.design_sha256),
        ("contrast_sha256", preparation.contrast_sha256),
        ("input_matrix_sha256", selected_counts_sha256),
    )
    (directory / "verification_manifest.tsv").write_text(
        "".join(f"{key}\t{value}\n" for key, value in manifest),
        encoding="utf-8",
    )
    return selected_counts_sha256


def _read_status(directory: Path) -> dict[str, str]:
    path = directory / "verification_status.tsv"
    try:
        if not path.is_file() or path.stat().st_size > _MAX_STATUS_BYTES:
            raise ValueError
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            pieces = line.split("\t")
            if len(pieces) != 2 or not pieces[0] or pieces[0] in fields:
                raise ValueError
            fields[pieces[0]] = pieces[1]
        if fields.get("protocol_version") != str(EDGER_PROTOCOL_VERSION):
            raise ValueError
        return fields
    except (OSError, UnicodeError, ValueError) as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "Verifier R status is invalid."
        ) from exc


def _read_binary(path: Path, dtype: str, count: int) -> np.ndarray:
    try:
        if not path.is_file() or path.stat().st_size != np.dtype(dtype).itemsize * count:
            raise ValueError
        result = np.fromfile(path, dtype=dtype, count=count)
    except (OSError, ValueError) as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "Verifier R output is invalid."
        ) from exc
    if result.size != count:
        raise _error("R_BACKEND_PROTOCOL_INVALID", "Verifier R output is truncated.")
    return result


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    count = p_values.size
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count, dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def _invoke_verifier_r(directory: Path, preparation: _Preparation) -> _RResult:
    rscript = _resolve_rscript()
    stdout_path = directory / "verification_stdout.log"
    stderr_path = directory / "verification_stderr.log"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [
                    str(rscript),
                    "--vanilla",
                    str(VERIFICATION_R_SCRIPT),
                    str(directory),
                ],
                cwd=directory,
                env=_subprocess_environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
            )
            return_code = process.wait()
    except FileNotFoundError as exc:
        raise _error("RSCRIPT_UNAVAILABLE", "Compatible Rscript is unavailable.") from exc
    except OSError as exc:
        raise _error(
            "R_BACKEND_EXECUTION_FAILED", "Verifier R process could not execute."
        ) from exc
    if (
        stdout_path.stat().st_size > _MAX_CAPTURE_BYTES
        or stderr_path.stat().st_size > _MAX_CAPTURE_BYTES
    ):
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "Verifier R diagnostics are oversized."
        )
    try:
        status = _read_status(directory)
    except DAVerificationError as exc:
        if return_code != 0:
            raise _error(
                "R_BACKEND_EXECUTION_FAILED", "Verifier R process failed."
            ) from exc
        raise
    if status.get("status") == "error":
        code = status.get("error_code")
        allowed = {
            "EDGER_PACKAGE_UNAVAILABLE",
            "EDGER_VERSION_UNSUPPORTED",
            "R_PACKAGE_VERSION_INCOMPATIBLE",
            "R_BACKEND_EXECUTION_FAILED",
            "DA_NO_FEATURES_AFTER_FILTER",
            "DA_FILTERED_LIBRARY_ZERO",
            "DA_NUMERICAL_RESULT_INVALID",
        }
        if set(status) != {"protocol_version", "status", "error_code"} or code not in allowed:
            raise _error(
                "R_BACKEND_PROTOCOL_INVALID", "Verifier R error protocol is invalid."
            )
        raise _error(str(code), "Verifier R reported a controlled failure.")
    if return_code != 0 or status.get("status") != "success":
        raise _error("R_BACKEND_EXECUTION_FAILED", "Verifier R process failed.")
    expected_fields = {
        "protocol_version",
        "status",
        "n_features",
        "n_samples",
        "n_tested",
        "n_filtered",
        "r_version",
        "bioconductor_version",
        "biocmanager_version",
        "edger_version",
        "limma_version",
        "locfit_version",
        "statmod_version",
        "lattice_version",
    }
    if set(status) != expected_fields:
        raise _error("R_BACKEND_PROTOCOL_INVALID", "Verifier R fields are invalid.")
    version_fields = {
        "r": "r_version",
        "bioconductor": "bioconductor_version",
        "biocmanager": "biocmanager_version",
        "edger": "edger_version",
        "limma": "limma_version",
        "locfit": "locfit_version",
        "statmod": "statmod_version",
        "lattice": "lattice_version",
    }
    versions = {key: status[field] for key, field in version_fields.items()}
    if versions != EXPECTED_BACKEND_VERSIONS:
        if versions.get("edger") != EXPECTED_BACKEND_VERSIONS["edger"]:
            raise _error("EDGER_VERSION_UNSUPPORTED", "edgeR is incompatible.")
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE", "Verifier R stack is incompatible."
        )
    try:
        n_features = int(status["n_features"])
        n_samples = int(status["n_samples"])
        n_tested = int(status["n_tested"])
        n_filtered = int(status["n_filtered"])
    except ValueError as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "Verifier dimensions are invalid."
        ) from exc
    if (
        n_features != len(preparation.pseudobulk.feature_ids)
        or n_samples != len(preparation.included_positions)
        or n_tested <= 0
        or n_tested + n_filtered != n_features
    ):
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "Verifier dimensions are inconsistent."
        )
    mask_values = _read_binary(
        directory / "verification_filter_mask.bin", "<i4", n_features
    )
    if not np.all(np.isin(mask_values, (0, 1))):
        raise _error("R_BACKEND_PROTOCOL_INVALID", "Verifier mask is invalid.")
    mask = mask_values.astype(bool)
    tested = _read_binary(
        directory / "verification_tested_indices.bin", "<i4", n_tested
    ).astype(np.int64)
    if not np.array_equal(tested, np.flatnonzero(mask)):
        raise _error("R_BACKEND_PROTOCOL_INVALID", "Verifier order is invalid.")
    libraries = _read_binary(
        directory / "verification_post_filter_library_sizes.bin", "<f8", n_samples
    )
    factors = _read_binary(
        directory / "verification_normalization_factors.bin", "<f8", n_samples
    )
    effective = _read_binary(
        directory / "verification_effective_library_sizes.bin", "<f8", n_samples
    )
    statistics = _read_binary(
        directory / "verification_statistics.bin", "<f8", n_tested * 5
    ).reshape(n_tested, 5)
    if (
        not np.isfinite(libraries).all()
        or np.any(libraries <= 0)
        or not np.isfinite(factors).all()
        or np.any(factors <= 0)
        or not np.isfinite(effective).all()
        or np.any(effective <= 0)
        or not np.allclose(
            effective, libraries * factors, rtol=1e-12, atol=1e-12
        )
        or abs(float(np.sum(np.log(factors)))) > 1e-10
        or not np.isfinite(statistics).all()
        or np.any(statistics[:, 2] < 0)
        or np.any(statistics[:, 3:] < 0)
        or np.any(statistics[:, 3:] > 1)
        or not np.allclose(
            statistics[:, 4],
            _bh_adjust(statistics[:, 3]),
            rtol=1e-12,
            atol=1e-300,
        )
    ):
        raise _error(
            "DA_NUMERICAL_RESULT_INVALID", "Verifier numerical output is invalid."
        )
    return _RResult(mask, tested, libraries, factors, effective, statistics, versions)


def _result_digests(preparation: _Preparation, result: _RResult) -> dict[str, str]:
    source = preparation.pseudobulk
    filter_digest = _binary_digest(
        "agent.replicate-da-filter-mask.v1",
        result.filter_mask.astype(np.uint8),
        dtype="u1",
        metadata={"feature_ids_sha256": source.feature_ids_sha256},
    )
    library_digest = _binary_digest(
        "agent.replicate-da-postfilter-library-sizes.v1",
        result.post_filter_library_sizes,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_ids)},
    )
    factor_digest = _binary_digest(
        "agent.replicate-da-normalization-factors.v1",
        result.normalization_factors,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_ids)},
    )
    effective_digest = _binary_digest(
        "agent.replicate-da-effective-library-sizes.v1",
        result.effective_library_sizes,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_ids)},
    )
    statistics_digest = _binary_digest(
        "agent.replicate-da-tested-statistics.v1",
        result.statistics,
        dtype="<f8",
        metadata={
            "feature_ids_sha256": source.feature_ids_sha256,
            "tested_indices": result.tested_indices.tolist(),
            "columns": ["logFC", "logCPM", "F", "PValue", "FDR"],
        },
    )
    result_digest = _domain_digest(
        "agent.replicate-da-result.v1",
        {
            "preparation_sha256": preparation.preparation_sha256,
            "filter_mask_sha256": filter_digest,
            "post_filter_library_sizes_sha256": library_digest,
            "normalization_factors_sha256": factor_digest,
            "effective_library_sizes_sha256": effective_digest,
            "tested_statistics_sha256": statistics_digest,
        },
    )
    return {
        "filter_mask_sha256": filter_digest,
        "post_filter_library_sizes_sha256": library_digest,
        "normalization_factors_sha256": factor_digest,
        "effective_library_sizes_sha256": effective_digest,
        "tested_statistics_sha256": statistics_digest,
        "result_sha256": result_digest,
    }


def _analysis_identity(preparation: _Preparation, production_sha256: str) -> str:
    source = preparation.pseudobulk
    return _domain_digest(
        "agent.replicate-differential-accessibility-analysis.v1",
        {
            "pseudobulk_sha256": source.sha256,
            "feature_space_identity_sha256": source.feature_space_identity_sha256,
            "preparation_sha256": preparation.preparation_sha256,
            "pipeline": EDGER_PIPELINE_ID,
            "filter": FILTER_CONFIGURATION,
            "normalization": NORMALIZATION_CONFIGURATION,
            "ql": QL_CONFIGURATION,
            "production_r_script_sha256": production_sha256,
            "backend_version_policy": EXPECTED_BACKEND_VERSIONS,
        },
    )


def _warning_summary(
    warning: tuple[str, Mapping[str, object]],
) -> dict[str, object]:
    code, metadata = warning
    if code == "DA_ONE_CELL_PSEUDOBULK":
        return {"code": code, "pseudobulk_count": int(metadata["pseudobulk_count"])}
    return {"code": code, **dict(metadata)}


def _warning_provenance(
    warnings: tuple[tuple[str, Mapping[str, object]], ...],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for warning in warnings:
        summary = _warning_summary(warning)
        code = str(summary.pop("code"))
        records[code] = summary
    return records


def _covariate_contract(preparation: _Preparation) -> list[dict[str, object]]:
    return [
        {
            "key": encoding.key,
            "kind": encoding.kind,
            "source_column": encoding.source_column,
            "design_columns": list(encoding.design_columns),
            "categorical_levels": [
                {
                    "value_type": value_type,
                    "value": value,
                    "design_column": column,
                }
                for value_type, value, column in encoding.categorical_levels
            ],
        }
        for encoding in preparation.encodings
    ]


def _validate_memory_record(
    value: object, preparation: _Preparation
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error("DA_ARTIFACT_INVALID", "Memory provenance is invalid.")
    expected_keys = {
        "policy",
        "n_features",
        "n_samples",
        "n_design_columns",
        "dense_dtype",
        "dense_matrix_bytes",
        "base_estimate_bytes",
        "dense_working_copies",
        "feature_result_columns",
        "design_working_copies",
        "fixed_overhead_per_process_bytes",
        "fixed_overhead_processes",
        "safety_factor",
        "maximum_available_fraction",
        "minimum_reserve_bytes",
        "estimated_peak_bytes",
        "available_bytes",
        "usable_bytes",
    }
    if set(value) != expected_keys:
        raise _error("DA_ARTIFACT_INVALID", "Memory provenance fields are invalid.")
    n_features = len(preparation.pseudobulk.feature_ids)
    n_samples = len(preparation.included_positions)
    n_columns = len(preparation.design_columns)
    dense_bytes = n_features * n_samples * 8
    base = (
        dense_bytes * _DENSE_WORKING_COPIES
        + n_features * _FEATURE_RESULT_COLUMNS * 8
        + n_samples * n_columns * 8 * _DESIGN_WORKING_COPIES
        + _FIXED_MEMORY_OVERHEAD_BYTES
    )
    estimated = math.ceil(base * _MEMORY_SAFETY_FACTOR)
    available = value.get("available_bytes")
    if isinstance(available, bool) or not isinstance(available, int) or available <= 0:
        raise _error("DA_ARTIFACT_INVALID", "Memory availability is invalid.")
    reserve = max(
        _MINIMUM_MEMORY_RESERVE_BYTES,
        math.ceil(available * (1.0 - _MAXIMUM_AVAILABLE_FRACTION)),
    )
    expected = {
        "policy": "dense-edger-v1",
        "n_features": n_features,
        "n_samples": n_samples,
        "n_design_columns": n_columns,
        "dense_dtype": "float64",
        "dense_matrix_bytes": dense_bytes,
        "base_estimate_bytes": base,
        "dense_working_copies": _DENSE_WORKING_COPIES,
        "feature_result_columns": _FEATURE_RESULT_COLUMNS,
        "design_working_copies": _DESIGN_WORKING_COPIES,
        "fixed_overhead_per_process_bytes": _FIXED_MEMORY_OVERHEAD_BYTES,
        "fixed_overhead_processes": 2,
        "safety_factor": _MEMORY_SAFETY_FACTOR,
        "maximum_available_fraction": _MAXIMUM_AVAILABLE_FRACTION,
        "minimum_reserve_bytes": _MINIMUM_MEMORY_RESERVE_BYTES,
        "estimated_peak_bytes": estimated,
        "available_bytes": available,
        "usable_bytes": max(0, available - reserve),
    }
    if _json_value(value) != expected or estimated > expected["usable_bytes"]:
        raise _error("DA_ARTIFACT_INVALID", "Memory provenance is inconsistent.")
    return value


def _expected_provenance(
    preparation: _Preparation,
    result: _RResult,
    *,
    selected_counts_sha256: str,
    production_sha256: str,
    analysis_sha256: str,
    digests: Mapping[str, str],
    memory_record: Mapping[str, object],
) -> dict[str, object]:
    source = preparation.pseudobulk
    return {
        "schema_version": DA_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": DA_ARTIFACT_TYPE,
        "stage": "replicate_differential_accessibility",
        "analysis_sha256": analysis_sha256,
        "source": {
            "pseudobulk_path": str(source.path),
            "pseudobulk_sha256": source.sha256,
            "feature_space_identity_sha256": source.feature_space_identity_sha256,
            "feature_ids_sha256": source.feature_ids_sha256,
            "pseudobulk_matrix_sha256": source.pseudobulk_matrix_sha256,
            "matrix_semantics": source.matrix_semantics,
            "output_value_semantics": source.output_value_semantics,
            "selected_counts_sha256": selected_counts_sha256,
        },
        "comparison": {
            "group_value": preparation.group_value,
            "condition_key": preparation.condition_key,
            "numerator_condition": preparation.numerator_condition,
            "denominator_condition": preparation.denominator_condition,
            "positive_logfc_meaning": "higher_in_numerator",
            "design_type": preparation.design_type,
            "n_samples": len(preparation.included_positions),
            "n_numerator_replicates": len(preparation.numerator_replicates),
            "n_denominator_replicates": len(preparation.denominator_replicates),
            "warnings": _warning_provenance(preparation.warnings),
        },
        "preparation": {
            "preparation_type": DA_PREPARATION_TYPE,
            "schema_version": DA_PREPARATION_SCHEMA_VERSION,
            "preparation_sha256": preparation.preparation_sha256,
            "inclusion_sha256": preparation.inclusion_sha256,
            "design_sha256": preparation.design_sha256,
            "contrast_sha256": preparation.contrast_sha256,
            "design_shape": list(preparation.design.shape),
            "design_rank": preparation.rank,
            "residual_degrees_of_freedom": preparation.residual_df,
            "rank_tolerance": preparation.rank_tolerance,
            "estimability_tolerance": preparation.estimability_tolerance,
            "covariate_contract": _covariate_contract(preparation),
        },
        "filter": {
            **FILTER_CONFIGURATION,
            "n_input_features": len(source.feature_ids),
            "n_tested_features": int(result.tested_indices.size),
            "n_filtered_features": int(
                len(source.feature_ids) - result.tested_indices.size
            ),
            "filter_mask_sha256": digests["filter_mask_sha256"],
            "post_filter_library_sizes_sha256": digests[
                "post_filter_library_sizes_sha256"
            ],
        },
        "normalization": {
            **NORMALIZATION_CONFIGURATION,
            "normalization_factors_sha256": digests[
                "normalization_factors_sha256"
            ],
            "effective_library_sizes_sha256": digests[
                "effective_library_sizes_sha256"
            ],
        },
        "statistical_test": {
            **QL_CONFIGURATION,
            "multiple_testing": "Benjamini-Hochberg",
            "multiple_testing_universe": "tested_features_this_group_contrast",
            "tested_statistics_sha256": digests["tested_statistics_sha256"],
            "result_sha256": digests["result_sha256"],
        },
        "backend": {
            "pipeline": EDGER_PIPELINE_ID,
            "protocol_version": EDGER_PROTOCOL_VERSION,
            "production_r_script_sha256": production_sha256,
            "versions": dict(result.versions),
        },
        "memory_preflight": dict(memory_record),
        "validation": {
            "all_source_pseudobulk_rows_preserved": True,
            "source_feature_order_preserved": True,
            "source_count_matrix_copied_to_artifact": False,
            "source_pseudobulk_modified": False,
            "sample_filtering_performed": False,
            "covariate_removal_performed": False,
            "design_reconstruction_performed": False,
            "individual_cells_used_as_observations": False,
        },
        "software_versions": _software_versions(),
    }


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    if isinstance(left.dtype, pd.CategoricalDtype) or isinstance(
        right.dtype, pd.CategoricalDtype
    ):
        return left.astype(str).tolist() == right.astype(str).tolist()
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.dtype.kind == "f" or right_values.dtype.kind == "f":
        return np.array_equal(left_values, right_values, equal_nan=True)
    return np.array_equal(left_values, right_values)


def _validate_da_artifact(
    artifact: ad.AnnData,
    preparation: _Preparation,
    r_result: _RResult,
    expected_provenance: Mapping[str, object],
) -> None:
    source = preparation.pseudobulk
    expected_obs_columns = (
        *source.obs.columns,
        "da_group_match",
        "da_condition_match",
        "da_analysis_included",
        "da_exclusion_reason",
        "da_design_row_index",
        "da_postfilter_library_size",
        "da_tmm_normalization_factor",
        "da_effective_library_size",
    )
    expected_var_columns = (
        *source.var.columns,
        "da_status",
        "logFC",
        "logCPM",
        "F",
        "PValue",
        "FDR",
        "effect_direction",
    )
    if (
        artifact.X is not None
        or artifact.raw is not None
        or tuple(artifact.obs.columns) != expected_obs_columns
        or tuple(artifact.var.columns) != expected_var_columns
        or tuple(str(value) for value in artifact.obs_names) != source.row_ids
        or tuple(str(value) for value in artifact.var_names) != source.feature_ids
        or set(artifact.uns) != {DA_PROVENANCE_KEY}
        or any(
            len(container)
            for container in (
                artifact.layers,
                artifact.obsm,
                artifact.obsp,
                artifact.varm,
                artifact.varp,
            )
        )
    ):
        raise _error("DA_ARTIFACT_INVALID", "DA H5AD structure is invalid.")
    for column in source.obs.columns:
        if not _series_equal(artifact.obs[column], source.obs[column]):
            raise _error("DA_ARTIFACT_INVALID", "Source obs metadata changed.")
    for column in source.var.columns:
        if not _series_equal(artifact.var[column], source.var[column]):
            raise _error("DA_ARTIFACT_INVALID", "Source feature metadata changed.")
    expected_group = np.asarray(
        [value == preparation.group_value for value in source.groups], dtype=bool
    )
    expected_condition = np.asarray(
        [
            value
            in {preparation.numerator_condition, preparation.denominator_condition}
            for value in source.conditions
        ],
        dtype=bool,
    )
    expected_included = np.asarray(
        [state[0] for state in preparation.row_states], dtype=bool
    )
    expected_reasons = [state[1] for state in preparation.row_states]
    expected_design_rows = np.full(len(source.row_ids), -1, dtype=np.int64)
    expected_libraries = np.full(len(source.row_ids), np.nan, dtype=np.float64)
    expected_factors = np.full(len(source.row_ids), np.nan, dtype=np.float64)
    expected_effective = np.full(len(source.row_ids), np.nan, dtype=np.float64)
    for design_row, source_position in enumerate(preparation.included_positions):
        expected_design_rows[source_position] = design_row
        expected_libraries[source_position] = r_result.post_filter_library_sizes[
            design_row
        ]
        expected_factors[source_position] = r_result.normalization_factors[design_row]
        expected_effective[source_position] = r_result.effective_library_sizes[
            design_row
        ]
    if not (
        np.array_equal(np.asarray(artifact.obs["da_group_match"]), expected_group)
        and np.array_equal(
            np.asarray(artifact.obs["da_condition_match"]), expected_condition
        )
        and np.array_equal(
            np.asarray(artifact.obs["da_analysis_included"]), expected_included
        )
        and artifact.obs["da_exclusion_reason"].astype(str).tolist()
        == expected_reasons
        and np.array_equal(
            np.asarray(artifact.obs["da_design_row_index"]), expected_design_rows
        )
        and np.allclose(
            np.asarray(artifact.obs["da_postfilter_library_size"]),
            expected_libraries,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        and np.allclose(
            np.asarray(artifact.obs["da_tmm_normalization_factor"]),
            expected_factors,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        and np.allclose(
            np.asarray(artifact.obs["da_effective_library_size"]),
            expected_effective,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    ):
        raise _error("DA_ARTIFACT_INVALID", "DA sample metadata changed.")
    status = artifact.var["da_status"].astype(str).tolist()
    expected_status = [
        "tested" if value else "filtered_by_expression"
        for value in r_result.filter_mask
    ]
    if status != expected_status:
        raise _error("DA_ARTIFACT_INVALID", "DA filter state changed.")
    for index, column in enumerate(("logFC", "logCPM", "F", "PValue", "FDR")):
        observed = np.asarray(artifact.var[column], dtype=np.float64)
        if (
            not np.isnan(observed[~r_result.filter_mask]).all()
            or not np.allclose(
                observed[r_result.filter_mask],
                r_result.statistics[:, index],
                rtol=1e-12,
                atol=1e-300 if column in {"PValue", "FDR"} else 1e-12,
            )
        ):
            raise _error("DA_ARTIFACT_INVALID", "DA statistics changed.")
    directions = np.full(len(source.feature_ids), "not_tested", dtype=object)
    logfc = r_result.statistics[:, 0]
    directions[r_result.tested_indices[logfc > 0]] = "higher_in_numerator"
    directions[r_result.tested_indices[logfc < 0]] = "higher_in_denominator"
    directions[r_result.tested_indices[logfc == 0]] = "no_change"
    if artifact.var["effect_direction"].astype(str).tolist() != directions.tolist():
        raise _error("DA_ARTIFACT_INVALID", "DA effect direction changed.")
    if _json_value(artifact.uns[DA_PROVENANCE_KEY]) != _json_value(
        expected_provenance
    ):
        raise _error("DA_ARTIFACT_INVALID", "DA provenance changed.")


def _expected_result(
    preparation: _Preparation,
    r_result: _RResult,
    *,
    da_path: Path,
    da_sha256: str,
    production_sha256: str,
    analysis_sha256: str,
) -> dict[str, object]:
    source = preparation.pseudobulk
    warning_codes = [code for code, _ in preparation.warnings]
    return {
        "status": "success",
        "da_path": str(da_path),
        "da_sha256": da_sha256,
        "artifact_type": DA_ARTIFACT_TYPE,
        "artifact_schema_version": DA_ARTIFACT_SCHEMA_VERSION,
        "pseudobulk_path": str(source.path),
        "pseudobulk_sha256": source.sha256,
        "preparation_sha256": preparation.preparation_sha256,
        "analysis_sha256": analysis_sha256,
        "group_value": preparation.group_value,
        "condition_key": preparation.condition_key,
        "numerator_condition": preparation.numerator_condition,
        "denominator_condition": preparation.denominator_condition,
        "design_type": preparation.design_type,
        "n_samples": len(preparation.included_positions),
        "n_numerator_replicates": len(preparation.numerator_replicates),
        "n_denominator_replicates": len(preparation.denominator_replicates),
        "design_rank": preparation.rank,
        "residual_degrees_of_freedom": preparation.residual_df,
        "warning_codes": warning_codes,
        "n_warnings": len(warning_codes),
        "n_input_features": len(source.feature_ids),
        "n_tested_features": int(r_result.tested_indices.size),
        "n_filtered_features": int(
            len(source.feature_ids) - r_result.tested_indices.size
        ),
        "filtering_method": "edgeR::filterByExpr",
        "normalization_method": "TMM",
        "backend_pipeline": EDGER_PIPELINE_ID,
        "production_r_script_sha256": production_sha256,
        "r_version": r_result.versions["r"],
        "bioconductor_version": r_result.versions["bioconductor"],
        "edger_version": r_result.versions["edger"],
        "package_versions": dict(r_result.versions),
    }


def verify_replicate_differential_accessibility(
    resolved_arguments: Mapping[str, object], result: Mapping[str, object]
) -> DAVerificationMetadata:
    """Reconstruct and recompute M8.2 without invoking production A/B code."""

    if not isinstance(resolved_arguments, Mapping) or not isinstance(result, Mapping):
        raise TypeError("DA verification requires argument and result mappings.")
    source = _verify_pseudobulk(resolved_arguments.get("pseudobulk_path"))
    preparation = _reconstruct_preparation(source, resolved_arguments)
    if not VERIFICATION_R_SCRIPT.is_file():
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE", "Verifier R script is unavailable."
        )
    if not PRODUCTION_R_SCRIPT.is_file():
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE", "Production R script is unavailable."
        )
    verifier_sha256 = hashlib.sha256(VERIFICATION_R_SCRIPT.read_bytes()).hexdigest()
    if verifier_sha256 != EXPECTED_VERIFICATION_R_SCRIPT_SHA256:
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE",
            "Independent verifier R script identity is incompatible.",
        )
    production_sha256 = hashlib.sha256(PRODUCTION_R_SCRIPT.read_bytes()).hexdigest()
    analysis_sha256 = _analysis_identity(preparation, production_sha256)
    _preflight_verifier_memory(preparation)
    output_dir = resolved_arguments.get("output_dir")
    if not isinstance(output_dir, (str, Path)):
        raise _error("RESULT_PATH_MISMATCH", "DA output directory is invalid.")
    expected_path = Path(output_dir).expanduser().resolve(strict=False) / (
        f"{source.path.stem}.da.{analysis_sha256}.h5ad"
    )
    result_path = result.get("da_path")
    if not isinstance(result_path, str):
        raise _error("RESULT_PATH_MISMATCH", "DA result path is invalid.")
    da_path = Path(result_path).expanduser().resolve()
    if da_path != expected_path or not da_path.is_file():
        raise _error("RESULT_PATH_MISMATCH", "DA result path differs.")
    da_sha256_before = _file_sha256(da_path)
    if result.get("da_sha256") != da_sha256_before:
        raise _error("ARTIFACT_SHA256_MISMATCH", "DA whole-file SHA differs.")

    with tempfile.TemporaryDirectory(prefix="agent-edger-verify-") as name:
        directory = Path(name)
        selected_counts_sha256 = _stage_verification(directory, preparation)
        r_result = _invoke_verifier_r(directory, preparation)
    digests = _result_digests(preparation, r_result)
    artifact: ad.AnnData | None = None
    try:
        artifact = ad.read_h5ad(da_path)
        provenance = artifact.uns.get(DA_PROVENANCE_KEY)
        if not isinstance(provenance, Mapping):
            raise _error("DA_ARTIFACT_INVALID", "DA provenance is unavailable.")
        memory_record = _validate_memory_record(
            provenance.get("memory_preflight"), preparation
        )
        expected_provenance = _expected_provenance(
            preparation,
            r_result,
            selected_counts_sha256=selected_counts_sha256,
            production_sha256=production_sha256,
            analysis_sha256=analysis_sha256,
            digests=digests,
            memory_record=memory_record,
        )
        _validate_da_artifact(
            artifact, preparation, r_result, expected_provenance
        )
    except DAVerificationError:
        raise
    except Exception as exc:
        raise _error("DA_ARTIFACT_INVALID", "DA artifact verification failed.") from exc
    finally:
        if artifact is not None:
            manager = getattr(artifact, "file", None)
            if manager is not None:
                manager.close()
    expected_result = _expected_result(
        preparation,
        r_result,
        da_path=da_path,
        da_sha256=da_sha256_before,
        production_sha256=production_sha256,
        analysis_sha256=analysis_sha256,
    )
    if _json_value(result) != _json_value(expected_result):
        raise _error("RESULT_IDENTITY_MISMATCH", "DA compact result changed.")
    if _file_sha256(source.path) != source.sha256:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed during DA verification."
        )
    if _file_sha256(da_path) != da_sha256_before:
        raise _error(
            "ARTIFACT_SHA256_MISMATCH", "DA artifact changed during verification."
        )
    return DAVerificationMetadata(
        verifier_sha256,
        digests["result_sha256"],
        preparation.preparation_sha256,
        analysis_sha256,
    )


__all__ = [
    "DAVerificationError",
    "DAVerificationMetadata",
    "EXPECTED_VERIFICATION_R_SCRIPT_SHA256",
    "VERIFICATION_R_SCRIPT",
    "verify_replicate_differential_accessibility",
]
