"""Deterministic Python preparation for replicate-aware accessibility testing.

Milestone 8.2-A stops at a validated numeric design and contrast.  This module
does not filter features, normalize counts, invoke R, fit a model, or publish a
differential-accessibility artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import anndata as ad
import numpy as np

from .replicate_pseudobulk import (
    M81ScientificError,
    PSEUDOBULK_ARTIFACT_TYPE,
    PSEUDOBULK_PROVENANCE_KEY,
    PSEUDOBULK_SCHEMA_VERSION,
    _canonical_covariate as _m81_canonical_covariate,
    _canonical_integer_chunk as _m81_canonical_integer_chunk,
    _file_sha256 as _m81_file_sha256,
    _output_value_semantics as _m81_output_value_semantics,
)


DA_PREPARATION_SCHEMA_VERSION = 1
DA_PREPARATION_TYPE = "agent.replicate-da-preparation"
DA_LOW_REPLICATION_WARNING = "DA_LOW_REPLICATION"
DA_ONE_CELL_PSEUDOBULK_WARNING = "DA_ONE_CELL_PSEUDOBULK"

_ELIGIBLE_MATRIX_SEMANTICS = frozenset({"fragment_counts", "insertion_counts"})
_DESIGN_TYPES = frozenset({"independent", "paired"})
_COVARIATE_KINDS = frozenset({"categorical", "numeric"})
_FLOAT64_EPSILON = float(np.finfo(np.float64).eps)


class M82ScientificError(ValueError):
    """M8.2 statistical-contract failure with a stable sanitized code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class DACovariateSpec:
    """One ordered additive covariate requested by the future public tool."""

    key: str
    kind: Literal["categorical", "numeric"]


@dataclass(frozen=True)
class DACategoricalLevel:
    """Type-preserving categorical level and its optional design column."""

    value_type: Literal["boolean", "integer", "float", "text"]
    value: bool | int | float | str
    design_column: str | None


@dataclass(frozen=True)
class DACovariateEncoding:
    """Reproducible encoding metadata for one included covariate."""

    key: str
    kind: Literal["categorical", "numeric"]
    source_column: str
    design_columns: tuple[str, ...]
    values: tuple[bool | int | float | str, ...]
    categorical_levels: tuple[DACategoricalLevel, ...]


@dataclass(frozen=True)
class DARowEligibility:
    """Deterministic inclusion state for one source pseudobulk row."""

    source_position: int
    pseudobulk_id: str
    group: str
    replicate: str
    condition: str
    n_cells: int
    library_size: int
    included: bool
    reason: Literal["included", "group_not_selected", "condition_not_selected"]


@dataclass(frozen=True)
class DAWarning:
    """Stable warning identity with immutable structured metadata."""

    code: Literal["DA_LOW_REPLICATION", "DA_ONE_CELL_PSEUDOBULK"]
    metadata: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class DifferentialAccessibilityPreparation:
    """Internal M8.2-A output; matrices intentionally stay out of run state."""

    schema_version: int
    preparation_type: str
    pseudobulk_path: str
    pseudobulk_sha256: str
    feature_space_identity_sha256: str
    feature_ids_sha256: str
    pseudobulk_matrix_sha256: str
    matrix_semantics: Literal["fragment_counts", "insertion_counts"]
    output_value_semantics: Literal["fragment_counts", "insertion_counts"]
    group_value: str
    condition_key: str
    numerator_condition: str
    denominator_condition: str
    design_type: Literal["independent", "paired"]
    source_row_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    row_eligibility: tuple[DARowEligibility, ...]
    included_source_positions: tuple[int, ...]
    included_pseudobulk_ids: tuple[str, ...]
    replicate_order: tuple[str, ...]
    numerator_replicates: tuple[str, ...]
    denominator_replicates: tuple[str, ...]
    covariate_specifications: tuple[DACovariateSpec, ...]
    covariate_encodings: tuple[DACovariateEncoding, ...]
    design_columns: tuple[str, ...]
    design_matrix: np.ndarray
    contrast: np.ndarray
    design_rank: int
    residual_degrees_of_freedom: int
    rank_tolerance: float
    estimability_tolerance: float
    warnings: tuple[DAWarning, ...]
    inclusion_sha256: str
    design_sha256: str
    contrast_sha256: str
    preparation_sha256: str


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
    source_covariate_keys: tuple[str, ...]
    source_covariate_columns: tuple[str, ...]
    row_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    groups: tuple[str, ...]
    replicates: tuple[str, ...]
    conditions: tuple[str, ...]
    n_cells: tuple[int, ...]
    library_sizes: tuple[int, ...]
    covariate_values: tuple[tuple[object, ...], ...]


def _error(code: str, message: str) -> M82ScientificError:
    return M82ScientificError(code, message)


def _strict_text(value: object, *, code: str, subject: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise _error(code, f"{subject} must be strict nonblank text.")
    return value


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite scientific value")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError("scientific mappings require string keys")
            converted[key] = _json_value(nested)
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(nested) for nested in value]
    raise TypeError("unsupported scientific value")


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
    if canonical.ndim != 2 or not np.isfinite(canonical).all():
        raise _error("DA_DESIGN_RANK_DEFICIENT", "Design values must be finite.")
    header = {
        "schema_version": 1,
        "shape": [int(canonical.shape[0]), int(canonical.shape[1])],
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


def _resolve_pseudobulk_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise M81ScientificError(
            "PSEUDOBULK_ARTIFACT_INVALID", "Pseudobulk path must be path-like."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise M81ScientificError(
            "PSEUDOBULK_ARTIFACT_INVALID", "Required pseudobulk H5AD is invalid."
        )
    return path


def _verified_pseudobulk(path_value: str | Path) -> _VerifiedPseudobulk:
    """Validate the supplied aggregate, without reconstructing its cell-level history.

    Accepted-output binding owns historical identity. This owner needs exact count
    values, ordered features, experimental units and the declared measurement.
    """
    from .replicate_pseudobulk import (
        _strict_ids, _ordered_strings_digest, _matrix_digest_and_nnz,
        _matrix_chunks, _source_matrix, _domain_digest, PSEUDOBULK_UNIT_DOMAIN,
    )
    path = _resolve_pseudobulk_path(path_value)
    digest_before = _m81_file_sha256(path)
    artifact = ad.read_h5ad(path, backed="r")
    try:
        provenance = _json_value(artifact.uns[PSEUDOBULK_PROVENANCE_KEY])
        if (provenance['schema_version'] != PSEUDOBULK_SCHEMA_VERSION
                or provenance['artifact_type'] != PSEUDOBULK_ARTIFACT_TYPE
                or provenance['stage'] != 'replicate_pseudobulk'):
            raise ValueError('Unsupported pseudobulk publication')
        biology, metadata = provenance['biology'], provenance['metadata']
        semantics = biology['matrix_semantics']
        if semantics not in _ELIGIBLE_MATRIX_SEMANTICS:
            raise _error('DA_MATRIX_SEMANTICS_INELIGIBLE', 'DA requires fragment or insertion counts.')
        output_semantics = _m81_output_value_semantics(semantics)
        if biology['output_value_semantics'] != output_semantics:
            raise ValueError('Pseudobulk measurement declaration changed')
        rows = _strict_ids(artifact.obs_names, 'pseudobulk')
        features = _strict_ids(artifact.var_names, 'feature')
        keys = tuple(metadata['covariate_keys']); columns = tuple(metadata['covariate_columns'])
        if (len(set(keys)) != len(keys) or columns != tuple(f'covariate_{i:03d}' for i in range(len(keys)))):
            raise ValueError('Pseudobulk covariate declarations changed')
        units = tuple(tuple(_strict_text(v, code='DA_METADATA_INVALID', subject='Experimental unit')
                            for v in values) for values in
                      artifact.obs[['group','replicate','condition']].itertuples(index=False, name=None))
        expected_ids = tuple('pb-' + _domain_digest(PSEUDOBULK_UNIT_DOMAIN,
            dict(feature_space_identity_sha256=provenance['source']['feature_space_identity_sha256'],
                 group=g, replicate=r, condition=c)) for g,r,c in units)
        if len(set(units)) != len(units) or rows != expected_ids:
            raise ValueError('Pseudobulk unit identity changed')
        n_cells = tuple(int(v) for v in artifact.obs['n_cells'])
        if any(v <= 0 or v != raw for v,raw in zip(n_cells, artifact.obs['n_cells'])):
            raise ValueError('Invalid pseudobulk cell counts')
        matrix = _source_matrix(artifact, 'X', None)
        logical, nnz = _matrix_digest_and_nnz(matrix, len(rows), len(features), output_semantics)
        libraries = []
        for first,last in _matrix_chunks(matrix):
            chunk = _m81_canonical_integer_chunk(matrix, first, last, output_semantics)
            for i in range(chunk.shape[0]):
                left,right = chunk.indptr[i:i+2]
                total = sum(int(v) for v in chunk.data[left:right])
                if total > np.iinfo(np.int64).max:
                    raise ValueError('Pseudobulk library size overflow')
                libraries.append(total)
        feature_digest = _ordered_strings_digest(features, domain='agent.regulatory-feature-identities.v1')
        digests = provenance['digests']; aggregation = provenance['aggregation']
        if (logical != digests['pseudobulk_matrix_sha256'] or feature_digest != digests['feature_ids_sha256']
                or tuple(libraries) != tuple(artifact.obs['library_size'])
                or any(aggregation[k] != v for k,v in dict(n_cells=sum(n_cells),n_features=len(features),
                    n_pseudobulks=len(rows),matrix_nnz=nnz,total_sum=sum(libraries)).items())):
            raise ValueError('Pseudobulk counts, features or summary changed')
        covariates = tuple(tuple(_m81_canonical_covariate(artifact.obs[col].iloc[i], key)
                          for key,col in zip(keys,columns)) for i in range(len(rows)))
        result = _VerifiedPseudobulk(path=path, sha256=digest_before,
            feature_space_identity_sha256=provenance['source']['feature_space_identity_sha256'],
            feature_ids_sha256=feature_digest, pseudobulk_matrix_sha256=logical,
            matrix_semantics=semantics, output_value_semantics=output_semantics,
            condition_key=metadata['condition_key'], source_covariate_keys=keys,
            source_covariate_columns=columns, row_ids=rows, feature_ids=features,
            groups=tuple(v[0] for v in units), replicates=tuple(v[1] for v in units),
            conditions=tuple(v[2] for v in units), n_cells=n_cells,
            library_sizes=tuple(libraries), covariate_values=covariates)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, (M81ScientificError, M82ScientificError)):
            raise
        raise M81ScientificError('PSEUDOBULK_ARTIFACT_INVALID',
                                 'Pseudobulk counts or experimental-unit metadata are invalid.') from exc
    finally:
        artifact.file.close()
    if _m81_file_sha256(path) != digest_before:
        raise M81ScientificError('SOURCE_CHANGED_DURING_READ', 'Pseudobulk changed during preparation.')
    return result


def _covariate_specifications(
    value: object,
    *,
    available_keys: Sequence[str],
) -> tuple[DACovariateSpec, ...]:
    if not isinstance(value, (list, tuple)):
        raise _error(
            "DA_COVARIATE_INVALID", "Covariates must be an ordered list or tuple."
        )
    specifications: list[DACovariateSpec] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"key", "kind"}:
            raise _error(
                "DA_COVARIATE_INVALID", "Each covariate specification is invalid."
            )
        key = _strict_text(
            item["key"], code="DA_COVARIATE_INVALID", subject="Covariate key"
        )
        kind = _strict_text(
            item["kind"], code="DA_COVARIATE_INVALID", subject="Covariate kind"
        )
        if kind not in _COVARIATE_KINDS:
            raise _error(
                "DA_COVARIATE_INVALID", "Covariate kind must be categorical or numeric."
            )
        specifications.append(DACovariateSpec(key, kind))
    keys = tuple(item.key for item in specifications)
    if len(set(keys)) != len(keys) or any(key not in available_keys for key in keys):
        raise _error(
            "DA_COVARIATE_INVALID", "Covariate keys must be unique frozen M8.1 covariates."
        )
    return tuple(specifications)


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
    raise _error("DA_COVARIATE_INVALID", "Categorical covariate value is invalid.")


def _encode_covariates(
    snapshot: _VerifiedPseudobulk,
    included_positions: Sequence[int],
    specifications: Sequence[DACovariateSpec],
) -> tuple[tuple[DACovariateEncoding, ...], tuple[str, ...], list[np.ndarray]]:
    encodings: list[DACovariateEncoding] = []
    design_columns: list[str] = []
    design_values: list[np.ndarray] = []
    key_to_index = {key: index for index, key in enumerate(snapshot.source_covariate_keys)}
    for covariate_index, specification in enumerate(specifications):
        source_index = key_to_index[specification.key]
        source_column = snapshot.source_covariate_columns[source_index]
        raw_values = tuple(
            snapshot.covariate_values[position][source_index]
            for position in included_positions
        )
        if specification.kind == "numeric":
            numeric: list[float] = []
            for value in raw_values:
                try:
                    canonical = _m81_canonical_covariate(value, specification.key)
                except M81ScientificError as exc:
                    raise _error(
                        "DA_COVARIATE_INVALID",
                        "Numeric covariate values must be finite numbers.",
                    ) from exc
                if isinstance(canonical, bool) or not isinstance(canonical, (int, float)):
                    raise _error(
                        "DA_COVARIATE_INVALID", "Numeric covariate values must be finite numbers."
                    )
                converted = float(canonical)
                if not math.isfinite(converted):
                    raise _error(
                        "DA_COVARIATE_INVALID", "Numeric covariate values must be finite numbers."
                    )
                numeric.append(converted)
            if len(set(numeric)) < 2:
                raise _error(
                    "DA_COVARIATE_INVARIANT", "Requested numeric covariate is invariant."
                )
            column = f"covariate_{covariate_index:03d}_numeric"
            design_columns.append(column)
            design_values.append(np.asarray(numeric, dtype=np.float64))
            encodings.append(
                DACovariateEncoding(
                    specification.key,
                    specification.kind,
                    source_column,
                    (column,),
                    tuple(numeric),
                    (),
                )
            )
            continue

        identities: list[tuple[str, bool | int | float | str]] = []
        levels: list[tuple[str, bool | int | float | str]] = []
        for value in raw_values:
            try:
                identity = _categorical_identity(value)
            except M81ScientificError as exc:
                raise _error(
                    "DA_COVARIATE_INVALID", "Categorical covariate values are invalid."
                ) from exc
            identities.append(identity)
            if identity not in levels:
                levels.append(identity)
        if len(levels) < 2:
            raise _error(
                "DA_COVARIATE_INVARIANT", "Requested categorical covariate is invariant."
            )
        level_records: list[DACategoricalLevel] = []
        generated_columns: list[str] = []
        for level_index, (value_type, level_value) in enumerate(levels):
            column = (
                None
                if level_index == 0
                else f"covariate_{covariate_index:03d}_level_{level_index:03d}"
            )
            level_records.append(
                DACategoricalLevel(value_type, level_value, column)
            )
            if column is not None:
                generated_columns.append(column)
                design_columns.append(column)
                design_values.append(
                    np.asarray(
                        [
                            1.0
                            if identity == (value_type, level_value)
                            else 0.0
                            for identity in identities
                        ],
                        dtype=np.float64,
                    )
                )
        encodings.append(
            DACovariateEncoding(
                specification.key,
                specification.kind,
                source_column,
                tuple(generated_columns),
                tuple(value for _, value in identities),
                tuple(level_records),
            )
        )
    return tuple(encodings), tuple(design_columns), design_values


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _warning_payload(warning: DAWarning) -> dict[str, object]:
    return {"code": warning.code, "metadata": dict(warning.metadata)}


def _preflight_design(
    design: np.ndarray,
    contrast: np.ndarray,
) -> tuple[int, int, float, float]:
    if design.ndim != 2 or contrast.ndim != 1 or contrast.shape[0] != design.shape[1]:
        raise _error("DA_DESIGN_RANK_DEFICIENT", "Design dimensions are invalid.")
    if not np.isfinite(design).all() or not np.isfinite(contrast).all():
        raise _error("DA_DESIGN_RANK_DEFICIENT", "Design values must be finite.")
    try:
        _, singular_values, right_vectors = np.linalg.svd(
            design, full_matrices=True
        )
    except np.linalg.LinAlgError as exc:
        raise _error(
            "DA_DESIGN_RANK_DEFICIENT", "Design rank could not be determined."
        ) from exc
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
        nonestimable = bool(
            null_space.size
            and float(np.linalg.norm(null_space @ contrast, ord=2))
            > estimability_tolerance
        )
        if nonestimable:
            raise _error(
                "DA_CONTRAST_NOT_ESTIMABLE", "Condition contrast is not estimable."
            )
        raise _error("DA_DESIGN_RANK_DEFICIENT", "Design matrix is rank deficient.")
    residual_degrees_of_freedom = design.shape[0] - design.shape[1]
    if design.shape[0] <= design.shape[1] or residual_degrees_of_freedom < 2:
        raise _error(
            "DA_RESIDUAL_DF_INSUFFICIENT", "Design has insufficient residual degrees of freedom."
        )
    return rank, residual_degrees_of_freedom, rank_tolerance, estimability_tolerance


def prepare_replicate_differential_accessibility(
    pseudobulk_path: str | Path,
    group_value: str,
    condition_key: str,
    numerator_condition: str,
    denominator_condition: str,
    design_type: Literal["independent", "paired"] | str,
    *,
    covariates: Sequence[Mapping[str, object]] = (),
) -> DifferentialAccessibilityPreparation:
    """Validate M8.1 input and construct the complete deterministic DA design."""

    snapshot = _verified_pseudobulk(pseudobulk_path)
    if (
        snapshot.matrix_semantics not in _ELIGIBLE_MATRIX_SEMANTICS
        or snapshot.output_value_semantics not in _ELIGIBLE_MATRIX_SEMANTICS
    ):
        raise _error(
            "DA_MATRIX_SEMANTICS_INELIGIBLE",
            "Pseudobulk matrix semantics are ineligible for edgeR QL.",
        )
    normalized_group = _strict_text(
        group_value, code="DA_GROUP_NOT_FOUND", subject="Group value"
    )
    normalized_condition_key = _strict_text(
        condition_key, code="DA_CONDITION_NOT_FOUND", subject="Condition key"
    )
    numerator = _strict_text(
        numerator_condition, code="DA_CONDITION_NOT_FOUND", subject="Numerator condition"
    )
    denominator = _strict_text(
        denominator_condition,
        code="DA_CONDITION_NOT_FOUND",
        subject="Denominator condition",
    )
    normalized_design_type = _strict_text(
        design_type, code="DA_DESIGN_INVALID", subject="Design type"
    )
    if normalized_design_type not in _DESIGN_TYPES:
        raise _error(
            "DA_DESIGN_INVALID", "Design type must be independent or paired."
        )
    if normalized_condition_key != snapshot.condition_key:
        raise _error(
            "DA_CONDITION_KEY_MISMATCH",
            "Requested condition key does not match M8.1 provenance.",
        )
    if numerator == denominator:
        raise _error(
            "DA_CONDITION_NOT_FOUND", "Numerator and denominator conditions must be distinct."
        )
    if normalized_group not in snapshot.groups:
        raise _error("DA_GROUP_NOT_FOUND", "Requested group is absent.")
    conditions_in_group = {
        condition
        for group, condition in zip(snapshot.groups, snapshot.conditions, strict=True)
        if group == normalized_group
    }
    if numerator not in conditions_in_group or denominator not in conditions_in_group:
        raise _error(
            "DA_CONDITION_NOT_FOUND", "Requested conditions are absent from the selected group."
        )

    row_eligibility: list[DARowEligibility] = []
    included_positions: list[int] = []
    for position, row_id in enumerate(snapshot.row_ids):
        group = snapshot.groups[position]
        condition = snapshot.conditions[position]
        if group != normalized_group:
            included = False
            reason = "group_not_selected"
        elif condition not in {numerator, denominator}:
            included = False
            reason = "condition_not_selected"
        else:
            included = True
            reason = "included"
            included_positions.append(position)
        row_eligibility.append(
            DARowEligibility(
                position,
                row_id,
                group,
                snapshot.replicates[position],
                condition,
                snapshot.n_cells[position],
                snapshot.library_sizes[position],
                included,
                reason,
            )
        )
    if any(snapshot.library_sizes[position] == 0 for position in included_positions):
        raise _error("DA_ZERO_LIBRARY", "An included pseudobulk has zero library size.")

    included_replicates = tuple(snapshot.replicates[position] for position in included_positions)
    included_conditions = tuple(snapshot.conditions[position] for position in included_positions)
    numerator_replicates = _ordered_unique(
        tuple(
            replicate
            for replicate, condition in zip(
                included_replicates, included_conditions, strict=True
            )
            if condition == numerator
        )
    )
    denominator_replicates = _ordered_unique(
        tuple(
            replicate
            for replicate, condition in zip(
                included_replicates, included_conditions, strict=True
            )
            if condition == denominator
        )
    )
    replicate_order = _ordered_unique(included_replicates)
    warnings: list[DAWarning] = []
    if normalized_design_type == "independent":
        if len(numerator_replicates) < 2 or len(denominator_replicates) < 2:
            raise _error(
                "DA_REPLICATION_INSUFFICIENT",
                "Independent design requires two replicates per condition.",
            )
        if set(numerator_replicates).intersection(denominator_replicates):
            raise _error(
                "DA_PAIRING_INVALID",
                "Independent-design replicate identifiers must be disjoint.",
            )
        if len(numerator_replicates) == 2 or len(denominator_replicates) == 2:
            warnings.append(
                DAWarning(
                    DA_LOW_REPLICATION_WARNING,
                    (
                        ("numerator_replicates", len(numerator_replicates)),
                        ("denominator_replicates", len(denominator_replicates)),
                        ("recommended_minimum_per_condition", 3),
                    ),
                )
            )
    else:
        pair_counts: dict[tuple[str, str], int] = {}
        for replicate, condition in zip(
            included_replicates, included_conditions, strict=True
        ):
            key = (replicate, condition)
            pair_counts[key] = pair_counts.get(key, 0) + 1
        if any(count != 1 for count in pair_counts.values()):
            raise _error(
                "DA_PAIRING_INVALID", "Paired design contains duplicate observations."
            )
        if set(numerator_replicates) != set(denominator_replicates):
            raise _error(
                "DA_PAIRING_INVALID", "Paired-design replicate sets do not match."
            )
        if len(numerator_replicates) < 3:
            raise _error(
                "DA_REPLICATION_INSUFFICIENT", "Paired design requires three complete pairs."
            )

    one_cell_positions = tuple(
        position for position in included_positions if snapshot.n_cells[position] == 1
    )
    if one_cell_positions:
        warnings.append(
            DAWarning(
                DA_ONE_CELL_PSEUDOBULK_WARNING,
                (
                    ("pseudobulk_count", len(one_cell_positions)),
                    (
                        "pseudobulk_ids",
                        tuple(
                            snapshot.row_ids[position]
                            for position in one_cell_positions
                        ),
                    ),
                    (
                        "cell_counts",
                        tuple(
                            snapshot.n_cells[position]
                            for position in one_cell_positions
                        ),
                    ),
                ),
            )
        )

    specifications = _covariate_specifications(
        covariates, available_keys=snapshot.source_covariate_keys
    )
    condition_values = np.asarray(
        [1.0 if condition == numerator else 0.0 for condition in included_conditions],
        dtype=np.float64,
    )
    design_columns: list[str] = ["intercept", "condition_numerator"]
    columns: list[np.ndarray] = [
        np.ones(len(included_positions), dtype=np.float64),
        condition_values,
    ]
    if normalized_design_type == "paired":
        for replicate_index, replicate in enumerate(replicate_order[1:], start=1):
            design_columns.append(f"replicate_{replicate_index:03d}")
            columns.append(
                np.asarray(
                    [1.0 if value == replicate else 0.0 for value in included_replicates],
                    dtype=np.float64,
                )
            )
    covariate_encodings, covariate_columns, covariate_values = _encode_covariates(
        snapshot, included_positions, specifications
    )
    design_columns.extend(covariate_columns)
    columns.extend(covariate_values)
    design = np.column_stack(columns).astype(np.float64, copy=False)
    contrast = np.zeros(design.shape[1], dtype=np.float64)
    contrast[1] = 1.0
    rank, residual_df, rank_tolerance, estimability_tolerance = _preflight_design(
        design, contrast
    )
    design.setflags(write=False)
    contrast.setflags(write=False)

    included_ids = tuple(snapshot.row_ids[position] for position in included_positions)
    inclusion_payload = {
        "schema_version": DA_PREPARATION_SCHEMA_VERSION,
        "pseudobulk_sha256": snapshot.sha256,
        "feature_space_identity_sha256": snapshot.feature_space_identity_sha256,
        "group": normalized_group,
        "condition_key": normalized_condition_key,
        "numerator": numerator,
        "denominator": denominator,
        "design_type": normalized_design_type,
        "rows": [
            {
                "source_position": row.source_position,
                "pseudobulk_id": row.pseudobulk_id,
                "group": row.group,
                "replicate": row.replicate,
                "condition": row.condition,
                "n_cells": row.n_cells,
                "library_size": row.library_size,
                "included": row.included,
                "reason": row.reason,
            }
            for row in row_eligibility
        ],
        "replicate_order": list(replicate_order),
        "covariates": [
            {"key": specification.key, "kind": specification.kind}
            for specification in specifications
        ],
        "warnings": [_warning_payload(warning) for warning in warnings],
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
                    "value_type": level.value_type,
                    "value": level.value,
                    "design_column": level.design_column,
                }
                for level in encoding.categorical_levels
            ],
        }
        for encoding in covariate_encodings
    ]
    preparation_sha256 = _domain_digest(
        "agent.replicate-da-preparation.v1",
        {
            "pseudobulk_sha256": snapshot.sha256,
            "feature_space_identity_sha256": snapshot.feature_space_identity_sha256,
            "feature_ids_sha256": snapshot.feature_ids_sha256,
            "pseudobulk_matrix_sha256": snapshot.pseudobulk_matrix_sha256,
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
            "warnings": [_warning_payload(warning) for warning in warnings],
        },
    )
    return DifferentialAccessibilityPreparation(
        schema_version=DA_PREPARATION_SCHEMA_VERSION,
        preparation_type=DA_PREPARATION_TYPE,
        pseudobulk_path=str(snapshot.path),
        pseudobulk_sha256=snapshot.sha256,
        feature_space_identity_sha256=snapshot.feature_space_identity_sha256,
        feature_ids_sha256=snapshot.feature_ids_sha256,
        pseudobulk_matrix_sha256=snapshot.pseudobulk_matrix_sha256,
        matrix_semantics=snapshot.matrix_semantics,
        output_value_semantics=snapshot.output_value_semantics,
        group_value=normalized_group,
        condition_key=normalized_condition_key,
        numerator_condition=numerator,
        denominator_condition=denominator,
        design_type=normalized_design_type,
        source_row_ids=snapshot.row_ids,
        feature_ids=snapshot.feature_ids,
        row_eligibility=tuple(row_eligibility),
        included_source_positions=tuple(included_positions),
        included_pseudobulk_ids=included_ids,
        replicate_order=replicate_order,
        numerator_replicates=numerator_replicates,
        denominator_replicates=denominator_replicates,
        covariate_specifications=specifications,
        covariate_encodings=covariate_encodings,
        design_columns=tuple(design_columns),
        design_matrix=design,
        contrast=contrast,
        design_rank=rank,
        residual_degrees_of_freedom=residual_df,
        rank_tolerance=rank_tolerance,
        estimability_tolerance=estimability_tolerance,
        warnings=tuple(warnings),
        inclusion_sha256=inclusion_sha256,
        design_sha256=design_sha256,
        contrast_sha256=contrast_sha256,
        preparation_sha256=preparation_sha256,
    )


__all__ = [
    "DACategoricalLevel",
    "DACovariateEncoding",
    "DACovariateSpec",
    "DA_LOW_REPLICATION_WARNING",
    "DA_ONE_CELL_PSEUDOBULK_WARNING",
    "DA_PREPARATION_TYPE",
    "DA_PREPARATION_SCHEMA_VERSION",
    "DARowEligibility",
    "DAWarning",
    "DifferentialAccessibilityPreparation",
    "M82ScientificError",
    "prepare_replicate_differential_accessibility",
]
