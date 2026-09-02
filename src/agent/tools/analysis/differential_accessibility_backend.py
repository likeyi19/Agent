"""Pinned edgeR execution and DA artifact publication for Milestone 8.2-B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import MappingProxyType
from typing import Literal, TypedDict

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .differential_accessibility import (
    DAWarning,
    DifferentialAccessibilityPreparation,
    M82ScientificError,
    _json_value,
    prepare_replicate_differential_accessibility,
)
from .replicate_pseudobulk import (
    _file_sha256 as _m81_file_sha256,
    _software_versions as _m81_software_versions,
)


DA_ARTIFACT_TYPE = "agent.replicate-differential-accessibility"
DA_ARTIFACT_SCHEMA_VERSION = 1
DA_PROVENANCE_KEY = "agent_milestone8_differential_accessibility"
EDGER_PROTOCOL_VERSION = 1
EDGER_PIPELINE_ID = "agent.edger-ql-native-v4.v1"
EDGER_RSCRIPT_ENVIRONMENT_VARIABLE = "AGENT_EDGER_RSCRIPT"
PRODUCTION_R_SCRIPT = Path(__file__).resolve().parent / "r" / "edger_ql_v1.R"

FILTER_CONFIGURATION = MappingProxyType(
    {
        "method": "edgeR::filterByExpr",
        "grouping": "condition_denominator_0_numerator_1",
        "min_count": 10,
        "min_total_count": 15,
        "large_n": 10,
        "min_prop": 0.7,
    }
)
NORMALIZATION_CONFIGURATION = MappingProxyType(
    {
        "method": "edgeR::normLibSizes",
        "method_argument": "TMM",
        "reference_column": "automatic",
        "logratio_trim": 0.30,
        "sum_trim": 0.05,
        "do_weighting": True,
        "a_cutoff": -1e10,
    }
)
QL_CONFIGURATION = MappingProxyType(
    {
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
)
EXPECTED_BACKEND_VERSIONS = MappingProxyType(
    {
        "r": "4.6.1",
        "bioconductor": "3.23",
        "biocmanager": "1.30.27",
        "edger": "4.10.4",
        "limma": "3.68.5",
        "locfit": "1.5.9.12",
        "statmod": "1.5.2",
        "lattice": "0.23.1",
    }
)

_MAX_CAPTURE_BYTES = 64 * 1024
_MAX_STATUS_BYTES = 64 * 1024
_DENSE_WORKING_COPIES = 12
_FEATURE_RESULT_COLUMNS = 6
_DESIGN_WORKING_COPIES = 4
_MEMORY_SAFETY_FACTOR = 1.25
_FIXED_MEMORY_OVERHEAD_BYTES = 64 * 1024 * 1024
_MINIMUM_MEMORY_RESERVE_BYTES = 512 * 1024 * 1024
_MAXIMUM_AVAILABLE_FRACTION = 0.70
_MAX_EXACT_FLOAT64_INTEGER = 2**53

_ERROR_CODES = frozenset(
    {
        "DA_NO_FEATURES_AFTER_FILTER",
        "DA_FILTERED_LIBRARY_ZERO",
        "RSCRIPT_UNAVAILABLE",
        "EDGER_PACKAGE_UNAVAILABLE",
        "EDGER_VERSION_UNSUPPORTED",
        "R_PACKAGE_VERSION_INCOMPATIBLE",
        "R_BACKEND_EXECUTION_FAILED",
        "R_BACKEND_PROTOCOL_INVALID",
        "DA_NUMERICAL_RESULT_INVALID",
        "HOST_MEMORY_EXHAUSTED",
        "ARTIFACT_WRITE_FAILED",
        "DISK_FULL",
        "SOURCE_CHANGED_DURING_READ",
    }
)


class ReplicateDifferentialAccessibilityToolResult(TypedDict):
    status: Literal["success"]
    da_path: str
    da_sha256: str
    artifact_type: str
    artifact_schema_version: int
    pseudobulk_path: str
    pseudobulk_sha256: str
    preparation_sha256: str
    analysis_sha256: str
    group_value: str
    condition_key: str
    numerator_condition: str
    denominator_condition: str
    design_type: str
    n_samples: int
    n_numerator_replicates: int
    n_denominator_replicates: int
    design_rank: int
    residual_degrees_of_freedom: int
    warning_codes: list[str]
    n_warnings: int
    n_input_features: int
    n_tested_features: int
    n_filtered_features: int
    filtering_method: str
    normalization_method: str
    backend_pipeline: str
    production_r_script_sha256: str
    r_version: str
    bioconductor_version: str
    edger_version: str
    package_versions: dict[str, str]


@dataclass(frozen=True)
class HostMemoryAssessment:
    n_features: int
    n_samples: int
    n_design_columns: int
    dense_matrix_bytes: int
    base_estimate_bytes: int
    safety_factor: float
    estimated_peak_bytes: int
    available_bytes: int
    reserved_bytes: int
    usable_bytes: int


@dataclass(frozen=True)
class _BackendResult:
    filter_mask: np.ndarray
    tested_indices: np.ndarray
    post_filter_library_sizes: np.ndarray
    normalization_factors: np.ndarray
    effective_library_sizes: np.ndarray
    statistics: np.ndarray
    versions: dict[str, str]


def _error(code: str, message: str) -> M82ScientificError:
    if code not in _ERROR_CODES:
        raise RuntimeError("Unregistered M8.2-B error code.")
    return M82ScientificError(code, message)


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


def _production_r_script_sha256() -> str:
    if not PRODUCTION_R_SCRIPT.is_file():
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE", "Production R script is unavailable."
        )
    return hashlib.sha256(PRODUCTION_R_SCRIPT.read_bytes()).hexdigest()


def _resolve_rscript() -> Path:
    raw = os.environ.get(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE)
    if raw is None or not raw.strip():
        raise _error("RSCRIPT_UNAVAILABLE", "Approved Rscript is not configured.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise _error("RSCRIPT_UNAVAILABLE", "Approved Rscript path must be absolute.")
    path = candidate.resolve()
    if not path.is_file() or not os.access(path, os.X_OK) or path.name != "Rscript":
        raise _error("RSCRIPT_UNAVAILABLE", "Approved Rscript is unavailable.")
    return path


def _host_available_memory_bytes() -> int:
    candidates: list[int] = []
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                fields = line.split()
                candidates.append(int(fields[1]) * 1024)
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
            "HOST_MEMORY_EXHAUSTED", "Available host memory could not be determined."
        )
    return min(candidates)


def assess_host_memory(
    n_features: int,
    n_samples: int,
    n_design_columns: int,
    *,
    available_bytes: int | None = None,
) -> HostMemoryAssessment:
    """Apply the frozen conservative edgeR host-memory policy."""

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (n_features, n_samples, n_design_columns)
    ):
        raise ValueError("Memory dimensions must be positive integers.")
    available = (
        _host_available_memory_bytes()
        if available_bytes is None
        else available_bytes
    )
    if not isinstance(available, int) or isinstance(available, bool) or available <= 0:
        raise ValueError("Available memory must be a positive integer.")
    dense_bytes = n_features * n_samples * 8
    feature_result_bytes = n_features * _FEATURE_RESULT_COLUMNS * 8
    design_bytes = n_samples * n_design_columns * 8
    base = (
        dense_bytes * _DENSE_WORKING_COPIES
        + feature_result_bytes
        + design_bytes * _DESIGN_WORKING_COPIES
        + _FIXED_MEMORY_OVERHEAD_BYTES
    )
    estimated = math.ceil(base * _MEMORY_SAFETY_FACTOR)
    reserve = max(
        _MINIMUM_MEMORY_RESERVE_BYTES,
        math.ceil(available * (1.0 - _MAXIMUM_AVAILABLE_FRACTION)),
    )
    usable = max(0, available - reserve)
    assessment = HostMemoryAssessment(
        n_features,
        n_samples,
        n_design_columns,
        dense_bytes,
        base,
        _MEMORY_SAFETY_FACTOR,
        estimated,
        available,
        reserve,
        usable,
    )
    if estimated > usable:
        raise _error(
            "HOST_MEMORY_EXHAUSTED", "Host memory preflight rejected edgeR execution."
        )
    return assessment


def _read_status(directory: Path) -> dict[str, str]:
    path = directory / "backend_status.tsv"
    try:
        if not path.is_file() or path.stat().st_size > _MAX_STATUS_BYTES:
            raise ValueError
        lines = path.read_text(encoding="utf-8").splitlines()
        fields: dict[str, str] = {}
        for line in lines:
            pieces = line.split("\t")
            if len(pieces) != 2 or not pieces[0] or pieces[0] in fields:
                raise ValueError
            fields[pieces[0]] = pieces[1]
        if fields.get("protocol_version") != str(EDGER_PROTOCOL_VERSION):
            raise ValueError
        return fields
    except (OSError, UnicodeError, ValueError) as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend status protocol is invalid."
        ) from exc


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


def _invoke_fixed_script(rscript: Path, mode: str, directory: Path) -> dict[str, str]:
    stdout_path = directory / "backend_stdout.log"
    stderr_path = directory / "backend_stderr.log"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [
                    str(rscript),
                    "--vanilla",
                    str(PRODUCTION_R_SCRIPT),
                    mode,
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
        raise _error("RSCRIPT_UNAVAILABLE", "Approved Rscript is unavailable.") from exc
    except OSError as exc:
        raise _error(
            "R_BACKEND_EXECUTION_FAILED", "R backend process could not be executed."
        ) from exc
    try:
        if (
            stdout_path.stat().st_size > _MAX_CAPTURE_BYTES
            or stderr_path.stat().st_size > _MAX_CAPTURE_BYTES
        ):
            raise _error(
                "R_BACKEND_PROTOCOL_INVALID", "R backend emitted oversized diagnostics."
            )
    except OSError as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend diagnostics are unavailable."
        ) from exc
    try:
        status = _read_status(directory)
    except M82ScientificError as exc:
        if return_code != 0:
            raise _error(
                "R_BACKEND_EXECUTION_FAILED", "R backend process failed."
            ) from exc
        raise
    if status.get("status") == "error":
        code = status.get("error_code")
        if set(status) != {"protocol_version", "status", "error_code"} or code not in _ERROR_CODES:
            raise _error(
                "R_BACKEND_PROTOCOL_INVALID", "R backend error protocol is invalid."
            )
        raise _error(str(code), "R backend reported a controlled failure.")
    if return_code != 0:
        raise _error("R_BACKEND_EXECUTION_FAILED", "R backend process failed.")
    if status.get("status") != "success":
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend success protocol is invalid."
        )
    return status


def _versions_from_status(status: Mapping[str, str], *, mode: str) -> dict[str, str]:
    field_names = {
        "r": "r_version",
        "bioconductor": "bioconductor_version",
        "biocmanager": "biocmanager_version",
        "edger": "edger_version",
        "limma": "limma_version",
        "locfit": "locfit_version",
        "statmod": "statmod_version",
        "lattice": "lattice_version",
    }
    versions = {name: status.get(field, "") for name, field in field_names.items()}
    if status.get("mode") != mode or versions != dict(EXPECTED_BACKEND_VERSIONS):
        edge_version = versions.get("edger")
        if edge_version and edge_version != EXPECTED_BACKEND_VERSIONS["edger"]:
            raise _error("EDGER_VERSION_UNSUPPORTED", "edgeR version is unsupported.")
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE", "R package stack is incompatible."
        )
    return versions


def _probe_backend(rscript: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="agent-edger-probe-") as name:
        directory = Path(name)
        status = _invoke_fixed_script(rscript, "probe", directory)
        expected_fields = {
            "protocol_version",
            "status",
            "mode",
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
            raise _error(
                "R_BACKEND_PROTOCOL_INVALID", "R backend probe fields are invalid."
            )
        return _versions_from_status(status, mode="probe")


def _write_binary(path: Path, values: np.ndarray, dtype: str) -> bytes:
    canonical = np.ascontiguousarray(values, dtype=dtype)
    payload = canonical.tobytes(order="C")
    try:
        with path.open("wb") as handle:
            handle.write(payload)
    except OSError as exc:
        code = "DISK_FULL" if exc.errno == errno.ENOSPC else "R_BACKEND_EXECUTION_FAILED"
        raise _error(code, "Unable to stage edgeR input.") from exc
    return payload


def _stage_inputs(
    directory: Path,
    preparation: DifferentialAccessibilityPreparation,
) -> str:
    """Stage exact prepared samples without densifying the raw cell matrix."""

    source_path = Path(preparation.pseudobulk_path)
    if _m81_file_sha256(source_path) != preparation.pseudobulk_sha256:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed before edgeR staging."
        )
    artifact: ad.AnnData | None = None
    counts_path = directory / "counts.bin"
    digest = hashlib.sha256()
    digest.update(b"agent.replicate-da-selected-counts.v1\0")
    digest.update(
        _canonical_json_bytes(
            {
                "shape": [
                    len(preparation.included_source_positions),
                    len(preparation.feature_ids),
                ],
                "dtype": "float64-little-endian",
                "row_ids": list(preparation.included_pseudobulk_ids),
                "feature_ids_sha256": preparation.feature_ids_sha256,
            }
        )
    )
    digest.update(b"\n")
    try:
        artifact = ad.read_h5ad(source_path, backed="r")
        if (
            artifact.X is None
            or not isinstance(artifact.X, ad.abc.CSRDataset)
            or tuple(str(value) for value in artifact.obs_names)
            != preparation.source_row_ids
            or tuple(str(value) for value in artifact.var_names)
            != preparation.feature_ids
        ):
            raise _error(
                "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed before edgeR staging."
            )
        with counts_path.open("wb") as handle:
            for position in preparation.included_source_positions:
                row = artifact.X[position : position + 1]
                if not sparse.issparse(row):
                    raise _error(
                        "SOURCE_CHANGED_DURING_READ", "Pseudobulk storage changed."
                    )
                canonical = row.tocsr(copy=True)
                canonical.sum_duplicates()
                canonical.sort_indices()
                canonical.eliminate_zeros()
                if canonical.data.size and (
                    np.any(canonical.data < 0)
                    or np.any(canonical.data > _MAX_EXACT_FLOAT64_INTEGER)
                ):
                    raise _error(
                        "DA_NUMERICAL_RESULT_INVALID",
                        "Selected counts exceed exact edgeR numeric representation.",
                    )
                dense_row = np.asarray(
                    canonical.toarray(), dtype="<f8", order="C"
                ).reshape(-1)
                payload = dense_row.tobytes(order="C")
                handle.write(payload)
                digest.update(payload)
    except M82ScientificError:
        raise
    except OSError as exc:
        code = "DISK_FULL" if exc.errno == errno.ENOSPC else "R_BACKEND_EXECUTION_FAILED"
        raise _error(code, "Unable to stage edgeR count input.") from exc
    except Exception as exc:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk could not be staged."
        ) from exc
    finally:
        if artifact is not None:
            artifact.file.close()

    design = np.asarray(preparation.design_matrix, dtype="<f8", order="C")
    contrast = np.asarray(preparation.contrast, dtype="<f8", order="C")
    conditions = np.asarray(
        [
            1
            if preparation.row_eligibility[position].condition
            == preparation.numerator_condition
            else 0
            for position in preparation.included_source_positions
        ],
        dtype="<i4",
    )
    _write_binary(directory / "design.bin", design, "<f8")
    _write_binary(directory / "contrast.bin", contrast, "<f8")
    _write_binary(directory / "condition.bin", conditions, "<i4")
    selected_counts_sha256 = digest.hexdigest()
    manifest = (
        ("protocol_version", str(EDGER_PROTOCOL_VERSION)),
        ("preparation_sha256", preparation.preparation_sha256),
        ("n_features", str(len(preparation.feature_ids))),
        ("n_samples", str(len(preparation.included_source_positions))),
        ("n_design_columns", str(len(preparation.design_columns))),
        ("design_sha256", preparation.design_sha256),
        ("contrast_sha256", preparation.contrast_sha256),
        ("input_matrix_sha256", selected_counts_sha256),
    )
    try:
        (directory / "input_manifest.tsv").write_text(
            "".join(f"{key}\t{value}\n" for key, value in manifest),
            encoding="utf-8",
        )
    except OSError as exc:
        code = "DISK_FULL" if exc.errno == errno.ENOSPC else "R_BACKEND_EXECUTION_FAILED"
        raise _error(code, "Unable to stage edgeR manifest.") from exc
    if _m81_file_sha256(source_path) != preparation.pseudobulk_sha256:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed during edgeR staging."
        )
    return selected_counts_sha256


def _read_binary(
    path: Path,
    *,
    dtype: str,
    count: int,
    code: str = "R_BACKEND_PROTOCOL_INVALID",
) -> np.ndarray:
    expected_bytes = np.dtype(dtype).itemsize * count
    try:
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError
        values = np.fromfile(path, dtype=dtype, count=count)
    except (OSError, ValueError) as exc:
        raise _error(code, "R backend binary output is invalid.") from exc
    if values.size != count:
        raise _error(code, "R backend binary output is truncated.")
    return values


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    count = p_values.size
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(count, dtype=np.float64)
    result[order] = adjusted
    return result


def _read_backend_result(
    directory: Path,
    preparation: DifferentialAccessibilityPreparation,
    status: Mapping[str, str],
) -> _BackendResult:
    expected_fields = {
        "protocol_version",
        "status",
        "mode",
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
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend result fields are invalid."
        )
    versions = _versions_from_status(status, mode="run")
    try:
        n_features = int(status["n_features"])
        n_samples = int(status["n_samples"])
        n_tested = int(status["n_tested"])
        n_filtered = int(status["n_filtered"])
    except (KeyError, ValueError) as exc:
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend dimensions are invalid."
        ) from exc
    if (
        n_features != len(preparation.feature_ids)
        or n_samples != len(preparation.included_source_positions)
        or n_tested <= 0
        or n_tested + n_filtered != n_features
    ):
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend dimensions differ from preparation."
        )
    mask_values = _read_binary(
        directory / "filter_mask.bin", dtype="<i4", count=n_features
    )
    if not np.all(np.isin(mask_values, (0, 1))):
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend filter mask is invalid."
        )
    filter_mask = mask_values.astype(bool)
    tested_indices = _read_binary(
        directory / "tested_indices.bin", dtype="<i4", count=n_tested
    ).astype(np.int64)
    expected_indices = np.flatnonzero(filter_mask)
    if not np.array_equal(tested_indices, expected_indices):
        raise _error(
            "R_BACKEND_PROTOCOL_INVALID", "R backend tested-feature order is invalid."
        )
    post_filter_library_sizes = _read_binary(
        directory / "post_filter_library_sizes.bin",
        dtype="<f8",
        count=n_samples,
    )
    normalization_factors = _read_binary(
        directory / "normalization_factors.bin", dtype="<f8", count=n_samples
    )
    effective_library_sizes = _read_binary(
        directory / "effective_library_sizes.bin", dtype="<f8", count=n_samples
    )
    statistics = _read_binary(
        directory / "statistics.bin", dtype="<f8", count=n_tested * 5
    ).reshape(n_tested, 5)
    if (
        not np.isfinite(post_filter_library_sizes).all()
        or np.any(post_filter_library_sizes <= 0)
    ):
        raise _error(
            "DA_FILTERED_LIBRARY_ZERO", "Post-filter library sizes are invalid."
        )
    if (
        not np.isfinite(normalization_factors).all()
        or np.any(normalization_factors <= 0)
        or not np.isfinite(effective_library_sizes).all()
        or np.any(effective_library_sizes <= 0)
        or not np.allclose(
            effective_library_sizes,
            post_filter_library_sizes * normalization_factors,
            rtol=1e-12,
            atol=1e-12,
        )
        or abs(float(np.sum(np.log(normalization_factors)))) > 1e-10
    ):
        raise _error(
            "DA_NUMERICAL_RESULT_INVALID", "TMM normalization results are invalid."
        )
    if (
        not np.isfinite(statistics).all()
        or np.any(statistics[:, 2] < 0)
        or np.any(statistics[:, 3:] < 0)
        or np.any(statistics[:, 3:] > 1)
    ):
        raise _error(
            "DA_NUMERICAL_RESULT_INVALID", "Differential statistics are invalid."
        )
    expected_fdr = _bh_adjust(statistics[:, 3])
    if not np.allclose(
        statistics[:, 4], expected_fdr, rtol=1e-12, atol=1e-300
    ):
        raise _error(
            "DA_NUMERICAL_RESULT_INVALID", "BH-adjusted values are invalid."
        )
    for values in (
        filter_mask,
        tested_indices,
        post_filter_library_sizes,
        normalization_factors,
        effective_library_sizes,
        statistics,
    ):
        values.setflags(write=False)
    return _BackendResult(
        filter_mask,
        tested_indices,
        post_filter_library_sizes,
        normalization_factors,
        effective_library_sizes,
        statistics,
        versions,
    )


def _warning_summary(warning: DAWarning) -> dict[str, object]:
    metadata = dict(warning.metadata)
    if warning.code == "DA_ONE_CELL_PSEUDOBULK":
        return {
            "code": warning.code,
            "pseudobulk_count": int(metadata["pseudobulk_count"]),
        }
    return {"code": warning.code, **metadata}


def _warning_provenance(
    warnings: tuple[DAWarning, ...],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for warning in warnings:
        summary = _warning_summary(warning)
        code = str(summary.pop("code"))
        records[code] = summary
    return records


def _result_digests(
    preparation: DifferentialAccessibilityPreparation,
    backend: _BackendResult,
) -> dict[str, str]:
    filter_digest = _binary_digest(
        "agent.replicate-da-filter-mask.v1",
        backend.filter_mask.astype(np.uint8),
        dtype="u1",
        metadata={"feature_ids_sha256": preparation.feature_ids_sha256},
    )
    post_library_digest = _binary_digest(
        "agent.replicate-da-postfilter-library-sizes.v1",
        backend.post_filter_library_sizes,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_pseudobulk_ids)},
    )
    normalization_digest = _binary_digest(
        "agent.replicate-da-normalization-factors.v1",
        backend.normalization_factors,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_pseudobulk_ids)},
    )
    effective_library_digest = _binary_digest(
        "agent.replicate-da-effective-library-sizes.v1",
        backend.effective_library_sizes,
        dtype="<f8",
        metadata={"row_ids": list(preparation.included_pseudobulk_ids)},
    )
    statistical_digest = _binary_digest(
        "agent.replicate-da-tested-statistics.v1",
        backend.statistics,
        dtype="<f8",
        metadata={
            "feature_ids_sha256": preparation.feature_ids_sha256,
            "tested_indices": backend.tested_indices.tolist(),
            "columns": ["logFC", "logCPM", "F", "PValue", "FDR"],
        },
    )
    result_digest = _domain_digest(
        "agent.replicate-da-result.v1",
        {
            "preparation_sha256": preparation.preparation_sha256,
            "filter_mask_sha256": filter_digest,
            "post_filter_library_sizes_sha256": post_library_digest,
            "normalization_factors_sha256": normalization_digest,
            "effective_library_sizes_sha256": effective_library_digest,
            "tested_statistics_sha256": statistical_digest,
        },
    )
    return {
        "filter_mask_sha256": filter_digest,
        "post_filter_library_sizes_sha256": post_library_digest,
        "normalization_factors_sha256": normalization_digest,
        "effective_library_sizes_sha256": effective_library_digest,
        "tested_statistics_sha256": statistical_digest,
        "result_sha256": result_digest,
    }


def _analysis_identity(
    preparation: DifferentialAccessibilityPreparation,
    *,
    production_script_sha256: str,
) -> str:
    return _domain_digest(
        "agent.replicate-differential-accessibility-analysis.v1",
        {
            "pseudobulk_sha256": preparation.pseudobulk_sha256,
            "feature_space_identity_sha256": (
                preparation.feature_space_identity_sha256
            ),
            "preparation_sha256": preparation.preparation_sha256,
            "pipeline": EDGER_PIPELINE_ID,
            "filter": dict(FILTER_CONFIGURATION),
            "normalization": dict(NORMALIZATION_CONFIGURATION),
            "ql": dict(QL_CONFIGURATION),
            "production_r_script_sha256": production_script_sha256,
            "backend_version_policy": dict(EXPECTED_BACKEND_VERSIONS),
        },
    )


def _artifact_provenance(
    preparation: DifferentialAccessibilityPreparation,
    backend: _BackendResult,
    memory: HostMemoryAssessment,
    *,
    selected_counts_sha256: str,
    production_script_sha256: str,
    analysis_sha256: str,
    digests: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": DA_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": DA_ARTIFACT_TYPE,
        "stage": "replicate_differential_accessibility",
        "analysis_sha256": analysis_sha256,
        "source": {
            "pseudobulk_path": preparation.pseudobulk_path,
            "pseudobulk_sha256": preparation.pseudobulk_sha256,
            "feature_space_identity_sha256": (
                preparation.feature_space_identity_sha256
            ),
            "feature_ids_sha256": preparation.feature_ids_sha256,
            "pseudobulk_matrix_sha256": preparation.pseudobulk_matrix_sha256,
            "matrix_semantics": preparation.matrix_semantics,
            "output_value_semantics": preparation.output_value_semantics,
            "selected_counts_sha256": selected_counts_sha256,
        },
        "comparison": {
            "group_value": preparation.group_value,
            "condition_key": preparation.condition_key,
            "numerator_condition": preparation.numerator_condition,
            "denominator_condition": preparation.denominator_condition,
            "positive_logfc_meaning": "higher_in_numerator",
            "design_type": preparation.design_type,
            "n_samples": len(preparation.included_source_positions),
            "n_numerator_replicates": len(preparation.numerator_replicates),
            "n_denominator_replicates": len(preparation.denominator_replicates),
            "warnings": _warning_provenance(preparation.warnings),
        },
        "preparation": {
            "preparation_type": preparation.preparation_type,
            "schema_version": preparation.schema_version,
            "preparation_sha256": preparation.preparation_sha256,
            "inclusion_sha256": preparation.inclusion_sha256,
            "design_sha256": preparation.design_sha256,
            "contrast_sha256": preparation.contrast_sha256,
            "design_shape": list(preparation.design_matrix.shape),
            "design_rank": preparation.design_rank,
            "residual_degrees_of_freedom": (
                preparation.residual_degrees_of_freedom
            ),
            "rank_tolerance": preparation.rank_tolerance,
            "estimability_tolerance": preparation.estimability_tolerance,
            "covariate_contract": [
                {
                    "key": value.key,
                    "kind": value.kind,
                    "source_column": value.source_column,
                    "design_columns": list(value.design_columns),
                    "categorical_levels": [
                        {
                            "value_type": level.value_type,
                            "value": level.value,
                            "design_column": level.design_column,
                        }
                        for level in value.categorical_levels
                    ],
                }
                for value in preparation.covariate_encodings
            ],
        },
        "filter": {
            **dict(FILTER_CONFIGURATION),
            "n_input_features": len(preparation.feature_ids),
            "n_tested_features": int(backend.tested_indices.size),
            "n_filtered_features": int(
                len(preparation.feature_ids) - backend.tested_indices.size
            ),
            "filter_mask_sha256": digests["filter_mask_sha256"],
            "post_filter_library_sizes_sha256": (
                digests["post_filter_library_sizes_sha256"]
            ),
        },
        "normalization": {
            **dict(NORMALIZATION_CONFIGURATION),
            "normalization_factors_sha256": (
                digests["normalization_factors_sha256"]
            ),
            "effective_library_sizes_sha256": (
                digests["effective_library_sizes_sha256"]
            ),
        },
        "statistical_test": {
            **dict(QL_CONFIGURATION),
            "multiple_testing": "Benjamini-Hochberg",
            "multiple_testing_universe": "tested_features_this_group_contrast",
            "tested_statistics_sha256": digests["tested_statistics_sha256"],
            "result_sha256": digests["result_sha256"],
        },
        "backend": {
            "pipeline": EDGER_PIPELINE_ID,
            "protocol_version": EDGER_PROTOCOL_VERSION,
            "production_r_script_sha256": production_script_sha256,
            "versions": dict(backend.versions),
        },
        "memory_preflight": {
            "policy": "dense-edger-v1",
            "n_features": memory.n_features,
            "n_samples": memory.n_samples,
            "n_design_columns": memory.n_design_columns,
            "dense_dtype": "float64",
            "dense_matrix_bytes": memory.dense_matrix_bytes,
            "base_estimate_bytes": memory.base_estimate_bytes,
            "dense_working_copies": _DENSE_WORKING_COPIES,
            "feature_result_columns": _FEATURE_RESULT_COLUMNS,
            "design_working_copies": _DESIGN_WORKING_COPIES,
            "fixed_overhead_per_process_bytes": _FIXED_MEMORY_OVERHEAD_BYTES,
            "fixed_overhead_processes": 2,
            "safety_factor": memory.safety_factor,
            "maximum_available_fraction": _MAXIMUM_AVAILABLE_FRACTION,
            "minimum_reserve_bytes": _MINIMUM_MEMORY_RESERVE_BYTES,
            "estimated_peak_bytes": memory.estimated_peak_bytes,
            "available_bytes": memory.available_bytes,
            "usable_bytes": memory.usable_bytes,
        },
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
        "software_versions": _m81_software_versions(),
    }


def _build_artifact(
    preparation: DifferentialAccessibilityPreparation,
    backend: _BackendResult,
    provenance: Mapping[str, object],
) -> ad.AnnData:
    source = ad.read_h5ad(preparation.pseudobulk_path, backed="r")
    try:
        obs = source.obs.copy()
        var = source.var.copy()
    finally:
        manager = getattr(source, "file", None)
        if manager is not None:
            manager.close()
    eligibility = preparation.row_eligibility
    obs["da_group_match"] = np.asarray(
        [row.group == preparation.group_value for row in eligibility], dtype=bool
    )
    obs["da_condition_match"] = np.asarray(
        [
            row.condition
            in {preparation.numerator_condition, preparation.denominator_condition}
            for row in eligibility
        ],
        dtype=bool,
    )
    obs["da_analysis_included"] = np.asarray(
        [row.included for row in eligibility], dtype=bool
    )
    obs["da_exclusion_reason"] = pd.Categorical(
        [row.reason for row in eligibility],
        categories=["included", "group_not_selected", "condition_not_selected"],
        ordered=True,
    )
    design_row_index = np.full(len(eligibility), -1, dtype=np.int64)
    post_filter_libraries = np.full(len(eligibility), np.nan, dtype=np.float64)
    normalization_factors = np.full(len(eligibility), np.nan, dtype=np.float64)
    effective_libraries = np.full(len(eligibility), np.nan, dtype=np.float64)
    for design_index, source_position in enumerate(
        preparation.included_source_positions
    ):
        design_row_index[source_position] = design_index
        post_filter_libraries[source_position] = (
            backend.post_filter_library_sizes[design_index]
        )
        normalization_factors[source_position] = (
            backend.normalization_factors[design_index]
        )
        effective_libraries[source_position] = (
            backend.effective_library_sizes[design_index]
        )
    obs["da_design_row_index"] = design_row_index
    obs["da_postfilter_library_size"] = post_filter_libraries
    obs["da_tmm_normalization_factor"] = normalization_factors
    obs["da_effective_library_size"] = effective_libraries

    n_features = len(preparation.feature_ids)
    tested = backend.filter_mask
    var["da_status"] = pd.Categorical(
        np.where(tested, "tested", "filtered_by_expression"),
        categories=["tested", "filtered_by_expression"],
        ordered=True,
    )
    statistic_names = ("logFC", "logCPM", "F", "PValue", "FDR")
    for statistic_index, name in enumerate(statistic_names):
        values = np.full(n_features, np.nan, dtype=np.float64)
        values[backend.tested_indices] = backend.statistics[:, statistic_index]
        var[name] = values
    directions = np.full(n_features, "not_tested", dtype=object)
    logfc = backend.statistics[:, 0]
    directions[backend.tested_indices[logfc > 0]] = "higher_in_numerator"
    directions[backend.tested_indices[logfc < 0]] = "higher_in_denominator"
    directions[backend.tested_indices[logfc == 0]] = "no_change"
    var["effect_direction"] = pd.Categorical(
        directions,
        categories=[
            "higher_in_numerator",
            "higher_in_denominator",
            "no_change",
            "not_tested",
        ],
        ordered=True,
    )
    return ad.AnnData(
        X=None,
        obs=obs,
        var=var,
        uns={DA_PROVENANCE_KEY: dict(provenance)},
    )


def _validate_temporary_artifact(
    path: Path,
    expected: ad.AnnData,
    preparation: DifferentialAccessibilityPreparation,
) -> None:
    try:
        observed = ad.read_h5ad(path)
    except Exception as exc:
        raise _error(
            "ARTIFACT_WRITE_FAILED", "Temporary DA artifact could not be reopened."
        ) from exc
    try:
        if (
            observed.X is not None
            or observed.raw is not None
            or tuple(str(value) for value in observed.obs_names)
            != preparation.source_row_ids
            or tuple(str(value) for value in observed.var_names)
            != preparation.feature_ids
            or tuple(observed.obs.columns) != tuple(expected.obs.columns)
            or tuple(observed.var.columns) != tuple(expected.var.columns)
            or set(observed.uns) != {DA_PROVENANCE_KEY}
            or any(
                len(container)
                for container in (
                    observed.layers,
                    observed.obsm,
                    observed.obsp,
                    observed.varm,
                    observed.varp,
                )
            )
        ):
            raise _error(
                "ARTIFACT_WRITE_FAILED", "Temporary DA artifact structure changed."
            )
        for column in (
            "da_group_match",
            "da_condition_match",
            "da_analysis_included",
            "da_design_row_index",
            "da_postfilter_library_size",
            "da_tmm_normalization_factor",
            "da_effective_library_size",
        ):
            left = np.asarray(observed.obs[column])
            right = np.asarray(expected.obs[column])
            if not np.array_equal(left, right, equal_nan=True):
                raise _error(
                    "ARTIFACT_WRITE_FAILED", "Temporary DA sample metadata changed."
                )
        if (
            observed.obs["da_exclusion_reason"].astype(str).tolist()
            != expected.obs["da_exclusion_reason"].astype(str).tolist()
        ):
            raise _error(
                "ARTIFACT_WRITE_FAILED", "Temporary DA exclusion states changed."
            )
        for column in ("logFC", "logCPM", "F", "PValue", "FDR"):
            if not np.array_equal(
                np.asarray(observed.var[column]),
                np.asarray(expected.var[column]),
                equal_nan=True,
            ):
                raise _error(
                    "ARTIFACT_WRITE_FAILED", "Temporary DA statistics changed."
                )
        if (
            observed.var["da_status"].astype(str).tolist()
            != expected.var["da_status"].astype(str).tolist()
            or observed.var["effect_direction"].astype(str).tolist()
            != expected.var["effect_direction"].astype(str).tolist()
            or _json_value(observed.uns[DA_PROVENANCE_KEY])
            != _json_value(expected.uns[DA_PROVENANCE_KEY])
        ):
            raise _error(
                "ARTIFACT_WRITE_FAILED", "Temporary DA result metadata changed."
            )
    finally:
        manager = getattr(observed, "file", None)
        if manager is not None:
            manager.close()


def _atomic_write_artifact(
    artifact: ad.AnnData,
    output_path: Path,
    preparation: DifferentialAccessibilityPreparation,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Scientific output already exists: {output_path}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.stem}.",
            suffix=".tmp.h5ad",
        )
        os.close(descriptor)
        temporary = Path(name)
        artifact.write_h5ad(temporary)
        _validate_temporary_artifact(temporary, artifact, preparation)
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
    except (FileExistsError, M82ScientificError):
        raise
    except OSError as exc:
        code = "DISK_FULL" if exc.errno == errno.ENOSPC else "ARTIFACT_WRITE_FAILED"
        raise _error(code, "DA artifact publication failed.") from exc
    except Exception as exc:
        raise _error("ARTIFACT_WRITE_FAILED", "DA artifact publication failed.") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolve_output_directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError("`output_dir` must be path-like.")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise ValueError("`output_dir` must be a directory.")
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        code = "DISK_FULL" if exc.errno == errno.ENOSPC else "ARTIFACT_WRITE_FAILED"
        raise _error(code, "DA output directory could not be created.") from exc
    return path


def run_replicate_differential_accessibility(
    pseudobulk_path: str | Path,
    group_value: str,
    condition_key: str,
    numerator_condition: str,
    denominator_condition: str,
    design_type: Literal["independent", "paired"] | str,
    output_dir: str | Path,
    *,
    covariates: Sequence[Mapping[str, object]] = (),
    overwrite: bool = False,
) -> ReplicateDifferentialAccessibilityToolResult:
    """Run the frozen edgeR QL backend and publish one compact DA artifact."""

    if not isinstance(overwrite, bool):
        raise ValueError("`overwrite` must be boolean.")
    preparation = prepare_replicate_differential_accessibility(
        pseudobulk_path,
        group_value,
        condition_key,
        numerator_condition,
        denominator_condition,
        design_type,
        covariates=covariates,
    )
    memory = assess_host_memory(
        len(preparation.feature_ids),
        len(preparation.included_source_positions),
        len(preparation.design_columns),
    )
    production_script_sha256 = _production_r_script_sha256()
    analysis_sha256 = _analysis_identity(
        preparation, production_script_sha256=production_script_sha256
    )
    directory = _resolve_output_directory(output_dir)
    output_path = directory / (
        f"{Path(preparation.pseudobulk_path).stem}.da.{analysis_sha256}.h5ad"
    )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Scientific output already exists: {output_path}")
    rscript = _resolve_rscript()
    probed_versions = _probe_backend(rscript)
    with tempfile.TemporaryDirectory(prefix="agent-edger-run-") as name:
        staging = Path(name)
        selected_counts_sha256 = _stage_inputs(staging, preparation)
        status = _invoke_fixed_script(rscript, "run", staging)
        backend = _read_backend_result(staging, preparation, status)
    if backend.versions != probed_versions:
        raise _error(
            "R_PACKAGE_VERSION_INCOMPATIBLE",
            "R package stack changed between probe and analysis.",
        )
    if _m81_file_sha256(Path(preparation.pseudobulk_path)) != preparation.pseudobulk_sha256:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed during edgeR analysis."
        )
    digests = _result_digests(preparation, backend)
    provenance = _artifact_provenance(
        preparation,
        backend,
        memory,
        selected_counts_sha256=selected_counts_sha256,
        production_script_sha256=production_script_sha256,
        analysis_sha256=analysis_sha256,
        digests=digests,
    )
    artifact = _build_artifact(preparation, backend, provenance)
    _atomic_write_artifact(
        artifact, output_path, preparation, overwrite=overwrite
    )
    if _m81_file_sha256(Path(preparation.pseudobulk_path)) != preparation.pseudobulk_sha256:
        raise _error(
            "SOURCE_CHANGED_DURING_READ", "Pseudobulk changed during DA publication."
        )
    artifact_sha256 = _m81_file_sha256(output_path)
    warning_codes = [warning.code for warning in preparation.warnings]
    return {
        "status": "success",
        "da_path": str(output_path),
        "da_sha256": artifact_sha256,
        "artifact_type": DA_ARTIFACT_TYPE,
        "artifact_schema_version": DA_ARTIFACT_SCHEMA_VERSION,
        "pseudobulk_path": preparation.pseudobulk_path,
        "pseudobulk_sha256": preparation.pseudobulk_sha256,
        "preparation_sha256": preparation.preparation_sha256,
        "analysis_sha256": analysis_sha256,
        "group_value": preparation.group_value,
        "condition_key": preparation.condition_key,
        "numerator_condition": preparation.numerator_condition,
        "denominator_condition": preparation.denominator_condition,
        "design_type": preparation.design_type,
        "n_samples": len(preparation.included_source_positions),
        "n_numerator_replicates": len(preparation.numerator_replicates),
        "n_denominator_replicates": len(preparation.denominator_replicates),
        "design_rank": preparation.design_rank,
        "residual_degrees_of_freedom": (
            preparation.residual_degrees_of_freedom
        ),
        "warning_codes": warning_codes,
        "n_warnings": len(warning_codes),
        "n_input_features": len(preparation.feature_ids),
        "n_tested_features": int(backend.tested_indices.size),
        "n_filtered_features": int(
            len(preparation.feature_ids) - backend.tested_indices.size
        ),
        "filtering_method": str(FILTER_CONFIGURATION["method"]),
        "normalization_method": "TMM",
        "backend_pipeline": EDGER_PIPELINE_ID,
        "production_r_script_sha256": production_script_sha256,
        "r_version": backend.versions["r"],
        "bioconductor_version": backend.versions["bioconductor"],
        "edger_version": backend.versions["edger"],
        "package_versions": dict(backend.versions),
    }


__all__ = [
    "DA_ARTIFACT_SCHEMA_VERSION",
    "DA_ARTIFACT_TYPE",
    "DA_PROVENANCE_KEY",
    "EDGER_PIPELINE_ID",
    "EDGER_PROTOCOL_VERSION",
    "EDGER_RSCRIPT_ENVIRONMENT_VARIABLE",
    "EXPECTED_BACKEND_VERSIONS",
    "FILTER_CONFIGURATION",
    "HostMemoryAssessment",
    "NORMALIZATION_CONFIGURATION",
    "PRODUCTION_R_SCRIPT",
    "QL_CONFIGURATION",
    "ReplicateDifferentialAccessibilityToolResult",
    "assess_host_memory",
    "run_replicate_differential_accessibility",
]
