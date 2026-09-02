"""Milestone 8.2-B pinned edgeR backend and DA artifact tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.orchestration import build_default_tool_registry
from agent.tools.analysis import differential_accessibility_backend as backend
from agent.tools.analysis.differential_accessibility import (
    M82ScientificError,
    prepare_replicate_differential_accessibility,
)
from agent.tools.analysis.differential_accessibility_backend import (
    DA_ARTIFACT_TYPE,
    DA_PROVENANCE_KEY,
    EDGER_RSCRIPT_ENVIRONMENT_VARIABLE,
    EXPECTED_BACKEND_VERSIONS,
    assess_host_memory,
    run_replicate_differential_accessibility,
)
from agent.tools.analysis.replicate_pseudobulk import (
    build_replicate_pseudobulk,
    validate_scATAC_feature_space,
)


R_SCRIPT = Path("/home/likeyi/anaconda3/envs/agent-edger/bin/Rscript")
REFERENCE_SCRIPT = Path(__file__).parent / "fixtures" / "edger_ql_reference.R"
GROUP = "T;$(touch should-not-exist)"
NUMERATOR = "treated;`touch should-not-exist`"
DENOMINATOR = "control && touch should-not-exist"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _error_code(callback) -> str:
    with pytest.raises(M82ScientificError) as caught:
        callback()
    return caught.value.code


def _selected_counts(kind: str = "standard") -> np.ndarray:
    counts = np.zeros((6, 40), dtype=np.int64)
    if kind == "no_features":
        counts[:, :] = 1
        return counts
    for sample in range(6):
        counts[sample, :5] = 1
        counts[sample, 5:15] = 45 + sample + np.arange(10) % 3
        if sample in (1, 3, 5):
            counts[sample, 15:25] = 120 + sample + np.arange(10) % 5
            counts[sample, 25:35] = 25 + sample + np.arange(10) % 4
        else:
            counts[sample, 15:25] = 30 + sample + np.arange(10) % 5
            counts[sample, 25:35] = 100 + sample + np.arange(10) % 4
        counts[sample, 35:40] = 40 + sample + np.arange(5)
    if kind == "postfilter_zero":
        counts[0, 5:] = 0
    return counts


def _fixture(tmp_path: Path, *, kind: str = "standard") -> tuple[Path, Path, np.ndarray]:
    selected = _selected_counts(kind)
    # Selected order after M8.2-A filtering is positions 1, 3, 4, 5, 6, 7.
    units = [
        ("B", "excluded-b", DENOMINATOR, np.full(40, 20, dtype=np.int64)),
        (GROUP, "control-1", DENOMINATOR, selected[0]),
        (GROUP, "excluded-condition", "other", np.full(40, 30, dtype=np.int64)),
        (GROUP, "treated-1", NUMERATOR, selected[1]),
        (GROUP, "control-2", DENOMINATOR, selected[2]),
        (GROUP, "treated-2", NUMERATOR, selected[3]),
        (GROUP, "control-3", DENOMINATOR, selected[4]),
        (GROUP, "treated-3", NUMERATOR, selected[5]),
    ]
    rows: list[np.ndarray] = []
    observations: list[dict[str, str]] = []
    cell_ids: list[str] = []
    for unit_index, (group, replicate, condition, values) in enumerate(units):
        halves = (values // 2, values - values // 2)
        for cell_index, row in enumerate(halves):
            rows.append(row)
            observations.append(
                {"group": group, "replicate": replicate, "condition": condition}
            )
            cell_ids.append(f"cell-{unit_index:02d}-{cell_index}")
    var = pd.DataFrame(
        {
            "chromosome": ["chr1"] * 20 + ["chr2"] * 20,
            "start": np.arange(40, dtype=np.int64) * 100,
            "end": np.arange(40, dtype=np.int64) * 100 + 50,
            "fixture_note": [f"note-{index}" for index in range(40)],
        },
        index=[f"peak-{index:03d}" for index in range(40)],
    )
    raw_path = tmp_path / "raw.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray(rows, dtype=np.int64)),
        obs=pd.DataFrame(observations, index=cell_ids),
        var=var,
    ).write_h5ad(raw_path)
    feature = validate_scATAC_feature_space(
        raw_path,
        tmp_path / "feature",
        matrix_source="X",
        matrix_semantics="fragment_counts",
        species="human",
        genome_assembly="hg38",
        coordinate_source="var_columns",
        feature_chrom_key="chromosome",
        feature_start_key="start",
        feature_end_key="end",
        coordinate_system="zero_based_half_open",
    )
    result = build_replicate_pseudobulk(
        feature["feature_space_path"],
        "replicate",
        "group",
        "condition",
        tmp_path / "pseudobulk",
        group_source="raw_obs",
    )
    return raw_path, Path(str(result["pseudobulk_path"])), selected


def _prepare(path: Path):
    return prepare_replicate_differential_accessibility(
        path,
        GROUP,
        "condition",
        NUMERATOR,
        DENOMINATOR,
        "independent",
    )


def _real_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("RUN_EDGER_INTEGRATION") != "1":
        pytest.skip("set RUN_EDGER_INTEGRATION=1 for the pinned edgeR tests")
    if not R_SCRIPT.is_file():
        pytest.skip("the isolated agent-edger Rscript is unavailable")
    monkeypatch.setenv(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE, str(R_SCRIPT))


def _fake_rscript(tmp_path: Path, body: str) -> Path:
    directory = tmp_path / "fake-bin"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "Rscript"
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_memory_policy_safe_estimate_and_unsafe_rejection() -> None:
    result = assess_host_memory(40, 6, 2, available_bytes=2 * 1024**3)
    assert result.dense_matrix_bytes == 40 * 6 * 8
    assert result.safety_factor == 1.25
    assert result.estimated_peak_bytes < result.usable_bytes
    assert _error_code(
        lambda: assess_host_memory(10_000_000, 100, 10, available_bytes=1024**3)
    ) == "HOST_MEMORY_EXHAUSTED"


def test_unsafe_preflight_and_a_failure_never_resolve_or_start_r(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk, _ = _fixture(tmp_path)
    resolution = pytest.fail
    monkeypatch.setattr(backend, "_resolve_rscript", resolution)
    monkeypatch.setattr(
        backend,
        "_host_available_memory_bytes",
        lambda: 128 * 1024**2,
    )
    assert _error_code(
        lambda: run_replicate_differential_accessibility(
            pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
            "independent", tmp_path / "out"
        )
    ) == "HOST_MEMORY_EXHAUSTED"

    monkeypatch.setattr(backend, "assess_host_memory", pytest.fail)
    assert _error_code(
        lambda: run_replicate_differential_accessibility(
            pseudobulk, "absent", "condition", NUMERATOR, DENOMINATOR,
            "independent", tmp_path / "out"
        )
    ) == "DA_GROUP_NOT_FOUND"


def test_rscript_resolution_is_absolute_approved_and_missing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE, raising=False)
    assert _error_code(backend._resolve_rscript) == "RSCRIPT_UNAVAILABLE"
    monkeypatch.setenv(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE, "Rscript")
    assert _error_code(backend._resolve_rscript) == "RSCRIPT_UNAVAILABLE"
    executable = _fake_rscript(tmp_path, "")
    monkeypatch.setenv(EDGER_RSCRIPT_ENVIRONMENT_VARIABLE, str(executable))
    assert backend._resolve_rscript() == executable.resolve()


@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        ({**EXPECTED_BACKEND_VERSIONS, "edger": "4.8.0"}, "EDGER_VERSION_UNSUPPORTED"),
        (
            {**EXPECTED_BACKEND_VERSIONS, "limma": "0.0.0"},
            "R_PACKAGE_VERSION_INCOMPATIBLE",
        ),
        (
            {
                key: value
                for key, value in EXPECTED_BACKEND_VERSIONS.items()
                if key != "locfit"
            },
            "R_PACKAGE_VERSION_INCOMPATIBLE",
        ),
    ],
)
def test_backend_version_policy_fails_closed(
    versions: dict[str, str], expected: str
) -> None:
    fields = {"mode": "probe"}
    for key, value in versions.items():
        field = "biocmanager_version" if key == "biocmanager" else f"{key}_version"
        fields[field] = value
    assert _error_code(
        lambda: backend._versions_from_status(fields, mode="probe")
    ) == expected


def test_protocol_maps_controlled_failure_crash_sanitizes_and_bounds_diagnostics(
    tmp_path: Path,
) -> None:
    controlled = _fake_rscript(
        tmp_path,
        "import pathlib, sys\n"
        "d=pathlib.Path(sys.argv[-1])\n"
        "(d/'backend_status.tsv').write_text("
        "'protocol_version\\t1\\nstatus\\terror\\n"
        "error_code\\tEDGER_PACKAGE_UNAVAILABLE\\n')\n"
        "sys.stderr.write('SECRET biological path and stack trace')\n"
        "raise SystemExit(41)\n",
    )
    directory = tmp_path / "controlled"
    directory.mkdir()
    with pytest.raises(M82ScientificError) as caught:
        backend._invoke_fixed_script(controlled, "probe", directory)
    assert caught.value.code == "EDGER_PACKAGE_UNAVAILABLE"
    assert "SECRET" not in str(caught.value)

    crash = _fake_rscript(
        tmp_path / "crash",
        "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n",
    )
    crash_dir = tmp_path / "crash-run"
    crash_dir.mkdir()
    assert _error_code(
        lambda: backend._invoke_fixed_script(crash, "probe", crash_dir)
    ) == "R_BACKEND_EXECUTION_FAILED"

    oversized = _fake_rscript(
        tmp_path / "oversized", "import sys\nsys.stderr.write('x' * 70000)\n"
    )
    oversized_dir = tmp_path / "oversized-run"
    oversized_dir.mkdir()
    assert _error_code(
        lambda: backend._invoke_fixed_script(oversized, "probe", oversized_dir)
    ) == "R_BACKEND_PROTOCOL_INVALID"


def test_malformed_duplicate_and_oversized_status_fail_closed(tmp_path: Path) -> None:
    status = tmp_path / "backend_status.tsv"
    status.write_text("protocol_version\t1\nstatus\tsuccess\nstatus\tsuccess\n")
    assert (
        _error_code(lambda: backend._read_status(tmp_path))
        == "R_BACKEND_PROTOCOL_INVALID"
    )
    status.write_bytes(b"x" * 70000)
    assert (
        _error_code(lambda: backend._read_status(tmp_path))
        == "R_BACKEND_PROTOCOL_INVALID"
    )


def test_truncated_binary_result_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "values.bin"
    path.write_bytes(np.asarray([1.0], dtype="<f8").tobytes())
    assert _error_code(
        lambda: backend._read_binary(path, dtype="<f8", count=2)
    ) == "R_BACKEND_PROTOCOL_INVALID"


def test_staging_consumes_exact_a_positions_design_and_contrast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk, selected = _fixture(tmp_path)
    preparation = _prepare(pseudobulk)
    staging = tmp_path / "stage"
    staging.mkdir()
    before = _sha(pseudobulk)
    read_h5ad = backend.ad.read_h5ad
    opened: list[tuple[Path, str | None]] = []

    def guarded_read_h5ad(path, *, backed=None):
        opened.append((Path(path), backed))
        return read_h5ad(path, backed=backed)

    monkeypatch.setattr(backend.ad, "read_h5ad", guarded_read_h5ad)
    backend._stage_inputs(staging, preparation)
    assert opened == [(pseudobulk, "r")]
    assert preparation.included_source_positions == (1, 3, 4, 5, 6, 7)
    np.testing.assert_array_equal(
        np.fromfile(staging / "counts.bin", dtype="<f8").reshape(6, 40), selected
    )
    np.testing.assert_array_equal(
        np.fromfile(staging / "design.bin", dtype="<f8").reshape(
            preparation.design_matrix.shape
        ),
        preparation.design_matrix,
    )
    np.testing.assert_array_equal(
        np.fromfile(staging / "contrast.bin", dtype="<f8"), preparation.contrast
    )
    assert (staging / "condition.bin").read_bytes() == np.asarray(
        [0, 1, 0, 1, 0, 1], dtype="<i4"
    ).tobytes()
    assert _sha(pseudobulk) == before
    assert all(
        value not in path.name
        for path in staging.iterdir()
        for value in (GROUP, NUMERATOR, DENOMINATOR)
    )


def test_atomic_write_failure_leaves_no_authoritative_or_temporary_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pseudobulk, _ = _fixture(tmp_path)
    preparation = _prepare(pseudobulk)
    output = tmp_path / "failed.h5ad"
    artifact = ad.AnnData(
        X=None,
        obs=pd.DataFrame(index=preparation.source_row_ids),
        var=pd.DataFrame(index=preparation.feature_ids),
    )

    def fail_write(*args, **kwargs) -> None:
        raise OSError("simulated artifact write failure")

    monkeypatch.setattr(ad.AnnData, "write_h5ad", fail_write)
    assert _error_code(
        lambda: backend._atomic_write_artifact(
            artifact, output, preparation, overwrite=False
        )
    ) == "ARTIFACT_WRITE_FAILED"
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp.h5ad"))


def _run_reference(
    directory: Path, preparation, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.savetxt(directory / "counts.tsv", counts.T, delimiter="\t", fmt="%d")
    np.savetxt(
        directory / "design.tsv",
        preparation.design_matrix,
        delimiter="\t",
        fmt="%.17g",
    )
    np.savetxt(directory / "condition.tsv", [0, 1, 0, 1, 0, 1], fmt="%d")
    np.savetxt(directory / "contrast.tsv", preparation.contrast, fmt="%.17g")
    subprocess.run(
        [str(R_SCRIPT), "--vanilla", str(REFERENCE_SCRIPT), str(directory)],
        check=True,
        cwd=directory,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=backend._subprocess_environment(),
    )
    mask = np.loadtxt(directory / "filter_mask.tsv", dtype=np.int64, ndmin=1)
    samples = np.loadtxt(
        directory / "sample_results.tsv", dtype=np.float64, ndmin=2
    )
    features = np.loadtxt(
        directory / "feature_results.tsv", dtype=np.float64, ndmin=2
    )
    return mask, samples, features


def test_real_backend_matches_independent_oracle_and_artifact_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_backend(monkeypatch)
    raw, pseudobulk, selected = _fixture(tmp_path)
    preparation = _prepare(pseudobulk)
    raw_before, pseudobulk_before = _sha(raw), _sha(pseudobulk)

    result = run_replicate_differential_accessibility(
        pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
        "independent", tmp_path / "da"
    )
    oracle_dir = tmp_path / "oracle"
    oracle_dir.mkdir()
    oracle_mask, oracle_samples, oracle_features = _run_reference(
        oracle_dir, preparation, selected
    )

    assert result["package_versions"] == dict(EXPECTED_BACKEND_VERSIONS)
    assert result["n_input_features"] == 40
    assert result["n_tested_features"] == 35
    assert result["n_filtered_features"] == 5
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    forbidden = {"statistics", "feature_ids", "design_matrix", "contrast", "normalization_factors"}
    assert forbidden.isdisjoint(result)
    artifact_path = Path(result["da_path"])
    artifact = ad.read_h5ad(artifact_path)
    source = ad.read_h5ad(pseudobulk)
    assert result["da_sha256"] == _sha(artifact_path)
    assert artifact.X is None and artifact.raw is None
    assert len(artifact.layers) == len(artifact.obsm) == len(artifact.obsp) == 0
    assert artifact.obs_names.tolist() == source.obs_names.tolist()
    assert artifact.var_names.tolist() == source.var_names.tolist()
    for column in source.obs.columns:
        assert artifact.obs[column].tolist() == source.obs[column].tolist()
    for column in source.var.columns:
        assert artifact.var[column].tolist() == source.var[column].tolist()
    assert artifact.obs["da_analysis_included"].tolist() == [
        False,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]
    assert artifact.obs["da_design_row_index"].tolist() == [-1, 0, -1, 1, 2, 3, 4, 5]
    assert artifact.obs["da_exclusion_reason"].astype(str).tolist() == [
        "group_not_selected", "included", "condition_not_selected", "included",
        "included", "included", "included", "included",
    ]
    np.testing.assert_array_equal(
        artifact.var["da_status"].astype(str) == "tested",
        oracle_mask.astype(bool),
    )
    assert artifact.var.loc[
        artifact.var["da_status"].astype(str) == "filtered_by_expression",
        ["logFC", "logCPM", "F", "PValue", "FDR"],
    ].isna().all().all()
    assert artifact.var["effect_direction"].astype(str).iloc[:5].tolist() == ["not_tested"] * 5
    np.testing.assert_allclose(
        artifact.obs.loc[
            artifact.obs["da_analysis_included"],
            [
                "da_postfilter_library_size",
                "da_tmm_normalization_factor",
                "da_effective_library_size",
            ],
        ],
        oracle_samples,
        rtol=1e-13,
        atol=1e-13,
    )
    tested = artifact.var["da_status"].astype(str) == "tested"
    np.testing.assert_array_equal(np.flatnonzero(tested), oracle_features[:, 0].astype(int))
    np.testing.assert_allclose(
        artifact.var.loc[tested, ["logFC", "logCPM", "F", "PValue", "FDR"]],
        oracle_features[:, 1:],
        rtol=1e-12,
        atol=1e-300,
    )
    assert (
        artifact.var.iloc[15:25]["effect_direction"].astype(str)
        == "higher_in_numerator"
    ).all()
    assert (
        artifact.var.iloc[25:35]["effect_direction"].astype(str)
        == "higher_in_denominator"
    ).all()
    provenance = artifact.uns[DA_PROVENANCE_KEY]
    assert provenance["artifact_type"] == DA_ARTIFACT_TYPE
    assert provenance["analysis_sha256"] == result["analysis_sha256"]
    assert provenance["statistical_test"]["estimate_disp_called"] is False
    assert provenance["validation"]["source_count_matrix_copied_to_artifact"] is False
    assert len(json.dumps(backend._json_value(provenance))) < 20_000
    assert _sha(raw) == raw_before and _sha(pseudobulk) == pseudobulk_before
    assert not (tmp_path / "should-not-exist").exists()


def test_real_backend_is_deterministic_and_overwrite_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _real_backend(monkeypatch)
    _, pseudobulk, _ = _fixture(tmp_path)
    first = run_replicate_differential_accessibility(
        pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
        "independent", tmp_path / "first"
    )
    second = run_replicate_differential_accessibility(
        pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
        "independent", tmp_path / "second"
    )
    assert first["analysis_sha256"] == second["analysis_sha256"]
    first_provenance = ad.read_h5ad(first["da_path"]).uns[DA_PROVENANCE_KEY]
    second_provenance = ad.read_h5ad(second["da_path"]).uns[DA_PROVENANCE_KEY]
    assert (
        first_provenance["statistical_test"]["result_sha256"]
        == second_provenance["statistical_test"]["result_sha256"]
    )
    with pytest.raises(FileExistsError):
        run_replicate_differential_accessibility(
            pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
            "independent", tmp_path / "first"
        )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("no_features", "DA_NO_FEATURES_AFTER_FILTER"),
        ("postfilter_zero", "DA_FILTERED_LIBRARY_ZERO"),
    ],
)
def test_real_filtering_controlled_failures_leave_no_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, expected: str
) -> None:
    _real_backend(monkeypatch)
    _, pseudobulk, _ = _fixture(tmp_path, kind=kind)
    output = tmp_path / "da"
    before = _sha(pseudobulk)
    assert _error_code(
        lambda: run_replicate_differential_accessibility(
            pseudobulk, GROUP, "condition", NUMERATOR, DENOMINATOR,
            "independent", output
        )
    ) == expected
    assert not list(output.glob("*.h5ad"))
    assert _sha(pseudobulk) == before


def test_m82c_registers_only_the_public_da_callable() -> None:
    names = set(build_default_tool_registry().names())
    assert "run_replicate_differential_accessibility" in names
    assert "differential_accessibility" not in names


def test_production_r_script_freezes_the_audited_pipeline() -> None:
    script = backend.PRODUCTION_R_SCRIPT.read_text(encoding="utf-8")
    for required in (
        "DGEList(",
        "filterByExpr(",
        "group = condition_group",
        "keep.lib.sizes = FALSE",
        "normLibSizes(",
        'method = "TMM"',
        "glmQLFit(",
        "dispersion = NULL",
        "legacy = FALSE",
        "robust = TRUE",
        "glmQLFTest(",
        "contrast = contrast",
        "poisson.bound = TRUE",
        'p.adjust(table[, "PValue"], method = "BH")',
    ):
        assert required in script
    assert "estimateDisp(" not in script
