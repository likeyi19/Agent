from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

import agent.report.visualization as visualization_module
from agent.orchestration import ToolRegistry, build_default_tool_registry
from agent.report import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
    ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
    AnalysisVisualizationError,
    build_analysis_visualizations,
    verify_analysis_visualizations,
)
from agent.schemas import (
    AgentPlan,
    AgentRunResult,
    PlanStep,
    RunStatus,
    VerificationCheck,
    VerificationResult,
)


def _passed() -> VerificationResult:
    return VerificationResult(
        True,
        "analysis_evidence",
        "request-1:run",
        (VerificationCheck("accepted", True, "Accepted."),),
    )


def _guarded_registry() -> ToolRegistry:
    registry = build_default_tool_registry()

    def forbidden(**_: object) -> object:
        raise AssertionError("Visualization invoked a scientific callable.")

    return ToolRegistry(
        tuple(replace(registry.get(name), function=forbidden) for name in registry.names())
    )


def _write_umap(path: Path, *, labels: tuple[str, ...] = ("2", "1", "2")) -> None:
    obs = pd.DataFrame(
        {"leiden": pd.Categorical(labels)},
        index=[f"cell-{index}" for index in range(len(labels))],
    )
    artifact = ad.AnnData(obs=obs)
    artifact.obsm["X_umap"] = np.asarray(
        [[float(index), float(index) - 0.5] for index in range(len(labels))],
        dtype=np.float32,
    )
    artifact.write_h5ad(path)


def _write_confusion(path: Path, *, count: int = 3) -> None:
    payload = {
        "confusion": {
            "rows": {"labels": ["B", "A"]},
            "columns": [
                {"kind": "biological_label", "label": "A"},
                {"kind": "biological_label", "label": "external"},
                {"kind": "structural_unassigned", "label": None},
            ],
            "counts": [[count, 1, 0], [0, 2, 1]],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(step_id: str, tool_name: str, kind: str, path: Path) -> dict[str, object]:
    return {
        "producing_step_id": step_id,
        "tool_name": tool_name,
        "result_field": "analysis_path" if tool_name == "compute_cell_umap" else "report_path",
        "artifact_kind": kind,
        "artifact_path": str(path.resolve()),
        "integrity": {"basis": ["fresh_tool_verifier"]},
    }


def _source(
    tmp_path: Path,
    *,
    tools: tuple[str, ...] = (
        "compute_cell_umap",
        "evaluate_cell_clustering",
        "evaluate_cell_annotation",
    ),
    duplicate_umap: bool = False,
) -> tuple[AgentRunResult, dict[str, object], Path, Path]:
    umap_path = tmp_path / "source.umap.h5ad"
    metrics_path = tmp_path / "source.clustering_metrics.json"
    confusion_path = tmp_path / "source.annotation_evaluation.json"
    _write_umap(umap_path)
    metrics_path.write_text("{}", encoding="utf-8")
    _write_confusion(confusion_path)

    definitions: list[tuple[str, str, dict[str, object], dict[str, object] | None]] = []
    counters: dict[str, int] = {}
    paths = {
        "compute_cell_umap": (
            "cell_umap_h5ad",
            umap_path,
            {"n_cells": 3},
        ),
        "evaluate_cell_clustering": (
            "clustering_evaluation_json",
            metrics_path,
            {"nmi": 0.7, "ari": -0.2, "ami": -0.1, "homogeneity": 0.8},
        ),
        "evaluate_cell_annotation": (
            "annotation_evaluation_json",
            confusion_path,
            {},
        ),
        "inspect_scATAC": ("", tmp_path / "input.h5ad", {}),
    }
    selected = tools + (("compute_cell_umap",) if duplicate_umap else ())
    artifacts: list[dict[str, object]] = []
    for tool_name in selected:
        counters[tool_name] = counters.get(tool_name, 0) + 1
        step_id = f"{tool_name}-{counters[tool_name]}"
        kind, path, facts = paths[tool_name]
        definitions.append((step_id, tool_name, facts, None))
        if kind:
            artifacts.append(_artifact(step_id, tool_name, kind, path))

    plan = AgentPlan(
        "plan-1",
        "request-1",
        "test-planner",
        tuple(PlanStep(step_id, tool_name, {}) for step_id, tool_name, _, _ in definitions),
    )
    run = AgentRunResult(
        "request-1:run",
        "request-1",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        verification=_passed(),
    )
    evidence_payload = {
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "status": "success",
        "run": {
            "run_id": run.run_id,
            "request_id": run.request_id,
            "plan_id": plan.plan_id,
        },
        "workflow": {
            "ordered_steps": [
                {"step_id": step_id, "tool_name": tool_name}
                for step_id, tool_name, _, _ in definitions
            ]
        },
        "steps": [
            {"step_id": step_id, "tool_name": tool_name, "facts": facts}
            for step_id, tool_name, facts, _ in definitions
        ],
        "artifacts": artifacts,
    }
    evidence_path = tmp_path / "analysis_evidence.json"
    evidence_bytes = json.dumps(
        evidence_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence_path.write_bytes(evidence_bytes)
    evidence_result: dict[str, object] = {
        "status": "success",
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "run_id": run.run_id,
        "request_id": run.request_id,
        "plan_id": plan.plan_id,
        "n_steps": len(definitions),
        "tool_names": [tool_name for _, tool_name, _, _ in definitions],
        "all_steps_verified": True,
    }
    return run, evidence_result, umap_path, confusion_path


@pytest.fixture(autouse=True)
def _verified_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(visualization_module, "verify_analysis_evidence", lambda *a, **k: _passed())
    monkeypatch.setattr(
        visualization_module,
        "_load_annotation_evaluation_report",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )


def _build(tmp_path: Path, **source_options: object) -> tuple[object, ...]:
    run, evidence, umap_path, confusion_path = _source(tmp_path, **source_options)
    result = build_analysis_visualizations(
        run,
        evidence,
        tmp_path / "output",
        registry=_guarded_registry(),
    )
    return run, evidence, umap_path, confusion_path, result


def test_builds_exact_v1_bundle_and_verifies_without_tool_execution(tmp_path: Path) -> None:
    run, evidence, _, _, result = _build(tmp_path)

    assert result["status"] == "success"
    assert result["artifact_type"] == ANALYSIS_VISUALIZATION_ARTIFACT_TYPE
    assert result["schema_version"] == ANALYSIS_VISUALIZATION_SCHEMA_VERSION
    assert result["n_figures"] == 3
    assert [value["figure_kind"] for value in result["figures"]] == [
        "umap_leiden",
        "clustering_metrics",
        "annotation_confusion",
    ]
    assert json.loads(json.dumps(result)) == result
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["validation"]["scientific_tools_invoked"] is False
    assert manifest["validation"]["transferred_label_umap_included"] is False
    assert manifest["plotting"]["renderer"]["backend"] == "Agg"
    for figure in result["figures"]:
        path = Path(figure["figure_path"])
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == figure["png_sha256"]

    verification = verify_analysis_visualizations(
        run, evidence, result, registry=_guarded_registry()
    )
    assert verification.passed


def test_projection_preserves_umap_values_order_and_fixed_presentations(tmp_path: Path) -> None:
    run, evidence, _, _, _ = _build(tmp_path)
    snapshot = visualization_module._verified_evidence_snapshot(
        run, evidence, _guarded_registry()
    )
    umap, metrics, confusion = visualization_module._derive_figure_projections(snapshot)

    assert umap.data.cell_ids == ("cell-0", "cell-1", "cell-2")
    np.testing.assert_array_equal(
        umap.data.coordinates,
        np.asarray([[0.0, -0.5], [1.0, 0.5], [2.0, 1.5]], dtype=np.float32),
    )
    assert umap.data.labels == ("2", "1", "2")
    assert umap.data.categories == ("2", "1")
    assert umap.presentation["jitter"] is False
    assert umap.presentation["subsampling"] is False
    assert umap.presentation["coordinate_transform"] is False
    assert metrics.data.names == ("NMI", "ARI", "AMI", "Homogeneity")
    assert metrics.data.values == (0.7, -0.2, -0.1, 0.8)
    assert metrics.presentation["y_axis"] == (-1.0, 1.0)
    assert metrics.presentation["ranking"] is False
    assert confusion.data.row_labels == ("B", "A")
    assert confusion.data.column_labels == ("A", "external", "Unassigned (structural)")
    np.testing.assert_array_equal(confusion.data.counts, [[3, 1, 0], [0, 2, 1]])
    assert confusion.presentation["normalization"] is False


def test_metric_renderer_uses_fixed_axis_and_zero_line() -> None:
    figure = Figure()
    visualization_module._render_metrics(
        figure,
        visualization_module._MetricData(
            ("NMI", "ARI", "AMI", "Homogeneity"), (0.5, -0.4, -0.2, 0.7)
        ),
    )
    axes = figure.axes[0]
    assert axes.get_ylim() == (-1.0, 1.0)
    assert any(np.allclose(line.get_ydata(), [0.0, 0.0]) for line in axes.lines)


def test_palette_extension_is_fixed_and_deterministic() -> None:
    first = visualization_module._categorical_palette(27)
    second = visualization_module._categorical_palette(27)
    assert first == second
    assert first[:20] == visualization_module._PRIMARY_PALETTE
    assert len(first) == len(set(first)) == 27


def test_confusion_cell_annotation_threshold_is_fixed(tmp_path: Path) -> None:
    report = tmp_path / "small.json"
    _write_confusion(report)
    small = visualization_module._read_confusion_presentation(report)
    assert small.annotate_cells
    large_report = tmp_path / "large.json"
    large_report.write_text(
        json.dumps(
            {
                "confusion": {
                    "rows": {"labels": [f"row-{index}" for index in range(21)]},
                    "columns": [
                        {"kind": "biological_label", "label": f"column-{index}"}
                        for index in range(19)
                    ]
                    + [{"kind": "structural_unassigned", "label": None}],
                    "counts": np.zeros((21, 20), dtype=int).tolist(),
                }
            }
        ),
        encoding="utf-8",
    )
    assert not visualization_module._read_confusion_presentation(
        large_report
    ).annotate_cells


def test_umap_presentation_reader_does_not_access_scientific_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AllowedMapping:
        def __init__(self, key: str, value: object) -> None:
            self.key = key
            self.value = value

        def __getitem__(self, key: str) -> object:
            assert key == self.key
            return self.value

    class File:
        def close(self) -> None:
            pass

    class Guarded:
        obs_names = pd.Index(["c1", "c2"])
        obsm = AllowedMapping("X_umap", np.asarray([[0, 1], [2, 3]], dtype=np.float32))
        obs = AllowedMapping("leiden", pd.Series(["x", "y"]))
        file = File()

        @property
        def X(self) -> object:
            raise AssertionError(".X must not be accessed")

        @property
        def obsp(self) -> object:
            raise AssertionError("neighbor graphs must not be accessed")

    monkeypatch.setattr(visualization_module.ad, "read_h5ad", lambda *a, **k: Guarded())
    data = visualization_module._read_umap_presentation(Path("ignored"), expected_n_cells=2)
    assert data.cell_ids == ("c1", "c2")


def test_multiple_supported_steps_have_stable_order_and_safe_names(tmp_path: Path) -> None:
    _, _, _, _, result = _build(tmp_path, duplicate_umap=True)
    assert [value["figure_kind"] for value in result["figures"]] == [
        "umap_leiden",
        "clustering_metrics",
        "annotation_confusion",
        "umap_leiden",
    ]
    for index, figure in enumerate(result["figures"], start=1):
        name = Path(figure["figure_path"]).name
        assert name.startswith(f"{index:03d}_{figure['figure_kind']}_")
        assert name.endswith(".png")
        assert "/" not in name


def test_no_supported_figure_fails_closed(tmp_path: Path) -> None:
    run, evidence, _, _ = _source(tmp_path, tools=("inspect_scATAC",))
    with pytest.raises(AnalysisVisualizationError, match="no supported") as caught:
        build_analysis_visualizations(
            run, evidence, tmp_path / "out", registry=_guarded_registry()
        )
    assert caught.value.code == "VISUALIZATION_NO_SUPPORTED_FIGURES"


def test_missing_expected_umap_source_fails_entire_build(tmp_path: Path) -> None:
    run, evidence, umap_path, _ = _source(tmp_path)
    umap_path.unlink()
    with pytest.raises(AnalysisVisualizationError) as caught:
        build_analysis_visualizations(
            run, evidence, tmp_path / "out", registry=_guarded_registry()
        )
    assert caught.value.code == "VISUALIZATION_SOURCE_INVALID"
    assert not (tmp_path / "out" / "analysis_visualizations").exists()


def test_only_explicit_evidence_artifacts_are_considered(tmp_path: Path) -> None:
    run, evidence, umap_path, confusion_path = _source(tmp_path)
    _write_umap(umap_path.parent / "plausible-extra.umap.h5ad")
    (confusion_path.parent / "plausible-extra.json").write_text(
        '{"confusion":"not-authoritative"}', encoding="utf-8"
    )
    result = build_analysis_visualizations(
        run, evidence, tmp_path / "out", registry=_guarded_registry()
    )
    assert result["n_figures"] == 3


def test_repeated_same_environment_render_has_identical_pngs(tmp_path: Path) -> None:
    run, evidence, _, _ = _source(tmp_path)
    first = build_analysis_visualizations(
        run, evidence, tmp_path / "one", registry=_guarded_registry()
    )
    second = build_analysis_visualizations(
        run, evidence, tmp_path / "two", registry=_guarded_registry()
    )
    assert [value["png_sha256"] for value in first["figures"]] == [
        value["png_sha256"] for value in second["figures"]
    ]
    assert Path(first["manifest_path"]).read_bytes() == Path(second["manifest_path"]).read_bytes()


def test_evidence_result_digest_is_authoritative(tmp_path: Path) -> None:
    run, evidence, _, _ = _source(tmp_path)
    evidence["evidence_sha256"] = "0" * 64
    with pytest.raises(AnalysisVisualizationError) as caught:
        build_analysis_visualizations(
            run, evidence, tmp_path / "out", registry=_guarded_registry()
        )
    assert caught.value.code == "EVIDENCE_SHA256_MISMATCH"


def test_source_change_is_detected_by_visualization_verification(tmp_path: Path) -> None:
    run, evidence, umap_path, confusion_path, result = _build(tmp_path)
    _write_umap(umap_path, labels=("9", "9", "8"))
    assert not verify_analysis_visualizations(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    ).passed
    _write_umap(umap_path)
    _write_confusion(confusion_path, count=99)
    assert not verify_analysis_visualizations(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    ).passed


def test_source_verification_failure_before_publication_leaves_no_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _, _ = _source(tmp_path)
    calls = 0

    def verify(*_: object, **__: object) -> VerificationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            return VerificationResult(
                False,
                "analysis_evidence",
                run.run_id,
                (VerificationCheck("fresh", False, "Changed."),),
            )
        return _passed()

    monkeypatch.setattr(visualization_module, "verify_analysis_evidence", verify)
    with pytest.raises(AnalysisVisualizationError):
        build_analysis_visualizations(
            run, evidence, tmp_path / "out", registry=_guarded_registry()
        )
    assert not (tmp_path / "out" / "analysis_visualizations").exists()
    assert not list((tmp_path / "out").glob(".analysis_visualizations.*"))


def test_valid_source_mutation_between_verifications_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, umap_path, _ = _source(tmp_path)
    calls = 0

    def mutate_on_second_verification(*_: object, **__: object) -> VerificationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            artifact = ad.read_h5ad(umap_path)
            artifact.obsm["X_umap"][0, 0] += np.float32(2.0)
            artifact.write_h5ad(umap_path)
        return _passed()

    monkeypatch.setattr(
        visualization_module,
        "verify_analysis_evidence",
        mutate_on_second_verification,
    )
    with pytest.raises(AnalysisVisualizationError) as caught:
        build_analysis_visualizations(
            run, evidence, tmp_path / "out", registry=_guarded_registry()
        )
    assert caught.value.code == "VISUALIZATION_SOURCE_CHANGED"
    assert not (tmp_path / "out" / "analysis_visualizations").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "renamed", "tampered"])
def test_figure_set_and_png_integrity_fail_closed(tmp_path: Path, mutation: str) -> None:
    run, evidence, _, _, result = _build(tmp_path)
    figure_path = Path(result["figures"][0]["figure_path"])
    if mutation == "missing":
        figure_path.unlink()
    elif mutation == "extra":
        (figure_path.parent / "extra.png").write_bytes(b"not a figure")
    elif mutation == "renamed":
        figure_path.rename(figure_path.with_name("renamed.png"))
    else:
        figure_path.write_bytes(figure_path.read_bytes() + b"tamper")
    assert not verify_analysis_visualizations(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    ).passed


def test_png_dimensions_are_verified_independently(tmp_path: Path) -> None:
    run, evidence, _, _, result = _build(tmp_path)
    umap_path = Path(result["figures"][0]["figure_path"])
    metrics_path = Path(result["figures"][1]["figure_path"])
    umap_path.write_bytes(metrics_path.read_bytes())
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["figures"][0]["png_sha256"] = hashlib.sha256(
        umap_path.read_bytes()
    ).hexdigest()
    manifest_path.write_bytes(visualization_module._canonical_json_bytes(manifest))
    verification = verify_analysis_visualizations(
        run, evidence, manifest_path, registry=_guarded_registry()
    )
    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == "VISUALIZATION_FIGURE_DIMENSIONS_INVALID"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"schema_version":NaN}',
    ],
)
def test_manifest_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path, payload: bytes) -> None:
    run, evidence, _, _, result = _build(tmp_path)
    Path(result["manifest_path"]).write_bytes(payload)
    assert not verify_analysis_visualizations(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    ).passed


def test_result_figure_metadata_is_authoritative(tmp_path: Path) -> None:
    run, evidence, _, _, result = _build(tmp_path)
    result["figures"][0]["figure_kind"] = "wrong"
    assert not verify_analysis_visualizations(
        run, evidence, result, registry=_guarded_registry()
    ).passed


def test_overwrite_false_and_failed_overwrite_preserve_existing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _, _ = _source(tmp_path)
    output = tmp_path / "out"
    original = build_analysis_visualizations(
        run, evidence, output, registry=_guarded_registry()
    )
    original_manifest = Path(original["manifest_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        build_analysis_visualizations(run, evidence, output, registry=_guarded_registry())

    real_replace = os.replace

    def fail_staged_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.endswith(".tmp") and destination_path.name == "analysis_visualizations":
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(visualization_module.os, "replace", fail_staged_replace)
    with pytest.raises(AnalysisVisualizationError) as caught:
        build_analysis_visualizations(
            run, evidence, output, registry=_guarded_registry(), overwrite=True
        )
    assert caught.value.code == "VISUALIZATION_PUBLISH_FAILED"
    assert Path(original["manifest_path"]).read_bytes() == original_manifest
    assert verify_analysis_visualizations(
        run, evidence, original, registry=_guarded_registry()
    ).passed
    assert not list(output.glob(".analysis_visualizations.*"))


def test_successful_overwrite_replaces_complete_bundle(tmp_path: Path) -> None:
    run, evidence, umap_path, _ = _source(tmp_path)
    output = tmp_path / "out"
    first = build_analysis_visualizations(
        run, evidence, output, registry=_guarded_registry()
    )
    first_hash = first["figures"][0]["png_sha256"]
    _write_umap(umap_path, labels=("1", "2", "1"))
    second = build_analysis_visualizations(
        run, evidence, output, registry=_guarded_registry(), overwrite=True
    )
    assert second["figures"][0]["png_sha256"] != first_hash
    assert verify_analysis_visualizations(
        run, evidence, second, registry=_guarded_registry()
    ).passed
    assert set(Path(second["bundle_path"]).iterdir()) == {
        Path(second["bundle_path"]) / "figures",
        Path(second["bundle_path"]) / ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    }
