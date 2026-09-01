from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

import agent.report.analysis_report as report_module
from agent.orchestration import ToolRegistry, build_default_tool_registry
from agent.report import (
    ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
    ANALYSIS_EVIDENCE_SCHEMA_VERSION,
    ANALYSIS_REPORT_ARTIFACT_TYPE,
    ANALYSIS_REPORT_MANIFEST_FILENAME,
    ANALYSIS_REPORT_SCHEMA_VERSION,
    ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
    ANALYSIS_VISUALIZATION_MANIFEST_FILENAME,
    ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
    AnalysisReportError,
    build_analysis_report,
    verify_analysis_report,
)
from agent.schemas import (
    AgentPlan,
    AgentRunResult,
    PlanStep,
    RunStatus,
    VerificationCheck,
    VerificationResult,
)


_FACT_VALUES: dict[str, dict[str, object]] = {
    "inspect_scATAC": {
        "input_path": "/data/input.h5ad",
        "n_cells": 2000,
        "n_features": 1_341_077,
        "x_storage_type": "csr_matrix",
        "x_is_sparse": True,
        "x_dtype": "float32",
        "nnz": 123456,
        "density": 0.0000460321,
    },
    "epizoo_embed_cells": {
        "n_cells": 2000,
        "embedding_dim": 512,
        "embedding_dtype": "float32",
        "species": "mouse",
        "backend": "EpiZoo",
        "checkpoint_path": "/models/epizoo.pth",
        "device": "cuda:0",
        "finite": True,
        "cell_order_preserved": True,
    },
    "build_cell_neighbors": {
        "n_cells": 2000,
        "embedding_dim": 512,
        "n_neighbors": 15,
        "metric": "euclidean",
        "neighbors_method": "umap",
        "transformer": None,
        "random_seed": 0,
        "connectivities_nnz": 40000,
        "distances_nnz": 30000,
        "backend": "scanpy",
    },
    "cluster_cells": {
        "n_cells": 2000,
        "n_clusters": 21,
        "cluster_key": "leiden",
        "algorithm": "leiden",
        "resolution": 1.0,
        "random_seed": 0,
        "backend": "scanpy",
    },
    "compute_cell_umap": {
        "n_cells": 2000,
        "n_components": 2,
        "umap_key": "X_umap",
        "coordinate_dtype": "float32",
        "min_dist": 0.5,
        "spread": 1.0,
        "random_seed": 0,
        "backend": "scanpy",
    },
    "evaluate_cell_clustering": {
        "n_cells": 2000,
        "n_reference_classes": 20,
        "n_predicted_clusters": 21,
        "nmi": 0.8642463249536162,
        "ari": 0.746014277040041,
        "ami": 0.8591719263671213,
        "homogeneity": 0.854796248075491,
        "average_method": "arithmetic",
        "metric_backend": "sklearn",
    },
    "transfer_cell_labels": {
        "checkpoint_path": "/models/epizoo.pth",
        "reference_label_key": "celltype",
        "n_reference_cells": 1400,
        "n_query_cells": 600,
        "n_reference_classes": 20,
        "assigned_count": 596,
        "unassigned_count": 4,
        "assignment_rate": 0.9933333333,
        "embedding_dim": 512,
        "n_neighbors": 20,
        "metric": "euclidean",
        "voting_method": "uniform_plurality",
        "min_confidence": 0.0,
        "species": "mouse",
        "species_compatible": True,
        "checkpoint_compatible": True,
        "backend": "sklearn",
    },
    "evaluate_cell_annotation": {
        "annotation_sha256": "a" * 64,
        "ground_truth_label_key": "celltype",
        "n_cells": 600,
        "n_ground_truth_classes": 20,
        "n_assigned_predicted_classes": 19,
        "assigned_count": 596,
        "unassigned_count": 4,
        "assignment_rate": 0.9933333333,
        "correct_assigned_count": 543,
        "incorrect_assigned_count": 53,
        "overall_accuracy": 0.905,
        "assigned_accuracy": None,
        "macro_f1": 0.861567991,
        "median_confidence": 1.0,
        "median_assigned_confidence": 1.0,
        "median_correct_assigned_confidence": 1.0,
        "median_incorrect_assigned_confidence": 0.6,
        "metric_backend": "sklearn",
        "macro_average": "ground_truth_classes",
        "zero_division": 0,
    },
}

_ALL_TOOLS = tuple(_FACT_VALUES)


def _passed(target_type: str = "analysis_evidence") -> VerificationResult:
    return VerificationResult(
        True,
        target_type,
        "run-1",
        (VerificationCheck("accepted", True, "Accepted."),),
    )


def _guarded_registry(calls: list[str] | None = None) -> ToolRegistry:
    registry = build_default_tool_registry()

    def forbidden(**_: object) -> object:
        if calls is not None:
            calls.append("forbidden")
        raise AssertionError("Report processing invoked a scientific callable.")

    return ToolRegistry(
        tuple(replace(registry.get(name), function=forbidden) for name in registry.names())
    )


@pytest.fixture(autouse=True)
def _fresh_sources_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        report_module,
        "verify_analysis_evidence",
        lambda *args, **kwargs: _passed(),
    )
    monkeypatch.setattr(
        report_module,
        "verify_analysis_visualizations",
        lambda *args, **kwargs: _passed("analysis_visualizations"),
    )


def _source(
    tmp_path: Path,
    tools: tuple[str, ...],
    *,
    facts_override: dict[str, dict[str, object]] | None = None,
    step_ids: tuple[str, ...] | None = None,
) -> tuple[AgentRunResult, dict[str, object], dict[str, object]]:
    ids = step_ids or tuple(f"step-{index}" for index in range(len(tools)))
    plan = AgentPlan(
        "plan-1",
        "request-1",
        "test-planner",
        tuple(PlanStep(step_id, tool, {}) for step_id, tool in zip(ids, tools, strict=True)),
    )
    run = AgentRunResult(
        "run-1",
        "request-1",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        verification=_passed("agent_run"),
    )
    evidence_path = tmp_path / "analysis_evidence.json"
    steps = []
    for step_id, tool in zip(ids, tools, strict=True):
        values = dict(_FACT_VALUES.get(tool, {}))
        if facts_override and tool in facts_override:
            values.update(facts_override[tool])
        steps.append({"step_id": step_id, "tool_name": tool, "facts": values})
    payload = {
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "status": "success",
        "run": {"run_id": run.run_id, "request_id": run.request_id, "plan_id": plan.plan_id},
        "workflow": {
            "ordered_steps": [
                {"step_id": step_id, "tool_name": tool}
                for step_id, tool in zip(ids, tools, strict=True)
            ]
        },
        "steps": steps,
        "artifacts": [],
    }
    evidence_bytes = report_module._canonical_json_bytes(payload)
    evidence_path.write_bytes(evidence_bytes)
    evidence = {
        "status": "success",
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "schema_version": ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_EVIDENCE_ARTIFACT_TYPE,
        "run_id": run.run_id,
        "request_id": run.request_id,
        "plan_id": plan.plan_id,
        "n_steps": len(tools),
        "tool_names": list(tools),
        "all_steps_verified": True,
    }
    return run, evidence, payload


def _visualization(
    tmp_path: Path,
    run: AgentRunResult,
    evidence: dict[str, object],
    *,
    kinds: tuple[str, ...] = (
        "umap_leiden",
        "clustering_metrics",
        "annotation_confusion",
    ),
) -> tuple[Path, tuple[bytes, ...]]:
    bundle = tmp_path / "analysis_visualizations"
    figures_dir = bundle / "figures"
    figures_dir.mkdir(parents=True)
    step_by_kind = {
        "umap_leiden": "step-4",
        "clustering_metrics": "step-5",
        "annotation_confusion": "step-7",
    }
    figures = []
    payloads = []
    for index, kind in enumerate(kinds, start=1):
        figure_id = f"FIG{index:04d}"
        filename = f"{index:02d}_{kind}.png"
        content = b"\x89PNG\r\n\x1a\n" + figure_id.encode("ascii")
        (figures_dir / filename).write_bytes(content)
        payloads.append(content)
        figures.append(
            {
                "figure_id": figure_id,
                "figure_kind": kind,
                "source_step_ids": [step_by_kind[kind]],
                "relative_path": f"figures/{filename}",
                "png_sha256": hashlib.sha256(content).hexdigest(),
                "width_px": 800,
                "height_px": 600,
                "dpi": 100,
            }
        )
    manifest = {
        "schema_version": ANALYSIS_VISUALIZATION_SCHEMA_VERSION,
        "artifact_type": ANALYSIS_VISUALIZATION_ARTIFACT_TYPE,
        "status": "success",
        "source": {
            "run_id": run.run_id,
            "request_id": run.request_id,
            "plan_id": run.plan.plan_id if run.plan is not None else "",
            "evidence_path": evidence["evidence_path"],
            "evidence_sha256": evidence["evidence_sha256"],
        },
        "figures": figures,
    }
    manifest_path = bundle / ANALYSIS_VISUALIZATION_MANIFEST_FILENAME
    manifest_path.write_bytes(report_module._canonical_json_bytes(manifest))
    return manifest_path, tuple(payloads)


def _build(
    tmp_path: Path,
    tools: tuple[str, ...] = ("inspect_scATAC",),
    *,
    with_visualization: bool = False,
    facts_override: dict[str, dict[str, object]] | None = None,
    step_ids: tuple[str, ...] | None = None,
    output_name: str = "output",
) -> tuple[AgentRunResult, dict[str, object], Path | None, dict[str, object]]:
    run, evidence, _ = _source(
        tmp_path, tools, facts_override=facts_override, step_ids=step_ids
    )
    visualization_path = (
        _visualization(tmp_path, run, evidence)[0] if with_visualization else None
    )
    result = build_analysis_report(
        run,
        evidence,
        tmp_path / output_name,
        registry=_guarded_registry(),
        visualization=visualization_path,
    )
    return run, evidence, visualization_path, result


def test_inspection_only_report_is_strict_and_verifiable(tmp_path: Path) -> None:
    calls: list[str] = []
    run, evidence, _, result = _build(tmp_path)
    assert result["artifact_type"] == ANALYSIS_REPORT_ARTIFACT_TYPE
    assert result["schema_version"] == ANALYSIS_REPORT_SCHEMA_VERSION
    assert result["section_ids"] == [
        "analysis_summary",
        "dataset",
        "methods",
        "provenance",
    ]
    assert result["n_figures"] == 0
    assert not (Path(result["bundle_path"]) / "figures").exists()
    assert verify_analysis_report(
        run, evidence, result, registry=_guarded_registry(calls)
    ).passed
    assert calls == []


@pytest.mark.parametrize(
    ("tools", "required", "absent"),
    [
        (("cluster_cells", "compute_cell_umap"), "Clustering and UMAP", "Clustering Evaluation"),
        (("evaluate_cell_clustering",), "Clustering Evaluation", "Cell Annotation"),
        (("transfer_cell_labels",), "Cell Annotation", "Annotation Evaluation"),
        (("evaluate_cell_annotation",), "Annotation Evaluation", "Cell Annotation"),
    ],
)
def test_conditional_sections_are_exact(
    tmp_path: Path, tools: tuple[str, ...], required: str, absent: str
) -> None:
    _, _, _, result = _build(tmp_path, tools)
    markdown = Path(result["report_path"]).read_text(encoding="utf-8")
    assert f"## {required}" in markdown
    assert f"## {absent}" not in markdown


def test_full_report_has_fixed_sections_facts_and_exact_values(tmp_path: Path) -> None:
    run, evidence, visualization, result = _build(
        tmp_path, _ALL_TOOLS, with_visualization=True
    )
    assert result["section_ids"] == [
        "analysis_summary",
        "dataset",
        "epizoo_representation",
        "clustering_umap",
        "clustering_evaluation",
        "cell_annotation",
        "annotation_evaluation",
        "figures",
        "methods",
        "provenance",
    ]
    markdown = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "0.8642463249536162" in markdown
    assert '` null ` (undefined)' in markdown
    assert "well separated" not in markdown.lower()
    assert "excellent" not in markdown.lower()
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    facts = manifest["content"]["facts"]
    assert [fact["fact_id"] for fact in facts] == [
        f"F{index:04d}" for index in range(1, len(facts) + 1)
    ]
    assert manifest["content"]["section_bindings"][0]["fact_ids"] == [
        fact["fact_id"] for fact in facts
    ]
    assert verify_analysis_report(
        run,
        evidence,
        result,
        registry=_guarded_registry(),
        visualization=visualization,
    ).passed


def test_frozen_field_order_multiple_steps_and_future_fields(tmp_path: Path) -> None:
    run, evidence, _ = _source(
        tmp_path,
        ("cluster_cells", "cluster_cells"),
        step_ids=("z-step", "a-step"),
        facts_override={"cluster_cells": {"future_interpretation": "excellent"}},
    )
    result = build_analysis_report(
        run, evidence, tmp_path / "out", registry=_guarded_registry()
    )
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    facts = manifest["content"]["facts"]
    expected_fields = list(report_module._REPORT_FIELDS["cluster_cells"])
    assert [fact["source_step_id"] for fact in facts] == ["z-step"] * len(
        expected_fields
    ) + ["a-step"] * len(expected_fields)
    assert [fact["field"] for fact in facts] == expected_fields * 2
    assert "future_interpretation" not in Path(result["report_path"]).read_text(
        encoding="utf-8"
    )


def test_markdown_injection_is_inert_inline_code(tmp_path: Path) -> None:
    malicious = "source`\n## Injected\n![x](bad.png)"
    _, _, _, result = _build(
        tmp_path,
        ("inspect_scATAC",),
        facts_override={"inspect_scATAC": {"input_path": malicious}},
        step_ids=("step`\n# forged",),
    )
    markdown = Path(result["report_path"]).read_text(encoding="utf-8")
    assert "\n## Injected\n" not in markdown
    assert "\n# forged\n" not in markdown
    assert json.dumps(malicious, ensure_ascii=False) in markdown


def test_unsupported_tool_fails_closed(tmp_path: Path) -> None:
    run, evidence, _ = _source(tmp_path, ("future_scientific_tool",))
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(run, evidence, tmp_path / "out", registry=_guarded_registry())
    assert caught.value.code == "REPORT_NO_REPORTABLE_CONTENT"


def test_visualization_copies_every_figure_in_exact_order_and_bytes(tmp_path: Path) -> None:
    run, evidence, visualization, result = _build(
        tmp_path, _ALL_TOOLS, with_visualization=True
    )
    assert visualization is not None
    source_manifest = json.loads(visualization.read_text(encoding="utf-8"))
    report_manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert [value["figure_id"] for value in report_manifest["figures"]] == [
        value["figure_id"] for value in source_manifest["figures"]
    ]
    for source, copied in zip(source_manifest["figures"], report_manifest["figures"], strict=True):
        source_bytes = (visualization.parent / source["relative_path"]).read_bytes()
        copied_bytes = (Path(result["bundle_path"]) / copied["report_relative_path"]).read_bytes()
        assert source_bytes == copied_bytes
        assert copied["source_png_sha256"] == copied["copied_png_sha256"]
        assert copied["associated_fact_ids"]
    assert verify_analysis_report(
        run,
        evidence,
        result,
        registry=_guarded_registry(),
        visualization=visualization,
    ).passed


def test_repeated_builds_are_byte_identical(tmp_path: Path) -> None:
    run, evidence, _ = _source(tmp_path, _ALL_TOOLS)
    visualization, _ = _visualization(tmp_path, run, evidence)
    first = build_analysis_report(
        run,
        evidence,
        tmp_path / "out-1",
        registry=_guarded_registry(),
        visualization=visualization,
    )
    second = build_analysis_report(
        run,
        evidence,
        tmp_path / "out-2",
        registry=_guarded_registry(),
        visualization=visualization,
    )
    assert Path(first["report_path"]).read_bytes() == Path(second["report_path"]).read_bytes()
    assert Path(first["manifest_path"]).read_bytes() == Path(second["manifest_path"]).read_bytes()
    assert b"timestamp" not in Path(first["manifest_path"]).read_bytes().lower()


@pytest.mark.parametrize("mutation", ["markdown", "manifest", "extra", "missing_figure", "copied_figure"])
def test_exact_bundle_mutations_fail_verification(tmp_path: Path, mutation: str) -> None:
    run, evidence, visualization, result = _build(
        tmp_path, _ALL_TOOLS, with_visualization=True
    )
    bundle = Path(result["bundle_path"])
    if mutation == "markdown":
        Path(result["report_path"]).write_bytes(Path(result["report_path"]).read_bytes() + b"extra\n")
    elif mutation == "manifest":
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        manifest["content"]["ordered_sections"].append("extra")
        Path(result["manifest_path"]).write_bytes(report_module._canonical_json_bytes(manifest))
    elif mutation == "extra":
        (bundle / "extra.txt").write_text("extra", encoding="utf-8")
    else:
        figure = Path(result["figures"][0]["figure_path"])
        if mutation == "missing_figure":
            figure.unlink()
        else:
            figure.write_bytes(figure.read_bytes() + b"changed")
    assert not verify_analysis_report(
        run,
        evidence,
        result["manifest_path"],
        registry=_guarded_registry(),
        visualization=visualization,
    ).passed


@pytest.mark.parametrize(
    "payload",
    [b'{"schema_version":1,"schema_version":1}', b'{"schema_version":NaN}'],
)
def test_manifest_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path: Path, payload: bytes
) -> None:
    run, evidence, _, result = _build(tmp_path)
    Path(result["manifest_path"]).write_bytes(payload)
    assert not verify_analysis_report(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    ).passed


def test_result_mapping_is_authoritative(tmp_path: Path) -> None:
    run, evidence, _, result = _build(tmp_path)
    result["n_sections"] = 999
    assert not verify_analysis_report(run, evidence, result, registry=_guarded_registry()).passed


def test_source_binding_and_source_figure_integrity_fail_closed(tmp_path: Path) -> None:
    run, evidence, visualization, _ = _build(
        tmp_path, _ALL_TOOLS, with_visualization=True
    )
    assert visualization is not None
    source_manifest = json.loads(visualization.read_text(encoding="utf-8"))
    source_figure = visualization.parent / source_manifest["figures"][0]["relative_path"]
    source_figure.write_bytes(source_figure.read_bytes() + b"tampered")
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(
            run,
            evidence,
            tmp_path / "second-output",
            registry=_guarded_registry(),
            visualization=visualization,
        )
    assert caught.value.code == "REPORT_SOURCE_VISUALIZATION_INVALID"


def test_visualization_run_binding_mismatch_is_rejected(tmp_path: Path) -> None:
    run, evidence, _ = _source(tmp_path, _ALL_TOOLS)
    visualization, _ = _visualization(tmp_path, run, evidence)
    payload = json.loads(visualization.read_text(encoding="utf-8"))
    payload["source"]["run_id"] = "another-run"
    visualization.write_bytes(report_module._canonical_json_bytes(payload))
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(
            run,
            evidence,
            tmp_path / "out",
            registry=_guarded_registry(),
            visualization=visualization,
        )
    assert caught.value.code == "REPORT_SOURCE_BINDING_MISMATCH"


def test_report_declaring_visualization_requires_explicit_source(tmp_path: Path) -> None:
    run, evidence, _, result = _build(tmp_path, _ALL_TOOLS, with_visualization=True)
    verification = verify_analysis_report(
        run, evidence, result["manifest_path"], registry=_guarded_registry()
    )
    assert not verification.passed
    assert verification.error is not None
    assert verification.error.code == "REPORT_SOURCE_VISUALIZATION_INVALID"


def test_source_verifier_exception_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _ = _source(tmp_path, ("inspect_scATAC",))

    def fail(*args: object, **kwargs: object) -> VerificationResult:
        raise RuntimeError("secret-bearing source failure")

    monkeypatch.setattr(report_module, "verify_analysis_evidence", fail)
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(run, evidence, tmp_path / "out", registry=_guarded_registry())
    assert caught.value.code == "REPORT_SOURCE_EVIDENCE_INVALID"
    assert "secret-bearing" not in str(caught.value)


def test_source_change_on_second_fresh_verification_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _ = _source(tmp_path, ("inspect_scATAC",))
    calls = 0

    def verify(*args: object, **kwargs: object) -> VerificationResult:
        nonlocal calls
        calls += 1
        if calls == 2:
            return VerificationResult(
                False,
                "analysis_evidence",
                run.run_id,
                (VerificationCheck("changed", False, "Changed."),),
            )
        return _passed()

    monkeypatch.setattr(report_module, "verify_analysis_evidence", verify)
    output = tmp_path / "out"
    with pytest.raises(AnalysisReportError):
        build_analysis_report(run, evidence, output, registry=_guarded_registry())
    assert not (output / "analysis_report").exists()
    assert not list(output.glob(".analysis_report.*"))


def test_overwrite_conflict_and_failed_overwrite_restore_previous_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _ = _source(tmp_path, ("inspect_scATAC",))
    output = tmp_path / "out"
    first = build_analysis_report(run, evidence, output, registry=_guarded_registry())
    original_manifest = Path(first["manifest_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        build_analysis_report(run, evidence, output, registry=_guarded_registry())

    real_replace = os.replace

    def fail_staged_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.endswith(".tmp") and destination_path.name == "analysis_report":
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_staged_replace)
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(
            run, evidence, output, registry=_guarded_registry(), overwrite=True
        )
    assert caught.value.code == "REPORT_PUBLISH_FAILED"
    assert Path(first["manifest_path"]).read_bytes() == original_manifest
    assert verify_analysis_report(run, evidence, first, registry=_guarded_registry()).passed
    assert not list(output.glob(".analysis_report.*"))


def test_failed_initial_publication_leaves_no_completed_or_staged_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _ = _source(tmp_path, ("inspect_scATAC",))
    output = tmp_path / "out"
    real_replace = os.replace

    def fail_staged_replace(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.endswith(".tmp") and destination_path.name == "analysis_report":
            raise OSError("simulated publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_staged_replace)
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(run, evidence, output, registry=_guarded_registry())
    assert caught.value.code == "REPORT_PUBLISH_FAILED"
    assert not (output / "analysis_report").exists()
    assert not list(output.glob(".analysis_report.*"))


def test_rollback_failure_has_distinct_fail_closed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, evidence, _ = _source(tmp_path, ("inspect_scATAC",))
    output = tmp_path / "out"
    build_analysis_report(run, evidence, output, registry=_guarded_registry())
    real_replace = os.replace

    def fail_publication_and_rollback(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "analysis_report" and (
            source_path.name.endswith(".tmp") or source_path.name.endswith(".backup")
        ):
            raise OSError("simulated replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(report_module.os, "replace", fail_publication_and_rollback)
    with pytest.raises(AnalysisReportError) as caught:
        build_analysis_report(
            run, evidence, output, registry=_guarded_registry(), overwrite=True
        )
    assert caught.value.code == "REPORT_ROLLBACK_FAILED"


def test_successful_overwrite_replaces_complete_report(tmp_path: Path) -> None:
    run, evidence, payload = _source(tmp_path, ("inspect_scATAC",))
    output = tmp_path / "out"
    first = build_analysis_report(run, evidence, output, registry=_guarded_registry())
    first_report = Path(first["report_path"]).read_bytes()
    payload["steps"][0]["facts"]["n_cells"] = 7
    evidence_bytes = report_module._canonical_json_bytes(payload)
    Path(evidence["evidence_path"]).write_bytes(evidence_bytes)
    evidence["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    second = build_analysis_report(
        run, evidence, output, registry=_guarded_registry(), overwrite=True
    )
    assert Path(second["report_path"]).read_bytes() != first_report
    assert verify_analysis_report(run, evidence, second, registry=_guarded_registry()).passed
    assert {path.name for path in Path(second["bundle_path"]).iterdir()} == {
        "analysis_report.md",
        ANALYSIS_REPORT_MANIFEST_FILENAME,
    }


def test_no_scientific_callable_is_invoked(tmp_path: Path) -> None:
    calls: list[str] = []
    run, evidence, _ = _source(tmp_path, _ALL_TOOLS)
    registry = _guarded_registry(calls)
    result = build_analysis_report(run, evidence, tmp_path / "out", registry=registry)
    assert verify_analysis_report(run, evidence, result, registry=registry).passed
    assert calls == []
