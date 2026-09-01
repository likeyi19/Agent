from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.orchestration import (
    AgentRuntime,
    FileRunStore,
    ToolRegistry,
    build_default_tool_registry,
    verify_run,
    verify_step,
)
from agent.report import (
    build_analysis_evidence,
    build_analysis_report,
    build_analysis_visualizations,
    verify_analysis_report,
)
from agent.schemas import (
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    PlanStep,
    RunStatus,
    StepExecutionResult,
    StepStatus,
)
from agent.tools import inspect_scATAC

from test_milestone7_analysis_visualization import _scientific_source_run


class _StaticPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self.plan_value


def _guarded_registry(calls: list[str]) -> ToolRegistry:
    registry = build_default_tool_registry()

    def forbidden(**_: object) -> object:
        calls.append("forbidden")
        raise AssertionError("Report processing invoked a scientific callable.")

    return ToolRegistry(
        tuple(replace(registry.get(name), function=forbidden) for name in registry.names())
    )


def _inspection_run(tmp_path: Path) -> tuple[AgentRunResult, Path]:
    input_path = tmp_path / "inspection.h5ad"
    artifact = ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[0, 1, 0], [2, 0, 3]], dtype=np.int32)),
        obs=pd.DataFrame(index=pd.Index(["cell-1", "cell-2"], dtype="object")),
        var=pd.DataFrame(index=pd.Index(["peak-1", "peak-2", "peak-3"], dtype="object")),
    )
    artifact.write_h5ad(input_path)
    arguments = {"path": str(input_path)}
    result = inspect_scATAC(**arguments)
    registry = build_default_tool_registry()
    step = PlanStep("inspect", "inspect_scATAC", arguments)
    plan = AgentPlan(
        "milestone7-report-inspection-plan",
        "milestone7-report-inspection",
        "offline-test-planner",
        (step,),
    )
    verification = verify_step(step, arguments, result, registry)
    assert verification.passed
    step_result = StepExecutionResult(
        step.step_id,
        step.tool_name,
        StepStatus.SUCCEEDED,
        1,
        arguments,
        result,
        verification,
    )
    run = AgentRunResult(
        "milestone7-report-inspection:run",
        "milestone7-report-inspection",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=(step_result,),
        verification=verify_run(plan, (step_result,)),
    )
    return run, input_path


def test_offline_inspection_evidence_to_verified_figureless_report(tmp_path: Path) -> None:
    run, source = _inspection_run(tmp_path)
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    calls: list[str] = []
    registry = _guarded_registry(calls)

    evidence = build_analysis_evidence(run, tmp_path / "evidence", registry=registry)
    report = build_analysis_report(
        run, evidence, tmp_path / "report", registry=registry
    )
    verification = verify_analysis_report(run, evidence, report, registry=registry)

    assert verification.passed
    assert calls == []
    assert report["section_ids"] == [
        "analysis_summary",
        "dataset",
        "methods",
        "provenance",
    ]
    assert report["n_figures"] == 0
    assert json.loads(json.dumps(report)) == report
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before


def test_offline_full_evidence_visualization_report_chain(tmp_path: Path) -> None:
    run, _, scientific_sources = _scientific_source_run(tmp_path)
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in scientific_sources
    }
    calls: list[str] = []
    registry = _guarded_registry(calls)

    evidence = build_analysis_evidence(run, tmp_path / "evidence", registry=registry)
    visualization = build_analysis_visualizations(
        run, evidence, tmp_path / "visualization", registry=registry
    )
    first = build_analysis_report(
        run,
        evidence,
        tmp_path / "report-1",
        registry=registry,
        visualization=visualization,
    )
    second = build_analysis_report(
        run,
        evidence,
        tmp_path / "report-2",
        registry=registry,
        visualization=visualization,
    )

    assert verify_analysis_report(
        run,
        evidence,
        first,
        registry=registry,
        visualization=visualization,
    ).passed
    assert calls == []
    assert first["section_ids"] == [
        "analysis_summary",
        "clustering_umap",
        "clustering_evaluation",
        "cell_annotation",
        "annotation_evaluation",
        "figures",
        "methods",
        "provenance",
    ]
    assert [figure["figure_kind"] for figure in first["figures"]] == [
        "annotation_confusion",
        "umap_leiden",
        "clustering_metrics",
    ]
    assert Path(first["report_path"]).read_bytes() == Path(second["report_path"]).read_bytes()
    assert Path(first["manifest_path"]).read_bytes() == Path(second["manifest_path"]).read_bytes()

    visualization_manifest = json.loads(
        Path(visualization["manifest_path"]).read_text(encoding="utf-8")
    )
    report_manifest = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    for source_figure, copied_figure in zip(
        visualization_manifest["figures"], report_manifest["figures"], strict=True
    ):
        source_path = Path(visualization["bundle_path"]) / source_figure["relative_path"]
        copied_path = Path(first["bundle_path"]) / copied_figure["report_relative_path"]
        assert source_path.read_bytes() == copied_path.read_bytes()
        assert source_figure["png_sha256"] == copied_figure["copied_png_sha256"]
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in scientific_sources
    } == before


def test_terminal_resume_report_processing_leaves_durable_state_unchanged(
    tmp_path: Path,
) -> None:
    source_run, _ = _inspection_run(tmp_path)
    assert source_run.plan is not None
    source_step = source_run.steps[0]
    assert source_step.result is not None
    execution_calls: list[str] = []
    default_registry = build_default_tool_registry()

    def return_inspection(**_: object) -> object:
        execution_calls.append("inspect")
        return source_step.result

    execution_registry = ToolRegistry(
        tuple(
            replace(
                default_registry.get(name),
                function=(
                    return_inspection
                    if name == "inspect_scATAC"
                    else default_registry.get(name).function
                ),
            )
            for name in default_registry.names()
        )
    )
    planner = _StaticPlanner(source_run.plan)
    store = FileRunStore(tmp_path / "run-store")
    runtime = AgentRuntime(planner=planner, registry=execution_registry, run_store=store)
    completed = runtime.run(
        AgentRequest(source_run.request_id, "Inspect the supplied dataset.", {})
    )
    terminal = runtime.resume(completed.run_id)
    assert terminal.status is RunStatus.SUCCEEDED
    assert execution_calls == ["inspect"]
    assert planner.calls == 1
    state_before = store.load(terminal.run_id).to_dict()

    report_calls: list[str] = []
    guarded_registry = _guarded_registry(report_calls)
    evidence = build_analysis_evidence(
        terminal, tmp_path / "terminal-evidence", registry=guarded_registry
    )
    report = build_analysis_report(
        terminal, evidence, tmp_path / "terminal-report", registry=guarded_registry
    )
    assert verify_analysis_report(
        terminal, evidence, report, registry=guarded_registry
    ).passed
    assert report_calls == []
    assert store.load(terminal.run_id).to_dict() == state_before
