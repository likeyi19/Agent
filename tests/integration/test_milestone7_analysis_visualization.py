from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

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
    build_analysis_visualizations,
    verify_analysis_visualizations,
)
from agent.schemas import (
    AgentPlan,
    AgentRequest,
    AgentRunResult,
    PlanStep,
    RunStatus,
    StepExecutionResult,
    StepOutputRef,
    StepStatus,
)
from agent.tools import (
    build_cell_neighbors,
    cluster_cells,
    compute_cell_umap,
    evaluate_cell_annotation,
    evaluate_cell_clustering,
    transfer_cell_labels,
)


def _guarded_registry(calls: list[str]) -> ToolRegistry:
    registry = build_default_tool_registry()

    def forbidden(**_: object) -> object:
        calls.append("forbidden")
        raise AssertionError("Reporting invoked a scientific callable.")

    return ToolRegistry(
        tuple(replace(registry.get(name), function=forbidden) for name in registry.names())
    )


class _StaticPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self._plan = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self._plan


def _write_ids(path: Path, values: tuple[str, ...]) -> Path:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    return path


def _scientific_source_run(
    tmp_path: Path,
) -> tuple[AgentRunResult, ToolRegistry, tuple[Path, ...]]:
    registry = build_default_tool_registry()

    cell_ids = tuple(f"cell-{index}" for index in range(24))
    embeddings = np.zeros((24, 512), dtype=np.float32)
    embeddings[:12, 0] = np.linspace(0.0, 0.3, 12, dtype=np.float32)
    embeddings[12:, 0] = np.linspace(8.0, 8.3, 12, dtype=np.float32)
    embeddings[:, 1] = np.linspace(-0.2, 0.2, 24, dtype=np.float32)
    embedding_path = tmp_path / "cells.npy"
    ids_path = tmp_path / "cells.txt"
    np.save(embedding_path, embeddings, allow_pickle=False)
    _write_ids(ids_path, cell_ids)

    neighbors_arguments = {
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(ids_path),
        "output_dir": str(tmp_path / "analysis"),
        "n_neighbors": 4,
    }
    neighbors = build_cell_neighbors(**neighbors_arguments)
    cluster_arguments = {
        "analysis_path": neighbors["analysis_path"],
        "output_dir": str(tmp_path / "analysis"),
    }
    clustered = cluster_cells(**cluster_arguments)
    umap_arguments = {
        "analysis_path": clustered["analysis_path"],
        "output_dir": str(tmp_path / "analysis"),
    }
    umap = compute_cell_umap(**umap_arguments)

    cluster_reference_path = tmp_path / "cluster-reference.h5ad"
    cluster_reference = ad.AnnData(
        obs=pd.DataFrame(
            {"celltype": pd.Categorical(["A"] * 12 + ["B"] * 12)},
            index=pd.Index(cell_ids, dtype="object"),
        )
    )
    cluster_reference.write_h5ad(cluster_reference_path)
    cluster_eval_arguments = {
        "analysis_path": clustered["analysis_path"],
        "reference_h5ad_path": str(cluster_reference_path),
        "label_key": "celltype",
        "output_dir": str(tmp_path / "cluster-evaluation"),
    }
    cluster_evaluation = evaluate_cell_clustering(**cluster_eval_arguments)

    reference_ids = tuple(f"reference-{index}" for index in range(8))
    query_ids = tuple(f"query-{index}" for index in range(5))
    reference_embedding = np.zeros((8, 512), dtype=np.float32)
    reference_embedding[4:, 0] = 10.0
    query_embedding = np.zeros((5, 512), dtype=np.float32)
    query_embedding[2:, 0] = 10.0
    reference_embedding_path = tmp_path / "reference.npy"
    query_embedding_path = tmp_path / "query.npy"
    reference_ids_path = _write_ids(tmp_path / "reference.txt", reference_ids)
    query_ids_path = _write_ids(tmp_path / "query.txt", query_ids)
    np.save(reference_embedding_path, reference_embedding, allow_pickle=False)
    np.save(query_embedding_path, query_embedding, allow_pickle=False)
    reference_h5ad_path = tmp_path / "reference.h5ad"
    query_h5ad_path = tmp_path / "query.h5ad"
    ad.AnnData(
        obs=pd.DataFrame(
            {"celltype": pd.Categorical(["A"] * 4 + ["B"] * 4)},
            index=pd.Index(reference_ids, dtype="object"),
        )
    ).write_h5ad(reference_h5ad_path)
    ad.AnnData(obs=pd.DataFrame(index=pd.Index(query_ids, dtype="object"))).write_h5ad(
        query_h5ad_path
    )
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"fixed-checkpoint-identity")
    transfer_arguments = {
        "reference_embedding_path": str(reference_embedding_path),
        "reference_cell_ids_path": str(reference_ids_path),
        "reference_h5ad_path": str(reference_h5ad_path),
        "reference_label_key": "celltype",
        "query_embedding_path": str(query_embedding_path),
        "query_cell_ids_path": str(query_ids_path),
        "query_h5ad_path": str(query_h5ad_path),
        "output_dir": str(tmp_path / "transfer"),
        "reference_species": "mouse",
        "query_species": "mouse",
        "reference_checkpoint_path": str(checkpoint_path),
        "query_checkpoint_path": str(checkpoint_path),
        "n_neighbors": 1,
    }
    transfer = transfer_cell_labels(**transfer_arguments)
    truth_path = tmp_path / "truth.h5ad"
    ad.AnnData(
        obs=pd.DataFrame(
            {"truth": pd.Categorical(["A", "A", "B", "B", "A"])},
            index=pd.Index(query_ids, dtype="object"),
        )
    ).write_h5ad(truth_path)
    annotation_eval_arguments = {
        "annotation_path": transfer["annotation_path"],
        "ground_truth_h5ad_path": str(truth_path),
        "ground_truth_label_key": "truth",
        "output_dir": str(tmp_path / "annotation-evaluation"),
    }
    annotation_evaluation = evaluate_cell_annotation(**annotation_eval_arguments)

    plan = AgentPlan(
        "milestone7-visualization-plan",
        "milestone7-visualization",
        "offline-test-planner",
        (
            PlanStep("neighbors", "build_cell_neighbors", neighbors_arguments),
            PlanStep(
                "cluster",
                "cluster_cells",
                {
                    "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                    "output_dir": str(tmp_path / "analysis"),
                },
                ("neighbors",),
            ),
            PlanStep(
                "umap",
                "compute_cell_umap",
                {
                    "analysis_path": StepOutputRef("cluster", "analysis_path"),
                    "output_dir": str(tmp_path / "analysis"),
                },
                ("cluster",),
            ),
            PlanStep(
                "cluster-evaluation",
                "evaluate_cell_clustering",
                {
                    "analysis_path": StepOutputRef("cluster", "analysis_path"),
                    "reference_h5ad_path": str(cluster_reference_path),
                    "label_key": "celltype",
                    "output_dir": str(tmp_path / "cluster-evaluation"),
                },
                ("cluster",),
            ),
            PlanStep("transfer", "transfer_cell_labels", transfer_arguments),
            PlanStep(
                "annotation-evaluation",
                "evaluate_cell_annotation",
                {
                    "annotation_path": StepOutputRef("transfer", "annotation_path"),
                    "ground_truth_h5ad_path": str(truth_path),
                    "ground_truth_label_key": "truth",
                    "output_dir": str(tmp_path / "annotation-evaluation"),
                },
                ("transfer",),
            ),
        ),
    )
    values = (
        (plan.steps[0], neighbors_arguments, neighbors, {}),
        (plan.steps[1], cluster_arguments, clustered, {"neighbors": neighbors}),
        (plan.steps[2], umap_arguments, umap, {"cluster": clustered}),
        (
            plan.steps[3],
            cluster_eval_arguments,
            cluster_evaluation,
            {"cluster": clustered},
        ),
        (plan.steps[4], transfer_arguments, transfer, {}),
        (
            plan.steps[5],
            annotation_eval_arguments,
            annotation_evaluation,
            {"transfer": transfer},
        ),
    )
    step_results: list[StepExecutionResult] = []
    for step, arguments, result, dependencies in values:
        verification = verify_step(
            step,
            arguments,
            result,
            registry,
            dependency_results=dependencies,
        )
        assert verification.passed
        step_results.append(
            StepExecutionResult(
                step.step_id,
                step.tool_name,
                StepStatus.SUCCEEDED,
                1,
                arguments,
                result,
                verification,
            )
        )
    run = AgentRunResult(
        "milestone7-visualization:run",
        "milestone7-visualization",
        RunStatus.SUCCEEDED,
        False,
        plan=plan,
        steps=tuple(step_results),
        verification=verify_run(plan, tuple(step_results)),
    )
    sources = (
        embedding_path,
        ids_path,
        Path(neighbors["analysis_path"]),
        Path(clustered["analysis_path"]),
        Path(umap["analysis_path"]),
        Path(cluster_evaluation["report_path"]),
        reference_embedding_path,
        query_embedding_path,
        reference_ids_path,
        query_ids_path,
        reference_h5ad_path,
        query_h5ad_path,
        Path(transfer["annotation_path"]),
        Path(annotation_evaluation["report_path"]),
    )
    return run, registry, sources


def test_offline_verified_evidence_to_verified_visualizations(tmp_path: Path) -> None:
    run, _, sources = _scientific_source_run(tmp_path)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    calls: list[str] = []
    guarded_registry = _guarded_registry(calls)

    evidence = build_analysis_evidence(
        run, tmp_path / "evidence", registry=guarded_registry
    )
    visualization = build_analysis_visualizations(
        run,
        evidence,
        tmp_path / "visualization",
        registry=guarded_registry,
    )
    verification = verify_analysis_visualizations(
        run, evidence, visualization, registry=guarded_registry
    )

    assert verification.passed
    assert calls == []
    assert visualization["n_figures"] == 3
    assert [value["figure_kind"] for value in visualization["figures"]] == [
        "annotation_confusion",
        "umap_leiden",
        "clustering_metrics",
    ]
    assert all(Path(value["figure_path"]).is_file() for value in visualization["figures"])
    assert json.loads(json.dumps(visualization)) == visualization
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources} == before


def test_terminal_resume_reporting_is_fresh_zero_execution_and_state_immutable(
    tmp_path: Path,
) -> None:
    source_run, source_registry, _ = _scientific_source_run(tmp_path)
    assert source_run.plan is not None
    result_by_tool = {
        step.tool_name: step.result for step in source_run.steps if step.result is not None
    }
    calls: list[str] = []

    def replacement(name: str):
        def return_existing(**_: object) -> object:
            calls.append(name)
            return result_by_tool[name]

        return return_existing

    execution_registry = ToolRegistry(
        tuple(
            replace(
                source_registry.get(name),
                function=(
                    replacement(name)
                    if name in result_by_tool
                    else source_registry.get(name).function
                ),
            )
            for name in source_registry.names()
        )
    )
    planner = _StaticPlanner(source_run.plan)
    store = FileRunStore(tmp_path / "run-store")
    runtime = AgentRuntime(
        planner=planner,
        registry=execution_registry,
        run_store=store,
    )
    request = AgentRequest(
        source_run.request_id,
        "Render accepted downstream analysis.",
        {},
    )
    completed = runtime.run(request)
    assert completed.status is RunStatus.SUCCEEDED
    assert len(calls) == len(source_run.plan.steps)
    assert planner.calls == 1

    terminal = runtime.resume(completed.run_id)
    assert terminal.status is RunStatus.SUCCEEDED
    assert len(calls) == len(source_run.plan.steps)
    assert planner.calls == 1
    state_before = store.load(terminal.run_id).to_dict()

    report_calls: list[str] = []
    guarded_registry = _guarded_registry(report_calls)
    evidence = build_analysis_evidence(
        terminal, tmp_path / "terminal-evidence", registry=guarded_registry
    )
    visualization = build_analysis_visualizations(
        terminal,
        evidence,
        tmp_path / "terminal-visualization",
        registry=guarded_registry,
    )
    assert verify_analysis_visualizations(
        terminal, evidence, visualization, registry=guarded_registry
    ).passed
    assert report_calls == []
    assert store.load(terminal.run_id).to_dict() == state_before

    umap_step = next(step for step in terminal.steps if step.tool_name == "compute_cell_umap")
    assert umap_step.result is not None
    umap_path = Path(str(umap_step.result["analysis_path"]))
    artifact = ad.read_h5ad(umap_path)
    artifact.obsm["X_umap"][0, 0] += np.float32(1.0)
    artifact.write_h5ad(umap_path)
    assert not verify_analysis_visualizations(
        terminal, evidence, visualization, registry=guarded_registry
    ).passed
    assert report_calls == []
    assert store.load(terminal.run_id).to_dict() == state_before
