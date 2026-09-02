from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
from pathlib import Path
import shutil

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from agent.application import ApplicationStatus, ResearchAgentApplication
from agent.orchestration import LLMPlanner, ToolRegistry, build_default_tool_registry
from agent.report import verify_analysis_report
from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepOutputRef


class _DownstreamPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        output_dir = request.inputs["output_dir"]
        return AgentPlan(
            f"{request.request_id}:downstream-plan",
            request.request_id,
            "offline-downstream-planner",
            (
                PlanStep(
                    "neighbors",
                    "build_cell_neighbors",
                    {
                        "embedding_path": request.inputs["embedding_path"],
                        "cell_ids_path": request.inputs["cell_ids_path"],
                        "output_dir": output_dir,
                        "n_neighbors": 4,
                    },
                ),
                PlanStep(
                    "cluster",
                    "cluster_cells",
                    {
                        "analysis_path": StepOutputRef("neighbors", "analysis_path"),
                        "output_dir": output_dir,
                    },
                    ("neighbors",),
                ),
                PlanStep(
                    "umap",
                    "compute_cell_umap",
                    {
                        "analysis_path": StepOutputRef("cluster", "analysis_path"),
                        "output_dir": output_dir,
                    },
                    ("cluster",),
                ),
            ),
        )


class _FixedPlanningModel:
    model_id = "milestone7-application-fake-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls.append((prompt, response_schema))
        return json.dumps(
            {
                "schema_version": 2,
                "status": "plan",
                "steps": [
                    {
                        "step_id": "inspect",
                        "tool_name": "inspect_scATAC",
                        "arguments": [
                            {
                                "name": "path",
                                "binding_type": "input",
                                "input_name": "input_path",
                                "ref_step_id": None,
                                "ref_output_key": None,
                            }
                        ],
                        "depends_on": [],
                        "description": "Inspect the supplied scATAC dataset.",
                    }
                ],
                "reason": None,
            }
        )


def _counting_registry(calls: list[str]) -> ToolRegistry:
    default = build_default_tool_registry()

    def wrap(name: str):
        original = default.get(name).function

        def counted(**arguments: object) -> object:
            calls.append(name)
            return original(**arguments)

        return counted

    counted = {"build_cell_neighbors", "cluster_cells", "compute_cell_umap"}
    return ToolRegistry(
        tuple(
            replace(spec, function=wrap(spec.name)) if spec.name in counted else spec
            for spec in (default.get(name) for name in default.names())
        )
    )


def _embedding_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cell_ids = tuple(f"cell-{index}" for index in range(24))
    embedding = np.zeros((24, 512), dtype=np.float32)
    embedding[:12, 0] = np.linspace(0.0, 0.2, 12, dtype=np.float32)
    embedding[12:, 0] = np.linspace(8.0, 8.2, 12, dtype=np.float32)
    embedding[:, 1] = np.linspace(-0.2, 0.2, 24, dtype=np.float32)
    embedding_path = tmp_path / "cells.npy"
    cell_ids_path = tmp_path / "cells.obs_names.txt"
    np.save(embedding_path, embedding, allow_pickle=False)
    cell_ids_path.write_text(
        "".join(f"{cell_id}\n" for cell_id in cell_ids), encoding="utf-8"
    )
    return embedding_path, cell_ids_path


def _tiny_h5ad(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(np.asarray([[1, 0], [0, 1]], dtype=np.float32)),
        obs=pd.DataFrame(index=pd.Index(["cell-1", "cell-2"], dtype="object")),
        var=pd.DataFrame(index=pd.Index(["peak-1", "peak-2"], dtype="object")),
    ).write_h5ad(path)
    return path


def _contains_callable(value: object) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(
            _contains_callable(key) or _contains_callable(nested)
            for key, nested in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_callable(nested) for nested in value)
    return False


def test_real_application_runtime_to_visualized_report_and_terminal_reuse(
    tmp_path: Path,
) -> None:
    embedding_path, cell_ids_path = _embedding_inputs(tmp_path)
    calls: list[str] = []
    planner = _DownstreamPlanner()
    registry = _counting_registry(calls)
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=planner, registry=registry
    )
    request = AgentRequest(
        "milestone7-application-downstream",
        "Build neighbors, perform Leiden clustering and UMAP, and generate a report.",
        {
            "embedding_path": str(embedding_path),
            "cell_ids_path": str(cell_ids_path),
        },
    )

    result = application.run(request)

    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.visualization is not None
    assert result.report is not None
    manifest = json.loads(Path(result.visualization.path).read_text(encoding="utf-8"))
    assert [figure["figure_kind"] for figure in manifest["figures"]] == [
        "umap_leiden"
    ]
    assert calls == ["build_cell_neighbors", "cluster_cells", "compute_cell_umap"]
    assert planner.calls == 1
    run_root = Path(result.workspace_path)
    for reference in (result.evidence, result.visualization, result.report):
        assert reference is not None
        assert Path(reference.path).resolve().is_relative_to(run_root.resolve())
    report_manifest = Path(result.report.path).parent / "report_manifest.json"
    assert verify_analysis_report(
        result.run_result,
        result.evidence.path,
        report_manifest,
        registry=registry,
        visualization=result.visualization.path,
    ).passed

    resumed = application.resume(result.run_id)

    assert resumed.status is ApplicationStatus.SUCCEEDED
    assert planner.calls == 1
    assert calls == ["build_cell_neighbors", "cluster_cells", "compute_cell_umap"]
    assert resumed.report == result.report


def test_missing_visualization_is_rebuilt_but_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    embedding_path, cell_ids_path = _embedding_inputs(tmp_path)
    calls: list[str] = []
    application = ResearchAgentApplication(
        tmp_path / "workspace",
        planner=_DownstreamPlanner(),
        registry=_counting_registry(calls),
    )
    request = AgentRequest(
        "milestone7-visualization-reuse",
        "Build neighbors, cluster, UMAP, and a report.",
        {
            "embedding_path": str(embedding_path),
            "cell_ids_path": str(cell_ids_path),
        },
    )
    first = application.run(request)
    assert first.visualization is not None
    shutil.rmtree(Path(first.visualization.path).parent)

    rebuilt = application.resume(first.run_id)

    assert rebuilt.status is ApplicationStatus.SUCCEEDED
    assert rebuilt.visualization is not None
    manifest = json.loads(Path(rebuilt.visualization.path).read_text(encoding="utf-8"))
    figure = Path(rebuilt.visualization.path).parent / manifest["figures"][0]["relative_path"]
    figure.write_bytes(b"tampered PNG")

    failed = application.resume(first.run_id)

    assert failed.status is ApplicationStatus.FAILED
    assert failed.error is not None
    assert failed.error.code == "APP_VISUALIZATION_FAILED"
    assert calls == ["build_cell_neighbors", "cluster_cells", "compute_cell_umap"]


def test_llm_planner_report_language_stays_outside_tool_plan(tmp_path: Path) -> None:
    source = _tiny_h5ad(tmp_path / "tiny.h5ad")
    model = _FixedPlanningModel()
    application = ResearchAgentApplication(
        tmp_path / "workspace", planner=LLMPlanner(model)
    )
    request = AgentRequest(
        "milestone7-natural-language",
        "Inspect this scATAC-seq dataset and generate a scientific report.",
        {"input_path": str(source)},
    )

    result = application.run(request)

    assert result.status is ApplicationStatus.SUCCEEDED
    assert result.run_result.plan is not None
    assert [step.tool_name for step in result.run_result.plan.steps] == [
        "inspect_scATAC"
    ]
    assert result.report is not None
    assert len(model.calls) == 1
    prompt, schema = model.calls[0]
    assert "output_dir" in prompt
    assert isinstance(schema, Mapping)
    assert schema["properties"]["schema_version"]["enum"] == (2,)
    assert not _contains_callable(schema)
    assert len(application.registry.names()) == 10
