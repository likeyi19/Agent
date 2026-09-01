from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from agent.orchestration import (
    AgentRuntime,
    FileRunStore,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.report import build_analysis_evidence, verify_analysis_evidence
from agent.schemas import AgentPlan, AgentRequest, PlanStep, RunStatus, StepStatus


def _tiny_sparse_h5ad(path: Path) -> Path:
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 1],
                ],
                dtype=np.float32,
            )
        )
    )
    adata.obs_names = ["cell-1", "cell-2", "cell-3"]
    adata.var_names = ["peak-1", "peak-2", "peak-3", "peak-4"]
    adata.write_h5ad(path)
    return path


def _counting_registry(calls: list[str]) -> ToolRegistry:
    default = build_default_tool_registry()
    original = default.get("inspect_scATAC")

    def counting_inspection(**arguments: object) -> object:
        calls.append(str(arguments["path"]))
        return original.function(**arguments)

    return ToolRegistry(
        tuple(
            replace(spec, function=counting_inspection)
            if spec.name == "inspect_scATAC"
            else spec
            for spec in (default.get(name) for name in default.names())
        )
    )


class _StaticPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self.plan_value


def test_offline_inspection_to_verified_analysis_evidence(tmp_path: Path) -> None:
    source = _tiny_sparse_h5ad(tmp_path / "tiny.h5ad")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    calls: list[str] = []
    registry = _counting_registry(calls)
    runtime = AgentRuntime(registry=registry)
    result = runtime.run(
        AgentRequest(
            "milestone7-offline",
            "Inspect this scATAC dataset.",
            {"input_path": str(source)},
        )
    )
    assert result.status is RunStatus.SUCCEEDED
    assert len(calls) == 1

    first = build_analysis_evidence(
        result, tmp_path / "evidence-first", registry=registry
    )
    second = build_analysis_evidence(
        result, tmp_path / "evidence-second", registry=registry
    )
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert verify_analysis_evidence(result, first, registry=registry).passed
    assert verify_analysis_evidence(result, second["evidence_path"], registry=registry).passed
    assert len(calls) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256


def test_evidence_after_nonterminal_and_terminal_resume_does_not_rerun_successes(
    tmp_path: Path,
) -> None:
    source = _tiny_sparse_h5ad(tmp_path / "durable.h5ad")
    request = AgentRequest(
        "milestone7-durable",
        "Inspect this dataset twice for a durability exercise.",
        {"input_path": str(source)},
    )
    plan = AgentPlan(
        "milestone7-durable-plan",
        request.request_id,
        "static-test-planner",
        (
            PlanStep("inspect-first", "inspect_scATAC", {"path": str(source)}),
            PlanStep("inspect-second", "inspect_scATAC", {"path": str(source)}),
        ),
    )
    planner = _StaticPlanner(plan)
    calls: list[str] = []
    registry = _counting_registry(calls)
    store = FileRunStore(tmp_path / "run-store")
    original_update = store.update

    def interrupt_after_first_success(state, *, expected_revision: int):
        persisted = original_update(state, expected_revision=expected_revision)
        statuses = tuple(step.status for step in persisted.steps)
        if statuses == (StepStatus.SUCCEEDED, StepStatus.PENDING):
            raise KeyboardInterrupt("simulated process interruption")
        return persisted

    store.update = interrupt_after_first_success  # type: ignore[method-assign]
    runtime = AgentRuntime(planner=planner, registry=registry, run_store=store)
    try:
        runtime.run(request)
    except KeyboardInterrupt:
        pass
    else:  # pragma: no cover - test invariant
        raise AssertionError("The durability test did not interrupt after step one.")
    store.update = original_update  # type: ignore[method-assign]
    assert len(calls) == 1
    assert planner.calls == 1

    resumed = AgentRuntime(registry=registry, run_store=store).resume(
        f"{request.request_id}:run"
    )
    assert resumed.status is RunStatus.SUCCEEDED
    assert len(calls) == 2
    assert planner.calls == 1
    state_before = store.load(resumed.run_id).to_dict()

    first = build_analysis_evidence(
        resumed, tmp_path / "resumed-evidence", registry=registry
    )
    assert verify_analysis_evidence(resumed, first, registry=registry).passed
    assert len(calls) == 2
    assert store.load(resumed.run_id).to_dict() == state_before

    terminal = AgentRuntime(registry=registry, run_store=store).resume(resumed.run_id)
    assert terminal.status is RunStatus.SUCCEEDED
    assert len(calls) == 2
    second = build_analysis_evidence(
        terminal, tmp_path / "terminal-evidence", registry=registry
    )
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert verify_analysis_evidence(terminal, second, registry=registry).passed
    assert len(calls) == 2
    assert store.load(resumed.run_id).to_dict() == state_before
