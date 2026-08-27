"""Portable real-tool acceptance tests for the Milestone 3 Agent stack."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import scipy.sparse as sp

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    RunMode,
    RunStatus,
    StepStatus,
    TraceEventType,
)


def _write_sparse_scatac(tmp_path):
    input_path = tmp_path / "tiny_scATAC.h5ad"
    matrix = sp.csr_matrix(
        np.array(
            [
                [1, 0, 0, 2, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 5],
            ],
            dtype=np.float32,
        )
    )
    adata = ad.AnnData(matrix)
    adata.obs_names = ["cell-1", "cell-2", "cell-3"]
    adata.var_names = [f"feature-{index}" for index in range(1, 6)]
    adata.obs["batch"] = ["a", "a", "b"]
    adata.var["reference"] = [True, True, False, False, True]
    adata.write_h5ad(input_path)
    return input_path, matrix


def test_real_inspect_scatac_executes_through_agent_runtime(tmp_path) -> None:
    input_path, matrix = _write_sparse_scatac(tmp_path)
    request = AgentRequest(
        request_id="integration-execute",
        prompt="Inspect this scATAC-seq dataset",
        inputs={"input_path": str(input_path)},
        mode=RunMode.EXECUTE,
    )

    result = AgentRuntime().run(request)

    assert result.status is RunStatus.SUCCEEDED
    assert result.planning_only is False
    assert result.plan is not None
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].tool_name == "inspect_scATAC"
    assert len(result.steps) == 1
    step_result = result.steps[0]
    assert step_result.status is StepStatus.SUCCEEDED
    assert step_result.verification is not None
    assert step_result.verification.passed
    assert result.verification is not None
    assert result.verification.passed
    assert result.errors == ()

    inspection = step_result.result
    assert inspection is not None
    assert inspection["input_path"] == str(input_path.resolve())
    assert inspection["n_cells"] == 3
    assert inspection["n_features"] == 5
    assert inspection["x_is_sparse"] is True
    assert inspection["nnz"] == matrix.nnz == 5
    assert inspection["density"] == matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    assert inspection["obs_columns"] == ("batch",)
    assert inspection["var_columns"] == ("reference",)

    event_types = {event.event_type for event in result.trace}
    assert TraceEventType.PLANNING in event_types
    assert TraceEventType.PLAN_VALIDATION in event_types
    assert TraceEventType.STEP_EXECUTION in event_types
    assert TraceEventType.VERIFICATION in event_types
    assert TraceEventType.RUN_COMPLETION in event_types
    json.dumps(result.to_dict())


def test_real_inspect_scatac_plan_only_has_no_execution_or_mutation(tmp_path) -> None:
    input_path, _ = _write_sparse_scatac(tmp_path)
    before_files = {path.name for path in tmp_path.iterdir()}
    before_stat = input_path.stat()
    request = AgentRequest(
        request_id="integration-plan-only",
        prompt="Inspect this scATAC-seq dataset",
        inputs={"input_path": str(input_path)},
        mode=RunMode.PLAN_ONLY,
    )

    result = AgentRuntime().run(request)

    assert result.status is RunStatus.PLANNED
    assert result.planning_only is True
    assert result.plan is not None and len(result.plan.steps) == 1
    assert result.steps == ()
    assert result.errors == ()
    assert result.verification is not None and result.verification.passed
    assert all(
        event.event_type is not TraceEventType.STEP_EXECUTION
        for event in result.trace
    )
    assert {path.name for path in tmp_path.iterdir()} == before_files
    after_stat = input_path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    json.dumps(result.to_dict())
