"""Offline Milestone 4.1 acceptance through the existing Agent runtime."""

from __future__ import annotations

from dataclasses import replace
import json
from unittest.mock import Mock

import anndata as ad
import numpy as np
import scipy.sparse as sp

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    RunMode,
    RunStatus,
    StepStatus,
    ToolRegistry,
    build_default_tool_registry,
)


class FakePlanningModel:
    model_id = "offline-integration-v1"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = json.dumps(response)
        self.calls = 0

    def complete(self, *, prompt: str, response_schema) -> str:
        self.calls += 1
        json.loads(prompt)
        json.dumps(response_schema)
        return self.response


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


def _inspect_step() -> dict[str, object]:
    return {
        "step_id": "inspect",
        "tool_name": "inspect_scATAC",
        "arguments": {
            "path": {
                "binding_type": "input",
                "input_name": "input_path",
            }
        },
        "depends_on": [],
        "description": "Inspect the dataset safely.",
    }


def _embed_step() -> dict[str, object]:
    return {
        "step_id": "embed",
        "tool_name": "epizoo_embed_cells",
        "arguments": {
            "input_path": {
                "binding_type": "ref",
                "ref_step_id": "inspect",
                "ref_output_key": "input_path",
            },
            "output_dir": {
                "binding_type": "input",
                "input_name": "output_dir",
            },
            "species": {
                "binding_type": "input",
                "input_name": "species",
            },
            "checkpoint_path": None,
            "device": None,
            "overwrite": None,
        },
        "depends_on": ["inspect"],
        "description": "Compute EpiZoo embeddings.",
    }


def _response(*steps: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 3,
        "status": "plan",
        "steps": list(steps),
        "reason": None,
    }


def test_llm_planner_executes_real_inspection_through_agent_runtime(tmp_path) -> None:
    input_path, matrix = _write_sparse_scatac(tmp_path)
    model = FakePlanningModel(_response(_inspect_step()))
    request = AgentRequest(
        "milestone4-execute",
        "Please inspect this sparse scATAC-seq dataset.",
        {"input_path": str(input_path)},
    )

    result = AgentRuntime(planner=LLMPlanner(model)).run(request)

    assert model.calls == 1
    assert result.status is RunStatus.SUCCEEDED
    assert result.plan is not None
    assert result.plan.planner_name == "llm:offline-integration-v1"
    assert result.steps[0].status is StepStatus.SUCCEEDED
    assert result.steps[0].verification is not None
    assert result.steps[0].verification.passed
    inspection = result.steps[0].result
    assert inspection is not None
    assert inspection["input_path"] == str(input_path.resolve())
    assert inspection["n_cells"] == 3
    assert inspection["n_features"] == 5
    assert inspection["x_is_sparse"] is True
    assert inspection["nnz"] == matrix.nnz
    assert result.verification is not None and result.verification.passed
    json.dumps(result.to_dict())


def test_llm_embedding_plan_only_runs_full_preflight_with_zero_tool_calls(
    tmp_path,
) -> None:
    input_path, _ = _write_sparse_scatac(tmp_path)
    default = build_default_tool_registry()
    inspect_call = Mock(side_effect=AssertionError("inspect must not execute"))
    embed_call = Mock(side_effect=AssertionError("embed must not execute"))
    guarded_registry = ToolRegistry(
        (
            replace(default.get("inspect_scATAC"), function=inspect_call),
            replace(default.get("epizoo_embed_cells"), function=embed_call),
        )
    )
    model = FakePlanningModel(_response(_inspect_step(), _embed_step()))
    request = AgentRequest(
        "milestone4-plan-only",
        "Inspect the dataset and plan EpiZoo cell embeddings.",
        {
            "input_path": str(input_path),
            "output_dir": str(tmp_path / "embeddings"),
            "species": "mouse",
        },
        RunMode.PLAN_ONLY,
    )

    result = AgentRuntime(
        planner=LLMPlanner(model), registry=guarded_registry
    ).run(request)

    assert model.calls == 1
    assert result.status is RunStatus.PLANNED
    assert result.planning_only is True
    assert result.plan is not None and len(result.plan.steps) == 2
    assert result.steps == ()
    assert result.verification is not None and result.verification.passed
    inspect_call.assert_not_called()
    embed_call.assert_not_called()
    assert not (tmp_path / "embeddings").exists()


def test_invalid_later_llm_step_preflight_prevents_earlier_real_tool(
    tmp_path,
) -> None:
    input_path, _ = _write_sparse_scatac(tmp_path)
    before_stat = input_path.stat()
    default = build_default_tool_registry()
    inspect_call = Mock(wraps=default.get("inspect_scATAC").function)
    guarded_registry = ToolRegistry(
        (replace(default.get("inspect_scATAC"), function=inspect_call),)
    )
    invalid_step = {
        "step_id": "unsafe",
        "tool_name": "arbitrary_python",
        "arguments": {},
        "depends_on": ["inspect"],
        "description": None,
    }
    model = FakePlanningModel(_response(_inspect_step(), invalid_step))
    request = AgentRequest(
        "milestone4-preflight-failure",
        "Inspect the dataset, then do an unsupported operation.",
        {"input_path": str(input_path)},
    )

    result = AgentRuntime(
        planner=LLMPlanner(model), registry=guarded_registry
    ).run(request)

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "UNKNOWN_TOOL"
    inspect_call.assert_not_called()
    after_stat = input_path.stat()
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
