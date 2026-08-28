"""Explicitly opt-in real OpenAI planning smoke test."""

from __future__ import annotations

import importlib.util
import os

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    RunMode,
    RunStatus,
)
from agent.providers import OpenAIPlanningModel


def test_real_openai_planner_plan_only() -> None:
    if os.environ.get("RUN_OPENAI_PLANNER_TEST") != "1":
        pytest.skip("Set RUN_OPENAI_PLANNER_TEST=1 to enable the OpenAI smoke test.")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the OpenAI smoke test.")
    model = os.environ.get("AGENT_OPENAI_MODEL")
    if not model:
        pytest.skip("AGENT_OPENAI_MODEL is required for the OpenAI smoke test.")
    if importlib.util.find_spec("openai") is None:
        pytest.skip("The optional `openai` package is not installed.")

    request = AgentRequest(
        request_id="openai-planner-smoke",
        prompt="Inspect the supplied scATAC-seq dataset.",
        inputs={"input_path": "/tmp/openai-planner-smoke-input.h5ad"},
        mode=RunMode.PLAN_ONLY,
    )
    runtime = AgentRuntime(
        planner=LLMPlanner(OpenAIPlanningModel(model=model, timeout=60.0))
    )

    result = runtime.run(request)

    assert result.status is RunStatus.PLANNED
    assert result.planning_only is True
    assert result.plan is not None
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].tool_name == "inspect_scATAC"
    assert result.steps == ()
    assert result.verification is not None and result.verification.passed
