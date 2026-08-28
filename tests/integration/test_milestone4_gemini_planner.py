"""Explicitly opt-in real Gemini planning smoke test."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest,
    AgentRuntime,
    LLMPlanner,
    RunMode,
    RunStatus,
    ToolRegistry,
    build_default_tool_registry,
)
from agent.providers import GeminiPlanningModel


def _gemini_sdk_available() -> bool:
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ModuleNotFoundError:
        return False


def test_real_gemini_planner_plan_only() -> None:
    if os.environ.get("RUN_GEMINI_PLANNER_TEST") != "1":
        pytest.skip("Set RUN_GEMINI_PLANNER_TEST=1 to enable the Gemini smoke test.")
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is required for the Gemini smoke test.")
    model = os.environ.get("AGENT_GEMINI_MODEL")
    if not model:
        pytest.skip("AGENT_GEMINI_MODEL is required for the Gemini smoke test.")
    if not _gemini_sdk_available():
        pytest.skip("The optional `google-genai` package is not installed.")

    default = build_default_tool_registry()
    inspect_call = Mock(side_effect=AssertionError("inspect must not execute"))
    embed_call = Mock(side_effect=AssertionError("embed must not execute"))
    guarded_registry = ToolRegistry(
        (
            replace(default.get("inspect_scATAC"), function=inspect_call),
            replace(default.get("epizoo_embed_cells"), function=embed_call),
        )
    )
    request = AgentRequest(
        request_id="gemini-planner-smoke",
        prompt="Inspect the supplied scATAC-seq dataset.",
        inputs={"input_path": "/tmp/gemini-planner-smoke-input.h5ad"},
        mode=RunMode.PLAN_ONLY,
    )
    runtime = AgentRuntime(
        planner=LLMPlanner(GeminiPlanningModel(model=model, timeout=60.0)),
        registry=guarded_registry,
    )

    result = runtime.run(request)

    assert result.status is RunStatus.PLANNED
    assert result.planning_only is True
    assert result.plan is not None
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].tool_name == "inspect_scATAC"
    assert result.steps == ()
    assert result.verification is not None and result.verification.passed
    inspect_call.assert_not_called()
    embed_call.assert_not_called()
