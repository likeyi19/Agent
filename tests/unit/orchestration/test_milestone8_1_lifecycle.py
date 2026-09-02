"""Durability, recovery, cancellation, and PLAN_ONLY checks for Milestone 8.1."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import anndata as ad
import pandas as pd
import pytest
from scipy import sparse

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    FileRunStore,
    PlanStep,
    RecoveryPolicyIncompatibleError,
    RunLifecycleStatus,
    RunMode,
    RunStatus,
    StepOutputRef,
    StepStatus,
    ToolRegistry,
    build_default_tool_registry,
)


class SimulatedProcessExit(BaseException):
    pass


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.plan_value


class InterruptAfterFeatureStore:
    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.triggered = False

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (
            not self.triggered
            and saved.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(saved.steps) == 2
            and saved.steps[0].status is StepStatus.SUCCEEDED
            and saved.steps[1].status is StepStatus.PENDING
        ):
            self.triggered = True
            raise SimulatedProcessExit()
        return saved


def _source(path: Path) -> Path:
    ad.AnnData(
        X=sparse.csr_matrix(
            [[1, 0, 2], [0, 3, 0], [4, 0, 5], [0, 6, 0]], dtype="int32"
        ),
        obs=pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "replicate": ["r1", "r1", "r2", "r2"],
                "condition": ["control", "control", "treated", "treated"],
            },
            index=["c1", "c2", "c3", "c4"],
        ),
        var=pd.DataFrame(index=["p1", "p2", "p3"]),
    ).write_h5ad(path)
    return path


def _plan(tmp_path: Path, source: Path) -> AgentPlan:
    output = tmp_path / "outputs"
    return AgentPlan(
        "m81-plan",
        "request-1",
        "fixed",
        (
            PlanStep(
                "feature",
                "validate_scATAC_feature_space",
                {
                    "input_path": str(source),
                    "output_dir": str(output),
                    "matrix_source": "X",
                    "matrix_semantics": "fragment_counts",
                    "species": "human",
                    "genome_assembly": "hg38",
                    "coordinate_source": "none",
                },
            ),
            PlanStep(
                "pseudobulk",
                "build_replicate_pseudobulk",
                {
                    "feature_space_path": StepOutputRef(
                        "feature", "feature_space_path"
                    ),
                    "replicate_key": "replicate",
                    "group_key": "group",
                    "condition_key": "condition",
                    "output_dir": str(output),
                    "group_source": "raw_obs",
                },
                ("feature",),
            ),
        ),
    )


def _interrupted_after_feature(tmp_path: Path):
    source = _source(tmp_path / "raw.h5ad")
    plan = _plan(tmp_path, source)
    default = build_default_tool_registry()
    feature_call = Mock(wraps=default.get("validate_scATAC_feature_space").function)
    pseudobulk_call = Mock(wraps=default.get("build_replicate_pseudobulk").function)
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=feature_call)
            if name == "validate_scATAC_feature_space"
            else replace(default.get(name), function=pseudobulk_call)
            if name == "build_replicate_pseudobulk"
            else default.get(name)
            for name in default.names()
        )
    )
    store = FileRunStore(tmp_path / "store")
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan),
            registry=registry,
            run_store=InterruptAfterFeatureStore(store),
        ).run(AgentRequest("request-1", "fixed", {}))
    return source, plan, registry, store, feature_call, pseudobulk_call


def test_feature_checkpoint_is_revalidated_and_reference_restored_on_resume(
    tmp_path: Path,
) -> None:
    _, plan, registry, store, feature_call, pseudobulk_call = (
        _interrupted_after_feature(tmp_path)
    )
    planner = FixedPlanner(plan)
    result = AgentRuntime(
        planner=planner, registry=registry, run_store=store
    ).resume("request-1:run")
    assert result.status is RunStatus.SUCCEEDED
    assert planner.calls == 0
    feature_call.assert_called_once()
    pseudobulk_call.assert_called_once()
    assert result.steps[1].resolved_arguments["feature_space_path"] == (
        result.steps[0].result["feature_space_path"]  # type: ignore[index]
    )


@pytest.mark.parametrize("mutation", ["source", "manifest"])
def test_changed_feature_evidence_blocks_resume_before_pseudobulk(
    tmp_path: Path, mutation: str
) -> None:
    source, plan, registry, store, feature_call, pseudobulk_call = (
        _interrupted_after_feature(tmp_path)
    )
    if mutation == "source":
        raw = ad.read_h5ad(source)
        raw.uns["changed"] = True
        raw.write_h5ad(source)
    else:
        state = store.load("request-1:run")
        Path(state.steps[0].result["feature_space_path"]).unlink()  # type: ignore[index]

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).resume("request-1:run")
    assert result.status is RunStatus.FAILED
    assert any(
        error.code == "PERSISTED_STEP_REVALIDATION_FAILED" for error in result.errors
    )
    feature_call.assert_called_once()
    pseudobulk_call.assert_not_called()


def test_recovery_identities_are_authoritative_and_have_no_retry_codes(
    tmp_path: Path,
) -> None:
    _, plan, registry, store, feature_call, pseudobulk_call = (
        _interrupted_after_feature(tmp_path)
    )
    snapshot = store.load("request-1:run").recovery_policy_snapshot
    assert snapshot is not None
    assert tuple(
        (tool.tool_name, tool.classifier_version, tool.retryable_error_codes)
        for tool in snapshot.tools
    ) == (
        (
            "build_replicate_pseudobulk",
            "build-replicate-pseudobulk-v1",
            (),
        ),
        (
            "validate_scATAC_feature_space",
            "validate-scatac-feature-space-v1",
            (),
        ),
    )
    guard = Mock(side_effect=AssertionError("policy drift invoked a tool"))
    incompatible = ToolRegistry(
        tuple(
            replace(
                registry.get(name),
                function=guard,
                recovery_policy_version="validate-scatac-feature-space-v2",
            )
            if name == "validate_scATAC_feature_space"
            else replace(registry.get(name), function=guard)
            for name in registry.names()
        )
    )
    with pytest.raises(RecoveryPolicyIncompatibleError):
        AgentRuntime(
            planner=FixedPlanner(plan), registry=incompatible, run_store=store
        ).resume("request-1:run")
    feature_call.assert_called_once()
    pseudobulk_call.assert_not_called()
    guard.assert_not_called()


def test_cancellation_after_feature_prevents_pseudobulk(tmp_path: Path) -> None:
    source = _source(tmp_path / "raw.h5ad")
    plan = _plan(tmp_path, source)
    store = FileRunStore(tmp_path / "store")
    default = build_default_tool_registry()

    def validate_then_cancel(**kwargs):
        result = default.get("validate_scATAC_feature_space").function(**kwargs)
        store.request_cancellation("request-1:run")
        return result

    pseudobulk_call = Mock(
        side_effect=AssertionError("cancelled run invoked pseudobulk")
    )
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=validate_then_cancel)
            if name == "validate_scATAC_feature_space"
            else replace(default.get(name), function=pseudobulk_call)
            if name == "build_replicate_pseudobulk"
            else default.get(name)
            for name in default.names()
        )
    )
    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).run(AgentRequest("request-1", "fixed", {}))
    assert result.status is RunStatus.CANCELLED
    assert tuple(step.status for step in result.steps) == (
        StepStatus.SUCCEEDED,
        StepStatus.SKIPPED,
    )
    pseudobulk_call.assert_not_called()


def test_plan_only_preflights_without_scientific_execution(tmp_path: Path) -> None:
    source = _source(tmp_path / "raw.h5ad")
    plan = _plan(tmp_path, source)
    guard = Mock(side_effect=AssertionError("PLAN_ONLY invoked a scientific tool"))
    default = build_default_tool_registry()
    registry = ToolRegistry(
        tuple(
            replace(default.get(name), function=guard)
            if name in {
                "validate_scATAC_feature_space",
                "build_replicate_pseudobulk",
            }
            else default.get(name)
            for name in default.names()
        )
    )
    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry
    ).run(AgentRequest("request-1", "fixed", {}, RunMode.PLAN_ONLY))
    assert result.status is RunStatus.PLANNED
    guard.assert_not_called()
