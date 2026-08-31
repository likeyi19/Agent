"""Offline integration tests for durable AgentRuntime resume semantics."""

from __future__ import annotations

from pathlib import Path
import threading
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentPlan,
    AgentRequest,
    AgentRuntime,
    ArgumentSpec,
    ErrorCategory,
    ErrorClassification,
    FileRunStore,
    PlanStep,
    PlanExecutor,
    PlannerError,
    ResultContract,
    RunAlreadyActiveError,
    RunLifecycleStatus,
    RunMode,
    RunStatus,
    StepOutputRef,
    StepStatus,
    ToolRegistry,
    ToolSpec,
)


class SimulatedProcessExit(BaseException):
    pass


def _classify(exception: Exception) -> ErrorClassification:
    return ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, "FAKE_FAILURE")


def _spec(
    name: str,
    function,
    *,
    arguments: dict[str, ArgumentSpec] | None = None,
    result_fields: dict[str, tuple[type, ...]] | None = None,
) -> ToolSpec:
    return ToolSpec(
        name,
        function,
        arguments or {},
        {},
        ResultContract(f"{name}Result", result_fields or {"value": (str,)}),
        _classify,
    )


def _request(*, mode: RunMode = RunMode.EXECUTE) -> AgentRequest:
    return AgentRequest("request-1", "fixed workflow", {}, mode)


def _two_step_plan() -> AgentPlan:
    return AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
        (
            PlanStep("producer", "producer", {}),
            PlanStep(
                "consumer",
                "consumer",
                {"source": StepOutputRef("producer", "value")},
                ("producer",),
            ),
        ),
    )


class FixedPlanner:
    def __init__(self, plan: AgentPlan) -> None:
        self.plan_value = plan
        self.calls = 0

    def plan(self, request: AgentRequest, registry: ToolRegistry) -> AgentPlan:
        self.calls += 1
        return self.plan_value


class InterruptingStore:
    """Delegate store that exits only after the selected state was saved."""

    def __init__(self, delegate: FileRunStore, predicate) -> None:
        self.delegate = delegate
        self.predicate = predicate
        self.triggered = False

    def create(self, state):
        return self.delegate.create(state)

    def execution_lease(self, run_id):
        return self.delegate.execution_lease(run_id)

    def load(self, run_id):
        return self.delegate.load(run_id)

    def request_cancellation(self, run_id):
        return self.delegate.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.delegate.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if not self.triggered and self.predicate(saved):
            self.triggered = True
            raise SimulatedProcessExit()
        return saved


class FailingSuccessCheckpointStore:
    """Fail before saving the first verified-success transition."""

    def __init__(self, delegate: FileRunStore) -> None:
        self.delegate = delegate
        self.failed = False

    def create(self, state):
        return self.delegate.create(state)

    def execution_lease(self, run_id):
        return self.delegate.execution_lease(run_id)

    def load(self, run_id):
        return self.delegate.load(run_id)

    def request_cancellation(self, run_id):
        return self.delegate.request_cancellation(run_id)

    def load_cancellation(self, run_id):
        return self.delegate.load_cancellation(run_id)

    def update(self, state, *, expected_revision):
        if (
            not self.failed
            and state.lifecycle_status is RunLifecycleStatus.RUNNING
            and state.steps
            and state.steps[0].status is StepStatus.SUCCEEDED
        ):
            self.failed = True
            raise RuntimeError("simulated checkpoint failure")
        return self.delegate.update(state, expected_revision=expected_revision)


def _registry(producer, consumer) -> ToolRegistry:
    return ToolRegistry(
        (
            _spec("producer", producer),
            _spec(
                "consumer",
                consumer,
                arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )


def _interrupt_after_first_success(
    tmp_path: Path,
    producer: Mock,
    consumer: Mock,
) -> tuple[FileRunStore, ToolRegistry]:
    base_store = FileRunStore(tmp_path)
    interrupting = InterruptingStore(
        base_store,
        lambda state: (
            state.lifecycle_status is RunLifecycleStatus.RUNNING
            and len(state.steps) == 2
            and state.steps[0].status is StepStatus.SUCCEEDED
            and state.steps[1].status is StepStatus.PENDING
        ),
    )
    registry = _registry(producer, consumer)
    runtime = AgentRuntime(
        planner=FixedPlanner(_two_step_plan()),
        registry=registry,
        run_store=interrupting,
    )
    with pytest.raises(SimulatedProcessExit):
        runtime.run(_request())
    assert interrupting.triggered
    return base_store, registry


def test_fresh_run_persists_request_plan_and_successful_step(
    tmp_path: Path,
) -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).run(_request())
    state = store.load(result.run_id)

    assert result.status is RunStatus.SUCCEEDED
    assert state.request == _request()
    assert state.plan == plan
    assert state.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert state.steps[0].status is StepStatus.SUCCEEDED
    assert state.steps[0].result == {"value": "done"}
    assert state.steps[0].verification is not None
    assert state.steps[0].verification.passed


def test_request_is_persisted_before_planner_execution(tmp_path: Path) -> None:
    store = FileRunStore(tmp_path)
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )

    class ObservingPlanner:
        def plan(self, request, registry):
            observed = store.load("request-1:run")
            assert observed.request == request
            assert observed.plan is None
            assert observed.lifecycle_status is RunLifecycleStatus.PLANNING
            return plan

    registry = ToolRegistry((_spec("tool", Mock(return_value={"value": "ok"})),))
    AgentRuntime(
        planner=ObservingPlanner(), registry=registry, run_store=store
    ).run(_request())


def test_planner_failure_leaves_valid_terminal_record(tmp_path: Path) -> None:
    class FailingPlanner:
        def plan(self, request, registry):
            raise PlannerError("UNSUPPORTED_REQUEST", "Not supported.")

    store = FileRunStore(tmp_path)
    result = AgentRuntime(
        planner=FailingPlanner(), registry=ToolRegistry(()), run_store=store
    ).run(_request())
    state = store.load(result.run_id)

    assert result.status is RunStatus.FAILED
    assert state.lifecycle_status is RunLifecycleStatus.FAILED
    assert state.plan is None
    assert state.errors[0].code == "UNSUPPORTED_REQUEST"


def test_resume_skips_checkpointed_step_and_restores_output_reference(
    tmp_path: Path,
) -> None:
    producer = Mock(return_value={"value": "persisted-upstream"})
    consumer = Mock(return_value={"value": "complete"})
    store, registry = _interrupt_after_first_success(
        tmp_path, producer, consumer
    )
    resume_planner = FixedPlanner(_two_step_plan())

    result = AgentRuntime(
        planner=resume_planner,
        registry=registry,
        run_store=store,
    ).resume("request-1:run")

    assert result.status is RunStatus.SUCCEEDED
    assert producer.call_count == 1
    consumer.assert_called_once_with(source="persisted-upstream")
    assert resume_planner.calls == 0
    assert [step.status for step in result.steps] == [
        StepStatus.SUCCEEDED,
        StepStatus.SUCCEEDED,
    ]


def test_success_checkpoint_failure_prevents_downstream_execution(
    tmp_path: Path,
) -> None:
    producer = Mock(return_value={"value": "not-durable"})
    consumer = Mock(return_value={"value": "must-not-run"})
    base_store = FileRunStore(tmp_path)
    failing_store = FailingSuccessCheckpointStore(base_store)

    result = AgentRuntime(
        planner=FixedPlanner(_two_step_plan()),
        registry=_registry(producer, consumer),
        run_store=failing_store,
    ).run(_request())

    assert result.status is RunStatus.FAILED
    assert failing_store.failed
    assert producer.call_count == 1
    consumer.assert_not_called()


def test_successful_terminal_resume_executes_zero_tools(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )
    first = runtime.run(_request())
    resumed = runtime.resume(first.run_id)

    assert resumed.to_dict() == first.to_dict()
    assert tool.call_count == 1


def test_failed_terminal_resume_executes_zero_tools(tmp_path: Path) -> None:
    tool = Mock(side_effect=RuntimeError("failed"))
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )
    first = runtime.run(_request())
    resumed = runtime.resume(first.run_id)

    assert first.status is RunStatus.FAILED
    assert resumed.to_dict() == first.to_dict()
    assert tool.call_count == 1


def test_plan_only_persists_planned_and_never_executes(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    runtime = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )

    first = runtime.run(_request(mode=RunMode.PLAN_ONLY))
    resumed = runtime.resume(first.run_id)

    assert first.status is RunStatus.PLANNED
    assert resumed.to_dict() == first.to_dict()
    assert store.load(first.run_id).lifecycle_status is RunLifecycleStatus.PLANNED
    tool.assert_not_called()


def test_nonterminal_plan_only_resume_reuses_plan_and_only_runs_preflight(
    tmp_path: Path,
) -> None:
    guarded_tool = Mock(side_effect=AssertionError("PLAN_ONLY invoked a tool"))
    registry = ToolRegistry((_spec("tool", guarded_tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    interrupting = InterruptingStore(
        store,
        lambda state: (
            state.lifecycle_status is RunLifecycleStatus.PLANNING
            and state.plan is not None
        ),
    )
    initial_planner = FixedPlanner(plan)

    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=initial_planner,
            registry=registry,
            run_store=interrupting,
        ).run(_request(mode=RunMode.PLAN_ONLY))

    persisted = store.load("request-1:run")
    assert persisted.plan == plan
    assert persisted.lifecycle_status is RunLifecycleStatus.PLANNING
    resume_planner = FixedPlanner(plan)
    executor = PlanExecutor(registry)
    preflight = Mock(wraps=executor.preflight)
    executor.preflight = preflight

    result = AgentRuntime(
        planner=resume_planner,
        registry=registry,
        executor=executor,
        run_store=store,
    ).resume("request-1:run")

    assert result.status is RunStatus.PLANNED
    assert result.planning_only
    assert result.plan == plan
    assert store.load(result.run_id).lifecycle_status is RunLifecycleStatus.PLANNED
    assert initial_planner.calls == 1
    assert resume_planner.calls == 0
    preflight.assert_called_once_with(plan)
    guarded_tool.assert_not_called()


def test_overlapping_resume_cannot_interrupt_active_durable_run(
    tmp_path: Path,
) -> None:
    tool_entered = threading.Event()
    release_tool = threading.Event()

    def blocking_tool() -> dict[str, str]:
        tool_entered.set()
        if not release_tool.wait(timeout=10):
            raise TimeoutError("test did not release blocking tool")
        return {"value": "done"}

    tool = Mock(side_effect=blocking_tool)
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    runtime_a = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    )
    outcomes = []
    failures: list[BaseException] = []

    def run_a() -> None:
        try:
            outcomes.append(runtime_a.run(_request()))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=run_a, daemon=True)
    worker.start()
    try:
        assert tool_entered.wait(timeout=10)
        active = store.load("request-1:run")
        assert active.lifecycle_status is RunLifecycleStatus.RUNNING
        assert active.steps[0].status is StepStatus.RUNNING

        resume_planner = FixedPlanner(plan)
        runtime_b = AgentRuntime(
            planner=resume_planner,
            registry=registry,
            run_store=FileRunStore(tmp_path),
        )
        with pytest.raises(RunAlreadyActiveError, match="already active"):
            runtime_b.resume("request-1:run")

        unchanged = store.load("request-1:run")
        assert unchanged == active
        assert resume_planner.calls == 0
        assert tool.call_count == 1
    finally:
        release_tool.set()
        worker.join(timeout=10)

    assert not worker.is_alive()
    assert not failures
    assert len(outcomes) == 1
    assert outcomes[0].status is RunStatus.SUCCEEDED
    final_state = store.load("request-1:run")
    assert final_state.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    terminal_resume = runtime_a.resume("request-1:run")
    assert terminal_resume.to_dict() == outcomes[0].to_dict()
    assert tool.call_count == 1


def test_resume_registry_preflight_failure_prevents_all_side_effects(
    tmp_path: Path,
) -> None:
    producer = Mock(return_value={"value": "persisted"})
    consumer = Mock(return_value={"value": "unused"})
    store, _ = _interrupt_after_first_success(tmp_path, producer, consumer)
    resume_producer = Mock(return_value={"value": "must-not-run"})
    incompatible_registry = ToolRegistry((_spec("producer", resume_producer),))

    result = AgentRuntime(
        planner=FixedPlanner(_two_step_plan()),
        registry=incompatible_registry,
        run_store=store,
    ).resume("request-1:run")

    assert result.status is RunStatus.FAILED
    assert any(error.code == "UNKNOWN_TOOL" for error in result.errors)
    resume_producer.assert_not_called()
    assert consumer.call_count == 0


def test_prior_result_contract_change_fails_before_pending_tool(
    tmp_path: Path,
) -> None:
    producer = Mock(return_value={"value": "persisted"})
    consumer = Mock(return_value={"value": "unused"})
    store, _ = _interrupt_after_first_success(tmp_path, producer, consumer)
    resume_producer = Mock(return_value={"value": 7})
    resume_consumer = Mock(return_value={"value": "must-not-run"})
    registry = ToolRegistry(
        (
            _spec(
                "producer",
                resume_producer,
                result_fields={"value": (int,)},
            ),
            _spec(
                "consumer",
                resume_consumer,
                arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )

    result = AgentRuntime(
        planner=FixedPlanner(_two_step_plan()),
        registry=registry,
        run_store=store,
    ).resume("request-1:run")

    assert result.status is RunStatus.FAILED
    assert any(
        error.code == "PERSISTED_STEP_REVALIDATION_FAILED"
        for error in result.errors
    )
    resume_producer.assert_not_called()
    resume_consumer.assert_not_called()


def _embedding_result(
    input_path: Path, embedding_path: Path, cell_ids_path: Path
) -> dict[str, object]:
    return {
        "status": "success",
        "input_path": str(input_path.resolve()),
        "embedding_path": str(embedding_path),
        "cell_ids_path": str(cell_ids_path),
        "n_cells": 2,
        "embedding_dim": 512,
        "embedding_dtype": "float32",
        "finite": True,
        "cell_order_preserved": True,
        "backend": "EpiZoo",
        "species": "mouse",
        "checkpoint_path": "/models/epizoo.pth",
        "device": "cuda:0",
    }


def test_missing_prior_artifact_fails_resume_revalidation(tmp_path: Path) -> None:
    input_path = tmp_path / "input.h5ad"
    input_path.write_bytes(b"input")
    embedding_path = tmp_path / "embedding.npy"
    cell_ids_path = tmp_path / "ids.txt"
    embedding_path.write_bytes(b"embedding")
    cell_ids_path.write_text("cell-1\ncell-2\n", encoding="utf-8")
    embed = Mock(
        return_value=_embedding_result(input_path, embedding_path, cell_ids_path)
    )
    consumer = Mock(return_value={"value": "unused"})
    result_fields = {
        key: (type(value),)
        for key, value in _embedding_result(
            input_path, embedding_path, cell_ids_path
        ).items()
    }
    registry = ToolRegistry(
        (
            _spec(
                "epizoo_embed_cells",
                embed,
                arguments={
                    "input_path": ArgumentSpec((str, Path)),
                    "output_dir": ArgumentSpec((str, Path)),
                    "species": ArgumentSpec((str,), choices=("mouse", "human")),
                },
                result_fields=result_fields,
            ),
            _spec(
                "consumer",
                consumer,
                arguments={"source": ArgumentSpec((str,))},
            ),
        )
    )
    plan = AgentPlan(
        "plan-1",
        "request-1",
        "fixed",
        (
            PlanStep(
                "embed",
                "epizoo_embed_cells",
                {
                    "input_path": str(input_path),
                    "output_dir": str(tmp_path),
                    "species": "mouse",
                },
            ),
            PlanStep(
                "consumer",
                "consumer",
                {"source": StepOutputRef("embed", "embedding_path")},
                ("embed",),
            ),
        ),
    )
    store = FileRunStore(tmp_path / "store")
    interrupting = InterruptingStore(
        store,
        lambda state: (
            len(state.steps) == 2
            and state.steps[0].status is StepStatus.SUCCEEDED
            and state.steps[1].status is StepStatus.PENDING
        ),
    )
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan),
            registry=registry,
            run_store=interrupting,
        ).run(_request())
    embedding_path.unlink()

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).resume("request-1:run")

    assert result.status is RunStatus.FAILED
    assert any(
        error.code == "PERSISTED_STEP_REVALIDATION_FAILED"
        for error in result.errors
    )
    assert embed.call_count == 1
    consumer.assert_not_called()


def test_stale_running_step_becomes_interrupted_without_tool_rerun(
    tmp_path: Path,
) -> None:
    tool = Mock(return_value={"value": "must-not-run"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )
    store = FileRunStore(tmp_path)
    interrupting = InterruptingStore(
        store,
        lambda state: (
            len(state.steps) == 1
            and state.steps[0].status is StepStatus.RUNNING
        ),
    )
    with pytest.raises(SimulatedProcessExit):
        AgentRuntime(
            planner=FixedPlanner(plan),
            registry=registry,
            run_store=interrupting,
        ).run(_request())
    tool.assert_not_called()

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry, run_store=store
    ).resume("request-1:run")
    state = store.load("request-1:run")

    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION"
    assert state.lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert state.steps[0].status is StepStatus.FAILED
    tool.assert_not_called()


def test_no_store_runtime_remains_in_memory_only(tmp_path: Path) -> None:
    tool = Mock(return_value={"value": "done"})
    registry = ToolRegistry((_spec("tool", tool),))
    plan = AgentPlan(
        "plan-1", "request-1", "fixed", (PlanStep("step", "tool", {}),)
    )

    result = AgentRuntime(
        planner=FixedPlanner(plan), registry=registry
    ).run(_request())

    assert result.status is RunStatus.SUCCEEDED
    assert not tuple(tmp_path.iterdir())
    with pytest.raises(RuntimeError, match="requires an injected RunStore"):
        AgentRuntime(planner=FixedPlanner(plan), registry=registry).resume(
            result.run_id
        )
