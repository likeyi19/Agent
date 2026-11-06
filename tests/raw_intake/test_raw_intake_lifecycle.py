from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from agent.orchestration import (AgentPlan, AgentRequest, AgentRuntime, FileRunStore,
    PlanStep, RunMode, RunStatus, RunLifecycleStatus, StepStatus, ToolRegistry, build_default_tool_registry)
from agent.tools.data import raw_scatac as raw


class FixedPlanner:
    def __init__(self, plan):
        self.value = plan
        self.calls = 0

    def plan(self, request, registry):
        self.calls += 1
        return self.value


def setup(args, *, twice=False):
    default = build_default_tool_registry()
    call = Mock(wraps=default.get('inspect_raw_scATAC').function)
    registry = ToolRegistry(tuple(replace(default.get(n), function=call)
        if n == 'inspect_raw_scATAC' else default.get(n) for n in default.names()))
    steps = (PlanStep('first', 'inspect_raw_scATAC', args),)
    if twice:
        steps += (PlanStep('second', 'inspect_raw_scATAC', args, ('first',)),)
    planner = FixedPlanner(AgentPlan('raw-plan', 'raw-request', 'fixed', steps))
    return registry, planner, call


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('malformed', [False, True])
def test_runtime_success_persistence_and_terminal_resume(raw_factory, tmp_path, kind, malformed):
    args = raw_factory(kind, malformed=malformed)
    registry, planner, call = setup(args)
    store = FileRunStore(tmp_path / 'store')
    runtime = AgentRuntime(registry=registry, planner=planner, run_store=store)
    result = runtime.run(AgentRequest('raw-request', 'inspect', {}))
    assert result.status is RunStatus.SUCCEEDED, result.errors
    assert result.steps[0].verification.passed
    assert result.steps[0].result['readiness'] == ('INVALID' if malformed else 'READY')
    assert store.load(result.run_id).to_run_result() == result
    # Terminal resume keeps the existing immutable-return contract, even after drift.
    path = next(Path(args['raw_input_paths']).iterdir())
    path.write_bytes(b'changed after terminal success')
    assert runtime.resume(result.run_id) == result
    assert call.call_count == planner.calls == 1
    assert len(list(Path(args['output_dir']).iterdir())) == 1


class ProcessExit(BaseException):
    pass


class InterruptStore:
    def __init__(self, delegate):
        self.delegate = delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def update(self, state, *, expected_revision):
        saved = self.delegate.update(state, expected_revision=expected_revision)
        if (saved.lifecycle_status is RunLifecycleStatus.RUNNING and len(saved.steps) == 2
                and saved.steps[0].status is StepStatus.SUCCEEDED
                and saved.steps[1].status is StepStatus.PENDING):
            raise ProcessExit()
        return saved


@pytest.mark.parametrize('kind', ['fastq', 'bam'])
@pytest.mark.parametrize('drift', [False, True])
def test_nonterminal_resume_revalidates_without_rerunning_completed_tool(raw_factory, tmp_path, kind, drift):
    args = raw_factory(kind)
    registry, planner, call = setup(args, twice=True)
    store = FileRunStore(tmp_path / 'store')
    with pytest.raises(ProcessExit):
        AgentRuntime(registry=registry, planner=planner, run_store=InterruptStore(store)).run(
            AgentRequest('raw-request', 'inspect', {}))
    assert call.call_count == 1
    if drift:
        next(Path(args['raw_input_paths']).iterdir()).write_bytes(b'changed')
    forbidden_planner = Mock()
    forbidden_planner.plan.side_effect = AssertionError('planner called on resume')
    result = AgentRuntime(registry=registry, planner=forbidden_planner, run_store=store).resume('raw-request:run')
    assert result.status is (RunStatus.FAILED if drift else RunStatus.SUCCEEDED), result.errors
    assert call.call_count == (1 if drift else 2)  # Second pending step only.
    assert planner.calls == 1 and forbidden_planner.plan.call_count == 0
    assert len(list(Path(args['output_dir']).iterdir())) == 1


def test_runtime_detects_source_changed_after_tool_before_verification(raw_factory):
    args = raw_factory()
    registry, planner, call = setup(args)
    def mutate(**kwargs):
        result = raw.inspect_raw_scATAC(**kwargs)
        next(Path(args['raw_input_paths']).iterdir()).write_bytes(b'changed')
        return result
    call.side_effect = mutate
    result = AgentRuntime(registry=registry, planner=planner).run(AgentRequest('raw-request', 'inspect', {}))
    assert result.status is RunStatus.FAILED
    assert not result.steps[0].verification.passed
    assert call.call_count == 1
