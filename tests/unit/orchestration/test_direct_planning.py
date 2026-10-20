"""Single-call Registry-derived planning, exact bindings and bounded recovery."""
from dataclasses import replace
import json

from agent.orchestration import (AgentRequest, AgentRuntime, LLMPlanner, RunMode,
    RunStatus, ToolRegistry, build_default_tool_registry, PlanningModelError)
from agent.orchestration.planning_scope import PlanningScope


class Model:
    model_id = 'scripted-direct-planning'
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
    def complete(self, *, prompt, response_schema):
        self.calls.append((json.loads(prompt), response_schema))
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return json.dumps(result)


def plan(tool='inspect_scATAC'):
    return dict(schema_version=4, decision=dict(kind='plan', steps=[
        dict(step_id='inspect', tool=tool, sources=[], control_dependencies=[])]))


def test_one_semantic_call_with_all_registered_tools_and_no_paths():
    registry = build_default_tool_registry()
    model = Model(plan())
    request = AgentRequest('direct', 'Inspect the supplied matrix.',
                           {'input_path':'/private/source.h5ad'}, RunMode.PLAN_ONLY)
    result = AgentRuntime(planner=LLMPlanner(model), registry=registry).run(request)
    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 1
    prompt, schema = model.calls[0]
    assert 'selection_schema_version' not in prompt
    assert '/private/source.h5ad' not in json.dumps(model.calls)
    assert {v['properties']['tool']['enum'][0] for v in schema['$defs']['step']['anyOf']} == set(PlanningScope(registry).visible_tool_names)
    assert result.plan.steps[0].arguments['path'] == '/private/source.h5ad'


def test_registered_extension_needs_no_central_family_description():
    registry = build_default_tool_registry()
    original = registry.get('inspect_scATAC')
    extended = ToolRegistry((replace(original, planning=replace(original.planning,
        capability_ids=('new_scientific_family',))),))
    assert PlanningScope(extended).visible_tool_names == ('inspect_scATAC',)
    model = Model(plan())
    accepted = LLMPlanner(model).plan(AgentRequest('extension', 'Inspect.', {'input_path':'private'}), extended)
    assert accepted.steps[0].tool_name == 'inspect_scATAC' and len(model.calls) == 1


def test_transport_recovery_does_not_reintroduce_classification():
    model = Model(PlanningModelError('timeout', code='PROVIDER_TIMEOUT'), plan())
    request = AgentRequest('retry', 'Inspect.', {'input_path':'/private/source.h5ad'}, RunMode.PLAN_ONLY)
    result = AgentRuntime(planner=LLMPlanner(model, retry_sleeper=lambda seconds: None)).run(request)
    assert result.status is RunStatus.PLANNED and len(model.calls) == 2
    assert all('semantic_prompt_version' in p for p, _ in model.calls)


def test_retired_scope_policy_is_explicitly_rejected_for_new_plans():
    import pytest
    from agent.orchestration import PlanningRecoveryPolicy
    from agent.orchestration.planning_recovery import SCOPED_PLANNING_RECOVERY_POLICY_VERSION
    policy = PlanningRecoveryPolicy(policy_version=SCOPED_PLANNING_RECOVERY_POLICY_VERSION)
    with pytest.raises(ValueError, match='retired'):
        LLMPlanner(Model(plan()), recovery_policy=policy)
