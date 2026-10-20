"""Offline acceptance of capability-scoped planning and the shared session budget."""
from dataclasses import FrozenInstanceError, replace
import json
from unittest.mock import Mock

import pytest

from agent.orchestration import (
    AgentRequest, AgentRuntime, FileRunStore, LLMPlanner, PlannerError,
    PlanningModelError, PlanningModelProfile, RunMode, RunStatus, ToolRegistry,
    build_default_tool_registry,
)
from agent.orchestration.planning_scope import (
    PlanningScope, capability_index, parse_selection, selection_request,
)
from agent.orchestration.semantic_prompt import build_semantic_planning_catalog, build_semantic_planning_prompt
from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema, parse_semantic_wire_v4
from agent.orchestration.semantic_compiler import build_semantic_compiler_contract, compile_semantic_plan
from agent.providers import PlanningModelFactoryRegistry


def selected(*ids):
    return json.dumps({"selection_schema_version": 1, "decision": {"kind": "select", "capability_ids": ids}})


def plan(tool="inspect_scATAC", sources=()):
    return json.dumps({"schema_version": 4, "decision": {"kind": "plan", "steps": [
        {"step_id": "inspect", "tool": tool, "sources": sources, "control_dependencies": []}]}})


class Model:
    model_id = "scoped-test"
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
    def complete(self, *, prompt, response_schema):
        self.calls.append((prompt, response_schema))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def registry():
    original = build_default_tool_registry()
    guard = Mock(side_effect=AssertionError("Planning executed science"))
    return ToolRegistry(tuple(replace(original.get(n), function=guard) for n in original.names()))


@pytest.fixture
def request_value():
    return AgentRequest("scope-test", "Inspect supplied data", {"input_path": "/PRIVATE-SECRET.h5ad"}, RunMode.PLAN_ONLY)


def diagnostics(result):
    return [e.details for e in result.trace if "diagnostic_schema_version" in e.details]


def test_metadata_scope_union_order_overlap_and_immutability(registry):
    index = capability_index(registry)
    assert set().union(*map(set, index.values())) == set(registry.names())
    first = PlanningScope(registry, ("embedding_analysis", "reference_annotation"))
    other = PlanningScope(registry, ("reference_annotation", "embedding_analysis"))
    assert first == other
    assert first.visible_tool_names == tuple(sorted(set(index['embedding_analysis']) | set(index['reference_annotation'])))
    assert first.visible_tool_names.count("epizoo_embed_cells") == 1
    with pytest.raises(FrozenInstanceError):
        first.capability_ids = ()
    assert first.scope_fingerprint != PlanningScope(registry, ("embedding_analysis",)).scope_fingerprint


def test_projection_preserves_guidance_and_full_compiler(registry, request_value):
    names = ('inspect_scATAC',)
    full = build_semantic_planning_catalog(request_value, registry)
    scoped = build_semantic_planning_catalog(request_value, registry, visible_tool_names=names)
    assert scoped['tools'] == {n:full['tools'][n] for n in names}
    candidate = parse_semantic_wire_v4(plan(), request_value, registry, visible_tool_names=names)
    contract = build_semantic_compiler_contract(registry)
    assert any(rule.tool_name=='epizoo_embed_cells' for rule in contract.request_bindings)
    assert compile_semantic_plan(request_value,candidate,registry,contract) == compile_semantic_plan(
        request_value,parse_semantic_wire_v4(plan(),request_value,registry),registry,contract)
    with pytest.raises(PlannerError, match="outside"):
        parse_semantic_wire_v4(plan('epizoo_embed_cells'),request_value,registry,visible_tool_names=names)


def test_durable_accepted_plan_resume_needs_no_provider(registry,request_value,tmp_path):
    model=Model(plan())
    runtime=AgentRuntime(planner=LLMPlanner(model),registry=registry,run_store=FileRunStore(tmp_path))
    result=runtime.run(request_value)
    assert result.status is RunStatus.PLANNED
    assert runtime.resume(result.run_id)==result
    assert len(model.calls)==1


def test_narrowing_preserves_registry_wide_optional_ambiguity(registry):
    tool = registry.get('epizoo_embed_cells')
    clone = replace(tool, name='another_embedding', planning=replace(tool.planning, capability_ids=('marker_annotation',)))
    expanded = ToolRegistry(tuple(registry.get(n) for n in registry.names()) + (clone,))
    request = AgentRequest('optional', 'embed', {'input_path':'/x','embedding_overwrite':True})
    full = build_semantic_planning_catalog(request, expanded)
    narrowed = build_semantic_planning_catalog(request, expanded, visible_tool_names=('epizoo_embed_cells',))
    assert narrowed['tools']['epizoo_embed_cells'] == full['tools']['epizoo_embed_cells']
    assert narrowed['tools']['epizoo_embed_cells'][1]['overwrite'][1] == 'optional_explicit'


def test_schema_scope_has_no_producer_completion(registry, request_value):
    scope = PlanningScope(registry, ('marker_annotation',))
    assert scope.visible_tool_names == ('annotate_scATAC_cell_types',)
    spec = registry.get('annotate_scATAC_cell_types').semantic_planning
    assert not next(p for p in spec.consumer_ports if p.name=='matrix').accepted_upstream_types
    schema=build_semantic_wire_v4_schema(registry,request_value,visible_tool_names=scope.visible_tool_names)
    assert len(schema['$defs']['step']['anyOf']) == 1


def body_bytes(prompt, schema):
    return len(json.dumps({'model':'openai/gpt-oss-120b','input':prompt,'text':{'format':{
        'type':'json_schema','name':'agent_plan','strict':True,'schema':schema}}},
        ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())


def test_scope_fingerprint_binds_semantic_request_contract(registry):
    tool=registry.get('inspect_scATAC')
    semantic=tool.semantic_planning
    port=semantic.consumer_ports[0]
    source=port.request_sources[0]
    member=replace(source.members[0],input_name='future_input')
    changed_source=replace(source,selector='future_input',members=(member,))
    changed_port=replace(port,request_sources=(changed_source,))
    changed=replace(tool,semantic_planning=replace(semantic,consumer_ports=(changed_port,)))
    new_registry=ToolRegistry(tuple(changed if n==tool.name else registry.get(n) for n in registry.names()))
    old_scope=PlanningScope(registry,('processed_inspection',))
    assert old_scope.scope_fingerprint != PlanningScope(new_registry,('processed_inspection',)).scope_fingerprint
    with pytest.raises(PlannerError):old_scope.validate(new_registry)
