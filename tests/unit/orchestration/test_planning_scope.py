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


@pytest.mark.parametrize("response", [
    selected("unknown"), selected("processed_inspection", "processed_inspection"), selected(),
    "{}", "not json", '{"selection_schema_version":1,"selection_schema_version":1,"decision":{}}',
    '{"selection_schema_version":true,"decision":{"kind":"unsupported"}}',
    '{"selection_schema_version":1,"decision":{"kind":"unsupported","capability_ids":[]}}',
    '{"selection_schema_version":1,"decision":{"kind":"select","capability_ids":[1]}}',
    '{"selection_schema_version":1,"decision":{"kind":"select","capability_ids":["processed_inspection"],"reason":"no"}}',
])
def test_invalid_selection_terminal_before_stage_b(registry, request_value, response):
    model = Model(response, plan())
    result = AgentRuntime(planner=LLMPlanner(model), registry=registry).run(request_value)
    assert result.status is RunStatus.FAILED
    assert result.errors[0].code == "INVALID_PLANNING_SCOPE"
    assert len(model.calls) == 1
    assert diagnostics(result)[-1]["total_provider_call_count"] == 1


@pytest.mark.parametrize("outcome", [
    PlanningModelError("PRIVATE", code="PROVIDER_TIMEOUT"),
    PlanningModelError("PRIVATE", code="PROVIDER_REQUEST_TOO_LARGE"),
    '{"selection_schema_version":1,"decision":{"kind":"unsupported"}}',
])
def test_stage_a_never_recovers(registry, request_value, outcome):
    model = Model(outcome, selected("processed_inspection"), plan())
    result = AgentRuntime(planner=LLMPlanner(model), registry=registry).run(request_value)
    assert result.status is RunStatus.FAILED
    assert len(model.calls) == 1
    assert not any(d.get("retry_used") or d.get("repair_used") or d.get("failover_used") for d in diagnostics(result))


def test_normal_session_scoped_safe_and_diagnosable(registry, request_value):
    model = Model(selected("processed_inspection"), plan())
    result = AgentRuntime(planner=LLMPlanner(model), registry=registry).run(request_value)
    assert result.status is RunStatus.PLANNED
    assert len(model.calls) == 2
    a, b = [json.loads(call[0]) for call in model.calls]
    assert set(a) == {"selection_schema_version", "instructions", "user_request", "request_inputs", "capabilities"}
    assert set(b['catalog']['tools']) == {'inspect_scATAC'}
    variants = model.calls[1][1]['$defs']['step']['anyOf']
    assert [v['properties']['tool']['enum'][0] for v in variants] == ['inspect_scATAC']
    text = json.dumps([dict(d) for d in diagnostics(result)])
    assert "PRIVATE-SECRET" not in text
    assert "PRIVATE-SECRET" not in json.dumps(model.calls)
    calls = [d for d in diagnostics(result) if d['code']=='PROVIDER_CALL_STARTED']
    assert [d['phase'] for d in calls] == ['scope_selection','detailed_planning']
    assert [d['provider_call_index'] for d in calls] == [1,2]
    assert all(d['session_call_ceiling']==4 for d in calls)
    assert all(d['prompt_fingerprint'] and d['schema_fingerprint'] for d in calls)
    assert calls[0]['planning_wire_schema_version']==1
    assert calls[1]['planning_wire_schema_version']==4
    assert calls[1]['scope_fingerprint']==PlanningScope(registry,('processed_inspection',)).scope_fingerprint
    assert diagnostics(result)[-1]['total_provider_call_count']==2


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


def test_future_tool_metadata_and_unrelated_stage_b_size(registry, request_value):
    tool = registry.get('inspect_scATAC')
    future = replace(tool, name='future_inspection', planning=replace(tool.planning,
        capability_ids=('marker_annotation',)))
    larger = ToolRegistry(tuple(registry.get(n) for n in registry.names())+(future,))
    assert 'future_inspection' in PlanningScope(larger,('marker_annotation',)).visible_tool_names
    names=('inspect_scATAC',)
    assert build_semantic_planning_prompt(request_value,registry,visible_tool_names=names)==build_semantic_planning_prompt(request_value,larger,visible_tool_names=names)
    assert build_semantic_wire_v4_schema(registry,request_value,visible_tool_names=names)==build_semantic_wire_v4_schema(larger,request_value,visible_tool_names=names)
    missing=replace(future,planning=replace(future.planning,capability_ids=()))
    model=Model(selected('processed_inspection'))
    result=AgentRuntime(planner=LLMPlanner(model),registry=ToolRegistry((missing,))).run(request_value)
    assert result.errors[0].code=='PLANNER_CATALOG_INVALID'
    assert model.calls==[]


@pytest.mark.parametrize('failure', [PlanningModelError('secret',code='PROVIDER_TIMEOUT'),'not json'])
def test_stage_b_local_recovery_and_final_failover_share_four_calls(registry, request_value, failure):
    primary=Model(selected('processed_inspection'),failure,failure)
    secondary=Model(plan())
    first=PlanningModelProfile('first','custom','first-model')
    last=PlanningModelProfile('last','custom','last-model')
    planner=LLMPlanner(primary,profile=first,recovery_profiles=(last,),retry_sleeper=lambda _:None,
        model_factory_registry=PlanningModelFactoryRegistry({'custom':lambda _:secondary}))
    result=AgentRuntime(planner=planner,registry=registry).run(request_value)
    assert result.status is RunStatus.PLANNED
    assert len(primary.calls)+len(secondary.calls)==4
    records=diagnostics(result)
    assert records[-1]['total_provider_call_count']==4
    assert [d['provider_call_index'] for d in records if d['code']=='PROVIDER_CALL_STARTED']==[1,2,3,4]
    identities={d['scope_fingerprint'] for d in records if d['phase']=='detailed_planning'}
    assert len(identities)==1
    assert primary.calls[1][1]==primary.calls[2][1]==secondary.calls[0][1]
    assert set(json.loads(secondary.calls[0][0])['catalog']['tools'])=={'inspect_scATAC'}


def test_no_fifth_call_and_out_of_scope_repair(registry, request_value):
    primary=Model(selected('processed_inspection'),plan('epizoo_embed_cells'),plan('epizoo_embed_cells'))
    secondary=Model(plan('epizoo_embed_cells'),plan())
    planner=LLMPlanner(primary,profile=PlanningModelProfile('p','custom','p'),
        recovery_profiles=(PlanningModelProfile('s','custom','s'),),
        model_factory_registry=PlanningModelFactoryRegistry({'custom':lambda _:secondary}))
    result=AgentRuntime(planner=planner,registry=registry).run(request_value)
    assert result.status is RunStatus.FAILED
    assert len(primary.calls)+len(secondary.calls)==4
    assert diagnostics(result)[-1]['total_provider_call_count']==4


@pytest.mark.parametrize('boundary', ['cancel','checkpoint'])
def test_between_stage_boundary_blocks_stage_b(registry,request_value,boundary):
    from agent.orchestration.planning_recovery import PlanningRecoveryCancelled
    model=Model(selected('processed_inspection'),plan())
    planner=LLMPlanner(model)
    cancelled=[]
    def sink(values):
        if any(d.code=='PLANNING_SCOPE_ACCEPTED' for d in values):
            if boundary=='checkpoint':raise OSError('checkpoint failed')
            cancelled.append('now')
    with pytest.raises(PlanningRecoveryCancelled if boundary=='cancel' else OSError):
        planner.plan_with_recovery(request_value,registry,validate_candidate=Mock(side_effect=AssertionError()),
            diagnostic_sink=sink,should_cancel=lambda:cancelled[0] if cancelled else None)
    assert len(model.calls)==1


def test_durable_accepted_plan_resume_needs_no_provider(registry,request_value,tmp_path):
    model=Model(selected('processed_inspection'),plan())
    runtime=AgentRuntime(planner=LLMPlanner(model),registry=registry,run_store=FileRunStore(tmp_path))
    result=runtime.run(request_value)
    assert result.status is RunStatus.PLANNED
    assert runtime.resume(result.run_id)==result
    assert len(model.calls)==2


def test_interrupted_scope_checkpoint_resume_never_replans(registry,request_value,tmp_path):
    store=FileRunStore(tmp_path)
    model=Model(selected('processed_inspection'),plan())
    runtime=AgentRuntime(planner=LLMPlanner(model),registry=registry,run_store=store)
    original=store.update
    def crash(state, **kwargs):
        if any(e.details.get('code')=='PLANNING_SCOPE_ACCEPTED' for e in state.trace):
            raise KeyboardInterrupt()
        return original(state,**kwargs)
    # RunStore updates are intercepted without changing production persistence.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store,'update',crash)
        with pytest.raises(KeyboardInterrupt):runtime.run(request_value)
    result=runtime.resume(request_value.request_id+':run')
    assert result.status is RunStatus.FAILED
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


@pytest.mark.parametrize('families', [
    ('processed_inspection',), ('processed_inspection','embedding_analysis'),
    ('raw_preprocessing',), ('marker_annotation',), ('exact_matrix_adoption',),
    ('differential_accessibility',), ('raw_preprocessing','marker_annotation'),
])
def test_real_scoped_payload_reduction(registry, families):
    request=AgentRequest('measurement','Inspect this scATAC dataset',{'input_path':'/audit/cells.h5ad'})
    scope=PlanningScope(registry,families)
    prompt=build_semantic_planning_prompt(request,registry,visible_tool_names=scope.visible_tool_names)
    schema=build_semantic_wire_v4_schema(registry,request,visible_tool_names=scope.visible_tool_names)
    full=body_bytes(build_semantic_planning_prompt(request,registry),build_semantic_wire_v4_schema(registry,request))
    assert body_bytes(prompt,schema)<full
    a_prompt,a_schema=selection_request(request,registry)
    assert body_bytes(a_prompt,a_schema)<full
    assert not set(registry.names()).intersection(json.loads(a_prompt)['capabilities'])
    print(f'families={families} stage_a_body={body_bytes(a_prompt,a_schema)} prompt={len(prompt.encode())} '
          f'schema={len(json.dumps(schema,sort_keys=True,separators=(",", ":")).encode())} stage_b_body={body_bytes(prompt,schema)}')


def test_stage_b_413_terminal_and_local_failure_does_not_spend_call(registry,request_value,monkeypatch):
    model=Model(selected('processed_inspection'),PlanningModelError('secret',code='PROVIDER_REQUEST_TOO_LARGE'),plan())
    result=AgentRuntime(planner=LLMPlanner(model),registry=registry).run(request_value)
    assert len(model.calls)==2
    assert result.errors[0].code=='PROVIDER_REQUEST_TOO_LARGE'
    assert diagnostics(result)[-1]['total_provider_call_count']==2
    import agent.orchestration.semantic_prompt as projection
    original=projection.build_semantic_planning_prompt
    def broken(*args,**kwargs):
        raise PlannerError('PLANNER_CATALOG_INVALID','local projection failure')
    monkeypatch.setattr(projection,'build_semantic_planning_prompt',broken)
    model=Model(selected('processed_inspection'),plan())
    result=AgentRuntime(planner=LLMPlanner(model),registry=registry).run(request_value)
    assert len(model.calls)==1
    assert result.errors[0].code=='PLANNER_CATALOG_INVALID'
    assert diagnostics(result)[-1]['total_provider_call_count']==1
    monkeypatch.setattr(projection,'build_semantic_planning_prompt',original)


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
