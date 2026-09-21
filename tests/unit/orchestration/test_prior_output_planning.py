"""Bounded context wire/compiler and backwards-compatible plan identity."""
from dataclasses import replace
import json

import pytest

from agent.schemas import AgentRequest, AgentPlan, PlanStep
from agent.schemas.prior_output import PriorOutputBinding, PriorOutputRef
from agent.schemas.run_state import _decode_plan, fingerprint_plan
from agent.orchestration import build_default_tool_registry
from agent.orchestration.active_context import ActiveContextItem, ActivePlanningContext, planning_context
from agent.orchestration.planning_scope import selection_request
from agent.orchestration.semantic_prompt import build_semantic_planning_prompt
from agent.orchestration.semantic_wire_v4 import build_semantic_wire_v4_schema, parse_semantic_wire_v4
from agent.orchestration.semantic_compiler import compile_semantic_plan, build_semantic_compiler_contract


def binding():
    return PriorOutputBinding('PRIVATE-RUN:run', 'PRIVATE-STEP', 'compute_scATAC_qc',
        'barcode_qc', 'scatac_barcode_qc.v1', '1'*64, '2'*64, '3'*64)


def context():
    return ActivePlanningContext('private-session', 'private-revision', 9,
                                  (ActiveContextItem('ctx.0', binding()),))


def request():
    return AgentRequest('request', 'Using the current QC result, select using the supplied thresholds.',
                        {'min_qc_fragment_records': 10, 'min_tss_enrichment': '2', 'output_dir': '/PRIVATE/OUT'})


def wire(handle='ctx.0', target='barcode_qc'):
    return {'schema_version': 4, 'decision': {'kind': 'plan', 'steps': [{
        'step_id': 'selection', 'tool': 'select_scATAC_cells',
        'sources': [{'target': target, 'source': {'kind': 'context', 'handle': handle}}],
        'control_dependencies': []}]}}


def test_scoped_context_private_identity_never_enters_prompts():
    registry = build_default_tool_registry()
    with planning_context(context()):
        a, _ = selection_request(request(), registry)
        b = build_semantic_planning_prompt(request(), registry, visible_tool_names=('select_scATAC_cells',))
        schema = build_semantic_wire_v4_schema(registry, request(), visible_tool_names=('select_scATAC_cells',))
    for value in (a,b,json.dumps(schema)):
        for private in ('PRIVATE', 'private-session', 'private-revision', '1'*64, '2'*64, '3'*64):
            assert private not in value
    assert json.loads(a)['active_artifact_types'] == ['scatac_barcode_qc.v1']
    assert len(a.encode()) < 5000 and len(b.encode()) < 10000
    assert schema['$defs']['context_source']['properties']['handle']['enum'] == ('ctx.0',)


def test_context_lowering_roundtrip_fingerprint():
    registry = build_default_tool_registry()
    with planning_context(context()):
        candidate = parse_semantic_wire_v4(json.dumps(wire()), request(), registry)
        plan = compile_semantic_plan(request(), candidate, registry, build_semantic_compiler_contract(registry))
    assert _decode_plan(plan.to_dict()) == plan
    refs = [a for a in plan.steps[0].arguments.values() if isinstance(a, PriorOutputRef)]
    assert len(refs) == 2 and refs[0].binding == refs[1].binding
    assert plan.steps[0].depends_on == ()
    other = replace(binding(), run_id='OTHER:run')
    changed = replace(plan, steps=(replace(plan.steps[0], arguments={
        k: replace(v, binding=other) if isinstance(v, PriorOutputRef) else v
        for k,v in plan.steps[0].arguments.items()}),))
    assert fingerprint_plan(plan) != fingerprint_plan(changed)
    assert fingerprint_plan(_decode_plan(plan.to_dict())) == fingerprint_plan(plan)


@pytest.mark.parametrize('handle,target', [('ctx.99','barcode_qc'), ('ctx.0','output_dir')])
def test_unknown_and_incompatible_context_rejected(handle, target):
    registry = build_default_tool_registry()
    with planning_context(context()), pytest.raises(ValueError):
        candidate = parse_semantic_wire_v4(json.dumps(wire(handle,target)), request(), registry)
        compile_semantic_plan(request(), candidate, registry, build_semantic_compiler_contract(registry))


def test_unoffered_or_injected_exact_identity_rejected():
    registry = build_default_tool_registry()
    with pytest.raises(ValueError): parse_semantic_wire_v4(json.dumps(wire()), request(), registry)
    value = wire()
    value['decision']['steps'][0]['sources'][0]['source']['run_id'] = 'other-session:run'
    with planning_context(context()), pytest.raises(ValueError):
        parse_semantic_wire_v4(json.dumps(value), request(), registry)


def test_old_plan_shape_and_fingerprint_unchanged():
    from agent.schemas import StepOutputRef
    plan = AgentPlan('p','r','legacy', (PlanStep('a','producer', {'path':'/input'}),
        PlanStep('b','consumer',{'value':StepOutputRef('a','value')}, ('a',))))
    expected = {'plan_id':'p','request_id':'r','planner_name':'legacy','steps':[
        {'step_id':'a','tool_name':'producer','arguments':{'path':'/input'},'depends_on':[],'description':None},
        {'step_id':'b','tool_name':'consumer','arguments':{'value':{'$ref':{'step_id':'a','output_key':'value'}}},
         'depends_on':['a'],'description':None}]}
    import hashlib
    assert plan.to_dict() == expected
    assert _decode_plan(expected) == plan
    assert fingerprint_plan(plan) == hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with planning_context(context()): assert plan.to_dict() == expected


def test_weak_verification_mode_never_upgraded():
    with pytest.raises(ValueError): replace(binding(), verification_mode='verified=true')


def test_unicode_step_anchor_matches_existing_session_locator_contract():
    from agent.schemas import StepExecutionResult, StepStatus
    from agent.application.session_state import digest as locator_digest
    from agent.orchestration.prior_outputs import accepted_step_digest
    step = StepExecutionResult('检查', 'inspect_scATAC', StepStatus.SUCCEEDED,
        result={'path': '/数据/样本.h5ad'}, resolved_arguments={'input_path': '/数据/样本.h5ad'})
    assert accepted_step_digest(step) == locator_digest(step.to_dict())
