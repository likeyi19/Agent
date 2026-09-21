from dataclasses import replace
import json
from fractions import Fraction
import pytest
from agent.application.turn_decisions import *
from agent.application.turn_context import parameter_specs
from agent.orchestration import build_default_tool_registry


@pytest.mark.parametrize('utterance,operation,literal,current,expected', [
    ('Set the TSS threshold to 7','set','7',6,7),
    ('Increase the TSS threshold by 1','add','1',6,7),
    ('Decrease it by 1','subtract','1',7,6),
    ('Increase TSS by 0.1','add','0.1','1/3','13/30'),
    ('try min_tss = 7','set','7',6,7),
])
def test_exact_grounded_arithmetic(utterance, operation, literal, current, expected):
    delta=IntentDelta('min_tss_enrichment',operation,literal,utterance)
    result=admit_delta(delta,utterance,{'min_tss_enrichment':current},
        parameter_specs(build_default_tool_registry()),focus='min_tss_enrichment')
    assert result == expected


@pytest.mark.parametrize('utterance,delta', [
    ('Make TSS stricter',IntentDelta('min_tss_enrichment','set','7','Make TSS stricter')),
    ('Set TSS to 7',IntentDelta('min_tss_enrichment','set','8','Set TSS to 7')),
    ('Increase TSS by 1',IntentDelta('min_tss_enrichment','set','1','Increase TSS by 1')),
    ('Decrease TSS by 9',IntentDelta('min_tss_enrichment','subtract','9','Decrease TSS by 9')),
    ('Do not set TSS to 7',IntentDelta('min_tss_enrichment','set','7','set TSS to 7')),
    ('Set threshold to 7',IntentDelta('min_tss_enrichment','set','7','Set threshold to 7')),
    ('Set foo_threshold to 7',IntentDelta('min_tss_enrichment','set','7','Set foo_threshold to 7')),
    ('Decrease it by 1',IntentDelta('min_tss_enrichment','subtract','1','Decrease it by 1')),
    ('Increase TSS by 10%',IntentDelta('min_tss_enrichment','add','10','Increase TSS by 10%')),
])
def test_unadmitted_changes(utterance,delta):
    with pytest.raises(IntentError):
        admit_delta(delta,utterance,{'min_tss_enrichment':6},parameter_specs(build_default_tool_registry()))


def test_strict_decision_parser_and_bounded_schema():
    assert parse_decision(json.dumps({'turn_schema_version':1,'decision':{'kind':'navigate','relation':'parent'}})) == Navigate('parent')
    for raw in ('{}','{"kind":"navigate","relation":"parent","revision_id":"invented"}',
                '{"kind":"clarify","reason":"oops"}', '{"kind":"navigate","kind":"clarify","relation":"parent"}'):
        with pytest.raises(IntentError): parse_decision('{"turn_schema_version":1,"decision":'+raw+'}')


def test_parent_previous_distinction():
    snap={'relations':{'current':'R3','parent':'R1','previous_active':'R2'}}
    assert resolve_relation('parent',snap,'use parent version') == 'R1'
    assert resolve_relation('previous_active',snap,'use the version I was using before') == 'R2'
    with pytest.raises(IntentError): resolve_relation('previous',snap,'use the previous version')
    with pytest.raises(IntentError): resolve_relation('parent',snap,'use the previous version')
    with pytest.raises(IntentError): resolve_relation('parent',snap,'use the earlier version')
    with pytest.raises(IntentError): resolve_relation('previous',snap,'use the earlier version')


def test_additive_session_format_stays_closed_and_empty_compatible():
    from agent.application.session_state import AnalysisSession
    state=AnalysisSession('session')
    assert 'interactions' not in state.to_dict()
    assert AnalysisSession.from_dict(json.loads(json.dumps(state.to_dict()))) == state


def test_planner_effect_guard_preserves_parameters_sources_and_arbitrary_step_ids():
    from agent.application.turns import _check_plan
    from agent.schemas import AgentPlan, PlanStep, PriorOutputBinding, PriorOutputRef
    from agent.orchestration.planner import PlannerError
    binding=PriorOutputBinding('r','q','compute_scATAC_qc','barcode_qc','scatac_barcode_qc.v1','1'*64,'2'*64,'3'*64)
    parameters={'min_qc_fragment_records':3000,'min_tss_enrichment':7}
    admitted=dict(delta={},parameters=parameters,target='selection',operation={'source':{'name':'chosen'}})
    args=dict(parameters,barcode_qc_manifest_path=PriorOutputRef(binding,'manifest_path'),
        barcode_qc_manifest_sha256=PriorOutputRef(binding,'manifest_sha256'),output_dir='/tmp/test-intent')
    plan=AgentPlan('p','r','fixture',(PlanStep('arbitrary-output-step','select_scATAC_cells',args),))
    registry=build_default_tool_registry()
    assert _check_plan(plan,admitted,parameters,{'qc':binding},registry)[0].step_id=='arbitrary-output-step'
    for changed in (args|{'min_qc_fragment_records':3001},args|{'max_qc_fragment_records':9000},
                    args|{'barcode_qc_manifest_path':PriorOutputRef(replace(binding,run_id='other'),'manifest_path')}):
        with pytest.raises(PlannerError):
            _check_plan(replace(plan,steps=(replace(plan.steps[0],arguments=changed),)),admitted,parameters,{'qc':binding},registry)
