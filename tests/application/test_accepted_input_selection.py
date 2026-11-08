"""Scientific selections are independent of metadata/figure references; no science."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from agent.application.turn_decisions import (Execute, RequestRoles, IntentError,
    decision_schema, parse_decision)
from agent.application.raw_handoff import AnnotationDeclaration
from agent.application.dialogue_execution import _research_intent, _validate_research_intent, planning_task
from agent.application.session_state import Interaction
from test_interpretation_protocol import accepts


def interaction(admitted=None):
    return Interaction('new', 'Reuse the accepted groups', 'revision', 1,
        dict(dialogue=dict(results=[dict(handle='g0', tool='run_snapatac2', cell_capabilities=dict(coordinates='X_umap'))])),
        admitted=admitted)


@pytest.mark.parametrize('context', ['none', 'current'])
def test_exact_scientific_input_does_not_depend_on_coarse_context(context):
    decision = Execute('current', 'plan', 'run_scanpy', None,
        context=context, input_results=('g0',), references=('g0',))
    value = dict(kind='execute', tool='run_scanpy', context=context,
        **_research_intent(interaction(), decision))
    _validate_research_intent(interaction(value))
    assert planning_task(interaction(value), SimpleNamespace(interactions=()))['input_results'] == ['g0']


def test_metadata_and_visualization_never_become_scientific_inputs():
    decision = Execute('current', 'plan', 'run_scanpy', None, references=('g0',),
        visualization=dict(coordinates='g0', labels='new', field='result:primary_annotation'))
    assert 'input_results' not in _research_intent(interaction(), decision)


@pytest.mark.parametrize('handles', [('g1',), ('g0', 'g0'), ('ctx.0',)])
def test_invalid_or_unoffered_scientific_inputs_are_rejected(handles):
    with pytest.raises((ValueError, IntentError)):
        _research_intent(interaction(), Execute('current', 'plan', 'run_scanpy', None, input_results=handles))


def test_new_annotation_schema_can_express_the_documented_context_and_input_selection():
    # Before the correction this branch omitted context despite instructions.
    decision = dict(kind='request_roles', target='annotate_scATAC_cell_types',
        required_roles=[], roles=[], arguments=[], context='none', input_results=['g0'],
        references=[], delivery=[], unfulfilled=[], visualization=None, comparison=None,
        retain_parameters=[], annotation=asdict(AnnotationDeclaration(grouping_strategy='reuse_only')))
    # The current schema uses the existing raw schema vocabulary, including optional fields.
    from agent.application.raw_handoff import schema
    for key in set(decision['annotation'])-set(schema('annotation')['properties']):
        decision['annotation'].pop(key)
    wire=dict(turn_schema_version=1, decision=decision)
    assert accepts(decision_schema(), wire)
    parsed=parse_decision(json.dumps(wire))
    assert isinstance(parsed, RequestRoles) and parsed.input_results == ('g0',)


def test_old_annotation_wire_stays_readable_with_no_new_implicit_inputs():
    wire=dict(turn_schema_version=1, decision=dict(kind='request_roles',
        target='annotate_scATAC_cell_types', required_roles=[], roles=[],
        annotation=asdict(AnnotationDeclaration())))
    parsed=parse_decision(json.dumps(wire))
    assert parsed.context == 'none' and parsed.input_results == ()


def test_nonplanning_operation_cannot_gain_scientific_input_selection():
    with pytest.raises(ValueError):
        Execute('current', 'selection', 'selection', None, input_results=('g0',))


@pytest.fixture
def captured_sources(monkeypatch):
    """Two accepted-metadata fixtures; binding authenticity is simulated here."""
    from agent.application.sessions import AnalysisSessions
    from agent.application.session_state import OutputLocator
    from agent.schemas.prior_output import PriorOutputBinding
    from agent.orchestration import prior_outputs
    from agent.application import turn_context
    outputs=tuple(OutputLocator(f'file{i}',f'run{i}:run',f'step{i}','matrix_path',str(i+1)*64) for i in range(2))
    descriptions=[dict(handle=f'g{i}', relations=['current'], derived_from=[]) for i in range(2)]
    shown=[dict(handle=f'r{i}',revision_id='revision',output_name=o.name,
        accepted_step_sha256=o.accepted_step_sha256,tool='run_scanpy') for i,o in enumerate(outputs)]
    captured=dict(bases=dict(revision=dict(revision_id='revision',outputs=[asdict(o) for o in outputs])),
        relations=dict(current='revision'),dialogue=dict(outputs=shown,results=descriptions))
    item=Interaction('new','Use selected accepted data','revision',1,captured)
    revision=SimpleNamespace(revision_id='revision',outputs=outputs)
    state=SimpleNamespace(interactions=(item,),revisions=(revision,))
    sessions=AnalysisSessions.__new__(AnalysisSessions)
    sessions._application=SimpleNamespace(run_store=object(),registry=object())
    sessions.load=lambda sid: state
    reads=[]
    def binding(output,*args):
        reads.append(output.name)
        return PriorOutputBinding(output.run_id,output.step_id,'run_scanpy','dataset',
            'native_matrix_h5ad.v1','1'*64,'2'*64,'3'*64)
    monkeypatch.setattr(prior_outputs,'optional_binding_for_locator',binding)
    monkeypatch.setattr(prior_outputs,'matrix_dataset_binding',lambda *a:None)
    monkeypatch.setattr(prior_outputs,'validate_binding',lambda *a,**k:None)
    monkeypatch.setattr(turn_context,'result_descriptions',lambda *a:descriptions)
    return sessions,reads,outputs


@pytest.mark.parametrize('broad', ['none','current'])
def test_exact_generic_selection_excludes_unrelated_accepted_results(captured_sources,broad):
    sessions,reads,_=captured_sources
    context,_=sessions.execution_context_for_interaction('session','new',admitted=dict(
        operation='plan',context=broad,input_results=['g1'],references=['g0']))
    assert reads==['file1']
    assert [i.binding.run_id for i in context.items]==['run1:run']
    assert not context.public_research_context()['results'][0]['reusable_inputs']


def test_broad_reuse_retains_existing_contract_without_exact_selection(captured_sources):
    sessions,reads,_=captured_sources
    context,_=sessions.execution_context_for_interaction('session','new',admitted=dict(context='current'))
    assert reads==['file0','file1'] and len(context.items)==2


def test_presentation_only_context_never_checks_or_offers_executable_sources(captured_sources):
    sessions,reads,_=captured_sources
    context,_=sessions.execution_context_for_interaction('session','new',admitted=dict(
        context='none',references=['g0'],visualization=dict(coordinates='g0',labels='new')))
    assert not context.items and not reads


def test_selected_binding_authority_failure_is_not_hidden_by_another_result(captured_sources,monkeypatch):
    from agent.orchestration import prior_outputs
    from agent.schemas.verification_authority import AuthorityError
    sessions,reads,_=captured_sources
    def corrupt(*a,**kw):raise AuthorityError('Missing selected accepted authority')
    monkeypatch.setattr(prior_outputs,'validate_binding',corrupt)
    with pytest.raises(AuthorityError):
        sessions.execution_context_for_interaction('session','new',admitted=dict(input_results=['g1']))
    assert reads==['file1']


def test_unoffered_selection_never_falls_back_to_current(captured_sources):
    sessions,reads,_=captured_sources
    with pytest.raises(IntentError):
        sessions.execution_context_for_interaction('session','new',admitted=dict(context='current',input_results=['g99']))
    assert not reads


def test_source_locator_cannot_cross_captured_revision(captured_sources):
    from agent.application.session_state import SessionError
    sessions,reads,outputs=captured_sources
    foreign=replace(outputs[0],run_id='foreign:run')
    with pytest.raises(SessionError):
        sessions.planning_context_for_interaction('session','new',selected_outputs=(foreign,))
    assert not reads


def test_historical_description_cannot_alias_a_same_named_current_output(captured_sources):
    sessions,reads,outputs=captured_sources
    state=sessions.load('session')
    historical=replace(outputs[1],run_id='older:run',accepted_step_sha256='9'*64)
    state.revisions+=(SimpleNamespace(revision_id='historical',outputs=(historical,)),)
    item=state.interactions[0]
    from agent.schemas.orchestration import _serialize
    captured=_serialize(item.snapshot)
    captured['bases']['historical']=dict(revision_id='historical',outputs=[asdict(historical)])
    captured['dialogue']['outputs'].append(dict(handle='r2',revision_id='historical',output_name=historical.name,
        accepted_step_sha256=historical.accepted_step_sha256,tool='run_scanpy'))
    captured['dialogue']['results'].append(dict(handle='g2',relations=['historical'],derived_from=[]))
    state.interactions=(replace(item,snapshot=captured),)
    with pytest.raises(IntentError):
        sessions.execution_context_for_interaction('session','new',admitted=dict(input_results=['g2']))
    assert not reads


def test_exact_inputs_survive_prerequisite_continuation_and_handle_reordering(captured_sources):
    sessions,reads,_=captured_sources
    state=sessions.load('session')
    origin=replace(state.interactions[0],admitted=dict(kind='execute',operation='plan',input_results=['g0']))
    from agent.schemas.orchestration import _serialize
    captured=_serialize(origin.snapshot)
    captured['dialogue']['outputs'].reverse()
    reply=Interaction('reply','Use the requested parameter','revision',1,captured,admitted=dict(
        kind='execute',operation='plan',continuation=dict(origin_turn_id='new',binding_turn_id='new',field='parameters')))
    state.interactions=(origin,reply)
    context,_=sessions.execution_context_for_interaction('session','reply')
    assert context.task['input_results']==('g1',)
    assert reads==['file0'] and context.items[0].binding.run_id=='run0:run'
