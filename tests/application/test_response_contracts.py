"""One reviewed response owner; deterministic controls and honest status fallback."""
from dataclasses import replace
from types import SimpleNamespace as N
import json
import pytest
from agent.application import Answer, TurnOutcome
from agent.application.turn_decisions import Clarify, IntentError, parse_decision, decision_schema
from agent.application.responses import summarize, conversation_utterances, CONVERSATION_LANGUAGE
from agent.application import scientific_dialogue as sd
from agent.application.interactive_schemas import ClientError
from agent.orchestration.planning_model import PlanningModelError
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, question, ask


@pytest.mark.parametrize('intent', ['version','matrix','selection','thresholds','changes','execution','reuse','comparison','provenance','verification','unsupported'])
def test_retired_answers_are_not_active_decisions(intent):
    with pytest.raises(IntentError):
        parse_decision(json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer',intent=intent,relation='current',technical=False))))
    with pytest.raises(ValueError): Answer(intent)


def test_retired_execution_and_answer_grammar_removed():
    variants=decision_schema()['properties']['decision']['anyOf']
    kinds={k for v in variants for k in v['properties']['kind']['enum']}
    assert 'answer' not in kinds and 'execute' not in kinds
    assert {'answer_scientific','answer_guidance','execute_plan'} <= kinds
    with pytest.raises(IntentError):
        parse_decision(json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute',base='current',operation='op.0',target='selection',delta=None))))
    from agent.orchestration import build_default_tool_registry
    registry=build_default_tool_registry()
    for name in ('select_scATAC_cells','build_scATAC_cell_by_ccre','build_scATAC_cell_by_features','pca_embed_cells'):
        assert name not in registry.names()
        assert registry.get(name).name == name
        assert not hasattr(registry.get(name), 'function')


class OutcomeModel:
    def __init__(self, text='请提供所需参数。', fail=False):
        self.text,self.fail,self.calls=text,fail,[]
    def complete(self, *, prompt, response_schema):
        data=json.loads(prompt);self.calls.append(data)
        if self.fail: raise PlanningModelError('private provider payload',code='PROVIDER_RATE_LIMITED')
        if 'grounding_review_version' in data:
            return json.dumps(dict(verdicts={p['id']:dict(verdict='entailed',independent=True) for p in data['paragraphs']}))
        assert data['focus']=='outcome' and CONVERSATION_LANGUAGE in data['instructions']
        return json.dumps(dict(support='supported',paragraphs=[dict(text=self.text,kind='limitation',citations=[],comparisons=[])]))


def outcome_session():
    interaction=N(turn_id='q',utterance='请用中文说明原因。',admitted={'tool':'run_snapatac2'},prerequisite=None,submission=None)
    state=N(interactions=(interaction,),turns=())
    return N(load=lambda _:state),interaction


@pytest.mark.parametrize('outcome', [TurnOutcome('clarify','clarification',Clarify('invalid_parameter_value',('run_snapatac2.kwargs',),True)),
    TurnOutcome('execute','failed',error=ClientError('NATIVE_FUNCTION_FAILED','Native failure'))])
def test_clarification_and_failure_share_generation_and_review(outcome):
    sessions,interaction=outcome_session();model=OutcomeModel()
    result=summarize(sessions,'s','q',outcome,model=model)
    assert result.text=='请提供所需参数。' and result.clarification==outcome.clarification and result.error==outcome.error
    assert len(model.calls)==2 and model.calls[1]['grounding_review_version']==4
    facts=model.calls[0]['evidence']['outcome']
    assert facts['science_started'] is False and facts['accepted_revision'] is None
    assert facts['requested_operation']=='run_snapatac2'
    assert facts['error']==(None if outcome.error is None else {'code':outcome.error.code,'message':outcome.error.message,'category':None})
    assert model.calls[1]['evidence']['outcome']==facts


def test_provider_failure_never_recurses_and_preserves_original_error():
    sessions,_=outcome_session();model=OutcomeModel(fail=True)
    native=TurnOutcome('execute','failed',error=ClientError('NATIVE_FUNCTION_FAILED','Native failure'))
    result=summarize(sessions,'s','q',native,model=model)
    assert len(model.calls)==1
    assert result.error==native.error and 'PROVIDER_RATE_LIMITED' in result.text and 'NATIVE_FUNCTION_FAILED' in result.text
    assert 'private' not in result.text
    model.calls.clear()
    provider=TurnOutcome('execute','failed',error=ClientError('PROVIDER_RATE_LIMITED','Provider failed'))
    result=summarize(sessions,'s','q',provider,model=model)
    assert not model.calls and result.error==provider.error


def test_reviewed_answer_and_exact_result_edit_do_not_generate_again():
    sessions,interaction=outcome_session();model=OutcomeModel()
    reviewed=TurnOutcome('answer','answered',text='已有回答',scientific=object())
    assert summarize(sessions,'s','q',reviewed,model=model) is reviewed
    interaction.admitted={'result_edit':True}
    assert summarize(sessions,'s','q',TurnOutcome('execute','activated'),model=model).text=='Analysis accepted.'
    assert not model.calls


@pytest.mark.parametrize('tool,function,parameters', [
    ('run_snapatac2','tl.leiden',{'resolution':0.8,'random_state':7}),
    ('run_snapatac2','tl.spectral',{'n_comps':17}),
    ('run_scanpy','pp.scale',{'zero_center':False,'max_value':3}),
    ('run_snapatac2','pp.filter_cells',{'min_counts':45,'min_tsse':2.5}),
    ('compute_motif_accessibility',None,{'niterations':50,'p_cutoff':0.00005}),
])
def test_native_effective_parameters_and_exact_repeated_steps(app,monkeypatch,tool,function,parameters):
    rid,_,_=accepted(app,tool,overrides={'function_name':function,'parameters':parameters},copies=2)
    app=forbid_work(app,monkeypatch)
    model=Model(question('r1'),['parameters'])
    result=ask(app,'q','这个步骤实际使用了什么参数？',model)
    assert result.status=='answered',result
    assert result.scientific.claims[0].value==parameters
    assert result.scientific.claims[0].source['output_locator']['step_id']=='step1'
    state=app.sessions.load('session');interaction=state.interactions[-1]
    public,_=sd.context(app.sessions,'session',interaction.admitted)
    target=public['targets'][0]
    assert target['identity']['revision_id']==rid and target['identity']['step_id']=='step1'
    assert [s['step_id'] for s in target['execution']['steps']]==['step0','step1']
    assert all(s['attempt_count']==1 for s in target['execution']['steps'])


def test_response_language_context_stops_at_captured_turn():
    state=N(interactions=[N(turn_id=str(i),utterance=t) for i,t in enumerate(['请分析','用这个','Please answer in English.','194','后来的请求'])])
    assert conversation_utterances(state,'3')==['请分析','用这个','Please answer in English.','194']


def test_native_method_comparison_uses_shared_field_meanings(app,monkeypatch):
    old,_,_=accepted(app,'run_snapatac2',overrides={'function_name':'tl.leiden','parameters':{'resolution':0.8,'random_state':7}})
    new,_,_=accepted(app,'run_snapatac2',name='edited',overrides={'function_name':'tl.leiden','parameters':{'resolution':1.1,'random_state':7}})
    app=forbid_work(app,monkeypatch)
    model=Model(question('r1',comparison={'output':'r0','subject':None}),['parameters'])
    result=ask(app,'compare','比较这两个已接受结果的参数。',model)
    assert result.status=='answered'
    interaction=app.sessions.load('session').interactions[-1]
    public,claims=sd.context(app.sessions,'session',interaction.admitted)
    assert public['comparison']['mode']=='descriptive_accepted_operands'
    pairs=public['comparison']['pairs'];by_id={c.claim_id:c for c in claims}
    pair=next(p for p in pairs if by_id[p['left']].field=='parameters')
    assert {by_id[pair[k]].source['revision_id'] for k in ('left','right')}=={old,new}
    assert {by_id[pair[k]].value['resolution'] for k in ('left','right')}=={0.8,1.1}


def test_web_boundary_explains_recorded_failure_without_new_execution(tmp_path):
    from test_structured_interaction import setup
    app, executor, model, inputs = setup(tmp_path)
    normal = model.complete
    language = OutcomeModel('本次执行失败，错误代码为 TEST_SCIENCE_WITHHELD；没有接受新的结果。')
    def complete(*,prompt,response_schema):
        data=json.loads(prompt)
        if data.get('focus')=='outcome' or 'grounding_review_version' in data:
            return language.complete(prompt=prompt,response_schema=response_schema)
        return normal(prompt=prompt,response_schema=response_schema)
    model.complete=complete
    result=app.submit_turn('session','failed','请分析这个数据。',expected_generation=0,execution_inputs=inputs)
    assert result.status=='failed' and result.error.code=='TEST_SCIENCE_WITHHELD'
    assert result.response.text==language.text and len(executor.plans)==1
    assert len(language.calls)==2
    facts=language.calls[0]['evidence']['outcome']
    assert facts['error']['code']=='TEST_SCIENCE_WITHHELD' and facts['accepted_revision'] is None
    assert facts['science_started'] is False  # This fixture withholds native science.
    before=app._application.sessions.load('session')
    assert app.submit_turn('session','failed','请分析这个数据。',expected_generation=0,execution_inputs=inputs)==result
    assert app._application.sessions.load('session')==before and len(language.calls)==2
