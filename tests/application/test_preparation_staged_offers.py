"""Exact answers may retransmit only their question's neutral staged catalog.

Genuine B2 shape: a treatment was selected earlier; a later upload/recheck
offers only a new control. Web keeps that control staged on the exact click.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from agent.application import ResearchAgentApplication, preparation_state as owner
from agent.application.session_state import Interaction, SessionConflictError
from agent.schemas.orchestration import _serialize
from test_preparation_owner_contracts import Model, setup


def pending(tmp_path, *, extra=False):
    model=Model();app,executor=setup(tmp_path,model)
    resources={}
    # Deliberately equal bytes: source identities must not collapse by hash.
    for name in ('treatment','control','unused') if extra else ('treatment','control'):
        path=tmp_path/(name+'.bedgraph');path.write_text('chr1\t0\t10\t1\n')
        resources[name]=app.resources.register(name,path,input_type='attachment',label=name,
            session_id='session',attribution='Independent controlled source identity')
    first=app.submit_turn('session','task','Compare these tracks.',expected_generation=0,
        registered_input=app.resources.compose_group([resources['treatment'].resource_id],attachment_candidates=True))
    assert first.status=='clarification',first.to_dict()
    staged=[resources['control'].resource_id]
    if extra:staged.append(resources['unused'].resource_id)
    checked=app.submit_turn('session','recheck','',expected_generation=0,
        resume_task_id=first.preparation_task_id,attachment_ids=staged)
    assert checked.status=='clarification',checked.to_dict()
    assert not executor.plans and not app._application.sessions.load('session').turns
    return app,model,executor,resources,checked,staged


@pytest.mark.parametrize('extra',[False,True])
def test_web_shaped_staged_answer_keeps_exact_original_and_new_source(tmp_path,extra):
    app,model,executor,resources,checked,staged=pending(tmp_path,extra=extra)
    before=app._application.sessions.load('session')
    question=before.interactions[-1]
    offered=question.snapshot['dialogue']['source_view']['registered_input']
    assert offered['attachment_candidates'] and not offered['roles']
    assert offered['members'][0]['resource_id']==resources['control'].resource_id
    assert question.admitted['registered_input']['members'][0]['resource_id']==resources['treatment'].resource_id
    choice=next(c for c in checked.clarification_choices if c.get('source_input') and c['label']=='control')
    calls=len(model.calls)
    result=app.submit_turn('session','answer','',expected_generation=0,attachment_ids=staged,
        clarification_choice=dict(question_turn_id='recheck',handle=choice['handle']))
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    state=app._application.sessions.load('session')
    assert state.interactions[:-1]==before.interactions
    assert state.interactions[-1].submission['registered_input']==offered
    answer=state.interactions[-1].preparation['answers'][0]
    assert answer['value']['component']['member']['resource_id']==resources['control'].resource_id
    assert len({i.preparation['candidate_id'] for i in state.interactions})==1
    assert len({i.preparation['task_id'] for i in state.interactions})==1
    args=executor.plans[0].steps[0].arguments
    assert args['input_path']==resources['treatment'].source_path
    assert args['control_path']==resources['control'].source_path
    assert resources['control'].source_sha256==resources['treatment'].source_sha256
    assert resources['control'].source_path!=resources['treatment'].source_path
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    assert all('output_selection_schema_version' in c for c in model.calls[calls:])
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state


@pytest.fixture(scope='module')
def staged_question(tmp_path_factory):
    app,model,executor,resources,checked,staged=pending(tmp_path_factory.mktemp('staged-offer'),extra=True)
    state=app._application.sessions.load('session');question=state.interactions[-1]
    choice=next(c for c in owner.choices(question) if c.get('source_input') and c['label']=='control')
    requirement=next(r for r in owner.observation(question)['requirements'] if r['requirement_id']==choice['requirement_id'])
    values={requirement['requirement_id']:_serialize(requirement['offers'][0])}
    submission=dict(clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle']),
        registered_input=_serialize(question.snapshot['dialogue']['source_view']['registered_input']),execution_inputs={})
    answer=Interaction('answer','',None,0,{},submission=submission)
    return state,question,answer,values


@pytest.mark.parametrize('change',['member_path','member_id','member_record','member_sha','order','role',
    'resource_choice','new_member','input_value','unoffered_answer','field_value'])
def test_staged_offer_cannot_change_any_source_or_answer_identity(staged_question,change):
    state,question,answer,values=staged_question
    received=_serialize(answer.submission);group=received['registered_input']
    supplied={key:_serialize(value) for key,value in values.items()}
    if change=='member_path':group['members'][0]['source_path']='/foreign/control.bedgraph'
    elif change=='member_id':group['members'][0]['resource_id']='local-'+'f'*64
    elif change=='member_record':group['members'][0]['record_sha256']='f'*64
    elif change=='member_sha':group['members'][0]['source_sha256']='f'*64
    elif change=='order':group['members'].reverse()
    elif change=='role':group['roles']=[dict(tool='run_macs3',port='control',source='source.0')]
    elif change=='resource_choice':group['resource_choice']={'resource':'changed'}
    elif change=='new_member':
        added=dict(group['members'][0],resource_id='local-'+'f'*64,source_path='/foreign/extra')
        group['members'].append(added)
    elif change=='input_value':received['execution_inputs']={'input_path':'/foreign/treatment'}
    elif change=='unoffered_answer':next(iter(supplied.values()))['component']['path']='/foreign/control'
    elif change=='field_value':next(iter(supplied.values()))['value']='forged-field'
    forged=replace(answer,submission=received)
    with pytest.raises(SessionConflictError):
        owner.answer_question(state.interactions,forged,values=supplied)
    # Failed reductions never mutate the question or reserve execution.
    assert state.turns==state.revisions==()
    assert question.preparation['answers']==()


def test_only_neutral_question_catalog_is_an_extra_allowed_identity(staged_question):
    state,question,answer,values=staged_question
    for field,value in [('attachment_candidates',False),('roles',[dict(tool='run_macs3',port='control',source='source.0')]),
                        ('resource_choice',dict(resource='changed'))]:
        snapshot=_serialize(question.snapshot)
        snapshot['dialogue']['source_view']['registered_input'][field]=value
        changed_question=replace(question,snapshot=snapshot)
        received=_serialize(answer.submission)
        received['registered_input']=snapshot['dialogue']['source_view']['registered_input']
        with pytest.raises(SessionConflictError):
            owner.validate_exact_submission(state.interactions,replace(answer,submission=received),changed_question)
