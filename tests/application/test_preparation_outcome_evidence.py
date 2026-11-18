"""Canonical UI choices and active answers share the grounded response evidence."""
from dataclasses import replace
import json

from agent.application import preparation_state as records
from agent.application.scientific_dialogue import _preparation_outcome, explain_outcome
from agent.application.session_state import Interaction
from agent.application.turn_decisions import Clarify
from agent.application.turns import TurnOutcome
from agent.orchestration.semantic_compiler import ScientificRequirement
from test_preparation_source_contracts import app_fixture
from test_response_contracts import OutcomeModel


def outcome_prompts(model):
    return [c for c in model.calls if c.get('focus')=='outcome' or 'grounding_review_version' in c]


def test_existing_field_offers_and_selected_answer_reach_generation_and_review(tmp_path,monkeypatch):
    app,model,executor,source,_=app_fixture(tmp_path,monkeypatch)
    first=app.submit_turn('session','task','请注释人类数据。',expected_generation=0,
        registered_input=app.resources.compose_group([source.resource_id],attachment_candidates=True))
    human=next(c for c in first.clarification_choices if c.get('value')=='Human')
    second=app.submit_turn('session','human','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=human['handle']))
    fields=[c for c in second.clarification_choices if c.get('source_field')]
    assert len(fields)==2,second.to_dict()
    language=OutcomeModel()
    state=app._application.sessions.load('session')
    explain_outcome(app._application.sessions,'session',state.interactions[-1],
        TurnOutcome('clarify','clarification',Clarify('missing_parameter_value',(),True)),language)
    pending=language.calls
    facts=pending[0]['evidence']['outcome']['preparation']
    assert pending[1]['evidence']['outcome']['preparation']==facts
    assert {c['label'] for r in facts['requirements'] for c in r['offered_choices']}=={
        'obs.cellranger_graphclust','obs.other_groups'}
    assert facts['resolved_inputs'][0]['label']=='Human'
    assert str(tmp_path) not in json.dumps(facts)
    choice=next(c for c in fields if c['value']=='other_groups')
    before=app._application.sessions.load('session')
    final=app.submit_turn('session','field','',expected_generation=0,
        clarification_choice=dict(question_turn_id='human',handle=choice['handle']))
    assert final.status=='clarification' and not executor.plans,final.to_dict()
    language=OutcomeModel()
    state=app._application.sessions.load('session')
    explain_outcome(app._application.sessions,'session',state.interactions[-1],
        TurnOutcome('clarify','clarification',Clarify('missing_parameter_value',(),True)),language)
    pending=language.calls
    facts=pending[0]['evidence']['outcome']['preparation']
    assert pending[1]['evidence']['outcome']['preparation']==facts
    assert {r['argument'] for r in facts['requirements']}=={'gene_anno_path'}
    selected=next(r for r in facts['resolved_inputs'] if r['argument']=='group_column')
    assert selected['label']=='obs.other_groups'
    assert selected['step_id']=='annotation' and selected['tool_name']=='annotate_scATAC_cell_types'
    assert str(tmp_path) not in json.dumps(facts)
    assert app._application.sessions.load('session').interactions[:2]==before.interactions


def test_outcome_projection_reuses_answer_invalidation_and_keeps_independent_value():
    # Two genuinely independent controls with different source dependencies.
    reqs=[]
    for name in ('left','right'):
        req=ScientificRequirement('candidate',name,'run_scanpy','pp.scale','kwargs','kwargs','scalar',member='max_value',
            dependencies=(name+':declaration:assembly',)).to_dict()
        req['source_dependencies']=['/exact/'+name]
        reqs.append(req)
    origin=Interaction('origin','Analyze both sources.',None,0,{})
    record=records.start('session',origin,origin.utterance)
    origin=replace(origin,preparation=record,status='clarification',admitted=dict(preparation_result=records.question(record,origin,reqs)))
    chain=[origin]
    for index,req in enumerate(reqs):
        previous=chain[-1]
        option=next(c for c in records.choices(previous) if c['requirement_id']==req['requirement_id'])
        turn=Interaction('answer'+str(index),'',None,0,{},submission=dict(
            clarification_choice=dict(question_turn_id=previous.turn_id,handle=option['handle']),
            parameter_overrides=[dict(tool='run_scanpy',step_id=req['step_id'],argument='kwargs',value={'max_value':index+1})]))
        record=records.answer_question(chain,turn,values={req['requirement_id']:index+1})
        turn=replace(turn,preparation=record,admitted=dict(preparation_result=records.question(record,turn,reqs)),status='clarification')
        chain.append(turn)
    old=_preparation_outcome(chain,chain[-1],records.observation(chain[-1]))
    assert {(r['step_id'],r['value']) for r in old['resolved_inputs']}=={('left',1),('right',2)}
    correction=Interaction('correction','The left assembly is now hg38.',None,0,{})
    record=records.record_answers(chain[-1],correction,{},provenance='interpreted_source')
    record['declarations']=[dict(scope=dict(owner='annotation',sources=[dict(source_path='/exact/left')]),
        member='assembly',old_value='hg19',value='hg38')]
    correction=replace(correction,preparation=record,admitted=dict(preparation_result=records.question(record,correction,reqs)))
    current=_preparation_outcome(chain,correction,records.observation(correction))
    assert [(r['step_id'],r['value']) for r in current['resolved_inputs']]==[('right',2)]
    assert _preparation_outcome(chain,chain[-1],records.observation(chain[-1]))==old


def test_external_upload_controls_and_bounded_private_offers_share_projection():
    origin=Interaction('origin','Compare these tracks.',None,0,{})
    record=records.start('s',origin,origin.utterance)
    req=ScientificRequirement('candidate','compare','run_macs3','bdgcmp','control','control_path','external_input').to_dict()
    req['offers']=[dict(handle=str(i),label='control_'+str(i),component={'path':'/private/data','sha256':'a'*64}) for i in range(40)]
    origin=replace(origin,preparation=record,admitted=dict(preparation_result=records.question(record,origin,[req])))
    facts=_preparation_outcome([origin],origin,records.observation(origin))
    row=facts['requirements'][0]
    assert row['offered_choices'][0]==dict(label='Upload control_path',action='upload_source')
    assert len(row['offered_choices'])==32 and row['choices_omitted']==9
    assert '/private' not in json.dumps(facts) and 'sha256' not in json.dumps(facts)


def test_bound_annotation_attachment_uses_exact_group_identity_and_leaves_other_staged(tmp_path,monkeypatch):
    app,model,executor,source,_=app_fixture(tmp_path,monkeypatch)
    path=tmp_path/'unselected.h5ad';path.write_bytes(open(source.source_path,'rb').read())
    other=app.resources.register('unselected',path,input_type='h5ad',session_id='session',label='Unselected',attribution='Fixture')
    first=app.submit_turn('session','task','请注释人类数据。',expected_generation=0,
        registered_input=app.resources.compose_group([source.resource_id,other.resource_id],attachment_candidates=True))
    assert [a['use'] for a in first.attachments]==['offered_for_analysis','staged']
    state=app._application.sessions.load('session');turn=state.interactions[-1]
    from agent.schemas.orchestration import _serialize
    # Persisted history is untouched. The read-only view sees a repacked group
    # where source.0 names the other member, not declaration-group source.0.
    raw=_serialize(turn.admitted)
    raw['registered_input']['members']=list(reversed(raw['registered_input']['members']))
    from agent.application.interactive import _selected_attachment_sources
    assert _selected_attachment_sources(raw,raw['annotation_sources']['members'])=={'source.1'}
    assert app._application.sessions.load('session')==state and not executor.plans


def test_exact_scope_selection_is_visible_without_answer_tokens():
    origin=Interaction('origin','Adjust one selected operation.',None,0,{})
    record=records.start('s',origin,origin.utterance)
    req=ScientificRequirement('candidate','scope','run_scanpy',None,'kwargs','kwargs','scope',
        choices=('left','right')).to_dict()
    origin=replace(origin,preparation=record,status='clarification',
        admitted=dict(preparation_result=records.question(record,origin,[req])))
    choice=records.choices(origin)[1]
    turn=Interaction('selected','',None,0,{},submission=dict(
        clarification_choice=dict(question_turn_id='origin',handle=choice['handle'])))
    record=records.answer_question([origin],turn,values={req['requirement_id']:'right'})
    turn=replace(turn,preparation=record,admitted=dict(preparation_result=records.question(record,turn,[])))
    facts=_preparation_outcome([origin,turn],turn,records.observation(turn))
    assert facts['resolved_inputs'][0]['value']=='right'
    assert 'question_version' not in json.dumps(facts) and 'requirement_id' not in json.dumps(facts)
