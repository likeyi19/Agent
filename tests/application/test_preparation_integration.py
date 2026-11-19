"""R2 production preparation boundaries; scripted provider, withheld science."""
import json
from pathlib import Path

import pytest

from agent.application import InteractiveAgentApplication, ResearchAgentApplication
from agent.providers import PlanningModelFactoryRegistry
from test_structured_interaction import setup, PROFILE


def reopen(app, executor, model):
    return InteractiveAgentApplication(app._application.workspace_root, registry=app._application.registry,
        executor=executor, model_profiles=(PROFILE,), default_profile_id=PROFILE.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _:model}))


def submit_member(app, question, name, member, value):
    choice = next(c for c in question.clarification_choices if c.get('member') == member)
    return app.submit_turn('session', name, '', expected_generation=0,
        clarification_choice=dict(question_turn_id=question.turn_id, handle=choice['handle']),
        parameter_overrides=[dict(tool='run_scanpy', native_function='pp.scale', argument='kwargs',
            step_id='scale', value={member:value})])


@pytest.mark.parametrize('textual', [False, True])
def test_independent_members_keep_remaining_question_after_restart(tmp_path, textual):
    app, executor, model, inputs = setup(tmp_path, missing=tuple(dict(tool='run_scanpy',
        native_function='pp.scale', argument='kwargs', member=m) for m in ('zero_center','max_value')))
    first = app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    before = app._application.sessions.load('session')
    assert first.status == 'clarification' and len(first.clarification_choices) == 2
    app = reopen(app, executor, model)
    if textual:
        complete = model.complete
        def answer(**kwargs):
            if 'turn_schema_version' in json.loads(kwargs['prompt']):
                model.calls.append(json.loads(kwargs['prompt']))
                return json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_parameters',
                    pending='@parameters', arguments=[dict(tool='run_scanpy',native_function='pp.scale',
                        argument='kwargs',members=[dict(name='max_value',kind='number',literal='4',evidence='max_value=4')])])) )
            return complete(**kwargs)
        model.complete = answer
        second = app.submit_turn('session','partial','Use max_value=4.',expected_generation=0)
    else:
        second = submit_member(app,first,'partial','max_value',4)
    assert second.status == 'clarification', second.to_dict()
    assert [c['member'] for c in second.clarification_choices] == ['zero_center']
    assert not executor.plans and not app._application.sessions.load('session').turns
    final = submit_member(app,second,'complete','zero_center',False)
    assert final.error.code == 'TEST_SCIENCE_WITHHELD', final.to_dict()
    assert len(executor.plans) == 1
    assert dict(executor.plans[0].steps[0].arguments['kwargs']) == {'max_value':4,'zero_center':False}
    state = app._application.sessions.load('session')
    assert state.interactions[0] == before.interactions[0]
    assert len({i.preparation['task_id'] for i in state.interactions}) == 1
    assert sum('candidate' in i.preparation for i in state.interactions) == 1
    assert sum('semantic_prompt_version' in c for c in model.calls) == 1
    assert sum('turn_schema_version' in c for c in model.calls) == (2 if textual else 1)
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session') == state


@pytest.mark.parametrize('when', ['before_answer','output_selection'])
@pytest.mark.parametrize('change_kind', ['counts','feature_axis'])
def test_direct_source_pin_blocks_changed_bytes_before_reservation(tmp_path, when, change_kind):
    import anndata as ad
    app, executor, model, inputs = setup(tmp_path,missing=(dict(tool='run_scanpy',
        native_function='pp.scale',argument='kwargs',member='max_value'),))
    first = app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    original = app._application.sessions.load('session').interactions[0]
    def change():
        matrix=ad.read_h5ad(inputs['input_path'])
        if change_kind == 'counts':
            matrix.X=matrix.X*2
        else:
            matrix.var_names=[f'changed-feature-{i}' for i in range(matrix.n_vars)]
        matrix.write_h5ad(inputs['input_path'])
    if when == 'before_answer':
        change()
    else:
        complete=model.complete
        def during_selection(**kwargs):
            if 'output_selection_schema_version' in json.loads(kwargs['prompt']):change()
            return complete(**kwargs)
        model.complete=during_selection
    result=submit_member(app,first,'answer','max_value',4)
    assert result.status == 'failed' and result.error.code == 'LOCAL_RESOURCE_INTEGRITY_INVALID',result.to_dict()
    state=app._application.sessions.load('session')
    assert not state.turns and not state.revisions and not executor.plans
    assert state.interactions[0] == original
    assert sum('semantic_prompt_version' in c for c in model.calls) == 1
    runs=list((Path(app._application.workspace_root)/'run_state').glob('*.json'))
    if when == 'before_answer':
        assert not runs
    else:
        # An actual post-provider integrity failure retains Runtime failure evidence.
        assert len(runs)==1
        saved=json.loads(runs[0].read_text())
        assert not saved.get('record',saved)['steps']


def test_resume_keeps_original_numeric_grounding(tmp_path):
    text='Scale with max_value=9.'
    declaration=dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',
        members=[dict(name='max_value',kind='number',literal='9',evidence='max_value=9')])
    app,executor,model,inputs=setup(tmp_path,declaration=declaration,
        missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='count_slot'),))
    first=app.submit_turn('session','task',text,expected_generation=0,execution_inputs=inputs)
    original=app._application.sessions.load('session').interactions[0]
    calls=len(model.calls)
    result=reopen(app,executor,model).submit_turn('session','resume','',expected_generation=0,
        resume_task_id=first.preparation_task_id)
    assert result.status == 'clarification',result.to_dict()
    assert len(model.calls)==calls and not executor.plans
    assert app._application.sessions.load('session').interactions[0]==original


@pytest.mark.parametrize("changed", [False, True])
def test_resume_recaptures_added_reference_without_replanning(tmp_path, monkeypatch, changed):
 from types import SimpleNamespace
 import hashlib
 from agent.application.input_sets import ScientificInputSet
 from agent.application.raw_handoff import AnnotationDeclaration, BamDeclaration, declaration_dict
 from agent.orchestration import PlanningModelProfile, build_default_tool_registry
 from test_human_epizoo_mapping import WithholdScience
 from test_native_candidate_continuation import _proposal

 root=tmp_path
 from agent.tools.analysis import sequence_prediction
 from agent.tools.data import cellmarker
 monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
 monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
 calls=[]
 class Model:
  model_id='review-reference-catalog'
  def complete(self,*,prompt,response_schema):
   value=json.loads(prompt);calls.append(value)
   if 'turn_schema_version' in value:
    return json.dumps(dict(turn_schema_version=1,decision=dict(kind='request_roles',target='run_snapatac2',required_roles=[],roles=[],bam=declaration_dict(BamDeclaration(source='source.0',assay='SCATAC',species='Homo sapiens',assembly='hg19')),annotation=declaration_dict(AnnotationDeclaration(species='Homo sapiens')))))
   if 'semantic_prompt_version' in value:return json.dumps(_proposal())
   raise AssertionError(list(value))
 registry=build_default_tool_registry();executor=WithholdScience(registry)
 app=InteractiveAgentApplication(root/'workspace',registry=registry,executor=executor,model_profiles=(PlanningModelProfile('offline','scripted',Model.model_id),),default_profile_id='offline',approved_source_roots=(root,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:Model()}))
 app.create_session('session');bam=root/'input.bam';bam.write_bytes(b'BAM\x01')
 source=app.resources.register('bam',bam,input_type='attachment',session_id='session',label='BAM',attribution='Review fixture')
 group=app.resources.compose_group([source.resource_id],attachment_candidates=True)
 def ref(name,**ports):
  values=dict(species='human',assembly='hg19')
  for port in ports:values.update({port+'_path':'/refs/'+name+'/'+port,port+'_sha256':hashlib.sha256(name.encode()).hexdigest()})
  return ScientificInputSet(name,name,values)
 refs=(ref('genome',chrom_sizes=True,reference=True),ref('other-genome',chrom_sizes=True,reference=True),ref('genes',gene_anno=True),ref('other-genes',gene_anno=True),ref('peaks',peaks=True))
 initial=app.submit_turn('session','task','Analyze and annotate this BAM.',expected_generation=0,registered_input=group,input_sets=refs)
 first_identity=app._application.sessions.load('session').interactions[0]
 added=(*refs,ref('new-genome',chrom_sizes=True,reference=True))
 if changed:
  from dataclasses import replace
  added=(replace(refs[0],label='Changed existing reference'),*refs[1:])
 resumed=app.submit_turn('session','resume','',expected_generation=0,resume_task_id=initial.preparation_task_id,input_sets=added)
 state=app._application.sessions.load('session')

 if changed:
  assert resumed.status=='failed' and resumed.error.code=='PREPARATION_RESOURCE_CHANGED',resumed.to_dict()
  assert state.interactions[-1].admitted['preparation_result']['status']=='INVALID'
 else:
  assert resumed.status=='clarification',resumed.to_dict()
  assert any(c.get('choice_identity')=='resource.new-genome' for c in resumed.clarification_choices)
 assert first_identity==state.interactions[0]
 assert initial.preparation_task_id==resumed.preparation_task_id
 assert sum('semantic_prompt_version' in c for c in calls)==1
 assert sum('turn_schema_version' in c for c in calls)==1
 assert not state.turns and not executor.plans
 assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state


@pytest.mark.parametrize('reserved', [False, True])
def test_interrupted_ready_reservation_requires_explicit_recovery_and_never_reexecutes(tmp_path,monkeypatch,reserved):
    from agent.application.session_store import FileSessionStore
    class StopProcess(BaseException):pass
    app,executor,model,inputs=setup(tmp_path,missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value'),))
    first=app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    before=app._application.sessions.load('session').interactions[0]
    update=FileSessionStore._update
    def stop(self,session_id,change):
        if change.__name__=='link':
            if reserved:update(self,session_id,change)
            raise StopProcess()
        return update(self,session_id,change)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(FileSessionStore,'_update',stop)
        with pytest.raises(StopProcess):submit_member(app,first,'answer','max_value',4)
    fresh=reopen(app,executor,model)
    state=fresh._application.sessions.load('session')
    assert state.interactions[0]==before
    assert len(state.turns)==int(reserved) and not executor.plans and not state.revisions
    assert state.interactions[-1].admitted['preparation_result']['status']=='READY'
    calls=len(model.calls)
    assert fresh.status('session','answer').status=='interrupted'
    assert submit_member(fresh,first,'answer','max_value',4).status=='interrupted'
    recovered=fresh.recover_turn('session','answer')
    if reserved:
        assert recovered.response is None
        assert fresh.status('session','answer').status=='interrupted'
    else:
        assert recovered.response is not None
    turns=fresh._application.sessions.load('session').turns
    if reserved:
        result=fresh.submit_turn('session','resume','',expected_generation=0,resume_task_id=first.preparation_task_id)
        assert result.status=='failed' and result.error.code=='INTERACTIVE_TURN_CONFLICT',result.to_dict()
    else:
        result=fresh.submit_turn('session','resume','',expected_generation=0,resume_task_id=first.preparation_task_id)
        assert result.status=='answered',result.to_dict()
        assert fresh._application.sessions.load('session').interactions[-1].admitted['preparation_result']['status']=='READY'
    assert fresh._application.sessions.load('session').turns==turns
    assert not executor.plans and len(model.calls)==calls


def test_http_competing_answers_exact_replay_and_restart(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from agent.web.app import create_app
    from helpers import wait_turn
    app,executor,model,inputs=setup(tmp_path,missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value'),))
    first=app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    original=app._application.sessions.load('session').interactions[0]
    choice,=first.clarification_choices
    body=dict(turn_id='answer',utterance='',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=choice['handle']),
        parameter_overrides=[dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',step_id='scale',value={'max_value':4})])
    with TestClient(create_app(reopen(app,executor,model))) as client:
        with ThreadPoolExecutor(2) as threads:
            responses=list(threads.map(lambda _:client.post('/api/v1/sessions/session/turns',json=body),range(2)))
        assert all(r.status_code==202 for r in responses),[r.text for r in responses]
        final=wait_turn(client,'answer')
        assert final['error']['code']=='TEST_SCIENCE_WITHHELD',final
    with TestClient(create_app(reopen(app,executor,model))) as client:
        assert client.get('/api/v1/sessions/session/turns/answer').json()==final
        replay=client.post('/api/v1/sessions/session/turns',json=body)
        assert replay.status_code==202,replay.text
        assert wait_turn(client,'answer')==final
        conflicting=dict(body,parameter_overrides=[dict(body['parameter_overrides'][0],value={'max_value':8})])
        assert client.post('/api/v1/sessions/session/turns',json=conflicting).status_code==409
    state=app._application.sessions.load('session')
    assert state.interactions[0]==original and len(state.turns)==len(executor.plans)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    assert sum('turn_schema_version' in c for c in model.calls)==1


def test_secondary_direct_source_is_pinned_across_parameter_answers(tmp_path):
    import anndata as ad
    import numpy as np
    app,executor,model,inputs=setup(tmp_path,missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value'),))
    other=tmp_path/'query.h5ad';ad.AnnData(np.eye(4)).write_h5ad(other)
    inputs['query_input_path']=str(other)
    complete=model.complete
    def branches(**kwargs):
        if 'semantic_prompt_version' in json.loads(kwargs['prompt']):
            model.calls.append(json.loads(kwargs['prompt']))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[
                dict(step_id='left',tool='inspect_scATAC',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[]),
                dict(step_id='right',tool='inspect_scATAC',sources=[dict(target='dataset',source=dict(kind='input',input='query_input_path'))],control_dependencies=[]),
                dict(step_id='scale',tool='run_scanpy',sources=[dict(target='dataset',source=dict(kind='step_port',step='right',source_port='dataset')),
                    dict(target='function_name',source=dict(kind='registered_choice',choice='pp.scale'))],control_dependencies=[])])))
        return complete(**kwargs)
    model.complete=branches
    first=app.submit_turn('session','task','Inspect both sources and scale the query.',expected_generation=0,execution_inputs=inputs)
    assert first.status=='clarification',first.to_dict()
    matrix=ad.read_h5ad(other);matrix.X*=2;matrix.write_h5ad(other)
    final=submit_member(app,first,'answer','max_value',4)
    assert final.status=='failed' and final.error.code=='LOCAL_RESOURCE_INTEGRITY_INVALID',final.to_dict()
    assert not executor.plans and not app._application.sessions.load('session').turns


@pytest.mark.parametrize('correction,declared',[('Source A uses hg19.',dict(assembly='hg19')),
    ('Source A is Mouse.',dict(species='Mouse'))])
def test_public_two_source_annotation_changes_preserve_independent_answers(tmp_path,monkeypatch,correction,declared):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import hashlib
    from types import SimpleNamespace as N
    from agent.tools.analysis import annotation_input,sequence_prediction
    from agent.tools.data import cellmarker
    from agent.application.input_sets import ScientificInputSet
    from agent.application.raw_handoff import AnnotationDeclaration,declaration_dict
    from agent.application import preparation_state as records
    from agent.orchestration import build_default_tool_registry
    from test_human_epizoo_mapping import WithholdScience
    monkeypatch.setattr(annotation_input,'observe',annotation_input._observe)
    monkeypatch.setattr(sequence_prediction,'library',lambda:N(entries=lambda:[dict(kind='cellmarker',id='fixture')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:N(identity={},metadata={'species':['Human','Mouse']}))
    calls=[]
    class Model:
        model_id=PROFILE.model_id
        def complete(self,*,prompt,response_schema):
            p=json.loads(prompt);calls.append(p)
            if 'turn_schema_version' in p:
                if p['utterance']==correction:
                    d=dict(kind='answer_roles',pending='@roles',roles=[],annotation=declaration_dict(AnnotationDeclaration(**declared)))
                else:
                    d=dict(kind='request_roles',target='annotate_scATAC_cell_types',required_roles=[],roles=[],
                        annotation=declaration_dict(AnnotationDeclaration(species='Human',assembly='hg38',generate_umap=False)))
                return json.dumps(dict(turn_schema_version=1,decision=d))
            assert 'semantic_prompt_version' in p,list(p)
            steps=[]
            for side,key in [('left','input_path'),('right','query_input_path')]:
                steps += [dict(step_id=side,tool='inspect_scATAC',control_dependencies=[],sources=[dict(target='dataset',source=dict(kind='input',input=key))]),
                    dict(step_id=side+'_annotation',tool='annotate_scATAC_cell_types',control_dependencies=[],sources=[
                        dict(target='source',source=dict(kind='step_port',step=side,source_port='dataset')),
                        dict(target='group_column',source=dict(kind='input',input='group_column')),
                        dict(target='activity_method',source=dict(kind='registered_choice',choice='episcanpy.tl.geneactivity'))])]
            # An independently missing native function keeps preparation open
            # while both annotation resource answers are exercised.
            steps.append(dict(step_id='followup',tool='run_scanpy',control_dependencies=[],
                sources=[dict(target='dataset',source=dict(kind='step_port',step='right',source_port='dataset'))]))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
    registry=build_default_tool_registry();executor=WithholdScience(registry);model=Model()
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        model_profiles=(PROFILE,),default_profile_id=PROFILE.profile_id,approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session');sources=[];paths=[]
    for name in ('left','right'):
        path=tmp_path/(name+'.h5ad');ad.AnnData(np.eye(3),obs=pd.DataFrame({'groups':['a','a','b']}),uns={'assembly':'mm10'} if name=='right' else {}).write_h5ad(path)
        paths.append(path);sources.append(app.resources.register(name,path,input_type='h5ad',session_id='session',label=name,attribution='Fixture').resource_id)
    resources=[]
    for name,species,assembly in [('human1','human','hg38'),('human2','human','hg38'),('oldhuman','human','hg19'),('mouse','mouse','mm10'),('mouse2','mouse','mm10')]:
        path=tmp_path/(name+'.gtf');path.write_text(name)
        resources.append(ScientificInputSet(name,name,dict(species=species,assembly=assembly,
            gene_anno_path=str(path),gene_anno_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    references=tuple(resources)
    first=app.submit_turn('session','task','Annotate these two sources.',expected_generation=0,
        execution_inputs=dict(input_path=str(paths[0]),
            query_input_path=str(paths[1]),group_column='groups'),input_sets=references)
    assert first.status=='clarification',first.to_dict()
    origin=app._application.sessions.load('session').interactions[0]
    question=first
    for sid,identity in [('left_annotation','resource.human1'),('right_annotation','resource.mouse')]:
        choice=next(c for c in question.clarification_choices if c['step_id']==sid and c.get('choice_identity')==identity)
        question=app.submit_turn('session',sid,'',expected_generation=0,input_sets=references,
            clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle']))
        assert question.status=='clarification',question.to_dict()
    before=app._application.sessions.load('session')
    result=app.submit_turn('session','correction',correction,expected_generation=0,input_sets=references)
    assert result.status=='clarification',result.to_dict()
    state=app._application.sessions.load('session');record=state.interactions[-1].preparation
    active=records.resource_bindings(state.interactions,record)
    assert 'left_annotation' not in active
    assert active['right_annotation']['gene_anno_path']['handle']=='resource.mouse'
    assert state.interactions[:len(before.interactions)]==before.interactions and state.interactions[0]==origin
    assert len({i.preparation['task_id'] for i in state.interactions})==1
    assert sum('candidate' in i.preparation for i in state.interactions)==1
    assert sum('semantic_prompt_version' in c for c in calls)==1
    assert not state.turns and not executor.plans
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state


@pytest.mark.parametrize('reduce_before_competition',[False,True])
def test_http_distinct_competing_answers_reserve_only_one_plan(tmp_path, monkeypatch, reduce_before_competition):
    from threading import Event
    from agent.application import scientific_preparation
    from fastapi.testclient import TestClient
    from agent.web.app import create_app
    from helpers import wait_turn
    app,executor,model,inputs=setup(tmp_path,missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value'),))
    first=app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    choice,=first.clarification_choices
    captured,finished=Event(),Event()
    typed=scientific_preparation.typed_answer
    def gated(sessions,interaction):
        if interaction.turn_id=='answer_a':
            prepared = typed(sessions,interaction) if reduce_before_competition else None
            captured.set()
            assert finished.wait(10)
            if reduce_before_competition:
                return prepared
        return typed(sessions,interaction)
    monkeypatch.setattr(scientific_preparation,'typed_answer',gated)
    execute=executor.execute
    def completed(*args,**kwargs):
        result=execute(*args,**kwargs)
        finished.set()
        return result
    monkeypatch.setattr(executor,'execute',completed)
    bodies=[dict(turn_id=turn,utterance='',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=choice['handle']),
        parameter_overrides=[dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',step_id='scale',value={'max_value':value})])
        for turn,value in [('answer_a',4),('answer_b',8)]]
    with TestClient(create_app(reopen(app,executor,model))) as client:
        responses=[client.post('/api/v1/sessions/session/turns',json=bodies[0])]
        assert captured.wait(10)
        responses.append(client.post('/api/v1/sessions/session/turns',json=bodies[1]))
        assert all(r.status_code==202 for r in responses),[r.text for r in responses]
        accepted=[body for body,r in zip(bodies,responses) if r.status_code==202]
        assert accepted
        results=[wait_turn(client,body['turn_id']) for body in accepted]
        assert [r['error']['code'] for r in results]==['INTERACTIVE_TURN_CONFLICT','TEST_SCIENCE_WITHHELD'],results
        assert client.post('/api/v1/sessions/session/turns',json=bodies[0]).status_code==202
        assert wait_turn(client,'answer_a')==results[0]
    state=app._application.sessions.load('session')
    assert len(executor.plans)==len(state.turns)==1 and not state.revisions
    assert executor.plans[0].steps[0].arguments['kwargs']['max_value']==8
    assert next(i for i in state.interactions if i.turn_id=='answer_a').preparation is None
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert ResearchAgentApplication(app._application.workspace_root).sessions.load('session')==state


@pytest.mark.parametrize('missing_at_capture',[None,0,1,2,3,4,5])
def test_direct_multilane_sources_pin_each_consumed_file(tmp_path,missing_at_capture):
    app,executor,model,_=setup(tmp_path,missing=(dict(tool='run_chromap',native_function='map',argument='kwargs',member='num-threads'),))
    inputs={}
    for key in ('reference_path','index_path','read1_path','read2_path'):
        paths=[]
        for lane in range(2 if key in ('read1_path','read2_path') else 1):
            path=tmp_path/f'{key}-{lane}.fixture';path.write_text('Opaque file identity; science is withheld.')
            paths.append(str(path))
        inputs[key]=paths if key in ('read1_path','read2_path') else paths[0]
    all_paths=[inputs['reference_path'],inputs['index_path'],*inputs['read1_path'],*inputs['read2_path']]
    absent=None if missing_at_capture is None else Path(all_paths[missing_at_capture])
    absent_bytes=None if absent is None else absent.read_bytes()
    complete=model.complete
    def candidate(**kwargs):
        prompt=json.loads(kwargs['prompt'])
        if 'turn_schema_version' in prompt:
            value=json.loads(complete(**kwargs));value['decision']['target']='run_chromap'
            return json.dumps(value)
        if 'semantic_prompt_version' in prompt:
            model.calls.append(prompt)
            if absent is not None:
                absent.unlink()
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(
                step_id='map',tool='run_chromap',control_dependencies=[],sources=[
                    dict(target=port,source=dict(kind='input',input=key)) for port,key in
                    [('reference','reference_path'),('index','index_path'),('read1','read1_path'),('read2','read2_path')]]+[
                    dict(target='function_name',source=dict(kind='registered_choice',choice='map'))])])) )
        return complete(**kwargs)
    model.complete=candidate
    first=app.submit_turn('session','task','Map these explicitly ordered lanes with the supplied reference; ask for threads.',
        expected_generation=0,execution_inputs=inputs)
    state=app._application.sessions.load('session')
    pins=state.interactions[-1].preparation['source_pins']
    assert {p['source_path'] for p in pins}==set(all_paths)
    if absent is not None:
        assert first.status=='failed' and first.error.code=='LOCAL_RESOURCE_INTEGRITY_INVALID',first.to_dict()
        assert [p['source_path'] for p in pins if 'source_sha256' not in p]==[str(absent)]
        absent.write_bytes(absent_bytes)
        reopened=ResearchAgentApplication(app._application.workspace_root).sessions.load('session')
        assert reopened==state
        resumed=app.submit_turn('session','resume','',expected_generation=0,
            resume_task_id=state.interactions[-1].preparation['task_id'])
        assert resumed.status=='failed' and resumed.error.code=='PREPARATION_SOURCE_CHANGED',resumed.to_dict()
        assert not executor.plans and not app._application.sessions.load('session').turns
        assert app._application.sessions.load('session').interactions[0]==state.interactions[0]
        assert sum('semantic_prompt_version' in c for c in model.calls)==1
        return
    assert first.status=='clarification',first.to_dict()
    Path(inputs['read2_path'][1]).write_text('Changed second read lane')
    choice,=first.clarification_choices
    result=app.submit_turn('session','answer','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=choice['handle']),
        parameter_overrides=[dict(tool='run_chromap',native_function='map',argument='kwargs',step_id='map',value={'num-threads':2})])
    assert result.status=='failed' and result.error.code=='LOCAL_RESOURCE_INTEGRITY_INVALID',result.to_dict()
    assert not executor.plans and not app._application.sessions.load('session').turns
    assert sum('semantic_prompt_version' in c for c in model.calls)==1


@pytest.mark.parametrize('mutation',[None,'planned','resolved','numeric_type'])
def test_canonical_answer_parameter_evidence_checks_exact_accepted_values(tmp_path,mutation):
    """Real answer ledger; controlled Run projection, with native science withheld."""
    from types import SimpleNamespace
    from dataclasses import replace
    from agent.application.dialogue_evidence import _explicit_user_parameter_facts, _Unavailable
    app,executor,model,inputs=setup(tmp_path,missing=(dict(tool='run_scanpy',native_function='pp.scale',argument='kwargs',member='max_value'),))
    first=app.submit_turn('session','task','Scale these counts.',expected_generation=0,execution_inputs=inputs)
    result=submit_member(app,first,'answer','max_value',4)
    assert result.error.code=='TEST_SCIENCE_WITHHELD'
    state=app._application.sessions.load('session')
    plan=executor.plans[0];planned=plan.steps[0]
    step=SimpleNamespace(tool_name=planned.tool_name,step_id=planned.step_id,resolved_arguments=dict(planned.arguments))
    run=SimpleNamespace(run_id=state.turns[-1].run_id,request_id=plan.request_id,plan=plan)
    calls=len(model.calls)
    if mutation=='planned':
        planned=replace(planned,arguments={**planned.arguments,'kwargs':{'max_value':5}})
    elif mutation in ('resolved','numeric_type'):
        step.resolved_arguments={**step.resolved_arguments,'kwargs':{'max_value':5 if mutation=='resolved' else 4.0}}
    if mutation:
        with pytest.raises(_Unavailable,match='session_parameter_binding_mismatch'):
            _explicit_user_parameter_facts(app._application.sessions,state,run,step,planned)
    else:
        fact,=_explicit_user_parameter_facts(app._application.sessions,state,run,step,planned)
        assert fact.value['parameters']=={'kwargs':'explicit_user_instruction'}
        assert fact.reason=='accepted_structured_session_declarations'
    assert len(model.calls)==calls and app._application.sessions.load('session')==state
