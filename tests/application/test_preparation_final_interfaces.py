"""Genuine CP4 interpretation witnesses and achievable corrected interfaces.

Saved wires remain evidence, not language templates or production repair rules.
All models below are offline; execution is withheld at the shared Executor.
"""
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from agent.application.turn_decisions import (
    DeclarationGroundingError, IntentError, RequestRoles, _intent_instructions,
    decision_schema, parse_decision,
)
from agent.schemas import StepOutputRef
from test_interpretation_protocol import accepts, context, registry
from test_preparation_owner_contracts import Model, setup


SAVED = json.loads((Path(__file__).parent/'fixtures/spp_final_saved_interpretations.json').read_text())


def corrected(case):
    decision=json.loads(SAVED[case]['response'])['decision']
    if case=='B':
        # Preferred current intent route: Preparation derives control applicability.
        decision['kind']='execute_plan'
        decision.pop('required_roles')
        decision['missing_parameters']=[]
    for argument in decision.get('arguments', ()):
        if argument['argument']=='function_name':
            argument['native_function']=argument['value']
    if case=='D':
        decision['arguments']=[a for a in decision['arguments'] if a['argument']!='parameter_selection']
        decision['arguments'][0]['literal']='构建邻居图'
    return decision


def raw(decision):
    return json.dumps(dict(turn_schema_version=1,decision=decision),ensure_ascii=False)


def test_saved_wires_retain_exact_bytes_and_invalid_grounding():
    for value in SAVED.values():
        assert hashlib.sha256(value['response'].encode()).hexdigest()==value['response_sha256']
    for case in ('B','C'):
        with pytest.raises(IntentError) as error:
            parse_decision(SAVED[case]['response'],utterance=SAVED[case]['utterance'])
        assert error.value.reason=='invalid_decision'
    with pytest.raises(DeclarationGroundingError) as error:
        parse_decision(SAVED['D']['response'],utterance=SAVED['D']['utterance'])
    assert error.value.parameters==('run_scanpy.function_name','run_scanpy.parameter_selection')
    # Historical no-choice wires stay readable; they are no longer offered live.
    assert parse_decision(SAVED['A']['response']).reason=='ambiguous_parameter'


def test_field_choice_is_not_an_interpreter_no_choice_decision(registry):
    public=context(registry)
    schema=decision_schema(public,utterance=SAVED['A']['utterance'])
    assert not accepts(schema,json.loads(SAVED['A']['response']))
    public['dialogue']['pending_prerequisite']=dict(handle='@parameters',field='parameters',
        tool='run_scanpy',missing=['run_scanpy.kwargs'])
    schema=decision_schema(public,utterance='Which value?')
    assert not accepts(schema,json.loads(SAVED['A']['response']))
    assert accepts(schema,dict(turn_schema_version=1,decision=dict(kind='clarify',
        reason='ambiguous_parameter',choices=['run_scanpy.kwargs'])))


@pytest.mark.parametrize('case',['B','C','D'])
def test_corrected_wires_are_grounded_and_current_schema_achievable(registry,case):
    public=context(registry)
    if case=='D':
        public['dialogue']['planning_context']=[dict(handle='ctx.0')]
        public['dialogue']['result_groups']=[dict(handle='g0',tool='run_scanpy',outputs=['r0'],canonical_output='r0')]
        public['dialogue']['outputs']=[dict(handle='r0',subjects=dict(candidates=[]))]
    decision=corrected(case)
    # D's existing use_rep is a reviewed choice; adding function scope preserves it.
    schema=decision_schema(public,utterance=SAVED[case]['utterance'])
    if case=='D':
        from agent.application.turn_decisions import _interpretation_public
        cap=_interpretation_public(public,SAVED[case]['utterance'])['dialogue']['capabilities']['run_scanpy']
        assert 'pp.neighbors' in cap['functions'] and 'pp.neighbors' not in cap.get('function_parameters',{})
        assert 'pp.neighbors' in schema['$defs']['offered_function']['enum']
    assert accepts(schema,json.loads(raw(decision)))
    parsed=parse_decision(raw(decision),utterance=SAVED[case]['utterance'])
    assert parsed.target==decision['target']
    assert all(a.native_function==a.value for a in parsed.arguments if a.argument=='function_name')
    if case=='D':
        assert {a.argument:a.literal for a in parsed.arguments}==dict(function_name='构建邻居图',use_rep='X_pca')


def test_complete_missing_role_contract_is_preserved(registry):
    value=json.loads(SAVED['B']['response'])['decision']
    value['required_roles'].insert(0,dict(tool='run_macs3',port='dataset'))
    assert isinstance(parse_decision(raw(value),utterance=SAVED['B']['utterance']),RequestRoles)
    assert not accepts(decision_schema(context(registry),utterance=SAVED['B']['utterance']),json.loads(raw(value)))
    resources=dict(kind='request_resources',target='annotate_scATAC_cell_types',arguments=[])
    assert parse_decision(raw(resources)).target=='annotate_scATAC_cell_types'


def test_model_visible_schema_keeps_scope_and_literal_guidance(registry):
    schema=decision_schema(context(registry),utterance=SAVED['C']['utterance'])
    forms=schema['$defs']['declaration']['anyOf']
    assert all('Verbatim substring' in f['properties']['literal']['description']
        for f in forms if 'literal' in f['properties'])
    assert all('same canonical value' in f['properties']['native_function']['description']
        for f in forms if 'native_function' in f['properties'])
    roles=[f for f in schema['properties']['decision']['anyOf']
        if f['properties']['kind']['enum']==['request_roles']]
    assert roles and all(f['properties']['required_roles']['maxItems']==0 for f in roles)
    assert all(f['properties']['kind']['enum']!=['request_resources'] for f in schema['properties']['decision']['anyOf'])
    instructions=' '.join(_intent_instructions())
    assert 'omit its declaration' in instructions and 'remaining kwargs omitted' in instructions


class WitnessModel(Model):
    def __init__(self,case,candidate):
        super().__init__(tool='run_macs3' if case=='B' else 'run_scanpy');self.case=case;self.candidate=candidate
    def complete(self,*,prompt,response_schema):
        value=json.loads(prompt)
        if 'turn_schema_version' in value:
            self.calls.append(value)
            return raw(corrected(self.case))
        if 'semantic_prompt_version' in value:
            self.calls.append(value)
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=self.candidate)))
        result=super().complete(prompt=prompt,response_schema=response_schema)
        if 'retained_outputs' in value:
            result=json.dumps(dict(json.loads(result),retain=[]))
        return result


def step(sid,function,parent=None):
    source=(dict(kind='input',input='input_path') if parent is None else
        dict(kind='step_port',step=parent,source_port='dataset'))
    return dict(step_id=sid,tool='run_scanpy',sources=[dict(target='dataset',source=source),
        dict(target='function_name',source=dict(kind='input',input='run_scanpy__'+function.replace('.','_')+'__function_name'))],control_dependencies=[])


def test_corrected_multistep_witness_has_exact_scopes_and_symbolic_producer(tmp_path):
    pca=step('pca','pp.pca');neighbors=step('method','pp.neighbors','pca')
    model=WitnessModel('C',[pca,neighbors]);app,executor=setup(tmp_path,model)
    source=tmp_path/'source.h5ad';ad.AnnData(np.eye(12,6)).write_h5ad(source)
    rec=app.resources.register('input',source,input_type='h5ad',label='matrix',session_id='session',attribution='Technical fixture')
    result=app.submit_turn('session','task',SAVED['C']['utterance'],expected_generation=0,
        registered_input=app.resources.compose_group([rec.resource_id],attachment_candidates=True))
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    pca,neighbors=executor.plans[0].steps
    assert pca.arguments['kwargs']=={'n_comps':3}
    assert neighbors.arguments['kwargs']=={'n_neighbors':3}
    assert neighbors.arguments['use_rep']=='X_pca'
    assert neighbors.arguments['input_path']==StepOutputRef('pca','matrix_path')
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1


def test_corrected_missing_file_witness_upload_answer_reuses_candidate(tmp_path):
    candidate=dict(step_id='method',tool='run_macs3',sources=[
        dict(target='dataset',source=dict(kind='input',input='input_path')),
        dict(target='function_name',source=dict(kind='input',input='run_macs3__bdgcmp__function_name'))],control_dependencies=[])
    model=WitnessModel('B',[candidate]);app,executor=setup(tmp_path,model)
    source=tmp_path/'treatment.bdg';source.write_text('chr1\t0\t10\t1\n')
    rec=app.resources.register('treatment',source,input_type='attachment',label='treatment',session_id='session',attribution='Technical fixture')
    first=app.submit_turn('session','task',SAVED['B']['utterance'],expected_generation=0,
        registered_input=app.resources.compose_group([rec.resource_id],attachment_candidates=True))
    assert first.status=='clarification',first.to_dict()
    assert any(c.get('action')=='upload_source' for c in first.clarification_choices)
    original=app._application.sessions.load('session').interactions[0]
    control=tmp_path/'control.bdg';control.write_text('chr1\t0\t10\t2\n')
    rec=app.resources.register('control',control,input_type='attachment',label='control',session_id='session',attribution='Technical fixture')
    recheck=app.submit_turn('session','recheck','',expected_generation=0,resume_task_id=first.preparation_task_id,
        attachment_ids=[rec.resource_id])
    assert not executor.plans
    choice=next(c for c in recheck.clarification_choices if c.get('source_input') and c['label']=='control')
    result=app.submit_turn('session','answer','',expected_generation=0,
        clarification_choice=dict(question_turn_id='recheck',handle=choice['handle']))
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    args=executor.plans[0].steps[0].arguments
    assert args['control_path']==str(control) and args['kwargs']==dict(method='subtract')
    state=app._application.sessions.load('session')
    assert state.interactions[0]==original
    assert len({i.preparation['candidate_id'] for i in state.interactions})==1
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1


@pytest.mark.parametrize('wire_case',['no_choices','ordinary_roles','ordinary_resources'])
def test_schema_ignoring_no_choice_provider_cannot_bypass_candidate_owner(tmp_path,wire_case):
    class NoChoice(Model):
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(value)
                if wire_case=='no_choices':return SAVED['A']['response']
                if wire_case=='ordinary_resources':return raw(dict(kind='request_resources',target='run_macs3',arguments=[]))
                d=json.loads(SAVED['B']['response'])['decision']
                d['required_roles'].insert(0,dict(tool='run_macs3',port='dataset'))
                return raw(d)
            assert 'semantic_prompt_version' not in value, 'Malformed clarification reached planning'
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=NoChoice();app,executor=setup(tmp_path,model)
    result=app.submit_turn('session','task',SAVED['A' if wire_case=='no_choices' else 'B']['utterance'],expected_generation=0)
    assert result.error.code=='INTERACTIVE_INTENT_BINDING_INVALID'
    assert result.status=='failed',result.to_dict()
    assert not executor.plans and not app._application.sessions.load('session').turns


def test_existing_field_choice_follows_candidate_and_exact_answer(tmp_path,monkeypatch):
    from test_preparation_source_contracts import app_fixture
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch)
    complete=model.complete
    def intent(**kwargs):
        if 'turn_schema_version' in json.loads(kwargs['prompt']):
            model.calls.append(json.loads(kwargs['prompt']))
            return raw(dict(kind='execute_plan',target='annotate_scATAC_cell_types',
                roles=[dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')],
                arguments=[dict(tool='annotate_scATAC_cell_types',argument='species',literal='Human',evidence='这是 Human 的 GRCh38 单细胞 ATAC H5AD。')]))
        return complete(**kwargs)
    model.complete=intent
    first=app.submit_turn('session','task',SAVED['A']['utterance'],expected_generation=0,
        registered_input=app.resources.compose_group([source.resource_id],attachment_candidates=True),input_sets=refs)
    fields=[c for c in first.clarification_choices if c.get('source_field')]
    assert {c['value'] for c in fields}=={'cellranger_graphclust','other_groups'},first.to_dict()
    original=app._application.sessions.load('session').interactions[0]
    assert original.preparation['candidate_id'] is not None
    choice=next(c for c in fields if c['value']=='other_groups')
    result=app.submit_turn('session','field','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=choice['handle']))
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    assert executor.plans[0].steps[0].arguments['group_column']=='other_groups'
    assert app._application.sessions.load('session').interactions[0]==original
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    assert sum('turn_schema_version' in c for c in model.calls)==1


def test_accepted_artifact_literal_uses_published_pca_without_upstream_replanning(tmp_path,monkeypatch):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.application.workspace import ManagedWorkspace
    from agent.schemas import AgentPlan,AgentRequest,PlanStep,PriorOutputRef
    from agent.tools.analysis import native_matrix
    from agent.tools import _native_process
    model=WitnessModel('D',[step('method','pp.neighbors')]);app,executor=setup(tmp_path,model)
    path=tmp_path/'source.h5ad';ad.AnnData(np.eye(12,6)).write_h5ad(path)
    # Simulated accepted native publication, not a biological/native qualification.
    def native(self,args,stage,matrix):
        matrix.obsm['X_pca']=np.ones((12,3));matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded',versions={'scanpy':'SIMULATED'},implementation=[],native_output_type='NoneType',result_slots=[],measurement='unchanged fixture')
    monkeypatch.setattr(native_matrix.MatrixPackage,'native',native)
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    root=app._application.workspace_root
    output=root/'runs'/ManagedWorkspace.run_digest('pca:run')/'pca'
    class PCA:
        def plan(self,request,registry):
            return AgentPlan('pca:plan',request.request_id,'Accepted publication fixture',(
                PlanStep('pca','run_scanpy',dict(input_path=str(path),function_name='pp.pca',kwargs={'n_comps':3},output_dir=str(output))),))
    seed=ResearchAgentApplication(root,planner=PCA())
    state=seed.sessions.run('session','pca',AgentRequest('pca','Produce PCA fixture',{}),
        (OutputSelection('dataset','pca','matrix_path'),),expected_generation=0)
    assert state.generation==1
    first_revision=state.revisions
    from agent.orchestration.prior_outputs import bind_output
    published=bind_output(seed.run_store,seed.registry,'pca:run','pca','dataset')
    source_bytes=path.read_bytes()
    # Accepted output is selected through the current exact context, never input_path.
    candidate=model.candidate[0]
    candidate['sources'][0]['source']=dict(kind='context',handle='ctx.0')
    result=app.submit_turn('session','neighbors',SAVED['D']['utterance'],expected_generation=1)
    assert result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    operation=executor.plans[0].steps[0]
    assert operation.arguments['input_path']==PriorOutputRef(published,'matrix_path')
    assert operation.arguments['use_rep']=='X_pca'
    assert 'parameter_selection' not in operation.arguments and not operation.arguments.get('kwargs')
    assert len(executor.plans)==1 and path.read_bytes()==source_bytes
    assert app._application.sessions.load('session').revisions==first_revision
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1


def test_zero_uploaded_files_acquires_candidate_then_exact_source_answers(tmp_path):
    # The operation is explicit even before either external input is uploaded.
    candidate=dict(step_id='method',tool='run_macs3',sources=[
        dict(target='function_name',source=dict(kind='input',input='run_macs3__bdgcmp__function_name'))],control_dependencies=[])
    class MissingBoth(WitnessModel):
        def complete(self,*,prompt,response_schema):
            value=json.loads(prompt)
            if 'turn_schema_version' in value:
                self.calls.append(value);decision=corrected('B');decision['roles']=[]
                return raw(decision)
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=MissingBoth('B',[candidate]);app,executor=setup(tmp_path,model)
    first=app.submit_turn('session','task',SAVED['B']['utterance'],expected_generation=0)
    assert first.status=='clarification',first.to_dict()
    origin=app._application.sessions.load('session').interactions[0]
    assert origin.preparation['candidate_id'] and not executor.plans
    assert any(c.get('action')=='upload_source' for c in first.clarification_choices)
    record_ids=[]
    for name in ('treatment','control'):
        path=tmp_path/(name+'.bdg');path.write_text('chr1\t0\t10\t1\n')
        rec=app.resources.register(name,path,input_type='attachment',label=name,session_id='session',attribution='Technical fixture')
        record_ids.append(rec.resource_id)
    question=app.submit_turn('session','recheck','',expected_generation=0,resume_task_id=first.preparation_task_id,
        attachment_ids=record_ids)
    from agent.application import preparation_state as records
    from agent.application.dialogue_execution import captured_input_group
    from agent.application.session_state import Interaction,SessionConflictError
    from agent.schemas.orchestration import _serialize
    captured=app._application.sessions.load('session')
    ref=records.capture_intake(captured.interactions,0,None,{},None)
    group=app.resources.compose_group(record_ids,attachment_candidates=True)
    for change in ('order','identity'):
        metadata=_serialize(group.attribution())
        if change=='order':metadata['members'].reverse()
        else:metadata['members'][0]['source_sha256']='f'*64
        bad=Interaction('bad-'+change,'Use the provided sources.',None,0,dict(preparation=ref),submission=dict(
            registered_input=metadata,execution_inputs={}))
        with pytest.raises(SessionConflictError):captured_input_group(bad,captured.interactions)
    assert app._application.sessions.load('session')==captured
    for turn,name in [('treatment','treatment'),('control','control')]:
        choices=[c for c in question.clarification_choices if c.get('source_input')]
        argument='input_path' if turn=='treatment' else 'control_path'
        choice=next((c for c in choices if c['argument']==argument and c['label']==name),None)
        assert choice is not None,question.to_dict()
        question=app.submit_turn('session',turn,'',expected_generation=0,
            clarification_choice=dict(question_turn_id=question.turn_id,handle=choice['handle']))
    assert question.error.code=='TEST_SCIENCE_WITHHELD',question.to_dict()
    assert len(executor.plans)==1
    assert executor.plans[0].steps[0].arguments['input_path']==str(tmp_path/'treatment.bdg')
    assert executor.plans[0].steps[0].arguments['control_path']==str(tmp_path/'control.bdg')
    state=app._application.sessions.load('session')
    assert state.interactions[0]==origin
    assert len({i.preparation['candidate_id'] for i in state.interactions})==1
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
