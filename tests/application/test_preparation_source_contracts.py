"""SPP-1F source/axis/reference boundaries; no native scientific execution."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from agent.application import InteractiveAgentApplication, preparation_state as records
from agent.application.input_sets import ScientificInputSet
from agent.application.native_raw_handoff import gene_annotation_component, _component, bind_plan, _species
from agent.application.raw_handoff import AnnotationDeclaration, declaration_dict
from agent.application.session_state import SessionConflictError
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.orchestration.semantic_compiler import ScientificInputRequired
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas.orchestration import _serialize
from test_human_epizoo_mapping import WithholdScience


def source_file(tmp_path, *, genomes=('GRCh38',)*3, uns=None):
    path=tmp_path/'source.h5ad'
    data=ad.AnnData(np.eye(4,3), obs=pd.DataFrame({
        'cellranger_graphclust':pd.Categorical(['a','a','b','b']),
        'other_groups':pd.Categorical(['a','b','a','b'])}, index=['a','b','c','d']),
        var=pd.DataFrame({'genome':list(genomes),'feature_identity':['f1','f2','f3']},index=['f1','f2','f3']),
        uns={} if uns is None else uns)
    data.write_h5ad(path)
    return path


def pin(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(tmp_path, *, assembly=None, count=1):
    catalog={}
    for i in range(count):
        path=tmp_path/f'genes{i}.gtf';path.write_text(f'# controlled reference {i}\n')
        catalog[str(i)]=dict(species=_species('human'),assembly='GRCh38',label=f'GTF {i}',default=False,
            component_identity_version=2,components={'gene_anno':dict(path=str(path),sha256=pin(path))})
    return dict(catalog=catalog,selected=None,custom={},assembly=assembly,species='human')


@pytest.mark.parametrize('mode', ['consistent','unknown','mismatch','conflict','mixed','empty'])
def test_reference_qualification_uses_source_claims_not_species(tmp_path, mode):
    genomes={'mixed':('GRCh38','hg19','GRCh38'),'empty':('',)*3,'unknown':('',)*3}.get(mode,('GRCh38',)*3)
    path=source_file(tmp_path,genomes=genomes)
    if mode=='unknown':
        with h5py.File(path,'r+') as f: del f['var/genome']
    rec=receipt(tmp_path,assembly='hg19' if mode in ('mismatch','conflict') else None)
    if mode=='conflict':
        data=ad.read_h5ad(path);data.uns['assembly']='hg19';data.write_h5ad(path)
    if mode=='consistent':
        component,chosen,reason,observed,assembly=gene_annotation_component(rec,path)
        assert chosen=='0' and assembly=='GRCh38' and reason=='automatic'
        assert observed['source_sha256']==pin(path)
        assert observed['observations']==[dict(origin='var.genome',values=['GRCh38'],complete=True)]
        assert component==rec['catalog']['0']['components']['gene_anno']
    else:
        with pytest.raises(ScientificInputRequired): gene_annotation_component(rec,path)


def test_unknown_cannot_select_default_and_multiple_are_exact_choices(tmp_path):
    rec=receipt(tmp_path,count=2)
    with pytest.raises(ScientificInputRequired) as unknown: _component(rec,'gene_anno')
    assert not unknown.value.requirements[0].offers
    path=source_file(tmp_path)
    with pytest.raises(ScientificInputRequired) as multiple: gene_annotation_component(rec,path)
    offers=multiple.value.requirements[0].offers
    assert {o['handle'] for o in offers}=={'resource.0','resource.1'}
    for key in ('0','1'):
        component,chosen,*_=gene_annotation_component(dict(rec,selected=key),path,explicit=True)
        assert chosen==key and component==rec['catalog'][key]['components']['gene_anno']
    with pytest.raises(ScientificInputRequired): gene_annotation_component(dict(rec,selected='absent'),path,explicit=True)


def test_exact_reference_alias_witness_survives_supplied_selection(tmp_path):
    path=source_file(tmp_path);rec=receipt(tmp_path)
    rec['catalog']['0']['assembly']='hg38'
    identity=dict(reference=dict(path='/genome',sha256='a'*64),chrom_sizes=dict(path='/fai',sha256='b'*64,ordered_sizes='c'*64))
    rec['catalog']['0']['components'].update(identity)
    rec['catalog']['alias']=dict(rec['catalog']['0'],assembly='GRCh38',components=identity)
    component=rec['catalog']['0']['components']['gene_anno']
    assert gene_annotation_component(rec,path,supplied=component['path'],supplied_sha256=component['sha256'])[1]=='0'
    rec['catalog']['alias']['components']=dict(identity,chrom_sizes=dict(identity['chrom_sizes'],sha256='d'*64))
    with pytest.raises(ScientificInputRequired): gene_annotation_component(rec,path)


@pytest.mark.parametrize('changed', ['source','gtf','supplied_sha','fai'])
def test_source_reference_integrity_and_no_silent_substitution(tmp_path, changed):
    path=source_file(tmp_path);source_sha=pin(path);rec=receipt(tmp_path)
    component=rec['catalog']['0']['components']['gene_anno'];kwargs={}
    if changed=='source': path.write_bytes(path.read_bytes()+b'changed')
    elif changed=='gtf': Path(component['path']).write_text('changed')
    elif changed=='supplied_sha': kwargs=dict(supplied=component['path'],supplied_sha256='f'*64)
    else:
        rec['assembly']='hg19'
    with pytest.raises((ScientificInputRequired,ValueError,OSError)):
        gene_annotation_component(rec,path,source_sha256=source_sha,**kwargs)


def test_saved_plan_replay_never_calls_current_source_observation(tmp_path,monkeypatch):
    from agent.schemas import AgentPlan, PlanStep
    from agent.application import native_raw_handoff as owner
    rec=receipt(tmp_path,assembly='GRCh38');component=rec['catalog']['0']['components']['gene_anno']
    def forbidden(*a,**kw): pytest.fail('History reopened current source files')
    monkeypatch.setattr(owner,'gene_annotation_component',forbidden)
    # Saved binding replay is deliberately separate from live qualification.
    plan=SimpleNamespace(steps=())
    from agent.schemas import AgentRequest
    from agent.orchestration.semantic_compiler import SemanticPlanCandidate,SemanticPlanStep,SemanticRequestInputSource as Input,SemanticRegisteredChoiceSource as Choice,analyze_candidate,compile_analysis
    registry=build_default_tool_registry()
    request=AgentRequest('history','Recorded native activity',dict(input_path='/unavailable/source.h5ad',gene_anno_path=component['path'],gene_anno_sha256=component['sha256'],output_dir='/out'))
    candidate=SemanticPlanCandidate((SemanticPlanStep('activity','run_episcanpy',(Input('dataset','input_path'),Choice('function_name','tl.geneactivity'),Input('gene_anno','gene_anno_path'))),))
    analysis=analyze_candidate(request,candidate,registry)
    plan=compile_analysis(request,analysis,registry,planner_name='recorded')
    assert bind_plan(plan,rec)==plan


class Model:
    model_id='source-contract-offline'
    def __init__(self, *, finite=False, inspect=False): self.calls=[];self.finite=finite;self.inspect=inspect
    def complete(self, *, prompt,response_schema):
        value=json.loads(prompt);self.calls.append(value)
        if 'grounding_review_version' in value:
            return json.dumps(dict(verdicts={r['id']:dict(verdict='entailed',independent=True) for r in value['paragraphs']}))
        if value.get('focus')=='outcome':
            return json.dumps(dict(support='supported',paragraphs=[dict(text='请提供所需输入，科学执行尚未开始。',kind='limitation',citations=[],comparisons=[])]))
        if 'turn_schema_version' in value:
            if 'cellranger_graphclust' in value['utterance']:
                # Genuine response005 decision, replayed through current admission.
                return json.dumps(dict(turn_schema_version=1,decision=dict(kind='answer_roles',pending='@roles',roles=[dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')],arguments=[dict(tool='annotate_scATAC_cell_types',argument='group_column',literal='cellranger_graphclust',evidence='请沿用文件中的 cellranger_graphclust 作为初始分组。',source='source.0',mode='offered',value='cellranger_graphclust')])))
            decision=(dict(kind='execute_plan',target='run_scanpy',roles=[dict(tool='run_scanpy',port='dataset',source='source.0')]) if self.finite else
                dict(kind='execute_plan',target='annotate_scATAC_cell_types',roles=[dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')],arguments=[dict(tool='annotate_scATAC_cell_types',argument='species',literal='人类',evidence=value['utterance'])]))
            return json.dumps(dict(turn_schema_version=1,decision=decision))
        if 'semantic_prompt_version' in value:
            if self.finite:
                steps=[dict(step_id='method',tool='run_scanpy',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[])]
            else:
                source=dict(kind='step_port',step='inspect',source_port='dataset') if self.inspect else dict(kind='input',input='input_path')
                steps=[dict(step_id='annotation',tool='annotate_scATAC_cell_types',control_dependencies=[],sources=[dict(target='source',source=source),dict(target='activity_method',source=dict(kind='registered_choice',choice='episcanpy.tl.geneactivity'))])]
                if self.inspect:steps.insert(0,dict(step_id='inspect',tool='inspect_scATAC',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[]))
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='result',step_id='method' if self.finite else 'annotation',output_key='matrix_path' if self.finite else 'hierarchy_path')]))
        raise AssertionError(list(value))


def app_fixture(tmp_path,monkeypatch,*,finite=False,inspect=False):
    from agent.tools.analysis import sequence_prediction
    from agent.tools.data import cellmarker
    monkeypatch.setattr(sequence_prediction,'library',lambda:SimpleNamespace(entries=lambda:[dict(kind='cellmarker',id='installed')]))
    monkeypatch.setattr(cellmarker,'CellMarkerEvidence',lambda *a:SimpleNamespace(identity={},metadata={'species':['Human','Mouse']}))
    registry=build_default_tool_registry();executor=WithholdScience(registry);model=Model(finite=finite,inspect=inspect)
    profile=PlanningModelProfile('offline','scripted',model.model_id)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,model_profiles=(profile,),default_profile_id='offline',approved_source_roots=(tmp_path,),planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session');path=source_file(tmp_path)
    source=app.resources.register('source',path,input_type='h5ad',session_id='session',label='Source',attribution='Controlled source contract fixture')
    rec=receipt(tmp_path)
    refs=(ScientificInputSet('genes','Genes',dict(species='human',assembly='GRCh38',gene_anno_path=rec['catalog']['0']['components']['gene_anno']['path'],gene_anno_sha256=rec['catalog']['0']['components']['gene_anno']['sha256'])),)
    (app._application.workspace_root/'uploads').mkdir()
    return app,model,executor,source,refs


def wait(client,turn):
    import time
    for _ in range(400):
        response=client.get('/api/v1/sessions/session/turns/'+turn);assert response.status_code==200,response.text
        value=response.json()
        if value.get('response') is not None or value['status'] in ('failed','interrupted','cancelled'):return value
        time.sleep(.02)
    pytest.fail('HTTP interaction did not finish')


@pytest.mark.parametrize('textual',[False,True])
@pytest.mark.parametrize('inspect',[False,True])
def test_http_human_then_source_field_converge_without_candidate_regeneration(tmp_path,monkeypatch,textual,inspect):
    from agent.web.app import create_app
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch,inspect=inspect)
    uploads=H5ADUploadAdmission(app.resources,app._application.workspace_root/'uploads')
    web=create_app(app,uploads=uploads,input_sets=refs)
    with TestClient(web) as client:
        body=dict(turn_id='task',expected_generation=0,utterance='请注释人类数据。',attachment_ids=[source.resource_id])
        assert client.post('/api/v1/sessions/session/turns',json=body).status_code==202
        first=wait(client,'task');human=next(c for c in first['clarification_choices'] if c.get('value')=='Human')
        captured=app._application.sessions.load('session').interactions[0]
        assert client.post('/api/v1/sessions/session/turns',json=dict(turn_id='human',expected_generation=0,utterance='',clarification_choice=dict(question_turn_id='task',handle=human['handle']))).status_code==202
        second=wait(client,'human');fields=[c for c in second['clarification_choices'] if c.get('source_field')]
        assert {c['value'] for c in fields}=={'cellranger_graphclust','other_groups'}
        assert all(c['source_field']['axis']=='obs' and c['source_field']['port']=='source' for c in fields)
        assert str(tmp_path) not in json.dumps(second['clarification_choices'])
        selected=next(c for c in fields if c['value']=='cellranger_graphclust')
        reply=dict(turn_id='field',expected_generation=0,utterance='请沿用文件中的 cellranger_graphclust 作为初始分组。' if textual else '')
        if not textual:reply['clarification_choice']=dict(question_turn_id='human',handle=selected['handle'])
        assert client.post('/api/v1/sessions/session/turns',json=reply).status_code==202
        final=wait(client,'field')
        assert final['error']['code']=='TEST_SCIENCE_WITHHELD',final
        assert len(executor.plans)==1
        annotation=next(s for s in executor.plans[0].steps if s.tool_name=='annotate_scATAC_cell_types')
        assert annotation.arguments['group_column']=='cellranger_graphclust'
        state=app._application.sessions.load('session')
        assert state.interactions[0]==captured
        answers=state.interactions[-1].preparation['answers']
        assert len(answers)==1 and answers[0]['requirement_id']==selected['requirement_id']
        assert sum('semantic_prompt_version' in c for c in model.calls)==1
        assert sum('turn_schema_version' in c for c in model.calls)==(2 if textual else 1)
        assert not state.revisions


def test_http_non_field_finite_choice_uses_same_question_contract(tmp_path,monkeypatch):
    from agent.web.app import create_app
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch,finite=True)
    with TestClient(create_app(app,uploads=H5ADUploadAdmission(app.resources,app._application.workspace_root/'uploads'))) as client:
        client.post('/api/v1/sessions/session/turns',json=dict(turn_id='task',expected_generation=0,utterance='Choose a native method.',attachment_ids=[source.resource_id]))
        first=wait(client,'task');choice=next(c for c in first['clarification_choices'] if c.get('value')=='pp.scale')
        assert choice['handle'].startswith('spp.') and 'source_field' not in choice
        client.post('/api/v1/sessions/session/turns',json=dict(turn_id='method',expected_generation=0,utterance='',clarification_choice=dict(question_turn_id='task',handle=choice['handle'])))
        result=wait(client,'method')
        assert result['error']['code']=='TEST_SCIENCE_WITHHELD',result
        assert executor.plans[0].steps[0].arguments['function_name']=='pp.scale'
        assert sum('turn_schema_version' in c for c in model.calls)==1
        assert sum('semantic_prompt_version' in c for c in model.calls)==1


def test_canonical_role_resolution_is_exact_and_consumer_scoped():
    from agent.application.session_state import SessionError
    members=[dict(resource_id=f'id{i}',record_sha256=str(i)*64,source_path=f'/source{i}',source_sha256=str(i+1)*64) for i in range(2)]
    admitted=dict(tool='annotate_scATAC_cell_types',annotation={'source':None},annotation_sources={'members':members},
        roles=[dict(tool='annotate_scATAC_cell_types',port='source',source='source.1')])
    captured={'members':members}
    scope=records.declaration_source(admitted,'annotation',captured)
    assert scope['sources']==[members[1]]
    explicit=dict(admitted,annotation={'source':'source.0'},roles=[dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')])
    assert records.declaration_source(explicit,'annotation',captured)['sources']==[members[0]]
    for roles in ([dict(tool='annotate_scATAC_cell_types',port='groups',source='source.1')],
                  [dict(tool='run_scanpy',port='dataset',source='source.1')],
                  [dict(tool='annotate_scATAC_cell_types',port='source',source=f'source.{i}') for i in range(2)]):
        with pytest.raises(SessionConflictError): records.declaration_source(dict(admitted,roles=roles),'annotation',captured)
    forged=[members[0],dict(members[1],source_sha256='f'*64)]
    from agent.application.session_state import Interaction
    previous=Interaction('previous','Original choice',None,0,{},admitted=admitted)
    interaction=Interaction('changed','Changed choice',None,0,{})
    with pytest.raises(SessionConflictError):
        records.declaration_changes(previous,interaction,dict(admitted,annotation_sources={'members':forged}),'annotation',captured)
    accepted=dict(accepted_results=[{'run_id':'exact','authority_sha256':'a'*64}])
    no_roles=dict(tool='annotate_scATAC_cell_types',annotation={'source':None})
    assert records.declaration_source(no_roles,'annotation',accepted)['sources']==[{'accepted_binding':accepted['accepted_results'][0]}]


@pytest.mark.parametrize('invalid',['forged','changed_bytes','stale'])
def test_http_field_answers_reject_forgery_changes_and_staleness(tmp_path,monkeypatch,invalid):
    from agent.web.app import create_app
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch)
    with TestClient(create_app(app,uploads=H5ADUploadAdmission(app.resources,app._application.workspace_root/'uploads'),input_sets=refs)) as client:
        client.post('/api/v1/sessions/session/turns',json=dict(turn_id='task',expected_generation=0,utterance='请注释人类数据。',attachment_ids=[source.resource_id]))
        first=wait(client,'task');human=next(c for c in first['clarification_choices'] if c.get('value')=='Human')
        client.post('/api/v1/sessions/session/turns',json=dict(turn_id='human',expected_generation=0,utterance='',clarification_choice=dict(question_turn_id='task',handle=human['handle'])))
        second=wait(client,'human');selected=next(c for c in second['clarification_choices'] if c.get('value')=='cellranger_graphclust')
        before=app._application.sessions.load('session')
        choice=dict(question_turn_id='human',handle=selected['handle'])
        if invalid=='changed_bytes':Path(source.source_path).write_bytes(Path(source.source_path).read_bytes()+b'changed')
        elif invalid=='forged':choice['handle']+='forged'
        else:
            client.post('/api/v1/sessions/session/turns',json=dict(turn_id='valid',expected_generation=0,utterance='',clarification_choice=choice))
            assert wait(client,'valid')['error']['code']=='TEST_SCIENCE_WITHHELD'
        client.post('/api/v1/sessions/session/turns',json=dict(turn_id='invalid',expected_generation=0,utterance='',clarification_choice=choice))
        result=wait(client,'invalid')
        assert result['error']['code']==('LOCAL_RESOURCE_INTEGRITY_INVALID' if invalid=='changed_bytes' else 'INTERACTIVE_TURN_CONFLICT'),result
        assert len(executor.plans)==(1 if invalid=='stale' else 0)
        assert app._application.sessions.load('session').interactions[:2]==before.interactions
        assert sum('semantic_prompt_version' in c for c in model.calls)==1
        assert sum('turn_schema_version' in c for c in model.calls)==1


def test_metadata_axes_are_bounded_private_and_never_read_the_count_matrix(tmp_path,monkeypatch):
    from agent.application.local_resources import source_field_choices,source_field_inputs
    from agent.orchestration.registry import ToolRegistry
    from agent.tools.analysis.categorical import has_field
    app,model,executor,registered,refs=app_fixture(tmp_path,monkeypatch)
    path=tmp_path/'metadata.h5ad'
    data=ad.AnnData(np.eye(3),obs=pd.DataFrame({f'group_{i}':['a','b','b'] for i in range(70)}),
        var=pd.DataFrame({'chrom':['chr1']*3,'start':[1,2,3],'end':[2,3,4]}),
        uns={'assembly':'GRCh38','nested':{'payload':'/private/secret'},'api_key':'private-secret'})
    data.obs['api_key']=['s','s','s'];data.write_h5ad(path)
    registered=app.resources.register('metadata',path,input_type='h5ad',session_id='session',label='Metadata',attribution='Controlled metadata fixture')
    binding=app.resources.compose_group([registered.resource_id],attachment_candidates=True)
    getitem=h5py.Dataset.__getitem__
    def guarded(dataset,*args,**kwargs):
        assert not dataset.name.startswith('/X'),'Field UI read the count matrix'
        return getitem(dataset,*args,**kwargs)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
    observed=source_field_choices(binding);row,=observed
    assert len(row['fields'])==64 and row['complete'] is False
    assert 'api_key' not in json.dumps(observed) and 'private-secret' not in json.dumps(observed)
    assert row['axes']['var']['fields']==['chrom','start','end']
    assert row['axes']['uns']['fields']==['assembly']
    assert has_field(path,'chrom',axis='var') and not has_field(path,'chrom',axis='obs')
    assert has_field(path,'assembly',axis='uns') and not has_field(path,'nested',axis='uns')
    registry=build_default_tool_registry();tool=registry.get('compute_motif_accessibility')
    spec=tool.optional_arguments['feature_chrom_key']
    # Applicability-ready Registry contract for an existing actual var consumer;
    # no future native metadata/capability is added to production Registry here.
    tool=replace(tool,optional_arguments=dict(tool.optional_arguments,feature_chrom_key=replace(spec,
        planning=replace(spec.planning,observation_source='dataset',observation_axis='var'))))
    scoped=ToolRegistry(tuple(tool if n==tool.name else registry.get(n) for n in registry.names()))
    roles=[dict(tool=tool.name,port='dataset',source='source.0')]
    values=source_field_inputs(scoped,binding,roles,observed)
    assert values[(tool.name,'feature_chrom_key')]==('chrom','start','end')
    assert all(name.startswith('group_') for name in source_field_inputs(registry,binding,
        [dict(tool='annotate_scATAC_cell_types',port='source',source='source.0')],observed)[('annotate_scATAC_cell_types','group_column')])


def test_empty_metadata_does_not_override_valid_researcher_assembly(tmp_path):
    path=source_file(tmp_path,genomes=('',)*3)
    assert gene_annotation_component(receipt(tmp_path,assembly='GRCh38'),path)[-1]=='GRCh38'
    with pytest.raises(ScientificInputRequired):gene_annotation_component(receipt(tmp_path),path)


def test_future_count_branch_retains_conflicting_source_declaration(tmp_path):
    from agent.schemas import StepOutputRef
    rec=receipt(tmp_path,assembly='GRCh38')
    with pytest.raises(ScientificInputRequired,match='conflicts'):
        gene_annotation_component(dict(rec,source_assembly='hg19'),StepOutputRef('matrix','matrix_path'))
    # Reference selection and source declaration are separate authorities.
    assert gene_annotation_component(dict(rec,source_assembly='GRCh38'),StepOutputRef('matrix','matrix_path'))[-1]=='GRCh38'


def test_planned_mapping_reference_establishes_assembly_only_on_its_exact_lineage(tmp_path):
    from agent.schemas import StepOutputRef
    from agent.application.native_raw_handoff import mapped_source_reference
    rec=receipt(tmp_path);fasta=tmp_path/'genome.fa';fasta.write_text('>chr1\nACGT\n')
    fasta_pin=dict(path=str(fasta),sha256=pin(fasta))
    rec['catalog']['0']['components'].update(reference=fasta_pin,
        chrom_sizes=dict(path='/fai',sha256='a'*64,ordered_sizes='b'*64,fasta_sha256=pin(fasta)))
    selected={'map':('run_chromap',dict(function_name='map',reference_path=str(fasta),reference_sha256=pin(fasta))),
              'import':('run_snapatac2',dict(function_name='pp.import_fragments',input_path=StepOutputRef('map','fragments_path')))}
    proof=mapped_source_reference(StepOutputRef('import','matrix_path'),selected,build_default_tool_registry())
    assert proof==dict(step_id='map',**fasta_pin)
    assert mapped_source_reference('/existing/fragments',selected,build_default_tool_registry()) is None
    resources=dict(reference_path=str(fasta),reference_sha256=pin(fasta))
    result=gene_annotation_component(dict(rec,resources=resources,source_reference=proof),StepOutputRef('import','matrix_path'))
    assert result[-1]=='GRCh38' and result[-2]['planned_mapping_reference']==proof
    assert rec['assembly'] is None
    with pytest.raises(ScientificInputRequired,match='differs'):
        gene_annotation_component(dict(rec,resources=dict(resources,reference_sha256='f'*64),source_reference=proof),StepOutputRef('import','matrix_path'))
    with pytest.raises(ScientificInputRequired,match='assembly'):
        gene_annotation_component(dict(rec,resources=resources),StepOutputRef('import','matrix_path'))


def test_a_single_observed_group_is_still_a_researcher_choice(tmp_path,monkeypatch):
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch)
    path=tmp_path/'single.h5ad';data=ad.read_h5ad(source.source_path)
    del data.obs['other_groups'];data.write_h5ad(path)
    registered=app.resources.register('single',path,input_type='h5ad',session_id='session',label='Single field',attribution='Controlled fixture')
    first=app.submit_turn('session','task','请注释人类数据。',expected_generation=0,attachment_ids=[registered.resource_id],input_sets=refs)
    human=next(c for c in first.clarification_choices if c.get('value')=='Human')
    second=app.submit_turn('session','human','',expected_generation=0,input_sets=refs,
        clarification_choice=dict(question_turn_id='task',handle=human['handle']))
    fields=[c for c in second.clarification_choices if c.get('source_field')]
    assert second.status=='clarification' and len(fields)==1
    assert fields[0]['value']=='cellranger_graphclust'
    assert not executor.plans and not app._application.sessions.load('session').turns


def test_selected_assembly_metadata_never_follows_external_ancestors(tmp_path,monkeypatch):
    from agent.tools.analysis.sequence_reference import assembly_observations
    path=source_file(tmp_path);external=tmp_path/'external.h5'
    with h5py.File(external,'w') as f:f.create_group('source').create_dataset('reference',data='GRCh38')
    with h5py.File(path,'r+') as f:f['uns']['source']=h5py.ExternalLink(str(external),'/source')
    getitem=h5py.Group.__getitem__
    def guarded(group,name):
        assert not (group.name=='/uns' and name=='source'),'External metadata ancestor was opened'
        return getitem(group,name)
    monkeypatch.setattr(h5py.Group,'__getitem__',guarded)
    with pytest.raises(ValueError,match='Indirect HDF5 metadata'):
        assembly_observations(path,pin(path))


def test_later_authorized_source_answers_unselected_task_without_replacing_history(tmp_path,monkeypatch):
    from agent.tools.analysis import annotation_input
    app,model,executor,source,refs=app_fixture(tmp_path,monkeypatch)
    normal=model.complete
    def complete(*,prompt,response_schema):
        value=json.loads(prompt)
        if 'turn_schema_version' not in value:return normal(prompt=prompt,response_schema=response_schema)
        model.calls.append(value)
        if value['utterance']=='Annotate whichever dataset I identify next.':
            decision=dict(kind='request_roles',target='annotate_scATAC_cell_types',required_roles=[],roles=[],arguments=[],annotation=declaration_dict(AnnotationDeclaration()))
        else:
            decision=dict(kind='answer_roles',pending='@roles',roles=[],annotation=declaration_dict(AnnotationDeclaration(source='source.1',species='Human',assembly='GRCh38',generate_umap=False)),arguments=[dict(tool='annotate_scATAC_cell_types',argument='group_column',source='source.1',mode='offered',value='cellranger_graphclust',literal='cellranger_graphclust',evidence=value['utterance'])])
        return json.dumps(dict(turn_schema_version=1,decision=decision))
    model.complete=complete
    first=app.submit_turn('session','task','Annotate whichever dataset I identify next.',expected_generation=0,
        registered_input=app.resources.compose_group([source.resource_id],attachment_candidates=True),input_sets=refs)
    before=app._application.sessions.load('session');origin=before.interactions[0]
    assert first.status=='clarification' and origin.preparation['candidate_id'] is None
    data=ad.read_h5ad(source.source_path);path=tmp_path/'later.h5ad';data.write_h5ad(path)
    later=app.resources.register('later',path,input_type='h5ad',session_id='session',label='Later supplied source',attribution='Authorized append fixture')
    monkeypatch.setattr(annotation_input,'observe',annotation_input._observe)
    result=app.submit_turn('session','select','Use the new dataset and cellranger_graphclust.',expected_generation=0,
        attachment_ids=[later.resource_id],input_sets=refs)
    assert result.status=='failed' and result.error.code=='TEST_SCIENCE_WITHHELD',result.to_dict()
    state=app._application.sessions.load('session')
    assert state.interactions[0]==origin and not state.revisions
    assert state.interactions[-1].preparation['task_id']==origin.preparation['task_id']
    assert len(executor.plans)==1 and executor.plans[0].steps[0].arguments['input_path']==str(path)
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    assert state.interactions[-1].admitted['annotation_sources']['members'][0]['source_sha256']==source.source_sha256
    assert state.interactions[-1].admitted['annotation_sources']['members'][1]['source_sha256']==later.source_sha256
