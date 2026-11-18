"""Selected-owner/source contracts with canonical questions and real acceptance."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import StepOutputRef, PriorOutputRef
from agent.application.session_state import Interaction
from agent.application import preparation_state as records
from test_human_epizoo_mapping import WithholdScience
from test_response_contracts import OutcomeModel


class Model:
    model_id='selected-owner-contract'
    def __init__(self, tool='run_macs3', function='bdgcmp', roles=True):
        self.tool=tool; self.function=function; self.roles=roles; self.calls=[]
    def complete(self, *, prompt, response_schema):
        value=json.loads(prompt); self.calls.append(value)
        if 'turn_schema_version' in value:
            d=dict(kind='execute_plan',target=self.tool)
            if self.roles: d['roles']=[dict(tool=self.tool,port='dataset',source='source.0')]
            return json.dumps(dict(turn_schema_version=1,decision=d))
        if 'semantic_prompt_version' in value:
            return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(
                step_id='method',tool=self.tool,sources=[dict(target='dataset',source=dict(kind='input',input='input_path')),
                dict(target='function_name',source=dict(kind='registered_choice',choice=self.function))],control_dependencies=[])])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='result',step_id='method',output_key='matrix_path' if self.tool in ('run_scanpy','run_episcanpy','run_snapatac2') else 'table_path')]))
        return OutcomeModel('Provide the requested input. Scientific execution has not started.').complete(prompt=prompt,response_schema=response_schema)
    def close(self): pass


def setup(tmp_path, model, registry=None):
    registry=registry or build_default_tool_registry(); executor=WithholdScience(registry)
    app=InteractiveAgentApplication(tmp_path/'workspace',registry=registry,executor=executor,
        approved_source_roots=(tmp_path,),model_profiles=(PlanningModelProfile('offline','scripted',model.model_id),),
        default_profile_id='offline',planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('session')
    return app,executor


@pytest.mark.parametrize('supplied',[False,True])
def test_missing_macs_control_has_file_control_and_exact_candidate_continuation(tmp_path,supplied):
    model=Model(); app,executor=setup(tmp_path,model)
    members=[]
    for name in ('treatment','control') if supplied else ('treatment',):
        path=tmp_path/(name+'.tsv');path.write_text('chr1\t0\t10\t1\n')
        record=app.resources.register(name,path,input_type='attachment',label=name,session_id='session',attribution='controlled fixture')
        members.append(record.resource_id)
    group=app.resources.compose_group(members,attachment_candidates=True)
    first=app.submit_turn('session','task','Compare these signal tracks.',expected_generation=0,registered_input=group)
    assert first.status=='clarification',first.to_dict()
    state=app._application.sessions.load('session'); candidate=state.interactions[-1].preparation['candidate_id']
    options=[o for o in first.clarification_choices if o['argument']=='control_path' and o.get('source_input')]
    assert options and all(o.get('source_input') for o in options)
    chosen=next(o for o in options if o['label']==('control' if supplied else 'treatment'))
    done=app.submit_turn('session','answer','',expected_generation=0,
        clarification_choice=dict(question_turn_id='task',handle=chosen['handle']))
    assert done.error.code=='TEST_SCIENCE_WITHHELD',done.to_dict()
    assert len(executor.plans)==1
    assert executor.plans[0].steps[0].arguments['control_path']==str(tmp_path/('control.tsv' if supplied else 'treatment.tsv'))
    assert sum('turn_schema_version' in c for c in model.calls)==1
    assert sum('semantic_prompt_version' in c for c in model.calls)==1
    state=app._application.sessions.load('session')
    assert state.interactions[-1].preparation['candidate_id']==candidate


def test_empty_external_file_requirement_has_upload_action(tmp_path):
    from agent.orchestration.scientific_inputs import parameter_requirements
    tool=build_default_tool_registry().get('run_macs3')
    requirement,=parameter_requirements(tool,('control_path',))
    assert requirement.kind=='external_input'
    from agent.application.scientific_preparation import _owner_requirements
    r,=_owner_requirements('candidate','step',tool,{'function_name':'bdgcmp'},
        __import__('agent.orchestration.scientific_inputs',fromlist=['ScientificInputRequired']).ScientificInputRequired(requirement))
    turn=Interaction('q','compare',None,0,{})
    start=records.start('s',turn,'compare')
    turn=replace(turn,preparation=start,admitted=dict(preparation_result=records.question(start,turn,[r])))
    action,=records.choices(turn)
    assert action['action']=='upload_source' and action['argument']=='control_path'


def test_byte_transparency_is_explicit_and_invalid_metadata_is_rejected():
    from agent.application.source_observations import identity_source
    from agent.orchestration.registry import ToolRegistry
    registry=build_default_tool_registry(); tool=registry.get('inspect_scATAC')
    selected={'inspect':('inspect_scATAC',dict(path='/same.h5ad')),
        'transform':('run_scanpy',dict(input_path='/same.h5ad',function_name='pp.log1p'))}
    assert identity_source(StepOutputRef('inspect','input_path'),selected,registry)=='/same.h5ad'
    assert identity_source(StepOutputRef('transform','matrix_path'),selected,registry) is None
    producer=replace(tool.semantic_planning.producer_ports[0],identity_from_port='absent')
    with pytest.raises(ValueError,match='transparent'):
        ToolRegistry((replace(tool,semantic_planning=replace(tool.semantic_planning,producer_ports=(producer,))),))


def test_cicero_existing_representation_offers_actual_obsm(tmp_path):
    from agent.orchestration.scientific_inputs import OperationPreparationContext,ScientificInputRequired
    path=tmp_path/'source.h5ad'; data=ad.AnnData(np.eye(4));data.obsm['X_umap']=np.ones((4,2));data.write_h5ad(path)
    registry=build_default_tool_registry(); tool=registry.get('run_cicero_cds')
    args=dict(input_path=str(path),function_name='make_cicero_cds')
    context=OperationPreparationContext(tool,'method',registry,{}, {},{},[],None,None,{})
    with pytest.raises(ScientificInputRequired) as missing:tool.preparation_owner(args,context)
    req,=missing.value.requirements
    from agent.application.scientific_preparation import source_field_offers
    offers=source_field_offers(registry,Interaction('q','make cicero',None,0,{}),(),tool,req.argument,args,{})
    assert [o['value'] for o in offers]==['X_umap']
    assert offers[0]['source_field']['axis']=='obsm'


def test_accepted_h5ad_observation_uses_actual_published_object_and_explicit_store(tmp_path,monkeypatch):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.orchestration.prior_outputs import bind_output
    from agent.tools.analysis import native_matrix
    from agent.tools import _native_process
    from agent.application.source_observations import existing_object
    path=tmp_path/'source.h5ad';ad.AnnData(np.eye(3),obs=pd.DataFrame({'original':['a','b','a']},index=['a','b','c'])).write_h5ad(path)
    def native(self,args,stage,matrix):
        matrix.obs['produced_group']=pd.Categorical(['x','y','x']);del matrix.obs['original']
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded',versions={'scanpy':'SIMULATED'},implementation=[],native_output_type='NoneType',result_slots=[],measurement='unchanged fixture')
    monkeypatch.setattr(native_matrix.MatrixPackage,'native',native)
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    root=tmp_path/'workspace';out=root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'method'
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('p',request.request_id,'fixture',(PlanStep('method','run_scanpy',dict(input_path=str(path),output_dir=str(out),function_name='pp.log1p')),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('s')
    state=app.sessions.run('s','first',AgentRequest('first','produce fixture',{}),(OutputSelection('result','method','matrix_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    binding=bind_output(app.run_store,app.registry,'first:run','method','dataset')
    reference=PriorOutputRef(binding,'matrix_path')
    observed=existing_object(reference,{},app.registry,store=app.run_store)
    assert observed.path!=str(path) and observed.accepted_binding==binding
    assert observed.fields('obs')==('produced_group',)
    from agent.application.scientific_preparation import source_field_offers
    tool=app.registry.get('evaluate_cell_clustering')
    offers=source_field_offers(app.registry,Interaction('q','evaluate',None,1,{}),(),tool,'label_key',
        dict(reference_h5ad_path=reference),{},store=app.run_store)
    assert [o['value'] for o in offers]==['produced_group']
    assert offers[0]['source_field']['accepted_binding']==binding.to_dict()
    Path(observed.path).write_bytes(Path(observed.path).read_bytes()+b'changed')
    with pytest.raises(ValueError):observed.fields('obs')
    with pytest.raises(ValueError):existing_object(reference,{},app.registry,store=app.run_store)


def test_selected_chromvar_owner_has_exact_slots_and_qualified_resources(tmp_path, monkeypatch):
    from agent.orchestration.scientific_inputs import OperationPreparationContext, ScientificInputRequired
    from agent.tools.analysis import sequence_reference
    path=tmp_path/'source.h5ad'; data=ad.AnnData(np.eye(3))
    data.uns.update(assembly='hg38', coordinate_system='zero_based_half_open', matrix_semantics='normalized')
    data.layers['counts']=np.eye(3);data.write_h5ad(path)
    entries=[dict(kind='genome',id='genome.'+str(i),label='genome '+str(i),files=[],
        resource=dict(assembly='hg38',species=dict(taxonomy_id=9606))) for i in range(2)]
    monkeypatch.setattr(sequence_reference,'qualified_genome_entries',lambda:entries)
    registry=build_default_tool_registry();tool=registry.get('compute_motif_accessibility')
    context=OperationPreparationContext(tool,'method',registry,{}, {},{},[],None,None,{})
    with pytest.raises(ScientificInputRequired) as missing:
        tool.preparation_owner(dict(input_path=str(path)),context)
    reqs={r.argument:r for r in missing.value.requirements}
    assert reqs['count_slot'].choices==('X','layers/counts')
    assert reqs['matrix_semantics'].choices==('peak_counts','fragment_counts','insertion_counts','binary_accessibility')
    assert [o['value'] for o in reqs['genome_resource'].offers]==['genome.0','genome.1']
    assert {'source:bytes','declaration:assembly','declaration:taxonomy_id'} <= set(reqs['genome_resource'].dependencies)
    assert missing.value.bindings==dict(input_path=str(path))


def test_selected_genome_retains_finite_motif_choices_and_taxonomy_conflict(tmp_path, monkeypatch):
    from agent.orchestration.scientific_inputs import OperationPreparationContext, ScientificInputRequired, ScientificDiagnostic
    from agent.tools.analysis import sequence_reference
    path=tmp_path/'source.h5ad';data=ad.AnnData(np.eye(3));data.uns.update(assembly='hg38',coordinate_system='zero_based_half_open');data.write_h5ad(path)
    entries=[dict(kind='genome',id='genome.'+str(i),label='genome '+str(i),files=[],resource=dict(assembly='hg38',species=dict(taxonomy_id=9606))) for i in range(2)]
    entries += [dict(kind='motif_collection',id='motif.'+str(i),label='motif '+str(i),files=[],resource=dict(applicable_taxonomy_ids=[9606])) for i in range(2)]
    monkeypatch.setattr(sequence_reference,'qualified_genome_entries',lambda:entries)
    registry=build_default_tool_registry();tool=registry.get('run_motifmatchr')
    context=OperationPreparationContext(tool,'method',registry,{}, {},{},[],None,None,{})
    args=dict(input_path=str(path),genome_resource='genome.0')
    with pytest.raises(ScientificInputRequired) as missing:tool.preparation_owner(args,context)
    req,=missing.value.requirements
    assert req.argument=='motif_collection' and [o['value'] for o in req.offers]==['motif.0','motif.1']
    with pytest.raises(ScientificDiagnostic,match='Declared species'):
        tool.preparation_owner(dict(args,taxonomy_id=10090),context)

    monkeypatch.setattr(sequence_reference,'qualified_genome_entries',lambda:[entries[0],entries[2]])
    with pytest.raises(ScientificDiagnostic,match='Declared species'):
        tool.preparation_owner(dict(args,taxonomy_id=10090),context)
