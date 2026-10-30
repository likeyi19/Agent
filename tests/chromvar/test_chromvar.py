"""Owner/authority regression with simulated native outputs, not science qualification."""
from pathlib import Path
import json
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread,mmwrite
from agent.tools.analysis import chromvar as owner


@pytest.fixture
def source(tmp_path):
    x=np.array([[2,0,0,1],[0,3,0,2],[1,1,0,1],[0,0,0,0]])
    a=ad.AnnData(sparse.csr_matrix(x),obs=pd.DataFrame({'cluster':pd.Categorical(['1','1','2','2'])},index=['a','b','c','d']),
        var=pd.DataFrame({'chrom':['chr1']*4,'start':[0,200,400,600],'end':[100,300,500,700]},index=['p1','p2','zero','p4']))
    a.uns.update(assembly='test',coordinate_system='zero_based_half_open',matrix_semantics='insertion_counts')
    path=tmp_path/'ordinary.data';a.write_h5ad(path)
    return a,dict(input_path=str(path),output_dir=str(tmp_path/'out'),group_key='cluster')


@pytest.fixture
def native(tmp_path,monkeypatch):
    calls=[];metadata=tmp_path/'motif_metadata.tsv'
    metadata.write_text('matrix_id\tclass\tfamily\ttax_id\tspecies\nMA1.1\tx\ty\t9606\thuman\nMA2.1\tx\ty\t9606\thuman\n')
    resource=dict(metadata=dict(path=str(metadata)),release='simulated',collection='CORE',tax_group='vertebrates',license='simulated',citation='simulated')
    monkeypatch.setattr(owner,'resolve_resources',lambda args,peaks,stage:('',dict(assembly='test'),resource,dict(id='simulated'),owner.file_closure([metadata])))
    def run(args,stage,fasta,motif):
        calls.append(1);x=mmread(stage/'counts.mtx').tocsr();keep=np.flatnonzero(np.asarray(x.sum(1)).ravel()>0)
        n=x.shape[1];m=2
        pd.DataFrame(dict(source_index=keep,gc=[.3,.4,.5])).to_csv(stage/'retained.tsv',sep='\t',index=False)
        pd.DataFrame(dict(motif_id=['MA1.1','MA2.1'],name=['TF1','TF2'],fraction_matches=[.3,.5],fraction_background_overlap=[.2,.4])).to_csv(stage/'motifs.tsv',sep='\t',index=False)
        z=np.arange(n*m,dtype=float).reshape(m,n);z[:,-1]=np.nan
        z.ravel(order='F').astype('<f8').tofile(stage/'z.bin');(z/10).ravel(order='F').astype('<f8').tofile(stage/'deviations.bin')
        np.tile(np.arange(1,len(keep)+1),2).astype('<i4').tofile(stage/'background.bin')
        mmwrite(stage/'matches.mtx',sparse.csr_matrix(np.ones((len(keep),m),dtype=int)))
        pd.DataFrame(dict(source_index=keep,sequence_sha256=['0'*64]*len(keep))).to_csv(stage/'sequence-identities.tsv',sep='\t',index=False)
        return dict(completion='succeeded',input_dim=list(x.shape),retained_dim=[len(keep),n],background_iterations=2,seed=1,p_cutoff=5e-5,motif_background='subject',filtering={'sample_filter':'none'},versions={'chromVAR':'simulated'},implementation=[])
    monkeypatch.setattr(owner,'native',run)
    return calls


@pytest.mark.parametrize('kind',['dense','csr','csc'])
def test_native_anndata_counts_and_slot_scope(source,tmp_path,kind):
    a,args=source;raw=a.X.toarray()
    a.X=np.log1p(raw);a.uns['matrix_semantics']='log1p_counts'
    a.layers['counts']=raw if kind=='dense' else getattr(sparse,kind+'_matrix')(raw)
    a.write_h5ad(args['input_path']);stage=tmp_path/'exchange';stage.mkdir()
    effective,obs,peaks,values=owner.prepare_input(dict(args,count_slot='layers/counts',matrix_semantics='insertion_counts'),stage)
    np.testing.assert_array_equal(values.toarray(),raw)
    assert list(obs.index)==list(a.obs_names) and effective['assembly']=='test'
    assert list(peaks.start)==[0,200,400,600]
    with pytest.raises(ValueError,match='raw peak'):owner.prepare_input(args,stage)


def test_invalid_counts_coordinates_and_missing_group(source,tmp_path):
    a,args=source;stage=tmp_path/'exchange';stage.mkdir()
    for bad in [-1,np.nan,.5]:
        a.X=a.X.astype(float);a.X.data[0]=bad;a.write_h5ad(args['input_path'])
        with pytest.raises(ValueError,match='nonnegative integer'):owner.prepare_input(args,stage)
    a.X.data[0]=1;a.uns.pop('coordinate_system');a.write_h5ad(args['input_path'])
    with pytest.raises(ValueError):owner.prepare_input(args,stage)


def test_ordinary_result_undefined_mapping_and_tamper(source,native):
    a,args=source;before=Path(args['input_path']).read_bytes();r=owner.execute(args)
    out=ad.read_h5ad(r['scores_path']);peaks=pd.read_csv(Path(args['output_dir'])/'peaks.tsv',sep='\t')
    assert list(out.obs_names)==list(a.obs_names) and list(out.var_names)==['MA1.1','MA2.1']
    assert np.isnan(out.X[-1]).all() and np.isnan(out.layers['z_scores'][-1]).all()
    assert r['undefined_values']==dict(deviations=2,z_scores=2)
    assert list(peaks.retained)==[True,True,False,True]
    assert Path(args['input_path']).read_bytes()==before
    assert owner.recover(args,None)==r and len(native)==1
    with pytest.raises(ValueError,match='not empty'):owner.execute(args)
    Path(r['scores_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):owner.verify_publication(r['manifest_path'],expected_sha256=r['manifest_sha256'])


def test_accepted_delivery_restart_does_not_run_native(source,native,tmp_path):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.dialogue_evidence import figure_inventory
    from agent.application.chromvar_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.orchestration.prior_outputs import bind_output
    a,args=source;root=tmp_path/'workspace'
    args['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'motifs')
    class Planner:
        def plan(self,request,registry):return AgentPlan('test',request.request_id,'Simulated motif test',(PlanStep('motifs',owner.TOOL,args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('motifs')
    state=app.sessions.run('motifs','first',AgentRequest('first','Examine motif accessibility',{}),
        (OutputSelection('motifs','motifs','manifest_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    revision=state.revisions[-1];turn=state.turn('first')
    assert bind_output(app.run_store,app.registry,turn.run_id,'motifs','motifs').semantic_type=='motif_accessibility.v1'
    assert len(figure_inventory(app.sessions,turn,revision))==1
    items=accepted_artifacts(app.sessions,'motifs',revision,turn,None,step_id='motifs')
    assert len(items)==11
    for item in items:
        prepared=MatrixDownloadPreparer().prepare(item)
        try:assert b''.join(prepared.iter_chunks())
        finally:prepared.close()
    reopened=ResearchAgentApplication(root,planner=Planner()).sessions.load('motifs')
    assert reopened.to_dict()==state.to_dict() and len(native)==1
    # Historical accepted reuse does not require the original upload to survive.
    Path(args['input_path']).unlink()
    assert len(accepted_artifacts(app.sessions,'motifs',revision,turn,None,step_id='motifs'))==11
    assert len(native)==1


def test_h5ad_companions_follow_registry_and_preserve_selected_tool_constraints(tmp_path):
    from agent.application.local_resources import LocalResourceAdmission,ResourceAdmissionError
    from agent.orchestration.registry import build_default_tool_registry
    path=tmp_path/'source';path.write_bytes(b'not parsed during registration')
    admission=LocalResourceAdmission(tmp_path/'workspace',approved_source_roots=(tmp_path,))
    record=admission.register('ordinary',path,input_type='h5ad',label='Ordinary',attribution='user')
    values=admission._h5ad_inputs(record,dict(matrix_semantics='peak_counts',count_slot='layers/counts',p_cutoff=.0001))
    assert values['matrix_semantics']=='peak_counts'
    assert values['input_path']==str(path)
    # No source locator can sneak in as a generic scientific parameter.
    with pytest.raises(ResourceAdmissionError):admission._h5ad_inputs(record,dict(genome_path='/unselected/genome.fa'))
    registry=build_default_tool_registry()
    old=registry.get('adopt_scATAC_cell_by_features')
    with pytest.raises(ValueError):old.required_arguments['matrix_semantics'].validate('matrix_semantics','peak_counts')


def test_metadata_observations_are_scoped_and_share_owner_reading(source):
    a,args=source
    observed=owner.observe_input(args['input_path'])
    assert observed['missing_declarations']==[]
    assert observed['recorded_declarations']['matrix_semantics']=='insertion_counts'
    a.layers['raw']=a.X.copy();a.uns.pop('coordinate_system');a.write_h5ad(args['input_path'])
    observed=owner.observe_input(args['input_path'])
    assert observed['missing_declarations']==['coordinate_system']
    assert observed['slots']==['X','layers/raw']


def test_task_genome_uses_private_index_and_exact_contigs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    fa=tmp_path/'ordinary.fa';fa.write_text('>chr1\n'+('ACGT'*200)+'\n');before=fa.read_bytes()
    motif=dict(applicable_taxonomy_ids=[9606])
    entry=dict(kind='motif_collection',id='collection',label='test',resource=motif,files=[])
    monkeypatch.setattr(owner,'library',lambda:SimpleNamespace(entries=lambda:[entry],check=lambda e:e['resource']))
    stage=tmp_path/'stage';stage.mkdir()
    peaks=pd.DataFrame(dict(feature_id=['p'],chrom=['chr1'],start=[1],end=[10]))
    args=dict(genome_path=str(fa),assembly='declared',taxonomy_id=9606)
    fasta,genome,_,_,_=owner.resolve_resources(args,peaks,stage)
    assert Path(fasta).is_relative_to(stage) and genome['source']=='task FASTA'
    assert not Path(str(fa)+'.fai').exists() and fa.read_bytes()==before
    assert len(genome['fai_sha256'])==64
    other=tmp_path/'invalid';other.mkdir();peaks.chrom='missing'
    with pytest.raises(ValueError,match='absent contigs'):owner.resolve_resources(args,peaks,other)


def test_named_clarification_uses_existing_contract_and_plain_scientific_wording():
    from agent.application.turn_decisions import parse_decision,Clarify,IntentError
    from agent.application.responses import clarification_text
    names=['compute_motif_accessibility.matrix_semantics','compute_motif_accessibility.coordinate_system']
    raw=json.dumps(dict(turn_schema_version=1,decision=dict(kind='clarify',reason='missing_parameter_value',choices=names)))
    value=parse_decision(raw)
    assert value==Clarify('missing_parameter_value',tuple(names),True)
    text=clarification_text(value)
    assert 'cut-site' in text and 'zero-based half-open' in text and owner.TOOL not in text
    old=json.dumps(dict(turn_schema_version=1,decision=dict(kind='clarify',reason='missing_parameter_value')))
    assert parse_decision(old)==Clarify('missing_parameter_value',(),True)
    with pytest.raises(IntentError):parse_decision(raw.replace('compute_motif_accessibility.matrix_semantics','/untrusted/path'))


def test_undefined_group_means_and_missing_labels_remain_explicit():
    z=np.array([[1.,2.],[np.nan,np.nan],[3.,4.]])
    motifs=pd.DataFrame(dict(motif_id=['M1','M2'],name=['T1','T2']))
    obs=pd.DataFrame({'group':pd.Series(['positive','empty',pd.NA],dtype='string')})
    table,groups,top,examples,order=owner._summaries(z,motifs,obs,'group')
    empty=[r for r in examples if r['group']=='empty']
    assert len(empty)==2 and all(r['finite_cells']==0 and r['mean_z'] is None for r in empty)
    json.dumps(examples,allow_nan=False)


def test_receipt_download_omits_embedded_runtime_paths():
    import io
    from agent.application.chromvar_delivery import prepare
    receipt=dict(session=['BLAS: /private/runtime/libblas.so'],versions={'chromVAR':'1.32.0'},p_cutoff=5e-5)
    for value in (receipt,dict(receipt=receipt)):
        public=json.loads(prepare(io.BytesIO(json.dumps(value).encode()),'native-receipt.json'))
        record=public.get('receipt',public)
        assert 'session' not in record and record['p_cutoff']==5e-5
        assert record['versions']==receipt['versions']


def test_exact_layer_name_needs_no_internal_path_expression(source,tmp_path):
    a,args=source;raw=a.X.copy();a.layers['raw_counts']=raw;a.X=np.log1p(raw.toarray())
    a.uns['matrix_semantics']='log1p_counts';a.write_h5ad(args['input_path'])
    stage=tmp_path/'exchange';stage.mkdir()
    effective,obs,peaks,x=owner.prepare_input(dict(args,count_slot='raw_counts',matrix_semantics='insertion_counts'),stage)
    assert effective['count_slot']=='layers/raw_counts' and (x!=raw).nnz==0
    with pytest.raises(ValueError,match='Declare actual raw'):
        owner.prepare_input(dict(args,count_slot='raw_counts'),stage)


@pytest.mark.parametrize("provider_interruption,transformed",[(False,True),(True,True),(False,False)])
def test_parameter_question_preserves_objective_source_and_natural_layer_reply(source,native,tmp_path,provider_interruption,transformed):
    from agent.application import InteractiveAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    from agent.providers import PlanningModelFactoryRegistry
    from agent.orchestration import PlanningModelProfile
    from agent.application.session_state import AnalysisSession,SessionError
    from agent.schemas.orchestration import _serialize
    a,args=source
    if transformed:
        raw=a.X.copy();a.layers['raw_counts']=raw;a.X=np.log1p(raw.toarray())
        a.uns['matrix_semantics']='log1p_counts';a.write_h5ad(args['input_path'])
    prompt='Analyze motif accessibility across these cells using seed 7.'
    reply='My raw insertion counts are in the raw_counts layer.'
    decisions={prompt:dict(kind='execute_plan',target=owner.TOOL,roles=[dict(tool=owner.TOOL,port='dataset',source='source.0')],
        missing_parameters=[],arguments=[dict(tool=owner.TOOL,argument='seed',literal='7',evidence='seed 7')]),reply:dict(kind='answer_parameters',pending='@parameters',arguments=[
            dict(tool=owner.TOOL,argument='count_slot',literal='raw_counts',evidence='raw_counts'),
            dict(tool=owner.TOOL,argument='matrix_semantics',literal='raw insertion counts',evidence='raw insertion counts',value='insertion_counts')])}
    calls=[];interrupted=[]
    class Model:
        model_id='scripted/motif'
        def complete(self,*,prompt,response_schema):
            v=json.loads(prompt);calls.append(v)
            if 'turn_schema_version' in v:
                if provider_interruption and v['utterance']==reply and not interrupted:
                    from agent.orchestration.planning_model import PlanningModelError
                    interrupted.append(True)
                    raise PlanningModelError('Simulated incomplete response',code='PROVIDER_COMPLETION_INCOMPLETE')
                return json.dumps(dict(turn_schema_version=1,decision=decisions[v['utterance']]))
            if 'output_selection_schema_version' in v:return json.dumps(dict(outputs=[dict(name='motifs',step_id='motifs',output_key='manifest_path')]))
            if 'schema_version' in response_schema.get('properties',{}):
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='motifs',tool=owner.TOOL,
                    sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[])])))
            raise RuntimeError('Optional prose is outside this simulated-native continuation test')
    profile=PlanningModelProfile('test','scripted','scripted/motif');root=tmp_path/'work'
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='test',approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:Model()}))
    app.create_session('session');local=LocalResourceAdmission(root,approved_source_roots=(tmp_path,))
    record=local.register('matrix',args['input_path'],input_type='h5ad',label='ordinary',attribution='researcher')
    group=local.compose_group([record.resource_id],attachment_candidates=True)
    first=app.submit_turn('session','first',prompt,expected_generation=0,registered_input=group).to_dict()
    if not transformed:
        state=app._application.sessions.load('session')
        assert len(native)==1 and len(state.revisions)==1,first
        assert state.interactions[0].prerequisite is None
        return
    assert first['response']['kind']=='clarify',first
    state=app._application.sessions.load('session');assert not state.turns and not state.revisions and not native
    assert 'count' in first['response']['text'] and 'compute_motif_accessibility' not in first['response']['text']
    assert AnalysisSession.from_dict(_serialize(state.to_dict()))==state
    forged=_serialize(state.to_dict());forged['interactions'][0]['prerequisite']['missing']=['genome_path']
    with pytest.raises(SessionError):AnalysisSession.from_dict(forged)
    if provider_interruption:
        failed=app.submit_turn('session','failed_reply',reply,expected_generation=0).to_dict()
        interrupted_state=app._application.sessions.load('session')
        assert failed['response']['error']['code']=='PROVIDER_COMPLETION_INCOMPLETE'
        assert interrupted_state.interactions[-1].prerequisite['origin_turn_id']=='first'
        assert not interrupted_state.turns and not native
        assert AnalysisSession.from_dict(_serialize(interrupted_state.to_dict()))==interrupted_state
    done=app.submit_turn('session','reply',reply,expected_generation=0).to_dict()
    state=app._application.sessions.load('session')
    assert len(native)==1 and len(state.revisions)==1,done
    run=app._application.run_store.load(state.turns[-1].run_id)
    assert run.steps[0].resolved_arguments['seed']==7
    assert run.steps[0].resolved_arguments['count_slot']=='raw_counts'
    assert run.steps[0].result['count_slot']=='layers/raw_counts'
    assert run.steps[0].resolved_arguments['input_path']==args['input_path']
    assert prompt in run.request.prompt


def test_missing_parameter_wire_does_not_change_source_role_question_schema():
    from agent.application.turn_decisions import decision_schema,parse_decision,IntentError
    public=dict(bases={},relations=[],dialogue=dict(outputs=[],tools=[owner.TOOL],scientific_parameters=[
        dict(tool=owner.TOOL,argument='count_slot',required=False,description='Selected count slot',exact_string=True)],
        selected_sources=[dict(source='source.0',position=1,label='matrix',input_type='h5ad')],
        input_roles=[dict(tool=owner.TOOL,port='dataset')],attachment_candidates=True))
    variants=decision_schema(public)['properties']['decision']['anyOf']
    for variant in variants:
        properties=variant['properties'];kind=properties['kind']['enum'][0]
        if kind=='execute_plan':assert 'missing_parameters' in properties
        if kind=='request_roles':assert 'missing_parameters' not in properties
    with pytest.raises(IntentError):
        parse_decision(json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target=owner.TOOL,
            missing_parameters=['count_slot','count_slot']))))


def test_owner_questions_follow_selected_slot_metadata_without_layer_inference(source):
    from agent.orchestration.registry import build_default_tool_registry
    a,args=source
    assert owner.parameter_prerequisites(args)==()
    a.layers['raw_counts']=a.X.copy();a.X=np.log1p(a.X.toarray());a.uns['matrix_semantics']='log1p_counts'
    a.write_h5ad(args['input_path'])
    assert owner.parameter_prerequisites(args)==('count_slot','matrix_semantics')
    assert owner.parameter_prerequisites(dict(args,count_slot='raw_counts'))==('matrix_semantics',)
    assert owner.parameter_prerequisites(dict(args,count_slot='raw_counts',matrix_semantics='insertion_counts'))==()
    with pytest.raises(ValueError,match='absent'):
        owner.parameter_prerequisites(dict(args,count_slot='invented'))
    a.uns.clear();a.write_h5ad(args['input_path'])
    assert owner.parameter_prerequisites(args)==('assembly','coordinate_system','matrix_semantics')
    assert owner.parameter_prerequisites({})==()
    registry=build_default_tool_registry()
    assert [n for n in registry.names() if registry.get(n).parameter_prerequisites is not None]==[owner.TOOL]
