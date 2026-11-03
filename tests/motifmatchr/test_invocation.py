"""Simulated native outputs test integration; genuine native parity is separate."""
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from agent.tools.analysis import motifmatchr as owner


@pytest.fixture
def source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    entries = [dict(kind='genome',id='test-genome',label='Test genome',resource=dict(assembly='test',species=dict(taxonomy_id=9606))),
        dict(kind='motif_collection',id='test-motifs',label='Test motifs',resource=dict(applicable_taxonomy_ids=[9606]))]
    monkeypatch.setattr(owner,'library',lambda:SimpleNamespace(entries=lambda:entries))
    a = ad.AnnData(np.array([[1.,0.,-3.],[0.,0.,np.nan]]),
        obs=pd.DataFrame(index=['cell1','cell2']),var=pd.DataFrame(
            dict(chrom=['chr1']*3,start=[0,100,200],end=[50,150,250]),index=['peak1','zero','peak3']))
    a.uns.update(assembly='test',coordinate_system='zero_based_half_open',matrix_semantics='transformed')
    path = tmp_path/'ordinary.data';a.write_h5ad(path)
    return dict(input_path=str(path),output_dir=str(tmp_path/'out'),function_name='matchMotifs',
        kwargs={'p.cutoff':.0001,'bg':'even','out':'matches'})


@pytest.fixture
def native(tmp_path,monkeypatch):
    calls = [];metadata = tmp_path/'motifs.tsv'
    metadata.write_text('matrix_id\tclass\nMA1.1\tclass1\nMA2.1\tclass2\n')
    resource = dict(metadata=dict(path=str(metadata)),release='simulated',collection='CORE',tax_group='vertebrates',
        citation='simulated',license='simulated')
    monkeypatch.setattr(owner.sequence_reference,'resolve_motif_resources',lambda args,peaks,stage:
        ('',dict(assembly='test'),resource,dict(id='simulated'),owner.file_closure([metadata])))
    def run(args,stage,fasta,motif):
        calls.append(args)
        assert not (stage/'counts.mtx').exists()
        peaks = pd.read_csv(stage/'peaks.tsv',sep='\t')
        pd.DataFrame(dict(native_name=['MA2.1_TF2','MA1.1_TF1'],motif_id=['MA2.1','MA1.1'],name=['TF2','TF1'])).to_csv(
            stage/'motifs.tsv',sep='\t',index=False)
        mmwrite(stage/'matches.mtx',sparse.csc_matrix(np.array([[1,0],[1,1],[0,1]],dtype=bool)))
        return dict(completion='succeeded',dim=[len(peaks),2],nnz=4,rows_preserved=True,assays='motifMatches',
            versions={'motifmatchr':'simulated'},implementation=[])
    monkeypatch.setattr(owner,'native',run)
    return calls


@pytest.mark.parametrize('matrix',['transformed','absent','empty_cells'])
def test_ranges_need_no_counts_or_positive_count_filter(source,tmp_path,matrix):
    if matrix=='absent':
        with h5py.File(source['input_path'],'r+') as handle:
            del handle['X']
    if matrix=='empty_cells':
        a = ad.read_h5ad(source['input_path']);a[:0].copy().write_h5ad(source['input_path'])
    stage = tmp_path/'exchange';stage.mkdir()
    effective,peaks = owner.prepare_input(source,stage)
    assert list(peaks.feature_id)==['peak1','zero','peak3']
    assert 'matrix_semantics' not in effective and not (stage/'counts.mtx').exists()
    assert owner.parameter_prerequisites(source)==()


def test_missing_declarations_and_coordinate_conflict(source,tmp_path):
    a = ad.read_h5ad(source['input_path']);a.uns.pop('coordinate_system');a.write_h5ad(source['input_path'])
    assert owner.parameter_prerequisites(source)==('coordinate_system',)
    stage=tmp_path/'stage';stage.mkdir()
    with pytest.raises(ValueError):
        owner.prepare_input(source,stage)
    with pytest.raises(ValueError,match='conflicts'):
        owner.prepare_input(dict(source,assembly='other',coordinate_system='zero_based_half_open'),stage)


def test_missing_resources_are_specific_parameter_questions(source,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(owner,'library',lambda:SimpleNamespace(entries=lambda:[]))
    assert owner.parameter_prerequisites(source)==('genome_resource',)
    assert owner.parameter_prerequisites(dict(source,genome_path='/authorized/genome.fa'))==('taxonomy_id',)
    assert owner.parameter_prerequisites(dict(source,genome_path='/authorized/genome.fa',taxonomy_id=9606))==('motif_collection',)


def test_owner_native_condition_survives_classification_without_chromvar_requirements():
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.error_policy import public_error_message
    registry=build_default_tool_registry()
    message="'arg' should be one of “subject”, “genome”, “even”"
    error=registry.classify_exception(owner.TOOL,owner.NativeInvocationError(message))
    assert message in error.message and public_error_message(error)==error.message
    assert not error.recoverable
    unsupported=registry.classify_exception(owner.TOOL,ValueError('unreviewed detail /private/file'))
    assert '/private/file' not in public_error_message(unsupported)
    assert 'Counts and clustering are not required' in public_error_message(unsupported)


def test_cancellation_terminates_native_process_and_publishes_nothing(source,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agent.tools._cancellation import ToolWorkCancelled
    script=tmp_path/'Rscript';script.write_text('simulated process; not executed')
    monkeypatch.setenv('AGENT_CHROMVAR_RSCRIPT',str(script))
    monkeypatch.setattr(owner,'library',lambda:SimpleNamespace(root=tmp_path/'library'))
    stage=tmp_path/'stage';stage.mkdir();calls=[]
    from agent.tools import _native_process as process_owner
    class Process:
        pid=12345
        stopped=False
        def poll(self):return 0 if self.stopped else None
        def terminate(self):self.stopped=True;calls.append('terminate')
        def wait(self,timeout):calls.append('wait');return 0
    process=Process()
    monkeypatch.setattr(process_owner.subprocess,'Popen',lambda *a,**k:process)
    monkeypatch.setattr(process_owner.os,'killpg',lambda *a:process.terminate())
    def cancel():raise ToolWorkCancelled('simulated cancellation')
    monkeypatch.setattr(process_owner,'cancellation_checkpoint',cancel)
    with pytest.raises(ToolWorkCancelled):
        owner.native(source,stage,'/authorized/genome.fa',dict(pfm=dict(path='/authorized/motifs.pfm')))
    assert calls==['terminate','wait'] and not (stage/'native-receipt.json').exists()
    request=json.loads((stage/'request.json').read_text())
    assert request['kwargs']==source['kwargs']


@pytest.mark.parametrize('kwargs',[{'out':'unknown'},{'genome':'/etc/passwd'},
    {'subject':'system("touch unsafe")'},{'p.cutoff':float('nan')},{'bg':{'file':'/etc/passwd'}}])
def test_unreviewed_outputs_objects_and_non_json_values_do_not_execute(source,kwargs):
    with pytest.raises(ValueError):
        owner.execute(dict(source,kwargs=kwargs))
    assert not Path(source['output_dir']).exists()


def test_scientific_invalid_values_remain_native_responsibility(source):
    for kwargs in ({'p.cutoff':-1},{'bg':'not-a-background'},{'bg':[1,2]}):
        assert owner.native_arguments(dict(source,kwargs=kwargs))==kwargs
    with pytest.raises(ValueError,match='Only motifmatchr'):
        owner.execute(dict(source,function_name='computeDeviations'))


def test_publication_order_recovery_integrity(source,native):
    before = Path(source['input_path']).read_bytes();result = owner.execute(source,'execution-1')
    assert result['n_peaks']==3 and result['n_matches']==4
    assert result['parameters']['kwargs']==source['kwargs']
    assert pd.read_csv(Path(source['output_dir'])/'motifs.tsv',sep='\t')['class'].tolist()==['class2','class1']
    assert sparse.load_npz(result['matches_path']).toarray().tolist()==[[True,False],[True,True],[False,True]]
    assert owner.recover(source,'execution-1')==result and len(native)==1
    with pytest.raises(ValueError,match='retry'):
        owner.recover(source,'execution-2')
    assert Path(source['input_path']).read_bytes()==before
    Path(result['matches_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        owner.verify_publication(result['manifest_path'],expected_sha256=result['manifest_sha256'])


def test_accepted_delivery_restart_and_historical_sources(source,native,tmp_path):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    from agent.orchestration.prior_outputs import bind_output
    root=tmp_path/'workspace';args=dict(source,output_dir=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'matches'))
    class Planner:
        def plan(self,request,registry):
            return AgentPlan('test',request.request_id,'Simulated native motif test',(PlanStep('matches',owner.TOOL,args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('matches')
    state=app.sessions.run('matches','first',AgentRequest('first','Invoke matchMotifs',{}),
        (OutputSelection('matches','matches','manifest_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    revision=state.revisions[-1];turn=state.turn('first')
    assert bind_output(app.run_store,app.registry,turn.run_id,'matches','matches').semantic_type=='native_motif_matches.v1'
    items=accepted_artifacts(app.sessions,'matches',revision,turn,None,step_id='matches');assert len(items)==6
    for item in items:
        prepared=MatrixDownloadPreparer().prepare(item)
        try:
            import hashlib
            delivered=b''.join(prepared.iter_chunks())
            assert delivered
            assert prepared.size_bytes==len(delivered)
            assert prepared.sha256==hashlib.sha256(delivered).hexdigest()
            assert prepared.source_sha256==item.sha256
        finally:
            prepared.close()
    reopened=ResearchAgentApplication(root,planner=Planner()).sessions.load('matches')
    assert reopened.to_dict()==state.to_dict() and len(native)==1
    Path(source['input_path']).unlink()
    assert len(accepted_artifacts(app.sessions,'matches',revision,turn,None,step_id='matches'))==6
    assert len(native)==1


@pytest.mark.parametrize('literal',['{"p.cutoff": 1e-4, "bg": "even"}', '{}', '{"bg": [0.2, 0.3, 0.3, 0.2]}'])
def test_exact_json_declaration_through_existing_binder(literal):
    from agent.application.dialogue_execution import _bind_arguments,scientific_parameters
    from agent.application.turn_decisions import ScopedArgument
    from agent.orchestration import build_default_tool_registry
    registry=build_default_tool_registry();utterance='Use kwargs '+literal+'.'
    declaration=ScopedArgument(owner.TOOL,'kwargs',literal,11,11+len(literal))
    assert _bind_arguments(registry,utterance,{},(declaration,))==dict(kwargs=json.loads(literal))
    assert next(p for p in scientific_parameters(registry) if p['tool']==owner.TOOL and p['argument']=='kwargs')['json_object']


@pytest.mark.parametrize('literal',['{"bg":"even", "bg":"subject"}','{"p.cutoff":NaN}','{"p.cutoff":1e999}',
    "{'p.cutoff':0.001}",'__import__("os").system("true")','[]'])
def test_json_lexical_admission_rejects_loss_or_code(literal):
    from agent.application.turn_decisions import ScopedArgument,admit_scientific_argument,IntentError
    from agent.orchestration import build_default_tool_registry
    spec=build_default_tool_registry().get(owner.TOOL).optional_arguments['kwargs']
    declaration=ScopedArgument(owner.TOOL,'kwargs',literal,0,len(literal))
    with pytest.raises(IntentError):
        admit_scientific_argument(declaration,literal,spec)


@pytest.mark.parametrize('structured', [False, True])
@pytest.mark.parametrize('initial_coordinate', [False, True])
@pytest.mark.parametrize('native_scope', [False, True])
def test_normal_interpreter_compiler_clarification_and_exact_kwargs(source,native,tmp_path,structured,initial_coordinate,native_scope,monkeypatch):
    from agent.application import InteractiveAgentApplication
    from agent.application.local_resources import LocalResourceAdmission
    from agent.providers import PlanningModelFactoryRegistry
    from agent.orchestration import PlanningModelProfile
    from agent.application.session_state import AnalysisSession
    from agent.schemas.orchestration import _serialize
    a=ad.read_h5ad(source['input_path']);a.uns.pop('coordinate_system')
    if structured:a.X=None
    a.write_h5ad(source['input_path'])
    literal='{"p.cutoff": 0.0001, "bg": "even", "out": "matches"}'
    prompt='Run motifmatchr::matchMotifs() with kwargs '+literal+'.'
    reply='These peaks use zero-based half-open coordinates.'
    decisions={prompt:dict(kind='execute_plan',target=owner.TOOL,missing_parameters=[],
        roles=[dict(tool=owner.TOOL,port='dataset',source='source.0')],arguments=[
            dict(tool=owner.TOOL,argument='function_name',literal='motifmatchr::matchMotifs()',evidence='motifmatchr::matchMotifs()',value='matchMotifs'),
            dict(tool=owner.TOOL,argument='kwargs',literal=literal,evidence='kwargs '+literal)]),
        reply:dict(kind='answer_parameters',pending='@parameters',arguments=[dict(tool=owner.TOOL,argument='coordinate_system',
            literal='zero-based half-open',evidence='zero-based half-open',value='zero_based_half_open')])}
    if structured:
        old=prompt
        prompt="Run motifmatchr::matchMotifs() with p.cutoff=0.0001, bg='even', out='matches'."
        decision=decisions.pop(old)
        decision['arguments'][1]=json.loads(structured_wire(prompt,[
            ('p.cutoff','number','0.0001','p.cutoff=0.0001'),
            ('bg','string','even',"bg='even'"),('out','string','matches',"out='matches'")],scoped=True))['decision']['arguments'][0]
        decisions[prompt]=decision
    if initial_coordinate:
        decision=decisions.pop(prompt)
        prompt += ' Coordinates are zero-based half-open.'
        decision['arguments'].append(dict(tool=owner.TOOL,argument='coordinate_system',
            literal='zero-based half-open',evidence='zero-based half-open',value='zero_based_half_open'))
        decisions[prompt]=decision
    if native_scope:
        for decision in decisions.values():
            for declaration in decision['arguments']:
                declaration['native_function']='matchMotifs'
    class Model:
        model_id='scripted/native-motif'
        def complete(self,*,prompt,response_schema):
            v=json.loads(prompt)
            if 'turn_schema_version' in v:
                return json.dumps(dict(turn_schema_version=1,decision=decisions[v['utterance']]))
            if 'output_selection_schema_version' in v:
                return json.dumps(dict(outputs=[dict(name='matches',step_id='matches',output_key='manifest_path')]))
            if 'schema_version' in response_schema.get('properties',{}):
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(step_id='matches',tool=owner.TOOL,
                    sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[])])))
            raise RuntimeError('Optional prose is outside simulated integration qualification')
    profile=PlanningModelProfile('test','scripted','scripted/native-motif');root=tmp_path/'work'
    app=InteractiveAgentApplication(root,model_profiles=(profile,),default_profile_id='test',approved_source_roots=(tmp_path,),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda p:Model()}))
    app.create_session('session');local=LocalResourceAdmission(root,approved_source_roots=(tmp_path,))
    record=local.register('peaks',source['input_path'],input_type='h5ad',label='ordinary',attribution='researcher')
    group=local.compose_group([record.resource_id],attachment_candidates=True)
    first=app.submit_turn('session','first',prompt,expected_generation=0,registered_input=group).to_dict()
    state=app._application.sessions.load('session')
    if initial_coordinate:
        done=first
        assert first['response']['kind']!='clarify',first
    else:
        assert first['response']['kind']=='clarify',first
        assert not state.turns and not native and 'coordinate' in first['response']['text']
        assert AnalysisSession.from_dict(_serialize(state.to_dict()))==state
        if native_scope and structured:
            from agent.application import dialogue_execution as execution
            with monkeypatch.context() as old:
                old.setattr(execution,'_missing_parameter_inputs',lambda registry,tool,names,values:
                    [name for name in names if execution._parameter_input(registry,tool,name)[0] not in values])
                partial=app.submit_turn('session','legacy-partial',reply,expected_generation=0).to_dict()
                assert partial['response']['kind']=='clarify' and not native
                historical=app._application.sessions.load('session')
            assert app._application.sessions.load('session')==historical
        done=app.submit_turn('session','reply',reply,expected_generation=0).to_dict()
        state=app._application.sessions.load('session')
        if native_scope and structured:
            assert state.interactions[:len(historical.interactions)]==historical.interactions
    assert len(native)==1 and len(state.revisions)==1,done
    run=app._application.run_store.load(state.turns[-1].run_id)
    assert run.steps[0].resolved_arguments['kwargs']==source['kwargs']
    assert run.steps[0].resolved_arguments['input_path']==source['input_path']
    assert AnalysisSession.from_dict(_serialize(state.to_dict()))==state


def structured_wire(text, members, *, scoped=False):
    declaration=dict(tool=owner.TOOL,argument='kwargs',
        members=[dict(name=name,kind=kind,literal=literal,evidence=evidence)
                 for name,kind,literal,evidence in members])
    if scoped:declaration.update(source='source.0',port='dataset')
    return json.dumps(dict(turn_schema_version=1,decision=dict(kind='execute_plan',target=owner.TOOL,
        arguments=[declaration])))


@pytest.mark.parametrize('text,members,expected',[
    ("Use p.cutoff=.0001, bg='even', out='matches'.",
     [('p.cutoff','number','.0001','p.cutoff=.0001'),('bg','string','even',"bg='even'"),
      ('out','string','matches',"out='matches'")],{'p.cutoff':.0001,'bg':'even','out':'matches'}),
    ('Set p cutoff to 0.0001, background even, and return the matches matrix.',
     [('p.cutoff','number','0.0001','p cutoff to 0.0001'),('bg','string','even','background even'),
      ('out','string','matches','return the matches matrix')],{'p.cutoff':.0001,'bg':'even','out':'matches'}),
    ('Use p.cutoff=2.', [('p.cutoff','number','2','p.cutoff=2')],{'p.cutoff':2}),
    ("Use p.cutoff='not-a-number'.", [('p.cutoff','string','not-a-number',"p.cutoff='not-a-number'")],{'p.cutoff':'not-a-number'}),
    ('Use background [0.2, 0.3, 0.3, 0.2].', [('bg','json','[0.2, 0.3, 0.3, 0.2]','background [0.2, 0.3, 0.3, 0.2]')],{'bg':[.2,.3,.3,.2]}),
])
def test_grounded_members_reuse_the_parameter_binder(text,members,expected):
    from dataclasses import asdict
    from agent.application.turn_decisions import parse_decision,_scientific_arguments
    from agent.application.dialogue_execution import _bind_arguments
    from agent.orchestration import build_default_tool_registry
    decision=parse_decision(structured_wire(text,members,scoped=True),utterance=text)
    declaration=decision.arguments[0]
    assert declaration.source=='source.0' and declaration.port=='dataset'
    assert _scientific_arguments([asdict(declaration)])==decision.arguments
    actual=_bind_arguments(build_default_tool_registry(),text,{},decision.arguments)
    assert actual=={'kwargs':expected}
    assert owner.native_arguments(dict(function_name='matchMotifs',**actual))==expected


@pytest.mark.parametrize('text,members',[
    ('Use background even.', [('bg','string','genome','background even')]),
    ('Use background even.', [('genome','string','even','background even')]),
    ('Use p.cutoff=0.0001.', [('p.cutoff','number','0.000','p.cutoff=0.0001')]),
    ('Use background even or genome.', [('bg','string','even','background even'),('bg','string','genome','or genome')]),
    ('Use background [0.2, NaN].', [('bg','json','[0.2, NaN]','background [0.2, NaN]')]),
])
def test_structured_members_do_not_repair_forgery_conflicts_or_unsafe_data(text,members):
    from agent.application.turn_decisions import parse_decision,IntentError
    from agent.application.dialogue_execution import _bind_arguments
    from agent.orchestration import build_default_tool_registry
    with pytest.raises(IntentError):
        d=parse_decision(structured_wire(text,members),utterance=text)
        _bind_arguments(build_default_tool_registry(),text,{},d.arguments)


def test_missing_file_offers_distinguish_required_input_from_optional_genome():
    from agent.application.local_resources import group_choices,RegisteredInputGroup
    from agent.orchestration import build_default_tool_registry
    roles=group_choices(RegisteredInputGroup((),{},attachment_candidates=True),build_default_tool_registry())['roles']
    roles={r['port']:r for r in roles if r['tool']==owner.TOOL}
    assert roles['dataset']['required'] is True
    assert roles['genome']['required'] is False
    assert 'Optional' in roles['genome']['description']



def test_missing_file_schema_can_request_one_required_dataset_without_optional_resources():
    from agent.application.turn_decisions import decision_schema
    schema=decision_schema()
    requests=[v['properties'] for v in schema['properties']['decision']['anyOf']
        if v['properties']['kind']['enum']==['request_roles'] and 'required_roles' in v['properties']]
    assert requests and requests[0]['required_roles']['minItems']==1


@pytest.mark.parametrize('mode,files',[('scores',9),('positions',6)])
def test_mode_specific_publication_acceptance_and_download(source,native,tmp_path,monkeypatch,mode,files):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    old_native=owner.native
    def invoke(args,stage,*resources):
        receipt=old_native(args,stage,*resources)
        (stage/'native-result.rds').write_bytes(b'SIMULATED native result; parity qualified separately')
        if mode=='scores':
            mmwrite(stage/'scores.mtx',sparse.csc_matrix([[1.,0.],[2.,3.],[0.,4.]]))
            mmwrite(stage/'counts.mtx',sparse.csc_matrix([[1,0],[2,1],[0,1]]))
            receipt['assays']=['motifScores','motifMatches','motifCounts']
        else:
            (stage/'positions.tsv').write_text('motif_index\tnative_name\tseqnames\tstart\tend\twidth\tstrand\tscore\n0\tMA2.1_TF2\tchr1\t1\t10\t10\t-\t7.5\n')
            receipt['n_positions']=1
        return receipt
    monkeypatch.setattr(owner,'native',invoke)
    root=tmp_path/'workspace';args=dict(source,kwargs=dict(source['kwargs'],out=mode),output_dir=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'motifs'))
    class Planner:
        def plan(self,request,registry):return AgentPlan('first',request.request_id,'simulated mode',(PlanStep('native',owner.TOOL,args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('modes')
    state=app.sessions.run('modes','first',AgentRequest('first','Native '+mode,{}),(OutputSelection('result','native','manifest_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    path=Path(args['output_dir'])/'manifest.json';value,sha=owner.load(path);out=owner.result(value,path,sha)
    if mode=='positions':
        assert out['n_positions']==1 and out['n_matches'] is None and out['matches_path'] is None
        assert not (path.parent/'motif-matches.npz').exists()
    else:assert out['n_matches']==4
    revision=state.revisions[-1];turn=state.turn('first')
    artifacts=accepted_artifacts(app.sessions,'modes',revision,turn,None,step_id='native')
    assert len(artifacts)==files
    for artifact in artifacts:
        prepared=MatrixDownloadPreparer().prepare(artifact)
        try:
            import hashlib
            data=b''.join(prepared.iter_chunks())
            assert len(data)==prepared.size_bytes and hashlib.sha256(data).hexdigest()==prepared.sha256
        finally:prepared.close()
    Path(source['input_path']).unlink()
    reopened=ResearchAgentApplication(root,planner=Planner())
    assert reopened.sessions.load('modes').to_dict()==state.to_dict()
    assert len(accepted_artifacts(reopened.sessions,'modes',revision,turn,None,step_id='native'))==files
    assert len(native)==1
    (path.parent/'native-result.rds').write_bytes(b'changed')
    with pytest.raises((ValueError,RuntimeError)):
        MatrixDownloadPreparer().prepare(next(a for a in artifacts if a.role=='native-result.rds'))
