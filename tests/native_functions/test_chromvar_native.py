"""Software-boundary tests; installed-package parity lives in acceptance evidence."""
import hashlib
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmread
from agent.tools.analysis import chromvar_native as owner


@pytest.fixture
def source(tmp_path):
    a=ad.AnnData(sparse.csr_matrix([[0,2,0],[3,0,1]],dtype=np.int32),
        obs=pd.DataFrame({'group':['B','A']},index=['002','001']),var=pd.DataFrame(index=['03','01','02']))
    a.uns['matrix_semantics']='insertion_counts'
    p=tmp_path/'ordinary.data';a.write_h5ad(p)
    return dict(input_path=str(p),output_dir=str(tmp_path/'out'),function_name='computeExpectations',kwargs={'norm':False})


@pytest.fixture
def native(monkeypatch):
    calls=[]
    def invoke(args,stage):
        calls.append(args)
        function=args['function_name'];cells=pd.read_csv(stage/'cells.tsv',sep='\t',dtype=str)
        features=pd.read_csv(stage/'features.tsv',sep='\t',dtype=str)
        axis=cells.cell_id.tolist() if function=='getFragmentsPerSample' else features.feature_id.tolist()
        if function=='computeVariability':
            table=pd.DataFrame(dict(source_index=range(len(axis)),source_id=axis,variability=[1.]*len(axis)))
        else:
            x=mmread(stage/'counts.mtx').tocsr();values=np.asarray(x.sum(0 if function=='getFragmentsPerSample' else 1)).ravel()
            if function=='computeExpectations':values=values/x.sum()
            table=pd.DataFrame(dict(source_index=range(len(axis)),source_id=axis,value=values))
        table.to_csv(stage/'result.tsv',sep='\t',index=False)
        return dict(completion='succeeded',input_dim=[len(features),len(cells)],rows=len(table),columns=list(table),
            native_output_type='data.frame' if function=='computeVariability' else 'numeric',undefined_values=0,
            result_axis='cells' if function=='getFragmentsPerSample' else 'features',
            versions={'chromVAR':'SIMULATED'},implementation=[])
    monkeypatch.setattr(owner,'native',invoke)
    return calls


@pytest.mark.parametrize('storage',['dense','csr','csc'])
def test_counts_axes_zero_features_and_no_genome_requirement(source,tmp_path,storage):
    a=ad.read_h5ad(source['input_path']);a.X={'dense':a.X.toarray(),'csr':a.X,'csc':a.X.tocsc()}[storage];a.write_h5ad(source['input_path'])
    stage=tmp_path/'exchange';stage.mkdir();effective,n,m=owner.prepare_input(source,stage)
    assert (n,m)==(2,3) and effective['measurement']=='insertion_counts'
    assert mmread(stage/'counts.mtx').toarray().tolist()==[[0,3],[2,0],[0,1]]
    assert pd.read_csv(stage/'features.tsv',sep='\t',dtype=str).feature_id.tolist()==['03','01','02']


def test_layers_keep_independent_measurement(source,tmp_path):
    a=ad.read_h5ad(source['input_path']);a.layers['counts']=a.X;a.X=np.log1p(a.X.toarray());a.uns['matrix_semantics']='log1p_counts';a.write_h5ad(source['input_path'])
    stage=tmp_path/'exchange';stage.mkdir()
    effective,*_=owner.prepare_input(dict(source,count_slot='counts'),stage)
    assert effective['measurement']=='unspecified_measurement'
    effective,*_=owner.prepare_input(dict(source,count_slot='counts',matrix_semantics='insertion_counts'),stage)
    assert effective['count_slot']=='layers/counts'


@pytest.mark.parametrize('function,kwargs',[
    ('computeExpectations',{'object':'/etc/passwd'}),('computeExpectations',{'bootstrap_error':False}),
    ('getFragmentsPerPeak',{'norm':True}),('computeVariability',{'norm':False}),
    ('computeVariability',{'bootstrap_quantiles':{'eval':'system()'}}),
    ('eval',{}),
    ('computeExpectations',{'norm':float('nan')}),
])
def test_closed_dispatch_and_kwargs(source,function,kwargs):
    with pytest.raises(ValueError):owner.execute(dict(source,function_name=function,kwargs=kwargs))
    assert not Path(source['output_dir']).exists()


def test_no_scientific_parameter_repair(source):
    assert owner.native_arguments(dict(source,function_name='computeVariability',kwargs={'bootstrap_samples':10001}))=={'bootstrap_samples':10001}
    assert owner.native_arguments(dict(source,kwargs={'norm':'invalid'}))=={'norm':'invalid'}
    assert owner.native_arguments(dict(source,function_name='computeVariability',kwargs={'bootstrap_samples':-1}))=={'bootstrap_samples':-1}
    assert owner.native_arguments(dict(source,kwargs={}))=={}


@pytest.mark.parametrize('function',['computeExpectations','getFragmentsPerPeak','getFragmentsPerSample'])
def test_publication_recovery_and_integrity(source,native,function):
    source.update(function_name=function,kwargs={} if function!='computeExpectations' else {'norm':False})
    before=Path(source['input_path']).read_bytes();result=owner.execute(source,'exact-execution')
    assert result['result_rows']==(2 if function=='getFragmentsPerSample' else 3)
    assert result['parameters']['kwargs']==source['kwargs']
    assert Path(source['input_path']).read_bytes()==before
    assert owner.recover(source,'exact-execution')==result and len(native)==1
    with pytest.raises(ValueError,match='retry'):owner.recover(source,'different-execution')
    with pytest.raises(ValueError):owner.recover(dict(source,kwargs={'norm':True}),'exact-execution')
    Path(result['table_path']).write_bytes(b'changed')
    with pytest.raises(ValueError):owner.verify_publication(result['manifest_path'],expected_sha256=result['manifest_sha256'])


def test_variability_requires_actual_score_slots_not_dimensions(source,tmp_path,native):
    args=dict(source,function_name='computeVariability',kwargs={'bootstrap_error':False})
    stage=tmp_path/'exchange';stage.mkdir()
    with pytest.raises(ValueError,match='requires supplied'):owner.prepare_input(args,stage)
    a=ad.read_h5ad(source['input_path']);a.X=a.X.toarray().astype(float);a.layers['z_scores']=np.array([[np.nan,1,2],[3,4,5]])
    a.uns=dict(matrix_semantics='chromVAR_bias_corrected_deviations',layer_semantics=dict(z_scores='chromVAR_background_standardized_z_scores'))
    a.write_h5ad(source['input_path']);owner.prepare_input(args,stage)
    assert np.isnan(np.fromfile(stage/'z.bin',dtype='<f8')[0])
    with pytest.raises(ValueError,match='not a declared count slot'):owner.prepare_input(dict(args,count_slot='X'),stage)
    result=owner.execute(args,'scores-execution');assert result['result_rows']==3 and len(native)==1


def test_shared_delivery_accepted_reopen_private_bytes(source,native,tmp_path):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    root=tmp_path/'workspace';source['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'/'native')
    class Planner:
        def plan(self,request,registry):return AgentPlan('test',request.request_id,'Simulated native function',(PlanStep('native',owner.TOOL,source),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('session')
    state=app.sessions.run('session','first',AgentRequest('first','Run computeExpectations',{}),
        (OutputSelection('result','native','manifest_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    items=accepted_artifacts(app.sessions,'session',state.revisions[-1],state.turn('first'),None,step_id='native')
    assert len(items)==5
    for item in items:
        output=MatrixDownloadPreparer().prepare(item)
        try:
            data=b''.join(output.iter_chunks())
            assert output.size_bytes==len(data) and output.sha256==hashlib.sha256(data).hexdigest()
            assert output.source_sha256==item.sha256
        finally:output.close()
    Path(source['input_path']).unlink()
    reopened=ResearchAgentApplication(root,planner=Planner()).sessions.load('session')
    assert reopened.to_dict()==state.to_dict() and len(native)==1
    assert len(accepted_artifacts(app.sessions,'session',state.revisions[-1],state.turn('first'),None,step_id='native'))==5


def test_structured_native_members_use_existing_grounding():
    from agent.application.turn_decisions import parse_decision
    from agent.application.dialogue_execution import _bind_arguments
    from agent.orchestration import build_default_tool_registry
    text='Call computeVariability with bootstrap_error=false and bootstrap_samples=25.'
    wire=dict(turn_schema_version=1,decision=dict(kind='execute_plan',target=owner.TOOL,arguments=[
        dict(tool=owner.TOOL,argument='kwargs',members=[dict(name='bootstrap_error',kind='json',literal='false',evidence='bootstrap_error=false'),
            dict(name='bootstrap_samples',kind='number',literal='25',evidence='bootstrap_samples=25')])]))
    decision=parse_decision(json.dumps(wire),utterance=text)
    assert _bind_arguments(build_default_tool_registry(),text,{},decision.arguments)=={'kwargs':{'bootstrap_error':False,'bootstrap_samples':25}}




def test_variability_uses_explicit_assay_slots_without_metadata_certificate(source,tmp_path):
    a=ad.read_h5ad(source['input_path']);a.X=np.array([[1.,2.,3.],[3.,2.,1.]])
    a.layers['z_scores']=a.X*2;a.uns.clear();a.write_h5ad(source['input_path'])
    stage=tmp_path/'exchange';stage.mkdir()
    effective,_,_=owner.prepare_input(dict(source,function_name='computeVariability',kwargs={}),stage)
    assert effective['measurement']=='supplied_chromVAR_z_assay'
    np.testing.assert_array_equal(np.fromfile(stage/'z.bin',dtype='<f8'),a.layers['z_scores'].ravel())


def test_background_requires_supplied_ordered_bias_without_fabrication(source,tmp_path):
    args=dict(source,function_name='getBackgroundPeaks',kwargs={'niterations':-2})
    assert owner.native_arguments(args)=={'niterations':-2}
    stage=tmp_path/'bias';stage.mkdir()
    with pytest.raises(ValueError,match='actual per-peak bias'):owner.prepare_input(args,stage)
    a=ad.read_h5ad(source['input_path']);a.var['provided_bias']=[.3,.1,.8];a.write_h5ad(source['input_path'])
    owner.prepare_input(dict(args,bias_key='provided_bias'),stage)
    features=pd.read_csv(stage/'features.tsv',sep='\t',dtype={'feature_id':str})
    assert features.feature_id.tolist()==['03','01','02'] and features.bias.tolist()==[.3,.1,.8]


def test_background_bias_does_not_lose_integer_precision_in_r_exchange(source,tmp_path):
    a=ad.read_h5ad(source['input_path']);a.var['bias']=np.array([2**60+1,1,2],dtype=np.int64);a.write_h5ad(source['input_path'])
    stage=tmp_path/'bias';stage.mkdir()
    with pytest.raises(ValueError,match='exactly'):
        owner.prepare_input(dict(source,function_name='getBackgroundPeaks',kwargs={}),stage)
