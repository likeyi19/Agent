"""Safe CDS exchange and shared authority; numerical parity is opt-in."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import cicero_cds as owner


@pytest.fixture
def source(tmp_path):
    a=ad.AnnData(sparse.csr_matrix([[1.,-1.],[2.,.5],[0.,1.]]),
        obs=pd.DataFrame({'depth':[1.,2.,3.]},index=['c3','c1','c2']),
        var=pd.DataFrame(index=['chr1_1_10','chr1_20_30']))
    a.obsm['provided']=np.array([[2.,1.],[1.,4.],[3.,2.]])
    path=tmp_path/'input.data';a.write_h5ad(path)
    return dict(input_path=str(path),output_dir=str(tmp_path/'out'),function_name='make_cicero_cds',reduced_coordinates_key='provided')


@pytest.fixture
def simulated(tmp_path,monkeypatch):
    executable=tmp_path/'Rscript';executable.write_text('SIMULATED')
    monkeypatch.setenv('AGENT_CICERO_RSCRIPT',str(executable));calls=[]
    def run(command,log,**kwargs):
        reqpath=Path(command[-1]);req=json.loads(reqpath.read_text());stage=reqpath.parent;calls.append(req)
        for name in owner.MEMBERS:(stage/('out-'+name)).write_bytes((stage/name).read_bytes())
        obj=json.loads((stage/'object.json').read_text())
        (stage/'aggregation.tsv').write_text('cell\tagg_cell\n')
        (stage/'result.tsv').write_text('Peak1\tPeak2\tcoaccess\n')
        owner.write(stage/'native-receipt.json',dict(request_sha256=hashlib.sha256(reqpath.read_bytes()).hexdigest(),completion='succeeded',function_name=req['function_name'],versions={'cicero':'1.28.3'},n_observations=len(obj['observations']['ids']),n_regions=len(obj['regions']['ids']),native_output_type='SIMULATED'))
        log.write_text('SIMULATED');return 0
    monkeypatch.setattr(owner,'run',run);return calls


def test_sparse_signed_input_and_coordinates_keep_exact_axes(source,tmp_path):
    stage=tmp_path/'stage';stage.mkdir();before=Path(source['input_path']).read_bytes()
    obj=owner.prepare(dict(source,kwargs={'summary_stats':['depth']}),stage)
    assert list(obj['observations']['ids'])==['c3','c1','c2']
    assert obj['observations']['columns']['depth']['values']==[1.,2.,3.]
    from scipy.io import mmread
    np.testing.assert_array_equal(mmread(stage/'exprs.mtx').toarray(),[[1.,2.,0.],[-1.,.5,1.]])
    np.testing.assert_array_equal(np.loadtxt(stage/'coordinates.tsv'),[[2.,1.],[1.,4.],[3.,2.]])
    assert Path(source['input_path']).read_bytes()==before


def test_summary_logical_type_is_not_changed_to_numeric(source,tmp_path):
    a=ad.read_h5ad(source['input_path']);a.obs['flag']=[True,False,True];a.write_h5ad(source['input_path'])
    stage=tmp_path/'stage';stage.mkdir()
    obj=owner.prepare(dict(source,kwargs={'summary_stats':['flag']}),stage)
    assert obj['observations']['columns']['flag']==dict(type='logical',values=[True,False,True])


@pytest.mark.parametrize('members',[{'../escape':'x'},{'exprs.mtx':'x','object.json':'{}','evil.rds':'x'},{'object.json':'{}'}])
def test_no_arbitrary_archive_members_or_object_deserialization(tmp_path,members):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        for key,value in members.items():z.writestr(key,value)
    stage=tmp_path/'stage';stage.mkdir()
    with pytest.raises(ValueError,match='exactly'):owner.unpack(path,stage)
    assert not (tmp_path/'escape').exists()


def test_native_values_not_prevalidated_and_objects_are_protected():
    assert owner.native_arguments(dict(function_name='make_cicero_cds',kwargs={'k':-2}))=={'k':-2}
    for kw in ({'cds':'anything'},{'reduced_coordinates':'/private/file'},{'source':'system()'}):
        with pytest.raises(ValueError):owner.native_arguments(dict(function_name='make_cicero_cds',kwargs=kw))
    assert owner.parameter_prerequisites({'function_name':'make_cicero_cds'})==('reduced_coordinates_key',)
    assert owner.parameter_prerequisites({'function_name':'run_cicero'})==('genomic_coords_path',)
    assert owner.parameter_prerequisites({'function_name':'estimate_distance_parameter'})==()


@pytest.mark.parametrize('input_kind',['h5ad','cds'])
def test_cds_and_genome_ports_bind_ordinary_uploads(source,tmp_path,input_kind):
    from agent.orchestration import build_default_tool_registry
    from agent.application.local_resources import LocalResourceAdmission, group_choices
    registry=build_default_tool_registry()
    local=LocalResourceAdmission(tmp_path/'roles',approved_source_roots=(tmp_path,),registry=registry)
    dataset=Path(source['input_path'])
    if input_kind=='cds':
        stage=tmp_path/'exchange';stage.mkdir();owner.prepare(source,stage)
        dataset=tmp_path/'ordinary.zip'
        with zipfile.ZipFile(dataset,'w') as bundle:
            for name in owner.MEMBERS:bundle.write(stage/name,name)
    genome=tmp_path/'genome.tsv';genome.write_text('chr1\t100\n')
    entries=[local.register(str(i),p,label=p.name,attribution='fixture',input_type='attachment')
        for i,p in enumerate((dataset,genome))]
    group=local.compose_group([e.resource_id for e in entries],attachment_candidates=True)
    offered={(p['tool'],p['port']) for p in group_choices(group,registry)['roles']}
    assert {('run_cicero_cds','dataset'),('run_cicero_cds','genomic_coords'),('run_scanpy','dataset')} <= offered
    bound=local.bind_group(group,[dict(source='source.0',tool=owner.TOOL,port='dataset'),
        dict(source='source.1',tool=owner.TOOL,port='genomic_coords')],verify_source=True)
    assert bound.execution_inputs['input_path']==str(dataset)
    assert bound.execution_inputs['genomic_coords_path']==str(genome)


def test_reuse_does_not_reaggregate_and_shared_delivery_restart(source,simulated,tmp_path):
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    root=tmp_path/'workspace'
    source['output_dir']=str(root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific')
    class Planner:
        selected=AgentPlan('first','first','Simulated CDS',(PlanStep('native',owner.TOOL,source),))
        def plan(self,request,registry):return self.selected
    planner=Planner();app=ResearchAgentApplication(root,planner=planner);app.sessions.create('reuse')
    state=app.sessions.run('reuse','first',AgentRequest('first','Make CDS',{}),(OutputSelection('cds','native','cds_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    binding=bind_output(app.run_store,app.registry,'first:run','native','cds')
    assert binding.semantic_type=='cicero_cds_exchange.v1'
    Path(source['input_path']).unlink()
    genome=tmp_path/'genome.tsv';genome.write_text('chr1\t100\n')
    args=dict(input_path=PriorOutputRef(binding,'cds_path'),function_name='run_cicero',genomic_coords_path=str(genome),output_dir=str(root/'runs'/ManagedWorkspace.run_digest('second:run')/'scientific'))
    planner.selected=AgentPlan('second','second','Reuse CDS',(PlanStep('native',owner.TOOL,args),))
    state=app.sessions.run('reuse','second',AgentRequest('second','Run',{}),(OutputSelection('cds','native','cds_path'),),expected_generation=1)
    assert state.turn('second').status=='activated',app.run_store.load('second:run').errors
    reopened=ResearchAgentApplication(root,planner=planner)
    assert reopened.sessions.load('reuse').to_dict()==state.to_dict()
    artifacts=accepted_artifacts(reopened.sessions,'reuse',state.revisions[-1],state.turn('second'),None,step_id='native')
    assert len(artifacts)==5
    for artifact in artifacts:
        prepared=MatrixDownloadPreparer().prepare(artifact)
        try:
            data=b''.join(prepared.iter_chunks())
            assert hashlib.sha256(data).hexdigest()==prepared.sha256 and len(data)==prepared.size_bytes
        finally:prepared.close()
    assert [r['function_name'] for r in simulated]==['make_cicero_cds','run_cicero']
