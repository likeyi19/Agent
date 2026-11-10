"""Native raw files use the ordinary Executor, RunStore and Session lifecycle."""
import os
from pathlib import Path
import pytest

from agent.application import ResearchAgentApplication, OutputSelection
from agent.application.workspace import ManagedWorkspace
from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
from agent.orchestration.verification_authority import _load_context
from agent.orchestration.prior_outputs import bind_output


@pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Explicit native runtime qualification')
def test_native_matrix_acceptance_restart_reuse_and_integrity(tmp_path,monkeypatch):
    source=tmp_path/'fragments';source.write_text('chr1\t10\t30\tB\t300\nchr1\t20\t40\tA\t1\n')
    peaks=tmp_path/'peaks';peaks.write_text('chr1\t15\t25\nchr1\t35\t50\n')
    sizes=tmp_path/'sizes';sizes.write_text('chr1\t100\n')
    root=tmp_path/'workspace';scientific=root/'runs'/ManagedWorkspace.run_digest('first:run')/'scientific'
    plan=AgentPlan('first','first','Native import and exact feature counts',(
        PlanStep('import','run_snapatac2',dict(input_path=str(source),chrom_sizes_path=str(sizes),
            function_name='pp.import_fragments',kwargs=dict(min_num_fragments=0,sorted_by_barcode=False),output_dir=str(scientific/'import'))),
        PlanStep('matrix','run_snapatac2',dict(input_path=StepOutputRef('import','matrix_path'),peaks_path=str(peaks),
            function_name='pp.make_peak_matrix',kwargs=dict(counting_strategy='fragment'),output_dir=str(scientific/'matrix')),depends_on=('import',))))
    class Planner:
        def plan(self,request,registry): return plan
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('native')
    state=app.sessions.run('native','first',AgentRequest('first','Make a raw-fragment matrix',{}),
        (OutputSelection('dataset','matrix','matrix_path'),),expected_generation=0)
    assert state.turn('first').status=='activated',app.run_store.load('first:run').errors
    run=app.run_store.load('first:run')
    assert all(step.verification.passed and step.attempt_count==1 for step in run.steps)
    from agent.tools.analysis.native_matrix import SNAPATAC2
    monkeypatch.setattr(SNAPATAC2,'native',lambda *a:pytest.fail('Accepted retrieval repeated native science'))
    for p in (source,sizes,peaks): p.unlink()
    restarted=ResearchAgentApplication(root)
    assert restarted.sessions.load('native')==state
    context=_load_context(restarted.run_store,'first:run','historical_verified_sources.v1');context.validate_anchors()
    assert bind_output(restarted.run_store,restarted.registry,'first:run','matrix','dataset').semantic_type=='native_matrix_h5ad.v1'
    # The same accepted publications use the real HTTP privacy/download path
    # after restart, without interpretation or native execution.
    import io
    import anndata as ad
    from fastapi.testclient import TestClient
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from agent.web.app import create_app
    facade=InteractiveAgentApplication(root,model_profiles=(PlanningModelProfile('offline','scripted','offline'),),
        default_profile_id='offline',planning_model_factory_registry=PlanningModelFactoryRegistry(
            {'scripted':lambda _:pytest.fail('Read-only HTTP retrieval contacted a model')}))
    route='/api/v1/sessions/native/revisions/'+state.revisions[0].revision_id+'/scientific-artifacts'
    with TestClient(create_app(facade)) as client:
        response=client.get(route);assert response.status_code==200,response.text
        artifacts=response.json()['artifacts'];shapes=[]
        for artifact in artifacts:
            if artifact['filename']!='result.h5ad':continue
            response=client.get(route+'/'+artifact['handle']);assert response.status_code==200,response.text
            shapes.append(ad.read_h5ad(io.BytesIO(response.content)).shape)
        assert sorted(shapes)==[(2,0),(2,2)]
    from agent.schemas.prior_output import PriorOutputRef
    binding=bind_output(restarted.run_store,restarted.registry,'first:run','matrix','dataset')
    class NativePlanner:
        def plan(self,request,registry):
            return AgentPlan('pca','pca','Native feature selection consumes the accepted matrix',(
                PlanStep('pca','run_snapatac2',dict(input_path=PriorOutputRef(binding,'matrix_path'),
                    function_name='pp.select_features',output_dir=str(Path(request.inputs['output_dir'])/'pca'))),))
    downstream=ResearchAgentApplication(root,planner=NativePlanner()).run(AgentRequest('pca','Select native accessibility features from the accepted matrix',{}))
    assert downstream.status.value=='SUCCEEDED',downstream.run_result.errors
    import anndata as ad
    assert 'selected' in ad.read_h5ad(downstream.run_result.steps[0].result['matrix_path']).var
    matrix=Path(run.steps[-1].result['matrix_path']);matrix.write_bytes(b'corrupt')
    with pytest.raises(ValueError): _load_context(restarted.run_store,'first:run','historical_verified_sources.v1')

@pytest.mark.parametrize('compression', ['blosc','zstd'])
def test_native_fragment_storage_and_plugin_matrix_privacy_exports(tmp_path,compression):
    import io
    import anndata as ad
    import h5py
    import hdf5plugin
    import numpy as np
    import pandas as pd
    from agent.application.statistical_delivery import create_statistical_export
    path=tmp_path/'native.h5ad'
    data=ad.AnnData(obs=pd.DataFrame(index=['B','A']))
    data.obsm['fragment_paired']=np.array([[0,4],[2,6]],dtype='uint32')
    data.write_h5ad(path)
    with h5py.File(path,'a') as h:
        del h['var']
        del h['obsm/fragment_paired']
        h['obsm'].create_dataset('fragment_paired',data=data.obsm['fragment_paired'],
            **(hdf5plugin.Blosc() if compression=='blosc' else hdf5plugin.Zstd()))
        h['obsm/fragment_paired'].attrs.update({'encoding-type':'array','encoding-version':'0.2.0'})
    before=path.read_bytes()
    target=io.BytesIO()
    with path.open('rb') as stream:
        create_statistical_export(stream,target,dict(tool_name='run_snapatac2',accepted_step_sha256='a'*64),
            source_sha256='b'*64,source_size_bytes=path.stat().st_size,manifest_sha256='c'*64,max_bytes=1024**3)
    with h5py.File(target,'r') as h:
        assert h['obsm/fragment_paired'].compression=='gzip'
        np.testing.assert_array_equal(h['obsm/fragment_paired'][:],data.obsm['fragment_paired'])
    assert path.read_bytes()==before
