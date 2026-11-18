"""Shared canonical controls with a real failed Run, simulated native algorithm."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest
from agent.tools.analysis import native_matrix,native_clustering
from agent.tools import _native_process
from agent.tools._file_identity import file_sha256
from agent.tools.analysis.sequence_prediction import write


@pytest.mark.parametrize('function',['tl.kmeans','tl.hc'])
def test_failed_native_count_uses_canonical_answer_and_exact_failed_run(tmp_path,monkeypatch,function):
    spec=importlib.util.spec_from_file_location('count_fixture',Path(__file__).parents[1]/'native_raw/test_asw_selection.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    calls=[];frozen=[]
    def native(self,args,stage,matrix):
        name=args['function_name'];slots=[];selection=None
        if name in ('tl.kmeans','tl.hc'):
            explicit='num_clusters' in args.get('kwargs',{})
            if not explicit and not frozen:
                from agent.application import ResearchAgentApplication
                from agent.schemas.orchestration import _serialize
                captured=ResearchAgentApplication(tmp_path/'app').sessions.load('asw').interactions[-1]
                frozen.append((_serialize(captured.admitted),_serialize(captured.preparation)))
            def cluster(a,num_clusters):
                calls.append((name,num_clusters,explicit))
                a.obs[name.split('.')[1]]=pd.Categorical([str(i%num_clusters) if explicit else '0' for i in range(a.n_obs)])
            try:
                _,selection=native_clustering.invoke(cluster,matrix,self.package,name,args.get('kwargs',{}),args.get('parameter_selection'),[])
            except native_clustering.RequiredClusterCount as exc:
                write(stage.parent/'native-error.json',dict(type='RequiredClusterCount',parameter_selection=exc.evidence,source_sha256=file_sha256(args['input_path'])))
                raise native_matrix.NativeInvocationError(str(exc),code='NATIVE_CLUSTER_COUNT_REQUIRED')
            slots=['obs.'+name.split('.')[1]]
        matrix.write_h5ad(stage/'result.h5ad')
        receipt=dict(completion='succeeded',versions={self.package:'SIMULATED'},implementation=[],native_output_type='NoneType',result_slots=slots,measurement='unchanged controlled fixture')
        if selection is not None:receipt['parameter_selection']=selection
        return receipt
    monkeypatch.setattr(native_matrix.MatrixPackage,'native',native)
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    result=helper.run_count_fallback(tmp_path,function,check_edit=function=='tl.hc')
    assert result['provider_calls_on_answer']==0 and result['failed_run_unchanged']
    from agent.application import ResearchAgentApplication
    state=ResearchAgentApplication(tmp_path/'app').sessions.load('asw')
    question=next(i for i in state.interactions if i.turn_id=='automatic')
    answer=next(i for i in state.interactions if i.turn_id=='explicit')
    from agent.application.preparation_state import choices,observation
    assert choices(question)[0]['handle'].startswith('spp.')
    from agent.schemas.orchestration import _serialize
    assert (_serialize(question.admitted),_serialize(question.preparation))==frozen[0]
    assert answer.preparation['task_id']==question.preparation['task_id']
    assert answer.preparation['candidate_id']==question.preparation['candidate_id']
    assert observation(question)['requirements'][0]['applicability']['native_failure']['run_id']==state.turn('automatic').run_id
    assert answer.preparation['answers'][0]['provenance']=='exact_submission'
    assert all(explicit for _,_,explicit in calls[-(2 if function=='tl.hc' else 1):])
