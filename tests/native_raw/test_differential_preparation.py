"""Installed package forwarding, empty tests and genuine rejection propagation."""
import os
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from agent.orchestration import AgentRuntime, FileRunStore, build_default_tool_registry
from agent.schemas import AgentRequest, AgentPlan, PlanStep
from agent.tools.analysis.local_subdivision import testable_features as select
from agent.tools._file_identity import file_sha256

pytestmark=pytest.mark.skipif(os.environ.get('RUN_RAW_NATIVE_QUALIFICATION')!='1',reason='Installed native science')


def test_native_empty_and_explicit_constant_remain_native(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_NATIVE_MEMORY_BYTES',str(4*1024**3))
    x=sparse.csr_matrix([[8,0],[8,0],[1,0],[1,0]],dtype=np.uint32)
    source=tmp_path/'source.h5ad';ad.AnnData(x,var=pd.DataFrame(index=['constant-one','constant-zero'])).write_h5ad(source)
    before=file_sha256(source)
    features,info=select(x,[0,1,2,3],['constant-one','constant-zero'])
    assert features==[]
    runtime=AgentRuntime(registry=build_default_tool_registry(),run_store=FileRunStore(tmp_path/'state'))
    def call(name,kwargs):
        request=AgentRequest(name,None,{})
        plan=AgentPlan(name+':plan',name,'Native differential edge cases',(PlanStep('diff','run_snapatac2',dict(
            input_path=str(source),output_dir=str(tmp_path/name),function_name='tl.diff_test',kwargs=dict(
                cell_group1=[0,1],cell_group2=[2,3],direction='positive',min_log_fc=0,**kwargs))),))
        return runtime.run_prepared(request,plan)
    empty=call('icta-empty',dict(features=features))
    assert empty.status.value=='SUCCEEDED',empty.errors
    result=ad.read_h5ad(empty.steps[0].result['matrix_path'])
    assert result.uns['agent_native_return']['value'].empty
    # A direct explicit native selection is forwarded unchanged, and still fails.
    explicit=call('native-explicit',dict(features=['constant-one']))
    assert explicit.status.value=='FAILED'
    assert explicit.steps[0].error.code=='NATIVE_FUNCTION_REJECTED'
    assert 'at least 2 classes' in explicit.steps[0].error.message
    assert explicit.steps[0].resolved_arguments['kwargs']['features']==('constant-one',)
    # Other genuine package errors are never turned into no-evidence stopping.
    rejected=call('native-other-error',dict(features=[0],covariates=['anything']))
    assert rejected.status.value=='FAILED' and rejected.steps[0].error.code=='NATIVE_FUNCTION_REJECTED'
    assert file_sha256(source)==before
