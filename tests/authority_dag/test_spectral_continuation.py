"""Real native science on simulated counts; scripted semantic planning."""
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from agent.application import OutputSelection, ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner
from agent.schemas.prior_output import PriorOutputRef


class Model:
    model_id='scripted-spectral-acceptance'
    def __init__(self,later=False): self.later=later; self.graph_handle=None
    def complete(self,*,prompt,response_schema):
        value=json.loads(prompt)
        if self.later:
            rows=[r for r in value['active_outputs'] if r['producer_tool']=='run_snapatac2']
            # The caller supplies the exact graph-containing H5AD through captured context.
            graph=next(r for r in rows if r['handle']==self.graph_handle)
            steps=[dict(step_id='clusters',tool='run_snapatac2',sources=[
                dict(target='dataset',source=dict(kind='context',handle=graph['handle'])),
                dict(target='function_name',source=dict(kind='registered_choice',choice='tl.leiden'))],control_dependencies=[])]
        else:
            steps=[]
            for i,(name,fn) in enumerate([('features','pp.select_features'),('spectral','tl.spectral'),
                    ('neighbors','pp.knn'),('clusters','tl.leiden'),('umap','tl.umap')]):
                source=dict(kind='input',input='input_path') if i==0 else dict(kind='step_port',step=steps[-1]['step_id'],source_port='dataset')
                steps.append(dict(step_id=name,tool='run_snapatac2',sources=[dict(target='dataset',source=source),
                    dict(target='function_name',source=dict(kind='registered_choice',choice=fn))],control_dependencies=[]))
        return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))


@pytest.fixture
def native_runtime(monkeypatch):
    executable=Path(os.environ.get('AGENT_SNAPATAC2_PYTHON','/home/likeyi/program/agent-data/resources/runtimes/native-matrix-1/env/bin/python'))
    if not executable.is_file(): pytest.skip('Optional isolated SnapATAC2 runtime is not installed')
    monkeypatch.setenv('AGENT_SNAPATAC2_PYTHON',str(executable))


def test_spectral_authority_report_and_restart_graph_reuse(tmp_path,monkeypatch,native_runtime):
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    x=sparse.csr_matrix(np.random.default_rng(4).poisson(.4,(40,71)))
    path=tmp_path/'input.h5ad';ad.AnnData(x,obs=pd.DataFrame(index=[f'c{i}' for i in range(40)])).write_h5ad(path)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model()));app.sessions.create('analysis')
    state=app.sessions.run('analysis','first',AgentRequest('spectral-first','Native scATAC clustering and UMAP.',dict(input_path=str(path))),
        (OutputSelection('umap','umap','matrix_path'),),expected_generation=0)
    run=app.run_store.load('spectral-first:run')
    assert state.turn('first').status=='activated',run.to_dict()
    assert all(s.verification.artifact_authority is not None for s in run.steps)
    output=ad.read_h5ad(run.steps[-1].result['matrix_path'])
    assert output.obsm['X_spectral'].shape[0]==40 and 0<output.obsm['X_spectral'].shape[1]<=30 and np.isfinite(output.obsm['X_umap']).all()
    assert (output.X != x).nnz==0
    paths=app._workspace.run_paths(run.run_id)
    report='\n'.join(p.read_text() for p in paths.report.rglob('*.md'))
    assert 'snapatac2.tl.spectral' in report and 'Centered PCA' not in report
    before={p:p.read_bytes() for root in (paths.scientific,paths.evidence,paths.report) for p in root.rglob('*') if p.is_file()}
    model=Model(later=True)
    later=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(model))
    from test_captured_planning_context import capture
    from agent.orchestration.active_context import planning_context
    capture(later.sessions,'analysis');context,_=later.sessions.planning_context_for_interaction('analysis','future')
    model.graph_handle=next(i.handle for i in context.items if i.binding.step_id=='umap' and i.binding.source_port=='dataset')
    # Only downstream native Leiden is allowed: native code cannot be invoked
    # for feature selection, Spectral or KNN during accepted-result reuse.
    from agent.tools.analysis.native_matrix import SNAPATAC2
    native=SNAPATAC2.native
    def only_leiden(args,*rest):
        assert args['function_name']=='tl.leiden'
        return native(args,*rest)
    monkeypatch.setattr(SNAPATAC2,'native',only_leiden)
    with planning_context(context):
        state=later.sessions.run('analysis','later',AgentRequest('spectral-later','Cluster the same graph.',{}),
            (OutputSelection('clusters','clusters','matrix_path'),),expected_generation=1)
    continuation=later.run_store.load('spectral-later:run')
    assert state.turn('later').status=='activated',continuation.to_dict()
    assert isinstance(continuation.plan.steps[0].arguments['input_path'],PriorOutputRef)
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert ResearchAgentApplication(tmp_path/'app').sessions.load('analysis')==state
