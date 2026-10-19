"""Simulated matrix, real Scanpy science, scripted semantic plans."""
import json
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from agent.application import OutputSelection, ResearchAgentApplication
from agent.orchestration import AgentRequest, LLMPlanner
from agent.schemas.prior_output import PriorOutputRef

class Model:
    model_id="scripted-pca-acceptance"
    def __init__(self,later=False):self.later=later
    def complete(self,*,prompt,response_schema):
        value=json.loads(prompt)
        if 'selection_schema_version' in value:
            return json.dumps(dict(selection_schema_version=1,decision=dict(kind='select',capability_ids=['embedding_analysis'])))
        if self.later:
            graph=next(x for x in value['active_outputs'] if x['producer_tool']=='build_cell_neighbors')
            steps=[dict(step_id='clusters',tool='cluster_cells',sources=[dict(target='analysis',source=dict(kind='context',handle=graph['handle']))],control_dependencies=[])]
        else:
            steps=[dict(step_id='pca',tool='pca_embed_cells',sources=[dict(target='dataset',source=dict(kind='input',input='input_path'))],control_dependencies=[])]
            for name,tool,target,source in [('neighbors','build_cell_neighbors','embedding','pca'),('clusters','cluster_cells','analysis','neighbors'),('umap','compute_cell_umap','analysis','clusters')]:
                steps.append(dict(step_id=name,tool=tool,sources=[dict(target=target,source=dict(kind='step',step=source))],control_dependencies=[]))
        return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=steps)))


def test_pca_authority_report_graph_reuse_and_recovery(tmp_path,monkeypatch):
    x=sparse.csr_matrix(np.random.default_rng(4).poisson(.4,(24,71)))
    path=tmp_path/'input.h5ad';ad.AnnData(x,obs=pd.DataFrame(index=[f'c{i}' for i in range(24)])).write_h5ad(path)
    app=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model()))
    app.sessions.create('analysis')
    state=app.sessions.run('analysis','first',AgentRequest('pca-first','Compute PCA, clusters and UMAP.',dict(input_path=str(path))),
        (OutputSelection('umap','umap','analysis_path'),),expected_generation=0)
    run=app.run_store.load('pca-first:run')
    assert state.turn('first').status=='activated',run.to_dict()
    assert run.steps[0].verification.artifact_authority is not None
    assert run.steps[1].verification.artifact_authority is not None
    paths=app._workspace.run_paths(run.run_id)
    report='\n'.join(p.read_text() for p in paths.report.rglob('*.md'))
    assert 'Centered PCA' in report and 'X_pca' in report
    assert 'EpiZoo representation produced' not in report
    before={p:p.read_bytes() for root in (paths.scientific,paths.evidence,paths.report) for p in root.rglob('*') if p.is_file()}
    # Exercise the actual application delivery gate, not just the owner exporter.
    from agent.application.interactive import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    facade = InteractiveAgentApplication(tmp_path/'app', model_profiles=(
        PlanningModelProfile(profile_id='read', provider_id='openrouter', model_id='unused'),),
        default_profile_id='read')
    artifacts = facade.scientific_artifact_handles('analysis', state.active_revision_id)
    graph = next(a for a in artifacts if a.filename == 'neighbors.h5ad')
    with facade.prepare_scientific_artifact('analysis', state.active_revision_id, graph.handle) as delivered:
        import io
        data = ad.read_h5ad(io.BytesIO(b''.join(delivered.iter_chunks())))
        assert data.uns['agent_scientific_export']['embedding_method'] == 'PCA'
        assert data.obsm['X_pca'].shape == (24, 23)
    # Only the final output was displayed: R1 exact lineage must expose the graph.
    later=ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model(later=True)))
    def forbidden(*args,**kwargs):raise AssertionError('upstream scientific work repeated')
    import scanpy as sc
    monkeypatch.setattr(sc.pp,'pca',forbidden);monkeypatch.setattr(sc.pp,'neighbors',forbidden)
    from agent.tools.analysis import pca_embedding,neighbors_publication
    monkeypatch.setattr(pca_embedding,'_verify_result',forbidden)
    monkeypatch.setattr(neighbors_publication,'_verify',forbidden)
    from test_captured_planning_context import capture
    from agent.orchestration.active_context import planning_context
    capture(later.sessions,'analysis')
    context,_=later.sessions.planning_context_for_interaction('analysis','future')
    with planning_context(context):
        state=later.sessions.run('analysis','later',AgentRequest('pca-later','Cluster the same graph.',{}),
            (OutputSelection('clusters','clusters','analysis_path'),),expected_generation=1)
    continuation=later.run_store.load('pca-later:run')
    assert state.turn('later').status=='activated',continuation.to_dict()
    assert isinstance(continuation.plan.steps[0].arguments['analysis_path'],PriorOutputRef)
    assert all(p.read_bytes()==raw for p,raw in before.items())
    # Persisted state loading performs no providers or science.
    assert ResearchAgentApplication(tmp_path/'app',planner=LLMPlanner(Model())).sessions.load('analysis')==state
