"""Small real Scanpy runs; scripted semantic decisions, no GPU or biological claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pytest

from agent.application import OutputSelection, ResearchAgentApplication
from agent.application.analysis_delivery import accepted_analysis
from agent.application.matrix_delivery import MatrixDownloadPreparer
from agent.orchestration import AgentRequest, LLMPlanner
from agent.schemas.prior_output import PriorOutputRef
from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
from test_processed_prior_outputs import (
    ScientificPlanModel, processed_source, processed_work, preserved_run_bytes,
)


class UmapModel(ScientificPlanModel):
    def __init__(self, *, clustered=False, historical=True):
        super().__init__(consumer=historical)
        self.clustered, self.historical = clustered, historical

    def complete(self, *, prompt, response_schema):
        result = json.loads(super().complete(prompt=prompt, response_schema=response_schema))
        if result.get('schema_version') == 4:
            steps = result['decision']['steps']
            if self.historical and not self.clustered:
                steps[0].update(step_id='umap', tool='compute_cell_umap')
            else:
                source_step = steps[-1]['step_id']
                if self.clustered and not self.historical:
                    steps.append(dict(step_id='clustering', tool='cluster_cells',
                        sources=[dict(target='analysis', source=dict(kind='step', step=source_step))],
                        control_dependencies=[]))
                steps.append(dict(step_id='umap', tool='compute_cell_umap',
                    sources=[dict(target='analysis', source=dict(kind='step', step=source_step))],
                    control_dependencies=[]))
        return json.dumps(result)


@pytest.mark.parametrize('clustered', [False, True])
def test_accepted_graph_umap_evidence_download_and_recovery(processed_source, monkeypatch, clustered):
    source = processed_source
    preserved = preserved_run_bytes(source)
    model = UmapModel(clustered=clustered)
    app = ResearchAgentApplication(source.app.workspace_root, planner=LLMPlanner(model))
    request = AgentRequest('umap-request',
        'Compute UMAP from the current neighbors' + (' and Leiden clusters.' if clustered else ' without clustering.'), {})
    import scanpy as sc
    umap_calls = []
    native = sc.tl.umap
    def counted(*args, **kwargs):
        umap_calls.append(kwargs)
        return native(*args, **kwargs)
    monkeypatch.setattr(sc.tl, 'umap', counted)
    with processed_work() as counts:
        state = app.sessions.run('analysis', 'umap-turn', request,
            (OutputSelection('umap', 'umap', 'analysis_path'),),
            expected_generation=1, use_active_context=True)
    run = app.run_store.load('umap-request:run')
    assert state.turn('umap-turn').status == 'activated', run.to_dict()
    assert counts['production.neighbors'] == counts['owner.neighbors'] == 0
    assert counts['production.leiden'] == int(clustered)
    assert len(umap_calls) == 1
    assert [s.tool_name for s in run.plan.steps] == (
        ['cluster_cells', 'compute_cell_umap'] if clustered else ['compute_cell_umap'])
    assert isinstance(run.plan.steps[0].arguments['analysis_path'], PriorOutputRef)
    step = run.steps[-1]
    assert step.verification.passed
    assert step.result.get('clustering_present', True) is clustered
    artifact = ad.read_h5ad(step.result['analysis_path'])
    assert ('leiden' in artifact.obs) is clustered
    assert ('clustering' in artifact.uns[PROVENANCE_KEY]['parameters']) is clustered
    assert tuple(artifact.obs_names) == source.ids
    np.testing.assert_array_equal(artifact.obsm['X_epizoo'], source.values)
    graph = ad.read_h5ad(source.run.steps[0].result['analysis_path'])
    assert (artifact.obsp['distances'] != graph.obsp['distances']).nnz == 0
    connections = artifact.obsp['connectivities']
    assert connections.multiply(connections != graph.obsp['connectivities']).count_nonzero() == 0
    assert artifact.obsm['X_umap'].shape == (32, 2)
    assert np.isfinite(artifact.obsm['X_umap']).all()

    revision = state.revisions[-1]
    view = app.sessions.evidence('analysis', revision.revision_id, 'umap')
    assert view.status == 'available', view
    facts = {f.field: f for f in view.facts}
    if clustered:
        assert facts['clustering_parameters'].status == 'available'
        assert 'clustering_present' not in facts  # Legacy projection stays exact.
    else:
        assert facts['clustering_present'].value is False
        assert facts['clustering_parameters'].reason == 'clustering_not_performed'
        assert facts['umap_neighbors_consumption'].value == 'qualified_historical_output'
        assert facts['umap_neighbors_identity_sha256'].status == 'available'
    paths = app._workspace.run_paths(run.run_id)
    manifests = [json.loads(p.read_text()) for p in paths.visualizations.rglob('*.json')]
    figures = [f for m in manifests for f in m.get('figures', [])]
    assert len(figures) == 1
    assert figures[0]['figure_kind'] == ('umap_leiden' if clustered else 'umap_coordinates')
    report = '\n'.join(p.read_text() for p in paths.report.rglob('*.md'))
    assert ('colored by Leiden' in report) is clustered
    if not clustered:
        assert 'Leiden clustering was not performed' in report
        assert 'Clustering and UMAP' not in report

    export = accepted_analysis(SimpleNamespace(_application=app), 'analysis', revision,
        state.turn('umap-turn'), revision.outputs[0])
    assert ('Leiden' in export.reference.label) is clustered
    with MatrixDownloadPreparer().prepare(export) as download:
        # Exercise the real pinned-download path before reading its safe H5AD.
        raw = b''.join(download.iter_chunks())
    from io import BytesIO
    with h5py.File(BytesIO(raw), 'r') as handle:
        from anndata.io import read_elem
        exported = read_elem(handle)
    assert exported.obs_names.equals(artifact.obs_names)
    assert ('leiden' in exported.obs) is clustered
    metadata = exported.uns['agent_scientific_export']
    assert bool(metadata['clustering_present']) is clustered
    assert metadata['contract'] == export.contract
    assert metadata['source_sha256'] == step.result['analysis_sha256']
    assert metadata['methods']['embedding']['representation'] == 'X_epizoo'
    assert metadata['methods']['embedding']['embedding_dim'] == 512
    assert ('clustering' in metadata['methods']) is clustered
    assert metadata['methods']['umap']['parameters']['n_components'] == 2
    for key in ('revision_id', 'run_id', 'step_id', 'accepted_step_sha256'):
        assert metadata['accepted_result'][key] == export.manifest['provenance'][key]
    if clustered:
        assert exported.obs['leiden'].equals(artifact.obs['leiden'])
    for key in ('X_epizoo', 'X_umap'):
        np.testing.assert_array_equal(exported.obsm[key], artifact.obsm[key])

    scientific = {p: p.read_bytes() for p in paths.root.rglob('*') if p.is_file()
                  and p.suffix in ('.h5ad', '.png', '.md')}
    restarted = ResearchAgentApplication(app.workspace_root)
    with processed_work() as recovered:
        assert restarted.resume(run.run_id).status.value == 'SUCCEEDED'
        assert restarted.sessions.recover('analysis', 'umap-turn') == state
        assert restarted.sessions.evidence('analysis', revision.revision_id, 'umap').status == 'available'
    assert recovered['production.neighbors'] == recovered['production.leiden'] == recovered['owner.neighbors'] == 0
    assert len(umap_calls) == 1
    assert all(p.read_bytes() == data for p, data in {**preserved, **scientific}.items())
    script = '''
import sys
import scanpy as sc
from agent.application import ResearchAgentApplication
def forbidden(*args, **kwargs):
    raise AssertionError('Terminal recovery invoked science')
sc.pp.neighbors = sc.tl.leiden = sc.tl.umap = forbidden
app = ResearchAgentApplication(sys.argv[1])
assert app.resume('umap-request:run').status.value == 'SUCCEEDED'
assert app.sessions.recover('analysis', 'umap-turn').turn('umap-turn').status == 'activated'
'''
    child = subprocess.run([sys.executable, '-B', '-c', script, str(app.workspace_root)],
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    assert all(p.read_bytes() == data for p, data in {**preserved, **scientific}.items())


@pytest.mark.parametrize('sibling_clustering', [False, True])
def test_same_plan_neighbors_to_umap_has_independent_dependencies(processed_source, tmp_path, sibling_clustering):
    source = processed_source
    model = UmapModel(historical=False, clustered=sibling_clustering)
    app = ResearchAgentApplication(tmp_path/'same-plan', planner=LLMPlanner(model))
    prompt = ('Compute neighbors, then independent Leiden and UMAP from that graph.' if sibling_clustering
              else 'Compute a neighbor graph and UMAP, without Leiden.')
    result = app.run(AgentRequest('same-plan', prompt,
        dict(embedding_path=str(source.embedding), cell_ids_path=str(source.cell_ids))))
    assert result.status.value == 'SUCCEEDED', result.to_dict()
    run = app.run_store.load(result.run_id)
    assert [s.tool_name for s in run.plan.steps] == (
        ['build_cell_neighbors', 'cluster_cells', 'compute_cell_umap'] if sibling_clustering
        else ['build_cell_neighbors', 'compute_cell_umap'])
    assert run.plan.steps[-1].depends_on == ('neighbors',)
    assert run.steps[-1].result['clustering_present'] is False


def test_tampered_accepted_graph_fails_before_umap(processed_source, monkeypatch):
    source = processed_source
    path = Path(source.run.steps[0].result['analysis_path'])
    path.write_bytes(path.read_bytes() + b'corruption')
    import scanpy as sc
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid prior source reached science')
    monkeypatch.setattr(sc.tl, 'umap', forbidden)
    app = ResearchAgentApplication(source.app.workspace_root, planner=LLMPlanner(UmapModel()))
    from agent.schemas.verification_authority import AuthorityError
    with pytest.raises(AuthorityError, match='bytes changed'):
        app.sessions.run('analysis', 'invalid', AgentRequest('invalid', 'Compute UMAP without clustering.', {}),
            (OutputSelection('umap', 'umap', 'analysis_path'),), expected_generation=1, use_active_context=True)
    assert app.sessions.load('analysis').generation == 1
