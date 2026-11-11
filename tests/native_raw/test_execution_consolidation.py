"""Production catalog/semantic contracts; real native numerical work on small inputs."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication
from agent.application.structured_parameters import result_parameters
from agent.orchestration import AgentRequest, LLMPlanner, PlanningModelProfile, build_default_tool_registry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import PriorOutputRef
from agent.tools.analysis.native_matrix import SNAPATAC2, SCANPY, EPISCANPY


def test_catalog_retires_agent_implementations_and_retains_native_methods():
    registry = build_default_tool_registry()
    for tool in ('build_cell_neighbors', 'cluster_cells', 'compute_cell_umap'):
        assert tool not in registry.names()
        from agent.orchestration.registry import UnknownToolError
        with pytest.raises(UnknownToolError): registry.get(tool)
    for owner in (SCANPY, EPISCANPY):
        assert {'pp.neighbors', 'tl.leiden', 'tl.umap'} <= owner.FUNCTIONS.keys()
        assert 'resolution' in owner.FUNCTIONS['tl.leiden']
        assert owner.native_arguments(dict(function_name='tl.leiden', kwargs={'resolution': 1.7}))['resolution'] == 1.7
    assert {'tl.pca', 'tl.dpt', 'tl.tsne', 'pp.log1p', 'pp.normalize_total'} <= EPISCANPY.FUNCTIONS.keys()
    assert {'pp.knn', 'tl.leiden', 'tl.umap', 'tl.dbscan'} <= SNAPATAC2.FUNCTIONS.keys()
    assert {'pp.tfidf', 'pp.lsi', 'tl.louvain', 'tl.kmeans', 'tl.hc', 'tl.rank_features'} <= EPISCANPY.FUNCTIONS.keys()
    assert {'pp.pca', 'tl.diffmap', 'tl.dpt', 'tl.paga', 'tl.tsne'} <= SCANPY.FUNCTIONS.keys()
    assert {'run_signac', 'run_macs3', 'call_scATAC_peaks', 'run_chromvar', 'compute_motif_accessibility',
            'run_motifmatchr', 'run_cicero', 'run_cicero_cds', 'run_archr', 'run_monalisa', 'run_chromap'} <= set(registry.names())

    from agent.orchestration.semantic_compiler import build_semantic_compiler_contract
    contract = build_semantic_compiler_contract(registry)
    for consumer in ('annotate_scATAC_cell_types', 'call_scATAC_peaks', 'evaluate_cell_clustering'):
        assert any(r.producer_tool_name == 'run_snapatac2' and r.consumer_tool_name == consumer
                   for r in contract.step_output_channels)


class Model:
    model_id = 'scripted-consolidation'

    def __init__(self, embedding=False, infer=False, umap=True, package="snapatac2"):
        self.package = package
        self.embedding, self.infer, self.umap = embedding, infer, umap
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='run_' + self.package)))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='analysis', step_id='umap' if self.umap else 'clusters', output_key='matrix_path')]))
        steps = []
        if self.infer:
            steps.append(dict(step_id='embed', tool='epizoo_embed_cells', sources=[
                dict(target='dataset', source=dict(kind='input', input='input_path')),
                dict(target='species', source=dict(kind='input', input='species'))], control_dependencies=[]))
        if self.package == 'snapatac2':
            operations = ([] if self.embedding else [('features', 'pp.select_features'), ('spectral', 'tl.spectral')])
            operations += [('neighbors', 'pp.knn')]
        else:
            operations = ([] if self.embedding else [('pca', 'pp.pca' if self.package == 'scanpy' else 'tl.pca')])
            operations += [('neighbors', 'pp.neighbors')]
        operations += [('clusters', 'tl.leiden')]
        if self.umap:
            operations += [('umap', 'tl.umap')]
        for name, fn in operations:
            source = (dict(kind='step_port', step=steps[-1]['step_id'], source_port='dataset') if steps
                      else dict(kind='input', input='input_path'))
            sources = [dict(target='dataset', source=source),
                dict(target='function_name', source=dict(kind='registered_choice', choice=fn))]
            if fn in ('pp.knn', 'pp.neighbors') or fn == 'tl.umap' and self.package == 'snapatac2':
                sources.append(dict(target='use_rep', source=dict(kind='registered_choice',
                    choice='X_epizoo' if self.embedding else 'X_spectral' if self.package == 'snapatac2' else 'X_pca')))
            steps.append(dict(step_id=name, tool='run_' + self.package, sources=sources, control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@pytest.mark.parametrize('package', ['scanpy', 'episcanpy', 'snapatac2'])
def test_epizoo_producer_compiles_to_native_dataset_without_counts(tmp_path, package):
    registry = build_default_tool_registry()
    plan = LLMPlanner(Model(embedding=True, infer=True, package=package)).plan(AgentRequest('epizoo', 'Embed then cluster.',
        dict(input_path=str(tmp_path/'input.h5ad'), output_dir=str(tmp_path/'out'), species='human')), registry)
    assert [s.tool_name for s in plan.steps] == ['epizoo_embed_cells'] + ['run_' + package]*3
    assert plan.steps[1].arguments['input_path'].output_key == 'analysis_path'
    assert plan.steps[1].arguments['use_rep'] == 'X_epizoo'
    if package == 'snapatac2':
        assert plan.steps[-1].arguments['use_rep'] == 'X_epizoo'
    else:
        assert 'use_rep' not in plan.steps[-1].arguments
    assert 'tl.spectral' not in [s.arguments.get('function_name') for s in plan.steps]


@pytest.mark.parametrize('embedding', [False, True])
@pytest.mark.parametrize('package', ['snapatac2', 'scanpy', 'episcanpy'])
def test_native_plan_edit_read_download_and_exact_upstream_reuse(tmp_path, monkeypatch, embedding, package):
    runtime = Path('/home/likeyi/program/agent-data/resources/runtimes/native-matrix-1/env/bin/python')
    if not runtime.is_file():
        pytest.skip('Optional SnapATAC2 runtime unavailable')
    monkeypatch.setenv('AGENT_SNAPATAC2_PYTHON', str(runtime))
    _scanpy_runtime(monkeypatch)
    # Only the outer process boundary is simulated in this regression. The
    # installed package still runs in the real native subprocess. The external
    # acceptance run separately exercises the unmodified isolation owner.
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    # epiScanpy's native PCA default is 50 components; supply a fixture large
    # enough for that default instead of changing the native request.
    n_cells = 64 if package == 'episcanpy' and not embedding else 40
    ids = [f'cell-{i:03d}' for i in reversed(range(n_cells))]
    data = ad.AnnData(obs=pd.DataFrame(index=ids)) if embedding else ad.AnnData(
        sparse.csr_matrix(np.random.default_rng(4).poisson(.4, (n_cells, 71)).astype('float32')), obs=pd.DataFrame(index=ids))
    if embedding:
        data.obsm['X_epizoo'] = np.random.default_rng(8).normal(size=(n_cells, 512)).astype('float32')
    source = tmp_path/'source.h5ad'; data.write_h5ad(source); original = source.read_bytes()
    profile = PlanningModelProfile('consolidation', 'scripted', Model.model_id)
    model = Model(embedding=embedding, umap=False, package=package)
    app = InteractiveAgentApplication(tmp_path/'app', model_profiles=(profile,), default_profile_id=profile.profile_id,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _:model}))
    app.create_session('analysis')
    first = app.submit_turn('analysis', 'initial', 'Cluster the supplied representation.', expected_generation=0,
        execution_inputs=dict(input_path=str(source)))
    assert first.status == 'succeeded', first.to_dict()
    state = app._application.sessions.load('analysis'); revision = state.revisions[-1]
    run = app._application.run_store.load(revision.run_id)
    functions = [s.arguments['function_name'] for s in run.plan.steps]
    assert [s.result['function_name'] for s in run.steps] == [package + '.' + f for f in functions]
    assert functions[-2:] == ['pp.knn' if package == 'snapatac2' else 'pp.neighbors', 'tl.leiden']
    assert run.steps[-2].result['parameters']['kwargs']['use_rep'] == (
        'X_epizoo' if embedding else 'X_spectral' if package == 'snapatac2' else 'X_pca')
    original_run = run.to_dict()
    pins = {p:p.read_bytes() for p in app._application._workspace.run_paths(run.run_id).scientific.rglob('*') if p.is_file()}
    fields = result_parameters(app._application.sessions, 'analysis', revision.revision_id)
    assert any(s['step_id'] == 'clusters' for s in fields['steps'])
    def forbidden(**kwargs):
        pytest.fail('Result-scoped parameter edit called a model')
    model.complete = forbidden
    result = app.submit_turn('analysis', 'edit', '', expected_generation=1, rerun_revision_id=revision.revision_id,
        parameter_overrides=[dict(tool='run_' + package, native_function='tl.leiden', step_id='clusters',
            argument='kwargs', value={'resolution': 1.7})])
    assert result.status == 'succeeded', result.to_dict()
    state = app._application.sessions.load('analysis'); edited = state.revisions[-1]
    later = app._application.run_store.load(edited.run_id)
    assert [s.step_id for s in later.steps] == ['clusters']
    assert isinstance(later.plan.steps[0].arguments['input_path'], PriorOutputRef)
    assert later.steps[0].result['parameters']['kwargs']['resolution'] == 1.7
    assert edited.parent_revision_id == revision.revision_id
    assert all(p.read_bytes() == raw for p,raw in pins.items())
    assert app._application.run_store.load(revision.run_id).to_dict() == original_run
    output = ad.read_h5ad(later.steps[0].result['matrix_path'])
    assert output.obs_names.tolist() == ids
    if embedding:
        assert output.X is None and output.n_vars == 0
        np.testing.assert_array_equal(output.obsm['X_epizoo'], data.obsm['X_epizoo'])
    else:
        assert (output.X != data.X).nnz == 0
    from agent.tools.analysis.categorical import observations
    observed_ids, labels = observations(later.steps[0].result['matrix_path'], later.steps[0].result['matrix_sha256'], 'leiden')
    assert list(observed_ids) == ids and len(labels) == n_cells
    artifact = next(a for a in app.scientific_artifact_handles('analysis', edited.revision_id) if a.filename.endswith('.h5ad'))
    download = app.prepare_scientific_artifact('analysis', edited.revision_id, artifact.handle)
    try:
        (tmp_path/'download.h5ad').write_bytes(b''.join(download.iter_chunks()))
    finally:
        download.close()
    delivered = ad.read_h5ad(tmp_path/'download.h5ad')
    assert delivered.obs_names.tolist() == ids
    assert source.read_bytes() == original



def test_compact_epizoo_export_preserves_values_and_cells_without_counts(tmp_path):
    from agent.tools.analysis.epizoo_embedding import _publish_compact_embedding
    from agent.application.analysis_delivery import create_analysis_export, EXPORT_KEY
    values = np.random.default_rng(3).normal(size=(3,512)).astype('float32')
    np.save(tmp_path/'embedding.npy',values)
    (tmp_path/'cells.txt').write_text('c3\nc1\nc2\n')
    result = _publish_compact_embedding(dict(embedding_path=str(tmp_path/'embedding.npy'),
        cell_ids_path=str(tmp_path/'cells.txt')), overwrite=False)
    source=Path(result['analysis_path']); before=source.read_bytes()
    destination=tmp_path/'download.h5ad'
    with source.open('rb') as supplied, destination.open('w+b') as output:
        create_analysis_export(supplied,output,dict(stage='embedding',n_cells=3),
            source_sha256=result['analysis_sha256'],source_size_bytes=len(before),
            manifest_sha256=None,max_bytes=1024*1024)
    delivered=ad.read_h5ad(destination)
    assert delivered.X is None and delivered.n_vars==0
    assert delivered.obs_names.tolist()==['c3','c1','c2']
    np.testing.assert_array_equal(delivered.obsm['X_epizoo'],values)
    assert not delivered.uns[EXPORT_KEY]['clustering_present']
    assert source.read_bytes()==before


@pytest.mark.parametrize('owner,function', [(SCANPY, 'pp.neighbors'), (EPISCANPY, 'pp.neighbors'),
    (SNAPATAC2, 'pp.knn')])
def test_representation_binding_uses_the_selected_native_parameter(owner, function):
    assert owner.native_arguments(dict(function_name=function, use_rep='X_epizoo')) == {'use_rep': 'X_epizoo'}
    assert owner.native_arguments(dict(function_name=function, use_rep='X_epizoo',
        kwargs={'use_rep': 'researcher_embedding'})) == {'use_rep': 'researcher_embedding'}
    with pytest.raises(ValueError, match='native use_rep'):
        owner.native_arguments(dict(function_name='tl.leiden', use_rep='X_epizoo'))
    if owner is not SNAPATAC2:
        with pytest.raises(ValueError, match='native use_rep'):
            owner.native_arguments(dict(function_name='tl.umap', use_rep='X_epizoo'))


class DependencyModel:
    """Fixed provider responses; the production compiler owns all binding."""
    model_id = 'scripted-native-dependencies'

    def __init__(self, operations):
        self.operations = operations

    def complete(self, **kwargs):
        steps = []
        for index, (package, function, parameters) in enumerate(self.operations):
            step = 'step' + str(index)
            sources = [dict(target='function_name', source=dict(kind='registered_choice', choice=function)),
                dict(target='dataset', source=dict(kind='step_port', step=steps[-1]['step_id'], source_port='dataset')
                    if steps else dict(kind='input', input='input_path'))]
            if parameters:
                sources.append(dict(target='kwargs', source=dict(kind='input',
                    input='run_' + package + '__' + function.replace('.', '_') + '__kwargs')))
            steps.append(dict(step_id=step, tool='run_' + package, sources=sources, control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


@pytest.mark.parametrize('method', ['paga', 'dpt'])
def test_native_prerequisite_composition(tmp_path, monkeypatch, method):
    from agent.application import ResearchAgentApplication, ApplicationStatus
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    _scanpy_runtime(monkeypatch)
    run_dependency_composition(tmp_path, method)


def run_dependency_composition(tmp_path, method):
    """Also called by isolated acceptance, without the test's process-boundary stub."""
    from agent.application import ResearchAgentApplication, ApplicationStatus
    source = tmp_path / 'source.h5ad'
    data = ad.AnnData(sparse.csr_matrix(np.random.default_rng(17).poisson(2, (60, 24)).astype('float32')),
        obs=pd.DataFrame(index=[f'c{i}' for i in range(60)]))
    data.uns['iroot'] = 3  # A supplied fixture root, never inferred by the planner.
    data.write_h5ad(source)
    before = source.read_bytes()
    operations = ([('scanpy', 'pp.pca', {'n_comps': 10}),
        ('scanpy', 'pp.neighbors', {'use_rep': 'X_pca', 'n_neighbors': 8}),
        ('scanpy', 'tl.leiden', {'resolution': 0.8}),
        ('scanpy', 'tl.paga', {'groups': 'leiden'})] if method == 'paga' else [
        ('episcanpy', 'tl.pca', {'n_comps': 10}),
        ('episcanpy', 'pp.neighbors', {'use_rep': 'X_pca', 'n_neighbors': 8}),
        ('scanpy', 'tl.diffmap', {'n_comps': 8}),
        ('episcanpy', 'tl.dpt', {'n_dcs': 6})])
    inputs = {'input_path': str(source)}
    inputs.update({'run_' + p + '__' + f.replace('.', '_') + '__kwargs': kw for p, f, kw in operations})
    profile = PlanningModelProfile('dependency', 'scripted', DependencyModel.model_id)
    app = ResearchAgentApplication(tmp_path / 'app', primary_planning_profile=profile,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: DependencyModel(operations)}))
    assert isinstance(app.runtime.planner, LLMPlanner)
    result = app.run(AgentRequest(method, 'Execute the supplied native dependency composition.', inputs))
    assert result.status is ApplicationStatus.SUCCEEDED, result.to_dict()
    run = result.run_result
    assert [s.result['function_name'] for s in run.steps] == [p + '.' + f for p, f, _ in operations]
    for index, step in enumerate(run.steps):
        expected = __import__('hashlib').sha256(before).hexdigest() if index == 0 else run.steps[index-1].result['matrix_sha256']
        assert step.result['source_identity']['sha256'] == expected
        assert dict(step.result['parameters']['kwargs']) == operations[index][2]
    out = ad.read_h5ad(run.steps[-1].result['matrix_path'])
    assert out.obs_names.equals(data.obs_names) and (out.X != data.X).nnz == 0
    assert source.read_bytes() == before
    if method == 'paga':
        assert 'connectivities' in out.obsp and 'connectivities' in out.uns['paga']
        assert 'leiden' in out.obs
    else:
        assert out.uns['iroot'] == 3 and 'X_diffmap' in out.obsm
        assert np.isfinite(out.obs['dpt_pseudotime']).all()
        assert out.obs['dpt_pseudotime'].iloc[3] == 0
    return result.to_dict()


def _scanpy_runtime(monkeypatch):
    runtime = Path('/home/likeyi/program/agent-data/resources/runtimes/episcanpy-0.4.0/env/bin/python')
    if not runtime.is_file():
        pytest.skip('Optional Scanpy/epiScanpy runtime unavailable')
    for package in ('SCANPY', 'EPISCANPY'):
        monkeypatch.setenv('AGENT_' + package + '_PYTHON', str(runtime))


@pytest.mark.parametrize('tool', ['run_scanpy', 'run_episcanpy', 'run_snapatac2'])
def test_epizoo_handoff_accepts_supported_native_downstream_arguments(tool):
    from agent.application.raw_handoff import validate_arguments
    from agent.application.turn_decisions import NativeChoiceArgument
    # This is the actual handoff admission boundary, before normal Registry binding.
    argument = NativeChoiceArgument(tool, 'function_name', 'tl.leiden', 0, 9,
        'tl.leiden', native_function='tl.leiden')
    validate_arguments((argument,), 'epizoo')
    from types import SimpleNamespace
    with pytest.raises(ValueError):
        validate_arguments((SimpleNamespace(tool='unregistered_execution'),), 'epizoo')
