"""ICTA-2B: exact native input adaptations and one accepted marker interface."""
import hashlib
import io
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.schemas.verification_authority import AuthorityError
from agent.schemas.orchestration import _serialize
from agent.tools.analysis.native_matrix import SCANPY, EPISCANPY, SNAPATAC2, NativeInvocationError, prepare_input
from agent.tools.analysis.native_gene_activity import axis_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


@pytest.fixture
def native():
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native runtime qualification')


@pytest.fixture
def source(tmp_path):
    a = ad.AnnData(sparse.csr_matrix([[8, 1], [6, 1], [7, 2], [9, 1], [1, 8], [2, 6], [1, 7], [1, 9]], dtype=float),
        obs=pd.DataFrame({'group': pd.Categorical(['left']*4 + ['right']*4)}, index=[f'cell{i}' for i in range(8)]),
        var=pd.DataFrame(index=['Alpha', 'Beta']))
    a.layers['scores'] = a.X.copy() / 2
    a.obsm['X_umap'] = np.arange(16).reshape(8, 2).astype(float)
    a.obsp['global_graph'] = sparse.eye(8, format='csr')
    a.uns['matrix_semantics'] = 'gene_activity'
    path = tmp_path/'supplied.h5ad'; a.write_h5ad(path)
    return path


def call(source, root, name='pp.calculate_qc_metrics', **args):
    return SCANPY.execute(dict(input_path=str(source), input_sha256=sha(source), output_dir=str(root),
        function_name=name, kwargs=dict(percent_top=None, log1p=False) if name == 'pp.calculate_qc_metrics' else {}, **args))


def selection(source, positions, **extra):
    a = ad.read_h5ad(source)
    return dict(source_sha256=sha(source), positions=positions, cell_ids=a.obs_names[positions].tolist(),
        **extra)


@pytest.mark.parametrize('dense,slot', [(False, 'X'), (True, 'X'), (False, 'scores'), (True, 'layers/scores')])
def test_existing_activity_exact_x_layer_unknown_history(native, source, tmp_path, dense, slot):
    a = ad.read_h5ad(source)
    # Missing IDs and duplicate symbols are valid native feature axes.
    a.var_names = ['Alpha', 'Alpha']
    if dense:
        a.X = array(a.X); a.layers['scores'] = array(a.layers['scores'])
        a.uns['matrix_semantics'] = 'gene_score'
        a.X[0, 0] = -0.5
        a.X[1, 1] = np.nan  # Preserve unusual values when native QC accepts them.
    # Deliberately forged provenance remains data; no proof is created from it.
    a.uns['agent_gene_activity'] = {'construction': 'pretend native', 'manifest_sha256': '0'*64}
    a.write_h5ad(source); before = sha(source)
    result = call(source, tmp_path/'reuse', gene_activity=True, count_slot=slot)
    b = ad.read_h5ad(result['matrix_path'])
    expected = a.X if slot == 'X' else a.layers['scores']
    np.testing.assert_array_equal(array(b.X), array(expected))
    np.testing.assert_array_equal(array(b.layers['scores']), array(a.layers['scores']))
    assert b.var_names.tolist() == ['Alpha', 'Alpha'] and not len(b.var.columns)
    desc = result['parameters']['gene_activity']
    assert desc['interface'] == 'gene-activity.v1'
    assert desc['construction'] == dict(origin='supplied_declaration', method=None, normalization_history='unknown')
    assert desc['axes'] == axis_identity(result['matrix_path'])
    assert desc['source_slot'] == ('X' if slot == 'X' else 'layers/scores')
    assert desc['source_history_completeness'] == 'unknown'
    assert 'accepted_publication' not in result['source_identity']
    assert result['source_identity']['sha256'] == before == sha(source)
    assert json.loads(b.uns['agent_gene_activity'])['description']['construction']['method'] is None


def test_explicit_transpose_preserves_selected_native_values(native, source, tmp_path):
    a = ad.read_h5ad(source)
    transposed = tmp_path/'features-by-cells.h5ad'; a.T.write_h5ad(transposed)
    before = sha(transposed)
    result = call(transposed, tmp_path/'transpose', gene_activity=True, input_orientation='features_by_cells', count_slot='scores')
    b = ad.read_h5ad(result['matrix_path'])
    np.testing.assert_array_equal(array(b.X), array(a.layers['scores']))
    assert b.obs_names.tolist() == a.obs_names.tolist() and b.var_names.tolist() == a.var_names.tolist()
    assert result['parameters']['input_orientation'].startswith('explicit native AnnData transpose')
    assert sha(transposed) == before


def test_native_subset_preserves_order_layers_features_and_global_slots(native, source, tmp_path):
    a = ad.read_h5ad(source); before = sha(source)
    choose = selection(source, [3, 0, 2, 1], groupby='group', groups=['left'])
    result = call(source, tmp_path/'subset', gene_activity=True, cell_subset=choose)
    b = ad.read_h5ad(result['matrix_path']); expected = a[[3, 0, 2, 1], :]
    assert b.obs_names.tolist() == choose['cell_ids']
    assert b.var_names.tolist() == a.var_names.tolist()
    np.testing.assert_array_equal(array(b.X), array(expected.X))
    np.testing.assert_array_equal(array(b.layers['scores']), array(expected.layers['scores']))
    np.testing.assert_array_equal(b.obsm['X_umap'], expected.obsm['X_umap'])
    np.testing.assert_array_equal(array(b.obsp['global_graph']), array(expected.obsp['global_graph']))
    recorded = result['parameters']['cell_subset']
    assert recorded['positions'] == choose['positions'] and recorded['source_axes'] == axis_identity(source)
    assert recorded['provenance'].startswith('exact source obs membership')
    assert not result['parameters'].get('graph_inputs')
    assert not {'obsm.X_umap', 'obsp.global_graph'} & set(result['result_slots'])
    assert sha(source) == before


@pytest.mark.parametrize('change', ['bytes', 'identities', 'positions', 'group'])
def test_subset_never_substitutes_cells(source, tmp_path, change):
    choose = selection(source, [0, 1, 2, 3], groupby='group', groups=['left'])
    if change == 'bytes': choose['source_sha256'] = '0'*64
    if change == 'identities': choose['cell_ids'][0] = 'absent'
    if change == 'positions': choose['positions'][0] = 7
    if change == 'group': choose['groups'] = ['right']
    stage = tmp_path/'stage'; stage.mkdir()
    with pytest.raises(AuthorityError):
        prepare_input(dict(input_path=str(source), cell_subset=choose), stage)


def test_supplied_activity_does_not_gain_authority_without_declaration(native, source, tmp_path):
    a = ad.read_h5ad(source); a.uns['agent_gene_activity'] = 'pretend accepted'; a.write_h5ad(source)
    result = call(source, tmp_path/'ordinary')
    assert 'gene_activity' not in result['parameters']
    assert 'accepted_publication' not in result['source_identity']


def test_single_cell_reaches_real_native_marker_failure(native, source, tmp_path):
    args = dict(input_path=str(source), output_dir=str(tmp_path/'failure'), function_name='tl.rank_genes_groups',
        gene_activity=True, cell_subset=selection(source, [0]), kwargs=dict(groupby='group', method='t-test', use_raw=False))
    with pytest.raises(NativeInvocationError): SCANPY.execute(args)
    log = (tmp_path/'failure/native-failure.log').read_text()
    assert 'sample' in log or 'group' in log
    assert not (tmp_path/'failure/manifest.json').exists()


@pytest.mark.parametrize('route', ['snapatac2', 'episcanpy', 'existing'])
def test_three_routes_one_registered_marker_path_and_prior_subset(native, source, tmp_path, route, monkeypatch):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
    from agent.orchestration.prior_outputs import bind_output, accepted_step_digest
    from agent.schemas.prior_output import PriorOutputRef
    from agent.application.statistical_delivery import create_statistical_export
    from agent.application.workspace import ManagedWorkspace
    a = ad.read_h5ad(source)
    gtf = tmp_path/'genes.gtf'
    gtf.write_text(''.join(f'chr1\tHAVANA\t{kind}\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gid}"; transcript_id "{tid}"; gene_name "{name}";\n'
        for kind in ('gene', 'transcript') for start,end,strand,gid,tid,name in
        [(101,200,'+','G1','T1','Alpha'), (801,1000,'-','G2','T2','Beta')]))
    peaks = a.copy(); peaks.var_names = ['chr1:100-120', 'chr1:900-920']; peaks.uns['matrix_semantics'] = 'peak_counts'
    peak_path = tmp_path/'peaks.h5ad'; peaks.write_h5ad(peak_path)
    if route == 'snapatac2':
        # A fresh researcher input with an explicit grouping. Never edit the accepted import.
        fragments = tmp_path/'fragments.tsv'
        fragments.write_text(''.join(f'chr1\t{100+j}\t{120+j}\t{cell}\t1\n' if feature == 0 else
            f'chr1\t{900+j}\t{920+j}\t{cell}\t1\n'
            for i,cell in enumerate(a.obs_names) for feature in range(2) for j in range(int(array(a.X)[i,feature]))))
        imported = SNAPATAC2.execute(dict(function_name='pp.import_fragments', input_path=str(fragments),
            output_dir=str(tmp_path/'import'), kwargs=dict(chrom_sizes={'chr1': 2000}, min_num_fragments=0)))
        import_sha = sha(imported['matrix_path'])
        grouped = ad.read_h5ad(imported['matrix_path']); grouped.obs['group'] = a.obs['group']
        input_path = tmp_path/'declared-group-fragments.h5ad'; grouped.write_h5ad(input_path)
        first_tool, first_args = 'run_snapatac2', dict(function_name='pp.make_gene_matrix', gene_anno_path=str(gtf), kwargs=dict(upstream=0))
    elif route == 'episcanpy':
        input_path = peak_path
        first_tool, first_args = 'run_episcanpy', dict(function_name='tl.geneactivity', gene_anno_path=str(gtf), kwargs=dict(upstream=0))
    else:
        input_path = source
        first_tool, first_args = 'run_scanpy', dict(function_name='pp.calculate_qc_metrics', gene_activity=True, kwargs=dict(percent_top=None, log1p=False))
    original_sha = sha(input_path)
    root = tmp_path/'workspace'
    scientific = root/'runs'/ManagedWorkspace.run_digest('common:run')/'scientific'
    plan = AgentPlan('common', 'common', 'Explicit native activity, log1p, markers', (
        PlanStep('activity', first_tool, dict(input_path=str(input_path), output_dir=str(scientific/'activity'), **first_args)),
        PlanStep('log', 'run_scanpy', dict(function_name='pp.log1p', input_path=StepOutputRef('activity', 'matrix_path'), output_dir=str(scientific/'log')), ('activity',)),
        PlanStep('markers', 'run_scanpy', dict(function_name='tl.rank_genes_groups', input_path=StepOutputRef('log', 'matrix_path'),
            kwargs=dict(groupby='group', method='t-test', use_raw=False), output_dir=str(scientific/'markers')), ('log',))))
    class Planner:
        def plan(self, request, registry): return plan
    app = ResearchAgentApplication(root, planner=Planner()); app.sessions.create('activity')
    state = app.sessions.run('activity', 'common', AgentRequest('common', 'Declared native marker comparison', {}),
        (OutputSelection('dataset', 'markers', 'matrix_path'),), expected_generation=0)
    run = app.run_store.load('common:run')
    assert state.turn('common').status == 'activated', run.errors
    results = [step.result for step in run.steps]
    activity_result, log_result, marker_result = results
    for result in results:
        assert result['parameters']['gene_activity']['interface'] == 'gene-activity.v1'
    np.testing.assert_allclose(array(ad.read_h5ad(log_result['matrix_path']).X), np.log1p(array(ad.read_h5ad(activity_result['matrix_path']).X)))
    marker = ad.read_h5ad(marker_result['matrix_path'])
    assert marker.uns['rank_genes_groups']['params']['groupby'] == 'group'
    assert marker.uns['rank_genes_groups']['params']['use_raw'] == False
    assert marker.uns['rank_genes_groups']['names']['left'][0] == 'Alpha'
    assert marker.uns['rank_genes_groups']['names']['right'][0] == 'Beta'
    assert marker_result['source_identity']['accepted_publication']['manifest_sha256'] == log_result['manifest_sha256']
    assert results[0]['parameters']['gene_activity']['construction']['origin'] == ('supplied_declaration' if route == 'existing' else 'recorded_native_execution')
    # Restart and use the same accepted activity for an exact local comparison.
    app = ResearchAgentApplication(root)
    assert app.sessions.load('activity') == state
    binding = bind_output(app.run_store, app.registry, run.run_id, 'log', 'dataset')
    local_selection = selection(log_result['matrix_path'], [2, 0, 6, 4])
    class LocalPlanner:
        def plan(self, request, registry):
            return AgentPlan('local', request.request_id, 'Exact selected comparison population', (
                PlanStep('local', 'run_scanpy', dict(function_name='tl.rank_genes_groups', input_path=PriorOutputRef(binding, 'matrix_path'),
                    cell_subset=local_selection, kwargs=dict(groupby='group', method='t-test', use_raw=False),
                    output_dir=str(Path(request.inputs['output_dir'])/'local'))),))
    # The constructors cannot run while accepted input is being reused.
    monkeypatch.setattr(EPISCANPY, 'native', lambda *args: pytest.fail('Recomputed epiScanpy activity'))
    monkeypatch.setattr(SNAPATAC2, 'native', lambda *args: pytest.fail('Recomputed SnapATAC2 activity'))
    outcome = ResearchAgentApplication(root, planner=LocalPlanner()).run(AgentRequest('local', 'Local markers', {}))
    assert outcome.status.value == 'SUCCEEDED', outcome.run_result.errors
    local = outcome.run_result.steps[0].result
    local_data = ad.read_h5ad(local['matrix_path'])
    assert local_data.obs_names.tolist() == local_selection['cell_ids']
    np.testing.assert_array_equal(array(local_data.X), array(marker.X)[[2, 0, 6, 4]])
    assert _serialize(local['parameters']['gene_activity']['axes']) == axis_identity(local['matrix_path'])
    assert local['source_identity']['accepted_publication']['manifest_sha256'] == log_result['manifest_sha256']
    # Explicit supplied association to original accessibility; equal sizes alone confer no relation.
    subset_peaks = call(peak_path, tmp_path/'local-peaks', cell_subset=selection(peak_path, [2, 0, 6, 4]))
    p = ad.read_h5ad(subset_peaks['matrix_path'])
    assert p.obs_names.tolist() == local_data.obs_names.tolist()
    assert p.var_names.tolist() != local_data.var_names.tolist()
    np.testing.assert_array_equal(array(p.X), array(peaks.X)[[2, 0, 6, 4]])
    assert subset_peaks['parameters']['cell_subset']['provenance'] == 'exact source positions'
    # Existing download owner delivers the exact native marker comparison and its description.
    target = io.BytesIO(); path = Path(local['matrix_path'])
    with path.open('rb') as stream:
        create_statistical_export(stream, target, dict(tool_name='run_scanpy', accepted_step_sha256=accepted_step_digest(outcome.run_result.steps[0])),
            source_sha256=local['matrix_sha256'], source_size_bytes=path.stat().st_size, manifest_sha256=local['manifest_sha256'], max_bytes=1024**3)
    delivered = ad.read_h5ad(target)
    assert delivered.obs_names.tolist() == local_selection['cell_ids']
    np.testing.assert_array_equal(delivered.uns['rank_genes_groups']['scores'], local_data.uns['rank_genes_groups']['scores'])
    assert json.loads(delivered.uns['agent_gene_activity'])['description']['interface'] == 'gene-activity.v1'
    assert sha(input_path) == original_sha
    if route == 'snapatac2': assert sha(imported['matrix_path']) == import_sha


def test_local_native_graph_replaces_inherited_graph_provenance(native, tmp_path):
    from agent.application import ResearchAgentApplication
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    rng = np.random.default_rng(17)
    x = rng.poisson(1 + np.repeat(np.eye(3), 20, axis=1)[np.arange(60) % 3] * 7).astype('float32')
    a = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame(index=[f'cell-{i}' for i in range(60)]))
    a.uns['matrix_semantics'] = 'peak_counts'
    source = tmp_path/'accessibility.h5ad'; a.write_h5ad(source)
    root = tmp_path/'app'
    class GlobalPlanner:
        def plan(self, request, registry):
            return AgentPlan('global', request.request_id, 'Explicit native global graph', (
                PlanStep('graph', 'run_scanpy', dict(input_path=str(source), function_name='pp.neighbors',
                    kwargs=dict(use_rep='X', n_neighbors=5), output_dir=str(Path(request.inputs['output_dir'])/'graph'))),))
    app = ResearchAgentApplication(root, planner=GlobalPlanner())
    global_run = app.run(AgentRequest('global', 'Build supplied global graph', {}))
    assert global_run.status.value == 'SUCCEEDED', global_run.run_result.errors
    accepted = global_run.run_result.steps[0].result
    before = sha(accepted['matrix_path'])
    binding = bind_output(app.run_store, app.registry, 'global:run', 'graph', 'dataset')
    selected = selection(accepted['matrix_path'], list(range(44, -1, -1)))
    class LocalPlanner:
        def plan(self, request, registry):
            out = Path(request.inputs['output_dir'])
            return AgentPlan('local', request.request_id, 'Explicit local native accessibility analysis', (
                PlanStep('subset', 'run_snapatac2', dict(input_path=PriorOutputRef(binding, 'matrix_path'), cell_subset=selected,
                    function_name='pp.select_features', kwargs=dict(n_features=40, filter_lower_quantile=0, filter_upper_quantile=1), output_dir=str(out/'subset'))),
                PlanStep('spectral', 'run_snapatac2', dict(input_path=StepOutputRef('subset', 'matrix_path'), function_name='tl.spectral',
                    kwargs=dict(n_comps=5, random_state=0), output_dir=str(out/'spectral')), ('subset',)),
                PlanStep('neighbors', 'run_snapatac2', dict(input_path=StepOutputRef('spectral', 'matrix_path'), function_name='pp.knn',
                    kwargs=dict(n_neighbors=5), output_dir=str(out/'neighbors')), ('spectral',)),
                PlanStep('clusters', 'run_snapatac2', dict(input_path=StepOutputRef('neighbors', 'matrix_path'), function_name='tl.leiden',
                    kwargs=dict(resolution=0.37), parameter_selection='asw', output_dir=str(out/'clusters')), ('neighbors',))))
    outcome = ResearchAgentApplication(root, planner=LocalPlanner()).run(AgentRequest('local', 'Assess this exact local population', {}))
    assert outcome.status.value == 'SUCCEEDED', outcome.run_result.errors
    subset, spectral, neighbors, clustered = [step.result for step in outcome.run_result.steps]
    assert not subset['parameters'].get('graph_inputs')
    assert 'obsp.connectivities' not in subset['result_slots']
    graph = neighbors['parameters']['graph_inputs'][0]
    assert graph['function'] == 'snapatac2.pp.knn'
    assert graph['source_sha256'] == spectral['matrix_sha256']
    assert graph['cells_sha256'] != accepted['parameters']['graph_inputs'][0]['cells_sha256']
    policy = clustered['parameters']['parameter_selection']
    assert policy['origin'] == 'explicit' and policy['selected_value'] == 0.37 and not policy['candidates']
    for result in (subset, spectral, neighbors, clustered):
        b = ad.read_h5ad(result['matrix_path'])
        assert b.obs_names.tolist() == selected['cell_ids']
        np.testing.assert_array_equal(array(b.X), x[selected['positions']])
    assert sha(accepted['matrix_path']) == before


def test_accepted_construction_uses_recorded_result_not_optional_metadata(source, tmp_path, monkeypatch):
    """The ICTA-2A descriptor already has authoritative native result facts."""
    from agent.tools.analysis import native_gene_activity as activity
    a = ad.read_h5ad(source); target = tmp_path/'result.h5ad'; a.write_h5ad(target)
    accepted = dict(function_name='episcanpy.tl.geneactivity', measurement='episcanpy_gene_domain_sum',
        source_identity=dict(sha256='1'*64), manifest_path='accepted/manifest.json', manifest_sha256='2'*64,
        parameters=dict(effective_native_parameters=dict(upstream=50, raw=False),
            gene_activity=dict(interface='gene-activity.v1', feature_type='gene')))
    monkeypatch.setattr(activity, 'accepted_source', lambda *args: accepted)
    summary = dict(function_name='scanpy.tl.rank_genes_groups', measurement='episcanpy_gene_domain_sum',
        parameters=dict(count_slot='X'), source_identity=dict(sha256=sha(source)), result_slots=[])
    activity.reuse_description(target, dict(input_path=str(source)), summary, dict(sha256=sha(source)))
    construction = summary['parameters']['gene_activity']['construction']
    assert construction == dict(origin='recorded_native_execution', method='episcanpy.tl.geneactivity',
        source_sha256='1'*64, measurement='episcanpy_gene_domain_sum', effective_parameters=dict(upstream=50, raw=False))
    assert summary['source_identity']['accepted_publication']['manifest_sha256'] == '2'*64


@pytest.mark.parametrize('positions,group', [([3, 1], True), ([3, 1, 3], False)])
def test_native_partial_and_repeated_positions_preserve_actual_rows(native, source, tmp_path, positions, group):
    a = ad.read_h5ad(source)
    choose = dict(positions=positions)
    if group: choose.update(groupby='group', groups=['left'])
    result = call(source, tmp_path/'partial', cell_subset=choose)
    b = ad.read_h5ad(result['matrix_path']); expected = a[positions, :]
    assert b.obs_names.tolist() == expected.obs_names.tolist()
    np.testing.assert_array_equal(array(b.X), array(expected.X))
    np.testing.assert_array_equal(array(b.layers['scores']), array(expected.layers['scores']))
    recorded = result['parameters']['cell_subset']
    assert recorded['positions'] == positions and recorded['cell_ids'] == expected.obs_names.tolist()
    assert recorded['source_sha256'] == sha(source)
    if group: assert recorded['complete_group_membership'] is False
    # Recovery uses the exact publication, including repeated observation names.
    args = dict(input_path=str(source), input_sha256=sha(source), output_dir=str(tmp_path/'partial'),
        function_name='pp.calculate_qc_metrics', kwargs=dict(percent_top=None, log1p=False), cell_subset=choose)
    assert SCANPY.recover(args, None) == result


def test_annotation_fixture_derives_cluster_population_for_native_children(native, source, tmp_path, monkeypatch):
    """One fixture parent, two real ICTA-1 children; no biological controller."""
    from agent.application import ResearchAgentApplication
    from agent.application.result_reading import handoff
    from agent.orchestration import AgentRuntime, AgentPlan, AgentRequest, PlanStep, StepOutputRef
    from agent.orchestration.annotation_runs import annotation_invocation
    from agent.orchestration.prior_outputs import bind_output
    from agent.orchestration.registry import (ToolRegistry, ToolSpec, ArgumentSpec, ResultContract,
        DurableToolHooks, ErrorClassification, SemanticToolSpec, SemanticConsumerPortSpec, SemanticPortMember)
    from agent.schemas import ErrorCategory, TraceEventType
    from agent.schemas.prior_output import PriorOutputRef
    from agent.tools.analysis.categorical import observations
    a = ad.read_h5ad(source); a.var_names = ['chr1:100-120', 'chr1:900-920']
    a.uns['matrix_semantics'] = 'peak_counts'; a.write_h5ad(source)
    before = sha(source)
    gtf = tmp_path/'genes.gtf'
    gtf.write_text('chr1\tHAVANA\tgene\t101\t200\t.\t+\t.\tgene_name "Alpha";\n'
                   'chr1\tHAVANA\tgene\t801\t1000\t.\t-\t.\tgene_name "Beta";\n')
    class Planner:
        def plan(self, request, registry):
            out = Path(request.inputs['output_dir'])
            return AgentPlan('accepted', request.request_id, 'Native clustering and associated activity', (
                PlanStep('neighbors', 'run_scanpy', dict(input_path=str(source), function_name='pp.neighbors',
                    kwargs=dict(use_rep='X', n_neighbors=3), output_dir=str(out/'neighbors'))),
                PlanStep('clusters', 'run_scanpy', dict(input_path=StepOutputRef('neighbors', 'matrix_path'),
                    function_name='tl.leiden', kwargs=dict(resolution=0.3, random_state=0), output_dir=str(out/'clusters')), ('neighbors',)),
                PlanStep('activity', 'run_episcanpy', dict(input_path=StepOutputRef('clusters', 'matrix_path'),
                    function_name='tl.geneactivity', gene_anno_path=str(gtf), kwargs=dict(upstream=0), output_dir=str(out/'activity')), ('clusters',))))
    app = ResearchAgentApplication(tmp_path/'app', planner=Planner())
    outcome = app.run(AgentRequest('accepted', 'Prepare exact scientific inputs', {}))
    assert outcome.status.value == 'SUCCEEDED', outcome.run_result.errors
    state = app.run_store.load('accepted:run'); original_state = state.to_dict()
    clustered, activity = [state.steps[i].result for i in (1, 2)]
    # The clustering publication retains original accessibility, not a gene proxy.
    np.testing.assert_array_equal(array(ad.read_h5ad(clustered['matrix_path']).X), array(a.X))
    assert activity['source_identity']['sha256'] == clustered['matrix_sha256']
    assert activity['source_identity']['accepted_publication']['manifest_sha256'] == clustered['manifest_sha256']
    bound = {name: bind_output(app.run_store, app.registry, state.run_id, step, 'dataset')
        for name,step in [('accessibility_path','clusters'), ('activity_path','activity')]}
    pins = {r['matrix_path']: sha(r['matrix_path']) for r in (clustered, activity)}
    supplied_intents = []
    def execute(args, identity):
        # The scientific owner supplies the actual accepted grouping field and
        # native labels. Choosing one label here is a test fixture, not recursion.
        actual = handoff(app.run_store, state.run_id, 'clusters')['groups']
        cells, labels = observations(actual['path'], actual['sha256'], actual['field'])
        chosen = labels.iloc[0]
        intent = dict(groupby=actual['field'], groups=[chosen])
        assert set(intent) == {'groupby', 'groups'}
        inv = annotation_invocation()
        for name in bound:
            supplied_intents.append(dict(intent))
            inv.invoke(branch_id='fixture-cluster', operation_id=name, tool_name='run_scanpy',
                arguments=dict(function_name='pp.calculate_qc_metrics', cell_subset=intent,
                    kwargs=dict(percent_top=None, log1p=False)), parent_inputs=dict(input_path=name))
        return dict(status='fixture_complete')
    parent = ToolSpec('fixture_population_handoff', lambda **args: pytest.fail('Durable hook required'),
        {name: ArgumentSpec((str,)) for name in (*bound, 'output_dir')}, {}, ResultContract('Fixture', {'status':(str,)}),
        lambda exc: ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'FIXTURE_FAILURE', str(exc)),
        durable_hooks=DurableToolHooks(execute, lambda *args: pytest.fail('Fixture parent has no scientific publication')),
        semantic_planning=SemanticToolSpec(consumer_ports=tuple(SemanticConsumerPortSpec(name,
            (SemanticPortMember('dataset',name),), True, accepted_upstream_types=('native_matrix_h5ad.v1',)) for name in bound)),
        annotation_children=frozenset({'run_scanpy'}))
    registry = ToolRegistry(tuple(app.registry.get(name) for name in app.registry.names()) + (parent,))
    monkeypatch.setattr('agent.orchestration.registry.build_default_tool_registry', lambda: registry)
    monkeypatch.setattr('agent.orchestration.verification_authority.build_default_tool_registry', lambda: registry)
    runtime = AgentRuntime(registry=registry, run_store=app.run_store)
    args = {name: PriorOutputRef(binding,'matrix_path') for name,binding in bound.items()}
    request = AgentRequest('parent', None, {})
    plan = AgentPlan('parent', 'parent', 'Fixture annotation-owned handoff', (
        PlanStep('annotate', parent.name, dict(args, output_dir=str(tmp_path/'parent'))),))
    result = runtime.run_prepared(request, plan)
    assert result.status.value == 'SUCCEEDED', result.errors
    parent_state = app.run_store.load('parent:run')
    assert len(parent_state.plan.steps) == 1
    children = [app.run_store.load(e.details['child_run_id']) for e in parent_state.trace
        if e.event_type is TraceEventType.ANNOTATION_PROGRESS and e.details['event'] == 'outcome']
    assert len(children) == 2 and supplied_intents[0] == supplied_intents[1]
    group = supplied_intents[0]
    clustered_data = ad.read_h5ad(clustered['matrix_path'])
    positions = np.flatnonzero(clustered_data.obs[group['groupby']].isin(group['groups'])).tolist()
    for child, parent_source in zip(children, (clustered, activity)):
        assert child.lifecycle_status.value == 'SUCCEEDED'
        step = child.steps[0]; selected = step.result
        assert step.verification.passed and step.verification.artifact_authority
        assert _serialize(step.resolved_arguments['cell_subset']) == group
        recorded = selected['parameters']['cell_subset']
        assert list(recorded['positions']) == positions
        assert list(recorded['cell_ids']) == clustered_data.obs_names[positions].tolist()
        assert recorded['source_sha256'] == parent_source['matrix_sha256']
        assert recorded['complete_group_membership'] is True
        assert selected['source_identity']['accepted_publication']['manifest_sha256'] == parent_source['manifest_sha256']
        expected = ad.read_h5ad(parent_source['matrix_path'])[positions, :]
        actual = ad.read_h5ad(selected['matrix_path'])
        assert actual.obs_names.tolist() == expected.obs_names.tolist()
        assert actual.var_names.tolist() == expected.var_names.tolist()
        np.testing.assert_array_equal(array(actual.X), array(expected.X))
    assert sha(source) == before and all(sha(path) == pin for path,pin in pins.items())
    assert app.run_store.load(state.run_id).to_dict() == original_state
    assert len(app.run_store.load('parent:run').steps) == 1
