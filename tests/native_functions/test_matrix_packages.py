"""Native adapter contracts; numerical parity is independently exercised opt-in."""
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import native_matrix as owner


@pytest.fixture
def source(tmp_path):
    a = ad.AnnData(sparse.csr_matrix([[0, 2, 1], [3, 0, 0]], dtype=np.int32),
        obs=pd.DataFrame({'group': ['B', 'A']}, index=['002', '001']),
        var=pd.DataFrame(index=['03', '01', '02']))
    a.uns['matrix_semantics'] = 'peak_counts'
    p = tmp_path / 'ordinary.data'; a.write_h5ad(p)
    return dict(input_path=str(p), output_dir=str(tmp_path / 'out'), function_name='pp.binarize')


@pytest.fixture
def simulated(monkeypatch):
    calls = []
    def native(self, args, stage, matrix):
        calls.append(args)
        a = ad.read_h5ad(stage / 'input.h5ad')
        a.X.data[:] = 1
        a.write_h5ad(stage / 'result.h5ad')
        return dict(completion='succeeded', versions={self.package: 'SIMULATED'}, implementation=[],
            native_output_type='NoneType', result_slots=['X'], measurement='binary_accessibility')
    monkeypatch.setattr(owner.MatrixPackage, 'native', native)
    return calls


@pytest.mark.parametrize('storage', ['dense', 'csr', 'csc'])
def test_axes_measurement_and_managed_copy(source, tmp_path, storage):
    a = ad.read_h5ad(source['input_path'])
    a.X = {'dense': a.X.toarray(), 'csr': a.X, 'csc': a.X.tocsc()}[storage]
    a.write_h5ad(source['input_path']); before = Path(source['input_path']).read_bytes()
    stage = tmp_path / 'stage'; stage.mkdir()
    prepared, effective = owner.prepare_input(source, stage)
    assert prepared.obs_names.tolist() == ['002', '001']
    assert prepared.var_names.tolist() == ['03', '01', '02']
    assert effective['matrix_semantics'] == 'peak_counts'
    np.testing.assert_array_equal(prepared.X.toarray() if sparse.issparse(prepared.X) else prepared.X,
        [[0, 2, 1], [3, 0, 0]])
    assert Path(source['input_path']).read_bytes() == before


def test_layer_requires_own_measurement_and_conflicts_fail(source, tmp_path):
    a = ad.read_h5ad(source['input_path']); a.layers['other'] = a.X.copy(); a.write_h5ad(source['input_path'])
    stage = tmp_path / 'stage'; stage.mkdir()
    _, effective = owner.prepare_input(dict(source, count_slot='other'), stage)
    assert effective['matrix_semantics'] == 'unspecified_retained_measurement'
    with pytest.raises(ValueError, match='conflicts'):
        owner.prepare_input(dict(source, matrix_semantics='insertion_counts'), stage)
    a, effective = owner.prepare_input(dict(source, count_slot='other', matrix_semantics='insertion_counts'), stage)
    assert effective['count_slot'] == 'layers/other' and a.uns['matrix_semantics'] == 'insertion_counts'


@pytest.mark.parametrize('package,function,kwargs', [
    (owner.EPISCANPY, 'eval', {}), (owner.EPISCANPY, 'pp.tfidf', {'adata': '/etc/passwd'}),
    (owner.SNAPATAC2, 'tl.spectral', {'features': '/etc/passwd'}),
    (owner.SNAPATAC2, 'pp.select_features', {'blacklist': '/etc/passwd'}),
    (owner.SIGNAC, 'RunTFIDF', {'idf': {'code': 'system()'}}),
])
def test_dispatch_has_no_unbound_objects_code_or_files(source, package, function, kwargs):
    with pytest.raises(ValueError):
        package.execute(dict(source, function_name=function, kwargs=kwargs))
    assert not Path(source['output_dir']).exists()


def test_explicit_values_and_omitted_defaults_are_not_repaired():
    assert owner.EPISCANPY.native_arguments({'function_name': 'pp.lsi', 'kwargs': {'n_components': 513}}) == {'n_components': 513}
    assert owner.SIGNAC.native_arguments({'function_name': 'RunTFIDF', 'kwargs': {'method': -7}}) == {'method': -7}
    assert owner.EPISCANPY.native_arguments({'function_name': 'pp.tfidf'}) == {}
    invalid = {'norm': ['native', {'must': 'reject'}]}
    assert owner.EPISCANPY.native_arguments({'function_name': 'pp.tfidf', 'kwargs': invalid}) == invalid
    with pytest.raises(ValueError, match='mask'):
        owner.SNAPATAC2.native_arguments({'function_name': 'tl.spectral', 'kwargs': {'features': [True, False]}})


@pytest.mark.parametrize('function', ['pp.log1p', 'pp.tfidf', 'pp.select_features', 'tl.spectral', 'RunTFIDF'])
def test_signed_numeric_values_reach_the_native_function(source, tmp_path, function):
    a = ad.read_h5ad(source['input_path'])
    a.X = a.X.astype(float); a.X.data[0] = -.5
    a.uns['matrix_semantics'] = 'transformed_numeric'; a.write_h5ad(source['input_path'])
    stage = tmp_path / 'stage'; stage.mkdir()
    args = dict(source, function_name=function)
    prepared, _ = owner.prepare_input(args, stage)
    assert prepared.X.data[0] == -.5


def test_native_copy_retains_scientific_anndata_contents(source, simulated):
    a = ad.read_h5ad(source['input_path'])
    a.var['selected'] = [True, False, True]
    a.layers['counts'] = a.X.copy(); a.layers['other'] = a.X.astype(float) * .5
    a.obsm['existing_embedding'] = np.array([[1., 2.], [3., 4.]])
    a.varm['loadings'] = np.ones((3, 2))
    a.obsp['distances'] = sparse.eye(2, format='csr')
    a.varp['correlation'] = sparse.eye(3, format='csr')
    a.uns['experiment'] = {'assay': 'ATAC', 'conditions': np.array(['control', 'treated'])}
    a.raw = a.copy(); a.write_h5ad(source['input_path'])
    before = Path(source['input_path']).read_bytes()
    result = owner.EPISCANPY.execute(source, 'fidelity')
    b = ad.read_h5ad(result['matrix_path'])
    pd.testing.assert_frame_equal(a.obs, b.obs); pd.testing.assert_frame_equal(a.var, b.var)
    for field in ('layers', 'obsm', 'varm', 'obsp', 'varp'):
        for key, value in getattr(a, field).items():
            actual = getattr(b, field)[key]
            if sparse.issparse(value):
                assert type(value) is type(actual) and value.dtype == actual.dtype and (value != actual).nnz == 0
            else:
                np.testing.assert_array_equal(value, actual)
    assert (a.raw.X != b.raw.X).nnz == 0
    assert a.raw.var_names.tolist() == b.raw.var_names.tolist()
    assert b.uns['experiment']['assay'] == 'ATAC'
    np.testing.assert_array_equal(b.uns['experiment']['conditions'], a.uns['experiment']['conditions'])
    assert 'not recomputed' in b.uns['agent_native_invocation']['annotation_policy']
    assert Path(source['input_path']).read_bytes() == before


def test_r_exchange_preserves_quotes_na_and_numeric_text_ids(source, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import csv
    a = ad.read_h5ad(source['input_path'])
    a.obs_names = ['"cell"', 'NA']; a.var_names = ['"peak"', 'NA', '001']
    a.write_h5ad(source['input_path'])
    runtime = tmp_path / 'runtimes/signac-1/env/bin/Rscript'; runtime.parent.mkdir(parents=True)
    runtime.write_text('simulated executable identity')
    monkeypatch.setattr(owner, 'library', lambda: SimpleNamespace(root=tmp_path / 'library'))
    def run(command, log, **unused):
        request_path = Path(command[-1]); req = json.loads(request_path.read_text()); stage = request_path.parent
        assert (stage / 'cells.tsv').read_text().splitlines() == ['cell_id', '"cell"', 'NA']
        assert (stage / 'features.tsv').read_text().splitlines() == ['feature_id', '"peak"', 'NA', '001']
        pd.DataFrame({'feature_id': ['001', 'NA', '"peak"'], 'count': [1., 2., 3.], 'percentile': [.3, .6, 1.]}).to_csv(
            stage / 'feature-statistics.tsv', sep='\t', index=False, quoting=csv.QUOTE_NONE)
        owner.write(stage / 'native-receipt.json', dict(completion='succeeded', package='Signac',
            function_name=req['function_name'], request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
            versions={'Signac': '1.16.0'}, native_output_type='data.frame', result_slots=['var.Signac_count']))
        log.write_text('simulated native diagnostic')
        return 0
    monkeypatch.setattr(owner, 'run', run)
    result = owner.SIGNAC.execute(dict(source, function_name='FindTopFeatures'), 'exact')
    output = ad.read_h5ad(result['matrix_path'])
    assert output.var_names.tolist() == ['"peak"', 'NA', '001']
    assert output.var.Signac_count.tolist() == [3., 2., 1.]
    assert output.var.Signac_native_rank.tolist() == [2, 1, 0]


def test_integrity_recovery_no_retry_and_original_bytes(source, simulated):
    before = Path(source['input_path']).read_bytes()
    result = owner.EPISCANPY.execute(source, 'exact')
    assert owner.EPISCANPY.recover(source, 'exact') == result and len(simulated) == 1
    with pytest.raises(ValueError, match='retry'):
        owner.EPISCANPY.recover(source, 'different')
    with pytest.raises(ValueError, match='not empty'):
        owner.EPISCANPY.execute(source, 'exact')
    assert Path(source['input_path']).read_bytes() == before
    Path(result['matrix_path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        owner.EPISCANPY.verify_publication(result['manifest_path'], expected_sha256=result['manifest_sha256'])


def test_catalog_growth_preserves_adapter_identity_and_old_profiles_are_read_only(source, simulated):
    from types import SimpleNamespace
    p = owner.EPISCANPY
    extended = owner.MatrixPackage(p.package, p.version, p.runtime,
        {**p.FUNCTIONS, 'test_only_declaration': ()}, p.NATIVE_ARGUMENTS, p.guidance)
    assert extended.PROFILE == p.PROFILE
    result = p.execute(source, 'exact')
    path = Path(result['manifest_path']); value = owner.read(path)
    value['profile'] = owner.HISTORICAL_PROFILES[p.package]
    path.unlink()  # Replace only this test-owned, unaccepted fixture.
    sha = owner.write(path, value)
    context = SimpleNamespace(files=owner.file_closure, execution_for=lambda *args: 'exact')
    description = p.publication.describe(path, sha, context)
    assert description['profile'] == description['proof_identity']['profile'] == value['profile']
    historical = p.result(value, path, sha)
    from agent.orchestration.native_delivery import delivery_spec
    spec = delivery_spec(p, 'test')
    authority = dict(artifact_contract=p.CONTRACT, science_profile=value['profile'], verifier=p.VERIFIER)
    assert spec.publication(source, historical, authority) == (value, sha)
    with pytest.raises(ValueError, match='producing request'):
        spec.publication(source, historical, dict(authority, science_profile=p.PROFILE))
    with pytest.raises(ValueError, match='producing execution'):
        p.publication.publication_record(context, p.TOOL, source, historical, 'exact', historical)
    value['profile'] = 'unknown-profile'; path.unlink(); sha = owner.write(path, value)
    with pytest.raises(ValueError, match='identity'):
        p.load(path, sha)


def test_axis_substitution_cannot_publish(source, monkeypatch):
    def corrupt(self, args, stage, matrix):
        matrix.obs_names = ['wrong1', 'wrong2']; matrix.write_h5ad(stage / 'result.h5ad')
        return {}
    monkeypatch.setattr(owner.MatrixPackage, 'native', corrupt)
    with pytest.raises(ValueError, match='ordered input axes'):
        owner.EPISCANPY.execute(source, 'exact')
    assert not (Path(source['output_dir']) / 'manifest.json').exists()


def test_accepted_delivery_restart_and_deleted_source(source, simulated, tmp_path):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    root = tmp_path / 'workspace'
    source['output_dir'] = str(root / 'runs' / ManagedWorkspace.run_digest('first:run') / 'scientific' / 'native')
    class Planner:
        def plan(self, request, registry):
            return AgentPlan('test', request.request_id, 'Simulated native function', (PlanStep('native', owner.EPISCANPY.TOOL, source),))
    app = ResearchAgentApplication(root, planner=Planner()); app.sessions.create('session')
    state = app.sessions.run('session', 'first', AgentRequest('first', 'Binarize', {}),
        (OutputSelection('result', 'native', 'manifest_path'),), expected_generation=0)
    assert state.turn('first').status == 'activated', app.run_store.load('first:run').errors
    for artifact in accepted_artifacts(app.sessions, 'session', state.revisions[-1], state.turn('first'), None, step_id='native'):
        output = MatrixDownloadPreparer().prepare(artifact)
        try:
            data = b''.join(output.iter_chunks())
            assert hashlib.sha256(data).hexdigest() == output.sha256 and len(data) == output.size_bytes
            assert output.source_sha256 == artifact.sha256
        finally:
            output.close()
    Path(source['input_path']).unlink()
    app = ResearchAgentApplication(root, planner=Planner())
    assert app.sessions.load('session').to_dict() == state.to_dict()
    assert len(accepted_artifacts(app.sessions, 'session', state.revisions[-1], state.turn('first'), None, step_id='native')) == 3
    assert len(simulated) == 1


def test_grounded_native_members_use_shared_binding():
    from agent.application.turn_decisions import parse_decision
    from agent.application.dialogue_execution import _bind_arguments
    from agent.orchestration import build_default_tool_registry
    text = 'Run Signac RunTFIDF with method=2 and scale.factor=5000.'
    wire = dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='run_signac', arguments=[
        dict(tool='run_signac', argument='kwargs', members=[
            dict(name='method', kind='number', literal='2', evidence='method=2'),
            dict(name='scale.factor', kind='number', literal='5000', evidence='scale.factor=5000')])]))
    decision = parse_decision(json.dumps(wire), utterance=text)
    assert _bind_arguments(build_default_tool_registry(), text, {}, decision.arguments) == {'kwargs': {'method': 2, 'scale.factor': 5000}}


@pytest.mark.parametrize('same_package', [False, True])
@pytest.mark.parametrize('select_scalar', [False, True])
def test_native_matrix_cross_package_reuse_without_original_source(source, simulated, tmp_path, same_package, select_scalar):
    from agent.application import ResearchAgentApplication, OutputSelection
    from agent.orchestration import AgentPlan, AgentRequest, PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.orchestration.prior_outputs import bind_output
    from agent.schemas.prior_output import PriorOutputRef
    root = tmp_path / 'workspace'
    source['output_dir'] = str(root / 'runs' / ManagedWorkspace.run_digest('first:run') / 'scientific')
    class Planner:
        selected = AgentPlan('first', 'first', 'Simulated native binary matrix', (PlanStep('native', owner.EPISCANPY.TOOL, source),))
        def plan(self, request, registry):
            return self.selected
    planner = Planner(); app = ResearchAgentApplication(root, planner=planner); app.sessions.create('reuse')
    state = app.sessions.run('reuse', 'first', AgentRequest('first', 'Binarize', {}),
        (OutputSelection('dataset', 'native', 'matrix_path'), OutputSelection('prior_cells', 'native', 'n_cells')), expected_generation=0)
    assert state.turn('first').status == 'activated'
    binding = bind_output(app.run_store, app.registry, 'first:run', 'native', 'dataset')
    assert binding.semantic_type == 'native_matrix_h5ad.v1'
    Path(source['input_path']).unlink()
    consumer = owner.EPISCANPY if same_package else owner.SIGNAC
    follow = dict(input_path=PriorOutputRef(binding, 'matrix_path'), function_name='pp.binarize' if same_package else 'BinarizeCounts',
        output_dir=str(root / 'runs' / ManagedWorkspace.run_digest('second:run') / 'scientific'))
    planner.selected = AgentPlan('second', 'second', 'Simulated compatible Signac call', (PlanStep('native', consumer.TOOL, follow),))
    state = app.sessions.run('reuse', 'second', AgentRequest('second', 'Use accepted binary matrix', {}),
        (OutputSelection('dataset', 'native', 'matrix_path'),) + ((OutputSelection('cells', 'native', 'n_cells'),) if select_scalar else ()), expected_generation=1, retain=('prior_cells',))
    assert state.turn('second').status == 'activated', app.run_store.load('second:run').errors
    assert len(simulated) == 2
    assert simulated[1]['input_path'] == app.run_store.load('first:run').steps[0].result['matrix_path']
    assert len(ResearchAgentApplication(root, planner=planner).sessions.load('reuse').revisions) == 2


def test_new_package_needs_no_fabricated_historical_profile():
    package = owner.MatrixPackage('new_package', '1.0', 'isolated', {'transform': ()}, (), 'Native transform')
    assert package.publication.historical_profiles == frozenset()
    grown = owner.MatrixPackage('new_package', '1.0', 'isolated', {'transform': (), 'another': ()}, (), 'More functions')
    assert grown.PROFILE == package.PROFILE
    for existing in (owner.EPISCANPY, owner.SIGNAC, owner.SNAPATAC2):
        assert owner.HISTORICAL_PROFILES[existing.package] in existing.publication.historical_profiles
        assert len(existing.publication.historical_profiles) == (4 if existing.package in ("Signac", "snapatac2") else 3)


def test_lsi_preserves_native_layer_values_without_numerical_prevalidation(source, tmp_path):
    a = ad.read_h5ad(source['input_path'])
    a.layers['tf-idf'] = a.X.astype(float)
    a.X = -a.X.astype(float)
    a.uns['matrix_semantics'] = 'signed_transformed'
    a.write_h5ad(source['input_path'])
    stage = tmp_path / 'stage'; stage.mkdir()
    args = dict(source, function_name='pp.lsi')
    prepared, _ = owner.prepare_input(args, stage)
    assert np.any(prepared.X.data < 0)
    np.testing.assert_array_equal(prepared.layers['tf-idf'].data, a.layers['tf-idf'].data)
    a.layers['tf-idf'].data[0] = np.nan; a.write_h5ad(source['input_path'])
    prepared, _ = owner.prepare_input(args, stage)
    assert np.isnan(prepared.layers['tf-idf'].data[0])


def test_expanded_catalog_does_not_expand_per_request_authority():
    from agent.orchestration import build_default_tool_registry
    from agent.application.turn_decisions import ArgumentMember, StructuredArgument
    registry = build_default_tool_registry()
    members = registry.get('run_snapatac2').optional_arguments['kwargs'].planning.conversational_members
    assert 32 < len(members) <= 128
    assert {name for name, _ in members} == set().union(*(set(args) for args in owner.SNAPATAC2.FUNCTIONS.values()))
    assignments = tuple(ArgumentMember(str(i), 'number', '1', 0, 1) for i in range(33))
    with pytest.raises(ValueError, match='structured argument'):
        StructuredArgument('run_snapatac2','kwargs','1',0,1,assignments)


@pytest.mark.parametrize('function,key', [('tl.umap','provided'),('pp.knn','provided'),('pp.lsi','tf-idf')])
def test_slot_consumers_do_not_require_count_eligibility(source,tmp_path,function,key):
    a = ad.read_h5ad(source['input_path']); a.X = -a.X.astype(float); a.uns.clear()
    if function == 'pp.lsi': a.layers[key] = -a.X
    else: a.obsm[key] = np.array([[1.,-2.],[3.,4.]])
    a.write_h5ad(source['input_path']); stage=tmp_path/'stage';stage.mkdir()
    args=dict(source,function_name=function,kwargs={} if function=='pp.lsi' else {'use_rep':key})
    prepared,effective=owner.prepare_input(args,stage)
    assert np.any(prepared.X.data < 0)
    assert effective['matrix_semantics']=='unspecified_retained_measurement'

@pytest.mark.parametrize('tamper', [False, True])
def test_lsi_retained_unconsumed_nan_is_not_a_computation_input(source, monkeypatch, tamper):
    a = ad.read_h5ad(source['input_path'])
    a.layers['tf-idf'] = a.X.astype(float)
    a.X = a.X.astype(float); a.X.data[0] = np.nan
    a.write_h5ad(source['input_path'])
    def native(self, args, stage, matrix):
        output = ad.read_h5ad(stage / 'input.h5ad')
        if tamper: output.X.data[-1] += 1
        output.obsm['X_lsi'] = np.ones((output.n_obs, 1))
        output.write_h5ad(stage / 'result.h5ad')
        return dict(completion='succeeded', versions={'epiScanpy':'SIMULATED'}, implementation=[],
            native_output_type='NoneType', result_slots=['obsm.X_lsi'], measurement='retained_input')
    monkeypatch.setattr(owner.MatrixPackage, 'native', native)
    args = dict(source, function_name='pp.lsi')
    if tamper:
        with pytest.raises(ValueError, match='changed source X'):
            owner.EPISCANPY.execute(args)
    else:
        result = owner.EPISCANPY.execute(args)
        assert np.isnan(ad.read_h5ad(result['matrix_path']).X.data[0])

@pytest.mark.parametrize('function', ['tl.umap', 'pp.knn'])
@pytest.mark.parametrize('use_dims', [1, [0], -1])
def test_embedding_consumers_validate_only_selected_dimensions(source, tmp_path, function, use_dims):
    a = ad.read_h5ad(source['input_path'])
    a.obsm['provided'] = np.array([[1., np.nan], [3., np.nan]])
    a.write_h5ad(source['input_path'])
    stage = tmp_path / 'stage'; stage.mkdir()
    args = dict(source, function_name=function, kwargs={'use_rep':'provided', 'use_dims':use_dims})
    prepared, _ = owner.prepare_input(args, stage)
    assert np.isnan(prepared.obsm['provided'][:, 1]).all()
    prepared, _ = owner.prepare_input(dict(args, kwargs={'use_rep':'provided', 'use_dims':[1]}), stage)
    assert np.isnan(prepared.obsm['provided'][:, 1]).all()


def test_r_double_conversion_checks_actual_fidelity_not_integer_range():
    from agent.tools.analysis.peak_h5ad import native_double_values
    native_double_values(np.array([-(2**60), 2**60], dtype=np.int64))
    native_double_values(np.array([2**63], dtype=np.uint64))
    with pytest.raises(ValueError, match='exactly'):
        native_double_values(np.array([2**60 + 1], dtype=np.int64))
    with pytest.raises(ValueError, match='exactly'):
        native_double_values(np.array([2**64 - 1], dtype=np.uint64))


def test_native_matrix_no_obsolete_scientific_prerequisite_callback():
    from agent.orchestration import build_default_tool_registry
    registry = build_default_tool_registry()
    for name in ('run_signac','run_episcanpy','run_snapatac2','run_scanpy','run_chromvar'):
        assert registry.get(name).parameter_prerequisites is None


def test_two_matrix_calls_reject_equal_size_different_feature_identity(source,tmp_path):
    from agent.tools.analysis.native_matrix import SIGNAC
    a=ad.read_h5ad(source['input_path']);a.var_names=list(reversed(a.var_names));second=tmp_path/'second.h5ad';a.write_h5ad(second)
    for fn in ('Jaccard','corSparse'):
        with pytest.raises(ValueError,match='identical ordered feature'):
            SIGNAC.execute(dict(source,function_name=fn,kwargs={},second_path=str(second),output_dir=str(tmp_path/fn)))
    with pytest.raises(ValueError,match='two actual matrix'):
        SIGNAC.native_arguments(dict(source,function_name='Jaccard',kwargs={}))
    with pytest.raises(ValueError,match='second matrix'):
        SIGNAC.native_arguments(dict(source,function_name='RunTFIDF',kwargs={},second_path=str(second)))


def test_pairwise_source_binding_and_rectangular_download(source,tmp_path,monkeypatch):
    from agent.application.local_resources import LocalResourceAdmission
    from agent.application import ResearchAgentApplication,OutputSelection
    from agent.orchestration import AgentPlan,AgentRequest,PlanStep
    from agent.application.workspace import ManagedWorkspace
    from agent.application.native_delivery import accepted_artifacts
    from agent.application.matrix_delivery import MatrixDownloadPreparer
    other=tmp_path/'other.h5ad';a=ad.read_h5ad(source['input_path']);a[:1].copy().write_h5ad(other)
    local=LocalResourceAdmission(tmp_path/'local',approved_source_roots=(tmp_path,))
    entries=[local.register('data'+str(i),p,input_type='h5ad',label='matrix'+str(i),attribution='simulated') for i,p in enumerate((source['input_path'],other))]
    group=local.compose_group([e.resource_id for e in entries],attachment_candidates=True)
    bound=local.bind_group(group,[dict(tool='run_signac',port=p,source='source.'+str(i)) for i,p in enumerate(('dataset','second'))],verify_source=True)
    assert bound.execution_inputs['input_path']==source['input_path'] and bound.execution_inputs['second_path']==str(other)
    def native(self,args,stage,matrix):
        matrix.uns['Signac_pairwise']=dict(values=np.array([[.2],[.7]]),row_cells=matrix.obs_names.to_numpy(dtype=str),column_cells=np.array(['002']),function_name='Jaccard')
        matrix.write_h5ad(stage/'result.h5ad')
        return dict(completion='succeeded',versions={'Signac':'SIMULATED'},implementation=[],native_output_type='matrix',result_slots=['uns.Signac_pairwise'])
    monkeypatch.setattr(owner.MatrixPackage,'native',native)
    root=tmp_path/'workspace';args=dict(bound.execution_inputs,function_name='Jaccard',output_dir=str(root/'runs'/ManagedWorkspace.run_digest('pair:run')/'scientific'))
    class Planner:
        def plan(self,request,registry):return AgentPlan('pair','pair','Simulated pairwise publication',(PlanStep('native','run_signac',args),))
    app=ResearchAgentApplication(root,planner=Planner());app.sessions.create('pair')
    state=app.sessions.run('pair','pair',AgentRequest('pair','Compare matrices',{}),(OutputSelection('matrix','native','matrix_path'),),expected_generation=0)
    assert state.turn('pair').status=='activated',app.run_store.load('pair:run').errors
    artifacts=accepted_artifacts(app.sessions,'pair',state.revisions[-1],state.turn('pair'),None,step_id='native')
    artifact=next(a for a in artifacts if a.role=='result.h5ad');prepared=MatrixDownloadPreparer().prepare(artifact)
    try:
        exported=tmp_path/'download.h5ad';exported.write_bytes(b''.join(prepared.iter_chunks()))
        pair=ad.read_h5ad(exported).uns['Signac_pairwise'];np.testing.assert_array_equal(pair['values'],[[.2],[.7]])
        assert pair['row_cells'].tolist()==['002','001'] and pair['column_cells'].tolist()==['002']
    finally:prepared.close()


def test_pre_raw_snapatac_profile_retains_exact_historical_result_shape(source, simulated):
    from types import SimpleNamespace
    from agent.orchestration.native_delivery import delivery_spec
    # Keep the simulated select_features fixture's X unchanged.
    data = ad.read_h5ad(source['input_path']); data.X.data[:] = 1; data.write_h5ad(source['input_path'])
    package = owner.SNAPATAC2
    args = dict(source, function_name='pp.select_features')
    current = package.execute(args, 'exact')
    path = Path(current['manifest_path'])
    value = owner.read(path)
    value['profile'] = owner.digest(dict(contract=package.CONTRACT, version=package.version,
        adaptation='ordinary AnnData; native returns and exact subset positions.v4'))
    path.unlink()  # This simulated fixture has never been accepted.
    sha = owner.write(path, value)
    historical = package.result(value, path, sha)
    assert 'fragments_path' not in historical and historical['matrix_path'] == current['matrix_path']
    authority = dict(artifact_contract=package.CONTRACT, science_profile=value['profile'], verifier=package.VERIFIER)
    before = path.read_bytes()
    assert delivery_spec(package, 'test').publication(args, historical, authority) == (value, sha)
    context = SimpleNamespace(files=owner.file_closure, execution_for=lambda *a: 'exact')
    with pytest.raises(ValueError, match='producing execution'):
        package.publication.publication_record(context, package.TOOL, args, historical, 'exact', historical)
    assert path.read_bytes() == before
