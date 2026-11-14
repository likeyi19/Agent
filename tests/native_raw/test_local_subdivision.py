"""Small genuine registered SnapATAC2 route; no biological accuracy claim."""
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import ResearchAgentApplication, OutputSelection
from agent.orchestration import AgentPlan, AgentRequest, PlanStep, StepOutputRef
from agent.orchestration.prior_outputs import bind_output
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.orchestration import _serialize
from agent.schemas.verification_authority import AuthorityError
from agent.tools.analysis.local_subdivision import assess_subdivision, differential_kwargs
from agent.tools._file_identity import file_sha256


def execute(app, name, specs, *, session=False, tool='run_snapatac2'):
    class Planner:
        def plan(self, request, registry):
            out = Path(request.inputs['output_dir'])
            return AgentPlan(name, request.request_id, 'Explicit small native qualification', tuple(
                PlanStep(step, tool, dict(output_dir=str(out/step), **args), deps)
                for step, args, deps in specs))
    app = ResearchAgentApplication(app._workspace.root, planner=Planner())
    if session:
        app.sessions.create(name)
        state = app.sessions.run(name, name, AgentRequest(name, 'Explicit native qualification', {}),
            (OutputSelection('dataset', specs[-1][0], 'matrix_path'),), expected_generation=0)
        run = app.run_store.load(name + ':run')
        assert state.turn(name).status == 'activated', run.errors
    else:
        outcome = app.run(AgentRequest(name, 'Explicit native qualification', {}))
        run = app.run_store.load(name + ':run')
    return app, run


@pytest.fixture(scope='module')
def accepted(tmp_path_factory):
    if os.environ.get('RUN_RAW_NATIVE_QUALIFICATION') != '1':
        pytest.skip('Explicit installed native runtime qualification')
    root = tmp_path_factory.mktemp('local-subdivision')
    rng = np.random.default_rng(2026)
    probabilities = np.full((140, 120), .2)
    probabilities[:60, :20] = .8; probabilities[:60, 20:40] = .04
    probabilities[60:120, :20] = .04; probabilities[60:120, 20:40] = .8
    x = rng.binomial(1, probabilities).astype('uint32')
    a = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame(index=[f'cell-{i}' for i in range(140)]),
        var=pd.DataFrame(index=[f'chr1:{1000+i*200}-{1100+i*200}' for i in range(120)]))
    a.uns['matrix_semantics'] = 'peak_counts'
    # Supplied graph has two disconnected parent populations. Native Leiden owns labels.
    a.obsp['distances'] = sparse.block_diag([np.ones((120,120))-np.eye(120), np.ones((20,20))-np.eye(20)], format='csr')
    source = root/'original-peaks.h5ad'; a.write_h5ad(source)
    original_sha = file_sha256(source)
    app = ResearchAgentApplication(root/'workspace')
    app, run = execute(app, 'parent', [('parent', dict(input_path=str(source), function_name='tl.leiden', kwargs=dict(resolution=.05)), ())], session=True)
    assert run.lifecycle_status.value == 'SUCCEEDED', run.errors
    parent = bind_output(app.run_store, app.registry, run.run_id, 'parent', 'dataset')
    parent_result = run.steps[0].result
    parent_data = ad.read_h5ad(parent_result['matrix_path'])
    group = parent_data.obs.leiden.iloc[0]
    assert (parent_data.obs.leiden == group).sum() == 120
    names = ['features','spectral','knn','leiden']
    steps = [
        ('features',dict(input_path=PriorOutputRef(parent,'matrix_path'), cell_subset=dict(groupby='leiden',groups=[group]),
            function_name='pp.select_features',kwargs=dict(n_features=100)),()),
        ('spectral',dict(input_path=StepOutputRef('features','matrix_path'),function_name='tl.spectral',kwargs=dict(n_comps=8,random_state=0)),('features',)),
        ('knn',dict(input_path=StepOutputRef('spectral','matrix_path'),function_name='pp.knn',kwargs=dict(n_neighbors=10)),('spectral',)),
        ('leiden',dict(input_path=StepOutputRef('knn','matrix_path'),function_name='tl.leiden',parameter_selection='asw'),('knn',))]
    app, run = execute(app, 'local', steps, session=True)
    assert run.lifecycle_status.value == 'SUCCEEDED', run.errors
    local = [bind_output(app.run_store,app.registry,run.run_id,step,'dataset') for step in names]
    data = ad.read_h5ad(run.steps[-1].result['matrix_path'])
    assert data.obs_names.tolist() == a.obs_names[:120].tolist()
    np.testing.assert_array_equal(data.X.toarray(), x[:120])
    selection = _serialize(run.steps[-1].result['parameters']['parameter_selection'])
    assert selection['origin'] == 'automatic' and selection['candidates']
    groups = list(dict.fromkeys(data.obs.leiden.tolist()))
    assert len(groups) > 1  # Genuine fixture check, never alter native outcomes to force a decision.
    differentials = {}
    for i, child in enumerate(groups):
        kwargs = differential_kwargs(local[-1], child, store=app.run_store, registry=app.registry)
        assert sorted(kwargs['cell_group1'] + kwargs['cell_group2']) == list(range(120))
        app, run = execute(app, 'diff' + str(i), [('diff', dict(input_path=PriorOutputRef(local[-1], 'matrix_path'),
                            function_name='tl.diff_test', kwargs=kwargs), ())])
        assert run.lifecycle_status.value == 'SUCCEEDED', run.errors
        differentials[child] = bind_output(app.run_store, app.registry, run.run_id, 'diff', 'dataset')
    return dict(root=root, app=app, parent=parent, local=local, differentials=differentials,
                groups=groups, source=source, original_sha=original_sha)


def test_genuine_local_decision_and_reinterpretation(accepted, monkeypatch):
    f = accepted; app = ResearchAgentApplication(f['root']/'workspace')
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda *a, **k: pytest.fail('Accepted evidence interpretation repeated native science'))
    kwargs = dict(store=app.run_store, registry=app.registry)
    result = assess_subdivision(f['parent'], f['local'], f['differentials'], **kwargs)
    assert result['outcome'] in ('supported_subdivision','partial_supported_subdivision','no_supported_subdivision')
    observed = []
    for child in result['children']:
        native = child['differential']['native_result']
        p = native['parameters']['effective_native_parameters']
        assert (p['direction'],p['min_log_fc'],p['min_pct'],p['covariates'],p['features']) == ('positive',.25,.05,None,None)
        table = ad.read_h5ad(native['matrix_path']).uns['agent_native_return']['value']
        # Independent Pandas expectation directly from the accepted native table.
        count = table.loc[(table['adjusted p-value'] < .01) & (table['log2(fold_change)'] > 0), 'feature name'].nunique() if len(table) else 0
        assert child['qualifying_peak_count'] == count and child['retained'] == (count >= 5)
        assert child['exploratory_unnamed']
        observed.extend(child['source_positions'])
    assert sorted(observed) == list(range(120))
    assert result['parent_membership']['source_sha256'] == result['parent']['native_result']['matrix_sha256']
    kept = [p for c in result['children'] if c['retained'] for p in c['source_positions']]
    assert sorted(kept + result['residual_parent']['positions']) == list(range(120))
    strict = assess_subdivision(f['parent'],f['local'],f['differentials'],adjusted_p_threshold=0, **kwargs)
    assert strict['outcome'] == 'no_supported_subdivision' and strict['residual_parent']['positions'] == list(range(120))
    threshold = max(c['qualifying_peak_count'] for c in result['children']) + 1
    high = assess_subdivision(f['parent'],f['local'],f['differentials'],minimum_peak_features=threshold, **kwargs)
    assert high['outcome'] == 'no_supported_subdivision'
    assert file_sha256(f['source']) == f['original_sha']
    (f['root']/'decision.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (f['root']/'bindings.json').write_text(json.dumps(dict(parent=f['parent'].to_dict(), local=[b.to_dict() for b in f['local']],
        differentials={k:v.to_dict() for k,v in f['differentials'].items()}),indent=2))


def test_genuine_native_failure_is_not_biological_stop(accepted):
    f=accepted; app=f['app']; group=f['groups'][0]
    kwargs=differential_kwargs(f['local'][-1], group, store=app.run_store, registry=app.registry)
    kwargs['covariates']=['depth']
    app, run=execute(app,'native-error',[('diff',dict(input_path=PriorOutputRef(f['local'][-1],'matrix_path'),
        function_name='tl.diff_test',kwargs=kwargs),())])
    assert run.lifecycle_status.value == 'FAILED'
    comparisons=dict(f['differentials']); comparisons[group]=dict(run_id=run.run_id,step_id='diff')
    decision=assess_subdivision(f['parent'],f['local'],comparisons,store=app.run_store,registry=app.registry)
    assert decision['outcome']=='native_execution_failure'
    assert 'covariates is not implemented' in json.dumps(run.to_dict())
    failed=next(c for c in decision['children'] if c['cluster']==group)
    assert failed['retained'] is None and failed['qualifying_peak_count'] is None
    (f['root']/'native-error-decision.json').write_text(json.dumps(decision,indent=2))


def test_accepted_publication_tamper_and_source_substitution(accepted):
    from dataclasses import replace
    f=accepted; app=f['app']; kwargs=dict(store=app.run_store,registry=app.registry)
    with pytest.raises(AuthorityError):
        assess_subdivision(replace(f['parent'],accepted_step_sha256='0'*64),f['local'],f['differentials'],**kwargs)
    wrong=dict(f['differentials']); wrong[f['groups'][0]]=f['differentials'][f['groups'][1]]
    with pytest.raises(AuthorityError,match='actual child'):
        assess_subdivision(f['parent'],f['local'],wrong,**kwargs)
    # Tamper only this test's disposable publication, restore exact bytes afterward.
    binding=f['differentials'][f['groups'][0]]
    path=Path(app.run_store.load(binding.run_id).steps[0].result['matrix_path']); original=path.read_bytes()
    try:
        path.write_bytes(original+b'tamper')
        with pytest.raises(AuthorityError): assess_subdivision(f['parent'],f['local'],f['differentials'],**kwargs)
    finally: path.write_bytes(original)


def test_genuine_annotation_handoff_is_independent(accepted):
    from agent.application.shared_resources import ScientificResourceLibrary
    from agent.tools.data.cellmarker import CellMarkerEvidence
    from agent.tools.analysis.marker_candidates import annotate_markers, marker_kwargs
    f=accepted; app=f['app']
    gtf=f['root']/'genes.gtf'
    genes=['Cd3d','Cd3e','Trac','Cd247']+[f'fixture_gene_{i}' for i in range(116)]
    gtf.write_text(''.join(f'chr1\tHAVANA\tgene\t{1001+i*200}\t{1100+i*200}\t.\t+\t.\tgene_id "g{i}"; gene_name "{gene}";\n' for i,gene in enumerate(genes)))
    app,run=execute(app,'activity',[('activity',dict(input_path=PriorOutputRef(f['local'][-1],'matrix_path'),
        function_name='tl.geneactivity',gene_anno_path=str(gtf),kwargs=dict(upstream=0)),())],tool='run_episcanpy')
    assert run.lifecycle_status.value=='SUCCEEDED', run.errors
    activity=bind_output(app.run_store,app.registry,run.run_id,'activity','dataset')
    app,run=execute(app,'markers',[('markers',dict(input_path=PriorOutputRef(activity,'matrix_path'),
        function_name='tl.rank_genes_groups',kwargs=marker_kwargs('leiden')),())],tool='run_scanpy')
    assert run.lifecycle_status.value=='SUCCEEDED', run.errors
    markers=bind_output(app.run_store,app.registry,run.run_id,'markers','dataset')
    lib=ScientificResourceLibrary(Path(os.environ['AGENT_SCIENTIFIC_RESOURCE_ROOT']))
    entry=next(r for r in lib.entries() if r['kind']=='cellmarker')
    evidence=annotate_markers(markers,store=app.run_store,registry=app.registry,
        evidence=CellMarkerEvidence(lib,entry['id']),species='Mouse')
    kwargs=dict(store=app.run_store,registry=app.registry)
    plain=assess_subdivision(f['parent'],f['local'],f['differentials'],**kwargs)
    annotated=assess_subdivision(f['parent'],f['local'],f['differentials'],annotation=evidence,
        annotation_bindings=[activity,markers],**kwargs)
    assert annotated['annotation']==evidence
    assert annotated['outcome']==plain['outcome']
    assert [(c['retained'],c['qualifying_peak_count']) for c in annotated['children']]==[(c['retained'],c['qualifying_peak_count']) for c in plain['children']]
    assert all(c['annotation_cluster'] is not None for c in annotated['children'])
    (f['root']/'annotated-decision.json').write_text(json.dumps(annotated,indent=2,allow_nan=False))


def test_native_small_feature_returns_and_failed_local_attempt(accepted):
    f=accepted; app=f['app']; group=f['groups'][0]
    comparison=differential_kwargs(f['local'][-1],group,store=app.run_store,registry=app.registry)
    for name,features in [('one-feature',[0]),('zero-features',[])]:
        app,run=execute(app,name,[('diff',dict(input_path=PriorOutputRef(f['local'][-1],'matrix_path'),
            function_name='tl.diff_test',kwargs=dict(comparison,features=features)),())])
        assert run.lifecycle_status.value=='SUCCEEDED', run.errors
        binding=bind_output(app.run_store,app.registry,run.run_id,'diff','dataset')
        comparisons=dict(f['differentials']); comparisons[group]=binding
        decision=assess_subdivision(f['parent'],f['local'],comparisons,store=app.run_store,registry=app.registry)
        child=next(c for c in decision['children'] if c['cluster']==group)
        assert child['evidence_status']=='accepted' and not child['retained']
        assert child['qualifying_peak_count'] <= len(features)
        (f['root']/(name+'-decision.json')).write_text(json.dumps(decision,indent=2))
    selection=app.run_store.load(f['local'][0].run_id).steps[0].result['parameters']['cell_subset']
    app,run=execute(app,'failed-local',[('features',dict(input_path=PriorOutputRef(f['parent'],'matrix_path'),
        cell_subset=dict(groupby=selection['groupby'],groups=list(selection['groups'])),
        function_name='pp.select_features',kwargs=dict(n_features=1.5)),())])
    assert run.lifecycle_status.value=='FAILED', run.errors
    decision=assess_subdivision(f['parent'],store=app.run_store,registry=app.registry,
        failed_local=dict(run_id=run.run_id,step_id='features'))
    assert decision['outcome']=='native_execution_failure' and decision['parent_membership'] is None
    (f['root']/'failed-local-decision.json').write_text(json.dumps(decision,indent=2))
