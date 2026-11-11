"""Bounded ASW policy plus genuine installed clustering through shared owners."""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.tools.analysis import native_clustering as asw
from agent.tools.analysis.cell_result import describe


def fixture_data(n=90):
    rng = np.random.default_rng(17)
    centers = np.repeat(np.eye(3), 20, axis=1) * 7
    x = rng.poisson(1 + centers[np.arange(n) % 3]).astype('float32')
    a = ad.AnnData(sparse.csr_matrix(x), obs=pd.DataFrame(index=[f'cell-{i:03}' for i in reversed(range(n))]))
    return a


def test_single_cluster_fallback_ties_fixed_sample_and_winner_not_replayed():
    a = fixture_data(12)
    a.obsm['X_pca'] = np.arange(24).reshape(12, 2).astype(float)
    calls = []
    def clustering(a, resolution=1):
        calls.append(resolution)
        a.obs['leiden'] = pd.Categorical(['0'] * 6 + ['1'] * 6)
    graph = dict(graph='connectivities', function='scanpy.pp.neighbors',
                 representation='X_pca', dimensions=None, metric='euclidean', metric_kwds={})
    result, evidence = asw.invoke(clustering, a, 'scanpy', 'tl.leiden', {}, 'asw', [graph])
    assert calls == asw.candidates('resolution', 12) and result is None
    assert evidence['selected_value'] == .1
    assert len({r['asw'] for r in evidence['candidates']}) == 1
    assert evidence['sample_count'] == 12
    calls.clear()
    def constant(a, resolution=1):
        calls.append(resolution)
        a.obs['leiden'] = pd.Categorical(['0'] * 12)
    _, unavailable = asw.invoke(constant, a, 'scanpy', 'tl.leiden', {}, 'asw', [graph])
    assert unavailable['status'] == 'unavailable' and unavailable['selected_value'] == 1
    assert all(r['asw'] is None for r in unavailable['candidates'])
    assert len(calls) == 6  # Retains the already executed native default.


def test_native_errors_are_not_scored_or_replaced():
    a = fixture_data(12)
    a.obsm['X_pca'] = np.ones((12, 3))
    def fail(a, num_clusters):
        raise RuntimeError('native failure')
    with pytest.raises(RuntimeError, match='native failure'):
        asw.invoke(fail, a, 'episcanpy', 'tl.kmeans', {}, 'asw', [])


@pytest.mark.parametrize('package,function,default', [('snapatac2', 'tl.leiden', 1),
    ('scanpy', 'tl.leiden', 1), ('episcanpy', 'tl.leiden', 1), ('episcanpy', 'tl.louvain', None)])
def test_undefined_asw_keeps_each_applicable_native_default(package, function, default):
    a = fixture_data(12); a.obsm['X_pca'] = np.ones((12, 2))
    calls = []
    key = function.split('.')[1]
    def native(a, resolution=default):
        calls.append(resolution)
        a.obs[key] = pd.Categorical(['0'] * a.n_obs)
    graph = dict(graph='distances' if package == 'snapatac2' else 'connectivities',
        function='snapatac2.pp.knn' if package == 'snapatac2' else package + '.pp.neighbors',
        representation='X_pca', dimensions=None, metric='euclidean', metric_kwds={})
    _, selection = asw.invoke(native, a, package, function, {}, 'asw', [graph])
    assert selection['status'] == 'unavailable' and selection['selected_value'] == default
    assert selection['origin'] == 'native_default'
    assert calls == asw.candidates('resolution', a.n_obs) + ([None] if default is None else [])


@pytest.mark.parametrize('function', ['tl.kmeans', 'tl.hc'])
def test_explicit_native_count_never_scans(function):
    a = fixture_data(12); a.obsm['X_pca'] = np.ones((12, 2))
    calls = []
    def native(a, num_clusters):
        calls.append(num_clusters)
    _, selection = asw.invoke(native, a, 'episcanpy', function, {'num_clusters': 7}, 'asw', [])
    assert calls == [7] and selection['origin'] == 'explicit' and not selection['candidates']


@pytest.mark.parametrize('kwargs,policy,origin', [({'resolution': 3.7}, 'asw', 'explicit'),
                                               ({}, 'native_defaults', 'native_default'), ({}, None, 'native_default')])
def test_explicit_or_native_default_never_scans(kwargs, policy, origin):
    calls = []
    def native(a, resolution=1):
        calls.append(resolution)
    _, evidence = asw.invoke(native, fixture_data(), 'scanpy', 'tl.leiden', kwargs, policy, [])
    assert calls == [kwargs.get('resolution', 1)]
    assert evidence['origin'] == origin and evidence['candidates'] == []


def test_no_default_does_not_invent_value_when_asw_is_undefined():
    a = fixture_data(2); a.obsm['X_pca'] = np.eye(2)
    def clustering(a, num_clusters):
        pytest.fail('No valid bounded candidates exist')
    with pytest.raises(ValueError, match='requires explicit num_clusters'):
        asw.invoke(clustering, a, 'episcanpy', 'tl.hc', {}, 'asw', [])


def run_count_fallback(root, function, *, check_edit=False):
    """Real native failure/question/answer; also usable with OS isolation intact."""
    from agent.application import InteractiveAgentApplication, OutputSelection
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from agent.schemas import AgentPlan, AgentRequest, PlanStep, PriorOutputRef
    from agent.schemas.orchestration import _serialize
    n = 12 if function == 'tl.kmeans' else 2
    source = root / 'source.h5ad'
    a = fixture_data(n)
    a.obsm['X_pca'] = np.ones((n, 2)) if n == 12 else np.eye(2)
    a.write_h5ad(source)
    original = source.read_bytes()
    model = Model('episcanpy', function, 'X_pca')
    complete = model.complete
    def select(**kwargs):
        value = json.loads(kwargs['prompt'])
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1,
                decision=dict(kind='execute_plan', target='run_episcanpy', context='current')))
        result = json.loads(complete(**kwargs))
        if 'output_selection_schema_version' in value:
            if 'retain' in kwargs['response_schema']['properties']:
                result['retain'] = []
        else:
            offered = [r for r in value['active_outputs'] if r['source_port'] == 'dataset']
            assert len(offered) == 1
            result['decision']['steps'][0]['sources'][0]['source'] = dict(kind='context', handle=offered[0]['handle'])
        return json.dumps(result)
    model.complete = select
    profile = PlanningModelProfile('asw', 'scripted', model.model_id)
    def open_app():
        return InteractiveAgentApplication(root/'app', model_profiles=(profile,), default_profile_id='asw',
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: model}))
    app = open_app()
    app.create_session('asw')
    # A previously accepted input, not a publication from the failed ASW run.
    owner = app._application
    request = AgentRequest('upstream', 'Native log1p on the supplied fixture.', {})
    _, paths = owner._prepare_request(request)
    plan = AgentPlan('upstream', request.request_id, request.prompt, (PlanStep('upstream', 'run_episcanpy',
        dict(input_path=str(source), output_dir=str(paths.scientific/'upstream'), function_name='pp.log1p')),))
    before = owner.sessions.run('asw', 'upstream', request, (OutputSelection('input', 'upstream', 'matrix_path'),),
        expected_generation=0, prepared_plan=plan)
    assert before.turn('upstream').status == 'activated'
    upstream_run = owner.run_store.load(before.revisions[0].run_id).to_dict()
    upstream_files = {p: p.read_bytes() for p in paths.scientific.rglob('*') if p.is_file()}
    first = app.submit_turn('asw', 'automatic', 'Cluster the current result.', expected_generation=1)
    assert first.status == 'clarification', first.to_dict()
    assert len(first.clarification_choices) == 1
    choice = first.clarification_choices[0]
    assert (choice['tool'], choice['native_function'], choice['argument'], choice['member']) == (
        'run_episcanpy', function, 'kwargs', 'num_clusters')
    state = owner.sessions.load('asw')
    assert state.revisions == before.revisions and state.generation == 1
    turn = state.turn('automatic')
    assert turn.status == 'failed' and turn.revision_id is None
    failed = owner.run_store.load(turn.run_id)
    assert failed.lifecycle_status.value == 'FAILED' and len(failed.steps) == 1
    assert failed.steps[0].result is None and failed.steps[0].error.code == 'NATIVE_CLUSTER_COUNT_REQUIRED'
    failed_dict = failed.to_dict()
    diagnostic_path = Path(failed.steps[0].resolved_arguments['output_dir'])/'native-error.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    candidates = diagnostic['parameter_selection']['candidates']
    assert len(candidates) == (6 if n == 12 else 0)
    assert all(row['asw'] is None for row in candidates)
    assert not (diagnostic_path.parent/'result.h5ad').exists()
    assert not (diagnostic_path.parent/'manifest.json').exists()
    original_binding = failed.plan.steps[0].arguments['input_path']
    assert isinstance(original_binding, PriorOutputRef)
    assert original_binding.binding.run_id == before.revisions[0].run_id
    def forbidden(**kwargs):
        raise AssertionError('The exact structured count answer must not call a provider')
    model.complete = forbidden
    app = open_app()  # Pending slots and failure survive an actual reload.
    answer = app.submit_turn('asw', 'explicit', '', expected_generation=1,
        parameter_overrides=[dict(tool='run_episcanpy', native_function=function,
            argument='kwargs', value={'num_clusters': 2})])
    assert answer.status == 'succeeded', answer.to_dict()
    owner = open_app()._application
    after = owner.sessions.load('asw')
    result = owner.run_store.load(after.revisions[-1].run_id)
    assert len(result.steps) == 1 and result.steps[0].resolved_arguments['function_name'] == function
    assert result.plan.steps[0].arguments['input_path'] == original_binding
    selection = _serialize(result.steps[0].result['parameters']['parameter_selection'])
    assert selection['origin'] == 'explicit' and selection['selected_value'] == 2 and selection['candidates'] == []
    assert result.request.prompt == failed.request.prompt
    assert after.revisions[0] == before.revisions[0] and after.turn('automatic') == turn
    assert owner.run_store.load(failed.run_id).to_dict() == failed_dict
    assert owner.run_store.load(before.revisions[0].run_id).to_dict() == upstream_run
    assert all(p.read_bytes() == data for p, data in upstream_files.items())
    assert source.read_bytes() == original
    evidence = dict(function=function, failed_run_id=failed.run_id, continued_run_id=result.run_id,
        failure_diagnostic=diagnostic, explicit_selection=selection, prior_binding=original_binding.to_dict(),
        previous_revision_unchanged=True, failed_run_unchanged=True, provider_calls_on_answer=0,
        continuation_steps=[s.step_id for s in result.steps])
    if check_edit:
        edited = app.submit_turn('asw', 'optimize-again', '', expected_generation=2,
            rerun_revision_id=after.revisions[-1].revision_id, parameter_overrides=[dict(tool='run_episcanpy',
                native_function=function, step_id='clusters', argument='parameter_selection', value='asw')])
        assert edited.status == 'clarification', edited.to_dict()
        answered = app.submit_turn('asw', 'edit-count', '', expected_generation=2,
            parameter_overrides=[dict(tool='run_episcanpy', native_function=function,
                argument='kwargs', value={'num_clusters': 2})])
        assert answered.status == 'succeeded', answered.to_dict()
        final = owner.sessions.load('asw')
        edited_run = owner.run_store.load(final.revisions[-1].run_id)
        assert edited_run.request.prompt == failed.request.prompt
        assert edited_run.plan.steps[0].arguments['input_path'] == original_binding
        assert final.revisions[:2] == after.revisions
        evidence['result_edit_count_continuation'] = 'passed without provider calls'
    (root/'fallback-evidence.json').write_text(json.dumps(evidence, indent=2))
    return evidence


@pytest.mark.parametrize('function', ['tl.kmeans', 'tl.hc'])
def test_unavailable_count_uses_scoped_question_and_exact_native_continuation(tmp_path, monkeypatch, function):
    from test_execution_consolidation import _scanpy_runtime
    from agent.tools import _native_process
    _scanpy_runtime(monkeypatch)
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    run_count_fallback(tmp_path, function, check_edit=function == 'tl.hc')


@pytest.mark.parametrize('function', ['tl.kmeans', 'tl.hc'])
def test_genuine_native_count_error_remains_failed_without_question(tmp_path, monkeypatch, function):
    from test_execution_consolidation import _scanpy_runtime
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from agent.tools import _native_process
    _scanpy_runtime(monkeypatch)
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    a = fixture_data(12)
    # Native input/parameter rejection must retain its ordinary failure class.
    a.obsm['X_pca'] = np.full((12, 2), np.nan)
    source = tmp_path/'source.h5ad'; a.write_h5ad(source)
    model = Model('episcanpy', function, 'X_pca')
    profile = PlanningModelProfile('asw', 'scripted', model.model_id)
    app = InteractiveAgentApplication(tmp_path/'app', model_profiles=(profile,), default_profile_id='asw',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': lambda _: model}))
    app.create_session('asw')
    # An explicit impossible count must reach the selected native function and
    # remain its failure, even though the ASW policy is present.
    outcome = app.submit_turn('asw', 'invalid', 'Cluster this input.', expected_generation=0,
        execution_inputs={'input_path': str(source)}, parameter_overrides=[dict(tool='run_episcanpy',
            native_function=function, argument='kwargs', value={'num_clusters': -1})])
    assert outcome.status == 'failed' and not outcome.clarification_choices, outcome.to_dict()
    state = app._application.sessions.load('asw')
    assert not state.revisions and state.turn('invalid').status == 'failed'
    assert state.interactions[-1].prerequisite is None
    run = app._application.run_store.load(state.turn('invalid').run_id)
    assert run.steps[0].error.code == 'NATIVE_FUNCTION_REJECTED'


def test_representation_graph_and_dimensions_are_exact():
    a = fixture_data(12)
    a.obsm['research_space'] = np.arange(60).reshape(12, 5).astype(float)
    a.obsp['distances'] = sparse.eye(12, format='csr')
    r = asw.graph_record(a, 'snapatac2', 'pp.knn', dict(use_rep='research_space', use_dims=[3, 0]))
    assert r['dimensions'] == [3, 0] and r['metric'] == 'euclidean'
    assert asw.valid_graphs(a, [r]) == [r]
    np.testing.assert_array_equal(asw.representation(a, r), a.obsm['research_space'][:, [3, 0]])
    assert asw.evaluation_input(a, 'scanpy', 'tl.leiden', {}, [r])[2] is not None
    a.obsm['research_space'][0, 3] += 1
    assert asw.valid_graphs(a, [r]) == []


@pytest.mark.parametrize('package,function,key', [('snapatac2','tl.dbscan','dbscan'),
    ('episcanpy','tl.kmeans','kmeans'), ('episcanpy','tl.hc','hc'), ('episcanpy','tl.louvain','louvain'),
    ('scanpy','tl.leiden','leiden')])
def test_grouping_requires_actual_written_slot(package, function, key):
    result = dict(contract_version='native-' + package + '-matrix.v1', matrix_path='result.h5ad',
                  function_name=package + '.' + function, parameters={'kwargs': {}}, result_slots=[])
    assert describe('run_' + package, result)['grouping'] is None
    result['result_slots'] = ['obs.' + key]
    assert describe('run_' + package, result)['grouping'] == key
    assert describe('run_' + package, result)['preserves_counts']


class Model:
    model_id = 'scripted-asw-selection'
    def __init__(self, package, function, representation=None, umap=False):
        self.package, self.function, self.representation, self.umap = package, function, representation, umap
        self.calls = []

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt); self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1, decision=dict(kind='execute_plan', target='run_' + self.package)))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='analysis', step_id='umap' if self.umap else 'clusters', output_key='matrix_path')]))
        ops = []
        if not self.representation:
            ops += [('features','pp.select_features'), ('reduce','tl.spectral')] if self.package == 'snapatac2' else [
                ('reduce', 'pp.pca' if self.package == 'scanpy' else 'tl.pca')]
        if self.function not in ('tl.kmeans', 'tl.hc', 'tl.dbscan'):
            ops += [('neighbors', 'pp.knn' if self.package == 'snapatac2' else 'pp.neighbors')]
        ops += [('clusters', self.function)]
        if self.umap:
            ops += [('umap', 'tl.umap')]
        steps = []
        for step, function in ops:
            sources = [dict(target='dataset', source=(dict(kind='input', input='input_path') if not steps else
                dict(kind='step_port', step=steps[-1]['step_id'], source_port='dataset'))),
                dict(target='function_name', source=dict(kind='registered_choice', choice=function))]
            if function in ('pp.knn','pp.neighbors'):
                sources.append(dict(target='use_rep', source=dict(kind='registered_choice', choice=self.representation or
                    ('X_spectral' if self.package == 'snapatac2' else 'X_pca'))))
            steps.append(dict(step_id=step, tool='run_' + self.package, sources=sources, control_dependencies=[]))
        return json.dumps(dict(schema_version=4, decision=dict(kind='plan', steps=steps)))


def run_case(root, package, function, source=None, representation=None, umap=False, *, unavailable=False, resume=False):
    """Also used without any process stubs by opt-in isolated acceptance."""
    from agent.application import InteractiveAgentApplication
    from agent.application.structured_parameters import result_parameters
    from agent.application.result_reading import handoff
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from agent.schemas import PriorOutputRef
    from agent.schemas.orchestration import _serialize
    if source is None:
        source = root/'source.h5ad'; fixture_data().write_h5ad(source)
    original = source.read_bytes()
    model = Model(package, function, representation, umap)
    profile = PlanningModelProfile('asw', 'scripted', model.model_id)
    def open_app():
        return InteractiveAgentApplication(root/'app', model_profiles=(profile,), default_profile_id='asw',
            planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app = open_app()
    if not resume:
        app.create_session('asw')
    first = app.submit_turn('asw', 'initial', 'Cluster the supplied data.', expected_generation=0,
                            execution_inputs={'input_path': str(source)})
    assert first.status == 'succeeded', first.to_dict()
    state = app._application.sessions.load('asw'); revision = state.revisions[-1]
    run = app._application.run_store.load(revision.run_id)
    step = next(s for s in run.steps if s.step_id == 'clusters')
    selection = _serialize(step.result['parameters']['parameter_selection'])
    if unavailable:
        assert selection['status']=='unavailable' and selection['origin']=='native_default', selection
        assert all(r['asw'] is None for r in selection['candidates'])
    else:
        assert selection['origin'] == 'automatic', selection
        assert selection['selected_value'] == max((r for r in selection['candidates'] if r['asw'] is not None), key=lambda r:r['asw'])['value']
    descriptor = handoff(app._application.run_store, run.run_id, 'clusters')
    assert descriptor['grouping'] == function.split('.')[1]
    from agent.orchestration.verification_authority import _load_context
    from agent.tools.analysis.cell_result import native_group_rows
    from agent.tools.analysis.categorical import observations
    ids, labels = observations(step.result['matrix_path'], step.result['matrix_sha256'], descriptor['grouping'])
    context = _load_context(app._application.run_store, run.run_id, 'historical_verified_sources.v1')
    projected = native_group_rows(context, step.result['matrix_path'], step.result['matrix_sha256'],
                                  descriptor['source']['sha256'], ids)
    assert projected == list(zip(ids, map(str, labels)))
    if umap:
        from agent.application.result_reading import cell_overlay
        from agent.orchestration.prior_outputs import accepted_step_digest
        def ref(s):
            return dict(run_id=run.run_id,step_id=s.step_id,output_key='matrix_path',accepted_step_sha256=accepted_step_digest(s))
        overlay=cell_overlay(app._application.sessions,ref(run.steps[-1]),ref(step),'result:'+descriptor['grouping'])
        assert len(overlay['labels'])==step.result['n_cells']
        assert list((root/'app').rglob('*.png'))
    output = ad.read_h5ad(step.result['matrix_path']); data = ad.read_h5ad(source)
    assert output.obs_names.equals(data.obs_names)
    parameter = selection['parameter']
    fields = result_parameters(app._application.sessions, 'asw', revision.revision_id)
    field = next(p for s in fields['steps'] if s['step_id']=='clusters' for p in s['parameters'] if p['argument']==parameter)
    assert field['value'] == selection['selected_value']
    original_run = run.to_dict()
    pins = {p:p.read_bytes() for p in (root/'app').rglob('*') if p.is_file() and 'scientific' in p.parts}
    model.complete = lambda **kw: (_ for _ in ()).throw(AssertionError('Result edit called model'))
    edits = [{'key_added':'edited_groups'}, {parameter: 1.7}] if parameter == 'resolution' else [{parameter: 5}]
    edit_evidence = []
    for index, change in enumerate(edits):
        changed = app.submit_turn('asw', 'edit'+str(index), '', expected_generation=index+1,
            rerun_revision_id=revision.revision_id, parameter_overrides=[dict(tool='run_'+package,
                native_function=function, step_id='clusters', argument='kwargs', value=change)])
        assert changed.status == 'succeeded', changed.to_dict()
        state = app._application.sessions.load('asw'); revision=state.revisions[-1]
        later = app._application.run_store.load(revision.run_id)
        assert [s.step_id for s in later.steps] == (['clusters', 'umap'] if umap else ['clusters'])
        assert isinstance(later.plan.steps[0].arguments['input_path'], PriorOutputRef)
        edited = later.steps[0].result['parameters']['parameter_selection']
        assert edited['origin'] == 'explicit' and not edited['candidates']
        assert edited['selected_value'] == change.get(parameter, selection['selected_value'])
        edit_evidence.append(_serialize(edited))
    artifact = next(a for a in app.scientific_artifact_handles('asw', revision.revision_id) if a.filename.endswith('.h5ad'))
    download = app.prepare_scientific_artifact('asw', revision.revision_id, artifact.handle)
    try:
        target=root/'download.h5ad'; target.write_bytes(b''.join(download.iter_chunks()))
    finally:
        download.close()
    delivered=ad.read_h5ad(target)
    assert delivered.obs_names.equals(data.obs_names)
    assert ('edited_groups' if parameter=='resolution' else function.split('.')[1]) in delivered.obs
    reopened = open_app()
    assert reopened._application.sessions.load('asw').revisions[-1] == revision
    assert reopened._application.run_store.load(run.run_id).to_dict() == original_run
    assert all(p.read_bytes()==raw for p,raw in pins.items()) and source.read_bytes()==original
    evidence = dict(package=package, function=function, selection=selection, edits=edit_evidence,
        source=str(source), source_sha256=__import__('hashlib').sha256(original).hexdigest(), n_cells=data.n_obs,
        initial_run_id=run.run_id, final_revision=revision.revision_id, preserved_files=len(pins),
        grouping=descriptor['grouping'], download=str(target), native_execution='real installed functions',
        planning='scripted responses; no live provider')
    (root/'evidence.json').write_text(json.dumps(evidence, indent=2))
    return evidence


@pytest.mark.parametrize('package,function', list(asw.ELIGIBLE))
def test_real_selected_native_functions_and_result_edits(tmp_path, monkeypatch, package, function):
    from test_execution_consolidation import _scanpy_runtime
    _scanpy_runtime(monkeypatch)
    monkeypatch.setenv('AGENT_SNAPATAC2_PYTHON', '/home/likeyi/program/agent-data/resources/runtimes/native-matrix-1/env/bin/python')
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    run_case(tmp_path, package, function)


def test_semantic_planning_uses_one_policy_and_grounded_values_only(tmp_path):
    from agent.orchestration import AgentRequest, LLMPlanner, build_default_tool_registry
    from agent.orchestration.native_parameters import input_name
    registry = build_default_tool_registry()
    base = dict(input_path=str(tmp_path/'input.h5ad'), output_dir=str(tmp_path/'out'))
    for extra, policy, kwargs in [({}, 'asw', {}),
        ({input_name('run_scanpy','tl.leiden','kwargs'): {'resolution': 4.2}}, 'asw', {'resolution': 4.2}),
        ({input_name('run_scanpy','tl.leiden','parameter_selection'): 'native_defaults'}, 'native_defaults', {})]:
        model = Model('scanpy','tl.leiden', 'X_epizoo')
        plan = LLMPlanner(model).plan(AgentRequest('policy', 'Cluster this input.', base | extra), registry)
        assert plan.steps[-1].arguments['parameter_selection'] == policy
        assert dict(plan.steps[-1].arguments.get('kwargs', {})) == kwargs
        assert len(model.calls) == 1
    assert 'parameter_selection' not in plan.steps[0].arguments


@pytest.mark.parametrize('function', ['tl.kmeans', 'tl.hc'])
def test_required_count_uses_existing_scoped_question_when_native_defaults_requested(tmp_path, function):
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    profile = PlanningModelProfile('asw','scripted',Model.model_id)
    model = Model('episcanpy', function)
    app = InteractiveAgentApplication(tmp_path/'app', model_profiles=(profile,), default_profile_id='asw',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('asw')
    source=tmp_path/'source.h5ad';fixture_data().write_h5ad(source)
    first=app.submit_turn('asw','defaults','Cluster this input.',expected_generation=0,
        execution_inputs={'input_path':str(source)}, parameter_overrides=[dict(tool='run_episcanpy',
        native_function=function, argument='parameter_selection', value='native_defaults')])
    assert first.status=='clarification', first.to_dict()
    assert len(first.clarification_choices)==1
    choice=first.clarification_choices[0]
    assert choice['native_function']==function and choice['member']=='num_clusters'
    state=app._application.sessions.load('asw')
    assert not state.turns and not state.revisions


def test_fixed_bounded_sample_scoring_is_deterministic():
    a=ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(2100)]))
    a.obsm['X_pca']=np.random.default_rng(17).normal(size=(2100,3))
    def native(a,num_clusters):
        a.obs['kmeans']=pd.Categorical(np.arange(a.n_obs)%num_clusters)
    _, first=asw.invoke(native,a,'episcanpy','tl.kmeans',{},'asw',[])
    _, second=asw.invoke(native,a,'episcanpy','tl.kmeans',{},'asw',[])
    assert first==second and first['sample_count']==2000


def test_real_dbscan_recognition_and_unadapted_defaults(tmp_path, monkeypatch):
    from agent.tools.analysis.native_matrix import SNAPATAC2
    from agent.tools import _native_process
    from agent.tools.data.authority_context import VerificationContext, authority_operation
    from agent.tools.analysis.cell_result import native_group_rows
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    monkeypatch.setenv('AGENT_SNAPATAC2_PYTHON', '/home/likeyi/program/agent-data/resources/runtimes/native-matrix-1/env/bin/python')
    a=fixture_data(18);a.obsm['X_spectral']=np.repeat(np.eye(3),6,axis=0).astype(float)
    source=tmp_path/'source.h5ad';a.write_h5ad(source)
    context=VerificationContext()
    with authority_operation(context):
        result=SNAPATAC2.execute(dict(input_path=str(source),output_dir=str(tmp_path/'dbscan'),function_name='tl.dbscan'))
    assert describe('run_snapatac2',result)['grouping']=='dbscan'
    assert 'parameter_selection' not in result['parameters']
    assert result['parameters']['effective_native_parameters']['eps']==.5
    rows=native_group_rows(context,result['matrix_path'],result['matrix_sha256'],result['source_identity']['sha256'],a.obs_names)
    assert [r[0] for r in rows]==a.obs_names.tolist()


def test_repeated_clustering_edit_targets_exact_step_and_pins_descendant_value(tmp_path, monkeypatch):
    from test_execution_consolidation import _scanpy_runtime
    _scanpy_runtime(monkeypatch)
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    from agent.application import InteractiveAgentApplication
    from agent.orchestration import PlanningModelProfile
    from agent.providers import PlanningModelFactoryRegistry
    from agent.schemas import PriorOutputRef
    model=Model('scanpy','tl.leiden')
    complete=model.complete
    def repeated(**kwargs):
        value=json.loads(complete(**kwargs))
        if value.get('schema_version')==4:
            steps=value['decision']['steps']
            previous=json.loads(json.dumps(steps[-1]));previous['step_id']='previous'
            steps.insert(-1,previous)
            steps[-1]['sources'][0]['source']=dict(kind='step_port',step='previous',source_port='dataset')
        return json.dumps(value)
    model.complete=repeated
    profile=PlanningModelProfile('asw','scripted',model.model_id)
    app=InteractiveAgentApplication(tmp_path/'app',model_profiles=(profile,),default_profile_id='asw',
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda _:model}))
    app.create_session('asw');source=tmp_path/'source.h5ad';fixture_data().write_h5ad(source)
    first=app.submit_turn('asw','initial','Cluster twice.',expected_generation=0,execution_inputs={'input_path':str(source)})
    assert first.status=='succeeded',first.to_dict()
    revision=app._application.sessions.load('asw').revisions[-1]
    run=app._application.run_store.load(revision.run_id)
    effective={s.step_id:s.result['parameters']['effective_native_parameters']['resolution'] for s in run.steps[-2:]}
    model.complete=lambda **kw: (_ for _ in ()).throw(AssertionError('Edit called model'))
    changed=app.submit_turn('asw','edit','',expected_generation=1,rerun_revision_id=revision.revision_id,
        parameter_overrides=[dict(tool='run_scanpy',native_function='tl.leiden',step_id='previous',argument='kwargs',value={'resolution':1.7})])
    assert changed.status=='succeeded',changed.to_dict()
    revision=app._application.sessions.load('asw').revisions[-1]
    later=app._application.run_store.load(revision.run_id)
    assert [s.step_id for s in later.steps]==['previous','clusters']
    assert isinstance(later.plan.steps[0].arguments['input_path'],PriorOutputRef)
    assert later.plan.steps[0].arguments['input_path'].binding.step_id=='neighbors'
    assert later.steps[0].result['parameters']['effective_native_parameters']['resolution']==1.7
    assert later.steps[1].result['parameters']['effective_native_parameters']['resolution']==effective['clusters']
    assert all(not s.result['parameters']['parameter_selection']['candidates'] for s in later.steps)
    defaults=app.submit_turn('asw','defaults','',expected_generation=2,rerun_revision_id=revision.revision_id,
        parameter_overrides=[dict(tool='run_scanpy',native_function='tl.leiden',step_id='previous',
                                  argument='parameter_selection',value='native_defaults')])
    assert defaults.status=='succeeded',defaults.to_dict()
    revision=app._application.sessions.load('asw').revisions[-1]
    default_run=app._application.run_store.load(revision.run_id)
    choice=default_run.steps[0].result['parameters']['parameter_selection']
    assert choice['origin']=='native_default' and choice['selected_value']==1 and not choice['candidates']
    assert choice['input']['representation']=='X_pca'


def test_real_metric_dimensions_custom_key_copy_and_independent_score(tmp_path, monkeypatch):
    from test_execution_consolidation import _scanpy_runtime
    _scanpy_runtime(monkeypatch)
    from agent.tools import _native_process
    monkeypatch.setattr(_native_process, 'call', lambda operation, arguments, local: local())
    from agent.tools.analysis.native_matrix import SCANPY
    from agent.tools.data.authority_context import VerificationContext, authority_operation
    from sklearn.metrics import silhouette_score
    a=fixture_data();a.obsm['custom_space']=np.random.default_rng(27).normal(size=(90,7))
    a.obsm['custom_space'][:30,:3]+=4
    source=tmp_path/'source.h5ad';a.write_h5ad(source)
    context=VerificationContext()
    with authority_operation(context):
        graph=SCANPY.execute(dict(input_path=str(source),output_dir=str(tmp_path/'neighbors'),function_name='pp.neighbors',
            kwargs={'use_rep':'custom_space','n_pcs':3,'metric':'cosine','n_neighbors':8,'key_added':'custom_neighbors'}))
        result=SCANPY.execute(dict(input_path=graph['matrix_path'],input_sha256=graph['matrix_sha256'],
            output_dir=str(tmp_path/'clusters'),function_name='tl.leiden',parameter_selection='asw',
            kwargs={'neighbors_key':'custom_neighbors','key_added':'custom_groups','copy':True}))
    selection=result['parameters']['parameter_selection'];record=selection['input']
    assert record['dimensions']==[0,1,2] and record['metric']=='cosine'
    assert record['graph']=='custom_neighbors_connectivities' and record['representation']=='custom_space'
    assert describe('run_scanpy',result)['grouping']=='custom_groups'
    out=ad.read_h5ad(result['matrix_path'])
    score=silhouette_score(a.obsm['custom_space'][:,:3],out.obs['custom_groups'],metric='cosine')
    selected=next(row for row in selection['candidates'] if row['value']==selection['selected_value'])
    assert score==pytest.approx(selected['asw'],abs=1e-12)
    np.testing.assert_array_equal(out.obsm['custom_space'],a.obsm['custom_space'])


def test_unqualified_or_sparse_precomputed_graph_does_not_invent_distances():
    a=fixture_data(12);a.obsm['distances_input']=sparse.eye(12,format='csr')
    record=dict(graph='connectivities',function='scanpy.pp.neighbors',representation='distances_input',
                dimensions=None,metric='precomputed',metric_kwds={})
    calls=[]
    def native(a,resolution=1):
        calls.append(resolution)
        a.obs['leiden']=pd.Categorical(['0']*6+['1']*6)
    _, evidence=asw.invoke(native,a,'scanpy','tl.leiden',{},'asw',[record])
    assert calls==[1] and evidence['status']=='unavailable' and evidence['sample_count']==0
    assert not evidence['candidates'] and 'complete pairwise' in evidence['reason']


def test_custom_umap_provenance_is_source_scoped_and_never_scored(tmp_path, monkeypatch):
    from agent.tools.analysis.native_matrix import SNAPATAC2
    from agent.tools import _native_process
    from agent.tools.data.authority_context import VerificationContext, authority_operation
    monkeypatch.setattr(_native_process,'call',lambda operation,arguments,local:local())
    monkeypatch.setenv('AGENT_SNAPATAC2_PYTHON','/home/likeyi/program/agent-data/resources/runtimes/native-matrix-1/env/bin/python')
    a=fixture_data(60);a.obsm['X_spectral']=np.random.default_rng(3).normal(size=(60,5))
    a.obsm['X_layout']=a.obsm['X_spectral'].copy()
    source=tmp_path/'source.h5ad';a.write_h5ad(source)
    context=VerificationContext()
    with authority_operation(context):
        layout=SNAPATAC2.execute(dict(input_path=str(source),output_dir=str(tmp_path/'layout'),function_name='tl.umap',
                                      kwargs={'key_added':'layout'}))
        graph=SNAPATAC2.execute(dict(input_path=layout['matrix_path'],output_dir=str(tmp_path/'layout-graph'),function_name='pp.knn',
                                     kwargs={'use_rep':'X_layout','n_neighbors':8}))
        result=SNAPATAC2.execute(dict(input_path=graph['matrix_path'],output_dir=str(tmp_path/'clusters'),function_name='tl.leiden',
                                      parameter_selection='asw'))
        unrelated=SNAPATAC2.execute(dict(input_path=str(source),output_dir=str(tmp_path/'supplied-graph'),function_name='pp.knn',
                                        kwargs={'use_rep':'X_layout','n_neighbors':8}))
    choice=result['parameters']['parameter_selection']
    assert choice['status']=='unavailable' and 'Visualization' in choice['reason'] and not choice['candidates']
    assert not unrelated['parameters']['graph_inputs'][0]['visualization']
