"""W3 completion/presentation integration: tiny science, scripted language only."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.application.interactive_schemas import PresentedResponse
from agent.application.sessions import AnalysisSessions
from agent.providers import PlanningModelFactoryRegistry
from agent.web.app import create_app
from test_interactive_boundary import PROFILES, ScriptedModel
from test_interactive_rich import read_only_boundary, accepted_png
from test_service import _tiny_h5ad, _counting_registry
from test_categorical_evidence import publication
from test_dialogue_evidence import app, accepted, forbid_work, files


class CompletionModel(ScriptedModel):
    def __init__(self, profile, *, explain=True, failure=None):
        delivery = ['explanation'] if explain else []
        super().__init__(profile, dict(kind='execute_plan', target='inspect_scATAC', delivery=delivery))
        self.phases, self.failure = [], failure

    def complete(self, *, prompt, response_schema):
        payload = json.loads(prompt)
        phase = next((key for key in ('turn_schema_version', 'semantic_prompt_version',
            'output_selection_schema_version', 'dialogue_schema_version', 'grounding_review_version')
                      if key in payload), 'unknown')
        self.phases.append(phase)
        if phase == 'output_selection_schema_version':
            self.calls.append(payload)
            return json.dumps(dict(outputs=[dict(name='cells', step_id='inspect', output_key='n_cells'),
                dict(name='features', step_id='inspect', output_key='n_features'),
                dict(name='sparse', step_id='inspect', output_key='x_is_sparse'),
                dict(name='source_file', step_id='inspect', output_key='input_path')]))
        if phase == 'dialogue_schema_version':
            assert payload['focus'] == 'completion'
            assert all(t['completion_role'] == 'created_by_this_run' for t in payload['evidence']['targets'])
            if self.failure == 'provider':
                raise RuntimeError('Private provider failure')
            cells = next(c for c in payload['evidence']['claims'] if c['field'] == 'n_cells')
            sparse = next(c for c in payload['evidence']['claims'] if c['field'] == 'x_is_sparse')
            assert cells['value'] == 2 and sparse['value'] is True
            refs = [cells['claim_id'], sparse['claim_id']]
            if self.failure == 'binding': refs = ['invented-claim']
            return json.dumps(dict(support='supported', paragraphs=[dict(
                text='The accepted inspection describes two cells in a sparse matrix. This does not establish biological quality.',
                kind='finding', citations=refs, comparisons=[])]))
        if phase == 'grounding_review_version' and self.failure == 'review':
            return json.dumps(dict(integrity='intact', verdicts={p['id']: dict(verdict='unsupported', independent=True) for p in payload['paragraphs']}))
        return super().complete(prompt=prompt, response_schema=response_schema)


def service(tmp_path, *, explain=True, failure=None):
    calls, models = [], []
    def factory(profile):
        model = CompletionModel(profile, explain=explain, failure=failure)
        models.append(model)
        return model
    facade = InteractiveAgentApplication(tmp_path/'workspace', model_profiles=PROFILES,
        default_profile_id='alpha', registry=_counting_registry(calls),
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    source = _tiny_h5ad(tmp_path/'tiny.h5ad')
    facade.create_session('session')
    return facade, source, calls, models


def submit(facade, source):
    return facade.submit_turn('session', 'first', 'Inspect the matrix and explain the accepted result.',
        expected_generation=0, execution_inputs={'input_path': str(source)})


@pytest.mark.parametrize('failure', [None, 'provider', 'binding', 'review'])
def test_completion_reuses_r5_and_never_revokes_scientific_success(tmp_path, failure):
    facade, source, calls, models = service(tmp_path, failure=failure)
    view = submit(facade, source)
    assert view.status == 'succeeded' and view.response.status == 'activated'
    assert view.response.version == 2 and view.profile_id == 'alpha'
    scientific = view.response.scientific
    assert scientific['availability'] == ('generated' if failure is None else 'unavailable')
    assert scientific['targets'] == (dict(revision_id=view.revision_id, output_name='cells', subject=None),)
    if failure is None:
        assert 'two cells' in view.response.text
        assert {c['field'] for c in scientific['citations']} == {'n_cells', 'x_is_sparse'}
        assert all(c['source']['revision_id'] == view.revision_id for c in scientific['citations'])
    else:
        assert 'explanation is unavailable' in view.response.text and scientific['citations'] == ()
    assert calls == ['inspect_scATAC']
    assert models[0].phases.count('dialogue_schema_version') == 1
    assert models[0].phases.count('grounding_review_version') == (failure not in ('provider', 'binding'))
    before = files(facade._application)
    restarted = read_only_boundary(facade._application)
    assert restarted.turn('session', 'first') == view
    assert restarted.reopen_session('session').turns[-1].response == view.response
    assert restarted.recover_turn('session', 'first') == view
    assert submit(restarted, source) == view and files(facade._application) == before
    assert facade._application.run_store.load(view.run_id).lifecycle_status.value == 'SUCCEEDED'
    assert restarted.revision('session', view.revision_id).artifacts


def test_simple_completion_groups_aliases_and_exposes_exact_fields(tmp_path):
    facade, source, calls, models = service(tmp_path, explain=False)
    view = submit(facade, source)
    assert view.response.scientific['availability'] == 'deterministic'
    assert 'n_cells = 2' in view.response.text and 'x_is_sparse = true' in view.response.text
    assert len(models[0].phases) == 3 and calls == ['inspect_scATAC']
    revision = facade.revision('session', view.revision_id)
    assert len(revision.outputs) == 4 and len(revision.result_groups) == 1
    group = revision.result_groups[0]
    assert group['canonical_output'] == 'cells'
    assert {o['name']: o['field'] for o in group['outputs']} == {
        'cells': 'n_cells', 'features': 'n_features', 'sparse': 'x_is_sparse', 'source_file': 'input_path'}
    with TestClient(create_app(facade)) as client:
        path = f'/api/v1/sessions/session/revisions/{view.revision_id}/evidence'
        selected = client.get(path, params={'output_name':'features', 'fields':'n_features'}).json()
        assert [(f['field'],f['value']) for f in selected['facts']] == [('n_features',3)]
        unsafe = client.get(path, params={'output_name':'source_file', 'fields':'input_path'}).json()
        assert unsafe['facts'][0]['status'] == 'omitted' and unsafe['facts'][0]['value'] is None


def test_accepted_science_without_saved_display_recovers_without_language(tmp_path, monkeypatch):
    facade, source, calls, models = service(tmp_path)
    original = AnalysisSessions.store_presentation
    def interrupted(*args, **kwargs): raise OSError('Interrupted display save')
    monkeypatch.setattr(AnalysisSessions, 'store_presentation', interrupted)
    with pytest.raises(InteractiveBoundaryError): submit(facade, source)
    phases = list(models[0].phases)
    monkeypatch.setattr(AnalysisSessions, 'store_presentation', original)
    recovered = read_only_boundary(facade._application).recover_turn('session', 'first')
    assert recovered.status == 'succeeded' and recovered.response.scientific['availability'] == 'unavailable'
    assert models[0].phases == phases and calls == ['inspect_scATAC']


def test_completion_keeps_its_captured_result_when_navigation_changes_during_language(tmp_path, monkeypatch):
    facade, source, calls, models = service(tmp_path)
    original = CompletionModel.complete
    def concurrent(self, *, prompt, response_schema):
        result = original(self, prompt=prompt, response_schema=response_schema)
        if 'dialogue_schema_version' in json.loads(prompt):
            # Script a second accepted metadata revision while language runs.
            # This fixture creates no second native scientific execution.
            accepted(facade._application, 'inspect_scATAC', name='concurrent', overrides={'n_cells':9})
        return result
    monkeypatch.setattr(CompletionModel, 'complete', concurrent)
    view = submit(facade, source)
    current = facade._application.sessions.load('session')
    assert view.response.scientific['availability'] == 'generated'
    assert view.base_generation == 0 and view.base_revision_id is None
    assert current.generation == 2 and current.active_revision_id != view.revision_id
    assert 'historical revision' in view.response.text and 'form the current revision' not in view.response.text
    assert all(c['source']['revision_id'] == view.revision_id for c in view.response.scientific['citations'])
    assert not facade.revision('session', view.revision_id).is_active and calls == ['inspect_scATAC']


def test_completion_discloses_distinct_undelivered_requirements(tmp_path, monkeypatch):
    facade, source, calls, models = service(tmp_path)
    original = CompletionModel.complete
    def compound(self, *, prompt, response_schema):
        if 'turn_schema_version' in json.loads(prompt):
            self.decision.update(delivery=['explanation','comparison','visualization'], unfulfilled=['marker analysis'])
        return original(self, prompt=prompt, response_schema=response_schema)
    monkeypatch.setattr(CompletionModel, 'complete', compound)
    view = facade.submit_turn('session','first',
        'Inspect and explain the accepted result, compare and visualize it; omit marker analysis if unavailable.',
        expected_generation=0, execution_inputs={'input_path':str(source)})
    assert view.response.scientific['availability'] == 'generated'
    assert 'exact comparison operands were not recorded' in view.response.text
    assert 'requested visualization has not been produced' in view.response.text
    assert 'Requirements not fulfilled: marker analysis' in view.response.text
    assert calls == ['inspect_scATAC'] and models[0].phases.count('dialogue_schema_version') == 1


def test_optional_figure_descriptions_do_not_displace_selected_scientific_facts(tmp_path,monkeypatch):
    from agent.application import dialogue_evidence
    facade,source,calls,models=service(tmp_path)
    def oversized_inventory(sessions,turn,revision):
        return (dict(metadata=dict(manifest_sha256='a'*64,figure_id='bounded-metadata-fixture',
            kind='fixture',sources=[dict(run_id=turn.run_id,step_id='inspect',tool_name='inspect_scATAC',
                field='x'*40000)])),)
    monkeypatch.setattr(dialogue_evidence,'figure_inventory',oversized_inventory)
    view=submit(facade,source)
    assert view.response.scientific['availability']=='generated' and 'two cells' in view.response.text
    assert any('Optional figure or download descriptions were omitted' in value
        for value in view.response.scientific['limitations'])
    assert calls==['inspect_scATAC'] and models[0].phases.count('dialogue_schema_version')==1


def test_v1_history_roundtrips_without_reinterpretation():
    legacy = PresentedResponse('answer','answered','Exact historical bytes.\n', version=1,
        scientific=dict(support='supported', limitations=[], evidence_scope='accepted_persisted_summary'))
    assert PresentedResponse.from_dict(legacy.to_dict()).to_dict() == legacy.to_dict()
    with pytest.raises(ValueError):
        replace(legacy, kind='execute')


def test_distinct_operations_keep_result_identity_without_alias_fanout(app, monkeypatch):
    from agent.application.scientific_dialogue import completion
    from agent.application.turns import TurnOutcome
    rid = accepted(app, 'inspect_scATAC', copies=2, overrides={'n_cells':7})[0]
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    groups = facade.revision('session', rid).result_groups
    assert len(groups) == 2 and {g['canonical_output'] for g in groups} == {'output0','output1'}
    result = completion(app.sessions, 'session', 'initial', TurnOutcome('execute','activated'))
    assert result.scientific.availability == 'deterministic'
    assert not any('figure relationships are unavailable' in value for value in result.scientific.limitations)
    assert {c.target for c in result.scientific.claims} == {'t0','t1'}
    assert result.text.count('Dataset inspection:') == 2


def test_http_categorical_tables_preserve_owner_denominators_and_partial_coverage(app, monkeypatch):
    rid, _, _ = publication(app)
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    before = files(app)
    with TestClient(create_app(facade)) as client:
        path = f'/api/v1/sessions/session/revisions/{rid}/evidence'
        query = dict(output_name='output0', detail_section='category_counts', field='result:primary_annotation', limit=1, offset=1)
        response = client.get(path, params=query)
        assert response.status_code == 200, response.text
        fact = response.json()['detail'][0]
        assert fact['value']['denominator_all'] == 5 and not fact['value']['complete']
        assert fact['value']['offset'] == 1 and fact['value']['fields'][0]['role'] == 'computational_assignment'
        query.update(detail_section='category_association', other_field='source:treatment', offset=0, limit=8,
                     comparison_output_name='output0')
        table = client.get(path, params=query).json()['detail'][0]
        assert table['value']['statistic'] == 'categorical_contingency' and table['value']['denominator'] == 3
        assert table['derivation']['comparison']['revision_id'] == rid
        query['comparison_output_name'] = 'missing'
        assert client.get(path, params=query).status_code == 400
    assert files(app) == before


def test_figure_identity_comes_from_pinned_manifest_and_legacy_png_remains_readable(app, monkeypatch):
    from agent.application.session_state import CompletionFile
    rid, png, data = accepted_png(app)
    state = app.sessions.load('session'); turn = state.turn('initial')
    path = png.parent/'visualization_manifest.json'
    path.write_text(json.dumps(dict(source=dict(run_id=turn.run_id), figures=[dict(
        figure_id='fixture-umap', relative_path=png.name, png_sha256=hashlib.sha256(data).hexdigest(), figure_kind='umap_coordinates',
        source_step_ids=['step0'], source_artifacts=[dict(producing_step_id='step0',
            tool_name='inspect_scATAC', artifact_path='/private/path', artifact_kind='fixture', integrity={})])])), encoding='utf-8')
    record = CompletionFile(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    app.sessions._store._write(replace(state, turns=(replace(turn, completion_files=turn.completion_files+(record,)),)))
    facade = read_only_boundary(forbid_work(app, monkeypatch))
    figure = facade.artifact_handles('session', rid)[0]
    assert figure.figure['output_names'] == ('output0',) and figure.figure['step_ids'] == ('step0',)
    assert '/private/path' not in json.dumps(figure.to_dict())
    assert facade.artifact_content('session', rid, figure.handle).data == data
    path.write_text('{}')
    with pytest.raises(InteractiveBoundaryError): facade.artifact_handles('session', rid)


def overlay_service(tmp_path, monkeypatch, *, failure=None, compare=False, explain=False, limit=8, compare_same=False):
    """Real dialogue/runtime/composer, scripted cluster owner + graph facts.

    As in W2's qualified-graph fixture, no native graph authority or scientific
    algorithm is established. Actual files, ordering, digests and readers run.
    """
    import shutil
    import anndata as ad
    import numpy as np
    import pandas as pd
    from agent.orchestration import ToolRegistry, build_default_tool_registry
    from agent.orchestration import verifier
    from agent.application import dialogue_evidence as de
    from agent.report import visualization as viz
    from agent.application.session_state import CompletionFile
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY, PROVENANCE_SCHEMA_VERSION

    tmp_path.mkdir(parents=True, exist_ok=True)
    ids = ['a','b','c','d','e']
    xy = np.array([[0,0],[0,1],[2,2],[3,2],[4,4]], dtype=np.float32)
    coords = tmp_path/'coordinates.h5ad'
    layout = ad.AnnData(obs=pd.DataFrame(index=ids)); layout.obsm['X_umap'] = xy
    layout.obs['cohort'] = pd.Categorical(['control', 'control', 'treated', 'treated', 'treated'])
    layout.uns[PROVENANCE_KEY] = dict(schema_version=PROVENANCE_SCHEMA_VERSION, clustering_present=False)
    layout.write_h5ad(coords)
    new_labels = tmp_path/'new-labels.h5ad'
    ad.AnnData(obs=pd.DataFrame(dict(cohort=pd.Categorical(['treated','control','treated',None,'control'],
        categories=['control','treated','unused'], ordered=True)),
        index=list(reversed(ids)) if failure == 'axis' else ids)).write_h5ad(new_labels)
    if compare_same:
        same = ad.read_h5ad(new_labels)
        same.obs['condition'] = pd.Categorical(['A', 'A', 'B', 'B', 'B'])
        same.write_h5ad(new_labels)
    default = build_default_tool_registry(); calls = []; models = []
    spec = default.get('cluster_cells')
    def labels_owner(**arguments):
        calls.append('scripted_cluster_owner')
        path = Path(arguments['output_dir'])/'labels.h5ad'
        path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(new_labels,path)
        values = {str:'fixture', int:2, float:.5, bool:True, dict:{}, list:[]}
        result = {k:values[t[0]] for k,t in spec.result_contract.required_fields.items()}
        result.update(status='success', analysis_path=str(path), analysis_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            input_path=str(coords), n_cells=5, n_clusters=2, cluster_key='cohort', algorithm='leiden',
            resolution=1.0, random_seed=0)
        return result
    registry = ToolRegistry(tuple(replace(default.get(n), function=labels_owner, scientific_authority=None)
        if n=='cluster_cells' else default.get(n) for n in default.names()))
    monkeypatch.setattr(verifier, '_verify_clustering', lambda arguments,result,checks: None)
    def graph_facts(sessions,run,step,planned):
        identity = ('b' if failure=='unqualified' and step.tool_name=='cluster_cells' else 'a')*64
        return (de.EvidenceFact('umap_neighbors_identity_sha256' if step.tool_name=='compute_cell_umap'
            else 'clustering_neighbors_identity_sha256', 'available', identity),)
    monkeypatch.setattr(de,'_clustering_neighbors_facts',graph_facts)
    class OverlayModel(ScriptedModel):
        def complete(self, *, prompt, response_schema):
            value=json.loads(prompt); self.calls.append(value)
            if 'turn_schema_version' in value:
                group=next(g for g in value['dialogue']['result_groups'] if g['tool']=='compute_cell_umap')
                decision = dict(kind='execute_plan', target='cluster_cells', references=[group['handle']],
                    delivery=['visualization'] + (['explanation'] if explain else []),
                    visualization=dict(coordinates=group['handle'],labels='new',field='result:cohort'))
                if compare:
                    decision['comparison'] = dict(left='new' if compare_same else group['handle'], right='new',
                        detail=dict(section='category_association', field='result:cohort',
                                    other_field='result:condition' if compare_same else
                                        'result:unknown' if failure == 'wrong_field' else 'result:cohort',
                                    offset=0, limit=limit))
                return json.dumps(dict(turn_schema_version=1, decision=decision))
            if 'semantic_prompt_version' in value:
                assert value['research_context']['task']['visualization']['field']=='result:cohort'
                return json.dumps(dict(schema_version=4,decision=dict(kind='plan',steps=[dict(
                    step_id='labels',tool='cluster_cells',sources=[dict(target='analysis',
                    source=dict(kind='input',input='analysis_path'))],control_dependencies=[])])))
            if 'output_selection_schema_version' in value:
                return json.dumps(dict(outputs=[dict(name='updated_labels',step_id='labels',output_key='analysis_path',
                    use=['visualization_labels', 'comparison_result'] if compare else 'visualization_labels'),
                    dict(name='label_count',step_id='labels',output_key='n_clusters',use=None)]))
            if 'dialogue_schema_version' in value:
                evidence = value['evidence']
                table = next(c for c in evidence['claims'] if c['field'] == 'detail.category_association')
                assert evidence['targets'][0]['completion_role'] == ('created_by_this_run' if compare_same else 'referenced_accepted_result')
                assert evidence['targets'][1]['completion_role'] == 'created_by_this_run'
                assert evidence['delivery']['comparison']['status'] == ('partial' if limit == 1 else 'available')
                figure = evidence['delivery']['figure_relationships'][0]
                assert figure['satisfies_request']
                assert figure['coordinate_reused'] and figure['correspondence']['basis'] == 'same_qualified_neighbors'
                assert {s['field'] for s in figure['sources']} == {'obsm:X_umap', 'result:cohort'}
                assert '/secret' not in json.dumps(evidence)
                return json.dumps(dict(support='supported', paragraphs=[dict(kind='finding',
                    text='The requested descriptive association uses the selected accepted ordered cells. The figure reuses the accepted coordinates unchanged.',
                    citations=[table['claim_id']], comparisons=[])]))
            if 'grounding_review_version' in value:
                assert value['evidence']['delivery']['figure_relationships'][0]['coordinate_reused']
                return json.dumps(dict(integrity='intact', verdicts={row['id']:dict(verdict='unsupported' if failure=='review' else 'entailed', independent=True)
                    for row in value['paragraphs']}))
            raise AssertionError('Unexpected provider phase')
    def factory(profile):
        model=OverlayModel(profile); models.append(model); return model
    facade=InteractiveAgentApplication(tmp_path/'workspace',model_profiles=PROFILES,default_profile_id='alpha',
        registry=registry,planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':factory}))
    facade.create_session('session'); app=facade._application
    old=accepted(app,'compute_cell_umap',name='old-layout',overrides=dict(
        analysis_path=str(coords),analysis_sha256=hashlib.sha256(coords.read_bytes()).hexdigest(),
        clustering_present=False,n_cells=5),extra_facts=dict(clustering_present=False))[0]
    # A real old PNG with scripted publication metadata, kept immutable below.
    state=app.sessions.load('session'); turn=state.turn('old-layout')
    root=app._workspace.run_paths(turn.run_id).visualizations
    png=root/'old-layout.png'; figure=Figure(figsize=(4,3)); FigureCanvasAgg(figure)
    viz._render_umap(figure,viz._read_umap_presentation(coords,expected_n_cells=5,clustered=False),legend_columns=1)
    figure.savefig(png,metadata={'Software':'Agent plotting spec v1'})
    manifest=root/'visualization_manifest.json'
    manifest.write_text(json.dumps(dict(source=dict(run_id=turn.run_id),figures=[dict(figure_id='old-layout',
        relative_path=png.name,png_sha256=hashlib.sha256(png.read_bytes()).hexdigest(),figure_kind='umap_coordinates',
        source_step_ids=['step0'],source_artifacts=[dict(producing_step_id='step0',tool_name='compute_cell_umap',
        artifact_kind='cell_umap_h5ad',artifact_path=str(coords),integrity={},
        internal=dict(path='/secret',credentials='private-token'))])])),encoding='utf-8')
    extra=tuple(CompletionFile(str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in (png,manifest))
    app.sessions._store._write(replace(state,turns=(replace(turn,completion_files=turn.completion_files+extra),)))
    if failure=='renderer':
        monkeypatch.setattr(viz,'_render_figure',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('Scripted renderer failure')))
    return facade, coords, xy, old, calls, models


def overlay_submit(facade, coords):
    return facade.submit_turn('session','recolor','Update the cell clusters and show cohort on the previous UMAP coordinates.',
        expected_generation=1,execution_inputs={'analysis_path':str(coords)})


@pytest.mark.parametrize('failure',[None,'unqualified','renderer'])
def test_new_labels_on_exact_old_coordinates_use_normal_conversation_and_preserve_success(tmp_path,monkeypatch,failure):
    from agent.report import visualization as viz
    facade,coords,xy,old,calls,models=overlay_service(tmp_path,monkeypatch,failure=failure)
    app=facade._application; before=files(app); coord_bytes=coords.read_bytes()
    from agent.application import scientific_dialogue as sd
    from agent.application.turns import TurnOutcome
    deliveries=[]; context=sd.context
    def capture_delivery(*args,**kwargs):
        public,claims=context(*args,**kwargs)
        if 'delivery' in public: deliveries.append(public['delivery'])
        return public,claims
    monkeypatch.setattr(sd,'context',capture_delivery)
    seen=[]; render=viz._render_figure
    def capture(projection,*args,**kwargs):
        if projection.figure_kind=='umap_categorical': seen.append(projection)
        return render(projection,*args,**kwargs)
    monkeypatch.setattr(viz,'_render_figure',capture)
    view=overlay_submit(facade,coords)
    assert view.status=='succeeded' and view.response.status=='activated',view
    assert app.run_store.load(view.run_id).lifecycle_status.value=='SUCCEEDED'
    assert calls==['scripted_cluster_owner'] and len(models[0].calls)==3
    assert coords.read_bytes()==coord_bytes
    for p,sha in before.items():
        if not p.startswith('sessions/'): assert files(app)[p]==sha
    revision=facade.revision('session',view.revision_id)
    assert not [a for a in revision.artifacts if a.figure]
    figures=app.sessions.presentation_figure_inventory('session','recolor')
    if failure:
        assert not figures and 'requested visualization has not been produced' in view.response.text
        assert deliveries[-1]['accepted_figures']==0 and deliveries[-1]['visualization']=='unavailable'
        assert view.response.version==2 and view.response.artifacts is None
    else:
        assert len(figures)==1 and len(seen)==1
        assert view.response.version==3 and len(view.response.artifacts)==1
        assert deliveries[-1]['accepted_figures']==0 and deliveries[-1]['visualization']=='delivered'
        import numpy as np
        np.testing.assert_array_equal(seen[0].data.coordinates,xy)
        assert seen[0].data.category_labels==('control','treated','Missing label (not an assigned category)')
        assert seen[0].presentation['category_order']==[dict(value='control',missing=False),dict(value='treated',missing=False),dict(value=None,missing=True)]
        assert 'not a new embedding fit' in view.response.text
        meta=figures[0]['metadata']
        assert meta['output_names']==[] and meta['revision_id'] is None
        assert meta['publisher']==dict(kind='interaction',session_id='session',turn_id='recolor')
        assert meta['coordinate_reused']
        assert meta['sources'][0]['output']['revision_id']==old
        assert meta['sources'][0]['result_field']=='analysis_path'
        assert meta['sources'][1]['output']['revision_id']==view.revision_id
        assert meta['sources'][1]['field']=='result:cohort'
        assert meta['correspondence']['basis']=='same_qualified_neighbors'
        assert str(tmp_path) not in json.dumps(revision.to_dict())
        handle=facade.presentation_artifact_handles('session','recolor')[0]
        assert handle.revision_id is None and handle.figure['publisher']['turn_id']=='recolor'
        assert facade.presentation_artifact_content('session','recolor',handle.handle).data.startswith(b'\x89PNG')
        # Explicit completion without a live model never adopts or regenerates
        # an existing Interaction bundle as a new delivered presentation.
        outcome=sd.completion(app.sessions,'session','recolor',TurnOutcome('execute','activated'))
        assert 'requested visualization has not been produced' in outcome.text
        assert not outcome.artifacts
        assert deliveries[-1]['accepted_figures']==0 and deliveries[-1]['visualization']=='unavailable'
    saved=files(app); restarted=read_only_boundary(app)
    assert restarted.turn('session','recolor')==view
    assert restarted.revision('session',view.revision_id).artifacts==revision.artifacts
    assert 'private-token' not in json.dumps(restarted.revision('session',old).to_dict())
    assert restarted.artifact_content('session',old, restarted.artifact_handles('session',old)[0].handle).data.startswith(b'\x89PNG')
    assert files(app)==saved


def test_review_withholding_keeps_independently_published_figure_pins(tmp_path,monkeypatch):
    facade,coords,_,_,calls,models=overlay_service(tmp_path,monkeypatch,compare=True,explain=True,failure='review')
    view=overlay_submit(facade,coords)
    assert view.status=='succeeded' and calls==['scripted_cluster_owner']
    assert view.response.scientific['availability']=='unavailable'
    assert view.response.scientific['citations']==()
    assert 'scientific explanation is unavailable' in view.response.text
    assert 'requested visualization is available' in view.response.text
    assert view.response.version==3 and len(view.response.artifacts)==1
    assert sum('dialogue_schema_version' in c for c in models[0].calls)==1
    assert sum('grounding_review_version' in c for c in models[0].calls)==1
    figure=facade.presentation_artifact_handles('session','recolor')[0]
    data=facade.presentation_artifact_content('session','recolor',figure.handle).data
    before=files(facade._application)
    restarted=read_only_boundary(forbid_work(facade._application,monkeypatch))
    assert restarted.recover_turn('session','recolor')==view
    assert restarted.turn('session','recolor')==view
    assert restarted.presentation_artifact_content('session','recolor',figure.handle).data==data
    assert files(facade._application)==before


def test_public_provenance_projects_only_known_source_fields():
    from agent.application.dialogue_evidence import public_citation_source
    locator=dict(name='labels',run_id='run',step_id='step',output_key='analysis_path',accepted_step_sha256='a'*64)
    value=dict(revision_id='revision',artifact_sha256='b'*64,output_locator=dict(locator,private='/secret'),
        private='/secret',derivation=dict(read_version='categorical-observations.v1',private='/secret',
        sources=[dict(run_id='run',field='cohort',artifact_sha256='c'*64,artifact_path='/secret',
            internal=dict(credentials='secret'))],request=dict(section='category_counts',field='result:cohort',
            internal='/secret'),comparison=None))
    public=public_citation_source(value)
    assert public['output_locator']==locator and public['derivation']['sources'][0]['field']=='cohort'
    assert 'secret' not in json.dumps(public) and 'internal' not in json.dumps(public)


@pytest.mark.parametrize('explain,limit', [(False, 8), (True, 8), (False, 1), (True, 1)])
def test_precise_comparison_completes_with_existing_readers_and_same_figure_facts(tmp_path, monkeypatch, explain, limit):
    facade, coords, _, old, calls, models = overlay_service(tmp_path, monkeypatch,
        compare=True, explain=explain, limit=limit)
    original = coords.read_bytes()
    view = overlay_submit(facade, coords)
    assert view.status == 'succeeded' and calls == ['scripted_cluster_owner']
    assert coords.read_bytes() == original
    assert 'requested comparison is included' in view.response.text.lower()
    if limit == 1:
        assert 'partial coverage' in view.response.text
    assert 'not a new embedding fit' in view.response.text
    scientific = view.response.scientific
    association = [c for c in scientific['citations'] if c['field'] == 'detail.category_association']
    assert len(association) == 1
    sources = association[0]['source']['derivation']['sources']
    assert len(sources) == 2 and sources[0]['artifact_sha256'] != sources[1]['artifact_sha256']
    assert association[0]['source']['revision_id'] == old
    phases = [next((k for k in ('turn_schema_version', 'semantic_prompt_version',
        'output_selection_schema_version', 'dialogue_schema_version', 'grounding_review_version') if k in c), '')
        for c in models[0].calls]
    assert phases.count('dialogue_schema_version') == int(explain)
    assert phases.count('grounding_review_version') == int(explain)
    before = files(facade._application)
    restarted = read_only_boundary(facade._application)
    assert restarted.turn('session', 'recolor') == view
    assert files(facade._application) == before


@pytest.mark.parametrize('failure,counts', [('unqualified', 2), ('axis', 2), ('digest', 1), ('wrong_field', 1)])
def test_unqualified_requested_comparison_keeps_independent_counts_without_matching(tmp_path, monkeypatch, failure, counts):
    facade, coords, _, _, calls, models = overlay_service(tmp_path, monkeypatch,
        compare=True, failure=failure)
    if failure == 'digest':
        from agent.application import result_reading
        resolve = result_reading.comparison_inputs
        def changed(*args):
            result = resolve(*args)
            coords.write_bytes(coords.read_bytes() + b' ')
            return result
        monkeypatch.setattr(result_reading, 'comparison_inputs', changed)
    view = overlay_submit(facade, coords)
    assert view.status == 'succeeded' and calls == ['scripted_cluster_owner']
    assert 'comparison is unavailable' in view.response.text
    assert 'Independent categorical counts' in view.response.text
    fields = [c['field'] for c in view.response.scientific['citations']]
    assert fields.count('detail.category_counts') == counts
    assert 'detail.category_association' not in fields
    assert len(models[0].calls) == 3


@pytest.mark.parametrize('explain', [False, True])
def test_precise_same_new_result_compares_distinct_fields_without_a_second_result(tmp_path, monkeypatch, explain):
    facade, coords, _, _, calls, models = overlay_service(tmp_path, monkeypatch,
        compare=True, compare_same=True, explain=explain)
    view = overlay_submit(facade, coords)
    assert view.status == 'succeeded' and calls == ['scripted_cluster_owner']
    assert 'requested comparison is included' in view.response.text.lower()
    claim = next(c for c in view.response.scientific['citations'] if c['field'] == 'detail.category_association')
    sources = claim['source']['derivation']['sources']
    assert sources[0]['artifact_sha256'] == sources[1]['artifact_sha256']
    assert {s['field'] for s in sources} == {'cohort', 'condition'}
    assert len(models[0].calls) == 3 + 2 * int(explain)


def test_scientific_answer_retains_exact_presentation_request_without_execution(tmp_path, monkeypatch):
    from test_scientific_dialogue import Model, ask
    facade, _, _, old, calls, _ = overlay_service(tmp_path, monkeypatch)
    app = forbid_work(facade._application, monkeypatch)
    def decision(payload):
        group = next(g['handle'] for g in payload['dialogue']['result_groups'] if g['tool'] == 'compute_cell_umap')
        return dict(kind='answer_scientific', target=dict(output=group, subject=None), comparison=None,
            focus='question', visualization=dict(coordinates=group, labels=group, field='result:cohort'))
    def reply(payload):
        assert payload['evidence']['delivery']['visualization'] == 'delivered'
        assert payload['evidence']['delivery']['requested_field'] == 'result:cohort'
        figure=payload['evidence']['delivery']['figure_relationships'][0]
        assert figure['satisfies_request'] and figure['coordinate_reused']
        assert figure['download'] == dict(format='PNG', status='available_with_response',
            actual_download_status='NOT_RECORDED')
        count=next(c for c in payload['evidence']['claims'] if c['field']=='n_cells')
        return dict(support='supported', paragraphs=[dict(kind='finding',
            text='The requested figure uses the accepted coordinates without fitting another embedding.',
            citations=[count['claim_id']], comparisons=[])])
    before = files(app)
    result = ask(app, 'presentation', 'Color the accepted layout by cohort without recomputation.', Model(decision, response=reply))
    assert result.status == 'answered' and calls == []
    state = app.sessions.load('session')
    assert state.active_revision_id == old and len(state.revisions) == 1
    admitted = state.interactions[-1].admitted
    assert admitted['visualization']['field'] == 'result:cohort' and admitted['references']
    assert 'requested figure is available' in result.text and len(result.artifacts)==1
    assert not app.sessions.presentation_figure_inventory('session','presentation')  # No final display commit at this low-level boundary.
    assert all(files(app)[p] == value for p, value in before.items() if not p.startswith('sessions/'))


@pytest.mark.parametrize('withhold',[False,True])
def test_standalone_source_coloring_preserves_registration_meaning_and_figure_when_prose_withheld(tmp_path,monkeypatch,withhold):
    from test_scientific_dialogue import Model
    from agent.application.local_resources import LocalResourceAdmission
    from agent.application import result_reading
    facade,coords,_,old,calls,_=overlay_service(tmp_path,monkeypatch)
    app=facade._application
    owner=LocalResourceAdmission(app._workspace,approved_source_roots=(tmp_path,))
    record=owner.register('source-labels',coords,label='Supplied cohort observations',
        attribution='Simulated original-source relationship; biological validity not assessed.',session_id='session')
    binding=owner.compose_group([record.resource_id],attachment_candidates=True)
    monkeypatch.setattr(result_reading,'original_source',lambda *a:dict(path=str(coords),sha256=record.source_sha256))
    def decision(payload):
        group=next(g['handle'] for g in payload['dialogue']['result_groups'] if g['tool']=='compute_cell_umap')
        return dict(kind='answer_scientific',target=dict(output=group,subject=None),comparison=None,focus='question',
            visualization=dict(coordinates=group,labels='source.0',field='source:cohort'))
    def reply(payload):
        evidence=payload['evidence']; figure=evidence['delivery']['figure_relationships'][0]
        assert evidence['delivery']['visualization']=='delivered' and figure['satisfies_request']
        assert figure['download'] == dict(format='PNG', status='available_with_response',
            actual_download_status='NOT_RECORDED')
        source=next(s for s in figure['sources'] if s['role']=='categories')
        assert source==dict(role='categories',field='source:cohort',origin='registered_source',
            observation_role='supplied_source_observation; biological validity not assessed')
        assert str(tmp_path) not in json.dumps(evidence)
        count=next(c for c in evidence['claims'] if c['field']=='n_cells')
        return dict(support='supported',paragraphs=[dict(kind='finding',
            text='The figure uses the supplied cohort observations on the accepted coordinates.',
            citations=[count['claim_id']],comparisons=[])])
    class PresentationModel(Model):
        def complete(self,*,prompt,response_schema):
            payload=json.loads(prompt)
            if 'grounding_review_version' in payload:
                self.calls.append(payload)
                return json.dumps(dict(integrity='intact', verdicts={r['id']:dict(verdict='unsupported' if withhold else 'entailed', independent=True)
                    for r in payload['paragraphs']}))
            return super().complete(prompt=prompt,response_schema=response_schema)
    model=PresentationModel(decision,response=reply)
    model.model_id=PROFILES[0].model_id
    live=InteractiveAgentApplication(app.workspace_root,model_profiles=PROFILES,default_profile_id='alpha',
        registry=app.registry,planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':lambda profile:model}))
    before=app.sessions.load('session')
    view=live.submit_turn('session','source-color','Color this UMAP by the supplied cohort observations.',
        expected_generation=1,registered_input=binding)
    after=app.sessions.load('session')
    assert view.status=='answered' and view.response.version==3 and len(view.response.artifacts)==1
    assert after.turns==before.turns and after.revisions==before.revisions
    assert after.generation==before.generation==1 and after.active_revision_id==old and calls==[]
    assert len(model.calls)==3  # Interpretation, generation and existing R5 review, no retry.
    assert ('could not ground' in view.response.text)==withhold
    assert 'requested figure is available' in view.response.text
    handle=live.presentation_artifact_handles('session','source-color')[0]
    label_source=next(s for s in handle.figure['sources'] if s['role']=='categories')
    assert dict(label_source['registered_source'])==dict(resource_id=record.resource_id,
        record_sha256=record.record_sha256,source_sha256=record.source_sha256)
    assert not {'output','tool_name','step_id','run_id'} & set(label_source)
    png=live.presentation_artifact_content('session','source-color',handle.handle).data
    saved=files(app); reopened=read_only_boundary(forbid_work(app,monkeypatch))
    assert reopened.turn('session','source-color')==view
    assert reopened.recover_turn('session','source-color')==view
    assert reopened.presentation_artifact_content('session','source-color',handle.handle).data==png
    assert files(app)==saved


@pytest.mark.parametrize('corrupt_figure', [False, True])
def test_read_only_answer_uses_same_pinned_relationship_or_reports_its_unavailability(tmp_path, monkeypatch, corrupt_figure):
    from test_scientific_dialogue import Model, ask
    facade, coords, _, _, calls, _ = overlay_service(tmp_path, monkeypatch)
    view = overlay_submit(facade, coords)
    if corrupt_figure:
        figure = facade._application.sessions.presentation_figure_inventory('session','recolor')[0]
        path = Path(figure['path'])
        path.write_bytes(path.read_bytes() + b' ')
    app = forbid_work(facade._application, monkeypatch)
    before = files(app)
    def decision(payload):
        group = next(g['handle'] for g in payload['dialogue']['result_groups'] if g['tool'] == 'cluster_cells')
        return dict(kind='answer_scientific', target=dict(output=group, subject=None), comparison=None, focus='question')
    def reply(payload):
        evidence = payload['evidence']
        count = next(c for c in evidence['claims'] if c['field'] == 'n_clusters')
        if corrupt_figure:
            assert not evidence.get('delivery', {}).get('figure_relationships')
            assert any('figure relationships are unavailable' in text for text in evidence['limitations'])
            text = 'The accepted clustering count remains available; its figure relationship cannot currently be checked.'
        else:
            figure = evidence['delivery']['figure_relationships'][0]
            assert figure['coordinate_reused'] and not figure['satisfies_request']
            assert figure['download'] == dict(format='PNG', status='available_with_response',
                actual_download_status='NOT_RECORDED')
            assert {s['version'] for s in figure['sources']} == {1, 2}
            text = 'The accepted figure uses the exact retained coordinates; recoloring did not fit a new embedding.'
        return dict(support='supported', paragraphs=[dict(kind='finding', text=text,
            citations=[count['claim_id']], comparisons=[])])
    outcome = ask(app, 'relationship', 'Explain the accepted figure sources without rerunning anything.', Model(decision, response=reply))
    assert outcome.status == 'answered' and calls == ['scripted_cluster_owner']
    assert all(files(app)[p] == value for p, value in before.items() if not p.startswith('sessions/'))


def test_generic_figure_delivery_recognizes_existing_native_inventory(tmp_path,monkeypatch):
    from agent.application import dialogue_evidence
    facade,source,calls,models=service(tmp_path)
    original=CompletionModel.complete
    def requested(self,*,prompt,response_schema):
        if 'turn_schema_version' in json.loads(prompt):self.decision['delivery']=['explanation','visualization']
        return original(self,prompt=prompt,response_schema=response_schema)
    monkeypatch.setattr(CompletionModel,'complete',requested)
    # Explicit simulated inventory; no biological figure qualification.
    monkeypatch.setattr(dialogue_evidence,'figure_inventory',lambda *a:(dict(metadata=dict(
        manifest_sha256='a'*64,figure_id='simulated-native',kind='umap',sources=[])),))
    view=submit(facade,source)
    assert 'requested visualization is available in the result view' in view.response.text
    assert 'has not been produced' not in view.response.text
    assert calls==['inspect_scATAC']
