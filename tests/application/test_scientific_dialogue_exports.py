"""Selected W4 recipes in ordinary R5 answers; scripted accepted PCA metadata.

The shared W4 fixture supplies real small H5AD storage and simulated accepted
science. No provider, native analysis, export preparation or download executes.
"""
import json

import pytest

from agent.application import analysis_delivery, scientific_dialogue
from agent.application.matrix_delivery import MatrixDownloadPreparer
from test_analysis_export import accepted_export
from test_dialogue_evidence import app, files, forbid_work
from test_scientific_dialogue import Model, ask


def selection(payload):
    group = next(g for g in payload['dialogue']['result_groups'] if g['tool'] == 'pca_embed_cells')
    assert any(row['field'] == 'obsm:X_pca' for row in group['export_fields'])
    handle = group['handle']
    return dict(kind='answer_scientific', target=dict(output=handle, subject=None),
        comparison=None, focus='question', export=dict(measurement=handle,
            observations=[dict(output=handle, field='source:cohort', destination='cohort_selected')],
            representations=[dict(output=handle, field='obsm:X_pca', destination='X_pca_selected')]))


class ExportModel(Model):
    def __init__(self, *, withhold=False, unavailable=False):
        self.withhold, self.unavailable = withhold, unavailable
        super().__init__(selection, response=self.reply)

    def complete(self, *, prompt, response_schema):
        payload = json.loads(prompt)
        if 'grounding_review_version' in payload:
            self.calls.append(payload)
            selected = payload['evidence']['delivery']['selected_export']
            assert selected['actual_download_status'] == 'NOT_RECORDED'
            return json.dumps(dict(integrity='intact', verdicts={row['id']:dict(verdict='unsupported' if self.withhold else 'entailed', independent=True)
                for row in payload['paragraphs']}))
        return super().complete(prompt=prompt, response_schema=response_schema)

    def reply(self, payload):
        evidence = payload['evidence']
        recipe = evidence['delivery']['selected_export']
        assert recipe['actual_download_status'] == 'NOT_RECORDED'
        if self.unavailable:
            assert recipe['status'] == 'unavailable'
            text = 'The accepted PCA cell count remains available; the selected export recipe is unavailable.'
        else:
            assert recipe['status'] == 'available_for_preparation'
            assert recipe['size_bytes'] is None and recipe['artifact_type'] == 'scientific_composition'
            assert recipe['payload_checks'] == 'performed_on_download'
            contents = recipe['contents']
            assert contents['measurement']['source'] == dict(version=1, output='output0')
            assert contents['observations'][0]['field'] == 'source:cohort'
            assert contents['observations'][0]['destination'] == 'cohort_selected'
            assert contents['representations'][0]['field'] == 'obsm:X_pca'
            assert contents['representations'][0]['destination'] == 'X_pca_selected'
            assert contents['representations'][0]['source'] == dict(version=1, output='output0')
            assert 'run_id' not in json.dumps(recipe) and 'step_id' not in json.dumps(recipe)
            text = 'The recipe retains the original measurements and adds the selected accepted PCA representation under its chosen name. Preparation occurs when downloaded; transfer completion is not recorded here.'
        cells = next(c for c in evidence['claims'] if c['field'] == 'n_cells')
        assert cells['value'] == 4
        return dict(support='supported', paragraphs=[dict(kind='finding', text=text,
            citations=[cells['claim_id']], comparisons=[])])


@pytest.mark.parametrize('withhold', [False, True])
def test_selected_export_uses_existing_r5_and_survives_withholding_without_preparing(app, monkeypatch, withhold):
    fixture, revision_id = accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    monkeypatch.setattr(MatrixDownloadPreparer, 'prepare',
        lambda *a, **k:pytest.fail('Explaining a recipe prepared download bytes.'))
    before = files(guarded)
    state = guarded.sessions.load('session')
    model = ExportModel(withhold=withhold)
    outcome = ask(guarded, 'export', 'Use the original measurements and selected PCA/cohort fields in one Scanpy H5AD.', model)
    assert outcome.status == 'answered' and not outcome.artifacts
    assert 'H5AD recipe is available for preparation when downloaded' in outcome.text
    assert ('could not ground' in outcome.text) == withhold
    assert (outcome.scientific is None) == withhold
    assert len(model.calls) == 3
    after = guarded.sessions.load('session')
    assert after.turns == state.turns and after.revisions == state.revisions
    assert after.generation == state.generation == 1 and after.active_revision_id == revision_id
    admitted = after.interactions[-1].admitted
    assert admitted['export']['representations'][0]['destination'] == 'X_pca_selected'
    assert admitted['delivery'] == ('export',) and admitted['references']
    assert not {'selected_export', 'contents', 'handle'} & set(admitted)
    assert all(files(guarded)[path] == sha for path, sha in before.items() if not path.startswith('sessions/'))


def test_optional_recipe_and_legacy_download_metadata_failure_preserve_primary_facts(app, monkeypatch):
    accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    def unavailable(*args, **kwargs):
        raise ValueError('/private/server/path must not leak')
    monkeypatch.setattr(type(guarded.sessions), 'composed_export', unavailable)
    monkeypatch.setattr(analysis_delivery, 'export_descriptions', unavailable)
    model = ExportModel(unavailable=True)
    outcome = ask(guarded, 'unavailable-export', 'Explain the accepted PCA and selected export.', model)
    assert outcome.status == 'answered' and outcome.scientific is not None
    assert 'accepted PCA cell count remains available' in outcome.text
    assert 'H5AD recipe is unavailable' in outcome.text and '/private/' not in outcome.text
    assert len(model.calls) == 3
    assert any('Optional download descriptions are unavailable' in text for text in outcome.scientific.limitations)


def test_stable_native_method_tampering_removes_recipe_but_preserves_accepted_numeric_facts(app, monkeypatch):
    import h5py
    from agent.tools.analysis import pca_embedding
    fixture, _ = accepted_export(app, monkeypatch)
    before = fixture.native.read_bytes()
    with h5py.File(fixture.native, 'r+') as handle:
        handle['uns'][pca_embedding.KEY]['parameters']['random_seed'][()] = 991
    assert fixture.native.read_bytes() != before
    guarded = forbid_work(app, monkeypatch)
    def choose(payload):
        group = next(g['handle'] for g in payload['dialogue']['result_groups'] if g['tool'] == 'pca_embed_cells')
        return dict(kind='answer_scientific', target=dict(output=group, subject=None), comparison=None, focus='question',
            export=dict(measurement=group, observations=[], representations=[dict(output=group,
                field='obsm:X_pca', destination='X_pca_selected')]))
    model = ExportModel(unavailable=True)
    model.decision = choose
    outcome = ask(guarded, 'changed-native-method', 'Explain the accepted PCA and select its H5AD representation.', model)
    assert outcome.status == 'answered' and outcome.scientific is not None
    assert any(c.field == 'n_cells' and c.value == 4 for c in outcome.scientific.claims)
    assert 'accepted PCA cell count remains available' in outcome.text and 'H5AD recipe is unavailable' in outcome.text
    assert len(model.calls) == 3


def test_known_collision_is_meaningful_without_publishing_private_exception_text(app, monkeypatch):
    accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    def conflict(*args, **kwargs):
        exc = analysis_delivery.ExportSelectionError('destination_collision')
        exc.args = ('/private/unsafe details',)
        raise exc
    monkeypatch.setattr(type(guarded.sessions), 'composed_export', conflict)
    outcome = ask(guarded, 'colliding-export', 'Explain the accepted PCA and selected export.', ExportModel(unavailable=True))
    assert outcome.status == 'answered' and outcome.scientific is not None
    assert 'Choose a distinct destination name' in outcome.text and '/private/' not in outcome.text


def test_file_followup_discusses_only_the_committed_predecessor_recipe_without_claiming_a_transfer(app, monkeypatch):
    from agent.application.interactive_schemas import PresentedResponse
    accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    monkeypatch.setattr(MatrixDownloadPreparer, 'prepare',
        lambda *a, **k:pytest.fail('A file explanation prepared derivative bytes.'))
    first = ask(guarded, 'selected-export', 'Select a reusable H5AD with original measurements and PCA.', ExportModel())
    guarded.sessions.store_presentation('session', 'selected-export',
        PresentedResponse('answer', 'answered', first.text).to_dict())
    before = files(guarded)
    def choose(payload):
        group = next(g['handle'] for g in payload['dialogue']['result_groups'] if g['tool'] == 'pca_embed_cells')
        return dict(kind='answer_scientific', target=dict(output=group, subject=None), comparison=None, focus='question')
    def reply(payload):
        evidence = payload['evidence']; delivery = evidence['delivery']
        assert 'selected_export' not in delivery
        recipes = [row for row in delivery['exports'] if row['artifact_type'] == 'scientific_composition']
        assert len(recipes) == 1 and recipes[0]['actual_download_status'] == 'NOT_RECORDED'
        recipe = recipes[0]
        assert recipe['status'] == 'available_for_preparation' and recipe['size_bytes'] is None
        assert recipe['contents']['representations'][0]['destination'] == 'X_pca_selected'
        cells = next(c for c in evidence['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[dict(kind='finding',
            text='Use composed-analysis.h5ad for the original measurements and selected PCA/cohort fields. It is prepared when downloaded; this context does not record whether a transfer was received.',
            citations=[cells['claim_id']], comparisons=[])])
    model = Model(choose, response=reply)
    result = ask(guarded, 'which-file', 'Which file should I use in Scanpy, and did the previous download finish?', model)
    assert result.status == 'answered' and 'Use composed-analysis.h5ad' in result.text
    assert 'H5AD recipe is available' not in result.text  # No new export fulfillment.
    latest = guarded.sessions.load('session').interactions[-1]
    assert latest.admitted['predecessor'] == 'selected-export' and 'export' not in latest.admitted
    assert all(files(guarded)[path] == sha for path, sha in before.items() if not path.startswith('sessions/'))


def test_selected_recipe_context_is_optional_when_its_details_exceed_the_bound(app, monkeypatch):
    from dataclasses import replace
    from agent.schemas.orchestration import _serialize
    accepted_export(app, monkeypatch)
    guarded = forbid_work(app, monkeypatch)
    owner = type(guarded.sessions).composed_export
    def large(self, *args, **kwargs):
        artifact = owner(self, *args, **kwargs)
        contents = _serialize(artifact.reference.contents)
        contents['representations'][0]['method'] = dict(description='x'*40000)
        return replace(artifact, reference=replace(artifact.reference, contents=contents))
    monkeypatch.setattr(type(guarded.sessions), 'composed_export', large)
    def response(payload):
        evidence = payload['evidence']; selected = evidence['delivery']['selected_export']
        assert selected['status'] == 'available_for_preparation' and 'contents' not in selected
        assert selected['contents_omitted']
        cells = next(c for c in evidence['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[dict(kind='finding',
            text='The accepted PCA has four cells and the recipe remains available; its optional details exceed this answer context.',
            citations=[cells['claim_id']], comparisons=[])])
    outcome = ask(guarded, 'bounded-export', 'Explain the accepted PCA and selected export.', Model(selection, response=response))
    assert outcome.status == 'answered' and outcome.scientific is not None
    assert 'H5AD recipe is available' in outcome.text


def test_recipe_components_keep_cross_revision_methods_separate_from_primary_umap_summary(app, monkeypatch):
    import anndata as ad
    import pandas as pd
    from agent.schemas import AgentRequest
    from agent.application import result_reading
    from agent.tools.analysis.embedding_analysis import PROVENANCE_KEY
    from test_analysis_export import export_fixture, sha
    from test_dialogue_evidence import accepted
    base, _ = accepted_export(app, monkeypatch)
    revisions = {}
    for name, tool, stage in (('old-layout', 'compute_cell_umap', 'umap'),
                              ('new-clusters', 'cluster_cells', 'clustering')):
        _, paths = app._prepare_request(AgentRequest(name, 'Simulated accepted component.', {}))
        fixture = export_fixture(paths.scientific, stage=stage, source=base.source)
        if stage == 'clustering':
            values = ad.read_h5ad(fixture.native)
            values.obs['leiden'] = pd.Categorical(['0', '1', '2', '2'])
            values.uns[PROVENANCE_KEY]['parameters']['clustering']['resolution'] = .5
            values.write_h5ad(fixture.native)
        revisions[name] = accepted(app, tool, name=name, overrides=dict(
            analysis_path=str(fixture.native), analysis_sha256=sha(fixture.native), n_cells=4,
            n_clusters=3 if stage == 'clustering' else 2, resolution=.5 if stage == 'clustering' else .75,
            cluster_key='leiden'))[0]
    # Accepted-source association is simulated, exactly as the W4 fixtures;
    # real pinned H5AD fields/methods and cross-Revision locators are consumed.
    original = result_reading.original_source
    monkeypatch.setattr(result_reading, 'original_source', lambda store, run, step:
        dict(path=str(base.source), sha256=sha(base.source)) if run in ('old-layout:run', 'new-clusters:run')
        else original(store, run, step))
    guarded = forbid_work(app, monkeypatch)
    def choose(payload):
        groups = payload['dialogue']['result_groups']
        old = next(g['handle'] for g in groups if g['tool'] == 'compute_cell_umap')
        new = next(g['handle'] for g in groups if g['tool'] == 'cluster_cells')
        return dict(kind='answer_scientific', target=dict(output=old, subject=None), comparison=None,
            focus='question', export=dict(measurement=new,
                observations=[dict(output=new, field='result:leiden', destination='leiden_selected')],
                representations=[dict(output=old, field='obsm:X_umap', destination='X_umap_selected')]))
    def response(payload):
        evidence = payload['evidence']; selected = evidence['delivery']['selected_export']
        assert selected['status'] == 'available_for_preparation'
        contents = selected['contents']; obs = contents['observations'][0]; coord = contents['representations'][0]
        assert obs['source'] == dict(version=3, output='output0')
        assert coord['source'] == dict(version=2, output='output0')
        assert obs['method']['clustering']['parameters']['resolution'] == .5
        assert coord['method']['clustering']['parameters']['resolution'] == .75
        assert obs['destination'] == 'leiden_selected' and coord['destination'] == 'X_umap_selected'
        claim = next(c for c in evidence['claims'] if c['field'] == 'n_cells')
        return dict(support='supported', paragraphs=[dict(kind='finding',
            text='The recipe retains original measurements and combines version3 clustering labels with version2 UMAP coordinates, preserving their distinct method settings.',
            citations=[claim['claim_id']], comparisons=[])])
    before = files(guarded)
    outcome = ask(guarded, 'cross-revision-export', 'Combine the new clusters with the previous UMAP and original measurements.',
        Model(choose, response=response))
    assert outcome.status == 'answered' and 'H5AD recipe is available' in outcome.text, outcome
    assert all(files(guarded)[path] == sha_bytes for path, sha_bytes in before.items() if not path.startswith('sessions/'))
