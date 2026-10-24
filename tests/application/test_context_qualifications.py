"""Qualified explanation context over bounded real readers and scripted publications.

These checks call no models or scientific algorithms. Accepted metadata fixtures
exercise exact source and presentation projection, not biological qualification.
"""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from agent.application import dialogue_evidence as evidence
from agent.application import result_reading, scientific_dialogue as dialogue
from test_dialogue_evidence import app, accepted, files, forbid_work
from test_dialogue_detail import publication


def target(app, revision_id):
    revision = next(r for r in app.sessions.load('session').revisions if r.revision_id == revision_id)
    output = revision.outputs[0]
    return dict(revision_id=revision_id, output_name=output.name,
                accepted_step_sha256=output.accepted_step_sha256, subject=None)


def test_only_catalog_guidance_accepts_empty_evidence_without_resolving_predecessor_exports(app, monkeypatch):
    before = files(app)
    state = app.sessions.load('session')
    # A read-only predecessor projection isolates the optional recipe boundary;
    # it is not written as an accepted Interaction or scientific state.
    previous = SimpleNamespace(turn_id='previous-export', presentation={}, admitted={'export': {}})
    projected = SimpleNamespace(revisions=state.revisions, turns=state.turns, interactions=(previous,))
    monkeypatch.setattr(app.sessions, 'load', lambda session_id: projected)
    monkeypatch.setattr(app.sessions, 'composed_export',
        lambda *args, **kwargs: pytest.fail('Empty guidance resolved an unrelated export recipe'))
    admitted = dict(intent='guidance', target=None, comparison=None, predecessor=previous.turn_id)
    public, claims = dialogue.context(app.sessions, 'session', admitted, evidence_targets=[])
    assert public['targets'] == public['claims'] == [] and claims == ()
    assert 'delivery' not in public
    for scientific in (dict(intent='scientific', target=None, comparison=None),
                       dict(target=None, comparison=None, completion_delivery={'requested': ['explanation']})):
        with pytest.raises(dialogue.IntentError) as exc:
            dialogue.context(app.sessions, 'session', scientific, evidence_targets=[])
        assert exc.value.reason == 'unavailable_context'
    assert files(app) == before


@pytest.mark.parametrize('tool', ['cluster_cells', 'compute_scATAC_qc'])
def test_result_limitations_are_projected_once_without_removing_qualified_meanings(app, monkeypatch, tool):
    from agent.tools.data.scatac_qc_profile import PROFILE, PROFILE_SHA256
    options = (dict(overrides={'science_profile_sha256': PROFILE_SHA256},
                    extra_facts={'tss_method': PROFILE.tss_method}) if tool == 'compute_scATAC_qc' else {})
    revision_id = accepted(app, tool, **options)[0]
    selected = target(app, revision_id)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    public, _ = dialogue.context(guarded.sessions, 'session', dict(target=selected, comparison=None))
    limitation = evidence.RESULT_LIMITATIONS[tool]
    assert public['limitations'].count(limitation) == 1
    assert limitation not in public['meanings'].values()
    if tool == 'compute_scATAC_qc':
        assert public['targets'][0]['reviewed_profile'] == asdict(PROFILE)
        assert any('endpoint incidence per base' in value for value in public['meanings'].values())
    assert files(app) == before


def test_annotation_rationale_context_keeps_exact_markers_and_consistent_computational_limits(app, monkeypatch):
    revision_id, _ = publication(app, annotation_version=2)
    selected = target(app, revision_id)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    public, claims = dialogue.context(guarded.sessions, 'session', dict(target=selected, comparison=None))
    candidates = [c for c in claims if c.field.startswith('detail.candidate.')]
    assert candidates and all(c.value['positive'] == ('MS4A1',) for c in candidates)
    assert all(c.source['revision_id'] == revision_id and c.source['artifact_sha256'] for c in candidates)
    assert any('Published annotation rationale' in value for value in public['meanings'].values())
    assert evidence.RESULT_LIMITATIONS['annotate_scATAC_cell_types'] in public['limitations']
    assert 'does not contain marker rationale' not in json.dumps(public)
    assert 'not independently validated biological cell identities' in json.dumps(public)
    assert files(app) == before


@pytest.mark.parametrize('field, expected_role', [
    ('result:leiden', 'computational_assignment'),
    ('source:cohort', 'supplied_source_observation; biological validity not assessed'),
])
def test_selected_figure_labels_keep_owner_roles_through_r5_projection(app, monkeypatch, field, expected_role):
    from test_composed_export_interaction import case
    current = case(app, monkeypatch)
    guarded = current.facade._application
    state = guarded.sessions.load('session')
    coordinate_revision, label_revision = state.revisions[0], state.revisions[-1]
    coordinate = dict(asdict(coordinate_revision.outputs[0]), revision_id=coordinate_revision.revision_id)
    categories = dict(asdict(label_revision.outputs[0]), revision_id=label_revision.revision_id)
    before = files(guarded)
    overlay = result_reading.cell_overlay(guarded.sessions, coordinate, categories, field)
    category_source = next(row for row in overlay['sources'] if row['role'] == 'categories')
    assert category_source['observation_role'] == expected_role
    metadata = evidence.figure_metadata(dict(figure_kind='umap_categorical', figure_id='overlay',
        source_step_ids=[coordinate['step_id'], categories['step_id']], source_artifacts=overlay['sources'],
        presentation=dict(delivery='accepted_cell_overlay', correspondence=overlay['correspondence'])),
        manifest_sha256='a' * 64, revision=label_revision)
    figure = dict(metadata=metadata)
    relationship, = dialogue._figure_relationships(state, (figure,), (figure,))
    projected = next(row for row in relationship['sources'] if row['role'] == 'categories')
    assert projected['observation_role'] == expected_role
    assert projected['version'] == 2 and projected['field'] == field
    assert relationship['satisfies_request'] and relationship['coordinate_reused']
    assert relationship['correspondence'] == dict(basis='exact_original_source', n_cells=4)
    assert str(app.workspace_root) not in json.dumps(relationship)
    assert not current.models and files(guarded) == before


def test_figure_relationships_keep_registered_roles_and_do_not_infer_legacy_roles():
    state = SimpleNamespace(revisions=[SimpleNamespace(revision_id='revision')])
    sources = [
        dict(role='categories', field='source:condition', registered_source={},
             observation_role='supplied_source_observation; biological validity not assessed'),
        dict(role='categories', field='result:legacy', tool_name='cluster_cells',
             output=dict(revision_id='revision', name='legacy_labels')),
    ]
    figure = dict(metadata=dict(kind='umap_categorical', manifest_sha256='a' * 64,
        figure_id='legacy', sources=sources))
    relationship, = dialogue._figure_relationships(state, (figure,))
    assert relationship['sources'][0]['origin'] == 'registered_source'
    assert relationship['sources'][0]['observation_role'].startswith('supplied_source_observation')
    assert 'observation_role' not in relationship['sources'][1]
    assert relationship['sources'][1]['version'] == 1
