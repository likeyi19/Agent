"""Bounded export intent and exact accepted identity, using simulated metadata.

Scientific source/format qualification remains tested by the W4 delivery owner.
No scientific algorithm or paid model executes in these identity checks.
"""
from dataclasses import replace
import json

import pytest

from agent.application import scientific_dialogue as dialogue
from agent.application.dialogue_execution import _research_intent
from agent.application.result_reading import export_inputs
from agent.application.session_state import Interaction, SessionError, digest
from agent.application.turn_context import result_groups, snapshot
from agent.application.turn_decisions import (
    IntentError, ScientificQuestion, ScientificTarget, decision_schema, parse_decision,
)
from agent.schemas.orchestration import _serialize
from test_dialogue_evidence import app, accepted, forbid_work


def selection(measurement='g0', observations='g1', representations='g0'):
    return dict(measurement=measurement,
        observations=[dict(output=observations, field='result:leiden', destination='selected_leiden')],
        representations=[dict(output=representations, field='obsm:X_umap', destination='X_selected_umap')])


def parse(export):
    return parse_decision(json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_scientific',
        target=dict(output='g0', subject=None), comparison=None, focus='question', export=export))))


def test_accepted_export_selection_is_immutable_and_preserves_legacy_questions():
    selected = selection()
    question = parse(selected).scientific
    assert _serialize(question.export) == selected
    with pytest.raises(TypeError):
        question.export['observations'][0]['destination'] = 'changed'
    old = parse_decision(json.dumps(dict(turn_schema_version=1, decision=dict(kind='answer_scientific',
        target=dict(output='g0', subject=None), comparison=None, focus='question'))))
    assert old.scientific == ScientificQuestion(ScientificTarget('g0'))
    assert old.scientific.export is None
    # One exact field can be intentionally copied to two explicit destinations.
    selected['observations'].append(selected['observations'][0] | dict(destination='second_leiden'))
    assert len(parse(selected).scientific.export['observations']) == 2


@pytest.mark.parametrize('mutation', [
    'new_measurement', 'new_output', 'wrong_observation_slot', 'wrong_representation_slot',
    'duplicate_destination', 'path_destination', 'reserved_destination', 'control_destination',
    'empty', 'over_bound', 'unexpected_key',
])
def test_export_rejects_unselected_unsafe_or_unbounded_operands(mutation):
    selected = selection()
    if mutation == 'new_measurement':
        selected['measurement'] = 'new'
    elif mutation == 'new_output':
        selected['observations'][0]['output'] = 'new'
    elif mutation == 'wrong_observation_slot':
        selected['observations'][0]['field'] = 'obsm:X_umap'
    elif mutation == 'wrong_representation_slot':
        selected['representations'][0]['field'] = 'result:X_umap'
    elif mutation == 'duplicate_destination':
        selected['observations'] *= 2
    elif mutation == 'path_destination':
        selected['observations'][0]['destination'] = '../leiden'
    elif mutation == 'reserved_destination':
        selected['observations'][0]['destination'] = '_index'
    elif mutation == 'control_destination':
        selected['observations'][0]['destination'] = 'leiden\n'
    elif mutation == 'empty':
        selected.update(observations=[], representations=[])
    elif mutation == 'over_bound':
        selected['observations'] = [selected['observations'][0] | dict(destination=f'label{i}') for i in range(32)]
    else:
        selected['observations'][0]['expression'] = 'guess_a_field()'
    with pytest.raises(IntentError, match='invalid_decision'):
        parse(selected)


def test_export_schema_is_accepted_only_and_does_not_add_planned_purposes():
    public = dict(bases={}, relations=[], dialogue=dict(outputs=[], tools=['compute_cell_umap'],
        result_groups=[dict(handle='g0', tool='compute_cell_umap'), dict(handle='g1', tool='cluster_cells')],
        selected_sources=[dict(source='source.0')]))
    variants = decision_schema(public)['properties']['decision']['anyOf']
    scientific, = [v['properties'] for v in variants if v['properties']['kind']['enum'] == ['answer_scientific']]
    export = scientific['export']['anyOf'][0]
    assert export['additionalProperties'] is False
    assert export['properties']['measurement']['enum'] == ['g0', 'g1', 'source.0']
    assert export['properties']['representations']['items']['properties']['output']['enum'] == ['g0', 'g1']
    assert all('export' not in v['properties'] for v in variants if v['properties']['kind']['enum'] != ['answer_scientific'])


def test_field_bound_keeps_the_distinct_original_measurement_reference():
    selected = dict(measurement='g0', representations=[], observations=[dict(
        output=f'g{i}', field='result:leiden', destination=f'labels{i}') for i in range(1, 33)])
    interaction = Interaction('export', 'Keep the exact selected fields.', None, 0, dict(
        dialogue=dict(results=[dict(handle=f'g{i}', tool='cluster_cells') for i in range(33)])))
    intent = _research_intent(interaction, ScientificQuestion(ScientificTarget('g0'), export=selected))
    assert intent['references'] == [f'g{i}' for i in range(33)]
    assert len(intent['export']['observations']) == 32


def captured(app, monkeypatch):
    """Two genuine Session Revisions with deliberately simulated native results."""
    labels = accepted(app, 'cluster_cells', name='labels')[0]
    coordinates = accepted(app, 'compute_cell_umap', name='coordinates')[0]
    app = forbid_work(app, monkeypatch)
    state = app.sessions.load('session')
    captured = snapshot(app.sessions, state)
    captured['dialogue'] = dialogue.capture(app.sessions, state, captured, None)
    groups = result_groups(captured, state)
    by_revision = {g['canonical']['revision_id']: g for g in groups}
    label, coordinate = by_revision[labels], by_revision[coordinates]
    requested = selection(coordinate['handle'], label['handle'], coordinate['handle'])
    interaction = Interaction('export', 'Give me the selected original data, Leiden labels and earlier coordinates.',
        state.active_revision_id, state.generation, captured)
    scientific = ScientificQuestion(ScientificTarget(coordinate['handle']), export=requested)
    item = coordinate['canonical']
    admitted = dict(kind='answer', intent='scientific', dialogue_version=1,
        target={k:item[k] for k in ('revision_id', 'output_name', 'accepted_step_sha256')} | dict(subject=None),
        comparison=None, previous_subject=None, focus=interaction.utterance, predecessor=None)
    admitted.update(_research_intent(interaction, scientific))
    interaction = replace(interaction, status='answered', admitted=admitted)
    app.sessions._store._write(replace(state, interactions=(interaction,)))
    return app, interaction, requested, labels, coordinates


def test_exact_export_binding_survives_session_reopen_across_revisions(app, monkeypatch):
    app, interaction, selected, labels, coordinates = captured(app, monkeypatch)
    saved = app.sessions.load('session').interactions[-1]
    assert _serialize(saved.admitted['export']) == selected
    assert saved.admitted['delivery'] == ('export',)
    assert set(saved.admitted['references']) == {selected['measurement'], selected['observations'][0]['output']}
    resolved = export_inputs(app.sessions, 'session', saved)
    assert resolved['measurement']['accepted_output']['revision_id'] == coordinates
    assert resolved['observations'][0]['accepted_output']['revision_id'] == labels
    assert resolved['representations'][0]['accepted_output']['revision_id'] == coordinates
    assert resolved['observations'][0]['field'] == 'result:leiden'
    assert resolved['representations'][0]['destination'] == 'X_selected_umap'
    assert len(app.sessions.load('session').turns) == len(app.sessions.load('session').revisions) == 2


@pytest.mark.parametrize('mutation', ['unoffered_group', 'destination_collision', 'missing_source'])
def test_rechecksummed_persisted_export_cannot_expand_captured_authority(app, monkeypatch, mutation):
    app, interaction, _, _, _ = captured(app, monkeypatch)
    path = app.sessions._store._path('session', '.json')
    original = path.read_bytes()
    envelope = json.loads(original)
    selected = envelope['record']['interactions'][0]['admitted']['export']
    if mutation == 'unoffered_group':
        selected['observations'][0]['output'] = 'g999'
    elif mutation == 'destination_collision':
        selected['observations'] *= 2
    else:
        selected['measurement'] = 'source.0'
    envelope['sha256'] = digest(envelope['record'])
    try:
        path.write_text(json.dumps(envelope))
        with pytest.raises(SessionError):
            app.sessions.load('session')
    finally:
        path.write_bytes(original)
    assert export_inputs(app.sessions, 'session', interaction)


def test_changed_accepted_result_digest_cannot_resolve_an_export(app, monkeypatch):
    app, interaction, _, _, _ = captured(app, monkeypatch)
    stored = app.run_store.load('labels:run')
    path = app.run_store.state_path(stored.run_id)
    original = path.read_bytes()
    changed = replace(stored.steps[0], result=dict(stored.steps[0].result, n_cells=999))
    try:
        app.run_store._write_atomic(path, replace(stored, steps=(changed,)))
        with pytest.raises(ValueError):
            export_inputs(app.sessions, 'session', interaction)
    finally:
        path.write_bytes(original)
    assert export_inputs(app.sessions, 'session', interaction)


def test_missing_captured_measurement_source_clarifies_at_existing_typed_boundary():
    interaction = Interaction('export', 'Use the uploaded measurements.', None, 0,
        dict(dialogue=dict(results=[dict(handle='g0', tool='cluster_cells')])))
    question = ScientificQuestion(ScientificTarget('g0'), export=selection('source.0', 'g0', 'g0'))
    with pytest.raises(IntentError, match='invalid_reference'):
        _research_intent(interaction, question)
