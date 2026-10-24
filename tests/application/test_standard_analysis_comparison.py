"""Accepted metadata tests; real scientific acceptance lives outside the repo."""
from dataclasses import replace

import pytest

from agent.application import scientific_dialogue as sd
from agent.application.turn_context import snapshot
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, ask, question


@pytest.mark.parametrize('tool,field,old,new', [
    ('compute_cell_umap', 'min_dist', .5, 0.),
    ('build_cell_neighbors', 'n_neighbors', 15, 5),
    ('cluster_cells', 'resolution', .5, 1.),
])
def test_descriptive_parameters_need_no_population_correspondence(app, monkeypatch, tool, field, old, new):
    revisions = [accepted(app, tool, name=name, overrides={field: value, 'n_cells': cells})[0]
                 for name, value, cells in [('old', old, 8), ('new', new, 12)]]
    def response(value):
        evidence = value['evidence']
        assert evidence['comparison']['mode'] == 'descriptive_accepted_operands'
        claims = {c['claim_id']: c for c in evidence['claims']}
        pairs = evidence['comparison']['pairs']
        assert not any(claims[p['left']]['field'] in ('n_cells', 'n_clusters') for p in pairs)
        pair = next(p for p in pairs if claims[p['left']]['field'] == field)
        assert pair['relationship'] == 'different'
        return dict(support='supported', paragraphs=[dict(
            text=' '.join(evidence['meanings'].values()), kind='finding',
            citations=[pair['left'],pair['right']], comparisons=[pair['id']])])

    guarded = forbid_work(app, monkeypatch)
    before = guarded.sessions.load('session')
    result = ask(guarded, 'compare', 'Compare the current settings with the previous accepted result.',
        Model(question('g0', comparison=dict(output='g1', subject=None)), response=response))
    assert result.status == 'answered', result
    assert {c.source['revision_id'] for c in result.scientific.claims} == set(revisions)
    assert {c.value for c in result.scientific.claims} == {old, new}
    after = guarded.sessions.load('session')
    assert after.revisions == before.revisions and after.generation == before.generation
    if tool == 'compute_cell_umap':
        from agent.application.dialogue_evidence import RESULT_LIMITATIONS
        assert RESULT_LIMITATIONS[tool] in result.scientific.limitations
        assert any('rotation' in limit and 'biological' in limit for limit in result.scientific.limitations)


def test_explicit_older_revision_is_offered_without_navigation(app, monkeypatch):
    revisions = [accepted(app, 'inspect_scATAC', name=f'run{i}')[0] for i in range(4)]
    def decision(value):
        groups = value['dialogue']['result_groups']
        old = next(g for g in groups if g['version'] == 1)
        current = next(g for g in groups if g['version'] == 4)
        return question(current['handle'], comparison=dict(output=old['handle'], subject=None))
    guarded = forbid_work(app, monkeypatch)
    result = ask(guarded, 'history', 'Discuss version 1 and the current result.', Model(decision, ['n_cells']))
    assert result.status == 'answered', result
    assert {c.source['revision_id'] for c in result.scientific.claims} == {revisions[0], revisions[-1]}
    assert guarded.sessions.load('session').active_revision_id == revisions[-1]


def test_model_and_feature_mismatches_allow_only_descriptive_settings(app, monkeypatch):
    for name, species, dimensions in [('human', 'human', 512), ('other', 'other', 256)]:
        accepted(app, 'epizoo_embed_cells', name=name,
                 overrides=dict(species=species, embedding_dim=dimensions))
    def response(value):
        evidence = value['evidence']
        assert evidence['comparison']['mode'] == 'descriptive_accepted_operands'
        claims = {c['claim_id']: c for c in evidence['claims']}
        pairs = [p for p in evidence['comparison']['pairs']
                 if claims[p['left']]['field'] in ('species', 'embedding_dim')]
        assert len(pairs) == 2 and all(p['relationship'] == 'different' for p in pairs)
        assert not any(claims[p['left']]['field'] == 'n_cells' for p in evidence['comparison']['pairs'])
        return dict(support='supported', paragraphs=[dict(
            text=evidence['meanings']['comparison_scope'], kind='finding',
            citations=list(dict.fromkeys(k for p in pairs for k in (p['left'],p['right']))),
            comparisons=[p['id'] for p in pairs])])

    result = ask(forbid_work(app, monkeypatch), 'models', 'Compare the two model outputs.',
        Model(question('g0', comparison=dict(output='g1', subject=None)), response=response))
    assert result.status == 'answered', result
    assert 'aligned coordinates' in result.text


@pytest.mark.parametrize('failure', ['metric', 'one_operand', 'corruption'])
def test_descriptive_comparison_keeps_binding_and_integrity_guards(app, monkeypatch, failure):
    _, path, _ = accepted(app, 'compute_cell_umap', name='first')
    accepted(app, 'compute_cell_umap', name='second')
    def response(value):
        pair = value['evidence']['comparison']['pairs'][0]
        refs, comparisons = [pair['left'],pair['right']], [pair['id']]
        if failure == 'metric': comparisons = ['silhouette']
        elif failure == 'one_operand': refs = [pair['left']]
        else: path.write_bytes(path.read_bytes() + b' ')
        return dict(support='supported', paragraphs=[dict(text='The accepted settings differ.',
            kind='finding', citations=refs, comparisons=comparisons)])

    guarded = forbid_work(app, monkeypatch)
    result = ask(guarded, failure, 'Compare the layouts.',
        Model(question('g0', comparison=dict(output='g1', subject=None)), response=response))
    assert result.status == 'unavailable' and result.scientific is None
    assert len(guarded.sessions.load('session').revisions) == 2


def test_historical_capture_rejects_foreign_or_future_revision(app):
    for i in range(4):
        accepted(app, 'inspect_scATAC', name=f'run{i}')
    result = ask(app, 'explain', 'Explain the current result.', Model(question('g0'), ['n_cells']))
    assert result.status == 'answered'
    state = app.sessions.load('session')
    interaction = state.interactions[-1]
    from agent.schemas.orchestration import _serialize
    data = _serialize(interaction.snapshot)
    data['dialogue']['context']['historical_revisions'].append('f' * 64)
    with pytest.raises(ValueError, match='historical revisions'):
        sd.validate_capture(state, replace(interaction, snapshot=data), state.interactions[:-1])


def test_history_inventory_is_bounded_without_dropping_current_scope(app):
    for i in range(5):
        accepted(app, 'inspect_scATAC', name=f'run{i}', copies=32)
    state = app.sessions.load('session')
    captured = snapshot(app.sessions, state)
    dialogue = sd.capture(app.sessions, state, captured, None)
    assert len(dialogue['outputs']) == 128
    assert {rid for rid in captured['bases']} <= {o['revision_id'] for o in dialogue['outputs']}
    assert state.revisions[0].revision_id not in {o['revision_id'] for o in dialogue['outputs']}
