"""Safe interaction metadata and exact typed result references; no science."""
from dataclasses import replace
import json

import pytest

from agent.application import SessionConflictError
from agent.application.sessions import AnalysisSessions
from agent.application.session_state import Interaction
from agent.application.turn_context import snapshot, supplied_input_context
from agent.schemas.orchestration import _serialize
from test_dialogue_evidence import app, accepted, forbid_work
from test_scientific_dialogue import Model, annotation, ask, question


def cooutputs(app, monkeypatch, *, names=('zeta', 'alpha'), tool='inspect_scATAC', **kwargs):
    """Select two existing members of one fixture step through normal completion."""
    from agent.application import OutputSelection
    original = AnalysisSessions.start_turn
    def select(self, session_id, turn_id, request, outputs, **options):
        step = outputs[0].step_id
        keys = ('n_cells', 'input_path') if tool == 'inspect_scATAC' else ('n_cells', 'n_groups')
        return original(self, session_id, turn_id, request,
            tuple(OutputSelection(name, step, key) for name, key in zip(names, keys)), **options)
    with monkeypatch.context() as patch:
        patch.setattr(AnalysisSessions, 'start_turn', select)
        return accepted(app, tool, **kwargs)


def pending(app, name, *, scientific=False):
    state = app.sessions.load('session')
    admitted = None
    if scientific:
        locator = state.revisions[-1].outputs[0]
        admitted = dict(kind='answer', intent='scientific', dialogue_version=1,
            target=dict(revision_id=state.active_revision_id, output_name=locator.name,
                accepted_step_sha256=locator.accepted_step_sha256, subject='3'),
            comparison=None, previous_subject=None, focus='Explain cluster 3.', predecessor=None)
    item = Interaction(name, 'Fixture interaction.', state.active_revision_id, state.generation,
        snapshot(app.sessions, state), status='admitted' if scientific else 'submitted', admitted=admitted)
    app.sessions._store._update('session', lambda s: replace(s, interactions=s.interactions+(item,)))


def finish_interaction(app, name, status):
    app.sessions._store._update('session', lambda s: replace(s, interactions=tuple(
        replace(i, status=status) if i.turn_id == name else i for i in s.interactions)))


def role_answer(app, turn, role, *, callback=None, fields=('n_cells',), subject=None):
    def decide(prompt):
        if callback: callback(prompt)
        return question(role, subject)
    model = Model(decide, fields)
    return ask(app, turn, 'Describe the requested accepted result.', model), model


def test_current_inputs_are_safe_registry_descriptors_in_interpretation(app, monkeypatch):
    app = forbid_work(app, monkeypatch)
    inputs = {'input_path': '/private/secret-matrix.h5ad',
        'min_tss_enrichment': 'secret-numeric-value',
        'annotation_inputs': {'private_nested_key': 'private_nested_value'},
        '/private/unknown-name': 'private_unknown_value'}
    def decide(prompt):
        supplied = prompt['dialogue']['supplied_inputs']
        fields = {f['name']: f for f in supplied['fields']}
        assert supplied['present'] and fields['input_path']['json_type'] == 'string'
        assert any(c['tool'] == 'inspect_scATAC' for c in fields['input_path']['consumers'])
        encoded = json.dumps(prompt)
        for private in ('secret-matrix', 'secret-numeric-value', 'private_nested_key',
                        'private_nested_value', '/private/unknown-name', 'private_unknown_value'):
            assert private not in encoded
        assert supplied['omitted_count'] >= 1
        return dict(kind='clarify', reason='unsupported_intent')
    model = Model(decide)
    result = ask(app, 'inputs', 'Discuss the supplied input.', model, execution_inputs=inputs)
    assert result.status == 'clarification' and len(model.calls) == 1
    state = app.sessions.load('session')
    assert state.generation == 0 and not state.revisions
    assert 'supplied_inputs' not in json.dumps(_serialize(state.interactions[0].snapshot))


def test_partial_grouped_sources_remain_partial_without_readiness_claim(app):
    candidates = [(t, p, s) for t in (app.registry.get(n) for n in app.registry.names())
        if t.semantic_planning for p in t.semantic_planning.consumer_ports
        for s in p.request_sources if len(s.members) > 1]
    tool, port, source = next((t, p, s) for t, p, s in candidates
        if len({m.input_name for m in s.members}) > 1)
    first = source.members[0].input_name
    inputs = {first: '/private/source'}
    partial = supplied_input_context(inputs, app.registry)
    consumer = next(c for f in partial['fields'] if f['name'] == first for c in f['consumers']
        if c['tool'] == tool.name and c['port'] == port.name)
    assert consumer['source_complete'] is False
    complete = supplied_input_context({m.input_name: 'private-value' for m in source.members}, app.registry)
    assert any(c['source_complete'] for f in complete['fields'] if f['name'] == first
        for c in f['consumers'] if c['tool'] == tool.name and c['port'] == port.name)
    assert supplied_input_context(None, app.registry) == dict(present=False, fields=[], omitted_count=0)
    assert 'private-value' not in json.dumps(complete) and 'ready' not in consumer


@pytest.mark.parametrize('names', [('zeta', 'alpha'), ('alpha', 'zeta')])
def test_same_step_cooutputs_share_exact_group_with_order_independent_canonical(app, monkeypatch, names):
    rid, _, _ = cooutputs(app, monkeypatch, names=names)
    app = forbid_work(app, monkeypatch)
    def decide(prompt):
        dialogue = prompt['dialogue']
        group, = dialogue['result_groups']
        outputs = {o['handle']: o for o in dialogue['outputs']}
        assert len(group['outputs']) == 2
        assert outputs[group['canonical_output']]['output_name'] == 'alpha'
        assert dialogue['result_referents']['@current_result']['status'] == 'available'
        return question(group['handle'])
    result = ask(app, 'group', 'Explain this result.', Model(decide, ['n_cells']))
    assert result.status == 'answered'
    claim, = result.scientific.claims
    assert claim.source['revision_id'] == rid and claim.source['output_locator']['name'] == 'alpha'
    assert app.sessions.load('session').generation == 1


def test_group_alias_evidence_mismatch_fails_closed(app, monkeypatch):
    cooutputs(app, monkeypatch)
    app = forbid_work(app, monkeypatch)
    original = AnalysisSessions.evidence
    def changed(self, session_id, revision_id, output_name, **kwargs):
        view = original(self, session_id, revision_id, output_name, **kwargs)
        return replace(view, facts=tuple(replace(f, value=999) if f.field == 'n_cells' else f
            for f in view.facts)) if output_name == 'zeta' else view
    monkeypatch.setattr(AnalysisSessions, 'evidence', changed)
    model = Model(question('g0'))
    result = ask(app, 'mismatch', 'Explain this result.', model)
    assert result.status == 'clarification' and result.clarification.reason == 'unavailable_context'
    assert not model.calls and result.scientific is None


@pytest.mark.parametrize('role', ['@current_result', '@most_recently_created'])
def test_independent_steps_remain_ambiguous(app, monkeypatch, role):
    accepted(app, 'inspect_scATAC', copies=2)
    app = forbid_work(app, monkeypatch)
    def check(prompt):
        assert len(prompt['dialogue']['result_groups']) == 2
        assert prompt['dialogue']['result_referents'][role]['status'] == 'ambiguous'
    result, model = role_answer(app, 'ambiguous', role, callback=check)
    assert result.status == 'clarification' and result.clarification.reason == 'ambiguous_subject'
    assert len(model.calls) == 1


def test_current_and_latest_creation_remain_distinct_after_rollback(app, monkeypatch):
    old = accepted(app, 'inspect_scATAC', overrides={'n_cells': 2})[0]
    newest = accepted(app, 'inspect_scATAC', name='newer', overrides={'n_cells': 5})[0]
    app.sessions.switch('session', 'rollback', old, expected_generation=2)
    app = forbid_work(app, monkeypatch)
    current, _ = role_answer(app, 'current', '@current_result')
    latest, _ = role_answer(app, 'latest', '@most_recently_created')
    assert current.scientific.claims[0].source['revision_id'] == old
    assert latest.scientific.claims[0].source['revision_id'] == newest
    assert latest.scientific.claims[0].value == 5 and 'historical' in latest.text
    assert app.sessions.load('session').active_revision_id == old


def test_creation_scope_excludes_retained_old_run_outputs(app, monkeypatch):
    accepted(app, 'compute_scATAC_qc')
    pending(app, 'selection')
    original = AnalysisSessions.start_turn
    from agent.application import OutputSelection
    def retaining(self, session_id, turn_id, request, outputs, **options):
        return original(self, session_id, turn_id, request,
            (OutputSelection('selection', outputs[0].step_id, outputs[0].output_key),),
            **dict(options, retain=('output0',)))
    with monkeypatch.context() as patch:
        patch.setattr(AnalysisSessions, 'start_turn', retaining)
        # The accepted() helper records scripted metadata without reusable
        # scientific authority. This test checks the read-only result scope;
        # executable retained-output authority is covered by its owner suite.
        from agent.orchestration import prior_outputs
        patch.setattr(prior_outputs, 'validate_active_outputs', lambda *args: None)
        rid = accepted(app, 'select_scATAC_cells', name='selection',
            arguments={'min_qc_fragment_records': 1, 'min_tss_enrichment': '4'})[0]
    app = forbid_work(app, monkeypatch)
    def check(prompt):
        roles = prompt['dialogue']['result_referents']
        assert roles['@current_result']['status'] == 'ambiguous'
        assert roles['@most_recently_created']['status'] == 'available'
        assert roles['@previous_turn_result']['status'] == 'available'
    result, _ = role_answer(app, 'created', '@previous_turn_result', callback=check, fields=('n_selected',))
    assert result.status == 'answered'
    assert result.scientific.claims[0].source['revision_id'] == rid
    assert result.scientific.claims[0].source['output_locator']['name'] == 'selection'


def test_previous_answer_preserves_subject_and_comparison_ambiguity(app, monkeypatch):
    annotation(app)
    app = forbid_work(app, monkeypatch)
    assert ask(app, 'first', 'Explain cluster 3.', Model(question(subject='3'), ['primary_annotation'])).status == 'answered'
    follow, _ = role_answer(app, 'previous', '@previous_turn_result', fields=('primary_annotation',))
    assert follow.scientific.claims[0].subject == '3'
    comparison = question('r0', '3', comparison=dict(output='r0', subject='7'))
    assert ask(app, 'pair', 'Compare cluster 3 and cluster 7.', Model(comparison, ['primary_annotation'])).status == 'answered'
    result, model = role_answer(app, 'ambiguous', '@previous_turn_result')
    assert result.status == 'clarification' and len(model.calls) == 1


def test_modern_canonical_alias_retains_focus_and_other_subject(app, monkeypatch):
    values = dict(n_groups=2, groups_omitted=0, validation_state='not_assessed', group_summary=[
        dict(group='3', n_cells=8, primary_annotation='CD8 T', status='assigned'),
        dict(group='7', n_cells=5, primary_annotation='B cell', status='assigned')])
    cooutputs(app, monkeypatch, tool='annotate_scATAC_cell_types', overrides=values)
    app = forbid_work(app, monkeypatch)
    assert ask(app, 'legacy', 'Explain zeta for cluster 3.', Model(question('r0', '3'), ['primary_annotation'])).status == 'answered'
    for turn, subject, expected in [('focus', '@focus', '3'), ('other', '@other', '7')]:
        result = ask(app, turn, 'Continue this discussion.', Model(question('g0', subject), ['primary_annotation']))
        assert result.status == 'answered' and result.scientific.claims[0].subject == expected
        assert result.scientific.claims[0].source['output_locator']['name'] == 'alpha'


@pytest.mark.parametrize('prior_kind', ['pending_answer', 'created_then_failed', 'already_failed', 'late_creation'])
def test_previous_result_availability_is_pinned_across_completion(app, monkeypatch, prior_kind):
    if prior_kind == 'pending_answer':
        annotation(app)
        pending(app, 'prior', scientific=True)
    else:
        accepted(app, 'inspect_scATAC')
        pending(app, 'prior')
        if prior_kind in ('created_then_failed', 'already_failed'):
            accepted(app, 'inspect_scATAC', name='prior')
        if prior_kind == 'already_failed': finish_interaction(app, 'prior', 'failed')
    app = forbid_work(app, monkeypatch)
    available = prior_kind == 'created_then_failed'
    def complete(prompt):
        assert prompt['dialogue']['result_referents']['@previous_turn_result']['status'] == (
            'available' if available else 'unavailable')
        if prior_kind == 'pending_answer': finish_interaction(app, 'prior', 'answered')
        elif prior_kind == 'created_then_failed': finish_interaction(app, 'prior', 'failed')
        elif prior_kind == 'late_creation': accepted(app, 'inspect_scATAC', name='prior')
    result, _ = role_answer(app, 'captured', '@previous_turn_result', callback=complete)
    assert result.status == ('answered' if available else 'clarification')
    restored = app.sessions.load('session')
    context = restored.interactions[-1].snapshot['dialogue']['context']
    assert context['previous_created_result'] is available
    assert context['previous_scientific_answer'] is False


def test_unrelated_previous_turn_does_not_scan_back_and_stale_unknown_fail(app, monkeypatch):
    accepted(app, 'inspect_scATAC')
    app = forbid_work(app, monkeypatch)
    assert ask(app, 'legacy', 'Explain this result.', Model(question(), ['n_cells'])).status == 'answered'
    ask(app, 'unrelated', 'Discuss something else.', Model(dict(kind='clarify', reason='unsupported_intent')))
    missing, _ = role_answer(app, 'previous', '@previous_turn_result')
    assert missing.status == 'clarification' and missing.clarification.reason == 'unavailable_context'
    unknown, _ = role_answer(app, 'unknown', 'g99')
    assert unknown.status == 'clarification'
    model = Model(question('@current_result'))
    with pytest.raises(SessionConflictError):
        ask(app, 'stale', 'Explain this result.', model, expected_generation=0)
    assert not model.calls
    def forbidden(*args, **kwargs): raise AssertionError('retry called a provider')
    exact = Model(question())
    result = ask(app, 'retry', 'Explain this result.', exact)
    from agent.application.interactive import _present
    shown = _present(result, app.sessions.load('session'), turn_id='retry')
    app.sessions.store_presentation('session', 'retry', shown.to_dict())
    exact.complete = forbidden
    assert ask(app, 'retry', 'Explain this result.', exact).text == result.text


def test_persisted_legacy_dialogue_snapshot_load_and_display_retry_do_no_work(app, monkeypatch, tmp_path):
    from shutil import copytree
    from agent.application import ResearchAgentApplication
    from agent.application.interactive import _present
    from agent.application.session_state import canonical, digest

    accepted(app, 'inspect_scATAC')
    app = forbid_work(app, monkeypatch)
    result = ask(app, 'historical', 'Explain this result.', Model(question(), ['n_cells']))
    shown = _present(result, app.sessions.load('session'), turn_id='historical')
    app.sessions.store_presentation('session', 'historical', shown.to_dict())
    original_path = app.sessions._store._path('session', '.json')
    original_bytes = original_path.read_bytes()
    copied_root = tmp_path / 'legacy-workspace'
    copytree(app.workspace_root, copied_root)
    copied_path = copied_root / original_path.relative_to(app.workspace_root)
    envelope = json.loads(copied_path.read_bytes())
    dialogue = envelope['record']['interactions'][0]['snapshot']['dialogue']
    del dialogue['context']
    assert set(dialogue) == {'outputs', 'predecessor', 'requested_predecessor', 'ambiguous'}
    envelope['sha256'] = digest(envelope['record'])
    copied_path.write_bytes(canonical(envelope))
    legacy_bytes = copied_path.read_bytes()
    legacy = ResearchAgentApplication(copied_root, registry=app.registry)
    restored = legacy.sessions.load('session')
    assert 'context' not in restored.interactions[0].snapshot['dialogue']
    assert _serialize(restored.interactions[0].presentation) == shown.to_dict()
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy display retry entered evidence or a provider')
    monkeypatch.setattr(AnalysisSessions, 'evidence', forbidden)
    model = Model(forbidden)
    retry = ask(legacy, 'historical', 'Explain this result.', model)
    assert retry.presentation == shown and retry.text == result.text and not model.calls
    assert copied_path.read_bytes() == legacy_bytes and original_path.read_bytes() == original_bytes


@pytest.mark.parametrize('racing_change', ['turn_only', 'stale_revision'])
def test_capture_conflicts_with_metadata_completion_without_generation_change(app, monkeypatch, racing_change):
    from agent.application import turns

    accepted(app, 'inspect_scATAC')
    if racing_change == 'stale_revision':
        accepted(app, 'inspect_scATAC', name='waiting', activate=False)
        accepted(app, 'inspect_scATAC', name='newer')
    app = forbid_work(app, monkeypatch)
    before = app.sessions.load('session')
    original_snapshot = turns.snapshot
    def racing_snapshot(sessions, state, **kwargs):
        captured = original_snapshot(sessions, state, **kwargs)
        if racing_change == 'turn_only':
            changed = sessions.record_no_run_turn('session', 'racing', outcome='clarification',
                expected_generation=state.generation)
        else:
            changed = sessions.recover('session', 'waiting')
            assert changed.turn('waiting').status == 'stale'
        assert changed.generation == state.generation and changed.interactions == state.interactions
        assert changed.turns != state.turns
        assert (changed.revisions != state.revisions) == (racing_change == 'stale_revision')
        return captured
    monkeypatch.setattr(turns, 'snapshot', racing_snapshot)
    model = Model(question('@current_result'))
    with pytest.raises(SessionConflictError, match='Interaction capture raced'):
        ask(app, 'captured', 'Explain this result.', model)
    after = app.sessions.load('session')
    assert after.generation == before.generation and after.interactions == before.interactions
    assert not model.calls
