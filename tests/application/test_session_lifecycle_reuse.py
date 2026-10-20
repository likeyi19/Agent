"""R7: exact-byte reuse, interruption visibility and single-owner clarification."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import pytest

from agent.application.session_state import AnalysisSession, Interaction, SessionError, canonical, digest
from agent.application.session_store import FileSessionStore
from test_interactive_boundary import boundary, first
from test_service import _tiny_h5ad, _counting_registry


def test_exact_byte_reuse_checks_changes_even_with_restored_stat(tmp_path, monkeypatch):
    store = FileSessionStore(tmp_path)
    original = store.create('s')
    path = store._path('s', '.json')
    raw, stat = path.read_bytes(), path.stat()
    def forbidden(*args, **kwargs):
        pytest.fail('Unchanged snapshot was decoded again')
    with monkeypatch.context() as m:
        m.setattr(AnalysisSession, 'from_dict', forbidden)
        assert store.load('s') is original
        assert store.load('s') is original
    # Exact bytes, not size/mtime, are the reuse boundary.
    changed = raw.replace(b'"generation":0', b'"generation":1')
    assert len(changed) == len(raw)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SessionError, match='checksum'):
        store.load('s')
    path.write_bytes(raw)
    assert store.load('s') == original
    assert FileSessionStore(tmp_path).load('s') == original


def test_other_writer_and_recomputed_checksum_still_validate_history(tmp_path):
    first_store = FileSessionStore(tmp_path)
    first_store.create('s')
    other = FileSessionStore(tmp_path)
    interaction = Interaction('t', 'Hello', None, 0, {'relations': {}, 'bases': {}},
                              status='clarification', admitted={'kind': 'clarify', 'reason': 'unsupported_intent'})
    other._update('s', lambda s: replace(s, interactions=(interaction,)))
    assert first_store.load('s').interactions == (interaction,)
    path = first_store._path('s', '.json')
    envelope = json.loads(path.read_bytes())
    envelope['record']['interactions'][0]['base_generation'] = 1
    envelope['sha256'] = digest(envelope['record'])
    path.write_bytes(canonical(envelope))
    with pytest.raises(SessionError):
        first_store.load('s')


def test_long_history_roundtrip_and_bounded_reuse(tmp_path):
    store = FileSessionStore(tmp_path)
    store.create('s')
    rows = tuple(Interaction(str(i), 'Question ' + str(i), None, 0,
        {'relations': {}, 'bases': {}}, status='clarification',
        admitted={'kind': 'clarify', 'reason': 'unsupported_intent'}) for i in range(1000))
    state = store._update('s', lambda s: replace(s, interactions=rows))
    assert not state.turns and not state.revisions
    assert FileSessionStore(tmp_path).load('s') == state
    for _ in range(10):
        assert store.load('s') is state
    # The store holds one snapshot, regardless of conversation count.
    store.create('other')
    assert store._cached[0] == 'other'
    assert store.load('s') == state


def test_dead_interpreter_read_and_explicit_close_do_not_call_model(tmp_path):
    service, models = boundary(tmp_path)
    service.create_session('s')
    sessions = service._application.sessions
    row = Interaction('lost', 'Inspect my data.', None, 0, {'relations': {}, 'bases': {}},
                      submission={'profile_id': 'alpha'})
    sessions._store._update('s', lambda s: replace(s, interactions=(row,)))
    path = sessions._store._path('s', '.json')
    before = path.read_bytes()
    assert service.status('s', 'lost').status == 'interrupted'
    assert path.read_bytes() == before
    with sessions.processing_lease('s', 'lost'):
        assert service.status('s', 'lost').status == 'interpreting'
        code = '''
import sys
from agent.application.session_store import FileSessionStore
print(FileSessionStore(sys.argv[1]).is_processing('s', 'lost'))
'''
        result = subprocess.run([sys.executable, '-c', code, str(sessions._store.root)],
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == 'True'
    assert not sessions._store.is_processing('s', 'lost')
    result = service.recover_turn('s', 'lost')
    assert result.response.status == 'unavailable'
    assert sessions.load('s').interactions[0].status == 'failed'
    assert not sessions.load('s').turns and not models
    assert service.recover_turn('s', 'lost') == result


def test_completed_retry_never_discovers_unrelated_shared_models(tmp_path, monkeypatch):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    calls = []
    service, models = boundary(tmp_path, registry=_counting_registry(calls))
    service.create_session('session')
    accepted = first(service, source)
    def forbidden(self):
        pytest.fail('Historical retry rediscovered shared models')
    monkeypatch.setattr(type(service), '_epizoo_resources', property(forbidden))
    assert first(service, source) == accepted
    service.validate_submission('session', 'first', 'Inspect the supplied matrix.',
        expected_generation=0, execution_inputs={'input_path': str(source)})
    assert calls == ['inspect_scATAC'] and len(models) == 1


def test_history_cursor_reaches_old_messages_without_execution(tmp_path):
    service, models = boundary(tmp_path)
    service.create_session('s')
    sessions = service._application.sessions
    rows = tuple(Interaction(str(i), 'Question ' + str(i), None, 0,
        {'relations': {}, 'bases': {}}, status='clarification',
        admitted={'kind': 'clarify', 'reason': 'unsupported_intent'}) for i in range(205))
    sessions._store._update('s', lambda s: replace(s, interactions=rows))
    before = sessions._store._path('s', '.json').read_bytes()
    pages = [service.reopen_session('s')]
    while pages[-1].history_truncated:
        pages.append(service.reopen_session('s', before_turn_id=pages[-1].turns[0].turn_id))
    assert [[t.turn_id for t in p.turns] for p in pages] == [
        [str(i) for i in range(105, 205)], [str(i) for i in range(5, 105)], [str(i) for i in range(5)]]
    assert not models and sessions._store._path('s', '.json').read_bytes() == before
    assert service._application.sessions is sessions
    from agent.application import InteractiveBoundaryError
    with pytest.raises(InteractiveBoundaryError):
        service.reopen_session('s', before_turn_id='unknown')


def test_bounded_revision_summary_keeps_selected_historical_result(tmp_path):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    service, _ = boundary(tmp_path)
    service.create_session('session')
    original = first(service, source)
    newer = service.submit_turn('session', 'second', 'Inspect the supplied matrix.',
        expected_generation=1, execution_inputs={'input_path': str(source)})
    assert newer.revision_id != original.revision_id
    service.activate_revision('session', 'back', original.revision_id, expected_generation=2)
    page = service.reopen_session('session', limit=1)
    assert page.active_revision_id == original.revision_id
    assert len(page.revisions) == 1 and page.revisions[0].is_active


def test_legacy_clarification_checkpoint_can_finish_missing_display(tmp_path):
    from agent.application.session_state import SessionTurn
    service, models = boundary(tmp_path)
    service.create_session('s')
    sessions = service._application.sessions
    row = Interaction('legacy', 'Old clarification.', None, 0, {'relations': {}, 'bases': {}},
        status='clarification', admitted={'kind': 'clarify', 'reason': 'unsupported_intent'},
        submission={'profile_id': 'alpha'})
    checkpoint = SessionTurn('legacy', None, 0, None, None, None, 'clarification')
    sessions._store._update('s', lambda s: replace(s, interactions=(row,), turns=(checkpoint,)))
    result = service.recover_turn('s', 'legacy')
    assert result.response is not None and result.response.status == 'unavailable'
    assert sessions.load('s').turns == (checkpoint,)
    assert sessions.load('s').interactions[0].status == 'clarification' and not models
    assert service.recover_turn('s', 'legacy') == result
