"""Saved historical prose is immutable after response consolidation."""
from agent.application import responses
from dataclasses import replace
from test_interactive_boundary import boundary


def test_old_clarification_stays_exact_on_reopen_and_retry(tmp_path,monkeypatch):
    decision=dict(kind='clarify',reason='invalid_parameter_value')
    service,models=boundary(tmp_path,decision=decision)
    service.create_session('session')
    old_text='Previously recorded clarification text.'
    with monkeypatch.context() as historical:
        historical.setattr(responses,'summarize',lambda s,i,t,o,**kw:replace(o,text=old_text))
        old=service.submit_turn('session','first','Use this value.',expected_generation=0)
    sessions=service._application.sessions
    path=sessions._store._path('session','.json');before=path.read_bytes()
    reopened,calls=boundary(tmp_path,decision=decision,workspace=service._application.workspace_root)
    assert reopened.reopen_session('session').turns[0]==old
    assert reopened.submit_turn('session','first','Use this value.',expected_generation=0)==old
    assert path.read_bytes()==before and not calls
    assert old.response.text==old_text
