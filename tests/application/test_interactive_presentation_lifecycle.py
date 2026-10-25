"""OPT-2E: synchronized publication windows over the actual interactive owner.

Tiny fixtures establish software lifecycle behavior, not biological validity.
No paid provider or preserved scientific workspace is involved.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Event

import pytest

from agent.application import InteractiveBoundaryError
from agent.application.local_resources import LocalResourceAdmission
from agent.application.sessions import AnalysisSessions
from agent.orchestration import ToolRegistry
from test_dialogue_evidence import app, accepted
from test_interactive_boundary import ScriptedModel, boundary, first
from test_scientific_dialogue import question
from test_service import _counting_registry, _tiny_h5ad


def pause_publication(monkeypatch):
    entered, release, committed = Event(), Event(), Event()
    original = AnalysisSessions.store_presentation

    def blocked(sessions, session_id, turn_id, presentation):
        entered.set()
        assert release.wait(20), 'Test publication boundary was not released'
        result = original(sessions, session_id, turn_id, presentation)
        committed.set()
        return result

    monkeypatch.setattr(AnalysisSessions, 'store_presentation', blocked)
    return entered, release, committed


def test_accepted_science_waits_for_immutable_response_without_historical_error(tmp_path, monkeypatch):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    calls = []
    service, models = boundary(tmp_path, registry=_counting_registry(calls))
    service.create_session('session')
    service.resources = LocalResourceAdmission(service._application._workspace,
        approved_source_roots=(tmp_path,), registry=service._application.registry)
    record = service.resources.register('matrix', source, label='Tiny input', attribution='Simulated fixture',
        session_id='session')
    binding = service.resources.resolve(record.resource_id, tool_name='inspect_scATAC')

    def submit_request():
        return service.submit_turn('session', 'first', 'Inspect the supplied matrix.', expected_generation=0,
            registered_input=binding, attachment_ids=(record.resource_id,))

    entered, release, _ = pause_publication(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(submit_request)
        try:
            assert entered.wait(20), 'Execution did not reach response publication'
            state = service._application.sessions.load('session')
            assert state.turn('first').status == 'activated' and len(state.revisions) == 1
            assert state.interactions[-1].presentation is None
            pending = service.status('session', 'first')
            assert pending.status == 'finalizing'
            assert pending.response is None and pending.error is None
            assert pending.revision_id == state.active_revision_id
            assert pending.attachments[0]['use'] == 'consumed'
            assert service.revision('session', pending.revision_id).revision_id == pending.revision_id
            assert service.reopen_session('session').turns[-1] == pending
            assert submit_request() == pending  # A duplicate reads, never creates another worker.
            assert all(service.status('session', 'first') == pending for _ in range(3))
            assert calls == ['inspect_scATAC'] and len(models) == 1
        finally:
            release.set()
        completed = future.result(timeout=20)
    assert completed.status == 'succeeded' and completed.response is not None
    assert service.status('session', 'first') == completed


@pytest.mark.parametrize('phase', ['dialogue_schema_version', 'grounding_review_version', 'publication'])
def test_read_only_answer_generation_review_and_commit_remain_pending(app, tmp_path, monkeypatch, phase):
    accepted(app, 'inspect_scATAC')
    service, models = boundary(tmp_path, workspace=app.workspace_root, decision=question())
    before = app.sessions.load('session')
    entered, release = Event(), Event()
    if phase == 'publication':
        entered, release, _ = pause_publication(monkeypatch)
    else:
        original = ScriptedModel.complete

        def blocked(model, *, prompt, response_schema):
            if phase in json.loads(prompt):
                entered.set()
                assert release.wait(20), 'Test model boundary was not released'
            return original(model, prompt=prompt, response_schema=response_schema)

        monkeypatch.setattr(ScriptedModel, 'complete', blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(service.submit_turn, 'session', 'discuss', 'Explain this result.', expected_generation=1)
        try:
            assert entered.wait(20)
            pending = service.status('session', 'discuss')
            assert pending.status == 'finalizing' and pending.response is None and pending.error is None
            assert pending.run_id is None and pending.revision_id is None
            assert service.reopen_session('session').turns[-1] == pending
            assert service.submit_turn('session', 'discuss', 'Explain this result.', expected_generation=1) == pending
            current = app.sessions.load('session')
            assert (current.turns, current.revisions, current.generation) == (before.turns, before.revisions, before.generation)
        finally:
            release.set()
        completed = future.result(timeout=20)
    assert completed.status == 'answered' and completed.response.kind == 'answer'
    assert service.status('session', 'discuss') == completed
    assert len(models) == 1 and len(models[0].calls) == 2


def test_figure_bytes_wait_for_committed_response_even_when_r5_withholds(tmp_path, monkeypatch):
    from test_interaction_figure_delivery import case, submit, unchanged_science, restarted

    current = case(tmp_path, monkeypatch)
    entered, release, _ = pause_publication(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(submit, current)
        try:
            assert entered.wait(20)
            assert len(current.rendered) == 1
            pending = current.facade.status('session', 'draw')
            assert pending.status == 'finalizing' and pending.error is None and pending.response is None
            assert current.facade.presentation_artifact_handles('session', 'draw') == ()
            assert current.models[0].phases == ['interpretation', 'generation', 'review']
            unchanged_science(current)
        finally:
            release.set()
        completed = future.result(timeout=20)
    assert 'could not ground' in completed.response.text
    reference, = current.facade.presentation_artifact_handles('session', 'draw')
    content = current.facade.presentation_artifact_content('session', 'draw', reference.handle)
    restored = restarted(current)
    assert restored.turn('session', 'draw') == completed
    assert restored.presentation_artifact_content('session', 'draw', reference.handle).data == content.data
    assert len(current.rendered) == 1 and len(current.models) == 1
    unchanged_science(current)


def test_qualified_export_is_not_announced_before_response_publication(app, monkeypatch):
    from test_composed_export_interaction import case, unchanged, REQUEST

    current = case(app, monkeypatch)
    entered, release, _ = pause_publication(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(current.facade.submit_turn, 'session', 'compose', REQUEST, expected_generation=2)
        try:
            assert entered.wait(20)
            pending = current.facade.status('session', 'compose')
            assert pending.status == 'finalizing' and pending.error is None and pending.response is None
            assert pending.export_requested is True
            assert current.facade.scientific_artifacts_for_turn('session', 'compose') == ()
            assert current.models[0].phases == ['interpretation', 'generation', 'review']
            unchanged(current)
        finally:
            release.set()
        completed = future.result(timeout=20)
    assert 'available for preparation' in completed.response.text.lower()
    assert len(current.facade.scientific_artifacts_for_turn('session', 'compose')) == 1
    assert current.facade.status('session', 'compose') == completed
    unchanged(current)


@pytest.mark.parametrize('observed_active', [False, True])
def test_response_published_between_snapshot_and_lease_probe_is_visible(tmp_path, monkeypatch, observed_active):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    calls = []
    service, models = boundary(tmp_path, registry=_counting_registry(calls))
    service.create_session('session')
    entered, release, committed = pause_publication(monkeypatch)
    store = service._application.sessions._store
    original = store.is_processing
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first, service, source)
        try:
            assert entered.wait(20)
            probes = []

            def publish_after_observation(session_id, turn_id):
                assert original(session_id, turn_id)
                probes.append((session_id, turn_id))
                release.set()
                assert committed.wait(20)
                future.result(timeout=20)  # The real lease has now released.
                return observed_active

            monkeypatch.setattr(store, 'is_processing', publish_after_observation)
            observed = service.status('session', 'first')
            assert observed.response is not None and observed.error is None
            assert observed == future.result(timeout=20)
            assert probes == [('session', 'first')]
        finally:
            release.set()
    assert calls == ['inspect_scATAC'] and len(models) == 1


@pytest.mark.parametrize('scientific_execution', [False, True])
def test_lost_publication_is_interrupted_and_only_explicit_recovery_closes_it(app, tmp_path, monkeypatch,
        scientific_execution):
    calls = []
    if scientific_execution:
        service, models = boundary(tmp_path, registry=_counting_registry(calls),
            workspace=tmp_path / 'execution-workspace')
        service.create_session('session')
        source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
        submit = lambda: first(service, source)
        turn_id = 'first'
    else:
        accepted(app, 'inspect_scATAC')
        service, models = boundary(tmp_path, workspace=app.workspace_root, decision=question())
        submit = lambda: service.submit_turn('session', 'discuss', 'Explain this result.', expected_generation=1)
        turn_id = 'discuss'
    with monkeypatch.context() as failing:
        def lost(*args, **kwargs):
            raise RuntimeError('Test interruption immediately before saved response')

        failing.setattr(AnalysisSessions, 'store_presentation', lost)
        with pytest.raises(InteractiveBoundaryError):
            submit()
    path = service._application.sessions._store._path('session', '.json')
    unchanged_bytes = path.read_bytes()
    for _ in range(3):
        interrupted = service.status('session', turn_id)
        assert interrupted.status == 'interrupted' and interrupted.response is None
        assert interrupted.error.code == 'INTERACTIVE_PROCESSING_INTERRUPTED'
    assert path.read_bytes() == unchanged_bytes
    assert submit() == interrupted
    recovered = service.recover_turn('session', turn_id)
    assert recovered.response is not None
    assert service.recover_turn('session', turn_id) == recovered
    assert submit() == recovered
    assert len(models) == 1
    assert calls == (['inspect_scATAC'] if scientific_execution else [])
    if scientific_execution:
        assert recovered.revision_id == interrupted.revision_id and recovered.status == 'succeeded'
        assert service._application.sessions.load('session').turn(turn_id).status == 'activated'
    else:
        assert recovered.response.status == 'unavailable'


def test_real_run_failure_remains_visible_during_publication_and_after_interruption(tmp_path, monkeypatch):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    counted = _counting_registry([])

    def failed_tool(**arguments):
        raise RuntimeError('Scripted scientific failure')

    registry = ToolRegistry(tuple(replace(counted.get(name), function=failed_tool)
        if name == 'inspect_scATAC' else counted.get(name) for name in counted.names()))
    service, models = boundary(tmp_path, registry=registry)
    service.create_session('session')
    entered, release = Event(), Event()

    def lost_after_pause(*args, **kwargs):
        entered.set()
        assert release.wait(20)
        raise RuntimeError('Test interruption after scientific failure')

    with monkeypatch.context() as publishing:
        publishing.setattr(AnalysisSessions, 'store_presentation', lost_after_pause)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(first, service, source)
            try:
                assert entered.wait(20)
                turn = service._application.sessions.load('session').turn('first')
                run = service._application.run_store.load(turn.run_id)
                pending = service.status('session', 'first')
                assert run.errors and pending.error.code == run.errors[0].code
                assert pending.status == 'finalizing' and pending.response is None
            finally:
                release.set()
            with pytest.raises(InteractiveBoundaryError):
                future.result(timeout=20)
    interrupted = service.status('session', 'first')
    assert interrupted.status == 'interrupted' and interrupted.error == pending.error
    recovered = service.recover_turn('session', 'first')
    assert recovered.response is not None and recovered.error.code == pending.error.code
    assert len(models) == 1


def test_missing_accepted_run_is_not_masked_by_live_presentation_lease(tmp_path, monkeypatch):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    service, models = boundary(tmp_path)
    service.create_session('session')
    entered, release, _ = pause_publication(monkeypatch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first, service, source)
        try:
            assert entered.wait(20)
            turn = service._application.sessions.load('session').turn('first')
            path = service._application.run_store.state_path(turn.run_id)
            original = path.read_bytes()
            path.unlink()
            try:
                missing = service.status('session', 'first')
                assert missing.status == 'unavailable' and missing.response is None
                assert missing.error.code == 'INTERACTIVE_REFERENCE_INVALID'
            finally:
                path.write_bytes(original)
        finally:
            release.set()
        assert future.result(timeout=20).response is not None
    assert len(models) == 1


def test_saved_figure_response_is_visible_while_publisher_still_holds_lease(tmp_path, monkeypatch):
    from test_interaction_figure_delivery import case, submit, unchanged_science

    current = case(tmp_path, monkeypatch)
    committed, release = Event(), Event()
    original = AnalysisSessions.store_presentation

    def after_commit(sessions, session_id, turn_id, presentation):
        result = original(sessions, session_id, turn_id, presentation)
        committed.set()
        assert release.wait(20)
        return result

    monkeypatch.setattr(AnalysisSessions, 'store_presentation', after_commit)
    with ThreadPoolExecutor(max_workers=1) as worker:
        future = worker.submit(submit, current)
        try:
            assert committed.wait(20)
            facade = current.facade
            assert facade._application.sessions._store.is_processing('session', 'draw')
            with ThreadPoolExecutor(max_workers=3) as readers:
                reads = [readers.submit(facade.status, 'session', 'draw'),
                         readers.submit(facade.turn, 'session', 'draw'), readers.submit(submit, current)]
                views = [read.result(timeout=20) for read in reads]
            assert all(view == views[0] for view in views)
            assert views[0].status == 'answered' and views[0].response is not None and views[0].error is None
            reference, = facade.presentation_artifact_handles('session', 'draw')
            delivered = facade.presentation_artifact_content('session', 'draw', reference.handle).data
            assert delivered.startswith(b'\x89PNG\r\n\x1a\n')
            assert facade.reopen_session('session').turns[-1] == views[0]
            assert len(current.rendered) == 1 and current.models[0].phases == ['interpretation', 'generation', 'review']
            unchanged_science(current)
        finally:
            release.set()
        assert future.result(timeout=20) == views[0]


def test_reopen_session_metadata_matches_turn_that_completes_during_projection(tmp_path, monkeypatch):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    planning, release, intercepted = Event(), Event(), Event()
    calls = []
    service, models = boundary(tmp_path, registry=_counting_registry(calls), started=planning, release=release)
    service.create_session('session')
    original = service._view
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first, service, source)
        try:
            assert planning.wait(20)

            def finish_during_projection(state, turn_id):
                if not intercepted.is_set():
                    intercepted.set()
                    assert state.generation == 0 and state.active_revision_id is None and not state.revisions
                    release.set()
                    future.result(timeout=20)
                return original(state, turn_id)

            monkeypatch.setattr(service, '_view', finish_during_projection)
            reopened = service.reopen_session('session')
            completed = future.result(timeout=20)
            assert reopened.generation == 1 and reopened.active_revision_id == completed.revision_id
            assert {revision.revision_id for revision in reopened.revisions} == {completed.revision_id}
            assert reopened.turns[-1] == completed
            assert len(models) == 1 and calls == ['inspect_scATAC']
        finally:
            release.set()
