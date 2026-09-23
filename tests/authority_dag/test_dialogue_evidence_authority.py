"""M16.1 reads real tiny accepted evidence without re-entering its owners."""
from agent.application import ResearchAgentApplication
from test_prior_outputs import source, application_chain, chain, continuation, all_work


def snapshot(app):
    return {str(p.relative_to(app.workspace_root)): p.read_bytes()
            for p in app.workspace_root.rglob('*') if p.is_file()}


def test_real_evidence_retained_sources_and_no_stale_matrix(source):
    _, _, original = source
    first = original.sessions.load('analysis').active_revision_id
    app, _, request, outputs, retain = continuation(source, matrix=False, name='selection-only')
    state = app.sessions.run('analysis', 'selection-only', request, outputs,
        expected_generation=1, use_active_context=True, retain=retain)
    second = state.active_revision_id
    assert second != first
    restarted = ResearchAgentApplication(app.workspace_root)
    before = snapshot(app)
    with all_work() as calls:
        qc = restarted.sessions.evidence('analysis', second, 'qc')
        selection = restarted.sessions.evidence('analysis', second, 'selection')
        absent = restarted.sessions.evidence('analysis', second, 'matrix')
        historical = restarted.sessions.evidence('analysis', first, 'matrix')
    assert not calls, calls
    assert qc.status == selection.status == historical.status == 'available', (qc, selection, historical)
    assert qc.source.output_locator['run_id'] == 'authority-request:run'
    assert selection.source.output_locator['run_id'] == 'selection-only:run'
    assert selection.source.prior_outputs
    assert qc.source.authority_sha256 and historical.source.authority_sha256
    assert qc.is_active and not historical.is_active
    assert absent.status == 'unavailable' and not absent.facts
    assert 'tss_method' in {f.field for f in qc.facts}
    assert 'effective_thresholds' in {f.field for f in selection.facts}
    assert 'matrix_semantics' in {f.field for f in historical.facts}
    assert snapshot(app) == before
