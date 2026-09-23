"""Real owner QC publication, read with all scientific work instrumented."""
from agent.application.dialogue_evidence import DetailRequest
from test_prior_outputs import source, application_chain, chain, all_work
from test_dialogue_evidence_authority import snapshot


def test_real_histogram_detail_requires_no_owner_work(source):
    _, _, app = source
    revision = app.sessions.load('analysis').active_revision_id
    before = snapshot(app)
    with all_work() as calls:
        view = app.sessions.evidence('analysis',revision,'qc',detail=DetailRequest('length_histogram',limit=3))
    assert not calls,calls
    assert view.status=='available',view
    assert view.source.authority_sha256
    assert [f.field for f in view.detail]==['length_histogram.1','length_histogram.2','length_histogram.3','length_histogram.records_omitted']
    assert view.detail[-1].value==998
    assert snapshot(app)==before
