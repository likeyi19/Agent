"""Contracts for rich state controls without a frontend build dependency."""
from html.parser import HTMLParser
from pathlib import Path
import re


STATIC = Path(__file__).resolve().parents[2] / 'src' / 'agent' / 'web' / 'static'


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.tags = []
        self.feed((STATIC / 'index.html').read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if 'id' in attrs:
            self.ids.add(attrs['id'])


def code():
    return (STATIC / 'app.js').read_text()


def test_scientific_state_stays_separate_from_conversation():
    page = Page()
    assert {'conversation-history', 'scientific-heading', 'revision-view',
            'result-choice', 'result-methods', 'result-panel',
            'evidence-panel', 'artifact-panel'} <= page.ids
    assert any(tag == 'aside' and attrs.get('aria-labelledby') == 'scientific-heading'
               for tag, attrs in page.tags)
    assert 'turn.utterance || turn.response' in code()
    assert '(turn.utterance || turn.response || turn.rerun_revision_id) && !terminal(turn)' in code()


def test_revision_browsing_is_read_only_and_has_one_selector():
    source = code()
    assert 'api(revisionPath(sessionId, revisionId))' in source
    viewing = source[source.index('async function loadRevision'):source.index('async function loadRevision') + 1200]
    assert '/activate' not in viewing
    continuation = source[source.index('async function continueFromResult'):source.index('async function openResultParameters')]
    assert '/activate`, "POST"' in continuation
    assert 'element("utterance").value = ""' in continuation
    assert 'postSubmission' not in continuation
    assert 'Continue from this result.' not in source
    assert 'result-selection' not in Page().ids


def test_guidance_uses_persisted_identity_and_semantic_turn_submission():
    source = code()
    guidance = source[source.index('function guidanceNode'):source.index('function textList')]
    assert 'card.dataset.candidateId = candidate.candidate_id' in guidance
    assert 'candidate.option' not in guidance and 'candidate.origin_turn_id' in guidance
    assert 'candidate.text' in guidance and 'candidate.limitations' in guidance
    assert 'candidate.readiness' in guidance and 'candidate.base_revision_id' in guidance
    assert 'origin_turn_id: candidate.origin_turn_id, candidate_id: candidate.candidate_id' in guidance
    assert 'Run option' not in guidance
    assert 'Readiness metadata was not stored for this historical candidate.' in guidance
    assert 'fetch(' not in guidance and 'api(' not in guidance
    assert 'body.predecessor_turn_id = predecessorTurnId' in source
    request = source[source.index('function submitRequest'):source.index('element("turn-form").addEventListener')]
    assert 'candidate_id' not in request and 'capability' not in request


def test_evidence_and_supported_detail_use_the_same_bounded_owner_endpoint():
    source = code()
    assert '}/evidence?${query}' in source
    assert 'new URLSearchParams({ output_name:' in source
    assert 'query.set("detail_section", element("detail-section").value)' in source
    assert 'view.supported_details.map' in source
    assert 'source.verification_checks' in source
    assert 'source.prior_outputs' in source
    assert 'source.output_locator.run_id' in source
    assert 'source.output_locator.step_id' in source
    assert 'source.output_locator.name' in source
    assert 'source.output_locator.accepted_step_sha256' in source
    assert 'view.limitations' in source
    assert 'fact.status === "available"' in source
    assert 'fact.reason' in source


def test_artifacts_use_handles_and_reports_download_without_duplicate_preview():
    source = code()
    assert '/artifacts/${encodeURIComponent(handle)}' in source
    assert '["analysis_report", "analysis_figure"].includes(artifact.artifact_type)' in source
    assert '?download=true' in source
    assert 'contentType.startsWith("image/png")' in source
    assert 'report-content' not in Page().ids
    assert 'Technical report' in source
    assert 'URL.revokeObjectURL(state.artifactUrl)' in source
    assert 'innerHTML' not in source and 'insertAdjacentHTML' not in source
    assert 'path=' not in source


def test_rich_view_is_reconstructed_without_persisting_scientific_state_locally():
    source = code()
    convenience = source[source.index('function saveConveniences'):source.index('async function api')]
    assert set(re.findall(r'^\s+(\w+):', convenience, re.M)) == {
        'sessionId', 'profileId', 'polledTurnId', 'draft', 'inputSetId', 'epizooResourceId',
        'viewedRevisionId', 'workspacePanel',
    }
    initialize = source[source.index('async function initialize'):]
    assert 'choice.input_set_id === conveniences.inputSetId' in initialize
    assert initialize.index('state.inputSets = inputs.choices') < initialize.index('resources = loadResources()')
    display = source[source.index('function displaySession'):source.index('async function refreshCurrentSession')]
    assert 'renderRevisionHistory()' in display
    assert 'if (refreshRevision) loadRevision(nextRevisionId)' in display
    assert 'state.session = view' in display


def test_captured_base_and_scientific_targets_are_not_inferred_from_prose():
    source = code()
    history = source[source.index('function renderHistory'):source.index('function updateControls')]
    assert 'turn.base_revision_id' in history
    assert 'Captured base revision:' in history
    assert 'scientific.targets.map' in history
    assert 'target.revision_id' in history and 'target.output_name' in history
    assert 'target.subject' in history
    assert 'View created result revision' in history
    assert 'Scientific target metadata was not stored in this historical display.' in history


def test_metadata_only_scientific_runs_do_not_block_conversation_reopen():
    source = code()
    reopen = source[source.index('async function openSession'):source.index('function newTurnId')]
    assert '(turn.utterance || turn.response || turn.rerun_revision_id) && !terminal(turn)' in reopen
    assert 'requested && (requested.utterance || requested.response)' in reopen
    assert 'requestedConversation && !terminal(requested)' in reopen
    assert 'resumeTurnId && !requested ? resumeTurnId' in reopen
    assert 'turn.run_id' not in reopen
