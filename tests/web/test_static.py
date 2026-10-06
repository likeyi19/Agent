"""Small contracts for the framework-free, same-origin browser client."""
from html.parser import HTMLParser
from pathlib import Path
import re


STATIC = Path(__file__).resolve().parents[2] / 'src' / 'agent' / 'web' / 'static'


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_main_page_resolves_only_local_static_assets():
    page = Page((STATIC / 'index.html').read_text())
    assets = [attrs[key] for tag, attrs in page.tags
              for key in ('src', 'href') if key in attrs]
    assert set(assets) == {'/static/app.js', '/static/styles.css'}
    for asset in assets:
        assert (STATIC / asset.removeprefix('/static/')).is_file()
    assert any(tag == 'meta' and attrs.get('name') == 'viewport' for tag, attrs in page.tags)


def test_required_browser_controls_and_dom_references_exist():
    page = Page((STATIC / 'index.html').read_text())
    ids = [attrs['id'] for _, attrs in page.tags if 'id' in attrs]
    assert len(ids) == len(set(ids))
    assert {
        'new-session', 'reopen-form', 'session-input', 'model-choice',
        'input-choice', 'conversation-history', 'turn-form', 'utterance',
        'submit-turn', 'cancel-turn', 'turn-status', 'client-error',
    } <= set(ids)
    code = (STATIC / 'app.js').read_text()
    references = set(re.findall(r'element\("([\w-]+)"\)', code))
    assert references <= set(ids)


def test_models_and_configured_scientific_inputs_are_api_populated():
    code = (STATIC / 'app.js').read_text()
    assert 'api("/models")' in code
    assert 'api("/input-sets")' in code
    assert 'option.value = choice.profile_id' in code
    assert 'option.value = choice.input_set_id' in code
    assert 'body.input_set_id' in code
    assert 'execution_inputs' not in code
    assert 'provider_endpoint' not in code


def test_session_submission_poll_and_cancellation_are_wired():
    code = (STATIC / 'app.js').read_text()
    assert 'api("/sessions", "POST", {})' in code
    assert 'api(sessionPath(sessionId))' in code
    assert 'expected_generation: state.session.generation' in code
    assert 'watchTurn(submission.sessionId, submission.body.turn_id)' in code
    assert '/status`)' in code
    assert '/cancel`, "POST", {})' in code
    assert '["planning", "validated", "running"].includes(state.activeTurn.status)' in code
    assert 'setTimeout(tick, interval)' in code
    assert 'POLL_INTERVAL_MS = 2000' in code


def test_exact_presentation_is_rendered_as_text():
    code = (STATIC / 'app.js').read_text()
    assert 'content.textContent = text' in code
    assert 'turn.response.text' in code
    assert 'innerHTML' not in code
    assert 'insertAdjacentHTML' not in code
    assert 'eval(' not in code
    assert 'RESPONSE_LABELS' in code


def test_refresh_reopens_instead_of_replaying_and_retry_keeps_identity():
    code = (STATIC / 'app.js').read_text()
    initialize = code[code.index('async function initialize()'):]
    assert 'await openSession(conveniences.sessionId' in initialize
    assert 'postSubmission(' not in initialize
    assert 'Object.freeze(body)' in code
    assert 'postSubmission(state.submission)' in code
    convenience_fields = code[code.index('const value = {', code.index('function saveConveniences')):]
    convenience_fields = convenience_fields[:convenience_fields.index('};')]
    assert set(re.findall(r'^\s+(\w+):', convenience_fields, re.M)) == {
        'sessionId', 'profileId', 'polledTurnId', 'draft',
    }


def test_terminal_poll_stops_and_transient_failure_remains_visible():
    code = (STATIC / 'app.js').read_text()
    assert 'if (terminal(turn))' in code
    assert 'showError(turn.error || turn.response && turn.response.error || null)' in code
    assert 'if (!state.durableTurnIds.has(turnId))' in code
    assert 'No durable turn was found. Reopen the session to continue.' in code


def test_scientific_completion_waits_for_persisted_display_without_replay():
    code = (STATIC / 'app.js').read_text()
    pending = code[code.index('const PRESENTATION_PENDING_STATES'):]
    pending = pending[:pending.index(']);')]
    assert set(re.findall(r'"([a-z]+)"', pending)) == {
        'succeeded', 'planned', 'activated', 'stale', 'failed', 'cancelled',
    }
    gate = code[code.index('function awaitingPresentation'):code.index('function checkpointText')]
    assert '!turn.response' in gate
    assert 'Boolean(turn.run_id)' in gate
    assert '!awaitingPresentation(turn) && TERMINAL_STATES.has(turn.status)' in gate
    assert 'PRESENTATION_POLL_INTERVAL_MS = 5000' in code
    assert '? PRESENTATION_POLL_INTERVAL_MS : POLL_INTERVAL_MS' in code
