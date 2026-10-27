"""Small contracts for the framework-free, same-origin browser client."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


STATIC = Path(__file__).resolve().parents[2] / 'src' / 'agent' / 'web' / 'static'


def test_coordinate_only_download_has_an_explicit_reviewed_label():
    code = (STATIC / 'app.js').read_text()
    predicate = code[code.index('function supportedScientificFile'):code.index('function scientificFileNode')]
    assert '"scientific_analysis"' in predicate
    from agent.application.interactive_schemas import ScientificArtifactHandle
    descriptor = ScientificArtifactHandle('a'*64, 'scientific_analysis', 'b'*64, 'revision', 'turn', 32,
                                        'analysis.h5ad', 'Embeddings and UMAP — safe export').to_dict()
    assert descriptor['title'] == 'UMAP analysis (compact H5AD)'
    assert 'X=None' in descriptor['description']


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
    assert 'setTimeout(tick, POLL_INTERVAL_MS)' in code
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
        'sessionId', 'profileId', 'polledTurnId', 'draft', 'inputSetId', 'epizooResourceId',
        'viewedRevisionId', 'workspacePanel', 'resultChoice',
    }


def test_terminal_poll_stops_and_transient_failure_remains_visible():
    code = (STATIC / 'app.js').read_text()
    assert 'if (terminal(turn))' in code
    assert 'showError(turn.error || turn.response && turn.response.error || null)' in code
    assert 'if (!state.durableTurnIds.has(turnId))' in code
    assert 'No durable turn was found. Reopen the session to continue.' in code


def test_browser_uses_application_status_without_reconstructing_presentation_lifecycle():
    code = (STATIC / 'app.js').read_text()
    assert 'PRESENTATION_PENDING_STATES' not in code
    assert 'awaitingPresentation' not in code
    assert 'PRESENTATION_POLL_INTERVAL_MS' not in code
    gate = code[code.index('function terminal'):code.index('function checkpointText')]
    assert 'Boolean(turn.response) || TERMINAL_STATES.has(turn.status)' in gate
    assert 'profile_id' not in gate and 'turn.error' not in gate and 'run_id' not in gate


def _javascript_predicates(code, turns):
    """Evaluate the actual shared predicates in an optional system JS runtime."""
    constants = code[code.index('const TERMINAL_STATES'):code.index('const RESPONSE_LABELS')]
    functions = code[code.index('function terminal'):code.index('function checkpointText')]
    script = ('(() => {\n' + constants + '\n' + functions + '\nreturn ' + json.dumps(turns) +
              '.map(turn => terminal(turn));})()')
    return _javascript_value(script)


def _javascript_value(script):
    runner = '''
import ctypes, ctypes.util, sys
library = ctypes.util.find_library('javascriptcoregtk-4.1') or ctypes.util.find_library('javascriptcoregtk-4.0')
if library is None:
    raise SystemExit(77)
try:
    runtime = ctypes.CDLL(library)
    runtime.jsc_context_new.restype = ctypes.c_void_p
    runtime.jsc_context_evaluate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_ssize_t]
    runtime.jsc_context_evaluate.restype = ctypes.c_void_p
    runtime.jsc_context_get_exception.argtypes = [ctypes.c_void_p]
    runtime.jsc_context_get_exception.restype = ctypes.c_void_p
    runtime.jsc_value_to_json.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    runtime.jsc_value_to_json.restype = ctypes.c_void_p
except (OSError, AttributeError):
    raise SystemExit(77)
context = runtime.jsc_context_new()
value = runtime.jsc_context_evaluate(context, sys.stdin.buffer.read(), -1)
if runtime.jsc_context_get_exception(context):
    raise RuntimeError('Production JavaScript predicates raised an exception')
result = runtime.jsc_value_to_json(value, 0)
if not result:
    raise RuntimeError('Production JavaScript predicates returned no JSON')
print(ctypes.string_at(result).decode())
'''
    interpreters = [sys.executable]
    # Isolate a system JS library from a Python environment's C++ runtime.
    system_python = Path('/usr/bin/python3')
    if system_python.is_file() and system_python.resolve() != Path(sys.executable).resolve():
        interpreters.append(str(system_python))
    for interpreter in interpreters:
        result = subprocess.run([interpreter, '-c', runner], input=script, text=True,
                                capture_output=True, timeout=15)
        if result.returncode == 77:
            continue
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    pytest.skip('Optional JavaScriptCore runtime is unavailable')


def test_application_finalizing_is_pending_and_saved_or_historical_outcomes_are_terminal():
    code = (STATIC / 'app.js').read_text()
    pending = dict(profile_id='captured-profile', status='finalizing', response=None)
    turns = [pending, dict(pending, profile_id=None),
             dict(pending, response={'kind': 'execute', 'error': {'code': 'APP_REPORT_FAILED'}})]
    for status in ('answered', 'clarification', 'navigated', 'succeeded', 'failed',
                   'cancelled', 'interrupted', 'unavailable', 'stale', 'planned'):
        turns.append(dict(pending, status=status))
    assert _javascript_predicates(code, turns) == [False, False] + [True] * 11


def test_failed_and_interrupted_states_are_not_overridden_by_profile_or_error_inference():
    code = (STATIC / 'app.js').read_text()
    pending = dict(profile_id='captured-profile', status='failed', run_id=None,
                   response=None, error={'code': 'INTERACTIVE_PRESENTATION_UNAVAILABLE'})
    published = dict(pending, response={'kind': 'answer', 'status': 'unavailable',
                                     'text': 'The recorded answer.'}, error=None)
    turns = [
        pending,
        published,
        dict(pending, error=None),
        dict(pending, status='cancelled'),
        dict(pending, status='cancelled', error=None),
        dict(pending, error={'code': 'INTERACTIVE_APPLICATION_FAILED'}),
        dict(pending, status='cancelled', error={'code': 'INTERACTIVE_APPLICATION_FAILED'}),
        dict(pending, profile_id=None),
        dict(pending, status='cancelled', profile_id=None),
        dict(pending, status='interrupted'),
        dict(pending, run_id='run'),
    ]
    assert _javascript_predicates(code, turns) == [
        True, True, True, True, True, True, True, True, True, True, True,
    ]


def test_status_labels_distinguish_saved_work_without_claiming_an_absent_result():
    code = (STATIC / 'app.js').read_text()
    label = code[code.index('function statusLabel'):code.index('function evidenceValueNode')]
    result = _javascript_value('(() => {\n' + label + '''
      return [statusLabel("finalizing", null, "accepted-revision"),
        statusLabel("finalizing"), statusLabel("finalizing", "execute", "accepted-revision"),
        statusLabel("failed", "answer"), statusLabel("failed", "answer", "accepted-revision")];
    })()''')
    assert result == [
        'Scientific analysis completed. Preparing the results and explanation.',
        'Preparing the scientific response.', 'Saved progress requires recovery',
        'Answer unavailable', 'Answer unavailable',
    ]


def _javascript_poll(turn, *, offered=(), refresh_error=None, reopen=False):
    """Run the production poll/control functions with a bounded DOM/HTTP transport.

    JavaScriptCore drains the async Promise jobs after evaluation. setTimeout is
    captured rather than run, so a pending response cannot finish by test timing.
    The Session endpoint supplies the independently authoritative accepted result.
    """
    code = (STATIC / 'app.js').read_text()
    constants = code[code.index('const POLL_INTERVAL_MS'):code.index('const RESPONSE_LABELS')]
    predicates = code[code.index('function terminal'):code.index('function shortIdentity')]
    labels = code[code.index('function readableLabel'):code.index('function evidenceValueNode')]
    controls = code[code.index('function updateControls'):code.index('function mergeTurn')]
    polling = code[code.index('function stopPolling'):code.index('// Display choice only:')]
    polling += code[code.index('async function refreshCurrentSession'):code.index('async function openSession')]
    if reopen:
        polling += code[code.index('async function openSession'):code.index('function newTurnId')]
    session = dict(session_id='session', generation=1, turns=[turn], active_revision_id=turn.get('revision_id'),
                   revisions=[dict(revision_id=turn['revision_id'])] if turn.get('revision_id') else [])
    setup = '''
      const result = {calls: [], errors: [], delays: [], displayed: [], merged: []};
      const dom = {};
      const element = id => dom[id] || (dom[id] = {
        options: [], parentElement: {}, replaceChildren() {},
      });
      const document = {querySelectorAll() {return [];}, createElement() {return {};}};
      const canAddScientificInput = () => true;
      const selectedResourceType = () => null;
      const clearTimeout = () => {};
      const setTimeout = (callback, delay) => {result.delays.push(delay); return "pending-timer";};
      const showError = error => {result.errors.push(error);};
      const saveConveniences = () => {};
      const sessionPath = () => "/session";
      const turnPath = () => "/turn";
      const mergeTurn = turn => {result.merged.push(turn);};
      const displaySession = view => {state.session = view; result.displayed.push(view); updateControls();};
      element("retry-turn").hidden = true;
    '''
    setup += '\nconst turn = ' + json.dumps(turn) + ';'
    setup += '\nconst refreshed = ' + json.dumps(session) + ';'
    setup += '\nconst refreshError = ' + json.dumps(refresh_error) + ';'
    setup += '\nconst state = ' + json.dumps(dict(
        session=dict(session_id='session', revisions=[dict(revision_id=value) for value in offered]),
        models=[{}], activeTurn=None, submission={}, epoch=1, pollEpoch=0, posting=False,
        attachments=[], resources=[], fastqMembers=[], epizooResources=[])) + ';'
    setup += '''
      state.durableTurnIds = new Set(["turn"]);
      state.scientificFiles = new Map();
      const api = async path => {
        result.calls.push(path);
        if (path === "/turn/status") return turn;
        if (refreshError) throw refreshError;
        return refreshed;
      };
      result.dom = dom;
      result.state = state;
    '''
    invocation = 'openSession("session", "turn")' if reopen else 'watchTurn("session", "turn")'
    return _javascript_value('(() => {\n' + constants + predicates + labels + controls + polling
                             + setup + '\n' + invocation + '; return result;})()')


def test_pending_poll_exposes_accepted_revision_and_keeps_submission_disabled():
    turn = dict(session_id='session', turn_id='turn', status='finalizing', revision_id='new-result',
                response=None, error=None, steps=[])
    pending = _javascript_poll(turn)
    assert pending['calls'] == ['/turn/status', '/session']
    assert pending['displayed'][0]['active_revision_id'] == 'new-result'
    assert pending['state']['activeTurn'] == turn and pending['state']['submission'] == {}
    assert pending['delays'] == [2000] and pending['errors'] == [None]
    assert pending['dom']['submit-turn']['disabled'] is True
    assert pending['dom']['cancel-turn']['hidden'] is True
    assert pending['dom']['retry-turn'].get('hidden') is not False
    assert pending['dom']['turn-status']['textContent'].startswith('Scientific analysis completed.')
    already_visible = _javascript_poll(turn, offered=('new-result',))
    assert already_visible['calls'] == ['/turn/status']
    assert already_visible['dom']['submit-turn']['disabled'] is True


def test_read_only_answer_poll_remains_pending_without_a_scientific_run():
    turn = dict(session_id='session', turn_id='turn', status='finalizing', revision_id=None,
                run_id=None, response=None, error=None, steps=[])
    result = _javascript_poll(turn)
    assert result['calls'] == ['/turn/status'] and result['delays'] == [2000]
    assert result['errors'] == [None] and result['dom']['submit-turn']['disabled'] is True
    assert result['dom']['turn-status']['textContent'] == 'Preparing the scientific response.'


def test_reopen_during_finalization_resumes_read_only_polling_without_submission():
    turn = dict(session_id='session', turn_id='turn', status='finalizing', revision_id='accepted-result',
                run_id='accepted-run', response=None, error=None, utterance='Explain this result.', steps=[])
    result = _javascript_poll(turn, reopen=True)
    assert result['calls'] == ['/session', '/turn/status'] and result['delays'] == [2000]
    assert result['state']['activeTurn'] == turn and result['state']['submission'] is None
    assert result['displayed'][0]['active_revision_id'] == 'accepted-result'
    assert result['dom']['submit-turn']['disabled'] is True
    assert all(error is None for error in result['errors'])


@pytest.mark.parametrize('status,error', [
    ('interrupted', 'INTERACTIVE_PROCESSING_INTERRUPTED'),
    ('unavailable', 'INTERACTIVE_PRESENTATION_UNAVAILABLE'),
    ('failed', 'INTERACTIVE_APPLICATION_FAILED'),
])
def test_actual_terminal_errors_stop_polling_and_remain_visible(status, error):
    turn = dict(session_id='session', turn_id='turn', status=status, revision_id=None,
                response=None, error=dict(code=error, message='Recorded failure.'), steps=[])
    result = _javascript_poll(turn)
    assert result['errors'] == [turn['error']]
    assert result['delays'] == [] and result['state']['activeTurn'] is None
    assert result['state']['submission'] is None
    assert result['dom']['submit-turn']['disabled'] is False


def test_saved_response_and_artifacts_complete_exactly_without_pending_inference():
    shown = dict(kind='answer', text='The exact saved response.', artifacts=[
        dict(artifact_id='pinned-figure', png_sha256='a' * 64)])
    turn = dict(session_id='session', turn_id='turn', status='answered', revision_id='result',
                response=shown, error=None, steps=[])
    result = _javascript_poll(turn)
    assert result['calls'] == ['/turn/status', '/session']
    assert result['merged'][0]['response'] == shown
    assert result['displayed'][0]['turns'][0]['response'] == shown
    assert result['state']['activeTurn'] is None and result['delays'] == []
    assert result['dom']['submit-turn']['disabled'] is False


def test_result_refresh_failure_does_not_stop_an_active_response_poll():
    turn = dict(session_id='session', turn_id='turn', status='finalizing', revision_id='result',
                response=None, error=None, steps=[])
    result = _javascript_poll(turn, refresh_error=dict(code='RESPONSE_UNAVAILABLE', message='Retry reading results.'))
    assert result['calls'] == ['/turn/status', '/session'] and result['delays'] == [2000]
    assert result['state']['activeTurn'] == turn
    assert result['errors'][-1]['code'] == 'RESPONSE_UNAVAILABLE'
    assert result['dom']['submit-turn']['disabled'] is True


def test_authoritative_interruption_offers_recovery_with_the_original_native_error():
    code = (STATIC / 'app.js').read_text()
    history = code[code.index('function renderHistory'):code.index('function updateControls')]
    condition = next(line.strip()[4:-3] for line in history.splitlines()
                     if line.strip().startswith('if ((!turn.response && turn.status === "interrupted")'))
    turns = [
        dict(status='interrupted', response=None, error=dict(code='NATIVE_RUN_FAILED')),
        dict(status='finalizing', response=None, error=dict(code='NATIVE_RUN_FAILED')),
        dict(status='unavailable', response=None, error=dict(code='INTERACTIVE_PRESENTATION_UNAVAILABLE')),
        dict(status='finalizing', response=dict(kind='execute'), error=dict(code='APP_REPORT_FAILED')),
    ]
    script = '(() => { return ' + json.dumps(turns) + '.map(turn => {const error=turn.error; return Boolean(' + condition + ');});})()'
    assert _javascript_value(script) == [True, False, False, True]


class HierarchyPage(Page):
    def __init__(self, text):
        self.stack, self.ancestors = [], {}
        super().__init__(text)

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ancestors[attrs['id']] = tuple(a.get('id') for _, a in self.stack)
        if tag not in {'input', 'meta', 'link', 'img', 'br', 'hr'}:
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break


def test_results_and_conversation_are_independent_workspace_panels():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    ids = [attrs['id'] for _, attrs in page.tags if 'id' in attrs]
    assert ids.index('scientific-details') < ids.index('conversation-history') < ids.index('turn-form')
    assert 'conversation-panel' in page.ancestors['conversation-history']
    assert 'conversation-panel' in page.ancestors['turn-form']
    assert 'conversation-panel' not in page.ancestors['scientific-details']
    assert 'research-layout' in page.ancestors['scientific-panel']
    details = next(attrs for tag, attrs in page.tags if attrs.get('id') == 'scientific-details')
    assert details['tabindex'] == '-1' and details['aria-labelledby'] == 'scientific-heading'
    css = (STATIC / 'styles.css').read_text()
    layouts = re.findall(r'\.research-layout\s*\{([^}]+)\}', css)
    assert any('grid-template-columns' in rule and '400px' in rule for rule in layouts)
    assert '@media (max-width: 1100px)' in css
    assert '[data-panel="conversation"]' in css and '[data-panel="results"]' in css


def test_technical_details_are_closed_and_result_summary_and_downloads_are_open():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    details = {attrs['id']: attrs for tag, attrs in page.tags if tag == 'details' and 'id' in attrs}
    for identifier in ('session-identities', 'revision-identities', 'revision-history-panel',
                       'result-methods', 'technical-report-record', 'evidence-panel'):
        assert identifier in details and 'open' not in details[identifier]
    for identifier in ('artifact-panel',):
        assert 'open' in details[identifier]
    for identifier in ('current-session', 'current-generation', 'current-revision'):
        assert 'session-identities' in page.ancestors[identifier]
    for identifier in ('viewed-revision', 'revision-summary'):
        assert 'revision-identities' in page.ancestors[identifier]
    assert 'revision-history-panel' in page.ancestors['revision-history']


def test_selected_input_label_and_result_details_use_existing_safe_read_paths():
    page = Page((STATIC / 'index.html').read_text())
    assert any(attrs.get('id') == 'selected-input-context' for _, attrs in page.tags)
    code = (STATIC / 'app.js').read_text()
    context = code[code.index('function inputContext()'):code.index('element("new-session")', code.index('function inputContext()'))]
    assert 'selectedOptions[0]' in context and 'choice.textContent' in context
    assert 'element("selected-input-context").textContent' in context
    assert 'inputContext();' in code
    button = code[code.index('result.className = "secondary view-result"'):code.index('(response || article).append(result)')]
    assert 'result.textContent = "View result details"' in button
    assert 'loadRevision(turn.revision_id)' in button
    assert 'showWorkspacePanel("results")' in button and 'details.focus' in button
    assert 'submitRequest(' not in button and 'postSubmission(' not in button
    assert 'execution_inputs' not in code and 'innerHTML' not in code


def test_compact_header_and_advanced_session_controls_preserve_existing_ids():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    details = {attrs['id']: attrs for tag, attrs in page.tags if tag == 'details' and 'id' in attrs}
    assert 'open' not in details['advanced-controls']
    for identifier in ('session-input', 'reopen-form', 'session-context', 'model-description', 'session-identities'):
        assert 'advanced-controls' in page.ancestors[identifier]
    for identifier in ('model-choice', 'new-session'):
        assert 'advanced-controls' not in page.ancestors[identifier]
    assert any(tag == 'section' and attrs.get('id') == 'scientific-panel' for tag, attrs in page.tags)
    assert 'scientific-panel' in page.ancestors['scientific-details']


def test_composer_keeps_message_selected_files_and_transfer_state_outside_disclosures():
    page = HierarchyPage((STATIC / 'index.html').read_text())
    ids = [attrs['id'] for _, attrs in page.tags if 'id' in attrs]
    assert ids.index('selected-scientific-inputs') < ids.index('utterance') < ids.index('attachment-panel')
    for identifier in ('utterance', 'selected-scientific-inputs', 'upload-status', 'upload-progress'):
        assert 'turn-form' in page.ancestors[identifier]
        assert 'attachment-panel' not in page.ancestors[identifier]
        assert 'scientific-input-settings' not in page.ancestors[identifier]
    for identifier in ('upload-file', 'resource-choice', 'fastq-fields'):
        assert 'attachment-panel' in page.ancestors[identifier]
    assert 'scientific-input-settings' in page.ancestors['input-choice']
    assert 'scientific-input-settings' in page.ancestors['epizoo-resource-choice']


def test_history_restores_details_and_focus_by_exact_turn_and_action_identity():
    code = (STATIC / 'app.js').read_text()
    capture = code[code.index('function captureConversationUi'):code.index('function renderHistory')]
    assert 'details.closest(".turn").dataset.turnId' in capture
    assert 'focused.dataset.actionKey' in capture
    assert 'node.dataset.turnId === captured.focus.turnId' in capture
    assert 'node.dataset.actionKey === captured.focus.actionKey' in capture
    assert 'target.focus({ preventScroll: true })' in capture
    assert 'restoreConversationUi(history, captured)' in code
    history = code[code.index('function renderHistory'):code.index('function updateControls')]
    assert 'status.textContent = statusLabel(turn.status, turn.response && turn.response.kind, turn.revision_id)' in history
    assert 'checkpointText(turn)' in history


@pytest.mark.parametrize('previous_active,viewed,active,offered,expected', [
    (None, None, 'one', ['one'], 'one'),
    ('one', 'one', 'two', ['one', 'two'], 'two'),
    ('two', 'one', 'three', ['one', 'two', 'three'], 'one'),
    ('two', 'two', 'two', ['one', 'two'], 'two'),
    ('two', 'one', 'two', ['one', 'two'], 'one'),
    ('two', 'removed', 'two', ['two'], 'two'),
    (None, None, None, [], None),
    (None, None, None, ['unactivated'], None),
    ('two', 'two', 'not-offered', ['two'], 'two'),
])
def test_result_display_follows_server_active_identity_without_discarding_preview(previous_active, viewed, active, offered, expected):
    code = (STATIC / 'app.js').read_text()
    function = code[code.index('function revisionToDisplay'):code.index('function displaySession')]
    previous = dict(session_id='s', active_revision_id=previous_active)
    view = dict(session_id='s', active_revision_id=active,
                revisions=[dict(revision_id=name) for name in offered])
    script = ('(() => {' + function + 'return revisionToDisplay(' + json.dumps(previous) + ','
              + json.dumps(view) + ',' + json.dumps(viewed) + ');})()')
    assert _javascript_value(script) == expected


def test_result_display_does_not_carry_preview_into_another_session():
    code = (STATIC / 'app.js').read_text()
    function = code[code.index('function revisionToDisplay'):code.index('function displaySession')]
    assert _javascript_value('(() => {' + function + "return revisionToDisplay({session_id:'old',active_revision_id:'two'}, {session_id:'new',active_revision_id:'two',revisions:[{revision_id:'one'},{revision_id:'two'}]}, 'one');})()") == 'two'


def test_answer_and_failed_turn_refresh_do_not_replace_same_revision_content():
    code = (STATIC / 'app.js').read_text()
    functions = code[code.index('function revisionToDisplay'):code.index('async function refreshCurrentSession')]
    # Run the production refresh path with the same accepted revision and changed
    # conversation status. No tool or provider interface exists in this renderer.
    script = """(() => {
      const nodes = new Map(); const element = id => {if(!nodes.has(id)) nodes.set(id, {});return nodes.get(id)};
      const state = {session:{session_id:'s',active_revision_id:'one',generation:1},
        attachmentSession:'s',viewedRevisionId:'one',revision:{revision_id:'one'}};
      const loads=[];
      const loadRevision=id=>loads.push(id), clearRevisionView=()=>loads.push(null);
      const rememberSession=()=>{},renderHistory=()=>{},renderRevisionHistory=()=>{},renderRevisionContext=()=>{},renderResultNavigation=()=>{},updateControls=()=>{};
      const shortIdentity=x=>x,resultLabel=x=>x;
    """ + functions + """
      for(const status of ['answered','clarification','running','failed','cancelled']) {
        displaySession({session_id:'s',active_revision_id:'one',generation:1,
          turns:[{turn_id:'t',status}],revisions:[{revision_id:'one'}]});
      }
      return loads;
    })()"""
    assert _javascript_value(script) == []
