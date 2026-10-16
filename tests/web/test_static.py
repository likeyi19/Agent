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
    assert 'artifact.artifact_type === "scientific_analysis"' in predicate
    assert 'artifact.filename === "analysis.h5ad"' in predicate
    assert 'artifact.label === "Embeddings and UMAP — safe export"' in predicate
    assert '"analysis.h5ad": "Embeddings, Leiden and UMAP — safe export"' in predicate


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
        'sessionId', 'profileId', 'polledTurnId', 'draft', 'inputSetId', 'epizooResourceId',
        'viewedRevisionId', 'workspacePanel',
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
    assert 'typeof turn.profile_id === "string"' in gate
    assert '!turn.response' in gate
    assert 'PRESENTATION_PENDING_STATES.has(turn.status) && !submissionFailure' in gate
    assert '!awaitingPresentation(turn) && TERMINAL_STATES.has(turn.status)' in gate
    assert 'PRESENTATION_POLL_INTERVAL_MS = 5000' in code
    assert '? PRESENTATION_POLL_INTERVAL_MS : POLL_INTERVAL_MS' in code


def _javascript_predicates(code, turns):
    """Evaluate the actual shared predicates in an optional system JS runtime."""
    constants = code[code.index('const TERMINAL_STATES'):code.index('const RESPONSE_LABELS')]
    functions = code[code.index('function awaitingPresentation'):code.index('function checkpointText')]
    script = ('(() => {\n' + constants + '\n' + functions + '\nreturn ' + json.dumps(turns) +
              '.map(turn => [awaitingPresentation(turn), terminal(turn)]);})()')
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


def test_no_run_failed_answer_waits_for_display_without_waiting_on_submission_errors():
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
        [True, False], [False, True], [True, False], [True, False], [True, False],
        [False, True], [False, True], [False, True], [False, True], [False, True],
        [True, False],
    ]


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
    assert 'status.textContent = statusLabel(turn.status)' in history
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
      const renderHistory=()=>{},renderRevisionHistory=()=>{},renderRevisionContext=()=>{},renderResultNavigation=()=>{},updateControls=()=>{};
      const shortIdentity=x=>x,resultLabel=x=>x;
    """ + functions + """
      for(const status of ['answered','clarification','running','failed','cancelled']) {
        displaySession({session_id:'s',active_revision_id:'one',generation:1,
          turns:[{turn_id:'t',status}],revisions:[{revision_id:'one'}]});
      }
      return loads;
    })()"""
    assert _javascript_value(script) == []
