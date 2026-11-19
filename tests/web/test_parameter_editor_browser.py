"""Real Firefox controls and HTTP; scripted planning, unopened tiny BAM, no science."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from threading import Thread
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from agent.schemas.orchestration import _serialize
from agent.application import preparation_state
from agent.web.app import create_app
from test_preparation_owner_contracts import Model, setup


def eventually(action, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        try:
            result = action()
            if result:
                return result
        except (AssertionError, OSError, RuntimeError):
            pass
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for browser/server state')
        time.sleep(.05)


class Firefox:
    """Minimal W3C WebDriver transport; no Selenium dependency or synthetic input."""
    def __init__(self, base):
        self.base = base
        self.path = ''
        self.actions = []

    def request(self, method, path, data=None):
        self.actions.append(dict(method=method, path=path, data=data))
        request = urllib.request.Request(self.base + path, method=method,
            data=None if data is None else json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)['value']
        except urllib.error.HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    def element(self, css):
        value = self.request('POST', self.path + '/element', dict(using='css selector', value=css))
        return self.path + '/element/' + value['element-6066-11e4-a52e-4f735466cecf']

    def click(self, css):
        def click_current():
            target = self.element(css)
            if not self.request('GET', target + '/enabled'):
                return False
            self.request('POST', target + '/click', {})
            return True
        eventually(click_current)

    def type(self, css, text):
        self.request('POST', self.element(css) + '/value', dict(text=text))


@pytest.fixture
def firefox(tmp_path):
    executable = shutil.which('geckodriver')
    if not executable or not shutil.which('firefox'):
        pytest.skip('Optional Firefox/geckodriver unavailable')
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    with (tmp_path / 'geckodriver.log').open('w') as log:
        process = subprocess.Popen([executable, '--host', '127.0.0.1', '--port', str(port)],
            stdout=log, stderr=subprocess.STDOUT)
        browser = Firefox(f'http://127.0.0.1:{port}')
        try:
            eventually(lambda: browser.request('GET', '/status')['ready'])
            created = browser.request('POST', '/session', dict(capabilities=dict(alwaysMatch={
                'browserName': 'firefox', 'moz:firefoxOptions': {'args': ['-headless', '--width=1440', '--height=1200']}})))
            browser.path = '/session/' + created['sessionId']
            yield browser
        finally:
            try:
                if browser.path:
                    browser.request('DELETE', browser.path)
            finally:
                process.terminate()
                process.wait(timeout=15)


@pytest.mark.parametrize('member,value', [
    ('barcode_tag', 'RG'), ('barcode_regex', '(?P<barcode>[ACGT]+)'),
])
def test_offered_member_editor_posts_and_persists_exact_answer(tmp_path, firefox, member, value):
    model = Model('run_snapatac2', 'pp.make_fragment_file', roles=False)
    original_complete = model.complete
    def complete(**kwargs):
        result = json.loads(original_complete(**kwargs))
        if result.get('schema_version') == 4:
            second = json.loads(json.dumps(result['decision']['steps'][0]))
            second['step_id'] = 'other'
            result['decision']['steps'].append(second)
        return json.dumps(result)
    model.complete = complete
    service, executor = setup(tmp_path, model)
    source = tmp_path / 'source.bam'
    source.write_bytes(b'Unopened technical browser fixture')
    initial = service.submit_turn('session', 'task', 'Convert this BAM to fragments.',
        expected_generation=0, execution_inputs=dict(input_path=str(source)))
    assert initial.status == 'clarification', initial.to_dict()
    sessions = service._application.sessions
    before = sessions.load('session')
    origin = before.interactions[0]
    mechanism = next(c for c in initial.clarification_choices
        if c.get('choice_identity') == 'native-parameter.' + member and c['step_id'] == 'method')
    application = create_app(service)
    requests = []
    @application.middleware('http')
    async def capture(request, call_next):
        body = await request.body()
        response = await call_next(request)
        if request.method == 'POST' and request.url.path.endswith('/turns'):
            requests.append(dict(path=request.url.path, body=json.loads(body), status=response.status_code))
        return response
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application, log_level='warning'))
    thread = Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        eventually(lambda: server.started)
        firefox.request('POST', firefox.path + '/url', dict(url=f'http://127.0.0.1:{port}/'))
        eventually(lambda: firefox.request('GET', firefox.element('#submit-turn') + '/enabled'))
        firefox.click('.conversation-navigation summary')
        firefox.request('POST', firefox.element('#session-input') + '/clear', {})
        firefox.type('#session-input', 'session')
        firefox.click('#reopen-form button[type=submit]')
        selector = lambda turn, handle: f'article[data-turn-id="{turn}"] button[data-action-key="clarification-choice:{handle}"]'
        eventually(lambda: firefox.element(selector('task', mechanism['handle'])))
        firefox.click(selector('task', mechanism['handle']))
        def mechanism_answer():
            rows = sessions.load('session').interactions
            return rows[-1] if len(rows) == 2 and rows[-1].status == 'clarification' else None
        selected = eventually(mechanism_answer)
        editor = next(c for c in service.turn('session', selected.turn_id).clarification_choices
            if c.get('member') == member and c['step_id'] == 'method')
        eventually(lambda: firefox.element(selector(selected.turn_id, editor['handle'])))
        firefox.click(selector(selected.turn_id, editor['handle']))
        eventually(lambda: firefox.element(f'#parameter-steps input[aria-label="method {member}"]'))
        firefox.type(f'#parameter-steps input[aria-label="method {member}"]', value)
        firefox.click('#run-parameters')
        def scalar_answer():
            rows = sessions.load('session').interactions
            return rows[-1] if len(rows) == 3 and rows[-1].status == 'clarification' else None
        actual = eventually(lambda: next((r for r in requests if r['body'].get('parameter_overrides')), None))
        assert actual['status'] == 202, actual
        answered = eventually(scalar_answer)
        assert actual['body']['parameter_overrides'] == [dict(tool='run_snapatac2',
            native_function='pp.make_fragment_file', step_id='method', argument='kwargs', value={member: value})]
        assert actual['body']['clarification_choice'] == dict(question_turn_id=selected.turn_id, handle=editor['handle'])
        persisted, = answered.preparation['answers']
        assert persisted['requirement_id'] == editor['requirement_id']
        assert persisted['value'] == value
        assert answered.preparation['task_id'] == origin.preparation['task_id']
        assert answered.preparation['candidate_id'] == origin.preparation['candidate_id']
        assert sessions.load('session').interactions[0] == origin
        remaining = service.turn('session', answered.turn_id)
        assert preparation_state.observation(answered)['status'] == 'NEEDS_INPUT'
        assert {c['step_id'] for c in remaining.clarification_choices} == {'other'}
        assert not executor.plans and not sessions.load('session').turns and not sessions.load('session').revisions
        # Recheck is another actual UI submission; no execution or reinterpretation.
        eventually(lambda: firefox.element(f'article[data-turn-id="{answered.turn_id}"] .assistant'))
        firefox.click(f'article[data-turn-id="{answered.turn_id}"] .assistant > button')
        rechecked = eventually(lambda: (rows[-1] if len(rows := sessions.load('session').interactions) == 4
            and rows[-1].status == 'clarification' else None))
        assert rechecked.preparation['candidate_id'] == origin.preparation['candidate_id']
        assert not executor.plans and not sessions.load('session').turns
        assert sum('turn_schema_version' in c for c in model.calls) == 1
        assert sum('semantic_prompt_version' in c for c in model.calls) == 1
        evidence = dict(requests=requests, initial=initial.to_dict(), remaining=remaining.to_dict(),
            persisted_answer=_serialize(persisted), final_session=_serialize(sessions.load('session').to_dict()),
            webdriver_actions=firefox.actions)
        destination = Path(os.environ.get('AGENT_EDITOR_EVIDENCE_DIR', tmp_path))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'{member}-http-continuation.json').write_text(json.dumps(evidence, indent=2))
    finally:
        destination = Path(os.environ.get('AGENT_EDITOR_EVIDENCE_DIR', tmp_path))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f'{member}-observed-http.json').write_text(json.dumps(dict(requests=requests,
            session=_serialize(sessions.load('session').to_dict()), webdriver_actions=firefox.actions), indent=2))
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
