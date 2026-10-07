"""Offline smoke lifecycle acceptance with real adapters and fake SDK clients."""

import json
from types import SimpleNamespace

import pytest

from benchmarks.interactive import run_smoke
from benchmarks.interactive.candidates import CANDIDATES, build_candidate


EXECUTE = json.dumps({'turn_schema_version': 1,
                     'decision': {'kind': 'execute_plan', 'target': 'inspect_scATAC'}})
CLARIFY = json.dumps({'turn_schema_version': 1,
                     'decision': {'kind': 'clarify', 'reason': 'unsupported_intent'}})


class SDKClient:
    def __init__(self, candidate, *, response=EXECUTE, available=True, error=None):
        self.candidate, self.response = candidate, response
        self.available, self.error = available, error
        self.events, self.completions, self.closed = [], [], False
        self.models = SimpleNamespace(list=self.list_models, get=self.get_model)
        self.responses = SimpleNamespace(create=self.complete_responses)
        self.interactions = SimpleNamespace(create=self.complete_responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete_chat))

    def list_models(self, **kwargs):
        self.events.append('availability')
        return {'data': [{'id': self.candidate.model_id if self.available else 'unrelated-model'}]}

    def get_model(self, **kwargs):
        assert kwargs['model'] == self.candidate.model_id
        self.events.append('availability')
        return {'name': 'models/' + (self.candidate.model_id if self.available else 'unrelated-model')}

    def _completion(self, kwargs):
        assert self.events == ['availability'], 'Completion before discovery or repeated completion'
        self.events.append('completion')
        self.completions.append(kwargs)
        assert kwargs['model'] == self.candidate.model_id
        if self.error is not None:
            raise self.error

    def complete_responses(self, **kwargs):
        self._completion(kwargs)
        return {'status': 'completed', 'output_text': self.response}

    def complete_chat(self, **kwargs):
        self._completion(kwargs)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': self.response}}]}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def offline_manifest(monkeypatch):
    monkeypatch.setattr(run_smoke, 'comparison_manifest', lambda: {
        'repository_commit': 'offline-fixture', 'qualification_claim': False})


def factory(clients):
    def construct(candidate):
        return build_candidate(candidate, client=clients[candidate.candidate_id], environment={
            'DASHSCOPE_BASE_URL': 'https://example.test/compatible-mode/v1',
        })
    return construct


def assert_zero_science(row):
    assert row['scientific_calls'] == row['execution_entry_attempts'] == row['scientific_step_results'] == 0
    if row['attempt'] is not None:
        assert all(value == 0 for value in row['attempt']['safety'].values())


def test_exact_five_candidate_smoke_uses_discovery_then_one_shared_contract(tmp_path):
    clients = {c.candidate_id: SDKClient(c) for c in CANDIDATES}
    report = run_smoke.smoke(output=tmp_path / 'smoke', factory=factory(clients))
    rows = report['candidates']
    assert [r['candidate']['candidate_id'] for r in rows] == [c.candidate_id for c in CANDIDATES]
    assert [r['candidate']['model_id'] for r in rows] == [c.model_id for c in CANDIDATES]
    prompt_ids, schema_ids = set(), set()
    for row, candidate in zip(rows, CANDIDATES):
        client = clients[candidate.candidate_id]
        assert client.events == ['availability', 'completion']
        assert client.closed is True
        assert row['status'] == 'structured_smoke_succeeded'
        assert row['availability_requests'] == row['completion_calls'] == 1
        assert row['attempt']['contract_success'] is row['attempt']['semantic_success'] is True
        call = row['attempt']['calls'][0]
        prompt_ids.add(call['prompt_fingerprint'])
        schema_ids.add(call['schema_fingerprint'])
        assert_zero_science(row)
    assert len(prompt_ids) == len(schema_ids) == 1
    assert rows[2]['candidate']['endpoint_identity'] == 'https://example.test/compatible-mode/v1'
    assert json.loads((tmp_path / 'smoke/results.json').read_text()) == report


def test_unavailable_exact_model_skips_completion_and_continues(tmp_path):
    candidates = CANDIDATES[:2]
    clients = {c.candidate_id: SDKClient(c, available=i != 0) for i, c in enumerate(candidates)}
    rows = run_smoke.smoke(output=tmp_path / 'smoke', candidates=candidates,
                           factory=factory(clients))['candidates']
    assert rows[0]['status'] == 'provider_model_unavailable'
    assert rows[0]['availability']['requested_model_id'] == 'openai/gpt-oss-120b'
    assert rows[0]['completion_calls'] == 0
    assert rows[0]['attempt'] is None
    assert clients[candidates[0].candidate_id].events == ['availability']
    assert rows[1]['status'] == 'structured_smoke_succeeded'
    for row in rows:
        assert_zero_science(row)


def test_genuine_clarify_is_contract_readiness_without_semantic_qualification(tmp_path):
    candidate = CANDIDATES[0]
    clients = {candidate.candidate_id: SDKClient(candidate, response=CLARIFY)}
    row = run_smoke.smoke(output=tmp_path / 'smoke', candidates=(candidate,),
                          factory=factory(clients))['candidates'][0]
    assert row['status'] == 'structured_smoke_succeeded'
    assert row['attempt']['contract_success'] is True
    assert row['attempt']['genuine_clarify'] is True
    assert row['attempt']['semantic_success'] is False
    assert row['attempt']['failure_category'] == 'semantic'
    assert row['completion_calls'] == 1
    assert_zero_science(row)


def test_contract_failure_does_not_retry_or_turn_into_genuine_clarification(tmp_path):
    candidate = CANDIDATES[0]
    clients = {candidate.candidate_id: SDKClient(candidate, response='not-json')}
    row = run_smoke.smoke(output=tmp_path / 'smoke', candidates=(candidate,),
                          factory=factory(clients))['candidates'][0]
    assert row['status'] == 'incompatible_transport_schema_behavior'
    assert row['attempt']['contract_success'] is False
    assert row['attempt']['genuine_clarify'] is False
    assert row['attempt']['failure_category'] == 'contract'
    assert row['completion_calls'] == 1
    assert_zero_science(row)


def test_operational_failure_does_not_retry_and_next_candidate_runs(tmp_path):
    class RateLimited(Exception):
        status_code = 429
    candidates = CANDIDATES[:2]
    clients = {c.candidate_id: SDKClient(c,
        error=RateLimited('secret-fixture-value') if i == 0 else None) for i, c in enumerate(candidates)}
    report = run_smoke.smoke(output=tmp_path / 'smoke', candidates=candidates, factory=factory(clients))
    first, second = report['candidates']
    assert first['status'] == 'operational_provider_failure'
    assert first['attempt']['failure_category'] == 'operational'
    assert first['attempt']['error_code'] == 'PROVIDER_RATE_LIMITED'
    assert first['completion_calls'] == second['completion_calls'] == 1
    assert second['status'] == 'structured_smoke_succeeded'
    assert 'secret-fixture-value' not in json.dumps(report)
    for row in report['candidates']:
        assert_zero_science(row)


def test_existing_output_directory_is_refused_before_candidate_construction(tmp_path):
    path = tmp_path / 'existing'
    path.mkdir()
    source = path / 'results.json'
    source.write_text('preserved')
    def forbidden(candidate):
        pytest.fail('Candidate constructed before output destination refusal')
    with pytest.raises(FileExistsError):
        run_smoke.smoke(output=path, factory=forbidden)
    assert source.read_text() == 'preserved'


def test_missing_live_flag_rejects_before_smoke_or_files(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail('Smoke invoked without explicit --live flag')
    monkeypatch.setattr(run_smoke, 'smoke', forbidden)
    with pytest.raises(SystemExit) as failure:
        run_smoke.main(['--output', str(tmp_path / 'no-live')])
    assert failure.value.code == 2
    assert not (tmp_path / 'no-live').exists()
