"""Offline exact-model/strict-contract admission for benchmark-only OpenRouter."""
import json
from types import SimpleNamespace

import pytest

from agent.orchestration.planning_model import PlanningModelError
from benchmarks.interactive.candidates import Availability, CANDIDATES
from benchmarks.interactive import run_openrouter_smoke, run_smoke
from benchmarks.interactive.openrouter import (
    OPENROUTER_CANDIDATES, Q1E_CANDIDATES, build_openrouter_candidate,
    classify_readiness, discover_models,
)


EXECUTE = json.dumps({"turn_schema_version": 1,
    "decision": {"kind": "execute_plan", "target": "inspect_scATAC"}})
SCHEMA = {"type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
    "required": ["status"], "additionalProperties": False}


class SDKError(Exception):
    def __init__(self, status, message="sanitized fixture", code=None):
        self.status_code = status
        self.body = {"error": {"code": code, "message": message}}
        self.response = SimpleNamespace(status_code=status, headers={})
        super().__init__("Raw provider exception must remain private")


class Client:
    """SDK-shaped client with observable requests and no real network."""

    def __init__(self, cards=None, *, catalog_error=None, completion_error=None, response=EXECUTE, metadata=None):
        self.cards = cards if cards is not None else [{"id": c.model_id,
            "supported_parameters": ["response_format", "structured_outputs"],
            "pricing": {"prompt": "0", "completion": "0"}} for c in OPENROUTER_CANDIDATES]
        self.catalog_error, self.completion_error, self.response = catalog_error, completion_error, response
        self.metadata = metadata
        self.catalog_calls, self.completion_calls, self.closed = [], [], False
        self.models = SimpleNamespace(list=self.list_models)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

    def list_models(self, **kwargs):
        assert not self.closed, "Closed catalog client was reused"
        self.catalog_calls.append(kwargs)
        if self.catalog_error:
            raise self.catalog_error
        return {"data": self.cards}

    def complete(self, **kwargs):
        assert not self.closed, "Closed completion client was reused"
        self.completion_calls.append(kwargs)
        if self.completion_error:
            raise self.completion_error
        answer = {"choices": [{"finish_reason": "stop", "message": {"content": self.response}}]}
        if self.metadata is not None:
            answer["openrouter_metadata"] = self.metadata
        return answer

    def close(self):
        self.closed = True


def availability(candidate):
    return Availability("available", "CANDIDATE_MODEL_AVAILABLE", candidate.model_id,
        "models.list", provider_model_id=candidate.model_id)


def test_openrouter_has_separate_exact_candidates_and_preserves_existing_cohort():
    assert [c.candidate_id for c in OPENROUTER_CANDIDATES] == [
        "openrouter-nemotron-3-super", "openrouter-nex-n2.5-pro", "openrouter-minimax-m3"]
    assert [c.model_id for c in OPENROUTER_CANDIDATES] == [
        "nvidia/nemotron-3-super-120b-a12b:free", "nex-agi/nex-n2.5-pro:free", "minimax/minimax-m3:free"]
    assert all(c.provider_id == "openrouter" and c.credential_env == "OPENROUTER_API_KEY"
        for c in OPENROUTER_CANDIDATES)
    assert all(c.provider_id != "openrouter" for c in CANDIDATES)


def test_q1e_has_only_the_two_requested_exact_free_candidates():
    assert [(c.candidate_id, c.provider_id, c.model_id) for c in Q1E_CANDIDATES] == [
        ("openrouter-nex-n2.5-mini", "openrouter", "nex-agi/nex-n2.5-mini:free"),
        ("openrouter-apodex-1.1-mini", "openrouter", "apodex/apodex-1.1-mini:free")]
    for candidate in Q1E_CANDIDATES:
        assert candidate.adapter_family == "openai-chat-completions"
        assert candidate.credential_env == "OPENROUTER_API_KEY"
        assert candidate.endpoint == "https://openrouter.ai/api/v1"
        assert candidate.timeout_seconds == OPENROUTER_CANDIDATES[0].timeout_seconds
        assert candidate not in OPENROUTER_CANDIDATES


def test_shared_catalog_discovery_uses_one_request_and_exact_ids():
    client = Client()
    discovery = discover_models(client)
    assert discovery["catalog_requests"] == len(client.catalog_calls) == 1
    assert client.completion_calls == []
    assert len(discovery["candidates"]) == 3
    for row, candidate in zip(discovery["candidates"], OPENROUTER_CANDIDATES):
        assert row["discovery_result"] == "EXACT_MODEL_AVAILABLE"
        assert row["candidate"]["provider_id"] == "openrouter"
        assert row["availability"]["requested_model_id"] == candidate.model_id
        assert row["availability"]["provider_model_id"] == candidate.model_id
        assert "response_format" in row["supported_parameters"]


@pytest.mark.parametrize("replacement", ["other-model:free", "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-super-120b-a12b:latest"])
def test_model_alias_or_suffix_change_never_substitutes_requested_exact_id(replacement):
    requested = OPENROUTER_CANDIDATES[0]
    client = Client(cards=[{"id": replacement, "aliases": [requested.model_id]}])
    row = discover_models(client, candidates=(requested,))["candidates"][0]
    assert row["discovery_result"] == "EXACT_MODEL_UNAVAILABLE"
    assert row["availability"]["provider_model_id"] is None
    assert client.completion_calls == []


def test_duplicate_exact_catalog_ids_fail_closed():
    candidate = OPENROUTER_CANDIDATES[0]
    client = Client(cards=[{"id": candidate.model_id}, {"id": candidate.model_id}])
    row = discover_models(client, candidates=(candidate,))["candidates"][0]
    assert row["discovery_result"] == "DISCOVERY_OPERATIONAL_FAILURE"
    assert row["availability"]["status"] != "available"
    assert client.completion_calls == []


@pytest.mark.parametrize("cards", ["not-a-list", [{"id": "irrelevant"}] * 4097])
def test_catalog_is_bounded_and_invalid_catalog_is_not_unavailable_evidence(cards):
    row = discover_models(Client(cards=cards), candidates=OPENROUTER_CANDIDATES[:1])["candidates"][0]
    assert row["discovery_result"] == "DISCOVERY_OPERATIONAL_FAILURE"


def test_catalog_transport_error_is_sanitized_and_not_retried():
    secret = "private-api-credential"
    client = Client(catalog_error=SDKError(429, "Bearer " + secret, "rate_limited"))
    result = discover_models(client, secrets=(secret,))
    assert len(client.catalog_calls) == 1
    assert all(row["discovery_result"] == "DISCOVERY_OPERATIONAL_FAILURE" for row in result["candidates"])
    assert secret not in json.dumps(result)
    assert all(row["diagnosis"]["http_status"] == 429 for row in result["candidates"])


@pytest.mark.parametrize("candidate", OPENROUTER_CANDIDATES + Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_generic_chat_transport_keeps_exact_identity_strict_schema_and_prompt(candidate):
    client = Client(response='{"status":"ok"}')
    runtime = build_openrouter_candidate(candidate, availability=availability(candidate), client=client)
    answer = runtime.complete(prompt="A frozen provider-neutral prompt.", response_schema=SCHEMA)
    assert answer == '{"status":"ok"}'
    assert runtime.manifest()["provider_id"] == "openrouter"
    assert runtime.manifest()["model_id"] == candidate.model_id
    assert runtime.model_id == "openrouter:" + candidate.model_id
    assert runtime.availability_requests == 0
    assert len(client.catalog_calls) == 0
    assert len(client.completion_calls) == 1
    call = client.completion_calls[0]
    assert call["model"] == candidate.model_id
    assert call["messages"] == [{"role": "user", "content": "A frozen provider-neutral prompt."}]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["schema"] == SCHEMA
    assert call["extra_body"]["provider"] == {"require_parameters": True, "allow_fallbacks": False,
        "max_price": {"prompt": 0, "completion": 0, "request": 0}}
    assert call["extra_body"]["plugins"] == [{"id": name, "enabled": False}
        for name in ("response-healing", "context-compression", "web")]
    assert call["extra_headers"] == {"X-OpenRouter-Metadata": "enabled"}
    assert "order" not in call["extra_body"]["provider"] and "only" not in call["extra_body"]["provider"]
    assert "tools" not in call


@pytest.mark.parametrize("admission", [
    Availability("unavailable", "CANDIDATE_MODEL_UNAVAILABLE", OPENROUTER_CANDIDATES[0].model_id, "models.list"),
    Availability("available", "CANDIDATE_MODEL_AVAILABLE", "different:free", "models.list", provider_model_id="different:free"),
    Availability("available", "CANDIDATE_MODEL_AVAILABLE", OPENROUTER_CANDIDATES[0].model_id, "models.list", provider_model_id="different:free"),
])
def test_completion_requires_same_exact_catalog_admission(admission):
    client = Client()
    with pytest.raises((PlanningModelError, ValueError)):
        runtime = build_openrouter_candidate(OPENROUTER_CANDIDATES[0], availability=admission, client=client)
        runtime.complete(prompt="x", response_schema=SCHEMA)
    assert client.completion_calls == []


@pytest.mark.parametrize("candidate", OPENROUTER_CANDIDATES[:1] + Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_transport_guard_prevents_a_second_generation_before_sdk_invocation(candidate):
    client = Client(response='{"status":"ok"}')
    runtime = build_openrouter_candidate(candidate, availability=availability(candidate), client=client)
    runtime.complete(prompt="x", response_schema=SCHEMA)
    with pytest.raises((PlanningModelError, AssertionError)):
        runtime.complete(prompt="x", response_schema=SCHEMA)
    assert len(client.completion_calls) == 1


def test_construction_disables_sdk_retries_without_persisting_credential(monkeypatch):
    import openai
    observed = []
    client = Client()
    def factory(**kwargs):
        observed.append(kwargs)
        return client
    monkeypatch.setattr(openai, "OpenAI", factory)
    candidate = OPENROUTER_CANDIDATES[0]
    secret = "private-openrouter-key"
    runtime = build_openrouter_candidate(candidate, availability=availability(candidate),
        environment={"OPENROUTER_API_KEY": secret})
    assert observed[0]["api_key"] == secret
    assert observed[0]["max_retries"] == 0
    assert observed[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert secret not in json.dumps(runtime.manifest())
    assert client.catalog_calls == client.completion_calls == []


@pytest.mark.parametrize("candidate", OPENROUTER_CANDIDATES[:1] + Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_transport_diagnostic_sanitizes_underlying_error_without_extra_call(candidate):
    secret = "private-openrouter-key"
    client = Client(completion_error=SDKError(400, "json_schema unsupported; API key=" + secret, "unsupported_format"))
    runtime = build_openrouter_candidate(candidate, availability=availability(candidate), client=client, secrets=(secret,))
    with pytest.raises(PlanningModelError):
        runtime.complete(prompt="x", response_schema=SCHEMA)
    assert len(client.completion_calls) == 1
    assert runtime.transport_diagnostic["http_status"] == 400
    assert secret not in json.dumps(runtime.transport_diagnostic)


def smoke_row(*, contract=True, failure=None):
    return {"status": "structured_smoke_succeeded" if contract else "operational_provider_failure",
        "attempt": {"contract_success": contract, "semantic_success": False if contract else None,
            "failure_category": failure}}


def discovered(candidate=OPENROUTER_CANDIDATES[0], result="EXACT_MODEL_AVAILABLE"):
    return {"candidate": candidate.manifest(), "discovery_result": result,
        "availability": availability(candidate).to_dict()}


def diagnostic(status, message):
    return {"http_status": status, "provider_error": {"code": None, "message": message}, "rate_limit": {}}


def test_typed_semantic_failure_is_readiness_only_without_semantic_qualification():
    outcome = classify_readiness(smoke_row(), discovered(), None)
    assert outcome["classification"] == "READY_FOR_Q2"


@pytest.mark.parametrize("status,message,classification", [
    (400, "json_schema response format is not supported", "CONTRACT_INCOMPATIBLE"),
    (422, "structured_outputs json_schema is unsupported", "CONTRACT_INCOMPATIBLE"),
    (404, "No endpoints found that support the requested parameters", "CONTRACT_INCOMPATIBLE"),
    (404, "No endpoints found", "OPERATIONALLY_BLOCKED"),
    (429, "Rate limit exceeded", "OPERATIONALLY_BLOCKED"),
    (401, "Invalid API key", "EXTERNAL_CONFIGURATION_REQUIRED"),
    (403, "Account permission denied", "EXTERNAL_CONFIGURATION_REQUIRED"),
])
def test_readiness_keeps_specific_transport_rejection_categories(status, message, classification):
    outcome = classify_readiness(smoke_row(contract=None, failure="operational"), discovered(), diagnostic(status, message))
    assert outcome["classification"] == classification


def test_parser_failure_is_contract_incompatible():
    outcome = classify_readiness(smoke_row(contract=False, failure="contract"), discovered(), None)
    assert outcome["classification"] == "CONTRACT_INCOMPATIBLE"


def test_unavailable_exact_candidate_is_not_a_schema_or_semantic_failure():
    outcome = classify_readiness({"attempt": None, "status": "provider_model_unavailable"},
        discovered(result="EXACT_MODEL_UNAVAILABLE"), None)
    assert outcome["classification"] == "EXACT_MODEL_UNAVAILABLE"


@pytest.mark.parametrize("candidate", OPENROUTER_CANDIDATES[:1] + Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_selected_catalog_fields_cannot_persist_known_secret_values(candidate):
    secret, operator = "secretlowercasecredential", "123456789"
    client = Client(cards=[{"id": candidate.model_id,
        "supported_parameters": ["response_format", secret],
        "pricing": {"prompt": "0", "completion": operator},
        "private_key": secret, "raw_headers": {"Authorization": "Bearer " + secret}}])
    result = discover_models(client, candidates=(candidate,), secrets=(secret,), sensitive_values=(operator,))
    assert result["candidates"][0]["discovery_result"] == "EXACT_MODEL_AVAILABLE"
    assert secret not in json.dumps(result)
    assert operator not in json.dumps(result)
    assert "raw_headers" not in json.dumps(result)


@pytest.fixture
def offline_manifest(monkeypatch):
    monkeypatch.setattr(run_smoke, "comparison_manifest", lambda: {
        "repository_commit": "offline-fixture", "qualification_claim": False})


def assert_zero_science(row):
    if row["attempt"] is not None:
        assert all(value == 0 for value in row["attempt"]["safety"].values())


def test_runner_one_shared_catalog_three_identical_interpreter_contracts(tmp_path, offline_manifest):
    client = Client()
    output = tmp_path / "openrouter"
    report = run_openrouter_smoke.smoke(output=output, client=client,
        environment={"OPENROUTER_API_KEY": "private-openrouter-key"})
    assert report["shared_catalog_requests"] == len(client.catalog_calls) == 1
    assert report["total_generation_requests"] == len(client.completion_calls) == 3
    assert report["retries"] == 0
    assert client.closed is True
    assert report["scientific_calls"] == report["execution_entry_attempts"] == report["scientific_step_results"] == 0
    assert report["reconstruction_entries"] == 0
    prompt_ids, schema_ids = set(), set()
    for row, candidate, call in zip(report["candidates"], OPENROUTER_CANDIDATES, client.completion_calls):
        assert row["classification"] == "READY_FOR_Q2"
        assert row["generation_requests"] == 1 and row["discovery_requests"] == 0
        assert row["candidate"]["provider_id"] == "openrouter"
        assert row["candidate"]["model_id"] == candidate.model_id
        assert row["attempt"]["contract_success"] is row["attempt"]["semantic_success"] is True
        prompt_ids.add(row["attempt"]["calls"][0]["prompt_fingerprint"])
        schema_ids.add(row["attempt"]["calls"][0]["schema_fingerprint"])
        assert call["response_format"]["type"] == "json_schema"
        assert call["response_format"]["json_schema"]["strict"] is True
        assert_zero_science(row)
    assert len(prompt_ids) == len(schema_ids) == 1
    assert "private-openrouter-key" not in json.dumps(report)
    assert json.loads((output / "results.json").read_text()) == report


def test_q1e_runner_uses_one_catalog_and_the_same_frozen_inspection_input(tmp_path, offline_manifest):
    baseline_client = Client()
    baseline = run_openrouter_smoke.smoke(output=tmp_path / "baseline", client=baseline_client,
        candidates=OPENROUTER_CANDIDATES[:1], environment={})["candidates"][0]["attempt"]
    client = Client(cards=[{"id": c.model_id, "supported_parameters": ["response_format", "structured_outputs"],
        "pricing": {"prompt": "0", "completion": "0"}} for c in Q1E_CANDIDATES])
    output = tmp_path / "q1e"
    secret = "private-q1e-openrouter-key"
    report = run_openrouter_smoke.smoke(output=output, candidates=Q1E_CANDIDATES, client=client,
        environment={"OPENROUTER_API_KEY": secret})
    assert report["shared_catalog_requests"] == len(client.catalog_calls) == 1
    assert report["total_generation_requests"] == len(client.completion_calls) == 2
    assert report["retries"] == 0 and report["q2_run"] is False
    assert len(report["candidates"]) == 2 and client.closed is True
    for row, candidate, call in zip(report["candidates"], Q1E_CANDIDATES, client.completion_calls):
        assert row["candidate"]["model_id"] == call["model"] == candidate.model_id
        assert row["discovery"]["discovery_result"] == "EXACT_MODEL_AVAILABLE"
        assert row["discovery"]["pricing"] == {"prompt": "0", "completion": "0"}
        assert row["classification"] == "READY_FOR_Q2"
        assert row["generation_requests"] == 1 and row["discovery_requests"] == row["retries"] == 0
        attempt = row["attempt"]
        assert attempt["contract_success"] is attempt["semantic_success"] is True
        assert attempt["context_fingerprint"] == baseline["context_fingerprint"]
        for key in ("prompt_fingerprint", "schema_fingerprint", "registry_fingerprint"):
            assert attempt["calls"][0][key] == baseline["calls"][0][key]
        assert call["messages"] == baseline_client.completion_calls[0]["messages"]
        assert call["response_format"] == baseline_client.completion_calls[0]["response_format"]
        assert_zero_science(row)
    assert secret not in json.dumps(report)
    assert json.loads((output / "results.json").read_text()) == report
    assert (output / ".gitignore").read_text() == "*\n"


@pytest.mark.parametrize("missing", Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_q1e_absent_exact_id_never_generates_for_paid_variant_or_advertised_alias(tmp_path, offline_manifest, missing):
    other = next(c for c in Q1E_CANDIDATES if c != missing)
    client = Client(cards=[{"id": other.model_id},
        {"id": missing.model_id.removesuffix(":free"), "aliases": [missing.model_id]},
        {"id": "openrouter/free", "aliases": [missing.model_id]}])
    report = run_openrouter_smoke.smoke(output=tmp_path / "q1e", candidates=Q1E_CANDIDATES,
        client=client, environment={})
    assert len(client.catalog_calls) == 1 and len(client.completion_calls) == 1
    assert client.completion_calls[0]["model"] == other.model_id
    for row in report["candidates"]:
        if row["candidate"]["candidate_id"] == missing.candidate_id:
            assert row["classification"] == "EXACT_MODEL_UNAVAILABLE"
            assert row["discovery"]["availability"]["provider_model_id"] is None
            assert row["generation_requests"] == 0 and row["attempt"] is None
        else:
            assert row["classification"] == "READY_FOR_Q2"
            assert row["generation_requests"] == 1
        assert_zero_science(row)


@pytest.mark.parametrize("status,message,classification", [
    (400, "json_schema response format is unsupported", "CONTRACT_INCOMPATIBLE"),
    (429, "Rate limit exceeded", "OPERATIONALLY_BLOCKED"),
])
def test_q1e_provider_failure_stops_after_one_call_per_candidate_and_redacts_credentials(
        tmp_path, offline_manifest, status, message, classification):
    secret = "private-q1e-openrouter-key"
    client = Client(cards=[{"id": c.model_id} for c in Q1E_CANDIDATES],
        completion_error=SDKError(status, message + "; API key=" + secret))
    report = run_openrouter_smoke.smoke(output=tmp_path / "q1e", candidates=Q1E_CANDIDATES,
        client=client, environment={"OPENROUTER_API_KEY": secret})
    assert len(client.catalog_calls) == 1 and len(client.completion_calls) == 2
    for row in report["candidates"]:
        assert row["classification"] == classification
        assert row["generation_requests"] == 1 and row["retries"] == 0
        assert row["attempt"]["semantic_success"] is None
        assert row["diagnostic"]["http_status"] == status
        assert_zero_science(row)
    assert secret not in json.dumps(report)
    for artifact in (tmp_path / "q1e").rglob("*.json"):
        assert secret not in artifact.read_text()


def test_runner_missing_exact_model_skips_only_that_candidate(tmp_path, offline_manifest):
    client = Client(cards=[{"id": c.model_id} for c in OPENROUTER_CANDIDATES[1:]])
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client, environment={})
    first, second, third = report["candidates"]
    assert first["classification"] == "EXACT_MODEL_UNAVAILABLE"
    assert first["generation_requests"] == 0 and first["attempt"] is None
    assert second["classification"] == third["classification"] == "READY_FOR_Q2"
    assert len(client.catalog_calls) == 1 and len(client.completion_calls) == 2
    for row in report["candidates"]:
        assert_zero_science(row)


def test_runner_rate_limit_stops_each_candidate_without_retry(tmp_path, offline_manifest):
    client = Client(completion_error=SDKError(429, "Rate limit exceeded", "rate_limited"))
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client, environment={})
    assert len(client.catalog_calls) == 1 and len(client.completion_calls) == 3
    for row in report["candidates"]:
        assert row["classification"] == "OPERATIONALLY_BLOCKED"
        assert row["failure_category"] == "operational"
        assert row["generation_requests"] == 1 and row["retries"] == 0
        assert row["attempt"]["contract_success"] is None
        assert row["diagnostic"]["http_status"] == 429
        assert_zero_science(row)


def test_runner_strict_schema_rejection_is_not_semantic_failure(tmp_path, offline_manifest):
    client = Client(completion_error=SDKError(400, "json_schema response format is unsupported", "unsupported_format"))
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client, environment={})
    assert len(client.completion_calls) == 3
    for row in report["candidates"]:
        assert row["classification"] == "CONTRACT_INCOMPATIBLE"
        assert row["failure_category"] == "contract"
        assert row["attempt"]["semantic_success"] is None
        assert row["generation_requests"] == 1 and row["retries"] == 0
        assert_zero_science(row)


def test_runner_catalog_failure_causes_zero_generation(tmp_path, offline_manifest):
    client = Client(catalog_error=SDKError(401, "API key invalid", "authentication_error"))
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client, environment={})
    assert len(client.catalog_calls) == 1 and client.completion_calls == []
    assert all(row["classification"] == "EXTERNAL_CONFIGURATION_REQUIRED" for row in report["candidates"])
    assert all(row["generation_requests"] == 0 for row in report["candidates"])


def test_existing_output_refusal_happens_before_catalog_request(tmp_path, offline_manifest):
    output = tmp_path / "existing"
    output.mkdir()
    preserved = output / "results.json"
    preserved.write_text("preserved")
    client = Client()
    with pytest.raises(FileExistsError):
        run_openrouter_smoke.smoke(output=output, client=client, environment={})
    assert client.catalog_calls == client.completion_calls == []
    assert preserved.read_text() == "preserved"


def test_cli_requires_live_flag_before_files_or_requests(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("OpenRouter was invoked without --live")
    monkeypatch.setattr(run_openrouter_smoke, "smoke", forbidden)
    output = tmp_path / "no-live"
    with pytest.raises(SystemExit) as exc:
        run_openrouter_smoke.main(["--output", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("metadata", [
    {"requested": OPENROUTER_CANDIDATES[0].model_id, "pipeline": [{"id": "response-healing",
        "private_metadata": "never-record-pipeline-payload"}], "attempt": 1},
    {"requested": "different-model:free", "pipeline": [], "attempt": 1},
    {"requested": OPENROUTER_CANDIDATES[0].model_id, "pipeline": [], "attempt": 2},
])
def test_observed_pipeline_or_model_substitution_is_not_admitted_as_ready(tmp_path, offline_manifest, metadata):
    client = Client(metadata=metadata)
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client,
        candidates=OPENROUTER_CANDIDATES[:1], environment={})
    row = report["candidates"][0]
    assert row["attempt"]["contract_success"] is True
    assert row["classification"] == "EXTERNAL_CONFIGURATION_REQUIRED"
    assert row["generation_requests"] == 1
    assert len(client.completion_calls) == 1
    assert "never-record-pipeline-payload" not in json.dumps(report)
    assert "different-model:free" not in json.dumps(report)
    assert_zero_science(row)


def test_safe_routing_metadata_is_only_bounded_facts(tmp_path, offline_manifest):
    client = Client(metadata={"requested": OPENROUTER_CANDIDATES[0].model_id,
        "pipeline": [], "attempt": 1, "account": "never-record-account", "raw_key": "never-record-key"})
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client,
        candidates=OPENROUTER_CANDIDATES[:1], environment={})
    row = report["candidates"][0]
    assert row["classification"] == "READY_FOR_Q2"
    assert row["routing_observation"] == {"requested_model_matches": True, "pipeline_modified": False,
        "upstream_attempt": 1}
    assert "never-record-account" not in json.dumps(report)
    assert "never-record-key" not in json.dumps(report)


def test_missing_credential_stops_before_sdk_construction_and_discovery(tmp_path, offline_manifest, monkeypatch):
    import openai
    def forbidden(**kwargs):
        pytest.fail("SDK constructed before required credential was present")
    monkeypatch.setattr(openai, "OpenAI", forbidden)
    report = run_openrouter_smoke.smoke(output=tmp_path / "openrouter", environment={})
    assert report["shared_catalog_requests"] == report["total_generation_requests"] == 0
    assert all(row["classification"] == "EXTERNAL_CONFIGURATION_REQUIRED" for row in report["candidates"])
    assert all(row["attempt"] is None for row in report["candidates"])


@pytest.mark.parametrize("candidate", OPENROUTER_CANDIDATES[:1] + Q1E_CANDIDATES, ids=lambda c: c.candidate_id)
def test_openrouter_path_keeps_executor_entry_guard_immediate(tmp_path, offline_manifest, candidate):
    from agent.orchestration import PlanExecutor, build_default_tool_registry
    from benchmarks.interactive.harness import ZeroScienceViolation
    client = Client(cards=[{"id": candidate.model_id}])
    def attack(**kwargs):
        PlanExecutor(build_default_tool_registry()).execute(None)
        pytest.fail("Scientific execution entry guard did not fire")
    client.chat.completions.create = attack
    with pytest.raises(ZeroScienceViolation):
        run_openrouter_smoke.smoke(output=tmp_path / "openrouter", client=client,
            candidates=(candidate,), environment={})
    assert client.closed is True
