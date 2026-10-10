"""Offline paid exact-model admission using the existing OpenRouter transport."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from agent.orchestration.planning_model import PlanningModelError
from benchmarks.interactive import harness, run_breadth
from benchmarks.interactive.candidates import Availability, Candidate
from benchmarks.interactive.openrouter import (
    OPENROUTER_CANDIDATES, OpenRouterPolicyViolation, _candidates,
    build_openrouter_candidate, discover_models,
)
from benchmarks.interactive.scenarios import scenario_by_id
from benchmarks.interactive import run_frontier


FRONTIER = Candidate(
    "openrouter-gpt-5.6-sol", "openrouter", "openai/gpt-5.6-sol",
    "openai-chat-completions", "OPENROUTER_API_KEY", "f1-openrouter-gpt56-sol",
    "https://openrouter.ai/api/v1",
)
PAID_POLICY = {
    "provider_max_price": {"prompt": 3.0, "completion": 18.0, "request": 0},
    "max_completion_tokens": 2048,
}
SCHEMA = {
    "type": "object", "properties": {"status": {"type": "string", "enum": ["ok"]}},
    "required": ["status"], "additionalProperties": False,
}
EXECUTE = json.dumps({"turn_schema_version": 1,
    "decision": {"kind": "execute_plan", "target": "inspect_scATAC"}})


def frontier_git(monkeypatch, *, head="a" * 40, branch="main", ancestor=True,
                 cached_origin=None, committed="", working="", tracked="", index=""):
    """Simulate source-equivalent commits using read-only Git path selection."""
    refs = {"HEAD": head, "main": head, "origin/main": cached_origin or head}
    frozen_names = ("benchmarks/interactive/openrouter.py", "benchmarks/interactive/harness.py",
        "benchmarks/interactive/scenarios.py", "benchmarks/interactive/run_breadth.py",
        "benchmarks/interactive/README.md")
    def git(*args):
        if args[:1] == ("rev-parse",):
            return refs[args[1]]
        if args == ("branch", "--show-current"):
            return branch
        if args[0] == "merge-base":
            assert args[1:] == (run_frontier.BASELINE, head)
            return run_frontier.BASELINE if ancestor else "b" * 40
        if args[0] == "ls-tree":
            assert run_frontier.BASELINE in args
            return "\n".join(frozen_names)
        if args == ("diff", "--name-only"):
            return tracked
        if args == ("diff", "--cached", "--name-only"):
            return index
        if args[:3] == ("diff", "--name-only", run_frontier.BASELINE):
            assert "src/agent" in args and "benchmarks/planner" in args
            assert "benchmarks/interactive/harness.py" in args
            assert "benchmarks/interactive/scenarios.py" in args
            assert "benchmarks/interactive/openrouter.py" not in args
            paths = args[args.index("--") + 1:]
            changed = working if args[3] == "--" else committed
            return "\n".join(name for name in changed.splitlines() if
                any(name == path or name.startswith(path + "/") for path in paths))
        pytest.fail(f"Unexpected frontier baseline Git read: {args!r}")
    monkeypatch.setattr(run_breadth, "_git", git)
    return refs


def test_frontier_experiment_baseline_survives_source_equivalent_descendant_commit(monkeypatch):
    refs = frontier_git(monkeypatch,
        committed="benchmarks/interactive/openrouter.py\nbenchmarks/interactive/run_breadth.py")
    result = run_frontier.baseline(require_clean=True)
    assert result["refs"] == refs
    assert result["refs"]["HEAD"] != run_frontier.BASELINE
    assert result["experiment_baseline"] == run_frontier.BASELINE
    assert result["tracked_clean"] is result["index_clean"] is True


@pytest.mark.parametrize("pending", ["benchmarks/interactive/README.md",
    "benchmarks/interactive/openrouter.py", "tests/benchmarks/test_interactive_frontier_transport.py"])
def test_frontier_offline_review_accepts_pending_infrastructure_but_live_requires_clean_tree(
        monkeypatch, pending):
    frontier_git(monkeypatch, working=pending, tracked=pending, index=pending)
    result = run_frontier.baseline(require_clean=False)
    assert result["tracked_clean"] is result["index_clean"] is False
    with pytest.raises(ValueError, match="clean tracked tree and index"):
        run_frontier.baseline(require_clean=True)


@pytest.mark.parametrize("drift", [
    {"ancestor": False}, {"branch": "other"}, {"cached_origin": "b" * 40},
    {"working": "src/agent/orchestration/semantic_prompt.py", "tracked":
        "src/agent/orchestration/semantic_prompt.py"},
    {"committed": "benchmarks/interactive/scenarios.py"},
])
def test_frontier_baseline_rejects_unrelated_or_semantically_changed_checkout_even_offline(
        monkeypatch, drift):
    frontier_git(monkeypatch, **drift)
    with pytest.raises(ValueError):
        run_frontier.baseline(require_clean=False)


def test_frontier_live_runner_checks_clean_checkout_before_reading_or_creating_evidence(tmp_path, monkeypatch):
    observed = []
    def reject(*, require_clean=False):
        observed.append(require_clean)
        raise ValueError("Live qualification requires a clean tracked tree and index.")
    monkeypatch.setattr(run_frontier, "baseline", reject)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="clean tracked tree and index"):
        run_frontier.run(output=output, live=True)
    assert observed == [True]
    assert not output.exists()


class SDKError(Exception):
    def __init__(self, secret):
        self.status_code = 400
        self.body = {"error": {"message": "Unsupported json_schema; API key=" + secret,
            "code": "unsupported_format"}}
        self.response = SimpleNamespace(status_code=400, headers={"Authorization": secret})
        super().__init__("Private raw error: " + secret)


class Client:
    """Observable injected SDK fixture; every request remains offline."""
    def __init__(self, *, cards=None, content='{"status":"ok"}',
                 returned_model=FRONTIER.model_id, usage=None, metadata=None, error=None):
        self.cards = cards if cards is not None else [{
            "id": FRONTIER.model_id,
            "supported_parameters": ["response_format", "structured_outputs"],
            "pricing": {"prompt": "0.000003", "completion": "0.000018", "request": "0"},
            "context_length": 131072,
            "top_provider": {"context_length": 131072, "max_completion_tokens": 8192},
        }]
        self.content, self.returned_model = content, returned_model
        self.usage, self.metadata, self.error = usage, metadata, error
        self.catalog_calls, self.completion_calls = [], []
        self.models = SimpleNamespace(list=self.list_models)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

    def list_models(self, **kwargs):
        self.catalog_calls.append(kwargs)
        return {"data": self.cards}

    def complete(self, **kwargs):
        self.completion_calls.append(deepcopy(kwargs))
        if self.error:
            raise self.error
        response = {"choices": [{"finish_reason": "stop",
            "message": {"content": self.content}}]}
        if self.returned_model is not None:
            response["model"] = self.returned_model
        if self.usage is not None:
            response["usage"] = self.usage
        if self.metadata is not None:
            response["openrouter_metadata"] = self.metadata
        return response


def availability(candidate=FRONTIER):
    return Availability("available", "CANDIDATE_MODEL_AVAILABLE", candidate.model_id,
        "models.list", candidate.model_id)


def runtime(client, **kwargs):
    return build_openrouter_candidate(FRONTIER, availability=availability(), client=client,
        environment={}, admitted_candidates=(FRONTIER,), paid_policy=deepcopy(PAID_POLICY), **kwargs)


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        pytest.fail("A paid transport test attempted a real network connection.")
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_frontier_candidate_preserves_exact_requested_identity_and_requires_allowlist():
    assert FRONTIER.model_id == "openai/gpt-5.6-sol"
    assert FRONTIER not in OPENROUTER_CANDIDATES
    with pytest.raises(ValueError):
        _candidates((FRONTIER,))
    assert _candidates((FRONTIER,), admitted_candidates=(FRONTIER,)) == (FRONTIER,)
    other = replace(FRONTIER, candidate_id="another-paid-candidate", model_id="openai/other-model")
    with pytest.raises(ValueError):
        _candidates((other,), admitted_candidates=(FRONTIER,))
    assert _candidates(OPENROUTER_CANDIDATES) == OPENROUTER_CANDIDATES


@pytest.mark.parametrize("change", [
    {"provider_id": "other-provider"}, {"adapter_family": "openai-responses"},
    {"endpoint": "https://different.example/api/v1"}, {"credential_env": "OTHER_API_KEY"},
])
def test_explicit_allowlist_cannot_admit_another_transport(change):
    candidate = replace(FRONTIER, **change)
    with pytest.raises(ValueError):
        _candidates((candidate,), admitted_candidates=(candidate,))


def test_new_exact_candidate_requires_paid_policy_before_sdk_dispatch():
    client = Client()
    with pytest.raises(ValueError):
        build_openrouter_candidate(FRONTIER, availability=availability(), client=client,
            environment={}, admitted_candidates=(FRONTIER,))
    assert client.catalog_calls == client.completion_calls == []


def test_paid_discovery_retains_exact_capabilities_prices_and_limits_without_generation():
    client = Client()
    report = discover_models(client, candidates=(FRONTIER,), admitted_candidates=(FRONTIER,))
    row = report["candidates"][0]
    assert report["catalog_requests"] == len(client.catalog_calls) == 1
    assert client.completion_calls == []
    assert row["discovery_result"] == "EXACT_MODEL_AVAILABLE"
    assert row["availability"]["provider_model_id"] == FRONTIER.model_id
    assert row["supported_parameters"] == ["response_format", "structured_outputs"]
    assert row["pricing"] == {"prompt": "0.000003", "completion": "0.000018", "request": "0"}
    assert row["limits"] == {"context_length": 131072, "top_provider_context_length": 131072,
        "top_provider_max_completion_tokens": 8192}


@pytest.mark.parametrize("replacement", [
    "openai/gpt-5.6-sol-pro", "openai/gpt-5.6-sol:latest", "openrouter/auto", "openai/other-model",
])
def test_paid_exact_discovery_never_uses_an_alias_or_substitute(replacement):
    client = Client(cards=[{"id": replacement, "aliases": [FRONTIER.model_id]}])
    row = discover_models(client, candidates=(FRONTIER,),
        admitted_candidates=(FRONTIER,))["candidates"][0]
    assert row["discovery_result"] == "EXACT_MODEL_UNAVAILABLE"
    assert row["availability"]["provider_model_id"] is None
    assert client.completion_calls == []


def test_paid_transport_keeps_frozen_prompt_strict_schema_and_exact_routing_with_output_cap():
    client = Client()
    model = runtime(client)
    assert model.complete(prompt="Frozen provider-neutral prompt.", response_schema=SCHEMA) == '{"status":"ok"}'
    assert len(client.completion_calls) == model.provider_completion_calls == 1
    call = client.completion_calls[0]
    assert call["model"] == FRONTIER.model_id
    assert call["messages"] == [{"role": "user", "content": "Frozen provider-neutral prompt."}]
    assert call["response_format"] == {"type": "json_schema",
        "json_schema": {"name": "agent_plan", "strict": True, "schema": SCHEMA}}
    assert call["max_completion_tokens"] == PAID_POLICY["max_completion_tokens"]
    assert call["extra_body"]["provider"] == {"require_parameters": True,
        "allow_fallbacks": False, "max_price": PAID_POLICY["provider_max_price"]}
    assert call["extra_body"]["plugins"] == [{"id": name, "enabled": False}
        for name in ("response-healing", "context-compression", "web")]
    assert "tools" not in call and "reasoning" not in call
    assert model.response_model_observation == {"present": True,
        "matches_requested": True, "model_id": FRONTIER.model_id}


def test_existing_free_transport_retains_zero_price_envelope_without_paid_output_setting():
    candidate = OPENROUTER_CANDIDATES[0]
    client = Client(returned_model=candidate.model_id)
    model = build_openrouter_candidate(candidate, availability=availability(candidate),
        client=client, environment={})
    model.complete(prompt="Unchanged free prompt.", response_schema=SCHEMA)
    call = client.completion_calls[0]
    assert call["extra_body"]["provider"] == {"require_parameters": True,
        "allow_fallbacks": False, "max_price": {"prompt": 0, "completion": 0, "request": 0}}
    assert "max_completion_tokens" not in call
    assert call["model"] == candidate.model_id


def test_existing_free_candidate_cannot_use_a_paid_policy():
    candidate = OPENROUTER_CANDIDATES[0]
    client = Client(returned_model=candidate.model_id)
    with pytest.raises(ValueError):
        build_openrouter_candidate(candidate, availability=availability(candidate),
            client=client, environment={}, paid_policy=PAID_POLICY)
    assert client.completion_calls == []


def test_paid_client_construction_disables_sdk_retries_and_keeps_credential_private(monkeypatch):
    import openai
    observed = []
    client = Client()
    def construct(**kwargs):
        observed.append(kwargs)
        return client
    monkeypatch.setattr(openai, "OpenAI", construct)
    secret = "private-frontier-sdk-key"
    model = build_openrouter_candidate(FRONTIER, availability=availability(),
        environment={"OPENROUTER_API_KEY": secret}, admitted_candidates=(FRONTIER,),
        paid_policy=PAID_POLICY)
    assert observed == [{"api_key": secret, "base_url": "https://openrouter.ai/api/v1", "max_retries": 0}]
    assert secret not in json.dumps(model.manifest())
    assert client.catalog_calls == client.completion_calls == []


@pytest.mark.parametrize("policy", [
    {}, {"provider_max_price": {"prompt": 3, "completion": 18}},
    {"provider_max_price": {"prompt": 3}, "max_completion_tokens": 2048},
    {"provider_max_price": {"prompt": 3, "completion": -1}, "max_completion_tokens": 2048},
    {"provider_max_price": {"prompt": float("nan"), "completion": 18}, "max_completion_tokens": 2048},
    {"provider_max_price": {"prompt": 3, "completion": float("inf")}, "max_completion_tokens": 2048},
    {"provider_max_price": {"prompt": True, "completion": 18}, "max_completion_tokens": 2048},
    {"provider_max_price": {"prompt": 3, "completion": 18}, "max_completion_tokens": 0},
    {"provider_max_price": {"prompt": 3, "completion": 18}, "max_completion_tokens": True},
    {"provider_max_price": {"prompt": 3, "completion": 18}, "max_completion_tokens": 2048,
        "allow_fallbacks": True},
])
def test_invalid_paid_limits_fail_before_any_sdk_call(policy):
    client = Client()
    with pytest.raises(ValueError):
        build_openrouter_candidate(FRONTIER, availability=availability(), client=client,
            environment={}, admitted_candidates=(FRONTIER,), paid_policy=policy)
    assert client.catalog_calls == client.completion_calls == []


def test_pre_dispatch_callback_validates_complete_transport_envelope_before_provider_call():
    client = Client()
    observed = []
    def check(*, request):
        assert client.completion_calls == []
        assert request["model"] == FRONTIER.model_id
        assert request["max_completion_tokens"] == PAID_POLICY["max_completion_tokens"]
        assert request["response_format"]["json_schema"]["strict"] is True
        observed.append(deepcopy(request))
    model = runtime(client, before_dispatch=check)
    model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert observed == client.completion_calls


def test_pre_dispatch_rejection_is_a_hard_stop_without_provider_call_or_retry():
    client = Client()
    class BudgetStop(BaseException):
        pass
    def reject(*, request):
        raise BudgetStop("Offline budget exhausted.")
    model = runtime(client, before_dispatch=reject)
    with pytest.raises(BudgetStop):
        model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert client.completion_calls == []
    assert model.provider_completion_calls == 0


def test_paid_single_call_bound_prevents_second_provider_dispatch():
    client = Client()
    model = runtime(client)
    model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    with pytest.raises(PlanningModelError):
        model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert len(client.completion_calls) == model.provider_completion_calls == 1


@pytest.mark.parametrize("returned", [None, "openai/gpt-5.6-sol-pro", "openai/other-model",
    "private-model-credential"])
def test_paid_returned_model_mismatch_hard_stops_and_redacts_secret_identity(returned):
    client = Client(returned_model=returned)
    model = runtime(client, secrets=("private-model-credential",))
    with pytest.raises(OpenRouterPolicyViolation):
        model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert len(client.completion_calls) == 1
    assert model.response_model_observation["matches_requested"] is False
    assert "private-model-credential" not in json.dumps(model.response_model_observation)


@pytest.mark.parametrize("metadata", [
    {"requested": "openai/other-model", "pipeline": [], "attempt": 1},
    {"requested": FRONTIER.model_id, "pipeline": ["response-healing"], "attempt": 1},
    {"requested": FRONTIER.model_id, "pipeline": [], "attempt": 2},
    {"requested": FRONTIER.model_id, "pipeline": [], "attempt": "private-value"},
    "malformed-routing-metadata",
])
def test_paid_modified_pipeline_or_upstream_fallback_hard_stops(metadata):
    client = Client(metadata=metadata)
    model = runtime(client)
    with pytest.raises(OpenRouterPolicyViolation):
        model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert len(client.completion_calls) == 1


def test_usage_retains_only_actual_bounded_provider_counts_cost_and_details():
    client = Client(usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
        "cost": "0.0009", "prompt_tokens_details": {"cached_tokens": 20, "private": "secret"},
        "completion_tokens_details": {"reasoning_tokens": 10, "private": "secret"},
        "Authorization": "secret", "account": "secret"})
    model = runtime(client)
    model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert model.usage_observation == {"prompt_tokens": 120, "completion_tokens": 30,
        "total_tokens": 150, "cost": "0.0009", "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 10}}
    assert "secret" not in json.dumps(model.usage_observation)


def test_absent_usage_does_not_infer_cost_or_tokens():
    client = Client()
    model = runtime(client)
    model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert model.usage_observation is None


def test_invalid_or_secret_usage_and_catalog_fields_never_become_evidence():
    secret, numeric_private = "privatesecretcredential", "987654321"
    client = Client(usage={"prompt_tokens": True, "completion_tokens": -1,
        "total_tokens": 987654321, "cost": float("nan"),
        "completion_tokens_details": {"reasoning_tokens": secret}})
    client.cards[0].update({"private": secret, "context_length": 987654321,
        "supported_parameters": ["response_format", secret],
        "pricing": {"prompt": secret, "completion": numeric_private, "request": "0"}})
    discovery = discover_models(client, candidates=(FRONTIER,), admitted_candidates=(FRONTIER,),
        secrets=(secret,), sensitive_values=(numeric_private,))
    model = runtime(client, secrets=(secret,), sensitive_values=(numeric_private,))
    model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    evidence = json.dumps({"discovery": discovery, "usage": model.usage_observation})
    assert secret not in evidence and numeric_private not in evidence
    assert model.usage_observation is None
    assert discovery["candidates"][0]["pricing"] == {"request": "0"}


def test_paid_error_diagnostics_strip_credentials_without_additional_provider_call():
    secret = "private-frontier-key"
    client = Client(error=SDKError(secret))
    model = runtime(client, secrets=(secret,))
    with pytest.raises(PlanningModelError):
        model.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert len(client.completion_calls) == 1
    assert model.transport_diagnostic["http_status"] == 400
    assert secret not in json.dumps(model.transport_diagnostic)
    assert secret not in json.dumps(model.manifest())


def test_paid_supported_inspection_smoke_reuses_frozen_contract_with_zero_science():
    smoke = replace(scenario_by_id("I01"), surface="interpreter",
        expected={"kind": "execute", "tool": "inspect_scATAC"})
    checks = []
    check = run_breadth._payload_guard(smoke, checks)
    def before_dispatch(*, request):
        check(prompt=request["messages"][0]["content"],
            response_schema=request["response_format"]["json_schema"]["schema"])
    client = Client(content=EXECUTE)
    model = runtime(client, before_dispatch=before_dispatch)
    attempt = harness.run_attempt(smoke, FRONTIER, model=model, attempt_id="offline-frontier-smoke").to_dict()
    assert attempt["contract_success"] is True
    assert attempt["semantic_success"] is True
    assert attempt["admission_success"] is True
    assert attempt["decision_kind"] == "execute"
    assert attempt["selected_tool"] == "inspect_scATAC"
    assert not any(attempt["safety"].values())
    assert len(client.completion_calls) == len(checks) == 1
    assert checks[0]["validated_before_sdk"] is True
    assert checks[0]["prompt_fingerprint"] == "7a4d10f3e530a43408c8a3d5f19caae11cad918ae976bb11172876dab581663f"
    assert checks[0]["schema_fingerprint"] == "795eb53cedd8983ac93d93ea5ca9fc342f92df010d5c5a21b8c29310fc7eeafb"


def spending_request(content="Frozen prompt."):
    return {
        "model": run_frontier.CANDIDATE.model_id,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "agent_plan", "strict": True, "schema": deepcopy(SCHEMA)}},
        "max_completion_tokens": run_frontier.OUTPUT_LIMIT,
        "extra_body": {"provider": {"require_parameters": True, "allow_fallbacks": False,
            "max_price": {"prompt": 2, "completion": 10, "request": 0}}},
    }


def test_frontier_run_matrix_is_only_single_smoke_and_four_canonical_attempts():
    plan = run_frontier.validate_matrix()
    assert run_frontier.SCHEDULE == ("admission", "I01", "I10", "I08", "I14")
    assert plan == {"admission_smokes": 1, "semantic_attempts": 4,
        "expected_completions": 7, "conservative_completion_ceiling": 9,
        "schedule": ["admission", "I01", "I10", "I08", "I14"]}
    assert run_frontier.CANDIDATE.model_id == "openai/gpt-5.6-sol"
    assert run_frontier.CANDIDATE.provider_id == "openrouter"


@pytest.mark.parametrize("schedule", [
    (), ("admission",), ("I01", "I10", "I08", "I14"),
    ("admission", "I01", "I10", "I08"),
    ("admission", "I01", "I10", "I08", "I14", "I02"),
    ("admission", "I01", "I10", "I08", "I14", "I14"),
    ("admission", "I01", "I08", "I10", "I14"),
    ("admission", "I01", "I10", "I08", "I12"),
    ("admission", "I01", "I10", "I08", "I13"),
    ("admission", *(f"I{i:02d}" for i in range(1, 15))),
])
def test_frontier_matrix_rejects_repeats_deferred_cases_omissions_and_full_breadth(schedule):
    with pytest.raises(ValueError):
        run_frontier.validate_matrix(schedule)


@pytest.mark.parametrize("ceiling", [Decimal(0), Decimal("-0.01"), Decimal("1.01"),
    Decimal("NaN"), Decimal("Infinity")])
def test_spending_ceiling_rejects_invalid_or_over_one_dollar_values(ceiling):
    with pytest.raises(ValueError):
        run_frontier.SpendingGuard(ceiling=ceiling)


def test_spending_guard_requires_exact_case_admission_and_single_schedule():
    guard = run_frontier.SpendingGuard()
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case="admission")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.begin("I01")
    guard.begin("admission")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.begin("admission")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case="I01")
    assert guard.requests == [] and guard.reserved == 0
    for case in run_frontier.SELECTED:
        guard.begin(case)
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.begin("I01")
    assert guard.started == list(run_frontier.SCHEDULE)


@pytest.mark.parametrize("mutation", ["model", "schema_type", "strict", "output_limit",
    "fallback", "required_parameters", "price"])
def test_spending_guard_rejects_transport_drift_before_reservation(mutation):
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    request = spending_request()
    if mutation == "model": request["model"] = "openai/gpt-5.6-sol-pro"
    elif mutation == "schema_type": request["response_format"]["type"] = "json_object"
    elif mutation == "strict": request["response_format"]["json_schema"]["strict"] = False
    elif mutation == "output_limit": request["max_completion_tokens"] += 1
    elif mutation == "fallback": request["extra_body"]["provider"]["allow_fallbacks"] = True
    elif mutation == "required_parameters": request["extra_body"]["provider"]["require_parameters"] = False
    else: request["extra_body"]["provider"]["max_price"]["completion"] = 11
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=request, case="admission")
    assert guard.requests == [] and guard.reserved == 0


def test_input_bound_counts_utf8_bytes_and_full_response_schema():
    request = spending_request("汉😀")
    request["response_format"]["json_schema"]["schema"]["description"] = "细胞"
    plain = json.dumps({"messages": request["messages"],
        "response_format": request["response_format"]}, ensure_ascii=False, separators=(",", ":"))
    assert run_frontier.input_token_bound(request) == len(plain.encode("utf-8")) + 2048
    ascii_request = spending_request("aa")
    assert run_frontier.input_token_bound(request) > run_frontier.input_token_bound(ascii_request)
    large_schema = spending_request()
    large_schema["response_format"]["json_schema"]["schema"]["description"] = "汉" * 10000
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=large_schema, case="admission")
    assert guard.requests == []


def test_spending_guard_rejects_oversized_input_before_reservation():
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    request = spending_request("x" * run_frontier.INPUT_BOUND)
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=request, case="admission")
    assert guard.requests == [] and guard.reserved == 0


def test_spending_guard_reserves_worst_case_provider_pricing_before_dispatch():
    request = spending_request()
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    guard.reserve(request=request, case="admission")
    expected = (run_frontier.PROMPT_PRICE * run_frontier.input_token_bound(request)
        + run_frontier.COMPLETION_PRICE * run_frontier.OUTPUT_LIMIT)
    assert guard.reserved == expected
    assert Decimal(guard.requests[0]["reserved_cost_usd"]) == expected
    assert guard.requests[0]["completion_token_upper_bound"] == 4096
    assert guard.requests[0]["input_token_upper_bound"] <= 30000
    assert expected <= Decimal("0.101")
    assert guard.manifest()["remaining_balance_estimate"] is None


def test_spending_guard_aggregate_budget_does_not_reclaim_or_allow_an_extra_call():
    guard = run_frontier.SpendingGuard(ceiling=Decimal("0.05"))
    guard.begin("admission")
    guard.reserve(request=spending_request(), case="admission")
    first = guard.reserved
    guard.begin("I01")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case="I01")
    assert guard.reserved == first
    assert len(guard.requests) == 1
    assert guard.reserved <= guard.ceiling


@pytest.mark.parametrize("case", ["admission", "I10", "I08", "I14"])
def test_spending_guard_each_single_call_surface_has_no_repeat(case):
    guard = run_frontier.SpendingGuard()
    for admitted in run_frontier.SCHEDULE:
        guard.begin(admitted)
        if admitted == case:
            break
    guard.reserve(request=spending_request(), case=case)
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case=case)
    assert len(guard.requests) == 1


def test_spending_guard_i01_preserves_only_production_recovery_ceiling():
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    guard.begin("I01")
    for _ in range(5):
        guard.reserve(request=spending_request(), case="I01")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case="I01")
    assert len(guard.requests) == 5


def test_nine_dispatches_are_absolute_total_ceiling_without_broadening_matrix():
    guard = run_frontier.SpendingGuard()
    for case in run_frontier.SCHEDULE:
        guard.begin(case)
        for _ in range(5 if case == "I01" else 1):
            guard.reserve(request=spending_request(), case=case)
    assert len(guard.requests) == run_frontier.CALL_CEILING == 9
    assert guard.reserved <= Decimal("1.00")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.reserve(request=spending_request(), case="I14")
    with pytest.raises(run_frontier.FrontierHardStop):
        guard.begin("I02")


def test_frontier_budget_hard_stop_bypasses_the_planner_exception_path():
    assert not issubclass(run_frontier.FrontierHardStop, Exception)
    assert not issubclass(OpenRouterPolicyViolation, Exception)


def test_frontier_transport_budget_rejection_reaches_no_provider_dispatch():
    client = Client()
    guard = run_frontier.SpendingGuard(ceiling=Decimal("0.001"))
    guard.begin("admission")
    transport = run_frontier.FrontierTransport(
        availability=availability(run_frontier.CANDIDATE), ledger=guard, case="admission",
        before_complete=lambda **kwargs: None, environment={}, client=client)
    with pytest.raises(run_frontier.FrontierHardStop):
        transport.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert client.completion_calls == []
    assert transport.diagnostics == guard.requests == []
    assert guard.reserved == 0


@pytest.mark.parametrize("usage", [
    {"prompt_tokens": 30001}, {"completion_tokens": 4097}, {"cost": "1.01"},
])
def test_frontier_transport_unexpected_reported_usage_stops_without_another_call(usage):
    client = Client(usage=usage)
    guard = run_frontier.SpendingGuard()
    guard.begin("admission")
    transport = run_frontier.FrontierTransport(
        availability=availability(run_frontier.CANDIDATE), ledger=guard, case="admission",
        before_complete=lambda **kwargs: None, environment={}, client=client)
    with pytest.raises(run_frontier.FrontierHardStop):
        transport.complete(prompt="Frozen prompt.", response_schema=SCHEMA)
    assert len(client.completion_calls) == len(guard.requests) == 1
    assert transport.diagnostics[0]["usage"] == usage


def valid_frontier_discovery():
    return {
        "candidate": run_frontier.CANDIDATE.manifest(),
        "discovery_result": "EXACT_MODEL_AVAILABLE",
        "matched_exact_model_id": "openai/gpt-5.6-sol", "exact_match_count": 1,
        "supported_parameters": ["response_format", "structured_outputs", "max_completion_tokens"],
        "pricing": {"prompt": "0.000002", "completion": "0.00001", "request": "0"},
    }


def test_frontier_discovery_admission_requires_exact_model_structured_output_and_live_pricing():
    run_frontier.validate_discovery(valid_frontier_discovery())


@pytest.mark.parametrize("change", ["model", "duplicate", "missing", "schema", "output_cap",
    "prompt_price", "completion_price", "request_price"])
def test_frontier_discovery_drift_stops_before_paid_generation(change):
    discovery = valid_frontier_discovery()
    if change == "model": discovery["matched_exact_model_id"] = "openai/gpt-5.6-sol-pro"
    elif change == "duplicate": discovery["exact_match_count"] = 2
    elif change == "missing": discovery["discovery_result"] = "EXACT_MODEL_UNAVAILABLE"
    elif change == "schema": discovery["supported_parameters"].remove("structured_outputs")
    elif change == "output_cap": discovery["supported_parameters"].remove("max_completion_tokens")
    elif change == "prompt_price": discovery["pricing"]["prompt"] = "0.000003"
    elif change == "completion_price": discovery["pricing"]["completion"] = "0.00002"
    else: discovery["pricing"]["request"] = "0.01"
    with pytest.raises(ValueError):
        run_frontier.validate_discovery(discovery)
