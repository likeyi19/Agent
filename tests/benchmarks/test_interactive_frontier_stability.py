"""Offline F2 qualification using frozen Agent contracts and exact paid fake SDKs."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from agent.orchestration import PlanExecutor, build_default_tool_registry
from benchmarks.interactive import harness, run_breadth, run_frontier, run_frontier_stability
from benchmarks.interactive.fixtures import scripted_answer
from benchmarks.interactive.run_stability import _frozen_context
from benchmarks.interactive.scenarios import scenario_by_id

from test_interactive_breadth import SDKError, inspection, scope, turn, witnesses
from test_interactive_frontier_transport import availability, spending_request, valid_frontier_discovery


F2 = run_frontier_stability
EXPECTED_SCHEDULE = tuple((sid, number) for sid in ("I01", "I10", "I08", "I14")
    for number in (2, 3))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("F2 offline tests attempted a real network connection")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(run_breadth, "comparison_manifest", lambda: {
        "repository_commit": run_frontier.BASELINE, "qualification_claim": False})


class PaidSDK:
    """Fresh SDK witness; exact responses carry bounded routing and usage data."""
    def __init__(self, outcomes, *, usage=None):
        self.outcomes = list(outcomes)
        self.calls = []
        self.closed = False
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 20,
            "total_tokens": 120, "cost": "0.0004"}
        self.models = SimpleNamespace(list=self.no_discovery)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def no_discovery(self, **kwargs):
        pytest.fail("F2 attempted to repeat F1 admission or discover another model")

    def close(self):
        self.closed = True

    def create(self, **request):
        self.calls.append(deepcopy(request))
        assert self.outcomes, "F2 added a benchmark retry"
        response = self.outcomes.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(json.loads(request["messages"][0]["content"]))
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": response if isinstance(response, str) else json.dumps(response)}}],
            "model": run_frontier.CANDIDATE.model_id, "usage": deepcopy(self.usage),
            "openrouter_metadata": {"requested": run_frontier.CANDIDATE.model_id,
                "pipeline": [], "attempt": 1}}


class RepeatClients:
    def __init__(self, *, overrides=None, usage=None):
        self.overrides = overrides or {}
        self.usage = usage
        self.created = []

    def __call__(self, *, candidate, case, attempt_number):
        assert candidate == run_frontier.CANDIDATE
        assert (case, attempt_number) == EXPECTED_SCHEDULE[len(self.created)]
        client = PaidSDK(self.overrides.get((case, attempt_number), witnesses(case)), usage=self.usage)
        self.created.append((case, attempt_number, client))
        return client


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def hash_manifest(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"}


@pytest.fixture(scope="module")
def f1_seed(tmp_path_factory):
    """Build historical F1 evidence entirely offline; real ignored evals are unused."""
    directory = tmp_path_factory.mktemp("synthetic-frontier-f1")
    ledger = run_frontier.SpendingGuard()
    rows = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_breadth, "comparison_manifest", lambda: {
            "repository_commit": run_frontier.BASELINE, "qualification_claim": False})
        for case in run_frontier.SCHEDULE:
            ledger.begin(case)
            scenario = run_frontier.smoke_scenario() if case == "admission" else scenario_by_id(case)
            client = PaidSDK([turn(target="inspect_scATAC")] if case == "admission" else witnesses(case))
            checks = []
            runtime = run_frontier.FrontierTransport(
                availability=availability(run_frontier.CANDIDATE), ledger=ledger, case=case,
                before_complete=run_breadth._payload_guard(scenario, checks), environment={}, client=client)
            attempt_id = "f1-admission-1" if case == "admission" else "f1-1"
            try:
                attempt = harness.run_attempt(scenario, run_frontier.CANDIDATE,
                    model=runtime, attempt_id=attempt_id).to_dict()
            finally:
                runtime.close()
            rows.append(dict(case=case, scenario_id=scenario.scenario_id, attempt_id=attempt_id,
                attempt=attempt, diagnostics=runtime.diagnostics, payload_checks=checks,
                status=run_breadth.status_for_attempt(attempt)))
    report = dict(phase="F1", candidate=run_frontier.CANDIDATE.manifest(), results=rows,
        admission_gate_passed=True, hard_stop=None, semantic_attempts=4,
        counts={"provider_calls": 7, "semantic_attempts": 4}, matrix=run_frontier.validate_matrix())
    validation = dict(baseline={"branch": "main", "refs": {
            name: run_frontier.BASELINE for name in ("HEAD", "main", "origin/main")}},
        candidate=run_frontier.CANDIDATE.manifest(), matrix=run_frontier.validate_matrix(),
        frozen_contexts={sid: _frozen_context(scenario_by_id(sid)) for sid in run_frontier.SELECTED},
        execution_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path("benchmarks/interactive/run_frontier.py"), Path("benchmarks/interactive/openrouter.py"))},
        historical_initial_payload_checks=[{"prompt_utf8_bytes": 100, "schema_utf8_bytes": 100}
            for _ in range(6)], historical_ledger_sha256="a" * 64,
        historical_audit_sha256="b" * 64, zero_science_guards_validated=True, provider_calls=0)
    report["offline_validation_fingerprint"] = F2.fingerprint(validation)
    write_json(directory / "results.json", report)
    write_json(directory / "offline_validation.json", validation)
    write_json(directory / "discovery.json", valid_frontier_discovery())
    write_json(directory / "frozen_input_audit.json", {"plan": validation})
    write_json(directory / "closeout.json", {"phase": "F1", "results": rows})
    write_json(directory / "independent_audit.json", {"passed": True})
    write_json(directory / "evidence_manifest.json", {"sha256": hash_manifest(directory)})
    return directory, report, validation


@pytest.fixture
def historical(tmp_path, f1_seed, monkeypatch):
    original, saved, validation = f1_seed
    directory = tmp_path / "f1"
    directory.mkdir()
    for path in original.iterdir():
        (directory / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(run_frontier, "frozen_validation", lambda: deepcopy(validation))
    monkeypatch.setattr(run_frontier, "baseline",
        lambda *, require_clean=False: deepcopy(validation["baseline"]))
    review = tmp_path / "external-review.json"
    answer = next(r for r in saved["results"] if r["case"] == "I08")["attempt"]["explanation"]
    write_json(review, {"scenario_id": "I08", "candidate_id": run_frontier.CANDIDATE.candidate_id,
        "historical_status": "HUMAN_REVIEW_PENDING", "external_human_semantic_review": "PASS",
        "retained_answer": answer, "review_authority": "Synthetic offline external review"})
    return directory, deepcopy(saved), deepcopy(validation), review


def execute(tmp_path, historical, *, factory=None, **kwargs):
    source, _, _, review = historical
    factory = factory or RepeatClients()
    output = tmp_path / "f2"
    output.mkdir()
    write_json(output / "discovery.json", valid_frontier_discovery())
    arguments = dict(output=output, original=source, human_review_path=review,
        environment={"OPENROUTER_API_KEY": "offline-private-credential"}, spacing=0,
        client_factory=factory)
    arguments.update(kwargs)
    F2.run(**arguments, live=False)
    return F2.run(**arguments, live=True), factory


def rows_by_case(report):
    return {sid: [r for r in report["results"] if r["scenario_id"] == sid]
        for sid in run_frontier.SELECTED}


def assert_safe(report):
    for row in report["results"]:
        if row.get("attempt"):
            assert not any(row["attempt"]["safety"].values())
    if "safety" in report:
        assert not any(report["safety"].values())


def test_repeat_matrix_has_one_unchanged_candidate_four_cases_and_exactly_eight_attempts():
    matrix = F2.validate_matrix()
    assert F2.SCHEDULE == EXPECTED_SCHEDULE
    assert F2.CANDIDATE == run_frontier.CANDIDATE
    assert F2.CANDIDATE.model_id == "openai/gpt-5.6-sol"
    assert F2.CANDIDATE.provider_id == "openrouter"
    assert matrix["total_new_attempts"] == 8
    assert matrix["expected_direct_completions"] == 12
    assert matrix["conservative_completion_ceiling"] == 16
    assert matrix["admission_smokes"] == 0


@pytest.mark.parametrize("schedule", [(), EXPECTED_SCHEDULE[:-1],
    (("admission", 1), *EXPECTED_SCHEDULE), (*EXPECTED_SCHEDULE, ("I01", 4)),
    (("I01", 1), *EXPECTED_SCHEDULE[1:]),
    (("I12", 2), *EXPECTED_SCHEDULE[1:]), (("I13", 2), *EXPECTED_SCHEDULE[1:]),
    (("I02", 2), *EXPECTED_SCHEDULE[1:]), tuple(reversed(EXPECTED_SCHEDULE))])
def test_repeat_matrix_rejects_breadth_admission_omissions_and_extra_attempts(schedule):
    with pytest.raises(ValueError):
        F2.validate_matrix(schedule)


@pytest.mark.parametrize("ceiling", [Decimal(0), Decimal("-1"), Decimal("1.01"),
    Decimal("NaN"), Decimal("Infinity")])
def test_cost_ceiling_is_finite_positive_and_at_most_one_dollar(ceiling):
    with pytest.raises(ValueError):
        F2.SpendingGuard(ceiling=ceiling)


def test_independent_attempts_have_separate_i01_production_recovery_bounds():
    ledger = F2.SpendingGuard()
    request = spending_request()
    for number in (2, 3):
        ledger.begin("I01", number)
        for _ in range(5):
            ledger.reserve(request=request, case="I01")
        with pytest.raises(run_frontier.FrontierHardStop):
            ledger.reserve(request=request, case="I01")
    assert len(ledger.requests) == 10
    assert ledger.reserved <= Decimal("1.00")


def test_each_single_call_attempt_remains_separate_and_total_ceiling_is_sixteen():
    ledger = F2.SpendingGuard()
    for sid, number in EXPECTED_SCHEDULE:
        ledger.begin(sid, number)
        for _ in range(5 if sid == "I01" else 1):
            ledger.reserve(request=spending_request(), case=sid)
        with pytest.raises(run_frontier.FrontierHardStop):
            ledger.reserve(request=spending_request(), case=sid)
    assert len(ledger.requests) == F2.COMPLETION_CEILING == 16
    assert ledger.reserved <= Decimal("1.00")
    with pytest.raises(run_frontier.FrontierHardStop):
        ledger.begin("I01", 4)


def test_spending_rejection_keeps_original_reservation_and_never_reclaims_actual_savings():
    ledger = F2.SpendingGuard(ceiling=Decimal("0.05"))
    ledger.begin("I01", 2)
    ledger.reserve(request=spending_request(), case="I01")
    reserved = ledger.reserved
    with pytest.raises(run_frontier.FrontierHardStop):
        ledger.reserve(request=spending_request(), case="I01")
    assert ledger.reserved == reserved
    assert len(ledger.requests) == 1


@pytest.mark.parametrize("mutation", ["oversized_input", "model", "reasoning", "output_cap",
    "fallback", "schema", "price"])
def test_per_request_input_price_and_fixed_configuration_guards_run_before_reservation(mutation):
    ledger = F2.SpendingGuard()
    ledger.begin("I01", 2)
    request = spending_request()
    if mutation == "oversized_input": request["messages"][0]["content"] = "x" * 30000
    elif mutation == "model": request["model"] = "openai/gpt-5.6-sol-pro"
    elif mutation == "reasoning": request["reasoning"] = {"effort": "high"}
    elif mutation == "output_cap": request["max_completion_tokens"] = 8192
    elif mutation == "fallback": request["extra_body"]["provider"]["allow_fallbacks"] = True
    elif mutation == "schema": request["response_format"]["json_schema"]["strict"] = False
    else: request["extra_body"]["provider"]["max_price"]["prompt"] = 3
    with pytest.raises(run_frontier.FrontierHardStop):
        ledger.reserve(request=request, case="I01")
    assert ledger.requests == [] and ledger.reserved == 0


def test_small_request_reserves_full_output_cap_without_balance_inference():
    ledger = F2.SpendingGuard()
    ledger.begin("I01", 2)
    request = spending_request()
    ledger.reserve(request=request, case="I01")
    expected = (run_frontier.PROMPT_PRICE * run_frontier.input_token_bound(request)
        + run_frontier.COMPLETION_PRICE * run_frontier.OUTPUT_LIMIT)
    assert Decimal(ledger.requests[0]["reserved_cost_usd"]) == expected <= Decimal("0.101")
    assert ledger.requests[0]["input_token_upper_bound"] <= 30000
    assert ledger.requests[0]["completion_token_upper_bound"] == 4096
    assert ledger.manifest()["balance_read"] is False
    assert ledger.manifest()["remaining_balance_estimate"] is None


def test_f1_manifest_and_external_review_are_separate_immutable_inputs(historical):
    source, saved, _, review = historical
    before = hash_manifest(source)
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity
    disposition = F2.validate_review(review, saved)
    assert disposition["human_semantic_review"] == "PASS"
    assert hash_manifest(source) == before
    historical_answer = next(r for r in returned["results"] if r["case"] == "I08")
    assert historical_answer["status"] == "HUMAN_REVIEW_PENDING"
    assert historical_answer["attempt"]["semantic_success"] is None


def rebind_offline_validation(source, offline):
    """Model a complete historical record with its original metadata binding."""
    write_json(source / "offline_validation.json", offline)
    saved = json.loads((source / "results.json").read_text())
    saved["offline_validation_fingerprint"] = F2.fingerprint(offline)
    write_json(source / "results.json", saved)
    write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    return saved


def test_f1_historical_source_hashes_survive_later_infrastructure_with_exact_metadata_binding(historical):
    source, _, offline, _ = historical
    historical_sources = {name: str(number) * 64 for number, name in
        enumerate(offline["execution_source_sha256"], 1)}
    offline["execution_source_sha256"] = historical_sources
    saved = rebind_offline_validation(source, offline)
    before = hash_manifest(source)
    assert any(F2._source_identity()[name] != digest for name, digest in historical_sources.items())
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity["execution_source_sha256"] == historical_sources
    assert hash_manifest(source) == before


def test_f1_actual_absolute_source_identity_is_retained_without_current_source_equality(historical):
    source, _, offline, _ = historical
    historical_sources = {str(Path(name).resolve()): str(number) * 64 for number, name in
        enumerate(offline["execution_source_sha256"], 1)}
    offline["execution_source_sha256"] = historical_sources
    saved = rebind_offline_validation(source, offline)
    returned, identity = F2.validate_f1(original=source)
    assert returned == saved
    assert identity["execution_source_sha256"] == historical_sources


@pytest.mark.parametrize("mutation", ["missing_source", "extra_source", "duplicate_alias", "invalid_digest"])
def test_f1_source_identity_shape_remains_closed_even_with_rebound_metadata(historical, mutation):
    source, _, offline, _ = historical
    sources = offline["execution_source_sha256"]
    if mutation == "missing_source":
        sources.pop("benchmarks/interactive/openrouter.py")
    elif mutation == "extra_source":
        sources["src/agent/orchestration/semantic_prompt.py"] = "a" * 64
    elif mutation == "duplicate_alias":
        sources.pop("benchmarks/interactive/openrouter.py")
        sources[str(Path("benchmarks/interactive/run_frontier.py").resolve())] = "a" * 64
    else:
        sources["benchmarks/interactive/run_frontier.py"] = "not-a-sha256"
    rebind_offline_validation(source, offline)
    with pytest.raises(ValueError):
        F2.validate_f1(original=source)


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_f1_offline_fingerprint_binding_is_required_even_with_valid_file_manifest(historical, mutation):
    source, saved, _, _ = historical
    if mutation == "missing":
        saved.pop("offline_validation_fingerprint")
    else:
        saved["offline_validation_fingerprint"] = "a" * 64
    write_json(source / "results.json", saved)
    write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    with pytest.raises(ValueError):
        F2.validate_f1(original=source)


@pytest.mark.parametrize("mutation", ["file_bytes", "missing_file", "extra_file", "manifest_path",
    "source_hash", "candidate", "admission", "context", "prompt", "schema"])
def test_f1_tampering_stops_before_any_sdk_client(tmp_path, historical, mutation):
    source, _, _, review = historical
    if mutation == "file_bytes":
        (source / "results.json").write_text((source / "results.json").read_text() + " ")
    elif mutation == "missing_file":
        (source / "independent_audit.json").unlink()
    elif mutation == "extra_file":
        (source / "extra.json").write_text("{}")
    elif mutation == "manifest_path":
        manifest = json.loads((source / "evidence_manifest.json").read_text())
        manifest["sha256"]["../outside.json"] = "a" * 64
        write_json(source / "evidence_manifest.json", manifest)
    elif mutation == "source_hash":
        offline = json.loads((source / "offline_validation.json").read_text())
        offline["execution_source_sha256"]["benchmarks/interactive/run_frontier.py"] = "a" * 64
        write_json(source / "offline_validation.json", offline)
        write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    else:
        saved = json.loads((source / "results.json").read_text())
        row = next(r for r in saved["results"] if r["case"] == "I10")
        if mutation == "candidate": saved["candidate"]["model_id"] = "openai/gpt-5.6-sol-pro"
        elif mutation == "admission": saved["admission_gate_passed"] = False
        elif mutation == "context": row["attempt"]["context_fingerprint"] = "changed"
        elif mutation == "prompt": row["attempt"]["calls"][0]["prompt_fingerprint"] = "changed"
        else: row["attempt"]["calls"][0]["schema_fingerprint"] = "changed"
        write_json(source / "results.json", saved)
        write_json(source / "evidence_manifest.json", {"sha256": hash_manifest(source)})
    factory = RepeatClients()
    with pytest.raises((ValueError, FileNotFoundError)):
        execute(tmp_path, historical, factory=factory)
    assert factory.created == []


@pytest.mark.parametrize("mutation", ["candidate", "scenario", "verdict", "historical_status", "answer"])
def test_external_review_must_match_exact_historical_pending_answer(historical, mutation):
    _, saved, _, review = historical
    disposition = json.loads(review.read_text())
    if mutation == "candidate": disposition["candidate_id"] = "groq-gpt-oss-120b"
    elif mutation == "scenario": disposition["scenario_id"] = "I09"
    elif mutation == "verdict": disposition["external_human_semantic_review"] = "PENDING"
    elif mutation == "historical_status": disposition["historical_status"] = "PASS"
    else: disposition["retained_answer"] = "Another answer."
    write_json(review, disposition)
    with pytest.raises(ValueError):
        F2.validate_review(review, saved)


def test_f2_live_without_injected_client_requires_clean_checkout_before_evidence_reads(tmp_path, monkeypatch):
    observed = []
    def reject(*, require_clean=False):
        observed.append(require_clean)
        raise ValueError("Live qualification requires a clean tracked tree and index.")
    monkeypatch.setattr(run_frontier, "baseline", reject)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="clean tracked tree and index"):
        F2.run(output=output, live=True, environment={})
    assert observed == [True]
    assert not output.exists()


def test_injected_f2_clients_remain_offline_when_exercising_dispatch_path(tmp_path, historical, monkeypatch):
    _, _, validation, _ = historical
    observed = []
    def baseline(*, require_clean=False):
        observed.append(require_clean)
        assert require_clean is False
        return deepcopy(validation["baseline"])
    monkeypatch.setattr(run_frontier, "baseline", baseline)
    report, factory = execute(tmp_path, historical)
    assert report["complete"]
    assert observed and not any(observed)
    assert sum(len(client.calls) for _, _, client in factory.created) == 12


def test_eight_independent_attempts_preserve_history_and_use_only_twelve_exact_paid_calls(
        tmp_path, historical, monkeypatch):
    source, saved, _, _ = historical
    before = hash_manifest(source)
    context_instances = []
    factory = RepeatClients()
    original_build = harness.build_context
    def capture_context(*args, **kwargs):
        context = original_build(*args, **kwargs)
        if len(factory.created) > len(context_instances):
            context_instances.append(context)
        return context
    monkeypatch.setattr(harness, "build_context", capture_context)
    report, factory = execute(tmp_path, historical, factory=factory)
    assert report["complete"] is True
    assert [(r["scenario_id"], r["attempt_number"]) for r in report["results"]] == list(EXPECTED_SCHEDULE)
    assert Counter(r["status"] for r in report["results"]) == {"PASS": 6, "HUMAN_REVIEW_PENDING": 2}
    assert len(factory.created) == len(context_instances) == 8
    assert len({id(context) for context in context_instances}) == 8
    assert len({id(context.sessions) for context in context_instances}) == 8
    assert len({id(client) for _, _, client in factory.created}) == 8
    assert all(client.closed for _, _, client in factory.created)
    requests = [call for _, _, client in factory.created for call in client.calls]
    assert len(requests) == 12
    assert report["counts"]["scenario_attempts"] == 8
    assert report["counts"]["completion_requests"] == 12
    assert report["counts"]["production_recovery_calls"] == report["counts"]["operational_events"] == 0
    assert report["actual_usage"] == {"prompt_tokens": 1200, "completion_tokens": 240,
        "total_tokens": 1440, "reported_cost_usd": "0.0048", "requests_with_usage": 12,
        "billing_inferred": False, "average_cost_per_scenario_attempt_usd": "0.0006"}
    assert all(not client.outcomes for _, _, client in factory.created)
    for call in requests:
        assert call["model"] == run_frontier.CANDIDATE.model_id
        assert call["max_completion_tokens"] == run_frontier.OUTPUT_LIMIT == 4096
        assert call["response_format"]["json_schema"]["strict"] is True
        assert call["extra_body"]["provider"] == {"require_parameters": True,
            "allow_fallbacks": False, "max_price": {"prompt": 2, "completion": 10, "request": 0}}
        assert "reasoning" not in call
        assert "f2-2" not in call["messages"][0]["content"]
        assert "f2-3" not in call["messages"][0]["content"]
    for sid, rows in rows_by_case(report).items():
        historical_row = next(r for r in saved["results"] if r["case"] == sid)
        for row in rows:
            for key in ("scenario_fingerprint", "context_fingerprint", "request_fingerprint", "registry_fingerprint"):
                assert row["attempt"][key] == historical_row["attempt"][key]
            assert len(row["observations"]) == (0 if sid == "I08" else 1)
            assert len(row["frozen_context_checks"]) == 1
            assert all(check["validated_before_sdk"] for check in row["payload_checks"])
            if sid == "I10":
                assert row["attempt"]["admitted"]["target"] == historical_row["attempt"]["admitted"]["target"]
            if sid == "I14":
                assert row["attempt"]["genuine_clarify"] is True
                assert row["attempt"]["clarification_reason"] == "ambiguous_subject"
        assert [(c["prompt_fingerprint"], c["schema_fingerprint"]) for c in rows[0]["attempt"]["calls"]] == [
            (c["prompt_fingerprint"], c["schema_fingerprint"]) for c in rows[1]["attempt"]["calls"]]
    for rows in rows_by_case(report)["I01"]:
        attempt = rows["attempt"]
        assert attempt["run_status"] == "PLANNED" and attempt["requested_mode"] == "PLAN_ONLY"
        assert attempt["compiler_success"] is attempt["preflight_success"] is True
        assert attempt["plan_tools"] == ["inspect_scATAC"]
        assert attempt["request_sources"] == {"inspect_scATAC.path": "input_path"}
        assert rows["stage_b_structures"][0]["steps"][0]["sources"][0]["input_name"] == "input_path"
    assert len(report["human_review_queue"]) == 2
    for row in rows_by_case(report)["I08"]:
        assert row["attempt"]["semantic_success"] is None
        assert row["attempt"]["contract_success"] is row["attempt"]["admission_success"] is True
        assert row["attempt"]["explanation"]
    for sid, stability in report["stability_matrix"].items():
        sequence = stability["attempts"]
        assert [entry["attempt_number"] for entry in sequence] == [1, 2, 3]
        assert sequence[0]["historical"] is True
        if sid == "I08":
            assert sequence[0]["external_human_semantic_review"] == "PASS"
            assert stability["classification"] == "INDETERMINATE"
        else:
            assert stability["classification"] == "STABLE_IN_SCREEN"
    assert hash_manifest(source) == before
    assert_safe(report)


@pytest.mark.parametrize("mutation", ["prompt", "schema", "private_context", "public_context"])
def test_frozen_payload_and_context_drift_hard_stop_before_sdk(
        tmp_path, historical, monkeypatch, mutation):
    factory = RepeatClients()
    if mutation in ("private_context", "public_context"):
        original_build = harness.build_context
        def altered(*args, **kwargs):
            context = original_build(*args, **kwargs)
            if not factory.created:
                return context
            if mutation == "private_context":
                return replace(context, interaction=replace(context.interaction,
                    utterance=context.interaction.utterance + " Changed private context."))
            return replace(context, public={**context.public, "unexpected": True})
        monkeypatch.setattr(harness, "build_context", altered)
    else:
        complete = run_frontier.FrontierTransport.complete
        def altered_complete(self, *, prompt, response_schema):
            if mutation == "prompt": prompt += " Changed provider-visible input."
            else: response_schema = {**response_schema, "unexpected": True}
            return complete(self, prompt=prompt, response_schema=response_schema)
        monkeypatch.setattr(run_frontier.FrontierTransport, "complete", altered_complete)
    report, factory = execute(tmp_path, historical, factory=factory)
    assert report["hard_stop"]
    assert len(factory.created) == 1
    assert factory.created[0][2].calls == []
    assert factory.created[0][2].closed
    assert len(report["results"]) == 1
    assert_safe(report)


def test_supported_inspection_unsupported_is_semantic_failure_without_retry(tmp_path, historical):
    refusal = {"schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported fixture."}}
    factory = RepeatClients(overrides={("I01", 2): [turn(target="inspect_scATAC"),
        scope("processed_inspection"), refusal, inspection()]})
    report, factory = execute(tmp_path, historical, factory=factory)
    refused, accepted = rows_by_case(report)["I01"]
    assert refused["status"] == "SEMANTIC_FAIL"
    assert refused["attempt"]["decision_kind"] == "unsupported"
    assert len(factory.created[0][2].calls) == 3
    assert factory.created[0][2].outcomes == [inspection()]
    assert accepted["status"] == "PASS"
    assert_safe(report)


def test_valid_provider_selected_scope_is_recorded_and_checked_with_its_actual_builder(tmp_path, historical):
    factory = RepeatClients(overrides={("I01", 2): [turn(target="inspect_scATAC"),
        scope("processed_inspection", "embedding_analysis"), inspection()]})
    report, _ = execute(tmp_path, historical, factory=factory)
    broad, narrow = rows_by_case(report)["I01"]
    assert broad["status"] == narrow["status"] == "PASS"
    broad_calls = broad["attempt"]["calls"]
    narrow_calls = narrow["attempt"]["calls"]
    assert set(broad_calls[1]["capability_ids"]) == {"processed_inspection", "embedding_analysis"}
    assert broad_calls[2]["prompt_fingerprint"] != narrow_calls[2]["prompt_fingerprint"]
    assert broad_calls[2]["schema_fingerprint"] != narrow_calls[2]["schema_fingerprint"]
    assert all(check["validated_before_sdk"] for check in broad["payload_checks"])
    assert_safe(report)


def test_existing_429_production_recovery_is_recorded_without_benchmark_retry(tmp_path, historical):
    factory = RepeatClients(overrides={("I01", 2): [turn(target="inspect_scATAC"),
        scope("processed_inspection"), SDKError(429, headers={"Retry-After": "1"}), inspection()]})
    report, factory = execute(tmp_path, historical, factory=factory)
    row = rows_by_case(report)["I01"][0]
    assert row["status"] == "PASS"
    assert len(factory.created[0][2].calls) == 4
    assert row["recovery_calls"] == 1
    assert row["attempt"]["calls"][-1]["recovery_kind"] == "transport_retry"
    assert row["attempt"]["initial_outcome"]["provider_outcome"] == "failed"
    assert rows_by_case(report)["I01"][1]["recovery_calls"] == 0
    assert rows_by_case(report)["I01"][1]["direct_or_recovered"] == "direct"
    assert sum(len(client.calls) for _, _, client in factory.created) == 13
    assert_safe(report)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_or_balance_failure_blocks_further_production_dispatch(tmp_path, historical, status):
    factory = RepeatClients(overrides={("I01", 2): [turn(target="inspect_scATAC"),
        SDKError(status, message="Private credential-bearing provider failure."), scope("processed_inspection")]})
    report, factory = execute(tmp_path, historical, factory=factory)
    assert report["hard_stop"]
    assert len(factory.created) == 1
    client = factory.created[0][2]
    assert len(client.calls) == 2
    assert client.outcomes == [scope("processed_inspection")]
    assert client.closed
    assert report["counts"]["completion_requests"] == 2
    assert report["counts"]["operational_events"] == 1
    assert_safe(report)


def test_production_retry_then_bad_contract_does_not_add_benchmark_repair(tmp_path, historical):
    invalid = inspection()
    invalid["decision"]["steps"][0]["sources"][0]["source"]["input"] = "unoffered-input"
    factory = RepeatClients(overrides={("I01", 2): [turn(target="inspect_scATAC"),
        scope("processed_inspection"), SDKError(429, headers={"Retry-After": "1"}), invalid, inspection()]})
    report, factory = execute(tmp_path, historical, factory=factory)
    row = rows_by_case(report)["I01"][0]
    assert row["status"] == "CONTRACT_FAIL"
    assert row["recovery_calls"] == 1
    assert len(factory.created[0][2].calls) == 4
    assert factory.created[0][2].outcomes == [inspection()]
    assert len(factory.created) == 8
    assert report["counts"]["completion_requests"] == 13
    assert_safe(report)


@pytest.mark.parametrize("response, status, genuine", [
    ({}, "CONTRACT_FAIL", False),
    (turn("clarify", reason="ambiguous_predecessor"), "SEMANTIC_FAIL", True)])
def test_genuine_ambiguity_is_distinct_from_invalid_contract_fallback(
        tmp_path, historical, response, status, genuine):
    report, _ = execute(tmp_path, historical, factory=RepeatClients(overrides={("I14", 2): [response]}))
    row = rows_by_case(report)["I14"][0]
    assert row["status"] == status
    assert row["attempt"]["genuine_clarify"] is genuine
    assert row["attempt"]["semantic_success"] is not True
    assert_safe(report)


def test_newer_admissible_result_is_wrong_current_referent(tmp_path, historical):
    wrong = turn("answer_scientific", target={"output": "@most_recently_created", "subject": None},
        comparison=None, focus="question")
    report, _ = execute(tmp_path, historical, factory=RepeatClients(overrides={("I10", 2): [wrong]}))
    row = rows_by_case(report)["I10"][0]
    assert row["status"] == "SEMANTIC_FAIL"
    assert row["attempt"]["contract_success"] is row["attempt"]["admission_success"] is True
    assert row["attempt"]["admitted"]["target"]["revision_id"]
    assert_safe(report)


def test_invalid_i08_claim_is_contract_failure_and_never_human_queued(tmp_path, historical):
    def invalid(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"] = [{"kind": "claim", "id": "unoffered-claim"}]
        return response
    report, _ = execute(tmp_path, historical, factory=RepeatClients(overrides={("I08", 2): [invalid]}))
    bad, valid = rows_by_case(report)["I08"]
    assert bad["status"] == "CONTRACT_FAIL" and bad["attempt"]["semantic_success"] is None
    assert bad["attempt"]["contract_success"] is False
    assert valid["status"] == "HUMAN_REVIEW_PENDING"
    assert len(report["human_review_queue"]) == 1
    assert report["human_review_queue"][0]["attempt_number"] == 3
    assert_safe(report)


def test_scientific_executor_entry_guard_stops_without_following_attempt(tmp_path, historical):
    def enter_science(prompt):
        return PlanExecutor(build_default_tool_registry()).execute(None)
    report, factory = execute(tmp_path, historical,
        factory=RepeatClients(overrides={("I01", 2): [enter_science]}))
    assert report["hard_stop"]
    assert len(factory.created) == 1
    assert len(factory.created[0][2].calls) == 1
    assert report["results"][0]["status"] == "SAFETY_FAIL"


def test_private_provider_text_and_usage_are_not_persisted(tmp_path, historical):
    secret, account = "offline-private-credential", "offline-private-account"
    def explanation(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"].insert(0,
            {"kind": "text", "text": f"API key={secret}; account {account}."})
        return response
    factory = RepeatClients(overrides={("I08", 2): [explanation]}, usage={"prompt_tokens": 100,
        "completion_tokens": 20, "total_tokens": 120, "cost": "0.0004",
        "Authorization": secret, "account": account})
    report, _ = execute(tmp_path, historical, factory=factory,
        environment={"OPENROUTER_API_KEY": secret, "PROJECT_ID": account})
    for material in [json.dumps(report), *(p.read_text() for p in (tmp_path / "f2").rglob("*.json"))]:
        assert secret not in material and account not in material
    assert len(report["human_review_queue"]) == 2
    assert_safe(report)


def test_private_provider_error_is_sanitized_without_semantic_retry(tmp_path, historical):
    secret = "offline-private-credential"
    factory = RepeatClients(overrides={("I10", 2): [SDKError(400,
        message=f"Unsupported structured format; API key={secret}.")]})
    report, factory = execute(tmp_path, historical, factory=factory)
    row = rows_by_case(report)["I10"][0]
    assert row["status"] == "OPERATIONAL_FAIL"
    assert len(factory.created[2][2].calls) == 1
    assert report["counts"]["production_recovery_calls"] == 0
    for material in [json.dumps(report), *(p.read_text() for p in (tmp_path / "f2").rglob("*.json"))]:
        assert secret not in material
    assert_safe(report)


def test_current_catalog_price_drift_blocks_f2_before_generation(tmp_path, historical):
    source, _, _, review = historical
    directory = tmp_path / "f2"
    directory.mkdir()
    discovery = valid_frontier_discovery()
    discovery["pricing"]["prompt"] = "0.000003"
    write_json(directory / "discovery.json", discovery)
    factory = RepeatClients()
    with pytest.raises(ValueError):
        F2.run(output=directory, original=source, human_review_path=review,
            live=True, environment={"OPENROUTER_API_KEY": "offline-key"}, client_factory=factory)
    assert factory.created == []


def test_source_drift_stops_first_request_before_sdk(tmp_path, historical, monkeypatch):
    factory = RepeatClients()
    original_identity = F2._source_identity
    def changed():
        identity = original_identity()
        if factory.created:
            identity[next(iter(identity))] = "changed"
        return identity
    monkeypatch.setattr(F2, "_source_identity", changed)
    report, factory = execute(tmp_path, historical, factory=factory)
    assert report["hard_stop"]
    assert len(factory.created) == 1
    assert factory.created[0][2].calls == []
    assert factory.created[0][2].closed


def test_read_only_answer_reconstruction_entry_is_guarded(tmp_path, historical):
    def reconstruct(prompt):
        return PlanExecutor(build_default_tool_registry()).preflight(None)
    report, factory = execute(tmp_path, historical,
        factory=RepeatClients(overrides={("I08", 2): [reconstruct]}))
    assert report["hard_stop"]
    assert len(factory.created) == 5
    assert factory.created[-1][2].closed
    assert report["results"][-1]["status"] == "SAFETY_FAIL"
    assert report["human_review_queue"] == []


def test_pacing_waits_only_between_independent_attempts_without_real_sleep(tmp_path, historical):
    class Clock:
        value = 100.0
        waits = []
        def now(self): return self.value
        def sleep(self, seconds):
            assert 0 < seconds <= 60
            self.waits.append(seconds)
            self.value += seconds
    clock = Clock()
    report, _ = execute(tmp_path, historical, clock=clock.now, sleeper=clock.sleep, spacing=3)
    assert clock.waits == [3] * 7
    assert len(report["waits"]) == 7
    assert report["min_attempt_spacing_seconds"] == 3


def test_existing_live_evidence_is_not_replayed_or_overwritten(tmp_path, historical):
    _, factory = execute(tmp_path, historical)
    before = hash_manifest(tmp_path / "f2")
    source, _, _, review = historical
    with pytest.raises((ValueError, FileExistsError)):
        F2.run(output=tmp_path / "f2", original=source, live=True,
            human_review_path=review, environment={"OPENROUTER_API_KEY": "offline-key"},
            client_factory=factory, spacing=0)
    assert len(factory.created) == 8
    assert hash_manifest(tmp_path / "f2") == before
