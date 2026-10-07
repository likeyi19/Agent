"""Offline breadth acceptance through frozen production contracts and fake SDKs."""
from dataclasses import replace
from collections import Counter
from copy import deepcopy
import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from agent.orchestration import PlanExecutor, build_default_tool_registry
from agent.orchestration.planning_model import PlanningModelError
from agent.schemas import AgentError, ErrorCategory, VerificationResult
from benchmarks.interactive import run_breadth
from benchmarks.interactive.breadth_transport import COHORT, PayloadValidationFailure, build_transport
from benchmarks.interactive.candidates import Availability, CANDIDATES
from benchmarks.interactive.fixtures import scripted_answer, scripted_guidance
from benchmarks.interactive.harness import ZeroScienceViolation
from benchmarks.interactive.scenarios import canonical_scenarios


def turn(kind="execute_plan", **values):
    return {"turn_schema_version": 1, "decision": {"kind": kind, **values}}


def scope(*capabilities):
    return {"selection_schema_version": 1, "decision": {
        "kind": "select", "capability_ids": list(capabilities)}}


def inspection():
    return {"schema_version": 4, "decision": {"kind": "plan", "steps": [{
        "step_id": "inspect", "tool": "inspect_scATAC", "sources": [{
            "target": "dataset", "source": {"kind": "input", "input": "input_path"}}],
        "control_dependencies": []}]}}


def witnesses(scenario_id):
    """Legal scripted outputs consumed once, never feedback for model recovery."""
    if scenario_id == "I01":
        return [turn(target="inspect_scATAC"), scope("processed_inspection"), inspection()]
    if scenario_id == "I02":
        return [scope("processed_inspection")]
    if scenario_id in ("I03", "I05"):
        return [inspection()]
    if scenario_id == "I04":
        return [{"schema_version": 4, "decision": {"kind": "plan", "steps": [
            {"step_id": "embed", "tool": "epizoo_embed_cells", "sources": [], "control_dependencies": []},
            {"step_id": "neighbors", "tool": "build_cell_neighbors", "sources": [{
                "target": "embedding", "source": {"kind": "step_port", "step": "embed", "source_port": "embedding"}}],
                "control_dependencies": []}]}}]
    if scenario_id == "I06":
        return [{"selection_schema_version": 1, "decision": {"kind": "unsupported"}}]
    if scenario_id == "I07":
        return [turn("clarify", reason="missing_parameter_value")]
    if scenario_id == "I08":
        return [lambda prompt: scripted_answer(prompt, "nnz")]
    if scenario_id == "I09":
        return [lambda prompt: scripted_answer(prompt, "primary_annotation")]
    if scenario_id in ("I10", "I11", "I12"):
        referent = {"I10": "@current_result", "I11": "@most_recently_created", "I12": "@previous_turn_result"}[scenario_id]
        return [turn("answer_scientific", target={"output": referent, "subject": None},
            comparison=None, focus="question")]
    if scenario_id == "I13":
        return [scripted_guidance]
    if scenario_id == "I14":
        return [turn("clarify", reason="ambiguous_subject")]
    raise AssertionError("No witness for the frozen scenario")


class SDKError(Exception):
    def __init__(self, status, *, headers=None, message="Synthetic provider failure."):
        self.status_code = status
        self.body = {"error": {"message": message}}
        self.response = SimpleNamespace(status_code=status, headers=headers or {})
        super().__init__("Raw fake SDK body must remain private")


class SDKClient:
    """Each independent SDK client fails on extra calls or renewed discovery."""
    def __init__(self, candidate, outcomes):
        self.candidate = candidate
        self.outcomes, self.calls = list(outcomes), []
        self.models = SimpleNamespace(list=self.forbidden_discovery)
        self.responses = SimpleNamespace(create=self.create)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def forbidden_discovery(self, **kwargs):
        pytest.fail("An attempt repeated the shared exact model discovery")

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self.outcomes, "Breadth introduced an extra completion"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        prompt = json.loads(kwargs["input"] if "input" in kwargs else kwargs["messages"][0]["content"])
        if callable(outcome):
            outcome = outcome(prompt)
        response = outcome if isinstance(outcome, str) else json.dumps(outcome)
        if self.candidate.provider_id == "groq":
            return {"status": "completed", "output_text": response}
        return {"choices": [{"finish_reason": "stop", "message": {"content": response}}]}


def discovery(candidates=COHORT):
    return {c.candidate_id: {"availability": Availability("available", "CANDIDATE_MODEL_AVAILABLE",
        c.model_id, "models.list", provider_model_id=c.model_id).to_dict(), "diagnosis": None}
        for c in candidates}


class Factory:
    def __init__(self, scenarios, candidates=COHORT, *, overrides=None, skip=(), environment=None):
        self.schedule = [(s, c) for s in scenarios for c in candidates
            if (s.scenario_id, c.candidate_id) not in skip]
        self.overrides, self.environment = overrides or {}, environment or {}
        self.created = []

    def __call__(self, candidate, availability, max_calls, *, before_complete=None):
        scenario, expected_candidate = self.schedule[len(self.created)]
        assert candidate == expected_candidate, "Breadth changed the scenario-major candidate order"
        assert max_calls == (5 if scenario.surface == "layer2" else 1)
        outcomes = self.overrides.get((scenario.scenario_id, candidate.candidate_id), witnesses(scenario.scenario_id))
        client = SDKClient(candidate, outcomes)
        runtime = build_transport(candidate, availability=availability, client=client,
            environment=self.environment, max_calls=max_calls, before_complete=before_complete)
        self.created.append((scenario, candidate, client, runtime))
        return runtime


@pytest.fixture
def scenarios():
    return {s.scenario_id: s for s in canonical_scenarios()}


@pytest.fixture(autouse=True)
def offline_manifest(monkeypatch):
    monkeypatch.setattr(run_breadth, "comparison_manifest", lambda: {
        "repository_commit": run_breadth.BASELINE, "qualification_claim": False})


def assert_zero_science(report):
    assert all(value == 0 for value in report["safety"].values())
    for row in report["results"]:
        if row["attempt"] is not None:
            assert all(value == 0 for value in row["attempt"]["safety"].values())


def test_dry_validation_freezes_the_full_matrix_without_provider_calls():
    report = run_breadth.dry_validate(environment={})
    assert report["baseline"] == "48d83625e3da45aad5b7687d14ecf2118b29f036"
    assert report["expected_attempts"] == 42
    assert report["expected_direct_completions"] == 48
    assert report["conservative_completion_ceiling"] == 54
    assert report["zero_science_guards_validated"] is True and report["provider_calls"] == 0
    assert [(c["provider_id"], c["model_id"]) for c in report["candidates"]] == [
        ("groq", "openai/gpt-oss-120b"), ("groq", "qwen/qwen3.8-27b"),
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free")]
    expected = [[s.scenario_id, c.candidate_id] for s in canonical_scenarios() for c in COHORT]
    assert report["attempt_order"] == expected
    assert [s["fingerprint"] for s in report["scenarios"]] == [s.fingerprint() for s in canonical_scenarios()]


def frozen_git(monkeypatch, *, head="a" * 40, ancestor=True, committed="", working="", tracked="", index=""):
    """No Git mutations: model the original baseline and a new infrastructure commit."""
    calls = []
    def git(*args):
        calls.append(args)
        if args == ("rev-parse", "HEAD"): return head
        if args[0] == "merge-base": return run_breadth.BASELINE if ancestor else "b" * 40
        if args[0] == "ls-tree":
            return "\n".join(("benchmarks/interactive/README.md", "benchmarks/interactive/harness.py",
                "benchmarks/interactive/scenarios.py", "benchmarks/interactive/candidates.py",
                "benchmarks/interactive/openrouter.py", "benchmarks/interactive/run_smoke.py",
                "benchmarks/interactive/run_openrouter_smoke.py"))
        if args == ("diff", "--name-only"): return tracked
        if args == ("diff", "--cached", "--name-only"): return index
        if args[:3] == ("diff", "--name-only", run_breadth.BASELINE):
            assert "src/agent" in args and "benchmarks/planner" in args
            assert "benchmarks/interactive/harness.py" in args and "benchmarks/interactive/scenarios.py" in args
            assert "benchmarks/interactive/run_smoke.py" in args
            assert "benchmarks/interactive/run_openrouter_smoke.py" in args
            assert "benchmarks/interactive/README.md" not in args and "benchmarks/interactive/run_breadth.py" not in args
            assert "benchmarks/interactive/openrouter.py" not in args
            paths = args[args.index("--") + 1:]
            changed = working if args[3] == "--" else committed
            return "\n".join(name for name in changed.splitlines() if
                any(name == path or name.startswith(path + "/") for path in paths))
        pytest.fail(f"Unexpected baseline Git read: {args!r}")
    monkeypatch.setattr(run_breadth, "_git", git)
    return head, calls


@pytest.mark.parametrize("head", [run_breadth.BASELINE, "a" * 40])
def test_experiment_baseline_survives_new_source_equivalent_infrastructure_commit(monkeypatch, head):
    expected, _ = frozen_git(monkeypatch, head=head)
    result = run_breadth.repository_baseline(require_clean=True)
    assert result["experiment_baseline"] == run_breadth.BASELINE
    assert result["infrastructure_commit"] == expected
    assert result["tracked_clean"] is result["index_clean"] is True
    report = run_breadth.dry_validate(environment={})
    assert report["baseline"] == run_breadth.BASELINE
    assert report["infrastructure_commit"] == expected
    assert report["provider_calls"] == 0


def test_offline_checks_allow_pending_docs_and_new_benchmarks_but_live_requires_clean_tree(monkeypatch):
    head, _ = frozen_git(monkeypatch, tracked="benchmarks/interactive/README.md",
        index="benchmarks/interactive/run_stability.py")
    result = run_breadth.repository_baseline()
    assert result["infrastructure_commit"] == head and result["experiment_baseline"] == run_breadth.BASELINE
    assert result["tracked_clean"] is result["index_clean"] is False
    with pytest.raises(ValueError): run_breadth.repository_baseline(require_clean=True)


@pytest.mark.parametrize("committed", [False, True])
def test_openrouter_transport_infrastructure_can_change_without_unfreezing_semantic_owners(
        monkeypatch, committed):
    transport = "benchmarks/interactive/openrouter.py"
    frozen_git(monkeypatch, committed=transport if committed else "",
        working="" if committed else transport, tracked="" if committed else transport)
    result = run_breadth.repository_baseline(require_clean=False)
    assert result["experiment_baseline"] == run_breadth.BASELINE
    assert result["tracked_clean"] is committed
    if committed:
        assert run_breadth.repository_baseline(require_clean=True)["tracked_clean"]
    else:
        with pytest.raises(ValueError, match="clean tracked tree and index"):
            run_breadth.repository_baseline(require_clean=True)


@pytest.mark.parametrize("drift", [
    {"ancestor": False},
    {"working": "src/agent/resources/scientific-contract.R"},
    {"working": "benchmarks/planner/canonical-case.json"},
    {"working": "benchmarks/interactive/harness.py"},
    {"working": "benchmarks/interactive/scenarios.py"},
    {"working": "benchmarks/interactive/run_smoke.py"},
    {"committed": "benchmarks/interactive/run_openrouter_smoke.py"},
    {"committed": "benchmarks/interactive/openrouter.py\nbenchmarks/interactive/harness.py"},
    {"committed": "src/agent/orchestration/semantic_prompt.py"},
])
def test_unrelated_checkout_or_frozen_contract_drift_is_rejected_even_offline(monkeypatch, drift):
    frozen_git(monkeypatch, **drift)
    with pytest.raises(ValueError): run_breadth.repository_baseline(require_clean=False)


def test_unavailable_historical_baseline_fails_closed_without_fetch(monkeypatch):
    def git(*args):
        if args == ("rev-parse", "HEAD"): return "a" * 40
        raise subprocess.CalledProcessError(128, ["git", *args])
    monkeypatch.setattr(run_breadth, "_git", git)
    with pytest.raises(ValueError): run_breadth.repository_baseline()


def test_cached_discovery_does_not_bypass_live_clean_tree_admission(tmp_path, scenarios, monkeypatch):
    frozen_git(monkeypatch, tracked="benchmarks/interactive/README.md")
    def forbidden(*args, **kwargs): pytest.fail("Dirty live checkout reached transport construction")
    monkeypatch.setattr(run_breadth, "build_transport", forbidden)
    with pytest.raises(ValueError):
        run_breadth.run_breadth(output=tmp_path / "blocked", scenarios=(scenarios["I03"],),
            candidates=COHORT[:1], discovery=discovery(COHORT[:1]), environment={})
    assert not (tmp_path / "blocked").exists()


class CatalogClient:
    def __init__(self, cards, *, error=None):
        self.cards, self.error, self.calls, self.closed = cards, error, [], False
        self.models = SimpleNamespace(list=self.list_models)

    def list_models(self, **kwargs):
        assert not self.closed
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(**card) for card in self.cards])

    def close(self):
        self.closed = True


def inject_catalogs(monkeypatch, clients):
    constructed = []
    def build(candidate, *, environment):
        constructed.append(candidate)
        client = clients[candidate.provider_id]
        return SimpleNamespace(_client=client, close=client.close)
    monkeypatch.setattr(run_breadth, "build_candidate", build)
    return constructed


@pytest.mark.parametrize("alias_decoys", [False, True])
def test_discovery_shares_two_provider_catalogs_and_accepts_only_exact_ids(monkeypatch, alias_decoys):
    groq_cards = [{"id": c.model_id} for c in COHORT[:2]]
    router_cards = [{"id": COHORT[2].model_id, "supported_parameters": ["response_format"],
        "pricing": {"prompt": "0", "completion": "0"}, "private_metadata": "never-record-provider-metadata"}]
    if alias_decoys:
        groq_cards[1] = {"id": COHORT[1].model_id + ":latest", "aliases": [COHORT[1].model_id]}
        router_cards[0]["id"] = COHORT[2].model_id.removesuffix(":free")
        router_cards[0]["aliases"] = [COHORT[2].model_id]
    clients = {"groq": CatalogClient(groq_cards), "openrouter": CatalogClient(router_cards)}
    constructed = inject_catalogs(monkeypatch, clients)
    report = run_breadth._discover(COHORT, {})
    assert constructed == [COHORT[0], COHORT[2]]
    assert report["catalog_requests"] == sum(len(c.calls) for c in clients.values()) == 2
    assert all(c.calls == [{"timeout": 60.0}] and c.closed for c in clients.values())
    for index, candidate in enumerate(COHORT):
        admission = report["candidates"][candidate.candidate_id]["availability"]
        available = not alias_decoys or index == 0
        assert admission["status"] == ("available" if available else "unavailable")
        assert admission["requested_model_id"] == candidate.model_id
        assert admission["provider_model_id"] == (candidate.model_id if available else None)
    assert "never-record-provider-metadata" not in json.dumps(report)


def test_discovery_failures_are_sanitized_without_catalog_retry(monkeypatch):
    secret = "private-breadth-catalog-key"
    organization = "org_private_breadth_catalog_organization"
    clients = {provider: CatalogClient([], error=SDKError(429,
        message="Authorization: Bearer " + secret + "; quota reached for " + organization))
        for provider in ("groq", "openrouter")}
    inject_catalogs(monkeypatch, clients)
    report = run_breadth._discover(COHORT, {"GROQ_API_KEY": secret, "OPENROUTER_API_KEY": secret})
    assert report["catalog_requests"] == 2
    assert all(len(c.calls) == 1 and c.closed for c in clients.values())
    for evidence in report["candidates"].values():
        assert evidence["availability"]["status"] == "operational_failure"
        assert evidence["diagnosis"]["http_status"] == 429
    assert secret not in json.dumps(report)
    assert organization not in json.dumps(report)
    assert "Raw fake SDK body" not in json.dumps(report)


@pytest.mark.parametrize("invalid", ["changed_scenario", "duplicate_scenario", "deferred_candidate"])
def test_frozen_inventory_rejection_precedes_files_or_factory(tmp_path, scenarios, invalid):
    selected, candidates = (scenarios["I03"],), COHORT
    if invalid == "changed_scenario":
        selected = (replace(selected[0], utterance="A model-specific rewritten request."),)
    elif invalid == "duplicate_scenario":
        selected = selected * 2
    else:
        candidates = (CANDIDATES[2],)
    def forbidden(*args):
        pytest.fail("Invalid frozen inventory constructed a model")
    output = tmp_path / "breadth"
    with pytest.raises(ValueError):
        run_breadth.run_breadth(output=output, scenarios=selected, candidates=candidates,
            discovery=discovery(), factory=forbidden, environment={})
    assert not output.exists()


def test_full_breadth_matrix_uses_42_fresh_real_harness_attempts(tmp_path):
    canonical = canonical_scenarios()
    factory = Factory(canonical)
    output = tmp_path / "breadth"
    report = run_breadth.run_breadth(output=output, discovery=discovery(), factory=factory, environment={})
    assert report["complete"] is True and len(report["results"]) == len(factory.created) == 42
    assert len({id(client) for _, _, client, _ in factory.created}) == 42
    assert len({id(runtime) for _, _, _, runtime in factory.created}) == 42
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 48
    assert all(not client.outcomes and runtime._closed for _, _, client, runtime in factory.created)
    assert report["stability"] == "not_yet_measured" and report["q2_2_run"] is False
    assert report["blocked_providers"] == {}
    for scenario in canonical:
        rows = [r for r in report["results"] if r["scenario_id"] == scenario.scenario_id]
        assert [r["candidate"]["candidate_id"] for r in rows] == [c.candidate_id for c in COHORT]
        assert {r["status"] for r in rows} == {"HUMAN_REVIEW_PENDING" if scenario.human_review_required else "PASS"}
        assert {r["attempt"]["context_fingerprint"] for r in rows} == {
            next(s["context_fingerprint"] for s in report["preflight"]["scenarios"] if s["scenario_id"] == scenario.scenario_id)}
        for index in range(3 if scenario.surface == "layer2" else 1):
            assert len({r["attempt"]["calls"][index]["prompt_fingerprint"] for r in rows}) == 1
            assert len({r["attempt"]["calls"][index]["schema_fingerprint"] for r in rows}) == 1
        for row in rows:
            assert row["attempted"] is True and row["recovery_calls"] == 0
            assert row["attempt"]["scenario_fingerprint"] == scenario.fingerprint()
            if scenario.human_review_required:
                assert row["attempt"]["semantic_success"] is None
                assert row["attempt"]["semantic_properties_pass"] is True
            if scenario.surface == "layer2":
                assert row["attempt"]["planning_only"] is True
                assert row["attempt"]["preflight_success"] is True
                assert row["attempt"]["final_outcome"]["status"] == "PLANNED"
    for counts in report["candidate_counts"].values():
        assert counts["attempts"] == 14 and counts["completions"] == 16
        assert counts["statuses"] == {"PASS": 11, "HUMAN_REVIEW_PENDING": 3}
        assert counts["recovery_calls"] == counts["provider_failures"] == 0
    assert len(report["human_review_queue"]) == 9
    assert {r["scenario_id"] for r in report["human_review_queue"]} == {"I08", "I09", "I13"}
    assert all(r["explanation"] for r in report["human_review_queue"])
    assert json.loads((output / "results.json").read_text()) == report
    assert json.loads((output / "human_review.json").read_text()) == report["human_review_queue"]
    assert (output / ".gitignore").read_text() == "*\n"
    assert_zero_science(report)


@pytest.mark.parametrize("failure,status,contract,compiler", [
    ("invalid_json", "CONTRACT_FAIL", False, None),
    ("false_unsupported", "SEMANTIC_FAIL", True, None),
    ("unknown_dependency", "ADMISSION_FAIL", True, False),
    ("provider_failure", "OPERATIONAL_FAIL", None, None),
])
def test_real_attempt_failures_keep_separate_taxonomy(tmp_path, scenarios, failure, status, contract, compiler):
    outcome = inspection()
    if failure == "invalid_json": outcome = "Not a JSON response"
    elif failure == "false_unsupported":
        outcome = {"schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported witness."}}
    elif failure == "unknown_dependency": outcome["decision"]["steps"][0]["control_dependencies"] = ["nonexistent-step"]
    else: outcome = SDKError(500)
    selected, candidates = (scenarios["I03"],), COHORT[:1]
    factory = Factory(selected, candidates, overrides={("I03", candidates[0].candidate_id): [outcome]})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=factory, environment={})
    row = report["results"][0]
    assert row["status"] == status and row["provider_completions"] == 1 and row["recovery_calls"] == 0
    assert row["attempt"]["contract_success"] is contract
    assert row["attempt"]["compiler_success"] is compiler
    assert len(factory.created[0][2].calls) == 1
    assert_zero_science(report)


def test_preflight_failure_keeps_successful_contract_and_compiler(tmp_path, scenarios, monkeypatch):
    def reject(self, plan):
        return VerificationResult(False, "plan", plan.plan_id, error=AgentError(
            ErrorCategory.INTERNAL_AGENT_ERROR, "INVALID_TOOL_ARGUMENTS", "Synthetic rejection."))
    monkeypatch.setattr(PlanExecutor, "preflight", reject)
    selected, candidates = (scenarios["I03"],), COHORT[:1]
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=Factory(selected, candidates), environment={})
    row = report["results"][0]
    assert row["status"] == "ADMISSION_FAIL"
    assert row["attempt"]["contract_success"] is row["attempt"]["compiler_success"] is True
    assert row["attempt"]["preflight_success"] is False
    assert_zero_science(report)


@pytest.mark.parametrize("candidate", COHORT, ids=lambda c: c.candidate_id)
@pytest.mark.parametrize("failure,recovery", [("schema", "repair"), ("timeout", "transport_retry")])
def test_layer2_recovery_remains_one_independent_production_attempt(tmp_path, scenarios, candidate, failure, recovery):
    outcomes = witnesses("I01")
    outcomes.insert(2, "Not JSON" if failure == "schema" else TimeoutError("private-fake-timeout"))
    selected, candidates = (scenarios["I01"],), (candidate,)
    factory = Factory(selected, candidates, overrides={("I01", candidate.candidate_id): outcomes})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=factory, environment={})
    row = report["results"][0]
    assert len(factory.created) == 1 and row["status"] == "PASS"
    assert row["provider_completions"] == len(factory.created[0][2].calls) == 4
    assert row["recovery_calls"] == 1
    assert any(c["recovery_kind"] == recovery for c in row["attempt"]["calls"])
    assert row["attempt"]["initial_outcome"] != row["attempt"]["final_outcome"]
    assert row["attempt"]["final_outcome"]["status"] == "PLANNED"
    assert "private-fake-timeout" not in json.dumps(report)
    assert_zero_science(report)


@pytest.mark.parametrize("terminal", ["plan", "unsupported"])
def test_recovered_transient_429_records_event_without_halting_later_cells(tmp_path, scenarios, terminal):
    selected = (scenarios["I01"], scenarios["I02"])
    outcomes = witnesses("I01")
    outcomes.insert(2, SDKError(429, headers={"Retry-After": "1"}))
    if terminal == "unsupported":
        outcomes[-1] = {"schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported witness."}}
    factory = Factory(selected, overrides={("I01", COHORT[0].candidate_id): outcomes})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={})
    first, *later = report["results"]
    assert first["status"] == ("PASS" if terminal == "plan" else "SEMANTIC_FAIL")
    assert first["provider_completions"] == 4 and first["recovery_calls"] == 1
    assert first["diagnostics"][2]["http_status"] == 429
    assert first["diagnostics"][-1]["provider_outcome"] == "returned"
    assert all(row["status"] == "PASS" for row in later)
    assert len(factory.created) == 6
    assert report["blocked_providers"] == {} and report.get("blocked_candidates", {}) == {}
    assert_zero_science(report)


def test_transport_budget_rejection_is_not_counted_as_an_sdk_completion(tmp_path, scenarios):
    selected, candidates = (scenarios["I01"],), COHORT[:1]
    client = SDKClient(candidates[0], witnesses("I01")[:1])
    runtimes = []
    def limited(candidate, availability, max_calls):
        assert max_calls == 5
        runtime = build_transport(candidate, availability=availability, client=client, environment={}, max_calls=1)
        runtimes.append(runtime)
        return runtime
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=limited, environment={})
    row = report["results"][0]
    assert row["status"] == "OPERATIONAL_FAIL"
    assert row["provider_completions"] == len(client.calls) == len(row["diagnostics"]) == 1
    assert row["harness_completion_invocations"] == 2
    assert row["recovery_calls"] == 0
    assert report["candidate_counts"][candidates[0].candidate_id]["provider_failures"] == 0
    assert runtimes[0]._closed is True
    assert_zero_science(report)


def test_transport_construction_failure_has_explicit_zero_call_cell_and_continues(tmp_path, scenarios):
    selected = (scenarios["I03"],)
    secret = "private-factory-credential"
    remaining = Factory(selected, COHORT[1:])
    def factory(candidate, availability, max_calls):
        if candidate == COHORT[0]:
            raise PlanningModelError(secret, code="PLANNING_PROVIDER_CONFIGURATION_FAILED")
        return remaining(candidate, availability, max_calls)
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={"GROQ_API_KEY": secret})
    assert report["complete"] is True and len(report["results"]) == 3
    first, *others = report["results"]
    assert first["status"] == "OPERATIONAL_FAIL" and first["attempt"] is None
    assert first["reason"]["kind"] == "transport_configuration"
    assert first["provider_completions"] == 0 and first["attempted"] is False
    assert all(r["status"] == "PASS" for r in others)
    assert secret not in json.dumps(report)
    assert_zero_science(report)


@pytest.mark.parametrize("headers,reason", [
    ({"Retry-After": "120"}, "reset_outside_bounded_window"),
    ({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m"}, "reset_outside_bounded_window"),
])
def test_long_model_rate_window_does_not_suppress_the_other_groq_candidate(tmp_path, scenarios, headers, reason):
    selected = tuple(scenarios[k] for k in ("I03", "I07", "I14"))
    trigger = ("I03", COHORT[0].candidate_id)
    skip = [(s.scenario_id, c.candidate_id) for s in selected for c in COHORT
        if c == COHORT[0] and (s.scenario_id, c.candidate_id) != trigger]
    factory = Factory(selected, overrides={trigger: [SDKError(429, headers=headers)]}, skip=skip)
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={})
    assert len(report["results"]) == 9 and report["complete"] is True
    assert report["blocked_providers"] == {}
    assert report["blocked_candidates"][COHORT[0].candidate_id]["reason"] == reason
    for row in report["results"]:
        key = (row["scenario_id"], row["candidate"]["candidate_id"])
        if key == trigger: assert row["status"] == "OPERATIONAL_FAIL"
        elif row["candidate"]["candidate_id"] == COHORT[0].candidate_id:
            assert row["status"] == "NOT_RUN_PROVIDER_LIMIT" and row["attempt"] is None
            assert row["provider_completions"] == 0 and row["attempted"] is False
        else: assert row["status"] == "PASS"
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 7
    assert_zero_science(report)


def test_repeated_unrecovered_429_stops_only_the_evidenced_model(tmp_path, scenarios):
    selected = tuple(scenarios[k] for k in ("I03", "I07", "I14"))
    model = COHORT[0]
    overrides = {(sid, model.candidate_id): [SDKError(429)] for sid in ("I03", "I07")}
    factory = Factory(selected, overrides=overrides, skip=[("I14", model.candidate_id)])
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={})
    assert report["blocked_providers"] == {}
    assert report["blocked_candidates"][model.candidate_id]["reason"] == "consecutive_unrecovered_model_rate_limits"
    for row in report["results"]:
        if row["candidate"]["candidate_id"] == model.candidate_id:
            assert row["status"] == ("NOT_RUN_PROVIDER_LIMIT" if row["scenario_id"] == "I14" else "OPERATIONAL_FAIL")
        else:
            assert row["status"] == "PASS"
    assert len(factory.created) == 8
    assert_zero_science(report)


def test_explicit_account_quota_evidence_can_stop_both_groq_models(tmp_path, scenarios):
    selected = (scenarios["I03"], scenarios["I14"])
    trigger = ("I03", COHORT[0].candidate_id)
    error = SDKError(429)
    error.body["error"]["code"] = "account_quota_exceeded"
    skip = [(s.scenario_id, c.candidate_id) for s in selected for c in COHORT
        if c.provider_id == "groq" and (s.scenario_id, c.candidate_id) != trigger]
    factory = Factory(selected, overrides={trigger: [error]}, skip=skip)
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={})
    assert report["blocked_providers"]["groq"]["reason"] == "explicit_account_limit_or_block"
    assert report["blocked_candidates"] == {}
    assert [r["status"] for r in report["results"]] == [
        "OPERATIONAL_FAIL", "NOT_RUN_PROVIDER_LIMIT", "PASS", "NOT_RUN_PROVIDER_LIMIT", "NOT_RUN_PROVIDER_LIMIT", "PASS"]
    assert len(factory.created) == 3
    assert_zero_science(report)


def test_single_unqualified_429_does_not_block_later_independent_attempts(tmp_path, scenarios):
    selected = tuple(scenarios[k] for k in ("I03", "I14"))
    trigger = ("I03", COHORT[0].candidate_id)
    factory = Factory(selected, overrides={trigger: [SDKError(429)]})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=discovery(), factory=factory, environment={})
    assert report["blocked_providers"] == {} and len(factory.created) == 6
    assert report["results"][0]["status"] == "OPERATIONAL_FAIL"
    assert all(r["status"] == "PASS" for r in report["results"][1:])
    assert_zero_science(report)


def test_unavailable_frozen_candidate_has_explicit_cells_and_no_substitution(tmp_path, scenarios):
    selected = (scenarios["I03"], scenarios["I14"])
    missing = COHORT[2]
    evidence = discovery()
    evidence[missing.candidate_id]["availability"] = Availability("unavailable", "CANDIDATE_MODEL_UNAVAILABLE",
        missing.model_id, "models.list").to_dict()
    factory = Factory(selected, skip=[(s.scenario_id, missing.candidate_id) for s in selected])
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected,
        discovery=evidence, factory=factory, environment={})
    assert len(report["results"]) == 6 and len(factory.created) == 4
    assert report["candidate_counts"][missing.candidate_id]["completions"] == 0
    for row in report["results"]:
        if row["candidate"]["candidate_id"] == missing.candidate_id:
            assert row["status"] == "OPERATIONAL_FAIL" and row["attempt"] is None
            assert row["reason"]["kind"] == "model_availability"
            assert row["candidate"]["model_id"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert_zero_science(report)


def test_invalid_human_claim_is_contract_failure_without_semantic_prose_judgment(tmp_path, scenarios):
    def invalid(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"] = [{"kind": "claim", "id": "not-an-offered-claim"}]
        return response
    selected, candidates = (scenarios["I08"],), COHORT[:1]
    factory = Factory(selected, candidates, overrides={("I08", candidates[0].candidate_id): [invalid]})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=factory, environment={})
    row = report["results"][0]
    assert row["status"] == "CONTRACT_FAIL"
    assert row["attempt"]["human_review_required"] is True and row["attempt"]["semantic_success"] is None
    assert report["human_review_queue"] == []
    assert_zero_science(report)


def test_rejected_guidance_fallback_is_not_a_human_reviewable_model_output(tmp_path, scenarios):
    def invalid(prompt):
        response = scripted_guidance(prompt)
        response["candidates"][0]["explanation"]["paragraphs"][0]["parts"] = [
            {"kind": "claim", "id": "not-an-offered-claim"}]
        return response
    selected, candidates = (scenarios["I13"],), COHORT[:1]
    factory = Factory(selected, candidates, overrides={("I13", candidates[0].candidate_id): [invalid]})
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=factory, environment={})
    row = report["results"][0]
    assert row["status"] == "CONTRACT_FAIL" and row["attempt"]["explanation"]
    assert row["attempt"]["guidance_candidates"] is None
    assert report["human_review_queue"] == []
    assert_zero_science(report)


def test_human_review_prose_and_all_artifacts_remove_private_values(tmp_path, scenarios):
    secret, operator = "private-breadth-key", "private-project-identifier"
    organization = "org_private_breadth_answer_organization"
    def explanation(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"].insert(0,
            {"kind": "text", "text": "API key=" + secret + "; project " + operator + "; " + organization + "."})
        return response
    selected, candidates = (scenarios["I08"],), COHORT[:1]
    environment = {"GROQ_API_KEY": secret, "PROJECT_ID": operator}
    factory = Factory(selected, candidates, environment=environment,
        overrides={("I08", candidates[0].candidate_id): [explanation]})
    output = tmp_path / "breadth"
    report = run_breadth.run_breadth(output=output, scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=factory, environment=environment)
    assert report["results"][0]["status"] == "HUMAN_REVIEW_PENDING"
    assert "REDACTED" in report["human_review_queue"][0]["explanation"]
    for material in [json.dumps(report), *(p.read_text() for p in output.rglob("*.json"))]:
        assert secret not in material and operator not in material
        assert organization not in material
    assert_zero_science(report)


def test_scientific_entry_guard_aborts_immediately_and_closes_attempt(tmp_path, scenarios):
    def attack(prompt):
        PlanExecutor(build_default_tool_registry()).execute(None)
        pytest.fail("Scientific entry escaped its mandatory guard")
    selected, candidates = (scenarios["I03"],), COHORT[:1]
    factory = Factory(selected, candidates, overrides={("I03", candidates[0].candidate_id): [attack]})
    with pytest.raises(ZeroScienceViolation):
        run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
            discovery=discovery(candidates), factory=factory, environment={})
    assert len(factory.created) == 1 and len(factory.created[0][2].calls) == 1
    assert factory.created[0][3]._closed is True


def test_discovery_scientific_entry_guard_propagates_and_closes_catalog(monkeypatch):
    client = CatalogClient([])
    def attack(**kwargs):
        PlanExecutor(build_default_tool_registry()).execute(None)
        pytest.fail("Discovery escaped its mandatory scientific entry guard")
    client.models.list = attack
    inject_catalogs(monkeypatch, {"groq": client})
    with pytest.raises(ZeroScienceViolation):
        run_breadth._discover(COHORT[:2], {})
    assert client.closed is True


def test_existing_output_is_preserved_before_factory_construction(tmp_path, scenarios):
    output = tmp_path / "breadth"
    output.mkdir()
    preserved = output / "results.json"
    preserved.write_text("Preserved result.")
    factory = Factory((scenarios["I03"],))
    with pytest.raises(FileExistsError):
        run_breadth.run_breadth(output=output, scenarios=(scenarios["I03"],),
            discovery=discovery(), factory=factory, environment={})
    assert factory.created == [] and preserved.read_text() == "Preserved result."


def test_cli_requires_explicit_live_or_dry_validation(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        pytest.fail("Breadth calls were authorized without --live")
    monkeypatch.setattr(run_breadth, "run_breadth", forbidden)
    output = tmp_path / "breadth"
    with pytest.raises(SystemExit) as failure:
        run_breadth.main(["--output", str(output)])
    assert failure.value.code == 2 and not output.exists()


@pytest.fixture(scope="module")
def original_ledger(tmp_path_factory):
    """Synthetic historical ledger with the same authorized 15/27 identity split."""
    canonical = canonical_scenarios()
    outcomes = [turn(target="inspect_scATAC"), scope("processed_inspection", "raw_preprocessing"),
        SDKError(429, headers={"Retry-After": "1"}),
        {"schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported witness."}}]
    factory = Factory(canonical, overrides={("I01", COHORT[0].candidate_id): outcomes})
    output = tmp_path_factory.mktemp("original-ledger") / "q2-1"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_breadth, "comparison_manifest", lambda: {
            "repository_commit": run_breadth.BASELINE, "qualification_claim": False})
        report = run_breadth.run_breadth(output=output, discovery=discovery(), factory=factory, environment={})
    for row in report["results"]:
        preserve = (row["candidate"]["candidate_id"] == COHORT[2].candidate_id
            or (row["scenario_id"], row["candidate"]["candidate_id"]) == ("I01", COHORT[0].candidate_id))
        if not preserve:
            row.update(status="NOT_RUN_PROVIDER_LIMIT", reason={"reason": "historical_conservative_stop"},
                attempted=False, provider_completions=0, recovery_calls=0, attempt=None, diagnostics=[])
            row.pop("harness_completion_invocations", None)
    for candidate in COHORT:
        rows = [r for r in report["results"] if r["candidate"]["candidate_id"] == candidate.candidate_id]
        report["candidate_counts"][candidate.candidate_id].update(
            attempts=sum(r["attempted"] for r in rows), completions=sum(r["provider_completions"] for r in rows),
            recovery_calls=sum(r["recovery_calls"] for r in rows), statuses=dict(Counter(r["status"] for r in rows)),
            provider_failures=sum(d["provider_outcome"] == "failed" for r in rows for d in r["diagnostics"]))
    report["human_review_queue"] = [r for r in report["human_review_queue"]
        if r["candidate"]["candidate_id"] == COHORT[2].candidate_id]
    source = output / "results.json"
    source.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "human_review.json").write_text(json.dumps(report["human_review_queue"]) + "\n")
    assert sum(r["attempt"] is not None for r in report["results"]) == 15
    assert sum(r["provider_completions"] for r in report["results"]) == 20
    return source, report


def preserved_keys():
    return {("I01", COHORT[0].candidate_id)} | {
        (s.scenario_id, COHORT[2].candidate_id) for s in canonical_scenarios()}


def missing_keys():
    preserved = preserved_keys()
    return [(s.scenario_id, c.candidate_id) for s in canonical_scenarios() for c in COHORT
        if (s.scenario_id, c.candidate_id) not in preserved]


def test_resume_validation_schedules_exact_27_first_observations(original_ledger):
    source, original = original_ledger
    before = source.read_bytes()
    plan = run_breadth.resume_validate(source, environment={})
    assert plan["total_cells"] == 42 and plan["existing_attempts"] == 15 and plan["missing_attempts"] == 27
    assert plan["planned_cells"] == [list(key) for key in missing_keys()]
    assert {tuple(key) for key in plan["preserved_cells"]} == preserved_keys()
    assert plan["planned_nemotron_calls"] == plan["planned_gpt_oss_i01_calls"] == 0
    assert plan["expected_direct_completions"] == 29 and plan["conservative_completion_ceiling"] == 31
    assert plan["provider_calls"] == plan["catalog_requests"] == 0
    assert plan["original_identity"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert plan["zero_science_guards_validated"] is True
    assert source.read_bytes() == before
    assert original["results"][0]["status"] == "SEMANTIC_FAIL"


def test_resume_keeps_historical_experiment_binding_after_infrastructure_commit(original_ledger, monkeypatch):
    source, original = original_ledger
    before = source.read_bytes()
    head, _ = frozen_git(monkeypatch)
    plan = run_breadth.resume_validate(source, environment={})
    assert plan["baseline"] == run_breadth.BASELINE
    assert plan["preflight"]["infrastructure_commit"] == head
    assert plan["original_ledger_fingerprint"] == run_breadth.fingerprint(original)
    assert plan["existing_attempts"] == 15 and plan["missing_attempts"] == 27
    assert plan["provider_calls"] == plan["catalog_requests"] == 0
    assert source.read_bytes() == before


def test_groq_only_resume_needs_no_historical_openrouter_credential(original_ledger, monkeypatch):
    source, _ = original_ledger
    frozen_git(monkeypatch)
    plan = run_breadth.resume_validate(source, environment={"GROQ_API_KEY": "synthetic-offline-key"},
        require_credentials=True)
    assert plan["credentials_present"] == {"GROQ_API_KEY": True}
    assert plan["planned_nemotron_calls"] == plan["catalog_requests"] == plan["provider_calls"] == 0


def test_resume_preserves_all_15_records_and_runs_only_the_27_missing_cells(tmp_path, original_ledger, monkeypatch):
    source, original = original_ledger
    before = source.read_bytes()
    def forbidden(*args):
        pytest.fail("Continuation repeated cached exact model discovery")
    monkeypatch.setattr(run_breadth, "_discover", forbidden)
    factory = Factory(canonical_scenarios(), skip=preserved_keys())
    output = tmp_path / "continuation"
    result = run_breadth.resume_breadth(original=source, output=output, factory=factory, environment={})
    continuation, merged = result["continuation"], result["merged"]
    assert continuation["complete"] is True and len(continuation["results"]) == len(factory.created) == 27
    assert [(r["scenario_id"], r["candidate"]["candidate_id"]) for r in continuation["results"]] == missing_keys()
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 29
    assert all(c != COHORT[2] and (s.scenario_id, c.candidate_id) != ("I01", COHORT[0].candidate_id)
        for s, c, _, _ in factory.created)
    assert all(runtime._closed and not client.outcomes for _, _, client, runtime in factory.created)
    assert continuation["candidate_counts"][COHORT[0].candidate_id]["attempts"] == 13
    assert continuation["candidate_counts"][COHORT[1].candidate_id]["attempts"] == 14
    assert continuation["candidate_counts"][COHORT[2].candidate_id]["attempts"] == 0
    assert continuation["candidate_counts"][COHORT[2].candidate_id]["completions"] == 0
    assert continuation["catalog_requests"] == 0 and continuation["q2_2_run"] is False
    assert merged["comparative_breadth_complete"] is True and len(merged["results"]) == 42
    for old, new in zip(original["results"], merged["results"]):
        key = (old["scenario_id"], old["candidate"]["candidate_id"])
        if key in preserved_keys():
            assert old == new and merged["sources"][":".join(key)] == "Q2.1"
        else:
            assert new["attempted"] is True and merged["sources"][":".join(key)] == "Q2.1b"
            assert new["attempt_id"] == "completion-1"
    assert sum(len(checks) for checks in continuation["payload_checks"].values()) == 29
    assert all(check["validated_before_sdk"] for checks in continuation["payload_checks"].values() for check in checks)
    for scenario_id in ("I08", "I09", "I13"):
        original_call = next(r for r in original["results"] if r["scenario_id"] == scenario_id
            and r["candidate"]["candidate_id"] == COHORT[2].candidate_id)["attempt"]["calls"][0]
        for candidate in COHORT[:2]:
            check = continuation["payload_checks"][scenario_id + ":" + candidate.candidate_id][0]
            assert check["prompt_fingerprint"] == original_call["prompt_fingerprint"]
            assert check["schema_fingerprint"] == original_call["schema_fingerprint"]
            assert check["recovery_envelope"] == []
    assert len(merged["human_review_queue"]) == 9
    assert merged["stability"] == continuation["stability"] == "not_yet_measured"
    assert source.read_bytes() == before
    assert json.loads((output / "merged_results.json").read_text()) == merged
    assert json.loads((output / "continuation.json").read_text()) == continuation
    assert_zero_science(continuation)
    assert_zero_science(merged)


def test_resume_merged_human_queue_excludes_preserved_rejected_guidance(tmp_path, original_ledger):
    _, original = original_ledger
    source = deepcopy(original)
    rejected = next(r for r in source["results"] if r["scenario_id"] == "I13"
        and r["candidate"]["candidate_id"] == COHORT[2].candidate_id)
    rejected["status"] = "CONTRACT_FAIL"
    rejected["attempt"].update(contract_success=False, failure_category="contract",
        explanation="Synthetic deterministic Guidance fallback.", guidance_candidates=None)
    preserved = deepcopy(rejected)
    factory = Factory(canonical_scenarios(), skip=preserved_keys())
    result = run_breadth.resume_breadth(original=source, output=tmp_path / "continuation",
        factory=factory, environment={})
    merged = result["merged"]
    assert next(r for r in merged["results"] if r["scenario_id"] == "I13"
        and r["candidate"]["candidate_id"] == COHORT[2].candidate_id) == preserved
    assert not any(r["scenario_id"] == "I13" and r["candidate"]["candidate_id"] == COHORT[2].candidate_id
        for r in merged["human_review_queue"])
    assert len(merged["human_review_queue"]) == 8
    assert_zero_science(merged)


@pytest.mark.parametrize("tamper", ["completed_identity", "missing_observation", "context_fingerprint", "prompt_fingerprint",
    "call_candidate", "call_scenario", "call_attempt", "empty_calls", "zero_completions", "diagnostic_count", "display_status",
    "oversized_completions"])
def test_invalid_resume_identity_or_fingerprint_prevents_any_factory_or_sdk(tmp_path, original_ledger, tamper):
    _, original = original_ledger
    bad = deepcopy(original)
    if tamper == "completed_identity": bad["results"][0]["candidate"]["model_id"] = "different-model"
    elif tamper == "missing_observation":
        bad["results"][1]["status"] = "PASS"
        bad["results"][1]["attempted"] = True
    elif tamper == "context_fingerprint": bad["results"][0]["attempt"]["context_fingerprint"] = "0" * 64
    elif tamper == "prompt_fingerprint": bad["results"][0]["attempt"]["calls"][0]["prompt_fingerprint"] = "0" * 64
    elif tamper == "call_candidate": bad["results"][0]["attempt"]["calls"][0]["candidate"]["model_id"] = COHORT[1].model_id
    elif tamper == "call_scenario": bad["results"][0]["attempt"]["calls"][0]["scenario_id"] = "I12"
    elif tamper == "call_attempt": bad["results"][0]["attempt"]["calls"][0]["attempt_id"] = "different-attempt"
    elif tamper == "empty_calls": bad["results"][0]["attempt"]["calls"] = []
    elif tamper == "zero_completions": bad["results"][0]["provider_completions"] = 0
    elif tamper == "display_status": bad["results"][0]["status"] = "PASS"
    elif tamper == "oversized_completions":
        row = bad["results"][0]
        row["provider_completions"] = 6  # Coherent records still exceed Layer-2's five-call ceiling.
        row["attempt"]["calls"] += [deepcopy(row["attempt"]["calls"][-1])
            for _ in range(6 - len(row["attempt"]["calls"]))]
        row["diagnostics"] += [deepcopy(row["diagnostics"][-1])
            for _ in range(6 - len(row["diagnostics"]))]
    else: bad["results"][0]["diagnostics"].pop()
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid source ledger constructed a model")
    output = tmp_path / "continuation"
    with pytest.raises(ValueError):
        run_breadth.resume_breadth(original=bad, output=output, factory=forbidden, environment={})
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["prompt", "schema"])
def test_resume_payload_fingerprint_guard_stops_before_actual_sdk_call(tmp_path, original_ledger, tamper):
    source, _ = original_ledger
    before = source.read_bytes()
    factory = Factory(canonical_scenarios(), skip=preserved_keys())
    def altered(candidate, availability, max_calls, *, before_complete):
        runtime = factory(candidate, availability, max_calls, before_complete=before_complete)
        complete = runtime.complete
        def changed(*, prompt, response_schema):
            if tamper == "prompt":
                payload = json.loads(prompt)
                payload["unexpected_semantic_instruction"] = "Changed after resume validation."
                prompt = json.dumps(payload)
            else:
                response_schema = dict(response_schema, title="Changed after resume validation.")
            return complete(prompt=prompt, response_schema=response_schema)
        runtime.complete = changed
        return runtime
    with pytest.raises(PayloadValidationFailure):
        run_breadth.resume_breadth(original=source, output=tmp_path / "continuation", factory=altered, environment={})
    assert len(factory.created) == 1
    client, runtime = factory.created[0][2:]
    assert client.calls == [] and runtime.completion_calls == 0 and runtime.diagnostics == []
    assert runtime._closed is True and source.read_bytes() == before


def test_resume_new_persistent_model_limit_leaves_qwen_running_and_preserves_old_rows(tmp_path, original_ledger):
    source, original = original_ledger
    gpt = COHORT[0]
    skip = preserved_keys() | {(s.scenario_id, gpt.candidate_id) for s in canonical_scenarios()
        if s.scenario_id not in ("I01", "I02", "I03")}
    overrides = {(sid, gpt.candidate_id): [SDKError(429)] for sid in ("I02", "I03")}
    factory = Factory(canonical_scenarios(), skip=skip, overrides=overrides)
    result = run_breadth.resume_breadth(original=source, output=tmp_path / "continuation", factory=factory, environment={})
    continuation, merged = result["continuation"], result["merged"]
    assert continuation["blocked_providers"] == {}
    assert continuation["blocked_candidates"][gpt.candidate_id]["reason"] == "consecutive_unrecovered_model_rate_limits"
    assert continuation["candidate_counts"][COHORT[1].candidate_id]["attempts"] == 14
    assert continuation["candidate_counts"][gpt.candidate_id]["attempts"] == 2
    assert continuation["candidate_counts"][gpt.candidate_id]["statuses"] == {
        "OPERATIONAL_FAIL": 2, "NOT_RUN_PROVIDER_LIMIT": 11}
    assert merged["comparative_breadth_complete"] is False and len(merged["results"]) == 42
    for old, new in zip(original["results"], merged["results"]):
        if old["attempt"] is not None: assert old == new
    assert_zero_science(continuation)


def test_frozen_guard_accepts_a_new_legal_scope_without_overriding_semantics(tmp_path, scenarios, original_ledger):
    _, original = original_ledger
    selected, candidates = (scenarios["I01"],), COHORT[1:2]
    outcomes = [turn(target="inspect_scATAC"), scope("embedding_analysis"),
        {"schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported witness."}}]
    factory = Factory(selected, candidates, overrides={("I01", candidates[0].candidate_id): outcomes})
    checks = []
    guard = run_breadth._payload_guard(selected[0], checks)
    def guarded(candidate, availability, max_calls):
        return factory(candidate, availability, max_calls, before_complete=guard)
    report = run_breadth.run_breadth(output=tmp_path / "breadth", scenarios=selected, candidates=candidates,
        discovery=discovery(candidates), factory=guarded, environment={})
    row = report["results"][0]
    assert row["status"] == "SEMANTIC_FAIL" and row["provider_completions"] == 3
    assert len(checks) == 3 and all(check["validated_before_sdk"] for check in checks)
    previous_pairs = {(c["prompt_fingerprint"], c["schema_fingerprint"])
        for r in original["results"] if r["scenario_id"] == "I01" and r["attempt"]
        for c in r["attempt"]["calls"] if c["surface"] == "stage_b"}
    assert (checks[-1]["prompt_fingerprint"], checks[-1]["schema_fingerprint"]) not in previous_pairs
    assert_zero_science(report)


def test_resume_spacing_applies_between_same_model_attempts_without_real_sleep(tmp_path, original_ledger, monkeypatch):
    source, _ = original_ledger
    clock = {"now": 1000.0}
    sleeps, starts = [], []
    def sleep(seconds):
        assert 0 < seconds <= 60
        sleeps.append(seconds)
        clock["now"] += seconds
    monkeypatch.setattr(run_breadth.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(run_breadth.time, "sleep", sleep)
    factory = Factory(canonical_scenarios(), skip=preserved_keys())
    def timed(candidate, availability, max_calls, *, before_complete):
        starts.append((candidate.candidate_id, clock["now"]))
        return factory(candidate, availability, max_calls, before_complete=before_complete)
    result = run_breadth.resume_breadth(original=source, output=tmp_path / "continuation",
        factory=timed, environment={}, min_model_spacing_seconds=60)
    continuation = result["continuation"]
    assert continuation["min_model_spacing_seconds"] == 60
    assert sleeps and all(seconds == 60 for seconds in sleeps)
    assert starts[0] == (COHORT[1].candidate_id, 1000.0)
    assert starts[1] == (COHORT[0].candidate_id, 1000.0)
    for candidate in COHORT[:2]:
        times = [started for identity, started in starts if identity == candidate.candidate_id]
        assert all(later - earlier >= 60 for earlier, later in zip(times, times[1:]))
    assert len(factory.created[0][2].calls) == 3
    assert [d["request_started_monotonic"] for d in factory.created[0][3].diagnostics] == [1000.0] * 3
    assert_zero_science(continuation)


def test_resume_scientific_guard_remains_immediate_and_preserves_original(tmp_path, original_ledger):
    source, _ = original_ledger
    before = source.read_bytes()
    def attack(prompt):
        PlanExecutor(build_default_tool_registry()).execute(None)
        pytest.fail("Continuation entered scientific execution")
    factory = Factory(canonical_scenarios(), skip=preserved_keys(),
        overrides={("I01", COHORT[1].candidate_id): [attack]})
    with pytest.raises(ZeroScienceViolation):
        run_breadth.resume_breadth(original=source, output=tmp_path / "continuation", factory=factory, environment={})
    assert len(factory.created) == 1 and len(factory.created[0][2].calls) == 1
    assert factory.created[0][3]._closed is True and source.read_bytes() == before
