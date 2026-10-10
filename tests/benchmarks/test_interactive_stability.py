"""Focused offline Q2.2 repeats through existing harness and fake SDK witnesses."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agent.orchestration import PlanExecutor, build_default_tool_registry
from agent.orchestration.planning_model import PlanningModelError
from benchmarks.interactive import harness, run_breadth, run_stability
from benchmarks.interactive.breadth_transport import COHORT, PayloadValidationFailure, build_transport
from benchmarks.interactive.harness import ZeroScienceViolation
from benchmarks.interactive.fixtures import scripted_answer
from benchmarks.interactive.scenarios import canonical_scenarios

# Reuse the existing SDK-shaped transport fixtures and legal production witnesses.
from test_interactive_breadth import (
    Factory, SDKClient, SDKError, inspection, original_ledger,
    frozen_git, preserved_keys, scope, turn, witnesses,
)


SELECTED = ("I01", "I10", "I08", "I14")
GPT = COHORT[0]


@pytest.fixture(autouse=True)
def offline_manifest(monkeypatch):
    frozen_git(monkeypatch)
    manifest = lambda: {"repository_commit": run_breadth.BASELINE, "qualification_claim": False}
    monkeypatch.setattr(run_breadth, "comparison_manifest", manifest)
    if hasattr(run_stability, "comparison_manifest"):
        monkeypatch.setattr(run_stability, "comparison_manifest", manifest)


@pytest.fixture(scope="module")
def merged_ledger(tmp_path_factory, original_ledger):
    """A complete fake breadth ledger; no dependency on ignored real eval files."""
    source, _ = original_ledger
    factory = Factory(canonical_scenarios(), skip=preserved_keys(), overrides={
        ("I10", GPT.candidate_id): [SDKError(400, message="Structured JSON generation failed.")],
    })
    output = tmp_path_factory.mktemp("synthetic-merged-ledger") / "q2-1b"
    with pytest.MonkeyPatch.context() as patch:
        frozen_git(patch)
        patch.setattr(run_breadth, "comparison_manifest", lambda: {
            "repository_commit": run_breadth.BASELINE, "qualification_claim": False})
        result = run_breadth.resume_breadth(original=source, output=output,
            factory=factory, environment={})
    merged = result["merged"]
    historical = {r["scenario_id"]: r for r in merged["results"]
        if r["candidate"]["candidate_id"] == GPT.candidate_id}
    assert {s: historical[s]["status"] for s in SELECTED} == {
        "I01": "SEMANTIC_FAIL", "I10": "OPERATIONAL_FAIL",
        "I08": "HUMAN_REVIEW_PENDING", "I14": "PASS"}
    review = output / "human-disposition.json"
    review.write_text(json.dumps({
        "phase": "Q2.1c", "scenario_id": "I08", "candidate_id": GPT.candidate_id,
        "historical_status": "HUMAN_REVIEW_PENDING", "historical_status_unchanged": True,
        "external_human_semantic_review": "PASS", "review_authority": "Synthetic offline operator review",
        "retained_answer": historical["I08"]["attempt"]["explanation"],
        "live_provider_calls": 0,
    }) + "\n")
    return output / "merged_results.json", merged, review


def scheduled():
    scenarios = {s.scenario_id: s for s in canonical_scenarios()}
    return [(scenarios[scenario_id], number) for scenario_id in SELECTED for number in (2, 3)]


class RepeatFactory:
    """Fresh existing breadth transports; exact scenario-major eight-cell schedule."""
    def __init__(self, *, overrides=None, environment=None, tamper=None):
        self.overrides, self.environment = overrides or {}, environment or {}
        self.tamper = tamper
        self.created = []

    def __call__(self, candidate, availability, max_calls, *, before_complete):
        assert candidate == GPT, "Stability attempted another candidate"
        scenario, number = scheduled()[len(self.created)]
        assert max_calls == (5 if scenario.surface == "layer2" else 1)
        assert callable(before_complete), "Each SDK request requires the frozen payload guard"
        outcomes = self.overrides.get((scenario.scenario_id, number), witnesses(scenario.scenario_id))
        client = SDKClient(candidate, outcomes)
        guard = before_complete
        if self.tamper:
            def guard(*, prompt, response_schema):
                payload, schema = json.loads(prompt), deepcopy(response_schema)
                if self.tamper == "prompt":
                    key = "user_request" if "user_request" in payload else "utterance"
                    payload[key] = "Unexpected scenario change."
                    prompt = json.dumps(payload)
                else:
                    schema["unexpected_field"] = True
                before_complete(prompt=prompt, response_schema=schema)
        runtime = build_transport(candidate, availability=availability, client=client,
            environment=self.environment, max_calls=max_calls, before_complete=guard)
        self.created.append((scenario, number, client, runtime))
        return runtime


def execute(tmp_path, merged_ledger, factory=None, **kwargs):
    source, _, review = merged_ledger
    factory = factory or RepeatFactory()
    environment, spacing = kwargs.pop("environment", {}), kwargs.pop("spacing", 0)
    result = run_stability.run_stability(original=source, output=tmp_path / "stability",
        factory=factory, environment=environment, human_review_path=review, spacing=spacing, **kwargs)
    return result, factory


def rows_by_case(report):
    return {s: [r for r in report["results"] if r["scenario_id"] == s] for s in SELECTED}


def assert_safe(report):
    for row in report["results"]:
        if row.get("attempt"):
            assert not any(row["attempt"]["safety"].values())
    if "safety" in report:
        assert not any(report["safety"].values())


def test_validation_schedules_exact_eight_gpt_repeats_and_preserves_history(merged_ledger):
    source, original, review = merged_ledger
    before = source.read_bytes()
    plan = run_stability.stability_validate(source, environment={}, human_review_path=review)
    assert tuple(run_stability.SELECTED_SCENARIOS) == SELECTED
    assert plan["total_new_attempts"] == 8
    assert plan["expected_direct_completions"] == 12
    assert plan["conservative_completion_ceiling"] == 16
    assert plan["planned_attempts"] == [dict(scenario_id=s.scenario_id,
        attempt_id=f"stability-{number}", attempt_number=number,
        candidate_id=GPT.candidate_id) for s, number in scheduled()]
    assert source.read_bytes() == before
    assert next(r for r in original["results"] if r["scenario_id"] == "I08"
        and r["candidate"]["candidate_id"] == GPT.candidate_id)["status"] == "HUMAN_REVIEW_PENDING"


def test_stability_preserves_historical_identity_after_infrastructure_commit(merged_ledger, monkeypatch):
    source, original, review = merged_ledger
    before = source.read_bytes()
    head, _ = frozen_git(monkeypatch)
    plan = run_stability.stability_validate(source, environment={}, human_review_path=review)
    assert plan["baseline"] == run_breadth.BASELINE and plan["infrastructure_commit"] == head
    assert plan["git_baseline"] is None
    assert plan["total_new_attempts"] == 8
    for reference in plan["historical_attempts"]:
        historical = next(r for r in original["results"] if r["scenario_id"] == reference["scenario_id"]
            and r["candidate"]["candidate_id"] == GPT.candidate_id)
        assert reference["row_fingerprint"] == run_breadth.fingerprint(historical)
    assert source.read_bytes() == before


@pytest.mark.parametrize("origin_lag", [False, True])
def test_live_stability_requires_current_infrastructure_refs_aligned(monkeypatch, origin_lag):
    head, _ = frozen_git(monkeypatch)
    def git(command, *, text):
        assert command[0] == "git"
        if command[1] == "rev-parse":
            return (run_breadth.BASELINE if command[2] == "origin/main" and origin_lag else head) + "\n"
        if command[1] == "branch": return "main\n"
        if command[1] == "diff": return ""
        pytest.fail(f"Unexpected live baseline read: {command!r}")
    monkeypatch.setattr(run_stability.subprocess, "check_output", git)
    if origin_lag:
        with pytest.raises(ValueError): run_stability._baseline()
    else:
        baseline = run_stability._baseline()
        assert baseline["experiment_baseline"] == run_breadth.BASELINE
        assert baseline["infrastructure_commit"] == head
        assert set(baseline["refs"].values()) == {head}


def test_eight_independent_real_harness_attempts_use_twelve_sdk_calls(tmp_path, merged_ledger, monkeypatch):
    source, original, _ = merged_ledger
    before = source.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("Stability renewed shared catalog discovery")
    monkeypatch.setattr(run_breadth, "_discover", forbidden)
    report, factory = execute(tmp_path, merged_ledger)
    assert report["complete"] is True
    assert len(report["results"]) == len(factory.created) == 8
    assert [(r["scenario_id"], r["attempt_id"]) for r in report["results"]] == [
        (s.scenario_id, f"stability-{n}") for s, n in scheduled()]
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 12
    assert len({id(runtime) for _, _, _, runtime in factory.created}) == 8
    assert all(runtime._closed and not client.outcomes for _, _, client, runtime in factory.created)
    assert all(r["candidate"]["candidate_id"] == GPT.candidate_id for r in report["results"])
    assert Counter(r["status"] for r in report["results"]) == {"PASS": 6, "HUMAN_REVIEW_PENDING": 2}
    for scenario_id, rows in rows_by_case(report).items():
        old = next(r for r in original["results"] if r["scenario_id"] == scenario_id
            and r["candidate"]["candidate_id"] == GPT.candidate_id)
        assert {r["attempt"]["context_fingerprint"] for r in rows} == {old["attempt"]["context_fingerprint"]}
        assert {r["attempt"]["scenario_fingerprint"] for r in rows} == {old["attempt"]["scenario_fingerprint"]}
        assert {r["attempt"]["request_fingerprint"] for r in rows} == {old["attempt"]["request_fingerprint"]}
        first, second = rows
        assert [(c["prompt_fingerprint"], c["schema_fingerprint"]) for c in first["attempt"]["calls"]] == [
            (c["prompt_fingerprint"], c["schema_fingerprint"]) for c in second["attempt"]["calls"]]
    assert source.read_bytes() == before
    assert len(report["human_review_queue"]) == 2
    assert report["counts"]["attempts"] == 8 and report["counts"]["completions"] == 12
    assert report["counts"]["recovery_calls"] == report["counts"]["operational_events"] == 0
    assert report["counts"]["reserved_pre_sdk_checks"] == 12
    assert set(report["frozen_context_checks"]) == set(report["payload_checks"]) == {
        f"{s.scenario_id}:stability-{n}" for s, n in scheduled()}
    for checks in report["payload_checks"].values():
        assert checks and all(c["validated_before_sdk"] for c in checks)
    for scenario_id, sequence in report["stability_matrix"].items():
        historical = next(r for r in original["results"] if r["scenario_id"] == scenario_id
            and r["candidate"]["candidate_id"] == GPT.candidate_id)
        assert [r["attempt_number"] for r in sequence] == [1, 2, 3]
        assert sequence[0]["historical"] is True and sequence[0]["final_status"] == historical["status"]
        if scenario_id == "I08": assert sequence[0]["external_human_semantic_review"] == "PASS"
    for row in rows_by_case(report)["I08"]:
        assert row["attempt"]["semantic_success"] is None
        assert row["attempt"]["contract_success"] is True
        assert row["attempt"]["explanation"]
    assert_safe(report)


@pytest.mark.parametrize("tamper", ["prompt_fingerprint", "context_fingerprint", "scenario_id", "candidate",
    "call_candidate", "call_scenario", "call_attempt", "empty_calls", "zero_completions", "diagnostic_count", "display_status",
    "oversized_completions"])
def test_historical_fingerprint_or_identity_drift_stops_before_factory(tmp_path, merged_ledger, tamper):
    _, original, review = merged_ledger
    altered = deepcopy(original)
    row = next(r for r in altered["results"] if r["scenario_id"] == "I10"
        and r["candidate"]["candidate_id"] == GPT.candidate_id)
    if tamper == "prompt_fingerprint": row["attempt"]["calls"][0][tamper] = "changed"
    elif tamper == "context_fingerprint": row["attempt"][tamper] = "changed"
    elif tamper == "scenario_id": row["attempt"][tamper] = "I12"
    elif tamper == "candidate": row["attempt"]["candidate"]["model_id"] = COHORT[1].model_id
    elif tamper == "call_candidate": row["attempt"]["calls"][0]["candidate"]["model_id"] = COHORT[1].model_id
    elif tamper == "call_scenario": row["attempt"]["calls"][0]["scenario_id"] = "I12"
    elif tamper == "call_attempt": row["attempt"]["calls"][0]["attempt_id"] = "different-attempt"
    elif tamper == "empty_calls": row["attempt"]["calls"] = []
    elif tamper == "zero_completions": row["provider_completions"] = 0
    elif tamper == "display_status": row["status"] = "PASS"
    elif tamper == "oversized_completions":
        row["provider_completions"] = 2  # Coherent records still exceed isolated Interpreter's one-call ceiling.
        row["attempt"]["calls"].append(deepcopy(row["attempt"]["calls"][-1]))
        row["diagnostics"].append(deepcopy(row["diagnostics"][-1]))
    else: row["diagnostics"] = []
    def forbidden(*args, **kwargs): pytest.fail("Invalid history reached factory")
    with pytest.raises(ValueError):
        run_stability.run_stability(original=altered, output=tmp_path / "invalid",
            factory=forbidden, environment={}, human_review_path=review, spacing=0)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("tamper", ["prompt", "schema"])
def test_frozen_payload_guard_rejects_drift_before_actual_sdk(tmp_path, merged_ledger, tamper):
    factory = RepeatFactory(tamper=tamper)
    with pytest.raises(PayloadValidationFailure):
        execute(tmp_path, merged_ledger, factory)
    assert len(factory.created) == 1
    _, _, client, runtime = factory.created[0]
    assert client.calls == [] and runtime.completion_calls == 0
    assert runtime._closed


@pytest.mark.parametrize("terminal", ["plan", "unsupported"])
def test_recovered_429_records_production_recovery_without_halting(tmp_path, merged_ledger, terminal):
    outcome = inspection() if terminal == "plan" else {
        "schema_version": 4, "decision": {"kind": "unsupported", "reason": "Unsupported fixture."}}
    outcomes = [turn(target="inspect_scATAC"), scope("processed_inspection"),
        SDKError(429, headers={"Retry-After": "1"}), outcome]
    report, factory = execute(tmp_path, merged_ledger, RepeatFactory(overrides={("I01", 2): outcomes}))
    assert report["complete"] is True and len(factory.created) == 8
    row = report["results"][0]
    assert row["status"] == ("PASS" if terminal == "plan" else "SEMANTIC_FAIL")
    assert row["provider_completions"] == 4 and row["recovery_calls"] == 1
    assert row["attempt"]["initial_outcome"]["provider_outcome"] == "failed"
    assert row["attempt"]["calls"][-1]["provider_outcome"] == "returned"
    assert row["attempt"]["calls"][-1]["recovery_kind"] == "transport_retry"
    assert report["blocked_models"] == {} and report["blocked_providers"] == {}
    assert not factory.created[0][2].outcomes
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 13
    assert_safe(report)


def test_persistent_unrecovered_model_limit_suppresses_remaining_repeats(tmp_path, merged_ledger):
    outcomes = [turn(target="inspect_scATAC"), SDKError(429, headers={"Retry-After": "120"})]
    report, factory = execute(tmp_path, merged_ledger, RepeatFactory(overrides={("I01", 2): outcomes}))
    assert len(factory.created) == 1
    assert report["results"][0]["status"] == "OPERATIONAL_FAIL"
    assert all(not r["attempted"] and r["provider_completions"] == 0 for r in report["results"][1:])
    assert len(report["results"]) == 8
    assert GPT.candidate_id in report["blocked_models"]
    assert report["blocked_providers"] == {}
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 2
    assert_safe(report)


@pytest.mark.parametrize("response,status,genuine", [
    ({}, "CONTRACT_FAIL", False),
    (turn("clarify", reason="ambiguous_predecessor"), "SEMANTIC_FAIL", True),
])
def test_invalid_interpreter_fallback_cannot_count_as_correct_genuine_clarify(
        tmp_path, merged_ledger, response, status, genuine):
    report, _ = execute(tmp_path, merged_ledger, RepeatFactory(overrides={("I14", 2): [response]}))
    row = rows_by_case(report)["I14"][0]
    assert row["status"] == status
    assert row["attempt"]["genuine_clarify"] is genuine
    assert row["attempt"]["semantic_success"] is not True
    assert_safe(report)


def test_wrong_admissible_current_referent_remains_semantic_failure(tmp_path, merged_ledger):
    response = turn("answer_scientific", target={"output": "@most_recently_created", "subject": None},
        comparison=None, focus="question")
    report, _ = execute(tmp_path, merged_ledger, RepeatFactory(overrides={("I10", 2): [response]}))
    row = rows_by_case(report)["I10"][0]
    assert row["status"] == "SEMANTIC_FAIL"
    assert row["attempt"]["contract_success"] is True and row["attempt"]["admission_success"] is True
    assert row["attempt"]["admitted"]["target"]["revision_id"]
    assert_safe(report)


def test_scientific_entry_guard_stops_immediately_and_closes_transport(tmp_path, merged_ledger):
    def enter_science(prompt):
        return PlanExecutor(build_default_tool_registry()).execute(None)
    factory = RepeatFactory(overrides={("I01", 2): [enter_science]})
    before = merged_ledger[0].read_bytes()
    with pytest.raises(ZeroScienceViolation):
        execute(tmp_path, merged_ledger, factory)
    assert len(factory.created) == 1 and factory.created[0][3]._closed
    assert merged_ledger[0].read_bytes() == before


def test_existing_output_rejected_before_factory_without_overwrite(tmp_path, merged_ledger):
    output = tmp_path / "stability"
    output.mkdir()
    sentinel = output / "preserve.bin"
    sentinel.write_bytes(b"preserved evidence")
    def forbidden(*args, **kwargs): pytest.fail("Existing output reached factory")
    with pytest.raises((ValueError, FileExistsError)):
        execute(tmp_path, merged_ledger, forbidden)
    assert sentinel.read_bytes() == b"preserved evidence"


def test_i08_contract_failure_is_not_queued_as_admitted_human_output(tmp_path, merged_ledger):
    def invalid(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"] = [{"kind": "claim", "id": "never-offered-claim"}]
        return response
    report, _ = execute(tmp_path, merged_ledger, RepeatFactory(overrides={("I08", 2): [invalid]}))
    rejected, admitted = rows_by_case(report)["I08"]
    assert rejected["status"] == "CONTRACT_FAIL" and rejected["attempt"]["semantic_success"] is None
    assert rejected["attempt"]["contract_success"] is False
    assert admitted["status"] == "HUMAN_REVIEW_PENDING"
    assert len(report["human_review_queue"]) == 1
    assert report["human_review_queue"][0]["attempt_id"] == "stability-3"
    assert_safe(report)


def test_new_answer_and_artifacts_redact_private_operator_values(tmp_path, merged_ledger):
    secret, operator = "private-stability-key", "private-stability-project"
    def explanation(prompt):
        response = scripted_answer(prompt)
        response["paragraphs"][0]["parts"].insert(0,
            {"kind": "text", "text": f"API key={secret}; project {operator}."})
        return response
    environment = {"GROQ_API_KEY": secret, "PROJECT_ID": operator}
    factory = RepeatFactory(environment=environment, overrides={("I08", 2): [explanation]})
    report, _ = execute(tmp_path, merged_ledger, factory, environment=environment)
    assert rows_by_case(report)["I08"][0]["status"] == "HUMAN_REVIEW_PENDING"
    assert "REDACTED" in report["human_review_queue"][0]["explanation"]
    for material in [json.dumps(report), *(p.read_text() for p in (tmp_path / "stability").rglob("*.json"))]:
        assert secret not in material and operator not in material
    assert_safe(report)


def test_spacing_is_between_independent_attempts_and_never_sleeps_for_real(tmp_path, merged_ledger):
    class Clock:
        value = 100.0
        waits = []
        def now(self): return self.value
        def sleep(self, seconds):
            assert 0 < seconds <= 60
            self.waits.append(seconds)
            self.value += seconds
    clock = Clock()
    report, _ = execute(tmp_path, merged_ledger, clock=clock.now, sleeper=clock.sleep, spacing=60)
    assert len(report["waits"]) == len(clock.waits) == 7
    assert clock.waits == [60] * 7
    assert report["min_model_spacing_seconds"] == 60


def test_production_retry_then_rejection_does_not_add_a_benchmark_repair(tmp_path, merged_ledger):
    invalid = inspection()
    invalid["decision"]["steps"][0]["sources"][0]["source"]["input"] = "never-supplied"
    outcomes = [turn(target="inspect_scATAC"), scope("processed_inspection"),
        SDKError(429, headers={"Retry-After": "1"}), invalid, inspection()]
    report, factory = execute(tmp_path, merged_ledger, RepeatFactory(overrides={
        ("I01", 2): outcomes, ("I01", 3): outcomes}))
    assert report["complete"] is True and report["all_eight_attempted"] is True
    assert report["counts"]["completions"] == report["counts"]["reserved_pre_sdk_checks"] == 14
    assert report["counts"]["completions"] <= run_stability.COMPLETION_CEILING == 16
    assert report["counts"]["recovery_calls"] == 2 and report["counts"]["operational_events"] == 2
    for row in rows_by_case(report)["I01"]:
        assert row["status"] == "CONTRACT_FAIL" and row["provider_completions"] == 4
        assert [c["recovery_kind"] for c in row["attempt"]["calls"]] == [
            "initial", "initial", "initial", "transport_retry"]
    assert sum(len(client.calls) for _, _, client, _ in factory.created) == 14
    assert [len(client.outcomes) for _, _, client, _ in factory.created] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert_safe(report)


def test_source_identity_drift_is_rejected_before_sdk_without_editing_files(
        tmp_path, merged_ledger, monkeypatch):
    source_identity = run_stability._source_identity
    calls = []
    def changed():
        value = source_identity()
        calls.append(value)
        if len(calls) > 1:
            value["preserved_breadth"][next(iter(value["preserved_breadth"]))] = "changed"
        return value
    monkeypatch.setattr(run_stability, "_source_identity", changed)
    factory = RepeatFactory()
    with pytest.raises(PayloadValidationFailure): execute(tmp_path, merged_ledger, factory)
    assert len(factory.created) == 1 and factory.created[0][2].calls == []
    assert factory.created[0][3]._closed


def test_private_captured_context_drift_is_rejected_before_sdk(tmp_path, merged_ledger, monkeypatch):
    build = harness.build_context
    armed = []
    def changed(*args, **kwargs):
        context = build(*args, **kwargs)
        if not armed: return context
        return replace(context, interaction=replace(context.interaction,
            utterance=context.interaction.utterance + " Private fixture changed."))
    monkeypatch.setattr(harness, "build_context", changed)
    class ArmingFactory(RepeatFactory):
        def __call__(self, *args, **kwargs):
            armed.append(True)
            return super().__call__(*args, **kwargs)
    factory = ArmingFactory()
    with pytest.raises(PayloadValidationFailure): execute(tmp_path, merged_ledger, factory)
    assert len(factory.created) == 1 and factory.created[0][2].calls == []
    assert factory.created[0][3]._closed


def test_transport_construction_failure_remains_zero_call_operational_row(tmp_path, merged_ledger):
    class FailingFactory(RepeatFactory):
        def __call__(self, candidate, availability, max_calls, *, before_complete):
            if not self.created:
                scenario, number = scheduled()[0]
                self.created.append((scenario, number, SimpleNamespace(calls=[]),
                    SimpleNamespace(_closed=True, completion_calls=0)))
                raise PlanningModelError("Synthetic missing configuration.", code="PLANNING_PROVIDER_CONFIGURATION_FAILED")
            return super().__call__(candidate, availability, max_calls, before_complete=before_complete)
    report, factory = execute(tmp_path, merged_ledger, FailingFactory())
    assert len(factory.created) == 8
    first = report["results"][0]
    assert first["status"] == "OPERATIONAL_FAIL" and first["attempt"] is None
    assert first["provider_completions"] == 0 and not first["attempted"]
    assert all(r["attempted"] for r in report["results"][1:])
    assert report["counts"]["completions"] == 9 and report["counts"]["attempts"] == 7
    assert_safe(report)


def test_external_human_review_must_match_admitted_historical_answer(tmp_path, merged_ledger):
    source, _, review = merged_ledger
    altered = json.loads(review.read_text())
    altered["retained_answer"] = "Different accepted output."
    path = tmp_path / "wrong-review.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        run_stability.stability_validate(source, environment={}, human_review_path=path)
