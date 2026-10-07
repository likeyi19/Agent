"""Offline acceptance of interactive qualification boundaries and zero-science guards."""
from dataclasses import replace
import json

import pytest

from agent.orchestration import LLMPlanner, PlanExecutor, build_default_tool_registry
from agent.orchestration.planning_model import PlanningModelError
from agent.schemas import AgentError, AgentRequest, ErrorCategory, VerificationResult
from benchmarks.interactive.candidates import CANDIDATES, get_candidate
from benchmarks.interactive.fixtures import scripted_answer, scripted_guidance
from benchmarks.interactive.harness import ZeroScienceViolation, run_attempt
from benchmarks.interactive.scenarios import canonical_scenarios


class ScriptedModel:
    """Each response is consumed once; an accidental extra call fails immediately."""

    model_id = "offline:interactive-qualification"

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def complete(self, *, prompt, response_schema):
        self.calls.append((json.loads(prompt), response_schema))
        assert self.outcomes, "Qualification made an unexpected model call"
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            outcome = outcome(json.loads(prompt), response_schema)
        return outcome if isinstance(outcome, str) else json.dumps(outcome)


@pytest.fixture
def candidate():
    return get_candidate("groq-gpt-oss-120b")


@pytest.fixture
def scenarios():
    return {s.scenario_id: s for s in canonical_scenarios()}


def turn(kind="execute_plan", **values):
    return {"turn_schema_version": 1, "decision": {"kind": kind, **values}}


def selection(*capabilities):
    return {"selection_schema_version": 1, "decision": {
        "kind": "select", "capability_ids": list(capabilities)}}


def inspection(*, source="input_path", tool="inspect_scATAC", step_id="arbitrary-step"):
    return {"schema_version": 4, "decision": {"kind": "plan", "steps": [{
        "step_id": step_id, "tool": tool,
        "sources": [{"target": "dataset", "source": {"kind": "input", "input": source}}],
        "control_dependencies": [],
    }]}}


def unsupported():
    return {"schema_version": 4, "decision": {
        "kind": "unsupported", "reason": "This request cannot be supported."}}


def execute_response():
    return turn(target="inspect_scATAC")


def interpreter_case(scenarios):
    return replace(scenarios["I01"], surface="interpreter",
        expected={"kind": "execute", "tool": "inspect_scATAC"})


def result(scenario, candidate, *outcomes, **kwargs):
    model = ScriptedModel(*outcomes)
    record = run_attempt(scenario, candidate, model=model, **kwargs).to_dict()
    assert not model.outcomes
    return record, model


def assert_zero_science(record):
    assert record["safety"]["scientific_calls"] == 0
    assert record["safety"]["execution_entry_attempts"] == 0
    assert record["safety"]["scientific_step_results"] == 0


def test_same_contract_runs_through_different_candidate_identities(scenarios, candidate):
    other = get_candidate("dashscope-deepseek-v4.1-flash")
    a, ma = result(interpreter_case(scenarios), candidate, execute_response())
    b, mb = result(interpreter_case(scenarios), other, execute_response())
    assert ma.calls == mb.calls
    assert a["contract_success"] is b["contract_success"] is True
    assert a["semantic_success"] is b["semantic_success"] is True
    assert a["candidate"]["provider_id"] == "groq"
    assert b["candidate"]["provider_id"] == "dashscope"
    assert b["candidate"]["model_id"] == "deepseek-v4.1-flash"
    assert b["candidate"]["adapter_family"] == other.adapter_family
    assert a["calls"][0]["prompt_fingerprint"] == b["calls"][0]["prompt_fingerprint"]
    assert_zero_science(a)
    assert_zero_science(b)


def test_interpreter_only_never_enters_planning(scenarios, candidate, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Interpreter entered Planner")
    for name in ("plan", "plan_with_diagnostics", "plan_with_recovery", "_select_scope", "_plan_detailed"):
        monkeypatch.setattr(LLMPlanner, name, forbidden)
    record, model = result(interpreter_case(scenarios), candidate, execute_response())
    assert len(model.calls) == 1
    assert record["decision_kind"] == "execute"
    assert record["contract_success"] is True
    assert record["admission_success"] is True
    assert record["compiler_success"] is record["preflight_success"] is None
    assert_zero_science(record)


def test_stage_a_does_not_enter_stage_b(scenarios, candidate, monkeypatch):
    monkeypatch.setattr(LLMPlanner, "_plan_detailed", lambda *a, **k: pytest.fail("Stage A entered B"))
    record, model = result(scenarios["I02"], candidate, selection("processed_inspection"))
    assert len(model.calls) == 1
    assert "selection_schema_version" in model.calls[0][0]
    assert record["decision_kind"] == "select"
    assert record["contract_success"] is record["semantic_success"] is True
    assert record["compiler_success"] is record["preflight_success"] is None
    assert_zero_science(record)


def test_stage_b_uses_frozen_scope_without_stage_a(scenarios, candidate, monkeypatch):
    monkeypatch.setattr(LLMPlanner, "_select_scope", lambda *a, **k: pytest.fail("Isolated B entered A"))
    record, model = result(scenarios["I03"], candidate, inspection())
    assert len(model.calls) == 1
    prompt, schema = model.calls[0]
    assert "selection_schema_version" not in prompt
    assert set(prompt["catalog"]["tools"]) == {"inspect_scATAC"}
    assert [v["properties"]["tool"]["enum"][0] for v in schema["$defs"]["step"]["anyOf"]] == ["inspect_scATAC"]
    assert record["contract_success"] is record["compiler_success"] is True
    assert record["preflight_success"] is record["semantic_success"] is True
    assert_zero_science(record)


def test_equivalent_step_names_are_not_exact_json_scored(scenarios, candidate):
    first, _ = result(scenarios["I03"], candidate, inspection(step_id="one-valid-name"))
    second, _ = result(scenarios["I03"], candidate, inspection(step_id="another-valid-name"))
    assert first["semantic_success"] is second["semantic_success"] is True
    assert first["preflight_success"] is second["preflight_success"] is True


def test_omitted_unique_source_is_equivalent_under_compiler_authority(scenarios, candidate):
    response = inspection()
    response["decision"]["steps"][0]["sources"] = []
    record, model = result(scenarios["I03"], candidate, response)
    assert len(model.calls) == 1
    assert record["contract_success"] is record["compiler_success"] is True
    assert record["preflight_success"] is record["semantic_success"] is True


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda c: c.candidate_id)
def test_all_candidates_share_the_same_stage_b_contract(scenarios, candidate):
    record, model = result(scenarios["I03"], candidate, inspection())
    assert len(model.calls) == 1
    assert record["candidate"]["provider_id"] == candidate.provider_id
    assert record["candidate"]["model_id"] == candidate.model_id
    assert record["contract_success"] is record["semantic_success"] is True
    assert_zero_science(record)


def test_producer_consumer_uses_legal_grouped_channel(scenarios, candidate):
    response = {"schema_version": 4, "decision": {"kind": "plan", "steps": [
        {"step_id": "embed", "tool": "epizoo_embed_cells", "sources": [], "control_dependencies": []},
        {"step_id": "graph", "tool": "build_cell_neighbors", "sources": [{
            "target": "embedding", "source": {"kind": "step_port", "step": "embed", "source_port": "embedding"}}],
            "control_dependencies": []},
    ]}}
    record, model = result(scenarios["I04"], candidate, response)
    assert len(model.calls) == 1
    assert record["contract_success"] is record["compiler_success"] is True
    assert record["preflight_success"] is record["semantic_success"] is True
    assert_zero_science(record)


def test_unrelated_inputs_are_not_semantically_interchangeable(scenarios, candidate):
    record, _ = result(scenarios["I05"], candidate, inspection(source="reference_input_path"))
    assert record["contract_success"] is record["compiler_success"] is True
    assert record["preflight_success"] is True
    assert record["semantic_success"] is False
    assert record["failure_category"] == "semantic"


def test_genuinely_unsupported_stage_a_is_successful_typed_refusal(scenarios, candidate):
    record, model = result(scenarios["I06"], candidate,
        {"selection_schema_version": 1, "decision": {"kind": "unsupported"}})
    assert len(model.calls) == 1
    assert record["contract_success"] is record["semantic_success"] is True
    assert record["decision_kind"] == "unsupported"
    assert record["failure_category"] is None


def test_valid_unsupported_is_semantic_failure_without_retry(scenarios, candidate):
    record, model = result(scenarios["I03"], candidate, unsupported())
    assert len(model.calls) == 1
    assert record["decision_kind"] == "unsupported"
    assert record["contract_success"] is True
    assert record["semantic_success"] is False
    assert record["failure_category"] == "semantic"
    assert record["compiler_success"] is record["preflight_success"] is None
    assert_zero_science(record)


@pytest.mark.parametrize("capabilities", [("embedding_analysis",), ("raw_preprocessing",)])
def test_wrong_but_legal_scope_is_semantic_failure(scenarios, candidate, capabilities):
    record, model = result(scenarios["I02"], candidate, selection(*capabilities))
    assert len(model.calls) == 1
    assert record["contract_success"] is True
    assert record["semantic_success"] is False
    assert record["failure_category"] == "semantic"


@pytest.mark.parametrize("payload", ["not JSON", '{"schema_version":4,"decision":{"kind":"plan","steps":[]}}',
    '{"schema_version":true,"decision":{"kind":"unsupported","reason":"x"}}'])
def test_contract_rejections_are_distinct_from_semantics(scenarios, candidate, payload):
    record, model = result(scenarios["I03"], candidate, payload)
    assert len(model.calls) == 1
    assert record["contract_success"] is False
    assert record["failure_category"] == "contract"
    assert record["compiler_success"] is record["preflight_success"] is None


@pytest.mark.parametrize("surface,scenario_id", [("interpreter", "I01"), ("stage_a", "I02"), ("stage_b", "I03")])
def test_transport_failures_record_codes_without_secret_prose(scenarios, candidate, surface, scenario_id):
    secret = "credential-value-must-not-be-recorded"
    error = PlanningModelError(secret, code="PROVIDER_RATE_LIMITED")
    record, model = result(replace(scenarios[scenario_id], surface=surface), candidate, error)
    assert len(model.calls) == 1
    assert record["failure_category"] == "operational"
    assert record["contract_success"] is None
    assert secret not in json.dumps(record)
    assert "PROVIDER_RATE_LIMITED" in json.dumps(record)
    assert_zero_science(record)


@pytest.mark.parametrize("scenario_id", ["I01", "I02", "I03"])
def test_arbitrary_model_error_code_cannot_expose_credentials(scenarios, candidate, scenario_id):
    secret = "credential-value-must-not-be-recorded"
    scenario = interpreter_case(scenarios) if scenario_id == "I01" else scenarios[scenario_id]
    record, model = result(scenario, candidate, PlanningModelError(secret, code=secret))
    assert len(model.calls) == 1
    assert record["failure_category"] == "operational"
    assert record["error_code"] == "PLANNING_PROVIDER_ERROR"
    assert record["calls"][0]["error_code"] == "PLANNING_PROVIDER_ERROR"
    assert secret not in json.dumps(record)


def test_compiler_rejection_is_deterministic_admission(scenarios, candidate):
    base = scenarios["I03"]
    request = AgentRequest(base.request.request_id, base.request.prompt,
        dict(base.request.inputs, unrelated="present-but-unauthorized"), base.request.mode)
    record, model = result(replace(base, request=request), candidate, inspection(source="unrelated"))
    assert len(model.calls) == 1
    assert record["contract_success"] is True
    assert record["compiler_success"] is False
    assert record["preflight_success"] is None
    assert record["failure_category"] == "deterministic_admission"
    assert "UNAUTHORIZED_REQUEST_INPUT" in json.dumps(record)


def test_preflight_rejection_stays_separate_from_compilation(scenarios, candidate, monkeypatch):
    def reject(self, plan):
        return VerificationResult(False, "plan", plan.plan_id,
            error=AgentError(ErrorCategory.INTERNAL_AGENT_ERROR, "INVALID_TOOL_ARGUMENTS", "Rejected fixture."))
    monkeypatch.setattr(PlanExecutor, "preflight", reject)
    record, model = result(scenarios["I03"], candidate, inspection())
    assert len(model.calls) == 1
    assert record["contract_success"] is record["compiler_success"] is True
    assert record["preflight_success"] is False
    assert record["failure_category"] == "deterministic_admission"


def test_layer2_uses_actual_admission_and_planonly_preflight(scenarios, candidate):
    case = replace(scenarios["I01"], surface="layer2")
    record, model = result(case, candidate, execute_response(), selection("processed_inspection"), inspection())
    assert len(model.calls) == 3
    assert record["contract_success"] is record["admission_success"] is True
    assert record["compiler_success"] is record["preflight_success"] is True
    assert record["semantic_success"] is True
    assert record["final_outcome"]["status"] == "PLANNED"
    assert_zero_science(record)


@pytest.mark.parametrize("failure,action", [("not JSON", "repair"),
    (PlanningModelError("sanitized", code="PROVIDER_TIMEOUT"), "transport_retry")])
def test_layer2_preserves_existing_recovery_without_harness_retries(scenarios, candidate, failure, action):
    case = replace(scenarios["I01"], surface="layer2")
    record, model = result(case, candidate, execute_response(), selection("processed_inspection"), failure, inspection())
    assert len(model.calls) == 4
    assert record["final_outcome"]["status"] == "PLANNED"
    assert record["initial_outcome"] != record["final_outcome"]
    assert any(call.get("recovery_kind") == action for call in record["calls"])
    assert record["preflight_success"] is True
    assert_zero_science(record)


def test_layer2_records_initial_compiler_failure_before_production_repair(scenarios, candidate):
    case = scenarios["I01"]
    request = AgentRequest(case.request.request_id, case.request.prompt,
        dict(case.request.inputs, unrelated="present-but-unauthorized"), case.request.mode)
    record, model = result(replace(case, request=request), candidate,
        execute_response(), selection("processed_inspection"), inspection(source="unrelated"), inspection())
    assert len(model.calls) == 4
    initial = record["initial_outcome"]
    assert initial["contract_success"] is True
    assert initial["compiler_success"] is False
    assert initial["preflight_success"] is None
    assert initial["failure_code"] == "UNAUTHORIZED_REQUEST_INPUT"
    assert record["final_outcome"]["status"] == "PLANNED"
    assert record["compiler_success"] is record["preflight_success"] is True
    assert any(call.get("recovery_kind") == "repair" for call in record["calls"])
    assert_zero_science(record)


def test_layer2_does_not_retry_a_valid_false_unsupported(scenarios, candidate):
    record, model = result(replace(scenarios["I01"], surface="layer2"), candidate,
        execute_response(), selection("processed_inspection"), unsupported())
    assert len(model.calls) == 3
    assert record["contract_success"] is True
    assert record["failure_category"] == "semantic"
    assert record["semantic_success"] is False
    assert_zero_science(record)


def test_layer2_preserves_interpreter_target_vs_actual_plan_admission(scenarios, candidate):
    record, model = result(replace(scenarios["I01"], surface="layer2"), candidate,
        turn(target="epizoo_embed_cells"), selection("processed_inspection"), inspection())
    assert len(model.calls) == 3
    assert record["contract_success"] is True
    assert record["compiler_success"] is record["preflight_success"] is True
    assert record["admission_success"] is False
    assert record["failure_category"] == "deterministic_admission"
    assert_zero_science(record)


def test_genuine_clarify_is_distinct_from_invalid_decision_fallback(scenarios, candidate):
    case = scenarios["I14"]
    genuine, authored = result(case, candidate, turn("clarify", reason="ambiguous_subject"))
    invalid, rejected = result(case, candidate, "not a decision")
    assert len(authored.calls) == len(rejected.calls) == 1
    assert genuine["contract_success"] is True and genuine["genuine_clarify"] is True
    assert genuine["semantic_success"] is True
    assert invalid["contract_success"] is False and invalid["genuine_clarify"] is False
    assert invalid["failure_category"] == "contract"


def test_layer2_invalid_interpreter_output_records_actual_clarification_fallback(scenarios, candidate):
    record, model = result(scenarios["I01"], candidate, "not a decision")
    assert len(model.calls) == 1
    assert record["presentation_kind"] == "clarify"
    assert record["contract_success"] is False
    assert record["genuine_clarify"] is False
    assert record["failure_category"] == "contract"
    assert_zero_science(record)


def test_layer2_unreached_compiler_after_stage_a_transport_is_unobserved(scenarios, candidate):
    record, model = result(scenarios["I01"], candidate, execute_response(),
        PlanningModelError("sanitized", code="PROVIDER_TIMEOUT"))
    assert len(model.calls) == 2
    assert record["contract_success"] is None
    assert record["compiler_success"] is record["preflight_success"] is None
    assert record["failure_category"] == "operational"
    assert_zero_science(record)


def test_missing_indispensable_parameter_is_genuine_clarify(scenarios, candidate):
    record, model = result(scenarios["I07"], candidate, turn("clarify", reason="missing_parameter_value"))
    assert len(model.calls) == 1
    assert record["genuine_clarify"] is True
    assert record["contract_success"] is record["semantic_success"] is True
    assert_zero_science(record)


def test_attempt_ids_and_prior_attempts_do_not_change_context(scenarios, candidate):
    case = interpreter_case(scenarios)
    before = case.request.to_dict(), dict(case.expected)
    first, ma = result(case, candidate, execute_response(), attempt_id="first")
    second, mb = result(case, candidate, execute_response(), attempt_id="second")
    assert ma.calls == mb.calls
    assert first["calls"][0]["prompt_fingerprint"] == second["calls"][0]["prompt_fingerprint"]
    assert (case.request.to_dict(), dict(case.expected)) == before
    assert first["attempt_id"] != second["attempt_id"]
    assert_zero_science(first)
    assert_zero_science(second)


def test_registry_invocation_raises_immediate_zero_science_violation(scenarios, candidate, monkeypatch):
    def malicious_preflight(self, plan):
        return self.registry.get("inspect_scATAC").function(path="/never-read.h5ad")
    monkeypatch.setattr(PlanExecutor, "preflight", malicious_preflight)
    with pytest.raises(ZeroScienceViolation):
        run_attempt(scenarios["I03"], candidate, model=ScriptedModel(inspection()))


def test_executor_entry_is_blocked_even_when_called_from_model(scenarios, candidate):
    def attack(prompt, schema):
        PlanExecutor(build_default_tool_registry()).execute(None)
        return inspection()
    with pytest.raises(ZeroScienceViolation):
        run_attempt(scenarios["I03"], candidate, model=ScriptedModel(attack))


@pytest.mark.parametrize("scenario_id,helper", [("I08", scripted_answer), ("I09", scripted_answer),
    ("I13", scripted_guidance)])
def test_answer_and_guidance_are_independent_readonly_contracts(scenarios, candidate, scenario_id, helper):
    def response(prompt, schema):
        return helper(prompt)
    record, model = result(scenarios[scenario_id], candidate, response)
    assert len(model.calls) == 1
    assert record["contract_success"] is record["admission_success"] is True
    assert record["human_review_required"] is True
    assert record["semantic_properties_pass"] is True
    assert record["semantic_success"] is None
    assert record["compiler_success"] is record["preflight_success"] is None
    assert_zero_science(record)


@pytest.mark.parametrize("scenario_id", ["I08", "I13"])
def test_readonly_provider_failure_does_not_become_contract_failure(scenarios, candidate, scenario_id):
    record, model = result(scenarios[scenario_id], candidate,
        PlanningModelError("sanitized", code="PROVIDER_RATE_LIMITED"))
    assert len(model.calls) == 1
    assert record["failure_category"] == "operational"
    assert record["contract_success"] is None
    assert record["semantic_properties_pass"] is None
    assert record["semantic_success"] is None
    assert_zero_science(record)


@pytest.mark.parametrize("scenario_id,helper", [("I08", scripted_answer), ("I13", scripted_guidance)])
def test_answer_and_guidance_reject_invented_claim_references(scenarios, candidate, scenario_id, helper):
    def bad_reference(prompt, schema):
        response = helper(prompt)
        explanation = response if scenario_id == "I08" else response["candidates"][0]["explanation"]
        explanation["paragraphs"][0]["parts"] = [{"kind": "claim", "id": "never-offered-claim"}]
        return response
    record, model = result(scenarios[scenario_id], candidate, bad_reference)
    assert len(model.calls) == 1
    assert record["contract_success"] is False
    assert record["failure_category"] == "contract"
    assert record["human_review_required"] is True
    assert record["semantic_success"] is None
    assert record["semantic_adjudication"] == "human_review_pending"
    assert_zero_science(record)


@pytest.mark.parametrize("scenario_id", ["I08", "I13"])
def test_readonly_surfaces_block_owner_reconstruction(scenarios, candidate, scenario_id):
    def attack(prompt, schema):
        from agent.tools.data.authority_context import VerificationContext
        VerificationContext().verify("matrix", lambda: None, [], {})
    with pytest.raises(ZeroScienceViolation):
        run_attempt(scenarios[scenario_id], candidate, model=ScriptedModel(attack))


@pytest.mark.parametrize("scenario_id", ["I08", "I13"])
def test_readonly_surfaces_block_report_regeneration(scenarios, candidate, scenario_id):
    def attack(prompt, schema):
        from agent.application import service
        service.build_analysis_report(None)
    with pytest.raises(ZeroScienceViolation):
        run_attempt(scenarios[scenario_id], candidate, model=ScriptedModel(attack))


def test_guidance_persistence_is_disposable_between_attempts(scenarios, candidate):
    first, ma = result(scenarios["I13"], candidate, lambda prompt, schema: scripted_guidance(prompt), attempt_id="first")
    second, mb = result(scenarios["I13"], candidate, lambda prompt, schema: scripted_guidance(prompt), attempt_id="second")
    assert ma.calls == mb.calls
    assert first["context_fingerprint"] == second["context_fingerprint"]
    assert first["evidence_context_fingerprint"] == second["evidence_context_fingerprint"]
    assert_zero_science(first)
    assert_zero_science(second)


@pytest.mark.parametrize("scenario_id,referent", [("I10", "@current_result"),
    ("I11", "@most_recently_created"), ("I12", "@previous_turn_result")])
def test_referent_selection_resolves_exact_identity(scenarios, candidate, scenario_id, referent):
    response = turn("answer_scientific", target={"output": referent, "subject": None},
        comparison=None, focus="question")
    record, model = result(scenarios[scenario_id], candidate, response)
    assert len(model.calls) == 1
    assert record["contract_success"] is record["admission_success"] is True
    assert record["semantic_success"] is True
    assert record["compiler_success"] is record["preflight_success"] is None
    assert_zero_science(record)


@pytest.mark.parametrize("scenario_id,wrong_referent", [("I10", "@most_recently_created"),
    ("I11", "@current_result")])
def test_admissible_wrong_referent_is_semantic_failure(scenarios, candidate, scenario_id, wrong_referent):
    response = turn("answer_scientific", target={"output": wrong_referent, "subject": None},
        comparison=None, focus="question")
    record, model = result(scenarios[scenario_id], candidate, response)
    assert len(model.calls) == 1
    assert record["contract_success"] is record["admission_success"] is True
    assert record["semantic_success"] is False
    assert record["failure_category"] == "semantic"
    assert_zero_science(record)
