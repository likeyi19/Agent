"""Frozen interactive qualification metadata and production reference resolution."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from agent.application import scientific_dialogue as dialogue
from agent.application import scientific_guidance as guidance
from agent.application.turn_decisions import (
    GuidanceQuestion, IntentError, ScientificQuestion, ScientificTarget,
)
from agent.schemas import RunMode
from benchmarks.interactive.fixtures import (
    FIXTURES, build_context, scripted_answer, scripted_guidance,
)
from benchmarks.interactive.scenarios import SURFACES, Scenario, canonical_scenarios
from benchmarks.planner.benchmark import guarded_registry


class Witness:
    def __init__(self, response):
        self.response, self.calls = response, []

    def complete(self, *, prompt, response_schema):
        data = json.loads(prompt)
        self.calls.append((data, response_schema))
        return json.dumps(self.response(data))


def _admit(context, handle="@current_result", subject=None):
    return dialogue.admit(context.sessions, context.interaction,
        ScientificQuestion(ScientificTarget(handle, subject)))


def _forbid_generation_and_science(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata fixture entered science or report generation.")
    from agent.application import service
    from agent.orchestration import verifier
    from agent.orchestration.runtime import AgentRuntime
    from agent.orchestration.executor import PlanExecutor
    from agent.tools.data.authority_context import VerificationContext
    for owner, names in (
        (AgentRuntime, ("run", "resume")), (PlanExecutor, ("execute",)),
        (VerificationContext, ("verify",)),
        (verifier, ("verify_step", "verify_run")),
        (service, ("build_analysis_evidence", "verify_analysis_evidence", "build_analysis_report")),
        (ev_module(), ("verify_step", "verify_run", "build_analysis_evidence", "verify_analysis_evidence")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


def ev_module():
    from agent.report import evidence
    return evidence


def test_inventory_is_frozen_and_covers_all_isolated_surfaces():
    scenarios = canonical_scenarios()
    assert len(scenarios) == len({s.scenario_id for s in scenarios}) == 14
    assert {s.surface for s in scenarios} == SURFACES
    assert all(s.request.mode is RunMode.PLAN_ONLY for s in scenarios if s.request)
    assert all(s.human_review_required for s in scenarios if s.surface in ("answer", "guidance"))
    with pytest.raises(TypeError):
        scenarios[0].expected["kind"] = "unsupported"
    with pytest.raises(FrozenInstanceError):
        scenarios[0].utterance = "replacement"
    with pytest.raises(ValueError):
        Scenario("bad", "arbitrary-python", "Run code.")


@pytest.mark.parametrize("fixture", tuple(FIXTURES))
def test_fresh_attempts_have_identical_semantic_context_and_no_science(tmp_path, monkeypatch, fixture):
    registry, guard = guarded_registry()
    _forbid_generation_and_science(monkeypatch)
    first = build_context(tmp_path / "first", fixture, registry, utterance="Explain this result.")
    original = first.public
    leaked_copy = first.public
    leaked_copy["dialogue"]["tools"].clear()
    first.sessions._store._update(first.session_id, lambda state: replace(
        state, interactions=tuple(replace(i, status="failed") if i.turn_id == first.interaction.turn_id
            else i for i in state.interactions)))
    second = build_context(tmp_path / "second", fixture, registry, utterance="Explain this result.")
    assert second.public == original == first.public
    assert second.provenance == first.provenance
    assert second.sessions.load(second.session_id).interactions[-1].status == "interpreting"
    assert guard.count == 0
    assert second.provenance["source"] == "scripted_accepted_metadata"
    assert second.provenance["biological_qualification"] is False
    assert second.provenance["reusable_scientific_authority"] is False
    with pytest.raises(ValueError, match="fresh"):
        build_context(tmp_path / "second", fixture, registry)


def test_supplied_input_projection_never_exposes_values(tmp_path):
    context = build_context(tmp_path / "context", "empty", execution_inputs={
        "input_path": "/private/secret.h5ad", "unregistered-private-name": "secret-value"})
    visible = json.dumps(context.public)
    assert "secret.h5ad" not in visible and "secret-value" not in visible
    supplied = context.public["dialogue"]["supplied_inputs"]
    assert supplied["present"] and supplied["omitted_count"] == 1
    assert supplied["fields"][0]["name"] == "input_path"
    assert any(c["tool"] == "inspect_scATAC" for c in supplied["fields"][0]["consumers"])


def test_interaction_optional_for_real_layer2_submission(tmp_path):
    context = build_context(tmp_path / "context", include_interaction=False)
    assert context.interaction.turn_id == "qualification-turn"
    assert context.sessions.load(context.session_id).interactions == ()


def test_current_and_latest_referents_resolve_distinct_exact_revisions(tmp_path):
    context = build_context(tmp_path / "context", "rollback")
    state = context.sessions.load(context.session_id)
    current = _admit(context, "@current_result")["target"]
    recent = _admit(context, "@most_recently_created")["target"]
    assert current["revision_id"] == state.turn("older").revision_id == state.active_revision_id
    assert recent["revision_id"] == state.turn("newer").revision_id
    assert current["accepted_step_sha256"] != recent["accepted_step_sha256"]
    assert current["output_name"] == recent["output_name"] == "inspection"
    with pytest.raises(IntentError, match="unavailable_context"):
        _admit(context, "@previous_turn_result")


def test_previous_turn_preserves_exact_subject_and_source(tmp_path):
    context = build_context(tmp_path / "context", "previous_answer")
    admitted = _admit(context, "@previous_turn_result")
    prior = context.sessions.load(context.session_id).interactions[0]
    assert admitted["target"] == dict(prior.admitted["target"])
    assert admitted["target"]["subject"] == "3"
    assert admitted["predecessor"] == "previous-discussion"


def test_ambiguous_independent_outputs_cannot_resolve_by_first_match(tmp_path):
    context = build_context(tmp_path / "context", "ambiguous")
    groups = context.public["dialogue"]["result_groups"]
    assert len(groups) == 2
    for handle in ("@current_result", "@most_recently_created"):
        assert context.public["dialogue"]["result_referents"][handle]["status"] == "ambiguous"
        with pytest.raises(IntentError, match="ambiguous_subject"):
            _admit(context, handle)


@pytest.mark.parametrize("fixture,subject,field,value", (
    ("matrix", None, "nnz", 32),
    ("annotation", "3", "primary_annotation", "CD8 T"),
))
def test_scientific_generation_replays_production_claims_without_science(tmp_path, monkeypatch, fixture, subject, field, value):
    registry, guard = guarded_registry()
    _forbid_generation_and_science(monkeypatch)
    context = build_context(tmp_path / "context", fixture, registry, utterance="Explain cluster 3.")
    admitted = _admit(context, subject=subject)
    public, claims = dialogue.context(context.sessions, context.session_id, admitted)
    model = Witness(lambda p: scripted_answer(p["evidence"], field))
    response = dialogue.generate(model, context.interaction.utterance, admitted["focus"], public, claims)
    assert response.support == "supported"
    assert len(model.calls) == 1 and response.claims[0].value == value and guard.count == 0
    assert "source" not in model.calls[0][0]["evidence"]["claims"][0]
    assert response.claims[0].source["output_locator"]["accepted_step_sha256"] == admitted["target"]["accepted_step_sha256"]


def test_guidance_uses_disposable_metadata_and_fixed_authoritative_context(tmp_path, monkeypatch):
    registry, guard = guarded_registry()
    _forbid_generation_and_science(monkeypatch)
    context = build_context(tmp_path / "context", "annotation", registry, utterance="What analyses could help investigate cluster 3?")
    admitted = guidance.admit(context.sessions, context.interaction,
        GuidanceQuestion((ScientificTarget("@current_result", "3"),)))
    interaction = replace(context.interaction, admitted=admitted, status="admitted")
    context.sessions._store._update(context.session_id, lambda state: replace(state,
        interactions=tuple(interaction if i.turn_id == interaction.turn_id else i for i in state.interactions)))
    before = context.sessions.load(context.session_id)
    model = Witness(scripted_guidance)
    result = guidance.answer(context.sessions, context.session_id, interaction, model)
    after = context.sessions.load(context.session_id)
    assert result.status == "answered" and len(model.calls) == 1 and guard.count == 0
    assert result.guidance.candidates[0].reference["capability"] == "inspect_scATAC"
    assert result.guidance.candidates[0].reference["targets"][0]["subject"] == "3"
    assert after.generation == before.generation and after.revisions == before.revisions
    assert after.interactions[-1].guidance_candidates is not None
    fresh = build_context(tmp_path / "fresh", "annotation", registry, utterance=context.interaction.utterance)
    assert fresh.public == context.public and fresh.sessions.load(fresh.session_id).interactions[-1].guidance_candidates is None


def test_corrupt_disposable_evidence_never_changes_other_attempt(tmp_path):
    first = build_context(tmp_path / "first", "matrix")
    second = build_context(tmp_path / "second", "matrix")
    first_state = first.sessions.load(first.session_id)
    evidence_file = next(f for f in first_state.turns[0].completion_files if f.path.endswith("analysis_evidence.json"))
    from pathlib import Path
    path = Path(evidence_file.path)
    path.write_bytes(path.read_bytes() + b" ")
    output = first_state.revisions[0].outputs[0]
    corrupt = first.sessions.evidence(first.session_id, first_state.active_revision_id, output.name)
    assert corrupt.status == "unavailable" and not corrupt.facts
    second_state = second.sessions.load(second.session_id)
    valid = second.sessions.evidence(second.session_id, second_state.active_revision_id, output.name)
    assert valid.status == "available"
