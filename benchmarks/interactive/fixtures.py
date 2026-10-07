"""Disposable scripted accepted metadata, using actual Session/evidence readers.

Adapted from tests/application/test_dialogue_evidence.py's accepted fixture.
These records qualify dialogue contracts only. They are not owner-generated
scientific results, reusable scientific authority, or biological qualification.
No fixture invokes a scientific implementation or reconstructs owner outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from agent.application import (
    ApplicationResult, ApplicationStatus, ArtifactReference, OutputSelection,
    ResearchAgentApplication,
)
from agent.application.session_state import Interaction, canonical, digest
from agent.application.turn_context import public_context, snapshot
from agent.application import scientific_dialogue as dialogue
from agent.report import evidence as ev
from agent.schemas import (
    AgentPlan, AgentRequest, PlanStep, StepExecutionResult, StepStatus,
    VerificationCheck, VerificationResult,
)
from agent.schemas.orchestration import _serialize, freeze_json_mapping
from agent.schemas.run_state import (
    PersistedRunState, RecoveryPolicySnapshot, RunLifecycleStatus,
    ToolRecoveryPolicySnapshot, fingerprint_plan, fingerprint_recovery_policy,
)


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    description: str
    source: str = "scripted_accepted_metadata"
    biological_qualification: bool = False


FIXTURES = {
    name: FixtureSpec(name, description) for name, description in (
        ("empty", "No scientific Revision; current-turn inputs only."),
        ("selection", "Accepted explicit QC selection with a fixed TSS threshold."),
        ("matrix", "Accepted neutral matrix summary including a nonzero count."),
        ("annotation", "Accepted annotation summary with two exact subjects."),
        ("rollback", "Two created inspection revisions; the older is active."),
        ("previous_answer", "Immediately preceding scientific answer targets annotation subject 3."),
        ("ambiguous", "Two independent accepted scientific steps in one Revision."),
    )
}


@dataclass(frozen=True)
class FrozenContext:
    sessions: object
    session_id: str
    interaction: Interaction
    _public: object
    _provenance: object

    @property
    def public(self):
        """Return an independent JSON-native projection for each caller."""
        return _serialize(self._public)

    @property
    def provenance(self):
        return _serialize(self._provenance)


def _passed(kind, identity):
    return VerificationResult(True, kind, identity, (
        VerificationCheck("metadata_fixture", True, "Scripted metadata; no scientific verification."),))


def _accepted(app, session_id, tool, *, turn="accepted", output="inspection", overrides=None,
              arguments=None, copies=1):
    """Publish scripted result metadata through the normal completion records."""
    sessions = app.sessions
    state = sessions.load(session_id)
    spec = app.registry.get(tool)
    request = AgentRequest(turn, "Scripted accepted metadata fixture; not biological qualification.", {})
    defaults = {str: "fixture", int: 2, float: 0.5, bool: True, dict: {}, list: []}
    result = {key: defaults[types[0]] for key, types in spec.result_contract.required_fields.items()}
    result.update(overrides or {})
    key = "manifest_path" if "manifest_path" in result else "n_cells"
    selections = tuple(OutputSelection(output if copies == 1 else f"{output}{i}", f"step{i}", key)
                       for i in range(copies))
    sessions.start_turn(session_id, turn, request, selections, expected_generation=state.generation)
    sessions.link_run(session_id, turn)
    effective, paths = app._prepare_request(request)
    plan = AgentPlan(turn + ":plan", turn, "metadata-fixture", tuple(
        PlanStep(f"step{i}", tool, arguments or {}) for i in range(copies)))
    steps = tuple(StepExecutionResult(step.step_id, tool, StepStatus.SUCCEEDED, attempt_count=1,
        result=result, verification=_passed("step", step.step_id), resolved_arguments=arguments or {},
        started_at="2026-10-07T00:00:00+00:00", finished_at="2026-10-07T00:00:00+00:00", duration_seconds=0.0)
        for step in plan.steps)
    policies = (ToolRecoveryPolicySnapshot(tool, spec.recovery_policy_version),)
    policy = RecoveryPolicySnapshot("metadata-fixture", 1, policies,
        fingerprint_recovery_policy("metadata-fixture", 1, policies))
    stored = PersistedRunState(3, 0, turn + ":run", effective, RunLifecycleStatus.SUCCEEDED,
        "2026-10-07T00:00:00+00:00", "2026-10-07T00:00:00+00:00", plan=plan,
        plan_fingerprint=fingerprint_plan(plan), recovery_policy_snapshot=policy,
        preflight_verification=_passed("plan", plan.plan_id), steps=steps,
        run_verification=_passed("run", plan.plan_id))
    app.run_store.create(stored)
    run = stored.to_run_result()
    fields = ev._TOOL_PROJECTIONS[tool].fact_fields
    facts = {key: result[key] for key in fields}
    if tool == "select_scATAC_cells":
        facts["effective_thresholds"] = dict(arguments or {})
    payload = dict(schema_version=ev.ANALYSIS_EVIDENCE_SCHEMA_VERSION,
        artifact_type=ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, status="success",
        run=dict(run_id=run.run_id, request_id=turn, plan_id=plan.plan_id,
            planner_name=plan.planner_name, source_run_status="SUCCEEDED",
            plan_sha256=digest(plan.to_dict()), source_run_result_sha256=digest(run.to_dict())),
        workflow=dict(ordered_steps=[dict(step_id=s.step_id, tool_name=tool, depends_on=[]) for s in steps]),
        steps=[dict(step_id=s.step_id, tool_name=tool, attempt_count=1, facts=facts,
            verification=dict(passed=True, freshly_verified=True, check_names=["metadata_fixture"]),
            recovery_identity=spec.recovery_policy_version) for s in steps],
        artifacts=[], provenance=dict(trace_sha256=digest([]),
            evidence_projection_version=ev.ANALYSIS_EVIDENCE_SCHEMA_VERSION,
            registry_tool_identities=[dict(tool_name=tool, result_contract=spec.result_contract.name,
                recovery_identity=spec.recovery_policy_version)] * copies))
    evidence = paths.evidence / ev.ANALYSIS_EVIDENCE_FILENAME
    evidence.write_bytes(canonical(payload))
    report = paths.report / "metadata_fixture.json"
    report.write_bytes(b"{}")
    completed = ApplicationResult(turn, run.run_id, ApplicationStatus.SUCCEEDED, run.status,
        str(paths.root), run,
        ArtifactReference(ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, str(evidence), digest(payload)),
        report=ArtifactReference("metadata-fixture", str(report), digest({})))
    sessions._record_result(session_id, turn, completed)
    state = sessions.recover(session_id, turn)
    return state.turn(turn).revision_id


def _capture(sessions, session_id, utterance, turn="qualification-turn"):
    state = sessions.load(session_id)
    names = {}
    captured = snapshot(sessions, state, tool_names=names)
    captured["dialogue"] = dialogue.capture(sessions, state, captured, None, tool_names=names)
    return state, captured, Interaction(turn, utterance, state.active_revision_id, state.generation, captured)


def build_context(root, spec="empty", registry=None, *, utterance="Explain this result.",
                  execution_inputs=None, include_interaction=True):
    """Create a fresh disposable workspace; refuse reuse instead of contaminating attempts.

    Result identifiers, values and model-visible context do not contain root or
    attempt IDs. Private application paths stay local to each disposable store.
    """
    if isinstance(spec, str):
        spec = FIXTURES[spec]
    if not isinstance(spec, FixtureSpec) or spec != FIXTURES.get(spec.fixture_id):
        raise ValueError("Unknown canonical metadata fixture.")
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Qualification fixture root must be fresh and empty.")
    root.mkdir(parents=True, exist_ok=True)
    app = ResearchAgentApplication(root, registry=registry)
    session_id = "qualification"
    app.sessions.create(session_id)
    annotation = dict(n_groups=2, groups_omitted=0, validation_state="not_assessed", group_summary=[
        dict(group="3", n_cells=8, primary_annotation="CD8 T", status="assigned"),
        dict(group="7", n_cells=5, primary_annotation="B cell", status="assigned")])
    if spec.fixture_id == "selection":
        _accepted(app, session_id, "select_scATAC_cells", output="selection",
            overrides=dict(n_selected=8, n_rejected=5, cell_call_state="not_assessed"),
            arguments=dict(min_qc_fragment_records=1, min_tss_enrichment="4"))
    elif spec.fixture_id == "matrix":
        _accepted(app, session_id, "build_scATAC_cell_by_features", output="matrix",
            overrides=dict(n_cells=8, n_features=15, nnz=32,
                matrix_semantics="canonical-fragment-record-overlap-counts.v1"))
    elif spec.fixture_id in ("annotation", "previous_answer"):
        _accepted(app, session_id, "annotate_scATAC_cell_types", output="annotation", overrides=annotation)
    elif spec.fixture_id == "rollback":
        older = _accepted(app, session_id, "inspect_scATAC", turn="older", overrides=dict(n_cells=2))
        _accepted(app, session_id, "inspect_scATAC", turn="newer", overrides=dict(n_cells=5))
        app.sessions.switch(session_id, "rollback", older, expected_generation=2)
    elif spec.fixture_id == "ambiguous":
        _accepted(app, session_id, "inspect_scATAC", copies=2, overrides=dict(n_cells=2))
    sessions = app.sessions
    sessions._interaction_session_id = session_id
    if spec.fixture_id == "previous_answer":
        state, captured, prior = _capture(sessions, session_id, "Explain cluster 3.", "previous-discussion")
        locator = state.revisions[-1].outputs[0]
        admitted = dict(kind="answer", intent="scientific", dialogue_version=1,
            target=dict(revision_id=state.active_revision_id, output_name=locator.name,
                accepted_step_sha256=locator.accepted_step_sha256, subject="3"),
            comparison=None, previous_subject=None, focus=prior.utterance, predecessor=None)
        prior = replace(prior, status="answered", admitted=admitted)
        sessions._store._update(session_id, lambda s: replace(s, interactions=s.interactions + (prior,)))
    state, captured, interaction = _capture(sessions, session_id, utterance)
    public = public_context(captured)
    public["dialogue"] = dialogue.public(captured, state, app.registry, sessions=sessions,
        utterance=utterance, execution_inputs=execution_inputs)
    if include_interaction:
        sessions._store._update(session_id, lambda s: replace(s, interactions=s.interactions + (interaction,)))
    provenance = dict(fixture_id=spec.fixture_id, source=spec.source,
        biological_qualification=spec.biological_qualification,
        reusable_scientific_authority=False, semantic_context_sha256=digest(public))
    return FrozenContext(sessions, session_id, interaction,
        freeze_json_mapping(public, "fixture.public"), freeze_json_mapping(provenance, "fixture.provenance"))


def scripted_answer(public, field=None, *, support=None):
    """A legal contract witness, not an oracle for scientific prose quality."""
    if "evidence" in public:
        public = public["evidence"]
    claims = [c for c in public["claims"] if field is None or c["field"] == field]
    selected = claims[:1]
    return dict(support=support or ("supported" if selected else "insufficient_evidence"),
        paragraphs=[dict(parts=[dict(kind="text", text="The accepted summary records:")]
            + [dict(kind="claim", id=c["claim_id"]) for c in selected])])


def scripted_guidance(public, capability="inspect_scATAC"):
    """Three bounded conditional paragraphs using the production claim slots."""
    evidence = public["context"]["evidence"] if "context" in public else public["evidence"]
    explanation = scripted_answer(evidence)
    explanation["paragraphs"].extend([
        dict(parts=[dict(kind="text", text="This assumes an appropriate experimental design.")]),
        dict(parts=[dict(kind="text", text="Further inputs and scientific checks would be required.")])])
    return dict(candidates=[dict(capability=capability, explanation=explanation)])
