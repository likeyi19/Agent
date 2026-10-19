"""Captured Session projection and exact retained-source consistency."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.application import OutputSelection, ResearchAgentApplication
from agent.application.session_state import Interaction, OutputLocator, SessionError
from agent.application.turn_context import snapshot
from agent.orchestration import AgentPlan, AgentRequest, LLMPlanner, PlanStep, StepOutputRef
from agent.orchestration.prior_outputs import (
    accepted_step_digest, bind_output, optional_binding_for_locator, validate_active_outputs,
)
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.verification_authority import AuthorityError
from agent.tools.data.authority_context import VerificationContext
from test_processed_prior_outputs import processed_source, ScientificPlanModel, consumer
from test_propagation import chain
from test_qualification import legacy


def capture(sessions, session_id, turn_id='future'):
    state = sessions.load(session_id)
    interaction = Interaction(turn_id, 'Continue the current analysis.', state.active_revision_id,
                              state.generation, snapshot(sessions, state))
    sessions._store._update(session_id, lambda previous: replace(previous,
        interactions=previous.interactions + (interaction,)))
    return interaction


def test_persisted_interaction_context_needs_no_turn_and_performs_no_writes(processed_source):
    source = processed_source
    interaction = capture(source.app.sessions, 'analysis')
    before = source.app.sessions.load('analysis').to_dict()
    context, locators = source.app.sessions.planning_context_for_interaction('analysis', 'future')
    assert context.base_revision_id == interaction.base_revision_id
    assert context.base_generation == interaction.base_generation
    assert context.items[0].binding == source.binding
    assert locators == {'ctx.0': source.session.revisions[0].outputs}
    assert not any(turn.turn_id == 'future' for turn in source.app.sessions.load('analysis').turns)
    assert source.app.sessions.load('analysis').to_dict() == before
    with pytest.raises(TypeError):
        locators['ctx.1'] = ()


def test_metadata_projection_checks_records_without_reading_publication_bytes(processed_source, monkeypatch):
    source = processed_source
    capture(source.app.sessions, 'analysis')
    def forbidden(*args):
        pytest.fail('Metadata projection read scientific artifact integrity')
    monkeypatch.setattr(VerificationContext, 'files', forbidden)
    context, _ = source.app.sessions.planning_context_for_interaction('analysis', 'future', integrity=False)
    assert context.items[0].binding == source.binding


def test_corrupt_publication_is_never_silently_removed_from_execution_context(processed_source):
    source = processed_source
    capture(source.app.sessions, 'analysis')
    path = Path(source.run.steps[0].result['analysis_path'])
    path.write_bytes(path.read_bytes() + b'changed')
    context, _ = source.app.sessions.planning_context_for_interaction('analysis', 'future', integrity=False)
    assert context.items[0].binding == source.binding  # Historical metadata is not current readiness.
    with pytest.raises((AuthorityError, ValueError, OSError)):
        source.app.sessions.planning_context_for_interaction('analysis', 'future')


def test_captured_context_never_follows_a_later_navigation_pointer(processed_source):
    source = processed_source
    app, _, request, outputs = consumer(source)
    state = app.sessions.run('analysis', 'later', request, outputs, expected_generation=1,
                             use_active_context=True)
    interaction = capture(app.sessions, 'analysis')
    app.sessions.switch('analysis', 'back', state.revisions[0].revision_id, expected_generation=2)
    context, locators = app.sessions.planning_context_for_interaction('analysis', 'future')
    # The clustering remains unqualified. Its exact recorded graph dependency
    # is reusable even though the output selector kept only the clustering.
    assert context.items[0].binding == source.binding
    assert context.base_revision_id == interaction.base_revision_id
    assert locators == {'ctx.0': ()}  # No implicit retained output or revision edit.
    assert interaction.base_revision_id == state.revisions[1].revision_id


def test_empty_or_foreign_session_cannot_supply_another_sessions_context(processed_source):
    source = processed_source
    capture(source.app.sessions, 'analysis')
    source.app.sessions.create('empty')
    capture(source.app.sessions, 'empty', 'empty-turn')
    context, locators = source.app.sessions.planning_context_for_interaction('empty', 'empty-turn')
    assert context is None and not locators
    with pytest.raises(SessionError, match='Unknown captured interaction'):
        source.app.sessions.planning_context_for_interaction('empty', 'future')


def test_cooutput_aliases_share_one_handle_and_preserve_each_exact_locator(processed_source, tmp_path):
    source = processed_source
    app = ResearchAgentApplication(tmp_path / 'aliases', planner=LLMPlanner(ScientificPlanModel()))
    app.sessions.create('aliases')
    state = app.sessions.run('aliases', 'original', AgentRequest('aliases', 'Build neighbors.', {
        'embedding_path': str(source.embedding), 'cell_ids_path': str(source.cell_ids)}),
        (OutputSelection('graph-a', 'neighbors', 'analysis_path'),
         OutputSelection('graph-b', 'neighbors', 'analysis_path'),
         OutputSelection('cell-count', 'neighbors', 'n_cells')), expected_generation=0)
    assert state.turn('original').status == 'activated'
    capture(app.sessions, 'aliases')
    context, locators = app.sessions.planning_context_for_interaction('aliases', 'future')
    assert len(context.items) == 1
    assert locators['ctx.0'] == state.revisions[0].outputs[:2]


def test_missing_accepted_authority_is_nonreusable_without_implicit_upgrade(processed_source):
    source = processed_source
    stored = source.run.steps[0]
    previous = replace(stored, verification=replace(stored.verification, artifact_authority=None))
    state = replace(source.run, steps=(previous,))
    locator = replace(source.session.revisions[0].outputs[0], accepted_step_sha256=accepted_step_digest(previous))
    store = SimpleNamespace(load=lambda _: state)
    assert optional_binding_for_locator(locator, store, source.app.registry) is None
    with pytest.raises(AuthorityError):
        optional_binding_for_locator(source.session.revisions[0].outputs[0], store, source.app.registry)


def test_corrupt_recorded_qualification_never_becomes_ordinary_absence(processed_source):
    source = processed_source
    store = SimpleNamespace(load=lambda _: source.run,
        _load_qualified_authorities=lambda _: {'neighbors': {'schema_version': 2}})
    with pytest.raises(AuthorityError):
        optional_binding_for_locator(source.session.revisions[0].outputs[0], store, source.app.registry)


@pytest.fixture
def custom_report(tmp_path):
    from agent.orchestration import AgentRuntime, ErrorCategory, ErrorClassification, ResultContract, ToolRegistry, ToolSpec
    from agent.orchestration.run_store import FileRunStore
    registry = ToolRegistry((ToolSpec('custom_report', lambda: {'value': 'reviewed'}, {}, {},
        ResultContract('CustomReport', {'value': (str,)}),
        lambda _: ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'REPORT_ERROR')),))
    plan = AgentPlan('custom-plan', 'custom-request', 'Review an explicit report.',
                     (PlanStep('report', 'custom_report', {}),))
    planner = SimpleNamespace(plan=lambda request, registry: plan)
    store = FileRunStore(tmp_path / 'custom-runs')
    result = AgentRuntime(planner=planner, registry=registry, run_store=store).run(
        AgentRequest('custom-request', 'Review the report.', {}))
    assert result.status.value == 'SUCCEEDED'
    run = store.load(result.run_id)
    step = run.steps[0]
    assert step.verification.artifact_authority is None
    locator = OutputLocator('report', run.run_id, step.step_id, 'value', accepted_step_digest(step))
    return registry, store, run, locator


def test_custom_registered_nonreusable_report_needs_no_default_registry_authority(custom_report):
    registry, store, _, locator = custom_report
    assert optional_binding_for_locator(locator, store, registry) is None
    validate_active_outputs((locator,), store, registry)


@pytest.mark.parametrize('invalid', ['target', 'failed'])
def test_custom_nonreusable_report_still_requires_an_exact_accepted_record(custom_report, invalid):
    registry, _, run, locator = custom_report
    step = run.steps[0]
    verification = replace(step.verification, **({'target_id': 'other-step'} if invalid == 'target' else {'passed': False}))
    changed = replace(step, verification=verification)
    malformed = SimpleNamespace(run_id=run.run_id, plan=run.plan, steps=(changed,),
                                lifecycle_status=run.lifecycle_status)
    store = SimpleNamespace(load=lambda _: malformed)
    locator = replace(locator, accepted_step_sha256=accepted_step_digest(changed))
    with pytest.raises(AuthorityError, match='No recorded scientific authority'):
        optional_binding_for_locator(locator, store, registry)


def test_existing_explicit_legacy_qualification_is_admitted(legacy):
    from agent.orchestration import build_default_tool_registry
    from agent.orchestration.verification_authority import qualify_legacy_authorities
    _, _, state, store = legacy
    step = state.steps[0]
    locator = OutputLocator('fragments', state.run_id, step.step_id, 'manifest_path', accepted_step_digest(step))
    registry = build_default_tool_registry()
    assert optional_binding_for_locator(locator, store, registry) is None
    qualify_legacy_authorities(store, state.run_id)
    assert optional_binding_for_locator(locator, store, registry) == bind_output(
        store, registry, state.run_id, step.step_id, 'fragments')


class DownstreamPlanner:
    def __init__(self, binding, *, umap=False, control=None):
        self.binding, self.umap, self.control = binding, umap, control

    def plan(self, request, registry):
        root = request.inputs['output_dir']
        steps = []
        if self.control is not None:
            steps.append(PlanStep('unrelated', 'cluster_cells', {
                'analysis_path': PriorOutputRef(self.control, 'analysis_path'),
                'output_dir': str(Path(root) / 'unrelated')}))
        steps.append(PlanStep('cluster', 'cluster_cells', {
            'analysis_path': PriorOutputRef(self.binding, 'analysis_path'),
            'output_dir': str(Path(root) / 'cluster')}, ('unrelated',) if self.control else ()))
        if self.umap:
            steps.append(PlanStep('umap', 'compute_cell_umap', {
                'analysis_path': StepOutputRef('cluster', 'analysis_path'),
                'output_dir': str(Path(root) / 'umap')}, ('cluster',)))
        return AgentPlan(request.request_id + '-plan', request.request_id, 'Explicit accepted sources.', tuple(steps))


def second_binding(source):
    app = ResearchAgentApplication(source.app.workspace_root, planner=LLMPlanner(ScientificPlanModel()))
    result = app.run(AgentRequest('second-neighbors', 'Build another graph.', {
        'embedding_path': str(source.embedding), 'cell_ids_path': str(source.cell_ids)}))
    assert result.status.value == 'SUCCEEDED'
    return bind_output(app.run_store, app.registry, result.run_id, 'neighbors', 'neighbors')


@pytest.mark.parametrize('umap', [False, True])
@pytest.mark.parametrize('retain', [False, True])
def test_nonreusable_new_reports_can_retain_their_exact_qualified_source(processed_source, umap, retain):
    source = processed_source
    app = ResearchAgentApplication(source.app.workspace_root, planner=DownstreamPlanner(source.binding, umap=umap))
    step = 'umap' if umap else 'cluster'
    state = app.sessions.run('analysis', 'later', AgentRequest('retained', 'Continue.', {}),
        (OutputSelection('result', step, 'analysis_path'),), expected_generation=1, retain=('neighbors',) if retain else ())
    assert state.turn('later').status == 'activated'
    if retain:
        assert state.revisions[-1].outputs[-1] == source.session.revisions[0].outputs[0]
    else:
        assert len(state.revisions[-1].outputs) == 1
    validate_active_outputs(state.revisions[-1].outputs, app.run_store, app.registry)
    capture(app.sessions, 'analysis')
    context, locators = app.sessions.planning_context_for_interaction('analysis', 'future')
    assert len(context.items) == 1 and context.items[0].binding == source.binding
    assert locators == {'ctx.0': (source.session.revisions[0].outputs[0],) if retain else ()}


@pytest.mark.parametrize('umap', [False, True])
def test_incompatible_retention_fails_through_exact_prior_and_same_plan_references(processed_source, umap):
    source = processed_source
    different = second_binding(source)
    app = ResearchAgentApplication(source.app.workspace_root, planner=DownstreamPlanner(different, umap=umap))
    step = 'umap' if umap else 'cluster'
    with pytest.raises(AuthorityError, match='different active upstream publication'):
        app.sessions.run('analysis', 'invalid', AgentRequest('invalid-retention', 'Continue.', {}),
            (OutputSelection('result', step, 'analysis_path'),), expected_generation=1, retain=('neighbors',))
    assert app.sessions.load('analysis').generation == 1
    assert len(app.sessions.load('analysis').revisions) == 1


def test_control_dependencies_are_not_inferred_as_scientific_lineage(processed_source):
    source = processed_source
    different = second_binding(source)
    app = ResearchAgentApplication(source.app.workspace_root,
        planner=DownstreamPlanner(source.binding, control=different))
    state = app.sessions.run('analysis', 'later', AgentRequest('control', 'Continue.', {}),
        (OutputSelection('result', 'cluster', 'analysis_path'),), expected_generation=1, retain=('neighbors',))
    assert state.turn('later').status == 'activated'
    capture(app.sessions, 'analysis')
    context, _ = app.sessions.planning_context_for_interaction('analysis', 'future')
    assert tuple(item.binding for item in context.items) == (source.binding,)


def test_unselected_graph_dependency_preserves_metadata_and_integrity_boundaries(processed_source, monkeypatch):
    source = processed_source
    app = ResearchAgentApplication(source.app.workspace_root,
        planner=DownstreamPlanner(source.binding, umap=True))
    state = app.sessions.run('analysis', 'later', AgentRequest('lineage', 'Continue.', {}),
        (OutputSelection('result', 'umap', 'analysis_path'),), expected_generation=1)
    capture(app.sessions, 'analysis')
    before = app.sessions.load('analysis').to_dict()
    with monkeypatch.context() as patch:
        patch.setattr(VerificationContext, 'files', lambda *a, **kw: pytest.fail('Metadata read source bytes'))
        context, locators = app.sessions.planning_context_for_interaction('analysis', 'future', integrity=False)
        assert context.items[0].binding == source.binding and locators == {'ctx.0': ()}
    path = Path(source.run.steps[0].result['analysis_path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises((AuthorityError, ValueError, OSError)):
        app.sessions.planning_context_for_interaction('analysis', 'future')
    assert app.sessions.load('analysis').to_dict() == before


def test_changed_resolved_prior_identity_is_rejected_without_science(processed_source):
    source = processed_source
    app = ResearchAgentApplication(source.app.workspace_root, planner=DownstreamPlanner(source.binding))
    state = app.sessions.run('analysis', 'later', AgentRequest('resolved', 'Continue.', {}),
        (OutputSelection('result', 'cluster', 'analysis_path'),), expected_generation=1, retain=('neighbors',))
    run = app.run_store.load('resolved:run')
    step = run.steps[0]
    changed = replace(step, resolved_arguments={**step.resolved_arguments, 'analysis_path': '/different/source.h5ad'})
    mutated = replace(run, steps=(changed,))
    locator = replace(state.revisions[-1].outputs[0], accepted_step_sha256=accepted_step_digest(changed))
    store = SimpleNamespace(load=lambda run_id: mutated if run_id == run.run_id else app.run_store.load(run_id))
    with pytest.raises(AuthorityError, match='resolved identity'):
        validate_active_outputs((locator, state.revisions[-1].outputs[-1]), store, app.registry)
