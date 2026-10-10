"""Exact accepted metadata projection; these fixtures qualify no new science."""
from dataclasses import replace
import hashlib

import pytest

from agent.application import ApplicationResult, ApplicationStatus, ArtifactReference, OutputSelection
from agent.application.session_state import canonical, digest
from agent.report import evidence as ev
from agent.schemas import AgentPlan, AgentRequest, PlanStep, StepExecutionResult, StepOutputRef, StepStatus
from agent.schemas.run_state import (PersistedRunState, RecoveryPolicySnapshot, RunLifecycleStatus,
    ToolRecoveryPolicySnapshot, fingerprint_plan, fingerprint_recovery_policy)
from test_dialogue_evidence import app, accepted, files, forbid_work, passed


def analysis(app, *, resolution=.7, binding='exact', mutation=None):
    """Accept two distinct clustering steps and retain only the final UMAP."""
    request = AgentRequest('analysis', 'Accepted processed-analysis metadata fixture.', {})
    app.sessions.start_turn('session', 'analysis', request,
        (OutputSelection('result', 'umap', 'analysis_path'),), expected_generation=0)
    app.sessions.link_run('session', 'analysis')
    effective, paths = app._prepare_request(request)
    definitions = []
    for name, value in (('unrelated', .2), ('requested', resolution)):
        arguments = {'analysis_path': '/accepted/neighbors.h5ad', 'output_dir': '/accepted'}
        if value is not None:
            arguments['resolution'] = value
        definitions.append((PlanStep(name, 'cluster_cells', arguments), dict(
            status='success', input_analysis_path=arguments['analysis_path'],
            analysis_path=f'/accepted/{name}.h5ad', n_cells=8, n_clusters=3,
            cluster_key='leiden', algorithm='leiden', resolution=1.0 if value is None else float(value),
            random_seed=0, cell_order_preserved=True, backend='Scanpy', software_versions={})))
    source = (StepOutputRef('requested', 'analysis_path') if binding == 'exact'
              else '/accepted/requested.h5ad')
    definitions.append((PlanStep('umap', 'compute_cell_umap',
        {'analysis_path': source, 'output_dir': '/accepted'}, depends_on=('requested',)), dict(
            status='success', input_analysis_path='/accepted/requested.h5ad',
            analysis_path='/accepted/umap.h5ad', n_cells=8, n_components=2,
            umap_key='X_umap', coordinate_dtype='float32', finite=True, min_dist=.5,
            spread=1.0, random_seed=0, cell_order_preserved=True, backend='Scanpy', software_versions={})))
    plan = AgentPlan('analysis:plan', 'analysis', 'scripted', tuple(step for step, _ in definitions))
    steps = []
    for step, result in definitions:
        arguments = dict(step.arguments)
        if step.step_id == 'umap':
            arguments['analysis_path'] = '/accepted/requested.h5ad'
        if mutation == 'binding' and step.step_id == 'umap':
            arguments['analysis_path'] = '/accepted/unrelated.h5ad'
        if mutation == 'argument' and step.step_id == 'requested':
            arguments['resolution'] = .9
        steps.append(StepExecutionResult(step.step_id, step.tool_name, StepStatus.SUCCEEDED,
            attempt_count=1, result=result, resolved_arguments=arguments, verification=passed('step', step.step_id),
            started_at='2026-09-23T00:00:00+00:00', finished_at='2026-09-23T00:00:00+00:00',
            duration_seconds=0.0))
    policies = tuple(ToolRecoveryPolicySnapshot(tool, app.registry.get(tool).recovery_policy_version)
        for tool in ('cluster_cells', 'compute_cell_umap'))
    policy = RecoveryPolicySnapshot('fixture', 1, policies, fingerprint_recovery_policy('fixture', 1, policies))
    stored = PersistedRunState(3, 0, 'analysis:run', effective, RunLifecycleStatus.SUCCEEDED,
        '2026-09-23T00:00:00+00:00', '2026-09-23T00:00:00+00:00', plan=plan,
        plan_fingerprint=fingerprint_plan(plan), recovery_policy_snapshot=policy,
        preflight_verification=passed('plan', plan.plan_id), steps=tuple(steps),
        run_verification=passed('run', plan.plan_id))
    app.run_store.create(stored)
    run = stored.to_run_result()
    payload = dict(schema_version=1, artifact_type=ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, status='success',
        run=dict(run_id=run.run_id, request_id=run.request_id, plan_id=plan.plan_id,
            planner_name=plan.planner_name, source_run_status='SUCCEEDED', plan_sha256=digest(plan.to_dict()),
            source_run_result_sha256=digest(run.to_dict())),
        workflow=dict(ordered_steps=[dict(step_id=s.step_id, tool_name=s.tool_name,
            depends_on=list(s.depends_on)) for s in plan.steps]),
        steps=[dict(step_id=s.step_id, tool_name=s.tool_name, attempt_count=1,
            facts={key: s.to_dict()['result'][key] for key in ev._TOOL_PROJECTIONS[s.tool_name].fact_fields},
            verification=dict(passed=True, freshly_verified=True, check_names=['fixture']),
            recovery_identity=app.registry.get(s.tool_name).recovery_policy_version) for s in steps],
        artifacts=[], provenance=dict(trace_sha256=digest([]), evidence_projection_version=1,
            registry_tool_identities=[dict(tool_name=tool, result_contract=app.registry.get(tool).result_contract.name,
                recovery_identity=app.registry.get(tool).recovery_policy_version)
                for tool in ('cluster_cells', 'compute_cell_umap')]))
    path = paths.evidence / ev.ANALYSIS_EVIDENCE_FILENAME
    path.write_bytes(canonical(payload))
    report = paths.report / 'historical-report.txt'
    report.write_bytes(b'Original accepted report bytes.\n')
    completed = ApplicationResult('analysis', run.run_id, ApplicationStatus.SUCCEEDED, run.status,
        str(paths.root), run, ArtifactReference(ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, str(path), digest(payload)),
        report=ArtifactReference('fixture', str(report), hashlib.sha256(report.read_bytes()).hexdigest()))
    app.sessions._record_result('session', 'analysis', completed)
    state = app.sessions.recover('session', 'analysis')
    return state.turn('analysis').revision_id, path, payload, run


def repin(app, path, payload):
    """A pinned malformed summary still must match its accepted scientific source."""
    raw = canonical(payload)
    path.write_bytes(raw)
    state = app.sessions.load('session')
    app.sessions._store._write(replace(state, turns=tuple(replace(turn, completion_files=tuple(
        replace(item, sha256=hashlib.sha256(raw).hexdigest()) if item.path == str(path) else item
        for item in turn.completion_files)) for turn in state.turns)))


@pytest.mark.parametrize('resolution,origin', [(.7, 'explicit_execution_argument'),
    (1.0, 'explicit_execution_argument'), (None, 'existing_owner_default')])
def test_selected_cluster_resolution_origin_reads_exact_accepted_metadata(app, monkeypatch, resolution, origin):
    arguments = {} if resolution is None else {'resolution': resolution}
    rid, path, _ = accepted(app, 'cluster_cells', arguments=arguments,
        overrides={'resolution': 1.0 if resolution is None else resolution, 'random_seed': 4})
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session', rid, 'output0',
        fields=('clustering_parameters', 'clustering_resolution_origin'))
    assert view.status == 'available', view
    parameters, provenance = view.facts
    assert dict(parameters.value) == {'resolution': 1.0 if resolution is None else resolution, 'random_seed': 4}
    assert parameters.source_pointer == '/steps/0/facts'
    assert provenance.value == origin
    assert provenance.artifact_name == 'accepted_run_result'
    assert provenance.artifact_sha256 == view.source.source_run_result_sha256
    assert provenance.source_pointer == '/steps/0/resolved_arguments'
    assert files(app) == before
    # Presentation facts never materialize themselves into the historical artifact.
    assert b'clustering_parameters' not in path.read_bytes()
    assert b'clustering_resolution_origin' not in path.read_bytes()


def test_historical_effective_value_is_never_replaced_by_current_owner_default(app, monkeypatch):
    rid, _, _ = accepted(app, 'cluster_cells', overrides={'resolution': 1.3, 'random_seed': 7})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    facts = {fact.field: fact for fact in view.facts}
    assert facts['clustering_parameters'].value['resolution'] == 1.3
    assert facts['clustering_parameters'].value['random_seed'] == 7
    assert facts['clustering_resolution_origin'].value == 'existing_owner_default'


def test_corrupt_boolean_explicit_argument_is_not_equated_with_recorded_numeric_resolution(app, monkeypatch):
    rid, _, _ = accepted(app, 'cluster_cells', arguments={'resolution': True},
        overrides={'resolution': 1.0, 'random_seed': 0})
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and view.source is None and not view.facts


@pytest.mark.parametrize('resolution,expected,origin', [(.7, .7, 'explicit_execution_argument'),
    (None, 1.0, 'existing_owner_default')])
def test_final_umap_uses_only_exact_cluster_source_and_preserves_history(app, monkeypatch, resolution, expected, origin):
    rid, path, payload, run = analysis(app, resolution=resolution)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session', rid, 'result')
    assert view.status == 'available', view
    facts = {fact.field: fact for fact in view.facts}
    assert dict(facts['clustering_parameters'].value) == {'resolution': expected, 'random_seed': 0}
    assert facts['clustering_parameters'].source_pointer == '/steps/1/facts'
    assert facts['clustering_resolution_origin'].value == origin
    assert facts['clustering_resolution_origin'].source_pointer == '/steps/1/resolved_arguments'
    assert view.source.output_locator['step_id'] == 'umap'
    assert view.source.source_run_result_sha256 == digest(run.to_dict())
    assert files(app) == before
    assert path.read_bytes() == canonical(payload)


def test_matching_path_does_not_infer_clustering_source(app, monkeypatch):
    rid, _, _, _ = analysis(app, binding='path')
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'result')
    assert view.status == 'available', view
    facts = {fact.field: fact for fact in view.facts}
    for field in ('clustering_parameters', 'clustering_resolution_origin'):
        assert facts[field].status == 'unavailable'
        assert facts[field].reason == 'exact_clustering_source_not_recorded'
        assert facts[field].value is None


@pytest.mark.parametrize('mutation', ['binding', 'argument'])
def test_inconsistent_accepted_clustering_source_is_not_projected(app, monkeypatch, mutation):
    rid, _, _, _ = analysis(app, mutation=mutation)
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'result')
    assert view.status == 'unavailable' and view.source is None and not view.facts


@pytest.mark.parametrize('mutation', ['missing_resolution', 'wrong_resolution', 'missing_summary',
    'wrong_contract', 'verification'])
def test_missing_or_tampered_pinned_upstream_summary_does_not_supply_parameters(app, monkeypatch, mutation):
    rid, path, payload, _ = analysis(app)
    if mutation == 'missing_resolution':
        del payload['steps'][1]['facts']['resolution']
    elif mutation == 'wrong_resolution':
        payload['steps'][1]['facts']['resolution'] = .9
    elif mutation == 'missing_summary':
        del payload['steps'][1]
    elif mutation == 'verification':
        payload['steps'][1]['verification']['passed'] = False
    else:
        payload['provenance']['registry_tool_identities'][0]['result_contract'] = 'unsupported'
    repin(app, path, payload)
    view = forbid_work(app, monkeypatch).sessions.evidence('session', rid, 'result')
    assert view.status in ('unavailable', 'unsupported') and view.source is None and not view.facts


def test_projected_parameter_fields_obey_existing_fact_bounds(app, monkeypatch):
    rid, _, _, _ = analysis(app)
    guarded = forbid_work(app, monkeypatch)
    from agent.application import dialogue_evidence
    monkeypatch.setattr(dialogue_evidence, 'MAX_FACT_BYTES', 8)
    view = guarded.sessions.evidence('session', rid, 'result',
        fields=('clustering_parameters', 'clustering_resolution_origin'))
    assert view.status == 'available'
    assert all(fact.status == 'omitted' and fact.value is None and fact.reason == 'field_size_limit'
               for fact in view.facts)


def test_final_umap_parameter_followup_uses_accepted_values_without_science(app, monkeypatch):
    from test_scientific_dialogue import Model, question
    rid, path, _, run = analysis(app)
    guarded = forbid_work(app, monkeypatch)
    pinned = path.read_bytes()
    fields = ('clustering_parameters.resolution', 'clustering_resolution_origin')
    model = Model(question(), fields)
    response = guarded.sessions.respond('session', 'parameters', 'What resolution did you use for clustering?',
        interpreter=model)
    assert response.status == 'answered', response
    claims = {claim.field: claim for claim in response.scientific.claims}
    assert claims[fields[0]].value == .7
    assert claims[fields[0]].source['pointer'] == '/steps/1/facts/resolution'
    assert claims[fields[0]].source['revision_id'] == rid
    assert claims[fields[1]].value == 'explicit_execution_argument'
    assert claims[fields[1]].source['pointer'] == '/steps/1/resolved_arguments'
    assert claims[fields[1]].source['source_run_result_sha256'] == digest(run.to_dict())
    assert claims[fields[1]].source['artifact_name'] == 'accepted_run_result'
    assert path.read_bytes() == pinned
