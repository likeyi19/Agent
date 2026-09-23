"""Accepted metadata fixtures exercise access, not biological qualification.

Use real result shapes, FileRunStore, session completion and activation. Fixture
acceptance is scripted; real owner-produced evidence is tested separately in
authority_dag. No scientific result is established by these fixture builders.
"""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from agent.application import (ResearchAgentApplication, OutputSelection, ApplicationResult,
                               ApplicationStatus, ArtifactReference)
from agent.application.session_state import digest, canonical
from agent.application import dialogue_evidence as de
from agent.orchestration import ToolRegistry
from agent.report import evidence as ev
from agent.schemas import (AgentRequest, AgentPlan, PlanStep, StepExecutionResult,
                          StepStatus, VerificationResult, VerificationCheck)
from agent.schemas.run_state import (PersistedRunState, RunLifecycleStatus, fingerprint_plan,
    RecoveryPolicySnapshot, ToolRecoveryPolicySnapshot, fingerprint_recovery_policy)


def passed(kind, identity):
    return VerificationResult(True, kind, identity, (VerificationCheck('fixture', True, 'Fixture acceptance.'),))


def accepted(app, tool, *, name='initial', overrides=None, extra_facts=None, copies=1, activate=True, arguments=None):
    state = app.sessions.load('session')
    request = AgentRequest(name, 'Synthetic accepted metadata fixture.', {})
    spec = app.registry.get(tool)
    # Populate the real result shape; explicit probe values below carry the
    # assertions. These are metadata fixtures, not scientific-owner executions.
    default = {str: 'fixture', int: 2, float: 0.5, bool: True, dict: {}, list: []}
    result = {k: default[t[0]] for k, t in spec.result_contract.required_fields.items()}
    result.update(overrides or {})
    output_key = 'manifest_path' if 'manifest_path' in result else next(iter(result))
    selections = tuple(OutputSelection(f'output{i}', f'step{i}', output_key) for i in range(copies))
    app.sessions.start_turn('session', name, request, selections, expected_generation=state.generation)
    app.sessions.link_run('session', name)
    effective, paths = app._prepare_request(request)
    plan = AgentPlan(name + ':plan', name, 'fixture', tuple(PlanStep(f'step{i}', tool, arguments or {}) for i in range(copies)))
    steps = tuple(StepExecutionResult(s.step_id, tool, StepStatus.SUCCEEDED, attempt_count=1,
                    result=result, verification=passed('step', s.step_id), resolved_arguments=arguments or {},
                    started_at='2026-09-23T00:00:00+00:00',
                    finished_at='2026-09-23T00:00:00+00:00', duration_seconds=0.0) for s in plan.steps)
    policies = (ToolRecoveryPolicySnapshot(tool, spec.recovery_policy_version),)
    policy = RecoveryPolicySnapshot('fixture', 1, policies, fingerprint_recovery_policy('fixture', 1, policies))
    stored = PersistedRunState(3, 0, name + ':run', effective, RunLifecycleStatus.SUCCEEDED,
        '2026-09-23T00:00:00+00:00', '2026-09-23T00:00:00+00:00', plan=plan,
        plan_fingerprint=fingerprint_plan(plan), recovery_policy_snapshot=policy,
        preflight_verification=passed('plan', plan.plan_id), steps=steps,
        run_verification=passed('run', plan.plan_id))
    app.run_store.create(stored)
    run = stored.to_run_result()
    fields = ev._TOOL_PROJECTIONS[tool].fact_fields
    payload = dict(schema_version=1, artifact_type=ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, status='success',
        run=dict(run_id=run.run_id, request_id=name, plan_id=plan.plan_id, planner_name=plan.planner_name,
            source_run_status='SUCCEEDED', plan_sha256=digest(plan.to_dict()), source_run_result_sha256=digest(run.to_dict())),
        workflow=dict(ordered_steps=[dict(step_id=s.step_id, tool_name=tool, depends_on=[]) for s in steps]),
        steps=[dict(step_id=s.step_id, tool_name=tool, attempt_count=1,
                    facts={k: result[k] for k in fields} | (extra_facts or {}),
                    verification=dict(passed=True, freshly_verified=True, check_names=['fixture']),
                    recovery_identity=spec.recovery_policy_version) for s in steps],
        artifacts=[], provenance=dict(trace_sha256=digest([]), evidence_projection_version=1,
            registry_tool_identities=[dict(tool_name=tool, result_contract=spec.result_contract.name,
                recovery_identity=spec.recovery_policy_version)] * copies))
    evidence = paths.evidence / ev.ANALYSIS_EVIDENCE_FILENAME
    evidence.write_bytes(canonical(payload))
    report = paths.report / 'fixture.json'
    report.write_text('{}')
    completed = ApplicationResult(name, run.run_id, ApplicationStatus.SUCCEEDED, run.status,
        str(paths.root), run, ArtifactReference(ev.ANALYSIS_EVIDENCE_ARTIFACT_TYPE, str(evidence), digest(payload)),
        report=ArtifactReference('fixture', str(report), digest({})))
    app.sessions._record_result('session', name, completed)
    if activate:
        state = app.sessions.recover('session', name)
        revision = state.turn(name).revision_id
    else:
        revision = None
    return revision, evidence, payload


@pytest.fixture
def app(tmp_path):
    app = ResearchAgentApplication(tmp_path / 'workspace')
    app.sessions.create('session')
    return app


def forbid_work(app, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Evidence access entered science/presentation/execution')
    from agent.application import service
    from agent.orchestration import verifier
    from agent.orchestration.runtime import AgentRuntime
    from agent.orchestration.executor import PlanExecutor
    from agent.tools.data.authority_context import VerificationContext
    from agent.report import analysis_report
    for module, names in ((ev, ('verify_step', 'verify_run', 'build_analysis_evidence', 'verify_analysis_evidence')),
                          (verifier, ('verify_step', 'verify_run')),
                          (service, ('build_analysis_evidence', 'verify_analysis_evidence', 'build_analysis_report')),
                          (analysis_report, ('build_analysis_report', 'verify_analysis_report')),
                          (AgentRuntime, ('run', 'resume')), (PlanExecutor, ('execute',)),
                          (VerificationContext, ('verify',)),
                          (ResearchAgentApplication, ('run', 'resume', '_complete'))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    # An accidental registry call is forbidden too, even without the executor.
    registry = ToolRegistry(tuple(replace(app.registry.get(n), function=forbidden) for n in app.registry.names()))
    return ResearchAgentApplication(app.workspace_root, registry=registry)


def files(app):
    return {str(p.relative_to(app.workspace_root)): p.read_bytes()
            for p in app.workspace_root.rglob('*') if p.is_file()}


PROBES = [
    ('compute_scATAC_qc', {'n_observed_barcodes': 13, 'tss_undefined': 3}),
    ('select_scATAC_cells', {'n_selected': 8, 'n_rejected': 5, 'cell_call_state': 'not_assessed'}),
    ('annotate_scATAC_cell_types', {'group_summary': [dict(group='cluster/3', n_cells=8,
        primary_annotation='CD8 T', status='assigned')], 'groups_omitted': 7, 'validation_state': 'not_assessed'}),
    ('build_scATAC_cell_by_features', {'n_cells': 8, 'n_features': 15, 'nnz': 32,
        'matrix_semantics': 'canonical-fragment-record-overlap-counts.v1'}),
    ('adapt_epizoo_species', {'strategy': 'de_novo', 'purpose': 'qualification', 'completed_step': 10,
        'optimizer_steps': 10, 'amp_skipped_steps': 0}),
    # Legacy processed-result evidence has no reusable schema-2 authority.
    ('inspect_scATAC', {'n_cells': 8, 'n_features': 15, 'obs_names_sample': ['c1', 'c2']}),
]


@pytest.mark.parametrize('tool,values', PROBES)
def test_cross_capability_exact_immutable_read(app, monkeypatch, tool, values):
    rid, path, payload = accepted(app, tool, overrides=values)
    guarded = forbid_work(app, monkeypatch)
    before = files(app)
    view = guarded.sessions.evidence('session', rid, 'output0')
    assert view.status == 'available', view
    assert view.is_active and view.generation == 1
    assert view.source.output_locator['run_id'] == 'initial:run'
    assert view.source.output_locator['step_id'] == 'step0'
    assert view.source.evidence_sha256 == digest(payload)
    assert view.source.authority_sha256 is None
    observed = {f.field: f.to_dict()['value'] for f in view.facts}
    assert {k: observed[k] for k in values} == values
    assert all(f.source_pointer.startswith('/steps/0/facts/') for f in view.facts)
    with pytest.raises(TypeError):
        view.source.output_locator['run_id'] = 'other'
    if tool == 'annotate_scATAC_cell_types':
        assert view.coverage[0].value == 7
        with pytest.raises(TypeError):
            next(f for f in view.facts if f.field == 'group_summary').value[0]['status'] = 'new'
    assert files(app) == before


def test_field_selection_null_missing_and_omissions(app, monkeypatch):
    rid, _, _ = accepted(app, 'annotate_scATAC_cell_types', overrides=dict(groups_omitted=8,
        group_summary=[dict(group='3', primary_annotation=None, status='ambiguous')]))
    guard = forbid_work(app, monkeypatch)
    view = guard.sessions.evidence('session', rid, 'output0', fields=['group_summary', 'marker_ranking'])
    assert view.status == 'available' and view.coverage[0].value == 8
    assert view.facts[0].value[0]['primary_annotation'] is None
    assert view.facts[1].status == 'unavailable'
    assert view.facts[1].reason == 'not_in_accepted_summary'
    assert 'groups_omitted' in view.unselected_fields
    monkeypatch.setattr(de, 'MAX_FACT_BYTES', 10)
    omitted = guard.sessions.evidence('session', rid, 'output0', fields=['group_summary'])
    assert omitted.facts[0].status == 'omitted' and omitted.facts[0].value is None
    assert omitted.coverage[0].value == 8
    monkeypatch.setattr(de, 'MAX_VIEW_BYTES', 10)
    assert guard.sessions.evidence('session', rid, 'output0').reason == 'evidence_view_size_limit'


def test_repeated_tool_exact_step_and_historical_visibility(app, monkeypatch):
    r1, _, _ = accepted(app, 'inspect_scATAC', copies=2, overrides={'n_cells': 3})
    r2, _, _ = accepted(app, 'select_scATAC_cells', name='next', overrides={'n_selected': 2})
    guard = forbid_work(app, monkeypatch)
    before = files(app)
    old = guard.sessions.evidence('session', r1, 'output1')
    assert old.status == 'available' and not old.is_active
    assert old.source.step_pointer == '/steps/1'
    assert old.source.output_locator['step_id'] == 'step1'
    # This output exists historically, but is not an output of the current revision.
    missing = guard.sessions.evidence('session', r2, 'output1')
    assert missing.status == 'unavailable' and missing.source is None and not missing.facts
    assert guard.sessions.evidence('session', 'unknown', 'output0').status == 'unavailable'
    assert files(app) == before


@pytest.mark.parametrize('mutation', ['missing', 'digest', 'duplicate_json', 'wrong_run', 'wrong_step',
    'wrong_fact', 'wrong_contract', 'wrong_schema', 'wrong_lineage', 'ambiguous_step', 'symlink', 'oversize'])
def test_invalid_evidence_fails_closed(app, monkeypatch, mutation):
    rid, path, payload = accepted(app, 'inspect_scATAC')
    if mutation == 'missing':
        path.unlink()
        path.parent.rmdir()
    elif mutation == 'digest':
        path.write_bytes(path.read_bytes() + b' ')
    elif mutation == 'symlink':
        target = path.with_name('other.json')
        path.rename(target)
        path.symlink_to(target)
    elif mutation == 'oversize':
        monkeypatch.setattr(de, 'MAX_EVIDENCE_BYTES', 10)
    else:
        if mutation == 'wrong_run': payload['run']['run_id'] = 'other:run'
        if mutation == 'wrong_step': payload['steps'][0]['step_id'] = 'other'
        if mutation == 'wrong_fact': payload['steps'][0]['facts']['n_cells'] = 99
        if mutation == 'wrong_contract': payload['provenance']['registry_tool_identities'][0]['result_contract'] = 'future'
        if mutation == 'wrong_schema': payload['schema_version'] = 99
        if mutation == 'wrong_lineage': payload['workflow']['ordered_steps'][0]['depends_on'] = ['other']
        if mutation == 'ambiguous_step': payload['steps'] *= 2
        raw = canonical(payload) if mutation != 'duplicate_json' else b'{"status":1,"status":2}'
        path.write_bytes(raw)
        # Simulate incompatible/corrupt completion content even when its bytes
        # are pinned. The independent run/step identities must still agree.
        import hashlib
        original = app.sessions.load('session')
        changed = replace(original, turns=tuple(replace(t, completion_files=tuple(
            replace(f, sha256=hashlib.sha256(raw).hexdigest()) if f.path == str(path) else f
            for f in t.completion_files)) for t in original.turns))
        app.sessions._store._write(changed)
    guard = forbid_work(app, monkeypatch)
    before = files(app)
    view = guard.sessions.evidence('session', rid, 'output0')
    assert view.status in {'unavailable', 'unsupported'} and not view.facts and view.source is None
    assert files(app) == before
    if mutation == 'missing': assert not path.parent.exists()


def test_unknown_projection_and_contract_drift(app, monkeypatch):
    rid, _, _ = accepted(app, 'inspect_scATAC')
    spec = app.registry.get('inspect_scATAC')
    changed = replace(spec, result_contract=replace(spec.result_contract,
        required_fields=dict(spec.result_contract.required_fields, new_fact=(str,))))
    registry = ToolRegistry(tuple(changed if n == spec.name else app.registry.get(n) for n in app.registry.names()))
    other = ResearchAgentApplication(app.workspace_root, registry=registry)
    assert other.sessions.evidence('session', rid, 'output0').status == 'unsupported'
    monkeypatch.delitem(ev._TOOL_PROJECTIONS, 'inspect_scATAC')
    assert app.sessions.evidence('session', rid, 'output0').reason == 'unsupported_tool'


def test_derived_facts_and_no_arbitrary_result_fanout(app, monkeypatch):
    rid, _, _ = accepted(app, 'compute_scATAC_qc',
        overrides={'unreviewed_biology': 'must not be exposed'},
        extra_facts={'tss_method': {'profile': 'accepted-method'}, 'qc_summary': {'tss_defined': 2}})
    guarded = forbid_work(app, monkeypatch)
    view = guarded.sessions.evidence('session', rid, 'output0',
        fields=['tss_method', 'unreviewed_biology'])
    assert view.facts[0].value['profile'] == 'accepted-method'
    assert view.facts[1].status == 'unavailable'


@pytest.mark.parametrize('mutation', ['run', 'locator', 'completion', 'session'])
def test_invalid_source_anchors(app, monkeypatch, mutation):
    rid, path, _ = accepted(app, 'inspect_scATAC')
    guarded = forbid_work(app, monkeypatch)
    state = app.sessions.load('session')
    if mutation == 'run':
        original = app.run_store.load
        from agent.orchestration.run_store import FileRunStore
        def changed(store, run_id):
            run = original(run_id)
            return replace(run, steps=(replace(run.steps[0], result=dict(run.steps[0].result, n_cells=99)),))
        monkeypatch.setattr(FileRunStore, 'load', changed)
    elif mutation == 'session':
        app.sessions._store._path('session', '.json').write_text('{}')
    else:
        record = state.to_dict()
        if mutation == 'locator':
            record['revisions'][0]['outputs'][0]['accepted_step_sha256'] = '0' * 64
        else:
            record['turns'][0]['completion_files'] = [f for f in record['turns'][0]['completion_files']
                                                    if f['path'] != str(path)]
        app.sessions._store._path('session', '.json').write_bytes(canonical(dict(
            format='agent.analysis-session.v1', record=record, sha256=digest(record))))
    view = guarded.sessions.evidence('session', rid, 'output0')
    assert view.status == 'unavailable' and not view.facts and view.source is None


def test_fresh_process_recovery_then_read(app):
    _, _, _ = accepted(app, 'adapt_epizoo_species', activate=False)
    code = '''
import json, sys
from agent.application import ResearchAgentApplication
from agent.orchestration.runtime import AgentRuntime
from agent.tools.data.authority_context import VerificationContext
def forbidden(*a, **kw): raise AssertionError('scientific work')
AgentRuntime.run = AgentRuntime.resume = VerificationContext.verify = forbidden
app = ResearchAgentApplication(sys.argv[1])
state = app.sessions.recover('session', 'initial')
view = app.sessions.evidence('session', state.active_revision_id, 'output0')
assert view.status == 'available', view
print(json.dumps(view.to_dict()))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code, str(app.workspace_root)],
                            capture_output=True, text=True, check=True)
    view = json.loads(result.stdout)
    assert view['source']['tool_name'] == 'adapt_epizoo_species'
    assert view['evidence_scope'] == 'accepted_persisted_summary'


@pytest.mark.parametrize('fields', ['n_cells', ['n_cells', 'n_cells'], [1], ['']])
def test_invalid_field_request(app, fields):
    with pytest.raises(ValueError):
        app.sessions.evidence('session', 'revision', 'output', fields=fields)
