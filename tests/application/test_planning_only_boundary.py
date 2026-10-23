"""Operator-only planning stop through real upload, interpretation and compilation.

All model responses and matrix measurements here are simulated. No registered
scientific callable, scientific execution, durable Run or result publication is
permitted. The operator intercept is a verification harness, not a product mode.
"""
from dataclasses import replace
import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent.application import InteractiveAgentApplication, ResearchAgentApplication
from agent.application import dialogue_execution
from agent.application.sessions import AnalysisSessions
from agent.application.turns import _AdmittedPlanner
from agent.application.uploads import H5ADUploadAdmission
from agent.orchestration import (AgentRuntime, PlanningModelProfile, PlanningRecoveryPolicy,
                                 ToolRegistry, build_default_tool_registry)
from agent.orchestration.executor import PlanExecutor
from agent.orchestration.run_store import FileRunStore
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import RunMode, RunStatus, StepOutputRef


REQUEST = ('Use PCA with 3 components, construct neighbors, identify Leiden clusters '
           'and show a UMAP of the uploaded matrix.')
POLICY = PlanningRecoveryPolicy(max_transport_retries=0, max_repairs=0,
    max_profile_failovers=0, max_primary_local_recovery_actions=0,
    max_total_provider_calls=1, max_retry_delay_seconds=0)


class PlanningBoundaryReached(BaseException):
    """Harness-only stop which application exception recovery cannot swallow."""

    def __init__(self, result, request, planner):
        self.result, self.request, self.planner = result, request, planner


def install_operator_stop(monkeypatch, *, reserve=False):
    """Reuse the production planner and validation callback without a RunStore."""
    def stopped(application, request):
        admitted = application.runtime.planner
        assert isinstance(admitted, _AdmittedPlanner)
        assert admitted.accept.__qualname__ == 'execute.<locals>.accept'
        effective, _ = application._prepare_request(replace(request, mode=RunMode.PLAN_ONLY))
        def accept(candidate, plan):
            if reserve:
                # Deliberately exercise the normal default, not an explicit True.
                return admitted.accept(candidate, plan)
            return admitted.accept(candidate, plan, reserve=False)
        runtime = AgentRuntime(planner=_AdmittedPlanner(admitted.delegate, accept),
            registry=application.registry, executor=application.runtime.executor)
        assert runtime.run_store is None
        result = runtime.run(effective)
        raise PlanningBoundaryReached(result, effective, admitted.delegate)
    monkeypatch.setattr(ResearchAgentApplication, 'run', stopped)


def forbidden(*args, **kwargs):
    pytest.fail('Planning boundary reached scientific execution or publication')


@pytest.fixture(autouse=True)
def prevent_execution_and_publication(monkeypatch):
    monkeypatch.setattr(PlanExecutor, 'execute', forbidden)
    monkeypatch.setattr(AgentRuntime, '_run_execute', forbidden)
    monkeypatch.setattr(AgentRuntime, '_run_durable_execute', forbidden)
    monkeypatch.setattr(FileRunStore, 'create', forbidden)
    monkeypatch.setattr(AnalysisSessions, '_record_result', forbidden)
    monkeypatch.setattr(dialogue_execution, 'SessionTurn', forbidden)


def wire_steps():
    def step(name, tool, target=None, source=None):
        return dict(step_id=name, tool=tool, control_dependencies=[],
                    sources=[] if target is None else [dict(kind='step', target=target, step=source)])
    return [step('pca', 'pca_embed_cells'),
            step('graph', 'build_cell_neighbors', 'embedding', 'pca'),
            step('clusters', 'cluster_cells', 'analysis', 'graph'),
            step('umap', 'compute_cell_umap', 'analysis', 'clusters')]


class ScriptedPlanningModel:
    def __init__(self, profile, *, decision=None, steps=None, after_plan=None, after_selection=None):
        self.model_id, self.calls = profile.model_id, []
        self.decision, self.steps = decision, steps
        self.after_plan, self.after_selection = after_plan, after_selection

    def complete(self, *, prompt, response_schema):
        payload = json.loads(prompt)
        self.calls.append((payload, response_schema))
        if 'turn_schema_version' in payload:
            decision = self.decision or dict(kind='execute_plan', target='pca_embed_cells',
                roles=[dict(tool='pca_embed_cells', port='dataset', source='source.0')],
                arguments=[dict(tool='pca_embed_cells', argument='n_components', literal='3',
                                evidence='PCA with 3 components')])
            return json.dumps(dict(turn_schema_version=1, decision=decision))
        if 'semantic_prompt_version' in payload:
            if self.after_plan is not None:
                self.after_plan()
            return json.dumps(dict(schema_version=4, decision=dict(kind='plan',
                steps=wire_steps() if self.steps is None else self.steps)))
        if 'output_selection_schema_version' in payload:
            if self.after_selection is not None:
                self.after_selection()
            return json.dumps(dict(outputs=[dict(name='map', step_id='umap', output_key='analysis_path')]))
        pytest.fail('Unexpected extra model phase during planning qualification')


def setup_upload(tmp_path, **model_options):
    base = build_default_tool_registry()
    registry = ToolRegistry(tuple(replace(base.get(name), function=forbidden) for name in base.names()))
    assert registry.names() == base.names()
    models = []
    def factory(profile):
        model = ScriptedPlanningModel(profile, **model_options)
        models.append(model)
        return model
    workspace = tmp_path / 'workspace'
    root = workspace / 'uploads'
    root.mkdir(parents=True)
    facade = InteractiveAgentApplication(workspace,
        model_profiles=(PlanningModelProfile('local', 'scripted', 'test/planning'),),
        default_profile_id='local', registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}),
        planning_recovery_policy=POLICY, approved_source_roots=(root,))
    facade.create_session('session')
    uploads = H5ADUploadAdmission(facade.resources, root)
    source = tmp_path / 'simulated.h5ad'
    ad.AnnData(sparse.csr_matrix(np.arange(24*8).reshape(24, 8) % 7),
        obs=pd.DataFrame(index=[f'cell-{i}' for i in range(24)]),
        var=pd.DataFrame(index=[f'feature-{i}' for i in range(8)])).write_h5ad(source)
    original = source.read_bytes()
    async def chunks():
        yield original[:100]
        yield original[100:]
    record = asyncio.run(uploads.receive('matrix', 'simulated.h5ad', chunks(),
        expected_size=len(original), session_id='session'))
    return SimpleNamespace(facade=facade, uploads=uploads, record=record, models=models,
        original=original, original_sha256=hashlib.sha256(original).hexdigest())


def submit(case):
    return case.facade.submit_turn('session', 'request', REQUEST, expected_generation=0,
        attachment_ids=(case.record.resource_id,))


def assert_no_scientific_records(case):
    app = case.facade._application
    state = app.sessions.load('session')
    assert state.turns == state.revisions == ()
    assert state.active_revision_id is None and state.generation == 0
    assert not list(app._workspace.run_state.rglob('*.json'))
    assert not any(path.is_file() for path in app._workspace.runs.rglob('*'))
    return state


def test_uploaded_turn_reaches_same_detailed_planner_and_validated_plan_without_science(tmp_path, monkeypatch):
    case = setup_upload(tmp_path)
    install_operator_stop(monkeypatch)
    with pytest.raises(PlanningBoundaryReached) as caught:
        submit(case)
    stopped = caught.value
    result = stopped.result
    assert result.status is RunStatus.PLANNED and result.planning_only
    assert result.verification.passed and result.steps == ()
    assert result.plan.planner_name.endswith(':wire-v4')
    assert stopped.request.mode is RunMode.PLAN_ONLY and stopped.request.prompt == REQUEST
    assert len(case.models) == 1 and stopped.planner.model is case.models[0]
    phases = [next(key for key in ('turn_schema_version', 'semantic_prompt_version') if key in p)
              for p, _ in case.models[0].calls]
    assert phases == ['turn_schema_version', 'semantic_prompt_version']
    detailed = case.models[0].calls[1][0]
    assert detailed['user_request'] == REQUEST
    steps = {step.step_id: step for step in result.plan.steps}
    assert steps['pca'].arguments['input_path'] == case.record.source_path
    assert steps['pca'].arguments['n_components'] == 3
    assert steps['graph'].arguments['embedding_path'] == StepOutputRef('pca', 'embedding_path')
    assert steps['graph'].arguments['cell_ids_path'] == StepOutputRef('pca', 'cell_ids_path')
    assert steps['clusters'].arguments['analysis_path'] == StepOutputRef('graph', 'analysis_path')
    assert steps['umap'].arguments['analysis_path'] == StepOutputRef('clusters', 'analysis_path')
    assert {step.tool_name for step in steps.values()} == {
        'pca_embed_cells', 'build_cell_neighbors', 'cluster_cells', 'compute_cell_umap'}
    state = assert_no_scientific_records(case)
    assert state.interactions[-1].status == 'admitted'
    assert state.interactions[-1].admitted['inputs']['n_components'] == 3
    assert Path(case.record.source_path).read_bytes() == case.original
    assert case.record.source_sha256 == case.original_sha256
    # The deliberate operator stop uses the existing interruption closure. A
    # refresh or duplicate never invokes the models or resumes science.
    closed = case.facade.recover_turn('session', 'request')
    assert closed.status == 'failed' and closed.response.status == 'unavailable'
    assert 'Scientific execution has not been resumed' in closed.response.text
    assert submit(case) == closed
    assert len(case.models[0].calls) == 2
    assert_no_scientific_records(case)


@pytest.mark.parametrize('invalid', ['unknown_tool', 'missing_dependency'])
def test_invalid_detailed_plan_cannot_reach_planned_status(tmp_path, monkeypatch, invalid):
    steps = wire_steps()
    if invalid == 'unknown_tool':
        steps[-1]['tool'] = 'invented_science'
    else:
        steps[1]['sources'][0]['step'] = 'absent'
    case = setup_upload(tmp_path, steps=steps)
    install_operator_stop(monkeypatch)
    with pytest.raises(PlanningBoundaryReached) as caught:
        submit(case)
    assert caught.value.result.status is RunStatus.FAILED
    assert caught.value.result.errors
    assert len(case.models[0].calls) == 2
    assert_no_scientific_records(case)


def test_plan_only_retains_registered_source_integrity_check(tmp_path, monkeypatch):
    case = setup_upload(tmp_path, after_plan=lambda:
        Path(case.record.source_path).write_bytes(b'changed simulated source'))
    # The source changes during the last (detailed-planning) provider completion.
    install_operator_stop(monkeypatch)
    with pytest.raises(PlanningBoundaryReached) as caught:
        submit(case)
    assert caught.value.result.status is RunStatus.FAILED
    assert caught.value.result.errors[0].code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert len(case.models[0].calls) == 2
    assert_no_scientific_records(case)


def test_default_reservation_checks_source_after_output_selection(tmp_path, monkeypatch):
    case = setup_upload(tmp_path, after_selection=lambda:
        Path(case.record.source_path).write_bytes(b'changed after simulated output selection'))
    install_operator_stop(monkeypatch, reserve=True)
    with pytest.raises(PlanningBoundaryReached) as caught:
        submit(case)
    assert caught.value.result.status is RunStatus.FAILED
    assert caught.value.result.errors[0].code == 'LOCAL_RESOURCE_INTEGRITY_INVALID'
    assert 'output_selection_schema_version' in case.models[0].calls[-1][0]
    assert len(case.models[0].calls) == 3
    assert_no_scientific_records(case)


def test_genuine_model_clarification_never_reaches_detailed_planning(tmp_path, monkeypatch):
    case = setup_upload(tmp_path, decision=dict(kind='clarify', reason='ambiguous_subject'))
    install_operator_stop(monkeypatch)
    view = submit(case)
    assert view.status == 'clarification' and view.response.kind == 'clarify'
    assert view.response.clarification['reason'] == 'ambiguous_subject'
    assert len(case.models[0].calls) == 1
    assert_no_scientific_records(case)


DEFAULT_REQUEST = ('Inspect the uploaded matrix and explore cellular heterogeneity '
    'with PCA, neighbors, Leiden clustering and UMAP using reasonable defaults. '
    'Explain the methods used.')


def shared_source_decision(pca_source='source.0'):
    # Exact response structure authored by the real Interpreter in UAT-02.
    return dict(kind='execute_plan', target='pca_embed_cells', delivery=['explanation'],
        unfulfilled=[], arguments=[], roles=[
            dict(tool='inspect_scATAC', port='dataset', source='source.0'),
            dict(tool='pca_embed_cells', port='dataset', source=pca_source)])


def test_authentic_shared_source_roles_reach_inspection_pca_plan_without_science(tmp_path, monkeypatch):
    steps = wire_steps()
    steps[0]['sources'] = [dict(kind='step', target='dataset', step='inspect')]
    steps.insert(0, dict(step_id='inspect', tool='inspect_scATAC', sources=[], control_dependencies=[]))
    case = setup_upload(tmp_path, decision=shared_source_decision(), steps=steps)
    install_operator_stop(monkeypatch)
    with pytest.raises(PlanningBoundaryReached) as caught:
        case.facade.submit_turn('session', 'request', DEFAULT_REQUEST, expected_generation=0,
            attachment_ids=(case.record.resource_id,))
    stopped = caught.value
    assert stopped.result.status is RunStatus.PLANNED
    assert stopped.result.verification.passed and stopped.result.steps == ()
    assert stopped.request.prompt == DEFAULT_REQUEST
    assert 'n_components' not in stopped.request.inputs
    assert len(case.models[0].calls) == 2
    assert case.models[0].calls[1][0]['user_request'] == DEFAULT_REQUEST
    task = case.models[0].calls[1][0]['research_context']['task']
    assert task['delivery'] == ['explanation'] and task['delivery_status'] == 'deferred'
    compiled = {step.step_id: step for step in stopped.result.plan.steps}
    assert len(compiled) == 5
    assert compiled['inspect'].arguments['path'] == case.record.source_path
    assert compiled['pca'].arguments['input_path'] == StepOutputRef('inspect', 'input_path')
    assert compiled['pca'].depends_on == ('inspect',)
    assert compiled['pca'].arguments.get('n_components') is None
    assert compiled['graph'].arguments['embedding_path'] == StepOutputRef('pca', 'embedding_path')
    assert compiled['graph'].arguments['cell_ids_path'] == StepOutputRef('pca', 'cell_ids_path')
    assert compiled['clusters'].arguments['analysis_path'] == StepOutputRef('graph', 'analysis_path')
    assert compiled['umap'].arguments['analysis_path'] == StepOutputRef('clusters', 'analysis_path')
    state = assert_no_scientific_records(case)
    admitted = state.interactions[-1].admitted
    assert admitted['roles'] == tuple(shared_source_decision()['roles'])
    assert admitted['inputs'] == dict(input_path=case.record.source_path)
    assert len(admitted['registered_input']['members']) == 1
    assert Path(case.record.source_path).read_bytes() == case.original


def test_shared_request_alias_cannot_bind_different_registered_sources(tmp_path, monkeypatch):
    case = setup_upload(tmp_path, decision=shared_source_decision('source.1'))
    async def chunks():
        yield case.original
    other = asyncio.run(case.uploads.receive('other-matrix', 'other-simulated.h5ad', chunks(),
        expected_size=len(case.original), session_id='session'))
    # Identical simulated bytes at distinct registered locators cannot authorize
    # incompatible values for one request alias or collapse the selected files.
    assert other.source_sha256 == case.record.source_sha256
    assert other.source_path != case.record.source_path
    install_operator_stop(monkeypatch)
    view = case.facade.submit_turn('session', 'request', DEFAULT_REQUEST, expected_generation=0,
        attachment_ids=(case.record.resource_id, other.resource_id))
    assert view.status == 'failed' and view.error.code == 'MULTI_INPUT_ROLE_INVALID'
    assert len(case.models[0].calls) == 1
    assert_no_scientific_records(case)
