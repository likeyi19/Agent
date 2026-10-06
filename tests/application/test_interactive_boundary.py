"""M18.2 real application interfaces with tiny science and scripted LLMs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import sys
from threading import Event

import pytest

from agent.application import InteractiveAgentApplication, InteractiveBoundaryError, ResearchAgentApplication
from agent.orchestration import PlanningModelProfile, ToolRegistry
from agent.providers import PlanningModelFactoryRegistry
from agent.schemas import AgentError, AgentPlan, AgentRequest, PlanStep, ErrorCategory, StepExecutionResult, StepStatus
from agent.schemas.run_state import PersistedRunState, RunLifecycleStatus, fingerprint_plan, RecoveryPolicySnapshot, ToolRecoveryPolicySnapshot, fingerprint_recovery_policy
from agent.application.session_state import Interaction, digest
from test_service import _tiny_h5ad, _counting_registry, _semantic_planning_response
from test_scientific_dialogue import Model as ScientificModel, question
from test_scientific_guidance import Model as GuidanceModel, decision as guidance_decision
from test_dialogue_evidence import app, accepted, passed


PROFILES = tuple(PlanningModelProfile(key, 'scripted', 'model/' + key) for key in ('alpha', 'beta'))


class ScriptedModel:
    def __init__(self, profile, decision=None, *, started=None, release=None):
        self.model_id, self.calls = profile.model_id, []
        self.decision = decision or dict(kind='execute_plan', target='inspect_scATAC')
        self.started, self.release = started, release

    def complete(self, *, prompt, response_schema):
        value = json.loads(prompt)
        self.calls.append(value)
        if 'turn_schema_version' in value:
            return json.dumps(dict(turn_schema_version=1,
                decision=self.decision(value) if callable(self.decision) else self.decision))
        if 'selection_schema_version' in response_schema.get('properties', {}):
            if self.started:
                self.started.set()
                assert self.release.wait(20)
            return json.dumps(dict(selection_schema_version=1,
                decision=dict(kind='select', capability_ids=['processed_inspection'])))
        if 'output_selection_schema_version' in value:
            return json.dumps(dict(outputs=[dict(name='inspection', step_id='inspect', output_key='n_cells')]))
        if 'dialogue_schema_version' in value:
            return ScientificModel(question(), ['n_cells']).complete(prompt=prompt, response_schema=response_schema)
        if 'guidance_schema_version' in value:
            return GuidanceModel().complete(prompt=prompt, response_schema=response_schema)
        return _semantic_planning_response()


def boundary(tmp_path, *, registry=None, decision=None, started=None, release=None, workspace=None):
    models = []
    def factory(profile):
        model = ScriptedModel(profile, decision, started=started, release=release)
        models.append(model)
        return model
    service = InteractiveAgentApplication(workspace or tmp_path / 'workspace',
        model_profiles=PROFILES, default_profile_id='alpha', registry=registry,
        planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted': factory}))
    return service, models


def first(service, source, *, session='session', turn='first', profile=None):
    return service.submit_turn(session, turn, 'Inspect the supplied matrix.', expected_generation=0,
        execution_inputs={'input_path': str(source)}, profile_id=profile)


def test_first_real_turn_and_fresh_process_reopen_retry(tmp_path):
    calls = []
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    service, models = boundary(tmp_path, registry=_counting_registry(calls))
    assert service.create_session('session').active_revision_id is None
    choices = service.model_choices()
    assert [c.profile_id for c in choices] == ['alpha', 'beta'] and choices[0].is_default
    view = first(service, source)
    assert view.status == 'succeeded' and view.response.kind == 'execute'
    assert view.response.status == 'activated' and view.revision_id
    assert calls == ['inspect_scATAC']
    assert len(models) == 1 and len(models[0].calls) == 4
    assert service.status('session', 'first') == view
    assert first(service, source) == view and len(models) == 1 and calls == ['inspect_scATAC']
    encoded = json.dumps(view.to_dict())
    assert str(tmp_path) not in encoded
    assert not {'resolved_arguments', 'authority', 'workspace_path'} & view.to_dict().keys()
    code = '''
import json, sys
from agent.application import InteractiveAgentApplication, InteractiveBoundaryError
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
def forbidden(profile): raise AssertionError('Provider constructed during history/retry')
profiles=tuple(PlanningModelProfile(k,'scripted','model/'+k) for k in ('alpha','beta'))
app=InteractiveAgentApplication(sys.argv[1],model_profiles=profiles,default_profile_id='alpha',
    planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':forbidden}))
history=app.reopen_session('session')
retry=app.submit_turn('session','first','Inspect the supplied matrix.',expected_generation=0,
    execution_inputs={'input_path':sys.argv[2]})
assert history.turns[0].response == retry.response
assert history.active_revision_id == retry.revision_id
try:
    app.submit_turn('session','bad','Inspect the supplied matrix.',expected_generation=1,profile_id='unknown')
except InteractiveBoundaryError as exc:
    assert exc.error.code == 'INTERACTIVE_MODEL_UNAVAILABLE'
else: raise AssertionError('Unknown model admitted')
print(json.dumps(retry.to_dict()))
'''
    child = subprocess.run([sys.executable, '-B', '-c', code,
        str(service._application.workspace_root), str(source)], env=os.environ.copy(), capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == view.to_dict()
    evidence = service._application.sessions.evidence('session', view.revision_id, 'inspection')
    assert evidence.status == 'available' and calls == ['inspect_scATAC']


def test_nonexecuting_first_turn_and_stored_display(tmp_path):
    service, models = boundary(tmp_path, decision=dict(kind='clarify', reason='missing_parameter_value'))
    service.create_session('session')
    view = service.submit_turn('session', 'hello', 'Please clarify.', expected_generation=0)
    assert view.response.kind == 'clarify' and view.response.clarification['value_required']
    assert service.reopen_session('session').generation == 0 and view.revision_id is None
    assert service.submit_turn('session', 'hello', 'Please clarify.', expected_generation=0) == view
    assert len(models) == 1 and len(models[0].calls) == 1


@pytest.mark.parametrize('change', [dict(utterance='Inspect a different matrix.'),
    dict(expected_generation=1), dict(profile_id='beta'), dict(execution_inputs={}),
    dict(predecessor_turn_id='different')])
def test_conflicting_id_reuse_fails_before_factory(tmp_path, change):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    service, models = boundary(tmp_path)
    service.create_session('session')
    first(service, source)
    kwargs = dict(utterance='Inspect the supplied matrix.', expected_generation=0,
                  execution_inputs={'input_path': str(source)})
    kwargs.update(change)
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.submit_turn('session', 'first', **kwargs)
    assert caught.value.error.code == 'INTERACTIVE_TURN_CONFLICT' and len(models) == 1


def test_stale_generation_unknown_model_and_internal_inputs_fail_before_provider(tmp_path):
    service, models = boundary(tmp_path)
    service.create_session('session')
    cases = [(dict(expected_generation=1), 'INTERACTIVE_GENERATION_CONFLICT'),
             (dict(profile_id='unoffered'), 'INTERACTIVE_MODEL_UNAVAILABLE'),
             (dict(execution_inputs={'output_dir':'/server/private'}), 'INTERACTIVE_INPUT_INVALID'),
             (dict(execution_inputs={'x':{'$prior_output':{}}}), 'INTERACTIVE_INPUT_INVALID'),
             (dict(predecessor_turn_id='missing'), 'INTERACTIVE_REFERENCE_INVALID'),
             (dict(execution_inputs={'x':'large' * 20000}), 'INTERACTIVE_INPUT_INVALID')]
    for kwargs, expected in cases:
        args = dict(expected_generation=0)
        args.update(kwargs)
        with pytest.raises(InteractiveBoundaryError) as caught:
            service.submit_turn('session','invalid','Inspect the supplied matrix.',**args)
        assert caught.value.error.code == expected and '/server/private' not in json.dumps(caught.value.error.to_dict())
    assert not models and not service.reopen_session('session').turns


def test_concurrent_profiles_bind_all_roles_and_duplicate_pending(tmp_path):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    started, release = Event(), Event()
    calls = []
    service, models = boundary(tmp_path, registry=_counting_registry(calls), started=started, release=release)
    for name in ('one','two'):
        service.create_session(name)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first, service, source, session='one', profile='alpha')
        assert started.wait(20)
        pending = first(service, source, session='one', profile='alpha')
        assert pending.status == 'planning' and pending.response is None
        assert service.status('one','first').status == 'planning'
        b = pool.submit(first, service, source, session='two', profile='beta')
        release.set()
        av, bv = a.result(), b.result()
    assert av.profile_id == 'alpha' and bv.profile_id == 'beta'
    assert av.status == bv.status == 'succeeded' and calls == ['inspect_scATAC'] * 2
    assert {m.model_id for m in models} == {'model/alpha','model/beta'}
    for model in models:
        assert len(model.calls) == 4
        assert sum('turn_schema_version' in p for p in model.calls) == 1
        assert sum('output_selection_schema_version' in p for p in model.calls) == 1
    assert service.reopen_session('one').generation == service.reopen_session('two').generation == 1


def test_running_status_uses_checkpoint_and_cooperative_cancel(tmp_path):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    started, release = Event(), Event()
    counted = _counting_registry([])
    original = counted.get('inspect_scATAC')
    def blocked(**arguments):
        started.set()
        assert release.wait(20)
        return original.function(**arguments)
    registry = ToolRegistry(tuple(replace(counted.get(n), function=blocked) if n == 'inspect_scATAC'
                                  else counted.get(n) for n in counted.names()))
    service, models = boundary(tmp_path, registry=registry)
    service.create_session('session')
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first,service,source)
        assert started.wait(20)
        status = service.status('session','first')
        assert status.status == 'running' and status.steps[0].status == 'RUNNING'
        assert status.steps[0].attempts == 1 and status.state_revision is not None
        assert not {'percentage', 'eta', 'worker_alive'} & status.to_dict().keys()
        assert service.cancel_turn('session','first').disposition.value == 'REQUESTED'
        release.set()
        final = future.result()
    assert final.status == 'cancelled' and final.revision_id is None


@pytest.mark.parametrize('kind', ['scientific','guidance'])
def test_answer_and_guidance_history_does_not_regenerate(app, tmp_path, kind):
    accepted(app, 'inspect_scATAC')
    choice = question() if kind == 'scientific' else guidance_decision('r0')
    service, models = boundary(tmp_path, workspace=app.workspace_root, decision=choice)
    before = app.sessions.load('session')
    view = service.submit_turn('session','discuss','What does this result suggest?',expected_generation=1)
    assert view.response.kind == 'answer' and view.response.status == 'answered'
    assert getattr(view.response,kind) is not None
    assert len(models[0].calls) == 2
    reopened, unused = boundary(tmp_path, workspace=app.workspace_root)
    assert reopened.turn('session','discuss').response == view.response
    assert reopened.submit_turn('session','discuss','What does this result suggest?',expected_generation=1).response == view.response
    assert not unused and app.sessions.load('session').revisions == before.revisions
    assert 'claims' not in (view.response.scientific or {})
    script = '''
import json,sys
from agent.application import InteractiveAgentApplication
from agent.orchestration import PlanningModelProfile
from agent.providers import PlanningModelFactoryRegistry
def forbidden(profile): raise AssertionError('History must not construct a model')
app=InteractiveAgentApplication(sys.argv[1],model_profiles=(PlanningModelProfile('alpha','scripted','model/alpha'),),
 default_profile_id='alpha',planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':forbidden}))
print(json.dumps(app.turn('session','discuss').response.to_dict()))
'''
    child=subprocess.run([sys.executable,'-B','-c',script,str(app.workspace_root)],
        env=os.environ.copy(),capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    assert json.loads(child.stdout)==view.response.to_dict()
    interaction=app.sessions.load('session').interactions[-1]
    class Forbidden:
        def complete(self,**kwargs):
            pytest.fail('Direct completed interactive replay regenerated a model answer')
    replay=app.sessions.respond('session','discuss','What does this result suggest?',
        interpreter=Forbidden(),submission=interaction.submission)
    assert replay.text==view.response.text and replay.presentation==view.response


def test_legacy_missing_presentation_is_explicit_without_regeneration(app, tmp_path):
    accepted(app,'inspect_scATAC')
    app.sessions.respond('session','old','What does this result show?',
        interpreter=ScientificModel(question(), ['n_cells']))
    service, models = boundary(tmp_path,workspace=app.workspace_root)
    old = service.turn('session','old')
    assert old.response is None and old.error.code == 'INTERACTIVE_PRESENTATION_UNAVAILABLE'
    assert service.reopen_session('session').turns and not models


def test_artifact_handles_bind_exact_accepted_report_and_no_raw_paths(tmp_path):
    source = _tiny_h5ad(tmp_path / 'tiny.h5ad')
    calls=[]
    service, models = boundary(tmp_path,registry=_counting_registry(calls))
    service.create_session('session')
    result = first(service,source)
    handles = service.artifact_handles('session',result.revision_id)
    assert len(handles) == 1 and handles[0].artifact_type == 'analysis_report'
    assert str(tmp_path) not in json.dumps([h.to_dict() for h in handles])
    assert service.resolve_artifact('session',result.revision_id,handles[0].handle) == handles[0]
    assert calls == ['inspect_scATAC']
    with pytest.raises(InteractiveBoundaryError):
        service.resolve_artifact('session',result.revision_id,'/etc/passwd')
    files = service._application.sessions.load('session').turn('first').completion_files
    report = next(f for f in files if f.path.endswith('.md'))
    from pathlib import Path
    Path(report.path).write_bytes(Path(report.path).read_bytes()+b'changed')
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.resolve_artifact('session',result.revision_id,handles[0].handle)
    assert caught.value.error.code == 'INTERACTIVE_ARTIFACT_UNAVAILABLE'


@pytest.mark.parametrize('lifecycle,expected', [(RunLifecycleStatus.PLANNING,'planning'),
    (RunLifecycleStatus.VALIDATED,'validated'),(RunLifecycleStatus.RUNNING,'running'),
    (RunLifecycleStatus.FAILED,'failed'),(RunLifecycleStatus.INTERRUPTED,'interrupted'),
    (RunLifecycleStatus.SUCCEEDED,'finalizing')])
def test_status_projects_durable_lifecycle_without_terminal_conversion(tmp_path,lifecycle,expected):
    service, models = boundary(tmp_path)
    service.create_session('session')
    application = service._application
    interaction = Interaction('probe','Inspect.',None,0,{'relations':{},'bases':{}},status='admitted',
        admitted={'kind':'execute','request_id':'probe'})
    application.sessions._store._update('session',lambda s:replace(s,interactions=(interaction,)))
    request = AgentRequest('probe','Inspect.',{})
    plan = None if lifecycle is RunLifecycleStatus.PLANNING else AgentPlan('probe-plan','probe','fixture',
        (PlanStep('inspect','inspect_scATAC',{'path':'fixture'}),))
    error = None
    if lifecycle in {RunLifecycleStatus.FAILED,RunLifecycleStatus.INTERRUPTED}:
        error = AgentError(ErrorCategory.INTERNAL_AGENT_ERROR,
            'STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION' if lifecycle is RunLifecycleStatus.INTERRUPTED else 'PLANNER_UNEXPECTED_ERROR',
            'Fixture safe error.')
    status = (StepStatus.SUCCEEDED if lifecycle is RunLifecycleStatus.SUCCEEDED else
              StepStatus.RUNNING if lifecycle is RunLifecycleStatus.RUNNING else
              StepStatus.FAILED if error else StepStatus.PENDING)
    steps = () if plan is None else (StepExecutionResult('inspect','inspect_scATAC',status,
        attempt_count=0 if status is StepStatus.PENDING else 1,
        result={'n_cells':2} if status is StepStatus.SUCCEEDED else None,
        verification=passed('step','inspect') if status is StepStatus.SUCCEEDED else None,error=error,
        started_at=None if status is StepStatus.PENDING else '2026-09-23T00:00:00+00:00',
        finished_at='2026-09-23T00:00:00+00:00' if status in {StepStatus.SUCCEEDED,StepStatus.FAILED} else None,
        duration_seconds=0.0 if status in {StepStatus.SUCCEEDED,StepStatus.FAILED} else None),)
    policies = (ToolRecoveryPolicySnapshot('inspect_scATAC',application.registry.get('inspect_scATAC').recovery_policy_version),)
    policy = RecoveryPolicySnapshot('fixture',1,policies,fingerprint_recovery_policy('fixture',1,policies))
    state = PersistedRunState(3,0,'probe:run',request,lifecycle,'2026-09-23T00:00:00+00:00',
        '2026-09-23T00:00:00+00:00',plan=plan,plan_fingerprint=None if plan is None else fingerprint_plan(plan),
        recovery_policy_snapshot=None if plan is None else policy,
        preflight_verification=None if plan is None else passed('plan',plan.plan_id),steps=steps,
        errors=() if error is None else (error,),run_verification=passed('run',plan.plan_id)
        if lifecycle is RunLifecycleStatus.SUCCEEDED else None)
    application.run_store.create(state)
    view=service.status('session','probe')
    assert view.status==expected and not models
    assert view.state_revision==0


def test_display_ack_failure_and_explicit_recovery_never_repeat_science(tmp_path,monkeypatch):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    calls=[]
    service,models=boundary(tmp_path,registry=_counting_registry(calls))
    service.create_session('session')
    from agent.application.sessions import AnalysisSessions
    original=AnalysisSessions.store_presentation
    def missing(*args,**kwargs):
        raise OSError('Private filesystem details must not reach client')
    monkeypatch.setattr(AnalysisSessions,'store_presentation',missing)
    with pytest.raises(InteractiveBoundaryError) as caught:
        first(service,source)
    assert 'Private' not in caught.value.error.message
    assert calls==['inspect_scATAC'] and len(models)==1
    monkeypatch.setattr(AnalysisSessions,'store_presentation',original)
    retry=first(service,source)
    assert retry.status=='succeeded' and retry.response is None
    recovered=service.recover_turn('session','first')
    assert recovered.response.status=='activated' and recovered.response.text.startswith('The request followed')
    assert first(service,source)==recovered and len(models)==1 and calls==['inspect_scATAC']
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.recover_turn('session','first',complete_presentation='false')
    assert caught.value.error.code=='INTERACTIVE_INPUT_INVALID'


def test_execution_recovery_preserves_parent_and_historical_active_distinction(tmp_path,monkeypatch):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    calls=[]
    service,models=boundary(tmp_path,registry=_counting_registry(calls))
    service.create_session('session')
    from agent.application.sessions import AnalysisSessions
    original=AnalysisSessions.store_presentation
    def missing(*args,**kwargs): raise OSError('Lost completion')
    monkeypatch.setattr(AnalysisSessions,'store_presentation',missing)
    with pytest.raises(InteractiveBoundaryError): first(service,source)
    monkeypatch.setattr(AnalysisSessions,'store_presentation',original)
    next_turn=service.submit_turn('session','second','Inspect the supplied matrix.',expected_generation=1,
        execution_inputs={'input_path':str(source)})
    recovered=service.recover_turn('session','first')
    assert 'historical' in recovered.response.text and 'form the active revision' not in recovered.response.text
    assert service.reopen_session('session').active_revision_id==next_turn.revision_id
    assert calls==['inspect_scATAC']*2 and len(models)==2


def test_exact_empty_stored_display_is_not_rerendered(tmp_path):
    service,models=boundary(tmp_path,decision=dict(kind='clarify',reason='unsupported_intent'))
    service.create_session('session')
    # Represent an already admitted display with intentionally empty text.
    app=service._application
    model=ScriptedModel(PROFILES[0],dict(kind='clarify',reason='unsupported_intent'))
    submission=service._submission(PROFILES[0],0,{},None)
    outcome=app.sessions.respond('session','empty-text','Clarify.',interpreter=model,
        expected_generation=0,submission=submission)
    from agent.application.interactive import _present
    shown=replace(_present(outcome),text='')
    app.sessions.store_presentation('session','empty-text',shown.to_dict())
    class Forbidden:
        def complete(self,**kwargs): pytest.fail('Stored display regenerated')
    replay=app.sessions.respond('session','empty-text','Clarify.',interpreter=Forbidden(),submission=submission)
    assert replay.text=='' and replay.presentation==shown
    assert service.turn('session','empty-text').response==shown


def test_guidance_subset_display_retains_original_option_identity():
    from types import SimpleNamespace
    from agent.application import GuidanceCandidate, GuidanceResponse, ScientificResponse, TurnOutcome
    from agent.application.interactive import _present
    refs=[dict(candidate_id='candidate-one',origin_turn_id='origin',capability='cluster_cells'),
          dict(candidate_id='candidate-two',origin_turn_id='origin',capability='inspect_scATAC')]
    response=ScientificResponse((), 'insufficient_evidence', 'Conditional guidance.', ())
    selected=GuidanceCandidate(refs[1],{},response)
    outcome=TurnOutcome('answer','answered',text='Option 1: inspect_scATAC\nConditional guidance.',
        guidance=GuidanceResponse((selected,)))
    state=SimpleNamespace(interactions=(SimpleNamespace(turn_id='origin',guidance_candidates=refs),))
    shown=_present(outcome,state)
    assert shown.guidance['candidates'][0]['option']==2
    assert shown.text=='Option 2: inspect_scATAC\nConditional guidance.'


def test_application_composition_failure_is_safe_and_explicit(tmp_path,monkeypatch):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    service,models=boundary(tmp_path)
    service.create_session('session')
    from agent.application import service as service_module
    def fail(*args,**kwargs):
        raise RuntimeError('/private/server/path API_KEY=secret')
    monkeypatch.setattr(service_module,'build_analysis_report',fail)
    result=first(service,source)
    assert result.status=='finalizing'
    assert result.error.code=='APP_REPORT_FAILED' and result.response.error.code=='APP_REPORT_FAILED'
    assert 'secret' not in json.dumps(result.to_dict()) and '/private' not in json.dumps(result.to_dict())
    assert first(service,source).response==result.response and len(models)==1


def test_latest_interaction_and_missing_run_have_explicit_views(app,tmp_path):
    accepted(app,'inspect_scATAC')
    service,models=boundary(tmp_path,workspace=app.workspace_root,decision=question())
    service.submit_turn('session','latest','Explain this result.',expected_generation=1)
    history=service.reopen_session('session',limit=1)
    assert history.turns[0].turn_id=='latest' and history.history_truncated
    old=service.turn('session','initial')
    app.run_store.state_path(old.run_id).unlink()
    unavailable=service.turn('session','initial')
    assert unavailable.status=='unavailable' and unavailable.error.code=='INTERACTIVE_REFERENCE_INVALID'


def test_status_reuses_exact_session_request_anchor(tmp_path,monkeypatch):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    service,models=boundary(tmp_path)
    service.create_session('session')
    view=first(service,source)
    original=service._application.run_store.load
    def changed(run_id):
        run=original(run_id)
        return replace(run,request=replace(run.request,prompt='Replaced request /private/path'))
    monkeypatch.setattr(service._application.run_store,'load',changed)
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.status('session','first')
    assert caught.value.error.code=='INTERACTIVE_APPLICATION_FAILED'
    assert '/private/path' not in str(caught.value)
    assert len(models)==1


def test_model_selection_on_answers_preserves_scientific_state(tmp_path):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    def choose(value):
        return dict(kind='execute_plan',target='inspect_scATAC') if value['utterance'].startswith('Inspect') else question()
    service,models=boundary(tmp_path,decision=choose)
    service.create_session('session')
    first(service,source)
    before=service._application.sessions.load('session')
    reply=service.submit_turn('session','beta-answer','Explain this result.',expected_generation=1,profile_id='beta')
    after=service._application.sessions.load('session')
    assert reply.response.scientific is not None and reply.profile_id=='beta'
    assert models[-1].model_id=='model/beta' and len(models[-1].calls)==2
    assert after.revisions==before.revisions and after.active_revision_id==before.active_revision_id
    assert after.generation==before.generation


def test_invalid_ids_and_lease_filesystem_errors_are_sanitized(tmp_path,monkeypatch):
    service,models=boundary(tmp_path)
    with pytest.raises(InteractiveBoundaryError):
        service.create_session('invisible\u200b')
    service.create_session('session')
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.submit_turn('session','surrogate','Inspect.',expected_generation=0,execution_inputs={'text':'\ud800'})
    assert caught.value.error.code=='INTERACTIVE_INPUT_INVALID'
    from agent.application.session_store import FileSessionStore
    from contextlib import contextmanager
    @contextmanager
    def failed_lease(*args,**kwargs):
        raise OSError('/private/session/lease')
        yield
    monkeypatch.setattr(FileSessionStore,'processing_lease',failed_lease)
    with pytest.raises(InteractiveBoundaryError) as caught:
        service.submit_turn('session','first','Inspect.',expected_generation=0)
    assert caught.value.error.code=='INTERACTIVE_APPLICATION_FAILED' and '/private' not in str(caught.value)
    assert not models


def test_two_fresh_processes_complete_the_acceptance_sequence(tmp_path):
    source=_tiny_h5ad(tmp_path/'tiny.h5ad')
    workspace=tmp_path/'acceptance'
    producer='''
import json,sys
from test_interactive_boundary import boundary,first
from test_service import _counting_registry
from pathlib import Path
calls=[]
app,models=boundary(Path(sys.argv[1]).parent,workspace=Path(sys.argv[1]),registry=_counting_registry(calls))
assert app.create_session('accept').active_revision_id is None
assert app.model_choices()[0].is_default
result=first(app,Path(sys.argv[2]),session='accept')
assert result.status=='succeeded' and result.revision_id
assert app.status('accept','first')==result
assert first(app,Path(sys.argv[2]),session='accept')==result
assert calls==['inspect_scATAC'] and len(models)==1 and len(models[0].calls)==4
print(json.dumps(result.to_dict()))
'''
    consumer='''
import json,sys
from agent.application import InteractiveAgentApplication,InteractiveBoundaryError
from agent.providers import PlanningModelFactoryRegistry
from test_interactive_boundary import PROFILES,first
from pathlib import Path
def forbidden(profile): raise AssertionError('Provider invoked after restart')
app=InteractiveAgentApplication(sys.argv[1],model_profiles=PROFILES,default_profile_id='alpha',
 planning_model_factory_registry=PlanningModelFactoryRegistry({'scripted':forbidden}))
history=app.reopen_session('accept')
result=first(app,Path(sys.argv[2]),session='accept')
assert history.turns[0].response==result.response
assert app._application.sessions.evidence('accept',result.revision_id,'inspection').status=='available'
try: app.submit_turn('accept','unoffered','Inspect.',expected_generation=1,profile_id='unknown')
except InteractiveBoundaryError as exc: assert exc.error.code=='INTERACTIVE_MODEL_UNAVAILABLE'
else: raise AssertionError('Unknown model admitted')
print(json.dumps(result.to_dict()))
'''
    env=dict(os.environ,PYTHONPATH='src:tests/application',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    for code in (producer,consumer):
        result=subprocess.run([sys.executable,'-B','-c',code,str(workspace),str(source)],
            env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        results.append(json.loads(result.stdout))
    assert results[0]==results[1]
