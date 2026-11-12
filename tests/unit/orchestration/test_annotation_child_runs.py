"""ICTA-1 lifecycle fixtures, not production biological annotation acceptance."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading

import pytest

from agent.orchestration import AgentRuntime, FileRunStore, RunAlreadyActiveError
from agent.orchestration.annotation_runs import annotation_invocation, continuation_anchor
from agent.orchestration.durable_tool_recovery import execution_identity
from agent.orchestration.registry import (
    ArgumentSpec, ArgumentPlanningSemantics, DurableToolHooks, ErrorClassification,
    PlanningSourceEligibility, ResultContract, ScientificAuthorityHooks,
    SemanticConsumerPortSpec, SemanticPortMember, SemanticProducerPortSpec, SemanticToolSpec,
    ToolRegistry, ToolSpec,
)
from agent.orchestration.prior_outputs import bind_output, prior_output_store, validate_binding
from agent.schemas import (AgentPlan, AgentRequest, ErrorCategory, PlanStep,
    RunLifecycleStatus, RunStatus, StepOutputRef, StepStatus, TraceEventType)
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.verification_authority import AuthorityError
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools._file_identity import file_closure
from agent.tools.analysis.native_publication import NativePublication
from agent.tools.data.authority_context import current


class Crash(BaseException):
    pass


class FaultStore:
    """One injected boundary; all state/lease/CAS operations use FileRunStore."""
    def __init__(self, store, predicate, *, before=False):
        self.store, self.predicate, self.before = store, predicate, before
        self.triggered = False

    def __getattr__(self, name):
        return getattr(self.store, name)

    def update(self, state, *, expected_revision):
        hit = not self.triggered and self.predicate(state)
        if hit and self.before:
            self.triggered = True
            raise Crash()
        saved = self.store.update(state, expected_revision=expected_revision)
        if hit:
            self.triggered = True
            raise Crash()
        return saved

    def create(self, state):
        saved = self.store.create(state)
        if not self.triggered and self.predicate(saved):
            self.triggered = True
            raise Crash()
        return saved


def parameter(types):
    return ArgumentSpec(types, planning=ArgumentPlanningSemantics(
        'Fixture native parameter', PlanningSourceEligibility.REQUEST_INPUT_ONLY,
        scientific_parameter=True, conversational_json_object=types == (dict,)))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from types import SimpleNamespace
    h = SimpleNamespace(store=FileRunStore(tmp_path / 'run_state'), calls=[], identities=[],
        before_parent=None, after_child=None, child_action=None, publications={}, exceptions=[])
    source = tmp_path / 'source.txt'
    source.write_text('exact original scientific input')
    h.source = str(source)
    h.sha = hashlib.sha256(source.read_bytes()).hexdigest()

    def spec(name, coordinator=False):
        pub = NativePublication(tool=name, kind='fixture_' + name, contract=name + '.v1',
            profile='fixture.v1', verifier={'id': name, 'compatibility_version': '1'},
            files=('data.txt',), arguments=lambda a: dict(a), outputs={'data': 'data.txt'})
        h.publications[name] = pub

        def execute(args, identity):
            context = current()
            assert context.execution[2] == identity
            h.identities.append((name, identity, context))
            if coordinator:
                inv = annotation_invocation()
                if h.before_parent:
                    h.before_parent(inv)
                result = inv.invoke(branch_id='root', operation_id='first', tool_name='child',
                    arguments=args.get('parameters', {}),
                    parent_inputs={'input_path': 'input_path', 'input_sha256': 'input_sha256'})
                assert current() is context and context.execution[2] == identity
                if h.after_child:
                    h.after_child(inv, result)
            else:
                with pytest.raises(RuntimeError, match='No active'):
                    annotation_invocation()
                assert hashlib.sha256(Path(args['input_path']).read_bytes()).hexdigest() == args['input_sha256']
                h.calls.append(dict(args))
                if h.child_action:
                    h.child_action(args, identity)
            out = Path(args['output_dir'])
            out.mkdir(parents=True, exist_ok=False)
            (out / 'data.txt').write_text(identity)
            value = dict(contract=pub.contract, profile=pub.profile, arguments=dict(args),
                native_completion='succeeded', execution_identity=identity,
                files=file_closure([out / 'data.txt']), sources=file_closure([args['input_path']]), result={})
            path = out / 'manifest.json'
            path.write_text(json.dumps(value, sort_keys=True))
            return pub.verify_publication(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

        def recover(args, identity):
            return pub.recover(args, identity)

        def validate(result):
            assert pub.verify_publication(result['manifest_path'], expected_sha256=result['manifest_sha256']) == dict(result)

        members = (SemanticPortMember('path', 'input_path'), SemanticPortMember('sha256', 'input_sha256'))
        return ToolSpec(name, lambda **a: pytest.fail('must use durable hook'),
            {'input_path': ArgumentSpec((str,)), 'input_sha256': ArgumentSpec((str,)), 'output_dir': ArgumentSpec((str,))},
            {'parameters': parameter((dict,))} if coordinator else
                {'kwargs': parameter((dict,)), 'parameter_selection': parameter((str,))},
            ResultContract(name, {'manifest_path': (str,), 'manifest_sha256': (str,),
                                 'data_path': (str,), 'data_sha256': (str,)}, validator=validate),
            lambda e: h.exceptions.append(repr(e)) or ErrorClassification(ErrorCategory.TOOL_EXECUTION_ERROR, 'FIXTURE_FAILURE'),
            durable_hooks=DurableToolHooks(execute, recover),
            scientific_authority=ScientificAuthorityHooks(pub.kind, ('dataset',), pub.describe,
                pub.publication_record, pub.reuse_result, pub.proof_metadata),
            semantic_planning=SemanticToolSpec(consumer_ports=(SemanticConsumerPortSpec('dataset', members, True,
                accepted_upstream_types=('fixture.data.v1',)),), producer_ports=(
                SemanticProducerPortSpec('dataset', 'fixture.data.v1',
                    (SemanticPortMember('path', 'data_path'), SemanticPortMember('sha256', 'data_sha256'))),)),
            annotation_children=frozenset({'child'}) if coordinator else frozenset())

    h.registry = ToolRegistry((spec('parent', True), spec('child')))
    # Existing authority loaders deliberately resolve reviewed Registry owners.
    # This fixture substitutes that inventory, never their validation machinery.
    monkeypatch.setattr('agent.orchestration.registry.build_default_tool_registry', lambda: h.registry)
    monkeypatch.setattr('agent.orchestration.verification_authority.build_default_tool_registry', lambda: h.registry)
    h.runtime = lambda store=None: AgentRuntime(registry=h.registry, run_store=store or h.store)

    def prepare(request_id='parent-1', parameters=None):
        request = AgentRequest(request_id, None, {})
        args = dict(input_path=h.source, input_sha256=h.sha, output_dir=str(tmp_path / request_id))
        if parameters is not None:
            args['parameters'] = parameters
        return request, AgentPlan(request_id + ':plan', request_id, 'Fixture annotation', (PlanStep('annotate', 'parent', args),))
    h.prepare = prepare
    return h


def reservations(state):
    return [e.details for e in state.trace if e.event_type is TraceEventType.ANNOTATION_PROGRESS and e.details['event'] == 'reserved']


def child_state(h, parent='parent-1:run'):
    row = reservations(h.store.load(parent))[0]
    return h.store.load(row['request']['request_id'] + ':run')


def continue_parent(h, request_id='parent-2', parameters=None):
    runtime = h.runtime()
    runtime.resume('parent-1:run')  # Reconcile only; never re-enter the coordinator.
    anchor = continuation_anchor(h.store, 'parent-1:run', 'annotate')
    before = h.store.load('parent-1:run').to_dict()
    result = runtime.continue_annotation(*h.prepare(request_id, parameters), previous=anchor)
    assert h.store.load('parent-1:run').to_dict() == before
    return result


@pytest.mark.parametrize('boundary', ['before_reservation', 'reserved', 'child_created', 'child_running',
    'published_before_checkpoint', 'child_success', 'parent_before_publication'])
def test_crash_windows_and_exact_explicit_continuation(harness, boundary):
    h = harness
    store = h.store
    if boundary == 'before_reservation':
        h.before_parent = lambda inv: (_ for _ in ()).throw(Crash())
    elif boundary == 'parent_before_publication':
        h.after_child = lambda *a: (_ for _ in ()).throw(Crash())
    else:
        predicates = {
            'reserved': lambda s: s.run_id == 'parent-1:run' and bool(reservations(s)),
            'child_created': lambda s: s.run_id.startswith('annotation-child-') and s.plan is None,
            'child_running': lambda s: s.run_id.startswith('annotation-child-') and bool(s.steps) and s.steps[0].status is StepStatus.RUNNING,
            'published_before_checkpoint': lambda s: s.run_id.startswith('annotation-child-') and bool(s.steps) and s.steps[0].status is StepStatus.SUCCEEDED,
            'child_success': lambda s: s.run_id.startswith('annotation-child-') and s.lifecycle_status is RunLifecycleStatus.SUCCEEDED,
        }
        store = FaultStore(h.store, predicates[boundary], before=boundary == 'published_before_checkpoint')
    with pytest.raises(Crash):
        h.runtime(store).run_prepared(*h.prepare())
    old = h.store.load('parent-1:run')
    assert old.steps[0].status is StepStatus.RUNNING
    h.before_parent = h.after_child = None
    result = continue_parent(h)
    if boundary == 'child_running':
        assert result.status is RunStatus.FAILED
        assert len(h.calls) == 0  # RUNNING persisted, but fixture body never entered.
        child = child_state(h)
        assert child.lifecycle_status is RunLifecycleStatus.INTERRUPTED
        assert child.errors[0].code == 'STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION'
        unchanged = child.to_dict()
        h.runtime().resume(child.run_id)
        assert h.store.load(child.run_id).to_dict() == unchanged
    else:
        assert result.status is RunStatus.SUCCEEDED, h.exceptions
        assert len(h.calls) == 1
        if reservations(old):
            child = child_state(h)
            assert Path(child.steps[0].result['data_path']).is_file()
    assert h.store.load('parent-1:run').lifecycle_status is RunLifecycleStatus.INTERRUPTED


@pytest.mark.parametrize('parameters', [{}, {'kwargs': {'resolution': 0.37}},
    {'parameter_selection': 'native_defaults'}, {'parameter_selection': 'asw'},
    {'parameter_selection': 'asw', 'kwargs': {'resolution': 0.37}}])
def test_parameters_and_independent_execution_identities(harness, parameters):
    h = harness
    result = h.runtime().run_prepared(*h.prepare(parameters=parameters))
    assert result.status is RunStatus.SUCCEEDED, h.exceptions
    assert {k: h.calls[0][k] for k in parameters} == parameters
    assert set(h.calls[0]) == {'input_path', 'input_sha256', 'output_dir', *parameters}
    parent, child = h.identities
    assert parent[1] != child[1] and parent[2] is not child[2]
    state = child_state(h)
    assert state.steps[0].verification.artifact_authority['execution_identity'] == child[1]
    assert result.steps[0].verification.artifact_authority['execution_identity'] == parent[1]
    assert h.runtime().planner is None
    with pytest.raises(RuntimeError):
        annotation_invocation()


def test_verified_child_dependency_is_exact_and_reused(harness):
    h = harness
    def second(inv, result):
        binding = inv.output('root', 'first', 'dataset')
        inv.invoke(branch_id='branch-A', operation_id='second', tool_name='child', arguments={
            'input_path': PriorOutputRef(binding, 'data_path'),
            'input_sha256': PriorOutputRef(binding, 'data_sha256')})
        raise Crash()
    h.after_child = second
    with pytest.raises(Crash):
        h.runtime().run_prepared(*h.prepare())
    assert len(h.calls) == 2
    rows = reservations(h.store.load('parent-1:run'))
    assert rows[1]['dependencies'][0]['run_id'] == rows[0]['request']['request_id'] + ':run'
    def second_again(inv, result):
        binding = inv.output('root', 'first', 'dataset')
        inv.invoke(branch_id='branch-A', operation_id='second', tool_name='child', arguments={
            'input_path': PriorOutputRef(binding, 'data_path'),
            'input_sha256': PriorOutputRef(binding, 'data_sha256')})
    h.after_child = second_again
    assert continue_parent(h).status is RunStatus.SUCCEEDED
    assert len(h.calls) == 2


def test_parent_cancellation_reaches_active_child(harness):
    h = harness
    entered, release = threading.Event(), threading.Event()
    def blocking(*args):
        entered.set()
        assert release.wait(10)
        cancellation_checkpoint()
        pytest.fail('cancelled child must stop')
    h.child_action = blocking
    results = []
    worker = threading.Thread(target=lambda: results.append(h.runtime().run_prepared(*h.prepare())))
    worker.start()
    assert entered.wait(10)
    with pytest.raises(RunAlreadyActiveError):
        h.runtime().run_prepared(*h.prepare())
    h.runtime().cancel('parent-1:run')
    release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert results[0].status is RunStatus.CANCELLED
    child = child_state(h)
    assert child.lifecycle_status is RunLifecycleStatus.CANCELLED
    assert h.store.load_cancellation(child.run_id) is not None


def test_failure_preserves_children_and_cannot_complete_by_skipping_them(harness):
    h = harness
    def fail(*args):
        raise ValueError('parent publication failure')
    h.after_child = fail
    assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.FAILED
    child = child_state(h)
    original = Path(child.steps[0].result['data_path']).read_bytes()
    h.after_child = None
    assert continue_parent(h).status is RunStatus.SUCCEEDED
    assert len(h.calls) == 1
    assert Path(child.steps[0].result['data_path']).read_bytes() == original


def test_unrelated_prior_output_and_bare_path_are_rejected(harness):
    h = harness
    request = AgentRequest('unrelated', None, {})
    plan = AgentPlan('unrelated-plan', request.request_id, 'unrelated', (PlanStep('operation', 'child', {
        'input_path': h.source, 'input_sha256': h.sha, 'output_dir': str(Path(h.source).parent / 'other')}),))
    assert h.runtime().run_prepared(request, plan).status is RunStatus.SUCCEEDED
    with prior_output_store(h.store):
        binding = bind_output(h.store, h.registry, 'unrelated:run', 'operation', 'dataset')
    def try_escape(inv):
        with pytest.raises(AuthorityError, match='not captured'):
            inv.invoke(branch_id='escape', operation_id='unrelated', tool_name='child', arguments={
                'input_path': PriorOutputRef(binding, 'data_path'), 'input_sha256': PriorOutputRef(binding, 'data_sha256')})
        with pytest.raises(AuthorityError, match='captured input'):
            inv.invoke(branch_id='escape', operation_id='path', tool_name='child', arguments={'input_path': h.source})
    h.before_parent = try_escape
    assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.SUCCEEDED
    assert len(h.calls) == 2


def test_corrupt_child_publication_is_not_recomputed(harness):
    h = harness
    h.after_child = lambda *a: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        h.runtime().run_prepared(*h.prepare())
    child = child_state(h)
    Path(child.steps[0].result['data_path']).write_text('tampered')
    h.after_child = None
    assert continue_parent(h).status is RunStatus.FAILED
    assert len(h.calls) == 1
    assert h.store.load(child.run_id).to_dict() == child.to_dict()


@pytest.mark.parametrize('cancel', [False, True])
def test_failed_and_cancelled_children_remain_terminal(harness, cancel):
    h = harness
    def stop(*args):
        if cancel:
            h.runtime().cancel(child_state(h).run_id)
            cancellation_checkpoint()
        raise ValueError('native fixture failed')
    h.child_action = stop
    assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.FAILED
    child = child_state(h)
    assert child.lifecycle_status is (RunLifecycleStatus.CANCELLED if cancel else RunLifecycleStatus.FAILED)
    h.child_action = None
    assert continue_parent(h).status is RunStatus.FAILED
    assert len(h.calls) == 1
    assert h.store.load(child.run_id).to_dict() == child.to_dict()


def test_child_publication_wins_step_checkpoint_while_run_is_cancelled(harness):
    h = harness
    class CancelAtSuccess(FaultStore):
        def update(self, state, *, expected_revision):
            if (not self.triggered and state.run_id.startswith('annotation-child-')
                    and state.steps and state.steps[0].status is StepStatus.SUCCEEDED):
                self.triggered = True
                h.runtime().cancel('parent-1:run')
            return self.store.update(state, expected_revision=expected_revision)
    result = h.runtime(CancelAtSuccess(h.store, lambda s: False)).run_prepared(*h.prepare())
    assert result.status is RunStatus.CANCELLED
    child = child_state(h)
    assert child.lifecycle_status is RunLifecycleStatus.CANCELLED
    assert child.steps[0].status is StepStatus.SUCCEEDED
    assert child.steps[0].verification.passed
    assert Path(child.steps[0].result['data_path']).is_file()


def test_active_child_is_not_resumed_or_replaced(harness):
    h = harness
    fault = FaultStore(h.store, lambda s: s.run_id.startswith('annotation-child-') and bool(s.steps)
                       and s.steps[0].status is StepStatus.RUNNING)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(*h.prepare())
    child = child_state(h)
    with h.store.execution_lease(child.run_id):
        assert continue_parent(h).status is RunStatus.FAILED
    assert h.store.load(child.run_id).to_dict() == child.to_dict()
    assert not h.calls
    assert any('RunAlreadyActiveError' in e for e in h.exceptions)


def test_continuation_is_explicit_exact_and_read_only_until_execution(harness):
    h = harness
    h.after_child = lambda *a: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        h.runtime().run_prepared(*h.prepare(parameters={'parameter_selection': 'native_defaults'}))
    with pytest.raises(ValueError, match='terminal'):
        continuation_anchor(h.store, 'parent-1:run', 'annotate')
    h.runtime().resume('parent-1:run')
    before = h.store.load('parent-1:run').to_dict()
    anchor = continuation_anchor(h.store, 'parent-1:run', 'annotate')
    assert h.store.load('parent-1:run').to_dict() == before
    with pytest.raises(AuthorityError, match='anchor changed'):
        h.runtime().continue_annotation(*h.prepare('bad-anchor'), previous=replace(anchor, state_sha256='0' * 64))
    with pytest.raises(AuthorityError, match='exact captured'):
        h.runtime().continue_annotation(*h.prepare('changed'), previous=anchor)
    assert len(h.calls) == 1


def test_captured_input_guard_remains_in_force_for_children(harness):
    from agent.orchestration.executor import execution_input_guard
    h = harness
    guarded = []
    def guard(step, args):
        guarded.append(step.tool_name)
        assert args['input_path'] == h.source and args['input_sha256'] == h.sha
        if step.tool_name == 'child':
            raise AuthorityError('captured scientific resource revoked')
    with execution_input_guard(guard):
        result = h.runtime().run_prepared(*h.prepare())
    assert result.status is RunStatus.FAILED
    assert guarded == ['parent', 'child']
    assert not h.calls


def test_registry_recursion_and_unscoped_executor_are_rejected(harness):
    h = harness
    with pytest.raises(ValueError, match='non-coordinator'):
        ToolRegistry((h.registry.get('parent'), replace(h.registry.get('child'),
            annotation_children=frozenset({'parent'}))))
    outcome = h.runtime().executor.execute(h.prepare()[1], checkpoint=lambda p: None,
                                           durable_run_id='unscoped:run')
    assert outcome.errors and not h.calls
    assert any('owning Runtime' in e for e in h.exceptions)


def test_coordinator_cannot_enter_native_worker_or_escape_context(harness):
    from agent.tools._native_process import _CONTROLLED
    h = harness
    token = _CONTROLLED.set(True)
    try:
        assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.FAILED
    finally:
        _CONTROLLED.reset(token)
    assert not h.calls
    assert any('top-level durable service' in e for e in h.exceptions)


def test_failed_progress_checkpoint_stops_before_child_creation(harness):
    from agent.orchestration.annotation_runs import AnnotationCheckpointError
    h = harness
    class UnavailableStore(FaultStore):
        def update(self, state, *, expected_revision):
            if state.run_id == 'parent-1:run' and reservations(state):
                raise OSError('storage unavailable')
            return self.store.update(state, expected_revision=expected_revision)
    with pytest.raises(AnnotationCheckpointError):
        h.runtime(UnavailableStore(h.store, lambda s: False)).run_prepared(*h.prepare())
    assert not h.calls and not reservations(h.store.load('parent-1:run'))


def test_ordinary_registry_has_no_annotation_coordinator_opt_in():
    # The fixture's inventory is not used by this test.
    from agent.orchestration.registry import build_default_tool_registry
    registry = build_default_tool_registry()
    assert not any(registry.get(n).annotation_children for n in registry.names())
    assert registry.get('annotate_scATAC_cell_types').durable_hooks.execute.__module__ == 'agent.tools.analysis.scatac_annotation'


def test_one_session_turn_and_no_child_revision(harness, tmp_path):
    from agent.application import OutputSelection, ResearchAgentApplication
    h = harness
    app = ResearchAgentApplication(tmp_path / 'application', registry=h.registry)
    h.store = app.run_store
    app.sessions.create('research')
    request, plan = h.prepare()
    request = replace(request, prompt='Fixture annotation research task')
    state = app.sessions.run('research', 'annotation', request,
        (OutputSelection('annotation', 'annotate', 'data_path'),),
        expected_generation=0, prepared_plan=plan)
    assert len(state.turns) == 1
    # A fixture has no production report projection. Science succeeds without
    # inventing an accepted annotation presentation or a Revision for a child.
    assert state.turn('annotation').status == 'run_succeeded'
    assert not state.revisions
    outputs = app.sessions._locators(state.turn('annotation'), h.store.load('parent-1:run'))
    assert all(o.run_id == 'parent-1:run' for o in outputs)
    assert child_state(h).lifecycle_status is RunLifecycleStatus.SUCCEEDED


def test_continuation_restores_exact_parent_prerequisite_authority(harness, tmp_path):
    h = harness
    request, original = h.prepare()
    prerequisite = PlanStep('prerequisite', 'child', dict(input_path=h.source,
        input_sha256=h.sha, output_dir=str(tmp_path / 'prerequisite')))
    parent = replace(original.steps[0], arguments=dict(original.steps[0].arguments,
        input_path=StepOutputRef('prerequisite', 'data_path'),
        input_sha256=StepOutputRef('prerequisite', 'data_sha256')), depends_on=('prerequisite',))
    plan = replace(original, steps=(prerequisite, parent))
    h.before_parent = lambda inv: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        h.runtime().run_prepared(request, plan)
    h.runtime().resume('parent-1:run')
    old = h.store.load('parent-1:run')
    source = old.steps[0].result
    def check_source(inv):
        assert (source['manifest_path'], source['manifest_sha256']) in current().accepted
    h.before_parent = check_source
    request2, plan2 = h.prepare('parent-2')
    plan2 = replace(plan2, steps=(replace(plan2.steps[0], arguments=dict(plan2.steps[0].arguments,
        input_path=source['data_path'], input_sha256=source['data_sha256'])),))
    anchor = continuation_anchor(h.store, old.run_id, 'annotate')
    result = h.runtime().continue_annotation(request2, plan2, previous=anchor)
    assert result.status is RunStatus.SUCCEEDED, h.exceptions
    assert len(h.calls) == 2  # Original prerequisite once, newly required child once.
    assert h.store.load(old.run_id).to_dict() == old.to_dict()


def test_final_parent_publication_recovers_without_reentering_coordinator(harness):
    h = harness
    fault = FaultStore(h.store, lambda s: s.run_id == 'parent-1:run' and bool(s.steps)
                       and s.steps[0].status is StepStatus.SUCCEEDED, before=True)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(*h.prepare())
    child = child_state(h)
    pending = h.store.load('parent-1:run')
    assert pending.steps[0].status is StepStatus.RUNNING
    assert pending.steps[0].verification is None
    h.before_parent = lambda inv: pytest.fail('coordinator re-entered during publication recovery')
    result = h.runtime().resume('parent-1:run')
    assert result.status is RunStatus.SUCCEEDED, h.exceptions
    assert len(h.calls) == 1
    authority = result.steps[0].verification.artifact_authority
    assert authority is not None
    assert authority['schema_version'] == 2
    assert authority['execution_identity'] == execution_identity(
        pending.run_id, pending.plan, pending.plan.steps[0], h.registry.get('parent'))
    binding = bind_output(h.store, h.registry, result.run_id, 'annotate', 'dataset')
    validate_binding(binding, store=h.store, registry=h.registry, integrity=True)
    assert h.store.load(child.run_id).to_dict() == child.to_dict()


@pytest.mark.parametrize('damage', ['missing_manifest', 'missing_output', 'incomplete',
                                   'output_bytes', 'source_bytes', 'execution_identity'])
def test_coordinator_recovery_cannot_issue_authority_for_invalid_publication(harness, damage):
    h = harness
    fault = FaultStore(h.store, lambda s: s.run_id == 'parent-1:run' and bool(s.steps)
                       and s.steps[0].status is StepStatus.SUCCEEDED, before=True)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(*h.prepare())
    pending = h.store.load('parent-1:run')
    child = child_state(h)
    root = Path(pending.steps[0].resolved_arguments['output_dir'])
    if damage == 'missing_manifest':
        (root / 'manifest.json').unlink()
    elif damage == 'missing_output':
        (root / 'data.txt').unlink()
    elif damage in {'incomplete', 'execution_identity'}:
        path = root / 'manifest.json'
        value = json.loads(path.read_text())
        if damage == 'incomplete':
            value['native_completion'] = 'running'
        else:
            # A genuine other execution identity cannot authorize this parent.
            value['execution_identity'] = child.steps[0].verification.artifact_authority['execution_identity']
        path.write_text(json.dumps(value))
    else:
        Path(h.source if damage == 'source_bytes' else root / 'data.txt').write_text('changed')
    h.before_parent = lambda inv: pytest.fail('recovery re-entered coordinator')
    result = h.runtime().resume(pending.run_id)
    assert result.status is RunStatus.FAILED
    saved = h.store.load(pending.run_id)
    assert saved.lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert all(s.verification is None or s.verification.artifact_authority is None for s in saved.steps)
    with pytest.raises(AuthorityError):
        bind_output(h.store, h.registry, pending.run_id, 'annotate', 'dataset')
    assert h.store.load(child.run_id).to_dict() == child.to_dict()
    assert len(h.calls) == 1


def test_failed_child_blocks_recovery_even_if_coordinator_writes_publication(harness, monkeypatch):
    from agent.orchestration.annotation_runs import _AnnotationInvocation
    h = harness
    invoke = _AnnotationInvocation.invoke
    def ignore_child_failure(inv, **kwargs):
        with pytest.raises(Exception):
            invoke(inv, **kwargs)
        return {}  # Simulate a broken coordinator that continues after failure.
    def fail(*args):
        raise ValueError('native child failed')
    def crash_before_completion_check(inv):
        assert (Path(inv._arguments['output_dir']) / 'manifest.json').is_file()
        raise Crash()
    h.child_action = fail
    with monkeypatch.context() as patch:
        patch.setattr(_AnnotationInvocation, 'invoke', ignore_child_failure)
        patch.setattr(_AnnotationInvocation, '_require_complete', crash_before_completion_check)
        with pytest.raises(Crash):
            h.runtime().run_prepared(*h.prepare())
    child = child_state(h)
    assert child.lifecycle_status is RunLifecycleStatus.FAILED
    result = h.runtime().resume('parent-1:run')
    assert result.status is RunStatus.FAILED
    assert all(s.verification is None for s in result.steps)
    assert h.store.load(child.run_id).to_dict() == child.to_dict()
    assert len(h.calls) == 1


def test_terminal_success_without_authority_is_not_upgraded(harness):
    h = harness
    result = h.runtime().run_prepared(*h.prepare())
    assert result.status is RunStatus.SUCCEEDED
    state = h.store.load(result.run_id)
    step = replace(state.steps[0], verification=replace(state.steps[0].verification, artifact_authority=None))
    # A separate disposable historical fixture; never rewrite a terminal Run.
    historical = FileRunStore(h.store.root.parent / 'historical_run_state')
    saved = historical.create(replace(state, revision=0, steps=(step,)))
    before = historical.state_path(saved.run_id).read_bytes()
    h.before_parent = lambda inv: pytest.fail('terminal coordinator re-entered')
    assert h.runtime(historical).resume(saved.run_id).status is RunStatus.SUCCEEDED
    with pytest.raises(AuthorityError, match='legacy qualification is not implicit'):
        bind_output(historical, h.registry, saved.run_id, 'annotate', 'dataset')
    assert historical.state_path(saved.run_id).read_bytes() == before
    assert h.store.load(state.run_id).to_dict() == state.to_dict()
    assert len(h.calls) == 1


def test_unrelated_durable_owner_does_not_gain_recovery_authority(harness):
    h = harness
    request, original = h.prepare('unrelated')
    plan = replace(original, steps=(replace(original.steps[0], tool_name='child'),))
    fault = FaultStore(h.store, lambda s: s.run_id == 'unrelated:run' and bool(s.steps)
                       and s.steps[0].status is StepStatus.SUCCEEDED, before=True)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(request, plan)
    result = h.runtime().resume('unrelated:run')
    assert result.status is RunStatus.SUCCEEDED
    assert Path(result.steps[0].result['data_path']).is_file()
    assert result.steps[0].verification.artifact_authority is None
    with pytest.raises(AuthorityError, match='legacy qualification is not implicit'):
        bind_output(h.store, h.registry, result.run_id, 'annotate', 'dataset')
    assert len(h.calls) == 1


def test_recovered_coordinator_cannot_borrow_accepted_execution_authority(harness, monkeypatch):
    h = harness
    fault = FaultStore(h.store, lambda s: s.run_id == 'parent-1:run' and bool(s.steps)
                       and s.steps[0].status is StepStatus.SUCCEEDED, before=True)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(*h.prepare())
    assert h.runtime().resume('parent-1:run').status is RunStatus.SUCCEEDED
    state = h.store.load('parent-1:run')
    binding = bind_output(h.store, h.registry, state.run_id, 'annotate', 'dataset')
    child = child_state(h)
    borrowed = replace(state, steps=(replace(state.steps[0], verification=replace(
        state.steps[0].verification, artifact_authority=child.steps[0].verification.artifact_authority)),))
    load = h.store.load
    with monkeypatch.context() as patch:
        patch.setattr(h.store, 'load', lambda rid: borrowed if rid == state.run_id else load(rid))
        with pytest.raises(AuthorityError, match='different execution'):
            validate_binding(binding, store=h.store, registry=h.registry, integrity=True)
    assert h.store.load(state.run_id).to_dict() == state.to_dict()


def test_continuation_cannot_silently_skip_reserved_branch(harness):
    h = harness
    def second(inv, result):
        inv.invoke(branch_id='branch-B', operation_id='second', tool_name='child',
                   parent_inputs={'input_path': 'input_path', 'input_sha256': 'input_sha256'})
        raise Crash()
    h.after_child = second
    with pytest.raises(Crash):
        h.runtime().run_prepared(*h.prepare())
    h.after_child = None
    result = continue_parent(h)
    assert result.status is RunStatus.FAILED
    assert any('unfinished reserved children' in e for e in h.exceptions)
    assert len(h.calls) == 2
    assert all(s.verification is None for s in result.steps)


def test_cancellation_between_children_preserves_success_without_next_reservation(harness):
    h = harness
    def between(inv, result):
        h.runtime().cancel('parent-1:run')
        inv.invoke(branch_id='next', operation_id='second', tool_name='child',
                   parent_inputs={'input_path': 'input_path', 'input_sha256': 'input_sha256'})
    h.after_child = between
    assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.CANCELLED
    assert len(reservations(h.store.load('parent-1:run'))) == 1
    assert child_state(h).lifecycle_status is RunLifecycleStatus.SUCCEEDED
    assert len(h.calls) == 1


def test_captured_input_port_cannot_drop_its_checksum(harness):
    h = harness
    def incomplete(inv):
        with pytest.raises(AuthorityError, match='atomically'):
            inv.invoke(branch_id='bad', operation_id='missing-pin', tool_name='child',
                       parent_inputs={'input_path': 'input_path'})
    h.before_parent = incomplete
    assert h.runtime().run_prepared(*h.prepare()).status is RunStatus.SUCCEEDED
    assert len(h.calls) == 1


def test_multiple_explicit_continuations_keep_original_child_identity(harness):
    h = harness
    h.after_child = lambda *a: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        h.runtime().run_prepared(*h.prepare())
    with pytest.raises(Crash):
        continue_parent(h)
    h.runtime().resume('parent-2:run')
    anchor = continuation_anchor(h.store, 'parent-2:run', 'annotate')
    h.after_child = None
    result = h.runtime().continue_annotation(*h.prepare('parent-3'), previous=anchor)
    assert result.status is RunStatus.SUCCEEDED, h.exceptions
    assert len(h.calls) == 1
    assert not reservations(h.store.load('parent-3:run'))
    assert child_state(h).steps[0].verification.passed


@pytest.mark.parametrize('second_finished', [False, True])
def test_parent_publication_recovery_rejects_unfinished_reserved_child(harness, monkeypatch, second_finished):
    from agent.orchestration.annotation_runs import _AnnotationInvocation
    h = harness
    def second(inv, result):
        inv.invoke(branch_id='branch-B', operation_id='second', tool_name='child',
                   parent_inputs={'input_path': 'input_path', 'input_sha256': 'input_sha256'})
        raise Crash()
    h.after_child = second
    fault = FaultStore(h.store, lambda s: s.request.inputs.get('operation_id') == 'second'
                       and bool(s.steps) and s.steps[0].status is StepStatus.RUNNING)
    with pytest.raises(Crash):
        h.runtime(h.store if second_finished else fault).run_prepared(*h.prepare())
    h.runtime().resume('parent-1:run')
    original = h.store.load('parent-1:run')
    unfinished_id = reservations(original)[1]['request']['request_id'] + ':run'
    unfinished = h.store.load(unfinished_id)
    successful = child_state(h)
    child_bytes = Path(successful.steps[0].result['data_path']).read_bytes()
    expected_calls = 2 if second_finished else 1
    assert unfinished.steps[0].status is (StepStatus.SUCCEEDED if second_finished else StepStatus.RUNNING)
    assert len(h.calls) == expected_calls

    # A changed coordinator omits the inherited unfinished branch, publishes
    # its parent artifact, then loses the process before the in-memory check.
    h.after_child = None
    with monkeypatch.context() as patch:
        def crash_before_check(inv):
            assert (Path(inv._arguments['output_dir']) / 'manifest.json').is_file()
            raise Crash()
        patch.setattr(_AnnotationInvocation, '_require_complete', crash_before_check)
        with pytest.raises(Crash):
            h.runtime().continue_annotation(*h.prepare('parent-2'),
                previous=continuation_anchor(h.store, original.run_id, 'annotate'))
    parent = h.store.load('parent-2:run')
    assert parent.steps[0].status is StepStatus.RUNNING
    parent_path = Path(parent.steps[0].resolved_arguments['output_dir']) / 'manifest.json'
    parent_bytes = parent_path.read_bytes()
    # Establish that the receipt itself passes the actual NativePublication
    # recovery owner; rejection must come from the durable child invariant.
    spec = h.registry.get('parent')
    spec.durable_hooks.recover(parent.steps[0].resolved_arguments,
        execution_identity(parent.run_id, parent.plan, parent.plan.steps[0], spec))
    h.before_parent = lambda inv: pytest.fail('Recovery must not re-enter the coordinator')
    result = h.runtime().resume(parent.run_id)
    assert result.status is RunStatus.FAILED
    assert h.store.load(parent.run_id).lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert result.errors[0].code == 'STEP_OUTCOME_UNKNOWN_AFTER_INTERRUPTION'
    assert len(h.calls) == expected_calls
    assert h.store.load(unfinished_id).to_dict() == unfinished.to_dict()
    assert h.store.load(successful.run_id).to_dict() == successful.to_dict()
    assert h.store.load(original.run_id).to_dict() == original.to_dict()
    assert Path(successful.steps[0].result['data_path']).read_bytes() == child_bytes
    assert parent_path.read_bytes() == parent_bytes


@pytest.mark.parametrize('damage', ['outcome_anchor', 'child_publication'])
def test_recovered_parent_requires_exact_verified_child_outcome(harness, monkeypatch, damage):
    from agent.orchestration.annotation_runs import _AnnotationInvocation
    h = harness
    if damage == 'outcome_anchor':
        checkpoint = _AnnotationInvocation._checkpoint
        def wrong_anchor(inv, event, **details):
            if event == 'outcome':
                details['accepted_step_sha256'] = '0' * 64
            return checkpoint(inv, event, **details)
        monkeypatch.setattr(_AnnotationInvocation, '_checkpoint', wrong_anchor)
    fault = FaultStore(h.store, lambda s: s.run_id == 'parent-1:run' and bool(s.steps)
                       and s.steps[0].status is StepStatus.SUCCEEDED, before=True)
    with pytest.raises(Crash):
        h.runtime(fault).run_prepared(*h.prepare())
    child = child_state(h)
    assert child.lifecycle_status is RunLifecycleStatus.SUCCEEDED
    if damage == 'child_publication':
        Path(child.steps[0].result['data_path']).write_text('changed after parent publication')
    h.before_parent = lambda inv: pytest.fail('Recovery must not re-enter the coordinator')
    result = h.runtime().resume('parent-1:run')
    assert result.status is RunStatus.FAILED
    assert h.store.load('parent-1:run').lifecycle_status is RunLifecycleStatus.INTERRUPTED
    assert h.store.load(child.run_id).to_dict() == child.to_dict()
    assert len(h.calls) == 1
