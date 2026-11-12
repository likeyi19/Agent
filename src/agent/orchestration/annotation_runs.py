"""Internal annotation child invocation at the durable tool boundary.

No production coordinator is registered here. Runtime owns all Runs; the only
durable orchestration state is the parent's append-only execution trace.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from agent.schemas import AgentPlan, AgentRequest, PlanStep, RunLifecycleStatus, StepStatus, TraceEventType, fingerprint_plan
from agent.schemas.orchestration import _serialize
from agent.schemas.prior_output import PriorOutputRef
from agent.schemas.run_state import _decode_plan, _decode_request
from agent.schemas.verification_authority import AuthorityError, digest
from agent.tools._cancellation import cancellation_checkpoint
from agent.tools.data.authority_context import VerificationContext, authority_operation, current

from .durable_tool_recovery import execution_identity
from .prior_outputs import accepted_step_digest, bind_output, validate_binding
from .run_store import RunNotFoundError, RunStoreError

_RUNTIME = ContextVar('annotation_runtime', default=None)
_INVOCATION = ContextVar('annotation_invocation', default=None)
_CHILD = ContextVar('annotation_child', default=None)
_SCHEMA = 'annotation-child.v1'
_TERMINAL = frozenset({RunLifecycleStatus.FAILED, RunLifecycleStatus.INTERRUPTED,
                      RunLifecycleStatus.CANCELLED})


class AnnotationCheckpointError(RunStoreError):
    """A progress write failed; do not classify it as a scientific failure."""


@dataclass(frozen=True)
class AnnotationContinuation:
    run_id: str
    step_id: str
    state_sha256: str


def continuation_anchor(store, run_id, step_id):
    """Read-only backend selection. This neither resumes nor creates a Run."""
    state = store.load(run_id)
    if state.lifecycle_status not in _TERMINAL or state.plan is None:
        raise ValueError('Explicit continuation requires a terminal unfinished parent.')
    if not any(s.step_id == step_id for s in state.plan.steps):
        raise ValueError('Unknown continuation step.')
    return AnnotationContinuation(run_id, step_id, digest(state.to_dict()))


def _events(state, step_id, event):
    return [e.details for e in state.trace if e.event_type is TraceEventType.ANNOTATION_PROGRESS
            and e.step_id == step_id and e.details.get('schema') == _SCHEMA
            and e.details.get('event') == event]


def _reservation_history(store, state, step_id):
    """Read exact reservations and continuation ancestors from the existing trace."""
    reservations, ancestors, visited = {}, [], set()
    while True:
        if state.run_id in visited:
            raise AuthorityError('Cyclic annotation continuation.')
        visited.add(state.run_id)
        for row in _events(state, step_id, 'reserved'):
            key = (row['branch_id'], row['operation_id'])
            if key in reservations and reservations[key] != row:
                raise AuthorityError('Conflicting annotation child reservation.')
            reservations[key] = row
        parents = _events(state, step_id, 'continued')
        if not parents:
            return reservations, ancestors
        if len(parents) != 1:
            raise AuthorityError('Ambiguous annotation continuation.')
        parent = parents[0]
        state = store.load(parent['previous_run_id'])
        step_id = parent['previous_step_id']
        if continuation_anchor(store, state.run_id, step_id).state_sha256 != parent['state_sha256']:
            raise AuthorityError('Annotation continuation history changed.')
        ancestors.append(state)


def _reserved_plan(reservation):
    request = _decode_request(_serialize(reservation['request']))
    plan = _decode_plan(_serialize(reservation['plan']))
    if fingerprint_plan(plan) != reservation['plan_fingerprint']:
        raise AuthorityError('Reserved child plan fingerprint changed.')
    return request, plan


def _verified_child_anchor(runtime, state, plan):
    from .verifier import verify_step
    from .verification_authority import accepted_authorities
    if state.lifecycle_status is not RunLifecycleStatus.SUCCEEDED:
        raise AuthorityError('Annotation child is not terminal successful.')
    stored = state.steps[0]
    if stored.status is not StepStatus.SUCCEEDED or stored.verification is None or not stored.verification.passed:
        raise AuthorityError('Child has no accepted step.')
    spec = runtime.registry.get(stored.tool_name)
    with accepted_authorities(runtime._required_run_store(), state.run_id):
        checked = verify_step(plan.steps[0], stored.resolved_arguments, stored.result, runtime.registry,
            authority_execution_identity=execution_identity(state.run_id, plan, plan.steps[0], spec))
    if not checked.passed:
        raise AuthorityError('Child accepted output failed integrity verification.')
    return accepted_step_digest(stored)


def verify_recovered_completion(runtime, state, step):
    """Read-only parent recovery gate; never resume or execute a reserved child."""
    store = runtime._required_run_store()
    reservations, _ = _reservation_history(store, state, step.step_id)
    outcomes = {(row['branch_id'], row['operation_id']): row
                for row in _events(state, step.step_id, 'outcome')}
    if set(outcomes) != set(reservations):
        raise AuthorityError('Annotation recovery has unfinished reserved children.')
    for key, reservation in reservations.items():
        request, plan = _reserved_plan(reservation)
        outcome = outcomes[key]
        run_id = request.request_id + ':run'
        if outcome['child_run_id'] != run_id or outcome['status'] != RunLifecycleStatus.SUCCEEDED.value:
            raise AuthorityError('Annotation recovery lacks an exact successful child outcome.')
        try:
            child = store.load(run_id)
        except RunNotFoundError as exc:
            raise AuthorityError('Annotation recovery is missing its accepted child.') from exc
        if child.request != request or child.plan != plan:
            raise AuthorityError('Reserved child does not match its persisted execution.')
        runtime._validate_resume_policy(child)
        if _verified_child_anchor(runtime, child, plan) != outcome['accepted_step_sha256']:
            raise AuthorityError('Annotation child accepted outcome anchor changed.')


def checked_continuation(runtime, anchor, plan):
    """Pin old progress; exact inputs/parameters only, with a fresh destination."""
    store = runtime._required_run_store()
    if continuation_anchor(store, anchor.run_id, anchor.step_id) != anchor:
        raise AuthorityError('Annotation continuation anchor changed.')
    old = store.load(anchor.run_id)
    old_step = next(s for s in old.plan.steps if s.step_id == anchor.step_id)
    if len(plan.steps) != 1:
        raise ValueError('Annotation continuation requires one coordinator step.')
    step = plan.steps[0]
    spec = runtime.registry.get(step.tool_name)
    if not spec.annotation_children or old_step.tool_name != step.tool_name:
        raise ValueError('Continuation must select the same opted-in annotation owner.')
    runtime._validate_resume_policy(old)
    # Continuation is not branch editing. Existing exact input/parameter owners
    # must bind an edit in a later phase; do not silently change old decisions.
    saved = next((s for s in old.steps if s.step_id == anchor.step_id), None)
    expected = dict(saved.resolved_arguments if saved and saved.resolved_arguments else old_step.arguments)
    from .executor import _TraceRecorder
    actual = runtime.executor._resolve_arguments(step, {}, _TraceRecorder())
    previous_output = expected.pop('output_dir', None)
    output = actual.pop('output_dir', None)
    if expected != actual or output is None or output == previous_output:
        raise AuthorityError('Continuation requires exact captured arguments and a fresh output directory.')
    return dict(schema=_SCHEMA, event='continued', previous_run_id=anchor.run_id,
                previous_step_id=anchor.step_id, state_sha256=anchor.state_sha256)


@contextmanager
def runtime_scope(runtime):
    token = _RUNTIME.set(runtime)
    try:
        yield
    finally:
        _RUNTIME.reset(token)


def child_cancellation_parent():
    child = _CHILD.get()
    return None if child is None else child[0]


def child_source_context():
    child = _CHILD.get()
    return None if child is None else child[1]


def annotation_invocation():
    """Available only while the reviewed coordinator's durable hook executes."""
    invocation = _INVOCATION.get()
    if invocation is None or not invocation._active:
        raise RuntimeError('No active annotation invocation.')
    return invocation


@contextmanager
def tool_scope(plan, step, arguments, run_id, emit, *, required=False):
    runtime = _RUNTIME.get()
    if required and runtime is None:
        raise RuntimeError('Annotation invocation requires its owning Runtime.')
    spec = None if runtime is None else runtime.registry.get(step.tool_name)
    invocation = None
    if spec is not None and spec.annotation_children:
        from agent.tools._native_process import _CONTROLLED
        if run_id is None or _CHILD.get() is not None or _CONTROLLED.get():
            raise RuntimeError('Annotation coordination requires a top-level durable service invocation.')
        invocation = _AnnotationInvocation(runtime, plan, step, arguments, run_id, emit)
    token = _INVOCATION.set(invocation)
    try:
        yield
        if invocation is not None:
            invocation._require_complete()
    finally:
        if invocation is not None:
            invocation._active = False
        _INVOCATION.reset(token)


class _AnnotationInvocation:
    def __init__(self, runtime, plan, step, arguments, run_id, emit):
        self._runtime, self._step = runtime, step
        self._arguments, self._run_id, self._emit = arguments, run_id, emit
        self._store = runtime._required_run_store()
        self._spec = runtime.registry.get(step.tool_name)
        self._identity = execution_identity(run_id, plan, step, self._spec)
        self._active = True
        self._busy = False
        self._completed = {}
        self._reservations, ancestors = _reservation_history(self._store, self._store.load(run_id), step.step_id)
        for state in ancestors:
            # A failed parent DAG can contain accepted prerequisite steps. Its
            # exact continuation anchor, not copied path literals, carries those
            # proofs into the new coordinator. Never copy the execution field.
            from .verification_authority import _load_context
            from .prior_outputs import merge_context
            merge_context(current(), _load_context(self._store, state.run_id, 'historical_verified_sources.v1'))

    def _checkpoint(self, event, **details):
        try:
            self._emit(dict(schema=_SCHEMA, event=event, **details))
        except Exception as exc:
            from .run_store import CancellationRequestedError
            if isinstance(exc, CancellationRequestedError):
                raise
            raise AnnotationCheckpointError('Annotation progress was not durably saved.') from exc

    def check_cancelled(self):
        if not self._active or _INVOCATION.get() is not self:
            raise RuntimeError('Annotation invocation is outside its execution scope.')
        cancellation_checkpoint()

    def _require_complete(self):
        if set(self._completed) != set(self._reservations):
            raise RuntimeError('Annotation has unfinished reserved children.')

    def output(self, branch_id, operation_id, port_name):
        """An existing PriorOutputBinding, scoped to an observed verified child."""
        self.check_cancelled()
        run_id, _ = self._completed[branch_id, operation_id]
        return bind_output(self._store, self._runtime.registry, run_id, 'operation', port_name)

    def evidence_context(self):
        """Existing accepted-evidence readers use the owning store and Registry."""
        self.check_cancelled()
        return dict(store=self._store, registry=self._runtime.registry)

    def input_binding(self, name):
        """Return an exact captured historical input, never infer one from a path."""
        self.check_cancelled()
        reference = self._step.arguments.get(name)
        if not isinstance(reference, PriorOutputRef):
            raise AuthorityError('Annotation input requires an accepted PriorOutputRef: ' + name)
        validate_binding(reference.binding, **self.evidence_context(), integrity=True)
        return reference.binding

    def invoke(self, *, branch_id, operation_id, tool_name, arguments=None, parent_inputs=None):
        """One finite registered operation; parameters are passed exactly.

        Non-parameter inputs must explicitly name a captured parent argument or
        use an existing PriorOutputRef from that parent or a completed child.
        The caller cannot allocate Run IDs or output paths.
        """
        self.check_cancelled()
        if self._busy:
            raise RuntimeError('Annotation children must execute sequentially.')
        if not all(isinstance(v, str) and v.strip() for v in (branch_id, operation_id)):
            raise ValueError('Child branch and operation identities must be nonempty.')
        if tool_name not in self._spec.annotation_children:
            raise AuthorityError('Child operation is outside the reviewed annotation contract.')
        spec = self._runtime.registry.get(tool_name)
        if spec.annotation_children:
            raise AuthorityError('Annotation-in-annotation recursion is forbidden.')
        args = dict(arguments or {})
        sources = dict(parent_inputs or {})
        if 'output_dir' in args or 'output_dir' in sources or set(args) & set(sources):
            raise ValueError('Child destination is managed; input bindings cannot overlap.')
        allowed = {r.binding for r in self._step.arguments.values() if isinstance(r, PriorOutputRef)}
        dependencies = []
        for name, value in args.items():
            if isinstance(value, PriorOutputRef):
                binding = value.binding
                child = next((key for key, (rid, anchor) in self._completed.items()
                              if rid == binding.run_id and anchor == binding.accepted_step_sha256), None)
                if binding not in allowed and child is None:
                    raise AuthorityError('Child input is not captured by this parent or a verified child.')
                validate_binding(binding, store=self._store, registry=self._runtime.registry, integrity=True)
                if binding.to_dict() not in dependencies:
                    dependencies.append(binding.to_dict())
            else:
                contract = spec.required_arguments.get(name, spec.optional_arguments.get(name))
                if contract is None or contract.planning is None or not contract.planning.scientific_parameter:
                    raise AuthorityError('Child source arguments require exact captured input bindings.')
        for name, source in sources.items():
            if source == 'output_dir' or source not in self._arguments:
                raise AuthorityError('Unknown captured parent input.')
            args[name] = self._arguments[source]
        # Preserve each declared parent input as an atomic identity. In
        # particular, a coordinator must not forward a path while discarding
        # its captured checksum/resource members. Scientific eligibility still
        # belongs to the registered native owner and inherited input guard.
        semantic = self._spec.semantic_planning
        for port in (() if semantic is None else semantic.consumer_ports):
            members = {m.field_name for m in port.members if m.field_name in self._arguments}
            if members.intersection(sources.values()) and not members <= set(sources.values()):
                raise AuthorityError('Captured parent input ports must be passed atomically.')
        operation = dict(tool_name=tool_name, arguments=_serialize(args), parent_inputs=sources)
        key = (branch_id, operation_id)
        reservation = self._reservations.get(key)
        if reservation is None:
            token = digest(dict(parent_execution=self._identity, branch_id=branch_id,
                                operation_id=operation_id, operation=operation))
            output = Path(self._arguments['output_dir'])
            if not output.is_absolute() or output.resolve() != output:
                raise AuthorityError('Annotation requires an exact managed output destination.')
            # Sibling of the final publication, never owned by its staging cleanup.
            destination = output.parent / (output.name + '.children') / token
            if destination.resolve() != destination:
                raise AuthorityError('Child output destination must not traverse symlinks.')
            request = AgentRequest('annotation-child-' + token, None,
                dict(parent_run_id=self._run_id, parent_step_id=self._step.step_id,
                     parent_execution_identity=self._identity, branch_id=branch_id, operation_id=operation_id))
            plan = AgentPlan(request.request_id + ':plan', request.request_id, 'Annotation child operation',
                (PlanStep('operation', tool_name, dict(args, output_dir=str(destination))),))
            reservation = dict(schema=_SCHEMA, event='reserved', branch_id=branch_id, operation_id=operation_id,
                parent_run_id=self._run_id, parent_step_id=self._step.step_id, parent_execution_identity=self._identity,
                operation_sha256=digest(operation), parent_inputs=sources, request=request.to_dict(), plan=plan.to_dict(),
                plan_fingerprint=fingerprint_plan(plan), output_dir=str(destination), dependencies=dependencies)
            self._checkpoint('reserved', **{k: v for k, v in reservation.items() if k not in ('schema', 'event')})
            self._reservations[key] = reservation
        elif reservation['operation_sha256'] != digest(operation):
            raise AuthorityError('Reserved child operation or exact inputs changed.')
        request, plan = _reserved_plan(reservation)
        run_id = request.request_id + ':run'
        self._busy = True
        child_token = _CHILD.set((self._run_id, current()))
        try:
            # Neither child execution nor recovery may borrow the parent's mutable
            # execution field. The normal authority decorators restore child proofs.
            with authority_operation(VerificationContext()):
                try:
                    saved = self._store.load(run_id)
                except RunNotFoundError:
                    result = self._runtime.run_prepared(request, plan)
                else:
                    if saved.request != request or (saved.plan is not None and saved.plan != plan):
                        raise AuthorityError('Reserved child does not match its persisted execution.')
                    if saved.plan is None:
                        result = self._runtime._complete_annotation_child_creation(request, plan)
                    else:
                        self._runtime._validate_resume_policy(saved)
                        result = self._runtime.resume(run_id)
                saved = self._store.load(run_id)
                anchor = None
                if saved.lifecycle_status is RunLifecycleStatus.SUCCEEDED:
                    anchor = _verified_child_anchor(self._runtime, saved, plan)
        finally:
            _CHILD.reset(child_token)
            self._busy = False
        self._checkpoint('outcome', branch_id=branch_id, operation_id=operation_id, child_run_id=run_id,
                         status=saved.lifecycle_status.value, accepted_step_sha256=anchor)
        if anchor is None:
            self.check_cancelled()
            raise RuntimeError('Annotation child did not succeed: ' + saved.lifecycle_status.value)
        self._completed[key] = (run_id, anchor)
        return result
