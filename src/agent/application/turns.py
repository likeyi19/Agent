"""Small interaction adapter above the existing Planner, sessions and executor."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from .response_facts import TurnResponseFacts
import re

from agent.schemas import AgentRequest, PriorOutputRef, StepOutputRef
from agent.schemas.orchestration import _serialize
from agent.orchestration.active_context import ActiveContextItem, ActivePlanningContext, planning_context
from agent.orchestration.prior_outputs import binding_for_locator, validate_binding
from agent.orchestration.planner import PlannerError
from .session_state import Interaction, OutputLocator, OutputSelection, SessionTurn, SessionConflictError, digest
from .turn_context import snapshot, public_context, parameter_specs, stored_step, SELECTION, MATRIX
from .turn_decisions import (Execute, Navigate, Clarify, Answer, IntentDelta, IntentError, interpret, clauses,
                             resolve_relation, admit_delta)


@dataclass(frozen=True)
class TurnOutcome:
    kind: str
    status: str
    clarification: Clarify | None = None
    text: str = ""
    facts: "TurnResponseFacts | None" = None


def _update(sessions, turn_id, **changes):
    def change(state):
        return replace(state, interactions=tuple(replace(i, **changes) if i.turn_id == turn_id else i
                                                  for i in state.interactions))
    return sessions._store._update(sessions_id(sessions), change)


def sessions_id(sessions):
    return sessions._interaction_session_id


def _terminal(sessions, interaction, clarification):
    captured = _serialize(interaction.snapshot)
    choices = ()
    if clarification.reason == 'ambiguous_revision':
        choices = tuple(k for k in ('parent', 'previous_active') if k in captured['relations'])
    elif clarification.reason in {'ambiguous_parameter', 'missing_parameter_value'}:
        base = captured['bases'][interaction.base_revision_id]
        choices = tuple(dict.fromkeys(k for o in base['operations'] for k in o['parameters']))
    if clarification.reason == 'missing_parameter_value':
        from .turn_decisions import ALIASES
        mentioned = tuple(k for k in choices if any(re.search(r'\b' + re.escape(a) + r'\b',
            interaction.utterance, re.I) for a in ALIASES.get(k, ())))
        if len(mentioned) == 1: choices = mentioned
    clarification = replace(clarification, choices=choices,
                            value_required=clarification.reason == 'missing_parameter_value')
    admitted = dict(kind='clarify', **asdict(clarification))
    def change(state):
        turns = state.turns
        if not any(t.turn_id == interaction.turn_id for t in turns):
            turns += (SessionTurn(interaction.turn_id, interaction.base_revision_id,
                                   interaction.base_generation, None, None, None, 'clarification'),)
        return replace(state, turns=turns, interactions=tuple(
            replace(i, status='clarification', admitted=admitted) if i.turn_id == interaction.turn_id else i
            for i in state.interactions))
    sessions._store._update(sessions_id(sessions), change)
    return TurnOutcome('clarify', 'clarification', clarification)


def _outcome(sessions, interaction):
    admitted = {} if interaction.admitted is None else _serialize(interaction.admitted)
    if admitted.get('kind') == 'answer':
        from .responses import admitted_answer
        return admitted_answer(sessions, sessions_id(sessions), admitted)
    if interaction.status == 'clarification':
        return TurnOutcome('clarify', 'clarification', Clarify(admitted['reason'],
            tuple(admitted.get('choices', ())), admitted.get('value_required', False)))
    state = sessions.load(sessions_id(sessions))
    turn = next((t for t in state.turns if t.turn_id == interaction.turn_id), None)
    if turn is not None and turn.request_id is not None:
        state = sessions.recover(state.session_id, interaction.turn_id)
        return TurnOutcome('execute', state.turn(interaction.turn_id).status)
    if turn is not None and turn.request_id is None and turn.status == 'activated':
        return TurnOutcome('navigate', 'activated')
    return TurnOutcome(admitted.get('kind', 'pending'), interaction.status)


NAVIGATION = re.compile(r'(?:use|go back to|switch back to|switch to) (?:the )?(?:parent|previous|earlier|previously active) version|go back one version|(?:use|switch back to) the version i was using before', re.I)
MATRIX_CLAUSE = re.compile(r'(?:rebuild|build)(?: the)? matrix', re.I)
SELECTION_CLAUSE = re.compile(r'(?:only (?:redo|run) cell selection|(?:redo|run) cell selection only|keep (?:this|the) selection)', re.I)


def admit(sessions, interaction, decision):
    captured = _serialize(interaction.snapshot)
    parts = clauses(interaction.utterance)
    relation = decision.relation if isinstance(decision, Navigate) else decision.base
    rid = resolve_relation(relation, captured, interaction.utterance)
    if isinstance(decision, Navigate):
        if not parts or not all(NAVIGATION.fullmatch(p) for p in parts): raise IntentError('unsupported_intent')
        return dict(kind='navigate', revision_id=rid)
    base = captured['bases'][rid]
    candidates = [o for o in base['operations'] if o['handle'] == decision.operation]
    if len(candidates) != 1: raise IntentError('unavailable_context')
    operation = candidates[0]
    parameters = dict(operation['parameters'])
    delta_clause = None if decision.delta is None else decision.delta.evidence
    for part in parts:
        if part != delta_clause and not any(p.fullmatch(part) for p in (NAVIGATION, MATRIX_CLAUSE, SELECTION_CLAUSE)):
            raise IntentError('unsupported_intent')
    wants_matrix = any(MATRIX_CLAUSE.fullmatch(p) for p in parts)
    selection_only = any(SELECTION_CLAUSE.fullmatch(p) and not p.lower().startswith('keep') for p in parts)
    if wants_matrix and selection_only: raise IntentError('unsupported_intent')
    if (decision.target == 'matrix') != wants_matrix: raise IntentError('unsupported_intent')
    if decision.delta is None:
        if not wants_matrix or not any(p.lower().startswith('keep ') for p in parts):
            raise IntentError('missing_parameter_value')
    else:
        value = admit_delta(decision.delta, interaction.utterance, parameters,
                            parameter_specs(sessions._application.registry), base['focus'])
        parameters[decision.delta.parameter] = value
    if wants_matrix and base['matrix_source'] is None: raise IntentError('unavailable_context')
    return dict(kind='execute', revision_id=rid, operation=operation, parameters=parameters,
        request_id='turn-' + digest(dict(session=sessions_id(sessions), turn=interaction.turn_id)),
        target=decision.target, delta=None if decision.delta is None else asdict(decision.delta),
        parameter=None if decision.delta is None else decision.delta.parameter,
        matrix_source=base['matrix_source'])


def _context(sessions, interaction, admitted, generation):
    base = _serialize(interaction.snapshot)['bases'][admitted['revision_id']]
    app = sessions._application
    items = []
    locators = {}
    for raw in base['outputs']:
        locator = OutputLocator(**raw)
        binding = binding_for_locator(locator, app.run_store, app.registry)
        validate_binding(binding, store=app.run_store, registry=app.registry, integrity=True)
        if binding not in locators:
            items.append(ActiveContextItem(f'ctx.{len(items)}', binding))
            locators[binding] = locator
    return ActivePlanningContext(sessions_id(sessions), admitted['revision_id'], generation, tuple(items)), locators


def _execution_inputs(sessions, admitted, context, utterance):
    app = sessions._application
    source = stored_step(sessions, admitted['operation']['source'])
    original = {k:_serialize(source.resolved_arguments[k]) for k in parameter_specs(app.registry)
                if k in source.resolved_arguments}
    expected = dict(original)
    if admitted['delta'] is not None:
        delta = IntentDelta(**admitted['delta'])
        expected[delta.parameter] = admit_delta(delta, utterance, original,
            parameter_specs(app.registry), focus=admitted['parameter'])
    if expected != admitted['parameters']:
        raise IntentError('invalid_parameter_value')
    def exact(path, sha):
        matches = []
        for item in context.items:
            step = validate_binding(item.binding, store=app.run_store, registry=app.registry)
            if step.result.get('manifest_path') == path and step.result.get('manifest_sha256') == sha:
                matches.append(item.binding)
        if len(matches) != 1: raise IntentError('unavailable_context')
        return matches[0]
    qc = exact(source.resolved_arguments['barcode_qc_manifest_path'], source.resolved_arguments['barcode_qc_manifest_sha256'])
    qc_step = validate_binding(qc, store=app.run_store, registry=app.registry)
    fragments = exact(qc_step.resolved_arguments['fragments_manifest_path'], qc_step.resolved_arguments['fragments_manifest_sha256'])
    selection = exact(source.result['manifest_path'], source.result['manifest_sha256'])
    inputs = dict(admitted['parameters']) if admitted['delta'] is not None else {}
    if admitted['target'] == 'matrix':
        recipe = stored_step(sessions, admitted['matrix_source'])
        inputs.update({k:recipe.resolved_arguments[k] for k in ('reference_manifest_path','reference_manifest_sha256')})
    return inputs, dict(qc=qc, fragments=fragments, selection=selection)


def _check_plan(plan, admitted, inputs, bindings, registry):
    """Enforce admitted effects; never create, rewrite, or complete a workflow."""
    expected = ([SELECTION] if admitted['delta'] is not None else []) + ([MATRIX] if admitted['target'] == 'matrix' else [])
    if sorted(s.tool_name for s in plan.steps) != sorted(expected):
        raise PlannerError('TURN_EFFECT_MISMATCH', 'Plan differs from the admitted scientific effects.')
    by_tool = {s.tool_name:s for s in plan.steps}
    output = []
    def pair(step, prefix, binding):
        for suffix in ('path','sha256'):
            if step.arguments.get(prefix+suffix) != PriorOutputRef(binding, 'manifest_'+suffix):
                raise PlannerError('TURN_SOURCE_MISMATCH', 'Plan differs from the admitted historical source.')
    if SELECTION in by_tool:
        step = by_tool[SELECTION]
        if {k for k in step.arguments if k in parameter_specs(registry)} != set(admitted['parameters']):
            raise PlannerError('TURN_PARAMETER_MISMATCH', 'Plan added or removed an admitted parameter.')
        for name, value in admitted['parameters'].items():
            if step.arguments.get(name) != value:
                raise PlannerError('TURN_PARAMETER_MISMATCH', 'Plan changed an admitted scientific parameter.')
        pair(step, 'barcode_qc_manifest_', bindings['qc'])
        output.append(OutputSelection(admitted['operation']['source']['name'], step.step_id, 'manifest_path'))
    if MATRIX in by_tool:
        step = by_tool[MATRIX]
        pair(step, 'fragments_manifest_', bindings['fragments'])
        for key in ('reference_manifest_path','reference_manifest_sha256'):
            if step.arguments.get(key) != inputs[key]: raise PlannerError('TURN_SOURCE_MISMATCH', 'Matrix reference changed.')
        if SELECTION in by_tool:
            for suffix in ('path','sha256'):
                if step.arguments.get('selected_cells_manifest_'+suffix) != StepOutputRef(by_tool[SELECTION].step_id,'manifest_'+suffix):
                    raise PlannerError('TURN_SOURCE_MISMATCH', 'Matrix must consume the new selection.')
        else:
            pair(step, 'selected_cells_manifest_', bindings['selection'])
        output.append(OutputSelection(admitted['matrix_source']['name'], step.step_id, 'manifest_path'))
    return tuple(output)


class _AdmittedPlanner:
    """Delegate unchanged planning/recovery, then check effects before execution."""
    def __init__(self, delegate, accept):
        self.delegate, self.accept = delegate, accept

    def plan(self, request, registry):
        plan = self.delegate.plan(request, registry)
        self.accept(request, plan)
        return plan

    def __getattr__(self, name):
        if name not in ('plan_with_recovery', 'plan_with_diagnostics'): raise AttributeError(name)
        method = getattr(self.delegate, name)  # Preserve absence for legacy planners.
        def delegated(request, registry, **kwargs):
            result = method(request, registry, **kwargs)
            plan = result.attempt.plan if name == 'plan_with_recovery' else result.plan
            self.accept(request, plan)
            return result
        return delegated


def execute(sessions, interaction, admitted):
    app = sessions._application
    generation = interaction.base_generation
    if admitted['revision_id'] != interaction.base_revision_id:
        # An explicit combined navigation/change is two authorized operations.
        state = sessions.switch(sessions_id(sessions), interaction.turn_id + ':base', admitted['revision_id'],
                                expected_generation=generation)
        generation = state.generation
    context, locators = _context(sessions, interaction, admitted, generation)
    inputs, bindings = _execution_inputs(sessions, admitted, context, interaction.utterance)
    request_id = admitted['request_id']
    goal = ('Using the offered current QC result, select cells with exactly the supplied parameters.'
            if admitted['delta'] is not None else 'Keep the offered current selection.')
    if admitted['target'] == 'matrix': goal += ' Rebuild the matrix using the offered fragments and selected cells.'
    request = AgentRequest(request_id, goal, inputs)
    def accept(effective, plan):
        selections = _check_plan(plan, admitted, inputs, bindings, app.registry)
        retained = tuple(locators[b] for b in (bindings['fragments'], bindings['qc']))
        if admitted['delta'] is None: retained += (locators[bindings['selection']],)
        def change(state):
            if any(t.turn_id == interaction.turn_id for t in state.turns):
                raise SessionConflictError('Turn already submitted.')
            turn = SessionTurn(interaction.turn_id, admitted['revision_id'], generation, request_id,
                digest(effective.to_dict()), request_id+':run', 'linked', selections, retained_outputs=retained)
            return replace(state, turns=state.turns+(turn,))
        sessions._store._update(sessions_id(sessions), change)
        _update(sessions, interaction.turn_id, status='submitted')
    from .service import ResearchAgentApplication
    execution_app = ResearchAgentApplication(app.workspace_root,
        planner=_AdmittedPlanner(app.runtime.planner, accept), registry=app.registry, executor=app.runtime.executor)
    with planning_context(context):
        result = execution_app.run(request)
    state = sessions.load(sessions_id(sessions))
    if any(t.turn_id == interaction.turn_id for t in state.turns):
        sessions._record_result(state.session_id, interaction.turn_id, result)
        state = sessions.recover(state.session_id, interaction.turn_id)
        return TurnOutcome('execute', state.turn(interaction.turn_id).status)
    _update(sessions, interaction.turn_id, status='failed')
    return TurnOutcome('execute', 'failed')


def respond(sessions, session_id, turn_id, utterance, *, interpreter, expected_generation=None):
    # Each property access returns its own facade; no process-global turn context.
    sessions._interaction_session_id = session_id
    if type(utterance) is not str or not utterance.strip() or len(utterance) > 4096:
        raise ValueError('A bounded nonempty utterance is required.')
    try:
        state = sessions.load(session_id)
    except (ValueError, RuntimeError, OSError):
        from .responses import UNAVAILABLE
        return TurnOutcome('answer', 'unavailable', text=UNAVAILABLE)
    existing = next((i for i in state.interactions if i.turn_id == turn_id), None)
    if existing is not None:
        if existing.utterance != utterance: raise SessionConflictError('Interaction identity changed.')
        return _outcome(sessions, existing)
    if any(t.turn_id == turn_id for t in state.turns): raise SessionConflictError('Turn identity already used.')
    generation = state.generation if expected_generation is None else expected_generation
    if generation != state.generation: raise SessionConflictError('Session generation changed.')
    try:
        captured = snapshot(sessions, state)
    except (ValueError, RuntimeError, OSError, KeyError):
        from .responses import UNAVAILABLE
        return TurnOutcome('answer', 'unavailable', text=UNAVAILABLE)
    interaction = Interaction(turn_id, utterance, state.active_revision_id, generation, captured)
    def capture(current):
        if current.generation != generation or any(i.turn_id == turn_id for i in current.interactions):
            raise SessionConflictError('Interaction capture raced with another turn.')
        return replace(current, interactions=current.interactions+(interaction,))
    sessions._store._update(session_id, capture)
    try:
        decision = interpret(interpreter, utterance, public_context(captured))
        if isinstance(decision, Clarify): return _terminal(sessions, interaction, decision)
        if isinstance(decision, Answer):
            from .responses import admit_answer
            admitted = admit_answer(interaction, decision)
        else:
            admitted = admit(sessions, interaction, decision)
    except Exception as exc:
        reason = exc.reason if isinstance(exc, IntentError) else 'invalid_decision'
        return _terminal(sessions, interaction, Clarify(reason))
    _update(sessions, turn_id, admitted=admitted, status='admitted')
    try:
        if isinstance(decision, Answer):
            from .responses import admitted_answer
            result = admitted_answer(sessions, session_id, admitted)
            _update(sessions, turn_id, status='answered')
            return result
        if isinstance(decision, Navigate):
            sessions.switch(session_id, turn_id, admitted['revision_id'], expected_generation=generation)
            _update(sessions, turn_id, status='navigated')
            return TurnOutcome('navigate', 'activated')
        return execute(sessions, interaction, admitted)
    except Exception:
        _update(sessions, turn_id, status='failed')
        raise
